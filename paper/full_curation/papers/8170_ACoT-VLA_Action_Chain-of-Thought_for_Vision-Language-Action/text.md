# arXiv:2601.11404v2[cs.RO]30Mar2026

## ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models

Linqing Zhong1,2 Yi Liu2 Yifei Wei1,2 Ziyu Xiong2 Maoqing Yao2* Si Liu1* Guanghui Ren2*

1Beihang University 2AgiBot

### Abstract

Vision-Language-Action models have emerged as essential generalist robot policies for diverse manipulation tasks, conventionally relying on directly translating multimodal inputs into actions via Vision-Language Model embeddings. Recent advancements have introduced explicit intermediary reasoning—such as sub-task prediction (language) or goal image synthesis (vision)—to guide action generation. However, these intermediate reasoning are often indirect and inherently limited in their capacity to convey the full, granular information required for precise action execution. Instead, we posit that the most effective form of reasoning is one that deliberates directly in the action space. We introduce Action Chain-of-Thought (ACoT), a paradigm where the reasoning process itself is formulated as a structured sequence of coarse action intents that guide the final policy. In this paper, we propose ACoT-VLA, a novel architecture that materializes the ACoT paradigm. Specifically, we introduce two complementary components: an Explicit Action Reasoner (EAR) and Implicit Action Reasoner (IAR). The former proposes coarse reference trajectories as explicit action-level reasoning steps, while the latter extracts latent action priors from internal representations of multimodal input, co-forming an ACoT that conditions the downstream action head to enable grounded policy learning. Extensive experiments in real-world and simulation environments demonstrate the superiority of our proposed method. Code is available at: https://github.com/ AgibotTech/ACoT-VLA.

- (a) Pre-trained VLM Action Policy

Instruction

Sub-tasks

Observation

Actions

- (b) World Model Action Policy

Instruction

Goal-image

Observation

Actions

- (c) Pre-trained VLM Action Policy

[Figure 1]

[Figure 2]

Reference Actions Actions

[Figure 3]

Instruction Observation

Figure 1. Chain-of-Thought in different space. (a) Language CoT paradigm predicts sub-tasks as intermediate reasoning. (b) Visual CoT paradigm synthesizes a goal image to provide guidance for action policy. (c) Our proposed Action CoT directly operates in action space and provides homogeneous action guidance.

Recent advancements seek to improve the mapping from the input space to the action space by introducing the intermediate reasoning step by language generation, leading to more generalized and precise action outputs [22, 55], as visualized in Fig. 1(a). A parallel thrust leverages world models [16, 51, 67] to simulate environmental dynamics, directly enhancing the efficacy and goal-oriented nature of the generated action sequences [60, 64], as shown in Fig. 1(b).

### 1. Introduction

Despite the promising trajectory set by these paradigms, a critical challenge persists: existing generalist policies think predominantly in the vision-language (input) space, often failing to adequately address the inherent disparity between these rich, semantic representations and the requirements of precise, low-level action execution (output). Specifically, the knowledge encoded within the VLM backbone of VLA models is derived from pre-training on webscale datasets focused on semantic alignment and question-

To overcome the generalization limits of task-specific robot policies [11, 48, 63], recent work has converged on VisionLanguage-Action (VLA) models [5, 24, 33, 41], which always leverage a pre-trained Vision-Language Model (VLM) [1, 2, 46] to encode visual and linguistic inputs into a latent representation that conditions an action decoder.

*Corresponding authors.

answering, yielding representations optimized for linguistic understanding rather than physical dynamics. Similarly, while world models forecast future visual states conditioned on inputs, their guidance remains tethered to naturally visual representations. Crucially, both semantic and visual forms of reasoning only offer suboptimal, indirect guidance for generating the necessary action sequence. Consequently, these prevailing approaches rely on an inherently constrained information conduit, struggling to convey the full, granular knowledge of the action space essential for truly grounded and accurate robotic policy learning.

The inherent semantic-kinematic gap in existing policies, i.e., a fundamental disconnect between high-level, abstract inputs and low-level, executable motor commands, necessitates a paradigm shift in how guidance is provided. We contend that to bridge this chasm, policies require guidance that is kinematically coherent, rather than purely semantic or visual. This core principle underpins our novel framework: Action Chain-of-Thought (ACoT) (Fig. 1(c)). We redefine the “thought” process not as a sequence of linguistic tokens, but as a structured chain of explicit, kinematically-grounded action intents. This approach furnishes the policy with direct motion cues, supplanting indirect representations. In a manner analogous to learning from physical demonstration, this direct conditioning on action-space information enables a substantially more efficient and veridically grounded policy learning process.

This foundational shift, however, introduces a critical and distinct research challenge: “How can we robustly and efficiently synthesize the complex, high-dimensional motion cues required for ACoT reasoning from the raw, heterogeneous multimodal inputs?”

Action-related information manifests in two complementary forms, i.e., explicit or implicit. The explicit form corresponds to observable motion trajectories, such as those in human demonstrations, which directly encode executable patterns of behavior. In contrast, the implicit form resides in latent cues, e.g., linguistic expressions like “reach out” or “grasp”, as well as interaction intents embedded in visual contexts. Although these cues are not presented as explicit robotic trajectories, they implicitly define distributions over feasible actions within the action space. Building upon this insight, we introduce two synergistic mechanisms to generate both explicit and implicit guidance in the action space. We first propose the Explicit Action Reasoner (EAR), which is realized as a light-weight transformer. Particularly, EAR synthesizes coarse-grained motion trajectories conditioned on multimodal observations, offering direct and executable guidance within the action space. Secondly, we devise the Implicit Action Reasoner (IAR), which infers latent action priors through applying cross-attention modeling between downsampled multimodal representations and learnable queries, thereby providing implicit behavioral pri-

ors. Note that these two mechanisms are inherently complementary to each other. Subsequently, through jointly leveraging both EAR and IAR, we develop ACoT-VLA, an integrated Action Chain-of-Thought framework that enables grounded generalist robot policy learning. Extensive experiments across both real-world settings and three simulation benchmarks consistently demonstrate the effectiveness and versatility of our ACoT-VLA.

To summarize, our main contributions are as follows:

- • Conceptually, we introduce Action Chain of Thought (ACoT), a new paradigm for generalist robot policies. To the best of our knowledge, this is the first work to formulate the deliberative process as a structured chain of explicit action-space intents, rather than abstract linguistic or visual sub-goals.
- • We delve into essential action space guidance and propose the Explicit and Implicit Action Reasoners, which provide both explicit trajectory guidance and implicit behavioral inspiration for action prediction.
- • Building upon these two modules, we further propose ACoT-VLA, a unified framework for grounded generalist robot policy learning.
- • Empirically, we validate our approach through extensive simulation and real-world experiments, achieving stateof-the-art performance on multiple benchmarks.

### 2. Related Works

Vision-Language-Action Models. VLA models [14, 18, 19, 27] incorporate pre-trained VLM models to predict language-driven robotic action sequences. Early works [24, 69] formulate robot control as an autoregressive sequence generation problem, discretizing continuous actions into bins. Inspired by generative modeling [31, 38, 68], increasing works [5, 20, 33] adopt diffusion-based action policies to synthesize smooth and high-quality action trajectories. Given that robotic manipulation inherently occurs in threedimensional space, a line of studies [30, 54, 61] have sought to enhance the spatial reasoning capability of VLA models by integrating 3D priors. For instance, SpatialVLA [41] integrates spatial embeddings to endow model with 3D awareness, while 4D-VLA [57] incorporates both spatial and temporal information to enrich representations. Besides, due to the scarcity of large-scale real-world robot demonstrations, a series of efforts [6, 10, 12, 23, 37, 59] focus on datacentric solutions, constructing large-scale robotic datasets through simulation or real-world collection to scale up policy learning. Moreover, recent large-scale co-training approaches, such as π0.5 [22], GenieReasoner [35] and Gemini Robotics [47], demonstrate the potential of unifying web-scale language understanding with action learning, enhancing the policy’s generalization ability while retaining the reasoning capability of pre-trained foundation models.

##### World-Model-based Policies. Advances in world mod-

[Figure 4]

[Figure 5]

(c) Action-Guided Prediction

- (a) Explicit Action Reasoner

N×️

[Figure 6]

- (b) Implicit Action Reasoner

ex

Place the block into the paper cup

Text Encoder

CrossAttention

Proj. Proj.

Self-Attention FFN

Noise

Cross-Attention

LLM

at:t+Href−1

[Figure 7]

Noise

Proj.

Action Head

SelfAttention

Denoised actions

at:t+H−1

[Figure 8]

Visual Encoder

im

Q Proj.

Pooling & Proj.

[Figure 9]

CrossAttention

K Proj.

Cross-Attention

V Proj.

KV Cache

- Figure 2. Architectural Overview of ACoT-VLA. The framework consists of three main components operating on features from a shared VLM backbone. (a) The Explicit Action Reasoner (EAR) is a Transformer-based module that synthesizes a coarse reference trajectory, providing explicit action-space guidance. (b) The Implicit Action Reasoner (IAR) employs a cross-attention mechanism with learnable queries to extract latent action priors from the VLM’s internal representations. (c) The Action-Guided Prediction (AGP) head synergistically integrates both explicit and implicit guidances via cross-attention to condition the final denoising process, producing the executable action sequence.

els have demonstrated remarkable capability in synthesizing high-fidelity images and temporally coherent videos. Building upon such progress, emerging researches [26, 29, 36, 58] exploit their predictive dynamics to implicitly guide action generation. Specifically, CoT-VLA [62] introduces visual chain-of-thought reasoning by forecasting sub-goal images, explicitly integrating visual reasoning into action prediction. WorldVLA [8] employs an autoregressive architecture that unifies perception and action generation within a single framework. DreamVLA [60] extends beyond visual prediction and enriches world modeling with dynamic, depth, and semantic cues, improving the model’s physical consistency. Collectively, existing world-model-based methods adopt knowledge-forecasting perspective, incorporating primarily visual guidance into trajectory generation.

In contrast to previous works focusing on visual or linguistic intermediaries for robotic policy learning, our key insight lies in investigating guidance directly within the action space, which intrinsically mitigates the heterogeneity between perception and action, enabling the model to effectively learn action-relevant priors.

### 3. Methodology

In this section, we present a detailed investigation into how to generate effective action space guidance and integrate it into robotic policy learning. We first define the robotic manipulation problem and formulate our proposed approach in Sec. 3.1. The core of our method lies in two distinct action reasoners introduced in Sec. 3.2 and Sec. 3.3, which provide explicit and implicit guidance within the action space. We conclude by illustrating the policy prediction strategy that effectively integrates this action guidance during pol-

icy learning (Sec. 3.4).

#### 3.1. Problem Formulation

Given a natural language instruction l and current visual observation ot, the generalist robot policy πθ aims to predict action sequences at:t+H−1 that accomplishes the specified task. The process can be formally expressed as:

at:t+H−1 = πθ(ot,l), (1) where H represents the action horizon. Numerous works introduce additional guidance signals g, which encapsulates various forms of auxiliary information to enhance policy’s prediction ability. Specifically, these guidance signals can be broadly categorized into two types: language-level guidance glang and vision-level guidance gvis. The former is primarily adopted by VLM-based methods, e.g., leveraging LLMs’ reasoning capabilities to predict sub-tasks, while the latter is always employed by world-model-based approaches, such as simulating future observations. Such relationship can be formulated as:

πθ(at:t+H−1,g | ot,l) = πθ(at:t+H−1 | ot,l,g)πθ(g | ot,l),

(2) where g ∈ {glang, gvis}. Conversely, we shift the focus toward the action domain itself and investigate cues operating directly in the action space, symbolized as gaction. The above guidances are extended as g ∈ {glang, gvis, gaction}.

Such action guidance can intuitively be disentangled into

explicit and implicit forms. The explicit guidance gactionex provides direct priors in the form of reference action se-

quences, whereas the implicit guidance gactionim arises from contextual signals, e.g., action distribution inherently implied in linguistics.

#### 3.2. Explicit Action Reasoner

To incorporate explicit action trajectories into the thinking process of πθ to generate high-quality action predictions, we propose the Explicit Action Reasoner (EAR).

We design a mechanism that enables the model to autonomously synthesize reference action sequences as internal guidance for policy learning. Analogously, this formulation can be viewed as an action-space transfer of selfconditioning in generative models [9, 34], where incorporating prior estimates into the generation process has been shown to markedly improve sample quality. Building upon this principle, we instantiate EAR as a light-weight transformer, as shown in Fig. 2 (a), generating kinematically plausible action reference as explicit action-space guidance gactionex for downstream action policy.

Formally, given visual observation ot and language instruction l, a pre-trained VLM encodes them into a contextual key-value cache:

(K1:VLMN ,V1:VLMN ) = VLM(ot,l), (3) where N represents the number of layers of VLM. Subsequently, the EAR, denoted as πθref, takes a noisy action sequence a˜t:t+Href−1 as input, where Href indicates the horizon of reference actions. The sequence is first embedded into an initial hidden representation href0 , which serves as the input to EAR’s transformer layers. At each transformer layer i, we adopt self-attention, along with crossattention with the contextual key-value cache from the corresponding VLM layer:

h˜refi = Self-Attn(hrefi−1) + CrossAttn(hrefi−1,KiVLM,ViVLM),

(4) where self-attention module captures temporal dependencies within the action sequence and cross-attention mechanism injects multimodal contextual priors from the VLM. Then, the intermediate representation h˜refi is processed by a feed-forward network (FFN) in a residual-parallel manner, updating the i-th EAR representation hrefi :

hrefi = hrefi−1 + FFN(h˜refi ). (5)

Through training via flow matching, πθref learns a distribution over action trajectories, producing a denoised action sequence:

ref−1 = πθref(˜at:t+Href−1,K1:VLMN ,V1:VLMN ). (6)

areft:t+H

The generated sequence is then encoded via a MLP projector to obtain action embedding Zex, which serves as explicit action-space guidance gactionex for action policy learning.

#### 3.3. Implicit Action Reasoner

Beyond the explicit action trajectories, the multimodal latent space of VLM also encodes implicit motion cues [13,

40], e.g., visual affordances and action-related semantics. Effectively extracting these action-relevant representations potentially offers complementary guidance. To this end, we introduce an Implicit Action Reasoner (IAR), which directly operates on the VLM’s key–value cache.

Concretely, as presented in Fig. 2 (b), for each VLM layer i ∈ [1,N], we initialize a learnable matrix Qi ∈ RM×d, where M is a hyperparameter and d represents VLM’s hidden dimension. Considering the information redundancy within VLM’s key–value cache and computational efficiency, we first downsample the corresponding key–value pairs into a lower-dimensional space, which is formulated as:

Q′i = QiWQ(i), Ki′ = KiVLMWK(i), Vi′ = ViVLMWV(i),

(7) where WQ(i),WK(i),WV(i) ∈ Rd×d

′

are learnable linear projectors and d′ ≪ d.

Later, cross-attention is applied to extract action-relevant information from each Ki′ and Vi′. The resulting features are subsequently integrated via average pooling and transformed through a MLP projector, as visualized in Fig. 2 (b), producing compact representations that capture the implicit action semantics ziim embedded in VLM’s i-th layer:

ziim = MLP(Pool(CrossAttn(Q′i,Ki′,Vi′))). (8) Then, through aggregating these representations across layers, we obtain implicit action-related feature Zim, which serves as implicit action-space guidance gactionim , complementing the explicit motion priors.

#### 3.4. Action-Guided Prediction

Building upon the explicit action embedding Zex produced by EAR and implicit action-related feature Zim obtained in IAR, in this section, we introduce the Action-Guided Prediction (AGP) strategy to incorporate both action guidances into policy learning.

As illustrated in Fig. 2 (c), given a noisy action segment a˜t:t+H−1, we first encode it into noisy action embedding via a MLP projector. Particularly, unlike previous approaches

that directly feed this embedding into action head πθhead, we treat it as action query, denoted as Qaction, which interacts with both Zex and Zim to retrieve complementary priors for conditional prediction.

Specifically, we perform dual cross-attention operations:

Sex = CrossAttn(Qaction,Zex,Zex), (9) Sim = CrossAttn(Qaction,Zim,Zim), (10)

where Sex and Sim denote the attended representations guided by explicit and implicit priors, respectively. Note that although both encode action-relevant information, they may highlight different facets of the underlying motion. For

Spatial Object Goal Long Avg. SR ↑ Rank ↓ SR ↑ Rank ↓ SR ↑ Rank ↓ SR ↑ Rank ↓ SR ↑ Rank ↓

Methods Guidance

Diffusion Policy [11] – 78.3 26 92.5 18 68.3 27 50.5 27 72.4 27 Octo [48] – 78.9 25 85.7 26 84.6 20 51.1 26 75.1 25

CoT-VLA [62] Visual 87.5 20 91.6 20 87.6 17 69.0 19 81.1 20 WorldVLA [8](256*256) Visual 85.6 22 89.0 23 82.6 22 59.0 22 79.1 21 WorldVLA [8](512*512) Visual 87.6 19 96.2 15 83.4 21 60.0 21 81.8 19 DreamVLA [60] Visual 97.5 9 94.0 16 89.5 15 89.5 12 92.6 14 UniVLA [49] Visual 95.4 14 98.8 4 93.6 12 94.0 6 95.5 10 F1 [36] Visual 98.2 5 97.8 10 95.4 11 91.3 10 95.7 9 GE-Act [29] Visual 98.2 5 97.6 12 95.8 9 94.4 5 96.5 7

TraceVLA [66] Linguistics 84.6 24 85.2 27 75.1 26 54.1 24 74.8 26 OpenVLA [24] Linguistics 84.7 23 88.4 24 79.2 23 53.7 25 76.5 24 UniAct [65] Linguistics 77.0 27 87.0 25 77.0 25 70.0 18 77.8 23 SpatialVLA [41] Linguistics 88.2 18 89.9 22 78.6 24 55.5 23 78.1 22 ThinkAct [17] Linguistics 88.3 17 91.4 21 87.1 18 70.9 17 84.4 18 π0-FAST [39] Linguistics 96.4 12 96.8 14 88.6 16 60.2 20 85.5 17 FPC-VLA [52] Linguistics 87.0 21 92.0 19 86.2 19 82.2 15 86.9 16 SmolVLA [43] Linguistics 93.0 16 94.0 16 91.0 14 77.0 16 88.8 15 GR00T-N1 [4] Linguistics 94.4 15 97.6 12 93.0 13 90.6 11 93.9 13 π0 [5] Linguistics 96.8 11 98.8 4 95.8 9 85.2 14 94.1 12 GO-1 [6] Linguistics 96.2 13 97.8 10 96.0 8 89.2 13 94.8 11 DD-VLA [28] Linguistics 97.2 10 98.6 6 97.4 5 92.0 9 96.3 8 MemoryVLA [42] Linguistics 98.4 4 98.4 7 96.4 7 93.4 7 96.7 6 π0.5 [22] Linguistics 98.8 2 98.2 9 98.0 3 92.4 8 96.9 5 OpenVLA-OFT [25] Linguistics 97.6 8 98.4 7 97.9 4 94.5 4 97.1 4 VLA-Adapter [50] Linguistics 97.8 7 99.2 2 97.2 6 95.0 3 97.3 3 Ours⋄ Action 99.4 1 99.6 1 98.8 2 96.0 2 98.5 1 Ours Action 98.6 3 99.0 3 99.4 1 97.0 1 98.5 1

- Table 1. Comparison on the LIBERO benchmark. Our proposed approach is trained on the LIBERO dataset. ⋄ represents that the LLM backbone is frozen during training. All metrics are average success rates (%). The best results are highlighted in bold.

### 4. Experiments

instance, explicit priors provide kinematic cues, whereas implicit priors capture latent action tendencies. Hence, to effectively combine these complementary guidance, we concatenate the two attended features and process them through self-attention fusion block, which integrates the priors into a unified representation h¯:

In this section, we first outline the experimental setup in Sec. 4.1. Then, in Sec. 4.2, we evaluate our approach on three simulation benchmarks, followed by comprehensive ablation studies in Sec. 4.3. Moreover, we present realworld deployment results in Sec. 4.4 to evaluate real-world applicability.

h¯ = Self-Attn([Sex; Sim]). (11)

Eventually, the aggregated representation h¯ is fed into πθhead, which predicts the denoised action sequence at:t+H−1.

Training Objectives. The entire framework is optimized under a standard flow-matching mean-squared error (MSE) objective. The training losses consist of two parts, i.e., flowmatching MSE for both Explicit Action Reasoner πθref and action head πθhead, denoted as Lπθref and Lπθhead, respectively. Hence, the overall objective is:

Ltotal = λ1Lπθref + λ2Lπθhead, (12) where λ1 and λ2 are balance factors.

Teacher Forcing Stabilization. During training, the outputs of πθref can be unstable. To stabilize optimization, we compute Zex directly from ground-truth reference trajectories instead of from πθref predictions, preventing optimization interference to πθhead. During inference, the model switches to a fully self-conditioned mode, where πθref autonomously generates the reference actions to guide πθhead in action prediction.

#### 4.1. Experimental Setup

Data Sources. For simulation experiments, we strictly follow the official training splits provided by the corresponding benchmark (LIBERO [32], LIBERO-Plus [15], and VLABench [59]), and train our models exclusively on their standard demonstration datasets without introducing any additional data. For the real-world setting, all demonstrations used for model training are collected on our own robotic platform. More details about data sources are introduced in Appendix A.

Implementation Details. We implement our approach upon π0.5 [22]. Specifically, we adopt SigLIP [56] as the visual encoder, while the LLM backbone is instantiated as Gemma 2B architecture [3] with N = 18 layers and hidden size d = 2048. For frame processing, each input frame is resized to 224 × 224 prior to the visual encoder. Regarding the EAR, we employ a compact Transformer-based design composed of N = 18 layers. Concerning the IAR, each learnable query matrix Qi is configured with a row dimen-

Methods Guidance Camera Robot Language Light Background Noise Layout Avg. Zero-Shot Transfer

WorldVLA [8] Visual 0.1 27.9 41.6 43.7 17.1 10.9 38.0 25.0 OpenVLA [24] Linguistics 0.8 3.5 23.0 8.1 34.8 15.2 28.5 15.6 NORA [21] Linguistics 2.2 37.0 65.1 45.7 58.6 12.8 62.1 39.0 UniVLA [7] Linguistics 1.8 46.2 69.6 69.0 81.0 21.2 31.9 42.9 π0-Fast [39] Linguistics 65.1 21.6 61.0 73.2 73.2 74.4 68.8 61.6 RIPT-VLA [44] Linguistics 55.2 31.2 77.6 88.4 91.6 73.5 74.2 68.4 OpenVLA-OFT [25] Linguistics 56.4 31.9 79.5 88.7 93.3 75.8 74.2 69.6 π0∗ [5] Linguistics 61.0 40.8 63.5 89.3 84.1 80.1 76.4 69.4 π0∗.5 [22] Linguistics 75.8 79.4 83.3 95.5 95.0 89.6 87.0 85.7 Ours⋄ Action 68.9 80.3 84.1 95.6 93.1 81.5 88.3 83.6 Ours Action 72.6 82.6 87.5 97.7 96.5 87.8 88.1 86.6

Supervised Fine-Tuning

π0⋄ [5] Linguistics 79.6 21.1 72.5 84.7 86.2 68.3 69.4 67.4 π0⋄.5 [22] Linguistics 70.3 41.7 81.1 97.3 94.6 71.8 84.9 75.7 Ours⋄ Action 91.2 62.5 80.3 95.1 91.5 88.3 84.9 84.1 Ours Action 96.6 70.4 79.7 95.1 97.1 95.9 85.0 88.0

- Table 2. Comparison on the LIBERO-Plus benchmark. Methods under Zero-Shot Transfer are trained on LIBERO dataset and directly evaluated on LIBERO-Plus. Supervised Fine-Tuning denotes models trained on the LIBERO-Plus training set. An asterisk (*) denotes results reproduced by utilizing officially released checkpoints, while ⋄ represents that the LLM backbone is frozen during training. The best results are highlighted in bold.

sion of M = 1. The reduced dimension in the downsampling strategy is set to d′ = 128.

In terms of model training, unless explicitly specified, the horizon of predicted reference actions Href and action policy output H are fixed to 15 and 10, with action shift set to 2 and 1, respectively. To clarify, the action shift specifies the temporal interval relative to the expert demonstration. For instance, a shift of 1 yields frame-aligned predictions, whereas a shift of 2 skips one intermediate frame. We set the balance factors in training losses as λ1 = λ2 = 0.5.

Training Configuration. We adopt a unified set of training hyperparameters across all experiments unless explicitly specified. Concretely, the learning rate follows a cosinedecay schedule with a warm-up phase of 10K steps, a peak learning rate of 5e−5, and a decay toward 5e−5 over 10K steps. Optimization is performed with AdamW with gradient-norm clipping set to 1.0. An exponential moving average (EMA) of model parameters is maintained with a decay rate of 0.999. Regarding hardware settings, model training is performed on a single node equipped with 8 NVIDIA H100 GPUs using bfloat16 precision. And the inference is conducted on a single NVIDIA RTX 4090.

#### 4.2. Simulation Experiments

In this section, we conduct the simulation evaluations across three benchmarks, i.e., LIBERO [32], LIBERO-Plus [15], and VLABench [59], to comprehensively evaluate our approach’s performance and generalization capabilities under diverse task structures.

LIBERO Benchmark. We evaluate our approach on LIBERO benchmark, which targets four distinct robot ca-

pabilities: spatial awareness (Spatial), object manipulation (Object), goal completion (Goal), and long-horizon reasoning (Long). Each task suite consists of 10 tasks and provides 50 human-teleoperated demonstrations per task for policy training. The evaluation is conducted following the official evaluation protocol. For each task, the policy is evaluated over 50 trials, amounting to 2,000 total rollouts.

As reported in Table 1, the quantitative evaluation results demonstrate that our proposed approach outperforms existing methods across all tracks. Compared to previous stateof-the-art method π0.5, our approach achieves a 1.6% absolute improvement in average. Notably, we observe a pronounced improvement on LIBERO-Long suite, where tasks require long-horizon manipulation with strict error control. Particularly, unlike Language- or Visual-CoT, whose intermediate reasoning remains abstract or indirect with respect to action execution, our proposed ACoT naturally operates in precise representation. Through leveraging actions as intermediate reasoning, the model feeds the action head with structured action guidance, which significantly enhances the robustness in long-horizon manipulation tasks.

LIBERO-Plus Benchmark. Built upon LIBERO, LIBERO-Plus is designed to systematically evaluate robotic policies under controlled distribution shifts. Concretely, LIBERO-Plus introduces 7 perturbation dimensions, i.e., camera-viewpoints (Camera), robot-initialstates (Robot), language-variations (Language), lightingconditions (Light), background-textures (Background), sensor-noise (Noise) and object-layout (Layout), exposing hidden failure modes under standard evaluations. Notably, LIBERO-Plus consists of 10,030 evaluation episodes, pro-

In-dist. Category Commonsense Instruction Texture Avg. IS ↑ PS ↑ IS ↑ PS ↑ IS ↑ PS ↑ IS ↑ PS ↑ IS ↑ PS ↑ IS ↑ PS ↑

Methods Guidance

π0⋄ [5] Linguistics 67.8 62.7 44.0 33.6 54.9 43.0 58.0 38.7 50.6 42.5 55.0 44.1 π0⋄.5 [22] Linguistics 75.0 60.8 49.6 35.3 57.5 41.6 57.1 30.3 62.0 47.4 60.2 43.1

Ours⋄ Action 79.8 66.1 54.1 38.9 52.3 37.8 56.8 39.6 74.6 54.6 63.5 47.4

- Table 3. Comparison on the VLABench benchmark. IS and PS represent Intention score and Progress score, respectively. All models are trained for 60K steps. ⋄ indicates that the LLM backbone is frozen during training. The best results are highlighted in bold.

Name EAR IAR Spatial Object Goal Long Avg. Baseline 98.8 98.2 98.0 92.4 96.9

- #1 ✓ 99.0 99.4 98.0 96.6 98.3

- #2 ✓ 99.2 99.2 98.2 95.6 98.1

- #3 ✓ ✓ 99.4 99.6 98.8 96.0 98.5

- Table 4. Module ablations. The performance is gradually improved with the continuous addition of proposed methods.

Action shift

Action horizon

Equi. horizon

Spatial Object Goal Long Avg. Baseline 1 10 10 98.6 99.0 96.4 92.2 96.6

Name

- 1 10 10 99.4 99.4 98.8 95.0 98.2

- 2 5 10 99.6 99.6 98.4 94.4 98.0

- 1 30 30 99.2 99.2 97.6 95.6 97.9

+EAR 2 15 30 99.0 99.4 98.0 96.6 98.3

- 2 30 60 99.4 99.0 98.2 95.0 97.9

- 3 30 90 98.8 99.4 97.4 96.2 98.0

viding statistically reliable evaluation.

- Table 5. Reference action parameter ablation. We observe that different reference-action configurations within EAR generally lead to performance improvements.

Methods Spatial Object Goal Long Avg. Baseline 98.8 98.2 98.0 92.4 96.9

Query 98.8 99.0 97.2 92.8 97.0 Attention Pooling 99.4 98.6 98.2 92.8 97.3

Downsample 99.2 99.2 98.2 95.6 98.1

- Table 6. Comparison of KV-cache interaction strategies in IAR.

We consider two evaluation protocols: (i) Zero-Shot Transfer, where policies trained on the LIBERO dataset are directly evaluated on LIBERO-Plus to assess generalization. (ii) Supervised Fine-Tuning, where models are directly optimized on LIBERO-Plus dataset. Technically, we follow evaluation configuration in LIBERO-Plus [15], i.e., each episode is executed once without repeated rollouts.

As shown in Table 2, our method significantly boosts the policy’s performance, surpassing all previous methods in both settings. Specifically, under the Zero-Shot regime, our approach demonstrates pronounced robustness against distribution shifts such as robot initial-state perturbations (+3.2%) and language variations (+4.2%), where existing language-guided or vision-guided policies exhibit significant degradation. Furthermore, our method maintains exceptional performance under the Supervised Fine-Tuning setting, reaching an 88.0% average success rate. These results highlight the effectiveness of our action-space reasoning in improving generalization and robust policy learning. VLABench Benchmark. Built on ManiSkill3 [45], VLABench is designed to benchmark both VLAs and VLMs on diverse robotic tasks. The standard evaluation is organized into 5 public tracks, i.e., in-distribution, cross-category (category-level generalization), commonsense reasoning, semantic-instruction (language understanding), and unseen-texture (appearance robustness). Particularly, VLABench proposes Intention Score (IS) and Progress Score (PS) to evaluate robot policies.

#### 4.3. Ablation Study

We examine each component’s contribution via systematic ablation experiments on the LIBERO benchmark, which are shown in Table 4, Table 5, and Table 6. Note that we adopt π0.5 as the “Baseline” method. More ablations in different benchmarks are in Appendix C.

EAR. As shown in Table 4, compared with the baseline, the experiment “#1” introduces the Explicit Action Reasoner (EAR) module into policy learning, which lifts the average success rate from 96.9% to 98.3%, demonstrating that the explicit action-space guidance benefits the robotic action sequence prediction. A plausible explanation is that EAR introduces an intermediate reference action sequence, which injects strong inductive bias on the behavior and thereby reduces ambiguity in mapping from observations to actions.

IAR. Analogously, with the Implicit Action Reasoner (IAR) module added in “#2”, the average success rate increases from 96.9% to 98.1%. This gain suggests that exploiting the implicit action distribution encoded in vision–language representations can also provide effective guidance for policy learning. This performance gain can be partly attributed to the fact that IAR distills action-related clues implicitly encoded within the vision–language backbone, which potentially reflects the distribution of feasible actions. Such priors encourages the policy to remain closer to coherent, task-consistent behavioral patterns.

In our context, we train π0, π0.5, along with our method in a unified training setup on VLABench’s official training data. We present quantitative results in Table 3. Overall, our method achieves the best performance across both IS (63.5%) and PS (47.4%). Notably, under the challenging unseen-texture track, it delivers substantial gains, i.e., +12.6% in IS and +7.2% in PS, indicating strong robustness to distribution shifts. Together, these results further confirm the effectiveness of our proposed approach.

EAR + IAR. In Table 4, experiment “#3” incorporates both

Wipe the stain on the table

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Pour water into the cup

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Pick up the blue doll

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Figure 4. Evaluation results of real-world experiments.

Stain”, “Pour Water”, and “Open-set Pick”, which respectively assess contact-rich manipulation, fine-grained object handling, and instruction-following abilities.

- Figure 3. Visualization of three manipulation tasks in real world.

EAR and IAR, achieving the highest average success rate of 98.5%. The consistent improvements demonstrate that explicit action guidance and implicit action cues extracted from VLM’s key-value cache are complementary, jointly providing stronger guidance for accurate action prediction.

Reference Action Configurations. To further examine the effect of explicit action references in EAR, we investigate different settings of action shift and action horizon, as summarized in Table 5. We observe various parameter combinations consistently bring improvements over the baseline, indicating that providing action cues is broadly beneficial for policy learning. Besides, we find that shorter horizons combined with moderate shifts tend to produce relatively stronger gains. These observations offer further insight into how explicit action guidance influence policy learning.

KV-cache Interaction Strategies. We compare three strategies for extracting action-relevant cues from VLM’s key-value cache within IAR module, as presented in Table 6. Concretely, “Query” method uses learnable queries to attend to VLM’s original key-value cache. “Attention Pooling” method forms a pooled query by averaging key-value cache and then applies cross-attention operation. “Downsample” method first downsamples VLM’s key-value cache and then aggregates them using learnable matrix.

As shown in Table 6, all three variants outperform the baseline, indicating that extracting implicit action cues from VLM benefits policy learning. Notably, the “Downsample” strategy achieves the best performance, suggesting that VLM’s features may contain noisy information for action prediction. This also highlights the importance of designing appropriate interaction mechanisms to align visionlanguage and action.

- 4.4. Real-World Deployment

Specifically, as visualized in Fig. 3, (i) the “Wipe Stain” task requires the robot to pick up a sponge from the table and wipe away the stain until the surface is clean. (ii) The “Pour Water” task requires the robot to grasp the kettle by its handle, locate the target cup, pour water into it without causing overflow, and finally return the kettle to the table in a stable manner. (iii) The “Open-set Pick” task instructs the robot to pick up the correct tabletop object according to given natural-language command. Additionally, to examine the cross-embodiment adaptability, we also perform the “Open-set Pick” task on the AgileX robotic platform. More details about training and evaluation are provided in the supplementary material.

As shown in Fig. 4, our approach achieves consistently higher average success rates than both π0.5 and π0, i.e., 66.7% against 61.0% and 33.8%. These results demonstrate that the proposed framework maintains effectiveness under real-world sensing conditions. Moreover, the aligned improvements observed on both AgiBot G1 and AgileX also indicate that our method exhibits adaptability across different robotic embodiments.

### 5. Conclusion

In this work, we address the fundamental semantickinematic gap in modern robotic policies by proposing a new paradigm: Action Chain-of-Thought (ACoT). We argue that for physically grounded intelligence, deliberation should occur not in the abstract space of language or vision, but directly in the kinematically grounded space of actions. We materialize this concept in our ACoT-VLA framework, which leverages two synergistic modules—an Explicit Action Reasoner (EAR) and an Implicit Action Reasoner (IAR)—to generate and fuse both explicit trajectory plans and implicit behavioral priors. This action-centric guidance mechanism creates a direct, information-rich conduit between high-level intent and low-level motor control. Extensive experiments across multiple simulation and real-

To further validate the effectiveness of our framework, we conduct extensive real-world experiments on the AgiBot G1 robot. We consider three manipulation tasks, i.e., “Wipe

world benchmarks demonstrate that our proposed approach yields state-of-the-art performance. Through shifting the locus of reasoning from perception to action, our work not only provides a more effective and grounded method for robotic policy learning, but also opens a new avenue for research into more structured, interpretable, and capable embodied agents. We believe that learning to “think” in the language of actions is a critical step towards developing the next generation of generalist robots.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1
- [3] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 5
- [4] Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 5
- [5] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy

Groom, Karol Hausman, Brian Ichter, et al. π0: A visionlanguage-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164. 1, 2, 5, 6, 7

- [6] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025. 2, 5
- [7] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions, 2025. 6
- [8] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 3, 5, 6
- [9] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. arXiv preprint arXiv:2208.04202, 2022. 4
- [10] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization

- for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025. 2
- [11] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44 (10-11):1684–1704, 2025. 1, 5
- [12] Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Wenhao Zhang, et al. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. arXiv preprint arXiv:2505.03233, 2025. 2
- [13] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. 2023. 4
- [14] Jiafei Duan, Wentao Yuan, Wilbert Pumacay, Yi Ru Wang, Kiana Ehsani, Dieter Fox, and Ranjay Krishna. Manipulateanything: Automating real-world robots using visionlanguage models. arXiv preprint arXiv:2406.18915, 2024. 2
- [15] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. Libero-plus: In-depth robustness analysis of visionlanguage-action models. arXiv preprint arXiv:2510.13626,

2025. 5, 6, 7, 12

- [16] David Ha and J¨urgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3), 2018. 1
- [17] Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, YuChiang Frank Wang, and Fu-En Yang. Thinkact: Visionlanguage-action reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815, 2025. 5
- [18] Siyuan Huang, Haonan Chang, Yuhan Liu, Yimeng Zhu, Hao Dong, Peng Gao, Abdeslam Boularias, and Hongsheng Li. A3vlm: Actionable articulation-aware vision language model. arXiv preprint arXiv:2406.07549, 2024. 2
- [19] Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024. 2
- [20] Wenhui Huang, Changhe Chen, Han Qi, Chen Lv, Yilun Du, and Heng Yang. Motvla: A vision-language-action model with unified fast-slow reasoning. arXiv preprint arXiv:2510.18337, 2025. 2
- [21] Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U Tan, Navonil Majumder, Soujanya Poria, et al. Nora: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854, 2025.

- 6

[22] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. 1, 2, 5, 6,

- 7, 13

- [23] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu,

and Hang Zhao. Galaxea open-world dataset and g0 dualsystem vla model. arXiv preprint arXiv:2509.00576, 2025. 2

- [24] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 1, 2, 5, 6
- [25] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. 5, 6
- [26] Hengtao Li, Pengxiang Ding, Runze Suo, Yihao Wang, Zirui Ge, Dongyuan Zang, Kexian Yu, Mingyang Sun, Hongyin Zhang, Donglin Wang, et al. Vla-rft: Vision-language-action reinforcement fine-tuning with verified rewards in world simulators. arXiv preprint arXiv:2510.00406, 2025. 3
- [27] Xiaoqi Li, Mingxu Zhang, Yiran Geng, Haoran Geng, Yuxing Long, Yan Shen, Renrui Zhang, Jiaming Liu, and Hao Dong. Manipllm: Embodied multimodal large language model for object-centric robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18061–18070, 2024. 2
- [28] Zhixuan Liang, Yizhuo Li, Tianshuo Yang, Chengyue Wu, Sitong Mao, Liuao Pei, Xiaokang Yang, Jiangmiao Pang, Yao Mu, and Ping Luo. Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies. arXiv preprint arXiv:2508.20072, 2025. 5
- [29] Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang, Shengcong Chen, Yuxin Jiang, Yue Hu, Jingbin Cai, Si Liu, Jianlan Luo, et al. Genie envisioner: A unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635, 2025. 3, 5
- [30] Tao Lin, Gen Li, Yilei Zhong, Yanwen Zou, Yuxin Du, Jiting Liu, Encheng Gu, and Bo Zhao. Evo-0: Vision-languageaction model with implicit spatial understanding. arXiv preprint arXiv:2507.00416, 2025. 2
- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [32] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 5, 6, 12
- [33] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024. 1, 2
- [34] Yunzhe Liu, Rinon Gal, Amit H Bermano, Baoquan Chen, and Daniel Cohen-Or. Self-conditioned generative adversarial networks for image editing. arXiv preprint arXiv:2202.04040, 2022. 4
- [35] Yi Liu, Sukai Wang, Dafeng Wei, Xiaowei Cai, Linqing Zhong, Jiange Yang, Guanghui Ren, Jinyu Zhang, Maoqing Yao, Chuankang Li, et al. Unified embodied vlm reasoning with robotic action via autoregressive discretized pretraining. arXiv preprint arXiv:2512.24125, 2025. 2

- [36] Qi Lv, Weijie Kong, Hao Li, Jia Zeng, Zherui Qiu, Delin Qu, Haoming Song, Qizhi Chen, Xiang Deng, and Jiangmiao Pang. F1: A vision-language-action model bridging understanding and generation to actions. arXiv preprint arXiv:2509.06951, 2025. 3, 5
- [37] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 2
- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [39] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for visionlanguage-action models. arXiv preprint arXiv:2501.09747,

2025. 5, 6

- [40] Shengyi Qian, Weifeng Chen, Min Bai, Xiong Zhou, Zhuowen Tu, and Li Erran Li. Affordancellm: Grounding affordance from vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7587–7597, 2024. 4
- [41] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025. 1, 2, 5
- [42] Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. Memoryvla: Perceptual-cognitive memory in vision-language-action models for robotic manipulation. arXiv preprint arXiv:2508.19236, 2025. 5
- [43] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844,

2025. 5

- [44] Shuhan Tan, Kairan Dou, Yue Zhao, and Philipp Kr¨ahenb¨uhl. Interactive post-training for vision-languageaction models. arXiv preprint arXiv:2505.17016, 2025. 6
- [45] Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse-kai Chan, et al. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. arXiv preprint arXiv:2410.00425, 2024. 7
- [46] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1
- [47] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch,

- Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. 2
- [48] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 1, 5
- [49] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. arXiv preprint arXiv:2506.19850, 2025. 5
- [50] Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. Vla-adapter: An effective paradigm for tiny-scale vision-language-action model. In Proceedings of the AAAI conference on artificial intelligence, pages 18638– 18646, 2026. 5
- [51] Ziyang Yan, Wenzhen Dong, Yihua Shao, Yuhang Lu, Haiyang Liu, Jingwen Liu, Haozhe Wang, Zhe Wang, Yan Wang, Fabio Remondino, et al. Renderworld: World model with self-supervised 3d label. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 6063–6070. IEEE, 2025. 1
- [52] Yifan Yang, Zhixiang Duan, Tianshi Xie, Fuyu Cao, Pinxi Shen, Peili Song, Piaopiao Jin, Guokang Sun, Shaoqing Xu, Yangwei You, et al. Fpc-vla: A vision-language-action framework with a supervisor for failure prediction and correction. arXiv preprint arXiv:2509.04018, 2025. 5
- [53] Chenghao Yin, Da Huang, Di Yang, Jichao Wang, Nanshu Zhao, Chen Xu, Wenjun Sun, Linjie Hou, Zhijun Li, Junhui Wu, et al. Genie sim 3.0: A high-fidelity comprehensive simulation platform for humanoid robot. arXiv preprint arXiv:2601.02078, 2026. 13, 14
- [54] Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Zhuoguang Chen, Tao Jiang, and Hang Zhao. Depthvla: Enhancing vision-language-action models with depth-aware spatial reasoning. arXiv preprint arXiv:2510.13375, 2025. 2
- [55] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024. 1
- [56] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 5
- [57] Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Yanpeng Zhou, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, Xingyue Quan, Hang Xu, et al. 4d-vla: Spatiotemporal vision-language-action pretraining with cross-scene calibration. arXiv preprint arXiv:2506.22242, 2025. 2
- [58] Jianke Zhang, Yanjiang Guo, Yucheng Hu, Xiaoyu Chen, Xiang Zhu, and Jianyu Chen. Up-vla: A unified understanding and prediction model for embodied agent. arXiv preprint arXiv:2501.18867, 2025. 3
- [59] Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, YuGang Jiang, et al. Vlabench: A large-scale benchmark

- for language-conditioned robotics manipulation with longhorizon reasoning tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11142– 11152, 2025. 2, 5, 6, 12
- [60] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, Fan Lu, He Wang, et al. Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025. 1, 3, 5
- [61] Zhengshen Zhang, Hao Li, Yalun Dai, Zhengbang Zhu, Lei Zhou, Chenchen Liu, Dong Wang, Francis EH Tay, Sijin Chen, Ziwei Liu, et al. From spatial to actions: Grounding vision-language-action model in spatial foundation priors. arXiv preprint arXiv:2510.17439, 2025. 2
- [62] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025. 3, 5
- [63] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023. 1
- [64] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024. 1
- [65] Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, YaQin Zhang, and Xianyuan Zhan. Universal actions for enhanced embodied foundation models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22508–22519, 2025. 5
- [66] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´e III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024. 5
- [67] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In European conference on computer vision, pages 55–72. Springer,

2024. 1

- [68] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 2
- [69] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 2

### A. Dataset Description

In this section, we present a comprehensive characterization of the benchmark datasets and the custom-collected data used for model training in our experiments. We systematically report key statistics, including the total number of episodes, frame counts, and other relevant properties, which is summarized in Table 7 below:

Type Dataset Embodiment DoF Episodes Frames FPS

LIBERO Franka 7 1,693 273,465 10 LIBERO-Plus Franka 7 14,347 2,238,036 20 VLABench Franka 7 4,713 528,398 10

Simulation

Wipe Stain AgiBot G1 22 177 356,316 30 Pour Water AgiBot G1 22 1,821 5,062,506 30

Real-World

Open-set Pick AgiBot G1 22 1,936 219,824 30 Open-set Pick AgileX 14 962 251,283 30

Table 7. Dataset statistics.

Simulation Benchmarks. We utilize three publicly released simulation datasets, i.e., LIBERO [32], LIBEROPlus [15], and VLABench [59]. Specifically, the LIBERO dataset contains 1,693 episodes and 273,465 frames, recorded at a fixed 10 Hz. Its demonstrations exhibit relatively uniform trajectory lengths and smooth motion patterns, making it widely adopted benchmark in community.

However, due to the increasing performance saturation observed on LIBERO, LIBERO-Plus is recently introduced to provide a more challenging and diversified evaluation setting. LIBERO-Plus provides 14,347 episodes and 2,238,036 frames, captured at 20 Hz. In contrast to the homogeneous trajectories in LIBERO, LIBERO-Plus explicitly emphasizes a perturbation-oriented design. The demonstrations display substantially larger variations in motion magnitude and camera–robot viewpoint configuration. These characteristics make it a more suitable benchmark for evaluating policy generalization under structured distribution shifts.

Besides these two datasets, we further benchmark our method on VLABench, whose training set includes 4,713 episodes and 528,398 frames, recorded at 10 Hz, which requires a higher level of visual and physical understanding from the policy.

Real-World Experiment. For real-world deployment, we collect demonstrations across 3 tasks, i.e., Wipe Stain, Pour Water, and Open-set Pick, as shown in Table 7.

The “Wipe Stain” dataset contains 177 episodes with 356,316 frames, characterized by dense tool–surface contact and fine-grained force control. The “Pour Water” dataset includes 1,821 episodes and 5,062,506 frames. Its large scale stems from the task’s long-horizon and multistage nature. Regarding the “Open-set Pick” task, the AgiBot G1 subset provides 1,936 episodes with 219,824 frames, while the AgileX subset offers 962 episodes with 251,283 frames, both featuring diverse tabletop layouts and natural-language instructions.

Task Action Space Action Horizon State Batch Size Training Step LIBERO Delta EEF 10 × 128 40K LIBERO-Plus Delta EEF 10 × 128 100K VLABench Abs EEF 10 ✓ 128 60K Wipe Stain Abs Joint 30 ✓ 128 50K Pour Water Abs Joint 30 ✓ 128 240K Open-set Pick Abs Joint 30 ✓ 128 50K Open-set Pick† Abs Joint 30 ✓ 128 50K

Table 8. Training details. Note that the “Open-set Pick†” task is performed on AgileX platform.

### B. Training & Evaluation Details

Training Details. We describe the task-specific training configurations, e.g., action space and state usage, for better understanding.

As presented in Table 8, for the LIBERO and LIBEROPlus suites, the policy is trained using delta end-effector control (Delta EEF) with an action horizon of 10 steps. In particular, no privileged state information is provided during training. We utilize a global batch size of 128 and train the policies for 40K and 100K steps, respectively. Similarly, we train our models in VLABench for 60K steps, while adopting state input and absolute end-effector (Abs EEF) actions to align the benchmark’s control convention.

In terms of the real-world tasks, we utilize Abs Joint control with a longer action horizon of 30. Unlike the simulator benchmarks, these tasks additionally provide structured robot state observations to improve robustness under realworld sensing and actuation noise. Our models are trained for 50K, 240K, and 50K steps, in “Wipe Stain”, “Pour Water”, and “Open-set Pick” tasks, respectively, with same batch size of 128.

Evaluation Details. Next, we illustrate the evaluation protocols and success criteria for all real-world tasks. Each task is assessed using fixed and repeatable initializations to ensure reproducibility and reduce environmental variance.

Concretely, in terms of the “Wipe Stain” task, we predefine three initial sponge poses. For each pose, the robot is required to clean stains placed at four distinct table locations. Every configuration is executed twice, resulting in 24 trials in total. A trial is considered successful if the robot grasps the sponge and removes the stain from the specified location.

As for the “Pour Water”, we standardize six predefined relative configurations between the bottle and the glass. Then, each configuration is executed two times. A trial is counted as successful if the robot lifts the bottle, pours water into the cup, and places the bottle back onto the coaster. Note that minor spillage of water when pouring is allowed.

Eventually, regarding the “Open-set Pick” task, we initialize ten object arrangements on the table, containing both in-distribution and out-of-distribution instances. In each arrangement, the robot is instructed to grasp a specified target object using either its left or right arm, as indicated by the

Name EAR IAR Camera Robot Language Light Background Noise Layout Avg. Baseline 70.3 41.7 81.1 97.3 94.6 71.8 84.9 75.7

- #1 ✓ 88.7 63.5 80.4 94.0 90.2 89.5 84.2 83.7

- #2 ✓ 80.7 48.7 82.6 97.7 90.9 84.3 86.0 80.4

- #3 ✓ ✓ 91.2 62.5 80.3 95.1 91.5 88.3 84.9 84.1

- Table 9. Module ablations on LIBERO-Plus benchmark. The performance is gradually improved with the addition of proposed methods. Note that models are directly optimized on LIBERO-Plus dataset, with the LLM backbone frozen during training.

Name

Action Head EAR LIBERO LIBERO-Plus

Param. Denoise Param. Denoise Spatial Object Goal Long Avg. Camera Robot Language Light Background Noise Layout Avg. Baseline 300M 10 - - 98.6 99.0 96.4 92.2 96.6 70.3 41.7 81.1 97.3 94.6 71.8 84.9 75.7

- #1 600M 10 - - 97.6 98.4 97.8 96.4 97.6 68.7 44.8 83.1 96.4 92.7 66.6 84.1 74.9

- #2 600M 20 - - 97.8 98.8 98.0 95.2 97.5 70.0 44.8 82.7 97.6 93.1 66.7 83.2 75.1

- #3 300M 5 300M 5 98.6 99.6 97.8 95.4 97.9 88.2 62.4 81.5 95.0 91.5 88.6 85.3 83.9

- #4 300M 10 300M 10 99.0 99.4 98.0 96.6 98.3 88.7 63.5 80.4 94.0 90.2 89.5 84.2 83.7

- #4 300M 10 300M 10 99.0 99.4 98.0 96.6 98.3 88.7 63.5 80.4 94.0 90.2 89.5 84.2 83.7

- #5 300M 10 150M 10 99.2 99.2 97.8 94.2 97.6 86.4 54.3 81.7 92.2 91.4 89.1 82.1 81.7

- #6 300M 10 250M 10 99.0 98.2 98.6 94.2 97.5 87.2 59.7 81.1 95.0 93.7 87.4 83.5 83.1

- #7 300M 10 500M 10 98.4 99.4 96.6 94.2 97.0 80.8 57.6 84.1 95.6 92.1 79.8 83.7 80.9

- Table 10. Effects of parameters and denoise steps on policy performance. Note that the IAR module is not added in this experiment. The evaluation protocol in LIBERO-Plus is Supervised Fine-Tuning, i.e., models are directly optimized on LIBERO-Plus dataset. The LLM backbone is frozen during training. The best results are highlighted in bold, and the second-best results are underlined.

instruction. Each arm–object pair is evaluated twice, resulting in 40 trials overall. A trial is deemed successful if the robot grasps the instructed object with the correct arm.

Across all tasks, evaluations are carried out by trained operators with substantial prior testing experience, and success rates are computed as the proportion of successful trials relative to the total number of executed attempts.

### C. More Experimental Results

In this section, we provide additional quantitative experiments to substantiate the effectiveness of our proposed approach and to empirically uncover several insightful phenomena. Specifically, the experimental analyses comprise four parts: (1) ablation study conducted on the LIBEROPlus benchmark in Table 9, (2) an investigation of how the parameter sizes of the Action Head and Explicit Action Reasoner (EAR), as well as the number of denoising steps, influence policy performance in Table 10, (3) evaluation of proposed approach’s sim2real capability based on GenieSim 3.0 [53] in Table 11, and (4) a comparative study examining the relationship among inference latency, model size, and performance in Table 12. Note that we adopt π0.5 [22] as the baseline method, denoted as “Baseline”.

Module Ablation. As shown in Table 9, incorporating the proposed reasoning modules consistently improves policy performance on the LIBERO-Plus benchmark. Adding

the EAR module, i.e., experiment “#1”, yields a clear gain over the baseline, increasing the average success rate from 75.7% to 83.7%. This improvement can be attributed to EAR’s ability to generate an explicit reference action trajectory, which significantly reduces the ambiguity in mapping complex visual or linguistic observations to low-level actions, such as camera shifts and background changes. Meanwhile, incorporating only the IAR (“#2”) also improves the performance from 75.7% to 80.4%, indicating that decoding the latent action-related semantics within the vision–language backbone provides useful behavioral priors. Finally, combining EAR and IAR (“#3”) achieves the highest success rate of 84.1%, demonstrating their complementary effects, i.e., EAR provides explicit motion guidance, while IAR supplies dense representation-level priors.

Effect of Model Scaling & Denoising Budget. Then, we analyze the superiority of our method by comparing settings with matched total model parameters and denoising steps. As shown in Table 10, firstly, we enlarge the model size of the action head and increase the number of denoising steps in experiments “#1” and “#2”, to construct fair baselines for subsequent comparison. We observe a preliminary observation, i.e., increasing the model size or denoising steps does not reliably enhance performance. Specifically, compared with the baseline, while “#1” improves performance on the LIBERO benchmark, it simultaneously drops on LIBEROPlus. Next, comparing “#1” and “#2” reveals that further in-

Simulation Real-World π0.5 Ours π0.5 Ours

Tasks

Select Color 86.0 98.8 85.0 94.0 Recognize Size 93.0 96.0 94.0 94.0 Grasp Targets 71.7 68.0 70.8 75.0 Organize Objects 52.0 74.0 60.0 68.4

Avg. 75.7 84.2 77.5 82.9

- Table 11. Experimental results on Genie Sim 3.0 Simulation and Real-world Transfer. The best results are highlighted in bold.

creasing denoising steps yields only negligible fluctuations.

Subsequently, we incorporate the EAR module under fully matched overall parameterization and denoising budgets. Concretely, in both comparison pairs, “#1” with “#3” and “#2” with “#4”, we consistently observe notable performance improvements on both benchmarks, once the EAR module is introduced. This indicates that the performance gains originate from our proposed action chain-of-thought. The proposed mechanism supplies explicit reference actions that effectively mitigate the intrinsic instability of action prediction, especially under challenging external perturbations, as shown in the LIBERO-Plus, enabling a more reliable and grounded generalist robotic policy.

Effect of EAR Scale. Moreover, we investigate how various scale of the EAR module influences action prediction fidelity. To isolate the effect of EAR, we keep the action head parameters and the denoising schedule strictly fixed, while scaling the EAR module to 150M, 250M, 300M, and 500M parameters via adjusting hidden size. As presented in Table 10, through the comparison across experiments “#4”, “#5”, “#6”, and “#7”, we find that although all EARequipped variants outperform non-EAR baselines on both benchmarks, the performance trend is non-monotonic. Applying moderate EAR scales, e.g., 300M, yields the greatest improvement. Particularly, as evidenced in “#7” in Table 10, when the parameter of EAR module even exceeds that of the action head, we observe a marked drop in performance. We attribute this degradation to the tendency of an over-parameterized EAR to overfit spurious correlations during training. Therefore, it generates reference action trajectories that are systematically biased, which ultimately misdirect the action head toward suboptimal predictions.

Sim To Real Evaluation. To further assess the sim2real transferability of proposed method, we conduct evaluations on the Genie-Sim 3.0 benchmark [53]. It comprises 4 challenging tasks: picking objects based on specified colors, sizes, or categories, as well as a complex tabletop organization task. Technically, the model is trained on the officially released simulation-based datasets. Then, it is deployed and evaluated in both simulation and real-world environments.

As illustrated in Table 11, our approach consistently outperforms the baseline across both domains. Specifically, it achieves 84.2% in simulation and 82.9% in real-world

LIBERO LIBERO-Plus

Name EAR IAR Param. Latency

Avg. SR Avg. SR Baseline 3.35B 91ms 96.9 75.7

- #1 ✓ 3.80B 110ms 98.3 83.7

- #2 ✓ 3.36B 93ms 98.1 80.4

- #3 ✓ ✓ 3.81B 112ms 98.5 84.1

Table 12. Ablation experiment on model efficiency and performance. Note that the evaluation protocol in LIBERO-Plus is Supervised Fine-Tuning.

settings, representing absolute improvements of 8.5% and 5.4%, respectively. Notably, our method exhibits minimal performance degradation during the sim-to-real transition, which we attribute to the fundamental nature of ACoT paradigm. Specifically, while visual domain gap persists between synthetic and real-world observations, the underlying kinematically grounded action guidance remains consistent. By shifting the locus of reasoning from the perceptual space to the action space, our model extracts task-relevant motion priors that are invariant to low-level visual perturbations, effectively enhancing policy’s sim2real capability.

Latency Analysis. In Table 12, we further examine the inference efficiency of our approach in terms of both parameter count and end-to-end latency. As additional reasoning modules are introduced, we observe a slight increase in latency. Incorporating the EAR module raises latency from 91ms to 110ms, while adding the IAR module introduces only an additional 2ms. However, this marginal overhead is outweighed by the substantial improvement, which reflects a favorable trade-off.

### D. Limitations & Future Works

In this section, we discuss the limitations existing in our work and promising directions for future research.

Although our proposed action chain-of-thought (ACoT) substantially boosts policy performance, our framework still exhibits several constraints. The reasoning modules introduce additional computational cost, which, while relatively modest compared to the performance gains, may pose challenges for deployment on resource-constrained robotic platforms. Besides, another limitation stems from the fact that the prevailing action representation in the community is implemented as action chunks, i.e., sequences of lowlevel control commands such as joint angles or end-effector poses. While such representations faithfully encode the executed motions, they lack explicit geometric structure that would facilitate higher-level spatial reasoning, such as object-centric coordination and contact geometry. Hence, the potential of ACoT reasoning may not be fully unleashed. Enriching action representations with spatially grounded information to enable ACoT to operate in geometrically interpretable 3D space, constitutes an interesting and promising avenue for future exploration.

### E. LLM Usage Statement

In this paper, we employ Large Language Models (LLMs) solely for minor linguistic refinement during the manuscript preparation stage, such as correcting grammatical errors. None of the technical content, implementation details, or experimental results were generated by LLMs.

