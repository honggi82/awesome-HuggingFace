## A Mechanistic View on Video Generation as World Models: State and Dynamics

Luozhou Wang∗, Zhifei Chen∗, Yihua Du, Dongyu Yan, Wenhang Ge, Guibao Shen, Xinli Xu, Leyi Wu, Man Chen, Tianshuo Xu, Peiran Ren, Xin Tao, Pengfei Wan, and Ying-Cong Chen†

### arXiv:2601.17067v1[cs.CV]22Jan2026

Abstract—Large-scale video generation models have demonstrated emergent physical coherence, positioning them as potential world models. However, a gap remains between contemporary “stateless” video architectures and classic state-centric world model theories. This work bridges this gap by proposing a novel taxonomy centered on two pillars: State Construction and Dynamics Modeling. We categorize state construction into implicit paradigms (context management) and explicit paradigms (latent compression), while dynamics modeling is analyzed through knowledge integration and architectural reformulation. Furthermore, we advocate for a transition in evaluation from visual fidelity to functional benchmarks, testing physical persistence and causal reasoning. We conclude by identifying two critical frontiers: enhancing persistence via data-driven memory and compressed fidelity, and advancing causality through latent factor decoupling and reasoning-prior integration. By addressing these challenges, the field can evolve from generating visually plausible videos to building robust, general-purpose world simulators.

Index Terms—World Models, Video Generation, Generative AI, Physical Simulation, Diffusion Models, Foundation Models.

✦

1 INTRODUCTION

# T

HE field of video generation has witnessed a paradigm shift in recent years. Driven by the development of

large-scale video diffusion transformer models [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], the capability of generative models has evolved from producing short, lowresolution clips to rendering cinematic-quality sequences with unprecedented temporal consistency. Leading models such as Sora [14], Veo [20], Kling [21], Wan [7], and Gen3 [26] have demonstrated implications that extend far beyond mere visual fidelity. These models exhibit emergent physical coherence. For example, they learned to respect gravity, understand collision dynamics, and recognize object permanence without explicit physical instruction. As a result, video generation models are increasingly discussed not merely as content creation tools, but as potential world models [14], [20], [27], [28], [29], [30], [31], [32] that appear to simulate the physical evolution of the environment.

To critically assess this potential, it is necessary to revisit the theoretical origins of the concept of the “world model.” The evolution of world models has spanned multiple epochs—moving from cognitive science and control theory to the modern era of deep learning. The term has its roots in cognitive science, specifically in the theory of mental models [33], [34], [35]. These foundational works posited that intelligent organisms maintain a “small-scale model” of external reality to anticipate events, reason about

- • ∗ Luozhou Wang and Zhifei Chen contributed equally to this work.
- • L. Wang, Z. Chen, Y. Du, D. Yan, W. Ge, G. Shen, X. Xu, L. Wu, M. Chen, T. Xu, and Y.-C. Chen are with the Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China.
- • P. Ren is with Tongji University, Shanghai, China.
- • X. Tao and P. Wan are with Kuaishou Technology, Beijing, China.
- • Corresponding author: Ying-Cong Chen (yingcongchen@hkustgz.edu.cn).

counterfactuals, and simulate consequences before taking action.

Subsequently, in the field of control theory [36], [37], this internal simulation process was explicitly formalized through mathematical rules. This transition defined the concept of an internal state by manually selecting key variables to represent the environment. In this framework, a transition equation—derived from physical laws or established rules—is constructed to evolve the state during inference. In contrast an observation equation is used to map these states back to sensory inputs.

The concept has since evolved into the modern era of deep learning, with Model-Based Reinforcement Learning (MBRL) [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57] serving as one of the most representative methodologies. In these settings, the world model is a parameterized, data-driven system inextricably coupled with a policy agent in a closedloop interaction. By learning from data, the model captures environmental laws, allowing the agent to ”dream” or practice in a latent imagination space before execution in the real world.

All these discussions regarding world models can be synthesized into three core elements: observation, state, and dynamics. Observation refers to the perceptible modalities or raw data from the environment. The state, while lacking a singular universal definition, generally refers to a set of variables that provides a sufficient representation of the world by removing redundancies irrelevant to the current task. Dynamics is responsible for state transitions governed by underlying causal relationships.

Taking MBRL as a primary example, these approaches typically rely on an explicit State-Space Model (SSM). By constructing such explicit state representations, these models can achieve long-range reasoning with constant computational overhead. Furthermore, this structure facilitates

explicit causal decoupling, which effectively isolates internal state transitions from the influence of external control inputs.

However, video models operate very differently from the world models mentioned above. First, the training and inference of video models do not involve reinforcement learning or policy models. Instead, they work in an openloop manner, learning by passively observing large amounts of data [29], [30], [31], [32], [39]. Most video models [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25] use the transformer architecture, which models observation sequences directly. Because of its high parallelism and efficiency, this architecture scales easily with more data and computing power, leading to very strong emergent capabilities. However, these transformer-based models lack explicit state modeling. Because they do not construct a hidden state to represent the world, they must keep a very large context window during long-term reasoning. This creates a significant memory and computational burden as the video gets longer. Finally, the lack of an explicit state means that causal decoupling is not clearly achieved. Some video models [7], [12], [15], [16], [17], [58] use bidirectional attention, which causes them to lose the ability to perform causal inference along a sequence. In addition, external inputs, such as text prompts, can interfere with the model’s understanding of causality. It becomes unclear whether the model makes an apple fall because it understands the scene’s physical dynamics, or simply because the text prompt told it to.

Regarding the two aspects mentioned above—state and dynamics—this survey provides a detailed discussion on how current technical approaches address the challenges of causal reasoning and long-range inference in video models. Our core contributions are threefold:

- • From Video Model to World Model: We bridge the gap between high-level theory and practical implementation. Starting from the definition of a complete world model, we narrow our focus to the specific context of video generation, defining the position of video models within the broader framework of world models.
- • Novel Taxonomy: We propose a novel taxonomy based on the key constituents of a world model: State and Dynamics. For state, we categorize existing literature into two paradigms: Implicit State, where history is managed via context windows (e.g., KV Cache), and Explicit State, where history is compressed into compact variables akin to traditional state-space models. For dynamics, we categorize research into Causal Knowledge Integration, which combines video models with external, reasoningintensive models to improve inference, and Causal Architecture Reformulation, which redesigns the video model architecture to support causal reasoning inherently.
- • Evaluation and Future Work: We establish a threelevel classification for evaluating video models as world models. We summarize the evaluation of generated content across three stages: basic quality, physical persistence, and causality. Finally, we

highlight future directions, focusing on how video models can better solve the problems of persistence and causal consistency.

#### 2 RELATED SURVEY

Research on memory mechanisms for Large Language Models (LLMs) has transitioned from early explorations of incontext learning and knowledge editing [69], [70] to highly systematized, cognitive-inspired architectural theories. A seminal milestone in this evolution was the survey by Zhang et al. [71] on agentic memory, which established a taxonomy of memory sources, forms, and operations to enable self-evolving capabilities in agents. In the 2024- 2025 period, the field saw an explosion of systematic frameworks. For instance, the survey by Wu et al. [72] introduced the 3D-8Q taxonomy mapping human cognitive stages to AI memory quadrants across personal, system, and temporal dimensions. Similarly, the study by Jia et al. [73] delineated implicit, explicit, and agentic memory paradigms. In contrast, the work of Du et al. [74] defined the atomic dynamics of memory, such as consolidation and forgetting. Furthermore, recent surveys have addressed critical engineering and governance issues, including KV-cache optimization for long-context efficiency [75], memory governance for ”verifiable forgetting” [76], personalized preference learning [77], and multimodal retrieval-augmented generation [78].

Regarding the distinction between LLM (Text) and Video memory mechanisms: While text-based LLM memory primarily focuses on maintaining semantic consistency and factual accuracy across 1D sequences using RAG or KVcache management [69], [75], video memory must model high-dimensional spatial-temporal dynamics and manage extreme signal redundancy [79], [80]. Video LLMs typically employ streaming encoding or hierarchical memory banks to iteratively compress massive frames into compact representations [79]. This necessitates adaptive memory selection to pinpoint specific temporal ”moments” in sequences spanning hours, rather than relying solely on the semantic vector matching common in text retrieval [80].

#### 3 PRELIMINARIES

This section transitions from the motivational overview in the Introduction to a more rigorous theoretical framework. We first establish a formal definition of a world model and subsequently detail its fundamental components, core operational functions, and the primary paradigms through which such models are acquired.

3.1 The Constituents of World Models: Observation, State, and Dynamics

The conceptual foundation of the World Model is deeply rooted in cognitive science [35], where it is characterized as a simplified internal mental representation of reality constructed within the human psyche.

By perceiving external sensory manifestations (observations) and distilling them into abstract internal concepts (state), humans utilize these internal representations to simulate causal outcomes (dynamics)

Sec. 1 Introduction Evolution and Frontiers

- Sec. 3.1 The Constituents of World Model

Observations State Dynamics

- Sec. 3.2 The Operations of World Model

State Estimation State Transition

- Sec. 3.3 The Acquisition of World Models

Coupled Training

Sequential Architecture Unified Architecture

Decoupled Training Modular Architecture

- Sec. 3.4 Evolution of Video Model

Sec. 3 Preliminary

Observation Modality RGB

- Sec. 4 State Coupled State

- Sec. 4.1 Implicit State (Memory Mechanism)

Compression

Static Compression Dynamic Compression

Retrieval

Internal Retrieval External Retrieval

consolidation

- Sec. 4.2 Explicit State

Hidden-variable State Parametric State

Decoupled State

Semantic-oriented State Geometry-oriented State

- Sec. 5 Dynamics

Video World Model

Sec. 5.2 Causal Knowledge Integration

Video World Models

Video Model + VLM

Sec. 5.1 Causal Architecture Reformulation

Causal Video Model

- Sec. 6 Evaluation

- Sec. 6.1 Quality VBench/VBench++ [59], [60]

Temporal Quality Visual Quality Text Alignment

- Sec. 6.2 Persistence

Long-Horizon Coherence e.g. VBench-Long [60]

Memory Capacity Tasks

World Consistency & Object Perminence

e.g. World Consistency Score [61]

Revisition e.g. rFID [62] Spatial Memory Retrieval e.g. MemoryMaze [63]

- Sec. 6.3 Causality

Reasoning

Physical Dynamics e.g. Physics IQ [64]

Temporal Reasoning

e.g. ChronoMagicBench [65]

Intervention e.g World-in-world [66]

Planning

e.g. Video-GoBench [32] CALVIN [67] RLBench [68]

- Sec. 7 Future Work Causal Knowledge Integration

- Sec. 7.1 Persistence

Implicit State Explicit State

- Sec. 7.2 Causality

Causal Architecture Reformulation

Fig. 1: Overview of the paper structure following the mindmap “From Video Generation Model to World Model”.

within a mental “sandbox.” This internal simulation allows individuals to anticipate the potential consequences of various behaviors before executing physical actions, thereby serving as the fundamental cognitive substrate for proactive decision-making and strategic planning.

Beyond cognitive science, a closely related formalization of internal mental simulation emerged in control theory, where the behavior of the system being studied is modeled through a state-space representation [36], [37]. In this paradigm, the core variables summarizing the system’s status are selected and maintained as the internal state x, which can fully describe the system without redundancy. Its dynamics x˙ (the derivative of x) and observation y can be derived using differential equations, which are artificially predefined using physical laws:

x˙ = f(x,u), y = g(x), (1)

where u denotes the control input. This state-space formulation provides a mathematically rigorous definition of the mappings from the state to the dynamics and the observations, enabling forward simulation for prediction, planning, and control of the system.

Building upon insights from cognitive science and control theory, world models from MBRL formalize internal mental simulation using a learned latent state-space representation [38], [39], [40], [41], [42], [43], [44], [53], [54], [55], [56], [57]. In this framework, the operation of a world model is also distilled into observation, state, and dynamics. Specifically, the observation at time t, denoted as Ot, captures external sensory information and can be interpreted as the system output in a control-theoretic sense. In our context, Ot typically takes the form of high-dimensional pixel-level visual inputs, such as video frames, providing

only a partial and indirect view of the underlying environment. To bridge the gap between raw perception and internal reasoning, the state St is introduced as the agent’s latent representation of the environment. This state serves as a learned, high-dimensional generalization of the classical system state in control theory, summarizing all taskrelevant information needed for prediction and decisionmaking. Although its specific parameterization varies across architectures, St consistently serves as a compact, abstract encoding of the world’s state. Finally, the dynamics govern the causal evolution of this state under specific interventions or actions At, which correspond to control inputs. This evolution is modeled through a learned state transition function

##### St+1 = f(St,At), (2)

mirroring the state equation in classical dynamical systems. This predictive mechanism enables the model to simulate future states by internalizing the environment’s objective physical laws, thereby allowing the agent to “foresee” the consequences of its actions before they are executed.

Taken together, the discussions above suggest that, despite differences in domain and formalism, the operation of a world model consistently revolves around three tightly coupled components:

- • Observation: This refers to the raw, perceptible data from the environment. In the context of video models, observations are typically high-dimensional, pixel-level inputs (video frames) that provide a partial, indirect view of the world.
- • State: The state is the internal representation that aims to provide a comprehensive explanation of the current environment. To establish an accurate state, a model must distill and aggregate information from a vast history of observations. Its primary goal is to retain all task-relevant variables while filtering out irrelevant noise and redundancy.
- • Dynamics: The dynamics govern how the state evolves over time, often under the influence of specific actions or interventions. By modeling the transi-

tion St+1 = f(St,At), the world model internalizes the underlying physical and causal regularities of the environment.

- 3.2 The Operations of World Models: Estimation and Prediction

Building upon the fundamental components introduced above, a world model operates through two core processes: state estimation and state transition. Although their concrete implementations differ substantially across modeling paradigms, these processes share a common conceptual foundation. In what follows, we abstract away architectural details and focus on the essential logic underlying each operation.

- 1. State Estimation (Understanding the World). State estimation functions as the model’s abstraction mechanism. Its goal is to compress high-dimensional, sequential observations into a compact representation that captures the current status of the environment.

[Figure 1]

Fig. 2: World Model Inference Cycle. (1) Estimation: Maps historical observations O1:t to a latent representation St (Eq. 4). (2) Transition: Predicts the future state St+1 given current state and action At, facilitating mental rollouts and future prediction (Eq. 8).

Formally, this process approximates the posterior distribution of the latent state conditioned on the observation history:

##### St ∼ Pϕ(St | O1:t) (3)

In stateful world models, which are commonly adopted in MBRL methods [39], [40], [41], [42], [43], [44], state estimation is typically implemented through a recurrent aggregation mechanism. The latent state is updated incrementally by integrating new observations and actions over time:

##### St = Encϕ(St−1,Ot,At−1) (4)

This recurrent formulation enables the model to encode non-instantaneous physical properties—such as velocity, acceleration, and object permanence—which cannot be inferred from a single observation alone.

In contrast, many recent transformer-based architectures [14], [20], [21], [24], [25], [26], [81], [82], [83], [84], [85], [86], [87], [88] adopt a stateless paradigm. These models do not maintain an explicit recurrent hidden state. Instead, the latent representation is derived directly from a fixed temporal window of recent observations:

St ≈ Ot−k:t (5)

Here, the burden of temporal reasoning is shifted from an explicit state update to the attention mechanism operating over the observation sequence.

At the extreme end of this spectrum lie large-scale generative video models. Because these models operate directly on video frames—often via a Variational Autoencoder (VAE) whose latent space preserves fine-grained visual details—the notion of “state” undergoes a conceptual simplification. In such settings, the latent state is frequently identified with the entire observation history:

##### St ≡ O1:t (6)

Under this identity, state estimation reduces to maintaining the observation stream itself. The model’s “understanding” of the world is thus fully encoded in the sequence of past perceptions, which is treated as a sufficient statistic

- TABLE 1: Comparison of World Model Paradigms. We compare representations across Cognitive Science, Control Theory, RL, and Video Generation. The Trade-off row synthesizes strengths (+) and limitations (-).

Dimension Mental Model [35] Control Theory [36] MBRL World Model [42] Video World Model [14]

(Cognitive Science) (State-Space) (Latent Dynamics) (Generative Video) State Abstract Concepts Physical Variables Latent Vectors Visual Tokens Dynamics Heuristic inference Deterministic Equations RNN Transition Global Attention Observation Sensory Perception Sensor Readings Compressed Visual

Raw frames

Features

Prediction Mental imagination Numerical simulation Latent rollout Video generation based on

history Trade-off + Generalization

+ Precise, Interpretable

+ Planning Efficiency

+ Real-world Alignment – Computational Cost

– Unverifiable

– Poor Scalability

– Weak Interpretability

for the current environment. Subsequent reasoning and simulation are performed by conditioning directly on this raw or semi-raw history.

- 2. State Transition (Predicting the World). While state estimation captures the current configuration of the environment, state transition models its causal evolution over time. This process acts as an internal simulation engine, allowing the world model to predict future states or observations.

In stateful architectures with explicit latent representations [39], [40], [41], [42], [43], [44], transitions are governed by a parameterized latent dynamics model:

St+1 ∼ Pθ(St+1 | St,At) (7)

This formulation enforces a clear separation between representation learning and dynamics modeling, enabling efficient multi-step rollout in the latent space.

By contrast, in stateless models without an explicit state variable [14], [20], [21], [24], [25], [26], [81], [82], [83], [84], [85], [86], [87], [88], state transition is handled implicitly through the progressive expansion of the observation sequence.

Under the identity St ≡ O1:t, the dynamics reduce to an action-conditioned next-observation prediction problem:

Ot+1 ∼ Pθ(Ot+1 | O1:t,At) (8)

In this formulation, “transition” corresponds to an autoregressive update of the history buffer. By predicting the next high-fidelity observation Ot+1 and appending it to the existing sequence, the model advances the world state directly in observation space. Through repeated application, this process effectively turns the model into a data-driven visual simulator.

- 3.3 The Acquisition of World Models: Coupled vs. Decoupled Learning

Since world models primarily serve to facilitate downstream decision-making, their acquisition can be categorized based on the degree of coupling with the policy model during the learning process.

1. Closed-loop Learning (Coupled Training). The world model and the policy model are trained jointly, such that

the world model exhibits a direct gradient dependence on the policy’s objectives. This paradigm can be further distinguished into two structural forms. In a sequential composition, the models remain distinct modules but are optimized via a shared gradient flow. For instance, in “videoas-world-model” research such as UniPi [28] or GAIA-

- 1 [27], an inverse action model is employed to extract actions from generated video sequences. If these extracted actions deviate from the original instructions, gradients are backpropagated into the world model to enforce “actionexecutable” and physically consistent generations. Alternatively, a unified architecture integrates the world model and the policy into a single end-to-end framework, exemplified by WorldVLA [89], where perception, prediction, and action generation are optimized within a holistic system.
- 2. Open-loop Learning (Decoupled Training). They treats the world model as an independent entity learned primarily through the passive observation of large-scale datasets. In this paradigm, while the policy model may utilize the world model for internal “imagination” during its own optimization—as seen in classic MBRL frameworks [39], [43]—the parameters of the world model itself do not receive gradients from the policy’s reward signals or loss functions. Representative examples of this decoupled approach are modern video generation models [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25]. These models focus on mastering the objective statistical regularities of visual dynamics from vast amounts of data, providing a robust but fixed simulation substrate that remains unaffected by the specific decision-making logic of the downstream agent.
- 3.4 The Evolution of Video Models: Towards Robust World Simulators

Modern video generation models have emerged as a powerful candidate of the world models [14], [20], [27], [28], [29], [30], [31], [32]. However, despite their visual fidelity, they differ from the classical definition in several key aspects:

- 3.4.1 State Most contemporary video generation models lack an explicit, compressed latent state. Instead, the observation se-

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Gradient Unified

Gradient

Unified-Action World Model

World Model

Architecture

Loss

Loss

Visual Input

Visual Input

Modular

Architecture

“Grasp the water bottle”

[Figure 9]

[Figure 10]

[Figure 11]

“Grasp the water bottle”

[Figure 12]

[Figure 13]

Gradient

World Model

Action Model Loss

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Text Input

Gradient Sequential Architecture

[Figure 18]

Text Input

World Model

Action Model Loss

(a) Coupled Training

(b) Decoupled Training

- Fig. 3: Learning Paradigms: Coupled vs. Decoupled Training. (a) Coupled Training (Closed-loop): The world and policy models are optimized jointly via shared gradients. This often takes place in Unified Architectures or Sequential Architectures (distinct modules with continuous gradient flow). (b) Decoupled Training (Open-loop): The world model serves as a pre-trained, frozen simulator. It enables policy planning but remains fixed, receiving no gradient updates from the action model.

quence O1:t itself serves as an implicit state. However, the ever-expanding nature of these sequences imposes a significant computational burden and a growing memory footprint. This expansion makes long-term temporal reasoning increasingly difficult as the model must attend to a progressively larger context, often leading to a loss of persistence in long-horizon simulations. Consequently, recent literature focuses on enhancing persistence through two primary architectural strategies:

- (1) Memory Mechanisms: These approaches maintain the implicit nature of the state within the input sequence but introduce specialized mechanisms to manage temporal data more effectively. Rather than treating all past frames equally, these methods implement operations to selectively store, retrieve, or compress elements within the sequence, thereby optimizing the model’s ability to handle extended temporal contexts without linear increases in complexity [90], [91], [92], [93], [94], [95], [96].
- (2) Explicit States: In contrast, these methodologies move away from raw sequence dependence by explicitly constructing latent states that function as global, long-term memory structures. By distilling historical information into a fixed-size or hierarchical latent bottleneck, these models decouple the reasoning complexity from the sequence length, providing a stable foundation for maintaining environment consistency over distant time horizons [97], [98], [99], [100], [101], [102], [103], [104], [105].

- 3.4.2 Dynamics Standard video models [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25] often utilize bidirectional attention, generate high fidelity videos given text prompt. During the generation process, video models function primarily as “renderers” that produce a fixed-duration clip simultaneously, rather than exhibiting explicit temporal causality in their visual reasoning. To address the inherent lack of causality, current research follows two primary strategic paths to instill causality into video generation:

(1) Causal Architecture Reformulation. This path focuses on fundamental architectural shifts or optimization objectives to ensure the model respects temporal order. By training or distilling video models to be truly autoregressive or by employing causal masking mechanisms, researchers aim to transform the generation process from a simultaneous

“rendering” task into a sequential “forecasting” task [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116], [117], [118], [119], [120].

(2) Causal Knowledge Integration. This path leverages external models that inherently possess strong causal reasoning capabilities, such as Large Multimodal Models (LMMs), to guide the generation process. This integration typically manifests in two forms: (1) Sequential Decoupling: A highlevel LMM serves as a “planner” to determine the logic and dynamics of the scene, while the video model acts as a downstream “visualizer” to render the pixels [121], [122], [123], [124], [125], [126], [127]; (2) Unified Coupling: The capabilities of understanding and generation are fused within a single framework [128] where the causal reasoning of the LMM and the generative power of the video model are optimized jointly to ensure that the produced dynamics are grounded in logical world knowledge.

3.4.3 Summary

Regarding other dimensions, such as the learning paradigm, current video models predominantly rely on Open-loop Learning, where the model is optimized independently of specific agent policies. While a significant area of active exploration focuses on whether joint training with policy-level signals is essential to achieve higher grounding and action alignment, such investigations into coupled learning remain beyond the scope of this survey. Instead, this survey focuses on investigating the state-of-the-art in video generation as a substrate for world modeling. Specifically, we review recent advancements in state representation—addressing challenges in persistence and memory—and causal dynamics—addressing the transition from rendering to physical forecasting—to chart a comprehensive technical roadmap toward more robust world simulation.

#### 4 CATEGORIZATION - STATE

To investigate the evolution of video generation models toward becoming robust world models, we first analyze their internal representations, specifically focusing on the construction of the state. As introduced in the Preliminaries, our primary objective is not necessarily to compel the model to produce an explicit state variable, but rather to incorporate the concept of a ”state” as a sufficient statistic. By distilling historical context into such a representation,

we ensure that the model can maintain coherent, long-term simulations. In the following sections, we will delve into two distinct paradigms—Memory Mechanisms and Explicit Latent States—to analyze how they respectively address the challenges of long-term consistency and physical persistence.

###### 4.1 Implicit State - Memory Mechanism

- 4.1.1 Definition In this paradigm, the video generation model does not explicitly construct a compact, abstract latent variable to represent the world. Instead, it relies on implicit state construction, where the “state” is functionally equivalent to a managed context of historical observations.

Here, the state St is not a fixed-size vector but a dynamic collection of information derived from the history of observations O1:t. Crucially, St is not directly equivalent to the raw history (which would be computationally intractable) but is the result of an external Memory Mechanism (M).

This mechanism constitutes the functional backbone of the implicit state construction. Drawing parallels to cognitive memory systems [129]which rely on encoding to compress information, consolidation to stabilize storage, and retrieval to access relevant pasts, we can formalize M as a composite mechanism addressing the computational constraints of video generation through three functional primitives:

- • Compression: Given the high spatio-temporal redundancy in video streams, raw history is inefficient

to store. The mechanism compresses the context O1:t into compact representations (e.g., via token merging or summary vectors) to retain high-density information.

- • Retrieval: Not all historical information is equally relevant for generating the next frame. The mechanism employs retrieval strategies (e.g., sparse attention or key-value lookup) to selectively access specific segments of the past state based on the current generation intent.
- • Consolidation: Upon generating new content, the memory state must evolve. The update mechanism dictates how new observations are integrated and which obsolete information is evicted or re-weighted, enabling the model to support indefinite streaming.

Under this definition, the “state” represents the Active Working Memory of the video model, the specific subset of compressed and retrieved history required to generate the next frame.

- 4.1.2 Memory Mechanism - Compression In the context of video generation, Context Compression involves condensing long historical sequences into compact representations to mitigate computational bottlenecks. While these methods often prioritize inference efficiency over explicit dynamics modeling, they serve as a critical foundation for scaling world models. By reducing the quadratic complexity of attention mechanisms, compression enables the exploration of scaling laws over extended temporal horizons. We categorize these mechanisms into

Dynamic Compression and Static Compression based on the timing of the compression and whether the strategy is content-adaptive or pre-defined.

Dynamic Compression refers to the real-time merging or pruning of context during the model’s computation, where compression criteria are typically derived from semantic similarity or attention scores. This approach has been extensively validated in LLMs through methods such as ToMe [130], AdaptMerge [131], TokenFusion [132], MCTF [133], and DiffRate [134]. In the video domain, VidToMe [135] applies these principles by merging similar tokens across frames, enhancing temporal consistency while reducing the workload to allow self-attention modules to handle longer sequences. Furthermore, works like Sparse VideoGen (SVG) [136] leverage an online profiling strategy to capture dynamic sparse patterns, effectively compressing the attention calculation graph by skipping non-essential spatial or temporal paths in real-time. Since the redundancy is identified based on mid-run feature activations, these methods are often training-free and highly adaptive to specific video content.

Static Compression involves condensing the context according to pre-defined or heuristic strategies before the model’s core computation begins. While LLM-based works like Learning to Compress [137], RAPTOR [138], and Long [139] utilize specialized structures to manage longrange dependencies, video-centric methods explicitly leverage spatial-temporal redundancies. FramePack [140], Pyramidal Flow Matching [141], and LoViC [142] design compression ratios based on temporal importance, where the degree of downsampling varies across the timeline, typically maintaining higher fidelity for recent frames and aggressive compression for distant history. RELIC [143] adopts empirical spatial downsampling on the KV cache to accommodate longer horizons, whereas TempoMaster [144] implements a coarse-to-fine temporal strategy where the context is compressed via a low frame rate ”blueprint” before being refined at higher rates. Because these strategies alter the input distribution or representation density, they generally require the model to be trained or fine-tuned to adapt to the compressed latent context.

4.1.3 Memory Mechanism - Retrieval

External Retrieval involves the pre-selection of historical or external context before it is fed into the generative model, focusing on contextual relevance rather than just computational throughput. This paradigm is generally divided into two categories: maintaining self-consistency through historical memory and enhancing quality via external referenceaugmentation. To preserve spatial-temporal consistency, which is often referred to as an implicit geometry memory, methods like WorldMem [93], Context-as-Memory [94], VRAG [145], and WorldPack [91] query historical frames using low-dimensional physical states such as camera poses, 3D coordinates, and orientation. While Ctrl-World [146] simplifies this by sampling historical frames to reduce context length, MagicWorld [147] utilizes the first frame of an I2V sequence as a visual key to retrieve relevant data from the history cache. Beyond self-history, another branch of research focuses on reference-augmented generation to inject ”World Knowledge” or motion priors. Corgi [92] utilizes

|𝑂𝑡+1| |
|---|---|
| | |

|𝑂𝑡+1|
|---|

| | | | |
|---|---|---|---|
|𝑂𝑡+1| | | |

Previous Memory

…

…

e.g., Sliding Window

𝑂1 𝑂2 𝑂3 𝑂𝑡−1 𝑂𝑡

𝑂1 𝑂2 𝑂3 𝑂𝑡−1 𝑂𝑡

Video Generation Model

Video Generation Model

Video Generation Model

e.g.,

e.g., FramePack

WorldMem

Current Memory

(a) Compression

(b) Retrieval

(c) Consolidation

- Fig. 4: Functional Primitives of the Memory Mechanism. To manage the computational constraints of long-term video generation, the implicit state is governed by three operations: (a) Compression mitigates bottlenecks by condensing the raw

observation history O1:t into compact representations. (b) Retrieval prioritizes contextual relevance by selectively accessing specific historical segments via internal routing or external matching. (c) Consolidation follows a constant computational strategy by dynamically updating the memory buffer, integrating newly generated observations Ot+1, and evicting obsolete history for continuous streaming.

T2I-generated candidates to guide the model, while DiTMem [95], RAGME [148], and MotionRAG [149] leverage CLIP-based embeddings to retrieve external video clips or ”motion exemplars” that provide realistic physical dynamics. Finally, OneStory [96] advances this by introducing a learnable retrieval module, demonstrating that external controllers can evolve from passive search engines into active, intent-aware components of the world simulator by optimizing the retrieval policy end-to-end.

Internal Retrieval implements retrieval implicitly within the attention mechanism. Instead of managing the context buffer externally, these methods integrate the “retrieval” process directly into the attention calculation phases. To address the computational bottleneck of long-context modeling, a prominent paradigm reframes the world state as a dynamically retrieved context, where the internal memory is managed through learnable or prior-driven visibility operators. Early approaches in this category, such as MoC [150] and MoBA [151], implement learnable routing engines that partition the history into content-aligned or contiguous chunks, utilizing mean-pooled key descriptors for top- block selection; while MoC emphasizes content boundaries like shots and frames, MoBA focuses on the mathematical consistency of merging these routed outputs via Online-Softmax. To preserve global awareness often lost in sparse selection, VSA [152] introduces a geometryaligned dual-pathway architecture that fuses fine-grained local retrieval with a gated coarse global context, whereas VMOBA [153] tailors this multi-scale perspective for diffusion models through a 1D-2D-3D layer-wise recurrent partition and global thresholding. Diverging from purely learned routing, AdaSpa [154] and Sparse VideoGen [136] leverage structural priors, specifically attention stability across denoising steps and head-specific spatiotemporal roles, to achieve training-free inference-time sparsification. The granularity of retrieval is further refined in MoGA [155] and SVG2 [156], which transcend physical block boundaries by clustering tokens into semantically coherent groups or memory blocks via k-means centroids, ensuring highfidelity identity preservation across disjoint segments. Efficiency and robustness are addressed in BSA [157], which proposes bidirectional pruning of both query and key-value redundancies, and ReSA [158], which safeguards memory integrity by introducing periodic dense rectification

to counter cumulative approximation errors and memory drift. Finally, the retrieval logic becomes increasingly specialized in Video-XL-2 [159], which employs a task-aware mixed-resolution strategy via bi-level KV representations, and Radial Attention [160], which imposes a physics-driven distance prior to log-scale memory access based on the natural exponential decay of attention energy.

- 4.1.4 Memory Mechanism - Consolidation

Memory Consolidation refers to the real-time distillation of the context buffer following the generation of new content to preserve long-range consistency while maintaining a constant computational footprint. StreamingT2V [161] implements this by extracting features from the final 8 frames of a generated chunk as a short-term window while anchoring the entire sequence to the first frame’s global semantics. FreeLong [162] utilizes a spectral blending operation, using 3D FFT to isolate and merge global low-frequency structural layouts with local high-frequency details from the most recent window. For token-based modeling, Loong [164] employs a pixel-to-token re-encoding of the final 5 frames to reset distribution shifts before discarding previous history. In contrast, FAR [114] utilizes asymmetric patchify kernels to aggregate distant historical frames into low-resolution tokens within the KV cache. WorldWeaver [165] expands the consolidation modality by archiving generated depth cues into a structured memory bank to constrain geometric drift. Finally, EgoLCD [163] introduces an active update policy through importance-driven sparse KV compression, filtering out redundant tokens based on attention scores to retain only high-value semantic features.

- 4.1.5 Summary: Functional Primitives of Implicit State

In synthesizing the landscape of implicit state construction, we categorize these approaches based on their operational logic and functional roles in managing the trade-off between computational constraints and historical fidelity (see Table 2). While all three primitives serve to mitigate the bottleneck of processing raw history (O1:t), they fundamentally differ in how they define the relevance of information.

Compression functions as a mechanism for information digestion, implementing a “soft averaging” strategy that systematically reduces data density regardless of specific generation intent. In contrast, Retrieval acts as an active

- TABLE 2: Taxonomy of Memory Mechanisms in Implicit State. We categorize the management of historical context into three functional primitives: Compression, Retrieval, and Consolidation.

Feature Compression Retrieval Consolidation Core Logic Condense raw history via token

Filter context via heuristic rules Dynamically update buffer &

evict obsolete data Cost (Time/Space) Low / Low Medium / Low Low / Constant

merging/pruning

+ Infinite Streaming - Semantic Drift Examples ToMe [130], FramePack [140],

Trade-off + Computational Efficiency

+ Contextual Relevance - Rigid Rules

- Lossy (Details dropped)

WorldMem [93], Context-as-Memory [94]

StreamingT2V [161], FreeLong [162], EgoLCD [163]

VidToMe [135]

or retrieval) to a functional transition mapping, where the model learns to compress and propagate only the most salient information for future consistency. Depending on the architectural relationship between this transition function and the generative backbone, we categorize these works into two sub-paradigms:

St+1

Ot+1

Ot+1

St+1

Video Model & State Model

Video Model

State Model

St

Ot

Ot

St

Ot+1 St

- • Coupled States: The transition function is intrinsically fused within the generation backbone, allowing the model to function as a synchronous statetransition system. Here, the state manifests either as hidden-variable states (e.g., SSM hidden units, LSTM cells, or linear attention buffers) or as parametric states where history is encoded directly into the plasticity of the model’s weights through online optimization.
- • Decoupled States: The state is structurally decoupled from the backbone’s internal activations, existing as a standalone representation that is explicitly updated. This includes semantics-oriented approaches that utilize separate transition models (e.g., LLMs) to evolve latent world descriptions, and geometryoriented approaches where the state is an explicit 3D memory (e.g., point clouds or Gaussians) updated via iterative spatial fusion and back-projection.

(a) Coupled States

(b) Decoupled States

###### Fig. 5: Explicit State Architectures. Instead of buffering

raw history, these models maintain a compact variable St. (a) Coupled: State transition is fused within the generative backbone, where St evolves as internal hidden activations or weights. (b) Decoupled: Dynamics are structurally separated; a standalone transition model updates St before feeding it into the generator.

input selection filter. Rather than uniformly condensing history, it optimizes the information flow by proactively identifying and routing only the most contextually relevant segments based on heuristic or learned priors. Finally, Consolidation operates as a post-hoc distillation process. Distinct from the pre-selection nature of retrieval, it focuses on the dynamic maintenance of the memory buffer after generation, evicting redundant information to support infinite streaming without semantic drift. Collectively, these components transform a static history buffer into a dynamic working memory, ensuring that the implicit state remains a refined, high-density summary of the past.

4.2.2 Coupled States

Crucially, the recursive updatability of these methods allows the observation model to function inherently as a state transition model, where the mapping and the generation of new observations are performed synchronously by a single architecture. In this paradigm, the “memory” of the sequence is stored within the model itself, manifesting either as hidden-variable states or parametric states.

- 4.2 Explicit State 4.2.1 Definition In contrast to the “Implicit State” paradigm, which manages raw observations through external selection policies, this paradigm constructs the world state through internalized compression. Rather than maintaining a growing buffer of previous frames, these methods distill historical context into a globally updating latent variable, which serves as a comprehensive mathematical summary of the video’s evolution.

Hidden-Variable States. This class of states typically consists of hidden activation vectors maintained and updated through internal causal linear systems, such as State-Space Models (SSMs), Recurrent Neural Networks (RNNs), or linear attention mechanisms. For instance, MALT Diffusion [97] constructs a global state by maintaining a fixed-size memory latent vector within the model layers to compress long-range context into compact tokens, using a recurrent attention mechanism to distill visual features from each generated segment. Similarly, VideoSSM [102] utilizes a Mamba-based SSM branch where the hidden states serve as a global compressed memory to capture dependencies

Within this framework, the “State” is an abstract, compact representation, ranging from transient activation vectors and parametric weights to explicit 3D geometries that evolved through a formal transition process. Crucially, the update mechanism shifts from heuristic rules (such as FIFO

beyond local sliding windows. SANA-Video [99] leverages the cumulative properties of linear attention to maintain a constant-memory state, employing a block-wise autoregressive approach to integrate incremental features via the associative property. To resolve issues of forgetting, Recurrent Autoregressive Diffusion (RAD) [101] integrates LSTM modules into the Diffusion Transformer, using hidden and cell states as memory carriers updated through an RNN forward pass. Architecture-specific scanning methods further enhance these states; LinGen [103] utilizes Mamba2 hidden states initialized with ”review tokens” for global semantic priors and updates them via linear scanning guided by Rotary-Major Scan (RMS). Other representative works include Matten [104], which embeds the global state in hybrid Mamba-Attention latent variables, and DiM [105], which employs bidirectional SSM modules for spatio-temporal propagation. Finally, Po et al. [98] combine block-wise SSM states with local KV caches, using a spatial-major/timeminor scan order to evolve the state linearly as new frames arrive.

Parametric States. This paradigm represents a fundamental shift by treating a subset of model parameters as dynamic, input-dependent variables rather than fixed constants. Instead of storing history in activation buffers, specific weights serve as state carriers that are continuously updated based on the input stream, effectively allowing the model to ”learn” and solidify the dynamics of the current context on-the-fly. TTT-DiT [166] parameterizes the global state as weight matrices within Test-Time Training (TTT) layers, where an online self-supervised gradient optimization process ”writes” long-term historical information directly into the parameter space during inference. Building on this, Titans [167] constructs a neural memory module that utilizes a momentum-based update rule to consider both past trends and current “surprises” when updating memory weights. From a theoretical perspective, Nested Learning [168] defines the global state as the optimal solution to a nested optimization problem aimed at compressing the “Context Flow” of a sequence, distillating and solidifying context into multi-level state representations in real-time.

Summary. Hidden-variable states offer significant advantages in engineering feasibility and rapid architectural iteration, particularly through hybrid designs that combine stateless Transformer blocks with causal linear systems. However, effectively balancing state capacity with the frequency of memory updates remains a critical challenge that warrants further investigation for long-horizon modeling. In contrast, parametric states theoretically possess far greater capacity and representational power than hiddenvariable states. Nevertheless, their implementation involves substantial engineering overhead and complex optimization requirements, and empirical references within the current video generation literature remain relatively nascent.

- 4.2.3 Decoupled States Methods in this category maintain an explicit state that is structurally decoupled from the internal activations of the visual generation backbone. Instead of fusing dynamics into the model’s weights or hidden layers, these approaches preserve a standalone state representation, ranging from abstract latent variables to explicit 3D geometry that evolves

as new observations are produced. The mechanism of state evolution depends on the system’s orientation: while semantics-oriented methods typically utilize a standalone transition model to map states across time, geometryoriented methods treat the 3D representation itself as the state, updating it through iterative spatial fusion.

Semantics-oriented State. Semantics-oriented methods focus on maintaining high-level consistency in object identity, scene logic, and narrative progression. By abstracting historical context into symbolic or latent semantic representations, these approaches ensure that the “story” of the video remains coherent over long durations, even as visual details transform. For instance, Owl-1 [121] implements a standalone transition model by maintaining a compact latent world state external to the video backbone; following the rendering of each clip, a multimodal LLM (MLLM) summarizes observed dynamics into a naturallanguage description and maps the current state to the next state to guide subsequent generation. Similarly, Pack and Force [169] employs the MemoryPack mechanism, where the SemanticPack module functions as an external recurrent network. It iteratively updates a long-term memory vector by compressing historical segments and aligning them with global text-image guidance, effectively encapsulating minute-level context within a standalone recurrent state variable.

Geometry-oriented State. Geometry-oriented methods prioritize spatial consistency, view-invariance, and physical structure. A distinctive feature of this paradigm is that the 3D representation itself acts as the state, serving as a persistent, evolving memory rather than a static asset. In this framework, 3D structures, such as point clouds [170], [171], [172], [173], [174], [175], [176], 3D Gaussians [177], [178], meshes [179], or implicit fields [180] are progressively refined over the course of generation. Instead of relying on a learned transition model, the state is updated through direct fusion of new observations. The process typically begins with a coarse reconstruction estimated from sparse observations, followed by an iterative render–generate–update cycle: the current 3D memory is rendered from a new viewpoint to identify missing regions; a generative model synthesizes the novel view; and the synthesized image is subsequently back-projected and fused into the 3D representation to update geometry, appearance, and visibility. This iterative refinement allows the system to accumulate scene knowledge over time, ensuring rigorous structural integrity across multi-view predictions.

Summary. Decoupled states offer significant flexibility, allowing researchers to select specialized state spaces and transition models tailored to specific task requirements. Semantics-oriented states facilitate transitions within highly abstract LLM-driven semantic spaces, which excel at narrative logic but may lack fine-grained spatial reasoning. Conversely, geometry-oriented states leverage explicit 3D representations to ensure rigorous spatial consistency and structural integrity, though they often encounter difficulties in modeling rich and fluid temporal dynamics.

- TABLE 3: Comparison of Implicit vs. Explicit State Paradigms. Implicit states prioritize visual fidelity via history buffers, while Explicit states prioritize efficiency and reasoning via compact latent variables.

Feature Implicit State Explicit State

(Managed History / Buffer) (Condensed Variable / Latent)

Mechanism External Management (e.g. Compression, Retrieval) Internal Recurrence (e.g. St → St+1)

Context-Selection Heuristic Learning/Rule-based selection Learned physics/transitions Persistence Window-Limited, Local fidelity Global Continuity, Infinite horizon Complexity Context-Bound: O(N) or O(N2) Constant: O(1)

Trade-off + High Fidelity - High Cost + Memory Efficient - Information Loss Examples WorldMem [93], Context-as-Memory [94] VideoSSM [102], RAD [160], MALT [97]

###### 4.3 Summary: Implicit vs. Explicit States in World Modeling

Synthesizing the paradigms of state construction (see Table 3), we analyze the divergence between implicit and explicit approaches across four critical dimensions: Mechanism, Logic, Persistence, and Scalability.

Mechanism and Persistence. The fundamental distinction lies in how history is being stored. Implicit states operate as a managed History, utilizing external management mechanisms to maintain a buffer of raw observations. While this preserves high visual fidelity by keeping authentic tokens, it results in Window-Limited persistence, where the model is prone to forgetting once it exceeds the context window. Conversely, Explicit states employ Internal Recurrence, distilling history into a compact latent variable (St) through recursive updates. This formulation achieves Global Continuity, theoretically supporting infinite horizons, though the aggressive compression can lead to “information decay” and a loss of fine-grained details over time.

Logic and Causality. From a causal perspective, the two paradigms define information relevance differently. Implicit states are predominantly Heuristic-Driven, relying on human-engineered rules (e.g., similarity matching or time proximity) to select context. In contrast, Explicit states are Dynamics-Driven. By requiring the model to autonomously learn the state transition function (St → St+1), these systems move beyond pattern matching to internalize the underlying physics of the world, aligning more closely with the theoretical definition of a world simulator.

Scalability and Trade-offs. Finally, computational scalability dictates the deployment feasibility. Implicit states are Context-Bound, with inference costs growing linearly or quadratically (O(N) to O(N2)) relative to history length. Explicit states, however, offer Constant Scalability (O(1)), maintaining a fixed computational footprint regardless of simulation duration. Ultimately, the choice represents a trade-off: Implicit states currently offer the most reliable path for high-fidelity video synthesis, while Explicit states represent the frontier for efficient, autonomous agents capable of long-term reasoning.

#### 5 CATEGORIZATION - DYNAMICS

In this section, we primarily examine the sources of dynamic behavior within video generation models. To serve as effective world models for simulation, video models must possess a degree of reasoning capability. Specifically, a model must internalize underlying causal laws to ensure that its temporal rollouts remain physically plausible and logically consistent. Current research aimed at enhancing the causality of video models follows two predominant strategies.

###### 5.1 Causal Architecture Reformulation

To internalize underlying causal laws and ensure physically plausible temporal rollouts, current research focuses on Causal Architecture Reformulation, primarily by reengineering denoisers into strictly causal structures. Initial efforts in Causal-Masked Video Diffusion [108], [109], [110] bridge the gap between rendering and forecasting by eliminating future-to-past information leakage. Methods like AR-Diffusion [108] and MAGI-1 [109] enforce directional dependency via asynchronous or monotonic per-frame noise schedules, while Self-AR (NOVA) [110] and VideoMAR [111] integrate frame-wise causal attention masks and next-frame diffusion losses to predict masked tokens based on complete preceding contexts. This autoregressive paradigm is further extended by Video-GPT [112], which predicts sequential video clips through a clip-level causal mask, and FAR [114], which uses asymmetric patchify kernels to compress distant history and maintain longcontext causality. VideoPoet [115] and NOVA [116] reformulate generation as next-token or next-frame prediction tasks using discrete or non-quantized tokens, while Lumos-1 [117] and ART-V [118] implement strict interframe causality through token dependency strategies and short-term context conditioning. Beyond architecture, recent techniques like Diffusion Forcing [119] utilize progressive noise levels and causal masks, while CausVid [181] employs causal distillation to improve efficiency. To mitigate exposure bias, Self-Forcing [182] and its derivatives—such as LongLive [183], Self-Forcing++ [106], and Rolling Forcing [107]—condition the model on self-generated content or extended contexts, and Resampling Forcing [184] provides

a teacher-free alternative by training on degraded context to simulate inference-time errors.

Summary. The essence of these methodologies lies in integrating causal attention masks into diffusion-based frameworks and employing diversified noise-level schedules to enforce strict temporal dependency. By simulating inference-time challenges—such as error accumulation and exposure bias—through various forcing strategies, these approaches effectively mitigate the training-inference discrepancy, thereby ensuring robust physical consistency and logical plausibility during long-horizon temporal rollouts.

###### 5.2 Causal Knowledge Integration

To enhance the causal reasoning capabilities of video models, a prominent research direction focuses on Causal Knowledge Integration, a paradigm that delegates highlevel dynamics and planning to Large Multimodal Models (LMMs) while utilizing video models primarily as high-fidelity “renderers.” In these decoupled, sequential frameworks, models such as Owl-1 [121] and VLIPP [125] utilize VLMs as motion planners—often via chain-ofthought reasoning—to ensure physical plausibility before the video model generates the final pixels. This “directorled” approach is further exemplified by VIDEODIRECTORGPT [122], LVD [124], and DirectorLLM [123], which employ LLMs like GPT-4 or fine-tuned Llama 3 to perform complex spatiotemporal planning, structured layout generation, or human pose simulation, thereby offloading the burden of causal reasoning from the visual generator.

Moving beyond simple sequential execution, more advanced frameworks introduce tighter integration through gradient-based optimization or unified system guidance. For instance, CSVC [126] iteratively optimizes text prompts using VLM-based textual gradients to produce causally faithful counterfactuals, while UniVideo [127] leverages an MLLM to guide a Multimodal DiT (MMDiT) in a unified stream for understanding and generation. Similarly, SemanticGen [185] maintains long-term consistency by decoupling the process into high-level semantic planning and low-level detail refinement stages. Ultimately, the most sophisticated implementations achieve deep architectural coupling, as seen in BAGEL [128]. By employing a bottleneck-free Mixture-of-Transformer-Experts architecture with shared self-attention, BAGEL unifies multimodal understanding and generation within a single system, allowing emergent world-modeling capabilities and complex reasoning to flourish through a holistic fusion of planning and simulation.

#### 6 EVALUATION

Evaluating the leap from video generation to world simulation calls for different benchmarks. Traditional metrics (e.g., IS, FVD) mainly quantify perceptual realism over short clips under passive viewing, but a usable world model must (i) produce high-quality observations, (ii) preserve a persistent state over long horizons, and (iii) obey causal/physical structure—especially under interventions. Accordingly, we organize evaluation along three core axes: Quality (frame-level fidelity, short-range temporal

coherence, and conditioning alignment), Persistence (longhorizon consistency, revisitation, and memory-dependent continuity), and Causality (temporal reasoning, counterfactual response, and action-consistent dynamics).

###### 6.1 Quality

Evaluating the generative quality of a video model requires a multi-faceted approach that moves beyond simple perframe realism to encompass visual fidelity, temporal dynamics, and semantic alignment. Traditional metrics like Inception Score (IS) and Fr´echet Inception Distance (FID) gauge isolated frame realism, while Fr´echet Video Distance (FVD) [186] extends this to short clips to capture basic spatiotemporal consistency. However, because these metrics often miss fine-grained temporal errors or drift in longer sequences, researchers utilize specialized measures like Fr´echet Video Motion Distance (FVMD) [187] to penalize unnatural motion and optical flow-based metrics to ensure frame-to-frame smoothness. To unify these dimensions, comprehensive suites like VBench [59] hierarchically decompose quality into specific aspects such as motion smoothness, subject identity, and spatial relations, using pretrained models to score each automatically. This framework is further expanded in VBench++ [60], which covers a wider range of generative tasks and introduces a trustworthiness dimension alongside rigorous text alignment evaluations. Text alignment is particularly critical; modern benchmarks like WorldModelBench [30] go beyond simple CLIP similarity checks by incorporating instructionfollowing tests, ensuring the model not only produces realistic footage but also strictly adheres to the semantic constraints and narrative details defined in the input prompt.

###### 6.2 Persistence

A defining feature of a world simulator is persistence: the ability to maintain a coherent internal state over long horizons. We evaluate this capability through two primary lenses: long-horizon coherence and specific memory capacity tasks.

6.2.1 Long-horizon Coherence

This aspect evaluates the model’s stability as the generation length increases, ensuring that the simulation does not collapse, diverge, or drift over time. Since standard metrics like FVD are typically computed on short windows and fail to capture long-term degradation, researchers employ protocols like VBench-long [60]—an extension of VBench designed for longer videos—to monitor trends in ”subject appearance change” and ”background continuity” over hundreds of frames. Studies on architectures like StreamingT2V [161] and SANA-Video [99] highlight the necessity of this evaluation, demonstrating that while naive models often suffer from quality collapse or identity switches after roughly 600 frames, purpose-built persistent architectures can maintain competitive FVD and consistent scene elements across minute-long sequences (around 1800+ frames).

- 6.2.2 Memory Capacity Tasks

Beyond maintaining visual consistency, persistence requires the model to ”remember” specific states, causal logic, and spatial layouts, which is evaluated through distinct memory tasks. The World Consistency Score (WCS) [61] provides a holistic, no-reference metric that checks the internal logical integrity of the generated ”world,” specifically monitoring object permanence, relation stability, and causal compliance to ensure entities do not vanish or act erratically. To test specific environmental recall, the scene re-visitation test [93] requires the camera to return to a previous location; performance is quantified using reconstruction FID (rFID) [62], where a low score indicates the model effectively retrieved the correct latent state rather than hallucinating new details. Finally, persistence is stress-tested via functional navigation tasks like the Memory Maze [63] or benchmarks within VR-Bench [188], where the model must generate a traversal through a complex environment without redundant loops, demonstrating an implicit spatial memory that informs future frame generation.

###### 6.3 Causality

The third critical axis of evaluation is causality: the extent to which a generative model internalizes and adheres to the physical laws and logical progressions of the simulated environment. A robust world model must transcend mere visual plausibility to satisfy the causal constraints governing real-world events. We categorize these evaluations into three progressive levels: reasoning, intervention, and planning.

- 6.3.1 Temporal Reasoning and Physical Validity

Fundamental to causal understanding is the correct handling of event ordering and physical interactions. The ChronoMagic-Bench [65] assesses long-range temporal reasoning by focusing on transformations that require strict monotonic ordering, such as biological aging or object fabrication. Models are penalized for semantic scrambling—for instance, reverting a fully grown tree to a seedling. Metrics such as the Metamorphic Progression Score and Temporal Coherence Score quantify the realism of these temporal trajectories. High performance here indicates an implicit representation of the ”arrow of time,” distinguishing causal simulation from temporally agnostic video generation.

In parallel, physical consistency is evaluated through benchmarks like the Physics-IQ test suite [64]. This protocol conditions models on short video prompts involving deterministic physical events (e.g., collisions, fluid dynamics, gravity) and compares the generated rollout against ground truth. Accuracy is measured via four complementary metrics: Spatial IoU (localization of action), Spatio-temporal IoU (timing and location accuracy), Weighted Spatial IoU (magnitude of movement), and pixel-wise Mean Squared Error (MSE). These are aggregated into a normalized PhysicsIQ score, where 100% represents perfect physical variance replication. Current state-of-the-art models achieve scores as low as 24%, highlighting a significant gap between surfacelevel visual realism and the underlying modeling of physical dynamics.

- 6.3.2 Interventions and Evaluation

A rigorous test of causal modeling is the response to interventions: the ability to generate coherent rollouts when initial conditions or actions are altered. Systematic tests involve generating twin sequences that diverge at a specific control point (e.g., an object is pushed vs. left stationary). A valid world model must produce distinct, logically consistent outcomes for each intervention without introducing artifacts in unrelated scene elements.

To formalize this, World-in-World [66] introduces a unified platform for agent-in-the-loop evaluation. Unlike passive open-loop benchmarks, this framework couples the video generator with an embodied agent. The world model acts as the simulator, responding to agent actions with new frames. The primary metric shifts from visual fidelity to task success rate across navigation and manipulation scenarios. Results indicate that visual quality does not guarantee functional utility; models with superior consistency and predictability often enable higher task success. This underscores the necessity of ”intervention-centric” evaluation, where the model’s value is defined by its reliability as an interactive environment.

- 6.3.3 Planning and Embodied Task Performance

The ultimate validation of a world model lies in its utility for planning and decision-making. Evaluations in this domain are adapted from robotics, measuring the performance of policies acting within the generated world. Key metrics include Success Rate (SR). Additionally, normalized regret compares agent performance within the generated model against an oracle policy in a ground-truth simulator, quantifying the performance degradation attributable to model inaccuracies.

To diagnose the source of failures, researchers measure controllability: the alignment between the model’s predicted dynamics and real-world physics under identical action sequences. Formally, for a sequence of actions a1:T, the controllability score is defined as:

##### Ctrl(a1:T) = exp(−λ · DLPIPS (oˆ1:T(a1:T),o⋆1:T(a1:T))),

(9)

where oˆ1:T represents the observation sequence generated by the world model, o⋆1:T is the ground-truth sequence from a physics engine or real-world recording, and DLPIPS is the perceptual distance metric scaled by λ. A score approaching 1 implies high fidelity to real-world dynamics.

Finally, recent work explores emergent planning capabilities within video generation models. For instance, VideoWorld [32] demonstrates that large-scale video pretraining allows models to function as planners for complex tasks, such as the game of Go or robotic manipulation in CALVIN [67] and RLBench [68], without explicit reinforcement learning. Success in these decision-oriented benchmarks suggests that sufficiently advanced generative models effectively internalize the causal structure of the environment, allowing them to function not just as renderers, but as actionable world models.

#### 7 FUTURE WORK

In this survey, we have argued that the evolution of video generation into world simulation represents a fundamental shift in the field. We identified that for current models to truly function as world simulators, they must effectively address the twin challenges of persistence and causality.

###### 7.1 Persistence

Regarding persistence, future research directions diverge depending on whether the model relies on implicit or explicit state representations:

Implicit States. The critical imperative is to move beyond simplistic, heuristic-based memory strategies—such as fixed-length context windows—toward more advanced, data-driven memory mechanisms. These systems should leverage learned knowledge rather than hand-crafted rules, utilizing sophisticated attention-based operations to dynamically determine which information is essential for maintaining long-term consistency.

Explicit states. In cases such as State-Space Models, the focus should shift toward balancing computational efficiency with visual fidelity. While compact, fixed-size memory buffers offer extreme efficiency, they often act as a bottleneck for visual quality. Future work must explore hybrid strategies that preserve the efficiency of compressed states without sacrificing the fine-grained details necessary for high-fidelity simulation.

###### 7.2 Causality

To realize the ultimate vision of a world model, research must shift from capturing mere statistical correlations to understanding true causal mechanisms. We propose two parallel paths:

Causal Architecture Reformulation: This involves investigating how pre-training can bestow models with robust causal inference capabilities. This requires a fundamental exploration of model architectures and, more importantly, data preparation. Establishing sophisticated annotation methodologies and granularities to decouple latent causal factors within video data is paramount. Such decoupling is not only essential for causal reasoning but is also intrinsically linked to achieving precise control in video generation.

Causal Knowledge Integration: A promising short-term solution involves bridging the gap between generative and understanding models. By integrating strong reasoning priors from large-scale understanding models into the generative process, we can move toward a unified system where video creation is guided by a deep comprehension of underlying dynamics. However, effectively aligning these generative and understanding components remains a significant research challenge.

Ultimately, these advancements will enable video models to overcome the fundamental hurdles of persistence and causal consistency, bringing us closer to the realization of general-purpose world simulators.

#### 8 CONCLUSION

This survey examines the convergence of Video Generation and Model-Based Reinforcement Learning, identifying the World State (St) as the pivotal component distinguishing a simulator from a content generator. We argue that while current architectures excel at visual synthesis, their dependence on raw observation buffers (O1:t) imposes severe bottlenecks in reasoning and scalability. Through our taxonomy of Implicit and Explicit paradigms, we observe a clear trajectory toward compressing historical context into compact, persistent representations. Consequently, evaluation must shift from perceptual metrics to functional benchmarks that assess state consistency and physical dynamics. Looking ahead, the field must transcend passive, open-loop prediction. By enabling closed-loop interaction and causal intervention, future models will evolve from rendering pixels to simulating the governing laws of reality.

#### REFERENCES

- [1] A. Gupta et al. Photorealistic video generation with diffusion models. In European Conference on Computer Vision, pp. 393–411. Springer, 2024.
- [2] H. Lu et al. Vdt: General-purpose video diffusion transformers via mask modeling. arXiv preprint arXiv:2305.13311, 2023.
- [3] X. Ma et al. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [4] W. Fan et al. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453, 2025.
- [5] D. Liu et al. Lumina-video: Efficient and flexible video generation with multi-scale next-dit. arXiv preprint arXiv:2502.06782, 2025.
- [6] Y. Zhang et al. Waver: Wave your way to lifelike video generation. arXiv preprint arXiv:2508.15761, 2025.
- [7] T. Wan et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [8] G. Chen et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.
- [9] G. Ma et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.
- [10] N. Agarwal et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [11] Y. HaCohen et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.
- [12] W. Kong et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [13] G. Team. Mochi 1: A new sota in open text-to-video. https: //www.genmo.ai/blog, 2025.
- [14] OpenAI. Video generation models as world simulators, 2024.
- [15] B. Lin et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [16] Z. Zheng et al. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [17] Z. Yang et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [18] O. Bar-Tal et al. Lumiere: A space-time diffusion model for video generation. In ACM SIGGRAPH Conference and Exhibition on Computer Graphics and Interactive Techniques in Asia, 2024.
- [19] X. Chen et al. Seine: Short-to-long video diffusion model for generative transition and prediction. ArXiv, abs/2310.20700, 2023.
- [20] Google. Veo 3.1: Our state-of-the-art video generation model with expanded creative controls, 2025.
- [21] Kuaishou. Kling ai, 2024.
- [22] Y. Gao et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [23] MiniMax. Hailuo ai video generator – transform idea to visual with ai, 2025.
- [24] J. Bruce et al. Genie: Generative interactive environments. ArXiv, abs/2402.15391, 2024.
- [25] PikaLabs. Pika 1.5, 2024.
- [26] Runway. Gen-3, 2024.

- [27] A. Hu et al. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.
- [28] Y. Du et al. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023.
- [29] X. Huang. Towards video world models, 2025.
- [30] D. Li et al. Worldmodelbench: Judging video generation models as world models. arXiv preprint arXiv:2502.20694, 2025.
- [31] B. Kang et al. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024.
- [32] Z. Ren et al. Videoworld: Exploring knowledge learning from unlabeled videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 29029–29039, 2025.
- [33] K. J. W. Craik. The Nature of Explanation. Cambridge University Press; Macmillan, Cambridge, UK, 1943.
- [34] D. Gentner and A. L. Stevens. Mental models. Psychology Press, 2014.
- [35] P. Johnson-Laird. Mental models: Towards a cognitive science of language, inference, and consciousness. Harvard University Press, 1983.
- [36] R. E. Kalman. A new approach to linear filtering and prediction problems. Transactions of the ASME–Journal of Basic Engineering, 82(Series D):35–45, 1960.
- [37] K. J. Astr¨˚ om and R. Murray. Feedback systems: an introduction for scientists and engineers. Princeton university press, 2021.
- [38] J. W. Forrester. Counterintuitive behavior of social systems. Theory and decision, 2(2):109–140, 1971.
- [39] D. Ha and J. Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3), 2018.
- [40] D. Ha and J. Schmidhuber. Recurrent world models facilitate policy evolution. Advances in neural information processing systems, 31, 2018.
- [41] D. Hafner et al. Learning latent dynamics for planning from pixels. In International conference on machine learning, pp. 2555–

2565. PMLR, 2019.

- [42] D. Hafner et al. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020.
- [43] D. Hafner et al. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.
- [44] D. Hafner et al. Training agents inside of scalable world models. arXiv preprint arXiv:2509.24527, 2025.
- [45] V. Mnih et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.
- [46] V. Mnih et al. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pp. 1928–

1937. PmLR, 2016.

- [47] G. Barth-Maron et al. Distributed distributional deterministic policy gradients. arXiv preprint arXiv:1804.08617, 2018.
- [48] J. Schrittwieser et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.
- [49] J. Oh et al. Value prediction network. Advances in neural information processing systems, 30, 2017.
- [50] D. Silver et al. The predictron: End-to-end learning and planning. In International Conference on Machine Learning, pp. 3191–3199. PMLR, 2017.
- [51] N. Hansen et al. Temporal difference learning for model predictive control. arXiv preprint arXiv:2203.04955, 2022.
- [52] N. Hansen et al. Td-mpc2: Scalable, robust world models for continuous control. arXiv preprint arXiv:2310.16828, 2023.
- [53] V. Lee et al. Dreamsmooth: Improving model-based reinforcement learning via reward smoothing. arXiv preprint arXiv:2311.01450, 2023.
- [54] P. Mattes et al. Hieros: Hierarchical imagination on structured state space sequence world models. arXiv preprint arXiv:2310.05167, 2023.
- [55] Z. Liu et al. Continual reinforcement learning by planning with online world models. arXiv preprint arXiv:2507.09177, 2025.
- [56] S. Prasanna et al. Dreaming of many worlds: Learning contextual world models aids zero-shot generalization. arXiv preprint arXiv:2403.10967, 2024.
- [57] C. Hao et al. Neural motion simulator pushing the limit of world models in reinforcement learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 27608–27617, 2025.
- [58] X. Ren et al. Cosmos-drive-dreams: Scalable synthetic driving data generation with world foundation models. arXiv preprint arXiv:2506.09042, 2025.

- [59] Z. Huang et al. Vbench: Comprehensive benchmark suite for video generative models, 2023.
- [60] Z. Huang et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models, 2024.
- [61] A. Rakheja et al. World consistency score: A unified metric for video generation quality, 2025.
- [62] M. Heusel et al. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018.
- [63] J. Pasukonis et al. Evaluating long-term memory in 3d mazes, 2022.
- [64] S. Motamed et al. Do generative video models understand physical principles?, 2025.
- [65] S. Yuan et al. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation, 2024.
- [66] J. Zhang et al. World-in-world: World models in a closed-loop world, 2025.
- [67] O. Mees et al. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters (RA-L), 7(3):7327–7334, 2022.
- [68] S. James et al. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 2020.
- [69] W. X. Zhao et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2), 2023.
- [70] S. Wang et al. Knowledge editing for large language models: A survey. ACM Computing Surveys, 57(3):1–37, 2024.
- [71] Z. Zhang et al. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):1–47, 2025.
- [72] Y. Wu et al. From human memory to ai memory: A survey on memory mechanisms in the era of llms. arXiv preprint arXiv:2504.15965, 2025.
- [73] Z. Jia et al. The ai hippocampus: How far are we from human memory? Transactions on Machine Learning Research, 2025.
- [74] Y. Du et al. Rethinking memory in ai: Taxonomy, operations, topics, and future directions. arXiv preprint arXiv:2505.00675, 2025.
- [75] H. Li et al. A survey on large language model acceleration based on kv cache management. arXiv preprint arXiv:2412.19442, 2024.
- [76] D. Zhang et al. Memory in large language models: Mechanisms, evaluation and evolution. arXiv preprint arXiv:2509.18868, 2025.
- [77] J. Liu et al. A survey of personalized large language models: Progress and future directions. arXiv preprint arXiv:2502.11528, 2025.
- [78] M. M. Abootorabi et al. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented generation. arXiv preprint arXiv:2502.08826, 2025.
- [79] B. He et al. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13504–13514, 2024.
- [80] Y. Tang et al. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, 2025.
- [81] E. Decart et al. Oasis: A universe in a transformer. URL: https://oasis-model. github. io, 2024.
- [82] D. Valevski et al. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.
- [83] H. Che et al. Gamegen-x: Interactive open-world game video generation. arXiv preprint arXiv:2411.00769, 2024.
- [84] J. Yu et al. Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325, 2025.
- [85] X. He et al. Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.
- [86] S. Gao et al. Adaworld: Learning adaptable world models with latent actions. arXiv preprint arXiv:2503.18938, 2025.
- [87] S. Huang et al. Vid2world: Crafting video diffusion models to interactive world models. arXiv preprint arXiv:2505.14357, 2025.
- [88] E. Alonso et al. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024.
- [89] J. Cen et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025.
- [90] X. Wu et al. Pack and force your memory: Long-form and consistent video generation. arXiv preprint arXiv:2510.01784, 2025.

- [91] Y. Oshima et al. Worldpack: Compressed memory improves spatial consistency in video world modeling. arXiv preprint arXiv:2512.02473, 2025.
- [92] X. Wu et al. Corgi: Cached memory guided video generation. arXiv preprint arXiv:2508.16078, 2025.
- [93] Z. Xiao et al. Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025.
- [94] J. Yu et al. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025.
- [95] S. Song et al. Learning plug-and-play memory for guiding video diffusion models. arXiv preprint arXiv:2511.19229.
- [96] Z. An et al. Onestory: Coherent multi-shot video generation with adaptive memory. arXiv preprint arXiv:2512.07802, 2025.
- [97] S. Yu et al. Malt diffusion: Memory-augmented latent transformers for any-length video generation. arXiv preprint arXiv:2502.12632, 2025.
- [98] R. Po et al. Long-context state-space video world models. arXiv preprint arXiv:2505.20171, 2025.
- [99] J. Chen et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.
- [100] Y. Oshima et al. Ssm meets video diffusion models: Efficient longterm video generation with structured state spaces. arXiv preprint arXiv:2403.07711, 2024.
- [101] T. Chen et al. Recurrent autoregressive diffusion: Global memory meets local attention. arXiv preprint arXiv:2511.12940, 2025.
- [102] Y. Yu et al. Videossm: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519, 2025.
- [103] H. Wang et al. Lingen: Towards high-resolution minute-length text-to-video generation with linear computational complexity. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2578–2588, 2025.
- [104] Y. Gao et al. Matten: Video generation with mamba-attention. arXiv preprint arXiv:2405.03025, 2024.
- [105] S. Mo and Y. Tian. Scaling diffusion mamba with bidirectional ssms for efficient image and video generation. arXiv preprint arXiv:2405.15881, 2024.
- [106] J. Cui et al. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.
- [107] K. Liu et al. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.
- [108] M. Sun et al. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025.
- [109] H. Teng et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [110] H. Deng et al. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.
- [111] H. Yu et al. Videomar: Autoregressive video generatio with continuous tokens. arXiv preprint arXiv:2506.14168, 2025.
- [112] S. Zhuang et al. Video-gpt via next clip diffusion. arXiv preprint arXiv:2505.12489, 2025.
- [113] S. Lin et al. Autoregressive adversarial post-training for realtime interactive video generation. arXiv preprint arXiv:2506.09350, 2025.
- [114] Y. Gu et al. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.
- [115] D. Kondratyuk et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.
- [116] H. Deng et al. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.
- [117] H. Yuan et al. Lumos-1: On autoregressive video generation from a unified model perspective. arXiv preprint arXiv:2507.08801, 2025.
- [118] W. Weng et al. Art-v: Auto-regressive text-to-video generation with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7395–7405, 2024.
- [119] B. Chen et al. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [120] T. Yin et al. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.
- [121] Y. Huang et al. Owl-1: Omni world model for consistent long video generation. arXiv preprint arXiv:2412.09600, 2024.

- [122] H. Lin et al. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091, 2023.
- [123] K. Song et al. Llama learns to direct: Directorllm for humancentric video generation. arXiv preprint arXiv:2412.14484, 2024.
- [124] L. Lian et al. Llm-grounded video diffusion models. arXiv preprint arXiv:2309.17444, 2023.
- [125] X. Yang et al. Vlipp: Towards physically plausible video generation with vision and language informed physical prior. arXiv preprint arXiv:2503.23368, 2025.
- [126] N. Spyrou et al. Causally steered diffusion for automated video counterfactual generation. arXiv preprint arXiv:2506.14404, 2025.
- [127] C. Wei et al. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025.
- [128] C. Deng et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [129] L. Sherwood et al. Human physiology: from cells to systems. 2004.
- [130] D. Bolya et al. Token merging: Your vit but faster. In International Conference on Learning Representations (ICLR), 2023.
- [131] Z. Islam and M. Rochan. Adaptmerge: Inference time adaptive visual and language-guided token merging for efficient large multimodal models. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 7352–7361, 2025.
- [132] Y. Wang et al. Multimodal token fusion for vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12186–12195, 2022.
- [133] S. Lee et al. Multi-criteria token fusion with one-step-ahead attention for efficient vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15741–15750, 2024.
- [134] M. Chen et al. Diffrate: Differentiable compression rate for efficient vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 17164–17174, 2023.
- [135] X. Li et al. Vidtome: Video token merging for zero-shot video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [136] H. Xi et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.
- [137] J. Mu et al. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36:19327–19352, 2023.
- [138] P. Sarthi et al. Raptor: Recursive abstractive processing for treeorganized retrieval. In The Twelfth International Conference on Learning Representations, 2024.
- [139] P. Zhang et al. Long context compression with activation beacon. arXiv preprint arXiv:2401.03462, 2024.
- [140] L. Zhang and M. Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.
- [141] Y. Jin et al. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.
- [142] J. Jiang et al. Lovic: Efficient long video generation with context compression. arXiv preprint arXiv:2507.12952, 2025.
- [143] Y. Hong et al. Relic: Interactive video world model with longhorizon memory, 2025.
- [144] Y. Ma et al. Tempomaster: Efficient long video generation via next-frame-rate prediction. arXiv preprint arXiv:2511.12578, 2025.
- [145] T. Chen et al. Learning world models for interactive video generation. arXiv preprint arXiv:2505.21996, 2025.
- [146] Y. Guo et al. Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125, 2025.
- [147] G. Li et al. Magicworld: Interactive geometry-driven video world exploration. arXiv preprint arXiv:2511.18886, 2025.
- [148] E. Peruzzo et al. Ragme: Retrieval augmented video generation for enhanced motion realism. arXiv preprint arXiv:2504.06672, 2025.
- [149] C. Zhu et al. Motionrag: Motion retrieval-augmented image-tovideo generation. arXiv preprint arXiv:2509.26391, 2025.
- [150] S. Cai et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025.
- [151] E. Lu et al. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025.
- [152] P. Zhang et al. Faster video diffusion with trainable sparse attention. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

- [153] J. Wu et al. Vmoba: Mixture-of-block attention for video diffusion models. arXiv preprint arXiv:2506.23858, 2025.
- [154] Y. Xia et al. Training-free and adaptive sparse attention for efficient long video generation. arXiv preprint arXiv:2502.21079, 2025.
- [155] W. Jia et al. Moga: Mixture-of-groups attention for end-to-end long video generation. arXiv preprint arXiv:2510.18692, 2025.
- [156] S. Yang et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875, 2025.
- [157] C. Zhan et al. Bidirectional sparse attention for faster video diffusion training. arXiv preprint arXiv:2509.01085, 2025.
- [158] Y. Sun et al. Rectified sparse attention. arXiv preprint arXiv:2506.04108, 2025.
- [159] M. Qin et al. Video-xl-2: Towards very long-video understanding through task-aware kv sparsification. arXiv preprint arXiv:2506.19225, 2025.
- [160] X. Li et al. Radial attention: Sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025.
- [161] R. Henschel et al. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2568–2577, 2025.
- [162] Y. Lu et al. Freelong: Training-free long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, 37:131434–131455, 2024.
- [163] L. Zhang et al. Egolcd: Egocentric video generation with long context diffusion. arXiv preprint arXiv:2512.04515, 2025.
- [164] Y. Wang et al. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024.
- [165] Z. Liu et al. Worldweaver: Generating long-horizon video worlds via rich perception. arXiv preprint arXiv:2508.15720, 2025.
- [166] K. Dalal et al. One-minute video generation with test-time training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17702–17711, 2025.
- [167] A. Behrouz et al. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024.
- [168] A. Behrouz et al. Nested learning: The illusion of deep learning architectures. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [169] X. Wu et al. Pack and force your memory: Long-form and consistent video generation. arXiv preprint arXiv:2510.01784, 2025.
- [170] W. Yu et al. Viewcrafter: Taming video diffusion models for highfidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.
- [171] R. Li et al. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903, 2025.
- [172] X. Ren et al. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6121–6132, 2025.
- [173] C. Cao et al. Uni3c: Unifying precisely 3d-enhanced camera and human motion controls for video generation. arXiv preprint arXiv:2504.14899, 2025.
- [174] S. Wu et al. Genfusion: Closing the loop between reconstruction and generation via videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6078–6088, 2025.
- [175] T. Huang et al. Voyager: Long-range and world-consistent video diffusion for explorable 3d scene generation. arXiv preprint arXiv:2506.04225, 2025.
- [176] J. Wang et al. Evoworld: Evolving panoramic world generation with explicit 3d memory. arXiv preprint arXiv:2510.01183, 2025.
- [177] L. Chen et al. Flexworld: Progressively expanding 3d scenes for flexiable-view synthesis. arXiv preprint arXiv:2503.13265, 2025.
- [178] S. Zhang et al. Scene splatter: Momentum 3d scene generation from single image with video diffusion model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6089– 6098, 2025.
- [179] Z. Yang et al. Matrix-3d: Omnidirectional explorable 3d world generation. arXiv preprint arXiv:2508.08086, 2025.
- [180] S. Zhai et al. Stargen: A spatiotemporal autoregression framework with video diffusion model for scalable and controllable scene generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26822–26833, 2025.

- [181] T. Yin et al. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22963–22974, 2025.
- [182] X. Huang et al. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [183] S. Yang et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.
- [184] Y. Guo et al. End-to-end training for autoregressive video diffusion via self-resampling. arXiv preprint arXiv:2512.15702, 2025.
- [185] J. Bai et al. Semanticgen: Video generation in semantic space. arXiv preprint arXiv:2512.20619, 2025.
- [186] T. Unterthiner et al. Towards accurate generative models of video: A new metric & challenges, 2019.
- [187] J. Liu et al. Fr´echet video motion distance: A metric for evaluating motion consistency in videos, 2024.
- [188] C. Yang et al. Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks, 2025.

