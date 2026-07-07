[Figure 1]

IFM Technical Report: 2025-11-14

# arXiv:2511.09057v3[cs.CV]15Nov2025

## PAN: A World Model for General, Interactable, and Long-Horizon World Simulation

PAN Team, Institute of Foundation Models Mohamed bin Zayed University of Artificial Intelligence

#### Overview

A world model enables an intelligent agent to imagine, predict, and reason about how the world evolves in response to its actions, and accordingly to plan and strategize. While recent video generation models produce realistic visual sequences, they typically operate in the prompt-tofull-video manner without causal control, interactivity, or long-horizon consistency required for purposeful reasoning. Existing world modeling efforts, on the other hand, often focus on restricted domains (e.g., physical, game, or 3D-scene dynamics) with limited depth and controllability, and struggle to generalize across diverse environments and interaction formats. In this work, we introduce PAN, a general, interactable, and long-horizon world model that predicts future world states through high-quality video simulation conditioned on history and natural language actions. PAN employs the Generative Latent Prediction (GLP) architecture that combines an autoregressive latent dynamics backbone based on a large language model (LLM), which grounds simulation in extensive text-based knowledge and enables conditioning on language-specified actions, with a video diffusion decoder that reconstructs perceptually detailed and temporally coherent visual observations, to achieve a unification between latent space reasoning (imagination) and realizable world dynamics (reality). Trained on large-scale video-action pairs spanning diverse domains, PAN supports open-domain, action-conditioned simulation with coherent, long-term dynamics. Extensive experiments show that PAN achieves strong performance in action-conditioned world simulation, long-horizon forecasting, and simulative reasoning compared to other video generators and world models, taking a step towards general world models that enable predictive simulation of future world states for reasoning and acting.

#### 1 Introduction

An intelligent agent must be able to imagine how the world might unfold, reason about the consequences of its actions, and plan accordingly. A world model provides this capacity by maintaining an internal simulation of the environmental evolution. Its objective is not merely to produce plausible observations, but to sustain coherent internal dynamics that support reasoning, prediction, and interaction across space and time. Through such internal simulation, an agent can perform hypothetical thinking, explore counterfactual outcomes, and act with foresight, which is a capability that underlies general intelligence.

Recent video generation models have demonstrated stunning visual fidelity (Brooks et al., 2024; DeepMind, 2025; Yang et al., 2024; Kong et al., 2024; Wan et al., 2025b), but typically operate in an open-loop setting, producing complete videos from fixed prompts without real-time causal control or adaptive feedback.

In particular, they lack explicit notions of state, action, and possibly even object-level or conceptual representations within the video frames, and thus fall short of the full-world continuity and simulation control required for reasoning about causal and counterfactual outcomes, or evaluating different decision alternatives. Efforts in developing the more general "world models" have advanced along different directions but remain limited in various ways. Some emphasize high-quality visual generation yet operate only in a single-shot or short-horizon manner, lacking continuity across extended sequences (Agarwal et al., 2025). Others focus on interactive simulation but are constrained by their domain-specific and/or restrictive action spaces, which limit their generality (Parker-Holder et al., 2024; Feng et al., 2024). Meanwhile, 3D world models (World Labs, 2024) capture static or geometric aspects of the environment but lack fine-grained temporal dynamics and interactivity. All the existing models reveal a fragmented landscape: they achieve either visual realism without sustained dynamics, or limited controllable interaction without open-domain generality. A comprehensive world model must unify broad-domain generality and long-range interactive dynamics, ultimately enabling agents to reason, plan, and act within a coherent simulated world.

In this work, we introduce PAN, a general world model that simulates both real and counterfactual world possibilities, serving as a sandbox for simulative reasoning. To overcome the real world’s inherent stochasticity and data variability, PAN employs the Generative Latent Prediction (GLP) architecture, which unifies world model learning in latent and observation spaces. GLP defines a world model as a generative system that predicts future world states in a latent space and supervises the predicted states by anchoring them to observable data. Rather than treating uncertainties or unseen contents as obstacles to be short-circuited for model training, GLP absorbs and utilizes them during training as intrinsic aspects of the physical reality, recognizing that coherent simulation often involves generating novel viewpoints or regions beyond direct observation. This formulation separates the modeling of abstract causal dynamics from the generation of realistic and temporally consistent observations, establishing a coherent link between internal reasoning and perceptual realization. By framing world modeling as hierarchical generative prediction, GLP offers a principled foundation for developing general, interactive, and causally grounded world models.

Specifically, PAN models the evolution of latent world states conditioned on language actions and previous world states, while using video observations as the perceptual signal for learning and prediction. PAN incorporates a vision encoder to encode visual observations into latent representations, an autoregressive latent dynamics backbone to perform long-horizon world simulation in a unified multimodal latent space, and a video diffusion decoder to predict frame-level observations with local visual consistency. To address the long-standing challenge of information sparsity from raw perceptual data (e.g., videos), we adopt a large language model (LLM) as the world model backbone to ground the perceptual encodings in massive real-world context and knowledge acquired via text-based pretraining. The autoregressive world model backbone enables steerable and interactive world simulation, in contrast to existing video generation models that typically produce single, non-interactive video segments. Besides, conventional video generation methods also struggle to maintain stable performance over extended rollouts, often suffering from error accumulation and temporal drift. To address this issue, we introduce the Causal Shift-Window Denoising Process Model (Causal Swin-DPM) in the video diffusion decoder, which augments the denoising process proposed by Feng et al. (2024) with chunk-wise causal attention mask to ensure smooth transitions between consecutive chunks and reduce compounding artifacts during long-term simulation.

Training data plays a crucial role in the ability to model continuous and interactive world dynamics. Previous video datasets primarily consist of short, independent clips that lack action-conditioned temporal continuity, making them unsuitable for learning interactive world models. To overcome this limitation, PAN is trained on large-scale video–action pairs spanning diverse physical and embodied domains, providing the long-range temporal context and multimodal grounding necessary for coherent, action-responsive world simulation. The large-scale data and architectural integration together enable PAN to simulate visually

grounded worlds that evolve coherently under natural language control.

We demonstrate PAN’s strong quantitative performance across multiple evaluation settings in §7, and illustrate a wide range of qualitative results in §8. These results also indicate significant potential for further enhancement with more powerful GLP designs, continued scaling and expanded training in the future. PAN thus takes a crucial step toward general-purpose world models that unify grounded reasoning, physical simulation, and long-horizon interactivity.

#### 2 PAN World Model

We present PAN, a general world model designed for interactive prediction, long-horizon simulation, and multimodal reasoning. The real world often presents long and highly stochastic dynamics and noisy data measurements, which make modeling extremely challenging. To address these difficulties, PAN adopts a generative latent prediction (GLP) architecture (Xing et al., 2025), which mitigates error accumulation and signal variability through the strategy of hierarchical abstraction. In particular, PAN unifies latent dynamic modeling and perceptual grounding through three components that define a generative distribution over observed data: a vision encoder that transforms raw observations into structured latent representations, an LLM-based predictive backbone that evolves world states conditioned on actions and history, and a multi-granular diffusion decoder that reconstructs temporally coherent visual observations. Together, these components enable PAN to simulate the evolution of both latent and observable worlds in a coherent generative process.

###### 2.1 The GLP Architecture

A general world model must unify two complementary abilities: (1) to simulate future world evolutions through internal latent dynamics, and (2) to ground these dynamics in observable sensory experience. The GLP framework formalizes this principle as a generative architecture that learns structured latent dynamics while remaining anchored to empirical observations. It provides a universal design template for constructing world models that support simulative reasoning, counterfactual imagination, and agentic learning.

We denote by ot the current observable world state and by at the agent action at time t. GLP introduces a latent state sˆt that represents the agent’s internal belief of the world, and a predicted observation oˆt+1 that represents the model’s generative reconstruction of the next world state. The GLP architecture of PAN defines three core components:

Encoder h maps an observation to a latent representation, summarizing the multimodal world into a compact latent state:

sˆt ∼ ph(· | ot) (1)

Predictive Module f models the latent world dynamics, evolving the latent state forward under the influence of actions:1

sˆt+1 ∼ pf(· | sˆt,at) (2)

Decoder g reconstructs observable outcomes from latent states, anchoring the simulation back to the

1We adopt the Markov model here for clarity, but it is formally possible to represent non-Markovian dynamics by defining an augmented latent state s˜t = [ˆs1, a1, sˆ2, . . . , at−1, sˆt] which concatenates the states and actions of all previous time steps.

sensory domain:

oˆt+1 ∼ pg(· | sˆt+1) (3)

Together, these components define a generative process of the next observation ot+1 given the current observation ot and proposed action at as below:

(4)

ph(ˆst | ot) encoder

pf(ˆst+1 | sˆt,at) world model

pg(ot+1 | sˆt+1) decoder

pPAN(ot+1 | ot,at) =

s ˆt,sˆt+1

which jointly models the evolution of both latent world state and observable percceptual data. The key idea of GLP lies in how it couples latent prediction with generative supervision to maintain representational richness and training stability.

###### 2.2 Generative Supervision and Learning Objective

We consider a dataset D = {(ot,at,ot+1)} consisting of consecutive observations and agent actions, where ot and ot+1 denote the current and next world states, and at the action applied between them.

PAN is trained under a generative supervision objective that grounds latent predictions in the observation space. Specifically, PAN is trained to minimize the discrepancy disc(ˆot+1,ot+1) between the predicted next observation oˆt+1 = g ◦ f(h(ot),at) and the ground-truth observation o′:

LPAN = E(ot,at,ot+1)∼D disc o ˆt+1, ot+1 . (5)

By reconstructing observable outcomes, PAN ensures that each latent transition sˆt→sˆt+1 corresponds to a realizable sensory change, thereby keeping the learned dynamics interpretable and grounded.

A key distinction between PAN and prior encoder-only predictive models, such as the Joint Embedding Predictive Architecture (JEPA, Assran et al., 2023) lies in the nature of their learning objectives. JEPA and similar frameworks define learning as a latent-space matching problem, training an encoder and predictor to minimize the distance between encoded latents of consecutive observations:

LJEPA = E(ot,at,ot+1)∼D ∥f(h(ot),at) − h(ot+1)∥ (6)

As analyzed in Xing et al. (2025), this objective is prone to collapse: the model can trivially minimize the loss by mapping all observations to a constant vector and learning an invariant transition. This leads to the indefinability problem, where latent transitions are unconstrained by the true data distribution and may not correspond to any realizable world dynamics. To mitigate collapse, JEPA-style methods typically introduce additional regularizers, such as mutual-information maximization or energy-based constraints, but these heuristics complicate optimization and do not fundamentally resolve the indefinability of the latent space. Recent work such as DINO-WM (Zhou et al., a) alleviates the collapse issue by training the dynamics predictor on fixed DINOv2 (Oquab et al.) features. While this stabilizes the latent space, it does not address the deeper problem identified in Xing et al. (2025): the learned transitions remain ungrounded in the observation space and do not correspond to realizable world dynamics. Because the latent representations are fixed and not optimized for temporal consistency, the predictor can still produce transitions that are semantically valid in feature space but physically or causally implausible. Another related line of work in

###### 3D scene modeling (Mildenhall et al., 2021; Kerbl et al., 2023) optimizes reconstruction losses to learn spatially consistent radiance fields from multi-view observations. Unlike PAN, these models operate in static 3D feature space rather than latent dynamics, focusing on geometric consistency instead of predictive evolution. As a result, their learned representations capture scene appearance but lack temporal grounding

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

##### …

Obs 𝑜 Obs 𝑜 Obs 𝑜 Obs 𝑜

| | | | |
|---|---|---|---|
| | | | |

Short-Term … Consistency

Vision Encoder

Video Decoder

Latent World State 𝑠 Latent World State 𝑠 Latent World State 𝑠 Latent World State 𝑠

Long-Term

Autoregressive World Model

Consistency …

| | | | |
|---|---|---|---|
| | | | |

𝑎 = “drive through a snowy forest”

𝑎 = “drive through a field of flowers”

𝑎 = “drive through a futuristic city”

Figure 1: PAN model architecture. It consists of an autoregressive LLM-based world model backbone for long-horizon world simulation, and a video diffusion decoder for observation prediction.

and interactivity, limiting their ability to simulate agent-driven world transitions. By contrast, PAN’s generative supervision couples latent prediction with reconstruction in the sensory domain, grounding temporal evolution in observable data rather than static features or pre-defined embeddings. This enables the model to learn world dynamics that corresponds to observable and actionable world evolutions.

###### 2.3 Design Choices and Parameterization

A practical instantiation of the GLP architecture requires each component to satisfy distinct functional goals. PAN instantiates the GLP architecture with specific parameterizations: It employs a Vision-Language Model (VLM) for the encoder h and backbone f, which mitigates the information sparsity of raw perceptual data by grounding visual representations in linguistic and world knowledge, enabling interactive, longhorizon reasoning. Besides, a diffusion-based video generator serves as the decoder g, chosen for its strong visual fidelity and stability in modeling temporally coherent video dynamics. The overall structure thus integrates three complementary components as shown in Figure 1, with an overview provided below:

- • The Vision Encoder h adopts the vision tower of Qwen2.5-VL-7B-Instruct (Bai et al., 2025), an open-weight VLM. At time step t, given observation ot (e.g., images or video frames), the encoder h estimates the world state sˆt as spatially structured continuous embeddings using a Vision Transformer (Dosovitskiy, 2020).
- • The Autoregressive World Model Backbone f is based on the language model of Qwen2.5-VL-7BInstruct. Specifically, the backbone predicts the next latent world state sˆt+1 conditioned on the full history of latent states and actions (ˆs1,a1,sˆ2,...,sˆt,at), maintaining global temporal consistency across extended horizons.
- • The Video Diffusion Decoder g is adapted from Wan2.1-T2V-14B (Wan et al., 2025b) with a Causal Swin-DPM mechanism (§3.3.3), which generates the next predicted observation oˆt+1 conditioned on both the current latent state sˆt+1 and the most recent observation ot. The decoder refines each

predicted state into a temporally smooth and perceptually detailed visual segment, ensuring local temporal consistency and high-fidelity reconstruction in the observation space.

For supervision, PAN employs the flow matching loss as the generative objective disc(ˆo′,o′) , which has been shown to provide stronger supervision signals and yield higher generation quality in recent studies (Lipman et al., 2022; Liu et al., 2022; Zhou et al., b). It is worth noting that the GLP architecture itself is not limited to these specific instantiations. Each component can be independently enhanced to incorporate hierarchical embeddings, mixed discrete–continuous representations, or higher-order temporal dynamics, offering a flexible foundation for future extensions of the PAN framework. We describe more details on the model architecture, training methods, inference strategy, data construction, and experiment results in §3–7, respectively.

#### 3 Model Implementation

In this section, we provide concrete implementation details of each component of PAN, including Vision Encoder, Autoregressive World Model Backbone, and Video Diffusion Decoder, which together instantiate the GLP architecture for grounded world simulation.

###### 3.1 Vision Encoder

The vision encoder h provides the perceptual grounding of the PAN world model. It is adopted from the vision tower of Qwen2.5-VL-7B-Instruct (Bai et al., 2025), a Vision Transformer (ViT) optimized for high-resolution and sequential inputs. The encoder partitions each video frame into 14×14 spatial patches, applies windowed self-attention for computational efficiency, and encodes the resulting tokens with 2D rotary positional embeddings that preserve spatial structure. For video streams, consecutive frames are grouped through 3D patch partitioning, allowing the encoder to represent short-term motion and temporal coherence natively within its token sequence. The encoded visual tokens form the perceptual representations of the current world state, which retain spatial and temporal organization, enabling downstream modules to reason over structured latent features rather than unstructured pixel embeddings. Within the GLP framework, the encoder fulfills the role of h, transforming observations into latent world states that provide the foundation for predictive dynamics modeled by the backbone f.

###### 3.2 Autoregressive World Model Backbone

The autoregressive world model backbone f simulates the next latent world state from past states and natural language actions in a unified multimodal space, maintaining long-term temporal consistency throughout the simulation. It rolls forward the state of the world step by step in latent form, ensuring that each predicted state is informed by both historical context and the current action. The backbone is adapted from the language model of Qwen2.5-VL-7B-Instruct which, due to its pretraining on massive text corpora, helps ground the prediction of the next perceptual states in language-based real-world knowledge and rich contexts. Following the GLP paradigm, the backbone corresponds to the predictive module f, which evolves the latent state step by step, ensuring that each prediction is informed by both historical context and the current action. While the GLP formulation in Xing et al. (2025) envisions a mixed backbone combining large language models and diffusion embedders, with mixed representation that unifies discrete and continuous latent spaces, the present implementation adopts a more streamlined and unified autoregressive language backbone for clarity and scalability. Extending PAN toward the full mixed backbone remains a promising direction for future work. We describe more details about the operationalization of input conditioning and next-state prediction in §3.2.1 and §3.2.2 below.

###### 3.2.1 Observation and Action Conditioning

- As shown in Figure 1, at each timestep t, PAN takes the estimated world state sˆt represented by visual embeddings, the proposed action at represented by natural language, and a set of 256 learnable query

embeddings as inputs, and predicts the next latent world state sˆt+1. Inputs are arranged in a multi-turn conversational format, alternating between visual states sˆ1:t and next actions a1:t, aligning with the pretrained VLM dialogue structure:

Input template for the VLM in the PAN autoregressive backbone

<|user|> <image state (video state 1)> <action 1> <|assistant|> <query embedding * 256> <|user|> <video state 2> <action 2> <|assistant|> <query embedding * 256>

...... <|user|> <video state n-1> <action n-1> <|assistant|> <query embedding * 256>

This conversational prompting establishes a natural temporal structure: each assistant turn corresponds to the predicted latent state conditioned on the preceding history (ˆs1,a1,sˆ2,...,sˆt,at). During training, the backbone is teacher-forced using ground-truth states; during inference, it performs closed-loop rollouts by recursively feeding back its state prediction and taking action input, thereby realizing long-horizon simulation in the latent space.

- 3.2.2 Latent World Prediction and Simulation The output of the backbone is a compact set of 256 continuous tokens representing the next latent state

sˆt+1. Each state representation encodes high-level information about dynamics, causal relationships, and object–agent interactions in the scene. The latent representation evolves continuously across time steps, functioning as a form of associative memory (Suzuki, 2008) that preserves the global consistency of world dynamics. Because these latents reside in the shared multimodal space of the vision–language model, they inherit semantic grounding from natural language and perceptual grounding from the vision encoder, enabling multimodal reasoning and action-conditioned prediction. The latent world state is then passed to the video diffusion decoder, ensuring that the simulation of future visual observations is guided by a globally consistent world representation over extended horizons.

- 3.3 Video Diffusion Decoder

The video diffusion decoder g transforms the latent states into predicted observations, a sequence of video chunks that are perceptually detailed and smoothly connected to preceding content. While the backbone governs the global evolution of the world over extended horizons, the decoder focuses on the fine-grained spatial and motion details within each predicted state. In PAN, this module is adapted from Wan2.1-T2V-14B (Wan et al., 2025b), a large-scale diffusion transformer for text-to-video generation, and then extended with Causal Swin-DPM to support long-horizon, sequential action-conditioned simulation.

The video diffusion decoder embodies PAN’s principle of generative supervision by translating latent world dynamics into concrete perceptual realizations. While encoder-only predictive models like JEPA (LeCun, 2022) argue that reconstructing detailed future frames is infeasible due to the fine-grained variability and unseen content in visual data, PAN overcomes this challenge through a structured relaxation of the reconstruction task. Technically, a sliding-window process is employed to dampen minor errors from the previous generation and prevent them from spreading to future decoding, thus reducing error propagation over long-horizon generation. In effect, this design delegates the fine-grained variability to the decoder’s enhanced diffusion process, while allowing the backbone to model the latent dynamics that captures the structured evolution of the world states, thereby ensuring that generative reconstruction serves as a faithful and controllable reflection of the world evolution.

In the following subsections, we describe the training objective, architecture designs, and additional techniques that enable the decoder to achieve high-quality, temporally consistent simulation.

- 3.3.1 Flow Matching Objective We train the video diffusion decoder using the flow matching objective (Lipman et al., 2022), following the formulation used in Wan2.1-T2V-14B. The base model is a 14B-parameter Diffusion Transformer (DiT) adapted for our architecture. During training, given an observation x1 (a latent representation from an image or a video), a noise sample x0 ∼ N(0,1), and a denoising step k ∈ [0,1], the input latent xk is constructed via linear interpolation, following the Rectified Flow formulation (Liu et al., 2022):

xk = kx1 + (1 − k)x0. (7) The model is trained to predict the ground-truth velocity vk, defined as:

vk =

dxk dk

= x1 − x0. (8)

We use 1000 denoising steps in training, with k sampled from 1000 discrete values within the interval [0,1] using a shifted denoising step schedule (Esser et al., 2024). To accommodate the sliding window mechanism in our Causal Swin-DPM model, we incorporate specific modifications to the loss function, such

- as restricting the timestep sampling range, as described in §3.3.3.

- 3.3.2 Conditioning on Actions and World State The video diffusion decoder in PAN conditions its simulation on two complementary sources of information: the latent world state from the autoregressive backbone and the natural language action for the current step. The latent world state is first linearly projected to the conditioning dimension of the decoder, and then fed to a newly added cross-attention stream in each attention block. The output of this stream then passes through a second linear projection that is zero-initialized to ensure stable training, following the strategy proposed in (Zhang et al., 2023b). In parallel, the action text is encoded with umT5 (Chung et al., 2023a) and provided to the original text cross-attention pathway. Within each block, the post-projection output of the world-state stream is summed with the text-conditioned output, enabling the decoder to integrate global state context with action-specific visual changes while maintaining shortterm local consistency, which is illustrated in Figure

- 2.

Self-Attn

Cross-Attn

Cross-Attn (copied)

Linear

+ (zero-init)

Latent World State

Action

FFN

Hidden States

Hidden States

×𝑁

Figure 2: Transformer block of the video diffusion decoder in PAN.

- 3.3.3 Causal Swin-DPM Traditional video diffusion models (e.g., Sora) (Wan et al., 2025b; Kong et al., 2024; DeepMind, 2025; Brooks et al., 2024) are typically designed for single-shot generation. A straightforward way to extend them for sequential world simulation is to use the last frame from the last predicted observation as the condition for simulating the next one, repeating this process step by step. However, this naive extension causes two significant problems in long-horizon simulation. The first is local inconsistency between adjacent video chunks. When the model conditions only on a single final frame rather than the entire denoising trajectory

Noise Level T

Noise Level T/2

Noise Level 0

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

…

… …

Queue New Noise

###### Dequeue

T/2 Denoising Steps

[Figure 29]

Video Diffusion Decoder

### ……

Chunk-Wise Causal Attention Mask

Latent World State

Noise Level T

Noise Level T/2

Noise Level 0

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

…

… …

Queue New Noise

Dequeue

- Figure 3: Denoising process for Causal Swin-DPM.

of the preceding observation, the transition between the two video chunks may suffer from abrupt changes in motion or appearance. These discontinuities disrupt short-term coherence and make the simulation feel visually disconnected. The second is rapid quality degradation over extended rollouts. Conditioning on only the last frame means that any small artifacts, motion drift, or misalignments introduced in one predicted observation are directly passed to the next. As the number of steps increases, these imperfections can accumulate quickly, leading to noticeable deterioration in visual fidelity and stability.

To overcome these challenges, we introduce Causal Swin-DPM, a novel mechanism that augments the Shift-Window Denoising Process Model (Feng et al., 2024) with chunk-wise causal attention, as illustrated in Figure 3. This design ensures smooth local transitions and maintains high simulation quality over long durations with causal, interactive action control. Specifically, Causal Swin-DPM operates within a sliding temporal window that simultaneously holds two chunks of video frames at different noise levels. Suppose the total denoising steps are K, i.e., 1000 for our model. At the beginning, video frames for the earlier chunk are at a noise level of K/2, while those for the later chunk are at a full noise level of K. After K/2 denoising steps, the frames from earlier chunk are fully denoised and dequeued as the new video chunk. Concurrently, a new chunk of noisy frames initialized with Gaussian noise is enqueued at the end of the window. During the iterative denoising process, different chunks of video frames are conditioned on their corresponding natural language actions.

Beyond improving temporal continuity, Causal Swin-DPM also provides a structured way to handle uncertainty about unpredictable future details, which is one of the core concerns raised by encoder-only predictive models such as JEPA (LeCun, 2022). Rather than conditioning on perfectly sharp, fully specified histories, the mechanism operates on fuzzy, partially noised representations of preceding chunks. This design suppresses the history’s incidental pixel-level details that are not reliably predictive and emphasizes high-level, persistent semantic consistency (e.g., objects, scene structure, and motion). Consequently, the model is not “unfairly” penalized for details that are inherently unknowable from the current context (e.g., unseen side of an object or an occluded entity). By coupling this fuzzified conditioning with the diffusion denoising trajectory, Causal Swin-DPM encourages the model to focus on stable world dynamics while leaving fine-grained variability to the stochastic diffusion process, achieving both coherent long-horizon behavior and perceptually diverse local detail.

During training, we require two video frame chunks with a noise level difference of K/2, which corresponds

to half of the denoising step sampling interval [0,1]. To achieve this, we first subsample k from [0,0.5] for the first chunk, and assign k + 0.5 to the second chunk. The only exception occurs when generating the first video chunk since it must be fully denoised from pure noise only conditioned on the given initial frame. We subsample k from the full interval [0,1] in this case.

The ability of the sliding temporal window to see the context of the previous chunk of video frames allows for significant mitigation of errors between adjacent chunks. By mitigating errors at each local transition, Causal Swin-DPM effectively prevents the severe accumulation of degradation over extended horizons, significantly enhancing long-term performance and simulation quality.

In typical inference settings, the natural language action for the next chunk will not be available until the current chunk is fully generated. To maintain real-time interactivity, the model employs a chunk-wise causal attention mask, as shown in Figure 3, where green cells indicate visible context and white cells represent masked positions corresponding to future chunk with unseen actions. This masking ensures that the later video chunk can only attend to the previous one, preventing information leakage from future actions.

- 3.3.4 Variational Autoencoder Padding We use the Variational Autoencoder (VAE) from Wan2.1 for video frame compression. Wan2.1-VAE is a

3D causal VAE that performs both temporal and spatial compression. It compresses 1 + T frames into 1 + T/4 latent features, and our model operates within the VAE latent space. For our video diffusion decoder, the window size is 21, which corresponds to 81 real video frames. During compression, the latent representation of each frame is conditioned on its preceding frames. Since our model uses a sliding window mechanism, the latent output at a given timestep may be influenced by a large number of preceding frames. To simulate this behavior during training, we randomly pad each 81-frame clip with 0 to 122 preceding frames. These additional frames are encoded by the VAE to provide proper temporal context, but are discarded afterward.

- 3.3.5 Conditioning Frame and Noise Augmentation The sliding window in Causal Swin-DPM contains one conditioning frame at the beginning, followed by two video chunks of size 10. The conditioning frame is taken as the last frame from the previously dequeued video chunk. When the window advances, it shifts by 10 latent frames, naturally moving the last frame of the current chunk to the front to serve as the new conditioning frame. Empirically, we observe that using a fully denoised frame as the conditioning input can lead to error accumulation over long-horizon generation. To mitigate the problem, we apply noise augmentation to the conditioning frame. Specifically, we add Gaussian noise corresponding to a small fixed denoising step k = 0.055, which does not corrupt the conditioning frame excessively but still introduces sufficient stochasticity. During training, we do not compute loss on the conditioning frame.

#### 4 Model Training

Training a general-purpose world model requires more than direct supervision from raw video or shortterm observations. To achieve robust long-horizon reasoning and simulation, it is important to adopt a divide-and-conquer strategy in which individual modules are first optimized for their own roles before being integrated into a unified system. This staged approach allows each component to develop stable and specialized capabilities, such as high-fidelity short-term state simulation or accurate modeling of long-term causal dynamics, before joint optimization aligns them for end-to-end performance.

###### 4.1 Stage 1: Module-Wise Training

Training individual modules separately allows each to develop stable and specialized capabilities before the complexities of full system integration. By optimizing components in isolation, we allow each to master its

specific role and establish a strong initialization for later joint optimization.

For PAN, the vision encoder and the autoregressive backbone is built on Qwen2.5-VL-7B-Instruct, which has already undergone extensive pretraining and therefore requires no additional adaptation at this stage. The video diffusion decoder, on the other hand, is adapted from Wan2.1-T2V-14B, a non-causal video diffusion architecture originally designed for single-shot generation. In this stage, we adapt it into our Causal Swin-DPM architecture, which enables smooth temporal transitions and interactive action control over long-horizon world state simulation.

Since we freeze both the Wan-VAE and the text encoder in this stage, we precompute their latent features during data preprocessing. To reduce GPU memory usage, we adopt a combination of Data Parallelism (DP) and Fully Sharded Data Parallelism (FSDP). Specifically, FSDP is applied within each 8-GPU compute node, forming a Hybrid Sharded Data Parallel (HSDP) setup. We further enable activation checkpointing

- at the unit of each DiT block. To improve training efficiency, we apply FlashAttention-3 (Shah et al.,

- 2024) to the cross-attention layers and use FlexAttention (Dong et al., 2024) to compile custom kernels for chunk-wise causal attention.

Training is conducted using BFloat16 precision and the AdamW optimizer (Loshchilov and Hutter, 2017), with a learning rate of 1 × 10−5, a cosine decay schedule, and a linear warm-up over the first 5% of steps. Gradients are clipped to a maximum norm of 0.05. The model is trained for 5 epochs using 960 NVIDIA H200 Tensor Core GPUs.

###### 4.2 Stage 2: Joint Training

After the individual modules are prepared, we integrate them into a unified system and train them under a generative objective that couples latent prediction with reconstruction in the observation space. A frequent concern over a generative scheme for world model is that they are difficult to train because the fine details of future observations are inherently unpredictable from past inputs. Directly reconstructing raw pixels often leads to unstable objectives dominated by high-frequency noise and stochastic variation.

Building on the perspective outlined in §3.3, PAN treats this unpredictability not as evidence against generative modeling, but as a defining property of the physical world. In realistic world simulation, agents indeed routinely encounter novel viewpoints or unseen regions (e.g., turning around a corner or revealing an occluded object), yet coherence remains achievable if the underlying world dynamics are faithfully modeled. The goal of generative supervision in PAN is therefore not to reproduce the original world pixel-for-pixel, but to ensure that every generated observation remains semantically and physically consistent with the evolving latent world state.

This view is operationalized through the Generative Latent Prediction (GLP) framework, which separates structured world dynamics from unpredictable perceptual detail. Low-level stochastic variations are handled by the encoder–decoder pair (h,g), while the predictive module f focuses on modeling smooth and causally meaningful transitions in the latent space. The unified training objective is formulated as:

LGLP(h,f,g) = E(ot,at,ot+1)∼D disc g ◦ f(h(ot),at), ot+1 , (9)

where the discrepancy function disc(ˆot+1,ot+1) is instantiated as a flow-matching loss (§3.3.1). By grounding prediction in the observation space rather than matching latent embeddings, the model ensures that each transition corresponds to a realizable sensory change and avoids the collapse and indefinability issues that occur in purely latent-space objectives. This generative supervision stabilizes optimization and enables scalable, interpretable world-model training.

The backbone, built on Qwen2.5-VL-7B-Instruct, now produces latent world states that directly condition

the Causal Swin-DPM decoder. This allows the decoder to learn to interpret and render the compact world state representations produced by the backbone, while the backbone itself adapts to produce states that are most effective for guiding high-fidelity, temporally coherent simulation. We freeze the vision-language model and train only the query embeddings and the video diffusion decoder. To align with the context window size of Qwen2.5-VL-7B-Instruct, we restrict the world model history to the most recent 10 rounds during training.

We reuse the HSDP strategy and activation checkpointing introduced in Stage 1 to manage memory consumption. Given the large sequence lengths in both the VLM and DiT, we apply sequence parallelism (SP) (Li et al., 2021) to shard hidden states across GPUs along the sequence length dimension. For self-attention modules, we adopt the Ulysses method (Jacobs et al., 2023), which introduces all-to-all communication to change sharding across attention heads to calculate attention efficiently. To minimize communication overhead and maximize parallelism, we only use intra-node SP group with size 4, which is a common divisor of the 28 attention heads in Qwen2.5-VL-7B-Instruct and the 8 GPUs per compute node.

Similar to Stage 1, training is conducted with BFloat16 precision, the AdamW optimizer, a learning rate of 1 × 10−5, cosine learning rate scheduling, and a 5% warm-up. While the training schedule was set for 5 epochs, we perform early stopping after 1 epoch based on validation convergence.

- 5 Model Inference During inference, PAN performs autoregressive multi-step simulation of world dynamics: given initial observation image o1, PAN estimates the initial world state sˆ1 = h(o1) and, based on proposed natural language actions a1:T−1, predicts the future states sˆ2:T sequentially and reconstructs their corresponding observation videos oˆ2:T. For each step t, we describe the routine for PAN to evolve the latent world state and synthesize the next observation as follows:

To improve long-term consistency and generation fidelity, we augment the predicted states sˆ2:t with the encoder output of their corresponding reconstructed observations oˆ2:t, resulting in enhanced future states sˆ′k = [ˆsk,h(ˆok)] for k ∈ [2,t]. To better keep track of existing objects, agents, dynamics, and previous interactions, we concatenate the initial state with enhanced future states and all previous actions as s˜t = [ˆs1,a1,sˆ′2,a2,...,sˆ′t] to serve as the input context for the world model backbone f, which predicts the next world state sˆt+1 as below:

sˆt+1 = f(˜st,at), (10)

where the prediction is conditioned on the previously generated observations oˆ2:t rather than the ground truth, enabling open-loop simulation over extended horizons. For the decoder, we include the previous observation oˆt to better preserve perceptual continuity, and synthesize the next observation from the predicted latent state as follows:

oˆt+1 = g(ˆst+1,oˆt). (11)

This iterative process yields a coherent rollout of future world evolutions consistent across both latent and perceptual domains. By conditioning every prediction on its own generated history, PAN functions as a fully self-contained simulator capable of long-horizon imagination and reasoning.

###### 5.1 Causal Inference for Video Diffusion Decoder

With the use of chunk-wise causal attention in video diffusion decoder, the model can complete the generation of a new chunk before the natural language action is available for the subsequent chunk in the window. During inference, the outputs of the DiT are cached after every denoising step, which is

required for generating the following chunk. To balance sample quality and adherence to natural language actions, we apply classifier-free guidance (CFG, Ho and Salimans, 2022) with a guidance scale of 4. Several techniques are further employed to accelerate inference speed, which are described in §5.2.

###### 5.2 Inference Accelerations

Parallel strategy. In order to decrease the latency of each generation, we apply sequence parallel (SP, Li et al., 2021) to make use of multiple GPUs for each sample. Similar to training, we adopt the Ulysses method (Jacobs et al., 2023) for attention modules. To achieve the minimal latency, we set the SP group size to 8, the number of GPUs per compute node. Note that the number of attention heads in Qwen2.5VL-7B-Instruct 28 is not divisible by 8. Thus, we modify the Ulysses (Jacobs et al., 2023) implementation to allow for uneven sharding on the number of heads dimension.

SageAttention. We apply the SageAttention2++ (Zhang et al., 2025a) during inference for 8-bit computing and 4-bit quantization on the Hopper architecture GPUs with minimal performance drop. We modify the implementation to support our block-wise causal attention mask. We achieve 30.3% acceleration with SageAttention2++ compared to the FlexAttention and FlashAttention3 we use in the training.

6 Training Data

At the core of PAN training lies the principle that next-state prediction is the universal building block across reasoning tasks and environments. This means that any data sequence reflecting how the world evolves over time can serve as useful training material. To support this, we construct a dataset that represents the world through aligned video and text, ensuring that high-level semantic descriptions are paired with fine-grained visual dynamics in a way specifically tailored for world modeling.

The dataset comes from the widely-used publicly accessible data to capture diverse experiences across everyday activities, human–object interactions, natural environments, and multi-agent scenarios. Raw longform videos are first segmented using a dynamic shot-boundary detection strategy to produce temporally coherent clips (§6.1). Since these clips vary in quality, we design a filtering pipeline that emphasizes diversity and highlights segments rich in physical and causal dynamics while removing disruptive discontinuities such as scene cuts. The filtering process combines several off-the-shelf computer vision detectors with a custom-trained vision–language model (§6.2). Unlike conventional text-to-video datasets, PAN emphasizes temporal dynamics when aligning text with video for next-state prediction. To this end, we generate dense captions that focus on describing evolving dynamics rather than static scene attributes (§6.3).

###### 6.1 Video Segmentation

We adopt a video segmentation pipeline which operates in three stages. First, frame-level heuristics detect candidate scene boundaries, splitting each raw video into initial short clips. Next, temporally adjacent segments with similar content are merged to reduce over-fragmentation and ensure each clip represents a coherent event. Finally, a lightweight filter selects clips within a target duration range and with sufficient visual quality. These well-formed clips are then passed on to our downstream filtering and processing.

###### 6.2 Data Filtering Pipeline

Our filtering stage combines a set of off-the-shelf rule-based detectors, and a custom-trained VLM to remove low-quality or undesirable videos. Details of each filter are provided below.

- 6.2.1 Rule-Based Filters We first apply a set of handcrafted, rule-based filters designed to efficiently remove obvious low-quality or uninformative videos. These filters rely on simple, interpretable statistics computed directly from the video frames, such as motion magnitude, luminance variation, and structural changes. Because they do not

require model inference, these filters are computationally inexpensive and can be applied at scale in the early stages of the pipeline.

Extremely Static or Overly Dynamic High-quality videos typically exhibit natural and balanced motion patterns. To identify clips that are either nearly static or excessively erratic, we design motion-related metrics based on optical flow, edge differences, and luminance differences between consecutive frames. These metrics collectively measure both the magnitude and stability of motion over time. Videos that show little to no motion (e.g., frozen scenes or slideshows) or that display abrupt, unstable transitions (e.g., flashing lights, hard cuts, or camera flicker) are filtered out. This ensures that only clips with coherent and physically meaningful dynamics are preserved for training.

Trivial Motion and Pure Color We further remove clips dominated by trivial or uninformative motion patterns, such as uniform camera translation or zooming. To identify such cases, we estimate the overall motion field of each clip using sparse feature tracking and compute two scalar measures: a translation score that captures consistent directional shifts and a zoom score that reflects radial expansion or contraction. Clips exhibiting highly regular motion according to these measures are filtered out. In addition, videos that begin or end with nearly uniform color frames, often caused by fade-in or fade-out transitions or blank screens, are excluded to maintain visual and temporal consistency.

- 6.2.2 Pretrained Detectors Next, we employ existing pretrained models to identify quality issues that are difficult to capture with simple heuristics. These detectors include models for assessing aesthetic quality and detecting obstructive text. Leveraging pretrained models allows us to incorporate robust, generalizable visual understanding into the filtering process without additional training cost.

Low Aesthetic Quality Training on visually appealing and well-composed videos improves the quality and stability of generation. To assess the aesthetic quality of each clip, we uniformly sample several frames and evaluate them with a pretrained aesthetic scorer. The final video-level score is obtained by averaging the frame-level predictions, and videos with scores below a defined threshold are excluded from the dataset.

Obstructive Text Videos that contain large or persistent text regions, such as subtitles, banners, or watermarks, often distract the model and can lead to undesired text artifacts in generated outputs. To address this, we apply a scene-text detector to identify and measure the proportion of each frame covered by text regions. Videos that contain prominent or obstructive text across frames are removed from the dataset.

- 6.2.3 Custom VLM-Based Filter Finally, we train a custom vision–language model (VLM) filter to identify categories of low-quality content that earlier stages cannot reliably capture. This filter provides a more holistic understanding of both visual appearance and semantic context, enabling the removal of undesirable clips that may otherwise pass through simpler detectors. The VLM is designed to detect and exclude several types of content, including:

- 1. Lecture-type videos: Clips in which a person speaks directly to the camera without performing any meaningful actions.
- 2. Text-dominated videos: Videos where large text banners or overlays occupy a significant portion of the visual content.
- 3. Screen recordings or noisy screenshots: Clips showing desktop or mobile interfaces that lack real-world dynamics.

- 4. Low-quality content: Videos with severe blurriness, compression artifacts, or other forms of visual degradation.
- 5. Heavily edited clips: Videos that contain transitions, animations, or special effects not useful for modeling natural world dynamics.
- 6. Residual scene cuts: Clips with abrupt transitions or compositional changes that remain after the initial segmentation step.

This custom filter is trained on a curated set of labeled examples and fine-tuned to detect these patterns with high precision. During inference, the model outputs confidence scores for each property, and clips identified as positive for any of the above categories are discarded. This final stage ensures that the remaining data are not only visually coherent but also semantically consistent and physically grounded, providing a strong foundation for training the world model.

- 6.3 Dense Video Caption for Temporal Dynamics

The collected videos either lack captions or include only brief descriptions that fail to capture the rich visual content within each clip. To support world model training, we identify two key requirements for captions. First, inspired by findings from recent work such as DALL-E 3 (Betker et al., 2023), which demonstrates that dense captions significantly enhance generative performance, we require the captions to be factually rich and detailed rather than high-level or generic. Second, and more critically for world models, captions must focus on temporal dynamics, including motion, events, changes in the environment, and the emergence of new objects, rather than static background elements. The static content is typically available in the initial frame or previous video state, while the evolving dynamics of the video corresponds to the action inputs of the world model and drives state transitions. To meet these requirements, we re-caption the video data using a vision–language model specifically prompted to generate dense, temporally grounded descriptions. This ensures that each clip is paired with text that not only provides rich factual detail but also highlights the transitions and causal structure that underlie world model learning.

7 Experiments

To rigorously evaluate PAN as a general-purpose interactive world model, we conduct comprehensive experiments. In our formulation, a world model must accurately simulate future world states in response to natural language actions. Such capability allows an agent to perform thought experiments by planning and simulating actions in complex, dynamic scenarios. Guided by these principles, we design benchmarks that systematically evaluate the ability of the model to simulate and forecast future states, and to support simulative reasoning and planning by an external agent. The detailed evaluation setup and results are presented in the following subsections.

- 7.1 Baselines

We compare our PAN with competitive baselines, spanning from open-source to closed-source models. For open-source models, we select WAN-2.1 & 2.2 (Wan et al., 2025a), Cosmos 1 & 2 (NVIDIA, 2025; NVIDIA Cosmos, 2025), and V-JEPA 2 (Assran et al., 2025). For closed-source models, we select popular video generation models, including KLING (Kuaishou Technology, 2025), MiniMax (Hailuo) (MiniMax,

- 2025), and Gen-3 (Runway Research, 2024).

- WAN 2.1-I2V-14B: It is a large diffusion-transformer model for high-fidelity image-to-video generation. It focuses on photorealistic rendering and consistent scene dynamics, demonstrating long, coherent sequences and accurate prompt conditioning.
- WAN 2.2-I2V-14B: As an iterative successor to WAN 2.1, WAN 2.2 improves visual fidelity and temporal stability while expanding controllable prompting capabilities. It adopts a mixture-of-experts architecture

to better learn denoising strategies across diffusion steps.

- Cosmos1-14B: Cosmos-1 is NVIDIA’s first-generation World Foundation Model that learns physical dynamics and object affordances from large-scale video corpora. It predicts future states for actionconditioned rollouts, targeting general-purpose physical and interactive reasoning beyond pure video synthesis.
- Cosmos2-14B: The second-generation Cosmos Predict model scales capacity and controllability, improving long-horizon coherence and physics fidelity. It is designed to be post-trained for domain applications such as camera control, robotic manipulation, and autonomous driving, as well as synthetic-data generation for policy development.

V-JEPA 2: It extends Meta’s Joint Embedding Predictive Architecture to video. Rather than predicting pixels, it predicts future latent representations, enabling efficient temporal reasoning and transfer to control and perception tasks. It functions as a compact latent world model rather than a video generator.

KLING: It is a production-grade text- and image-to-video model developed by Kuaishou. It produces highquality 1080p outputs with controllable camera motion and temporal coherence, serving content-creation and post-production workflows.

MiniMax-Hailuo: It is a commercial video-generation API designed for high-resolution, minute-scale video synthesis. It supports common frame rates and user-prompted creative controls for rapid media production.

Gen-3: Runway’s Gen-3 model emphasizes reliability, character consistency, and fine-grained spatiotemporal control. It improves text-to-video quality and responsiveness compared to Gen-2 and underpins Runway’s professional visual-effects and pre-visualization tools.

###### 7.2 Evaluation Benchmarks

Existing evaluation protocols for video generation and predictive modeling mostly focus on short-horizon metrics such as frame-level fidelity or motion realism. While these measures capture visual quality, they fail to assess the fundamental competencies required of a general world model—namely, the ability to simulate coherent causal dynamics, maintain consistency across long temporal spans, and support reasoning grounded in internal simulation. Consequently, such benchmarks provide limited insight into whether a model can serve as a functional simulator for interaction, prediction, and planning.

To better evaluate these capabilities, we introduce a new benchmarking framework that measures the performance of world models along three complementary dimensions: Action Simulation Fidelity, Longhorizon Forecast, and Simulative Reasoning and Planning. These dimensions reflect a progressive hierarchy of world modeling competence—from accurately capturing and simulating world dynamics, to maintaining coherent evolution over extended horizons, and ultimately to supporting higher-level reasoning and decisionmaking grounded in internal simulation.

Action Simulation Fidelity evaluates how faithfully the model simulates language-specified actions and their causal consequences in future world states. Given an initial image observation, we employ GPT-4o (OpenAI, 2024) to propose several feasible, non-contradictory action sequences that are semantically coherent and causally applicable. The world model then simulates corresponding rollouts conditioned on these action sequences. Each rollout is scored by a VLM-based judge following Li et al. (2025) along the criteria of action faithfulness and precision. We consider two evaluation settings:

Agent Simulation. The model is required to drive a controllable entity according to the specified behaviors while maintaining background stability. For each initial state, we sample multiple distinct action sequences to induce counterfactual futures and assess both fidelity and diversity.

Environment Simulation. The model is instructed to perform scene-level interventions, such as adding, removing, or moving objects, or changing weather and lighting conditions. These settings mirror the agent planning task but emphasizes accurate simulation of scene dynamics and multi-future diversity.

Long-horizon Forecast evaluates models’ ability to maintain coherent, high-quality rollouts over extended action sequences. Starting from an initial image observation and a multi-step action script, we assess whether models can produce trajectories that remain visually faithful, dynamically plausible, and internally consistent as the prediction horizon extends. Specific, our evaluation protocol measures two complementary dimensions.

Transition Smoothness: To evaluate temporal continuity across consecutive actions, we construct multi-step action sequences that form coherent, smooth motions. The model is expected to generate predicted observations that maintain natural motion across the entire sequence, particularly around the boundaries between successive steps. We quantify this continuity using dense optical flow (Horn and Schunck, 1981) computed on the predicted video frames. From the flow fields, we derive frame-wise velocity and acceleration, and define the Transition Smoothness score as the inverse exponential of the acceleration magnitude. Higher scores indicate more continuous motion evolution and fewer abrupt or physically implausible transitions at step boundaries.

Simulation Consistency: To quantify error accumulation across extended horizons, we adopt several consistency metrics from WorldScore (Duan et al., 2025) and monitor performance degradation as the action sequences lengthen. Given the critical importance of late-stage prediction quality for downstream planning, we apply progressive penalties to later steps to emphasize temporal robustness and stability in long-horizon rollouts.

Together, these measurements evaluate multi-step dynamical continuity (does motion evolve smoothly across temporal boundaries?) and long-horizon robustness (does the simulation quality remain stable as the horizon extends?).

Simulative Reasoning and Planning evaluates whether a world model can serve as an internal simulator that enables an agent to reason about actions and plan toward a goal. A complete reasoning process involves using the model to predict the causal outcomes of actions, to compose such predictions into multi-step plans, and to guide external agents through imagined trials toward a target goal. Figure 4 shows one example for the process. Therefore, we first evaluate Step-Wise Simulation, which measures atomic causal prediction, and then assess two complementary multi-step reasoning and planning settings: Open-Ended Simulation and Planning and Structured Simulation and Planning.

Step-Wise Simulation: This task measures whether a world model can accurately predict the immediate consequence of a given action within a manipulation context. Each instance provides an initial observation and a single action, and the model must select the correct next observation from one ground-truth target and three adversarial distractors. Following WM-ABench (Gao et al., 2025), we use robotic manipulation scenarios from Agibot (Bu et al., 2025). For models capable of image or video synthesis such as PAN, human evaluators assess whether the predicted observation reflects correct object relations and physical effects. For embedding-based models (e.g., V-JEPA 2), we compute the similarity of predicted and reference latent world state. To ensure domain alignment, all models are finetuned on the Agibot dataset (Bu et al., 2025). For V-JEPA 2 specifically, we extend the model with the UMT5 encoder (Chung et al., 2023b) from WAN2.1 (Wan et al., 2025a) and finetune the model so that it can process language-based actions. This task captures the elementary causal reasoning capability that underlies more complex, multi-step simulative planning.

[Figure 45]

|Goal ! Match the number of yellow cans between the right blue tray and middle white tray<br><br>[Figure 46]|
|---|

Reasoning Step 1 Reasoning Step 2 Reasoning Step 3 Round 4

✅

[Figure 47]

[Figure 48]

Release the yellow can into the right blue tray.

[Figure 49]

[Figure 50]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Grasp a blue object from the left blue tray.

[Figure 51]

Move the yellow can to the right blue tray.

Grasp a yellow can from the middle white tray.

Grasp a yellow can from the right blue tray.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

|Grasp a yellow can from the middle white tray.|
|---|

Release the yellow can into the right blue tray.

Grasp a yellow can from the right blue tray.

Initial Observation

Release the yellow can into the middle white tray.

[Figure 57]

[Figure 58]

[Figure 59]

Release a yellow can from the right blue tray into the middle white tray.

Grasp another yellow can from the middle white tray.

Grasp a yellow can from the right blue tray.

- Figure 4: An example of a complete reasoning and planning process. Given an initial state and a goal specification, a VLM proposes candidate actions at each step. The world model simulates each candidate’s outcome as a “thought experiment", and the VLM selects the action whose predicted state shows maximal goal progress. This iterative process repeats until task completion or step limit reached. Gray arrows show explored alternatives; blue arrows trace the selected trajectory. The tree-structured search demonstrates how world models enable multi-step planning by simulating hypothetical futures before plan finalization.

Open-Ended Simulation and Planning: This setting evaluates goal-directed manipulation on diverse objects in open-ended environments. Starting from initial observations drawn from Agibot (Bu et al., 2025), a VLM agent (OpenAI-o3, OpenAI, 2025a)) proposes candidate actions, the world model simulates their consequences, and the agent selects the action whose predicted observation most closely approaches the goal. The loop continues until success or the action budget is exhausted. We curate 15 diverse scenarios and evaluate both trajectory-level success and rollout quality through blinded human assessment. For embedding-based models such as V-JEPA 2, we additionally generate plausible goal observations through minimal image editing to standardize evaluation targets.

Structured Simulation and Planning: This setting focuses on precise, language-grounded manipulation in highly structured tabletop environments containing regular objects such as colored cubes and spheres. We initialize from observations in the Language Table dataset (Lynch et al., 2023) and define goal observations through minimal spatial rearrangements. The same agent–world model planning loop is applied over 46 test cases.

Performance of planning task is measured by goal completion, defined as binary success over the full trajectory. This protocol examines not only whether a world model can simulate faithful outcomes of actions, but also whether its simulated rollouts provide sufficient reliable dynamics for iterative decision-making under open-ended goals.

11/11/25, 9:29 PM Comprehensive Evaluation Dashboard

Figure 5: Model performance on Action Simulation Fidelity, Long-Horizon Forecast and Simulative Planning tasks.

###### 7.3 Main Results

Figure 5 presents a comprehensive comparison of all evaluated models across our benchmark. Overall, PAN achieves state-of-the-art performance among open-source models and on par with the best performing closed-source model, demonstrating its effectiveness as a general-purpose world model. We now discuss results across the three evaluation dimensions, respectively.

127.0.0.1:5500/pan_eval_visual/paper_comprehensive_evaluation_dashboard.html 1/2

Action Simulation Fidelity. PAN achieves the highest overall fidelity among open-source world models, with 70.3% accuracy on agent simulation and 47.0% on environment simulation, yielding an overall score of 58.6%. It surpasses all open-source baselines and most commercial baselines. While prior work has

suggested that large pretrained video generation models could function as general world models, our results show that they struggle to maintain consistent multi-step action–effect dynamics. In contrast, PAN shows superior capacity to precisely simulate action-driven world evolution.

Long-horizon Forecast. To assess stability over extended horizons, the Transition Smoothness and Simulation Consistency metrics evaluate local motion continuity and performance consistency over long horizons. As shown in the figure, PAN achieves 53.6% on Transition Smoothness and 64.1% on Simulation Consistency, substantially surpassing all the baselines, including commercial models such as KLING and MiniMax. The results show that PAN is able to maintain visual clarity and temporal coherence throughout extended rollouts, while the baselines tend to amplify motion magnitudes and exhibit temporal drift, leading to less stable simulations.

Simulative Reasoning and Planning. The Step-Wise Simulation evaluation measures whether a world model can predict the immediate effect of an action within a physical manipulation setting. PAN achieves the highest score (56.1%) among all the open-source world model baselines. The results indicate that PAN effectively captures fine-grained causal dependencies between actions and outcomes, providing a reliable atomic step for multi-step reasoning. This capability underpins its performance in the subsequent planning tasks, where such local causal correctness accumulates into coherent long-horizon behavior.

Open-Ended and Structured Simulation and Planning evaluates whether world models can serve as internal simulators that enable downstream planning through “thought experiments,” allowing an agent to test alternative actions before execution. Using the same VLM agent (OpenAI-o3, OpenAI, 2025a)), the baselines show inconsistent performance. They improve performance in some scenarios but degrade it in others, as inaccurate or unstable simulations can mislead the agent. This indicates that realistic appearance alone is insufficient; reliable causal grounding is essential for effective plan-time reasoning. In contrast, integrating PAN with the same agent leads to consistent and substantial improvements, increasing task success by 26.7% in Open-Ended Planning and 23.4% in Structured Planning relative to the VLM-only baseline.

#### 8 Qualitative Results

We demonstrate our model’s abilities as a general-purpose world model in four domains: (i) Interactive Long World Simulation, (ii) Diverse Action and World Simulation, (iii) Rare Case Simulation, and (iv) Complex Planing and Reasoning.

###### 8.1 Interactive Long World Simulation

The model sustains long-horizon, closed-loop state evolution under real-time interventions, maintaining a consistent latent state that causally propagates actions while preserving identities and spatial relations without drift.

[Figure 63]

[Figure 65]

###### 8.2 Diverse Action and World Simulation

Our model follows diverse action instructions while generalizing across drastically different worlds; it disentangles action semantics from appearance and transfers behaviors across domains.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 72]

[Figure 73]

[Figure 74]

###### 8.3 Rare Case Simulation

The model synthesizes physically coherent tail events and counterfactuals on demand, enabling targeted robustness testing and safety-focused augmentation while keeping trajectories plausible and causally grounded.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

#### 9 Related Works

World models World models simulate the future state of the world based on its previous states and given actions (Tolman, 1948; Briscoe, 2011; Battaglia et al., 2013; Allen et al., 2020; Pramod et al., 2020). Previous world models in AI systems are usually designed for specific domains. For example, in robotics domain, world models are usually used for model-based reinforcement learning in specific simulators (Ha and Schmidhuber, 2018a,b; Matsuo et al., 2022; Chen et al., 2021; Kaiser et al., 2019). In robotics domain, world models (Yang et al., 2023; Zhen et al., 2024; Zhou et al., 2024; Ajay et al., 2023; Du et al., 2023) are capable of predicting future image or video states across diverse robotics environments. These predictive capabilities are important for robots to understand the environments, make informed decisions, and execute tasks accurately. Besides the robotics domain, world models are also widely used in autonomous driving (Wang et al., 2023b,a; Hu et al., 2023; Li et al., 2023; Zhao et al., 2024; Zhang et al., 2023a; Zheng et al., 2023), where they mainly focus on path planning and real-time decision-making, which is pivotal in enabling vehicles to navigate complex environments safely and efficiently. There are also world models for games (Bamford and Lucas, 2020; Chiappa et al., 2016; Eslami et al., 2018; Hafner et al., 2019; Kim et al., 2020; Micheli et al., 2022; Robine et al., 2022; Decart et al., 2024; Guo et al., 2025). For example, Genie 2 (Parker-Holder et al., 2024) and Matrix-Game (Zhang et al., 2025b) can simulate an interactive 3D game given a image using generative models. World Labs (World Labs, 2024, 2025) focuses on 3D scene generation and egocentric navigation, but appears to be limited to static environments without dynamic physics and interactivity. PAN, in contrast, makes a step towards building a more general world model that simulates any-domain states given any-text actions at any time.

Video generation models Video generation models aim to synthesize realistic videos given text prompts or initial frames. Recent successes in diffusion models (Ho et al., 2020; Rombach et al., 2022; SohlDickstein et al., 2015; Song et al., 2020) have paved the way for their application in the video generation domain (Khachatryan et al., 2023; Peng et al., 2023; Zhang et al., 2023d; Chen et al., 2023; Harvey et al.,

- 2022; Singer et al., 2022; Tang et al., 2024; Voleti et al., 2022; Wang et al., 2024). For example, additional modules are introduced into the existing image diffusion models (Ho et al., 2022b,a; Blattmann et al.,
- 2023; Xing et al., 2023; Zhang et al., 2023c, 2024) to facilitate video generation capabilities. However, the length of generated videos is limited due to the non-autoregressive nature. Consequently, the Diffusion Transformer (DiT) (Peebles and Xie, 2023) has been proposed to allow for autoregressive generation, and Sora 2 (Brooks et al., 2024; OpenAI, 2025b) has further scaled it up, achieving remarkable success in generating long, high-quality video. Furthermore, as the strong understanding and generation ability of LLMs, Team (2024); Yu et al. (2023) have explored the usage of LLMs in vision generation domain. Additionally, Kondratyuk et al. (2024); Sun et al. (2024) incorporate LLMs for video generation to enhance the semantic understanding. Previous models are designed to generate scenes from input descriptions, yet they frequently lack the ability to control actions or predict real-world states. On the contrary, PAN is a hybrid autoregressive-diffusion model, thus it is capable of on-the-fly control, serving for reasoning and thought-experiments.

#### 10 Conclusion

This work presents PAN, a general-purpose interactive world model that instantiates the Generative Latent Prediction (GLP) paradigm in a unified multimodal architecture. Comprehensive evaluations demonstrate PAN’s superiority among open-source world models and its competitiveness with leading commercial systems, achieving high action-simulation fidelity, stable long-horizon forecasting, and effective support for simulative reasoning and planning. Future work will explore scaling PAN to broader modalities and more interactive settings, further enhancing its capacity for causal reasoning, temporal abstraction, and real-time decision-making.

#### A Contributors

- A.1 Core Contributors Model Training Jiannan Xiang, Yi Gu, Zeyu Feng, Zihan Liu, Kun Zhou, Yichi Yang, Yiyan Hu, Benhao Huang, Guangyi Liu

Data Pipeline Jiannan Xiang, Yi Gu, Benhao Huang, Yiyan Hu, Zihan Liu

Infrastructure Yi Gu, Zihan Liu

Model Serving & Applications Zihan Liu, Yi Gu, Guangyi Liu, Jiannan Xiang

Evaluation Qiyue Gao, Jiannan Xiang, Zihan Liu

Conception, Design, and Leadership Zhiting Hu, Zhengzhong Liu, Eric P. Xing

- A.2 Contributors

Davit Abrahamyan, Arif Ahmad, Ganesh Bannur, Junrong Chen, Kimi Chen, Mingkai Deng, Ruobing Han, Xinqi Huang, Haoqiang Kang, Zheqi Liu, Enze Ma, Hector Ren, Yashowardhan Shinde, Rohan Shingre, Ramsundar Tanikella, Kaiming Tao, Dequan Yang, Xinle Yu, Cong Zeng, Binglin Zhou

#### References

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, and Pulkit Agrawal. Compositional foundation models for hierarchical planning, 2023.

Kelsey R Allen, Kevin A Smith, and Joshua B Tenenbaum. Rapid trial-and-error learning with simulation supports flexible tool use and physical reasoning. PNAS, 2020.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Chris Bamford and Simon M Lucas. Neural game engine: Accurate learning of generalizable forward models from pixels. In 2020 IEEE Conference on Games (CoG), pages 81–88. IEEE, 2020.

Peter W Battaglia, Jessica B Hamrick, and Joshua B Tenenbaum. Simulation as an engine of physical scene understanding. PNAS, 2013.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Robert Eamon Briscoe. Mental imagery and the varieties of amodal perception. Pacific Philosophical Quarterly, 92

(2):153–173, 2011.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/video-generation-models-as-world-simulators.

Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

Chang Chen, Jaesik Yoon, Yi-Fu Wu, and Sungjin Ahn. Transdreamer: Reinforcement learning with transformer world models. In Deep RL Workshop NeurIPS 2021, 2021.

Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023.

Silvia Chiappa, Sébastien Racaniere, Daan Wierstra, and Shakir Mohamed. Recurrent environment simulators. In International Conference on Learning Representations, 2016.

Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151, 2023a.

Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining, 2023b. URL https://arxiv.org/abs/2304.09151.

Etched Decart, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. URL: https://oasis-model. github. io, 2024.

Google DeepMind. Veo: A generative video model by google deepmind. https://deepmind.google/models/veo/, 2025.

Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496, 2024.

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation, 2023.

Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. arXiv preprint arXiv:2504.00983, 2025.

SM Ali Eslami, Danilo Jimenez Rezende, Frederic Besse, Fabio Viola, Ari S Morcos, Marta Garnelo, Avraham Ruderman, Andrei A Rusu, Ivo Danihelka, Karol Gregor, et al. Neural scene representation and rendering. Science, 360(6394):1204–1210, 2018.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568, 2024.

Qiyue Gao, Xinyu Pi, Kevin Liu, Junrong Chen, Ruolan Yang, Xinqi Huang, Xinyu Fang, Lu Sun, Gautham Kishore, Bo Ai, Stone Tao, Mengyang Liu, Jiaxi Yang, Chao-Jung Lai, Chuanyang Jin, Jiannan Xiang, Benhao Huang, Zeming Chen, David Danks, Hao Su, Tianmin Shu, Ziqiao Ma, Lianhui Qin, and Zhiting Hu. Do vision-language models have internal world models? towards an atomic evaluation, 2025. URL https://arxiv.org/abs/2506.2 1876.

Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, and Jiang Bian. Mineworld: a real-time and open-source interactive world model on minecraft. arXiv preprint arXiv:2504.08388, 2025.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. Advances in neural

information processing systems, 31, 2018a. David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018b. Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by

latent imagination. arXiv preprint arXiv:1912.01603, 2019. William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion

modeling of long videos. Advances in Neural Information Processing Systems, 35:27953–27965, 2022. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information

processing systems, 33:6840–6851, 2020.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022a.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video

diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022b. Berthold KP Horn and Brian G Schunck. Determining optical flow. Artificial intelligence, 17(1-3):185–203, 1981. Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca

Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.

Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al. Model-based reinforcement learning for atari. arXiv preprint arXiv:1903.00374, 2019.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023.

Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, and Sanja Fidler. Learning to simulate dynamic environments with gamegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1231–1240, 2020.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation, 2024.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Kuaishou Technology. Kling ai — app portal (chinese), 2025. URL https://app.klingai.com/cn/. Accessed 2025-10-06.

Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62, 2022. Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu

Yin, Joseph E. Gonzalez, Ion Stoica, Song Han, and Yao Lu. Worldmodelbench: Judging video generation models as world models, 2025. URL https://arxiv.org/abs/2502.20694.

Shenggui Li, Fuzhao Xue, Chaitanya Baranwal, Yongbin Li, and Yang You. Sequence parallelism: Long sequence training from system perspective. arXiv preprint arXiv:2105.13120, 2021.

Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: Layout-guided multi-view driving scene video generation with latent diffusion model. arXiv preprint arXiv:2310.07771, 2023.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. Corey Lynch, Ayzaan Wahid, Jonathan Tompson, Tianli Ding, James Betker, Robert Baruch, Travis Armstrong,

and Pete Florence. Interactive language: Talking to robots in real time. IEEE Robotics and Automation Letters, 2023.

Yutaka Matsuo, Yann LeCun, Maneesh Sahani, Doina Precup, David Silver, Masashi Sugiyama, Eiji Uchibe, and Jun Morimoto. Deep learning, reinforcement learning, and world models. Neural Networks, 2022.

Vincent Micheli, Eloi Alonso, and François Fleuret. Transformers are sample-efficient world models. In The Eleventh International Conference on Learning Representations, 2022.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:

Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. MiniMax. Hailuo ai (minimax) — official site, 2025. URL https://hailuoaiminimax.com/. Accessed 2025-10-06. NVIDIA. Cosmos world foundation model platform for physical ai, 2025. URL https://arxiv.org/abs/2501.03575. NVIDIA Cosmos. Cosmos-predict2: General-purpose world foundation models for physical ai, 2025. URL https:

//github.com/nvidia-cosmos/cosmos-predict2. GitHub repository. OpenAI. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774. OpenAI. Openai o3 and o4-mini system card. https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758

f3722c1/o3-and-o4-mini-system-card.pdf, April 2025a. Accessed May 31, 2025. OpenAI. Sora 2 is here — openai.com. https://openai.com/index/sora-2/, 2025b. Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research.

J Parker-Holder, P Ball, J Bruce, V Dasagi, K Holsheimer, C Kaplanis, A Moufarek, G Scully, J Shar, J Shi, et al. Genie 2: A large-scale foundation world model. URL: https://deepmind. google/discover/blog/genie-2-a-large-scale-foundation-world-model, 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. Bo Peng, Xinyuan Chen, Yaohui Wang, Chaochao Lu, and Yu Qiao. Conditionvideo: Training-free condition-guided

text-to-video generation. arXiv preprint arXiv:2310.07697, 2023.

RT Pramod, Michael Cohen, Kirsten Lydic, Josh Tenenbaum, and Nancy Kanwisher. Evidence that the brain’s physics engine runs forward simulations of what will happen next. Journal of Vision, 20(11):1521–1521, 2020.

Jan Robine, Marc Höftmann, Tobias Uelwer, and Stefan Harmeling. Transformer-based world models are happy with 100k interactions. In The Eleventh International Conference on Learning Representations, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Runway Research. Introducing gen-3 alpha, 2024. URL https://runwayml.com/research/introducing-gen-3-a lpha. Accessed 2025-10-06.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using

nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based

generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners, 2024. Wendy A Suzuki. Associative learning signals in the brain. Progress in brain research, 169:305–320, 2008. Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable

diffusion. Advances in Neural Information Processing Systems, 36, 2024. Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models, 2024. Edward C Tolman. Cognitive maps in rats and men. Psychological review, 55(4):189, 1948. Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal. Mcvd-masked conditional video diffusion for prediction,

generation, and interpolation. Advances in neural information processing systems, 35:23371–23385, 2022.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314,

- 2025a.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314,

- 2025b.

Weimin Wang, Jiawei Liu, Zhijie Lin, Jiangqiao Yan, Shuo Chen, Chetwin Low, Tuyen Hoang, Jie Wu, Jun Hao Liew, Hanshu Yan, et al. Magicvideo-v2: Multi-stage high-aesthetic video generation. arXiv preprint arXiv:2401.04468, 2024.

Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023a.

Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. arXiv preprint arXiv:2311.17918, 2023b.

World Labs. Generating worlds – world labs, December 2024. URL https://www.worldlabs.ai/blog/generati ng-worlds.

World Labs. Generating Bigger and Better Worlds – worldlabs.ai, September 2025. URL https://www.worldlabs. ai/blog/generating-worlds.

Eric Xing, Mingkai Deng, Jinyu Hou, and Zhiting Hu. Critiques of world models. arXiv preprint arXiv:2507.05169, 2025.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023.

Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023.

[Figure 87]

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, Candace Ross, Adam Polyak, Russell Howes, Vasu Sharma, Puxin Xu, Hovhannes Tamoyan, Oron Ashual, Uriel Singer, Shang-Wen Li, Susan Zhang, Richard James, Gargi Ghosh, Yaniv Taigman, Maryam Fazel-Zarandi, Asli Celikyilmaz, Luke Zettlemoyer, and Armen Aghajanyan. Scaling autoregressive multi-modal models: Pretraining and instruction tuning, 2023.

David Junhao Zhang, Dongxu Li, Hung Le, Mike Zheng Shou, Caiming Xiong, and Doyen Sahoo. Moonshot: Towards controllable video generation and editing with multimodal conditions. arXiv preprint arXiv:2401.01827, 2024.

Jintao Zhang, Xiaoming Xu, Jia Wei, Haofeng Huang, Pengle Zhang, Chendong Xiang, Jun Zhu, and Jianfei Chen.

Sageattention2++: A more efficient implementation of sageattention2. arXiv preprint arXiv:2505.21136, 2025a. Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun. Learning unsupervised world

models for autonomous driving via discrete diffusion. arXiv preprint arXiv:2311.01017, 2023a. Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023b.

Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023c.

Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023d.

Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025b.

Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer2: Llm-enhanced world models for diverse driving video generation. arXiv preprint arXiv:2403.06845, 2024.

Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024.

Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. arXiv preprint arXiv:2311.16038, 2023.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. In Forty-second International Conference on Machine Learning, a.

Linqi Zhou, Aaron Lou, Samar Khanna, and Stefano Ermon. Denoising diffusion bridge models. In The Twelfth International Conference on Learning Representations, b.

Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination, 2024.

