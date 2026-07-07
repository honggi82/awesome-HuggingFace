# arXiv:2603.19227v1[cs.CV]19Mar2026

## Bridging Semantic and Kinematic Conditions with Diffusion-based Discrete Motion Tokenizer

Chenyang Gu1,* , Mingyuan Zhang1,* , Haozhe Xie1,* , Zhongang Cai1 , Lei Yang2 , and Ziwei Liu1,

1S-Lab, Nanyang Technological University 2The Chinese University of Hong Kong https://rheallyc.github.io/projects/motok

0.2

InterControl

###### Perception Planning Control

Perception

Semantics

0.15

Coarse Cnstr.

CrowdMoGen

Generative Models

Conditions

MoTok Decoder

FID(↓)

(diffusion-based) 𝐱𝐱0

Constraints

Text Trajectory Keypoints

0.1

Kinematics

Planning

MaskControl

…

[Figure 1]

[Figure 2]

DDM

0.05

Fine Cnstr.

MoTok

𝐱𝐱𝑡𝑡

Constraints

OR

0

Motion Tokens

[Figure 3]

0 1 2 3

AR

Global Cond.

Local Cond.

Control

Pelvis

→Increased Controlled Joints

(c) Bridging Semantics and Kinematics

- (a) Unified Motion Generation

- (b) Efficient Motion Tokenization

[Figure 4]

c

(d) Low Error & High Fidelity

3 Joints, 25% Temporal Density Control

Left Foot

VQ-VAE

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

5Layers

Steps left,picks up, turns back.

…

0.25|0.0704|0.0640

Left Wrist on XY

Residual-VQ

Multi-scale Residual-VQ

MoTok

Controlled Joint

0.25|0.0244|0.0394

1.5|0.0190|0.0510

~0.47|NA|0.0690

Generated Joint

Ours MoMask MoMask++

InterControl MaskControl MoTok

#Tokens/# Frames|Reconstruction FID↓|Generation FID↓

Fig. 1: (a) A unified Perception–Planning–Control pipeline for conditional motion generation. (b) MoTok enables compact motion tokenization with fewer tokens while maintaining competitive performance. (c) Bridging semantics and kinematics by applying coarse constraints in planning and fine constraints in motion control. MoTok maintains and improves fidelity as more joints are controlled, rather than compromising between controllability and realism. (d) Left-wrist XY trajectories with sparse control (red triangles). MoTok yields the most natural trajectory and best alignment.

Abstract. Prior motion generation largely follows two paradigms: continuous diffusion models that excel at kinematic control, and discrete token-based generators that are effective for semantic conditioning. To combine their strengths, we propose a three-stage framework comprising condition feature extraction (Perception), discrete token generation (Planning), and diffusion-based motion synthesis (Control). Central to this framework is MoTok, a diffusion-based discrete motion tokenizer that decouples semantic abstraction from fine-grained reconstruction by delegating motion recovery to a diffusion decoder, enabling compact single-layer tokens while preserving motion fidelity. For kinematic conditions, coarse constraints guide token generation during planning, while fine-grained constraints are enforced during control through diffusionbased optimization. This design prevents kinematic details from disrupting semantic token planning. On HumanML3D, our method significantly improves controllability and fidelity over MaskControl while using only one-sixth of the tokens, reducing trajectory error from 0.72 cm to 0.08

* Equal Contribution Corresponding Author

cm and FID from 0.083 to 0.029. Unlike prior methods that degrade under stronger kinematic constraints, ours improves fidelity, reducing FID from 0.033 to 0.014.

Keywords: Motion Tokenization · Diffusion Models · Conditional Motion Generation

### 1 Introduction

Human motion generation underpins applications ranging from animation to robotics and embodied agents [39,42]. While recent conditional generative models [9, 36] enable realistic synthesis from high-level semantic inputs, practical scenarios often require additional fine-grained, time-varying kinematic control signals. Effectively integrating such low-level constraints while preserving semantic intent remains a central challenge.

Token-based motion generation [9, 45] compresses continuous motion into discrete tokens for conditional sequence modeling, enabling scalable architectures, flexible conditioning, and the reuse of language-model-style generators. However, existing motion tokenizers [9] often entangle high-level semantics with low-level motion details, requiring high token rates or hierarchical codes to ensure faithful reconstruction. This increases the burden on downstream generators and complicates controllable generation, as fine-grained kinematic condition signals may compete with or override semantic conditioning. In contrast, diffusion models [6, 46] excel at reconstructing continuous motion with smooth dynamics and rich local details. This suggests a division of labor in motion generation, where diffusion handles fine-grained reconstruction while discrete tokens capture semantic abstraction.

Motivated by this insight, we propose a Perception–Planning–Control paradigm for controllable motion generation (Fig. 1a). In Perception, heterogeneous conditions are encoded as either global conditions (e.g., text) that guide the overall motion, or local conditions (e.g., keypoint trajectories) that provide local constraints. In Planning, a token-space planner predicts a discrete motion token sequence under a unified interface supporting both autoregressive (AR) and discrete diffusion (DDM) generators. In Control, the final motion is synthesized via diffusion-based decoding while enforcing fine-grained kinematic constraints during denoising. This decomposition separates high-level planning from low-level kinematics, enabling the same pipeline to generalize across generator architectures and motion generation tasks.

Building on this paradigm, we introduce MoTok, a diffusion-based discrete motion tokenizer that decouples semantic abstraction from low-level reconstruction. MoTok employs a single-layer codebook to produce compact token sequences (Fig. 1b), while delegating motion recovery to a diffusion decoder. This design reduces the token budget for downstream planners and enables decodingtime refinement without forcing discrete tokens to encode fine-grained kinematic details. Furthermore, we propose a condition injection scheme that harmonizes

semantic cues and kinematic constraints by distributing control across stages (Fig. 1c). Kinematic conditions act as coarse constraints during the Planning stage to guide token generation, and as fine-grained constraints during the Control stage via optimization-based guidance in diffusion denoising. This coarse-tofine design prevents low-level kinematic details from interfering with token-space planning, avoiding a compromise between controllability and realism.

We evaluate our framework on text-and-trajectory controllable motion generation on HumanML3D [10]. Compared with MaskControl [29], our method substantially improves both controllability and fidelity, reducing trajectory error from 0.72 cm to 0.08 cm and FID from 0.083 to 0.029 while using only one-sixth of the tokens. As shown in Fig. 1, prior methods [4,29,38] degrade as more joints are controlled, whereas ours improves motion fidelity under stronger constraints. Beyond controllable generation, MoTok also improves standard textto-motion performance on HumanML3D under aggressive compression, achieving lower FID than strong token-based baselines with substantially fewer tokens (e.g., 0.039 vs. 0.045 with one-sixth tokens).

The contributions are summarized as follows:

- 1. We propose a three-stage Perception–Planning–Control paradigm for controllable motion generation that supports both autoregressive (AR) and discrete diffusion (DDM) planners under a unified interface.
- 2. We introduce MoTok, a diffusion-based discrete motion tokenizer that decouples semantic abstraction from low-level reconstruction by delegating motion recovery to diffusion decoding, enabling compact single-layer tokens with a dramatically reduced token budget.
- 3. We develop a coarse-to-fine conditioning scheme that injects kinematic signals as coarse constraints during token planning and enforces fine-grained constraints during diffusion denoising, improving controllability and fidelity.

### 2 Related Work

Motion Generative Model. Early motion generation research primarily focused on unconditional settings, with classical methods such as PCA [25] and Motion Graphs [23], followed by learning-based generative models including VAEs [12, 26], implicit functions [5], GANs [3, 13], and normalizing flows [14]. Subsequent text- and action-conditioned approaches [1,10,27,28,35] aligned motion and language representations via latent-space objectives, but often suffered from limited motion fidelity. Diffusion-based methods [20,36,46,47] significantly improved generation quality through iterative denoising [15], yet incur slow inference due to operating on raw motion sequences, while latent diffusion [6,33] accelerates generation at the cost of fine-grained details and editability. Autoregressive token-based models [16, 45, 49] further enhance controllability but introduce high computational overhead and limited bidirectional dependency modeling. Motivated by recent advances in masked modeling [9,29,30], recent works explore efficient and editable motion generation through discrete representations. MaskControl [29] designs a differentaible sampling strategy for discrete motion diffusion model, which can achieve spatio-temporal low-level control.

###### (a) Diffusion-based Discrete Motion Tokenizer (b) Unified Conditional Motion Generation

Motion Token

Masked Token

Motion Embed

Condition embed

𝐞𝐞𝑐𝑐

𝐬𝐬1:𝑇𝑇

AdaIN

scaleshift

Decoder 𝒟𝒟

MLP

⊕ ⊕ ⊕ ⊕

⊕

⨀

gate

𝐪𝐪1:𝑁𝑁

𝑡𝑡 Transformer

Transformer

𝐱𝐱𝑐𝑐

Add noise

𝒞𝒞

× 𝑁𝑁

𝐳𝐳1:𝑁𝑁

[Figure 5]

Codebook

Temporal Conv

Discrete Diffusion Planning

Autoregressive Planning

𝐞𝐞1:𝑇𝑇

𝐡𝐡1:𝑁𝑁

⊕ MLP

“walks forward” “sits down” …

Inference Time 𝐌𝐌1:𝑁𝑁𝑠𝑠 𝐜𝐜𝑔𝑔 ℰ𝑔𝑔

Encoder ℰ

ℰ𝑠𝑠

𝐜𝐜1:𝑇𝑇𝑠𝑠

𝐌𝐌𝑔𝑔

Linear

ℒ𝑐𝑐𝑐𝑐𝑐𝑐𝑐𝑐

Encoder Encoder

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Local Conditions

Global Conditions

ℒdiff

𝐱𝐱0 = 𝑓𝑓𝜙𝜙 𝐱𝐱𝑐𝑐,𝑡𝑡,𝐬𝐬1:𝑇𝑇

𝜽𝜽1:𝑇𝑇

- Fig. 2: Overview of MoTok and the unified motion generation framework. (a) MoTok factorizes motion representation into compact discrete tokens and diffusionbased reconstruction by decoding tokens into per-frame conditioning for conditional diffusion. (b) A unified conditional generation framework built on MoTok supports both discrete diffusion and autoregressive planners, integrating global and local conditions in a generator-agnostic manner.

Motion Tokenizer. Early discrete text-to-motion methods such as TM2T [11] introduce motion tokens by framing motion as a foreign language and learning text–motion translation with VQ-based tokenizers. Subsequent works advance along two main directions. One line focuses on tokenizer and generator design, improving convolutional tokenizers (e.g., T2M-GPT [45]), modeling full-body structure more explicitly (e.g., HumanTOMATO [21]), or extending tokenization to the spatio-temporal domain (e.g., MoGenTS [44]), often at the cost of increased modeling complexity. The other line explores improved quantization schemes. MoMask [9] introduces residual vector quantization to reduce reconstruction error but substantially increases token count and requires specialized generators, while later variants such as ScaMo [22] and MoMask++ [8] investigate alternative or hierarchical quantization strategies to balance efficiency and accuracy. Despite these advances, existing approaches still face a fundamental trade-off between token efficiency and generation quality, and remain limited in supporting fine-grained, low-level control.

### 3 Our Approach

#### 3.1 Problem Formulation

Motion Representation. A motion sequence is denoted as θ1:T = {θt}Tt=1, where T is the sequence length and θt ∈ RD represents the motion state at time t. The motion state can be instantiated using standard skeleton-based representations commonly adopted in text-to-motion benchmarks (e.g., joint rotations or positions with auxiliary signals), while our framework remains agnostic to the specific choice of parameterization.

Discrete Token Sequence. Each motion sequence is encoded as a shorter discrete token sequence z1:N = {zn}Nn=1, where each token zn ∈ 1,...,K indexes a shared codebook of size K. The token compression ratio is defined as ρ = T/N. A central goal of this work is to achieve high-quality motion generation under aggressive compression (large ρ), thereby reducing the sequence modeling burden of downstream generators.

Conditions and Taxonomy. Heterogeneous conditioning signals c are categorized into two types: 1) Global conditions cg provide sequence-level guidance without requiring frame-wise alignment, such as text descriptions or style labels; and 2) local conditions cs1:T = {cst}Tt=1 are aligned with the motion timeline and specify kinematic control signals, including target root trajectories, keyframes, contact hints, or motion rhythm. This taxonomy is used throughout the method to integrate semantic guidance and kinematic constraints in a unified and generator-agnostic manner.

Task Instantiations. Our formulation supports a range of conditional motion generation tasks, with additional task definitions and experimental results provided in the supplementary material. In the main text, we consider two representative settings: 1) text-to-motion, where text provides the global condition cg and no local conditions cs1:T is given and 2) text-and-trajectory control, where text serves as cg and a target trajectory is specified as the local conditions cs1:T, requiring the generated motion to follow the trajectory while maintaining semantic consistency.

#### 3.2 Diffusion-based Discrete Motion Tokenizer

MoTok is a diffusion-based discrete motion tokenizer that factorizes motion representation into a compact discrete code sequence and a diffusion decoder for fine-grained reconstruction. Unlike conventional VQ-VAE tokenizers that directly decode continuous motion from discrete codes, MoTok first maps the discrete codes to a per-frame conditioning signal and then employs a conditional diffusion model to reconstruct motion details. By explicitly offloading fine-grained reconstruction to diffusion-based decoding, discrete tokens are freed to focus on semantic structure, enabling a substantially reduced token budget.

As shown Fig. 2a, MoTok consists of three components: 1) a convolutional encoder E(·) that produces a temporally downsampled latent sequence; 2) a vector quantizer Q(·) that maps latents to discrete codes; and (3) a decoder with diffusion-based reconstruction, comprising a convolutional decoder D(·) and a conditional diffusion model Pϕ(·).

Convolutional Encoder. A convolutional encoder E(·) is used to obtain a compressed latent representation. Given a motion sequence θ1:T, latent features are extracted through progressive temporal downsampling:

h1:N = E(θ1:T), h1:N ∈ RN×d, (1)

where d denotes the latent dimension. The temporal length N is determined by the encoder downsampling factor r.

Vector Quantizer. A vector quantization (VQ) module Q(·) is applied to discretize the latent sequence h1:N. Let the codebook be C = {c}Kk=1, where K = 1024 and each code ck ∈ Rd. Each latent vector is assigned to its nearest codebook entry:

∥hn − ck∥22, qn = cz

, (2)

zn = arg min

n

k∈{1,...,K}

yielding a discrete token sequence z1:N and the quantized latents q1:N.

Decoder with Diffusion-based Reconstruction. Rather than directly regressing θ1:T from the quantized latents q1:N, MoTok decodes q1:N into a perframe conditioning sequence and reconstructs motion using conditional diffusion. Specifically, a convolutional decoder D(·) upsamples the quantized latents as:

s1:T = D(q1:N), s1:T ∈ RT×d, (3)

where s1:T serves as the conditioning signal for diffusion-based reconstruction. We then define a conditional diffusion decoder as a reverse diffusion process Pϕ(·) parameterized by a neural denoiser fϕ. Concretely, fϕ predicts the clean motion xˆ0 from a noisy input xt at diffusion timestep t:

xˆ0 = fϕ(xt,t,s1:T). (4)

This prediction defines the reverse transitions of the diffusion model, yielding a distribution pϕ(xt−1 | xt,s1:T). At inference, Pϕ samples the reconstructed motion by iteratively applying reverse steps from xT ∼ N(0,I) to obtain xˆ0. Architecturally, fϕ first projects xt to the latent dimension via a linear layer, followed by a stack of processing blocks. Each block contains a residual 1D convolution module for enhanced temporal modeling and an MLP that injects conditioning embeddings e1:T into motion features via an AdaIN-style transformation, where e1:T combines timestep embeddings with conditioning signals s1:T. This diffusion-based decoding provides a natural interface for enforcing additional fine-grained constraints during reconstruction (e.g., trajectories or joint-level hints), as such constraints can be applied throughout the denoising process rather than solely being imposed at the level of discrete token prediction.

Training Objectives. MoTok is trained end-to-end using a combination of a diffusion reconstruction objective and a VQ commitment loss, following the diffusion training strategy of MAR [18]. During diffusion training, a timestep t is sampled and the conditional denoising objective is optimized:

Ldiff = Et,ϵ ℓ x ˆ0,x0 , (5)

where ℓ(·,·) denotes the Smooth-ℓ1 loss and x0 = θ1:T is the clean motion sequence. In addition, the VQ commitment loss returned by the quantizer is included with weight λcommit = 0.02, yielding the overall training objective:

L = Ldiff + λcommitLcommit. (6)

#### 3.3 Unified Conditional Motion Generation

As shown in Fig. 2, MoTok enables a unified conditional motion generation pipeline by decoupling planning in discrete token space from control in diffusion-based decoding. Given a condition set c = {cg,cs1:T}, a token generator first produces a discrete sequence z1:N, which is then decoded into a continuous motion xˆ0 via diffusion conditioned on features derived from MoTok. This formulation supports both discrete diffusion and autoregressive token generators through a shared conditioning interface.

Conditions are categorized by their temporal characteristics into 1) global conditions cg, which provide sequence-level guidance without frame-wise alignment (e.g., text descriptions), and 2) local conditions cs1:T, which are aligned with the motion timeline and specify fine-grained control signals (e.g., target trajectories). Global conditions are encoded by Eg(·) into a sequence-level feature Mg ∈ Rd and used as a dedicated token during discrete planning, while local conditions are encoded by Es(·) into a feature sequence Ms1:N ∈ RN×d aligned with the token length N.

Planning in Discreate Token Space. Token-space planning generates discrete motion tokens under heterogeneous conditions and supports both discrete diffusion and autoregressive generators through a shared planning interface.

Discrete Diffusion Planning follows the masked-token diffusion paradigm introduced by MoMask [9], where subsets of tokens are iteratively predicted conditioned on observed tokens and external conditions. To inject conditions in a unified manner, a token embedding sequence of length N + 1 is constructed, with the first position reserved for the global condition feature and the remaining N positions corresponding to motion tokens:

H0 = [Mg; h1;...;hN ], (7)

where hn denotes the learnable embedding of token zn or a learned [MASK] embedding for masked positions. local conditions features are incorporated by additive fusion with positional embeddings at the motion-token positions:

H0[1 + n] ← H0[1 + n] + Msn + pn, n = 1,...,N, (8)

where pn is the standard positional embedding. Mg at position 1 attends to all motion tokens, providing sequence-level guidance throughout denoising.

Autoregressive Planning follows the same interface, with the global condition occupying the first position and motion tokens generated sequentially in a causal manner, as in T2M-GPT [45]. Due to the one-step shift inherent to nexttoken prediction, the local conditions embedding for the first token is added to the global-conditioning position, while the embedding for each subsequent token is added to the preceding token position. This design preserves temporal alignment of control signals and allows MoTok to be integrated into autoregressive backbones with minimal modification.

Classifier-free Guidance (CFG) is applied to token-space planning and extended to multiple conditions via alternating guidance pairs, following ReMoDiffuse [47]. Let ϵθ(z;c) denote the sampling output of the token generator under

conditions c = cg,cs1:T. For single-condition CFG, conditional and unconditional predictions are formed with ccond = g,∅ and cuncond = ∅,∅, and are combined as

ϵcfg = ϵθ(z;cuncond) + w ϵθ(z;ccond) − ϵθ(z;cuncond) , (9)

where w is the guidance scale. When both semantic and trajectory conditions are present, fully dropping conditions in the unconditional branch may bias generation toward a single modality. To balance semantic guidance and control fidelity, two CFG pairs are alternated with equal probability:

- (A) ccond = {g,s}, cuncond = {∅,s}, (10)
- (B) ccond = {g,s}, cuncond = {g,∅}. (11)

The same CFG combination rule is applied. This alternating strategy enables effective multi-condition guidance during planning without introducing additional networks or training objectives.

Control in Diffusion Decoding. After token-level planning, discrete tokens are decoded by MoTok into a per-frame conditioning sequence s1:T, and motion is reconstructed via conditional diffusion. Fine-grained control is enforced directly during denoising by optimizing an auxiliary control objective. At diffusion step k, given the current estimate xˆk of the full motion sequence, a control loss Lctrl(xˆk,cs1:T) measures deviation from local conditions (e.g., trajectory adherence), and the denoising update is refined via

xˆk ← xˆk − η∇xˆkLctrl(xˆk,cs1:T), (12)

where η controls the refinement strength. Enforcing constraints at the continuousmotion level enables precise low-level control without burdening the discrete planner with high-frequency details, and is critical for achieving both low trajectory error and improved motion fidelity when semantic and low-level conditions are jointly applied.

#### 3.4 Instantiation for Different Tasks

The unified framework is instantiated for two representative tasks, with additional tasks and results provided in the Appendix. Unless otherwise specified, global conditions are encoded into sequence-level features, while local conditions are encoded into token-aligned features. Both discrete diffusion and autoregressive token generators are supported through the same conditioning interface.

Text-to-Motion. In text-to-motion generation, conditioning is purely global. Given a text prompt t, a sequence-level embedding is extracted using a pretrained CLIP text encoder [32]:

Mg = Etext(t) ∈ Rd. (13)

Text and Trajectory Control. For joint text-and-trajectory generation, a global text embedding is combined with a time-synchronized trajectory embedding. Given a text prompt t and a target trajectory τ1:T ∈ RT×J×3, where J denotes the number of joints, the text prompt is encoded into a global feature Mg as in Eq. 13, while the trajectory is encoded into a token-aligned sequence using the same convolutional encoder as in MoTok:

Ms1:N = Etraj(τ1:T) ∈ RN×d. (14)

The trajectory features are injected as local conditions during token-space planning and further enforced during diffusion decoding through refinement. This design allows semantic planning to occur in token space, while precise trajectory adherence is handled at the continuous-motion level.

### 4 Experiments

#### 4.1 Experimental Setup

Datasets. Experiments are conducted on HumanML3D [10] and KIT-ML [31], which are widely used paired text–motion benchmarks. Each dataset provides natural language descriptions for every motion sequence, together with a standardized skeleton-based motion representation.

Text Conditioning. Text conditions are treated as global conditions and encoded into a sequence-level feature using a pretrained CLIP text encoder (ViTB/32) [32], producing a 512-dimensional embedding. This embedding serves as the global condition token during token-space planning for both discrete diffusion and autoregressive generators.

Trajectory Conditioning. Trajectory conditions are treated as local conditions and encoded into token-aligned feature sequences using a convolutional encoder that mirrors the motion encoder in MoTok. The encoder downsamples the trajectory to match the token length, with dropout (0.1) applied for robustness and classifier-free guidance.

Training. MoTok is evaluated with two token-space planning backbones: 1) a MoMask-style [9] discrete diffusion planner (dim 384, 6 layers) and 2) a T2MGPT-style [45] autoregressive transformer (dim 768, 9 layers). During training, condition dropout is applied with probability 0.1, and random token replacement with probability 0.1 (DDM) / 0.2 (AR). Four variants are reported: MoTokDDM-4/2 and MoTok-AR-4/2, where DDM/AR denote the generative model and 4/2 the temporal compression ratio.

Inference. During reconstruction and controllable generation, discrete tokens are decoded by MoTok and converted into continuous motion using conditional diffusion. When trajectory control is enabled, decoding-time refinement is applied by incorporating an auxiliary control objective at each denoising step, following the strategy of InterControl [38].

Table 1: Controllable motion generation quantitative results on HumanML3D test set. Note that “Traj. Err.”, “Loc. Err.”, and “Avg. Err.” denote “Trajectory Error”, “Localization Error”, and “Average Error”, respectively. ↑ (↓) indicates that the values are better if the metric is larger (smaller). ‘→’ means closer to real data is better. Red and Blue highlight the best and second-best results, respectively.

R-Precision ↑ (Top-3)

Foot Skating Ratio ↓

Traj. Err.↓ (50cm)

Loc. Err. ↓ (50cm)

Avg. Err. ↓ (m)

Method FID ↓

Diversity →

Real Motion 0.002 0.797 9.503 0.000 0.0000 0.0000 0.0000

PriorMDM [34] 0.475 0.583 9.156 0.0897 0.3457 0.2132 0.4417 GMD [17] 0.576 0.665 9.206 0.109 0.0931 0.0321 0.1439 OmniControl [43] 0.218 0.687 9.422 0.0547 0.0387 0.0096 0.0338 InterControl [38] 0.159 0.671 9.482 0.0729 0.0132 0.0004 0.0496 CrowdMoGen [4] 0.132 0.784 9.109 0.0762 0.0000 0.0000 0.0196 MaskControl [29] 0.061 0.809 9.496 0.0547 0.0000 0.0000 0.0098 MoTok-AR-2 0.046 0.767 9.516 0.0489 0.0000 0.0000 0.0052 MoTok-DDM-2 0.035 0.786 9.440 0.0453 0.0000 0.0000 0.0049 MoTok-AR-4 0.046 0.777 9.408 0.0605 0.0000 0.0000 0.0051 MoTok-DDM-4 0.029 0.794 9.476 0.0548 0.0000 0.0000 0.0049

Pelvis

OmniControl [43] 0.310 0.693 9.502 0.0608 0.0617 0.0107 0.0404 InterControl [38] 0.178 0.669 9.498 0.0968 0.0403 0.0031 0.0741 CrowdMoGen [4] 0.147 0.781 9.461 0.0829 0.0000 0.0000 0.0028 MaskControl [29] 0.083 0.805 9.395 0.0545 0.0000 0.0000 0.0072 MoTok-AR-2 0.056 0.744 9.586 0.0430 0.0000 0.0000 0.0009 MoTok-DDM-2 0.025 0.779 9.756 0.0436 0.0000 0.0000 0.0008 MoTok-AR-4 0.040 0.756 9.380 0.0496 0.0000 0.0000 0.0009 MoTok-DDM-4 0.029 0.777 9.297 0.0508 0.0000 0.0000 0.0008

RandomOne

InterControl [38] 0.184 0.670 9.410 0.0948 0.0475 0.0030 0.0911 CrowdMoGen [4] 0.178 0.777 9.149 0.0865 0.0000 0.0000 0.0027 MoTok-AR-2 0.037 0.765 9.452 0.0422 0.0000 0.0000 0.0006 MoTok-DDM-2 0.016 0.775 9.711 0.0456 0.0000 0.0000 0.0006 MoTok-AR-4 0.041 0.762 9.481 0.0515 0.0000 0.0000 0.0006 MoTok-DDM-4 0.022 0.784 9.562 0.0521 0.0000 0.0000 0.0006

RandomTwo

InterControl [38] 0.199 0.673 9.352 0.0930 0.0487 0.0026 0.0969 CrowdMoGen [4] 0.192 0.778 9.169 0.0871 0.0000 0.0000 0.0030 MoTok-AR-2 0.035 0.772 9.278 0.0482 0.0000 0.0000 0.0008 MoTok-DDM-2 0.014 0.786 9.519 0.0480 0.0000 0.0000 0.0007 MoTok-AR-4 0.045 0.769 9.519 0.0529 0.0000 0.0000 0.0008 MoTok-DDM-4 0.022 0.788 9.474 0.0523 0.0000 0.0000 0.0007

RandomThree

#### 4.2 Text and Trajectory Control

- Table 1 shows that MoTok achieves substantially improved controllable generation on HumanML3D. Compared to the strongest DDM-based baseline, MoTokDDM-4 reduces FID from 0.061 to 0.027 while nearly halving trajectory error (0.098 to 0.049) under the “Pelvis” setting. The improvement is even larger in the Random One” setting, where MoTok-DDM-2 reduces FID from 0.083 to 0.025 and average error from 0.0072 to 0.0008. Notably, these gains are achieved with substantially fewer tokens: MoTok uses only 1/6 of MaskControl’s token budget in the first setting and 1/3 in the second, demonstrating its superior efficiency.

While prior methods such as MaskControl [29] degrade in motion quality when trajectory constraints are introduced (FID 0.061 vs. 0.045 from MoMask in Table 2), both MoTok-DDM and MoTok-AR achieve lower FID under joint text-and-trajectory control than in text-only generation. This suggests that tra-

InterControl MaskControl MoTok

[Figure 11]

[Figure 12]

[Figure 13]

Top View Top View Top View

Top View Top View Top View

Controlled Joints (100% Frames): Pelvis | Left Foot

Text: Steps left, picks up, turns back.

[Figure 14]

[Figure 15]

InterControl MaskControl MoTok

[Figure 16]

Side View Side View Side View

Side View Side View Side View

Controlled Joints (100% Frames): Pelvis | Left Foot | Left Wrist

Text: Jumps left, jumps right.

[Figure 17]

[Figure 18]

[Figure 19]

InterControl MaskControl MoTok

Top View Top View Top View

Top View Top View Top View

Controlled Joints (25% Frames): Pelvis | Left Foot | Left Wrist

Text: Walks forward, turns back, walks forward.

- Fig. 3: Visual comparison with state-of-the-art methods for any-joint anyframe control. The right panels show trajectory views of two of the controlled joints. Red indicates the control signal and blue indicates the generated motion.

jectory information acts as complementary guidance rather than a competing constraint. This improvement stems from 1) a condition hierarchy that separates global semantic constraints from fine-grained trajectory constraints during token-space planning, and 2) decoding-time diffusion guidance that enforces low-level constraints without overloading the discrete planner.

Figure 3 presents a qualitative comparison across several methods. As shown, MoTok follows the provided trajectory most faithfully while producing smooth overall motion. Although MaskControl also yields high-quality motions, its trajectory alignment is limited by the capacity of its tokenizer, making it difficult to match fine-grained trajectory details.

Overall, MoTok better aligns semantic planning with low-level control, producing motions that are both more trajectory-accurate and closer to the real motion distribution. Although the AR variants perform slightly worse than the DDM versions, likely due to weaker global planning capacity, they still outperform prior methods, demonstrating the generality of the MoTok paradigm.

#### 4.3 Text-to-Motion Generation

As shown in Table 2, we evaluate MoTok on two standard text-to-motion benchmarks, HumanML3D and KIT-ML, where it achieves strong overall performance, particularly in motion realism measured by FID.

- Table 2: Quantitative results on the HumanML3D and KIT-ML test sets. For each metric, we conduct the evaluation 20 times and report the average with a 95% confidence interval. The right arrow (→) signifies that results closer to real motion are better. Red and Blue highlight the best and second-best results, respectively. Notably, MoTok-DDM operates with a token budget that is only one-sixth of that used by MoMask, while maintaining comparable or improved performance across most metrics.

Method

Top-1 ↑ R-PrecisionTop-2 ↑ ↑ Top-3 ↑ FID ↓ MM-Dist ↓ Diversity → MModality ↑

HumanML3D

Real Motion 0.511±.003 0.703±.003 0.797±.002 0.002±.000 2.974±.008 9.503±.065 MotionDiffuse [46] 0.491±.001 0.681±.001 0.782±.001 0.630±.001 3.113±.001 9.410±.049 1.553±.042 ReMoDiffuse [47] 0.510±.005 0.698±.006 0.795±.004 0.103±.004 2.974±.016 9.018±.075 1.795±.043 T2M-GPT [45] 0.492±.003 0.679±.002 0.775±.002 0.141±.005 3.121±.009 9.722±.082 1.831±.048 MoMask [9] 0.521±.002 0.713±.002 0.807±.002 0.045±.002 2.958±.008 - 1.241±.040 MotionLCM [7] 0.502±.003 0.698±.002 0.798±.002 0.304±.012 3.012±.007 9.607±.066 2.259±.092 MoTok-AR-2 0.483±.003 0.670±.002 0.770±.002 0.061±.005 3.128±.011 9.675±0.09 2.237±.050 MoTok-DDM-2 0.512±.002 0.705±.002 0.799±.002 0.033±.002 2.981±.010 9.523±0.09 2.639±.079 MoTok-AR-4 0.491±.004 0.676±.002 0.773±.003 0.053±.004 3.106±.014 9.691±0.07 2.183±.048 MoTok-DDM-4 0.500±.003 0.693±.002 0.793±.002 0.039±.002 3.030±.007 9.411±.078 2.639±.063

KIT-ML

Real Motion 0.424±.005 0.649±.006 0.779±.006 0.031±.004 2.788±.012 11.08±.097 MotionDiffuse [46] 0.417±.004 0.621±.004 0.739±.004 1.954±.062 2.958±.005 11.10±.143 0.730±.013 ReMoDiffuse [47] 0.427±.014 0.641±.004 0.765±.055 0.155±.006 2.814±.012 10.80±.105 1.239±.028 T2M-GPT [45] 0.416±.006 0.627±.006 0.745±.006 0.514±.029 3.007±.023 10.92±.108 1.570±.039 MoMask [9] 0.433±.007 0.656±.005 0.781±.005 0.204±.011 2.779±.022 - 1.131±.043 MoTok-AR-2 0.423±.006 0.633±.005 0.750±0.006 0.211±.003 3.048±.032 11.31±.101 1.154±.041 MoTok-DDM-2 0.427±.006 0.642±.006 0.762±.006 0.144±.005 2.856±.033 10.92±.163 1.325±.051 MoTok-AR-4 0.418±.007 0.619±.007 0.734±.005 0.297±.002 3.075±.036 11.15±.085 1.102±.039 MoTok-DDM-4 0.421±.007 0.638±.007 0.760±.006 0.192±.016 2.896±.028 10.81±.115 1.295±.046

On HumanML3D, MoTok-DDM consistently outperforms the MoMask baseline under significantly smaller token budgets: with only one-sixth of the tokens, MoTok-DDM-4 attains a lower FID than MoMask (0.039 vs. 0.045), and with higher capacity, MoTok-DDM-2 further reduces FID to 0.033, the best among all compared methods. MoTok also transfers effectively to autoregressive planning. Under the same token budget, MoTok-AR-4 reduces the FID of T2MGPT by nearly threefold (0.053 vs. 0.141), indicating a substantially reduced modeling burden for token-space generators.

On KIT-ML, MoTok achieves the lowest FID of 0.144, improving over the strongest baseline (0.155) while maintaining competitive performance on other metrics. Overall, these results show that MoTok produces high-quality motions with substantially fewer tokens, supporting the effectiveness of diffusion-based decoding for compact and semantically expressive motion representations.

4.4 Ablation Study

- Table 3 presents an ablation study of tokenizer configurations under both discrete diffusion (DDM) and autoregressive (AR) planners, analyzing decoder design, latent dimension, and temporal downsampling rate. Beyond raw performance differences, the results reveal a consistent pattern: tokenizer effectiveness is largely determined by how temporal structure is handled under noisy generation, rather than by reconstruction fidelity alone.

##### Decoder Design. Across all settings, decoder performance follows a clear hierarchy: diffusion-based decoders outperform plain convolutional decoding, and

- Table 3: Ablation study of tokenizer configurations across DDM and AR planners. Latent Dim is the codebook dimension, and Downrate denotes the temporal compression ratio (frames per latent). DiffusionHead (DH) applies independent, perframe diffusion decoding using AdaIN-conditioned MLP blocks, while DiffusionConv (DC) augments it with residual 1D convolutions inserted between MLP blocks to model local temporal dependencies (kernel size ∈ {3, 5, 7, 9}). “DR”, “Recon.”, “Ctrl.”, and “Err.” stand for downsample rate, reconstruction, control, and error, respectively. Red and Blue highlight the best and second-best results, respectively.

Kernel Size

Recon. FID ↓

DDM Planner AR Planner T2M FID ↓ Ctrl. FID ↓ Ctrl. Err. ↓ T2M FID ↓ Ctrl. FID ↓ Ctrl. Err. ↓

Latent Dim

#

Decoder DR

- 1 768

Conv

4

3 0.0704 0.0640 0.0601 0.2027 0.0935 0.0874 0.2604

- 2 DH N/A 0.0396 0.0678 0.0373 0.0051 0.0686 0.0739 0.0053

- 3 768

DC 4

3 0.0259 0.0500 0.0379 0.0047 0.0671 0.0469 0.0049

- 4 768 5 0.0244 0.0394 0.0292 0.0049 0.0535 0.0461 0.0051
- 5 768 7 0.0288 0.0458 0.0279 0.0047 0.0690 0.0454 0.0050
- 6 768 9 0.0400 0.0875 0.0400 0.0051 0.0720 0.0590 0.0054
- 7 512 5 0.0177 0.0505 0.0457 0.0049 0.0780 0.0649 0.0052
- 8 384 5 0.0292 0.0669 0.0428 0.0049 0.0581 0.0692 0.0051

- 9

768 DC

1

5 0.0207 0.1849 0.0734 0.0050 0.1379 0.0630 0.0054

- 10 7 0.0188 0.1551 0.0542 0.0052 0.1465 0.0597 0.0056

- 11 2

5 0.0240 0.0459 0.0395 0.0050 0.0773 0.0474 0.0052

- 12 7 0.0228 0.0332 0.0353 0.0049 0.0617 0.0468 0.0052

- 13 8

5 0.0429 0.0707 0.0304 0.0051 0.0682 0.0527 0.0052

- 14 7 0.0492 0.0816 0.0362 0.0054 0.0533 0.0708 0.0056

- 15 16

5 0.0600 0.0875 0.0474 0.0054 0.0788 0.1043 0.0054

- 16 7 0.0894 0.1748 0.0818 0.0054 0.1580 0.1486 0.0055

introducing explicit temporal modeling further improves over frame-wise diffusion. With the same compression ratio (downrate = 4), a convolutional decoder yields poor reconstruction (Recon FID = 0.0704), whereas diffusion-based decoding reduces Recon FID to below 0.04, demonstrating its advantage in recovering fine-grained motion details. A lightweight DiffusionHead [18] achieves strong reconstruction and is sufficient for controllable generation, but offers limited benefit for the planning stage. This gap reflects the difference between reconstruction on clean tokens and generation under stochastic noise. In contrast, DiffusionConv introduces local temporal correlations via residual 1D convolutions, consistently improving generation fidelity and controllability under both planners.

Latent Dimension. Fixing the decoder and temporal compression (downrate = 4), we vary the codebook dimension in Table 3 (rows 3–8). A latent dimension of d = 768 provides sufficient capacity for accurate reconstruction and stable token-space planning, whereas smaller dimensions reduce expressiveness and degrade both generation fidelity and controllability.

Temporal Downsampling Rate. Varying the temporal downsampling rate under a fixed decoder and latent dimension (Table 3, rows 9–16) reveals a nonmonotonic trade-off. Low downrates produce dense tokens that are harder to model under noisy generation, while excessive compression removes essential structure. Moderate downsampling (downrate 2 or 4) consistently yields the best balance between reconstruction quality and downstream planning.

Kernel Size in Temporal Diffusion. Fixing the latent dimension and downsampling rate (Table 3, rows 3–6), we observe that kernel size has a non-trivial

- Table 4: Effect of low-level control injection location. Comparison of injecting low-level control signals in the generator (token-space planning), the tokenizer decoder (diffusion decoding), evaluated with DDM and AR planners under “Pelvis” setting.

Low-level Condition DDM Planner AR Planner Generator Tok. Decoder Ctrl. FID ↓ Ctrl. Err. ↓ Ctrl. FID ↓ Ctrl. Err. ↓

✓ ✓ 0.029 0.0049 0.046 0.0051

✓ ✗ 0.028 0.2170 0.037 0.2770

- ✗ ✓ 0.365 0.0056 0.447 0.0063
- ✗ ✗ 0.039 0.6239 0.053 0.6009

effect on performance. A moderate temporal receptive field (k = 5) yields the best overall results, whereas larger kernels over-smooth sparse tokens and degrade generation quality. We also find that the optimal kernel size is closely tied to the temporal compression ratio. When the compression ratio is low (DR = 1 or 2), where the number of tokens is relatively large, a kernel size of 7 performs better than 5. However, under higher compression (fewer tokens), a kernel size of

- 5 yields better results. This suggests that a moderate kernel size strikes a better balance between modeling temporal dependencies and avoiding over-smoothing.

Dual-path Low-level Conditioning. As shown in Table 4, low-level conditioning is necessary and cannot be removed without harming controllable generation. Injecting low-level signals only in the generator forces token-space planning to simultaneously satisfy semantics and precise constraints, often increasing control error and degrading motion fidelity. Injecting them only in the tokenizer decoder lacks planning-time guidance and leads to unstable or weaker adherence. The best results are achieved only when low-level conditions are applied in both stages, where the generator provides coarse, condition-aware planning and the diffusion decoder enforces fine-grained constraints during denoising.

### 5 Conclusion

In this work, we bridge the strengths of continuous diffusion models for kinematic control and discrete token-based generators for semantic conditioning. We introduce a three-stage Perception–Planning–Control paradigm that encodes conditions, plans discrete motion tokens, and synthesizes motion via diffusionbased decoding with fine-grained constraint enforcement. Within this framework, we proposed MoTok, a diffusion-based discrete motion tokenizer that offloads low-level reconstruction to diffusion decoding, enabling compact and semantically informative tokens. Experiments on HumanML3D and KIT-ML demonstrate strong performance under aggressive token compression and significant gains on text-and-trajectory controllable generation, reducing trajectory error from 0.72 cm to 0.08 cm while using only one-sixth of the tokens.

Acknowledgements. This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOET2EP20221-0012, MOET2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

### References

- 1. Ahuja, C., Morency, L.: Language2Pose: Natural language grounded pose forecasting. In: 3DV (2019) 3
- 2. Athanasiou, N., Cseke, A., Diomataris, M., Black, M.J., Varol, G.: MotionFix: text-driven 3D human motion editing. In: SIGGRAPH. pp. 44:1–44:11 (2024) 21
- 3. Barsoum, E., Kender, J.R., Liu, Z.: HP-GAN: probabilistic 3D human motion prediction via GAN. In: CVPRW (2018) 3
- 4. Cao, Y., Guo, X., Zhang, M., Xie, H., Gu, C., Liu, Z.: CrowdMoGen: zero-shot text-driven collective motion generation. IJCV 134(1), 29 (2026) 3, 10, 20
- 5. Cervantes, P., Sekikawa, Y., Sato, I., Shinoda, K.: Implicit neural representations for variable length human motion generation. In: ECCV (2022) 3
- 6. Chen, X., Jiang, B., Liu, W., Huang, Z., Fu, B., Chen, T., Yu, G.: Executing your commands via motion diffusion in latent space. In: CVPR (2023) 2, 3
- 7. Dai, W., Chen, L., Wang, J., Liu, J., Dai, B., Tang, Y.: MotionLCM: Real-time controllable motion generation via latent consistency model. In: ECCV (2024) 12
- 8. Guo, C., Hwang, I., Wang, J., Zhou, B.: SnapMoGen: human motion generation from expressive texts. In: NeurIPS (2025) 4
- 9. Guo, C., Mu, Y., Javed, M.G., Wang, S., Cheng, L.: MoMask: Generative masked modeling of 3D human motions. In: CVPR (2024) 2, 3, 4, 7, 9, 12
- 10. Guo, C., Zou, S., Zuo, X., Wang, S., Ji, W., Li, X., Cheng, L.: Generating diverse and natural 3D human motions from text. In: CVPR (2022) 3, 9
- 11. Guo, C., Zuo, X., Wang, S., Cheng, L.: TM2T: stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In: ECCV (2022) 4, 25
- 12. Guo, C., Zuo, X., Wang, S., Zou, S., Sun, Q., Deng, A., Gong, M., Cheng, L.: Action2Motion: Conditioned generation of 3D human motions. In: ACM MM (2020) 3
- 13. Harvey, F.G., Yurick, M., Nowrouzezahrai, D., Pal, C.J.: Robust motion inbetweening. ACM TOG 39(4), 60 (2020) 3
- 14. Henter, G.E., Alexanderson, S., Beskow, J.: MoGlow: probabilistic and controllable motion synthesis using normalising flows. ACM TOG 39(6), 236:1–236:14 (2020) 3
- 15. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS

(2020) 3

- 16. Jiang, B., Chen, X., Liu, W., Yu, J., Yu, G., Chen, T.: MotionGPT: Human motion as a foreign language. In: NeurIPS (2023) 3
- 17. Karunratanakul, K., Preechakul, K., Suwajanakorn, S., Tang, S.: Guided motion diffusion for controllable human motion synthesis. In: ICCV (2023) 10
- 18. Li, T., Tian, Y., Li, H., Deng, M., He, K.: Autoregressive image generation without vector quantization. In: NeurIPS (2024) 6, 13
- 19. Li, Z., Yuan, W., He, Y., Qiu, L., Zhu, S., Gu, X., Shen, W., Dong, Y., Dong, Z., Yang, L.T.: LaMP: language-motion pretraining for motion generation, retrieval, and captioning. In: ICLR (2025) 25
- 20. Lin, J., Wang, R., Lu, J., Huang, Z., Song, G., Zeng, A., Liu, X., Wei, C., Yin, W., Sun, Q., Cai, Z., Yang, L., Liu, Z.: The quest for generalizable motion generation: Data, model, and evaluation. arXiv 2510.26794 (2025) 3
- 21. Lu, S., Chen, L., Zeng, A., Lin, J., Zhang, R., Zhang, L., Shum, H.: HumanTOMATO: text-aligned whole-body motion generation. In: ICML (2024) 4

- 22. Lu, S., Wang, J., Lu, Z., Chen, L., Dai, W., Dong, J., Dou, Z., Dai, B., Zhang, R.: ScaMo: exploring the scaling law in autoregressive motion generation model. In: CVPR (2025) 4
- 23. Min, J., Chai, J.: Motion graphs++: a compact generative model for semantic motion analysis and synthesis. ACM TOG 31(6), 153:1–153:12 (2012) 3
- 24. Nichol, A.Q., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In: ICML (2022) 24
- 25. Ormoneit, D., Black, M.J., Hastie, T., Kjellström, H.: Representing cyclic human motion using functional analysis. Image and Vision Computing 23(14), 1264–1276

(2005) 3

- 26. Petrovich, M., Black, M.J., Varol, G.: Action-conditioned 3D human motion synthesis with transformer VAE. In: ICCV (2021) 3
- 27. Petrovich, M., Black, M.J., Varol, G.: TEMOS: generating diverse human motions from textual descriptions. In: ECCV (2022) 3
- 28. Petrovich, M., Black, M.J., Varol, G.: TMR: text-to-motion retrieval using contrastive 3D human motion synthesis. In: ICCV (2023) 3
- 29. Pinyoanuntapong, E., Saleem, M.U., Karunratanakul, K., Wang, P., Xue, H., Chen, C., Guo, C., Cao, J., Ren, J., Tulyakov, S.: MaskControl: spatio-temporal control for masked motion synthesis. In: ICCV (2025) 3, 10, 20
- 30. Pinyoanuntapong, E., Wang, P., Lee, M., Chen, C.: MMM: generative masked motion model. In: CVPR (2024) 3
- 31. Plappert, M., Mandery, C., Asfour, T.: The KIT motion-language dataset. Big Data 4(4), 236–252 (2016) 9
- 32. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML (2021) 8, 9
- 33. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022) 3
- 34. Shafir, Y., Tevet, G., Kapon, R., Bermano, A.H.: Human motion diffusion as a generative prior. In: ICLR (2024) 10
- 35. Tevet, G., Gordon, B., Hertz, A., Bermano, A.H., Cohen-Or, D.: MotionCLIP: Exposing human motion generation to CLIP space. In: ECCV (2022) 3
- 36. Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., Bermano, A.H.: Human motion diffusion model. In: ICLR (2023) 2, 3
- 37. Wang, Y., Huang, D., Zhang, Y., Ouyang, W., Jiao, J., Feng, X., Zhou, Y., Wan, P., Tang, S., Xu, D.: MotionGPT-2: A general-purpose motion-language model for motion generation and understanding. arXiv 2410.21747 (2024) 25
- 38. Wang, Z., Wang, J., Li, Y., Lin, D., Dai, B.: InterControl: zero-shot human interaction generation by controlling every joint. In: NeurIPS (2024) 3, 9, 10, 20
- 39. Wen, B., Xie, H., Chen, Z., Hong, F., Liu, Z.: 3D scene generation: A survey. arXiv 2505.05474 (2025) 2
- 40. Wu, B., Xie, J., Shen, K., Kong, Z., Ren, J., Bai, R., Qu, R., Shen, L.: MGMotionLLM: A unified framework for motion comprehension and generation across multiple granularities. In: CVPR (2025) 25
- 41. Wu, Y., Ji, W., Zheng, K., Wang, Z., Xu, D.: MoTe: learning motion-text diffusion model for multiple generation tasks. arXiv 2411.19786 (2024) 25
- 42. Xie, H., Wen, B., Zheng, J., Chen, Z., Hong, F., Diao, H., Liu, Z.: DynamicVLA: a vision-language-action model for dynamic object manipulation. arXiv 2601.22153

(2026) 2

- 43. Xie, Y., Jampani, V., Zhong, L., Sun, D., Jiang, H.: OmniControl: control any joint at any time for human motion generation. In: ICLR (2024) 10, 20
- 44. Yuan, W., He, Y., Shen, W., Dong, Y., Gu, X., Dong, Z., Bo, L., Huang, Q.: MoGenTS: motion generation based on spatial-temporal joint modeling. In: NeurIPS

(2024) 4

- 45. Zhang, J., Zhang, Y., Cun, X., Zhang, Y., Zhao, H., Lu, H., Shen, X., Ying, S.: Generating human motion from textual descriptions with discrete representations. In: CVPR (2023) 2, 3, 4, 7, 9, 12
- 46. Zhang, M., Cai, Z., Pan, L., Hong, F., Guo, X., Yang, L., Liu, Z.: MotionDiffuse: Text-driven human motion generation with diffusion model. IEEE TPAMI 46(6), 4115–4128 (2024) 2, 3, 12
- 47. Zhang, M., Guo, X., Pan, L., Cai, Z., Hong, F., Li, H., Yang, L., Liu, Z.: ReMoDiffuse: Retrieval-augmented motion diffusion model. In: ICCV (2023) 3, 7, 12
- 48. Zhang, Y., Huang, D., Liu, B., Tang, S., Lu, Y., Chen, L., Bai, L., Chu, Q., Yu, N., Ouyang, W.: MotionGPT: finetuned LLMs are general-purpose motion generators. In: AAAI (2024) 25
- 49. Zhong, C., Hu, L., Zhang, Z., Xia, S.: AttT2M: Text-driven human motion generation with multi-perspective attention mechanism. In: ICCV (2023) 3

### A From Semantic Usability to Detail Recoverability: Why MoTok Bridges Both

In the main paper, we show that MoTok substantially improves downstream performance over existing tokenizers. A key reason is that MoTok upgrades the conventional “Decode” stage into a “Control” stage that can incorporate finegrained kinematic conditions. This division of labor allows the planning stage to focus more on high-level conditions, leading to better overall generation.

In this section, we reveal another important reason: MoTok serves as an effective bridge between the motion space and the semantic space. On the one hand, it provides stronger recovery of high-frequency motion details

- (Section A.1). On the other hand, it preserves richer low-frequency motion characteristics that facilitate mapping textual semantics to discrete motion tokens
- (Section A.2).

#### A.1 Better low-level detail recoverability with diffusion decoding

To isolate the effect of the decoder on fine-grained motion reconstruction, we design a two-stage training protocol on HumanML3D and compare a standard VQ-VAE tokenizer against our MoTok tokenizer. In Stage-1, we train the tokenizer end-to-end to learn an encoder and a discrete codebook. In Stage-2, we freeze the Stage-1 encoder and codebook so that all variants share the same discrete tokens, and train the decoder only. We further consider a 2×2 design that combines the decoder architecture used in Stage-1 and Stage-2 (Conv vs. Diffusion), which enables a controlled and fair evaluation of reconstruction capacity under identical token information. We report reconstruction quality (Recon FID), downstream text-to-motion performance (T2M FID), and motion-to-text metrics (R@1/R@3/BLEU@1/BLEU@4) to reflect both low-level fidelity and semantic usability of the learned discrete representations.

Table 5 shows that, with the encoder and codebook frozen in Stage-2, replacing a conventional convolutional decoder with MoTok’s diffusion-based decoder consistently improves both reconstruction and downstream performance. In particular, the diffusion decoder achieves lower Recon FID, indicating stronger recovery of fine-grained details given the same discrete token sequence. This protocol provides a more principled and fair comparison than end-to-end training alone, since the gains cannot be attributed to increased token information capacity. The benefits also translate to text-to-motion generation, where diffusionbased decoding yields consistently lower T2M FID. Overall, these results demonstrate that MoTok recovers motion details more faithfully under the same token information.

#### A.2 Better semantic usability of discrete tokens

To compare the semantic information carried by different tokenizers, we further evaluate motion-to-text (M2T) captioning on top of MotionGPT. Interestingly,

- Table 5: Two-stage training ablation of tokenizer decoders on HumanML3D. Stage-1 trains the tokenizer end-to-end. Stage-2 freezes the encoder and codebook and trains the decoder only. All variants share the same discrete tokens in Stage-2, enabling a fair comparison of reconstruction capability under identical token information.

|Reconstruction T2M|M2T|
|---|---|
|FID ↓ FID ↓|R@1 ↑ R@3 ↑ BLEU@1 ↑ BLEU@4 ↑|

Stage-1 Decoder Stage-2 Decoder

|Conv (VQ) Conv (VQ) Conv (VQ) Diffusion (MoTok)|0.0954 0.1271 0.0544 0.1098<br><br>|0.473 0.732 50.1 14.6|
|---|---|---|
|Diffusion (MoTok) Conv (VQ) Diffusion (MoTok) Diffusion (MoTok)|0.1347 0.0718 0.0433 0.0642<br><br>|0.488 0.743 51.3 15.5<br><br>|

although M2T does not involve motion generation or reconstruction, the captioning model trained with tokens produced by a standard VQ-VAE in Stage-1 consistently underperforms the one trained with MoTok tokens across nearly all metrics. This indicates that MoTok tokens encode richer and more usable semantic information, which improves text–motion alignment and effectively reduces the burden on the downstream planner.

A plausible explanation is that the stronger diffusion-based reconstruction shifts high-frequency detail recovery to the decoder, allowing MoTok’s discrete tokens to devote less capacity to fine-grained kinematic variations and instead preserve more low-frequency information that captures overall motion structure and trends.

As a result, MoTok serves as an effective bridge: it recovers high-frequency motion details more faithfully under the same token information, while simultaneously simplifying semantic alignment in the planning stage, thereby connecting semantic and kinematic constraints more efficiently.

### B From Richer Conditions to Broader Tasks: Generality of the Unified Paradigm

For different types of conditions, we define two forms in the main paper: global and local. Specifically, we treat text features as global conditions and kinematic signals as local conditions. In this section, we present additional variants and usages of conditions to further demonstrate the generality and flexibility of our proposed conditioning framework.

#### B.1 Inject Semantic Condition via Cross Attention

For text, which serves as a global condition, we adopt an in-context-learning-style training strategy in the main paper. Another common practice in the motion generation literature is to inject text features through cross-attention. Here, we compare against this alternative design. As shown in Table 6, compared with the in-context variant, the cross-attention version yields a clear improvement in RPrecision, indicating stronger text-motion alignment. However, this gain comes

- Table 6: Controllable motion generation quantitative results on HumanML3D test set. Note that “Traj. Err.”, “Loc. Err.”, and “Avg. Err.” denote “Trajectory Error”, “Localization Error”, and “Average Error”, respectively. ↑ (↓) indicates that the values are better if the metric is larger (smaller). ‘→’ means closer to real data is better. Red and Blue highlight the best and second-best results, respectively.

R-Precision ↑ (Top-3)

Foot Skating Ratio ↓

Traj. Err.↓ (50cm)

Loc. Err. ↓ (50cm)

Avg. Err. ↓ (m)

Method FID ↓

Diversity →

Real Motion 0.002 0.797 9.503 0.000 0.0000 0.0000 0.0000

OmniControl [43] 0.310 0.693 9.502 0.0608 0.0617 0.0107 0.0404 InterControl [38] 0.178 0.669 9.498 0.0968 0.0403 0.0031 0.0741 CrowdMoGen [4] 0.147 0.781 9.461 0.0829 0.0000 0.0000 0.0028 MaskControl [29] 0.083 0.805 9.395 0.0545 0.0000 0.0000 0.0072 MoTok-DDM-2 0.025 0.779 9.756 0.0436 0.0000 0.0000 0.0008 w/ Cross Att. 0.027 0.799 9.582 0.0432 0.0000 0.0000 0.0008 w/ Global Hint 0.028 0.775 9.876 0.0427 0.0000 0.0000 0.0008

One

InterControl [38] 0.184 0.670 9.410 0.0948 0.0475 0.0030 0.0911 CrowdMoGen [4] 0.178 0.777 9.149 0.0865 0.0000 0.0000 0.0027 MoTok-DDM-2 0.016 0.775 9.711 0.0456 0.0000 0.0000 0.0006 w/ Cross Att. 0.025 0.795 9.678 0.0444 0.0000 0.0000 0.0005 w/ Global Hint 0.021 0.776 9.608 0.0444 0.0000 0.0000 0.0005

Two

InterControl [38] 0.199 0.673 9.352 0.0930 0.0487 0.0026 0.0969 CrowdMoGen [4] 0.192 0.778 9.169 0.0871 0.0000 0.0000 0.0030 MoTok-DDM-2 0.014 0.786 9.519 0.0480 0.0000 0.0000 0.0007 w/ Cross Att. 0.017 0.799 9.440 0.0477 0.0000 0.0000 0.0007 w/ Global Hint 0.017 0.779 9.676 0.0479 0.0000 0.0000 0.0007

Three

at the cost of a noticeable drop in FID. This suggests that different conditioning injection strategies lead to different trade-offs between R-Precision and FID.

#### B.2 Global vs Local Hint Injection

For kinematic conditions, we also explore injecting them in a global form. Specifically, we use a motion Transformer to extract global features from the hint sequence, and then place these features after the text features for in-context sequence modeling. The results are reported in Table 6 under the row “w/ Global Hint”. Note that we still retain the original local injection pathway in this setting. Empirically, adding global kinematic conditions brings almost no improvement across the evaluation metrics. This suggests that, for the current task, local conditioning alone is already sufficient, and introducing additional global kinematic guidance is unnecessary.

#### B.3 Motion Editing as a new condition-driven task

Another important application of kinematic conditions is motion editing. In this task, the input typically consists of a source motion together with a text prompt describing the desired modification. We inject the source motion as a local condition, in a manner similar to trajectory control. Specifically, we adopt a motion encoder architecture similar to that in MoTok to extract features from the source motion, which are then used as position-wise embeddings. The text

###### Table 7: Ablation on CFG scale. We vary the CFG scale from 1.6 to 3.6 with a step size of 0.2. Each entry reports the evaluation result (e.g., FID; lower is better).

Variant 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6

- MoTok-DDM-1 0.5551 0.4139 0.3251 0.3130 0.2669 0.2093 0.2519 0.1730 0.1756 0.1769 0.1551
- MoTok-DDM-2 0.2792 0.2077 0.1741 0.1373 0.1069 0.0835 0.0630 0.0519 0.0332 0.0403 0.0349 MoTok-DDM-4 0.1430 0.1023 0.0753 0.0466 0.0396 0.0394 0.0423 0.0653 0.0798 0.0984 0.1272 MoTok-DDM-8 0.1031 0.0772 0.0707 0.0887 0.0993 0.1149 0.1477 0.1607 0.1949 0.2173 0.2755 MoTok-DDM-16 0.1036 0.0875 0.0975 0.1104 0.1148 0.1282 0.1335 0.1418 0.1550 0.1604 0.1878

MoTok-AR-2 0.0857 0.0852 0.0803 0.0799 0.0783 0.0710 0.0699 0.0696 0.0617 0.0692 0.0710 MoTok-AR-4 0.0907 0.0839 0.0535 0.0548 0.0629 0.0753 0.0974 0.1361 0.2071 0.2207 0.2582 MoTok-AR-8 0.0554 0.0533 0.0551 0.0669 0.1194 0.1457 0.2292 0.2652 0.3351 0.4201 0.5580 MoTok-AR-16 0.1580 0.1598 0.1842 0.1730 0.2512 0.3116 0.3257 0.3330 0.4119 0.4831 0.5364

features are injected in the same way as in our other tasks. We provide several examples of this application setting in the demo video, which is trained under MotionFix [2] dataset.

### C Additional Analyses on Downstream Behaviors

Efficiency Comparison. We evaluate our method on H100. It costs 2.63s to generate a single sequence with both textual prompt and trajectory conditions. As for MaskControl, the full version costs 32.79s in the same environment.

Ablation on CFG scale. The CFG-scale ablation exhibits two consistent insights. First, nearly all variants follow a clear unimodal pattern: performance improves as the CFG scale increases from a small value, reaches an optimum, and then degrades when the scale becomes overly large. This indicates that moderate guidance best balances semantic alignment and motion realism, while excessive guidance can over-constrain sampling and harm fidelity. Second, the optimal CFG scale shifts with the temporal compression rate. As compression decreases (i.e., more tokens), even under the same masking and replacement strategy during training, the absolute token count increases and thus injects more motion-side information into the planning process. Consequently, the generator tends to rely more on motion tokens and time-synchronized cues, weakening the relative influence of the text condition. To compensate for this imbalance, inference requires a larger CFG scale to amplify the contribution of the text embedding and enforce stronger adherence to the given prompt, especially for low-downrate (high-token) settings.

### D Implementation Details

- D.1 Evaluation Metrics In this work, we mainly focus on three types of metrics:

- 1. Text-to-Motion (T2M): We evaluate both motion realism and text–motion alignment. Motion realism is measured by Fréchet Inception Distance (FID). Text–motion consistency is assessed by R-Precision (R@1/2/3) and Multimodal Distance (MM Dist). We further report Diversity (Div) to quantify variation across generated motions, and Multi-Modality (MM) to measure the ability to generate multiple plausible motions for the same text.
- 2. Motion-to-Text (M2T): We adopt standard captioning metrics, including BLEU@1/4, ROUGE-L, CIDEr, and BERTScore. In addition, we report R-Precision to measure alignment between generated captions and their corresponding motions. Below we define each metric.
- 3. Controllable Motion Generation: We additionally evaluate spatial-level control accuracy, including Foot skating ratio, Trajectory error, Location error, and Average keyframe error. These metrics directly quantify whether generated motions satisfy the prescribed spatial control signals.

R-Precision measures retrieval accuracy by computing the fraction of relevant items among the top-R retrieved candidates:

R-Prec = |Rel ∩ Top-R| R

, (15)

where Rel denotes the set of ground-truth matches and Top-R denotes the retrieved set at rank R.

Fr’echet Inception Distance (FID) measures the distance between real and generated feature distributions:

FID = |µr − µg|22 + Tr Σr + Σg − 2(ΣrΣg)1/2 , (16)

where (µr,Σr) and (µg,Σg) are the mean and covariance of real and generated features, respectively.

Diversity quantifies sample-level variability by averaging pairwise distances:

2 M(M − 1) i<j |f(xi) − f(xj)|2. (17)

Div =

Multi-Modality measures conditional diversity among K samples generated for the same text:

2

K(K − 1) i<j |f(xti) − f(xtj)|2. (18) Multimodal Distance evaluates cross-modal alignment in a shared embed-

MM =

ding space:

N

1 N

d(ftext(ti),fmotion(mi)). (19) BLEU computes n-gram precision with a brevity penalty:

MM Dist =

i=1

BLEU@N = BP · exp

N

1 N

log pn , (20)

n=1

##### with modified precision

min CountC(ngram),maxR ∈ RefsCountR(ngram) ngram ∈ CCountC(ngram)

pn = ngram∈C

, (21)

and brevity penalty

1 if c > r, e(1−r/c) if c ≤ r.

BP =

(22)

ROUGE-L uses the longest common subsequence (LCS) and an F-measure:

(1 + β2) · PLCS · RLCS RLCS + β2 · PLCS

. (23) CIDEr computes TF-IDF weighted n-gram similarity:

ROUGE-L =

g(c) · g(s) |g(c)||g(s)|

1 |S| s∈S

CIDEr(c,S) =

. (24)

BERTScore measures semantic similarity via contextual embeddings:

1 |c| x∈c

BERTScore(c,r) =

cos f(x),f(y) . (25)

max

y∈r

Spatial-level Metrics. Let pk,k ∈ K and pˆk,k ∈ K denote the groundtruth and generated 3D positions at a set of keyframes K for the controlled joint (e.g., pelvis). We define the keyframe location error at keyframe k as

ek = |pˆk − pk|2 . (26) Given a threshold τ (e.g., 50,cm), we compute:

- (1) Average error. The mean keyframe position error is

AvgErr =

1 |K| k∈K

ek. (27)

- (2) Location error. The fraction of keyframes whose error exceeds the

threshold is

1 |K| k∈K

LocErr =

I ek > τ , (28)

where I[·] is the indicator function. violates the threshold, and report the failure ratio over N test samples:

- (3) Trajectory error. We mark a trajectory as unsuccessful if any keyframe

1 N

TrajErr =

N

e(kn) > τ . (29)

I max

k∈K

n=1

- (4) Foot skating ratio. Let F denote the set of frames where the foot is in

contact (estimated by a velocity/height heuristic), and vˆt be the generated foot horizontal velocity at frame t. Foot skating ratio is computed as

1 |F|

Skate =

t ∈ FI |vˆt|2 > ϵ , (30)

where ϵ is a small threshold for detecting undesired sliding during contact.

#### D.2 Details for Controllable Motion Generation

Training Strategies. For both the spatial control and text-to-motion (T2M) tasks, we adopt the same training protocol. Specifically, for DDM, AR, and MoTok, all models are trained for 24 epochs, with each epoch consisting of approximately 2,000 iterations.

For MoTok, training is conducted on 8 GPUs with a batch size of 512 per GPU. For DDM and AR, we also use 8 GPUs, but with a batch size of 64 per GPU. All models are optimized using AdamW, with an initial learning rate of 2e-4, which is decayed to 2e-5 at the 20th epoch.

Inference Strategies. During inference, MoTok adopts a spaced diffusion scheme and uses Fast27, provided by GLIDE [24], as the sampling strategy, thereby substantially reducing the original 1,000 diffusion steps used in training. The inference speed could be further improved in the future by incorporating consistency models, which we leave for future work. For DDM, inference is performed with 10 autoregressive denoising steps. In contrast, AR predicts the output one token at a time, generating the token sequence sequentially.

#### D.3 Details for Motion-to-Text

Task Formulation. For motion-to-text (M2T), the goal is to generate a natural-language description that faithfully summarizes a given motion sequence. Formally, given an input motion x = xtt = 1T, where xt denotes the pose representation at time step t, the model predicts a caption y = yii = 1L consisting of L tokens. We model M2T as conditional language generation by maximizing the likelihood p(y | x). In our framework, the motion sequence is first encoded by the MoTok tokenizer into a discrete token sequence z = znn = 1N, where N depends on the temporal downsampling rate. An autoregressive language decoder then generates the caption conditioned on z:

p(y | x) = p(y | z) = i = 1Lp(yi | y<i,z). (31)

This formulation allows the captioner to operate in a compact token space while leveraging MoTok to preserve semantically relevant motion content.

Implementation Details. We implement the M2T captioner as a standard autoregressive transformer decoder conditioned on MoTok tokens. Specifically, we

- Table 8: Quantitative results of Motion-to-Text (M2T) on HumanML3D. Note that “Sep.” and “Uni.” denote “Seperated Model” and “Unified Model”, respectively. ↑ indicates higher is better. Red and Blue highlight the best and second-best results, respectively.

Method R@1↑ R@3↑ BLEU@1↑ BLEU@4↑ ROUGE-L↑ CIDEr↑ BERTScore↑ Real Motion 0.523 0.828 - - - - -

TM2T [11] 0.516 0.823 48.9 7.0 38.1 16.8 32.2 LaMP [19] 0.547 0.831 47.8 13.0 37.1 28.9 MG-MotionLLM [40] 0.592 0.866 - 8.1 - - 36.7

Sep.

MotionGPT [48] 0.543 0.827 48.2 12.5 37.4 29.2 32.4 MotionGPT2 [37] 0.558 0.838 48.7 13.8 37.6 29.8 32.6 MoTe [41] 0.577 0.871 46.7 11.2 37.4 31.5 30.3

Uni.

Baseline-VQ 0.473 0.732 50.1 14.6 40.8 34.3 34.3 Baseline-MoTok 0.488 0.743 51.3 15.5 41.4 35.3 34.8

first encode motions into discrete tokens using the pretrained MoTok tokenizer (encoder + codebook), and keep the tokenizer fixed during captioning training. Token embeddings are obtained by table lookup and optionally augmented with positional embeddings. The captioning model takes the MoTok token sequence as conditioning context and generates text using teacher forcing during training, optimizing the cross-entropy loss over the ground-truth caption tokens. At inference time, we decode captions autoregressively using greedy decoding or beam search. Unless otherwise specified, we follow common M2T training practices in prior work, including using the HumanML3D captions as supervision, truncating/padding motion sequences to a maximum length, and applying standard text preprocessing (tokenization and vocabulary construction).

Full Quantitative Results. As shown in Table 8, replacing a conventional VQ-based motion tokenizer with MoTok consistently improves M2T captioning performance. This improvement suggests that MoTok produces discrete tokens that better capture motion semantics, enabling the language decoder to infer high-level action descriptions more effectively from the token sequence. By offloading fine-grained reconstruction to diffusion-based decoding, the discrete representation is freed from low-level details and can focus on semantic content. Consequently, the learned token space exhibits stronger semantic separability and serves as a more effective interface for text generation, supporting our design choice of using latent dimensionality as a semantically meaningful bottleneck rather than a reconstruction-driven representation.

