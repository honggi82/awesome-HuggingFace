# arXiv:2605.14278v1[cs.CV]14May2026

[Figure 1]

[Figure 2]

## KVPO: ODE-Native GRPO for Autoregressive Video Alignment via KV Semantic Exploration

Ruicheng Zhang1,3, Kaixi Cong1, Jun Zhou1, Zhizhou Zhong2,3, Zunnan Xu1, Shuiyang Mao3†, Wei Liu3, Xiu Li1‡

1Tsinghua University, 2HKUST, 3Video Rebirth

†Project leader. ‡Corresponding author.

Aligning streaming autoregressive (AR) video generators with human preferences is challenging. Existing reinforcement learning methods predominantly rely on noise-based exploration and SDE-based surrogate policies that are mismatched to the deterministic ODE dynamics of distilled AR models, and tend to perturb low-level appearance rather than the high-level semantic storyline progression critical for long-horizon coherence. To address these limitations, we present KVPO, an ODE-native online Group Relative Policy Optimization (GRPO) framework for aligning streaming video generators. For diversity exploration, KVPO introduces a causal-semantic exploration paradigm that relocates the source of variation from stochastic noise to the historical KV cache. By stochastically routing historical KV entries, it constructs semantically diverse generation branches that remain strictly on the data manifold. For policy modeling, KVPO introduces a velocity-field surrogate policy based on Trajectory Velocity Energy (TVE), which quantifies branch likelihood in flow-matching velocity space and yields a reward-weighted contrastive objective fully consistent with the native ODE formulation. Experiments on multiple distilled AR video generators demonstrate consistent gains in visual quality, motion quality, and text-video alignment across both single-prompt short-video and multi-prompt long-video settings.

Homepage: https://richard-zhang-ai.github.io/KVPO-Project Code: https://github.com/Richard-Zhang-AI/KVPO Date: May 15, 2026

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

0.03

MemFlow LongLive

0.03

0.06

0.04

0.02

0.02

TA

TA

[Figure 7]

1

Astrolable

MemFlow Astrolable

Self-Forcing

7

8.8 6

9.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

947

998

VQ

VQ

3

8.8

- 1

.0

- 2 1

QM

QM

08.1

78.1

Reward-Forcing

KVPO Reward-Forcing KVPO

+KVPO +Astrolable

28.1

58.1

78.1

[Figure 12]

[Figure 13]

98.1

###### Short-Video Single-Prompt Generation Long-Video Multi-Prompt Generation

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Baseline Baseline

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

###### +KVPO

+KVPO

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Baseline

Baseline

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

+KVPO

+KVPO

### 1 Introduction

Recent advances in video generation [13, 29, 3, 20, 27, 28] have substantially improved visual quality, yet deploying these models in real-time interactive settings remains challenging. Such settings demand not merely high-fidelity generation, but low-latency, streaming, long-horizon synthesis under causal temporal dependencies. To meet these requirements, recent work distills pretrained video diffusion models into few-step autoregressive (AR) video generators, enabling efficient streaming inference via causal attention and KV caching [24, 5, 12]. Nevertheless, aligning these AR video models with human preferences remains an open challenge, as preference-relevant qualities extend beyond frame-level fidelity to long-horizon coherence, subject consistency, and semantic progression.

Existing alignment methods for AR video generators predominantly fall into two categories, yet neither adequately addresses these challenges. The first relies on reward-weighted distillation [12], which upweights high-reward trajectories in the supervised objective but fundamentally lacks active exploration of diverse candidate behaviors. The second [10, 21] converts the deterministic ODE sampling into a stochastic SDE process and constructs exploration branches by injecting noise into the initial or intermediate latents. However, this strategy has been shown to be ill-suited to streaming AR video generators [1, 2, 33]. Recasting a few-step distilled generator as an SDE injects stochastic transitions into an originally deterministic probability flow, which breaks its native ODE formulation [2]. Moreover, noise-driven exploration primarily perturbs low-level appearance and local structure [2] rather than the high-level semantics, motion dynamics, and storyline evolution that are crucial for long-horizon video generation (Figure 1). Furthermore, intermediate noise injection induces off-manifold structural interference [33], exacerbating the risk of generative degradation and weakening exploration signal quality.

More recently, NeighborGRPO [1] reinterprets Group Relative Policy Optimization (GRPO) [16] as an implicit contrastive learning paradigm. It approximates the surrogate policy via Euclidean distances between samples generated under a pure ODE framework, with AR-CoPO [2] extending this approach to AR video generation. While this line of work offers useful insights into ODE-based policy optimization, surrogate policies grounded in latent Euclidean distances implicitly assume uniform geometry in the generation space, even though different latent dimensions may contribute unequally to policy probabilities. Therefore, such metrics may fail to faithfully capture the model’s intrinsic preferences structure over candidate trajectories.

To overcome these limitations, we propose KVPO, an ODE-native online GRPO framework tailored to streaming autoregressive video generation. KVPO pioneers causal-semantic exploration and surrogate policy modeling in the flow-matching [8] velocity-field space under a pure ODE paradigm. Unlike noise-driven perturbation approaches, we introduce a causal-semantic exploration paradigm that relocates the source of variation from stochastic noise to the historical KV cache. In streaming AR video generation, future content is causally conditioned on historical context, making differential reuse of historical information a natural mechanism for diversity exploration. Specifically, we design Causal History Routing (CHR), which stochastically routes historical KV entries to construct branch-specific local contexts. Consequently, exploration remains strictly on-manifold, and variation in semantic space naturally promotes more meaningful and causally coherent narrative progression. To optimize preferences over the explored branches, we further introduce an ODE-native surrogate policy formulation grounded in flow-matching dynamics. Rather than relying on external geometric distances or SDE transition kernels, we define a Gibbs-form surrogate policy based on Trajectory Velocity Energy (TVE) to quantify the likelihood of the current policy reproducing each branch directly in the velocity-field space. This yields a reward-weighted contrastive flow-matching objective that embeds preference optimization into the model’s native dynamics.

Experiments on multiple distilled AR video generators demonstrate consistent gains in humanpreference alignment across both single-prompt short-video and multi-prompt long-video settings. Our primary contributions are as follows:

- • We propose KVPO, an ODE-native online policy optimization framework for streaming AR video generation. To the best of our knowledge, KVPO is the first method to perform causalsemantic exploration and model the surrogate policy within the flow-matching velocity-field space under a pure ODE paradigm.
- • We introduce a causal-semantic exploration mechanism that shifts diversity generation from unstructured noise injection to historical KV-cache routing, intrinsically avoiding off-manifold distortion while promoting richer narrative progression and storyline diversity.

- • We introduce a velocity-field surrogate policy based on Trajectory Velocity Energy (TVE), which yields a reward-weighted contrastive flow-matching objective that embeds preference optimization into the model’s native ODE dynamics without relying on external geometric distances or SDE transition kernels.

### 2 Related Work

##### 2.1 Streaming Autoregressive Video Generation

Autoregressive (AR) models [25, 26] generate video in a causal, streaming fashion by conditioning each new frame on previously generated content. Recent acceleration and distillation techniques have substantially improved their practicality, compressing multi-step diffusion processes into efficient few-step variants while preserving visual quality [24, 17, 31]. By exploiting causal attention, dynamic key-value (KV) caching [7], and explicit memory architectures [6], these models enable interactive, real-time, and long-horizon video generation [5, 23, 6]. Despite these advances, explicit preference alignment for highly deterministic few-step AR models remains relatively underexplored.

##### 2.2 Preference Alignment for Generative Models

Post-training alignment for generative models typically leverages reward signals to steer model outputs toward human-preferred behaviors. This is commonly achieved by framing the sampling process as a policy rollout and optimizing the induced distribution via policy-gradient objectives. VideoAlign [11] introduces reward supervision for video generation. Flow-GRPO [10] and DanceGRPO [21] extend GRPO-style optimization to visual generative models by reformulating ODEs as SDEs. However, such noise-injection rollout strategies and SDE-based policy modeling paradigms are ill-suited for few-step AR video models [2]. These methods deviate from the native ODE formulation of AR generators and tend to alter low-level appearance more than high-level semantic development. SAGE-GRPO [33] further shows that noise-based exploration can induce off-manifold distortions, undermining the quality of candidate samples.

Recent works have begun to explore alignment techniques tailored for AR video models. Reward Forcing [12] performs reward-weighted distillation to amplify optimization signals from high-quality samples, but lacks active exploration. Astrolabe [30] applies forward-process reinforcement learning by contrasting positive and negative samples at inference endpoints, yet exploration remains confined to noise-endpoint perturbation rather than structured semantic branching. NeighborGRPO [1] offers an ODE-centric alternative by modeling preferences through latent-space neighborhood geometry, and AR-CoPO [2] extends this to AR video generation. Nevertheless, both depend on external geometric proximity to approximate surrogate preference ordering, which may not faithfully reflect the model’s intrinsic preferences over candidate trajectories. In contrast, KVPO performs causalsemantic exploration via stochastic KV routing and models the surrogate policy in the ODE-native flow-matching velocity-field space, offering a new perspective on AR preference alignment.

### 3 Methodology

- 3.1 Preliminaries: Block-wise Autoregressive Video Generation Mainstream streaming AR video generators synthesize long videos in a block-by-block manner. Given a video sequence V = [v1, v2, . . . , vB] partitioned into B blocks, the generation at block b is

formulated as pθ(vb | v<b,C), conditioned on the text prompt C and the historical context v<b. In Diffusion Transformer (DiT) [15] architectures, this historical context is materialized as a compressed

Key-Value (KV) cache K<b. In streaming implementations, the KV memory typically adopts

- a (sink, local) structure: the sink cache stores persistent global anchors for long-range temporal coherence, while the local cache maintains a sliding window of the N most recent frames for local motion modeling. Under the flow matching framework [8], the model is trained along the linear interpolation path between a clean sample x0 and a noise latent xT:

xt = t x0 + (1 − t)xT, xT ∼ N(0,I), t ∈ [0,1]. (1) A conditional velocity field vθ(xt, t,K<b) is learned by minimizing the expected squared error against the ground-truth velocity ut = x0 − xT. At inference, block xb0 is obtained by integrating the probability flow ODE from noise to clean:

dxt dt

= vθ(xt, t, K<b), xt=0 = xT ∼ N(0,I). (2) The ODE solver advances through S discrete timesteps {ts}Ss=1, yielding the generated block xb0 = vb.

##### 3.2 Causal-Semantic Exploration via Causal History Routing

(a) Semantic-Space Exploration

We redirect diversity exploration from noise-driven perturbations to causal-semantic exploration over the historical KV cache via Causal History Routing (CHR). Since future content in streaming AR video generation is strongly conditioned on the historical context K<b, perturbing the composition of local memory induces semantically diverse generation branches. Specifically, consider a pivot block b∗ at which L frames have already been generated. CHR leaves the sink KV unchanged, where the sink memory comprises the earliest three historical frames: Ksink = {(K1, V1),(K2, V2),(K3, V3)}. For local memory, CHR adopts a fixed 9-slot layout in which the last three slots always store the most recent frames, Knear = {(KL−2, VL−2),(KL−1, VL−1),(KL, VL)}, while the first six slots are branch-specific and stochastically refilled from the older non-sink history. Letting ΩL = {4,5, . . . , L − 3} denote the routable index set, CHR samples six indices r1g, . . . , r6g ∈ ΩL for each branch g ∈ {1, . . . , G} and constructs the branch-specific local cache as

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

- Branch 1

- Branch 2

- Branch 3

- Branch 4

- Branch 1

(b) Noise-Space Exploration

- Branch 2

- Branch 3

- Branch 4

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

- • Off-manifold distortion
- • Low-level structural perturbation
- • Unstructured stochastic diversity

- • On-manifold generation
- • High-level semantic variation
- • Causally structured branch diversity
- • On-manifold generation → preserved generation fidelity
- • High-level semantic variation → richer narrative progression and storyline diversity
- • Causally structured branch diversity → meaningful contrastive preference signal

[Figure 62]

[Figure 63]

Figure 1: Comparison of semantic-space and noise-space exploration.

- • Off-manifold distortion → degraded generation fidelity
- • Low-level structural perturbation → mainly affects local appearance and layout
- • Unstructured stochastic diversity → noisy preference optimization signal

K˜<bg,local∗ = (Krg

. (3)

; Knear

, Vrg

), . . . , (Krg

, Vrg

)

1

1

6

6

shared recent 3 slots

branch-specific routed 6 slots

For each candidate branch g, the attention output at block b∗ is computed using the current-block query Qb∗ against the concatenation of the sink cache, the branch-specific local cache, and the current-block KV entries:

Qgb∗ Ksink ; K˜<bg,local∗ ; Kbg∗ ⊤ √dk

Vsink ; V˜<bg,local∗ ; Vbg∗ , (4) where dk denotes the key dimension.

Attngb∗ = Softmax

Rollout and Replay. During rollout, semantic exploration branches from a randomly sampled pivot block b∗ under G distinct CHR refill decisions for the branch-specific local slots. Blocks preceding

- b∗ are generated once using the shared default KV cache, while CHR is applied exclusively within a contiguous window B = [b∗, b∗ + W), where W denotes the exploration window length in blocks. Beyond B, generation reverts to the standard local cache, yet the semantic variations introduced within the window propagate through subsequent blocks, as the perturbed KV states are written back into the cache. Within each perturbed block, CHR is restricted to the first half of the ODE steps, motivated by the observation that early-to-mid solver stages govern coarse semantic layout and motion, whereas late-stage perturbations contribute marginally to semantic diversity while incurring unnecessary replay cost [2]. The rollout produces G branch trajectories {Xg} with associated rewards {rg}, alongside an anchor trajectory X0 generated under the default local cache without CHR routing, yielding a baseline reward r0. For each branch g and solver step s, we cache replay tuples {zb,sg , uˆgb,s} over the perturbed window B, where zb,sg denotes the intermediate latent at block b and step s, and uˆgb,s the corresponding rollout velocity target.

During replay, the cached intermediate states zb,sg from each branch are reused as input under the restored unperturbed context K<b to predict replayed velocities vθ(zb,sg , ts,K<b), which are subsequently used for surrogate policy modeling (Section 3.3). This procedure assesses the current model’s generative tendency toward each branch trajectory under the unperturbed deployment-time semantics. Each replay step incurs the computational cost of a single forward pass and requires no specialized solver, making the replay stage as efficient as standard supervised fine-tuning. Gradient tracking is enabled exclusively for solver steps within the perturbed window b ∈ B.

##### 3.3 Velocity-Field Surrogate Policy Modeling and Optimization

Deterministic ODE generators do not expose an explicit policy distribution over candidate branches, making direct application of PPO intractable [1]. Prior work [1, 2] has shown that GRPO admits

(a) Causal-Semantic Exploration (b) Velocity-Field Surrogate Policy Modeling

###### Rollout Replay

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Causal History Routing

Trajectory Velocity Energy

- 1

 , 

- 2

- 1

[Figure 68]

[Figure 69]

[Figure 70]

 , 

- 2

KV Cache Pool

- 1

 , 

- 2

Sink KV

Text Prompt

Shared Noise

 , 

 , 

 , 

ℓ =−ℰ / 

Policies for Branch

4 5 6 ··· ···

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

1 2 3

AR Model

L-2 L-1 L

L-5 L-4 L-3

[Figure 77]

[Figure 78]

exp ℓ ℎ=1 exp ℓ ℎ

Gibbs Logit

=

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

AR Model

[Figure 83]

[Figure 84]

[Figure 85]

···

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

g

g

···

···

MSE

 , 

 , 

Cache Condition

 , 

[Figure 90]

[Figure 91]

[Figure 92]

KV Router g

KV Router 1

KV Router 2

Anchor

[Figure 93]

[Figure 94]

Condition

(c) Policy Optimization

Branch Trajectories

Block ∗ Block ∗ +1 Block ∗ +W

[Figure 95]

- Block 1
- Block 2

1

[Figure 96]

PPO Loss

ℒPPO =−

- 1
- 2

- 1
- 2

[Figure 97]

[Figure 98]

[Figure 99]

Reward Model

··· ···

[Figure 100]

··· ···

old

AdvantageComputation

 =1

Visual Quality

[Figure 101]

[Figure 102]

(VQ)

[Figure 103]

[Figure 104]

[Figure 105]

Update

Motion Quality

Velocity Target  , 

[Figure 106]

(MQ)

[Figure 107]

···

[Figure 108]

[Figure 109]

···

···

···

Cache Cache

Cache

···

···

Intermediate State  , 

g

···

[Figure 110]

[Figure 111]

[Figure 112]

Text Alignment

···

Perturbed Window B = [ ∗, ∗ +W]

Higher Rewards

(TA)

- Figure 2: Overview of the KVPO training pipeline. Starting from a shared initial noise, the model first performs causal-semantic exploration via stochastic KV routing within a perturbed window to produce diverse candidate branches (a). These branches are then replayed under the unperturbed deployment-time context, where the Trajectory Velocity Energy of each branch is computed and converted into Gibbs-form surrogate branch probabilities to measure their generation likelihood under the current policy (b). Finally, the branches are scored by the reward model, and PPO updates the AR generator toward higher-reward behaviors via a contrastive flow-matching objective (c)

an interpretation as implicit contrastive learning: the update promotes reward-preferred generations while suppressing reward-disfavored ones via their relative advantages. Guided by this insight, we introduce a branch-wise quantity that captures the current model’s generative likelihood under causal-semantic exploration and use it to construct a surrogate policy for preference optimization.

Trajectory Velocity Energy (TVE). In KVPO, causal-semantic exploration generates diverse candidate branches via stochastic local KV routing, while inference is performed under the unperturbed context K<b. The key quantity of interest is therefore the likelihood of the current policy reproducing the cached rollout velocities of a given branch under the unperturbed deployment-time semantics, which motivates the definition of Trajectory Velocity Energy. Formally, TVE for branch trajectory Xg is defined as the aggregated squared residual between the cached rollout velocity target uˆgb,s and the corresponding replayed velocity vθ(zb,sg , ts, K<b) across all perturbed blocks and solver steps:

S

2 F

1 d

vθ zb,sg , ts, K<b − uˆgb,s

Eθ Xg =

, (5)

s=1

b∈B

where d denotes the feature dimension. TVE directly reflects branch likelihood in the flow-matching velocity space: a lower TVE indicates that the current policy assigns stronger generative tendency toward that branch under the unperturbed deployment-time context K<b.

Surrogate Policy and Policy Ratio. Having defined TVE as a measure of branch likelihood under the unperturbed deployment-time context, we convert these energy values into a normalized branch distribution to construct a surrogate policy. Such a conversion should satisfy three requirements: (1) branches with lower TVE receive higher policy probability; (2) the policy is differentiable and amenable to gradient optimization; and (3) the policy depends only on relative TVE scores across branches, aligning with the contrastive learning objective. Gibbs parameterization naturally satisfies all three. Let ℓgθ = −Eθ(Xg)/τ, where τ is a temperature parameter. The current and previous policies for branch g are then defined as

exp ℓgold

exp ℓgθ

. (6)

, πold(g) =

πθ(g) =

G h=1 exp ℓhθ

G h=1 exp ℓhold

The resulting Gibbs distribution converts the model’s generative tendencies into a normalized branch distribution. Unlike geometry-based surrogate policies [1], our branch probabilities are grounded directly in replay-time compatibility, remaining faithful to the flow-matching model’s native dynamics. The PPO importance ratio ρg = πθ(g)/πold(g) is computed in the logarithmic domain as

G

G

exp(ℓhθ)) − (ℓgold − log

log ρg = log πθ(g) − log πold(g) = (ℓgθ − log

exp(ℓhold)). (7)

h=1

h=1

The generator parameters are then updated via the clipped PPO objective

G

1 G

min ρgAg,clip(ρg,1 − ϵlow,1 + ϵhigh)Ag , (8) where the normalized branch advantage is

LPPO(θ) = −

g=1

G

rg − r¯

1 G

rk, ϵ = 10−8. (9)

Ag =

, where r¯ =

G k=1 rk − r¯ 2 + ϵ

1 G

k=1

Here rg is the reward of branch g, clip(·) constrains the importance ratio within a trust region, and πold is updated once per optimization iteration. We adopt an asymmetric clipping range with ϵlow = 0.1 and ϵhigh = 0.2, which more aggressively promotes the optimization of high-reward branches while conservatively suppressing low-reward ones to prevent optimization collapse.

Derivation. We now verify that the velocity-field surrogate policy induces the desired preference optimization direction through its gradient.

- Step 1: We start from the unclipped Policy-Gradient (PG) objective underlying PPO: JPG(θ) = Eg∼πold π πθ(g)

old(g)Ag . Taking the gradient with respect to θ gives ∇θJPG(θ) = Eg∼πold[∇πθπθ(g)

old(g) Ag]. Applying the log-derivative identity ∇θ logπθ(g) = ∇πθπθ(g)

θ(g) , we obtain ∇θJPG(θ) = Eg∼πold[ππθ(g)

old(g)Ag ∇θ log πθ(g)]

=

G

g=1

πold(g)ππθ(g)

old(g)Ag ∇θ log πθ(g) =

G

g=1

πθ(g)Ag ∇θ log πθ(g). (10)

- Step 2: From Eq. (6), we have logπθ(g) = −Eθ(X

g)

τ − log h exp −Eθ(X

h)

τ . Therefore, ∇θ logπθ(g) = −τ1 ∇θEθ(Xg) − Gk=1 πθ(k)∇θEθ(Xk) . Substituting into (10) yields

∇θJPG(θ) = −

1 τ

G

g=1

πθ(g) Ag(∇θEθ(Xg) −

G

k=1

πθ(k)∇θEθ(Xk)) (11)

= −

1 τ

[

G

g=1

πθ(g) Ag ∇θEθ(Xg) − (

G

g=1

πθ(g) Ag)(

G

k=1

πθ(k) ∇θEθ(Xk))]

= −

1 τ

G

g=1

πθ(g)(Ag −

G

k=1

πθ(k)Ak)∇θEθ(Xg). (12)

- Step 3: Using the TVE definition in Eq. (5), its gradient with respect to θ is

S

2 d

J⊤vθ(vθ zb,sg , ts, K<b − uˆgb,s), (13)

∇θEθ(Xg) =

s=1

b∈B

where Jvθ denotes the Jacobian of the velocity field with respect to the network parameters. Substituting Eq. (13) into Eq. (12), we obtain

G

S

1 τ b∈B

2 d

πθ(g)(Ag − µπAθ)

J⊤vθ(vθ zb,sg , ts, K<b − uˆgb,s), (14)

∇θJPG(θ) = −

g=1

s=1

where µπAθ = Gk=1 πθ(k)Ak denotes the policy-weighted average advantage. Equation (14) reveals that policy optimization over TVE reduces to a reward-weighted contrastive flow-

matching objective. When branch g is superior (Ag > µπAθ), the gradient drives parameter updates toward reducing the replay residual against uˆgb,s, aligning the ODE dynamics more consistent with the high-reward trajectory. Conversely, when branch g is inferior (Ag < µπAθ), updates are steered away from reproducing that branch’s velocity targets, suppressing lowreward trajectory dynamics. This establishes TVE as a principled bridge between reinforcement learning and the generator’s native flow-matching objective [8].

##### 3.4 Reward Design and Regularization

Multi-reward Formulation. To mitigate reward hacking [9], we adopt a composite reward integrating three complementary dimensions: Visual Quality (VQ), Motion Quality (MQ), and Text-Video

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

LongLive

###### LongLive

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

###### +KVPO

+KVPO

[Seg4] Wet sidewalks reflect neon billboards while distant traffic hums in the damp air.

[Seg1] A sleek matte-black sci-fi drone gliding through a rain-soaked cyberpunk alley.

[Seg2] Its undercarriage LED lights pulse in rhythmic cyan and magenta sequences.

[Seg3] As the camera tilts upward, the drone flies past a towering holographic skyscraper.

[Seg2] The leaf lands on the water, creating concentric ripples that distort tree reflections.

[Seg3] Rustling leaves blend with soft rhythmic lapping of water against the rock.

[Seg1] A mossy stone in a sundappled clearing as a golden leaf spirals down from the upper left.

[Seg4] A smaller leaf drifts in from the right, joins the first, and both float in the current.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

LongLive

LongLive

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

+KVPO

+KVPO

[Seg1] A spaceship streaks through space, its hull gleaming under a nebula’s bright glare.

[Seg3] As the view widens, a massive gas giant fills the frame with swirling red and gold bands.

[Seg4] The spaceship flies toward the giant white storm eye, with galaxies faintly visible behind.

[Seg2] The ship races past a cluster of asteroids scarred with deep craters and dust trails.

[Seg1] Under a twilight sky, an astronaut in black walks among towering obsidian mushroom stalks.

[Seg3] The astronaut pulls up a mushroom and gently rotates it to examine the sample in hand.

[Seg2] The astronaut stops at the base of a mushroom, its cap pulsing with rhythmic purple light.

[Seg4] The mushroom releases spores that drift upward from his hand into the cool air.

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

LongLive

LongLive

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

+KVPO

+KVPO

[Seg1] The camera glides over a sea of colorful saris and denim jackets in a crowded Indian cricket stadium.

[Seg2] An eight-year-old boy in faded indigo shirt and straw hat crouches barefoot on soft soil.

[Seg4] The crowd sings the national anthem together under the waving flag.

[Seg4] The butterfly’s vivid orange and black wings glow warmly in the sunset light.

[Seg3] He slowly extends a finger toward a monarch butterfly on a tall blue cornflower.

[Seg1] Sunlight filters through trees, casting dappled shadows on a dirt path lined with wildflowers.

[Seg2] The audience suddenly stood up, leaning forward to cheer.

[Seg3] They unfurl a large red, white and green flag in the crowd.

- Figure 3: Qualitative comparison between LongLive and LongLive trained with KVPO. KVPO yields more faithful prompt grounding, cleaner object interactions, and smoother temporal evolution.

Alignment (TA). The VQ reward is computed as the average HPSv3 score [14], while MQ and TA rewards are obtained via the official VideoAlign configuration [11]. For long-video generation, rewards are computed per segment and averaged across segments.

KL Regularization. To prevent the surrogate policy from drifting excessively from the pretrained distribution, we augment the objective with a discrete KL divergence penalty:

DKL(πθ∥πref) =

G

g=1

πθ(g) [log πθ(g) − log πref(g)] . (15)

Here πref denotes the frozen reference policy constructed with the same surrogate mapping. The total training objective combines the PPO loss (Eq. 8) with the KL penalty (Eq. 15):

Ltotal = LPPO + β DKL(πθ∥πref), (16)

where β controls the KL penalty strength. To guard against occasional pathological exploration causing model degradation, KVPO zeros out the gradient for any iteration in which no candidate branch reward exceeds the anchor reward r0.

- 4 Experiments

##### 4.1 Experimental Setup

Implementation Details. We evaluate KVPO on two state-of-the-art autoregressive video generators, LongLive [23] and MemFlow [6]. Both are obtained via classical Self-Forcing-style [5] distillation and support single-prompt and multi-prompt generation. We also compare against Astrolabe [30], a state-of-the-art post-training method for AR video generation. Training prompts are sampled from the multi-prompt VidProM dataset [19] and further refined using Qwen3 [22]. Each video is uniformly segmented into groups of four prompts, with prompt switching every 588 frames (147 latent frames). For parameter-efficient fine-tuning, we apply LoRA [4] with rank r = 256 and scaling factor α = 256. All experiments are conducted on 32 NVIDIA H200 GPUs, where each training iteration processes 32 prompts with a candidate group size of G = 8. Each iteration takes approximately 960 seconds, and the best checkpoint typically emerges within 3,000–4,000 training samples, corresponding to roughly 30 hours of wall-clock time and about 1000 GPU-hours. Additional key training hyperparameters are summarized in Appendix G.

Evaluation Metrics. We evaluate KVPO under two settings: single-prompt short-video and multiprompt long-video generation. In addition to the three primary metrics used by our reward design, we report four complementary VBench [32] metrics, namely Quality, Semantic, Consistency Score, and CLIP Score, to provide a comprehensive assessment of model performance.

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

MemFlow

MemFlow

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

###### +KVPO

+KVPO

[Seg4] A soft glow from the UFO lights the misty treetops and golden foliage.

[Seg2] A musician in a worn denim jacket steps off the bus with a guitar.

[Seg3] The musician plays guitar by a crackling campfire.

[Seg1] Purple twilight spreads across the horizon, casting long slanted shadows on dewy grass.

[Seg3] The silver craft hovers above trees, reflecting sunset across the grass below.

[Seg2] A UFO descends from above, its amber underside lights growing brighter.

[Seg4] The camera slowly moves forward beneath a vast sky of orange and pink hues.

[Seg1] An old weathered bus is parked on a gravel mountain overlook.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

MemFlow

MemFlow

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

###### +KVPO

+KVPO

[Seg4] Shibuya Crossing glows behind skyscrapers, neon reflecting off wet pavement.

[Seg3] She pauses at a lit vendor stall, studying colorful trinkets as crowds blur past.

[Seg1] Chicago glows at twilight as a massive firestorm rolls in from the west.

[Seg3] The John Hancock Center darkens as its sharp reflections fade into smoke.

[Seg4] Black smoke and orange flame rise over avenues where headlights struggle through gloom.

[Seg1] A sleek black sedan glides along a wet Tokyo street at night.

[Seg2] The Willis Tower stands against smoke, its windows flickering with dying power.

[Seg2] The car stops gently as a young woman steps out in a cherry-blossom kimono.

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

MemFlow

MemFlow

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

###### +KVPO

###### +KVPO

[Seg2] A silver trout suddenly leaps from the depths, creating perfect concentric ripples.

[Seg3] A lone figure in a worn sweater sits on a wooden log by the water.

[Seg1] Snow-capped peaks catch the first faint light of dawn.

[Seg4] The rising sun casts long golden-orange beams across the illuminated peaks.

[Seg4] Amber light bathes the skeleton and spreads across the cracked earth.

[Seg3] The lens descends, revealing ribcage textures against the butterfly's vivid wings.

[Seg1] A weathered skeleton lies among wildflowers beneath a hazy afternoon light.

[Seg2] A delicate orange butterfly lands softly on the skull's hollow eye socket.

Figure 4: Qualitative comparison between MemFlow and MemFlow trained with KVPO. KVPO yields clearer semantic transitions, richer details, and stronger cross-segment consistency.

##### Table 1: Comparison of single-prompt short-video and multi-prompt long-video generation.

Method VQ↑ MQ↑ TA↑ Quality↑ Semantic↑ Consistency Score↑ CLIP Score↑

###### Single-prompt short-video generation

LongLive [23] 8.86 1.80 0.02 81.89 70.10 89.12 32.01 + Astrolabe 9.98 1.87 0.03 81.41 70.61 89.14 32.31 + KVPO 10.21 (⇑15.2%) 1.89 (⇑5.0%) 0.06 (⇑200.0%) 81.44 71.45 89.56 32.29

MemFlow [6] 8.83 1.82 0.02 80.72 71.31 88.74 31.96 + Astrolabe 9.47 1.85 0.03 80.13 71.42 88.87 32.04 + KVPO 9.71 (⇑9.1%) 1.87 (⇑2.7%) 0.03 (⇑50.0%) 80.91 71.65 89.08 32.17

###### Multi-prompt long-video generation

LongLive [23] 6.34 1.41 -0.19 78.42 67.88 88.37 31.90 + Astrolabe 7.26 1.44 -0.18 78.46 68.36 88.30 32.18 + KVPO 8.14 (⇑28.4%) 1.50 (⇑6.4%) -0.14 (⇑26.3%) 79.31 69.02 88.62 32.29

MemFlow [6] 6.30 1.39 -0.20 77.95 68.11 87.34 31.80 + Astrolabe 6.52 1.35 -0.23 78.02 67.94 87.35 31.86 + KVPO 6.96 (⇑10.5%) 1.44 (⇑3.6%) -0.17 (⇑15.0%) 78.36 68.74 87.52 32.34

##### 4.2 Quantitative and Qualitative Results

[Figure 209]

As shown in Table 1, KVPO achieves consistent improvements across all reward and VBench metrics in both single-prompt shortvideo and multi-prompt long-video settings. In the short-horizon setting, KVPO improves LongLive by 15.2%, 5.0%, and 200.0% on VQ, MQ, and TA respectively, and MemFlow by 9.1%, 2.7%, and 50.0%. In the long-horizon setting, LongLive improves by 28.4%, 6.4%, and 26.3%, while MemFlow improves by 10.5%, 3.6%, and 15.0% on the same metrics. KVPO also consistently outperforms Astrolabe [30], with the margin widening in the multi-prompt long-video setting. We attribute this to causal-semantic exploration, which yields richer and semantically coherent optimization signals that better guide storyline evolution, whereas Astrolabe’s noise-based exploration primarily affects low-level appearance rather than high-level semantic development.

[Figure 210]

MemFlow (Inner ring)

LongLive (Outer ring)

[Figure 211]

[Figure 212]

Base

Base

21.8

18.7

Base

18.7

25.0

- 0.52
- 1
- 2.5

6 2.5

5.6

###### 5 9.4

46.9

53.1

- 0.52
- 1

46.9

50.0

KVPO KVPO KVPO

VQ MQ TA

3.23

1.82

- 3
- 4.4

1.82

Astrolable Astrolable

[Figure 213]

[Figure 214]

Astrolable

Figure 5: Human study on long-video settings.

[Figure 215]

[Figure 216]

[Figure 217]

(a) Visual Quality (VQ)

(b) Motion Quality (MQ)

(c) Text-Video Alignment (TA)

Figures 3 and 4 visually confirm these improvements. Relative to the baselines, the optimized models exhibit more accurate prompt grounding, cleaner object boundaries, more plausible motion conti-

LongLive Base Model

LongLive Base Model

LongLive Base Model

Training Steps

Training Steps

Training Steps

Figure 6: Performance improvements on LongLive.

nuity, and better preservation of subject identity across segments, with fewer abrupt semantic shifts and less structural drift under viewpoint and prompt changes. We further evaluate perceptual quality via a human study in which 32 instructed participants performed preference voting among the baseline, Astrolabe [30], and KVPO across VQ, MQ, and TA. Figure 5 reports the proportion of samples voted best by participants under each metric, where KVPO secures a clear majority preference over all competing methods. We attribute these advantages to two design choices. Causal-Semantic Exploration enables semantically meaningful variation over causal history, improving storyline coherence in a way prior post-training methods struggle to achieve. The Velocity-Field Surrogate Policy provides an ODE-native optimization signal aligned with the generator’s intrinsic flow-matching dynamics, more effectively embedding human preferences into the model. Furthermore, Appendix H provides additional qualitative results.

- 4.3 Ablation Studies Table 2: Ablation of CHR and surrogate policy on LongLive in the multi-prompt long-video setting.

- Table 2 presents a comprehensive ablation of the core components of KVPO on LongLive [23] under the multi-prompt long-video setting, with additional ablations provided in Appendix E.

Factor Variant VQ↑ MQ↑ TA↑ Perturbed blocks 3 6.92 1.43 -0.18 Perturbed blocks 5 8.14 1.50 -0.14 Perturbed blocks 7 8.10 1.53 -0.16 Perturbed local KV slots 3 / 9 6.22 1.36 -0.20 Perturbed local KV slots 6 / 9 8.14 1.50 -0.14 Perturbed local KV slots 9 / 9 6.97 1.43 -0.17 Local KV length Fixed 9 8.14 1.50 -0.14 Local KV length Random {6,9,12} 8.11 1.48 -0.15

Causal History Routing (CHR). We ablate the main CHR hyperparameters, including the number of perturbed blocks, perturbed local KV slots, local KV window length, and perturbed denoising steps. First, perturbing 5 blocks yields the best overall trade-off, since using only 3 blocks likely provides insufficient semantic variation and weakens all metrics, while 7 blocks offer no consistent gain while incurring higher memory cost. Second, perturbing 6 of the 9 local KV slots is optimal. Too few perturbed slots produce overly similar branches and thus weak preference signals, whereas perturbing all 9 disrupts short-range temporal anchoring and destabilizes generation. Third, randomizing the local KV window length brings negligible improvement, suggesting that the default local window already balances causal context and exploration diversity while remaining matched to inference. Fourth, perturbing the first two denoising steps yields the most balanced improvement across all three metrics. Perturbing only one step provides insufficient intervention on coarse semantics and motion layout, whereas perturbing additional steps significantly degrades visual quality and substantially increases memory overhead.

- Perturbed solver steps 1 7.12 1.43 -0.18
- Perturbed solver steps 2 8.14 1.50 -0.14
- Perturbed solver steps 3 7.65 1.51 -0.12
- Perturbed solver steps 4 7.41 1.46 -0.17

Surrogate policy Geometric latent ℓ2 6.02 1.43 -0.21 Surrogate policy TVE 8.14 1.50 -0.14

TVE-based Surrogate Policy. Replacing TVE with a geometric latent-space ℓ2 surrogate, similar to that used in NeighborGRPO [1], substantially degrades performance across all metrics, demonstrating that velocity-field-space policy modeling is critical for effective preference optimization under autoregressive ODE replay. We further analyze the limitations of Euclidean-distance surrogate policies and the advantages of our formulation in the Appendix D.

### 5 Conclusion

We studied preference alignment for streaming autoregressive video generators under the deterministic ODE regime. To address the mismatch between existing noise-driven reinforcement learning methods and ODE-based generation, we introduced KVPO, which combines causal-semantic exploration through Causal History Routing (CHR) with a velocity-field surrogate policy based on Trajectory Velocity Energy (TVE). CHR redirects exploration to the historical key-value cache, inducing semantically meaningful and on-manifold candidate branches, while the TVE-based surrogate policy keeps preference optimization within the model’s native flow-matching dynamics. Experiments on distilled autoregressive video generators demonstrate consistent improvements in visual quality, motion quality, and text-video alignment across both short-video and long-video generation settings. More broadly, KVPO suggests that semantic-space exploration offers a principled alternative to noise-based perturbation for inducing on-manifold branch diversity, and that velocity-field space provides a natural domain for surrogate policy modeling faithful to the generator’s intrinsic dynamics. These insights may inform future alignment research on ODE-based generative architectures.

### References

- [1] He, D., Feng, G., Ge, X., Niu, Y., Zhang, Y., Ma, B., Song, G., Liu, Y., Li, H.: Neighbor grpo: Contrastive ode policy optimization aligns flow models. arXiv preprint arXiv:2511.16955

- (2025)

[2] He, D., Feng, G., Ge, X., Zhang, Y., Ma, B., Song, G., Liu, Y., Li, H.: Ar-copo: Align autoregressive video generation with contrastive policy optimization. arXiv preprint arXiv:2603.17461

- (2026)

- [3] Hu, B., Qi, Z., Huang, G., Xu, Z., Zhang, R., Ye, C., Zhou, J., Li, X., Wang, J.: Identityconsistent video generation under large facial-angle variations. arXiv preprint arXiv:2603.21299

(2026)

- [4] Hu, J.E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Chen, W.: Lora: Lowrank adaptation of large language models. ArXiv abs/2106.09685 (2021), https://api. semanticscholar.org/CorpusID:235458009
- [5] Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009 (2025)
- [6] Ji, S., Chen, X., Yang, S., Tao, X., Wan, P., Zhao, H.: Memflow: Flowing adaptive memory for consistent and efficient long video narratives (2025), https://arxiv.org/abs/2512.14699
- [7] Ji, Y., Zhong, Z., Zhang, J., Yang, Q., Jin, X., Qin, Y., Luo, W., Mao, S., Liu, W., Li, H.: Forcing-kv: Hybrid kv cache compression for efficient autoregressive video diffusion models

(2026), https://arxiv.org/abs/2605.09681

- [8] Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum?id=PqvMRDCJT9t
- [9] Liu, G., Yang, B., Zhi, Y., Zhong, Z., Ke, L., Deng, D., Gao, H., Huang, Y., Zhang, K., Fu, H., et al.: Beyond vlm-based rewards: Diffusion-native latent reward modeling. arXiv preprint arXiv:2602.11146 (2026)
- [10] Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., Ouyang, W.: Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470 (2025)
- [11] Liu, J., Liu, G., Liang, J., Yuan, Z., Liu, X., Zheng, M., Wu, X., Wang, Q., Xia, M., Wang, X., et al.: Improving video generation with human feedback. arXiv preprint arXiv:2501.13918

(2025)

- [12] Lu, Y., Zeng, Y., Li, H., Ouyang, H., Wang, Q., Cheng, K.L., Zhu, J., Cao, H., Zhang, Z., Zhu, X., et al.: Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678 (2025)
- [13] Luo, J., Lin, J., Zhang, Z., Wu, B., Fang, M., Chen, L., Tang, H.: Univid: The open-source unified video model. arXiv preprint arXiv:2509.24200 (2025)
- [14] Ma, Y., Wu, X., Sun, K., Li, H.: Hpsv3: Towards wide-spectrum human preference score (2025), https://arxiv.org/abs/2508.03789
- [15] Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)
- [16] Shao, Z., Wang, P., Zhu, Q., Yang, R., Xu, J., Li, M., Zhang, Y.W., Li, Y.K., Gao, Y., Ma, D., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)
- [17] Song, Y., Dhariwal, P., Chen, M., Sutskever, I.: Consistency models. In: International Conference on Machine Learning (2023)

- [18] Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-based generative modeling through stochastic differential equations. In: International Conference on Learning Representations (2021)
- [19] Wang, W., Yang, Y.: Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. In: Thirty-eighth Conference on Neural Information Processing Systems

(2024), https://openreview.net/forum?id=pYNl76onJL

- [20] Xu, Z., Yu, Z., Zhou, Z., Zhou, J., Jin, X., Hong, F.T., Ji, X., Zhu, J., Cai, C., Tang, S., et al.: Hunyuanportrait: Implicit condition control for enhanced portrait animation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15909–15919 (2025)
- [21] Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al.: Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818

(2025)

- [22] Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., Qiu, Z.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)
- [23] Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., Chen, S.H.Y.: Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622 (2025)
- [24] Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. In: Conference on Computer Vision and Pattern Recognition (2024)
- [25] Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. In: CVPR (2024)
- [26] Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., Shechtman, E., Huang, X.: From slow bidirectional to fast autoregressive video diffusion models. In: CVPR (2025)
- [27] Zhang, R., Chen, G., Xu, Z., Liu, Z., Zhong, Z., Zhang, M., Zhou, J., Li, X.: Robostereo: Dual-tower 4d embodied world models for unified policy optimization. arXiv preprint arXiv:2603.12639 (2026)
- [28] Zhang, R., Zhang, M., Zhou, J., Guo, Z., Liu, X., Xu, Z., Zhong, Z., Yan, P., Luo, H., Li, X.: Mind-v: Hierarchical video generation for long-horizon robotic manipulation with rl-based physical alignment. arXiv preprint arXiv:2512.06628 (2025)
- [29] Zhang, R., Zhou, J., Xu, Z., Liu, Z., Huang, J., Zhang, M., Sun, Y., Li, X.: Zo3t: Zero-shot 3d-aware trajectory-guided image-to-video generation via test-time training. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 12708–12716 (2026)
- [30] Zhang, S., Xue, Z., Fu, S., Huang, J., Kong, X., Ma, Y., Huang, H., Duan, N., Rao, A.: Astrolabe: Steering forward-process reinforcement learning for distilled autoregressive video models. arXiv preprint arXiv:2603.17051 (2026)
- [31] Zhao, W., Bai, L., Rao, Y., Zhou, J., Lu, J.: Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems 36, 49842–49869 (2023)
- [32] Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Zhang, Y., He, J., Zheng, W.S., Qiao, Y., Liu, Z.: VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755 (2025)
- [33] Zheng, M., Kong, W., Wu, Y., Jiang, D., Ma, Y., He, X., Lin, B., Gong, K., Zhong, Z., Bo, L., et al.: Manifold-aware exploration for reinforcement learning in video generation. arXiv preprint arXiv:2603.21872 (2026)

## Appendix

### A KVPO Training Pipeline

We summarize the full training procedure of KVPO. At a high level, each iteration consists of three tightly coupled stages: semantic exploration, replay-based surrogate policy construction, and policy optimization. KVPO first shares the initial latent xT and perturbs the composition of the local KV cache to generate a group of semantically diverse candidate branches through Causal History Routing (CHR). These branches are then evaluated by the reward model to produce group advantages. The same trajectories are replayed under the unperturbed context to compute Trajectory Velocity Energy (TVE), from which KVPO constructs the Gibbs-form surrogate policy and the PPO importance ratios. Finally, the generator is updated using the clipped PPO objective together with KL regularization toward the reference policy. Algorithm 1 gives the complete procedure.

Algorithm 1: KVPO Training

Input: Generator vθ, old policy πold, reference policy πref, reward model R, prompt C, latent xT, history cache K<b, branch number G, solver steps S, perturbation window B, PPO clips ϵlow, ϵhigh, KL weight β, temperature τ

Output: Optimized generator vθ∗ for iteration n = 1,2, . . . do

- // Step 1: CHR-based semantic exploration

Sample pivot block b∗ and anchor video X0 ← Φ(xT; θ,K<b); for g = 1, . . . , G do

Sample CHR refill sets {Jmg }6m=1 and form K˜<bg,local∗ ← CHR(K<bhist∗; J1:6g ); Generate branch Xg ← Φ(xT; θ,K˜<bg,local∗ ,B);

end

- // Step 2: Cache rollout states for replay

For each branch g and replay step (b, s) ∈ B × [S], cache T g ← T g ∪ {(zb,sg ,uˆgb,s)}, then restore the default local cache after B;

- // Step 3: Compute group-normalized advantages Evaluate rewards rg ← R(Xg,C) for all g; Compute r¯ ← G1 Gk=1 rk and σr ← G1 Gk=1(rk − r¯)2 + ϵ; Set Ag ← rgσ−r¯

r

for each branch g;

- // Step 4: Replay and construct the Gibbs surrogate

Replay all branches under the shared unperturbed context K<b; for g = 1, . . . , G do

Eθ(Xg) ← b∈B

S s=1

1 d ∥vθ(zb,sg , ts,K<b) − uˆgb,s∥2F;

ℓgθ ← −Eθ(Xg)/τ, πθ(g) ← exp(ℓ

g θ)

G

h=1 exp(ℓhθ), ρg ← ππθ(g)

old(g); end

- // Step 5: PPO-KL update

LPPO ← −G1 Gg=1 min ρgAg, clip(ρg,1 − ϵlow,1 + ϵhigh)Ag ; DKL ← Gg=1 πθ(g) log πθ(g) − log πref(g) ; Ltotal ← LPPO + β DKL; θ ← θ − η ∇θLtotal, πold ← πθ;

end return vθ∗

### B Why KV Exploration?

We clarify why the historical KV cache is a particularly suitable locus for exploration. For streaming autoregressive video generation, an effective exploration locus should satisfy three criteria: it should meaningfully influence future generation, remain compatible with the model’s native causal generation pathway, and induce diversity at the level most relevant to preference optimization. The historical KV cache is well aligned with these criteria. First, because future content is strongly conditioned on accumulated historical context, modifying routed local KV memory directly affects subsequent

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Anchor

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

KV Exploration

KV Exploration

##### Figure 7: Visualization of the effectiveness of causal-semantic exploration.

continuation. Second, this intervention operates on the model’s native conditioning state rather than introducing external stochastic perturbations into the latent trajectory. Third, because preferences for long-horizon video depend heavily on temporal coherence, subject consistency, and semantic progression, varying historical KV composition is better suited to induce semantically distinct branches than perturbations that primarily alter local appearance.

### C Conditional Marginal Preservation

We show that CHR preserves the model’s conditional marginal distribution. For an ODE-based generator, this means that, for any fixed conditioning context K, the deterministic generation map transports the base noise distribution to the corresponding conditional model distribution.

Under the probability-flow ODE framework [18], if generation is written as

x0 = Φ(xT; θ,K), xT ∼ N(0,I), (17)

then, for fixed K, the induced sample distribution of x0 is exactly the model’s conditional distribution pθ(x0 | K). In this sense, the conditional marginal is preserved because sampling is still performed by solving the original deterministic ODE from a Gaussian initial latent.

The role of CHR is therefore not to modify this noise-to-sample transport itself, but only to change the conditioning context. Since CHR leaves the initial latent xT unchanged, each branch is generated as

xg0 = Φ(xT; θ,K˜<bg,local∗ ), (18) and thus remains an exact sample from the corresponding conditional model distribution

xg0 ∼ pθ x0 | K˜<bg,local∗ . (19)

Accordingly, CHR changes the branchwise conditioning context and hence the semantic trajectory, but it does not introduce an off-manifold perturbation to the underlying generative process.

### D Rethinking ODE-based Policy Optimization

We discuss and compare KVPO with recent purely ODE-based policy optimization methods. Recent advances in purely ODE-based GRPO optimization (e.g., NeighborGRPO [1]) construct an exploration neighborhood by perturbing the initial latent and approximate the surrogate policy via Euclidean distance between generated latents. In this section, we analyze the theoretical limitations of such noise-driven exploration in autoregressive video generation and highlight the advantages of KVPO from two perspectives: diversity exploration and surrogate policy modeling.

##### D.1 Diversity Exploration: Disentangling Policy from Noise Variance

Diverse candidate samples are a prerequisite for effective policy optimization. Let the generation process be denoted by x0 = Φ(xT;θ,K). In noise-driven exploration, candidate samples are generated from a perturbed initial latent xiT = x∗T + σδi, where δi ∼ N(0,I). Applying a first-order Taylor expansion with respect to the initial latent, the variation in the generated sample ∆xi0 = xi0 − x∗0 can be approximated as:

Φ(x∗T;θ,K) · (σδi) (20)

∆xi0 ≈ ∇xT

When prior works model the surrogate policy πθ(i) using the Euclidean distance ∥∆xi0∥22, a fundamental confounding issue arises. The variation in ∆xi0 is inherently coupled with the random noise seed δi. Therefore, a larger distance does not necessarily imply a lower likelihood under the current surrogate policy θ; it may simply correspond to a larger sampled noise magnitude. This conflation corrupts the true policy gradient signal.

Advantage of Causal-Semantic Exploration. KVPO resolves this ambiguity by shifting exploration from noise perturbation to causal-semantic exploration over the historical KV cache. Concretely, it keeps the initial latent xT fixed across all candidates, while CHR instantiates the branch-specific local memory K˜<bg,local∗ . Because the initial latent is shared, any structural or semantic difference among the generated trajectories can be attributed solely to the model’s deterministic response to the distinct context. This disentanglement ensures that the exploration space directly reflects the surrogate preference ordering induced by changes in the historical KV cache rather than by differences in noise magnitude.

##### D.2 Surrogate Policy Modeling: Overcoming Geometric Distortion

Modeling the surrogate policy via unweighted Euclidean distance intrinsically imposes an isotropic assumption, treating every feature dimension as contributing equally to likelihood evaluation:

- 1

- 2σπ2 ∥xi0 − x∗0∥22 (21)

πnoise(i) ∝ exp −

This implies a local covariance of Σassume = σπ2I.

However, the probability flow ODE Φ(·;θ) mapping xT to x0 is nonlinear. By the change-of-variables theorem, the true local covariance pushed forward from the isotropic prior is governed by the Jacobian JΦ:

Σtrue = σ2JΦ(x∗T)JΦ(x∗T)⊤ (22) In AR models spanning multiple continuous blocks and diffusion steps, this cascaded nonlinear transformation yields an anisotropic true distribution (JΦJ⊤Φ ̸= I). The unweighted Euclidean distance implicitly ignores this Riemannian curvature, introducing geometric distortion that unjustly penalizes valid semantic variations along the principal axes of the flow.

Advantage of velocity-field surrogate policy modeling. KVPO resolves this by constructing the surrogate policy via trajectory replay in the velocity-field space, with TVE serving as the energy functional evaluating each candidate under the unperturbed deployment-time context:

πθ(i) ∝ exp −Eθ(Xi | K<b)/τ (23)

Since TVE is defined through velocity-field MSE accumulated along the replayed ODE trajectory, it directly mirrors the model’s native flow-matching objective and provides a geometrically sound, algorithmically aligned gradient signal for AR video policy optimization.

### E Ablation on the KL Penalty Weight

We study the effect of the KL regularization coefficient β in Eq. (15). This term anchors the learned surrogate policy to the frozen reference and is essential for stable online preference optimization.

Without it (β = 0), the policy drifts aggressively toward noisy high-reward branches, the PPO update becomes poorly regularized, and the replay-time distribution diverges too far from the pretrained generator — in practice, this leads to training collapse rather than meaningful reward optimization. Conversely, excessively large β keeps training stable but renders the update too conservative. Optimal performance is thus expected at an intermediate regularization strength.

- Table 3 confirms this on LongLive [23]. At β = 0, all metrics drop substantially below the base model in both settings, indicating collapse in both alignment quality and generation stability. Introducing even a moderate KL constraint restores stable learning: β = 1 recovers most of the gains, and β = 3–5 yields the strongest overall trade-off. The default β = 5 achieves the most balanced performance across reward metrics and auxiliary VBench metrics. Beyond this, gains diminish gradually as the optimization grows increasingly conservative.

Table 3: Ablation on the KL penalty weight β for LongLive.

KL Weight β VQ↑ MQ↑ TA↑ Quality↑ Semantic↑ Consistency Score↑ CLIP Score↑

Single-prompt short-video generation

Base 8.86 1.80 0.02 81.89 70.10 89.12 32.01

- β = 0 6.92 1.36 -0.07 77.42 65.96 85.84 30.31
- β = 1 10.04 1.87 0.07 81.21 71.19 89.28 32.23 β = 3 10.22 1.88 0.06 81.37 71.33 89.46 32.27 β = 5 10.21 1.89 0.06 81.44 71.45 89.56 32.29 β = 10 10.13 1.87 0.05 81.30 71.18 89.34 32.17 β = 20 9.68 1.84 0.04 81.08 70.84 89.23 32.10

Multi-prompt long-video generation

Base 6.34 1.41 -0.19 78.42 67.88 88.37 31.90

- β = 0 4.96 1.18 -0.33 76.11 64.88 85.17 30.08
- β = 1 8.02 1.48 -0.14 78.98 68.79 88.36 32.19 β = 3 8.13 1.49 -0.14 79.20 68.93 88.55 32.25 β = 5 8.14 1.50 -0.14 79.31 69.02 88.62 32.29 β = 10 7.96 1.48 -0.15 79.05 68.84 88.50 32.18 β = 20 7.58 1.45 -0.16 78.73 68.46 88.34 32.06

- F Limitations

Although KVPO shows consistent gains on the distilled autoregressive video generators studied in this paper, it still has several limitations. First, the proposed causal-semantic exploration mechanism is designed for autoregressive generators with KV-cache-based memory. While this design is natural for many mainstream autoregressive video models, extending it to models without KV caches or with substantially different memory mechanisms (e.g., Mamba-style state-space models) may require additional adaptation. Second, although KVPO introduces a novel ODE-native velocity-field surrogate for preference optimization, training incurs additional computational and memory cost due to the need to cache replay states. Although this cost grows with the number of branches, the replay horizon, and the solver steps, it remains small relative to the peak memory required for backpropagation. Finally, the optimization quality depends on the fidelity of the reward model. If the reward does not adequately capture long-range visual quality, narrative consistency, or subtle motion realism, the resulting KVPO updates may not fully reflect human preferences. Future work will focus on extending KVPO to a broader range of video generation architectures and developing stronger reward models for long-horizon semantic and motion consistency.

- G Key KVPO Training Hyperparameters

- Table 4 summarizes the key hyperparameters used in KVPO training. Figure 8 reports the GPU memory footprint and compute utilization during LongLive training, with peak memory usage of approximately 130GB and a per-step runtime of approximately 960 seconds.

Table 4: Comprehensive hyperparameters for KVPO training. We summarize the configurations used for model setup, optimization, replay-based reinforcement learning, and streaming rollout.

Category Setting Value Model & Video Specs Denoising Timesteps (T) 4 from [1000,750,500,250] Local Attention Size 12 Sink Size 3 Local Size 9 Frames per Block 3

LoRA Fine-Tuning Rank (r) 256 Scaling Factor (α) 256 Dropout Rate 0.0 Gradient Checkpointing True EMA Decay Rate 0.999

Optimization Hardware 32 NVIDIA H200 GPUs Optimizer AdamW Learning Rate (η) 1 × 10−5 LR Scheduler constant_with_warmup Warmup Steps 5 Weight Decay 0.01 Gradient Accumulation Steps 2 Max Gradient Norm 1.0 Max Train Steps 1500

Replay-Based RL Branch Number (K) 8 Perturbed Blocks (P) 5 Nearest Preserved Frames (m) 3 Perturb Search Range first 7 chunks Reward Signals local + global Local / Global Loss Weights 1.0 / 0.5 Gradient-Carrying Replay Steps 2 PPO Epochs 1 Clip Range (ϵlow, ϵhigh) (0.1, 0.2) Advantage Clip Max 2.5 KL Penalty Weight 5 KL Reference Policy frozen initialization policy

Streaming Rollout Chunk Size 21 Prompt Segments 4 Switch Frame Indices [39, 75, 111] VAE Temporal Stride 4

(a) GPU Memory Used

[Figure 242]

(b) GPU Utilization

[Figure 243]

##### Figure 8: GPU memory usage and compute utilization of KVPO during LongLive training.

### H More Qualitative Results

This section provides additional qualitative comparisons complementing the quantitative results in the main paper, covering both LongLive [23] and MemFlow [6] settings. The comparisons span diverse prompts and scene transitions, enabling evaluation of long-horizon temporal coherence, subject identity preservation, motion plausibility, and text-video alignment. Across all presented cases, KVPO consistently exhibits stronger semantic consistency and more stable narrative progression than the compared baselines.

#### LongLive +KVPO

#### MemFlow

#### +KVPO

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

A ten-headed crimson demon emerges from smoke behind the throne, roaring as the ground trembles.

A woman restores a drooping crimson rose, removing a withered petal and pressing in a fresh dewy one.

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

A timelapse of messy notes transforming into paintings and sculptures, ending with polished works displayed in warm light.

A black diagonal band with a bright white stripe cuts across a flag, fluttering sharply in the wind.

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

A bird's path sends ripples across a lake, warping reflections of fiery clouds and distant pine silhouettes.

A lone figure stands on the edge of a dark concrete rooftop, gazing out over a city glowing in the night.

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

A woman in a blue suit stands before a screen, analyzing fluctuating green stock market curves for her colleagues.

A blonde girl steps into the morning light, holding a gently swaying red balloon.

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

A small medieval town with castle-like buildings, red and white maple leaf flags swaying gently in the breeze.

Schools of fish drift through the water as a large silver fish glides past, surrounded by a cluster of smaller fish.

Figure 9: Additional qualitative results on short-video generation.

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

LongLive

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

###### +KVPO

[Seg1] The camera pulls back from a glowing India map.

[Seg3] A satellite line appears and traces an elliptical orbit.

[Seg2] Political borders and terrain glow under amber and teal light.

[Seg4] The view settles on Earth's curvature and moving trajectory.

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

LongLive

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

+KVPO

[Seg3] She walks past pastel cottages with flowered window boxes.

[Seg1] A blonde girl enters holding a swaying red balloon.

[Seg4] The camera tracks her toward a quiet park entrance.

[Seg2] Morning light filters through old oaks onto mossy cobblestones.

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

LongLive

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

+KVPO

[Seg3] Wind whistles through broken windows as plants reclaim cracks.

[Seg1] Twilight reveals a shattered skyscraper and debris strewn street.

[Seg4] A dark bird lands near a discarded flickering phone.

[Seg2] Rusted electric vehicles and circuit boards litter the cobblestones.

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

LongLive

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

+KVPO

[Seg3] Fires and ruins cast long shadows across broken shields.

[Seg1] Dusk falls over a chaotic medieval battlefield.

[Seg4] He raises his sword as the sunset flashes brightly.

[Seg2] A silver knight rides in on a white stallion.

##### Figure 10: Additional qualitative results on LongLive.

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

LongLive

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

+KVPO

[Seg4] Smoke, forge sounds, and roosters fill the waking settlement.

[Seg3] Children chase a hoop while elders drink tea nearby.

[Seg2] A blue bird glides over rooftops and cobblestones.

[Seg1] Morning mist clings to green hills around the village.

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

LongLive

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

+KVPO

[Seg2] A devotee in white walks barefoot toward the flagstaff.

[Seg1] Golden mist lights the Bhagava Dhvaja above the Yamuna valley.

[Seg4] The camera reveals Ram Mandir's marble carvings and golden domes.

[Seg3] He drapes a marigold garland with reverent precision.

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

LongLive

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

+KVPO

[Seg3] She takes his gloved hand with quiet reverence.

[Seg1] Golden light dapples an oak forest around a waiting prince.

[Seg2] A barefoot peasant girl approaches in rough linen.

[Seg4] Blue butterflies rise as the camera closes on them.

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

LongLive

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

+KVPO

[Seg1] Grainy 1983 footage shows a city under nuclear devastation.

[Seg4] A military helicopter sweeps searchlight through the haze.

[Seg2] Skyscrapers collapse as smoke and debris fill the streets.

[Seg3] Civilians flee while emergency vehicles fight through gridlock.

##### Figure 11: Additional qualitative results on LongLive.

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

LongLive

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

+KVPO

[Seg1] A metallic robot drifts through a clear alpine lake.

[Seg3] Sunbeams and mist reflect across cyan circuitry.

[Seg4] Pines frame the robot as the camera tilts upward.

[Seg2] Its fin like limb steers ripples across the surface.

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

LongLive

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

+KVPO

[Seg1] Five stormtroopers wade waist-deep into churning dark ocean.

[Seg4] Astronauts walk ashore and open their helmets.

[Seg2] Camera tilts down, revealing towering waves surrounding them.

[Seg3] Astronauts adjust helmets seal as water pours down.

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

LongLive

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

+KVPO

[Seg3]The camera tilts toward a solitary rooftop figure.

[Seg2] Empty avenues glow faintly beneath jagged skyscraper silhouettes.

[Seg4] The figure shines a flashlight through the fog into the distance.

[Seg1] A drone sweeps across a fog blanketed metropolis.

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

LongLive

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

+KVPO

[Seg2] She spins slowly in an emerald silk dress.

[Seg3] Baskets nearby overflow with red geraniums and hibiscus.

[Seg4] Bees hum as the camera reveals sudden blooming.

[Seg1] Morning mist lifts from ferns around a young woman.

##### Figure 12: Additional qualitative results on LongLive.

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

LongLive

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

###### +KVPO

[Seg3] Pickaxes strike earth as pine forest dust hangs nearby.

[Seg4] The camera pans across hills, smoke, and long shadows.

[Seg2] Workers climb timber supports with iron buckets ready.

[Seg1] Sunset lights an 1859 Pennsylvania oil derrick.

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

LongLive

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

###### +KVPO

[Seg2] A rescue worker enters pushing a covered stretcher.

[Seg3] The camera tightens on their determined expression.

[Seg1] A damaged Gaza building burns under twilight smoke.

[Seg4] Fire, rubble, explosions, and wet pavement surround them.

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

LongLive

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

###### +KVPO

[Seg1] A drone crosses Rajasthan dunes toward Punjab valleys.

[Seg2] A red tractor pulls farmers through emerald wheat.

[Seg3] Irrigation canals cut silver lines through the fields.

[Seg4] Windmills hum as crops ripple in midday sun.

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

LongLive

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

+KVPO

[Seg1] Heat shimmers across the Giza construction site.

[Seg2] A linen kilted worker enters through blowing sand.

[Seg3] He signals laborers around a massive granite block.

[Seg4] The team strains forward beneath the unfinished pyramid.

##### Figure 13: Additional qualitative results on LongLive.

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

LongLive

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

+KVPO

[Seg3] A golden retriever puppy bounds in from the beach.

[Seg4] It shakes sparkling droplets beneath the canopy.

[Seg2] A purple umbrella ripples in the dunes.

[Seg1] A sunny cove stretches toward turquoise water.

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

LongLive

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

+KVPO

[Seg2] He reaches toward a curious duck waddling closer.

[Seg1] A boy crouches beside a calm pond.

[Seg4] Other ducks ripple across the water behind them.

[Seg3] The duck nuzzles his hand in golden light.

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

LongLive

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

+KVPO

[Seg4] The knight narrows his eyes with impossible determination.

[Seg1] A battered crimson knight stands in an ancient courtyard.

[Seg2] His nervous squire holds a startled white horse.

[Seg3] Three windmills loom across dusty rolling hills.

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

LongLive

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

+KVPO

[Seg2] Long shadows stretch across polished wooden desks.

[Seg1] Golden sunset fills an empty classroom.

[Seg3] A foreground drawer slowly creaks open.

[Seg4] Leather notebooks and a faded pencil wait inside.

##### Figure 14: Additional qualitative results on LongLive.

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

MemFlow

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

+KVPO

[Seg1] A macro lens studies a crimson peony.

[Seg2] Outer petals slowly unfurl in morning light.

[Seg4] The camera tilts down toward dew, stem, and soil.

[Seg3] Golden stamens emerge as the bloom sways.

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

MemFlow

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

+KVPO

[Seg1] A rugged motorcyclist leans into a mountain hairpin.

[Seg3] Pine needles blur past the vibrating handlebars.

[Seg4] Sunlight flashes across chrome and a weathered jacket.

[Seg2] The rear tire kicks red dust and gravel.

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

MemFlow

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

+KVPO

[Seg3] A hand paints Alif in black ink.

[Seg1] Candlelight warms a dark indigo background.

[Seg4] Flowing Thuluth script forms Allah as gold highlights shimmer.

[Seg2] A weathered wooden tablet gleams with metallic accents.

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

MemFlow

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

+KVPO

[Seg3] They kneel to install a silver cylinder.

[Seg1] Harsh sunlight cuts across the lunar surface.

[Seg2] An astronaut enters with heavy magnetic boots.

[Seg4] Dust settles around the device as another astronaut watches.

##### Figure 15: Additional qualitative results on MemFlow.

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

MemFlow

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

+KVPO

[Seg2] A woman stands in a pale blue silk dress.

[Seg1] Morning mist lifts from lavender rows.

[Seg4] The camera pans across endless violet fields.

[Seg3] She steps forward as embroidery catches light.

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

MemFlow

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

+KVPO

[Seg3] A breeze shifts dappled patterns below.

[Seg2] Emerald leaves fill the forest canopy.

[Seg1] Sunset beams pierce scattered clouds above.

[Seg4] Pine scented stillness glows with soft warmth.

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

MemFlow

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

+KVPO

[Seg4]Villagers exchange gifts under bells, carols, and falling snow.

[Seg1] A child's glove catches a glowing snowflake.

[Seg3]A towering Christmas tree glows in the square.

[Seg2] Children sled down a moonlit village slope.

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

MemFlow

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

+KVPO

[Seg2] A waterfall tumbles over wet granite.

[Seg1] Sunlight pierces a humid rainforest canopy.

[Seg4] A butterfly glides toward orchids beside ferns.

[Seg3] Clear pool ripples reflect towering trees.

##### Figure 16: Additional qualitative results on MemFlow.

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

MemFlow

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

+KVPO

[Seg4] Screen light reveals the user's quiet envy.

[Seg1] A manicured hand grips a sleek smartphone.

[Seg2] A thumb scrolls through a social feed.

[Seg3] It stops on a friend's lavish birthday post.

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

MemFlow

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

+KVPO

[Seg1] Indigo waves roll beneath a fading sunset.

[Seg4] Gold and violet reflections shimmer around the hull.

[Seg3] The captain tightens canvas against evening wind.

[Seg2] A wooden sailboat drifts into view.

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

MemFlow

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

+KVPO

[Seg2] The car dives into a violent turn.

[Seg4] Carnival tents blur far below the track.

[Seg1] A roller coaster hangs high under blue sky.

[Seg3] Riders scream, laugh, and grip safety bars.

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

MemFlow

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

+KVPO

[Seg1] A lunar sunset casts violet shadows.

[Seg3] The car crosses a bridge over flowing molten lava below.

[Seg2] A scratched rover drives through powdery dust.

[Seg4] Its headlight reveals tracks and suspension details.

##### Figure 17: Additional qualitative results on MemFlow.

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

MemFlow

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

+KVPO

[Seg1] Children gather in a warm circle.

[Seg2] A girl in yellow spins playfully. [Seg3] Streamers flutter beneath

[Seg4] A laughing boy enters clapping along.

striped balloons.

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

MemFlow

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

+KVPO

[Seg2] Sunlight strikes mixed towers and stone spires.

[Seg3] The view settles into a busy market.

[Seg4] An indigo vendor arranges woven baskets.

[Seg1] A crane descends through amber city clouds.

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

MemFlow

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

+KVPO

[Seg4] Sunlight, flowers, and birds create a magical calm.

[Seg1] Sam and Lily face each other smiling.

[Seg2] A glowing amulet hovers between them.

[Seg3] Flowers bloom in full color all around them.

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

MemFlow

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

+KVPO

[Seg2] The platform turns through a confectionary city.

[Seg3] Lollipop towers rise under cotton candy clouds.

[Seg4] A gummy bear villain watches from a lamppost.

[Seg1] Animal friends leap onto a candy carousel.

##### Figure 18: Additional qualitative results on MemFlow.

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

MemFlow

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

+KVPO

[Seg3] Silver trout leap and splash nearby.

[Seg1] A fisherman stands waist deep in a river.

[Seg4] A fish is pulled from the water as a butterfly drifts over nearby wildflowers.

[Seg2] He casts beneath warm willow shadows.

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

MemFlow

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

+KVPO

[Seg2] A man reaches across the table.

[Seg3] A woman comforts his hand gently.

[Seg1] Amber light warms a quiet restaurant.

[Seg4] Diners murmur softly in the blurred background.

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

MemFlow

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

+KVPO

[Seg4] The camera pans across tranquil valleys.

[Seg1] A summit view frames distant mountains.

[Seg2] Clouds drift around towering peaks.

[Seg3] Forests and rocks glow at sunset.

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

MemFlow

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

+KVPO

[Seg1] Jagged peaks stretch beneath fiery dusk.

[Seg2] A hawk dives from the upper left.

[Seg3] Sunset beams cut through alpine pines.

[Seg4] The camera pulls back across misty ridges.

##### Figure 19: Additional qualitative results on MemFlow.

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

MemFlow

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

+KVPO

[Seg4] A vendor hands her a warm pastry.

[Seg1] A village fair glows in afternoon sun.

[Seg2] A girl carries a blue starfish balloon.

[Seg3] A Ferris wheel turns behind her.

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

MemFlow

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

+KVPO

[Seg2] Vines sway against its thatched roof.

[Seg1] Pines shade a wooden mountain cottage.

[Seg4] A bluebird lifts off before snowy peaks.

[Seg3] Wildflowers and ferns fill the valley.

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

MemFlow

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

+KVPO

[Seg1] Sunlight enters the crumbling Colosseum arches.

[Seg2] Moss and wildflowers reclaim limestone cracks.

[Seg4] A bird ascends from the shadowed tiers.

[Seg3] Old weapons and skulls lie in dust.

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

MemFlow

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

+KVPO

[Seg1] An upscale dining hall glows at dusk.

[Seg2] A tuxedoed waiter carries a gourmet burger.

[Seg3] He sets it on white marble. [Seg4] Diners sip cocktails beneath

warm pendant lights.

##### Figure 20: Additional qualitative results on MemFlow.

