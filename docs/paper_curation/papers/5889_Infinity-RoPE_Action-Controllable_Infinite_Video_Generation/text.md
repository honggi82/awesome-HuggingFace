# arXiv:2511.20649v3[cs.CV]19Mar2026

## Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout

Hidir Yesiltepe1 Tuna Han Salih Meral1 Adil Kaan Akan2 Kaan Oktay2 Pinar Yanardag1 1Virginia Tech 2fal

Project Page: https://infinity-rope.github.io

[Figure 1]

Figure 1. ∞-RoPE demonstrates three core capabilities: Infinite-length video generation enabled by Block-Relativistic RoPE, fine-grained action-control through KV Flush, and cinematic multi-cut scene composition via RoPE Cut.

#### Abstract

Current autoregressive video diffusion models are constrained by three core bottlenecks: (i) the finite temporal horizon imposed by the base model’s 3D Rotary Positional Embedding (3D-RoPE), (ii) slow prompt responsiveness in maintaining fine-grained action control during long-form rollouts, and (iii) the inability to realize discontinuous cinematic transitions within a single generation stream. We introduce ∞-RoPE, a unified inference-time framework that addresses all three limitations through three interconnected components: Block-Relativistic RoPE, KV Flush, and RoPE Cut. Block-Relativistic RoPE reformulates temporal encoding as a moving local reference frame, where each newly generated latent block is rotated relative to the base model’s maximum frame horizon while earlier blocks are rotated

backward to preserve relative temporal geometry. This relativistic formulation eliminates fixed temporal positions, enabling continuous video generation far beyond the base positional limits. To obtain fine-grained action control without re-encoding, KV Flush renews the KV cache by retaining only two latent frames, the global sink and the last generated latent frame, thereby ensuring immediate prompt responsiveness. Finally, RoPE Cut introduces controlled discontinuities in temporal RoPE coordinates, enabling multicut scene transitions within a single continuous rollout. Together, these components establish ∞-RoPE as a trainingfree foundation for infinite-horizon, controllable, and cinematic video diffusion. Comprehensive experiments show that ∞-RoPE consistently surpasses previous autoregressive models in overall VBench scores.

#### 1. Introduction

In recent years, video diffusion models have advanced rapidly, progressing from early pixel-space denoisers [12, 14, 33] to transformer-based architectures [2, 3, 13, 32]. The introduction of Diffusion Transformers (DiTs) [10, 28] has further shifted the scaling frontier of generative video modeling, enabling high-resolution, multi-shot synthesis and significantly reducing the quality gap between synthetic and real-world video. Despite these advances, existing models are still limited to short video durations.

To overcome these shortcomings, recent research has shifted toward autoregressive video generation frameworks [5–7, 16, 20, 25, 37, 42] that align with the causal nature of time. CausVid [42] first demonstrated that a bidirectional DiT could be distilled into a causal student through Distribution Matching Distillation (DMD) [40, 41], converting dense bidirectional attention into block-causal attention. However, its asymmetric teacher–student supervision introduced a train–test mismatch leading to exposure bias and degraded visual quality in long rollouts. SelfForcing [16] addressed this issue by performing autoregressive self-rollout during training, using its own generated frames and a rolling KV cache to align training with inference and achieve temporally consistent short-form generation. Building upon this paradigm, Self-Forcing++ [6] extended the framework to minute-scale horizons through long rollouts and extended DMD, adapting the temporal 3D RoPE [34] across the full 1024-frame dimension while still supervising only short slices. Rolling Forcing [25] further refined this direction by jointly denoising multiple consecutive frames with progressively increasing noise levels within a rolling window.

Despite these advances, prior autoregressive diffusion methods remain fundamentally constrained by the architectural limits of their positional strategies. Along the temporal dimension, 3D-RoPE restricts autoregressive diffusion transformers to a fixed horizon of 1024 latent frames, beyond which temporal encoding collapses and attention degenerates. At the same time, extending the Self-Forcing training paradigm to such horizons incurs prohibitively slow optimization and high computational cost due to sequential self-rollouts, as reported in [6].

In this work, we revisit the challenge of long-form video generation from a fundamentally different standpoint. Prior autoregressive diffusion frameworks extend temporal scalability through longer self-rollouts [6, 25] or retraining on long-video data [5], yet they are limited by the absolute frame indexing of 3D-RoPE and the heavy memory footprint of uncompressed KV caches. Instead, we investigate whether it is possible to overcome these bottlenecks purely through relativistic adaptation and training-free architectural reparameterization, without relying on any longvideo supervision. We show that autoregressive video

[Figure 2]

Figure 2. Motivation. Thirty-second video generation with Self-Forcing combined with our method (Top) Self-Forcing alone cannot sustain dynamic long-form generation. (Bottom) When augmented with Block-Relativistic RoPE, a Self-Forcing model trained only on five-second videos produces highly dynamic, highquality long-form sequences.

DiTs trained under the Self-Forcing paradigm for only 5second clips already possess the capacity for highly dynamic infinite-horizon generation, as illustrated in Fig. 2.

To address these challenges, we introduce BlockRelativistic RoPE, a training-free relativistic positional encoding that redefines temporal coordinates as a moving reference frame, allowing each newly generated block to be rotated relative to the base model’s maximum frame horizon while earlier blocks are rotated backward to preserve relative geometry. This formulation removes fixed temporal positions and enables continuous generation far beyond the 1024-frame RoPE limit. We further propose KV Flush, which leverages this relativistic geometry to renew the KV cache and resume generation from past temporal indices, providing prompt-responsive action control with constant memory. Finally, RoPE Cut performs controlled discontinuities in temporal coordinates to introduce cinematic multi-cut transitions within a single rollout. Altogether, these components transform short-horizon SelfForcing models into infinite-horizon, scene-aware generators. Our contributions are summarized as follows:

- • We propose ∞-RoPE, a training-free methodology that converts existing short-horizon autoregressive self-rollout diffusion models into action-controllable infinite-horizon generators.
- • We introduce Block-Relativistic RoPE, a relativistic positional encoding that reformulates temporal structure as a moving reference frame, enabling generation far beyond the 1024-frame RoPE limit.
- • We develop two inference-time operators, KV Flush and RoPE Cut, which respectively provide promptresponsive action control and cinematic multi-cut transitions at constant memory.
- • We conduct extensive qualitative and quantitative evaluations on VBench, showing that ∞-RoPE achieves stateof-the-art long-video subject, background consistency, motion smoothness and dynamic degree.

#### 2. Related Work

Bidirectional Video Generation. Recent years have witnessed remarkable progress in video generation, largely driven by the success of denoising diffusion models [12, 24, 33]. These approaches have evolved from early pixelspace diffusion methods to more efficient latent-space formulations [2, 3, 13, 32], supported by increasingly powerful architectures ranging from space–time U-Nets [2, 15] to diffusion transformers (DiT) [9, 10, 15, 21, 27, 28, 36, 39]. These large-scale systems typically operate as bidirectional video diffusion models, meaning they leverage information from both past and future frames during the denoising process. This bidirectional conditioning yields coherent and visually consistent results. However, their reliance on full temporal context prevents deployment in real-time scenarios, where future frames are unavailable at inference time.

Autoregressive Video Generation. SkyReels-V2 [5] integrates Diffusion Forcing [4] with RL and a non-decreasing noise schedule for infinite-length synthesis. CausVid [42] distilled bidirectional DiT into causal generators using asymmetric DMD [40, 41], achieving streaming video synthesis but limited consistency beyond short horizons. SelfForcing [16] mitigated this train–test mismatch by conditioning on self-generated frames during training, improving temporal stability but still degrading beyond the teacher’s horizon. Self-Forcing++ [6] extended this with long-horizon distillation on self-rollouts via a rolling KV cache, enabling minute-scale synthesis. Rolling Forcing [25] jointly denoises consecutive frames with progressively increasing noise and maintains a persistent attentionsink cache for global consistency. LongLive [38] complements this with frame-level autoregression and KV recache tuning for interactive generation. Stable Video Infinity (SVI) [23] bridges the train-test gap via error-recycling fine-tuning, which injects self-generated errors during training. BAgger [29] constructs corrective trajectories by reversing the model’s own rollouts. Reward Forcing [26] enhances motion dynamics through an EMA sink mechanism and Rewarded Distribution Matching Distillation (ReDMD), which biases the generator toward high-reward motion regions. Block Cascading [1] introduces a trainingfree acceleration strategy that parallelizes block-causal inference by allowing future blocks to begin generation conditioned on the partially denoised latents of preceding blocks. NOVA [7] performs non-quantized frame-by-frame and setby-set prediction, removing vector quantization for continuous latent autoregression but limited in long-context scalability. MAGI-1 [35] adopts chunk-wise autoregressive denoising with block-causal attention and parallel chunk generation at high infrastructure cost. In contrast to these methods, which primarily extend temporal horizons via new training procedures, distillation strategies, or large-scale infrastructure, ∞-RoPE explores what already distilled mod-

els can achieve by reparameterizing temporal RoPE and KV caching at inference time, and can be applied in a plug-andplay fashion on top of existing Self-Forcing variants to enable effectively infinite-horizon, controllable video generation. FLEX [22] subsequently introduced frequency-aware RoPE modulation and antiphase noise sampling to mitigate the spectral bias in long video generation.

#### 3. Background

##### 3.1. Base Model

Our work builds upon the pretrained Wan2.1-T2V-1.3B [36] distilled from Wan2.1-T2V-14B via Self-Forcing. The model operates in a latent space encoded by a 3D Variational Auto-Encoder (VAE) [19], which compresses the input video by 4× along the temporal dimension and 8× along each spatial dimension. Formally, each input video V ∈ RF×H×W×3 is encoded into a latent tensor x0 of size 1 + F-14 × C × H8 × W8 , where C denotes the latent channel dimension. The model follows the Rectified Flow formulation [8], where the forward process interpolates between a clean latent x0 and Gaussian noise ϵ ∼ N(0,I) as xt = (1 − t)x0 + tϵ, where t ∈ [0,1], and the reverse process is parameterized by a neural velocity field vθ as an ordinary differential equation (ODE): dxt = vθ(xt,t)dt, where θ denotes model parameters. During inference, Euler discretization is applied over t to iteratively solve the ODE and recover x0 from x1.

##### 3.2. 3D Rotary Position Embedding (3D-RoPE)

Wan [36] employs 3D-RoPE to encode the temporal and spatial coordinates of query and key tokens before selfattention. Let the latent features be

x ∈ RB×S×C, S = F × H × W.

For simplicity, we omit the timestep index t. The channel dimension is divided into three groups for the temporal, height, and width axes, such that

x = xf ∥xh ∥xw ,

C 2

= Cf + Ch + Cw.

For each token at coordinates (f,h,w), 3D-RoPE applies rotary embeddings along the corresponding axes and concatenates the results:

RoPE3D x(f,h,w) = RoPE(xf)∥RoPE(xh)∥RoPE(xw)

= RoPE(xf)∥RoPE2D[x(h,w)]

Temporal Extrapolation Limit. In Wan, each RoPE dimension is configured with a fixed maximum sequence length of 1024, which defines the representable positional range for all three axes. However, the base model is trained only on short temporal horizons, so it never adapts to RoPE

[Figure 3]

- Figure 3. Block-Relativistic RoPE. (a) Fixed cache size. As new latent blocks are generated, their temporal RoPE coordinates are rotated relative to the teacher’s maximum horizon flimit, while earlier latents are rotated backward to preserve their relative temporal geometry within the fixed cache window. (b) Unbounded cache size. When the KV cache grows beyond flimit, earlier latents undergo semanticization: Temporally distant tokens collapse into abstract semantic memory, while recent high-SNR tokens retain precise temporal geometry. See Sec. 4.1 for details.

indices beyond those seen during training. As a result, even when generation exceeds the model’s short training horizon, rather than the full 1024-frame limit, the model enters unseen positional regimes and attention quality degrades even though the RoPE formulation itself remains valid.

#### 4. Methodology

##### 4.1. Block-Relativistic RoPE

Fixed Cache Size. We first consider the case where the KV cache horizon is smaller than or equal to the base model’s native generation horizon flimit. In our experiments, the base model is trained on 5-second clips corresponding to flimit = 21 latent frames. During autoregressive self-rollout, video generation proceeds in blocks of three latent frames, i.e.,

Bf = {f − 2, f − 1, f }. (1)

Once the cache becomes full, the oldest three latent frames are evicted and replaced by the newly generated block. In

- our design, although only fixed size latent frames are resident in KV cache at any time, the temporal RoPE indices assigned to the current block are not bounded by this cache size but instead by the base model’s full generation capacity

of flimit latent frames as demonstrated in Fig. 3(a). Consequently, even as tokens are rolled within the cache, their temporal embeddings can continue to advance beyond the cache length, fully harnessing the global temporal continuity within the teacher’s horizon. In conventional absolute RoPE3D, temporal indices are defined globally. For temporal index i ≫ flimit and current generation block

Bi = {i − 2, i − 1, i}

i,h,w) = RoPE(xB

RoPE3D x(B

)∥RoPE2D[x(h,w)] .

i

In contrast, Block-Relativistic RoPE defines temporal coordinates within a moving local reference frame while keeping spatial coordinates fixed:

Bi = {i − 2, i − 1, i}, if i ≤ f0, Bf

B˜i =

(2)

= {f0 − 2, f0 − 1, f0 }, otherwise,

0

i,h,w) f0 = RoPE(xB˜

)∥RoPE2D[x(h,w)] ,

RoPE3D x(B

i

where f0 ≤ flimit denotes the onset index of the current cache window. As generation advances, previously cached latent frames are re-anchored relative to the f0, effectively rotating their temporal phase backward within the RoPE space.

Unbounded Cache Size. When the KV cache extends beyond the teacher model’s temporal horizon flimit, the temporal encoding must transition from a strictly temporal regime to a semantically abstract one. Drawing inspiration from the semanticization process [31] in cognitive neuroscience, we reinterpret this transition as a form of temporal decontextualization. In the human brain, semanticization describes how episodic memories, which are initially tied to specific times and places, gradually lose their spatiotemporal precision while preserving their conceptual and self-relevant meaning. Through consolidation, these details become integrated into the brain’s semantic networks, forming personal semantics: knowledge of one’s past detached from its original temporal context.

[Figure 4]

- Figure 4. KV Flush. KV Flush resets the KV cache to only two tokens, the global sink and the last latent frame, so that a new prompt takes effect immediately without carrying over old semantics. Compared to no-cache (abrupt changes), full-cache (semantic lag), and KV re-cache (high latency), KV Flush achieves instant, clean action responsiveness with smooth temporal continuity, as shown in the prompt sequence: standing → jumping → sitting → singing.

Analogously, when the KV cache surpasses flimit, earlier latent frames undergo semanticization within the BlockRelativistic RoPE formulation as shown in Fig. 3(b). Rather than preserving unique temporal indices for the earliest frames, we collapse their temporal coordinates to a shared minimum index, effectively anchoring them as temporally invariant yet semantically influential memories. Concretely, when the current generation block is Bi with i > flimit, the earliest cached block is remapped as

###### B3 = {1,2,3} −→ B¯1 = {1,1,1}. (3)

Hence, preserving what occurred in earlier frames (their visual–semantic content) rather than when it exactly occurred, mirroring the abstraction of episodic detail into decontextualized knowledge. Consequently, the system’s temporal representation evolves from an episodic phase, where frame order and timing dominate, to a semantic phase, where distant history persists only as contextual priors shaping ongoing generation. Within Block-Relativistic RoPE, this mechanism allows the temporal axis to expand indefinitely, maintaining precise relative encoding within the current active window Bi while assimilating all preceding frames into a unified, temporally agnostic semantic field.

##### 4.2. Action Control via KV Flush

In autoregressive video diffusion, transitioning between prompts during inference, known as action-controllable video generation, requires balancing immediate semantic

responsiveness with temporal continuity. KV Flush mechanism performs cache renewal with constant memory and zero latency as described in Fig. 4. When a new prompt arrives, all cached tokens are flushed except two anchors, the global sink latent frame and the last generated latent frame, which respectively stabilize attention normalization and preserve local temporal continuity. The next action is then conditioned directly on these minimal anchors, allowing the scene to evolve smoothly in motion while instantly adopting the new semantics outperforming prior cache management paradigms in both efficiency and controllability.

##### 4.3. Multi-Cut Scenes via RoPE Cut

While Block-Relativistic RoPE and KV Flush enable continuous and controllable generation, cinematic narratives often require abrupt temporal discontinuities such as scene cuts, flashbacks, or cross-location transitions that cannot be represented by smooth temporal evolution alone. To achieve such discontinuities within a single continuous rollout, we introduce RoPE Cut, a training-free operation that performs controlled discontinuous jumps in temporal RoPE coordinates.

For the current generation block Bf = {f−2, f−1, f}, RoPE Cut redefines the temporal coordinate mapping as

Bf→f+∆ = {f−2, f+∆−1, f+∆},

where ∆ denotes an offset representing the temporal gap between the current and the next scene. Instead of extend-

[Figure 5]

- Figure 5. RoPE Cut. RoPE Cut enables a discontinuous jump along the temporal RoPE axis. In the first rollout (second row), the active latent block B6 = {4, 5, 6} is reassigned to a new RoPE-local frame, becoming B4→21 = {4, 20, 21}: the token “4” is kept as the local anchor while the next two tokens are reassigned to the high-SNR positions 20 and 21 for continued denoising. In the subsequent rollout (third row), the block {4, 20, 21} is treated as past context after the cut, and generation proceeds again from the original temporal location with a fresh block B6 = {4, 5, 6} inside the fixed cache horizon.

[Figure 6]

(a) Block-Relativistic RoPE (b) KV Flush (c) RoPE Cut

[Figure 7]

[Figure 8]

- Figure 6. Attention behavior of ∞-RoPE across interventions. Frame-to-frame attention maps from the 13th DiT layer, shown as headaveraged, pixel-summed attention with query frame index q on the y-axis and key frame index k on the x-axis, for (a) a baseline ∞-RoPE rollout, (b) KV Flush at an action change, and (c) RoPE Cut. These visualizations summarize how our method structures long-horizon temporal dependencies; see Sec. 4.4 for a detailed mechanistic interpretation.

ing temporal indices sequentially, we cut and re-anchor the temporal RoPE phase by offsetting the current block by ∆ frames:

RoPE3D x(B

f→f+∆,h,w) . (4) As illustrated in Fig. 5 (second row), when a discontinu-

f,h,w) −→ RoPE3D x(B

- ous jump occurs, the subsequent frames (third row) are generated as if the jumped segment were repositioned into the past temporal indices. This re-indexing allows the system to fully reuse the base model’s generation horizon flimit with-
- out violating its temporal range. Thanks to our relativistic formulation, there is no fixed or absolute position along the temporal axis—the coordinate system itself shifts with each cut, allowing identity preservation even after large temporal or semantic jumps.

##### 4.4. Mechanistic Interpretation of ∞-RoPE

We analyze frame-to-frame self-attention maps to interpret how ∞-RoPE structures long-horizon temporal dependencies. During long video generation via Block-Relativistic RoPE (Fig. 6a), attention forms a sharp diagonal band, showing that each frame attends primarily to its recent predecessors, and a persistent early-time sink column that aggregates global scene information. This pattern remains

stable far beyond the original 3D-RoPE horizon, indicating that the relativistic re-anchoring of RoPE preserves relative temporal geometry and prevents attention collapse. Interventions reshape this structure in a controllable way. With KV Flush (Fig. 6b), attention from new frames to intermediate past frames is largely suppressed, and mass is redirected toward the sink and the last few pre-flush frames, enabling instant prompt responsiveness while maintaining local motion continuity. With RoPE Cut (Fig. 6c), the map splits into two nearly disjoint diagonal blocks, with the new segment attending only to itself and the sink, effectively severing temporal context and realizing a clean scene cut. Overall, these patterns provide a mechanistic explanation of how ∞-RoPE supports stable long-horizon rollouts, prompt-responsive control, and cinematic transitions within a single autoregressive generation.

#### 5. Experiments

Implementation Details. We implement ∞-RoPE on SelfForcing [16], a causal four-step generator distilled from Wan2.1-T2V-1.3B [36] that natively produces five-second videos at 16 FPS with a resolution of 832×480. All quantitative experiments use a KV cache size of 6, an onset index

- Table 1. Performance comparisons on 5s and 60s videos. For 5s generations, several baselines report high temporal-quality metrics largely due to stagnation effects reflected in their low dynamic degree, whereas ∞-RoPE maintains strong temporal stability without sacrificing motion richness. On 60s videos, ∞-RoPE demonstrates superior long-horizon coherence, subject and background consistency, and overall visual quality.

Model Throughput↑ Results on 5s ↑ Results on 60s ↑

(FPS)

Aesthetic Quality

Background Consistency

Dynamic Degree

Imaging Quality

Motion Smoothness

Subject Consistency

Temporal Flickering Overall↑

Aesthetic Quality

Background Consistency

Dynamic Degree

Imaging Quality

Motion Smoothness

Subject Consistency

Temporal Flickering Overall↑

Bidirectional models

LTX-Video [11] 8.98 0.6209 0.9530 0.64 0.7289 0.9941 0.9501 0.9812 0.8439 - - - - - - - Wan2.1 [36] 0.78 0.6675 0.9696 0.44 0.7195 0.9851 0.9599 0.9745 0.8319 - - - - - - - -

Autoregressive models NOVA [7] 0.88 0.5819 0.9516 0.28 0.6338 0.9908 0.9338 0.9833 0.7915 0.4753 0.8806 0.12 0.4497 0.9894 0.7750 0.9827 0.6901 Pyramid Flow [18] 6.70 0.6377 0.9609 0.44 0.6650 0.9935 0.9608 0.9861 0.8266 - - - - - - - MAGI-1 [35] 0.19 0.6360 0.9683 0.44 0.6022 0.9945 0.9583 0.9900 0.8199 0.5210 0.8776 0.56 0.5454 0.9926 0.7946 0.9848 0.7511 SkyReels-V2 [5] 0.49 0.6621 0.9683 0.48 0.7016 0.9884 0.9607 0.9757 0.8335 0.5764 0.8995 0.44 0.6667 0.9867 0.8499 0.9760 0.7768 CausVid [42] 17.01 0.6443 0.9512 0.48 0.7012 0.9832 0.9596 0.9730 0.8231 0.6288 0.8985 0.52 0.6747 0.9847 0.8675 0.9755 0.7940 Self-Forcing [16] 17.01 0.6576 0.9598 0.60 0.7323 0.9841 0.9629 0.9677 0.8398 0.5895 0.8784 0.32 0.6971 0.9890 0.8360 0.9830 0.7715 Rolling-Forcing [25] 17.01 0.6655 0.9652 0.40 0.7273 0.9871 0.9721 0.9771 0.8332 0.6350 0.9447 0.36 0.7242 0.9865 0.9409 0.9769 0.8146 ∞-RoPE (Ours) 17.01 0.6308 0.9720 0.48 0.7141 0.9872 0.9787 0.9845 0.8377 0.6325 0.9490 0.52 0.7010 0.9901 0.9444 0.9852 0.8298

- Table 2. Performance comparisons on 120s and 240s videos. Across both horizons, ∞-RoPE consistently ranks first or second in every metric and achieves the strongest overall score, demonstrating stable long-horizon quality, high subject and background consistency, and robust motion dynamics over extended rollouts.

Model Results on 120s ↑ Results on 240s ↑

Background Consistency

Imaging Quality

Subject Consistency

Temporal Flickering Overall↑

Background Consistency

Imaging Quality

Subject Consistency

Temporal Flickering Overall↑

Dynamic Degree

Dynamic Degree

Aesthetic Quality

Motion Smoothness

Aesthetic Quality

Motion Smoothness

Autoregressive models

NOVA [7] 0.4563 0.8629 0.16 0.4294 0.9908 0.7394 0.9845 0.6785 0.4345 0.8533 0.24 0.3854 0.9897 0.7116 0.9827 0.6662 SkyReels-V2 [5] 0.5136 0.8571 0.40 0.6154 0.9856 0.7700 0.9765 0.7326 0.4787 0.8421 0.40 0.6005 0.9841 0.7384 0.9750 0.7147 CausVid [42] 0.6296 0.8902 0.40 0.6754 0.9847 0.8485 0.9757 0.7799 0.6300 0.8837 0.36 0.6876 0.9845 0.8399 0.9759 0.7758 Self-Forcing [16] 0.5135 0.8321 0.32 0.6496 0.9877 0.7528 0.9829 0.7271 0.4286 0.7973 0.24 0.6141 0.9881 0.6893 0.9853 0.6850 Rolling-Forcing [25] 0.6319 0.9374 0.44 0.7242 0.9862 0.9272 0.9769 0.8162 0.6156 0.9248 0.40 0.7128 0.9855 0.9080 0.9753 0.8017 ∞-RoPE (Ours) 0.6199 0.9392 0.56 0.6904 0.9893 0.9302 0.9835 0.8236 0.6235 0.9361 0.64 0.7028 0.9898 0.9256 0.9831 0.8309

[Figure 9]

- Figure 7. Cache Size vs. Performance across VBench dimensions. Metrics are shown as a function of KV Cache size for (a) Overall, (b) Aesthetic Quality, (c) Dynamic Degree, and (d) Imaging Quality, evaluated on 30s, 60s and 120s sequences.

of f0 = 21, a classifier-free guidance scale of 3.0, and a timestep shift of 5.0.

Baselines. We compare our method against state-of-theart open-source bidirectional and autoregressive video generation models. Specifically, we include 2 bidirectional diffusion models, Wan2.1-1.3B [36] and LTX-Video [11], and 7 autoregressive models, including Pyramid Flow [18], NOVA [7], SkyReels-V2 [5], MAGI-1 [35], CausVid [42], Self-Forcing [16], and Rolling-Forcing [25].

Evaluation. Following [6, 16, 25], we evaluate ∞RoPE using VBench [17], which measures subject consistency, background consistency, motion smoothness, temporal flickering, dynamic degree, aesthetic quality, and imaging quality. We randomly sample prompts from MovieGenBench [30] and generate over 100 videos across four durations: 5, 60, 120, and 240 seconds.

##### 5.1. Qualitative Experiments

Qualitative Results on Long Video Generation. We provide qualitative comparisons with previous state-of-the-art approaches in Fig. 8. Visually, ∞-RoPE produces frames that are notably sharper and more temporally coherent than baseline methods. A key advantage of our model is its ability to maintain superior identity consistency throughout the entire video duration, successfully mitigating the temporal drift and identity loss often observed in extended autoregressive rollouts. Furthermore, our method excels at generating highly dynamic scenes, a key advantage compared to other methods. This is quantitatively supported by our state-of-the-art Dynamic Degree scores, particularly in the long-horizon 60s 120s and 240s benchmarks (Tab. 2). This sustained quality can be observed in both continuous longform generation and in complex multi-prompt scenarios.

Qualitative Results on Fine-Grained Action Control. Figure 1 and Figure 4 show that ∞-RoPE enables precise action control in single-subject and multi-subject streaming. Given a sequence of evolving action prompts such as standing → jumping → sitting → singing, the model produces continuous rollouts in which the subjects respond immediately to each new instruction while maintaining appearance, pose style, and scene layout across long horizons, for both single subject and multi-subject scenes. KV Flush plays a central role by resetting the cache to only the global sink token and the most recent latent frame, which removes stale context from earlier segments and allows the new action to take effect instantly.

Qualitative Results on Cinematic Transitions. For cinematic scene transitions, we apply RoPE Cut to introduce controlled discontinuities in the temporal RoPE coordinates

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

PyramidFlowSkyReels-V2OursRolling Forcing

CausVidSelf-ForcingMAGI-1NOVA

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

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

- Figure 8. Qualitative Comparison on 60-Second Generation. We present 60-second generations. While prior models exhibit identity drift, temporal inconsistencies, or notable degradation in visual fidelity, our approach produces sequences that remain highly subject consistent and temporally coherent, with sustained image quality across the entire one-minute horizon.

while keeping the autoregressive rollout continuous shown in Fig. 1 and Fig. 9. Qualitatively, this enables multi-cut compositions such as indoor-to-outdoor changes, time-ofday shifts, or cross-location jumps within a single generated video: the background, lighting, and context change abruptly at the cut, yet the main character’s identity, clothing, and coarse pose remain coherent before and after the transition, even when new actors enter the scene.

##### 5.2. Quantitative Experiments

Quantitative Results on Long Video Generation. We evaluate ∞-RoPE on both short-horizon (5 seconds) and long-horizon (60, 120, and 240 seconds) video generation using the MovieGenBench [30] prompts. As shown in Tab. 1-2, our method achieves state-of-the-art results compared to baselines, particularly in subject consistency and motion smoothness. Since ∞-RoPE is not limited by the RoPE dimension [6], it maintains stable identity representation across arbitrarily long sequences. This architectural advantage explains its sustained quality in extended generations such as the 120- and 240-second benchmarks.

Quantitative Results on Action Controlled Video Generation. Beyond autoregressive baselines, we also conduct a user study to compare ∞-RoPE with LongLive [38], SkyReels-V2 [5], and Self-Forcing [16] for actioncontrolled video generation in Table 3. LongLive introduces KV-Recache, a cache management mechanism designed to enable prompt-dependent action transitions in autoregressive models. At each transition point, KV-Recache extracts all cached tokens, applies cross-attention with the new prompt, rebuilds a modified cache, and conditions subsequent frames on these recached latent tokens as demonstrated in Fig. 4. However, the approach presents two main limitations: (1) During long rollouts, embeddings from

Subject Consist. ↑

Video Quality ↑ Autoregressive models

Text Align. ↑

Motion Smoothness ↑

Method

Self Forcing 1.88 2.00 1.81 1.64 SkyReels-V2 2.21 2.12 2.14 1.81 LongLive 3.19 3.29 3.10 2.98 Ours 3.86 3.95 3.74 3.38

Table 3. User Study on Action Controlled Video Generation. ∞-RoPE consistently obtain higher Text Alignment, Subject Consistency, Motion Smoothness and Video Quality scores.

earlier prompts accumulate in the cache and are not fully erased. This reduces prompt responsiveness over time and increases the delay before the new action appears. The qualitative comparison can be found in the project page. (2) KVRecache requires reconstructing the entire cache at every prompt change. The resulting overhead increases with the cache size and the number of action transitions. By contrast, ∞-RoPE achieves instant prompt responsiveness by simply flushing the stale cache content, which is implemented as a local update to the cache end index without proportional computation. As shown in Table 3, ∞-RoPE yields the best Text Alignment, Subject Consistency, Motion Smoothness and Video Quality scores.

Ablation on KV Cache Size. We analyze the impact of KV cache size on Dynamic Degree, Imaging Quality, Subject Consistency, and Background Consistency in Fig. 7. In all experiments, we fix f0 = 21 and vary only the cache size. The results show that while Imaging Quality and Dynamic Degree gradually decrease as the cache grows, both Subject Consistency and Background Consistency remain stable, indicating that ∞-RoPE effectively preserves longrange identity and scene structure.

Ablation on Temporal Jump Index ∆. We evaluate ∆ ∈ {6,21,45,90}. Notably, ∆ = 6 and ∆ = 21 lie

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

- Figure 9. Dynamic Scene Cut. RoPE Cut enables controlled cinematic transitions by introducing discontinuities in temporal RoPE coordinates. This allows a single autoregressive rollout to produce diverse environments and background changes while preserving subject identity and temporal coherence.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Δ=21Δ=45Δ=90Δ=6

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Figure 10. Temporal Jump Index ∆ Ablation.

within the training horizon of the pretrained model, whereas ∆ = 45 and ∆ = 90 fall outside that range. For in-horizon values, transitions remain smooth with minimal artifacts. For out-of-horizon values, the model produces more dramatic scene transitions at the cost of a visible transition edge. This edge effect arises because the block Bf→f+∆ is rotated by a RoPE angle that does not appear in the training data and must therefore be extrapolated. The resulting artifact reflects the inherent extrapolation behavior of the base model’s RoPE formulation. In Table 4, we present the quantitative results for temporal jump indices ∆ ∈ {6,21,45,90} and qualitative results in Fig. 10 and Fig. 9. The results show that increasing the temporal jump index leads to lower background consistency due to the more abrupt scene transitions. However, this reduction occurs while subject consistency and temporal smoothness remain high, indicating that the model preserves identity and motion stability even under large scene changes.

Long Video Generation User Study. We conducted a user study with 50 participants recruited from Prolific, who rated each video independently on a 5-point Likert scale for two aspects: (i) overall quality given the prompt, and (ii) temporal consistency from start to end. We report the mean scores across all prompts and videos for each model. ∞-RoPE achieves the highest average overall quality (3.91) and temporal consistency (3.71), outperforming the strongest baseline, Rolling-Forcing (3.55 / 3.42), as well as CausVid,

Subject Consistency ↑

Background Consistency ↑

Temporal Smoothness ↑

Temporal Jump Index ∆

∆ = 6 90.74 88.98 0.98 ∆ = 21 90.04 87.57 0.98 ∆ = 45 88.47 84.48 0.96 ∆ = 90 87.69 82.27 0.95

- Table 4. Ablation Study on Temporal Jump Index ∆. We evaluate Subject Consistency, Background Consistency, and Temporal Smoothness for different values of ∆ on 40-second video generation. A total of 20 videos are generated, and each video undergoes a scene cut every 10 seconds. Smaller jump values (∆ = 6 and

21) fall within the training horizon and yield smoother transitions, while larger jump values (∆ = 45 and 90) produce more pronounced scene changes accompanied by stronger transition-edge artifacts.

Model

Overall Quality

Temporal

Consistency Avg.↑ Autoregressive models

CausVid [42] 3.113 3.131 3.122 NOVA [7] 1.333 1.286 1.310 SkyReels-V2 [5] 2.579 2.175 2.377 Self-Forcing [16] 2.458 2.087 2.273 Rolling-Forcing [25] 3.554 3.423 3.488 ∞-RoPE (Ours) 3.911 3.708 3.810

- Table 5. User Study. Evaluating 60 second video generations on a 5 point Likert scale, with higher scores reflecting better performance.

SkyReels-V2, Self-Forcing, and NOVA. These results align with our automatic metrics and indicate that users perceive ∞-RoPE ’s long videos as both higher quality and more temporally stable.

- 6. Limitations and Conclusion

As a training-free method, ∞-RoPE inherits limitations of its base model, such as imperfect physics. Nonetheless, our results show that a simple training-free mechanism can significantly extend autoregressive video diffusion models without additional data, model updates. This work takes a practical step toward user-controllable long video generation and future scalable, temporally robust video models.

#### References

- [1] Hmrishav Bandyopadhyay, Nikhil Pinnaparaju, Rahim Entezari, Jim Scott, Yi-Zhe Song, and Varun Jampani. Block cascading: Training free acceleration of block-causal video models. arXiv preprint arXiv:2511.20426, 2025. 3
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023. 2, 3
- [4] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 3
- [5] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025. 2, 3, 7, 8, 9, 12, 13
- [6] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 2, 3, 7, 8
- [7] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169,

2024. 2, 3, 7, 9, 12

- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. 2024. 3
- [9] Google DeepMind. Veo. https://deepmind. google/models/veo/, 2024. 3
- [10] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. In European Conference on Computer Vision, pages 393–411. Springer, 2024. 2, 3
- [11] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103,

2024. 7

- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3
- [13] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben

- Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2, 3
- [14] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 2
- [15] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3
- [16] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3, 6, 7, 8, 9, 12, 13
- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 7
- [18] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 7

- [19] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [20] Akio Kodaira, Tingbo Hou, Ji Hou, Masayoshi Tomizuka, and Yue Zhao. Streamdit: Real-time streaming text-to-video generation. arXiv preprint arXiv:2507.03745, 2025. 2
- [21] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3
- [22] Jia Li, Xiaomeng Fu, Xurui Peng, Weifeng Chen, Youwei Zheng, Tianyu Zhao, Jiexi Wang, Fangmin Chen, Xing Wang, and Hayden Kwok-Hay So. Train short, inference long: Training-free horizon extension for autoregressive video generation. arXiv preprint arXiv:2602.14027, 2026. 3
- [23] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025. 3
- [24] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [25] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025. 2, 3, 7, 9, 12
- [26] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Ef-

- ficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025. 3
- [27] OpenAI. Sora. https://openai.com/sora, 2024. 3
- [28] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2, 3

- [29] Ryan Po, Eric Ryan Chan, Changan Chen, and Gordon Wetzstein. Bagger: Backwards aggregation for mitigating drift in autoregressive video diffusion models. arXiv preprint arXiv:2512.12080, 2025. 3
- [30] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 7, 8

- [31] Louis Renoult, Patrick SR Davidson, Daniela J Palombo, Morris Moscovitch, and Brian Levine. Personal semantics: at the crossroads of semantic and episodic memory. Trends in cognitive sciences, 16(11):550–558, 2012. 4
- [32] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2, 3

- [33] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 3
- [34] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 2

- [35] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 3, 7, 12
- [36] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3, 6, 7
- [37] Dirk Weissenborn, Oscar T¨ackstr¨om, and Jakob Uszkoreit. Scaling autoregressive video models. arXiv preprint arXiv:1906.02634, 2019. 2
- [38] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622,

2025. 3, 8, 12, 13

- [39] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3

- [40] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024. 2, 3
- [41] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 2, 3
- [42] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025. 2, 3, 7, 9, 12

## Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout

### Supplementary Material

[Figure 83]

Figure 11. User Study Interface. User Study Interface for Long Video Generation

#### A. Videos and Website

To support thorough evaluation and improve the accessibility of our findings, we provide more than one hundred video results totaling over two hours of content. These include motivation examples, detailed qualitative demonstrations, ablation studies, and side by side comparisons. All videos are available on our Project Page: https://infinityrope.github.io.

#### B. User Studies

In Figure 11 and Figure 12, we present the interfaces used in our two user studies. For the long-form video generation user study, participants were shown videos generated from provided prompts and were asked two questions: ”Overall, how good is this video given the prompt?” and ”How consistent does the video stay from start to end?” These questions were designed to evaluate prompt adherence and temporal consistency. For the action-controlled video generation user study, we compare our method against LongLive [38], SkyReels-V2 [5], and Self-Forcing [16]. In this study, participants were asked four questions: ”Did the video perform the specific sequence of actions described in the text?” to measure prompt responsiveness, ”Did the main character or object look like the same person or thing from start to finish?” to assess subject consistency, ”Did the video

[Figure 84]

Figure 12. User Study Interface. User Study Interface for Action Controlled Long Video Generation

flow smoothly when the instructions changed?” to examine motion smoothness at action transition points, and ”How realistic and artifact-free are the individual frames?” to evaluate overall video quality.

#### C. More Discussion on Qualitative Results

##### C.1. Discussion on Long Video Generation Results

In the main paper, we compare ∞-RoPE against NOVA [7], MAGI-1 [35], SkyReels-V2 [5], CausVid [42], SelfForcing [16], and Rolling-Forcing [25] across both short (5s) and long (60s, 120s, 240s) generation settings. Our quantitative evaluations consistently show that, in the long-duration regime, ∞-RoPE outperforms prior autoregressive approaches in terms of Subject Consistency, Background Consistency, and Dynamic Degree, while ranking first or second in Motion Smoothness and Temporal Flickering.

To validate that these quantitative trends align with perceptual quality, we provide project page with qualitative comparisons at all four durations. The qualitative results corroborate the numerical findings. As rollout length increases, Rolling-Forcing tends to repeatedly re-

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Figure 13. Ultra-Long Video Generation Enabled by Block-Relativistic RoPE. Block-Relativistic RoPE reformulates temporal encoding as a moving frame of reference, allowing the model to preserve relative temporal geometry far beyond the base model’s positional horizon. This enables continuous, stable, and fully coherent video generation over extremely long durations without retraining or increased cache size.

generate/spawn similar characters with minimal scene evolution, a limitation stemming from its training paradigm. SkyReels-V2 exhibits large, unstable camera motions that reduce subject consistency in long sequences. Pyramidal Flow frequently resets scene content every 5 seconds, resulting in low subject and background continuity. Meanwhile, both CausVid and Self-Forcing gradually accumulate exposure bias in extended rollouts.

In contrast, ∞-RoPE maintains highly dynamic scenes with stable subject and background appearance across all tested durations, despite relying solely on the pretrained Self-Forcing model, which natively supports only 5-second generation at 16FPS. These results highlight the robustness and scalability of our method for long-form autoregressive video generation.

##### C.2. Discussion on Action Control Results

Beyond autoregressive baselines, we also compare ∞RoPE with LongLive [38], SkyReels-V2 [5], and SelfForcing [16] for action-controlled video generation. LongLive introduces KV-Recache, a cache management mechanism designed to enable prompt-dependent action transitions in autoregressive models. At each transition point, KV-Recache extracts all cached tokens, applies crossattention with the new prompt, rebuilds a modified cache,

and conditions subsequent frames on these recached latent tokens. This procedure aims to overwrite residual semantics from the previous prompt and improve responsiveness to new user instructions. However, the approach presents two main limitations:

- 1. Incomplete removal of previous prompt content. During long rollouts, embeddings from earlier prompts accumulate in the cache and are not fully erased. This reduces prompt responsiveness over time and increases the delay before the new action appears. As demonstrated in the Action Control Comparison section of the project page, LongLive shows reduced responsiveness and significant identity and background drift, while ∞-RoPE preserves subject identity and background stability and responds to new prompts immediately.
- 2. Additional latency proportional to cache size and number of transitions. KV-Recache requires reconstructing the entire cache at every prompt change. The resulting overhead increases with the cache size and the number of action transitions. By contrast, ∞-RoPE achieves instant prompt responsiveness by simply flushing the stale cache content, which is implemented as a local update to the cache end index without proportional computation.

##### C.3. Discussion on Dynamic Scene Cut Results

Autoregressive video diffusion models naturally produce temporally smooth sequences, which often results in limited scene dynamism. To introduce cinematic variation without compromising coherence, we propose RoPE Cut, a mechanism that applies controlled discontinuities to the temporal RoPE coordinates. This technique enables intentional scene shifts while preserving overall generative stability.

We present our Dynamic Scene Cut results in the project page and in Fig. 9. Using RoPE Cut, we generate trailerstyle sequences for several films, including Harry Potter, Titanic, Game of Thrones, The Shawshank Redemption, Barbie, and Interstellar. As demonstrated by these results, RoPE Cut produces dynamic, diverse scenes with varying backgrounds and environments within a single continuous generation stream, while consistently maintaining subject identity and visual fidelity.

#### D. Interpretability via Attention Maps

Construction of frame-level attention maps. For each video, we extract self-attention weights from the middle transformer block of the denoiser at a fixed denoising step and aggregate them at the frame level. Let the video consist of T frames, and let each frame t be represented by a set of latent tokens. Denote by a(t,i)→(s,j) the self-attention weight from query token (t,i) (frame t, spatial index i) to key token (s,j) (frame s, spatial index j), averaged over attention heads. We then construct a T ×T frame–frame attention matrix M by summing over all token pairs between frames:

###### Mt,s =

a(t,i)→(s,j).

i∈frame t j∈frame s

Each cell (t,s) in the attention map therefore corresponds to the total attention mass from all tokens of frame t (query frame) to all tokens of frame s (key frame). Since the underlying self-attention weights are row-normalized by the softmax, each row of M is naturally normalized as well and can be interpreted as a frame-level attention distribution over the video history. A sharp diagonal structure means that each frame mainly attends to itself and its immediate temporal neighbors, while vertical stripes or off-diagonal blocks indicate longer-range dependencies or special tokens (e.g., sink tokens).

Block-Relativistic RoPE for Infinite-length Video Generation. For standard infinite-length generation, the Block-Relativistic RoPE map (Fig. 6a) exhibits two main structures: (i) a sharp diagonal band around the main diagonal, and (ii) a persistent bright column corresponding to the global attention sink token. The diagonal band shows that each query frame primarily attends to a small window of its

recent predecessors and itself, which captures local temporal continuity and smooth motion. The bright sink column indicates that the model also consistently attends to a global sink token that provides a stable global context over time.

Crucially, all tokens that lie within the active KV window share a consistent local RoPE coordinate frame: their relative temporal indices stay within the range seen during pretraining (the teacher horizon), even though the absolute video length keeps growing. The attention maps do not show any drift of attention towards extremely early frames or degenerate patterns as the sequence becomes long. This supports our claim that Block-Relativistic RoPE re-anchors temporal indices within the teacher horizon instead of letting them drift to unseen absolute positions, effectively bypassing the 1024-index limit and enabling continuous, stable infinite-horizon rollouts.

KV Flush for Action-controllable Long Video Generation. For action-controllable generation, the KV Flush map (Fig. 6b) visualizes the effect of our selective cache renewal strategy when the prompt is changed mid-generation. At the moment of a prompt change, we flush the KV cache and retain only two anchors: the global attention sink token and the last few generated frames (e.g., the last one or two frames before the change). All earlier frames are removed from the cache.

In the attention map, this appears as: (i) a strong vertical column corresponding to the sink token, and (ii) a narrow band of attention centered around the last pre-flush frame(s), while attention to older frames is strongly suppressed and appears almost dark. After the flush, new frames attend primarily to the sink and these recent anchors, rather than to the distant history. This pattern confirms that the model re-orients its temporal context around a very short window of frames while keeping a stable global reference via the sink token.

Mechanistically, this behavior matches the intended design: the model preserves immediate temporal continuity (short-term motion and appearance consistency) through the last cached frames, but rapidly adapts to the new semantic guidance specified by the updated text prompt. The attention maps therefore make explicit how KV Flush balances short-term temporal coherence with fast semantic resteering.

RoPE Cut for Dynamic Scene Transitions. For dynamic scene transitions, the RoPE Cut map (Fig. 6c) shows what happens when we perform a scene cut. At the cut point, we apply two operations simultaneously: (i) we flush the KV cache, and (ii) we discontinuously advance the RoPE indices for the new frames, assigning them to a new temporal region that does not overlap with the pre-cut segment.

In the frame-level attention maps, this dual action produces two almost disjoint diagonal blocks. The first block corresponds to the pre-cut scene: frames in this block attend strongly to each other but receive very little attention from the post-cut frames. The second block corresponds to the post-cut scene: new frames attend mainly to themselves, their nearby neighbors, and the sink token, while giving only weak attention to frames from the first block. The weak residual attention to the pre-cut frames appears as a faint background, indicating that the old scene is still technically part of the extended context but is functionally de-emphasized.

This pattern shows that RoPE Cut forces the model to reset its effective temporal context at the cut: the post-cut segment behaves like a new scene with its own local temporal structure, rather than a continuation of the old one. At the same time, the sink-mediated pathway still provides a stable identity signal, which helps preserve subject identity across the scene boundary. In other words, the attention maps directly visualize how RoPE Cut implements a hard scene transition in the temporal representation, while still maintaining the global subject and style consistency learned by the model.

