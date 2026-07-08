# arXiv:2605.15190v1[cs.CV]14May2026

[Figure 1]

## RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO

Yanzuo Lu Ronglai Zuo Jiankang Deng Imperial College London https://yanzuo.lu/raven

### Abstract

Causal autoregressive video diffusion models support real-time streaming generation by extrapolating future chunks from previously generated content. Distilling such generators from high-fidelity bidirectional teachers yields competitive fewstep models, yet a persistent gap between the history distributions encountered during training and those arising at inference constrains generation quality over long horizons. We introduce the Real-time Autoregressive Video Extrapolation Network (RAVEN), a training-time test framework that repacks each self rollout into an interleaved sequence of clean historical endpoints and noisy denoising states. This formulation aligns training attention with inference-time extrapolation and allows downstream chunk losses to supervise the history representations on which future predictions depend. We further propose Consistency-model Group Relative Policy Optimization (CM-GRPO), which reformulates a consistency sampling step as a conditional Gaussian transition and applies online Reinforcement Learning (RL) directly to this kernel, avoiding the Euler-Maruyama auxiliary process adopted in prior flow-model RL formulations. Experiments demonstrate that RAVEN surpasses recent causal video distillation baselines across quality, semantic, and dynamic degree evaluations, and that CM-GRPO provides further gains when combined with RAVEN.

### 1 Introduction

Recent progress in video diffusion has established bidirectional models as the dominant paradigm for high-fidelity generation [1–15]. Their reliance on bidirectional context and a large number of denoising steps, however, limits their suitability for real-time generation, where video must be produced continuously as a stream. This requirement has motivated causal autoregressive architectures that extrapolate future chunks from previously generated content [16–30]. The strongest generation capability still largely resides in high-step bidirectional models, and recent work has studied asymmetric distillation, which transfers knowledge from such bidirectional teachers to causal student generators [31–36]. The resulting few-step generators achieve real-time generation speeds while retaining much of the visual fidelity of their teachers.

A central challenge in autoregressive video diffusion distillation lies in how the model represents and reuses historical chunks, as each generated chunk becomes the context on which all subsequent predictions depend. As illustrated in Figure 1, existing training paradigms differ in both the source of historical states and whether those states receive end-to-end supervision from later chunks. Teacher Forcing trains with real historical chunks, which provides clean supervision but does not expose the generator to its own test-time history. Diffusion Forcing [37, 38] trains causal diffusion models by assigning each token an independently sampled Signal-to-Noise Ratio (SNR), and CausVid [35] adapts this construction to autoregressive video distillation by incorporating Distribution Matching Distillation (DMD) [39, 40]. This formulation optimizes the causal generator under a history

Preprint.

distribution that does not match inference, and the resulting discrepancy can accumulate across autoregressive rollouts. Self Forcing [31] reduces this discrepancy by conditioning the DMD objective on self rollouts, yet the historical cache is reused as detached context, so the history representations receive no end-to-end supervision from subsequent chunk losses.

We propose the Real-time Autoregressive Video Extrapolation Network (RAVEN), a training-time test framework that directly supervises the history construction used during autoregressive extrapolation. Starting from self rollouts of the few-step causal generator, RAVEN repacks the sampled trajectory into an interleaved sequence of clean historical endpoints and noisy denoising states. Within this sequence, clean rollout chunks provide the causal history for subsequent predictions, while noisy states from the same rollout remain the supervised denoising inputs. The resulting attention computation aligns more closely with inference than Teacher Forcing or Diffusion Forcing and keeps history representations inside the supervised forward pass, as shown in Figure 1(d). This design enables gradients from later chunks to shape the cached representations on which future predictions depend, while avoiding the cost of backpropagating through an entire autoregressive sampling trajectory.

Reinforcement learning (RL) has become an influential post-training paradigm for large generative models, and recent work has begun to adapt it to diffusion and flow models. Flow-GRPO [41] demonstrates this direction for flow matching, addressing the conflict between deterministic Ordinary Differential Equation (ODE) sampling and the stochastic exploration required by policy optimization through an ODE-to-Stochastic Differential Equation (SDE) conversion followed by Euler-Maruyama discretization. The causal generator in RAVEN employs a few-step consistency sampler, for which Euler-Maruyama introduces a train-test discrepancy by optimizing over stochastic transitions that differ from the deterministic sampling used at inference. We observe that a consistency sampling step can be cast as a conditional Gaussian transition parameterized by the predicted clean endpoint, enabling the policy objective to be defined on the same update rule used during generation without an auxiliary stochastic process. This correspondence is especially consequential for autoregressive video generation, where each generated chunk alters the history on which subsequent predictions depend. We therefore propose Consistency-model Group Relative Policy Optimization (CM-GRPO), which applies group relative policy optimization directly to this consistency transition kernel.

Our contributions are as follows.

- • We identify a history supervision gap in autoregressive video diffusion distillation, where existing methods are either optimized under history distributions that differ from inference or conditioned on rollout history without end-to-end supervision.
- • We introduce RAVEN, a training-time test framework that repacks self rollouts into an interleaved sequence of clean historical endpoints and noisy denoising states, allowing supervision to propagate through the history representations used during extrapolation.
- • We propose CM-GRPO, which reformulates a consistency sampling step as a conditional Gaussian transition kernel and applies group relative policy optimization directly to this kernel, matching the sampler interface used at inference.
- • We demonstrate that RAVEN surpasses recent causal video distillation baselines and that CMGRPO provides complementary gains when combined with RAVEN.

### 2 Related Work

Autoregressive Video Diffusion Distillation. Autoregressive video generation encompasses several parallel directions beyond the causal distillation setting studied in this paper. One line of work explores the design of the autoregressive rollout itself, either extending the prediction window for longer sequences or conditioning on intermediate noisy latents rather than fully denoised outputs as historical context [42, 43, 32, 44, 34, 45]. Although our current implementation conditions on clean latents, the training-time test paradigm can simulate these alternative history mechanisms to provide end-to-end supervision. A separate direction develops architectures with dedicated temporal memory for managing long-range context during training [42, 46–50], while a complementary body of training-free methods adapts models at inference time for length extrapolation [51–56]. Our framework is orthogonal to both families, as any strategy that generates and caches the next chunk through specialized memory designs can be executed within the self-rollout phase and benefit from the subsequent interleaved optimization.

[Figure 2]

Real Data Self Rollout Attention with Gradient Attention without Gradient No Attention

[Figure 3]

Noisy State Varying SNR

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

1 1 2 2 3

1 1 2 2 3

1 1 2 2 3

𝐾𝑉

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

[Figure 13]

[Figure 14]

[Figure 15]

- 1

- 1

[Figure 16]

- 2

- 2

[Figure 17]

- 3

- 1

- 1

[Figure 18]

- 2

- 2

[Figure 19]

- 3

- 1

- 1

[Figure 20]

- 2

- 2

[Figure 21]

- 3

[Figure 22]

[Figure 23]

[Figure 24]

1 2 3

| | | |
|---|---|---|
| | | |
| | | |

[Figure 25]

- 1

[Figure 26]

- 2

[Figure 27]

- 3

𝑄

(a) Teacher Forcing (b) Diffusion Forcing (c) Self Forcing (d) RAVEN (Ours)

- Figure 1: Attention Mask Configuration. Autoregressive video diffusion training paradigms differ in how historical states enter attention and whether those states receive end-to-end supervision from later chunks. Teacher Forcing and Diffusion Forcing rely on data-driven historical states, inducing a training distribution that differs from inference. Self Forcing shifts the history distribution toward inference but reuses the historical cache as detached context. RAVEN instead repacks each self rollout into clean historical endpoints and noisy denoising states, allowing later chunks to attend to the same history used during extrapolation while their losses supervise the cached representations.

Online RL in Diffusion Model. Online RL has become a practical paradigm for aligning diffusion and flow models after pretraining, beginning with reward-guided optimization for image generation and gradually evolving into policy optimization methods tailored to diffusion and flow trajectories [57, 41, 58–60]. This approach has since been extended to autoregressive generators and world models, where reinforcement learning serves not only for preference alignment but also for preserving pretrained capabilities and improving controllable generation over long horizons [61–66]. Parallel work applies online RL to distilled and few-step generators, where the central challenge is to improve alignment without sacrificing the efficiency that makes these models practical [67–69]. Much of the follow-up work has focused on refining the policy objective itself. Some methods revisit regularization to control reward hacking and distribution drift [70–73], while others study how the stochasticity or numerical form of the sampler shapes policy optimization [74–80]. A separate direction makes more deliberate use of the denoising trajectory, for instance through branching, tree search, or stepwise credit assignment [81–97]. Our method is most closely related to the literature on few-step generation and sampler design. Rather than adopting the Euler-Maruyama discretization used in prior online RL formulations for flow models, CM-GRPO formulates the policy objective directly on the consistency transition kernel and combines it with the training-time test framework of RAVEN, more closely matching the inference-time behavior of autoregressive video extrapolation.

### 3 Methodology

#### 3.1 Preliminaries

Let x1:T denote a sequence of latent video chunks and c the text condition, with hats used for student-generated quantities. Throughout the paper, the subscript t indexes the chunk position, while a superscript in parentheses, such as (n), (u), or (s), denotes the noise level. We write the autoregressive video diffusion model as

T

pθ(xt | ht,c), ht = H(x<t). (1)

pθ(x1:T | c) =

t=1

The operator H(·) denotes the history representation encoded by the model via its cache. For a noise level n, we define the noisy current chunk as zt(n) = αnxt + σnϵ, with ϵ ∼ N(0,I). Training paradigms are distinguished primarily by how the history ht is constructed from past chunks, and we detail this distinction in the following subsections.

History Formulation in Diffusion Forcing and Self Forcing. Recent methods for autoregressive video diffusion distillation are largely built on either Diffusion Forcing [37] or Self Forcing [31]. In CausVid [35], training follows Diffusion Forcing and represents the history as

hDFt = H z1(n1),...,zt(−nt1−1) , perturbing each ground-truth prefix chunk with an independently sam-

1-1 1-2 1-3

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Add Noise ➕

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

1 2 3 4

1 2 3 4

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Noisy State

[Figure 40]

Bidirectional Fake-Score Critic

[Figure 41]

[Figure 42]

Causal Student Generator

Clean Endpoint

Reuse Self Rollouts

Update

1-x KV Cache Denoise Encode Pack

1 2 3 4

1 2 3 4 MSE

Fake-Score Step

Generator Step

Chunk-wise Loss Scaling

[Figure 43]

[Figure 44]

- 1 1 2 2 3 3 4

[Figure 45]

[Figure 46]

[Figure 47]

Causal Student Generator

[Figure 48]

- 1 2 3 4 Bidirectional Fake-Score Critic

[Figure 49]

Attention Mask

Frozen Trainable Stop // Gradient

Update

[Figure 50]

reverseKL

[Figure 51]

Bidirectional Real-Score Teacher

[Figure 52]

Add Noise ➕

//

[Figure 53]

[Figure 54]

[Figure 55]

- Figure 2: Training Pipeline. RAVEN builds on score distillation with a training-time test formulation that aligns the generator’s training context with inference. In the fake-score step, the frozen generator performs autoregressive self rollout with KV cache reuse, producing the clean endpoints and noisy denoising states that are subsequently reused in the generator step. Rather than discarding these rollout states after critic training, RAVEN repacks them into an interleaved sequence of clean historical endpoints and noisy denoising states, processed under a causal attention mask so that each noisy state attends to the clean history the generator itself produced. This allows later chunk losses, scaled chunk-wise, to supervise the history representations on which future predictions depend.

pled noise level before entering the causal context. Self Forcing [31] instead unrolls the autoregressive generator at training time and reuses detached cache representations written as hSFt = sg(H(ˆx<t)), where the stop-gradient operator sg(·) treats historical chunks as fixed context for subsequent denoising steps. Both formulations therefore leave the cache construction outside end-to-end supervision, motivating the training-time test formulation introduced next.

Euler-Maruyama Discretization in Flow-GRPO. Flow-GRPO [41] starts from the rectified-flow ODE dyτ = vθ(yτ,τ,c)dτ, where yτ denotes the latent variable at denoising time τ ∈ [0,1]. To inject the stochasticity required for policy optimization, it introduces an ODE-to-SDE conversion and operates on the reverse-time SDE dyτ = bθ(yτ,τ,c)dτ + στ dw, where bθ(yτ,τ,c) is the drift term and στ dw the diffusion term. The drift term is given by

στ2 2τ

yτ + (1 − τ)vθ(yτ,τ,c) . (2) Applying Euler-Maruyama discretization yields

bθ(yτ,τ,c) = vθ(yτ,τ,c) +

√

∆τ ϵ, ϵ ∼ N(0,I). (3) Equivalently, the Euler-Maruyama step defines an isotropic Gaussian policy kernel,

yτ+∆τ = yτ + bθ(yτ,τ,c)∆τ + στ

πθEM yτ+∆τ | yτ,c = N yτ+∆τ;yτ + bθ(yτ,τ,c)∆τ,στ2∆τI . (4)

This auxiliary kernel makes the policy ratio and the KL term tractable in closed form, but its stochastic transitions remain absent from the deterministic ODE sampler used at inference. ODE-based samplers are typically deterministic [98–104], while the consistency sampler [105–109, 40, 110–112] is a notable exception in the few-step regime, remaining defined on the probability flow ODE trajectory while still yielding stochastic transitions that can serve as the policy interface directly.

#### 3.2 Training-Time Test via RAVEN

RAVEN is a training-time test framework for autoregressive video diffusion that aligns the training procedure with inference-time extrapolation. Building upon the asymmetric distillation formulated by CausVid [35], the pipeline distills knowledge from a frozen bidirectional teacher into the causal student generator. As illustrated in Figure 2, training alternates between a fake-score step and a generator step. In the fake-score step, the bidirectional fake-score critic is updated on self-rollout

samples perturbed with Gaussian noise. In the generator step, the causal student generator is updated via a reverse Kullback-Leibler (KL) score gradient computed from evaluations by both the bidirectional real-score teacher and the learned fake-score critic.

Let τ1 > ··· > τK = 0 denote the few-step sampling timesteps of the consistency sampler adopted by the generator. During the fake-score step, the frozen causal student generator autoregressively

produces, for each chunk index t, a full denoising trajectory {zˆt(τk)}Kk=1 along with the clean endpoint xˆt = zˆt(0). These clean endpoints are perturbed with Gaussian noise to form the training inputs for the fake-score critic. During the generator step, the same self rollout is reused and the noisy state at denoising level τk is taken directly from each chunk’s sampled trajectory. These rollout states are then packed into an input sequence processed under the attention mask illustrated in Figure 1(d). Specifically, for a sampled timestep u ∈ {τ1,...,τK−1}, the interleaved sequence takes the form

Iu = z ˆ1(u),xˆ1,zˆ2(u),xˆ2,...,zˆT(u−)1,xˆT−1,zˆT(u) , (5)

where zˆt(u) is the noisy state of chunk t at denoising level u and xˆt is the corresponding clean endpoint. Within this sequence, the noisy states serve as supervised denoising targets, while the

clean endpoints preceding chunk t constitute its history hRAVENt = H(ˆx<t). The causal student generator encodes these clean endpoints as history representations within the same forward pass, allowing later noisy states to attend to them under the causal attention structure employed during autoregressive extrapolation. The resulting predictions are subsequently perturbed with Gaussian noise and evaluated by the bidirectional real-score teacher and the fake-score critic to compute the reverse KL score gradient.

Reuse Self Rollouts. The formulation is inspired by the training-time test principle of EAGLE-

- 3 [113], where the model is trained on the context it will produce and encounter during speculative decoding. In language generation, this amounts to feeding a predicted draft token representation into the next simulated drafting step. The analogous construction is substantially more involved for autoregressive video diffusion, since each chunk is the endpoint of a multi-step denoising trajectory and future chunks depend on the resulting cache. A direct simulation would require unrolling the generator across all chunks and denoising steps within a single computation graph, incurring backpropagation through both autoregressive recursion and sampler dynamics. RAVEN avoids this cost by exploiting the self rollout already produced during the fake-score step, which is precisely the process that defines future context at inference. Repacking its states into an interleaved sequence, where generated clean chunks supply context and later noisy states remain supervised targets, reduces training-time test to a reorganization of existing self rollouts rather than an additional mechanism layered on top of score distillation, while faithfully preserving the dependency structure of autoregressive extrapolation.

Chunk-wise Loss Scaling. Within the interleaved training sequence, chunks along the autoregressive horizon are exposed to qualitatively different denoising conditions. Earlier chunks operate under limited historical context, whereas later chunks condition on richer accumulated history and must simultaneously maintain contextual consistency and suppress error propagation. To account for this positional asymmetry, we introduce a future participation score. For a sequence of J chunks, let mj denote the number of scalar elements in chunk j and let ℓj denote its summed loss. The future participation score is defined as pj = Jk=j mk / Jk=1 mk, namely the fraction of supervised elements contributed by chunk j and all subsequent chunks, which is larger for earlier chunks and decreases monotonically toward later ones. The resulting profile P = (p1,...,pJ) is passed to a predefined weighting function gη to produce nonnegative raw weights w˜1:J = gη(P), whose specific form is examined in the ablation studies. For any choice of gη, the normalized per-chunk weights and the aggregate chunk loss are given by wj = w˜j Jk=1 mk / Jk=1 w˜kmk and Lchunk =

J j=1 wjℓj / Jj=1 wjmj. The normalization ensures that the average element-wise weight is

preserved, so gη governs only the relative distribution of gradient emphasis across chunk positions. The complete training procedure is summarized in Algorithm 1 of Appendix A.

#### 3.3 Online RL via CM-GRPO

CM-GRPO is an online policy optimization method for few-step consistency generators. As discussed in the preliminaries, Flow-GRPO [41] achieves tractable policy optimization for flow matching by converting the deterministic ODE into an auxiliary SDE via Euler-Maruyama discretization, yet the

resulting stochastic transitions are absent from the ODE sampler used at inference. A consistency sampler, by contrast, inherently yields stochastic Gaussian transitions through its predicted clean endpoint, enabling CM-GRPO to formulate the policy objective directly on the consistency transition kernel without introducing any auxiliary stochastic process.

Consider a single consistency sampling step from noise level u to a lower level s. Given the current latent z˜(u) and condition c, the model predicts a clean endpoint xˆθ = fθ(˜z(u),u,c), from which the next latent is drawn as z˜(s) = αsxˆθ + σsϵ with ϵ ∼ N(0,I), where αs and σs are the noise schedule coefficients. This sampling rule induces the Gaussian transition probability

##### πθ z ˜(s) | z˜(u),c = N z ˜(s);µuθ→s,σs2I , µuθ→s = αsxˆθ, (6) which constitutes the policy interface in CM-GRPO.

To instantiate group relative policy optimization on this kernel, for each condition c the generator runs G independent consistency trajectories, each terminating in a clean output xˆi on which a scalar reward Ri is evaluated. Following GRPO [114], the group-normalized advantage is computed as Aˆi = Ri − mean({Rj}Gj=1) / std({Rj}Gj=1) + ϵ . This advantage is broadcast to all consistency sampling transitions within the same trajectory, converting the endpoint reward into a per-transition objective. For a transition from u to s, dropping the Gaussian normalization constant and terms independent of θ, the log probability under the consistency kernel reduces to

∥z˜i(s) − µuθ→s∥2 2σs2

log πθ z ˜i(s) | z˜i(u),ci = −

. (7)

Because µuθ→s = αsxˆθ, the gradient of the advantage-weighted log probability with respect to the predicted clean endpoint takes the form

z˜i(s) − µuθ→s σs2

∇xˆθ −Aˆi log πθ(˜zi(s) | z˜i(u),ci) = −Aˆiαs

. (8) CM-GRPO implements this update through the stop-gradient regression objective

  x ˆθ − sg x ˆθ +

2 , (9)

Aˆiαs 2σs2

z ˜i(s) − µuθ→s

LCM-GRPO = Ei,u,s

whose gradient with respect to xˆθ recovers exactly the endpoint gradient derived above, matching the score gradient update used in our implementation. The same formulation also admits reference policy

KL regularization. If a reference consistency model produces a clean endpoint xˆref under the same noisy state z˜(u), the KL divergence between the two Gaussian kernels reduces to

αs2∥xˆθ − xˆref∥2 2σs2

DKL πθ(· | z˜(u),c) πref(· | z˜(u),c) =

. (10)

This regularizer is tractable in principle, but in our current implementation the bidirectional teacher cannot be sampled through the consistency interface and therefore does not provide xˆref on this policy interface. We therefore derive this closed-form expression for completeness, leaving its practical application to future work in which a compatible reference consistency model is accessible. The complete training procedure is summarized in Algorithm 2 of Appendix A.

Reward Composition. Autoregressive video reinforcement learning requires reward signals that jointly capture motion dynamics, visual fidelity, and semantic alignment. We empirically find that overweighting visual fidelity or semantic alignment tends to encourage static generations, whereas an overly strong motion reward degrades the remaining two aspects, making reward design challenging. This difficulty is compounded by the limited availability of reliable holistic metrics for few-step video generation. Reward models based on vision-language models (VLMs) [115, 116] supply useful scalar preferences, yet their preference data are typically collected from high-step or high-quality generators, introducing a distribution shift when applied to outputs of few-step distilled models. We therefore combine VLM-based rewards with rewards derived from representation models [117–120]. Each reward component is normalized within the sampled group before weighted summation, balancing reward scales and preventing any single metric from dominating the group-relative advantage.

[Figure 56]

Figure 3: Qualitative comparisons. See supplementary for playble video clips.

### 4 Experiments

Implementation. All experiments are built upon Wan2.1-T2V-1.3B [13] as the base model with 3 latent frames per chunk, consistent with existing baselines. RAVEN adopts the same initialization as Causal Forcing [36], which grounds the causal student in ODE distillation from an autoregressive teacher to satisfy frame-level injectivity. The CM-GRPO stage then proceeds from the RAVEN checkpoint. For reward composition, we integrate several representation models derived from VBench [121] consisting of complementary dimensions with a VLM-based reward trained on human feedback for Text Alignment. The representation rewards span both temporal dynamics (i.e., Dynamic Degree and Motion Smoothness) and frame-level visual quality (i.e., Aesthetic Quality and Imaging Quality). More implementation details can be found at Appendix B.

Evaluation. We adopt VBench [121] and report the Total Score along with the Quality Score and Semantic Score following the protocol shared by all compared baselines. Because the Dynamic Degree dimension in VBench is computed from RAFT [120] optical-flow magnitudes and therefore picks up camera jitter and temporal drift alongside genuine motion, we additionally adopt UnifiedReward-32B [116] to evaluate dynamic degree on the full collection of 6,220 videos generated under the VBench prompt suite, as this reward model is trained on human preference annotations and carries complementary prior knowledge from its underlying VLM. We also conduct a user study to complement these evaluations in Appendix C.

Table 1: Quantitative comparison results.

Total Score

Qual. Score

Sem. Score

Dyn. Deg.

- 4.1 Comparison with Baselines

Method

#### Quantitative Comparisons. As reported

CausVid [35] 83.01 84.18 78.34 2.340 LongLive [34] 83.05 83.70 80.46 2.277 Rolling Forcing [32] 83.25 84.00 80.25 2.536 Self Forcing [31] 84.27 85.10 80.97 2.543 Reward Forcing [33] 84.39 85.27 80.87 2.508 Causal Forcing [36] 84.96 86.00 80.76 2.669

- in Table 1, earlier causal distillation baselines cluster within a narrow range on the total score, and gains on quality or semantic alignment are typically offset by reduced motion or the reverse. Causal Forcing [36] pushes the overall scores higher and partially recovers motion among the baselines, yet the same trade-off persists. RAVEN

+ CM-GRPO 85.08 86.12 80.96 2.829 RAVEN 85.15 86.18 81.04 2.951

+ CM-GRPO 85.46 86.54 81.17 2.962

[Figure 57]

Figure 4: Qualitative ablation on Training-time Test. See supplementary for playable video clips.

surpasses every prior baseline across all four dimensions, with the largest margin on dynamic degree, indicating that supervising the cached history alleviates this trade-off rather than redistributing it. Adding CM-GRPO to Causal Forcing yields a smaller gain concentrated on motion, whereas its combination with RAVEN produces the leading entry on every dimension, suggesting that the two contributions are complementary and that the policy update benefits from a generator already aligned with inference.

Qualitative Comparisons. Figure 3 contrasts Causal Forcing and RAVEN, each with and without CM-GRPO, across four prompts spanning animal motion, urban scenery, and human subjects. Causal Forcing exhibits severe structural failures under motion, stretching the corgi’s body into an unnaturally long shape, detaching the lion’s head from its body, and rendering colors in an over-saturated tone that looks unnatural. Adding CM-GRPO repairs these structural breakdowns and the facial distortion of the running child, while RAVEN avoids the same failures from the start and produces more realistic colors and proportions, with only mild blurring around fast-moving regions such as the boy’s face and the lion’s body. Combining RAVEN with CM-GRPO yields the most coherent results, sustaining structural stability and temporal smoothness through continuous motion such as the woman’s turning hair and the child running in the rain.

Table 2: Ablation on Training-time Test.

Total Score

Qual. Score

Sem. Score

Dyn. Deg.

4.2 Ablation Studies

Method

#### Effect of Training-time Test. All entries

Teacher Forcing (TF) 82.64 83.11 80.73 3.000 Diffusion Forcing (DF) 84.09 84.75 81.45 2.743 Self Forcing (SF) 84.06 84.68 81.56 2.347 DF w/ Self Rollout 83.30 83.96 80.65 2.979 RAVEN 85.15 86.18 81.04 2.951

- in Table 2 share the same ODE-distilled initialization and chunk-wise loss scaling, isolating the effect of how history is formed and supervised. TF achieves the strongest motion at the cost of the other two dimensions, while SF leads on semantic alignment but yields the weakest motion, reflecting how a detached cache withholds gradient from the history. Replacing real prefixes in DF with self-rollout endpoints recovers motion close to TF yet erodes quality and semantic alignment, indicating that aligning the history distribution without supervising it merely redistributes the error. RAVEN routes gradients from later chunks through the clean rollout endpoints used as history, attaining the leading total score while retaining motion close to TF. These quantitative tradeoffs are visually validated in Figure 4, where RAVEN consistently maintains structural coherence and object identity across frames, avoiding the severe structural distortions of TF and the blurring artifacts of SF.

Effect of Chunk-wise Loss Scaling. We compare candidate weighting functions gη that map the future participation score pj to a per-chunk weight, summarized in Figure 5. Two candidates are inherited from SD3 [122], the mode density at s ∈ {−0.54,0.81} and the logit-normal density at (µ,σ) = (0,1), each integrated over the participation interval covered by chunk j to produce its raw weight. The remaining family follows the shift parameterization πα(p) = αp/(1 + (α − 1)p) borrowed from flow-matching timestep schedules, where α = 1 biases mass toward early chunks and α = 0 recovers a uniform schedule, while a negative α applies π|α| to the reversed coordinate pJ/pj so that emphasis concentrates on later chunks instead. Profiles peaked near the middle of the rollout or biased toward early chunks fall below the uniform baseline on total and quality, with the shift at α = 1 trading semantic alignment for the highest dynamic degree. The shift at α = −1, the setting we adopt throughout RAVEN, instead raises the total score by roughly 1.3 points while keeping motion and semantic alignment competitive.

3.0

Mode (s = −0.54)

Shift (α = 1) Shift (α = 0) Shift (α = -1, Ours)

Total Score

Qual. Score

Sem. Score

Dyn. Deg.

Mode (s = 0.81)

Weighting Function gη

2.5

Logit-Normal (μ = 0, σ = 1)

Normalizedweightwj

2.0

Mode (s = −0.54) 82.58 82.86 81.43 2.996 Mode (s = 0.81) 83.08 83.74 80.47 2.971 Logit-Normal (µ = 0, σ = 1) 83.31 83.97 80.65 2.963

1.5

1.0

Shift (α = 1) 83.79 84.87 79.46 3.000 Shift (α = 0) 83.82 84.67 80.42 2.924 Shift (α = −1, Ours) 85.15 86.18 81.04 2.951

0.5

0.0

1 2 3 4 5 6 7

Chunk index j

Figure 5: Ablation on Chunk-wise Loss Scaling.

Effect of Reward Composition. Table 3 ablates each reward dimension around our adopted composition. Halving the Text Alignment (TA) weight lowers the total and quality scores without a semantic gain, and reducing the Dynamic Degree (DD) weight weakens both the total score and motion, indicating that the optical flow term carries the bulk of motion supervision rather than acting as a redundant complement to Motion Smoothness (MS). Raising the MS weight produces a smaller motion regression, while doubling Aesthetic Quality (AQ) and Imaging Quality (IQ) lifts semantic alignment to the highest entry at the cost of motion. The adopted composition therefore best balances the three aspects identified in the methodology.

Table 3: Ablation on Reward Composition.

Reward Dimensions Total Score

Dyn. TA DD MS AQ IQ Deg.

Qual. Score

Sem. Score

RAVEN 85.15 86.18 81.04 2.951

- 1 0.35 0.75 1 1 84.82 85.77 80.99 2.913
- 2 0.30 0.75 1 1 85.07 86.14 80.81 2.957 2 0.35 1.00 1 1 85.24 86.27 81.14 2.936 2 0.35 0.75 2 2 84.92 85.82 81.33 2.914 2 0.35 0.75 1 1 85.46 86.54 81.17 2.962

Effect of Consistency Policy. All entries in Table 4 share the RAVEN initialization and reward composition, differing only in the policy objective. The Euler-Maruyama (EM) interfaces follow the discretization of Flow-GRPO [41], with σ denoting the auxiliary diffusion noise level and β the reference KL weight, and across the resulting grid the EM scores stay within a narrow range, with the strongest setting only marginally exceeding RAVEN. CM-GRPO instead defines the policy objective on the consistency transition kernel used at inference and lifts the total, quality, and dynamic dimensions above every EM variant, surrendering only a small margin on semantic alignment. This indicates that an auxiliary stochastic process is unnecessary once the policy kernel matches the inference sampler, which also relieves the joint tuning of the diffusion noise and KL weight.

Table 4: Ablation on Consistency Policy.

Total Score

Qual. Score

Sem. Score

Dyn. Deg.

Policy Interface

RAVEN 85.15 86.18 81.04 2.951 + EM (σ = 0.1, β = 0) 85.06 86.10 80.92 2.949 + EM (σ = 0.4, β = 0) 85.15 86.20 80.95 2.949 + EM (σ = 0.8, β = 0) 85.22 86.31 80.84 2.951 + EM (σ = 0.1, β = 0.004) 85.03 86.07 80.86 2.947 + EM (σ = 0.4, β = 0.004) 85.14 86.16 81.09 2.950 + EM (σ = 0.8, β = 0.004) 85.27 86.26 81.29 2.950 + CM-GRPO (Ours) 85.46 86.54 81.17 2.962

### 5 Conclusion

We presented RAVEN, a training-time test framework that repacks each self rollout into an interleaved sequence of clean historical endpoints and noisy denoising states so that supervision propagates through the cached history used during autoregressive extrapolation. We further proposed CM-GRPO, which formulates the policy objective directly on the consistency transition kernel used at inference rather than on an auxiliary Euler-Maruyama process. Together they surpass recent causal video distillation baselines across quality, semantic, and dynamic dimensions, with their combination yielding the strongest results on every metric considered. Both formulations also admit broader scope than the setting evaluated here, and we discuss their generalizability in greater detail in Appendix D.

### 6 Acknowledgements

We thank Fei Ni and Changrui Chen for their valuable discussions and assistance. J. Deng was supported by the NVIDIA Academic Grant. The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR). Isambard-AI [123] is operated by the University of Bristol and is funded by the UK Government’s Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023].

### References

- [1] Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko, Viacheslav Vasilev, Alexey Letunovskiy, Nikolai Vaulin, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Nikita Kiselev, et al. Kandinsky 5.0: A family of foundation models for image and video generation. arXiv preprint arXiv:2511.14993, 2025.
- [2] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, et al. Sana-video: Efficient video generation with block linear diffusion transformer. In ICLR, 2026.
- [3] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, Xunsong Li, Yifu Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [4] Genmo. Mochi 1: Ai video generator. https://www.genmo.ai/blog/mochi-1-a-new-sota-in-o pen-text-to-video, 2024.
- [5] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.
- [6] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.
- [7] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2025.
- [8] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, Yu Zhou, Deshan Sun, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.
- [9] Meta. Movie gen: A cast of media foundation models. https://ai.meta.com/static-resource/ movie-gen-research-paper, 2024.
- [10] NVIDIA. Cosmos world foundation model platform for physical ai. https://research.nvidia.co m/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai, 2025.
- [11] Team Seedance, Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Tianheng Cheng, Xinqi Cheng, Xuyan Chi, Jian Cong, et al. Seedance 1.5 pro: A native audio-visual joint generation foundation model. arXiv preprint arXiv:2512.13507, 2025.
- [12] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, Mojie Chi, Xuyan Chi, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.
- [13] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [14] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, Linus, Patrol, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.
- [15] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In ICLR, 2025.

- [16] Sand ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [17] Xingjian Bai, Guande He, Zhengqi Li, Eli Shechtman, Xun Huang, and Zongze Wu. Causality in video diffusers is separable from denoising. arXiv preprint arXiv:2602.10095, 2026.
- [18] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.
- [19] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In ICLR, 2025.
- [20] Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, and Dahua Lin. End-to-end training for autoregressive video diffusion via self-resampling. arXiv preprint arXiv:2512.15702, 2025.
- [21] Junchao Huang, Ziyang Ye, Xinting Hu, Tianyu He, Guiyu Zhang, Shaoshuai Shi, Jiang Bian, and Li Jiang. Live: Long-horizon interactive video world modeling. arXiv preprint arXiv:2602.03747, 2026.
- [22] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In ICLR, 2025.
- [23] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinitelength video generation with error recycling. In ICLR, 2026.
- [24] Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Infinitystar: Unified spacetime autoregressive modeling for visual generation. In NeurIPS, 2025.
- [25] Ryan Po, Eric Ryan Chan, Changan Chen, and Gordon Wetzstein. Bagger: Backwards aggregation for mitigating drift in autoregressive video diffusion models. arXiv preprint arXiv:2512.12080, 2025.
- [26] Xiaofei Wu, Guozhen Zhang, Zhiyong Xu, Yuan Zhou, Qinglin Lu, and Xuming He. Pack and force your memory: Long-form and consistent video generation. arXiv preprint arXiv:2510.01784, 2025.
- [27] Xunzhi Xiang, Yabo Chen, Guiyu Zhang, Zhongyu Wang, Zhe Gao, Quanming Xiang, Gonghu Shang, Junqi Liu, Haibin Huang, Yang Gao, Chi Zhang, Qi Fan, et al. Macro-from-micro planning for high-quality and parallelized autoregressive long video generation. arXiv preprint arXiv:2508.03334, 2025.
- [28] Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. Helios: Real real-time long video generation model. arXiv preprint arXiv:2603.04379, 2026.
- [29] Zeyu Zhang, Shuning Chang, Yuanyu He, Yizeng Han, Jiasheng Tang, Fan Wang, and Bohan Zhuang. Blockvid: Block diffusion for high-quality and consistent minute-long video generation. arXiv preprint arXiv:2511.22973, 2025.
- [30] Lvmin Zhang, Shengqu Cai, Muyang Li, Chong Zeng, Beijia Lu, Anyi Rao, Song Han, Gordon Wetzstein, and Maneesh Agrawala. Pretraining frame preservation for lightweight autoregressive video history embedding. arXiv preprint arXiv:2512.23851, 2026.
- [31] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In NeurIPS, 2025.
- [32] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. In ICLR, 2026.
- [33] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, Yujun Shen, and Min Zhang. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. In CVPR, 2026.
- [34] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. Longlive: Real-time interactive long video generation. In ICLR, 2026.
- [35] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.

- [36] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. In ICML, 2026.
- [37] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024.
- [38] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. Historyguided video diffusion. In ICML, 2025.
- [39] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, pages 6613–6623, 2024.
- [40] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.
- [41] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. In NeurIPS, 2025.
- [42] Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028, 2026.
- [43] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.
- [44] Jinxiu Liu, Xuanming Liu, Kangfu Mei, Yandong Wen, Ming-Hsuan Yang, and Weiyang Liu. Streaming autoregressive video generation via diagonal distillation. In ICLR, 2026.
- [45] Kai Zou, Dian Zheng, Hongbo Liu, Tiankai Hang, Bin Liu, and Nenghai Yu. Hiar: Efficient autoregressive long video generation via hierarchical denoising. arXiv preprint arXiv:2603.08703, 2026.
- [46] Hanmo Chen, Chenghao Xu, Xu Yang, Xuan Chen, and Cheng Deng. Past- and future-informed kv cache policy with salience estimation in autoregressive video diffusion. arXiv preprint arXiv:2601.21896, 2026.
- [47] Sihui Ji, Xi Chen, Shuai Yang, Xin Tao, Pengfei Wan, and Hengshuang Zhao. Memflow: Flowing adaptive memory for consistent and efficient long video narratives. arXiv preprint arXiv:2512.14699, 2025.
- [48] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, and Xun Huang. Motionstream: Real-time video generation with interactive motion controls. In ICLR, 2026.
- [49] Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, and Xiaojuan Qi. Videossm: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519, 2025.
- [50] Tianrui Zhu, Shiyi Zhang, Zhirui Sun, Jingqi Tian, and Yansong Tang. Memorize-and-generate: Towards long-term consistency in real-time video generation. arXiv preprint arXiv:2512.18741, 2025.
- [51] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Lol: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026.
- [52] Jia Li, Xiaomeng Fu, Xurui Peng, Weifeng Chen, Youwei Zheng, Tianyu Zhao, Jiexi Wang, Fangmin Chen, Xing Wang, and Hayden Kwok-Hay So. Train short, inference long: Training-free horizon extension for autoregressive video generation. arXiv preprint arXiv:2602.14027, 2026.
- [53] Xunzhi Xiang, Zixuan Duan, Guiyu Zhang, Haiyu Zhang, Zhe Gao, Junta Wu, Shaofeng Zhang, Tengfei Wang, Qi Fan, and Chunchao Guo. Pathwise test-time correction for autoregressive long video generation. arXiv preprint arXiv:2602.05871, 2026.
- [54] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Action-controllable infinite video generation emerges from autoregressive self-rollout. In CVPR, 2026.
- [55] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.

- [56] Zengqun Zhao, Yanzuo Lu, Ziquan Liu, Jifei Song, Jiankang Deng, and Ioannis Patras. Relax forcing: Relaxed kv-memory for consistent long video generation. arXiv preprint arXiv:2603.21366, 2026.
- [57] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In ICLR, 2024.
- [58] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2026.
- [59] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. In NeurIPS, 2023.
- [60] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. In ICLR, 2026.
- [61] Xiaoxiao Ma, Haibo Qiu, Guohui Zhang, Zhixiong Zeng, Siqi Yang, Lin Ma, and Feng Zhao. Stage: Stable and generalizable grpo for autoregressive image generation. arXiv preprint arXiv:2509.25027, 2025.
- [62] Zehan Wang, Tengfei Wang, Haiyu Zhang, Xuhui Zuo, Junta Wu, Haoyuan Wang, Wenqiang Sun, Zhenwei Wang, Chenjie Cao, Hengshuang Zhao, Chunchao Guo, and Zhou Zhao. Worldcompass: Reinforcement learning for long-horizon world models. arXiv preprint arXiv:2602.09022, 2026.
- [63] Jialong Wu, Shaofeng Yin, Ningya Feng, and Mingsheng Long. Rlvr-world: Training world models with reinforcement learning. In ICLR, 2025.
- [64] Yang Ye, Tianyu He, Shuo Yang, and Jiang Bian. Reinforcement learning with inverse rewards for world model post-training. arXiv preprint arXiv:2509.23958, 2025.
- [65] Songchun Zhang, Zeyue Xue, Siming Fu, Jie Huang, Xianghao Kong, Y. Ma, Haoyang Huang, Nan Duan, and Anyi Rao. Astrolabe: Steering forward-process reinforcement learning for distilled autoregressive video models. arXiv preprint arXiv:2603.17051, 2026.
- [66] Kesen Zhao, Jiaxin Shi, Beier Zhu, Junbao Zhou, Xiaolong Shen, Yuan Zhou, Qianru Sun, and Hanwang Zhang. Real-time motion-controllable autoregressive video diffusion. In ICLR, 2026.
- [67] Guanjie Chen, Shirui Huang, Kai Liu, Jianchen Zhu, Xiaoye Qu, Peng Chen, Yu Cheng, and Yifu Sun. Flash-dmd: Towards high-fidelity few-step image generation with efficient distillation and joint reinforcement learning. arXiv preprint arXiv:2511.20549, 2025.
- [68] Xiefan Guo, Xinzhu Ma, Haoxiang Ma, Zihao Zhou, and Di Huang. Erudiff: Refactoring knowledge in diffusion models for advanced text-to-image synthesis. arXiv preprint arXiv:2603.20828, 2026.
- [69] Yihong Luo, Tianyang Hu, Weijian Luo, and Jing Tang. Tdm-r1: Reinforcing few-step diffusion models with non-differentiable reward. arXiv preprint arXiv:2603.07700, 2026.
- [70] Haoran He, Yuxiao Ye, Jie Liu, Jiajun Liang, Zhiyong Wang, Ziyang Yuan, Xintao Wang, Hangyu Mao, Pengfei Wan, and Ling Pan. Gardo: Reinforcing diffusion models without reward hacking. arXiv preprint arXiv:2512.24138, 2025.
- [71] Jie Liu, Zilyu Ye, Linxiao Yuan, Shenhan Zhu, Yu Gao, Jie Wu, Kunchang Li, Xionghui Wang, Xiaonan Nie, Weilin Huang, and Wanli Ouyang. Unigrpo: Unified policy optimization for reasoning-driven visual generation. arXiv preprint arXiv:2603.23500, 2026.
- [72] Haotian Ye, Kaiwen Zheng, Jiashu Xu, Puheng Li, Huayu Chen, Jiaqi Han, Sheng Liu, Qinsheng Zhang, Hanzi Mao, Zekun Hao, Prithvijit Chattopadhyay, Dinghao Yang, et al. Data-regularized reinforcement learning for diffusion models at scale. arXiv preprint arXiv:2512.04332, 2025.
- [73] Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, and Vicky Kalogeiton. Diffusion reinforcement learning via centered reward distillation. arXiv preprint arXiv:2603.14128, 2026.
- [74] Dailan He, Guanlin Feng, Xingtong Ge, Yazhe Niu, Yi Zhang, Bingqi Ma, Guanglu Song, Yu Liu, and Hongsheng Li. Neighbor grpo: Contrastive ode policy optimization aligns flow models. In CVPR, 2026.
- [75] Yihong Luo, Tianyang Hu, and Jing Tang. Reinforcing diffusion models by direct group preference optimization. In ICLR, 2026.

- [76] Jiayuan Sheng, Hanyang Zhao, Haoxian Chen, David D. Yao, and Wenpin Tang. Understanding sampler stochasticity in training diffusion models for rlhf. arXiv preprint arXiv:2510.10767, 2025.
- [77] Feng Wang and Zihao Yu. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.
- [78] Shaomeng Wang, He Wang, Longquan Dai, and Jinhui Tang. Pc-flow: Preference alignment in flow matching via classifier. In AAAI, 2026.
- [79] Shengjun Zhang, Zhang Zhang, Chensheng Dai, and Yueqi Duan. E-grpo: High entropy steps drive effective reinforcement learning for flow models. In CVPR Findings, 2026.
- [80] Mingzhe Zheng, Weijie Kong, Yue Wu, Dengyang Jiang, Yue Ma, Xuanhua He, Bin Lin, Kaixiong Gong, Zhao Zhong, Liefeng Bo, Qifeng Chen, and Harry Yang. Manifold-aware exploration for reinforcement learning in video generation. arXiv preprint arXiv:2603.21872, 2026.
- [81] Kaijie Chen, Zhiyang Xu, Ying Shen, Zihao Lin, Yuguang Yao, and Lifu Huang. Superflow: Training flow matching models with rl on the fly. arXiv preprint arXiv:2512.17951, 2026.
- [82] Zheng Ding and Weirui Ye. Treegrpo: Tree-advantage grpo for online rl post-training of diffusion models. In ICLR, 2026.
- [83] Xiaolong Fu, Lichen Ma, Zipeng Guo, Gaojing Zhou, Chongxiao Wang, ShiPing Dong, Shizhe Zhou, Shizhe Zhou, Ximan Liu, Jingling Fu, Tan Lit Sin, Yu Shi, et al. Dynamic-treerpo: Breaking the independent trajectory bottleneck with structured sampling. arXiv preprint arXiv:2509.23352, 2025.
- [84] Xiaoxuan He, Siming Fu, Yuke Zhao, Wanli Li, Jian Yang, Dacheng Yin, Fengyun Rao, and Bo Zhang. Tempflow-grpo: When timing matters for grpo in flow models. In ICLR, 2026.
- [85] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. In ICLR, 2026.
- [86] Zhanhao Liang, Tao Yang, Jie Wu, Chengjian Feng, and Liang Zheng. Leapalign: Post-training flow matching models at any generation step by building two-step trajectories. arXiv preprint arXiv:2604.15311, 2026.
- [87] Yifu Luo, Penghui Du, Bo Li, Sinan Du, Tiantian Zhang, Yongzhe Chang, Kai Wu, Kun Gai, and Xueqian Wang. Sample by step, optimize by chunk: Chunk-level grpo for text-to-image generation. arXiv preprint arXiv:2510.21583, 2025.
- [88] Qiang Lyu, Zicong Chen, Chongxiao Wang, Haolin Shi, Shibo Gao, Ran Piao, Youwei Zeng, Jianlou Si, Fei Ding, Jing Li, Chun Pong Lau, and Weiqiang Wang. Multi-grpo: Multi-group advantage estimation for text-to-image generation with tree-based trajectories and multiple rewards. arXiv preprint arXiv:2512.00743, 2025.
- [89] David McAllister, Miika Aittala, Tero Karras, Janne Hellsten, Angjoo Kanazawa, Timo Aila, and Samuli Laine. Finite difference flow optimization for rl post-training of text-to-image models. arXiv preprint arXiv:2603.12893, 2026.
- [90] Yash Savani, Branislav Kveton, Yuchen Liu, Yilin Wang, Jing Shi, Subhojyoti Mukherjee, Nikos Vlassis, and Krishna Kumar Singh. Stepwise credit assignment for grpo on flow-matching models. In CVPR, 2026.
- [91] Yawen Shao, Jie Xiao, Kai Zhu, Yu Liu, Wei Zhai, Yang Cao, and Zheng-Jun Zha. Anchoring values in temporal and group dimensions for flow matching model alignment. arXiv preprint arXiv:2512.12387, 2025.
- [92] Sophia Tang, Yuchen Zhu, Molei Tao, and Pranam Chatterjee. Tr2-d2: Tree search guided trajectory-aware fine-tuning for discrete diffusion. arXiv preprint arXiv:2509.25171, 2025.
- [93] Yunze Tong, Mushui Liu, Canyu Zhao, Wanggui He, Shiyi Zhang, Hongwei Zhang, Peng Zhang, Jinlong Liu, Ju Huang, Jiamang Wang, Hao Jiang, and Pipei Huang. Alleviating sparse rewards by modeling step-wise and long-term sampling effects in flow-based grpo. arXiv preprint arXiv:2602.06422, 2026.
- [94] Jing Wang, Jiajun Liang, Jie Liu, Henglin Liu, Gongye Liu, Jun Zheng, Wanyuan Pang, Ao Ma, Zhenyu Xie, Xintao Wang, Meng Wang, Pengfei Wan, et al. Grpo-guard: Mitigating implicit over-optimization in flow matching via regulated clipping. arXiv preprint arXiv:2510.22319, 2025.

- [95] Zhixiong Yue, Zixuan Ni, Feiyang Ye, Jinshan Zhang, Sheng Shen, and Zhenpeng Mi. Know your step: Faster and better alignment for flow matching models via step-aware advantages. arXiv preprint arXiv:2602.01591, 2026.
- [96] Liyu Zhang, Kehan Li, Tingrui Han, Tao Zhao, Yuxuan Sheng, Shibo He, and Chao Li. Op-grpo: Efficient off-policy grpo for flow-matching models. arXiv preprint arXiv:2604.04142, 2026.
- [97] Yujie Zhou, Pengyang Ling, Jiazi Bu, Yibin Wang, Yuhang Zang, Jiaqi Wang, Li Niu, and Guangtao Zhai. Fine-grained grpo for precise preference alignment in flow models. In CVPR, 2026.
- [98] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.
- [99] Yanzuo Lu, Xin Xia, Manlin Zhang, Huafeng Kuang, Jianbin Zheng, Yuxi Ren, and Xuefeng Xiao. Hyper-bagel: A unified acceleration framework for multimodal understanding and generation. arXiv preprint arXiv:2509.18824, 2025.
- [100] Yanzuo Lu, Yuxi Ren, Xin Xia, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Andy J Ma, Xiaohua Xie, and Jian-Huang Lai. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In ICCV, 2025.
- [101] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. In NeurIPS, 2024.
- [102] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022.
- [103] Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, Hongsheng Li, and Xiaogang Wang. Phased consistency model. In NeurIPS, 2024.
- [104] Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. In NeurIPS, 2024.
- [105] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. In ICLR, 2024.
- [106] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [107] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.
- [108] Weijian Luo, Zemin Huang, Zhengyang Geng, J. Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. In NeurIPS, 2024.
- [109] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023.
- [110] Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao, and TatJen Cham. Trajectory consistency distillation: Improved latent consistency distillation by semi-linear consistency function with trajectory mapping. arXiv preprint arXiv:2402.19159, 2024.
- [111] Mingyuan Zhou, Huangjie Zheng, Yi Gu, Zhendong Wang, and Hai Huang. Adversarial score identity distillation: Rapidly surpassing the teacher in one step. In ICLR, 2025.
- [112] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In ICML, 2024.
- [113] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. In NeurIPS, 2025.
- [114] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [115] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, et al. Improving video generation with human feedback. In NeurIPS, 2025.
- [116] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2026.
- [117] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In ICCV, pages 5128–5137, 2021.
- [118] LAION-AI. aesthetic-predictor. https://github.com/LAION-AI/aesthetic-predictor, 2022.
- [119] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In CVPR, pages 9801–9810, 2023.
- [120] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In ECCV, volume 12347, pages 402–419, 2020.
- [121] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.
- [122] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.
- [123] Simon McIntosh-Smith, Sadaf Alam, and Christopher Woods. Isambard-ai: a leadership-class supercomputer optimised specifically for artificial intelligence. arXiv preprint arXiv:2410.11199, 2024.
- [124] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. In NeurIPS Datasets and Benchmarks Track, 2024.
- [125] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. In ICLR, 2025.
- [126] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In ECCV, 2022.
- [127] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2022.
- [128] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
- [129] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.
- [130] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [131] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.

Algorithm 1 RAVEN training iteration Require: causal student fθ, fake-score critic fϕ, bidirectional teacher fψ Require: text condition c, chunk count T, noise schedule (αn,σn) Require: sampling timesteps τ1 > ··· > τK = 0 Require: chunk-wise weighting function gη, learning rates ηθ and ηϕ Require: iteration index i, TTUR ratio r (critic-to-generator update frequency)

#### Stage 1: Self Rollout no gradient through fθ

- 1: for t = 1,...,T do
- 2: sample initial noisy state zˆt(τ1) ∼ N(0,I)
- 3: for k = 1,...,K − 1 do
- 4: predict clean endpoint xˆ(tk) ← Endpoint fθ(ˆzt(τk),τk,c,ht)
- 5: sample ϵ ∼ N(0,I)
- 6: consistency transition zˆt(τk+1) ← ατ

k+1

xˆ(tk) + στ

k+1

ϵ ▷ Eq. (6)

- 7: end for
- 8: final endpoint xˆt ← Endpoint fθ(ˆzt(τK),τK,c,ht)
- 9: update KV cache with xˆt so that ht+1 = H(ˆx≤t)
- 10: end for

- Stage 2: Fake-Score Step

11: concatenate the rollout endpoints into the full clean sequence xˆ = (ˆx1,...,xˆT) 12: sample n ∼ U(0,1) and ϵ ∼ N(0,I)

13: perturb z(n) ← αnxˆ + σnϵ 14: bidirectional forward pass of fϕ on z(n) 15: update critic ϕ ← ϕ − ηϕ∇ϕ fϕ(z(n),n,c) − Target(ˆx) 2

- Stage 3: Generator Step on Iu runs only when i mod r = 0

- 16: if i mod r = 0 then
- 17: sample sampling level u ∈ {τ1,...,τK−1}
- 18: sample score level s ∼ U(0,u)
- 19: assemble interleaved sequence Iu from the rollout ▷ Eq. (5)
- 20: causal forward of fθ on Iu, yielding xˆθ
- 21: sample ϵ ∼ N(0,I)
- 22: consistency transition zˆ(s) ← sg αsxˆθ + σsϵ ▷ Eq. (6)
- 23: teacher endpoint xˆψ ← Endpoint(fψ(ˆz(s),s,c))
- 24: critic endpoint xˆϕ ← Endpoint(fϕ(ˆz(s),s,c))
- 25: per-element DMD loss ℓ ← x ˆθ − sg x ˆθ +

xˆψ − xˆϕ ∥xˆθ − xˆψ∥1

2

- 26: split ℓ along the chunk axis into per-chunk losses ℓt
- 27: compute future participation scores pt and chunk weights wt ∝ gη(pt)
- 28: chunk-weighted DMD loss LDMD ← t

wtℓt t wtmt

- 29: θ ← θ − ηθ∇θLDMD
- 30: end if

### A Algorithm Formulations

- Algorithm 1 formalizes one RAVEN training iteration. The iteration begins with a self rollout of fθ under the consistency sampler, producing per-chunk denoising trajectories {zˆt(τk)} together with

clean endpoints xˆt = zˆt(0). The fake-score critic fϕ is updated on noised endpoints every iteration, whereas the generator fθ is updated only once every r iterations on the interleaved sequence Iu, with a reverse-KL score gradient defined by the teacher fψ and the updated critic fϕ. A single causal forward pass over Iu then routes gradients from later chunks through the cached history hRAVENt = H(ˆx<t) used during extrapolation.

Building on the RAVEN checkpoint, Algorithm 2 formalizes one CM-GRPO training iteration. The iteration draws a group of G independent consistency rollouts under a shared text condition c,

- Algorithm 2 CM-GRPO training iteration

Require: consistency generator fθ initialized from a RAVEN checkpoint, reward composition {Rm}

with weights {λm} Require: text condition c, group size G, chunk count T, noise schedule (αn,σn) Require: sampling timesteps τ1 > ··· > τK = 0 Require: chunk-wise weighting function gη, advantage clip Amax, learning rate ηθ Require: normalization stabilization constant ε

#### Stage 1: Group Rollouts no gradient through fθ

- 1: for i = 1,...,G do
- 2: for t = 1,...,T do
- 3: sample initial noisy state z˜i,t(τ1) ∼ N(0,I)
- 4: for k = 1,...,K − 1 do
- 5: predict clean endpoint xˆ(i,tk) ← Endpoint fθ(˜zi,t(τk),τk,c,hi,t)
- 6: sample ϵ ∼ N(0,I)
- 7: consistency transition z˜i,t(τk+1) ← ατ

k+1

xˆ(i,tk) + στ

k+1

ϵ ▷ Eq. (6)

- 8: end for
- 9: final endpoint xˆi,t ← Endpoint fθ(˜zi,t(τK),τK,c,hi,t)
- 10: update KV cache so that hi,t+1 = H(ˆxi,≤t)
- 11: end for
- 12: trajectory endpoint xˆi ← (ˆxi,1,...,xˆi,T)
- 13: end for

- Stage 2: Reward Evaluation and Group-Relative Advantages

14: for i = 1,...,G do 15: evaluate each reward dimension Rmi ← Rm(ˆxi,c) 16: end for 17: per-dimension group normalization R¯mi ← Rmi − mean({Rmj }Gj=1) / std({Rmj }Gj=1) + ε 18: composite reward Ri ← m λmR¯mi / m |λm| 19: group-relative advantage Aˆi ← Ri − mean({Rj}Gj=1) / std({Rj}Gj=1) + ε 20: clip advantage Aˆi ← clip(Aˆi,−Amax,Amax)

- Stage 3: Policy Update on Sampled Transitions

- 21: for i = 1,...,G do
- 22: sample ki ∼ U{1,...,K − 1} and let ui ← τk

i

, si ← τk

i+1

- 23: retrieve z˜i(ui),z˜i(si) from the cached trajectory
- 24: end for
- 25: predict clean endpoint xˆiθ ← Endpoint fθ(˜zi(ui),ui,c) for all i
- 26: consistency kernel mean µu

i→si

θ ← αs

i

xˆiθ ▷ Eq. (6)

- 27: per-trajectory stop-gradient regression loss

ℓi ← x ˆiθ − sg x ˆiθ +

Aˆi αs

i

2σs2

i

z ˜i(si) − µu

i→si θ

2

▷ Eq. (9)

- 28: split each ℓi along the chunk axis into per-chunk losses ℓit
- 29: compute future participation scores pt and chunk weights wt ∝ gη(pt)
- 30: chunk-weighted CM-GRPO loss LCM-GRPO ← i,t

wtℓit i,t wtmt

- 31: θ ← θ − ηθ∇θLCM-GRPO

scores each rollout with the composite reward, and converts the endpoint scores into a group-relative advantage following Eq. (9). A single transition u → s is then sampled along each rollout, and the policy update is realized as the stop-gradient regression on the predicted clean endpoint of the consistency kernel, inheriting the chunk-wise weighting introduced by RAVEN so that supervision remains aligned with the autoregressive horizon. The transition index is restricted to ki ∈ {1,...,K− 1} rather than the full set of consistency steps, since the final transition lands at τK = 0 with στ

= 0, where the kernel collapses to a Dirac delta and the policy log-probability is undefined.

K

[Figure 58]

Figure 6: User study preference rates on Quality, Semantic, and Overall.

### B More Implementation Details

Dataset. Both RAVEN and CM-GRPO are trained exclusively on text prompts drawn from VidProM [124], preprocessed through filtering and large language model extension, following the data protocol of Self Forcing [31]. Ablation experiments that require real video data draw from OpenVidHD-0.4M [125], with all video clips temporally upsampled via RIFE [126] prior to use.

Training Details. Most RAVEN training settings are inherited from Self Forcing [31] and Causal Forcing [36]. We disable weight decay throughout and reduce the two-time-scale update rule (TTUR) [40] ratio between critic and generator updates from 5 to 2. CM-GRPO is instead optimized in a parameter-efficient manner on top of the RAVEN checkpoint, applying LoRA [127] of rank 256 to all linear layers under AdamW [128, 129] with learning rate 5 × 10−6, zero weight decay, betas (0.0,0.999) and epsilon 10−10, where each policy update consumes a batch size of 8 paired with a group size of 32. The training-time test framework and chunk-wise loss scaling introduced by RAVEN are carried over unchanged into the CM-GRPO stage, so that the policy update inherits the same alignment between training context and inference-time extrapolation. The Causal Forcing [36] + CM-GRPO entry in Table 1 does not utilize these two designs, so as to isolate the contribution of the policy objective from RAVEN itself. For this entry, the learning rate is lowered to 2 × 10−6 and weight decay is reinstated at 0.01, while the reward weight on the Dynamic Degree (DD) dimension is raised from 0.35 to 2.35 to compensate for the weaker motion of the Causal Forcing checkpoint. RAVEN and CM-GRPO consume training budgets of approximately 70 and 170 NVIDIA H200 GPU hours respectively.

Reward Composition. Temporal dynamics are captured through dynamic degree and motion smoothness, where the former takes the top-5% mean of optical flow magnitudes estimated by RAFT [120] across consecutive frame pairs as an index of peak scene motion, and the latter is derived from the reconstruction error incurred when AMT [119] recovers artificially dropped frames from their temporal neighbors, with lower error reflecting greater temporal coherence. At the frame level, the LAION aesthetic predictor [118] applies a linear model on image embeddings to score each frame for compositional appeal and perceptual naturalness, while the multi-scale MUSIQ [117] evaluates low-level technical distortion. Text-video alignment is additionally assessed by VideoReward [115], a reward model built on Qwen2-VL [130] and trained via direct preference optimization [131] on 182K pairwise human preference annotations.

### C User Study

We conduct a user study on 100 long and detailed prompts drawn from the qualitative showcases of the existing baselines, generating 4 samples per prompt for each method. The study covers the four baselines designed for 5-second short video generation, namely CausVid [35], Self Forcing [31], Reward Forcing [33], and Causal Forcing [36]. For each sample pair, an individual user rates a RAVEN clip against its baseline counterpart presented in randomized order along Quality, Semantic, and Overall, following the instructions in Figure 7. Aggregate preference rates are reported in Figure 6, where RAVEN is preferred on every dimension against all four baselines, with a more pronounced lead on Semantic than on Quality and a clear margin on Overall.

[Figure 59]

Figure 7: User study instruction screenshot.

### D Discussion

Although RAVEN and CM-GRPO are presented with concrete design choices tailored to causal autoregressive video distillation, both formulations admit broader scope than the setting evaluated in our experiments. The interleaved sequence construction underlying RAVEN currently treats clean chunks as historical context, yet the supervised forward pass does not restrict the form of cached history. Arbitrary representations derived from the rollout can be substituted in its place, including intermediate noisy states from preceding chunks, memory tokens specific to the underlying architecture, and cache management strategies such as sliding windows or attention sinks, provided the same representation is consumed by subsequent chunks during self rollout. These representations can in principle be instantiated inside the supervised forward pass and optimized end-to-end through downstream losses on later noisy states, which would turn history handling into a learnable rather than fixed element of the rollout.

CM-GRPO admits an analogous extension along a different axis. Its policy interface depends only on the conditional Gaussian transition induced by the consistency sampler and is independent of the autoregressive structure of the generator on which it is applied. The same objective therefore applies to any few-step generator that draws samples through a stochastic consistency step, encompassing bidirectional video models as well as generators in other modalities that have been distilled through consistency training or distillation. Collectively, RAVEN constitutes a general training-time test interface for autoregressive few-step distillation, while CM-GRPO constitutes a general policy optimization interface for consistency generators, indicating that both contributions extend beyond the specific setting studied here.

### E Prompts

We list the text prompts behind the qualitative comparisons in the main text, indexed by their position within each figure grid.

#### Figure 3 (qualitative comparison).

- • Top-left. A cheerful, happy Corgi playing in a futuristic park during sunset, set in a cyberpunk style. The Corgi has a playful expression, wagging its tail and running around in the grass. The park is illuminated by neon lights and surrounded by towering skyscrapers with holographic advertisements flashing on their surfaces. The sky is a blend of orange and purple hues, creating a striking contrast against the dark cityscape. The Corgi is in the foreground, while the vibrant city lights and buildings create a dynamic background. The scene captures the essence of a cyberpunk world with natural elements intertwined. Medium close-up shot focusing on the Corgi’s joyful playfulness.
- • Top-right. A majestic African lion with golden fur and piercing green eyes grabs a small antelope tightly in its powerful jaws. The lion’s mane flows gracefully as it crouches on all fours, muscles tense and alert. After a moment of contemplation, the lion releases the antelope, which quickly runs away. The scene takes place in a dense savanna with tall grass and scattered acacia trees. The lion remains still for a few moments before turning its gaze back to the fleeing antelope. Wide shot, with a focus on the lion’s expressions and movements.
- • Bottom-left. A cheerful woman with a warm smile, standing outdoors under a clear blue sky. Her hair is flowing freely as she faces the wind, creating a gentle, swirling wind effect around her. She is wearing a light, flowy dress that moves gracefully with the breeze. The background includes green trees and blooming flowers, adding a vibrant and lively atmosphere. The scene captures a close-up of the woman, emphasizing her joyful expression and the dynamic wind movement around her.
- • Bottom-right. On a rainy day, a young boy with tousled brown hair is sprinting away from the rain, his clothes slightly damp and clinging to his body. His face is animated with excitement and joy as he runs barefoot through puddles, splashing water everywhere. The sky is overcast, with heavy raindrops falling steadily. In the background, blurred figures of people walking with umbrellas can be seen. The scene is captured in a medium shot, focusing on the boy’s dynamic movements and expressions.

#### Figure 4 (qualitative ablation).

- • Left. A documentary-style nature photography shot from a camera truck moving to the left, capturing a crab quickly scurrying into its burrow. The crab has a hard, greenish-brown shell and long claws, moving with determined speed across the sandy ground. Its body is slightly arched as it burrows into the sand, leaving a small trail behind. The background shows a shallow beach with scattered rocks and seashells, and the horizon features a gentle curve of the coastline. The photo has a natural and realistic texture, emphasizing the crab’s natural movement and the texture of the sand. A close-up shot from a slightly elevated angle.
- • Middle. A dynamic action shot of a surfer accelerating on a powerful wave, carving through the water with grace and agility. The surfer, with a tanned complexion and muscular build, rides the wave with one hand gripping the board while the other extends outwards for balance. The water splashes behind, creating a foamy trail, and the sun casts a golden glow over the scene. The background features a clear blue ocean and distant white-capped waves, with a few seagulls flying overhead. The surfer’s expression is one of exhilaration and focus. A mid-shot from a low-angle perspective capturing the surfer’s motion and the wave’s power.
- • Right. A scenic photograph capturing the moment a steam train departs from the Glenfinnan Viaduct, a historic railway bridge in Scotland. The train moves gracefully over the arch-covered viaduct, its smoke billowing into the air. The landscape is lush with greenery, and towering rocky mountains frame the scene, creating a picturesque backdrop. The sky is a clear, bright blue with the sun shining down, casting a warm glow on the train and the surrounding scenery. The viaduct itself is a striking feature, with intricate ironwork and a verdant setting. The photo has a classic, nostalgic feel, emphasizing the natural beauty and historical charm of the location. A wide-angle shot from a slightly elevated angle, capturing both the train and the expansive landscape.

### F Broader Impacts

RAVEN and CM-GRPO advance the practical viability of real-time autoregressive video generation, which carries both positive and negative societal implications. On the positive side, real-time lowlatency video synthesis can support creative tools, education, accessibility applications, and interactive simulation, while reducing the compute footprint of video generation pipelines and lowering the barrier for downstream research. On the negative side, more capable and more accessible video generators raise familiar misuse concerns, including the production of misleading or non-consensual synthetic media, identity impersonation, and content that could be used to spread misinformation. These risks are inherited from the underlying generative models rather than introduced by our contributions, but the gain in efficiency does broaden the population of users who can produce such content. We encourage downstream applications to combine model release with provenance tooling, watermarking, and content moderation, and to follow community norms for the responsible release of generative video models.

