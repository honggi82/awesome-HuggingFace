# arXiv:2606.18208v1[cs.LG]16Jun2026

## LOOPED WORLD MODELS

### FaceMind Research Asia

[Figure 1]

Leading Contributors Hongyuan Adam Lu* Z.L. Victor Wei Core Contributors Qun Zhang Jinrui Zeng Bowen Cao Lingwei Meng Mocheng Li Zezhong Wang Haonan Yin Naifu Xue Minyu Chen Cenyuan Zhang Zefan Zhang Hao Wei Jiawei Zhou Haoran Xu Hao Yang Ronglai Zuo Tongda Xu Yonghao Li Jian Chen Hebin Wang Zeyu Gao Yang Li Wei Zhao Qimin Zhong Siqi Liu Yumeng Zhang Leyan Cui Zhangyu Wang Wai Lam

ABSTRACT

Current world models face a fundamental tension: faithful long-horizon simulation demands deep computation, but deeper models are expensive to deploy and prone to compounding errors. We resolve this by introducing Looped World Models (LoopWM), which are the first looped architectures for world modelling. Our method iteratively refines latent environment states through a parameter-shared transformer block. This yield up to 100× parameter efficiency over conventional approaches with adaptive computation that automatically scales depth to match the complexity of each prediction step. Orthogonal to scaling model size and training data, LoopWM establishes iterative latent depth as a new scaling axis for world simulation, which might significantly push the community forward.

[Figure 2]

Figure 1: The overall framework of our proposed Looped World Models (LoopWM).

- 1 INTRODUCTION

World models (WM) learn to predict how an environment evolves in accordance with actions. WM has become a cornerstone of sample-efficient reinforcement learning and embodied intelligence (Ha & Schmidhuber, 2018; Hafner et al., 2019; Łukasz Kaiser et al., 2020). Remarkably, the Deep Planning Network (PlaNet) is a WM (Hafner et al., 2019) first demonstrated that agents can learn latent dynamics entirely from pixels and plan via online optimisation. This establishes the recurrent statespace model (RSSM) as a foundational architecture for world modelling. The Dreamer family of models then (Hafner et al., 2020; 2021; 2025) progressively refined this approach, culminating in DreamerV3 (Hafner et al., 2025). DreamerV3 masters over 150 different tasks with a single set of hyperparameters. Seeking to leverage the representational power of transformers, subsequent work replaced or augmented the recurrent backbone. IRIS (Micheli et al., 2023) showed that an autoregressive transformer over discrete latent tokens can serve as a highly data-efficient world model. TransDreamer (Chen et al., 2022) introduced a Transformer State-Space Model for tasks demanding long-range memory. ∆-IRIS (Micheli et al., 2024) improved efficiency via context-aware delta tokenisation. DIAMOND (Alonso et al., 2024) demonstrated that diffusion models can produce visually faithful world simulations, and EMERALD (Burchi & Timofte, 2025) achieved state-ofthe-art Crafter performance by combining masked generative transformers with spatial latent states. At a larger scale, Sora (OpenAI, 2024) and Genie (Bruce et al., 2024; Google DeepMind, 2025) demonstrated that video generation models and generative interactive environments can serve as general-purpose world simulators. And multiple surveys have charted the rapid expansion of world models into autonomous driving (Feng et al., 2025), embodied AI (Li et al., 2025b), and video generation (Dewi Puspitasari et al., 2024; Wang et al., 2026).

Despite this progress, faithful long-horizon simulation often requires deep or iterative computation. This is because physical dynamics unfold through repeated application of governing laws, whereas conventional fixed-depth architectures allocate the same amount of computation to every transition regardless of its difficulty. There are two typical failure modes. First, prediction errors cause trajectory quality to degrade rapidly over extended horizons across rollout steps (Xiao et al., 2020; Talvitie, 2017; Luo et al., 2022). Second, scaling model depth to combat this degradation proportionally usually increases parameter count and inference cost, which then makes real-time deployment on resource-constrained platforms prohibitively expensive (Feng et al., 2025; Hafner et al., 2025).

A parallel line of research has explored looped transformer architectures (LM). In LM, a shared set of transformer blocks is applied recurrently to the same latent representation. Such a concept was first proposed as the Universal Transformer (Dehghani et al., 2019), which introduced weightsharing across depth with an adaptive halting mechanism inspired by Adaptive Computation Time (Graves, 2016). Early theoretical work shows that LM can simulate arbitrary iterative algorithms. This includes gradient descent, Newton’s method, and dynamic programming, with constant parameter count (Giannou et al., 2023), and they achieve comparable in-context learning performance to standard transformers while using less than 10% of the parameters (Yang et al., 2023). ALBERT (Lan et al., 2020) shows the practical viability of cross-layer parameter sharing for language representation learning. MoEUT (Csord´as et al., 2024) combined mixture-of-experts with universal transformers.

More recently, looped architectures have been scaled to practical language models with promising results. Zhu et al. (2025) demonstrated that a looped language model can achieve about 2 to 3× parameter efficiency through iterative latent computation. Geiping et al. (2025) showed that recurrent-depth models can scale test-time compute by simply increasing the number of loop iterations at inference. Fan et al. (2024) demonstrated that looped transformers with adaptive stopping significantly improve length generalisation. Saunshi et al. (2025) provided theoretical and empirical evidence that looped models implicitly generate latent thoughts. Jeddi et al. (2026) introduced elastic-depth training with shortcut modulation for budget-conditioned latent reasoning. Bae et al. (2025) proposed per-token dynamic recursive depth allocation within a single recursive transformer. Prairie et al. (2026) addressed the training instability of looped models by recasting the looped forward pass as a nonlinear dynamical system over the residual stream and constraining the spectral norm of the state-transition matrix through a negative-diagonal parameterisation. Most recently, Hyperloop Transformers (Zeitoun et al., 2026) augmented the looped block with matrixvalued hyper-connected residual streams. This outperforms depth-matched standard transformers at half the parameter count. These developments connect looped transformers to a broader family of

depth-continuous and implicit-layer models, including Neural ODEs (Chen et al., 2018) and Deep Equilibrium Models (Bai et al., 2019), which likewise iterate a shared function toward a fixed point.

However, all of the above looped-architecture works have been developed and evaluated exclusively in the context of language modelling. Looped World Models (LoopWM) remain entirely unexplored.

We propose that looped transformers are a promising backbone for world models because they introduce an explicit iterative refinement mechanism while reusing parameters across depth. At a high level, environment dynamics can often be viewed as repeated application of a shared transition law, which motivates modelling a single-step transition through repeated application of a shared latent update operator. This correspondence is conceptual rather than exact, as the inner loop is not meant to represent physical time directly but to perform iterative refinement of a latent transition estimate. To improve the numerical stability of this recurrent computation, we adopt a spectrally constrained state-retention parameterisation inspired by looped architectures. This construction ensures that the linear retention component remains contractive, which helps keep recurrent latent updates bounded

- as the number of inner-loop iterations increases. Structurally, environment dynamics are themselves an iterative process: a state st evolves to st+1 through the repeated application of (approximately) stationary physical laws. The looped transformer’s computation graph, where a shared function fθ is applied recurrently to a latent state h,

ht+1 = Ah¯ t + B¯ e + R¯(ht,e), (1) with A¯ governing state retention, B¯ controlling input injection, and R¯ subsuming the transformer nonlinearities (Prairie et al., 2026) is directly isomorphic to this dynamics structure. Stability is guaranteed by parameterizing the continuous-time matrix as A := diag(−exp(a)) with learnable a, and discretizing via zero-order hold,

##### A¯ = exp(∆A), (2)

which constrains all eigenvalues of A¯ to the interval (0,1), ensuring bounded residual dynamics regardless of rollout length (Prairie et al., 2026).

Practically, the parameter efficiency of looped architectures is uniquely valuable for world models, because long-horizon rollouts require executing the dynamics model hundreds or thousands of times in sequence; a model that achieves the predictive quality of a much larger network with a fraction of the parameters yields compounding savings across every rollout step. Furthermore, the adaptivedepth property of looped models, allocating more iterations to complex transitions (e.g., collisions, contact events) and fewer to simple dynamics (e.g., free flight), maps directly onto the non-uniform computational demands of physical simulation. In the most favourable cases, where simple state transitions require only a single loop iteration compared to the full forward pass of a conventional fixed-depth model, this adaptive mechanism can substantially reduce average inference cost relative to a fixed-depth baseline. The magnitude of this reduction depends on the distribution of transition difficulty, the minimum useful loop depth, and the overhead of the exit mechanism.

In this work, we introduce Looped World Models (LoopWM), the first looped transformer architectures for environment simulation and dynamics prediction. Our approach combines a parametershared recurrent transformer block with spectrally-constrained residual dynamics, enabling provably stable state transitions across arbitrary rollout lengths. We demonstrate that Looped World Models achieve competitive or superior predictive accuracy to existing world model architectures while using significantly fewer parameters, maintain stable rollouts over substantially longer horizons, and support test-time adaptive computation that automatically matches computational depth to transition complexity. We also integrate residual connections to improve model performance. Our results establish iterative latent depth as a previously unexplored and highly effective scaling axis for world models, orthogonal to both model size and training data.

- 2 RELATED WORK

- 2.1 WORLD MODELS FOR REINFORCEMENT LEARNING AND EMBODIED AI

The idea of learning an internal model of environment dynamics dates back to early work on mental simulation and forward models in cognitive science and control theory. In deep reinforcement learning, Ha & Schmidhuber (2018) proposed learning a compressed spatial and temporal representation

of the environment using a variational autoencoder and an RNN, training a compact policy entirely within the learned “dream.” PlaNet (Hafner et al., 2019) formalised this via a latent dynamics model (RSSM) that plans directly in latent space from pixel observations. SimPLe (Łukasz Kaiser et al., 2020) demonstrated model-based Atari play by training a video-prediction model as a learned simulator. MuZero (Schrittwieser et al., 2020) showed that a learned dynamics model with Monte-Carlo tree search can master board games and Atari without access to the ground-truth rules.

The Dreamer family (Hafner et al., 2020; 2021; 2025) progressively refined the RSSM-based world model, culminating in DreamerV3 (Hafner et al., 2025). They achieve human-level performance across over 150 diverse tasks with a single set of hyperparameters. Transformer-based world models subsequently emerged: IRIS (Micheli et al., 2023) replaced the recurrent backbone with an autoregressive transformer over discrete tokens; TransDreamer (Chen et al., 2022) introduced a Transformer State-Space Model for memory-demanding tasks; ∆-IRIS (Micheli et al., 2024) improved tokenization efficiency via context-aware delta encoding; DIAMOND (Alonso et al., 2024) leveraged diffusion models to produce visually faithful world simulations; and EMERALD (Burchi & Timofte, 2025) achieved state-of-the-art Crafter performance using masked generative transformers over spatial latent states.

At a larger scale, video generation models have been cast as world simulators. OpenAI’s Sora (OpenAI, 2024) demonstrated long-form video generation with emergent 3D consistency, while Genie (Bruce et al., 2024) and Genie 3 (Google DeepMind, 2025) showed that text-conditioned generative models can produce interactive, explorable environments. Several surveys chart the rapid expansion of world models into autonomous driving (Feng et al., 2025; Guan et al., 2024), embodied AI (Li et al., 2025b), and video generation (Dewi Puspitasari et al., 2024; Wang et al., 2026).

A persistent challenge across all these approaches is compounding prediction error: small inaccuracies at each rollout step accumulate exponentially over long horizons, degrading trajectory fidelity (Xiao et al., 2020; Talvitie, 2017; Luo et al., 2022). Various mitigation strategies have been proposed, including short-horizon re-planning, self-correcting models (Talvitie, 2017), and physics-informed architectures (Li et al., 2025a; Wang et al., 2025), yet the fundamental tension between computational depth and rollout stability remains unresolved by existing architectures.

- 2.2 LOOPED AND RECURRENT-DEPTH TRANSFORMER ARCHITECTURES

Looped transformers reuse a shared set of transformer blocks across depth, decoupling effective computation from parameter count. The Universal Transformer (Dehghani et al., 2019) first proposed this idea, combining weight sharing with Adaptive Computation Time (ACT) (Graves, 2016) for input-dependent halting. ALBERT (Lan et al., 2020) demonstrated the practical viability of full cross-layer parameter sharing in BERT-scale models.

Theoretical analyses subsequently established the computational power of looped transformers. Giannou et al. (2023) proved that looped transformers can simulate arbitrary programs, functioning

- as programmable computers with constant parameter count. Yang et al. (2023) showed that looped transformers match standard transformer performance on in-context learning while using less than 10% of the parameters. Fan et al. (2024) demonstrated significant length generalisation improvements through adaptive loop counts. Saunshi et al. (2025) provided both theoretical and empirical evidence that looped models implicitly generate “latent thoughts,” enabling reasoning beyond their apparent depth. At a practical scale, Ouro (Zhu et al., 2025) trained looped language models (LoopLMs) through the full modern LLM pipeline with pre-training, instruction tuning, and RLHF, achieving 2–3× parameter efficiency with entropy-regularised adaptive computation. Geiping et al.

(2025) demonstrated that recurrent-depth models (RDMs) scale test-time compute by increasing loop count at inference, following predictable quality improvements. Pappone et al. (2025) analysed the geometry of latent dynamics in recurrent-depth transformers, identifying two-scale structure with fast intra-loop and slow inter-token dynamics. LoopFormer (Jeddi et al., 2026) introduced elasticdepth training with shortcut modulation for budget-conditioned reasoning. Mixture-of-Recursions (Bae et al., 2025) proposed per-token dynamic depth allocation within a single recursive framework. MoEUT (Csord´as et al., 2024) combined mixture-of-experts with universal transformers to balance specialisation and sharing.

- 2.3 ADAPTIVE COMPUTATION AND EARLY EXIT

Allocating variable computation to inputs of differing complexity has been studied across multiple paradigms. Graves (2016) introduced Adaptive Computation Time for RNNs, allowing per-step halting decisions. The early exit literature (Teerapittayanon et al., 2017; Bolukbasi et al., 2017; Jyoti Bajpai & Hanawal, 2025) enables inference to terminate at intermediate layers when confidence is sufficient. In the context of looped transformers, adaptive depth takes a particularly natural form: the model can halt after any number of loop iterations. Ouro (Zhu et al., 2025) introduced entropy-regularised early exit, where a token exits the loop when its prediction entropy drops below a learned threshold. Geiping et al. (2025) trained with stochastic depth sampling (Poisson-distributed loop counts) to induce robustness to variable test-time depth. LoopFormer (Jeddi et al., 2026) conditioned on a continuous “time budget” during training, enabling fine-grained compute allocation

- at inference. Pappone et al. (2025) proposed acceleration-based exit rules using second-order differences of hidden states. For world models specifically, adaptive computation is highly attractive: simple state transitions (e.g., free flight, static scenes) demand minimal processing, while complex events (e.g., multi-body collisions, contact dynamics) require deeper iterative refinement. To the best of our knowledge, no prior work has proposed adaptive-depth looped architectures with world modelling.

- 3 LOOPED WORLD MODEL

We present Looped World Models, a latent dynamics architecture that combines the iterative computation of looped transformers with the action-conditioned state prediction required for world modelling. Our design follows three principles: (i) structural alignment between the model’s computation graph and the iterative nature of physical dynamics, (ii) provable stability of latent state transitions across arbitrary rollout lengths, and (iii) adaptive computational depth that matches the complexity of each transition. We describe the overall architecture (§3.1), the stabilised looped dynamics core (§3.2), the training objective (§3.3), and the adaptive early-exit mechanism for inference (§3.4).

- 3.1 OVERALL ARCHITECTURE

At each environment time step k, the agent receives an observation ok ∈ O and selects an action ak ∈ A. The goal of the world model is to predict the next latent state, from which future observations, rewards, and termination signals can be decoded. Our architecture consists of four modules:

Observation Encoder Eϕ. A convolutional (or vision-transformer-based) encoder maps the raw observation ok into a latent embedding ek = Eϕ(ok) ∈ Rd.

Action Embedder Aψ. The action ak is projected into the same latent space via a learned embedding uk = Aψ(ak) ∈ Rd.

Looped Dynamics Core Lθ. This is the central contribution of our architecture. The dynamics core takes the previous latent state hk−1, the current observation embedding ek, and the action embedding uk, and produces the next latent state hk through T iterations of a parameter-shared transformer block with spectrally-constrained residual dynamics. We describe this module in detail in §3.2.

Prediction Heads Dξ. A set of lightweight MLPs decode the latent state hk into: (i) a reconstructed observation oˆk+1 or its latent target, (ii) a predicted reward rˆk, and (iii) a predicted continuation flag cˆk. These heads follow the standard design of prior latent world models (Hafner et al., 2020; 2021; 2025).

The full forward pass at environment step k proceeds as:

##### ek = Eϕ(ok), uk = Aψ(ak), hk = Lθ(hk−1, ek, uk), (ˆok+1, rˆk, cˆk) = Dξ(hk). (3)

During imagination-based training of the policy (as in Dreamer (Hafner et al., 2020)), the encoder is bypassed: the dynamics core autoregressively rolls out latent trajectories using only actions sampled

from the policy network, i.e., hk+1 = Lθ(hk, 0, uk), where observation injection is omitted or replaced by the model’s own prediction.

- 3.2 LOOPED DYNAMICS CORE WITH SPECTRAL STABILITY

The dynamics core is the heart of our architecture. Following the prelude recurrent coda design (Geiping et al., 2025; Prairie et al., 2026; Zeitoun et al., 2026), we partition the dynamics core into three blocks:

Prelude P. A small stack of LP transformer layers processes the concatenation of the previous latent state, the observation embedding, and the action embedding to produce the conditioning signal:

e = LN(P([hk−1; ek; uk])) ∈ Rd, (4) where LN(·) denotes layer normalization. The normalisation of e follows the Parcae design (Prairie et al., 2026) and prevents input magnitude from inducing late-stage loss spikes.

Recurrent Block R. A stack of LR transformer layers with shared parameters is applied iteratively for T loops. The hidden state is initialised as h(0) ∼ N(0,σ2I) (or, for temporal rollouts, from the previous time step’s final hidden state). At each loop iteration t = 0,1,...,T−1, the update rule is:

##### h(t+1) = Ah¯ (t) + B¯ e + R¯(h(t), e), (5)

where A¯ ∈ Rd×d is the state-retention matrix controlling how much of the previous hidden state is preserved, B¯ ∈ Rd×d is the input-injection matrix controlling the influence of the conditioning signal e, and R¯ subsumes the nonlinear transformer operations (multi-head attention and feed-forward layers) applied to the residual stream. The key distinction from conventional fixed-depth transformers is that the parameters of R are shared across all T iterations, making the computational depth independent of the parameter count.

Spectral Stability Constraint. To guarantee that the latent state does not explode regardless of the number of loop iterations T (which is critical for long-horizon rollouts in world modelling), we constrain the spectral norm of A¯ to be strictly less than 1. Following Parcae (Prairie et al., 2026), we parameterize A¯ through discretization of a continuous-time negative diagonal matrix:

A := diag(−exp(a)), a ∈ Rd (learnable), (6) A¯ = exp(∆ · A), ∆ ∈ Rd>0 (learnable). (7)

Since A has strictly negative diagonal entries, ∆ · A has strictly negative entries, and exp(·) maps these to the interval (0,1). Consequently, A¯ is a diagonal matrix with all entries in (0,1), guaranteeing ρ(A¯) < 1. This constraint holds by construction throughout training; no gradient clipping, post-hoc normalisation, or sensitive hyperparameter tuning is required.

The input-injection matrix is similarly discretised as B¯ = ∆·B with unconstrained B, but we apply layer normalisation to e (Eq. 4) to bound the injected signal’s magnitude.

Coda C. A final stack of LC transformer layers (with separate, non-shared parameters) processes the terminal hidden state h(T) through a learned projection:

##### hk = C(C h(T)), (8) where C ∈ Rd

c×d optionally adapts the embedding dimension. The output hk is then passed to the prediction heads and carried forward as the initial state for the next environment time step.

Cross-Timestep State Propagation. A distinctive property of our architecture is that the terminal hidden state h(T) from environment step k can serve as the initialization h(0) for step k+1, enabling a dual-loop structure: the inner loop (iterations of R) refines the latent state within a single transition, while the outer loop (sequential environment steps) propagates information across time. The spectral constraint on A¯ ensures that both loops remain bounded, encouraging continuity while keeping propagated hidden states numerically well behaved.

- 3.3 TRAINING OBJECTIVE

Variable-Depth Training. We train with stochastic loop depth. At each training step, the loop count T is sampled from a Poisson distribution with learnable mean µrec:

T ∼ Poisson(µrec). (9) We sample T independently per sequence within each micro-batch, rather than per micro-batch as in prior work (Geiping et al., 2025). This reduces variance in the training objective and empirically eliminates most loss spikes.

World Model Loss. The overall training objective combines observation prediction, reward prediction, and continuation prediction:

Lwm = ET∼Poisson(µ

rec)

 

K

k=1

 Lobs(ok+1, oˆk+1)

observation loss

+λr Lrew(rk, rˆk)

reward loss

+λc Lcont(ck, cˆk)

continuation loss

 

 ,

(10)

- where K is the sequence length, λr and λc are balancing coefficients, and the specific form of Lobs depends on the observation space (e.g., MSE for continuous states, cross-entropy for discrete tokens). Backpropagation through the loop iterations is truncated at µbwd = ⌈µrec/2⌉ steps to limit memory cost.

Entropy-Regularised Adaptive Depth. When adaptive early exit is enabled (see §3.4), we augment the loss with an entropy-regularisation term that prevents the exit gate from collapsing to trivial solutions (always exiting at the first iteration or never exiting). The regularisation takes the form:

Lent = −α E

T

t=1

H g(t) , (11)

where g(t) ∈ [0,1] is the exit probability at loop iteration t, H(·) denotes binary entropy, and α is a regularization coefficient. The total training loss is L = Lwm + Lent.

- 3.4 ADAPTIVE EARLY EXIT FOR INFERENCE

At inference time, the looped dynamics core can adaptively terminate the inner loop early for transitions that converge quickly, and allocate additional iterations to complex transitions. We implement this via a lightweight exit gate, a single-layer MLP followed by a sigmoid:

g(t) = σ wg⊤ h(t) + bg , (12)

where wg ∈ Rd and bg ∈ R are learned parameters. At each loop iteration t, if g(t) exceeds a threshold τ, the loop terminates and h(t) is used as the final hidden state. This mechanism is complementary to the convergence-based exit criteria studied by Pappone et al. (2025), which halt when the second-order difference ∥h(t) − 2h(t−1) + h(t−2)∥ falls below a threshold.

In the world-modelling setting, adaptive exit yields particularly large savings. Consider a 100-layer fixed-depth baseline: for a simple free-flight trajectory segment, our model may exit after a single loop of LR layers (e.g., 4 layers), reducing inference FLOPs by a factor of ∼25× for that step. Over a long rollout containing many simple transitions interspersed with occasional complex events, the aggregate FLOPs reduction can reach up to two orders of magnitude compared to a fixed-depth model of equivalent quality.

The maximum loop count Tmax at inference time can also exceed the training-time mean µrec, enabling test-time compute scaling: the model produces progressively refined predictions as more iterations are allocated.

- 3.5 DEFERRED DECODING: ACTION-CONDITIONED LATENT ROLLOUT

- 3.5.1 MOTIVATION

In standard world models (Hafner et al., 2020; 2021; 2025), the prediction heads Dξ are applied

- at every environment step k to produce intermediate observation reconstructions oˆk+1, reward pre-

dictions rˆk, and continuation signals cˆk. This per-step decoding introduces two inefficiencies: (i) it forces the latent state to allocate representational capacity to pixel-level reconstruction at every intermediate step, even when only the final prediction matters for planning; (ii) it prevents the dynamics core from performing uninterrupted latent reasoning across a multi-step action sequence.

Recent work in language modelling has demonstrated that deferring decoding to the end of a latent reasoning process—allowing the model to encode, think, then decode—substantially improves reasoning quality (Koishekenov et al., 2026; Geiping et al., 2025; Saunshi et al., 2025). MuZero (Schrittwieser et al., 2020) similarly operates entirely in latent space without observation reconstruction, predicting only value, reward, and policy. Dreamer’s own “imagination” rollouts (Hafner et al., 2020) propagate latent states without re-encoding real observations, yet still apply reward and value heads at each step.

We propose Deferred Decoding, a modification to the Looped World Model that eliminates all intermediate observation decoding during multi-step rollouts. Given a sequence of ground-truth or planned actions, the model injects each action into the looped dynamics core and advances the latent state purely in the continuous hidden space. Observation, reward, and continuation predictions are produced only at the final step, reducing computation and encouraging the latent trajectory to encode temporally extended, action-relevant structure rather than per-step visual detail.

- 3.5.2 FORMULATION

Consider a planning or evaluation horizon of K steps. Let h0 be the initial latent state (obtained from encoding a real observation o0 through the encoder Eϕ and the prelude block of the looped dynamics core), and let (a0,a1,...,aK−1) be a sequence of actions.

Standard per-step decoding (baseline). At each step k = 0,1,...,K − 1, the baseline model performs:

uk = Aψ(ak), (13)

hk+1 = Lθ(hk,uk), (14) (ˆok+1, rˆk, cˆk) = Dξ(hk+1), (15)

- where Lθ denotes the full looped dynamics core (prelude, T-step recurrent block, coda) described in Section 3.2. This yields K sets of decoded predictions.

Deferred decoding. We replace the per-step decoding with a decode-free latent rollout followed by a single terminal decoding:

uk = Aψ(ak), k = 0,1,...,K − 1, (16)

hk+1 = Lcoreθ (hk,uk), k = 0,1,...,K − 1, (17) (ˆoK, rˆK, cˆK) = Dξ(hK). (18)

The key difference is that Eqs. equation 16, equation 17 are applied K times without invoking Dξ, and the decoder is called exactly once at step K. Between steps, the model ingests a new action

embedding uk and advances the latent state through the looped recurrent block, but produces no intermediate observation, reward, or continuation output.

Interaction between inner and outer loops. Recall from Section 3.2 that each invocation of Lcoreθ itself involves T inner-loop iterations of the shared transformer block. With deferred decoding, the

overall computation becomes a nested loop:

- • Outer loop (action steps): k = 0,...,K − 1. At each step, the action uk is injected.
- • Inner loop (latent refinement): t = 0,...,T − 1. Within each action step, the recur-

rent block refines h via h(t+1) = Ah¯ (t) + B¯ [uk; hk] + R¯(h(t),uk) with spectral-normconstrained A¯.

The total effective depth is K × T shared-parameter transformer applications, but only one forward pass through the decoder.

- 3.5.3 TRAINING OBJECTIVE FOR DEFERRED DECODING

Training the deferred-decoding variant requires the model to maintain accurate latent representations across K action-conditioned transitions without intermediate reconstruction supervision. We define a terminal prediction loss and a latent trajectory regularizer.

Terminal prediction loss. Given a training trajectory (o0,a0,o1,a1,...,oK) where all intermediate actions are ground-truth, the model encodes o0 to obtain h0, performs K latent transitions via Eqs. equation 16, equation 17, then decodes hK:

Lterminal = λo ℓobs(ˆoK,oK) + λr ℓrew(ˆrK,rK−1) + λc ℓcont(ˆcK,cK−1), (19)

where ℓobs may be a reconstruction loss (MSE, perceptual loss) or, in the decoder-free setting, a next-embedding alignment loss analogous to NE-Dreamer (Bredis et al., 2026).

Latent trajectory regularizer. Without intermediate decoding, the latent states at steps 1,...,K − 1 are unsupervised and could drift into regions that are spectrally stable yet semantically meaningless. We introduce two lightweight constraints:

- 1. Latent consistency loss. We encode each intermediate ground-truth observation ok (k =

1,...,K−1) with the frozen encoder Eϕ to obtain reference embeddings e⋆k = sg(Eϕ(ok)), then align:

Lconsist =

1 K − 1

K−1

k=1

∥gω(hk) − e⋆k ∥22 , (20)

where gω is a lightweight projection head and sg(·) denotes stop-gradient. This loss provides soft guidance without requiring a full decoder at each step, analogous to the latent overshooting technique in PlaNet (Hafner et al., 2019).

- 2. Spectral contraction budget. The spectral-norm constraint on A¯ (Section 3.2) already ensures bounded latent evolution per inner loop. Over K outer steps, we additionally monitor the cumulative contraction:

∥hK − h0 ∥2 ≤

K−1

k=0

∥hk+1 − hk ∥2 ≤ K · Cmax, (21)

where Cmax is a soft upper bound enforced as a penalty term. This prevents latent explosion over long deferred horizons while still permitting meaningful state changes induced by actions.

The full training objective for the deferred-decoding variant is:

LDD = Lterminal + α Lconsist + β max 0, k ∥hk+1 − hk∥2 − K · Cmax , (22) where α and β are balancing coefficients.

- 3.5.4 CURRICULUM OVER DEFERRAL HORIZON K

Training directly with a large K is unstable because gradients must back-propagate through K × T shared-parameter applications. We adopt a progressive horizon curriculum: training begins with K = 1 (equivalent to standard per-step decoding) and gradually increases K during training according to a schedule K(step) = min(Kmax, 1 + ⌊step/∆⌋), where ∆ is the number of training steps between increments. This allows the latent dynamics to first learn accurate single-step transitions before being challenged with longer decode-free rollouts.

- 3.5.5 INFERENCE MODES Deferred decoding naturally supports two inference modes:

Planning mode. Given a candidate action sequence (a0,...,aK−1) from a planner (e.g., CEM, MPPI), the model performs a single decode-free rollout and evaluates only the terminal state hK. This reduces decoder invocations from K to 1, saving approximately (K−1)×cost(Dξ) FLOPs per candidate sequence. When combined with adaptive early exit within each inner loop (Section 3.4), the total FLOP reduction can reach up to two orders of magnitude for long-horizon planning with simple transitions.

Monitoring mode. When intermediate state inspection is needed (e.g., for safety-critical applications), the lightweight projection head gω can be applied at any step k to produce a low-dimensional state summary e˜k = gω(hk) without invoking the full decoder. The full decoder Dξ remains available as an optional diagnostic tool but is not required for the planning loop.

- 3.5.6 RELATIONSHIP TO PRIOR WORK

- Table 1 summarizes the key distinctions:

Table 1: Comparison of intermediate decoding strategies across world-model architectures.

Method Latent dynamics Intermediate decode Action injection Looped depth

Dreamer (Hafner et al., 2020) RSSM reward + value at each step per step – MuZero (Schrittwieser et al., 2020) learned MLP policy + value + reward per step – PlaNet (Hafner et al., 2019) RSSM reconstruction at each step per step – ETD (Koishekenov et al., 2026) looped layers decode only at end – (language) ✓ NE-Dreamer (Bredis et al., 2026) RSSM embedding alignment per step –

LoopWM-DD (ours) looped transformer decode only at step K per step in latent ✓

Dreamer’s imagination rollout already avoids re-encoding real observations but still applies reward and value heads at every imagined step (Hafner et al., 2020). MuZero dispenses with observation reconstruction entirely but uses a non-looped, fixed-depth dynamics function (Schrittwieser et al., 2020). ETD (Koishekenov et al., 2026) demonstrates the encode-think-decode paradigm for language reasoning with looped layers, but does not handle action-conditioned state transitions or environment simulation. Our deferred decoding unifies these insights: it applies the looped transformer’s iterative refinement at each action step in latent space (inheriting the parameter efficiency and spectral stability of the LoopWM) while deferring all observation-space computation to the terminal step, yielding a clean separation between latent dynamics reasoning (inner + outer loops) and observation grounding (single terminal decode).

4 RESULTS

- 4.1 MAIN RESULTS ON SCIENCEWORLD

Table 2 presents the results on the ScienceWorld dataset of our models against claude-opus-4-6-max. From the results, it is clear that our model surpasses the strong claude-opus-4-6-max. In the most extreme cases, it improves the scores on Lifespan from 0% to 100%, denoting the underlying strong capacity of our model. On average, our model shows a promising capability, clearly surpassing the baseline by 21.2% on EM, and clearly on other metrics. Further, we note that our model is a small AI model with around 1B parameters, which is much smaller than those strong closed-source API models such as claude-opus-4-6-max by more than 100x. This suggests our proposed model has a promising parameter efficiency to be deployed on downstream applications. Note that Table 3 presents more baselines, which lead to the same conclusions.

We also note that qwen-3.5-flash and gemini-3-flash-preview seem to be clearly worse than other baselines and our models across most metrics. This is reasonable as they are considered smaller than the other baseline models. Our proposed models are still competitive and much stronger than them across the metrics.

- 4.2 MAIN RESULTS ON ALFWORLD

Table 4 presents evaluation results on the AlfWorld dataset. On this dataset, we see that the trend can still be promising, as our proposed model gives a promising overall result, given the fact that it is pretty small in terms of model size, with around 1B parameters. Notably, it gives the best result on the BLEU metrics (Papineni et al., 2002) among four models, and ranks in second place on EM and Token F1. Further, by inspecting the detailed action categories, we found that our model seems to

- Table 2: Comparison of our proposed looped world model against claude-opus-4-6-max (Anthropic,

2026) on ScienceWorld dataset (Wang et al., 2022) world modelling task. The accuracy is calculated based on feeding consecutive five actions, and obtaining the final scores on world modelling. Note that our model is a model with about 1B model parameters. Refer to Table 3 for more baselines.

Task Type EM F1 BLEU Entity Task Type EM F1 BLEU Entity Looped World Model (Ours)

Boil 66.7% 79.0% 75.3% 77.5% Chemistry 44.4% 64.4% 54.2% 57.9% Conductivity 87.0% 89.0% 87.8% 87.9% Find 76.9% 90.4% 82.7% 85.8% Freeze 25.0% 59.7% 31.2% 54.8% Genetics 78.3% 80.2% 78.9% 79.8% Grow 73.8% 80.0% 75.5% 79.8% Incline 59.3% 95.3% 90.4% 93.4% LifeStages 0.0% 18.3% 6.1% 10.2% Lifespan 100.0% 100.0% 100.0% 100.0% Melt 73.0% 91.9% 85.7% 91.6% Power 57.1% 63.9% 60.8% 61.5% StateChange 80.0% 83.1% 80.0% 80.0% Thermometer 83.3% 85.3% 83.3% 83.3%

Overall: EM: 68.4%; Token F1: 85.3%; BLEU-4: 80.7%; Entity: 83.9% claude-opus-4-6-max

Boil 22.2% 33.3% 30.2% 32.3% Chemistry 44.4% 59.8% 46.3% 59.2% Conductivity 47.8% 67.2% 53.1% 72.7% Find 69.2% 83.8% 78.9% 84.6% Freeze 12.5% 33.6% 21.6% 37.3% Genetics 59.2% 71.8% 65.7% 71.6% Grow 70.8% 81.6% 76.1% 80.7% Incline 34.0% 86.5% 76.2% 83.8% LifeStages 0.0% 10.6% 6.0% 6.0% Lifespan 0.0% 61.4% 0.0% 58.3% Melt 36.5% 52.4% 46.3% 53.4% Power 42.9% 47.3% 45.3% 45.8% StateChange 40.0% 65.5% 44.6% 73.3% Thermometer 83.3% 98.1% 93.5% 97.2%

Overall: EM: 47.2%; Token F1: 72.8%; BLEU-4: 64.4%; Entity: 72.3%

have low entity scores, and it seems valid for most action categories. Such an error analysis indicates that future optimization can focus on the entity scores to further enhance the model.

- 4.3 DEEP ANALYSIS ON DEFERRED DECODING

Across the tables, we conclude that the deferred decoding is useful, and it tends to be more useful when the rollouts are accumulated.

- 5 CONCLUSIONS

We have presented Looped World Models, the first application of looped transformer architectures to world modelling. Our approach addresses a central tension in current world models: generating faithful long-horizon simulations demands deep computation, yet deeper models incur prohibitive deployment costs and are susceptible to compounding rollout errors. By iteratively refining latent environment states through a parameter-shared transformer block with stabilised residual dynamics, LoopWM structurally mirrors the recurrence inherent in physical systems while maintaining a compact parameter footprint. Empirically, LoopWM achieve up to 100× parameter efficiency over conventional approaches without sacrificing prediction quality. Theoretically, we show that spectralnorm constraints on state transitions yield provably stable rollouts, providing formal guarantees that are absent in standard autoregressive world models. Furthermore, our adaptive computation mecha-

- Table 3: Baseline results on ScienceWorld dataset (Wang et al., 2022) world modelling task. The accuracy is calculated based on feeding consecutive five actions, and obtaining the final scores on world modelling. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Task Type EM F1 BLEU Entity qwen-3.5-flash (Qwen Team, 2026)

Boil 0.0% 44.6% 15.1% 39.6% Chemistry 3.7% 28.8% 4.1% 41.3% Conductivity 0.0% 28.2% 0.8% 49.2% Find 0.0% 25.1% 0.0% 51.0% Freeze 0.0% 29.8% 9.8% 44.0% Genetics 7.5% 30.5% 11.5% 53.2% Grow 4.6% 28.3% 7.2% 58.3% Incline 20.7% 81.3% 63.9% 84.0% LifeStages 0.0% 24.8% 0.0% 10.2% Lifespan 0.0% 12.1% 0.0% 25.0% Melt 9.5% 44.1% 24.6% 64.7% Power 0.0% 27.5% 1.4% 38.9% StateChange 0.0% 25.1% 5.7% 70.0% Thermometer 0.0% 32.7% 2.1% 56.9%

Overall: EM: 10.0%; Token F1: 46.9%; BLEU-4: 26.7%; Entity: 63.0% gemini-3-flash-preview-thinking (Gemini Team, Google DeepMind, 2025)

Boil 22.2% 61.5% 41.9% 64.1% Chemistry 22.2% 54.8% 27.6% 57.9% Conductivity 17.4% 55.4% 21.9% 60.6% Find 15.4% 65.2% 40.3% 80.7% Freeze 12.5% 35.0% 22.2% 37.5% Genetics 41.7% 65.6% 48.5% 71.1% Grow 47.7% 72.6% 55.4% 75.8% Incline 32.7% 88.5% 76.0% 88.6% LifeStages 0.0% 15.1% 6.0% 8.1% Lifespan 0.0% 38.8% 0.0% 58.3% Melt 7.9% 47.9% 29.6% 62.5% Power 14.3% 42.7% 16.7% 45.9% StateChange 20.0% 57.8% 23.9% 70.0% Thermometer 33.3% 68.8% 45.3% 86.1%

Overall: EM: 30.8%; Token F1: 68.9%; BLEU-4: 51.1%; Entity: 73.8%

nism automatically scales the effective depth of the model to match the complexity of each prediction step, allocating more refinement iterations to dynamically challenging transitions and fewer to predictable ones. Beyond the specific results reported here, we believe this work identifies iterative latent depth as a new scaling axis for world simulation, one that is orthogonal to the conventional axes of model size and data volume. We hope that this perspective opens new directions for building world models that are simultaneously more capable, more efficient, and more stable over extended horizons.

- 6 BROADER IMPACTS

While the present paper already provides strong evidence for the effectiveness of LoopWM, the current manuscript is intentionally selective in disclosure scope. In this version, our goal is to establish the core architectural thesis that looped latent refinement, deferred decoding, and stabilized dynamics together define a viable and promising design space for world modelling, rather than to exhaustively present every supporting result we have already obtained.

First, the current paper already demonstrates the value of iterative latent computation through deferred decoding, which gives concrete evidence that preserving and refining latent computation across rollout steps is beneficial. We view this as a direct and meaningful manifestation of the looped design. At the same time, it represents only one visible entry point into a broader body of

- Table 4: Comparison of our proposed looped world model against claude-opus-4-6-max (Anthropic,

2026) and other baselines on AlfWorld dataset (Cˆot´e et al., 2018) world modelling task. The accuracy is calculated based on feeding consecutive five actions, and obtaining the final scores on world modelling. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Task Type EM F1 BLEU Entity Looped World Model (Ours)

- clean 60.4% 81.7% 75.0% 81.3% cool 50.0% 81.7% 72.6% 78.5%

- heat 55.0% 81.8% 76.6% 81.2% look 60.5% 78.9% 73.1% 82.0% pick 46.7% 79.7% 69.1% 81.5% - - - - -

Overall: EM: 51.6%; Token F1: 80.4%; BLEU-4: 71.6%; Entity: 81.1% claude-opus-4-6-max

clean 57.3% 73.8% 68.9% 77.4% cool 50.0% 73.0% 68.2% 72.8% heat 52.5% 67.6% 64.4% 74.1% look 60.5% 73.2% 68.9% 78.6% pick 51.0% 72.8% 65.7% 78.1% - - - - -

Overall: EM: 53.0%; Token F1: 72.6%; BLEU-4: 66.8%; Entity: 77.0% qwen-3.5-flash (Qwen Team, 2026)

clean 36.5% 71.9% 55.6% 90.1% cool 27.3% 72.0% 52.6% 85.1% heat 27.5% 70.1% 49.5% 91.2% look 27.9% 66.2% 46.3% 92.8% pick 21.2% 64.1% 43.5% 87.5% - - - - -

Overall: EM: 26.0%; Token F1: 67.3%; BLEU-4: 47.7%; Entity: 88.4% gemini-3-flash-preview-thinking (Gemini Team, Google DeepMind, 2025) clean 61.5% 88.2% 79.9% 90.5% cool 54.5% 88.1% 77.0% 88.3%

- heat 55.0% 81.9% 73.3% 90.6% look 55.8% 86.6% 74.4% 97.7% pick 42.7% 80.2% 65.1% 89.3% - - - - -

Overall: EM: 50.0%; Token F1: 83.5%; BLEU-4: 71.0%; Entity: 90.2%

evidence supporting the effectiveness of looping, and a more explicit decomposition of these gains can be disclosed in the future.

Second, although the present manuscript emphasizes the principal task domains reported here, our empirical validation is not confined to these settings. We have also verified in continuous visual environments that optimization is feasible and that the training loss is consistently reducible, which supports the practicality of the proposed architecture beyond the environments highlighted in this paper. The main limitation at this stage is therefore not a lack of empirical support, but that the manuscript does not yet fully expose the breadth of validation already completed.

Third, LoopWM is best understood as a distinct point in the broader world model landscape. The current paper makes clear that its emphasis differs from major existing families, including RSSM style latent dynamics models, autoregressive video token world models, and diffusion based world models. A more explicit positioning analysis would further sharpen this distinction and make the contribution even easier to interpret. We therefore see clear value in more directly situating

- Table 5: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on average over all the tasks, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

- Step1 +73.2% +16.4% +47.0% +9.7%
- Step2 +54.5% +21.4% +41.7% +18.0%
- Step3 +103.6% +28.1% +65.0% +19.0%
- Step4 +82.9% +29.0% +55.5% +20.7%
- Step5 +113.8% +22.4% +54.6% +12.8%

- Table 6: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Boil, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours−Baselines)/Baselines∗100%. Note that our model is a model with about 1B model parameters. ‘—‘ represents that the baseline score is 0 and LoopWM score is not zero. ‘0%‘ means that both of them has a score of 0.

#### Task Type EM F1 BLEU Entity

- Step1 +100.0% +3.5% +39.6% -8.0%
- Step2 +50.2% +33.4% +61.3% +54.9%
- Step3 +250.5% +57.5% +136.2% +39.3%
- Step4 +700.9% +120.0% +503.5% +121.0%
- Step5 +500.9% +29.9% +101.9% +20.0%

LoopWM among these families and clarifying the regimes in which iterative latent depth is the most natural scaling axis.

Finally, our current step 1 to step 5 experiments already indicate that iterative latent depth behaves as a meaningful scaling dimension, and we consider this one of the central implications of the work. The remaining limitation is not whether such a scaling trend exists, but that the present paper stops short of providing a more complete scaling law characterization across broader task and compute ranges. Similarly, from an optimization perspective, our experience suggests that training can benefit from curriculum like engineering strategies that progressively unlock the architecture’s capability. We regard this not as a weakness of the method, but as part of the practical recipe for making a new architectural regime reliably trainable at scale.

Overall, the main limitation of the current paper is one of presentation scope rather than conceptual or empirical foundation. The paper establishes the core case for LoopWM, while broader cross family positioning, richer scaling analysis, and more extensive optimization disclosure can further strengthen the story in the future.

- Table 7: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Chemistry, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

Step1 +267.1% +52.4% +197.2% +56.9% Step2 +140.3% +42.2% +120.0% +33.5% Step3 +110.3% +34.1% +92.5% +18.5% Step4 +367.6% +57.0% +224.3% +62.6% Step5 +100.0% +15.0% +78.9% 0.0%

- Table 8: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Conductivity, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours−Baselines)/Baselines∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

- Step1 +78.0% +17.5% +58.8% +2.4%
- Step2 +183.1% +44.5% +190.5% +42.5%
- Step3 +220.7% +57.5% +249.8% +39.0%
- Step4 +183.1% +53.2% +194.8% +48.7%
- Step5 +233.3% +51.9% +218.1% +40.0%

- Table 9: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Find, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours−Baselines)/Baselines∗100%. Note that our model is a model with about 1B model parameters.

#### Task Type EM F1 BLEU Entity

- Step1 +166.2% +45.0% +141.2% +28.4%
- Step2 +79.7% +7.1% +40.6% -8.3%
- Step3 +266.2% +71.0% +253.7% +41.7%
- Step4 +71.6% +25.6% +56.8% +14.6%
- Step5 +399.4% +38.7% +105.2% +6.3%

- Table 10: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Freeze, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Step1 +100.0% -6.2% +63.5% -32.1% Step2 +50.0% +10.1% +62.2% -2.9% Step3 +250.0% +80.9% +112.6% +20.6% Step4 +400.0% +96.7% +303.0% +76.0% Step5 +100.0% +70.6% +40.5% +46.1%

- Table 11: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Genetics, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

Step1 +80.8% +31.9% +68.3% +27.0% Step2 +36.5% +24.5% +33.0% +20.3% Step3 +122.1% +36.3% +101.4% +24.3% Step4 +76.7% +30.5% +54.5% +22.6% Step5 +74.0% +19.5% +52.3% +10.7%

- Table 12: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Grow, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

#### Task Type EM F1 BLEU Entity

Step1 +109.6% +20.9% +89.1% +12.4% Step2 +59.6% +13.9% +46.9% +13.9% Step3 +48.4% +7.6% +41.1% +6.8% Step4 +16.3% -5.6% +5.8% -10.4% Step5 +50.0% +8.4% +34.1% +2.8%

- Table 13: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Incline, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Step1 +24.5% +1.9% +7.5% -0.1% Step2 +7.6% +2.3% +5.5% +2.1% Step3 +42.5% +6.6% +13.9% +3.8% Step4 +40.2% +6.8% +14.8% +3.3% Step5 +85.3% +8.4% +20.5% +6.1%

- Table 14: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Melt, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Step1 +112.6% -5.7% +27.9% -22.9% Step2 +343.2% +105.4% +201.4% +88.0% Step3 +349.6% +86.0% +172.7% +59.4% Step4 +585.3% +220.3% +467.1% +138.3% Step5 +557.7% +79.5% +161.3% +42.9%

- Table 15: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task on the task of Power, compared to gemini-3-flash-preview-thinking. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

#### Task Type EM F1 BLEU Entity

Step1 +499.3% +105.5% +499.3% +61.6% Step2 +99.8% +30.6% +80.9% +17.1% Step3 +299.3% +61.0% +347.5% +24.5% Step4 +66.4% +12.4% +66.4% -9.8% Step5 +299.3% +46.2% +264.1% +34.0%

- Table 16: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on all tasks, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

Step1 +143.5% +32.4% +93.8% +26.3% Step2 +86.4% +33.9% +72.5% +27.8% Step3 +136.1% +42.0% +106.0% +30.2% Step4 +78.1% +35.5% +74.0% +25.1% Step5 +104.8% +32.2% +79.3% +22.1%

- Table 17: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Boil, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Step1 — +38.6% +356.5% +9.2% Step2 — +116.7% +2428.1% +140.3% Step3 — +100.2% +438.1% +55.7% Step4 — +203.1% — +194.7% Step5 +500.9% +57.1% +184.2% +86.7%

- Table 18: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Chemistry, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

#### Task Type EM F1 BLEU Entity

Step1 +450.7% +114.6% +369.8% +103.3% Step2 +500.7% +111.4% +487.7% +86.4% Step3 +600.9% +87.3% +411.0% +38.1% Step4 +250.7% +61.6% +217.6% +43.0% Step5 +300.9% +43.4% +290.6% +15.3%

- Table 19: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Conductivity, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Step1 +100.0% +19.3% +71.3% -2.4% Step2 +240.5% +40.8% +209.7% +26.0% Step3 +166.3% +27.0% +169.3% +1.3% Step4 +143.1% +45.3% +191.0% +23.9% Step5 +233.3% +39.6% +191.7% +10.6%

- Table 20: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Find, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

Step1 +699.2% +75.0% +400.0% +58.9% Step2 +349.4% +38.6% +196.7% -0.3% Step3 +449.4% +105.8% +440.5% +63.1% Step4 +139.7% +54.0% +133.3% +30.0% Step5 +149.4% +50.7% +108.8% +8.1%

- Table 21: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Freeze, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Step1 +100.0% -6.3% +308.7% -31.3% Step2 +200.0% +47.7% +116.6% +33.4% Step3 — +162.2% +750.0% +51.5% Step4 +400.0% +109.2% +219.7% +59.9% Step5 — +63.1% +235.5% +14.9%

- Table 22: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Genetics, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

Step1 +203.5% +47.2% +184.8% +53.5% Step2 +61.6% +23.0% +50.6% +23.2% Step3 +202.9% +44.6% +185.0% +39.9% Step4 +70.8% +25.8% +59.8% +22.6% Step5 +104.4% +28.5% +91.5% +21.1%

- Table 23: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Grow, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

Step1 +450.4% +63.1% +376.8% +45.1% Step2 +168.8% +46.5% +135.0% +34.6% Step3 +257.8% +45.7% +206.3% +31.5% Step4 +87.0% +15.4% +73.1% +0.8% Step5 +152.7% +33.3% +125.4% +20.4%

- Table 24: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Incline, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Step1 +40.2% +5.5% +12.9% +4.1% Step2 -1.0% +5.8% +8.4% +4.5% Step3 +7.4% +8.2% +13.5% +5.0% Step4 +2.4% +9.5% +13.6% +6.4% Step5 +9.8% +7.1% +13.0% +4.8%

- Table 25: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Melt, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Step1 +241.8% +25.0% +132.4% +5.6% Step2 +680.9% +143.0% +418.9% +125.0% Step3 +501.1% +152.9% +368.9% +115.7% Step4 +723.4% +198.4% +555.0% +150.8% Step5 +822.8% +128.0% +365.8% +112.0%

- Table 26: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged on Power, compared to qwen3.5-flash. The relative improvements are calculated using the absolute performance (Ours − Baselines)/Baselines ∗ 100%. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity

Step1 +499.3% +138.0% +350.9% +121.2% Step2 +499.3% +116.3% +439.4% +46.9% Step3 +299.3% +77.9% +300.6% +70.3% Step4 — +103.1% +891.7% +13.0% Step5 +299.3% +76.0% +272.1% +48.2%

- Table 27: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling task averaged, on our model. Note that our model is a model with about 1B model parameters.

#### Task Type EM F1 BLEU Entity

- Step 1 67.2% 78.0% 72.3% 77.9%
- Step 2 68.6% 86.2% 80.9% 86.4%
- Step 3 68.0% 87.5% 82.0% 87.1%
- Step 4 68.4% 87.1% 82.1% 85.6%
- Step 5 68.4% 85.3% 80.7% 83.9%

- Table 28: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 1, on our model. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Boil 44.4% 59.6% 52.5% 54.4% Chemistry 81.5% 85.2% 84.1% 85.2% Conductivity 69.6% 78.5% 71.6% 76.6% Find 61.5% 73.5% 61.5% 76.9% Freeze 25.0% 42.0% 34.0% 42.5% Genetics 78.3% 83.9% 78.6% 85.5% Grow 67.7% 75.7% 67.7% 76.3% Incline 74.7% 91.9% 87.7% 90.9% Melt 27.0% 39.5% 31.6% 38.0% Power 85.7% 97.6% 85.7% 100.0%

Table 29: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 2, on our model. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Boil 66.7% 81.9% 71.6% 88.9% Chemistry 88.9% 92.6% 91.1% 93.2% Conductivity 73.9% 83.5% 82.2% 81.8% Find 69.2% 78.3% 70.3% 70.3% Freeze 37.5% 69.7% 54.0% 66.7% Genetics 80.8% 84.9% 82.2% 85.9% Grow 78.5% 84.4% 79.9% 87.1% Incline 56.7% 93.4% 86.9% 92.0% Melt 49.2% 72.9% 63.3% 73.9% Power 85.7% 95.6% 89.0% 98.0%

- Table 30: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 3, on our model. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Boil 77.8% 97.5% 94.7% 98.1% Chemistry 77.8% 85.4% 80.1% 83.8% Conductivity 69.6% 82.7% 80.8% 80.6% Find 84.6% 96.1% 90.9% 96.2% Freeze 87.5% 98.6% 70.8% 97.9% Genetics 83.3% 86.3% 83.8% 85.0% Grow 66.2% 73.3% 68.3% 75.1% Incline 58.0% 96.6% 91.2% 94.6% Melt 57.1% 88.0% 76.9% 92.3% Power 57.1% 74.7% 72.5% 72.2%

Table 31: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 4, on our model. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Boil 88.9% 98.8% 85.1% 98.1% Chemistry 51.9% 72.7% 61.3% 66.5% Conductivity 73.9% 93.0% 90.2% 92.5% Find 92.3% 94.1% 93.3% 93.4% Freeze 62.5% 95.4% 66.5% 91.7% Genetics 82.5% 84.3% 83.1% 84.6% Grow 66.2% 72.1% 67.7% 71.3% Incline 60.7% 95.8% 90.9% 93.7% Melt 65.1% 91.6% 84.5% 90.3% Power 71.4% 79.0% 71.4% 71.4%

- Table 32: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 5, on our model. Note that our model is a model with about 1B model parameters.

Task Type EM F1 BLEU Entity Boil 66.7% 79.0% 75.3% 77.5% Chemistry 44.4% 64.4% 54.2% 57.9% Conductivity 87.0% 89.0% 87.8% 87.9% Find 76.9% 90.4% 82.7% 85.8% Freeze 25.0% 59.7% 31.2% 54.8% Genetics 78.3% 80.2% 78.9% 79.8% Grow 73.8% 80.0% 75.5% 79.8% Incline 59.3% 95.3% 90.4% 93.4% Melt 73.0% 91.9% 85.7% 91.6% Power 57.1% 63.9% 60.8% 61.5%

- Table 33: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks, on gemini.

Task Type EM F1 BLEU Entity

- Step 1 38.8% 67.0% 49.2% 71.0%
- Step 2 44.4% 71.0% 57.1% 73.2%
- Step 3 33.4% 68.3% 49.7% 73.2%
- Step 4 37.4% 67.5% 52.8% 70.9%
- Step 5 32.0% 69.7% 52.2% 74.4%

- Table 34: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 1 on gemini. Task Type EM F1 BLEU Entity

Boil 22.2% 57.6% 37.6% 59.1% Chemistry 22.2% 55.9% 28.3% 54.3% Conductivity 39.1% 66.8% 45.1% 74.8% Find 23.1% 50.7% 25.5% 59.9% Freeze 12.5% 44.8% 20.8% 62.6% Genetics 43.3% 63.6% 46.7% 67.3% Grow 32.3% 62.6% 35.8% 67.9% Incline 60.0% 90.2% 81.6% 91.0% Melt 12.7% 41.9% 24.7% 49.3% Power 14.3% 47.5% 14.3% 61.9%

- Table 35: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 2 on gemini. Task Type EM F1 BLEU Entity

Boil 44.4% 61.4% 44.4% 57.4% Chemistry 37.0% 65.1% 41.4% 69.8% Conductivity 26.1% 57.8% 28.3% 57.4% Find 38.5% 73.1% 50.0% 76.7% Freeze 25.0% 63.3% 33.3% 68.7% Genetics 59.2% 68.2% 61.8% 71.4% Grow 49.2% 74.1% 54.4% 76.5% Incline 52.7% 91.3% 82.4% 90.1% Melt 11.1% 35.5% 21.0% 39.3% Power 42.9% 73.2% 49.2% 83.7%

Table 36: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 3 on gemini. Task Type EM F1 BLEU Entity

Boil 22.2% 61.9% 40.1% 70.4% Chemistry 37.0% 63.7% 41.6% 70.7% Conductivity 21.7% 52.5% 23.1% 58.0% Find 23.1% 56.2% 25.7% 67.9% Freeze 25.0% 54.5% 33.3% 81.2% Genetics 37.5% 63.3% 41.6% 68.4% Grow 44.6% 68.1% 48.4% 70.3% Incline 40.7% 90.6% 80.1% 91.1% Melt 12.7% 47.3% 28.2% 57.9% Power 14.3% 46.4% 16.2% 58.0%

- Table 37: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 4 on gemini. Task Type EM F1 BLEU Entity

Boil 11.1% 44.9% 14.1% 44.4% Chemistry 11.1% 46.3% 18.9% 40.9% Conductivity 26.1% 60.7% 30.6% 62.2% Find 53.8% 74.9% 59.5% 81.5% Freeze 12.5% 48.5% 16.5% 52.1% Genetics 46.7% 64.6% 53.8% 69.0% Grow 56.9% 76.4% 64.0% 79.6% Incline 43.3% 89.7% 79.2% 90.7% Melt 9.5% 28.6% 14.9% 37.9% Power 42.9% 70.3% 42.9% 79.2%

Table 38: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 5 on gemini. Task Type EM F1 BLEU Entity

- Table 39: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world modelling tasks on qwen.

Boil 11.1% 60.8% 37.3% 64.6% Chemistry 22.2% 56.0% 30.3% 57.9% Conductivity 26.1% 58.6% 27.6% 62.8% Find 15.4% 65.2% 40.3% 80.7% Freeze 12.5% 35.0% 22.2% 37.5% Genetics 45.0% 67.1% 51.8% 72.1% Grow 49.2% 73.8% 56.3% 77.6% Incline 32.0% 87.9% 75.0% 88.0% Melt 11.1% 51.2% 32.8% 64.1% Power 14.3% 43.7% 16.7% 45.9%

#### Task Type EM F1 BLEU Entity

- Step 1 27.6% 58.9% 37.3% 61.7%
- Step 2 36.8% 64.4% 46.9% 67.6%
- Step 3 28.8% 61.6% 39.8% 66.9%
- Step 4 38.4% 64.3% 47.2% 68.4%
- Step 5 33.4% 64.5% 45.0% 68.7%

- Table 40: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 1 on qwen. Task Type EM F1 BLEU Entity

Boil 0.0% 43.0% 11.5% 49.8% Chemistry 14.8% 39.7% 17.9% 41.9% Conductivity 34.8% 65.8% 41.8% 78.5% Find 7.7% 42.0% 12.3% 48.4% Freeze 0.0% 34.3% 8.3% 49.3% Genetics 25.8% 57.0% 27.6% 55.7% Grow 12.3% 46.4% 14.2% 52.6% Incline 53.3% 87.1% 77.7% 87.3% Melt 7.9% 31.6% 13.6% 36.0% Power 14.3% 41.0% 19.0% 45.2%

Table 41: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 2 on qwen. Task Type EM F1 BLEU Entity

Boil 0.0% 37.8% 3.2% 37.0% Chemistry 14.8% 43.8% 15.5% 50.0% Conductivity 21.7% 59.3% 26.6% 64.9% Find 15.4% 56.5% 23.7% 70.5% Freeze 12.5% 47.2% 24.7% 50.0% Genetics 50.0% 69.0% 54.5% 69.7% Grow 29.2% 57.6% 34.0% 64.7% Incline 57.3% 88.3% 80.2% 88.0% Melt 6.3% 30.0% 12.2% 32.8% Power 14.3% 44.2% 16.5% 66.7%

- Table 42: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 3 on qwen. Task Type EM F1 BLEU Entity

Boil 0.0% 48.7% 17.6% 63.0% Chemistry 11.1% 45.6% 16.4% 61.4% Conductivity 26.1% 65.1% 30.0% 79.6% Find 15.4% 46.7% 16.8% 59.0% Freeze 0.0% 37.6% 8.3% 64.6% Genetics 27.5% 59.7% 29.4% 61.1% Grow 18.5% 50.3% 22.3% 57.1% Incline 54.0% 89.3% 80.5% 90.1% Melt 9.5% 34.8% 16.4% 42.8% Power 14.3% 42.0% 18.1% 42.4%

Table 43: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 4 on qwen. Task Type EM F1 BLEU Entity

Boil 0.0% 32.6% 0.0% 33.3% Chemistry 14.8% 45.0% 19.3% 46.5% Conductivity 30.4% 64.0% 31.0% 74.7% Find 38.5% 61.1% 40.0% 71.8% Freeze 12.5% 45.6% 20.8% 57.3% Genetics 48.3% 67.0% 52.0% 69.0% Grow 35.4% 62.5% 39.1% 70.7% Incline 59.3% 87.5% 80.0% 88.1% Melt 7.9% 30.7% 12.9% 36.0% Power 0.0% 38.9% 7.2% 63.1%

- Table 44: The effect of deferred decoding on the ScienceWorld dataset (Wang et al., 2022) world

- modelling tasks, on Step 5 on qwen. Task Type EM F1 BLEU Entity

Boil 11.1% 50.3% 26.5% 41.5% Chemistry 11.1% 44.9% 13.9% 50.2% Conductivity 26.1% 63.8% 30.1% 79.5% Find 30.8% 60.0% 39.6% 79.4% Freeze 0.0% 36.6% 9.3% 47.7% Genetics 38.3% 62.4% 41.2% 65.9% Grow 29.2% 60.0% 33.5% 66.3% Incline 54.0% 89.0% 80.0% 89.1% Melt 7.9% 40.3% 18.4% 43.2% Power 14.3% 36.3% 22.1% 41.5%

[Figure 3]

- Figure 2: Relative increase over Qwen3.7-max on automatic online performance, compared against baselines. Note that the results are obtained via online estimation, with the tasks of danmaku generation. LWM denotes LoopWM.

Human evaluation results on Danmaku Chan

(b) Overall comparison

Baseline VLM

LWM

| |
|---|

| |
|---|

Appropr.

(a) Human evaluation by metric

100

| |20 40 60 80<br><br>|
|---|---|
| | |

91 93

90

86

81

80

72

65

60

56

Score

Human-like 100

Inform.

40

20

0

Engaging ness

Appropriate ness

Informative ness

Humanlikeness

Engaging

- Figure 3: Human evaluation performance with our model, compared against baselines. Note that the results are obtained via online estimation, with the tasks of danmaku generation. LWM denotes LoopWM.

REFERENCES

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and Fran¸cois Fleuret. Diffusion for world modeling: Visual details matter in atari. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https:// openreview.net/forum?id=NadTwTODgC.

Anthropic. System card: Claude opus 4.6. Technical report, Anthropic, February 2026. URL https://www-cdn.anthropic.com/ 14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf.

Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, and Se-Young Yun. Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. arXiv e-prints, art. arXiv:2507.10524, July 2025. doi: 10.48550/arXiv.2507.10524.

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. Curran Associates Inc., Red Hook, NY, USA, 2019.

Tolga Bolukbasi, Joseph Wang, Ofer Dekel, and Venkatesh Saligrama. Adaptive neural networks for efficient inference. In Proceedings of the 34th International Conference on Machine Learning

- Volume 70, ICML’17, pp. 527–536. JMLR.org, 2017.

George Bredis, Nikita Balagansky, Daniil Gavrilov, and Ruslan Rakhimov. Next embedding prediction makes world models stronger. In ICLR 2026 the 2nd Workshop on World Models: Understanding, Modelling and Scaling, 2026. URL https://openreview.net/forum?id= SkAgjqPmhY.

Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge (Jimmy) Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando De Freitas, Satinder Singh, and Tim Rockt¨aschel. Genie: generative interactive environments. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Maxime Burchi and Radu Timofte. Accurate and efficient world modeling with masked latent transformers. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=zNUOZcAUxz.

Chang Chen, Jaesik Yoon, Yi-Fu Wu, and Sungjin Ahn. Transdreamer: Reinforcement learning with transformer world models, 2022. URL https://openreview.net/forum?id= s3K0arSRl4d.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, pp. 6572–6583, Red Hook, NY, USA, 2018. Curran Associates Inc.

Marc-Alexandre Cˆot´e, Akos´ K´ad´ar, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, et al. Textworld: A learning environment for text-based games. In Workshop on Computer Games, pp. 41–75. Springer, 2018.

R´obert Csord´as, Kazuki Irie, J¨urgen Schmidhuber, Christopher Potts, and Christopher D Manning. MoEUT: Mixture-of-experts universal transformers. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=ZxVrkm7Bjl.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=HyzdRiR9Y7.

Fachrina Dewi Puspitasari, Chaoning Zhang, Joseph Cho, Adnan Haider, Noor Ul Eman, Omer Amin, Alexis Mankowski, Muhammad Umair, Jingyao Zheng, Sheng Zheng, Lik-Hang Lee, Caiyan Qin, Tae-Ho Kim, Choong Seon Hong, Yang Yang, and Heng Tao Shen. Sora as a World Model? A Complete Survey on Text-to-Video Generation. arXiv e-prints, art. arXiv:2403.05131, March 2024. doi: 10.48550/arXiv.2403.05131.

Ying Fan, Yilun Du, Kannan Ramchandran, and Kangwook Lee. Looped Transformers for Length Generalization. arXiv e-prints, art. arXiv:2409.15647, September 2024. doi: 10.48550/arXiv. 2409.15647.

Tuo Feng, Wenguan Wang, and Yi Yang. A Survey of World Models for Autonomous Driving. arXiv e-prints, art. arXiv:2501.11260, January 2025. doi: 10.48550/arXiv.2501.11260.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv e-prints, art. arXiv:2502.05171, February 2025. doi: 10.48550/arXiv.2502.05171.

Gemini Team, Google DeepMind. Gemini 3 flash model card. Technical report, Google DeepMind, December 2025. URL https://deepmind.google/models/model-cards/ gemini-3-flash/.

Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Google DeepMind. Genie 3: A new frontier for world models, 2025. https://deepmind. google/blog/genie-3-a-new-frontier-for-world-models/.

Alex Graves. Adaptive Computation Time for Recurrent Neural Networks. arXiv e-prints, art. arXiv:1603.08983, March 2016. doi: 10.48550/arXiv.1603.08983.

Yanchen Guan, Haicheng Liao, Zhenning Li, Jia Hu, Runze Yuan, Yunjian Li, Guohui Zhang, and Chengzhong Xu. World Models for Autonomous Driving: An Initial Survey. arXiv e-prints, art. arXiv:2403.02622, March 2024. doi: 10.48550/arXiv.2403.02622.

David Ha and J¨urgen Schmidhuber. Recurrent world models facilitate policy evolution. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, pp. 2455–2467, Red Hook, NY, USA, 2018. Curran Associates Inc.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International Conference on Machine Learning, pp. 2555–2565, 2019.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=S1lOTC4tDS.

Danijar Hafner, Timothy P Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=0oabwyZbOu.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 2025. DOI: 10.1038/s41586-025-08744-2.

Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation. arXiv e-prints, art. arXiv:2602.11451, February 2026. doi: 10.48550/arXiv.2602.11451.

Divya Jyoti Bajpai and Manjesh Kumar Hanawal. A Survey of Early Exit Deep Neural Networks in NLP. arXiv e-prints, art. arXiv:2501.07670, January 2025. doi: 10.48550/arXiv.2501.07670.

Yeskendir Koishekenov, Aldo Lipani, and Nicola Cancedda. Encode, think, decode: Scaling testtime reasoning with recursive latent thoughts, 2026. URL https://openreview.net/ forum?id=jBSye8M3FQ.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum? id=H1eA7AEtvS.

Wenxuan Li, Hang Zhao, Zhiyuan Yu, Yu Du, Qin Zou, Ruizhen Hu, and Kai Xu. PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation. arXiv e-prints, art. arXiv:2504.16693, April 2025a. doi: 10.48550/arXiv.2504.16693.

Xinqing Li, Xin He, Le Zhang, Min Wu, Xiaoli Li, and Yun Liu. A Comprehensive Survey on World Models for Embodied AI. arXiv e-prints, art. arXiv:2510.16732, October 2025b. doi: 10.48550/arXiv.2510.16732.

Fan-Ming Luo, Tian Xu, Hang Lai, Xiong-Hui Chen, Weinan Zhang, and Yang Yu. A Survey on Model-based Reinforcement Learning. arXiv e-prints, art. arXiv:2206.09328, June 2022. doi: 10.48550/arXiv.2206.09328.

Vincent Micheli, Eloi Alonso, and Fran¸cois Fleuret. Transformers are sample-efficient world models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=vhFu1Acb0xb.

Vincent Micheli, Eloi Alonso, and Franc¸ois Fleuret. Efficient world models with context-aware tokenization. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id=BiWIERWBFX.

OpenAI. Video generation models as world simulators, 2024. https://openai.com/index/ video-generation-models-as-world-simulators/.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin (eds.), Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp. 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040/.

Francesco Pappone, Donato Crisostomi, and Emanuele Rodol`a. Two-Scale Latent Dynamics for Recurrent-Depth Transformers. arXiv e-prints, art. arXiv:2509.23314, September 2025. doi: 10.48550/arXiv.2509.23314.

Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, and Daniel Y. Fu. Parcae: Scaling Laws For Stable Looped Language Models. arXiv e-prints, art. arXiv:2604.12946, April 2026. doi: 10.48550/arXiv.2604.12946.

Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=din0lGfZFd.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588:604–609, 2020. https: //doi.org/10.1038/s41586-020-03051-4.

Erik Talvitie. Self-correcting models for model-based reinforcement learning. In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, AAAI’17, pp. 2597–2603. AAAI Press, 2017.

Surat Teerapittayanon, Bradley McDanel, and H. T. Kung. BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. arXiv e-prints, art. arXiv:1709.01686, September 2017. doi: 10.48550/arXiv.1709.01686.

Luozhou Wang, Zhifei Chen, Yihua Du, Dongyu Yan, Wenhang Ge, Guibao Shen, Xinli Xu, Leyi Wu, Man Chen, Tianshuo Xu, Peiran Ren, Xin Tao, Pengfei Wan, and Ying-Cong Chen. A Mechanistic View on Video Generation as World Models: State and Dynamics. arXiv e-prints, art. arXiv:2601.17067, January 2026. doi: 10.48550/arXiv.2601.17067.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Cˆot´e, and Prithviraj Ammanabrolu. ScienceWorld: Is your agent smarter than a 5th grader? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.775. URL https: //aclanthology.org/2022.emnlp-main.775/.

Zijun Wang, Panwen Hu, Jing Wang, Terry Jingchen Zhang, Yuhao Cheng, Long Chen, Yiqiang Yan, Zutao Jiang, Hanhui Li, and Xiaodan Liang. ProPhy: Progressive Physical Alignment for Dynamic World Simulation. arXiv e-prints, art. arXiv:2512.05564, December 2025. doi: 10. 48550/arXiv.2512.05564.

Chenjun Xiao, Yifan Wu, Chen Ma, Dale Schuurmans, and Martin M¨uller. Learning to combat compounding-error in model-based reinforcement learning, 2020. URL https:// openreview.net/forum?id=S1g_S0VYvr.

Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped Transformers are Better at Learning Learning Algorithms. arXiv e-prints, art. arXiv:2311.12424, November 2023. doi: 10.48550/arXiv.2311.12424.

Abbas Zeitoun, Lucas Torroba-Hennigen, and Yoon Kim. Hyperloop Transformers. arXiv e-prints, art. arXiv:2604.21254, April 2026. doi: 10.48550/arXiv.2604.21254.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, and Jason Eshraghian. Scaling Latent Reasoning via Looped Language Models. arXiv e-prints, art. arXiv:2510.25741, October 2025. doi: 10.48550/arXiv.2510.25741.

Łukasz Kaiser, Mohammad Babaeizadeh, Piotr Miłos, Bła˙zej Osi´nski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, Afroz Mohiuddin, Ryan Sepassi, George Tucker, and Henryk Michalewski. Model based reinforcement learning for atari. In International Conference on Learning Representations, 2020. URL https:// openreview.net/forum?id=S1xCPJHtDB.

