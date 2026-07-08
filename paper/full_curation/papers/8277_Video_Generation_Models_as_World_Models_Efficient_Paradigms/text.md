# Video Generation Models as World Models: Efficient Paradigms, Architectures and Algorithms

Muyang He, Hanzhong Guo, Junxiong Lin, Yizhou Yu, Fellow, IEEE

## arXiv:2603.28489v2[eess.IV]4May2026

Abstract—The rapid evolution of video generation has enabled models to simulate complex physical dynamics and long-horizon causalities, positioning them as potential world simulators. However, a critical gap still remains between the theoretical capacity for world simulation and the heavy computational costs of spatiotemporal modeling. To address this, we comprehensively and systematically review video generation frameworks and techniques that consider efficiency as a crucial requirement for practical world modeling. We introduce a novel taxonomy in three dimensions: efficient modeling paradigms, efficient network architectures, and efficient inference algorithms. We further show that bridging this efficiency gap directly empowers interactive applications such as autonomous driving, embodied AI, and game simulation. Finally, we identify emerging research frontiers in efficient video-based world modeling, arguing that efficiency is a fundamental prerequisite for evolving video generators into general-purpose, real-time, and robust world simulators.

Index Terms—Video Generation, World Models, Interactive Simulation, Diffusion Models, Embodied AI.

I. INTRODUCTION

In the rapidly evolving landscape of generative artificial intelligence, video generation has received remarkable attention due to its potential to simulate complex world dynamics. This field has undergone a transformative journey, progressing from early generative adversarial networks (GANs) [1], [2] and pixel-level auto-regressive (AR) models [3], [4] to high-fidelity diffusion-based approaches [5]–[13], and more recently to large-scale architectures that function as “World Simulators” capable of modeling physical laws and long-horizon causalities [14], [15]. This progression marks a substantial leap in generative capabilities, enabling models not only to synthesize visual content but to understand and predict the underlying physics of the environment, thereby paving the way for AGI [16], [17].

To fully appreciate this leap, it is essential to understand video generation has the potential to achieve world modeling. The concept of world modeling seeks to move beyond simple pattern matching toward a fundamental understanding of environmental dynamics. A world model is generally defined as an internal representation of environmental dynamics that enables the prediction of future states based on historical contexts and,

Muyang He and Yizhou Yu are with School of Computing and Data Science, The University of Hong Kong, Hong Kong SAR, China and Hong Kong Generative AI Research and Development Center, Hong Kong SAR, China (E-mail: muyanghe@connect.hku.hk; yizhouy@acm.org).

Hanzhong Guo and Junxiong Lin are with School of Computing and Data Science, The University of Hong Kong, Hong Kong SAR, China (E-mail: hanzhong@connect.hku.hk; junxionglin26@outlook.com).

(M. He, H. Guo and J. Lin contributed equally.) (Corresponding author: Yizhou Yu.)

optionally, actions [16]. In the context of visual synthesis, video-based world models treat the generative process as a simulation of the physical world, where the objective is to model the underlying causal mechanisms such as gravity, collision, and object permanence rather than just pixel transitions. Mathematically, this can be viewed as learning the transition function P(st+1|st,at), where s represents the state (video frames or latents) and a represents the conditions or actions (e.g., text prompts or camera trajectories). As emphasized in the development of Sora [14], scaling video generation models leads to the emergence of simulation capabilities, where the model demonstrates an initial comprehension of physical laws without explicit hard-coding.

This alignment between video generation and world modeling offers several advantages:

Emergent Physics: Large-scale training on diverse video data allows models to learn complex interactions, such as agent-environment interactions or fluid dynamics, which are difficult to model via traditional analytical engines.

Latent Imagination: Modern world models often operate in compact latent spaces [16], [17], allowing the imagination of future scenarios to occur at a lower computational cost than high-resolution pixel rendering. This inherently links the concept of world modeling to computational efficiency.

Unified Reasoning: By treating video generation as world modeling, the same architecture can be applied to diverse domains ranging from media production to autonomous driving [18], [19] and robotic manipulation [20], where the model acts as a general-purpose simulator for decision-making.

To function as effective world simulators, video generators must ensure long-term consistency, physical accuracy, and high-resolution interactivity [20], [21]. However, highdimensional video data and complex dynamics impose severe computational and memory bottlenecks. For example, autoregressive models face key-value (KV) cache explosion during long-sequence generation [22], [23], and diffusion models suffer from high latency due to iterative denoising. In addition, the vast redundancy in video frames must be reduced without losing semantics [24], [25]. To prevent these efficiency limitations from hindering scalability, the development of efficient architectures and algorithms is crucial for enabling real-time deployment.

Taxonomy. As shown in Figure 1, this article systematically investigates the role of efficiency in the aspects of modeling, architectures, and algorithms for video-based world models, covering the spectrum between AR-based and Diffusion-based paradigms. Our discussion is structured around three core dimensions: Efficient Modeling (covering efficiency-oriented

###### Video Generation Models as World Models

Efficient Modeling (Sec. III)

Efficient Architecture (Sec. IV) Hierarchical & VAE Designs Long Context & Memory Mechanisms Efficient Attention Extrapolation and RoPE

Efficient Inference (Sec. V) Parallelism Caching Pruning Quantization

Applications (Sec. VI) Autonomous Driving Embodied AI Game & Interactive World Simulation

More Related Work (Sec. VII)

Diffusion Model Distillation for Efficient Sampling

Interactive Talking Head Generation

Interactive Content Creation Video-Driven Scene Generation

Auto-Regressive and Hybrid Approaches

- Fig. 1. A taxonomy of representative topics related to efficiency improvement for video generation-based world models.

modeling paradigms), Efficient Architectures (designs such as VAEs, memory mechanisms, and efficient attention), and Efficient Inference (system deployment considerations including parallelism, caching, pruning, and quantization). Furthermore, this article also explores how these efficient models are used in downstream application scenarios, such as autonomous driving, embodied AI and games/interactive simulations. By reviewing comprehensive insights in this rapidly evolving field, we aim to catalyze new advances in video-based world models that leverage efficient computing to tackle increasingly sophisticated simulation challenges.

Within the existing literature, previous studies have primarily explored general video generation or specific diffusion model based techniques. More recently, amidst the significant advances in Sora-like models [14], some works have begun to address the computational demands of video generation. However, a systematic review specifically elucidating how efficiency improvement techniques can benefit a video-based world model is notably absent. To the best of our knowledge, this article presents the first systematic exploration dedicated to the intersection of efficiency improvement techniques and the multiple facets of video-based world models. The main contributions of this paper are summarized as follows:

- • We provide the first comprehensive review of the critical intersection between efficiency improvement techniques and video-based world models.
- • We introduce a novel taxonomy that provides a structured perspective on efficiency across three dimensions: modeling paradigms, architectural designs, and inference optimizations.
- • We detail how these efficiency improvement techniques empower critical applications such as autonomous driving, embodied AI, and interactive simulation.
- • We further discuss key challenges and future opportunities in efficient video-based world modeling.

In the remainder of this paper, we introduce background knowledge in Section II, efficient modeling paradigms in Section III, efficient architectures in Section IV, and efficient inference algorithms in Section V. In addition, promising applications and more related works are discussed in Section VI and Section VII. Finally, Section VIII concludes this paper.

II. BACKGROUND A. Generative Paradigms

Modern video generation models are largely built upon paradigms established in image synthesis. We introduce the mathematical formulations of these generative models, focusing on Diffusion Models and Flow Matching as the current dominant approaches, followed by Auto-regressive models.

1) Denoising Diffusion Probabilistic Models (DDPM): Diffusion models [26] formulate generation as a denoising process. To improve efficiency, most state-of-the-art models operate in the latent space of a pre-trained variational autoencoder (VAE), known as Latent Diffusion Models (LDMs) [6]. Forward Process. Given a data sample x0 ∼ q(x0) (or its latent representation z0), the forward process is a fixed Markov chain that gradually adds Gaussian noise according to a variance schedule βt ∈ (0,1). The transition probability is defined as:

q(xt|xt−1) = N(xt; 1 − βtxt−1,βtI) (1)

Using the notation αt = 1 − βt and α¯t = ts=1 αs, we can sample xt at any timestep t directly from x0:

xt = √α¯tx0 + √1 − α¯tϵ, where ϵ ∼ N(0,I) (2) Reverse Process and Training. The generative process reverses this noise addition. Since the true posterior q(xt−1|xt) is intractable, we approximate it with a parameterized distribution pθ(xt−1|xt). In practice, the model is trained to predict the added noise ϵ or the velocity v. The simplified training objective is often the mean squared error (MSE) between the actual noise ϵ and the predicted noise ϵθ:

0,t,ϵ ∥ϵ − ϵθ(xt,t)∥2 (3) Once trained, the model generates data by iteratively denoising pure Gaussian noise xT to x0.

Lsimple = Ex

2) Flow Matching: While DDPMs rely on a pre-defined forward process in Eq. (2), which transports samples through a fixed and typically curved noising trajectory, Flow Matching (FM) [27], [28] instead models generation as a continuoustime probability path governed by ordinary differential equations (ODEs). FM defines a probability density path pt that transforms a simple prior distribution into the data distribution through a time-dependent vector field vt(x):

d dt

ϕt(x) = vt(ϕt(x)), ϕ0(x) = x (4)

where ϕt maps samples from t = 0 to t. The goal is to learn a parameterized vector field vθ(x,t) that matches the target velocity field associated with the chosen probability path.

Since directly regressing the marginal target velocity field is generally intractable for complex data distributions, flow matching is commonly implemented in a conditional form. Given a source sample z0 and a target data sample x1, one defines a conditional probability path pt(x | z0,x1) together with a tractable conditional target vector field ut(x | z0,x1). The resulting conditional flow matching (CFM) objective is

LCFM = Et,z0,x1,xt∼pt(·|z0,x1) ∥vθ(xt, t) − ut(xt | z0, x1)∥2 .

(5)

In common straight-line path formulations, the conditional path is chosen as a linear interpolation between noise z0 and data x1, namely xt = tx1 + (1 − t)z0. In this case, the target velocity becomes a constant, i.e., ut = x1 − z0, and the objective reduces to

0,x1 ∥vθ(xt,t) − (x1 − z0)∥2 . (6)

LCFM = Et,z

3) Auto-regressive (AR) Models: AR models decompose the joint probability distribution of a sequence x into a product of conditional probabilities. In a canonical visual generation formulation, x represents a flattened sequence of discrete visual tokens derived from a VQ-VAE-style tokenizer [29], where an encoder maps patches or frames to continuous latents that are snapped to a learned finite codebook via nearestneighbor vector quantization (VQ), although more general autoregressive video models may also operate on other compressed latent token sequences. For a sequence of length N:

N

p(xi | x<i) (7)

p(x) =

i=1

Training maximizes the log-likelihood of the next token given the previous context. While training is efficient due to parallel teacher forcing, inference is inherently sequential and can become computationally expensive for long videos (O(N)).

B. From Image to Video Generation

Transitioning from image to video generation involves extending 2D spatial modeling to the 3D spatiotemporal domain (T × H × W). Efficient techniques largely focus on how to manage the cubic growth in complexity.

Inflation Early approaches directly inflated 2D kernels into 3D kernels (e.g., 3 × 3 → 3 × 3 × 3) [8]. While preserving spatial priors from pre-trained image models, this drastically increases parameter count and computational load.

Factorization To improve efficiency, modern architectures factorize 3D operations into separate 2D spatial and 1D temporal operations. For instance, Video LDM [10] inserts temporal attention layers after spatial blocks in a pre-trained image U-Net. This allows the model to learn motion dynamics without catastrophic forgetting of spatial concepts and reduces computational complexity in attention mechanisms from O((THW)2) to O(T(HW)2 + HW(T)2).

Spacetime Tokenization Emerging Transformer-based video models (e.g., Latte [30]) treat video as a unified volumetric sequence. Instead of processing frames individually, they extract

3D spacetime cubes (“tubelets”) as tokens, utilizing a spatial and temporal downsampling mechanism by encapsulating a local spatial region across multiple consecutive frames into a single token. Consequently, the model allows for jointly capturing spatial semantics and temporal evolution within a unified attention layer, although this necessitates sophisticated positional embeddings (e.g., 3D RoPE) to accurately preserve spatiotemporal geometry.

C. Architectures

Modern video generative frameworks typically follow a modular pipeline consisting of three core components.

Latent Compression Module (usually a VAE) To mitigate the high dimensionality of video, VAEs compress pixel data into a latent space [6]. Modern video generators often utilize 3D causal VAEs [31], [32] to jointly reduce spatial and temporal redundancy.

Generative Backbone The central component performs denoising or next-step prediction within the latent space. This backbone is primarily implemented using either a convolutional U-Net [33] or a Diffusion Transformer (DiT) [15]. DiT adopts 3D patchification and self-attention to capture longrange spatiotemporal dependencies.

Conditioning Module Modern video generators, especially video-based world models, are no longer conditioned by text alone, but increasingly support multimodal inputs such as reference images, video clips, audio, actions, trajectories, layouts, and other structured control signals. Textual guidance is commonly encoded by CLIP [34] or T5-XXL [35] and other vision-language models (VLMs) [36]–[39]. Beyond text prompts, structured conditions such as bounding boxes, road layouts, and ego trajectories can be injected to constrain scene geometry and motion, as demonstrated in drivingoriented models such as MagicDrive-V2 [40]. In interactive world models, action signals can be represented as discrete tokens, latent actions, or control embeddings, and integrated into generation to obtain action-conditioned rollouts, as in Genie [41], Matrix-Game 2.0 [42], and Cosmos-Predict [43]. Audio conditions are typically encoded by a speech or motion encoder and used to guide temporal dynamics such as lip motion, facial expression, or speech rhythm [44]–[49]. These conditions are injected into the generative backbone through cross-attention, adaptive normalization, or token merging. For example, autoregressive frameworks such as iVideoGPT [50] serialize heterogeneous conditions into a unified sequence, whereas diffusion-based models more often fuse them through cross-attention layers or a token merging mechanism [36], [37]. Overall, the conditioning module determines not only what should be generated, but also how the generated world should evolve under external instructions or interactions.

III. EFFICIENT MODELING

Efficient modeling is central to scaling video generation from short clips to long-horizon, high-resolution sequences under practical latency and memory constraints. This section reviews two major directions: (i) diffusion model distillation,

which reduces the number of sampling steps required for highfidelity generation, and (ii) long-horizon interactive modeling paradigms, including autoregressive, hybrid AR-diffusion, and streaming causal diffusion approaches that aim to support realtime interaction and persistent world simulation.

- A. Diffusion Model Distillation for Efficient Sampling

While architectural and system optimizations reduce wallclock latency per step, a complementary direction is posttraining acceleration that directly reduces the number of denoising steps. In diffusion-based video generation, the sampling cost scales linearly with the step count K. Distillation aims to train a student model that matches the teacher diffusion model’s sampling behavior with significantly fewer stepsdown to few-step or even one-step generation.

1) Step-Reduction Distillation: A direct approach distills a K-step teacher sampler into a K′-step student sampler (K′ ≪ K) [51], [52]. Let T denote a fixed teacher solver. Starting from xt, the teacher produces a target xt−∆ after ∆ steps. The student is trained to match this result in one macro-step:

Lstep(θ) = E x ˆ(t−S)∆ − xˆ(t−T)∆

2 2

, (8)

where xˆ(t−T)∆ = T ∆(xt,c) is the teacher rollout target. Progressive variants halve the step count iteratively. In video

generation, GPD [53] provides a representative example of this direction by progressively guiding the student model to operate with larger step sizes, reducing the sampling steps of Wan [38] from 48 to 6 while maintaining competitive quality.

2) Consistency Distillation: Consistency-style objectives learn a mapping fθ(xt,t) that maps any point on the trajectory to the origin x0. Consistency training enforces that predictions from two timepoints along the same trajectory agree [54], [55]:

#### Lcons(θ) = E ∥fθ(xt,t,c) − fθ(xs,s,c)∥22 , s < t, (9)

where xs is obtained by advancing from xt. This enables one-step generation. VideoLCM [56] and AnimateLCM [57] extend this to latent video models, enabling real-time synthesis. TurboDiffusion [58] introduces a unified framework that combines consistency models with reward-guided distillation, significantly enhancing the visual quality of one-step outputs. Similarly, open-source initiatives like FastVideo [59] provide optimized pipelines for distilling large-scale video models into few-step or one-step variants, democratizing real-time video generation capabilities.

3) Adversarial Distillation: To maintain perceptual fidelity under extremely small step budgets, recent distillation methods increasingly optimize the student at the distribution level rather than relying only on pointwise regression targets. A generic objective can be written as

D(pS(·|c)∥pT(·|c)), (10)

min

θ

where D denotes a generic discrepancy between the student distribution pS and the teacher distribution pT. Such discrepancies can be instantiated in three ways. First, D can be an explicit statistical divergence or its score-based surrogate, such

as approximate KL divergence or Fisher-type score matching. Second, D can be an implicitly learned discrepancy induced by a discriminator, as in GAN-style adversarial training. Third, practical systems often combine the two, using distribution/score matching to preserve teacher alignment while introducing adversarial supervision to improve realism and perceptual sharpness.

Representative examples of the first direction include DMD [60] and related DMD-style methods, which match the student to the teacher at the distribution level without enforcing a strict one-to-one correspondence with the teacher’s sampling trajectory. In the hybrid regime, DMD2 [61] further augments distribution matching with a GAN loss on real data, and AVDM2D [62] can also be viewed within this broader family of perceptually enhanced distribution-matching distillation.

For video generation, recent work increasingly moves toward pure adversarial post-training. Seaweed-APT [63] applies adversarial post-training against real data after diffusion pretraining, together with an approximated R1 regularization, enabling real-time one-step video generation.

However, these distillation-based approaches primarily improve step efficiency and wall-clock latency. They are usually insufficient to support persistent, long-horizon generation, which requires explicit mechanisms for causal inference, memory retention, and error control over long horizons.

B. Auto-Regressive and Hybrid Approaches

Autoregressive and hybrid approaches aim to overcome the limitation of traditional video diffusion models as mainly clip-based generators. By combining autoregressive temporal rollout with efficient video synthesis, these methods move toward persistent, interactive, and long-horizon world modeling. These methods focus on infinite-length generation with realtime interactivity by strategically combining AR scalability with diffusion fidelity.

1) Auto-Regressive Modeling: Treating video generation as a discrete token prediction problem allows models to inherit the scalability of autoregressive language models. A representative early work is VideoGPT [4], which employs a VQ-VAE to learn discrete spatiotemporal latent tokens from raw videos and then uses a GPT-like transformer to autoregressively model these tokens. This formulation establishes a clean and reproducible baseline for transformer-based video generation. More recent work extends this idea to large-scale multimodal and long-horizon generation. VideoPoet [64] adopts a decoderonly transformer architecture that processes multimodal inputs, including text, images, videos, and audio, in a unified autoregressive framework. By following an LLM-style pretraining and adaptation pipeline, it demonstrates strong capability in zero-shot video generation and high-fidelity motion synthesis. Loong [65] further pushes autoregressive generation toward minute-level long videos by modeling text tokens and video tokens as a unified sequence for autoregressive language models, together with progressive short-to-long training and inference strategies to mitigate error accumulation.

Pure autoregressive modeling is also highly relevant to interactive world modeling. Genie [41] introduces a generative

interactive environment model composed of a spatiotemporal video tokenizer, an autoregressive dynamics model, and a latent action model. This design enables frame-by-frame controllable generation and shows that autoregressive world models can support interactive environment simulation. Along a similar direction, iVideoGPT [50] formulates world modeling as next-token prediction over a unified sequence of visual observations, actions, and rewards. Its scalable autoregressive transformer architecture supports action-conditioned video generation, visual planning, and model-based reinforcement learning, making it a strong representative of autoregressive video-based world models.

- 2) Hybrid AR-Diffusion Modeling: Hybrid AR-diffusion

modeling aims to combine the long-horizon rollout capability of autoregressive generation with the high-fidelity synthesis ability of diffusion models. In this paradigm, temporal dependencies are modeled autoregressively along the time dimension, such that previously generated frames or chunks serve as the context for predicting subsequent content, while the current frame or chunk is still generated by a diffusion model rather than directly decoded by a pure AR backbone. Progressive Autoregressive Video Diffusion Models [66] is a representative work in this direction. It revisits autoregressive long video generation by assigning progressively increasing noise levels across frames and performing denoising together with temporal shifting in small intervals, thereby improving information propagation and generation fidelity over long horizons. FramePack [67] further develops this paradigm for next-frame or next-frame-section prediction by compressing historical frame contexts according to frame-wise importance, allowing much longer effective contexts under a fixed context length while introducing drift prevention strategies to reduce long-horizon error accumulation. Overall, this hybrid paradigm provides a practical compromise between temporal scalability and visual fidelity, making it an important direction for efficient long-horizon video world modeling.

- 3) Streaming Causal Diffusion Modeling: Streaming causal

diffusion modeling can be viewed as a complementary line of work to token-level autoregressive models and hybrid ARdiffusion pipelines. Instead of explicitly predicting discrete future tokens or introducing a separate autoregressive module or algorithm, it causalizes the diffusion model itself—typically through temporal causal attention or block-causal design, so that frames or chunks can be generated incrementally without relying on future context. In this way, conventional offline clip-wise diffusion is transformed into a streaming-friendly generation paradigm. For example, CausVid [68] reformulates video diffusion with causal attention masks, enabling frameby-frame generation and making diffusion models more suitable for autoregressive streaming scenarios.

However, causal attention alone is insufficient for stable long-horizon rollout, since continuous autoregressive generation still suffers from severe train-test mismatch and error accumulation. Diffusion Forcing [69] provides a representative training paradigm for this setting by combining causal nexttoken prediction with full-sequence diffusion and allowing different tokens to be denoised under independent noise levels. Building on this idea, Self Forcing [70] explicitly bridges the

train-test gap by conditioning training on self-generated histories rather than only on ground-truth contexts, thereby improving long-horizon stability. Rolling Forcing [71] further extends this principle with a rolling-window denoising strategy that jointly generates multiple future frames, substantially reducing long-range drift while enabling real-time multi-minute video generation. Follow-up work pushes the same forcing-based streaming line toward longer horizons and sharper training objectives. Self-Forcing++ [72] scales self-generated guidance to minute-scale video generation far beyond the original teacher horizon, while Reward Forcing [73] enhances motion dynamics through Rewarded Distribution Matching Distillation, biasing the model toward high-reward dynamic regions and thereby allocating limited generation capacity more effectively to behaviorally important events. Closely related to few-step interactive streaming, Causal Forcing [74] studies autoregressive diffusion distillation from pretrained bidirectional video diffusion models. It argues that initializing the causal student via ODE distillation from a bidirectional teacher can violate frame-level injectivity under the probability-flow ODE, leading to a biased conditional-expectation solution rather than the teacher’s flow map, and instead performs ODE initialization using an autoregressive teacher to align the distillation objective with causal generation. Orthogonal to further training-time refinements, Rolling Sink [75] targets the mismatch between finite clip-length training and open-ended test-time horizons, and proposes a training-free test-time procedure for autoregressive cache maintenance that scales autoregressive video diffusion models trained with Self Forcing to minute-scale generation while improving long-horizon fidelity and temporal consistency. Beyond likelihood- or distillation-based causal diffusion training, AAPT [76] further extends adversarial posttraining to autoregressive streaming generation, reducing error accumulation through student-forcing and demonstrating that diffusion-based video generators can move closer to interactive real-time rollout.

C. Discussion

Efficient modeling methods are driven by two equally fundamental objectives: per-step efficiency and long-horizon interactivity. Distillation-based approaches are highly effective in straightening the denoising trajectory and reducing latency from tens of steps to only a few or even a single step, making real-time generation increasingly practical. However, most of these methods remain centered on accelerating fixed-length generation, and therefore do not by themselves resolve the challenges of persistent rollout and long-term error accumulation. In contrast, autoregressive, hybrid AR-diffusion, and causal streaming diffusion paradigms, based on efficient perstep inference, explicitly target these world-model requirements by introducing causal generation interfaces.

While per-step efficiency and interactivity form the foundation of video-based world models, a more critical and unresolved goal is optimizing consistency, stability, and spatial understanding in long-duration scenarios. As illustrated in Figure 2, current state-of-the-art models still exhibit severe degradation over extended timelines. Over a 10-minute generation window, we observe that most methods struggle with

### 0s 15s 40s 2m 5m 10m

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Self Forcing [70]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

LongLive [77]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Rolling Forcing [71]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Deep Forcing [78]

Causal Forcing

- [74]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Rolling Sink

- [75]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

|Red boxes|
|---|

- Fig. 2. Qualitative comparison of long-video generation. We evaluate six methods built upon Wan2.1-T2V-1.3B [38] across a 10-minute timeline. are added to highlight specific failure modes for each method. Self Forcing [70]: inherently limited to a maximum generation length of 5 seconds (81 frames), leading to catastrophic structural collapse (e.g., at 40s), outfit inconsistencies (e.g., at 2m and 10m), and color artifacts (e.g., at 5m). LongLive [77]: severe structural hallucination and geometric distortion of the red bus (e.g., at 15s, 40s, and 5m). Rolling Forcing [71]: inconsistent spatial structures, such as detached skateboards or duplicated characters (e.g., at 15s, 2m, and 10m). Deep Forcing [78]: severe background hallucinations, with the clock tower disappearing or completely transforming (e.g., at 10m). Causal Forcing [74]: illogical 3D geometry and inconsistent backgrounds, such as a kiosk in the middle of the road (e.g., at 2m) and a mutating clock tower. Rolling Sink [75]: human body distortions (e.g., at 0s and 15s), scene flattening and incorrect spatial relationships (e.g., at 5m and 10m). Overall, maintaining complex interactions and persistent 3D structures over minutes remains a significant challenge.

object permanence and structural consistency in later stages. Furthermore, accurately synthesizing complex human-object interactions (e.g. skateboarding) and maintaining coherent relative spatial positioning within a 3D environment remains a significant obstacle.

Addressing these limitations requires a novel modeling paradigm. This involves designing algorithms to mitigate cumulative errors in long-term interactions and memory mechanisms to ensure spatial, logical, and physical consistency.

IV. EFFICIENT ARCHITECTURE

To overcome spatiotemporal redundancy and the quadratic cost of attention in long-horizon video generation, efficient architectural design is the most direct and effective method for enhancing video generation from short clips to persistent, high-fidelity world models. This section reviews four structural optimization paradigms: (i) Hierarchical and VAE Designs, which compress the world’s state into compact or coarse-tofine representations; (ii) Long Context and Memory Mechanisms, offering scalable alternatives for long-term consistency; (iii) Efficient Attention, accelerating computation via sparse, windowed, or linear mechanisms; and (iv) Extrapolation and RoPE, providing cost-effective methods to extend generation beyond training horizons.

A. Hierarchical & VAE Designs

The common framework for efficient video-based world modeling involves decomposing the high-dimensional spatiotemporal signal into a coarse-to-fine hierarchy or a compact latent space, reducing the state complexity that the model must simulate.

1) Hierarchical and Pyramidal Generation: This approach is a multi-stage refinement process where a base module establishes a general world model followed by specialized modules for detail enhancement. Pyramidal Flow Matching [79] and TPDiff [80] establish multiple stages with increasing frame rate; the former extends the flow matching algorithm to an efficient spatial pyramid representation and proposes a temporal pyramid design to further improve training efficiency, while the latter introduces a temporal pyramid to increase frame rate along the diffusion process. Waver [81] (Figure 3) and FlashVideo [82] adopt a cascade paradigm to upsample lowresolution videos generated by the DiT to a final resolution of 1080p. Compared to flat architectures, these hierarchical designs significantly reduce the redundant computation of details during the early semantic planning phase. SUPERGEN [83] also follows this two-stage paradigm, where a sketch provides an overview and iterative fine-grained tilebased refinement enriches details. PatchVSR [84] introduces inter-patch modulation during the detail generation process.

Stage I

Window Attention

Waver 1.0 Low Resolution Video

Noise

[Figure 37]

[Figure 38]

[Figure 39]

| | |
|---|---|
|Dual Stream|DiT Block x M<br><br>|
| | |
|Single Stream|DiT Block x N|
| | |

Up

Spatial Window Temporal Window

Sample

[Figure 40]

Waver Refiner

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

w

+

Dual Stream DiT Block x M

Single Stream DiT Block x N

Upsampled Video

1 - w

Noise

High Resolution Video

Stage II

- Fig. 3. Pipeline of cascaded video generation. Figure courtesy of [81].

In addition, on the parameter dimension, SRDiffusion [85] uses a large model during the early high-noise steps to generate higher-quality structures and motion while using a small model during the late low-noise steps to generate finer details, thus accelerating the overall diffusion process. This method offers a more flexible efficiency-quality trade-off than fixed hierarchical cascades, and switches models during different stages of the diffusion process. Conversely, on the spatial dimension, FlexiDiT [86] applies a reversed compute strategy. Observing that the early denoising steps focus on lowfrequency structures, it uses larger patch sizes (lower compute) for the initial denoising steps and switches to smaller patches (higher compute) for later refinement.

2) Efficient VAE and Latent Compression: To model a persistent world, the world state must be compressed into a manageable latent representation. DC-VideoGen [25] introduces a deep compression video autoencoder with a chunkcausal design to achieve up to 64× spatial compression and 4× temporal compression. REGEN [87] further expands this by relaxing the criterion for decoding from exact reproduction to plausible reconstruction. The decoder itself is generative, allowing the encoder to store ultra-compact semantic tokens only, achieving a temporal compression ratio of up to 32×. Considering that VAE’s fixed compression rates cannot capture the temporal non-uniformity of real-world video contents, DLFR-VAE [88] proposes a dynamic VAE that dynamically adjusts the optimal latent frame rate according to the content complexity. In addition, VGDFR [89] adaptively merges frames in the latent space, allowing subsequent denoising steps to be executed in a smaller latent space, significantly reducing computational costs. Recent works such as Turbo-VAED [90] distill heavy decoders into lightweight versions.

- B. Long Context & Memory Mechanisms

Video-based world modeling relies on maintaining consistency over long horizons. The common method augments the generative backbone with an external or implicit memory that serves as a persistent storage of the simulated world.

1) Visual Memory: Visual Memory retains raw or semicompressed keyframes as distinct reference points to anchor the generation. FramePack [67] compresses historical frames according to their relative importance to encode more frames within a fixed context length limit. Following this, WorldPack [23] combines trajectory packing, which enables efficient utilization of long-term history within a fixed-length

Video1 Text1 Videon-1 Textn-1

Output

……

FlexFormer Decoder

Feed Forward

+ …… +

Cross-Attention

Context1

Contextn-1

Textn

Copy Copy

QueryV QueryT

FlexFormer Encoder Self-Attention

DiT Block N

……

Noised Videon

Videon-1 Textn-1

Video1 Copy

Text1

Query

+ Concat

Feature from video VAE Feature from text encoder

Learnable token Reconstructed feature

Fig. 4. LoViC [101] introduces FlexFormer, a flexible encoder that compresses context of arbitrary length under an adaptive compression ratio. The resulting compressed context features are fed into a DiT-based decoder to generate the current video chunk. Figure courtesy of [101].

context by hierarchically compressing and allocating frames, and memory retrieval, which selectively recalls past scenes that share substantial visual overlap with the target scene. This design allows recent frames and frames recalled by memory retrieval to be stored at a high resolution, and the remaining frames to be stored at a lower resolution, enabling the model to retain long-term history while keeping computation efficient. Related works such as StoryMem [91] maintain a compact and dynamically updated memory bank of keyframes from historically generated shots. Astra [92] also adopts frame packing and uses a noise-augmented history memory to avoid over-reliance on past frames.

- 2) Spatial Memory: Spatial Memory utilizes explicit ge-

ometric representations (e.g., point clouds, meshes) to enforce strict physical consistency. EvoWorld [93] maintains an evolving panoramic world using a spherical 3D memory, while VMem [94] constructs a surfel-indexed global map for camera-pose queries. These works shift the model’s role from pixel generation to rendering from a consistent memory. Following these, HunyuanWorld-Voyager [95] and [96] align frames with persistent point clouds for spatial consistency. Memory Forcing [97] combines spatial and temporal memory for maintaining long-term geometric consistency.

- 3) Compressed Contexts: These approaches focus on re-

ducing historical information into compact latent vectors. SVI [22] utilizes an Error Replay Memory that dynamically banks and selectively resamples self-generated diffusion errors across discretized timesteps to simulate and correct accumulation artifacts during fine-tuning. MemFlow [98] also dynamically updates the memory bank by retrieving the most relevant historical frames with the text prompt of the current chunk. VideoSSM [99] introduces a global memory to absorb tokens evicted from the local window and relies on a state space model to recurrently compress them into a compact, fixedsize state. Other works such as Context as Memory [100], LoViC [101] (Figure 4), and Mixture of Contexts [102] refine how contexts are retrieved. Compared to spatial maps, compressed contexts are more flexible, but may struggle with precise geometric grounding.

- 4) Implicit Model Memory: Implicit Model Memory em-

beds historical contexts directly into the model’s weights via online updates (test-time training, TTT). TTT [103] has emerged as a promising approach for efficient sub-quadratic

sequence modeling, which extends the concept of recurrent states in RNNs to a small, adaptive sub-network. The weights of this sub-network are rapidly adapted online via self-supervised objectives to memorize in-context information. [104] incorporate TTT layers into DiT to capture global narrative dependencies for minute-long generation. Addressing the hardware inefficiency of frequent updates, LaCT [105] performs weight updates for massive token blocks rather than individual steps, enabling scalable autoregressive modeling with contexts exceeding 50k tokens. Although this paradigm offers the memorization of very long contexts, it incurs higher inference latency due to the cost of online optimization.

Trained with Full Attention, cost 276 GPU Hours

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Full Attention

VideoGenerationQualityGenerationLatency(s)

Score: 68.25

[Figure 54]

[Figure 55]

[Figure 56]

+ DiTFastAttn Score: 64.69

[Figure 57]

[Figure 58]

[Figure 59]

+ SVG

Score: 67.78

Sequence Length

(b) Comparison of inference time as sequence length increases

- C. Efficient Attention

Trained with MoBA, cost 226 GPU Hours

- Fig. 5. Comparison of inference time among full attention, SVG [107], MoBA [113] and VMoBA [112] as sequence length increases. Figure courtesy of [112].

0s

5s

10s

15s

20s

Short Window Attention

Frame Sink

Prompt 4

Prompt 1

Prompt 2

Prompt 3

|KV re-cache<br><br>to maintain adherence across prompt switches<br><br>Attention<br><br>Recached tokens<br><br>|New Prompt|
|---|
<br><br>|Previous Video|
|---|
|
|---|

Query

Key

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

- Fig. 6. LongLive [77] processes sequential user prompts and generates a corresponding long video using efficient short window attention and frame sink. Figure courtesy of [77].

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Full attention accounts for major end-to-end runtime in video generation. Due to the quadratic computational complexity with respect to context length, attention can be much more dominant as the resolution and number of frames increase. To handle this quadratic complexity of high-resolution or longhorizon world simulation, efficient architectures approximate full attention with sparse attention, window attention, or linear attention, or even replace the attention mechanism with linearcomplexity alternatives (e.g., SSMs).

VMoBA (Ours)

Full Attention

MoBA Score: 56.88

[Figure 76]

[Figure 77]

##### Trained with VMoBA, cost 187 GPU Hours

[Figure 78]

[Figure 79]

[Figure 80]

VMoBA (Ours) Score: 68.34

[Figure 81]

[Figure 82]

MoBA

GFLOPs

Prompt: “A fluffy white sheep dashes across a lush, green meadow.”

- 1) Sparse Attention: Attention in transformers is inherently

sparse [106], which offers a great opportunity to reduce computation. Sparse attention selectively restricts computation to highly relevant or local token pairs. SVG [107] and SVG2 [108] pioneer this direction; the former reveals inherent sparse patterns (e.g., temporal and spatial heads focusing on critical tokens), while the latter leverages semantics-aware permutation to maximize efficiency. Following this paradigm, several works identify similar structural sparsity patterns [109], [110], or directly unify SVG’s heads into scalable radial attention [111]. To address the costly dynamic detection overhead of SVG, VMoBA [112] adapts text-centric MoBA [113] to capture spatiotemporal locality via layer-wise recurrent block partitions. Figure 5 compares the inference time of some of the aforementioned methods. The aforementioned methods, along with several other variants [114]–[117], rely primarily on standard block sparse attention, which partitions queries/keys into blocks with a fixed size and computes attention for selected blocks only, skipping entire blocks to leverage hardware-efficient GPU kernels. However, recent advances explore new dimensions. For instance, FG-Attn [118] challenges the block paradigm with finer-grained slice-level sparsity; to address overlooked query-side redundancy, BSA [119] and Astraea [120] propose mechanisms to selectively prune queries alongside or instead of key-value pairs.

- 2) Windowed Attention: Exploiting inherent spatiotempo-

(a) Qualitative comparison of VMoBA and baseline methods

(c) Comparison of efficiency and quality

in Figure 6 that permanently caches initial frames as global anchors to maintain long-range coherence during infinitelength streaming generation. Alternatively, several approaches explicitly decompose attention into local and global branches. Specifically, UltraGen [122] enables native 4K generation via spatially compressed global attention alongside hierarchical cross-window local attention. Similarly, VORTA [123] and VideoNSA [124] employ learnable routing or gating mechanisms to dynamically fuse local sliding windows with global sparse or compressed tokens. Beyond receptive field limitations, standard sliding windows often suffer from hardware inefficiencies. To resolve this, STA [125] proposes a hardwareaware Sliding Tile Attention that aligns window strides with GPU tile sizes, successfully translating the theoretical FLOP reduction into significant wall-clock speedup.

ral locality, window-based attention restricts computation to local neighborhoods to mitigate the complexity of full 3D attention. To address potential quality degradation caused by limited receptive fields, various works propose combining local windows with global context mechanisms. For instance, DiTFastAttn [121] caches the residual difference between full and windowed attention, while LongLive [77] combines short window attention with a frame sink mechanism as shown

3) Linear Attention: Linear attention mitigates the O(N2) complexity of standard self-attention by employing a kernel feature map ϕ(·) and the associative property of matrix multiplication: O = ϕ(Q)(ϕ(K)⊤V ). This decoupling avoids the explicit N × N matrix, reducing complexity to O(N). Recent video generation models integrate linear attention

through various structural paradigms. At the global level, SANA-Video [126] entirely replaces vanilla attention with ReLU-based linear attention, enabling efficient block-wise autoregressive generation. At the layer level (serial integration), LinVideo [127] adapts pre-trained models by selectively substituting a subset of quadratic attention layers with linear ones, optimizing via distribution matching. At the token level (parallel routing), SLA [128] decomposes attention weights to apply exact sparse attention to critical tokens and linear attention to the marginal majority. Building on these efficient formulations, models like Yume-1.5 [129] further apply linear attention to interactive long-video generation.

4) State Space Models (SSMs): SSMs, particularly Mamba [130], offer a linear-complexity O(N) alternative to Transformers by modeling sequences through recurrent state transitions. LaMamba-Diff [131] designs a novel backbone for diffusion models for image generation. LinGen [132] introduces a hybrid linear-complexity block that couples a bidirectional Mamba2 [133] branch with a Temporal Swin Attention branch, achieving stable minute-length video generation with strictly linear scaling.

- D. Extrapolation and RoPE

- A true world model must simulate the future beyond its

seen horizon, requiring modifications to Rotary Positional Embeddings (RoPE) to prevent distribution drift.

- 1) Frequency-Based Extrapolation: Early adaptations fo-

cused on RoPE frequency scaling. RIFLEx [134] identifies that high-frequency components cause temporal repetition and proposes frequency shifting to enable 3× length extrapolation. This remains a simple, training-free baseline for extending temporal horizons.

- 2) Mitigating Attention Dispersion: UltraViCo [135] identi-

fies “attention dispersion”—where distant tokens dilute learned patterns—as the root cause of quality decay, introducing a constant decay factor to suppress distant scores. Compared to frequency-only scaling, this maintains better imaging quality at larger extrapolation limits.

- 3) From Long to Infinite: To enable effectively infinite

simulation, Infinity-RoPE [136] proposes Block-Relativistic RoPE, rotating new latent blocks relative to a moving local reference frame. This shifts from “extending a window” to a “sliding world” paradigm. Related works like FreeNoise [137] and Align your Latents [10] explore complementary tuningfree noise and attention rescheduling strategies.

- E. Discussion

Despite significant advances, existing efficient architectures face fundamental trade-offs between computational cost and spatiotemporal/causal integrity. Specifically, hierarchical compression often sacrifices long-term semantic consistency for visual refinement, while training-free extrapolation techniques fail to maintain causal progression over long horizons, inevitably leading to motion decay or temporal loops. Furthermore, memory mechanisms face the stability-plasticity dilemma: how to retain a persistent global map (stability) while rapidly adapting to new, unexpected environmental

changes (plasticity). At the operational level, efficient attention variants frequently fail to translate theoretical complexity reduction into wall-clock acceleration. To overcome these bottlenecks, future research must shift toward physics-aware and adaptive paradigms. Promising directions include exploring physically constrained latent spaces, designing hybrid memory hierarchies that couple slowly updated global maps with agile working memories, and using interactive causal chains to replace absolute frame indices. Ultimately, realizing realtime, physics-compliant generation will necessitate hardwaresoftware co-designed mechanisms that dynamically allocate compute based on semantic importance and motion dynamics.

V. EFFICIENT INFERENCE

As video generation models scale to billions of parameters and become capable of generating videos with extended durations (e.g., Seedance 1.0 [37] with 30B parameters), running inference on a single GPU often faces severe memory bottlenecks and unacceptable latency. To address these challenges, efficient inference strategies focus on distributing the computational load, reducing redundant calculations and quantization. This section reviews four critical strategies: (i) Parallelism, which distributes inference across multiple devices via spatial, sequence, and pipeline partitioning; (ii) Caching, which exploits spatial and temporal redundancy to accelerate generation; (iii) Pruning, which directly mitigates sequence length explosion and architectural redundancy by merging tokens and streamlining networks; and (iv) Quantization, which lowers the precision of weights and activations to reduce computational resource and memory cost.

A. Parallelism

Parallel inference is critical for scaling video generation to high resolution, long duration, and real-time inference, since the computational and memory costs of diffusion transformers grow rapidly with sequence length. In practice, existing systems mainly improve inference efficiency through sequencelevel partition, pipeline-style execution, and hybrid parallel frameworks, making it possible to generate in real-time with multi-GPUs. A straightforward strategy is to split spatial or temporal tokens across multiple devices so that memory and computation can be distributed. In diffusion inference, DistriFusion [138] shows that patch-wise distributed inference can be made efficient by reusing features from the previous denoising step, thereby overlapping communication with computation. For long-form video generation, Video-Infinity [139] further extends this idea with clip parallelism and dual-scope attention, enabling distributed long-video generation across multiple GPUs.

Another complementary strategy is to partition model execution into a pipeline so that different devices process different parts of the workload concurrently. Rather than fully relying on one type of parallelism, recent DiT inference systems increasingly exploit pipeline-style execution at the patch level to improve device utilization and reduce end-toend latency. Related system-oriented designs also appear in streaming avatar generation. For example, LiveAvatar [140]

introduces timestep-forcing pipeline parallelism, which assigns different denoising timesteps to different devices and converts the diffusion chain into a high-throughput streaming pipeline.

Since no single parallel strategy is optimal under all hardware and model settings, unified frameworks have recently emerged to combine multiple forms of parallelism. xDiT [141] is a representative example, which integrates sequence parallelism, PipeFusion-style pipeline parallelism, and classifierfree guidance (CFG) [142] parallelism into a scalable inference engine for diffusion transformers.

- B. Caching

Caching methods accelerate video generation by exploiting redundancy across adjacent denoising steps. As diffusion inference proceeds through a sequence of timesteps, intermediate activations often evolve gradually, making it unnecessary to recompute all features from scratch at every step. In recent video generation systems, this direction has rapidly evolved from coarse feature reuse to more adaptive and fine-grained caching strategies.

Representative recent methods include PAB [143], TeaCache [144], FasterCache [145], and PreciseCache [146]. PAB [143] accelerates video generation by broadcasting attention outputs in a pyramid manner across timesteps, based on the observation that attention redundancy varies across different stages and attention types. TeaCache [144] instead adopts a timestep-aware policy for video diffusion models rather than using a fixed cache interval. It estimates output variation from timestep-related signals and selectively reuses cached outputs when the predicted change is sufficiently small. FasterCache [145] further improves training-free acceleration by combining dynamic feature reuse with classifier-free guidance (CFG) [142]-aware caching, reducing redundancy both across timesteps and between conditional and unconditional branches. More recently, PreciseCache [146] combines stepwise and block-wise caching to skip only truly redundant computations, using low-frequency difference to identify steplevel redundancy and then performing additional block-level reuse within non-skipped steps. Table I summarizes a compact comparison under the unified 4 A800 GPU setting.

While Table I focuses on cache-based acceleration results for general video generation models, recent work has also begun to explore caching mechanisms for video-based world models. HERO [147] proposes a hybrid acceleration scheme for video generation based on multimodal data, such as depth and RGB views. It figures out that shallow layers, which exhibit larger variation, should be recomputed more frequently, whereas deeper and more stable layers can be accelerated through linear extrapolation from preceding timesteps, effectively reducing attention computation. More recently, WorldCache [148] explicitly targets video-based world models and identifies two world-model-specific obstacles for caching, namely heterogeneous token behavior caused by multimodal coupling and non-uniform temporal dynamics where a small subset of hard tokens dominates error accumulation. To address these issues, it introduces curvature-guided heterogeneous token prediction together with chaotic-prioritized

TABLE I COMPACT COMPARISON OF REPRESENTATIVE CACHE-BASED ACCELERATION METHODS UNDER THE UNIFIED 4 A800 GPU SETTING, REPORTED BY PRECISECACHE [146].

VBench [149] ↑

Benchmark Block Method

Speedup ↑ Latency (s) ↓

Open-Sora 1.2 [150] Original 78.79% 1.00× 47.23 PAB [143] 78.15% 1.26× 38.40 TeaCache [144] 78.23% 1.95× 24.73 FasterCache [145] 78.46% 1.67× 29.15 PreciseCacheFlash [146]

78.19% 2.60× 18.38

HunyuanVideo [36] Original 80.66% 1.00× 73.64 PAB [143] 79.37 1.35× 54.54 TeaCache [144] 80.51% 1.64× 44.90 FasterCache [145] 80.59% 1.43× 51.50 PreciseCacheTurbo [146]

80.49% 1.95× 37.76

adaptive skipping, achieving up to 3.7× end-to-end speedup while maintaining 98% rollout quality.

C. Pruning

Pruning techniques in video diffusion models aim to reduce computational burden by eliminating redundant information at the token, channel, or layer level. To tackle the huge computational overhead introduced by the spatial resolution and temporal depth of video data, recent approaches exploit redundancies across both the video content and the diffusion generation process. We categorize these techniques into two primary paradigms: token-level reduction, which addresses sequence length explosion by merging or dropping redundant visual tokens, and structural pruning, which streamlines the model architecture by removing network components or reallocating compute based on layer roles.

1) Token-Level Reduction: The high resolution and temporal depth of video data result in an explosion of tokens in diffusion models. To address this, early representative work such as VidToMe [24] utilizes ToMe [151]’s bipartite matching algorithm to merge redundant self-attention tokens across video frames (Figure 7)). The algorithm partitions the tokens into a pair of source (src) and destination (dst) sets, and the source tokens are linked to their most similar tokens in the destination set. Specifically, VidToMe divides the video into frame chunks, mapping temporally correlated tokens from adjacent frames into a single target frame (intra-chunk local merging), and further links them with a persistent set of tokens across different chunks (inter-chunk global merging). Building on this foundation, importance-based token merging [152] argues that dst tokens are chosen randomly within predefined regions in existing methods, degrading generation quality due to the elimination of critical semantic details. To explicitly preserve informative regions, it leverages classifier-free guidance (CFG) [142] magnitudes as a cost-free indicator to construct a dynamic pool of important tokens and samples destination tokens from this pool. To overcome the distortion and pixelation caused by extending ToMe methods, AsymRnR [153] introduces a training-free asymmetric reduction strategy that independently merges Q,K,V tokens at adaptive rates tailored for specific network blocks and denoising timesteps. Designed for resource-constrained mobile environments, On-

To “Van Gogh Style”

[Figure 83]

[Figure 84]

|Video Chunks<br><br>……<br><br>| |
|---|
<br><br>Off-the-shelf Diffusion Model<br><br>Residual Block<br><br>Self Attention<br><br>Cross<br><br>| |Attention|
|---|---|
|Token Unmerging| |
<br><br>Token Merging<br><br>|
|---|

……

###### … …

…

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

11

DDIM Inversion

Denoising × 𝑇𝑇 Steps

Input Video

Edited Video

y achieving extreme

|𝑓𝑓0 𝑓𝑓2 𝑓𝑓3<br><br>𝑓𝑓1<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>4 × 𝑁𝑁 × 𝐶𝐶 Token Unmerging<br><br>hardware (e.g., RTX 5090), effectively compression with negligible quality loss.<br><br>2) Post-Training Quantization: lenging due to significant outliers in robust post-training quantization (PTQ) Q [163] serves as a representative main, which addresses oscillating|
|---|

|[Figure 97]<br><br>Global Token Merging<br><br>global tokens<br><br>update<br><br>|
|---|

|[Figure 98]<br><br>Local Token Merging<br><br>𝑓𝑓0 𝑓𝑓2 𝑓𝑓3<br><br>𝑓𝑓1<br><br>|
|---|

.

| |
|---|

| |
|---|

Quantizing DiTs is chalactivations, requiring ) techniques. ViDiTmethod in this do-

Self Attention

| |
|---|

| |
|---|

4 × 𝑁𝑁 × 𝐶𝐶

1 × 𝑀𝑀 × 𝐶𝐶

Token Merging

activations in video models through specialized metric-aware rounding. Subsequent work further refines these methods: DVD-Quant [160] extends quantization to a data-free setting by reconstructing calibration data to handle temporal dependencies, and LRQ-DiT [161] tackles the dual challenges of long-tailed, Gaussian-like weight distributions and diverse activation outliers by introducing twin-log quantization along with an adaptive rotation scheme. These methods collectively pave the way for the deployment of INT4-level DiTs in production environments.

Merging 𝑠𝑠𝑠𝑠𝑠𝑠 to 𝑑𝑑𝑠𝑠𝑑𝑑 matching

- Fig. 7. VidToMe [24] first merges tokens locally and then combines the merged tokens with the maintained global tokens. Figure courtesy of [24].

device Sora [154] introduces temporal dimension token merging (TDTM) that explicitly averages consecutive tokens along the temporal axis to effectively halve the sequence length for attention computation.

- 3) Quantization-Aware Training: While PTQ methods offer

efficient deployment, they often suffer from severe performance degradation when pushing video generation models to ultra-low precision (e.g., ≤ 4-bit). To bridge this performance gap, quantization-aware training (QAT) has emerged as a promising direction. As a pioneering work in this paradigm, QVGen [162] introduces a novel QAT framework tailored for video diffusion models under extremely low-bit settings (e.g., W4A4 and W3A3). Table II shows a performance comparison of some PTQ and QAT methods.

- 4) Dynamic and Temporal Quantization Strategies: Video

2) Structural Pruning: Beyond reducing token counts, structural pruning directly targets the model architecture to reduce computation. Early representative strategies target macrolevel depth and temporal redundancy. For static architectural pruning, MobileVD [155] introduces a learnable gating mechanism to prune redundant temporal blocks alongside a channel funneling strategy to compress layer widths during inference. VDMini [156] empirically observes that shallow layers primarily focus on individual frame content, while deeper layers dictate temporal motion dynamics; consequently, it selectively prunes redundant shallow blocks and restores generation quality through a loss based on individual content and motion dynamics (ICMD). SnapGen-V [157] also performs a lot of architecture searches. Contrasting with VDMini’s macro-level block removal, Mobile Video DiT [158] executes a sensitivityaware tri-level static pruning, simultaneously targeting blocks, attention heads, and FFN channels. Concurrently, UniCP [159] introduces finer dynamic pruning at the attention-matrix level.

generation involves temporal redundancy and multi-step denoising, offering opportunities for adaptive precision. Focusing on the temporal dimension, QuantCache [169] advances this concept by implementing an adaptive importance-guided quantization specifically for the KV cache and hierarchical latents, effectively exploiting the similarity between video frames to reduce memory bandwidth. Conversely, addressing the temporal heterogeneity across denoising timesteps, AdaTSQ [170] introduces a timestep-dynamic quantization framework. By leveraging Fisher information to evaluate the varying sensitivity of different diffusion phases, AdaTSQ dynamically allocates bit-widths via Pareto-aware beam search. Coupled with a Fisher-guided temporal calibration mechanism, this strategy pushes the Pareto frontier of efficiency and quality for video generation models.

D. Quantization

Quantization reduces the precision of weights and activations to accelerate inference and lower memory usage, which is critical for deploying large-scale video-based world models. We categorize recent advances into attention-centric optimization, post-training quantization, quantization-aware training, and dynamic scheduling strategies.

E. Discussion

1) Attention-Centric Quantization: Attention mechanisms dominate the computational cost in video generation models, driving a rapid evolution from 8-bit to 4-bit precision. Early representative works such as SageAttention [164] and FPSAttention [165] established the baseline; SageAttention employs 8-bit quantization with smooth matrix to handle outliers, while FPSAttention co-designs FP8 quantization with sparsity constraints. Building on this, the SageAttention series has pushed the limits of low-bit inference: SageAttention2 [166] achieves INT4 precision by introducing per-thread quantization and thorough outlier smoothing; SageAttention2++ [167] further optimizes kernel performance by utilizing faster FP8 matrix multiplication instructions accumulated in FP16. The most recent member in this series, SageAttention3 [168], introduces FP4 microscaling attention tailored for next-generation

Efficient video inference focuses on reducing per-step latency and scaling to high-fidelity, long-horizon generation. The four directions reviewed in this section address this problem from complementary perspectives: parallelism distributes computation across devices, caching reuses intermediate features across denoising steps, pruning removes redundant tokens or network components, and quantization reduces the precision cost of weights and activations. However, these techniques are not independent. For instance, aggressive caching or pruning may amplify approximation or accumulation errors in dynamic regions, while low-bit quantization can further destabilize activations already altered by token reduction or feature reuse. For video-based world models, this problem is more challenging because inference must support not only short clips,

TABLE II PERFORMANCE COMPARISON OF QUANTIZATION METHODS ON VBENCH [149] ACROSS MULTIPLE BASE MODELS. REPORTED BY DVD-QUANT [160], LRQ-DIT [161] AND QVGEN [162].

Bit-width (W/A)

Aesthetic Quality ↑

Imaging Quality ↑

Overall Consistency ↑

Scene Consistency ↑

Background Consistency ↑

Subject Consistency ↑

Dynamic Degree ↑

Motion

Method

Smoothness ↑ Base Model: HunyuanVideo [36]

BF16 Baseline 16/16 62.53 64.78 25.86 42.81 97.01 96.05 51.39 99.30 ViDiT-Q [163] 4/8 57.01 59.74 24.77 27.11 97.37 95.16 48.61 99.06 DVD-Quant [160] 4/6 62.27 64.22 25.83 33.07 97.89 96.57 58.33 99.05 ViDiT-Q [163] 4/4 45.36 40.10 19.66 7.85 97.19 97.29 0.00 99.43 DVD-Quant [160] 4/4 61.96 61.82 25.68 29.94 97.82 96.61 56.94 99.15

Base Model: Open-Sora 1.2 [150]

ViDiT-Q [163] 4/6 50.89 55.57 25.98 36.77 96.52 94.83 52.77 98.66 LRQ-DiT [161] 4/6 52.25 56.57 26.68 41.28 96.90 95.28 48.62 98.85

ViDiT-Q [163] 4/4 47.30 51.60 25.84 35.61 95.27 92.01 54.16 98.14 LRQ-DiT [161] 4/4 47.95 51.79 25.87 37.80 95.56 92.87 55.56 98.34

Base Model: CogVideoX-2B [32]

BF16 Baseline 16/16 54.49 59.15 25.06 36.24 94.79 92.82 67.78 97.43 ViDiT-Q [163] 4/6 43.01 54.72 20.41 26.25 90.76 81.02 43.22 92.18 QVGen [162] 4/4 54.61 60.16 24.61 31.42 94.38 93.01 67.22 98.06

TABLE III APPLICATIONS OF VIDEO-BASED WORLD MODELS

Application Data Synthesis Interactive Simulation Generative Planning

GAIA [18], [171], [172], DriveDreamer4D [173], InfinityDrive [174], Glad [175], STAGE [176], UniScene [177], WorldSplat [178], EOT-WM [179], WoVoGen [180],Cosmos-DriveDreams [181]

Drive-WM [182], Vista [183], MiLA [184], ADriver-I [185], [186], Drivedreamer [19], MagicDrive-V2 [40], DriveArena [187], MAD [188]

Epona [189], GenAD [190], DriveLaW [191], DrivingGPT [192], VaVAM [193]

Autonomous Driving

GR-1 [212], VILP [213], UVA [214], RoboEnvision [215], GEVRM [216], EnerVerse [217], LingBot-VA [218], Cosmos Policy [219],Fast-WAM [220],LeWorldModel [221],DreamZero [222]

Vidar [194], DreamGen [195], GenMimic [196], RBench [197], GigaWorld-0 [198], RIGVid [199], LuciBot [200], Gen2Act [201], Dreamitate [202]

World-Env [203], EVAC [204], Ctrl-World [205], VideoAgent [206], VIPER [207], WorldEval [208], Genie Envisioner [209], World-Gymnast [210], DreamDojo [211]

Embodied AI

GameGen-X [223], GameFactory [224], MineWorld [225], Matrix-Game [42], [226], GenieRedux-G [227], Hunyuan-GameCraft [228], [229], PlayGen [230], WorldPlay [231], Yume1.5 [129], LingBot-World [232], Cosmos-Predict2.5 [43], Dreamer 4 [233], Genie 3 [21]

Game & Interactive World Simulation

but also long-horizon and interactive generation. In practice, this means that parallelism, caching, pruning, and quantization should work together rather than be applied separately. Future methods should therefore improve both efficiency and stability, especially for long-duration interactive scenarios.

VI. APPLICATIONS

World modeling via efficient video generation has been widely applied to domains including autonomous driving, embodied AI, and interactive game simulation, supporting tasks such as data synthesis, interactive simulation, and generative planning (Table III). In such applications, online generation is often used to facilitate reinforcement learning, while offline data are more commonly used for supervised training.

- A. Autonomous Driving

1) Data Synthesis: In autonomous driving, video-based world models improve coverage of long-tail and safety-critical scenarios by generating realistic, controllable driving videos that can be used as synthetic training data for perception,

prediction, and planning, as well as evaluation data for testing robustness and safety under rare or hazardous conditions (Figure 8). The GAIA series [18], [171], [172] advances generative world modeling for autonomous driving: GAIA-

- 1 demonstrated that models can learn from video, text, and actions to generate realistic driving scenarios; GAIA-2 added stronger controllability, broader geographic coverage, and multi-camera scene generation across diverse vehicle embodiments. GAIA-3 combines the realism of real-world driving data with the controllability of simulation, allowing authentic driving sequences to be replayed with modifications-

—for example, altering the trajectory of the ego vehicle while making every other element in the scene consistent. DriveDreamer4D [173] leverages world-model priors to enhance 4D driving scene representations. InfinityDrive [174] introduces a spatiotemporal co-modeling module and an extended temporal training strategy, producing high-resolution spatiotemporally consistent videos.

- 2) Interactive Simulation: Some works integrate video-

based world models into closed-loop interaction pipelines, rolling out action-conditioned generation for interactive sim-

[Figure 99]

[Figure 100]

+5s

[Figure 101]

+10s

SunnyRainy

[Figure 102]

[Figure 103]

+15s

[Figure 104]

+20s

[Figure 105]

[Figure 106]

+15s

[Figure 107]

- Fig. 8. Examples of street-scene videos generated by MagicDrive-V2, which supports conditional generation with multiple types of control signals (e.g., road maps, object boxes, ego trajectories, and text). Figure courtesy of [40].

ulation and evaluation. Vista [183] generates realistic and temporally continuous videos at high spatiotemporal resolution and supports diverse behavior-conditioned control. MiLA [184] adopts a coarse-to-fine approach to stabilize video generation and correct distortions in dynamic objects. ADriverI [185] enables infinite autonomous driving within a virtual world created by a video generation model.

3) Generative Planning: Some works explore generative planning by using video-based world models to assist action selection during inference, while others leverage them as an auxiliary training objective. Drive-WM [182] can roll out multiple trajectories under different driving actions and select the best trajectory using image-based rewards. As an auxiliary training signal, Epona [189] explicitly integrates trajectory prediction into a video generation framework, using a dualbranch diffusion model to separately generate trajectories and video frames, and supports trajectory-only planning to improve real-time performance. GenAD [190] can generalize in a zero-shot manner to diverse unseen driving datasets, and be adapted into a motion planner or an action-conditioned generation model for future frames, highlighting its great value for real-world autonomous driving applications. By sharing a latent space, DriveLAW [191] treats the video model as a feature generator by directly injecting the latent representation produced by the Video DiT into the Action DiT. This chained design allows the planner (Action DiT) to better exploit the world modeling capability of the video model.

- B. Embodied AI

1) Data Synthesis: In embodied AI, video world models can serve as a data engine to augment training data, covering broader distributions and rare cases to improve policy generalization in dynamic and long-tail tasks (Figure 9). GigaWorld0 [198] modifies real-world videos through text-guided editing and promotes sim-to-real transfer, helping bridge the simulation-to-reality gap. DreamGen [195] forms a syntheticdata loop by turning world-model rollouts into trajectory-style supervision, enabling diverse sample generation based on data from a single real environment. To mitigate the sim-to-real

[Figure 108]

Fig. 9. Comparison of videos generated by video-based world models on the same robot action sequence. Figure courtesy of [210].

gap, GenMimic [196] first lifts videos of human movements to 4D reconstructions, then retargets the extracted human motion to humanoid embodiments, and finally trains reinforcement learning policies for robust motion imitation.

- 2) Interactive Simulation: As an interactive environment

simulator, a video-based world model can support stable action-conditioned rollout generation for reinforcement learning and facilitate real-time evaluation of generated trajectories, allowing safe and reproducible policy testing and improvement. World-Env [203] couples a video simulator with VLMguided reflection to provide dense rewards and completionbased termination; EVAC [204] generates multi-view, controllable observations as a low-cost evaluation proxy. CtrlWorld [205] provides a world simulator that enables robots to evaluate and improve their manipulation skills in a virtual environment. DreamDojo [211] is pretrained on 44k hours of human videos to learn physical dynamics without explicit action labels, and is then post-trained on robot data for downstream adaptation.

- 3) Generative Planning: Video world models can be ex-

tended to world action models (WAM) that support robot policy learning and action generation. One line of work jointly models future video frames and actions, leveraging shared representations between video generation and action prediction to improve policy learning and enhance scene dynamics understanding. DreamZero [222] builds a large-scale WAM based on a pretrained video diffusion backbone and jointly predicts future video frames and actions, showing strong zeroshot generalization, real-time closed-loop control, and crossembodiment transfer. UVA [214] models video frames and actions in a shared latent space with two lightweight diffusion decoders, enabling action-only inference and flexible task switching via random masking. Fast-WAM [220] provides evidence through controlled ablation studies that the gains of video world models stem primarily from the video co-training objective shaping physical representations during training, rather than from explicit future imagination at test time. An-

other line of work first generates future visual trajectories and then predicts actions according to the generated trajectories. VILP [213] learns action prediction from generated videos via imitation learning, while enabling real-time recedinghorizon control. RoboEnvision [215] generates keyframes via instruction decomposition plus interpolation for long-horizon consistency, then regresses joint controls with a lightweight policy. Based on JEPA [234], LeWorldModel [221] proposes a stable end-to-end latent world model that avoids representation collapse using only a prediction loss and a SIGReg regularizer enforcing Gaussian-distributed embeddings, enabling efficient latent planning with only 15M parameters.

- C. Game & Interactive World Simulation

Efficient video generation provides critical support for interactive world simulation, and games have become a common deployment setting because of their well-defined interaction interfaces and controllable closed-loop evaluation.

Among representative works, GameGen-X [223] targets open-world game videos, injecting keyboard actions and multimodal instructions into the generation process to improve interactive responsiveness over long sequences. GameFactory [224] models action control independently of the game genre to enable action-conditioned interactive video generation for diverse open-world scenarios. Focusing on Minecraft, MineWorld [225] increases interactive frame rates by alleviating the throughput bottleneck of autoregressive tokens via parallel decoding. Matrix-Game 2.0 [42], trained on data from GTA5 and Unreal Engine, reports interactive generation at around 25 frames per second and supports minute-level long rollouts. DreamerV4 [233] uses a video-based world model as an interactive environment for reinforcement learning, allowing the agent to practice complex long-horizon tasks.

Toward more general interactive world generation, existing methods typically combine streaming generation with contextual memory to support long-term exploration, and rely on architectural choices and inference acceleration to meet real-time requirements. WorldPlay [231] emphasizes highresolution real-time generation and long-term consistency under action conditioning. Yume1.5 [129] focuses on text controllability and event editing, reducing long-context latency through context compression and distillation. LingBotWorld [232] is an open-source world simulator that combines a hierarchical semantic data engine with multi-stage training for low-latency interaction and long-term memory.

- D. Discussion

Video generation models, empowered by strong world modeling capabilities, can predict future observations conditioned on actions or instructions. In autonomous driving and embodied AI, a clear trend is the gradual convergence of data generation and interactive simulation: during closed-loop interaction and rollout-based prediction, the model continuously produces new samples and hard cases, forming a “generate–evaluate–retrain” loop for policy training and shifting data provisioning from offline augmentation to online iteration. Meanwhile, video-based world models are also moving toward

[Figure 109]

Fig. 10. Comparison of videos generated by methods designed for realtime talking head generation. Existing methods generally preserve identity consistency well, but failure cases remain: Ditto tends to produce limited facial motion, while LiveAvatar may introduce local factual inconsistencies or artifacts (highlighted in red). Figure courtesy of [46].

supporting the full pipeline of data, model, and evaluation. Representative works such as Genie Envisioner [209], Cosmos [43], and LingBot [218], [232] attempt to integrate data generation, interactive simulation, feedback, and policy optimization within a single generative framework, reducing cross-platform adaptation costs and enabling more systematic and reproducible evaluation and training paradigms.

VII. MORE RELATED WORK A. Interactive Talking Head Generation

Recent advances in talking head generation have increasingly focused on interaction, streaming, and long-horizon conversation rather than traditional offline portrait synthesis.

An early representative method, INFP [235], explicitly models speaker–listener interaction in dyadic conversations, capturing speaking and listening behaviors within a shared motion latent space. In parallel, efficiency-oriented methods aim to reduce diffusion cost rather than redesign generation paradigms. For instance, Ditto [45] performs diffusion in a compact motion space to achieve controllable real-time synthesis, while OSA-LCM [236] compresses multi-step diffusion into a one-step latent consistency model to further accelerate expressive portrait generation. More recent work extends beyond portrait-level interaction toward streaming diffusion frameworks. InfiniteTalk [47] adopts a sparse-frame paradigm, where key reference frames anchor identity and motion style, while context frames enable stable long-horizon synthesis. Similarly, SoulX-FlashTalk [46] adopts self-correcting bidirectional distillation to preserve temporal coherence during long-form avatar streaming. At the system and architecture level, LiveAvatar [140] demonstrates that algorithm–system

co-design by exploiting timestep-forcing pipeline parallelism and the rolling sink mechanism can enable large-scale diffusion models to operate in real-time streaming settings, while StreamAvatar [237] proposes a two-stage autoregressive adaptation framework that converts non-causal human video diffusion models into block-causal streaming generators with improved long-term stability.

As illustrated in Figure 10, recent advances mark a clear paradigm shift from traditional offline portrait synthesis to causal and streaming real-time talking head generation, although challenges such as limited motion diversity or local artifacts still remain in existing approaches.

- B. Interactive Content Creation

For AI content creation [238], creators often iterate rapidly by repeatedly refining prompts or other input to rewrite structure, swap characters, or adjust shot pacing, making efficient video generation crucial for shifting from offline processing to interactive workflows.

For video editing, Edit-Your-Interest [239] caches and dynamically updates spatial attention feature tokens from previous frames, enabling cross-frame information utilization without explicit temporal modeling, thereby effectively reducing computational cost and memory consumption. DiTCtrl [240] performs tuning-free controllable editing using maskguided KV sharing and latent blending for smooth transitions across semantic segments. For story generation, TaleCrafter [241] modularizes story visualization into four interconnected components—story-to-prompt generation, text-tolayout generation, controllable text-to-image generation, and image-to-video animation—enabling interactive edits on intermediate representations and avoiding repeated end-to-end resampling. Animate-A-Story [242] adopts retrieval-augmented narrative synthesis to offload complex motion structure to retrieved priors. For controllable production such as virtual tryon, ViViD [243] extends diffusion to video with garment/pose encoders and hierarchical temporal modules to strengthen spatiotemporal coherence. PlayerOne [244] formulates egocentric video generation as motion-conditioned world modeling, introducing a joint reconstruction framework for 4D scenes and video frames that supports real-time first-person exploration while ensuring scene consistency and temporal continuity.

- C. Video-Driven Scene Generation

Video-driven scene generation methods leverage the spatial priors embedded in video generation models to synthesize more coherent and realistic 3D/4D environments.

Some approaches decompose the pipeline into two stages: video generation and 3D optimization. They first use a video model to synthesize a reference video or multi-view sequence, and then recover scene structure via techniques such as 4D Gaussians. VividDream [245] introduces a novel pipeline that first constructs and expands a static 3D scene according to an input image, then generates dynamic multi-view videos with a video diffusion model, and finally makes use of them to optimize an explorable 4D scene. Similarly, 4Real [246] and Free4D [247] first generate a temporally consistent reference

video and then expand the viewpoint range through frameconditioned video generation. These methods benefit from a stable modular pipeline; however, because video generation and geometric reconstruction are decoupled, errors can accumulate progressively.

Other approaches aim to jointly model spatiotemporal information within a unified model, directly generating representations that are consistent across time and viewpoints. A promising direction is to combine the geometric priors of feed-forward 3D reconstruction models with the generative capability of video diffusion models. Gen3R [248] unifies feed-forward 3D reconstruction with video diffusion in a shared latent space, enabling the joint generation of temporally consistent RGB videos and their 3D geometry within a single framework. Other approaches, such as CAT4D [249], address dynamic scene generation by first expanding a monocular video into dynamic multi-view videos with a video diffusion model and then optimizing a deformable 3D Gaussian representation to recover the final 4D scene. StarGen [250] conditions each temporal sliding window on the overlapping frame from the preceding sliding window to maintain temporal consistency, and on images covering the largest common spatial area in the scene with the current sliding window to improve spatial consistency during long-range generation. Due to their potential advantages in consistency and generation efficiency, these unified approaches are increasingly becoming an important direction for video-driven scene generation.

VIII. CONCLUSIONS

In this paper, we provide a comprehensive and systematic review of the critical intersection between efficiency improvement techniques and video-based world models. We explore how efficiency-oriented designs empower video-based world simulators in three primary dimensions: efficient modeling paradigms, efficient architectures, and efficient algorithms. In addition, we investigate how these efficient video generation frameworks directly enhance downstream applications such as autonomous driving, embodied AI, and gaming. Based on this review, we summarize challenges and future opportunities in this rapidly developing field, offering potential solutions for next-generation models facing increasingly complex physical dynamics and substantial computational demands.

REFERENCES

- [1] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014.
- [2] M. Saito, E. Matsumoto, and S. Saito, “Temporal generative adversarial nets with singular value clipping,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2830–2839.
- [3] N. Kalchbrenner, A. Oord, K. Simonyan, I. Danihelka, O. Vinyals, A. Graves, and K. Kavukcuoglu, “Video pixel networks,” in International Conference on Machine Learning. PMLR, 2017, pp. 1771– 1779.
- [4] W. Yan, Y. Zhang, P. Abbeel, and A. Srinivas, “Videogpt: Video generation using vq-vae and transformers,” arXiv preprint arXiv:2104.10157, 2021.
- [5] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zero-shot text-to-image generation,” in International conference on machine learning. Pmlr, 2021, pp. 8821–8831.

- [6] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695.
- [7] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” Advances in neural information processing systems, vol. 35, pp. 36479–36494, 2022.
- [8] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” Advances in neural information processing systems, vol. 35, pp. 8633–8646, 2022.
- [9] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/ forum?id=PxTIG12RRHS
- [10] A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and K. Kreis, “Align your latents: High-resolution video synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 22563–22575.
- [11] J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo et al., “Improving image generation with better captions,” Computer Science. https://cdn. openai. com/papers/dalle-3. pdf, vol. 2, no. 3, p. 8, 2023.
- [12] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” in The Twelfth International Conference on Learning Representations, 2024.
- [13] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. M¨uller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first International Conference on Machine Learning, 2024.
- [14] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman et al., “Video generation models as world simulators,” 2024. [Online]. Available: https: //openai.com/research/video-generation-models-as-world-simulators
- [15] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.
- [16] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, vol. 2, no. 3, p. 440, 2018.
- [17] S. Yang, Y. Du, S. K. S. Ghasemipour, J. Tompson, L. P. Kaelbling, D. Schuurmans, and P. Abbeel, “Learning interactive real-world simulators,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https: //openreview.net/forum?id=sFyTZEqmUY
- [18] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado, “Gaia-1: A generative world model for autonomous driving,” arXiv preprint arXiv:2309.17080, 2023.
- [19] X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu, “Drivedreamer: Towards real-world-drive world models for autonomous driving,” in European conference on computer vision. Springer, 2024, pp. 55–72.
- [20] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel, “Learning universal policies via text-guided video generation,” Advances in neural information processing systems, vol. 36, pp. 9156–9172, 2023.
- [21] Google DeepMind. Genie 3. Google DeepMind. [Online]. Available: https://deepmind.google/models/genie/
- [22] W. Li, W. Pan, P.-C. Luan, Y. Gao, and A. Alahi, “Stable video infinity: Infinite-length video generation with error recycling,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=X96Ei9n34a
- [23] Y. Oshima, Y. Iwasawa, M. Suzuki, Y. Matsuo, and H. Furuta, “Worldpack: Compressed memory improves spatial consistency in video world modeling,” arXiv preprint arXiv:2512.02473, 2025.
- [24] X. Li, C. Ma, X. Yang, and M.-H. Yang, “Vidtome: Video token merging for zero-shot video editing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7486–7495.
- [25] J. Chen, W. He, Y. Gu, Y. Zhao, J. Yu, J. Chen, D. Zou, Y. Lin, Z. Zhang, M. Li et al., “Dc-videogen: Efficient video generation with deep compression video autoencoder,” arXiv preprint arXiv:2509.25182, 2025.
- [26] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33,

pp. 6840–6851, 2020.

- [27] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=PqvMRDCJT9t
- [28] X. Liu, X. Zhang, J. Ma, J. Peng, and qiang liu, “Instaflow: One step is enough for high-quality diffusion-based text-to-image generation,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/ forum?id=1k4yZbbDqX
- [29] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12873–12883.
- [30] X. Ma, Y. Wang, X. Chen, G. Jia, Z. Liu, Y.-F. Li, C. Chen, and Y. Qiao, “Latte: Latent diffusion transformer for video generation,” Transactions on Machine Learning Research, 2025. [Online]. Available: https://openreview.net/forum?id=ntGPYNUF3t
- [31] L. Yu, Y. Cheng, K. Sohn, J. Lezama, H. Zhang, H. Chang, A. G. Hauptmann, M.-H. Yang, Y. Hao, I. Essa et al., “Magvit: Masked generative video transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10459–10469.
- [32] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-tovideo diffusion models with an expert transformer,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=LQzN6TRFg9
- [33] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.
- [34] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.
- [35] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of machine learning research, vol. 21, no. 140, pp. 1–67, 2020.
- [36] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu et al., “Hunyuanvideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024.
- [37] Y. Gao, H. Guo, T. Hoang, W. Huang, L. Jiang, F. Kong, H. Li, J. Li, L. Li, X. Li et al., “Seedance 1.0: Exploring the boundaries of video generation models,” arXiv preprint arXiv:2506.09113, 2025.
- [38] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.
- [39] B. Wu, C. Zou, C. Li, D. Huang, F. Yang, H. Tan, J. Peng, J. Wu, J. Xiong, J. Jiang et al., “Hunyuanvideo 1.5 technical report,” arXiv preprint arXiv:2511.18870, 2025.
- [40] R. Gao, K. Chen, B. Xiao, L. Hong, Z. Li, and Q. Xu, “Magicdrivev2: High-resolution long video generation for autonomous driving with adaptive control,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 28135–28144.
- [41] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps et al., “Genie: Generative interactive environments,” in Forty-first International Conference on Machine Learning, 2024.
- [42] X. He, C. Peng, Z. Liu, B. Wang, Y. Zhang, Q. Cui, F. Kang, B. Jiang et al., “Matrix-game 2.0: An open-source real-time and streaming interactive world model,” arXiv preprint arXiv:2508.13009, 2025.
- [43] A. Ali, J. Bai, M. Bala, Y. Balaji, A. Blakeman, T. Cai, J. Cao, T. Cao, E. Cha, Y.-W. Chao et al., “World simulation with video foundation models for physical ai,” arXiv preprint arXiv:2511.00062, 2025.
- [44] L. Tian, Q. Wang, B. Zhang, and L. Bo, “Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions,” in European Conference on Computer Vision. Springer, 2024, pp. 244–260.
- [45] T. Li, R. Zheng, M. Yang, J. Chen, and M. Yang, “Ditto: Motionspace diffusion for controllable realtime talking head synthesis,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 9704–9713.
- [46] L. Shen, Q. Qiao, T. Yu, K. Zhou, T. Yu, Y. Zhan, Z. Wang, M. Tao, S. Yin, and S. Liu, “Soulx-flashtalk: Real-time infinite streaming of audio-driven avatars via self-correcting bidirectional distillation,”

2025. [Online]. Available: https://arxiv.org/abs/2512.23379

- [47] S. Yang, Z. Kong, F. Gao, M. Cheng, X. Liu, Y. Zhang, Z. Kang, W. Luo, X. Cai, R. He, and X. Wei, “Infinitetalk: Audio-driven video generation for sparse-frame video dubbing,” 2025. [Online]. Available: https://arxiv.org/abs/2508.14033
- [48] L. Zheng, Y. Zhang, H. Guo, J. Pan, Z. Tan, J. Lu, C. Tang, B. An, and S. YAN, “MEMO: Memory-guided diffusion for expressive talking video generation,” Transactions on Machine Learning Research, 2026, j2C Certification. [Online]. Available: https://openreview.net/forum?id=uBcHcM7Kzi
- [49] H. Yi, T. Ye, S. Shao, X. Yang, J. Zhao, H. Guo, T. Wang, Q. Yin, Z. Xie, L. Zhu et al., “Magicinfinite: Generating infinite talking videos with your words and voice,” arXiv preprint arXiv:2503.05978, 2025.
- [50] J. Wu, S. Yin, N. Feng, X. He, D. Li, J. HAO, and M. Long, “ivideoGPT: Interactive videoGPTs are scalable world models,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id= 4TENzBftZR
- [51] T. Salimans and J. Ho, “Progressive distillation for fast sampling of diffusion models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/ forum?id=TIdIXIpzhoI
- [52] K. Frans, D. Hafner, S. Levine, and P. Abbeel, “One step diffusion via shortcut models,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=OlzB6LnXcS
- [53] X. Liang, Y. Zhang, and L. Zhu, “Gpd: Guided progressive distillation for fast and high-quality video generation,” arXiv preprint arXiv:2602.01814, 2026.
- [54] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in Proceedings of the 40th International Conference on Machine Learning, 2023, pp. 32211–32252.
- [55] S. Luo, Y. Tan, L. Huang, J. Li, and H. Zhao, “Latent consistency models: Synthesizing high-resolution images with few-step inference,” arXiv preprint arXiv:2310.04378, 2023.
- [56] X. Wang, S. Zhang, H. Zhang, Y. Liu, Y. Zhang, C. Gao, and N. Sang, “Videolcm: Video latent consistency model,” arXiv preprint arXiv:2312.09109, 2023.
- [57] F.-Y. Wang, Z. Huang, W. Bian, X. Shi, K. Sun, G. Song, Y. Liu, and H. Li, “Animatelcm: Computation-efficient personalized style video generation without personalized video data,” in SIGGRAPH Asia 2024 Technical Communications, 2024, pp. 1–5.
- [58] J. Zhang, K. Zheng, K. Jiang, H. Wang, I. Stoica, J. E. Gonzalez, J. Chen, and J. Zhu, “Turbodiffusion: Accelerating video diffusion models by 100-200 times,” arXiv preprint arXiv:2512.16093, 2025.
- [59] F. Team, “Fastvideo,” 2025, https://huggingface.co/FastVideo.
- [60] T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, and T. Park, “One-step diffusion with distribution matching distillation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 6613–6623.
- [61] T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and B. Freeman, “Improved distribution matching distillation for fast image synthesis,” Advances in neural information processing systems, vol. 37, pp. 47455–47487, 2024.
- [62] Y. Zhu, H. Yan, H. Yang, K. Zhang, and J. Li, “Accelerating video diffusion models via distribution matching,” arXiv preprint arXiv:2412.05899, 2024.
- [63] S. Lin, X. Xia, Y. Ren, C. Yang, X. Xiao, and L. Jiang, “Diffusion adversarial post-training for one-step video generation,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=AAgzsnhc28
- [64] D. Kondratyuk, L. Yu, X. Gu, J. Lezama, J. Huang, G. Schindler, R. Hornung, V. Birodkar, J. Yan, M.-C. Chiu et al., “Videopoet: A large language model for zero-shot video generation,” in Forty-first International Conference on Machine Learning, 2024. [Online]. Available: https://openreview.net/forum?id=LRkJwPIDuE
- [65] Y. Wang, T. Xiong, D. Zhou, Z. Lin, Y. Zhao, B. Kang, J. Feng, and X. Liu, “Loong: Generating minute-level long videos with autoregressive language models,” arXiv preprint arXiv:2410.02757, 2024.
- [66] D. Xie, Z. Xu, Y. Hong, H. Tan, D. Liu, F. Liu, A. Kaufman, and Y. Zhou, “Progressive autoregressive video diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2025, pp. 6376–6386.
- [67] L. Zhang, S. Cai, M. Li, G. Wetzstein, and M. Agrawala, “Frame context packing and drift prevention in next-frame-prediction video diffusion models,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=J8JCF64aEn

- [68] T. Yin, Q. Zhang, R. Zhang, W. T. Freeman, F. Durand, E. Shechtman, and X. Huang, “From slow bidirectional to fast autoregressive video diffusion models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 22963–22974.
- [69] B. Chen, D. M. Mons´o, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann, “Diffusion forcing: Next-token prediction meets full-sequence diffusion,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=yDo1ynArjj
- [70] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman, “Self forcing: Bridging the train-test gap in autoregressive video diffusion,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id= mSiN7i0BYH
- [71] K. Liu, W. Hu, J. Xu, Y. Shan, and S. Lu, “Rolling forcing: Autoregressive long video diffusion in real time,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=IAyzXjbfwo
- [72] J. Cui, J. Wu, M. Li, T. Yang, X. Li, R. Wang, A. Bai, Y. Ban, and C.-J. Hsieh, “Self-forcing++: Towards minute-scale high-quality video generation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=DzvPiqh23f
- [73] Y. Lu, Y. Zeng, H. Li, H. Ouyang, Q. Wang, K. L. Cheng, J. Zhu, H. Cao, Z. Zhang, X. Zhu et al., “Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation,” arXiv preprint arXiv:2512.04678, 2025.
- [74] H. Zhu, M. Zhao, G. He, H. Su, C. Li, and J. Zhu, “Causal forcing: Autoregressive diffusion distillation done right for high-quality realtime interactive video generation,” arXiv preprint arXiv:2602.02214, 2026.
- [75] H. Li, S. Liu, Z. Lin, and M. Chandraker, “Rolling sink: Bridging limited-horizon training and open-ended testing in autoregressive video diffusion,” arXiv preprint arXiv:2602.07775, 2026. [Online]. Available: https://arxiv.org/abs/2602.07775
- [76] S. Lin, C. Yang, H. He, J. Jiang, Y. Ren, X. Xia, Y. Zhao, X. Xiao, and L. Jiang, “Autoregressive adversarial post-training for real-time interactive video generation,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=lF6SHARvmG
- [77] S. Yang, W. Huang, R. Chu, Y. Xiao, Y. Zhao, X. Wang, M. Li, E. Xie, Y.-C. Chen, Y. Lu et al., “Longlive: Real-time interactive long video generation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=nCAODkpsPJ
- [78] J. Yi, W. Jang, P. H. Cho, J. Nam, H. Yoon, and S. Kim, “Deep forcing: Training-free long video generation with deep sink and participative compression,” arXiv preprint arXiv:2512.05081, 2025.
- [79] Y. Jin, Z. Sun, N. Li, K. Xu, K. Xu, H. Jiang, N. Zhuang, Q. Huang, Y. Song, Y. MU, and Z. Lin, “Pyramidal flow matching for efficient video generative modeling,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=66NzcRQuOq
- [80] L. Ran and M. Z. Shou, “TPDiff: Temporal pyramid video diffusion model,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=Eg3KqoI9tS
- [81] Y. Zhang, H. Yang, Y. Zhang, Y. Hu, F. Zhu, C. Lin, X. Mei, Y. Jiang, B. Peng, and Z. Yuan, “Waver: Wave your way to lifelike video generation,” arXiv preprint arXiv:2508.15761, 2025.
- [82] S. Zhang, W. Li, S. Chen, C. Ge, P. Sun, Y. Zhang, Y. Jiang, Z. Yuan, B. Peng, and P. Luo, “Flashvideo: Flowing fidelity to detail for efficient high-resolution video generation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 15, 2026, pp. 12735– 12743.
- [83] F. Ye, Z. Zhao, Y. Mu, J. Shen, R. Li, K. Wang, D. Sun, S. Agarwal, M. Lee, T. Cao et al., “Supergen: An efficient ultra-high-resolution video generation system with sketching and tiling,” arXiv preprint arXiv:2508.17756, 2025.
- [84] S. Du, M. Xia, C. Liu, X. Wang, J. Wang, P. Wan, D. Zhang, and X. Ji, “Patchvsr: Breaking video diffusion resolution limits with patch-wise video super-resolution,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 17799–17809.
- [85] S. Cheng, Y. Wei, L. Diao, Y. Liu, B. Chen, L. Huang, Y. Liu, W. Yu, J. Du, W. Lin et al., “Srdiffusion: Accelerate video diffusion inference via sketching-rendering cooperation,” arXiv preprint arXiv:2505.19151, 2025.

- [86] S. Anagnostidis, G. Bachmann, Y. Kim, J. Kohler, M. Georgopoulos, A. Sanakoyeu, Y. Du, A. Pumarola, A. Thabet, and E. Sch¨onfeld, “Flexidit: Your diffusion transformer can easily generate high-quality samples with less compute,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 28316–28326.
- [87] Y. Zhang, L. Mai, A. Mahapatra, D. Bourgin, Y. Hong, J. Casebeer, F. Liu, and Y. Fu, “Regen: Learning compact video embedding with (re-)generative decoder,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 18453– 18462.
- [88] Z. Yuan, S. Wang, Y. Shang, H. Zhang, T. Fang, R. Xie, S. Yan, G. Dai, and Y. Wang, “Dlfr-vae: Dynamic latent frame rate vae for video generation,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 10388–10397.
- [89] Z. Yuan, R. Xie, Y. Shang, H. Zhang, S. Wang, S. Yan, G. Dai, and Y. Wang, “Dlfr-gen: Diffusion-based video generation with dynamic latent frame rate,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 16410– 16419.
- [90] Y. Zou, J. Yao, S. Yu, S. Zhang, W. Liu, and X. Wang, “Turbovaed: Fast and stable transfer of video-vaes to mobile devices,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 16, 2026, pp. 14086–14094.
- [91] K. Zhang, L. Jiang, A. Wang, J. Z. Fang, T. Zhi, Q. Yan, H. Kang, X. Lu, and X. Pan, “Storymem: Multi-shot long video storytelling with memory,” arXiv preprint arXiv:2512.19539, 2025.
- [92] Y. Zhu, F. Jiaqi, W. Zheng, Y. Gao, X. Tao, P. Wan, J. Lu, and J. Zhou, “Astra: General interactive world model with autoregressive denoising,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=8UZpmrxoLG
- [93] J. Wang, L. Ye, T. Lu, J. Xiao, J. Zhang, Y. Guo, X. Liu, R. Chellappa, C. Peng, A. Yuille et al., “Evoworld: Evolving panoramic world generation with explicit 3d memory,” arXiv preprint arXiv:2510.01183, 2025.
- [94] R. Li, P. Torr, A. Vedaldi, and T. Jakab, “Vmem: Consistent interactive video scene generation with surfel-indexed view memory,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 25690–25699.
- [95] T. Huang, W. Zheng, T. Wang, Y. Liu, Z. Wang, J. Wu, J. Jiang, H. Li, R. Lau, W. Zuo et al., “Voyager: Long-range and world-consistent video diffusion for explorable 3d scene generation,” ACM Transactions on Graphics (TOG), vol. 44, no. 6, pp. 1–15, 2025.
- [96] T. Wu, S. Yang, R. Po, Y. Xu, Z. Liu, D. Lin, and G. Wetzstein, “Video world models with long-term spatial memory,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=HbTxc6U1fO
- [97] J. Huang, X. Hu, B. Han, S. Shi, Z. Tian, T. He, and L. Jiang, “Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft,” arXiv preprint arXiv:2510.03198, 2025.
- [98] S. Ji, X. Chen, S. Yang, X. Tao, P. Wan, and H. Zhao, “Memflow: Flowing adaptive memory for consistent and efficient long video narratives,” arXiv preprint arXiv:2512.14699, 2025.
- [99] Y. Yu, X. Wu, X. Hu, T. Hu, Y. Sun, X. Lyu, B. Wang, L. Ma, Y. Ma, Z. Wang et al., “Videossm: Autoregressive long video generation with hybrid state-space memory,” arXiv preprint arXiv:2512.04519, 2025.
- [100] J. Yu, J. Bai, Y. Qin, Q. Liu, X. Wang, P. Wan, D. Zhang, and X. Liu, “Context as memory: Scene-consistent interactive long video generation with memory retrieval,” in Proceedings of the SIGGRAPH Asia 2025 Conference Papers, 2025, pp. 1–11.
- [101] J. Jiang, W. Li, J. Ren, Y. Qiu, Y. Guo, X. Xu, H. Wu, and W. Zuo, “Lovic: Efficient long video generation with context compression,” arXiv preprint arXiv:2507.12952, 2025.
- [102] S. Cai, C. Yang, L. Zhang, Y. Guo, J. Xiao, Z. Yang, Y. Xu, Z. Yang, A. Yuille, L. Guibas et al., “Mixture of contexts for long video generation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=y6XJZlEC2x
- [103] Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Chen, X. Wang, S. Koyejo et al., “Learning to (learn at test time): RNNs with expressive hidden states,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=wXfuOj9C7L
- [104] K. Dalal, D. Koceja, J. Xu, Y. Zhao, S. Han, K. C. Cheung, J. Kautz, Y. Choi, Y. Sun, and X. Wang, “One-minute video generation with test-time training,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 17702–17711.

- [105] T. Zhang, S. Bi, Y. Hong, K. Zhang, F. Luan, S. Yang, K. Sunkavalli, W. T. Freeman, and H. Tan, “Test-time training done right,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/ forum?id=Tb9qAxT3xv
- [106] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Re, C. Barrett et al., “H2o: Heavy-hitter oracle for efficient generative inference of large language models,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=RkRrPp7GKO
- [107] H. Xi, S. Yang, Y. Zhao, C. Xu, M. Li, X. Li, Y. Lin, H. Cai, J. Zhang, D. Li et al., “Sparse video-gen: Accelerating video diffusion transformers with spatial-temporal sparsity,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=u8CA3qIS0V
- [108] S. Yang, H. Xi, Y. Zhao, M. Li, J. Zhang, H. Cai, Y. Lin, X. Li, C. Xu, K. Peng et al., “Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=WPU17d1l7R
- [109] P. Chen, X. Zeng, M. Zhao, M. Shen, W. Cheng, G. Yu, and T. Chen, “Sparse-vdit: Unleashing the power of sparse attention to accelerate video diffusion transformers,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 4, 2026, pp. 2957–2965.
- [110] Q. Li, G. Zheng, Q. Zhao, J. Li, B. Dong, Y. Yao, and X. Li, “Compact attention: Exploiting structured spatio-temporal sparsity for fast video generation,” arXiv preprint arXiv:2508.12969, 2025.
- [111] X. Li, M. Li, T. Cai, H. Xi, S. Yang, Y. Lin, L. Zhang, S. Yang, J. Hu, K. Peng et al., “Radial attention: $\mathcal o(n \log n)$ sparse attention for long video generation,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=hYovE4nHTt
- [112] J. Wu, L. Hou, H. Yang, Y. Tian, P. Wan, D. ZHANG, and Y. Tong, “VMoBA: Mixture-of-block attention for video diffusion models,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=oQaRElUdmh
- [113] E. Lu, Z. Jiang, J. Liu, Y. Du, T. Jiang, C. Hong, S. Liu, W. He, E. Yuan, Y. Wang et al., “MoBA: Mixture of block attention for long-context LLMs,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=RlqYCpTu1P
- [114] Y. Gu, X. LI, Y. Hu, C. Minqi, and B. Zhuang, “BLADE: Block-sparse attention meets step distillation for efficient video generation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/ forum?id=O9J20MsmRl
- [115] J. Zhang, C. Xiang, H. Huang, J. wei, H. Xi, J. Zhu, and J. Chen, “Spargeattention: Accurate and training-free sparse attention accelerating any model inference,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https: //openreview.net/forum?id=74c3Wwk8Tc
- [116] J. Zhang, K. Jiang, C. Xiang, W. Feng, Y. Hu, H. Xi, J. Chen, and J. Zhu, “Spargeattention2: Trainable sparse attention via hybrid top-k+ top-p masking and distillation fine-tuning,” arXiv preprint arXiv:2602.13515, 2026.
- [117] Y. Zhang, J. Xing, B. Xia, S. Liu, B. PENG, X. Tao, P. Wan, E. Lo, and J. Jia, “Training-free efficient video generation via dynamic token carving,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=CdkFnJSG4G
- [118] S. Durvasula, K. Sreedhar, Z. Moustafa, S. Kothawade, A. Gondimalla, S. Subramanian, N. Shahidi, and N. Vijaykumar, “Fg-attn: Leveraging fine-grained sparsity in diffusion transformers,” arXiv preprint arXiv:2509.16518, 2025.
- [119] C. Zhan, W. Li, C. Shen, J. Zhang, S. Wu, and H. Zhang, “Bidirectional sparse attention for faster video diffusion training,” arXiv preprint arXiv:2509.01085, 2025.
- [120] H. Liu, Y. Cheng, W. Miao, Z. Liu, A. Chen, J. Lin, Y. Yao, C. Chen, J. Leng, Y. Feng, and M. Guo, “ASTRAEA: A token-wise acceleration framework for video diffusion transformers,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=e8P4Oo8S6U
- [121] Z. Yuan, H. Zhang, L. Pu, X. Ning, L. Zhang, T. Zhao, S. Yan, G. Dai, and Y. Wang, “DiTFastattn: Attention compression for diffusion transformer models,” in The Thirty-eighth Annual Conference

- on Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=51HQpkQy3t
- [122] T. Hu, J. Zhang, Z. Su, and R. Yi, “Ultragen: High-resolution video generation with hierarchical attention,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 6, 2026, pp. 4923– 4931.
- [123] W. Sun, R.-C. Tu, Y. Ding, J. Liao, Z. Jin, S. Liu, and D. Tao, “VORTA: Efficient video diffusion via routing sparse attention,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https: //openreview.net/forum?id=gY9yOGYB48
- [124] E. Song, W. Chai, S. Yang, E. J. Armand, X. Shan, H. Xu, J. Xie, and Z. Tu, “VideoNSA: Native sparse attention scales video understanding,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=zA2LbsUMDd
- [125] P. Zhang, Y. Chen, R. Su, H. Ding, I. Stoica, Z. Liu, and H. Zhang, “Fast video generation with sliding tile attention,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=U74MOXPEJd
- [126] J. Chen, Y. Zhao, J. YU, R. Chu, J. Chen, S. Yang, X. Wang, Y. Pan, D. Zhou, H. Ling et al., “SANA-video: Efficient video generation with block linear diffusion transformer,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=mzAchylAtf
- [127] Y. Huang, X. Ge, R. Gong, C. Lv, and J. Zhang, “Linvideo: A post-training framework towards o (n) attention in efficient video generation,” arXiv preprint arXiv:2510.08318, 2025.
- [128] J. Zhang, H. Wang, K. Jiang, S. Yang, K. Zheng, H. Xi, Z. Wang, H. Zhu, M. Zhao, I. Stoica et al., “SLA: Beyond sparsity in diffusion transformers via fine-tunable sparse–linear attention,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=eD8IPvNoZB
- [129] X. Mao, Z. Li, C. Li, X. Xu, K. Ying, T. He, J. Pang, Y. Qiao, and K. Zhang, “Yume-1.5: A text-controlled interactive world generation model,” arXiv preprint arXiv:2512.22096, 2025.
- [130] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in First Conference on Language Modeling, 2024. [Online]. Available: https://openreview.net/forum?id= tEYskw1VY2
- [131] Y. Fu, C. Chen, and Y. Yu, “Lamamba-diff: Linear-time high-fidelity diffusion models based on local attention and mamba,” in 36th British Machine Vision Conference 2025, BMVC 2025, Sheffield, UK, November 24-27, 2025. BMVA, 2025. [Online]. Available: https: //bmva-archive.org.uk/bmvc/2025/assets/papers/Paper 962/paper.pdf

- [132] H. Wang, C.-Y. Ma, Y.-C. Liu, J. Hou, T. Xu, J. Wang, F. JuefeiXu, Y. Luo, P. Zhang, T. Hou et al., “Lingen: Towards high-resolution minute-length text-to-video generation with linear computational complexity,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 2578–2588.
- [133] T. Dao and A. Gu, “Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality,” in Forty-first International Conference on Machine Learning, 2024. [Online]. Available: https://openreview.net/forum?id=ztn8FCR1td
- [134] M. Zhao, G. He, Y. Chen, H. Zhu, C. Li, and J. Zhu, “RIFLEx: A free lunch for length extrapolation in video diffusion transformers,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=v3B79m7t8Z
- [135] M. Zhao, H. Zhu, Y. Wang, B. Yan, J. Zhang, G. He, L. Yang, C. Li, and J. Zhu, “Ultravico: Breaking extrapolation limits in video diffusion transformers,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=fLLCmC53u9
- [136] H. Yesiltepe, T. H. S. Meral, A. K. Akan, K. Oktay, and P. Yanardag, “Infinity-rope: Action-controllable infinite video generation emerges from autoregressive self-rollout,” arXiv preprint arXiv:2511.20649, 2025.
- [137] H. Qiu, M. Xia, Y. Zhang, Y. He, X. Wang, Y. Shan, and Z. Liu, “Freenoise: Tuning-free longer video diffusion via noise rescheduling,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/ forum?id=ijoqFqSC7p
- [138] M. Li, T. Cai, J. Cao, Q. Zhang, H. Cai, J. Bai, Y. Jia, K. Li, and S. Han, “Distrifusion: Distributed parallel inference for high-resolution diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 7183–7193.

- [139] Z. Tan, X. Yang, S. Liu, and X. Wang, “Video-infinity: Distributed long video generation,” arXiv preprint arXiv:2406.16260, 2024.
- [140] Y. Huang, H. Guo, F. Wu, S. Zhang, S. Huang, Q. Gan, L. Liu, S. Zhao, E. Chen, J. Liu et al., “Live avatar: Streaming real-time audio-driven avatar generation with infinite length,” arXiv preprint arXiv:2512.04677, 2025.
- [141] J. Fang, J. Pan, X. Sun, A. Li, and J. Wang, “xdit: an inference engine for diffusion transformers (dits) with massive parallelism,” arXiv preprint arXiv:2411.01738, 2024.
- [142] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.
- [143] X. Zhao, X. Jin, K. Wang, and Y. You, “Real-time video generation with pyramid attention broadcast,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=hDBrQ4DApF
- [144] F. Liu, S. Zhang, X. Wang, Y. Wei, H. Qiu, Y. Zhao, Y. Zhang, Q. Ye, and F. Wan, “Timestep embedding tells: It’s time to cache for video diffusion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 7353–7363.
- [145] Z. Lv, C. Si, J. Song, Z. Yang, Y. Qiao, Z. Liu, and K.-Y. K. Wong, “Fastercache: Training-free video diffusion model acceleration with high quality,” arXiv preprint arXiv:2410.19355, 2025.
- [146] J. Wang, K. Zhao, J. Guo, J. Wang, H. Guo, C. Zhu, X. Li, and X. Yue, “Precisecache: Precise feature caching for efficient and high-fidelity video generation,” arXiv preprint arXiv:2603.00976, 2026.
- [147] Q. Song, X. Wang, D. Zhou, J. Lin, C. Chen, Y. Ma, and X. Li, “Hero: Hierarchical extrapolation and refresh for efficient world models,” arXiv preprint arXiv:2508.17588, 2025.
- [148] W. Feng, G. Fan, H. Qin, C. Yang, M. Wu, Y. Li, X. Li, Z. An, L. Huang, D. Wang et al., “Worldcache: Accelerating world models for free via heterogeneous token caching,” arXiv preprint arXiv:2603.06331, 2026.
- [149] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 21807–21818.
- [150] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You, “Open-sora: Democratizing efficient video production for all,” arXiv preprint arXiv:2412.20404, 2024.
- [151] D. Bolya and J. Hoffman, “Token merging for fast stable diffusion,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 4599–4603.
- [152] H. Wu, J. Xu, H. Le, and D. Samaras, “Importance-based token merging for efficient image and video generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 4983–4995.
- [153] W. Sun, R.-C. Tu, J. Liao, Z. Jin, and D. Tao, “Asymrnr: Video diffusion transformers acceleration with asymmetric reduction and restoration,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=5PiZevq9fY
- [154] B. Kim, K. Lee, I. Jeong, J. Cheon, Y. Lee, and S. Lee, “On-device sora: Enabling training-free diffusion-based text-to-video generation for mobile devices,” arXiv preprint arXiv:2502.04363, 2025.
- [155] H. Ben Yahia, D. Korzhenkov, I. Lelekas, A. Ghodrati, and A. Habibian, “Mobile video diffusion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 19450–19460.
- [156] Y. Wu, Z. Chen, H. Wang, and D. Xu, “Individual content and motion dynamics preserved pruning for video diffusion models,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 9714–9723.
- [157] Y. Wu, Z. Zhang, Y. Li, Y. Xu, A. Kag, Y. Sui, H. Coskun, K. Ma, A. Lebedev, J. Hu et al., “Snapgen-v: Generating a five-second video within five seconds on a mobile device,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 2479– 2490.
- [158] Y. Wu, Y. Li, A. Kag, I. Skorokhodov, W. Menapace, K. Ma, A. Sahni, J. Hu, A. Siarohin, D. Sagar et al., “Taming diffusion transformer for efficient mobile video generation in seconds,” arXiv preprint arXiv:2507.13343, 2025.
- [159] W. Sun, Q. Hou, D. Di, J. Yang, Y. Ma, and J. Cui, “Unicp: A unified caching and pruning framework for efficient video generation,” in Proceedings of the 7th ACM International Conference on Multimedia in Asia, 2025, pp. 1–7.

- [160] Z. Li, H. Li, J. Wu, K. Liu, H. Qin, L. Kong, G. Chen, Y. Zhang, and X. Yang, “DVD-quant: Data-free video diffusion transformers quantization,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=3AnRMvlVDw
- [161] L. Yang, H. Lin, T. Zhao, Y. Wu, H. Zhu, R. Xie, Z. Sun, Y. Wang, and Q. Gu, “Lrq-dit: Log-rotation post-training quantization of diffusion transformers for image and video generation,” arXiv preprint arXiv:2508.03485, 2025.
- [162] Y. Huang, R. Gong, J. Liu, Y. Ding, C. Lv, H. Qin, and J. Zhang, “QVGen: Pushing the limit of quantized video generative models,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/ forum?id=XJXZXuTj11
- [163] T. Zhao, T. Fang, H. Huang, R. Wan, W. Soedarmadji, E. Liu, S. Li, Z. Lin, G. Dai, S. Yan et al., “Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https: //openreview.net/forum?id=E1N1oxd63b
- [164] J. Zhang, J. wei, P. Zhang, J. Zhu, and J. Chen, “Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=OL44KtasKc
- [165] A. Liu, Z. Zhang, Z. Li, X. Bai, Y. Xing, Y. Han, J. Tang, J. Wu, M. Yang, W. Chen et al., “FPSAttention: Training-aware FP8 and sparsity co-design for fast video diffusion,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=T62TYoF8R3
- [166] J. Zhang, H. Huang, P. Zhang, J. wei, J. Zhu, and J. Chen, “Sageattention2: Efficient attention with thorough outlier smoothing and per-thread INT4 quantization,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https: //openreview.net/forum?id=nC8XliUxeg
- [167] J. Zhang, X. Xu, J. Wei, H. Huang, P. Zhang, C. Xiang, J. Zhu, and J. Chen, “Sageattention2++: A more efficient implementation of sageattention2,” arXiv preprint arXiv:2505.21136, 2025.
- [168] J. Zhang, J. wei, H. Wang, P. Zhang, X. Xu, H. Huang, K. Jiang, J. Zhu, and J. Chen, “Sageattention3: Microscaling FP4 attention for inference and an exploration of 8-bit training,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=JbJVWljk7r
- [169] J. Wu, Z. Li, Z. Hui, Y. Zhang, L. Kong, and X. Yang, “Quantcache: Adaptive importance-guided quantization with hierarchical latent and layer caching for video generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 15035–15044.
- [170] S. Zhang, Z. Ding, K. Yang, J. Wu, X. Yan, X. Li, B. Duan, J. Fang, and Y. Zhang, “Adatsq: Pushing the pareto frontier of diffusion transformers via temporal-sensitivity quantization,” arXiv preprint arXiv:2602.09883, 2026.
- [171] L. Russell, A. Hu, L. Bertoni, G. Fedoseev, J. Shotton, E. Arani, and G. Corrado, “Gaia-2: A controllable multi-view generative world model for autonomous driving,” arXiv preprint arXiv:2503.20523, 2025.
- [172] Wayve. Gaia-3: Scaling world models to power safety and evaluation. Wayve Blog (Research). [Online]. Available: https: //wayve.ai/thinking/gaia-3/
- [173] G. Zhao, C. Ni, X. Wang, Z. Zhu, X. Zhang, Y. Wang, G. Huang, X. Chen, B. Wang, Y. Zhang et al., “Drivedreamer4d: World models are effective data machines for 4d driving scene representation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 12015–12026.
- [174] X. Guo, C. Ding, H. Dou, X. Zhang, W. Tang, and W. Wu, “Infinitydrive: Breaking time limits in driving world models,” arXiv preprint arXiv:2412.01522, 2024.
- [175] B. Xie, Y. Liu, T. Wang, J. Cao, and X. Zhang, “Glad: A streaming scene generator for autonomous driving,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=ZFxpclrCCf
- [176] J. Wang, Y. Yao, X. Feng, H. Wu, Y. Wang, Q. Huang, Y. Ma, and X. Zhu, “Stage: A stream-centric generative world model for longhorizon driving-scene simulation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 14163–14169.
- [177] B. Li, J. Guo, H. Liu, Y. Zou, Y. Ding, X. Chen, H. Zhu, F. Tan, C. Zhang, T. Wang et al., “Uniscene: Unified occupancy-centric driving

- scene generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 11971–11981.
- [178] Z. Zhu, Z. Wu, Z. Zhu, L. Zhou, H. Sun, B. WANG, K. Ma, G. Chen, H. Ye, J. Xie, and jian Yang, “Worldsplat: Gaussian-centric feed-forward 4d scene generation for autonomous driving,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=KWeX6tYno6
- [179] J. Zhu, Z. Jia, T. Gao, J. Deng, S. Li, L. Zhang, F. Liu, P. Jia, and X. Lang, “Other vehicle trajectories are also needed: A driving world model unifies ego-other vehicle trajectories in video latent space,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 16, 2026, pp. 13934–13942.
- [180] J. Lu, Z. Huang, Z. Yang, J. Zhang, and L. Zhang, “Wovogen: World volume-aware diffusion for controllable multi-camera driving scene generation,” in European Conference on Computer Vision. Springer, 2024, pp. 329–345.
- [181] X. Ren, Y. Lu, T. Cao, R. Gao, S. Huang, A. Sabour, T. Shen, T. Pfaff, J. Z. Wu, R. Chen et al., “Cosmos-drive-dreams: Scalable synthetic driving data generation with world foundation models,”

2025. [Online]. Available: https://arxiv.org/abs/2506.09042

- [182] Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang, “Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14749–14759.
- [183] S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li, “Vista: A generalizable driving world model with high fidelity and versatile controllability,” Advances in Neural Information Processing Systems, vol. 37, pp. 91560–91596, 2024.
- [184] H. Wang, D. Liu, H. Xie, H. Liu, E. Ma, K. Yu, L. Wang, and B. Wang, “Mila: Multi-view intensive-fidelity long-term video generation world model for autonomous driving,” arXiv preprint arXiv:2503.15875, 2025.
- [185] F. Jia, W. Mao, Y. Liu, Y. Zhao, Y. Wen, C. Zhang, X. Zhang, and T. Wang, “Adriver-i: A general world model for autonomous driving,”

2023. [Online]. Available: https://arxiv.org/abs/2311.13549

- [186] E. Santana and G. Hotz, “Learning a driving simulator,” arXiv preprint arXiv:1608.01230, 2016.
- [187] X. Yang, L. Wen, T. Wei, Y. Ma, J. Mei, X. Li, W. Lei, D. Fu, P. Cai, M. Dou et al., “Drivearena: A closed-loop generative simulation platform for autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 26933– 26943.
- [188] A. Rahimi, V. Gerard, E. Zablocki, M. Cord, and A. Alahi, “Mad: Motion appearance decoupling for efficient driving world models,” arXiv preprint arXiv:2601.09452, 2026.
- [189] K. Zhang, Z. Tang, X. Hu, X. Pan, X. Guo, Y. Liu, J. Huang, L. Yuan, Q. Zhang, X.-X. Long et al., “Epona: Autoregressive diffusion world model for autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.
- [190] J. Yang, S. Gao, Y. Qiu, L. Chen, T. Li, B. Dai, K. Chitta, P. Wu, J. Zeng, P. Luo et al., “Generalized predictive model for autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 14662–14672.
- [191] T. Xia, Y. Li, L. Zhou, J. Yao, K. Xiong, H. Sun, B. Wang, K. Ma, H. Ye, W. Liu et al., “Drivelaw: Unifying planning and video generation in a latent driving world,” arXiv preprint arXiv:2512.23421, 2025.
- [192] Y. Chen, Y. Wang, and Z. Zhang, “Drivinggpt: Unifying driving world modeling and planning with multi-modal autoregressive transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 26890–26900.
- [193] F. Bartoccioni, E. Ramzi, V. Besnier, S. Venkataramanan, T.-H. Vu, Y. Xu, L. Chambon, S. Gidaris, S. Odabas, D. Hurych et al., “Vavim and vavam: Autonomous driving through video generative modeling,” arXiv preprint arXiv:2502.15672, 2025.
- [194] Y. Feng, H. Tan, X. Mao, C. Xiang, G. Liu, S. Huang, H. Su, and J. Zhu, “Vidar: Embodied video diffusion model for generalist manipulation,” arXiv preprint arXiv:2507.12898, 2025.
- [195] J. Jang, S. Ye, Z. Lin, J. Xiang, J. Bjorck, Y. Fang, F. Hu, S. Huang, K. Kundalia et al., “Dreamgen: Unlocking generalization in robot learning through video world models,” in 9th Annual Conference on Robot Learning, 2025. [Online]. Available: https: //openreview.net/forum?id=3CnxNqmklv
- [196] J. Ni, Z. Wang, W. Lin, A. Bar, Y. LeCun, T. Darrell, J. Malik, and R. Herzig, “From generated human videos to physically plausible robot trajectories,” arXiv preprint arXiv:2512.05094, 2025.

- [197] Y. Deng, Z. Pan, H. Zhang, X. Li, R. Hu, Y. Ding, Y. Zou, Y. Zeng, and D. Zhou, “Rethinking video generation model for the embodied world,” arXiv preprint arXiv:2601.15282, 2026.
- [198] G. Team, A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, J. Zhu, K. Li, M. Xu et al., “Gigaworld-0: World models as data engine to empower embodied ai,” arXiv preprint arXiv:2511.19861, 2025.
- [199] S. Patel, S. Mohan, H. Mai, U. Jain, S. Lazebnik, and Y. Li, “Robotic manipulation by imitating generated videos without physical demonstrations,” arXiv preprint arXiv:2507.00990, 2025.
- [200] X. Qiu, Y. Wang, J. Cai, Z. Chen, C. Lin, T.-H. Wang, and C. Gan, “Lucibot: Automated robot policy learning from generated videos,” arXiv preprint arXiv:2503.09871, 2025.
- [201] H. Bharadhwaj, D. Dwibedi, A. Gupta, S. Tulsiani, C. Doersch, T. Xiao, D. Shah, F. Xia, D. Sadigh, and S. Kirmani, “Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation,” in 9th Annual Conference on Robot Learning, 2025. [Online]. Available: https://openreview.net/forum?id=HprBJupvvM
- [202] J. Liang, R. Liu, E. Ozguroglu, S. Sudhakar, A. Dave, P. Tokmakov, S. Song, and C. Vondrick, “Dreamitate: Real-world visuomotor policy learning via video generation,” in 8th Annual Conference on Robot Learning, 2024. [Online]. Available: https://openreview.net/forum?id= InT87E5sr4
- [203] J. Xiao, Y. Yang, X. Chang, R. Chen, F. Xiong, M. Xu, W.-S. Zheng, and Q. Zhang, “World-env: Leveraging world model as a virtual environment for vla post-training,” arXiv preprint arXiv:2509.24948, 2025.
- [204] Y. Jiang, S. Chen, S. Huang, L. Chen, P. Zhou, Y. Liao, X. He, C. Liu, H. Li, M. Yao et al., “Enerverse-ac: Envisioning embodied environments with action condition,” arXiv preprint arXiv:2505.09723, 2025.
- [205] Y. Guo, L. X. Shi, J. Chen, and C. Finn, “Ctrl-world: A controllable generative world model for robot manipulation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=748bHL2BAv
- [206] A. Soni, S. Venkataraman, A. Chandra, S. Fischmeister, P. Liang, B. Dai, and S. Yang, “Videoagent: Self-improving video generation,” arXiv preprint arXiv:2410.10076, 2024.
- [207] A. Escontrela, A. Adeniji, W. Yan, A. Jain, X. B. Peng, K. Goldberg, Y. Lee, D. Hafner, and P. Abbeel, “Video prediction models as rewards for reinforcement learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 68760–68783, 2023.
- [208] Y. Li, Y. Zhu, J. Wen, C. Shen, and Y. Xu, “Worldeval: World model as real-world robot policies evaluator,” arXiv preprint arXiv:2505.19017, 2025.
- [209] Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, S. Liu, J. Luo, L. Chen et al., “Genie envisioner: A unified world foundation platform for robotic manipulation,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=fHLtSxDFKC
- [210] A. K. Sharma, Y. Sun, N. Lu, Y. Zhang, J. Liu, and S. Yang, “Worldgymnast: Training robots with reinforcement learning in a world model,” arXiv preprint arXiv:2602.02454, 2026.
- [211] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, K. Mo, C.-H. Lin et al., “Dreamdojo: A generalist robot world model from large-scale human videos,” 2026. [Online]. Available: https://arxiv.org/abs/2602.06949
- [212] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong, “Unleashing large-scale video generative pretraining for visual robot manipulation,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=NxoFmGgWC9
- [213] Z. Xu, Q. Qiu, and Y. She, “Vilp: Imitation learning with latent video planning,” IEEE Robotics and Automation Letters, 2025.
- [214] S. Li, Y. Gao, D. Sadigh, and S. Song, “Unified video action model,” arXiv preprint arXiv:2503.00200, 2025.
- [215] L. Yang, Y. Bai, G. Eskandar, F. Shen, M. Altillawi, D. Chen, S. Majumder, Z. Liu, G. Kutyniok, and A. Valada, “Roboenvision: A longhorizon video generation model for multi-task robot manipulation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 21281–21288.
- [216] H. Zhang, P. Ding, S. Lyu, Y. Peng, and D. Wang, “GEVRM: Goal-expressive video generation model for robust visual manipulation,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=hPWWXpCaJ7
- [217] S. Huang, L. Chen, P. Zhou, S. Chen, Y. Liao, Z. Jiang, Y. Hu, P. Gao, H. Li, M. Yao, and G. Ren, “Enerverse: Envisioning

- embodied future space for robotics manipulation,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=igtjRQfght
- [218] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu et al., “Causal world modeling for robot control,” arXiv preprint arXiv:2601.21998, 2026.
- [219] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M.-Y. Liu, C. Finn, and J. Gu, “Cosmos policy: Fine-tuning video models for visuomotor control and planning,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=wPEIStHxYH
- [220] T. Yuan, Z. Dong, Y. Liu, and H. Zhao, “Fast-wam: Do world action models need test-time future imagination?” 2026, arXiv preprint arXiv:2603.16666.
- [221] L. Maes, Q. L. Lidec, D. Scieur, Y. LeCun, and R. Balestriero, “Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels,” arXiv preprint arXiv:2603.19312, 2026.
- [222] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang et al., “World action models are zeroshot policies,” arXiv preprint arXiv:2602.15922, 2026.
- [223] H. Che, X. He, Q. Liu, C. Jin, and H. Chen, “Gamegen-x: Interactive open-world game video generation,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=8VG8tpPZhe
- [224] J. Yu, Y. Qin, X. Wang, P. Wan, D. Zhang, and X. Liu, “Gamefactory: Creating new games with generative interactive videos,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 11590–11599.
- [225] J. Guo, Y. Ye, T. He, H. Wu, Y. Jiang, T. Pearce, and J. Bian, “Mineworld: a real-time and open-source interactive world model on minecraft,” arXiv preprint arXiv:2504.08388, 2025.
- [226] Y. Zhang, C. Peng, B. Wang, P. Wang, Q. Zhu, Z. Gao, E. Li, Y. Liu, and Y. Zhou, “Matrix-game: Interactive world foundation model,” arXiv preprint arXiv:2506.18701, 2025.
- [227] N. Savov, N. Kazemi, M. Mahdi, D. P. Paudel, X. Wang, and L. Van Gool, “Exploration-driven generative interactive environments,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 27597–27607.
- [228] J. Li, J. Tang, Z. Xu, L. Wu, Y. Zhou, S. Shao, T. Yu, Z. Cao, and Q. Lu, “Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition,” 2025. [Online]. Available: https://arxiv.org/abs/2506.17201
- [229] J. Tang, J. Liu, J. Li, L. Wu, H. Yang, P. Zhao, S. Gong, X. Yuan, S. Shao, and Q. Lu, “Hunyuan-gamecraft-2: Instruction-following interactive game world model,” arXiv preprint arXiv:2511.23429, 2025.
- [230] M. Yang, J. Li, Z. Fang, S. Chen, Y. Yu, Q. Fu, W. Yang, and D. Ye, “Playable game generation,” 2024. [Online]. Available: https://arxiv.org/abs/2412.00887
- [231] W. Sun, H. Zhang, H. Wang, J. Wu, Z. Wang, Z. Wang, Y. Wang, J. Zhang, T. Wang, and C. Guo, “Worldplay: Towards long-term geometric consistency for real-time interactive world modeling,” arXiv preprint arXiv:2512.14614, 2025.
- [232] R. Team, “Advancing open-source world models,” 2026. [Online]. Available: https://arxiv.org/abs/2601.20540
- [233] D. Hafner, W. Yan, and T. Lillicrap, “Training agents inside of scalable world models,” arXiv preprint arXiv:2509.24527, 2025. [Online]. Available: https://arxiv.org/abs/2509.24527
- [234] Y. LeCun, “A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27,” Open Review, vol. 62, no. 1, pp. 1–62, 2022.
- [235] Y. Zhu, L. Zhang, Z. Rong, T. Hu, S. Liang, and Z. Ge, “Infp: Audio-driven interactive head generation in dyadic conversations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 10667–10677.
- [236] H. Guo, H. Yi, D. Zhou, A. W. Bergman, M. Lingelbach, and Y. Yu, “Real-time one-step diffusion-based expressive portrait videos generation,” arXiv preprint arXiv:2412.13479, 2024.
- [237] Z. Sun, Z. Peng, Y. Ma, Y. Chen, Z. Zhou, Z. Zhou, G. Zhang, Y. Zhang, Y. Zhou, Q. Lu, and Y.-J. Liu, “Streamavatar: Streaming diffusion models for real-time interactive human avatars,” 2025. [Online]. Available: https://arxiv.org/abs/2512.22065
- [238] H. Guo, J. Wu, J. Liu, Y. Gao, Z. Ye, L. Yuan, X. Wang, Y. Yu, and W. Huang, “Leveraging verifier-based reinforcement learning in image editing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Denver, Colorado, June 2026.
- [239] Y. Zuo, Z. Wang, L. Li, X. Liu, F. Liu, and L. Jiao, “Edit-your-interest: Efficient video editing via feature most-similar propagation,” arXiv preprint arXiv:2510.13084, 2025.

- [240] M. Cai, X. Cun, X. Li, W. Liu, Z. Zhang, Y. Zhang, Y. Shan, and X. Yue, “Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 7763–7772.
- [241] Y. Gong, Y. Pang, X. Cun, M. Xia, Y. He, H. Chen, L. Wang, Y. Zhang, X. Wang, Y. Shan et al., “Talecrafter: Interactive story visualization with multiple characters,” arXiv preprint arXiv:2305.18247, 2023.
- [242] Y. He, M. Xia, H. Chen, X. Cun, Y. Gong, J. Xing, Y. Zhang, X. Wang, C. Weng, Y. Shan et al., “Animate-a-story: Storytelling with retrievalaugmented video generation,” arXiv preprint arXiv:2307.06940, 2023.
- [243] Z. Fang, W. Zhai, A. Su, H. Song, K. Zhu, M. Wang, Y. Chen, Z. Liu, Y. Cao, and Z.-J. Zha, “Vivid: Video virtual try-on using diffusion models,” arXiv preprint arXiv:2405.11794, 2024.
- [244] Y. Tu, H. Luo, X. Chen, X. Bai, F. Wang, and H. Zhao, “Playerone: Egocentric world simulator,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=Gq4Gay8rDB
- [245] Y.-C. Lee, Y.-T. Chen, A. Wang, T.-H. Liao, B. Y. Feng, and J.-B. Huang, “Vividdream: Generating 3d scene with ambient dynamics,”

2024. [Online]. Available: https://arxiv.org/abs/2405.20334

- [246] H. Yu, C. Wang, P. Zhuang, W. Menapace, A. Siarohin, J. Cao, L. A. Jeni, S. Tulyakov, and H.-Y. Lee, “4real: Towards photorealistic 4d scene generation via video diffusion models,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=SO1aRpwVLk
- [247] T. Liu, Z. Huang, Z. Chen, G. Wang, S. Hu, L. Shen, H. Sun, Z. Cao, W. Li, and Z. Liu, “Free4d: Tuning-free 4d scene generation with spatial-temporal consistency,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 25571–25582.
- [248] J. Huang, Y. Yang, B. Yang, L. Ma, Y. Ma, and Y. Liao, “Gen3r: 3d scene generation meets feed-forward reconstruction,” 2026. [Online]. Available: https://arxiv.org/abs/2601.04090
- [249] R. Wu, R. Gao, B. Poole, A. Trevithick, C. Zheng, J. T. Barron, and A. Holynski, “Cat4d: Create anything in 4d with multi-view video diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 26057–26068.
- [250] S. Zhai, Z. Ye, J. Liu, W. Xie, J. Hu, Z. Peng, H. Xue, D. Chen, X. Wang, L. Yang, N. Wang, H. Liu, and G. Zhang, “Stargen: A spatiotemporal autoregression framework with video diffusion model for scalable and controllable scene generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 26822–26833.

