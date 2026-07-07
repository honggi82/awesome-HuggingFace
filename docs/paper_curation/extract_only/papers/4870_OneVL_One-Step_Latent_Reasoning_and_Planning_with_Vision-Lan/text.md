# arXiv:2604.18486v3[cs.CV]8May2026

Xiaomi EV

### Xiaomi OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation

#### Xiaomi Embodied Intelligence Team

See the Contributions and Acknowledgments section for a list of contributors.

#### Abstract

Chain-of-Thought (CoT) reasoning has become a powerful driver of trajectory prediction in VLAbased autonomous driving, yet its autoregressive nature imposes a latency cost that is prohibitive for real-time deployment. Latent CoT methods attempt to close this gap by compressing reasoning into continuous hidden states, but consistently fall short of their explicit counterparts. We suggest that this is due to purely linguistic latent representations compressing a symbolic abstraction of the world, rather than the causal dynamics that actually govern driving. Thus, we present OneVL (One-step latent reasoning and planning with Vision-Language explanations), a unified VLA and World Model framework that routes reasoning through compact latent tokens supervised by dual auxiliary decoders. Alongside a language decoder that reconstructs text CoT, we introduce a visual world model decoder that predicts future-frame tokens, forcing the latent space to internalize the causal dynamics of road geometry, agent motion, and environmental change. A three-stage training pipeline progressively aligns these latents with trajectory, language, and visual objectives, ensuring stable joint optimization. In inference, the auxiliary decoders are discarded, and all latent tokens are prefilled in a single parallel pass, matching the speed of answer-only prediction. Across four benchmarks, OneVL becomes the first latent CoT method to surpass explicit CoT, delivering superior accuracy at answer-only latency. These results show that with world model supervision, latent CoT produces more generalizable representations than verbose token-by-token reasoning.

- - Project Page: https://Xiaomi-Embodied-Intelligence.github.io/OneVL
- - GitHub Repository: https://github.com/xiaomi-research/OneVL

###### (a) NAVSIM

###### (b) ROADWork

###### (c) Impromptu

###### (d) Alpamayo-R1

3.0

3.75

25.0

90

12

12

- 3

- 4

- 5

- 6

- 7

- 8

3.50

22.5

- 2

- 3

- 4

- 5

2.5

10

10

88

3.25

20.0

###### Latency(s)

###### Latency(s)

###### Latency(s)

###### Latency(s)

PDM-score

ADE(px)

ADE(m)

ADE(m)

8

8

2.0

3.00

17.5

86

2.75

15.0

6

6

1.5

2.50

12.5

84

4

4

2.25

10.0

1.0

AdaThinkLaSTCOCOSIM-CoTARAnsARCoTOneVL

YNetCOCOSIM-CoTARAnsARCoTOneVL

Impro.COCOSIM-CoTARAnsARCoTOneVL

Alpa.COCOCODISIM-CoTARAnsARCoTOneVL

Prev. SOTA (latency N/A)

Latent CoT AR Baseline

OneVL (Ours)

| |
|---|

Prev. SOTA

Latency

| |
|---|

Figure 1 Accuracy and efficiency comparison across four benchmarks. Existing latent CoT methods underperform explicit CoT. OneVL is the first to surpass it while matching answer-only prediction latency.

#### Contents

- 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 2 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 2.1 Implicit and Latent Chain-of-Thought . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.2 VLM and VLA for Autonomous Driving . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.3 World Modeling for Autonomous Driving . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3 Model Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3.1 Main Vision-Language Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.2 Latent Token Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.3 Language Auxiliary Decoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.4 Visual Auxiliary Decoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.5 Combined Training Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.6 Prefill Inference . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 Three-Stage Training Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4.1 Preliminary: Visual Auxiliary Decoder Self-Supervised Pretraining . . . . . . . . . . . . . . . 9
- 4.2 Stage 0: Main Model Warmup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.3 Stage 1: Auxiliary Decoder Warmup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.4 Stage 2: Joint End-to-End Fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 5 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 5.1 Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.2 Experimental Setups . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 5.3 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.4 Explanation Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 5.5 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 5.6 Towards Real-World Deployment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 5.7 In-Depth Analysis: Where Does the Benefit Come From? . . . . . . . . . . . . . . . . . . . . 21

- 6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 7 Contributions and Acknowledgments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23 A Appendix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- A.1 Data Format Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- A.2 Training Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- A.3 CoT Annotation Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- A.4 LLM-as-Judge Evaluation Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- A.5 Reproducing Impromptu . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- A.6 NAVSIM qualitative examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- A.7 Roadwork qualitative examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- A.8 Impromptu qualitative examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- A.9 Alpamayo-R1 qualitative examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

#### 1 Introduction

Vision-Language Models (VLMs) [1, 4–6, 10, 15, 35, 37, 38, 49, 50, 67, 68, 79, 97, 100, 104] have rapidly become a foundational building block for autonomous driving, unifying holistic scene understanding, natural language reasoning, and end-to-end trajectory planning within a single model [43, 45, 47, 57, 65, 84, 89, 103, 113, 114]. When further extended to produce action outputs, such as trajectory waypoints or control signals, these models are known as Vision-Language-Action models (VLAs) [13, 17, 30, 31, 42, 48, 63, 80, 102].

A central driver of recent progress in VLA-based driving is Chain-of-Thought (CoT) reasoning [17, 78, 80, 116], where the model articulates intermediate reasoning steps before committing to a final trajectory, yielding substantial gains in prediction quality [45, 105]. By explicitly surfacing scene semantics, anticipated agent behaviors, and high-level driving intent, CoT supervision binds predictions into coherent causal chains and markedly reduces planning errors. This success echoes a broader body of LLM CoT research spanning mathematical reasoning [66], document understanding [27, 28, 73–76], code synthesis [14], multimodal QA [77, 92], RL-based deep reasoning [37, 53], and test-time scaling [52, 90]. A unifying explanation for why CoT works comes from the compression view of intelligence [21, 59]: under next-token supervision, a model forced to articulate intermediate steps must compress its understanding into structured, generalizable representations rather than memorize shallow input–output mappings.

Yet deploying CoT in real driving systems exposes a sharp tension between interpretability and efficiency. Standard autoregressive (AR) CoT generation must emit every reasoning token before the trajectory can be produced. This yields inference latency proportional to the chain length, which is far above that of answer-only prediction. In safety-critical real-time settings, this gap is prohibitive. At the same time, explicit CoT chains are strikingly redundant; for example, much of the sequence merely restates context or follows formulaic patterns. This redundancy suggests that the essential reasoning content can be compressed into a far more compact form [96] without sacrificing and even strengthening generalization, since tighter compression forces the model to retain only the causal structure that truly matters for prediction.

Latent CoT and Its Limitations. A growing line of work pursues exactly this direction, replacing explicit reasoning tokens with compact latent representations [115]. COCONUT [41] introduced curriculum learning over latent thought tokens, progressively replacing discrete reasoning steps with continuous vectors. CODI [87] extended this with self-distillation, training a student to mimic a teacher’s CoT behavior in latent space. SIMCoT [106] attached a separate text-decoding auxiliary decoder to enable direct text supervision during latent training. However, adapting these methods to VLA-based driving reveals critical shortcomings. COCONUT, CODI, and SIM-CoT were designed for language-only reasoning and make no use of the rich visual structure that defines driving scenes. As a result, their purely linguistic latents prove insufficient for the multimodal reasoning demanded by trajectory prediction, and as Figure 1 shows, every existing latent CoT method underperforms explicit CoT across all benchmarks.

More fundamentally, natural language descriptions of driving scenes are inherently abstract. They encode semantic labels rather than the spatiotemporal causal dynamics that actually determine future outcomes. A latent vector that compresses language is therefore compressing a symbolic abstraction of the world, not its underlying causal structure. A further limitation is that in prior methods, latent hidden states are still produced autoregressively (one latent hidden state at a time), leaving inference sequential. We instead aim to generate all latent hidden states in a single step via prefill. Figure 2 contrasts these three paradigms, that is, explicit CoT, prior implicit latent CoT, OneVL, and motivates our design.

OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanations.

We present OneVL, a framework that overcomes the limitations of prior latent CoT methods through two key innovations. First, we introduce dual-modal auxiliary decoders: a language auxiliary decoder that reconstructs human-readable CoT reasoning from compact language latent tokens, and a visual auxiliary decoder that predicts anticipated future frames [19, 71, 88] from visual latent representations. The visual decoder plays the role of a world model auxiliary. By forcing the compressed latents to anticipate what the scene will look like at future time steps, it ensures that the bottleneck encodes genuinely causal scene dynamics, such as agent trajectories, road geometry evolution, and emerging hazards, rather than abstract symbolic summaries. This is precisely the missing ingredient in language-only latent CoT. Future-frame prediction is a concrete

- a

- b

- c

✓ Interpretable

###### Explicit CoT

###### . . . T̂ y

input VLM c

Tc Tc Tc Tc

✗ Slow

(Traditional)

Explicit Reasoning Chain

###### Implicit CoT

✗ Not Interpretable

Z Z Z T̂ y Opaque Latent Vectors

input VLM

(Existing Work)

✓ Fast

[Figure 1]

##### OneVL

✓ Vision & Language

input VLM Zv T̂ y

Zv Zl Zl

(Ours)

✓ Faster (Preﬁll)

- v Stage 0 ✓ More Accurate Main Model Warm Up
- v Stage 1 Aux Decoder Warm Up
- v Stage 2 Joint E2E Fine-Tuning

Vis. Aux Dec Lang. Aux Dec

Future Frames CoT Text

Tc CoT Token Z Latent (z) Zv Vis. Latent Zl Lang. Latent T̂ y Answer

- Figure 2 Comparison of CoT paradigms. (a) Explicit CoT: the model generates a full chain of discrete reasoning tokens before the answer. (b) Implicit CoT: reasoning is compressed into a small number of opaque latent vectors Z.

(c) OneVL (Ours): two types of latent tokens (Zv, red) and language (Zl, salmon); during training, dual auxiliary decoders decode these into future-frame visual tokens and CoT text, respectively, providing rich text and world model supervision. During inference, the decoders are discarded, and the latent tokens are prefilled into the prompt context, matching the speed of answer-only prediction while keeping the interpretability of (a) in both vision and language.

compression target that directly reflects the causal structure of the physical world, satisfying the compression view of intelligence in a way that text descriptions alone cannot. The resulting framework simultaneously handles planning, language reasoning, and visual interpretation within a single model.

Beyond interpretability, the dual reconstruction objectives serve a deeper role: they ensure that the compressed latents encode genuinely generalizable structure rather than superficial correlations [21, 95]. If compact latent tokens can be decoded into both coherent language reasoning and plausible future frames, the model has necessarily discovered transferable representations of scene dynamics rather than memorized input-output mappings. Critically, the world model supervision (visual decoder) and the language supervision act as complementary forms of validation. Language grounds the latents in semantic intent, while visual prediction grounds them in physical scene dynamics. Together, they guarantee that the compressed representation satisfies both the semantic and causal requirements of robust trajectory planning.

Second, we design a Prefill Inference mechanism. At inference time, the latent tokens (both visual and language) are prefilled into the model’s context as fixed prompt inputs, enabling single-pass generation of all latent tokens. This eliminates the iterative latent token generation overhead and achieves inference speed essentially identical to answer-only AR prediction. The resulting model performs one-step latent reasoning (fast inference), vision-language explanation (interpretable reasoning), and finally planning in a unified sequence. Empirically, OneVL not only matches but surpasses explicit AR CoT in trajectory quality, demonstrating that compression, far from being a necessary compromise, is itself a driver of more effective reasoning [21, 59].

Contributions. The key contributions of this work are summarized as follows:

- • OneVL Framework: We introduce a latent CoT framework built on the principle that compression drives generalization, and identify a critical gap in prior latent CoT work, that is, purely linguistic latent representations are too abstract to satisfy this principle for planning tasks. We address this with dualmodal auxiliary decoders, a language decoder, and a visual world model decoder that jointly supervise compact latent tokens to encode both linguistic reasoning and future scene dynamics. The world model decoder provides the concrete, causal compression target that language alone cannot supply. A principled

- three-stage training pipeline progressively aligns the latent bottleneck with trajectory prediction, ensuring that the compressed representations capture causal structure rather than memorized patterns.
- • Superior performance across Four Benchmarks: OneVL achieves superior performance across many benchmarks. Notably, OneVL is the only latent CoT method that outperforms explicit autoregressive CoT, directly supporting our hypothesis that tighter compression encourages more generalizable reasoning. Ablation studies confirm each component’s contribution. Both the visual and language decoders yield consistent performance gains, and the staged training recipe is essential.
- • Prefill Inference: At inference time, the auxiliary decoders are discarded, and all latent tokens are prefilled into the prompt, enabling single-pass latent CoT reasoning with no iterative overhead. For example, on NAVSIM, the latency matches AR answer-only prediction and is 1.5× faster than explicit autoregressive CoT. On ROADWork, prefill latency is identical to answer-only and 2.3× faster than its explicit counterpart. For real-world deployment, appending an MLP head for producing trajectory further reduces latency to 0.24s (4.16 Hz), just 5.4% of the AR model’s latency, offering a practical deployment option.
- • Interpretable Explanations: The language auxiliary decoder recovers high-quality CoT text from compressed latents, while the visual auxiliary decoder generates spatially coherent future-frame previews, providing both linguistic and visual interpretability.

Organization. The remainder of this paper is organized as follows. Section 2 presents related work. Section 3 describes the OneVL architecture in detail, including the main VLM, latent token design, and auxiliary decoders. Section 4 elaborates on the three-stage training pipeline and the motivation for each stage. Section 5 presents the experimental setup, main results, and ablation studies. Section 6 concludes with a discussion of future directions.

#### 2 Related Work

###### 2.1 Implicit and Latent Chain-of-Thought

Since explicit CoT incurs inference overhead proportional to the length of the reasoning chain, a growing line of work internalizes reasoning into continuous latent representations [16, 22, 60, 61, 91, 112]. Deng et al. [22] proposed a step-by-step internalization curriculum that progressively replaces each explicit reasoning token with implicit internal computation, training the model to absorb CoT one step at a time. COCONUT [41] generalizes this idea to continuous latent thought tokens through a staged curriculum, enabling breadthfirst-like exploration of solution paths entirely within the LLM’s hidden state space. Compressed Chain of Thought [16] takes a complementary distillation approach, condensing an explicit CoT trace into a small set of dense summary vectors prepended to the input, achieving substantial length reduction at only modest accuracy cost. CODI [87] adopts sequence-level self-distillation, training a student model to align its anchor latent hidden state, typically the final hidden representation before the answer, with the teacher model’s full chain-of-thought sequence, narrowing the performance gap while preserving efficiency. Token Assorted [91] offers a flexible middle ground by interleaving discrete text tokens and continuous latent tokens within the same sequence, interpolating between fully explicit and fully implicit reasoning. SIM-CoT [106] identifies a latent instability problem, where representations collapse as the number of latent tokens grows without per-step supervision, and addresses it with a plug-and-play auxiliary decoder that aligns each implicit token with its corresponding explicit reasoning step at training time only.

- As discussed, all of these methods were developed for language-only tasks and do not transfer effectively to VLA-based autonomous driving.

###### 2.2 VLM and VLA for Autonomous Driving

Beyond the foundational VLM works and CoT-augmented driving models discussed in Section 1, a parallel line of research has focused on establishing richer evaluation and supervision signals for language-grounded driving [18, 45, 108]. MapLM [9] introduced a large-scale benchmark specifically targeting map and traffic scene understanding, probing whether VLMs can parse structured road topology from sensor data. Ding et al. [23] augmented VLMs with bird’s-eye-view feature injection, enabling holistic scene understanding that fuses

camera and top-down spatial context within a single multimodal model. Complementary efforts have explored corner-case evaluation [12] and risk localization [82] to stress-test VLM reasoning under rare or safety-critical scenarios.

Closer to trajectory prediction, recent VLA models pair language reasoning with waypoint or action outputs [69, 70, 117]. DriveVLA-W0 [62] employs world modeling to generate dense self-supervised signals that amplify data scaling laws in VLA-based driving. AdaThinkDrive [78] introduces adaptive CoT for driving decisions, LaST-VLA [80] trains a large vision-language-action model on driving data, and Alpamayo-R1 [105] explicitly bridges reasoning traces with long-tail action prediction. OneVL builds on these foundations by addressing the latency cost of explicit CoT through dual-modal latent supervision and prefill inference, delivering competitive performance without sacrificing interpretability.

###### 2.3 World Modeling for Autonomous Driving

The concept of the world model originates from model-based reinforcement learning, where it seeks to emulate human cognitive processes and predict the effects of actions on environmental evolution [39, 40, 55], particularly in 3D and 4D spaces [7, 56–58, 64, 111]. To further enhance spatial reasoning, several approaches incorporate advanced perception frameworks [101, 109, 110, 118, 119] to achieve a more robust understanding of 3D environments. With advances in video generation and the introduction of the Joint Embedding Predictive Architecture by Assran et al. [2], the scope and applications of world models have broadened considerably [24, 25, 44, 46, 93]. In autonomous driving, world models are typically applied to three ends: data generation, closed-loop evaluation, and representation learning [29, 36, 57, 65, 94].

For data generation, Cosmos [3] integrates multimodal inputs such as text, images, videos, and motion signals to synthesize consistent data for training robotic and autonomous driving systems. For closed-loop evaluation, DICC [33] leverages generative world models to produce realistic driving images and performs adversarial evaluation on end-to-end driving systems to improve safety and robustness. Similarly, AD-R1 [114] takes advantage of the high physical fidelity of world models, employing them as interactive simulators for reinforcement learning and thereby reducing safety violations in challenging scenarios. For representation learning, DriveVLA-W0 [62] incorporates future temporal information from a world model to improve trajectory planning, and DynVLA [86] reduces redundancy in generated images by modeling inter-frame similarity, achieving lower inference latency while maintaining competitive performance.

In contrast, OneVL uses short-horizon future visual token prediction as a training-only world model auxiliary paired with compressed latent CoT inside a single VLA. This auxiliary guides the bottleneck toward causal scene dynamics precisely where language-only implicit CoT is insufficient (Section 1), and is then discarded at inference so that prefilled latents yield answer-only latency. Prior work emphasizes data generation, simulators, or separate representation stacks, rather than this joint certification of language and visual latent bottlenecks.

#### 3 Model Architecture

OneVL augments a pretrained VLM with a compact latent token interface and dual auxiliary decoders for multimodal explanation. Figure 3 gives a complete overview. We describe each component in detail below.

###### 3.1 Main Vision-Language Model

The backbone of OneVL is Qwen3-VL-4B-Instruct [5], a VLM that processes interleaved image and text inputs. The model consists of three standard components: Vision Encoder (ViT), Visual Projector (MLP Aligner), and Large Language Model (LLM). All three components are initialized from the Qwen3-VL-4B-Instruct checkpoint and remain fully trainable in Stages 0 (Section 4.2) and 2 (Section 4.4). The backbone is primarily optimized via a standard next-token prediction objective, applying a cross-entropy loss (Lc) to both the trajectory answers and the latent reasoning tokens introduced below.

###### 3.2 Latent Token Design

A critical design decision in OneVL is the introduction of specialized latent tokens that serve as compact carriers of implicit reasoning. We define two classes of latent tokens:

[Figure 2]

###### Token Type

[Figure 3]

img Img Token txt Text Token Zv Visual Latent Zl Lang. Latent T̂ y Answer

[Figure 4]

[Figure 5]

## OneVL

[Figure 6]

Sensor Data

One-Step Reasoning, Planning & Explaining

Command Velocity Acceleration His. Traj.

Turn Right [ 1.5, 0.0 ]

img txt Zv Zv Zv Zv Zl Zl Zl T̂ y

[0.3, 0.1 ] [0, 2], [3 ,4], …

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Visual Reason Text Reason Plan

Meta Data

[Figure 13]

Visual Aux Decoder Language Aux Decoder Trajectory Predictor

[Figure 14]

Green trafﬁc light signals allow move at intersection…

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

| |
|---|

Current

[Figure 25]

h h

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Case A

[Figure 32]

[Figure 33]

CE

[Figure 34]

Line of trafﬁc cones placed in right lane…

[Figure 35]

[Figure 36]

Case B

t at 0.5s t at 1.0s

- Figure 3 OneVL architecture. An image and structured text prompt (ego state, command, historical trajectory) are fed into the VLM. The output hidden states shown below the VLM, contains image tokens (Tv), text tokens (Tl), visual latent tokens (Zv), language latent tokens (Zl), and trajectory answer tokens (Tˆy). During training, hidden states Hv and Hl at the latent positions are routed to two auxiliary decoders: the Visual Aux. Decoder (left) directly predicts future-frame visual tokens at 0.5s and 1.0s (Lv), and the Language Aux. Decoder (right) predicts chain-of-thought reasoning (Ll). During inference, both decoders are discarded; latent tokens are prefilled into the prompt, matching the answer-only AR prediction latency.

Language Latent Tokens (⟨|latent|⟩): A fixed-length sequence of Ct = 2 language latent tokens, framed by start and end delimiters (⟨|start-latent|⟩ and ⟨|end-latent|⟩). These tokens are placed in the assistant response before the trajectory answer, occupying the position where explicit CoT reasoning would appear in standard AR models. The hidden states extracted at these token positions after LLM processing encode the model’s implicit language-grounded reasoning.

Visual Latent Tokens (⟨|latent-vis|⟩): A fixed-length sequence of Cv = 4 visual latent tokens, similarly delimited (i.e., ⟨|start-latent-vis|⟩ and ⟨|end-latent-vis|⟩), placed before the language latent tokens in the response. These tokens are designed to encode spatial and temporal visual reasoning about the future scene state.

Both sets of latent tokens serve as reasoning carriers whose hidden states at the VLM output layer are fed into auxiliary decoders. Note that during implementation, we found that adding dedicated special tokens (e.g., <|latent-vis|>) to the VLM vocabulary causes performance degradation. Instead, we represent latent tokens using the original vocabulary. Concretely, the Cv visual latent tokens are realized as 35 tokens, and the Ct language latent tokens are realized as 20 tokens.

###### 3.3 Language Auxiliary Decoder

The language auxiliary decoder Dl aims to recover human-readable CoT reasoning text from the compact language latent hidden states.

Input Construction. For each training sample, the language latent tokens in the main model produce hidden states Hl ∈ RC

t×d, where d is the LLM hidden dimension. We additionally supply the current-frame ViT patch embeddings V ∈ RN

v×d from the backbone, where Nv denotes the length of vision tokens after ViT embedding. An MLP layer maps both branches into the auxiliary decoder’s embedding space; we then form

the multimodal input by concatenation:

Zl = Wl(V), Wl(Hl) , (1)

where Wl is MLP (with dimensions chosen so that Zl matches the LLM input dimension). The tensor Zl is fed into Dl.

Training Objective. The language auxiliary decoder is trained to predict the ground-truth CoT reasoning text Ty

given Zl, that is:

t

|Tyt |

t,<i . (2)

Ll = −

l Ty

t,i | Zl,Ty

log PD

i=1

This cross-entropy loss encourages the main model’s language latent tokens to encode semantically rich information about the driving scene that is decodable as natural language reasoning.

###### 3.4 Visual Auxiliary Decoder

The visual auxiliary decoder Dv aims to predict anticipated future-frame visual tokens.

Motivation. Autonomous driving is inherently a spatial-temporal prediction task. Future frame visual tokens, which represent what the driving scene will look like at near-term horizons, are a natural target for learning the visual latent representations. This visual prediction objective serves as a world model auxiliary, supplementing language-only latent CoT. This task acts as a rigorous test of generalization, as predicting unseen configurations requires a robust causal model rather than pattern memorization. By combining visual and language decoders, the framework supervises latents in both physical dynamics and semantic intent, imposing a multi-modal constraint that captures the shared causal structure of the environment.

Input Construction. Let V ∈ RN

v×d denote the ViT embeddings from the current frame (extracted from the main model’s visual encoder), and let Hv ∈ RC

v×d denote the visual latent token hidden states from the main model. Let Wv ∈ Rd×d be the MLP layer. The visual auxiliary decoder receives the concatenation:

Zv = Wv(V), Wv(Hv) . (3)

This conditioning on both the current visual context and the latent state allows the decoder to perform conditioned future-frame prediction.

Visual Tokenizer and Vocabulary Extension. To represent images as discrete token sequences, we adopt the IBQ (Index Backpropagation Quantization) visual tokenizer [88]. We use the Emu3.5 tokenizer [19, 88] with a codebook of 131,072 discrete visual codes. The images are resized to a maximum resolution of 512x512 pixels. To integrate this visual vocabulary into OneVL, the Qwen3-VL-4B base vocabulary is extended by 131,072 additional visual token IDs. The visual token sequences for training are constructed offline by running the IBQ tokenizer over the ground-truth future frames from the dataset, requiring no additional forward passes during training.

Training Objective. Let Tyv

= [Tyv,1, Tyv,2] be the concatenated discrete visual token sequence for the future frames at time steps Tyv,1 (+0.5s) and Tyv,2 (+1.0s). The visual loss is:

|Tyv|

(Tyv,t | Zv,Tyv,<t) . (4)

Lv = −

log PD

v

t=1

###### 3.5 Combined Training Objective The total training loss L is a weighted sum of three components:

L = Lc + λlLl + λvLv , (5)

where Lc is the main model’s cross-entropy loss, λl = 1.0 is the language explanation loss weight, and λv = 0.1 is the visual explanation loss weight. The lower weight on Lv reflects that visual token reconstruction is a harder task, and a smaller weight prevents it from dominating the training signal.

###### 3.6 Prefill Inference

- At inference time, the auxiliary decoders are discarded. The key efficiency insight is that the latent tokens, both visual and language, can be prefilled into the prompt context as fixed token sequences, because their specific vocabulary identities have been seen by the model during training. Concretely, the inference prompt is constructed as:

, ⟨|end-latent|⟩] . (6)

[System, User query, ⟨|start-latent-vis|⟩, ⟨|latent-vis|⟩ · · ·

, ⟨|end-latent-vis|⟩, ⟨|start-latent|⟩, ⟨|latent|⟩ · · ·

Cv

Ct

All latent tokens are included in the prefill phase rather than the decode phase. Since modern transformers [99] process the entire prefill in parallel, these additional tokens add negligible overhead compared to sequential autoregressive generation. The model then generates only the trajectory tokens autoregressively. This yields inference latency nearly identical to answer-only AR prediction, while the main model’s processing of the prefilled latent tokens still implicitly activates the reasoning pathways learned during training. To conclude, the model outputs:

- • Trajectory prediction: The primary output—future waypoints for autonomous driving.
- • Language explanation (optional, via aux decoder): Human-readable CoT reasoning describing the model’s interpretation of the scene and its driving decision rationale.
- • Visual explanation (optional, via visual aux decoder): future frame visual tokens, providing a spatial preview of the predicted scene evolution.

Items 2 and 3 are available during post-hoc explanation generation (e.g., for human-in-the-loop [72] debugging, safety auditing, or human-robot interaction), while item 1 is always generated during inference.

#### 4 Three-Stage Training Pipeline

Training OneVL presents a unique optimization challenge. The main VLM, the language auxiliary decoder, and the visual auxiliary decoder must all be jointly optimized, yet they have fundamentally different learning objectives and start from different relative states of alignment.

We address this challenge through a training pipeline consisting of a preliminary self-supervised pretraining step followed by three main stages, each with a clear purpose. The configuration is summarized in Table 9.

###### 4.1 Preliminary: Visual Auxiliary Decoder Self-Supervised Pretraining

Motivation. Before integrating the visual auxiliary decoder into the full OneVL pipeline, we first pretrain it independently as a future-frame generator. The intuition is straightforward. Asking the decoder to immediately predict future frames conditioned on latent tokens that carry no information yet (early in training) is an illposed task that impedes learning. Instead, we first train the decoder with a strong unconditional prior—given the current-frame ViT embeddings, predict what the scene will look like at the next two timestamps—before introducing the latent conditioning signal. This preliminary stage is conceptually analogous to self-supervised video prediction, where the decoder learns purely from visual observations without any reasoning supervision.

Training Objective. The visual auxiliary decoder Dv receives only the current-frame ViT embeddings V (projected via Projv) as input—the visual latent token hidden states Hv are absent at this stage (the main model is not yet connected). Using the same concatenated target Tyv

= [Tyv,1, Tyv,2] defined in Section 3.4, the pretraining loss is:

|Tyv|

(Tyv,t | V, Tyv,<t) . (7)

Lp = −

log PD

v

t=1

From Unconditioned to Action-Conditioned Generation World Model. After pretraining, the decoder has learned a robust prior for visual dynamics, enabling it to predict plausible future frames from the current

scene alone. This component functions as the model’s implicit world model, capturing the underlying rules of visual evolution. When it is subsequently connected to the main model, the visual latent tokens Hv are introduced as an additional conditioning signal alongside the ViT embeddings. Since these latent tokens encode the driving agent’s planned action—derived from the main model’s reasoning—the decoder effectively transitions from unconditioned next-frame generation to action-conditioned rollouts of the world model. This framing provides a principled interpretation where the visual latent tokens serve as a compact, actionable representation that steers the world model’s predictions.

###### 4.2 Stage 0: Main Model Warmup

Motivation. The fundamental prerequisite for auxiliary decoders to provide meaningful supervision is that the main model’s latent tokens (i.e., <|latent-vis|> or <|latent|>) carry information that is semantically aligned with reasoning content. Without a targeted warmup, these tokens would not produce meaningful hidden states that auxiliary decoders can decode.

Stage 0 addresses this by training the main VLM end-to-end on the trajectory prediction task, with latent tokens embedded in each training sample’s assistant response. The model learns to:

- • Predict accurate trajectories: The main CE loss LCE over the latent tokens and trajectory answer tokens ensures the model develops a strong base prediction capability.
- • Develop meaningful latent representations: By contextualizing the latent tokens within the prompt-response structure alongside trajectory targets, the model naturally learns to use the latent positions to encode intermediate representations useful for trajectory prediction. Besides, the attention mechanism allows trajectory tokens to attend to latent token positions, establishing the information routing pathways that the auxiliary decoders will later exploit.

###### 4.3 Stage 1: Auxiliary Decoder Warmup

Motivation. With the main model producing stable, meaningful latent representations (as established in Stage 0), Stage 1 focuses exclusively on training the auxiliary decoders to align with these representations. Crucially, we freeze the main model during this stage, ensuring that the auxiliary decoders optimize against a consistent semantic distribution. By maintaining this stability, the decoders can more effectively internalize the mapping from fixed latent features to visual and language reasoning. To conclude, Stage 1 trains:

- • Language auxiliary decoder Dl: Trained to decode the language CoT reasoning text and fine-tuned with Ll against the ground-truth reasoning annotations.
- • Visual auxiliary decoder Dv: Trained to predict two future frames with Lv.

###### 4.4 Stage 2: Joint End-to-End Fine-tuning

Motivation. Finally, Stage 2 jointly fine-tunes all three model components with the combined loss L (Eq. 5). The gradients from Ll and Lv now flow back into the main model, directly shaping the latent representations to simultaneously serve trajectory prediction, language explanation, and visual prediction objectives. This creates a virtuous cycle:

- • The richer latent representations enable the main model to make better trajectory predictions (as the latent tokens carry more useful intermediate representations).
- • The auxiliary decoders adapt to the updated latent representations, improving their explanation quality.

This joint optimization is possible in Stage 2 precisely because both the main model and the auxiliary decoders are already well-initialized. Ablation studies (see Section 5.5) confirm that skipping three-stage training leads to degraded performance.

#### 5 Experiments

In this section, we present a comprehensive quantitative evaluation of OneVL across four benchmarks, followed by ablation studies and analyses that isolate the source of each performance gain.

###### 5.1 Datasets

We evaluate OneVL on four complementary benchmarks: NAVSIM [20], ROADWork [34], Impromptu [17], and Alpamayo-R1. These datasets are chosen because they have been shown to be effective in settings where CoT reasoning is employed, or because they provide sufficient labels to construct explicit reasoning traces [17, 78, 80].

- • NAVSIM [20] is a large-scale autonomous driving benchmark derived from nuPlan [54] driving logs, providing real-world data for non-reactive simulation-based planning evaluation.
- • ROADWork [34] targets autonomous navigation in road construction zones, featuring temporary signage, non-standard lane configurations, dynamic obstructions such as cones and barriers, and worker presence. These are all scenarios that remain underrepresented in standard driving benchmarks.
- • Impromptu [17] is a large-scale vision-language-action benchmark distilled from eight open driving datasets [8, 20, 26, 32, 83, 85, 98, 107]. It focuses on four types of unstructured corner-case scenarios and provides planning-oriented Q&A annotations together with trajectory data for training and evaluating autonomous driving models.
- • Alpamayo-R1 [105] introduces the Chain of Causation (CoC) annotations, featuring decision-grounded reasoning traces aligned with complex driving behaviors to enhance the interpretability and generalization of VLA models.

CoT Annotation Construction. A key challenge in training OneVL is obtaining high-quality chain-of-thought reasoning annotations paired with each driving scenario. On NAVSIM, we leverage the CoT annotations released by AdaThinkDrive [78], the previous state-of-the-art method. These annotations provide naturallanguage reasoning traces that cover scene interpretation (such as lane boundaries), critical object analysis (including vehicles and pedestrians), and the final driving intent. They are synthesized by a VLM that converts raw detection labels (e.g., objects and lanes) into reasoning sequences, and serve as the supervision target for the language auxiliary decoder (Ll). On ROADWork, we construct CoT annotations using a similar in-house pipeline. On Impromptu, we build CoT annotations from the original dataset’s Q&A pairs, augmenting the data with explicit decision and root-cause labels for corner-case trajectory prediction. On Alpamayo-R1, we use the released checkpoint to predict the CoC labels for all training examples. These annotations supervise the language auxiliary decoder, enabling the model to learn robust reasoning and decision-making logic in unstructured driving scenarios. Further details are provided in Appendix A.3.

###### 5.2 Experimental Setups

Evaluation Metrics. On NAVSIM, all methods are evaluated using the Predictive Driver Model (PDM) score, a composite metric that jointly assesses trajectory safety, comfort, and progress. On ROADWork, we report ADE (Average Displacement Error) and FDE (Final Displacement Error) to measure waypoint accuracy. On Impromptu, in addition to ADE and FDE, we also report the trajectory prediction L2 error over the first four seconds, following the protocol of the original paper. On Alpamayo-R1, we report ADE and FDE as well. Across all methods, we additionally report the average inference latency.

Baselines. We compare OneVL against two categories of baselines, all built on Qwen3-VL-4B-Instruct [5], together with previous state-of-the-art methods as stronger reference points. The AR-based methods that use standard autoregressive generation are:

• AR Answer: Direct autoregressive trajectory prediction without any reasoning. The model receives the front-view image and ego state and directly outputs trajectory waypoints. This is the fastest baseline and defines the latency lower bound.

Table 1 Performance comparisons on the NAVSIM benchmark. PDM-score (higher is better) and average inference latency (lower is better) for all methods. Symbol ∗ indicates the result is derived from the corresponding paper. For OneVL, we only count the parameters of the main VLM, as the auxiliary decoders are discarded during inference. The same applies to all subsequent models.

Method Model Size PDM-score ↑ Latency (s) ↓ Interpretability

Previous State-of-the-Art

- ◦ AdaThinkDrive [78] 8B 86.20∗ – Language
- ◦ LaST-VLA [80] 8B 87.30∗ – – AR-based Baselines (4B, Qwen3-VL)

- ◦ AR Answer 4B 87.47 4.49 –

- ◦ AR CoT+Answer 4B 88.29 6.58 Language Latent CoT Baselines (4B, Qwen3-VL)

- ◦ COCONUT [41] 4B 84.84 5.93 –
- ◦ CODI [87] 4B 83.92 8.62 –
- ◦ SIM-CoT [106] 4B 84.21 10.86 Language

• OneVL 4B 88.84 4.46 Vision + Language

• AR CoT+Answer: Standard CoT reasoning followed by trajectory prediction. The model first generates a full reasoning chain and then produces the trajectory. This represents the performance upper bound for explicit reasoning, at the cost of substantially higher latency.

The Latent CoT methods that use continuous latent representations for implicit reasoning are:

- • COCONUT [41]: Adapted for VLA-based autonomous driving. It uses curriculum learning to replace discrete reasoning tokens with continuous latent vectors.
- • CODI [87]: A COCONUT variant based on self-distillation, where a teacher model provides full textual CoT supervision and the student reasons in latent space.
- • SIM-CoT [106]: A CODI variant that adds a separate text-decoding auxiliary decoder for language interpretability.

We also compare against previous state-of-the-art methods reported in the literature. On NAVSIM (supervised fine-tuning setting), these methods are:

- • AdaThinkDrive [78]: An 8B-parameter model with adaptive CoT reasoning for autonomous driving.
- • LaST-VLA [80]: An 8B-parameter vision-language-action model for autonomous driving.

On ROADWork, we compare against YNet [34] and on Impromptu, we compare against the Impromptu VLA [17]. For Impromptu VLA, we report the result from our own replication. We also include a result obtained using the provided model checkpoint, which is shown in Table 10. On Alpamayo-R1, we compare against the Cosmos-Reason, a flow-matching-based VLA model. AR-based baselines are trained for 2 epochs with a learning rate of 4 × 10−5 and batch size 64. Latent CoT baselines are trained for 6 epochs with a learning rate of 4 × 10−5 and batch size 64. The results of previous state-of-the-art methods, where not explicitly stated otherwise, are taken directly from the literature.

###### 5.3 Main Results

- Table 1 presents the NAVSIM comparison results. Table 2 reports ADE/FDE on ROADWork. Table 3 and Table 4 report the score on Impromptu. Table 5 reports the overall results on Alpamayo-R1. Figure 1 provides a visual overview of all the results. We can make the following observations from the results.

- Table 2 Performance comparisons on the ROADWork benchmark. ADE and FDE (pixels; lower is better), latency (lower is better). Symbol ∗ indicates the result is derived from the corresponding paper.

Method ADE(pixel) ↓ FDE(pixel) ↓ Latency (s) ↓ Interpretability

Previous State-of-the-Art

- ◦ YNet [34] 22.68∗ 80.78∗ – – AR-based Baselines (4B, Qwen3-VL)

- ◦ AR Answer 15.98 40.29 4.74 –

- ◦ AR CoT+Answer 13.18 29.98 10.74 Language Latent CoT Baselines (4B, Qwen3-VL)

- ◦ COCONUT [41] 15.44 38.60 6.06 –

- ◦ CODI [87] 16.45 44.28 6.73 –
- ◦ SIM-CoT [106] 16.49 44.32 6.19 Language

• OneVL 12.49 28.80 4.71 Vision + Language

- Table 3 Performance comparisons on the Impromptu benchmark. ADE and FDE (meters; lower is better), latency (lower is better).

Method Model Size ADE(m) ↓ FDE(m) ↓ Latency (s) ↓ Interpretability

Previous State-of-the-Art

- ◦ Impromptu VLA [17] 3B 1.60 4.28 6.10 – AR-based Baselines (4B, Qwen3-VL)

- ◦ AR Answer 4B 1.46 4.03 4.24 –

- ◦ AR CoT+Answer 4B 1.42 3.96 6.84 Language Latent CoT Baselines (4B, Qwen3-VL)

- ◦ COCONUT [41] 4B 1.49 4.07 5.27 –
- ◦ CODI [87] 4B 1.86 5.18 5.24 –
- ◦ SIM-CoT [106] 4B 2.43 6.10 5.09 Language

• OneVL 4B 1.34 3.70 4.02 Vision + Language

OneVL achieves best performance. Specifically, OneVL achieves 88.84 PDM-score, surpassing the previous supervised finetuned SOTA AdaThinkDrive (86.20) and LaST-VLA (87.30) by +2.64 and +1.54, respectively. On ROADWork, OneVL also achieves 12.49 ADE and 28.80 FDE, significantly surpassing the previous SOTA YNet (22.68/80.78), respectively. The same pattern is observed on the Impromptu and Alpamayo-R1 datasets. Besides, OneVL outperforms all AR-based, latent CoT, and notably, explicit CoT baselines. This consistent superiority demonstrates the effectiveness of our multimodal auxiliary supervision approach in advancing planning tasks. On the Alpamayo-R1 dataset, OneVL underperforms in FDE with a slight margin (7.53 vs. 7.42). This is expected, as Cosmos-Reason uses RL to further enhance its capabilities.

Prefill inference matches answer-only prediction speed. On NAVSIM, OneVL with prefill inference achieves

- 4.46 latency, essentially identical to AR answer-only prediction (4.49s). This validates the core efficiency claim of our approach. By prefilling latent tokens into the prompt context (which is processed in a single parallel prefill pass), we incur negligible additional latency compared to sequential autoregressive generation of explicit reasoning tokens. On ROADWork, Impromptu, and Alpamayo-R1, we observe a similar speed-up effect, where

- Table 4 Performance comparisons on the Impromptu benchmark. We report trajectory prediction L2 error following the benchmark setting (meters; lower is better).

Traj. Pred. L2 Error (m) Method 1s ↓ 2s ↓ 3s ↓ 4s ↓ Avg. ↓

Previous State-of-the-Art

- ◦ Impromptu VLA [17] 0.14 0.60 1.45 2.67 1.22 AR-based Baselines (4B, Qwen3-VL)

- ◦ AR Answer 0.13 0.51 1.29 2.46 1.11

- ◦ AR CoT+Answer 0.13 0.51 1.27 2.44 1.09 Latent CoT Baselines (4B, Qwen3-VL)

- ◦ COCONUT [41] 0.15 0.54 1.32 2.50 1.13
- ◦ CODI [87] 0.17 0.63 1.61 3.13 1.39
- ◦ SIM-CoT [106] 0.41 1.10 2.25 3.94 1.93

• OneVL 0.13 0.48 1.18 2.25 1.01

- Table 5 Performance comparisons on the Alpamayo-R1 benchmark. ADE and FDE (meters; lower is better), latency (lower is better).

Method Model Size ADE(m) ↓ FDE(m) ↓ Latency (s) ↓ Interpretability

Previous State-of-the-Art

- ◦ Cosmos-Reason 10B 2.86 7.42 – Language AR-based Baselines (4B, Qwen3-VL)

- ◦ AR Answer 4B 3.27 9.59 3.06 –
- ◦ AR CoT+Answer 4B 2.99 8.54 3.51 Language Latent CoT Baselines (4B, Qwen3-VL)

- ◦ COCONUT [41] 4B 3.29 9.48 3.76 –
- ◦ CODI [87] 4B 3.22 9.25 3.85 –
- ◦ SIM-CoT [106] 4B 3.40 9.85 3.78 Language

• OneVL 4B 2.62 7.53 3.23 Vision + Language

OneVL achieves 4.71s, 4.02s, and 3.23s latency respectively, which are comparable to or even lower than those of answer-only prediction. In all cases, OneVL inference is much faster than the AR CoT+Answer baseline.

Existing latent CoT methods fail on VLA-based autonomous driving. All three adapted latent CoT methods perform substantially worse than answer-only AR prediction, except for the Alpamayo-R1 dataset. This is a critical finding that purely linguistic latent CoT approaches that were designed for text-only reasoning tasks do not transfer effectively to the multimodal spatial-temporal domain of autonomous driving trajectory prediction. The compact linguistic latent spaces these methods define are insufficient to support the geometric reasoning required for precise waypoint prediction. From the compression view of intelligence, this failure is fundamental rather than incidental. Concretely, these methods compress language descriptions of the scene, but language is already an abstraction of the physical world—it encodes semantic labels and relationships, not the spatial-temporal causal dynamics that determine future outcomes. In other words, it satisfies the efficiency criterion of the compression principle but not the intelligence criterion. A further likely factor is that

these baselines are not trained with OneVL’s staged recipe (Section 4), so their latent streams may remain poorly aligned with trajectory and multimodal supervision when optimization starts fully coupled.

OneVL overcomes this limitation by pairing compression with verification: the auxiliary decoders ensure that the latent bottleneck preserves spatially and linguistically meaningful content, rather than collapsing to degenerate representations. This is why OneVL is the only latent CoT method that outperforms the AR baseline. The three-stage training recipe is equally crucial, as it progressively aligns the bottleneck before joint optimization. More analysis can be found in Section 5.7.

Explicit CoT supervision helps: AR CoT+Answer vs. AR Answer. Comparing AR CoT+Answer (88.29) to AR Answer (87.47) on NAVSIM confirms that explicit reasoning supervision provides meaningful trajectory improvements (+0.80 PDM-score). We can observe similar patterns on ROADWork, Impromptu, and Alpamayo-R1. These results confirm the core motivation for incorporating CoT reasoning into autonomous driving systems and establish the quality ceiling that latent CoT methods aim to approach while reducing latency.

###### 5.4 Explanation Quality

Beyond trajectory prediction accuracy, OneVL is unique in providing human-interpretable explanations in both language and vision. Figures 4, 5, 6 and 7 provide NAVSIM, ROADWork, Impromptu and Alpamayo-R1 qualitative examples, including a side-by-side trajectory prediction comparison (vs. AR Answer baseline), two future frames from the visual auxiliary decoder, and the text CoT from the language auxiliary decoder. We also quantitatively evaluate the quality of text explanations.

Text CoT Quality. We evaluate the language explanations produced by the language auxiliary decoder against the ground-truth CoT annotations on the 500 NAVSIM test set. We compare against the AR CoT+Answer baseline (which generates explicit CoT autoregressively) and SIM-CoT (the only prior latent CoT method with language interpretability). We report three complementary metrics that capture different dimensions of explanation quality:

- • Meta Action Accuracy: Following Luo et al. [81], we report the meta action accuracy. On NAVSIM, each CoT concludes with a high-level driving decision (e.g., “the ego should maintain speed and keep lane”). We extract this meta-action clause from both the ground-truth and the predicted CoT and compute exact string match accuracy. This metric directly measures whether the model’s reasoning arrives at the correct driving intent, which is the most safety-critical aspect of CoT quality.
- • STS Score (Semantic Textual Similarity Score): We compute a neural semantic similarity score between each predicted CoT and its ground-truth reference using a cross-encoder reranker (BGE-reranker-v2-m3 [11]). This evaluator is particularly suitable for computing a similarity score for templated CoTs where most of the words are identical. A cross-encoder concatenates the ground truth and the prediction, processing them simultaneously through full token-by-token cross-attention. This mechanism allows the model to perform a deep, comparative analysis, making it highly sensitive to critical localized contradictions—such as predicting “slowly” instead of “fastly”, or outputting hazardous numerical distances. For each ground-truth-prediction pair, the cross-encoder outputs a raw relevance logit. We apply global min-max normalization across all methods and examples to obtain scores in [0,1]. This metric captures fine-grained semantic alignment beyond surface-level n-gram overlap.
- • LLM-as-Judge Score: Following Ishaq et al. [51], Luo et al. [81], we also employ a state-of-the-art proprietary VLM, gemini-3.1-flash-lite-preview, as an automated evaluator. Given the front-camera image, the ground-truth CoT, and the predicted CoT, the judge model scores each prediction on a 0-100 scale based on four criteria: (1) perception accuracy; (2) motion state prediction; (3) ego decision correctness, and (4) language fluency. Scores are normalized to [0,1]. This metric provides a holistic evaluation that accounts for both visual grounding and reasoning quality. The full evaluation prompt is provided in Appendix A.4.

Quantitative results are shown in Table 6. OneVL consistently outperforms SIM-CoT across all evaluation dimensions. Specifically, OneVL achieves a Meta Action Accuracy of 71.00, a significant 3.8 improvement over

Baseline Front View Baseline BEV

[Figure 37]

[Figure 38]

OneVL Front View OneVL BEV

[Figure 39]

[Figure 40]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 41]

[Figure 42]

CoT: The right side of the lane where ego vehicle is located is close to the undrivable area, so I need to drive slightly to the left. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should should maintain speed and turn left.

- Figure 4 Visualizations of prediction on NAVSIM. Each plot overlays ground-truth (green) and predicted (red)

- trajectories on the front camera view. More examples can be found in Appendix A.6.

SIM-CoT (67.20). This indicates that our latent representation more effectively captures and decodes the critical high-level driving intents. In terms of semantic alignment, OneVL reaches an STS score of 78.26 and an LLM Judge score of 79.13, narrowing the gap to the autoregressive baseline (AR CoT+Answer). While the AR model maintains the highest scores across all metrics, it requires explicit sequential text generation. This results in higher inference latency as demonstrated in Table 1. In contrast, OneVL achieves competitive explanation quality while benefiting from the efficiency of a latent reasoning framework. These results demonstrate that the language auxiliary decoder in OneVL successfully enables faithful, human-readable explanations of the model’s reasoning.

Baseline Front View OneVL Front View

[Figure 43]

[Figure 44]

OneVL T+1 OneVL T+2

[Figure 45]

[Figure 46]

OneVL Reasoning

CoT: <critical_factors>A line of cones and orange barrels forms a taper and channelizes traffic away from the right side of the roadway, constraining the ego vehicle’s available path and prompting a leftward trajectory. A work/utility truck is stopped in the right lane/near the right curb within the coned-off area, creating a physical blockage that the ego vehicle must steer around.Two workers in high-visibility clothing are standing near the rear/right side of the work truck inside the work zone, increasing caution and reinforcing the need to keep left of the channelized area. Temporary fencing and additional barrels along the right sidewalk edge indicate an active work area and further reduce usable roadway space on the right. A vehicle is present ahead in the left lane, influencing the ego vehicle’s merge/shift timing and requiring the ego vehicle to follow the open lane through the work zone.</critical_factors> <cause>Work zone entry deceleration and channelized lane keeping because cones/barrels taper traffic away from the right and a stopped work truck blocks the right side of the roadway.</cause> <decision>longitudinal:Work zone entry deceleration lateral:Channelized lane keeping</decision>

- Figure 5 Visualizations of prediction on ROADWork. Each plot overlays ground-truth (green) and predicted (red)

- trajectories on the front camera view. More examples can be found in Appendix A.7.

###### 5.5 Ablation Study

To understand the contribution of each component, we conduct ablation experiments by training additional models with different components. OneVL w/o language decoder, OneVL w/o visual decoder, and OneVL w/o staged train are trained. For OneVL w/o language decoder, and OneVL w/o visual decoder, we follow the same training recipe as OneVL, but without the language or visual auxiliary decoders. For OneVL w/o staged train, we use the same architecture as OneVL, but directly optimize the model with the end-to-end joint learning with the same training setting as stage 3. Results are shown in Table 7.

Comparing OneVL w/o visual decoder (87.97) to the full OneVL model (88.84), we find that the visual auxiliary decoder contributes +0.87 score. This result confirms that visual latent supervision—requiring the model to encode anticipated future scene content—provides complementary and additive benefit beyond language reasoning alone. A comparison between OneVL without its language decoder (88.53) and the full

Baseline Front View OneVL Front View

[Figure 47]

[Figure 48]

[Figure 49]

OneVL T+1 OneVL T+2

[Figure 50]

[Figure 51]

/e2e-data/evad-techvla/guanjiayi/dataset/nuscenes/ proj_results/nuScences_868_10

192.CAM_FRONT.png_proj.png

OneVL Reasoning

CoT： <decision>longitudinal decision: keep speed, lateral decision: straight</decision>\n<cause> The decision to

maintain current speed and go straight is likely due to the presence of a large building and parked cars on the left, which would obstruct a left turn. Additionally, the wet road conditions suggest caution, making a straight path the safer option.</cause>

- Figure 6 Visualizations of prediction on Impromptu. Each plot overlays ground-truth (green) and predicted (red)

- trajectories on the front camera view. More examples can be found in Appendix A.8.

OneVL model (88.84) validates the contribution of the language auxiliary decoder and language latent tokens, yielding a modest performance gain of +0.31. While the absolute improvement is small, it demonstrates that encoding language-based reasoning into compact latent tokens can enhance trajectory prediction performance.

On the efficacy of three-stage training. The ablation study comparing OneVL without three-stage training to the full model yields a conclusive result: the proposed three-stage training strategy is not merely beneficial but essential. Direct end-to-end joint fine-tuning fails catastrophically, causing the PDM-Score to drop by 21.71 points (from 88.84 to 67.13). This drastic performance gap underscores the necessity of the staged approach.

A detailed examination of the training dynamics reveals the underlying causes. First, the direct approach suffers from severe “gradient shock” at initialization, with an exploding gradient norm of 378.22 that destabilizes the pre-trained backbone. In contrast, the three-stage strategy maintains a stable gradient norm of 0.28 by properly warming up the latent tokens. Second, the end-to-end method causes catastrophic task interference. The backbone struggles to optimize conflicting objectives simultaneously, resulting in a much higher final trajectory prediction loss (0.186 vs. 0.136), showing that the model converges into a local minimum and consequently has poor driving performance.

We further examine the visual quality of predicted future frames. Figure 8 compares decoded future frames

Baseline Front View OneVL Front View

[Figure 52]

[Figure 53]

OneVL T+1 OneVL T+2

[Figure 54]

[Figure 55]

OneVL Reasoning

CoT: Yield to the pedestrian crossing the crosswalk ahead.

- Figure 7 Visualizations of prediction on Alpamayo-R1. Each plot overlays ground-truth (green) and predicted (red)

- trajectories on the front camera view. More examples can be found in Appendix A.9.

Front View (with Traj.) BEV Reasoning T+1 T+2

(a) OneVL full receipt

(b) OneVL w/o three-stage training

CoT: I don't need to pay attention to any road boundaries. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should maintain speed and change to the left lane.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

CoT: The right side of the lane where ego vehicle is located is close to the undrivable area, so I need to drive slightly to the left. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should maintain speed and keep lane.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

- Figure 8 Visual CoT under full training vs. an overfitting visual auxiliary decoder. (a) Top row: with the full training recipe, latent visual tokens decode to future frames that remain scene-consistent and usable as spatial-temporal reasoning supervision. (b) Bottom row: when without three-stage training, decoding from the same input collapses to memorized artifacts rather than generalizing—the predicted “future” frames are visually irrelevant to the input image.

###### from the same NAVSIM test image. Under the full OneVL recipe (a), the visual auxiliary decoder produces

- Table 6 Text CoT quality on the NAVSIM test set. Meta Action Acc. measures exact match of the predicted driving decision against the ground-truth meta action. STS score reports the mean semantic similarity score (global min-max normalized). LLM Judge reports the mean VLM evaluation score (normalized to [0, 1]; see Appendix A.4 for the full prompt). Higher is better for all metrics.

Method Meta Action Acc. ↑ STS ↑ LLM Judge ↑ Avg. ↑

- ◦ AR CoT+Answer 73.20 79.75 81.86 78.27
- ◦ SIM-CoT 67.20 76.25 78.73 74.06

• OneVL (lang. aux.) 71.00 78.26 79.13 76.13

Table 7 Ablation study on each component in OneVL.

Model Variant Lang. Aux. Dec. Vis. Aux. Dec. Staged Train PDM-score ↑

- ◦ OneVL w/o vis. dec. – 87.97
- ◦ OneVL w/o lang.e dec. – 88.53

- ◦ OneVL w/o staged train – 67.13

• OneVL (full configuration) 88.84

spatially coherent previews for t+1 and t+2 that match plausible ego motion and scene layout, demonstrating a consistent and reasonable visual chain-of-thought. In contrast, for the ablated training setup (b), the decoded frames are totally irrelevant to the input, which introduces noise.

Besides, the language CoT reasoning in (b) is also erroneous. This demonstrates that without the staged training, the model memorized training patterns rather than learning real-world dynamics. We also observe that the full recipe’s visual explain loss (i.e., Lv) curve is smooth and stable across training, whereas the ablation curve exhibits a pronounced spike at the beginning. All observations indicate that the visual decoder fails to learn reliable scene dynamics without staged training. This indicates that without the three-stage curriculum, the visual auxiliary decoder overfits, taking shortcuts and memorizing data instead of learning to generalize.

Consequently, the visual supervision signal fails to improve the final trajectory prediction and instead introduces noise that degrades overall performance. This observation underscores that effective compression requires careful optimization: the three-stage curriculum ensures the bottleneck learns to encode generalizable scene dynamics rather than memorized shortcuts, a concrete instance of how the quality of compression determines the quality of intelligence.

###### 5.6 Towards Real-World Deployment

Closed-loop on-vehicle testing favors extremely low-latency trajectory prediction. However, autoregressive decoding waypoints still dominate the budget even when latent reasoning is prefilled. To explore a deploymentoriented variant, we append a compact MLP head on top of the same Qwen3-VL-4B-Instruct backbone (using the hidden state of the last latent token as the input). Therefore, the model can autoregressively predict the trajectory using the LLM head as well as predict the trajectory with a single feed-forward pass using the MLP head. This variant can still exploit multimodal latent supervision during training.

Table 8 compares this MLP variant against full OneVL on NAVSIM. The MLP head reaches 86.83 PDM-score with 0.24s inference latency, whereas full OneVL keeps 88.84 with 4.46s inference latency, achieving a 5.4% of the AR model’s latency. At 1/0.2405 ≈ 4.16Hz end-to-end, the MLP variant meets a typical on-vehicle few-Hz budget while remaining in a competitive performance relative to strong prior models (e.g., LaST-VLA). This promising outcome supports the potential for real-world vehicle deployment.

Table 8 Accuracy-latency trade-off for real-time deployment on NAVSIM.

Variant PDM-score ↑ latency (s) ↓

◦ OneVL (regression) 86.83 0.24 • OneVL (AR) 88.84 4.46

###### 5.7 In-Depth Analysis: Where Does the Benefit Come From?

Implicit Reasoning vs. Explicit Tokens. A natural question is why OneVL’s implicit latent reasoning achieves better performance than explicit AR CoT+Answer (88.84 vs. 88.29), despite using fewer tokens to express the reasoning. We hypothesize two mechanisms:

First, the compression benefit: compact latent tokens force the model to distill the most trajectory-relevant reasoning into a small representational bottleneck, filtering out irrelevant or redundant content. This is precisely the mechanism predicted by the information bottleneck principle [96]. Tighter compression discards noise and retains only the causal features that are predictive of the output, yielding representations that generalize better than verbose, free-form CoT chains, where tangential reasoning may introduce noise into the trajectory prediction.

Second, the world model grounding benefit: the visual auxiliary decoder objective explicitly requires the visual latent tokens to encode spatial-temporal scene dynamics (future frame content), which is directly relevant to trajectory prediction. This is a world model supervision signal—predicting what the scene will look like forces the compressed latents to internalize the causal dynamics of agent motion and road geometry. Explicit language CoT does not have an analogous spatial grounding mechanism; it describes the world symbolically, leaving the causal geometry implicit.

Why Visual Supervision Helps More Than Language Supervision. The visual auxiliary decoder contributes +0.87 PDM-score compared to +0.31 for the language auxiliary decoder. This asymmetry reflects the world model role the visual decoder plays. Autonomous driving trajectory prediction is fundamentally a spatial prediction task, and visual token reconstruction, i.e., predicting what the scene looks like 0.5–1.0 seconds later, provides a supervision signal that is inherently aligned with the geometric nature of trajectory prediction.

Crucially, future-frame prediction is a world model objective: to minimize reconstruction loss on unseen configurations of agents and road geometry, the visual latent tokens must encode the causal dynamics of the scene, not just its current appearance. Language CoT annotation describes the reasoning process in abstract, symbolic terms; it is valuable for semantic grounding but is one step removed from the physical dynamics that drive trajectory outcomes. The world model decoder thus provides a harder, more causally direct compression target that language supervision alone cannot supply, which is precisely why it contributes a larger performance gain.

Latency Analysis. The prefill inference mechanism achieves its speed advantage because modern transformer implementations process the prefill sequence in a single parallel forward pass, while autoregressive decoding requires sequential token generation. With several additional latent tokens in the prefill phase, the overhead is negligible compared to the much longer image patch sequence already in the prefill context. For example, on NAVSIM, the result is that prefill-mode OneVL (4.46s) is essentially indistinguishable from AR Answer (4.49s) in terms of latency, matching the speed of answer-only autoregressive prediction. Further experiment in Section 5.6 demonstrates that by introducing the MLP head, OneVL can achieve the optimal inference speed (0.24s) at the cost of performance.

Why Prior Latent CoT Methods Fail on Autonomous Driving. The catastrophic failure of COCONUT, CODI, and SIM-CoT warrants deeper analysis. These methods were designed under the assumption that latent representations can compress reasoning quality from teacher text generation into student latent tokens via semantic alignment. However, in the autonomous driving domain, these methods fail because:

- • Lack of visual world model supervision: Without the visual auxiliary decoder forcing the latent tokens

- to encode spatial-temporal scene content, the latent representations collapse to encoding only languagelevel abstractions, losing the geometric precision needed for trajectory prediction. In the framing of the compression view of intelligence, these methods compress a symbolic abstraction of the world rather than the world itself—the compression target (language) is too abstract to drive the model toward genuinely causal scene representations. The visual decoder functions as a world model auxiliary: it provides a concrete, physically grounded compression target (future visual observations) that cannot be satisfied by language-level memorization. This is directly supported by our ablation. OneVL w/o visual decoder scores 87.97, still below the full model (88.84), removing the world model supervision alone accounts for a −0.87 drop, even when all other components remain intact.
- • Absence of staged training: Prior methods are typically optimized without OneVL’s three-stage pipeline (Section 4), leaving language latents, visual latents, and trajectory prediction poorly aligned at the start of optimization. Our ablation isolates this effect. OneVL w/o visual decoder but with staged training still reaches 87.97, well above the no-staged-training collapse (67.13) and above prior latent CoT baselines, demonstrating that the curriculum alone provides substantial gains independent of the visual decoder. Combining both yields the full 88.84, confirming that staged training and world model supervision are complementary. The curriculum creates a stable latent space that the world model decoder can then push toward genuinely causal representations.

To conclude, OneVL addresses these failure modes as follows: (1) the visual auxiliary decoder provides precise spatial supervision and maintains spatial-temporal grounding in the latent representations; (2) the three-stage training recipe avoids the optimization conflict above by progressively aligning auxiliary decoders and latent tokens before full joint optimization.

#### 6 Conclusion

We presented OneVL, a framework for autonomous driving trajectory prediction built on a central hypothesis compression drives generalization. A key contribution lies in identifying why prior latent CoT methods fail on planning tasks. Compressing language is not the same as compressing scene dynamics. Natural language descriptions of driving scenes are inherently abstract, encoding semantic labels rather than the physical causal structure that determines future outcomes. As a result, compressing language satisfies the efficiency requirement of the compression principle but not its intelligence requirement. OneVL closes this gap by introducing a world model auxiliary in the form of a visual decoder that predicts future-frame visual tokens. Future-frame prediction is a concrete, causally grounded compression target: a model that can anticipate the scene’s visual evolution has necessarily internalized the dynamics governing agent motion and scene interaction, information that language alone cannot reliably encode.

At inference time, the decoders are discarded, and the prefilled latent tokens enable single-pass trajectory generation at a latency matching answer-only prediction. A three-stage training pipeline is essential for realizing this compression: Stage 0 establishes meaningful latent representations, Stage 1 aligns the decoders against a stable latent space, and Stage 2 tightens the bottleneck from both sides through bidirectional joint optimization. Ablations show that skipping this curriculum causes the compression to collapse into memorization, with a catastrophic performance drop.

Empirically, OneVL achieves state-of-the-art results on NAVSIM, ROADWork, Impromptu, and AlpamayoR1 while matching answer-only inference speed. Most notably, OneVL is the only latent CoT method to outperform explicit autoregressive CoT, providing direct evidence that tighter compression, when grounded in both linguistic and world-model supervision, yields more effective reasoning than verbose token-by-token generation. Finally, by appending a lightweight MLP regression head, OneVL retains competitive performance at only 5% of the original latency, pointing toward efficient real-world deployment.

Limitations. The current system requires roughly 3× memory during training, since three full 4B model instances must be held in memory. This is mitigated by DeepSpeed ZeRO-2 but still imposes nontrivial infrastructure requirements. In addition, the latent token count was chosen empirically. Thus, a systematic study of the trade-off between latent token count and representation capacity is left for future work.

Future Directions. Several directions emerge naturally from this work. First, while OneVL’s prefill mechanism eliminates latent CoT overhead, the trajectory tokens themselves are still generated autoregressively. As shown in Section 5.6, appending an MLP head can improve inference speed at some cost to performance, so the overall latency remains constrained by AR decoding.

Bridging this gap, for instance, through parallel or non-autoregressive trajectory decoding, is a key step toward true real-time deployment in safety-critical driving systems. Second, extending the world model decoder to multi-camera inputs would enable 360-degree future-scene prediction and more comprehensive causal scene understanding, further strengthening the compression targets available to the visual latents. Third, the dual-modal explanation framework could enable novel human-machine interface designs in which drivers receive real-time visual and verbal justifications for the vehicle’s planning decisions. Beyond enhancing transparency and trust, this also has the potential to supply richer training signals for reinforcement learning, closing the loop between world model prediction and policy improvement.

#### 7 Contributions and Acknowledgments

###### Core Contributors

- • Jinghui Lu
- • Jiayi Guan
- • Zhijian Huang
- • Jinlong Li
- • Guang Li
- • Lingdong Kong
- • Yingyan Li
- • Han Wang
- • Shaoqing Xu
- • Yuechen Luo
- • Fang Li
- • Chenxu Dang
- • Junli Wang
- • Tao Xu
- • Jing Wu
- • Jianhua Wu
- • Xiaoshuai Hao
- • Wen Zhang
- • Tianyi Jiang
- • Kuiyuan Yang
- • Hangjun Ye
- • Long Chen†

†Corresponding author.

- Contributors
- • Lingfeng Zhang
- • Lei Zhou
- • Yingbo Tang
- • Jie Wang
- • Yinfeng Gao
- • Xizhou Bu
- • Haochen Tian
- • Yihang Qiu
- • Feiyang Jia
- • Lin Liu
- • Yigu Ge
- • Hanbing Li
- • Yuannan Shen
- • Jianwei Cui
- • Hongwei Xie
- • Bing Wang
- • Haiyang Sun
- • Jingwei Zhao
- • Jiahui Huang
- • Pei Liu
- • Zeyu Zhu
- • Yuncheng Jiang
- • Zibin Guo
- • Chuhong Gong
- • Hanchao Leng
- • Kun Ma
- • Naiyan Wang
- • Guang Chen

Acknowledgments

In this work, we utilize the Qwen3-VL-4B-Instruct [5] model and the following datasets: NAVSIM [20], ROADWork [34], Impromptu [17], and Alpamayo-R1 [105]. The authors confirm that the use of these resources is strictly for academic research purposes and have not been involved in any commercial activities.

#### References

- [1] Anthropic. Claude 3.7 Sonnet and Claude Code. https://www.anthropic.com/news/claude-3-7-sonnet, 2025.
- [2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023.

- [3] Alisson Azzolini, Junjie Bai, Hannah Brandon, Jiaxin Cao, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, et al. Cosmos-Reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025.

- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

- [5] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

- [6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

- [7] Hengwei Bian, Lingdong Kong, Haozhe Xie, Liang Pan, Yu Qiao, and Ziwei Liu. DynamicCity: Large-scale 4D occupancy generation from dynamic scenes. In International Conference on Learning Representations, 2025.

- [8] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11621–11631, 2020.

- [9] Xu Cao, Tong Zhou, Yunsheng Ma, Wenqian Ye, Can Cui, Kun Tang, Zhipeng Cao, Kaizhao Liang, Ziran Wang, James M Rehg, et al. MapLM: A real-world large-scale vision-language benchmark for map and traffic scene understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21819–21830, 2024.

- [10] Jiahong Chen, Jing Wang, Long Chen, Chuwei Cai, and Jinghui Lu. NanoVLA: Routing decoupled vision-language understanding for nano-sized generalist robotic policies. arXiv preprint arXiv:2510.25122, 2025.

- [11] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-Embedding: Multilinguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024.

- [12] Kai Chen, Yanze Li, Wenhua Zhang, Yanxin Liu, Pengxiang Li, Ruiyuan Gao, Lanqing Hong, Meng Tian, Xinhai Zhao, Zhenguo Li, et al. Automated evaluation of large vision-language models on self-driving corner cases. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7817–7826, 2025.

- [13] Long Chen, Oleg Sinavski, Jan Hünermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, and Jamie Shotton. Driving with LLMs: Fusing object-level vector modality for explainable autonomous driving. In IEEE International Conference on Robotics and Automation, pages 14093–14100, 2024.

- [14] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

- [15] Qimao Chen, Fang Li, Shaoqing Xu, Zhiyi Lai, Zixun Xie, Yuechen Luo, Shengyin Jiang, Hanbing Li, Long Chen, Bing Wang, et al. VILTA: A VLM-in-the-loop adversary for enhancing driving policy robustness. arXiv preprint arXiv:2601.12672, 2026.

- [16] Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171, 2024.

- [17] Haohan Chi, Huan ang Gao, Ziming Liu, Jianing Liu, Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu, Zeda Wang, Wenyi Li, Leichen Wang, Xingtao Hu, Hao Sun, Hang Zhao, and Hao Zhao. Impromptu VLA: Open weights and open data for driving vision-language-action models. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), volume 38, 2025.

- [18] Meng Chu, Xuan Billy Zhang, Kevin Qinghong Lin, Lingdong Kong, Jize Zhang, Teng Tu, Weijian Ma, Ziqi Huang, Senqiao Yang, Wei Huang, Yeying Jin, Zhefan Rao, Jinhui Ye, Xinyu Lin, Xichen Zhang, Qisheng Hu, Shuai Yang, Leyang Shen, Wei Chow, Yifei Dong, Fengyi Wu, Quanyu Long, Bin Xia, Shaozuo Yu, Mingkang Zhu, Wenhu Zhang, Jiehui Huang, Haokun Gui, Haoxuan Che, Long Chen, Qifeng Chen, Wenxuan Zhang, Wenya Wang, Xiaojuan Qi, Yang Deng, Yanwei Li, Mike Zheng Shou, Zhi-Qi Cheng, See-Kiong Ng, Ziwei Liu, Philip Torr, and Jiaya Jia. Agentic world modeling: Foundations, capabilities, laws, and beyond. arXiv preprint arXiv:2604.22748, 2026.

- [19] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

- [20] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2024.

- [21] Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, et al. Language modeling is compression. arXiv preprint arXiv:2309.10668, 2023.

- [22] Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit CoT to implicit CoT: Learning to internalize CoT step by step. arXiv preprint arXiv:2405.14838, 2024.

- [23] Xinpeng Ding, Jianhua Han, Hang Xu, Xiaodan Liang, Wei Zhang, and Xiaomeng Li. Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13668–13677, 2024.

- [24] Yifei Dong, Fengyi Wu, Guangyu Chen, Lingdong Kong, Xu Zhu, Qiyu Hu, Yuxuan Zhou, Jingdong Sun, Jun-Yan He, Qi Dai, Alexander G. Hauptmann, and Zhi-Qi Cheng. Towards unified world models for visual navigation via memory-augmented planning and foresight. arXiv preprint arXiv:2510.08713, 2025.

- [25] Yifei Dong, Fengyi Wu, Yilong Dai, Lingdong Kong, Guangyu Chen, Xu Zhu, Qiyu Hu, Tianyu Wang, Johnalbert Garnica, Feng Liu, et al. Language-conditioned world modeling for visual navigation. arXiv preprint arXiv:2603.26741, 2026.

- [26] Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai, Ben Sapp, Charles R Qi, Yin Zhou, et al. Large scale interactive motion forecasting for autonomous driving: The Waymo open motion dataset. In IEEE/CVF International Conference on Computer Vision, pages 9710–9719, 2021.

- [27] Xiang Fei, Jinghui Lu, Qi Sun, Hao Feng, Yanjie Wang, Wei Shi, An-Lan Wang, Jingqun Tang, and Can Huang. Advancing sequential numerical prediction in autoregressive models. In Annual Meeting of the Association for Computational Linguistics, pages 562–574, 2025.

- [28] Hao Feng, Shu Wei, Xiang Fei, Wei Shi, Yingdong Han, Lei Liao, Jinghui Lu, Binghong Wu, Qi Liu, Chunhui Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. In Annual Meeting of the Association for Computational Linguistics, pages 21919–21936, 2025.

- [29] Tuo Feng, Wenguan Wang, and Yi Yang. A survey of world models for autonomous driving. arXiv preprint arXiv:2501.11260, 2025.

- [30] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. In IEEE/CVF International Conference on Computer Vision, pages 24823–24834, 2025.

- [31] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Hongwei Xie, Bing Wang, Guang Chen, Dingkang Liang, and Xiang Bai. MindDrive: A vision-language-action model for autonomous driving via online reinforcement learning. arXiv preprint arXiv:2512.13636, 2025.

- [32] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The KITTI dataset. International Journal of Robotics Research, 32(11):1231–1237, 2013.

- [33] Jiaheng Geng, Jiatong Du, Xinyu Zhang, Ye Li, Panqu Wang, and Yanjun Huang. Driving in corner case: A real-world adversarial closed-loop evaluation platform for end-to-end autonomous driving. arXiv preprint arXiv:2512.16055, 2025.

- [34] Anurag Ghosh, Shen Zheng, Robert Tamburo, Khiem Vuong, Juan Alvarez-Padilla, Hailiang Zhu, Michael Cardei, Nicholas Dunn, Christoph Mertz, and Srinivasa G. Narasimhan. ROADWork: A dataset and benchmark for learning to recognize, observe, analyze and drive through work zones. In IEEE/CVF International Conference on Computer Vision, pages 6132–6142, 2025.

- [35] Google. Gemini 2.5 Pro preview: even better coding performance. https://developers.googleblog.com/en/ gemini-2-5-pro-io-improved-coding-performance, 2025.
- [36] Yanchen Guan, Haicheng Liao, Zhenning Li, Jia Hu, Runze Yuan, Guohui Zhang, and Chengzhong Xu. World models for autonomous driving: An initial survey. IEEE Transactions on Intelligent Vehicles, pages 1–17, 2024.

- [37] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [38] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1.5-VL technical report. arXiv preprint arXiv:2505.07062, 2025.

- [39] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.

- [40] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.

- [41] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

- [42] Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, Yuchen Zhang, Jing Wu, Jinghui Lu, Chenxu Dang, Jiayi Guan, Jianhua Wu, Zhiyi Hou, Hanbing Li, Shumeng Xia, Mingliang Zhou, Yinan Zheng, Zihao Yue, Shuhao Gu, Hao Tian, Yuannan Shen, Jianwei Cui, Wen Zhang, Shaoqing Xu, Bing Wang, Haiyang Sun, Zeyu Zhu, Yuncheng Jiang, Zibin Guo, Chuhong Gong, Chaofan Zhang, Wenbo Ding, Kun Ma, Guang Chen, Rui Cai, Diyun Xiang, Heng Qu, Fuli Luo, Hangjun Ye, and Long Chen. MiMo-Embodied: X-embodied foundation model technical report. arXiv preprint arXiv:2511.16518, 2025.

- [43] Zhiyi Hou, Enhui Ma, Fang Li, Zhiyi Lai, Kalok Ho, Zhanqian Wu, Lijun Zhou, Long Chen, Chitian Sun, Haiyang Sun, et al. DriveMRP: Enhancing vision-language models with synthetic motion data for motion risk prediction. arXiv preprint arXiv:2507.02948, 2025.

- [44] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

- [45] Tianshuai Hu, Xiaolu Liu, Song Wang, Yiyao Zhu, Ao Liang, Lingdong Kong, Guoyang Zhao, Zeying Gong, Jun Cen, Zhiyu Huang, Xiaoshuai Hao, Linfeng Li, Hang Song, Xiangtai Li, Jun Ma, Shaojie Shen, Jianke Zhu, Dacheng Tao, Ziwei Liu, and Junwei Liang. Vision-language-action models for autonomous driving: Past, present, and future. arXiv preprint arXiv:2512.16760, 2025.

- [46] Tianshuai Hu, Zeying Gong, Lingdong Kong, Xiaodong Mei, Yiyi Ding, Qi Zeng, Ao Liang, Rong Li, Yangyi Zhong, and Junwei Liang. NavThinker: Action-conditioned world models for coupled prediction and planning in social navigation. arXiv preprint arXiv:2603.15359, 2026.

- [47] Zhijian Huang, Sihao Lin, Guiyu Liu, Mukun Luo, Chaoqiang Ye, Hang Xu, Xiaojun Chang, and Xiaodan Liang. Fuller: Unified multi-modality multi-task 3D perception via multi-level gradient calibration. In IEEE/CVF International Conference on Computer Vision, pages 3502–3511, 2023.

- [48] Zhijian Huang, Tao Tang, Shaoxiang Chen, Sihao Lin, Zequn Jie, Lin Ma, Guangrun Wang, and Xiaodan Liang. Making large language models better planners with reasoning-decision alignment. In European Conference on Computer Vision, pages 73–90. Springer, 2024.

- [49] Zhijian Huang, Chengjian Feng, Feng Yan, Baihui Xiao, Zequn Jie, Yujie Zhong, Xiaodan Liang, and Lin Ma. RoboTron-Drive: All-in-one large multimodal model for autonomous driving. In IEEE/CVF International Conference on Computer Vision, pages 8011–8021, 2025.

- [50] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [51] Ayesha Ishaq, Jean Lahoud, Ketan More, Omkar Thawakar, Ritesh Thawkar, Dinura Dissanayake, Noor Ahsan, Yuhao Li, Fahad Shahbaz Khan, Hisham Cholakkal, Ivan Laptev, Rao Muhammad Anwer, and Salman Khan. DriveLMM-o1: A step-by-step reasoning dataset and large multimodal model for driving scenario understanding. arXiv preprint arXiv:2503.10621, 2025.

- [52] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, et al. OpenAI o1 system card. arXiv preprint arXiv:2412.16720, 2024.

- [53] Weitao Jia, Jinghui Lu, Haiyang Yu, Siqi Wang, Guozhi Tang, An-Lan Wang, Weijie Yin, Dingkang Yang, Yuxiang Nie, Bin Shan, et al. MEML-GRPO: Heterogeneous multi-expert mutual learning for RLVR advancement. In AAAI Conference on Artificial Intelligence, volume 40, pages 31283–31291, 2026.

- [54] Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, et al. Towards learningbased planning: The nuPlan benchmark for real-world autonomous driving. In IEEE International Conference on Robotics and Automation, pages 629–636, 2024.

- [55] Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Yaru Niu, Wei Tsang Ooi, Benoit R. Cottereau, Lai Xing Ng, Yuexin Ma, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu, Weichao Qiu, Wei Zhang, Xu Cao, Hao Lu, Ying-Cong Chen, Caixin Kang, Xinning Zhou, Chengyang Ying, Wentao Shang, Xingxing Wei, Yinpeng Dong, Bo Yang, Shengyin Jiang, Zeliang Ma, Dengyi Ji, Haiwen Li, Xingliang Huang, Yu Tian, Genghua Kou, Fan Jia, Yingfei Liu, Tiancai Wang, Ying Li, Xiaoshuai Hao, Yifan Yang, Hui Zhang, Mengchuan Wei, Yi Zhou, Haimei Zhao, Jing Zhang, Jinke Li, Xiao He, Xiaoqiang Cheng, Bingyang Zhang, Lirong Zhao, Dianlei Ding, et al. The RoboDrive challenge: Drive anytime anywhere in any condition. arXiv preprint arXiv:2405.08816, 2024.

- [56] Lingdong Kong, Xiang Xu, Jiawei Ren, Wenwei Zhang, Liang Pan, Kai Chen, Wei Tsang Ooi, and Ziwei Liu. Multi-modal data-efficient 3D scene understanding for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(5):3748–3765, 2025.

- [57] Lingdong Kong, Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu, Mingkai Jia, Junyuan Deng, Kaiwen Zhang, Yang Wu, Tianyi Yan, Shenyuan Gao, Song Wang, Linfeng Li, Liang Pan, Yong Liu, Jianke Zhu, Wei Tsang Ooi, Steven C. H. Hoi, and Ziwei Liu. 3D and 4D world modeling: A survey. arXiv preprint arXiv:2509.07996, 2025.

- [58] Lingdong Kong, Xiang Xu, Youquan Liu, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Ziwei Liu. LargeAD: Large-scale cross-sensor data pretraining for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(2):1291–1308, 2026.

- [59] Shane Legg and Marcus Hutter. Universal intelligence: A definition of machine intelligence. Minds and Machines, 17(4):391–444, 2007.

- [60] Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing endto-end autonomous driving with latent world model. In International Conference on Learning Representations, 2025.

- [61] Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via BEV world model. In IEEE/CVF International Conference on Computer Vision, pages 27137–27146, 2025.

- [62] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, and Zhaoxiang Zhang. DriveVLA-W0: World models amplify data scaling law in autonomous driving. In International Conference on Learning Representations, 2026.

- [63] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Kun Ma, Guang Chen, Hangjun Ye, Wenyu Liu, and Xinggang Wang. ReCogDrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052, 2025.

- [64] Alan Liang, Youquan Liu, Yu Yang, Dongyue Lu, Linfeng Li, Lingdong Kong, Huaici Zhao, and Wei Tsang Ooi. LiDARCrafter: Dynamic 4D world modeling from LiDAR sequences. AAAI Conference on Artificial Intelligence, 40(22):18406–18414, 2026.

- [65] Ao Liang, Lingdong Kong, Tianyi Yan, Hongsi Liu, Wesley Yang, Ziqi Huang, Wei Yin, Jialong Zuo, Yixuan Hu, Dekai Zhu, Dongyue Lu, Youquan Liu, Guangfeng Jiang, Linfeng Li, Xiangtai Li, Long Zhuo, Lai Xing Ng, Benoit R. Cottereau, Changxin Gao, Liang Pan, Wei Tsang Ooi, and Ziwei Liu. WorldLens: Full-spectrum evaluations of driving world models in real world. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

- [66] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, 2024.

- [67] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pages 34892–34916, 2023.

- [68] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.

- [69] Lin Liu, Caiyan Jia, Guanyi Yu, Ziying Song, Junqiao Li, Feiyang Jia, Peiliang Wu, Xiaoshuai Hao, and Yadan Luo. GuideFlow: Constraint-guided flow matching for planning in end-to-end autonomous driving. arXiv preprint arXiv:2511.18729, 2025.

- [70] Lin Liu, Ziying Song, Caiyan Jia, Hangjun Ye, Xiaoshuai Hao, and Long Chen. DriveWorld-VLA: Unified latentspace world modeling with vision-language-action for autonomous driving. arXiv preprint arXiv:2602.06521, 2026.

- [71] Xueyi Liu, Zuodong Zhong, Junli Wang, Yuxin Guo, Zhiguo Su, Qichao Zhang, Yinfeng Gao, Yupeng Zheng, Donbin Zhao, et al. ReasonPlan: Unified scene prediction and decision reasoning for closed-loop autonomous driving. In Conference on Robot Learning, pages 3051–3068. PMLR, 2025.

- [72] Jinghui Lu, Linyi Yang, Brian Namee, and Yue Zhang. A rationale-centric framework for human-in-the-loop machine learning. In Annual Meeting of the Association for Computational Linguistics, pages 6986–6996, 2022.

- [73] Jinghui Lu, Rui Zhao, Brian Mac Namee, and Fei Tan. PUnifiedNER: a prompting-based unified ner system for diverse datasets. In Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23. AAAI Press, 2023. ISBN 978-157735-880-0. doi: 10.1609/aaai.v37i11.26564. URL https://doi.org/10.1609/aaai.v37i11.26564.

- [74] Jinghui Lu, Dongsheng Zhu, Weidong Han, Rui Zhao, Brian Mac Namee, and Fei Tan. What makes pre-trained language models better zero-shot learners? In Annual Meeting of the Association for Computational Linguistics, pages 2288–2303, 2023.

- [75] Jinghui Lu, Ziwei Yang, Yanjie Wang, Xuejing Liu, Brian Mac Namee, and Can Huang. PaDeLLM-NER: Parallel decoding in large language models for named entity recognition. In Advances in Neural Information Processing Systems, volume 37, pages 117853–117880, 2024.

- [76] Jinghui Lu, Haiyang Yu, Yanjie Wang, Yongjie Ye, Jingqun Tang, Ziwei Yang, Binghong Wu, Qi Liu, Hao Feng, Han Wang, Hao Liu, and Can Huang. A bounding box is worth one token - interleaving layout and text in a large language model for document understanding. In Annual Meeting of the Association for Computational Linguistics, pages 7252–7273, 2025.

- [77] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35, pages 2507–2521, 2022.

- [78] Yuechen Luo, Fang Li, Shaoqing Xu, Zhiyi Lai, Lei Yang, Qimao Chen, Ziang Luo, Zixun Xie, Shengyin Jiang, Jiaxin Liu, et al. AdaThinkDrive: Adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769, 2025.

- [79] Yuechen Luo, Qimao Chen, Fang Li, Shaoqing Xu, Jaxin Liu, Ziying Song, Zhi-xin Yang, and Fuxi Wen. Unleashing VLA potentials in autonomous driving via explicit learning from failures. arXiv preprint arXiv:2603.01063, 2026.

- [80] Yuechen Luo, Fang Li, Shaoqing Xu, Yang Ji, Zehan Zhang, Bing Wang, Yuannan Shen, Jianwei Cui, Long Chen, Guang Chen, Hangjun Ye, Zhi-Xin Yang, and Fuxi Wen. LaST-VLA: Thinking in latent spatio-temporal space for vision-language-action in autonomous driving. arXiv preprint arXiv:2603.01928, 2026.

- [81] Ziang Luo, Kangan Qian, Jiahua Wang, Yuechen Luo, Jinyu Miao, Zheng Fu, Yunlong Wang, Sicong Jiang, Zilin Huang, Yifei Hu, Yuhao Yang, Hao Ye, Mengmeng Yang, Xiaojian Dong, Kun Jiang, and Diange Yang. MTRDrive: Memory-tool synergistic reasoning for robust autonomous driving in corner cases. arXiv preprint arXiv:2509.20843, 2025.

- [82] Srikanth Malla, Chiho Choi, Isht Dwivedi, Joon Hee Choi, and Jiachen Li. DRAMA: Joint risk localization and captioning in driving. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1043–1052, 2023.

- [83] Jiageng Mao, Minzhe Niu, Chenhan Jiang, Hanxue Liang, Jingheng Chen, Xiaodan Liang, Yamin Li, Chaoqiang Ye, Wei Zhang, Zhenguo Li, et al. One million scenes for autonomous driving: ONCE dataset. arXiv preprint arXiv:2106.11037, 2021.

- [84] Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, Elahe Arani, and Oleg Sinavski. LingoQA: Visual question answering for autonomous driving. In European Conference on Computer Vision, pages 252–269. Springer, 2024.

- [85] Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulo, and Peter Kontschieder. The Mapillary Vistas dataset for semantic understanding of street scenes. In IEEE/CVF International Conference on Computer Vision, pages 4990–4999, 2017.

- [86] Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, et al. DynVLA: Learning world dynamics for action reasoning in autonomous driving. arXiv preprint arXiv:2603.11041, 2026.

- [87] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. CODI: Compressing chain-ofthought into continuous space via self-distillation. In Conference on Empirical Methods in Natural Language Processing, pages 677–693, 2025.

- [88] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization. In IEEE/CVF International Conference on Computer Vision, pages 16037–16046, 2025.

- [89] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. DriveLM: Driving with graph visual question answering. In European Conference on Computer Vision, pages 256–274. Springer, 2024.

- [90] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

- [91] Dijia Su, Hanlin Zhu, Yingchen Xu, Jiantao Jiao, Yuandong Tian, and Qinqing Zheng. Token assorted: Mixing latent and text tokens for improved language model reasoning. arXiv preprint arXiv:2502.03275, 2025.

- [92] Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, An-Lan Wang, Chunhui Lin, Hao Feng, Zhen Zhao, Yanjie Wang, Yuliang Liu, Hao Liu, Xiang Bai, and Can Huang. MTVQA: Benchmarking multilingual text-centric visual question answering. In Annual Meeting of the Association for Computational Linguistics, pages 7748–7763, 2025.

- [93] Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, and Yann LeCun. What drives success in physical planning with joint-embedding predictive world models? arXiv preprint arXiv:2512.24497, 2025.

- [94] Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang, Yihang Qiu, Guang Li, Junli Wang, Yinfeng Gao, Zhang Zhang, Liang Wang, Long Chen, Hongyang Li, et al. SimScale: Learning to drive via real-world simulation at scale. arXiv preprint arXiv:2511.23369, 2025.

- [95] Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In IEEE Information Theory Workshop, pages 1–5. IEEE, 2015.

- [96] Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. In Proc. of the 37-th Annual Allerton Conference on Communication, Control and Computing, pages 368–377, 1999. URL https://arxiv.org/abs/physics/0004057.

- [97] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In Advances in Neural Information Processing Systems, volume 37, pages 87310–87356, 2024.

- [98] Girish Varma, Anbumani Subramanian, Anoop Namboodiri, Manmohan Chandraker, and CV Jawahar. IDD: A dataset for exploring problems of autonomous navigation in unconstrained environments. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1743–1751, 2019.

- [99] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 6000–6010, 2017.

- [100] Han Wang, Yongjie Ye, Bingru Li, Yuxiang Nie, Jinghui Lu, Jingqun Tang, Yanjie Wang, and Can Huang. Vision as LoRA. arXiv preprint arXiv:2503.20680, 2025.

- [101] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5294–5306, 2025.

- [102] Jie Wang, Guang Li, Zhijian Huang, Chenxu Dang, Hangjun Ye, Yahong Han, and Long Chen. VGGDrive: Empowering vision-language models with cross-view geometric grounding for autonomous driving. arXiv preprint arXiv:2602.20794, 2026.

- [103] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. OmniDrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22442–22452, 2025.

- [104] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

- [105] Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, Liang Feng, Greg Heinrich, Jack Huang, Peter Karkus, Boyi Li, Pinyi Li, Tsung-Yi Lin, Dongran Liu, Ming-Yu Liu, Langechuan Liu, Zhijian Liu, Jason Lu, Yunxiang Mao, Pavlo Molchanov, Lindsey Pavao, Zhenghao Peng, Mike Ranzinger, Ed Schmerling, Shida Shen, Yunfei Shi, Sarah Tariq, Ran Tian, Tilman Wekel, Xinshuo Weng, Tianjun Xiao, Eric Yang, Xiaodong Yang, Yurong You, Xiaohui Zeng, Wenyuan Zhang, Boris Ivanovic, and Marco Pavone. Alpamayo-R1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025.

- [106] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. Sim-cot: Supervised implicit chain-of-thought. arXiv preprint arXiv:2509.20317, 2025.

- [107] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen

- Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, et al. Argoverse 2: Next generation datasets for self-driving perception and forecasting. arXiv preprint arXiv:2301.00493, 2023.
- [108] Shaoyuan Xie, Lingdong Kong, Yuhao Dong, Chonghao Sima, Wenwei Zhang, Qi Alfred Chen, Ziwei Liu, and Liang Pan. Are VLMs ready for autonomous driving? An empirical study from the reliability, data, and metric perspectives. In IEEE/CVF International Conference on Computer Vision, pages 6585–6597, 2025.

- [109] Shaoyuan Xie, Lingdong Kong, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Benchmarking and improving bird’s eye view perception robustness in autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(5):3878–3894, 2025.

- [110] Shaoqing Xu, Fang Li, Shengyin Jiang, Ziying Song, Li Liu, and Zhi-xin Yang. GaussianPretrain: A simple unified 3D Gaussian representation for visual pre-training in autonomous driving. arXiv preprint arXiv:2411.12452, 2024.

- [111] Xiang Xu, Ao Liang, Youquan Liu, Linfeng Li, Lingdong Kong, Ziwei Liu, and Qingshan Liu. U4D: Uncertaintyaware 4D world modeling from LiDAR sequences. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

- [112] Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. SoftCoT: Soft chain-of-thought for efficient reasoning with LLMs. In Annual Meeting of the Association for Computational Linguistics, pages 23336–23351, 2025.

- [113] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. DriveGPT4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 9(10):8186–8193, 2024.

- [114] Tianyi Yan, Tao Tang, Xingtai Gui, Yongkang Li, Jiasen Zheng, Weiyao Huang, Lingdong Kong, Wencheng Han, Xia Zhou, Xueyang Zhang, et al. AD-R1: Closed-loop reinforcement learning for end-to-end autonomous driving with impartial world models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

- [115] Xinlei Yu, Zhangquan Chen, Yongbo He, Tianyu Fu, Cheng Yang, Chengming Xu, Yue Ma, Xiaobin Hu, Zhe Cao, Jie Xu, et al. The latent space: Foundation, evolution, mechanism, ability, and outlook. arXiv preprint arXiv:2604.02029, 2026.

- [116] Shuang Zeng, Xinyuan Chang, Mengwei Xie, Xinran Liu, Yifan Bai, Zheng Pan, Mu Xu, Xing Wei, and Ning Guo. FutureSightDrive: Thinking visually with spatio-temporal CoT for autonomous driving. arXiv preprint arXiv:2505.17685, 2025.

- [117] Lingfeng Zhang, Xiaoshuai Hao, Yingbo Tang, Haoxiang Fu, Xinyu Zheng, Pengwei Wang, Zhongyuan Wang, Wenbo Ding, and Shanghang Zhang. nava3: Understanding any instruction, navigating anywhere, finding anything. arXiv preprint arXiv:2508.04598, 2025.

- [118] Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, and Jiwen Lu. DVGT: Driving visual geometry transformer. arXiv preprint arXiv:2512.16919, 2025.

- [119] Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Hanbing Li, Long Chen, Zhi-Xin Yang, and Jiwen Lu. DVGT-2: Vision-geometry-action model for autonomous driving at scale. arXiv preprint arXiv:2604.00813, 2026.

### Appendix

#### A Appendix

###### A.1 Data Format Example

Each NAVSIM training sample in OneVL follows the structure below. The assistant response contains visual latent tokens (4), language latent tokens (2), and the trajectory answer:

|User: <image> Front-view image of the driving scene. Command: MOVE FORWARD. Velocity: [1.5, 0.0]. Acceleration: [0.3, 0.0]. Historical trajectory: (0.0, 0.0, 0.0), (-0.75, 0.0, 0.0), (-1.5, 0.0, 0.0).<br><br>[Task instruction to output reasoning and predicted trajectory. . . ]<br><br>Assistant: <|start-latent-vis|> <|latent-vis|> <|latent-vis|> <|latent-vis|> <|latent-vis|><br><br><|end-latent-vis|> <|start-latent|> <|latent|> <|latent|> <|end-latent|> <answer>[0.75, 0.0, 0.0], [1.5, 0.0, 0.0], ...</answer><br><br>think_steps (label for Dl): “The road ahead is clear. There are no vehicles or pedestrians that would require braking. I should maintain current speed and continue straight along the lane.”<br><br>future_image_tokens (label for Dv): <|image_start|>...<|visual_token_XXXXXX|>...<|image_end|>|
|---|

###### A.2 Training Configuration

We begin with self-supervised pretraining of the visual auxiliary decoder: the model learns to predict the next frame from ViT features alone, without the autoregressive backbone in the loop. This phase uses 13040 optimizer steps and a global batch size of 256, rather than a fixed epoch count. We then move to Stage 0, which warms up the main VLM while introducing latent tokens, followed by Stage 1, where the auxiliary decoders train with the main model frozen and the visual decoder becomes action-conditioned through visual latent tokens. Finally, Stage 2 performs joint end-to-end fine-tuning with all parts of the stack updated together; details are shown in Table 9.

###### A.3 CoT Annotation Construction

The chain-of-thought reasoning annotations for the ROADWork dataset are constructed using an in-house annotation pipeline developed specifically for work-zone driving scenarios. Unlike the NAVSIM annotations (sourced from AdaThinkDrive [78]), ROADWork CoT reasoning requires domain-specific annotation of:

- • Work-zone hazard identification: Detection and description of cones, barriers, temporary signage, and worker presence.
- • Non-standard lane interpretation: Reasoning about temporary lane markings, reduced lane widths, and merge configurations.
- • Speed and clearance reasoning: Justification of appropriate speed reduction and lateral clearance decisions specific to work zones.

The CoT reasoning annotations for the Impromptu VLA Dataset are constructed using a VLM-centric annotation pipeline with Chain-of-Thought prompting developed specifically for unstructured driving scenarios (the core corner cases for autonomous driving). Unlike annotations for conventional structured driving datasets, Impromptu VLA CoT reasoning requires domain-specific annotation of:

- • Unstructured scenario classification: Identification and categorization of four core challenging unstructured scenarios, including roads with unclear boundaries, temporary traffic rule changes, unconventional dynamic obstacles, and challenging road conditions.

- • Complex scene element perception: Detection and description of ambiguous road boundaries, temporary traffic facilities, non-standard dynamic obstacles, and adverse road/environmental conditions.
- • Non-standard driving behavior reasoning: Interpretation of temporary traffic rules, obstacle avoidance strategies, and justification of speed adjustment and lateral clearance decisions in unstructured environments.
- • Planning-oriented decision reasoning: Generation of ego-vehicle meta-action plans, end-to-end trajectory predictions, and textual rationales for driving maneuvers.

Since the CoT reasoning labels for Alpamayo-R1 have not been released, we replicate these annotations ourselves. Specifically, we use the publicly released model checkpoint1 to reproduce the CoT labels for all training examples. Regarding the waypoint prediction, we applied a heuristic subsampling strategy to reduce the sequence from 64 to 8 points while ensuring the last points are retained, as 64 waypoints are too dense for autoregressive modeling (the original Alpamayo-R1 model employs flow matching). Furthermore, as the original paper does not release the official test set, we subsample 700 examples from the available video clips to construct our test set.

The following are four anonymized CoT examples:

NAVSIM Example.

|The right side of the lane where the ego vehicle is located is close to the undrivable area, so I need to drive slightly to the left. I should pay more attention to a vehicle, located 18.42 meters ahead of the ego vehicle and 0.49 meters to the left. It is driving forward in the same direction, and its motion state is moving fast. Based on the understanding of the driving scene and the navigation information, the ego should maintain speed and keep the lane.|
|---|

ROADWork Example.

|critical_factors: An orange work-zone warning sign placed in the roadway ahead indicates construction activity and prompts caution and adjustment in path. A line of cones along the left side of the roadway creates a temporary channelization/taper that narrows the usable lane and guides vehicles away from the left edge. Parked vehicles tightly line the right curb, reducing lateral clearance and constraining the ego vehicle available path through the work area. Scaffolding/sidewalk construction along the right side suggests active work frontage and encourages keeping distance from the curbside area. cause:Taper or channelization speed adjustment and channelized lane keeping because cones taper the left side and parked vehicles along the right curb narrow the usable lane. decision:longitudinal:Taper or channelization speed adjustment lateral:Channelized lane keeping|
|---|

Impromptu Example.

|<decision>longitudinal decision: stop, lateral decision: straight</decision><cause> The decision to stop the car is likely due to the presence of construction barriers and a pedestrian crossing the road, indicating a need to ensure safety and comply with traffic regulations.</cause>|
|---|

Alpamayo-R1 Example.

|Turn right at the intersection since the right-turn traffic light is green.|
|---|

1https://huggingface.co/nvidia/Alpamayo-R1-10B

Table 9 Training hyperparameters for OneVL.

Hyperparameter Pre-training Stage 0 Stage 1 Stage 2

Steps 13 040 – – – Epochs – 2 1 5 Batch (global) 256 64 64 64 Learning rate 1×10−4 4×10−5 1×10−4 1×10−4 LR schedule Cosine Cosine Cosine Cosine Optimizer AdamW AdamW AdamW AdamW Precision BF16 BF16 BF16 BF16 Parallelism Zero-2 Zero-2 Zero-2 Zero-2

Trainable Vis aux decoder ViT, LLM, aligner Lang. & vis aux dec All Frozen – – Main VLM –

λl – – 1.0 1.0 λv 1.0 – 0.1 0.1

###### A.4 LLM-as-Judge Evaluation Prompt

We employ a state-of-the-art proprietary VLM, gemini-3.1-flash-lite-preview2 as an automated evaluator for text CoT quality. For each sample, the judge receives the front-camera image alongside the following prompt (with the ground-truth and predicted CoT filled in):

|You are an expert evaluator for Autonomous Driving Systems. Your task is to evaluate the quality of a predicted driving Chain-of-Thought (CoT) against the Ground Truth CoT. You are also provided with the front-camera image for driving context.<br><br>Ground Truth CoT: “{gt_cot}” Predicted CoT: “{pred_cot}”<br><br>Evaluation Criteria:<br><br>1. Perception (Distance & Location): Minor numerical differences (e.g., 29.3m vs 27.5m) are acceptable and should only incur slight penalties.<br>2. Motion State (Prediction): Identifying the correct state (e.g., ‘moving fastly’, ‘moving slowly’, ‘keep static’) is highly critical. A mismatch here is a severe safety hazard and must heavily reduce the score.<br>3. Ego Decision (Planning): The final action (e.g., ‘accelerate’, ‘decelerate’, ‘maintain speed’, ‘keep lane’) MUST match the Ground Truth. Any deviation is a critical safety error and should result in a massive penalty.<br>4. Language Fluency: Penalize minor grammatical errors. Score the Predicted CoT on a scale of 0 to 100 based on the above criteria.<br><br><br>Output STRICTLY in the following JSON format without any markdown blocks, backticks, or extra text: {"reasoning": "Briefly explain the penalties applied based on the criteria.", "score": <integer_between_0_and_100>}|
|---|

The judge’s raw score (0–100) is normalized to [0, 1] by dividing by 100. We set the temperature to 0.1 to ensure scoring stability.

2https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview

Table 10 Performance comparisons on the Impromptu benchmark [17]. We report trajectory prediction L2 error following the benchmark setting (meters; lower is better).

Traj. Pred. L2 Error (m) Method 1s ↓ 2s ↓ 3s ↓ 4s ↓ Avg. ↓

- ◦ Impromptu VLA [17]* 0.90 2.80 3.75 5.89 3.16
- ◦ Impromptu VLA [17] 0.14 0.60 1.45 2.67 1.22

• OneVL 0.13 0.48 1.18 2.25 1.01

###### A.5 Reproducing Impromptu

To reproduce the optimal performance of the baseline model, we follow the official settings to conduct evaluations using both the training script released by Impromptu and the pre-trained checkpoint uploaded to Hugging Face, with the corresponding test results presented in Table 10. Concretely, the first row marked with an asterisk (*) reports the results evaluated on the author-provided checkpoint3, and the second row shows the outcomes of the model trained from scratch via the official training script. As neither of these results matches the performance claimed in the original paper, we adopt the better-performing one as the representative result of Impromptu for comparative analysis in the main experiments.

###### A.6 NAVSIM qualitative examples

Figure 9 to 17 show additional NAVSIM examples compare a baseline answer-only prediction against OneVL, along with two future frames decoded from the visual auxiliary decoder, the text CoT from the language decoder. The key reasoning that affects the decision is highlighted in bold. Each plot overlays ground-truth (green) and predicted (red) trajectories on the front camera view.

###### A.7 Roadwork qualitative examples

Figure 18 to 19 show additional ROADWork examples compare a baseline answer-only prediction against OneVL, along with two future frames decoded from the visual auxiliary decoder, the text CoT from the language decoder. Each plot overlays ground-truth (green) and predicted (red) trajectories on the front camera view.

###### A.8 Impromptu qualitative examples

Figures 20 and 21 present the cases of the answer-only baseline and the OneVL model on the impromptu task. Green lines in the figures denote the ground-truth trajectories provided by the dataset, and red lines represent the trajectories predicted by different models. As observed in Figure 20, the CoT prompt guides the model to avoid large vehicles merging into the traffic flow, enabling OneVL to generate trajectories that stay away from large vehicles and are closer to the ground truth. Furthermore, Figure 21 shows that when passing through a Y-shaped intersection, the right-turn CoT prompt allows OneVL to output a reasonable right-turn trajectory that remains more centered in the lane. Comprehensive analysis demonstrates that the information provided by CoT prompts can help produce trajectories that are closer to the ground truth and better satisfy driving requirements, especially in scenarios that require changes in current driving behavior and trajectories, such as crossing intersections and interacting with surrounding vehicles.

###### A.9 Alpamayo-R1 qualitative examples Figure 22 presents the cases of the answer-only baseline and the OneVL model on the Alpamayo-R1 dataset.

3https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/7B_AD

[Figure 64]

[Figure 65]

OneVL Front View OneVL BEV

[Figure 66]

[Figure 67]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 68]

[Figure 69]

CoT: I don't need to pay attention to any road boundaries. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should should maintain speed and turn right.

###### Figure 9 NAVSIM qualitative example 1.

[Figure 70]

[Figure 71]

OneVL Front View OneVL BEV

[Figure 72]

[Figure 73]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 74]

[Figure 75]

CoT: I don't need to pay attention to any road boundaries. I should pay more attention to a vehicle, located 22.58 meters ahead of ego vehicle and 0.98 meters to the left. It is not moving, and its motion state is keep static. Based on the understanding of the driving scene and the navigation information, the ego should should decelerate and keep lane.

###### Figure 10 NAVSIM qualitative example 2.

[Figure 76]

[Figure 77]

OneVL Front View OneVL BEV

[Figure 78]

[Figure 79]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 80]

[Figure 81]

CoT: I don't need to pay attention to any road boundaries. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should should maintain speed and change to the left lane.

###### Figure 11 NAVSIM qualitative example 3.

[Figure 82]

[Figure 83]

OneVL Front View OneVL BEV

[Figure 84]

[Figure 85]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 86]

[Figure 87]

CoT: I don't need to pay attention to any road boundaries. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should accelerate and turn right.

###### Figure 12 NAVSIM qualitative example 4.

[Figure 88]

[Figure 89]

OneVL Front View OneVL BEV

[Figure 90]

[Figure 91]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 92]

[Figure 93]

CoT: The right side of the lane where ego vehicle is located is close to the undrivable area, so I need to drive slightly to the left. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should decelerate and keep lane.

###### Figure 13 NAVSIM qualitative example 5.

[Figure 94]

[Figure 95]

OneVL Front View OneVL BEV

[Figure 96]

[Figure 97]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 98]

[Figure 99]

CoT: The right side of the lane where ego vehicle is located is close to the undrivable area, so I need to drive slightly to the left. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should accelerate and change to the right lane.

###### Figure 14 NAVSIM qualitative example 6.

[Figure 100]

[Figure 101]

OneVL Front View OneVL BEV

[Figure 102]

[Figure 103]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 104]

[Figure 105]

CoT: I don't need to pay attention to any road boundaries. I should pay more attention to a vehicle, located 25.81 meters ahead of ego vehicle and 0.51 meters to the right. It is not moving, and its motion state is keep static. Based on the understanding of the driving scene and the navigation information, the ego should decelerate and keep lane.

###### Figure 15 NAVSIM qualitative example 7.

[Figure 106]

[Figure 107]

OneVL Front View OneVL BEV

[Figure 108]

[Figure 109]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 110]

[Figure 111]

CoT: I don't need to pay attention to any road boundaries. I should pay more attention to a vehicle, located 18.41 meters ahead of ego vehicle and 5.98 meters to the left. It is driving left front in the same direction, and its motion state is moving fastly. Based on the understanding of the driving scene and the navigation information, the ego should should maintain speed and change to the left lane.

###### Figure 16 NAVSIM qualitative example 8.

[Figure 112]

[Figure 113]

OneVL Front View OneVL BEV

[Figure 114]

[Figure 115]

OneVL T+1 OneVL T+2 OneVL Reasoning

[Figure 116]

[Figure 117]

CoT: I don't need to pay attention to any road boundaries. There are no objects in the current scene that I need to pay attention to. Based on the understanding of the driving scene and the navigation information, the ego should should maintain speed and turn left.

###### Figure 17 NAVSIM qualitative example 9.

[Figure 118]

[Figure 119]

OneVL T+1 OneVL T+2

[Figure 120]

[Figure 121]

OneVL Reasoning

CoT: <critical_factors>Orange construction barrels/cones along the right edge of the roadway narrow/define the usable lane and encourage the ego vehicle to keep left of them while proceeding forward.. A stop sign at the right-side corner of the upcoming intersection requires the ego vehicle to approach cautiously and prepare to stop/yield at the intersection. Parked vehicles along the right curb near the intersection reduce lateral clearance and constrain the approach path, reinforcing a centered/leftward trajectory.</critical_factors> <cause>Stop at the control point while turning right because the stop sign at the upcoming intersection requires a controlled approach and full stop.</cause> <decision>longitudinal:Stop at control point lateral:Turn right</decision>

###### Figure 18 ROADWork qualitative example 1.

[Figure 122]

[Figure 123]

OneVL T+1 OneVL T+2

[Figure 124]

[Figure 125]

OneVL Reasoning

CoT: <critical_factors>A line of temporary green construction barriers and cones on the right edge narrows/defines the usable roadway and keeps the ego vehicle left of the work area.. Active roadside work area on the right with exposed dirt/uneven shoulder and construction materials behind the barriers, encouraging the ego vehicle to stay away from the edge. Parked vehicles along the left curb reduce lateral clearance, influencing the ego vehicle to hold a centered path within the remaining open lane. A lead vehicle ahead in the same direction provides traffic flow context and encourages maintaining lane position and spacing rather than changing lanes.</critical_factors> <cause>Channelized lane keeping with unconstrained speed tracking because temporary construction barriers and cones on the right narrow the usable roadway and guide the ego vehicle to stay left of the work area.</cause> <decision>longitudinal:Unconstrained speed tracking lateral:Channelized lane keeping</decision>

###### Figure 19 ROADWork qualitative example 2.

Baseline Front View OneVL Front View

[Figure 126]

[Figure 127]

[Figure 128]

OneVL T+1 OneVL T+2

[Figure 129]

[Figure 130]

/e2e-data/evad-techvla/guanjiayi/dataset/nuscenes/ proj_results/nuScences_678_81

98.CAM_FRONT.png_proj.png

OneVL Reasoning

CoT： <decision>longitudinal decision: keep speed, lateral decision: straight</decision>\n<cause> The decision to

maintain current speed and go straight is likely due to the presence of a large bus ahead, which may obstruct the left

turn path, and the need to ensure safe navigation around the bus. Additionally, the road markings and surrounding environment suggest a complex intersection ahead, requiring cautious straight-line driving to avoid potential hazards.</cause>

Figure 20 Impromptu qualitative example 1.

Baseline Front View OneVL Front View

[Figure 131]

[Figure 132]

[Figure 133]

OneVL T+1 OneVL T+2

[Figure 134]

[Figure 135]

/e2e-data/evad-techvla/guanjiayi/dataset/nuscenes/ proj_results/nuScences_927_10

990.CAM_FRONT.png_proj.png

OneVL Reasoning

CoT： <decision>longitudinal decision: keep speed, lateral decision: turn right</decision>\n<cause> The decision to turn right instead of left is likely due to the presence of an obstacle or a sharp curve ahead, which is not clearly visible in the image but could be inferred from the driving decision. Maintaining current speed ensures a smooth and safe maneuver around the potential hazard.</cause>

Figure 21 Impromptu qualitative example 2.

Baseline Front View OneVL Front View

[Figure 136]

[Figure 137]

OneVL T+1 OneVL T+2

[Figure 138]

[Figure 139]

OneVL Reasoning

CoT: Slow down due to the right turn.

###### Figure 22 Alpamayo-R1 qualitative example.

