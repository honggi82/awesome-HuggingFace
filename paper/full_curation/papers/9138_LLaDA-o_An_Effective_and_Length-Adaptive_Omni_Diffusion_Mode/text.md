## LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model

Zebin You123† Xiaolu Zhang4 JUN ZHOU4 Chongxuan Li123¶ Ji-Rong Wen123

# arXiv:2603.01068v1[cs.CV]1Mar2026

### Abstract

We present LLaDA-o, an effective and lengthadaptive omni diffusion model for multimodal understanding and generation. LLaDA-o is built on a Mixture of Diffusion (MoD) framework that decouples discrete masked diffusion for text understanding and continuous diffusion for visual generation, while coupling them through a shared, simple, and efficient attention backbone that reduces redundant computation for fixed conditions. Building on MoD, we further introduce a data-centric length adaptation strategy that enables flexible-length decoding in multimodal settings without architectural changes. Extensive experiments show that LLaDA-o achieves state-ofthe-art performance among omni-diffusion models on multimodal understanding and generation benchmarks, and reaches 87.04 on DPG-Bench for text-to-image generation, supporting the effectiveness of unified omni diffusion modeling. Code is available at https://github.com/ ML-GSAI/LLaDA-o.

### 1. Introduction

Masked diffusion models (MDMs) (Austin et al., 2021; Lou

- et al., 2024; Shi et al., 2024; Sahoo et al., 2024; Ou et al.,

2024) have recently emerged as a promising alternative to autoregressive (AR) language models. By iteratively denoising masked tokens in parallel, MDMs have demonstrated strong language modeling performance (Nie et al., 2025; Ye

- et al., 2025; Cheng et al., 2025; Bie et al., 2025) and offer appealing properties such as bidirectional context modeling and improved inference parallelism (Song et al., 2025; Khanna et al., 2025; Google, 2025), drawing increasing

†Work done during an internship at Ant Group 1Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China. 2Beijing Key Laboratory of Research on Large Models and Intelligent Governance. 3Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE. 4Ant Group. Correspondence to: Chongxuan Li <chongxuanli@ruc.edu.cn>.

Preprint. March 3, 2026.

attention to diffusion-based language modeling.

Despite this progress, the potential of diffusion language models as omni models for multimodal understanding and generation remains largely underexplored. A key challenge is that text and images favor fundamentally different diffusion dynamics: masked diffusion naturally operates over discrete language tokens, whereas for images, continuous diffusion in latent space has become the de facto standard (SohlDickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). Although attempts exists (Pynadath et al., 2025; Chang et al., 2022), these modality-specific preferences do not transfer trivially across domains, motivating a hybrid design that models text discretely and images continuously.

However, naively co-training both modalities within a single dense diffusion model is often ineffective. The heterogeneous state spaces and corruption processes can induce objective mismatch and gradient interference, leading to unstable optimization and suboptimal performance (Li et al., 2025c). In addition, existing omni-diffusion models (Xin et al., 2025; Li et al., 2025a; Yang et al., 2025a) often assume a fixed length for understanding, constraining their applicability open-ended settings.

To address these challenges, we propose LLaDA-o, an effective and length-adaptive omni diffusion model built upon the Mixture of Diffusion (MoD) framework. MoD decouples modality-appropriate diffusion processes into specialized experts while maintaining a shared attention backbone for cross-modality interaction. Specifically, it assigns masked diffusion to an understanding expert for text and visual encoder tokens, and continuous diffusion to a generation expert for visual latent tokens, mitigating optimization conflicts in dense co-training. Building on MoD, we further introduce an efficient attention mechanism and a data-centric length adaptation strategy, enabling scalable inference and flexible-length generation in multimodal settings.

We evaluate LLaDA-o from both qualitative and quantitative perspectives. Qualitatively, Fig. 1 shows that LLaDA-o achieves stronger multimodal understanding and produces images with richer fine-grained details than existing omnidiffusion models such as Lumina-DiMOO (Xin et al., 2025). Quantitatively, we evaluate LLaDA-o on ten multimodal understanding benchmarks (see Tab. 1), where it achieves state-of-the-art results among omni-diffusion models and

Multimodal Understanding

[Figure 1]

User: Describe this image in brief.

LLaDA-o: A red and white train is traveling along a curved railway track next to a calm river. The surrounding area is lush with trees displaying autumn foliage, featuring shades of yellow, orange, and green. The scene is serene, with the natural landscape framing the train.

Text-To-Image Generation

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Case by Case Comparison

[Figure 12]

User: What time is shown? Answer by typing a time word, not a number. It is (_) past six.

quarter 6:00 11:00 12:00

half 6:00

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

A astronaut's helmet on a museum pedestal, reflecting not the room but a vast, starry nebula in its visor, hyperrealistic, polished surface.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

A giant panda wearing a space suit, surfing on a galaxy wave.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Isometric view of a floating island with a magical library,

waterfalls cascading into the void, clouds swirling around, low poly 3d render, vibrant

colors, soft shadows.

LLaDA-o (ours) Lumina-DiMOO LaViDa-O

MMaDA

Figure 1. Overview of LLaDA-o’s capabilities. Top: multimodal understanding examples. Middle: text-to-image generation results following complex prompts (see Table 7 for the prompts). Bottom: case by case comparison with existing omni diffusion models, where LLaDA-o achieves stronger understanding performance and generates images with richer fine-grained details following the instructions.

other discrete-diffusion-based multimodal approaches. We further evaluate text-to-image generation on the widely used GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024) benchmarks against strong generation-only and unified multimodal models, where LLaDA-o achieves state-ofthe-art performance on DPG-Bench (87.04). Besides, we present a comprehensive analysis for LLaDA-o. Overall, these results demonstrate the effectiveness of LLaDA-o as a unified omni diffusion model.

- 2. Preliminaries We present preliminaries on diffusion models.

#### 2.1. Continuous Diffusion Models

Continuous diffusion models (CDMs) (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b) constitute a core paradigm in modern image generation, particularly when combined with diffusion Transformers (Peebles & Xie, 2023; Bao et al., 2023). Conceptually, a CDM specifies a forward stochastic process that gradually corrupts data into noise, and learns to reverse this process to generate samples from the noise distribution. This mechanism can be characterized using stochastic differential equations (Song et al., 2020b). At inference, the same dynamics admit an equivalent ordinary differential equation (ODE) formulation, leading to deterministic samplers (Song et al., 2020a; Lu et al., 2022). This ODE perspective has further inspired a series of variants, including rectified flow (RF) (Liu, 2022) and Flow Matching (Lipman et al., 2022).

In particular, RF (Liu, 2022) connects the noise distribution π0 and data distribution π1 via a deterministic linear path. Given a data-noise pair (x0,x1), the intermediate state at time t ∈ [0,1] is defined as strict linear interpolation:

xt = (1 − t)x0 + tx1. (1)

RF learns a velocity field pθ(x,t) to match the constant flow direction x1 − x0 by minimizing the following objective:

0,x1),t (x1 − x0) − pθ(xt,t) 22 . (2)

L(θ) = E(x

Sampling corresponds to solving the ODE dzt/dt = pθ(zt,t) initialized from z0 ∼ π0. Formally, RF discretizes the following integral of the learned velocity field over the time interval (via the Euler method, for instance):

z1 = z0 +

1

pθ(zt,t)dt. (3)

0

#### 2.2. Diffusion Language Models

Discrete diffusion models (Sohl-Dickstein et al., 2015; Austin et al., 2021) are generative models tailored to discrete data, such as token sequences. Analogous to CDMs,

they define a forward Markov process of discrete state that progressively corrupts the data toward a simple prior, typically a uniform distribution over tokens or an absorbing mask state. The model then learns the corresponding reverse dynamics to generate discrete data from this prior. A special case of discrete diffusion, masked diffusion models (MDMs) (Lou et al., 2024; Shi et al., 2024; Sahoo et al., 2024; Ou et al., 2024), have demonstrated strong potential and favorable scaling properties in language modeling.

Formally, let x0 = [xi]Ni=1 denote a sequence of N tokens, and let [M] represent a special mask token. In the forward process, tokens are independently corrupted based on a time step t uniformly sampled from [0,1]. Let αt be a continuous monotonically increasing function and α0 = 0 and α1 = 1. Each token either remains unchanged with probability αt or is replaced by [M] with probability 1 − αt as follows:

N−1

qt|0(xit|xi0),

qt|0(xt|x0) =

i=0

(4)

αt, xit = xi0, 1 − αt, xit = [M].

qt|0(xit|xi0) =

The reverse process reconstructs the sequence iteratively. In particular, consider the transformation from time t to s (where 0 ≤ s < t ≤ 1). Tokens that are already predicted remain fixed. Conversely, masked tokens either stay masked with probability 1−αs

1−αt or are decoded with probability αs−αt

1−αt based on the model prediction pθ(xi0|xt):

N−1

qs|t(xis|xt),

qs|t(xs|xt) =

i=0

 

1, xit ̸= [M],xis = xit,

- 1−αs

- 1−αt , xit = [M],xis = [M], αs−αt

qs|t(xis|xt) =

1−αt pθ(xi0|xt), xit = [M],xis ̸= [M], 0, otherwise.



(5) During inference, MDMs generate text by iteratively simulating this reverse transition, gradually converting a fully masked sequence into coherent text.

Intuitively, MDMs function as mask predictors, aiming to recover masked tokens from the observed context. Formally, the training objective of MDMs is defined as:

L(θ)=

1

- 0
- 1

Ex

0,xt∼qt|0

t

−log pθ(xi0|xt) dt. (6)

i:xit=[M]

Diffusion large language models (dLLMs) (Nie et al., 2025; Ye et al., 2025) demonstrate the viability of masked diffusion at scale, achieving competitive performance while enabling parallel decoding and flexible generation control.

Text De-Tokenizer Gen. Decoder

Mask Prediction Velocity Prediction

Und. MLP Gen. MLP

Intra-Modality Bidirectional Attention

x N

Und image token

| |
|---|

Noise image token

Und. QKV Gen. QKV

Clean image token

| |
|---|

Mask token

Und. Encoder Text Tokenizer Gen. Encoder

| |
|---|

Text token

Figure 2. Overview of LLaDA-o: the mixture of diffusion framework.

### 3. Method

and r0 denotes the ground-truth response, the entire expert is trained to optimize the variant of Eq. (6) as follows:

Motivated by the discussion in Sec. 1, LLaDA-o unifies multimodal understanding and generation within a single diffusion framework by separating discrete and continuous modalities, coupling them through a shared and efficient attention mechanism, and enabling adaptive-length training and inference, detailed as follows.

1

- 0
- 1

−log pθ(ri0|v,p,rt) dt.

E v,p

Lund =

t

r0,rt

i:rit=[M]

(7)

In parallel, the generation expert comprises a variational autoencoder (VAE) (Kingma & Welling, 2013) and a diffusion Transformer (Peebles & Xie, 2023). The VAE maps between images and visual latent tokens and its parameters are kept frozen during training. Given the relevant variables (p,v0,ϵ), where p denotes the prompt, v0 denotes the ground-truth image tokens from the VAE, and ϵ denotes a Gaussian noise, the training objective follows Eq. (2):

#### 3.1. The Mixture of Diffusion Framework

At the core of LLaDA-o is a hybrid diffusion design that treats discrete tokens and continuous visual latents with their respective optimal parameterizations. However, naively cotraining these modalities in a single dense model is often ineffective (Li et al., 2025c): the two branches operate on heterogeneous state spaces and corruption processes, which can induce objective mismatch and gradient interference, leading to training conflicts and suboptimal performance.

0,vt, t (v0 − ϵ) − pθ(p,vt,t) 22 . (8)

Lgen = Ep,ϵ,v

It is worth noting that both the input image and text in the generation task are also processed by the understanding expert. As a result, the corresponding parameters are jointly trained. Detailed training protocols for multi-turn dialogue and interleaved data are provided in Appendix B.2.

To address these challenges, we propose Mixture of Diffusion (MoD), a unified multi-modal diffusion framework illustrated in Fig. 2. Inspired by modality-factorized designs (Liang et al., 2024; Deng et al., 2025), MoD employs two diffusion experts to decouple the processing of discrete and continuous modalities: an understanding expert that handles text and visual encoder tokens via masked diffusion, and a generation expert that handles visual latent tokens via continuous diffusion. While the experts allow for specialized processing, they share the same self-attention backbone (see Sec. 3.2) to ensure effective cross-modality interaction.

#### 3.2. Intra-Modality Bidirectional Attention

Although MoD decouples modality-specific processing via different experts, they share a common attention backbone for cross-modality interaction. Using global attention in this setting is inefficient because it recomputes attention over the entire sequence at every denoising step, even when the condition (e.g., an input image or a text prompt) stays fixed.

Specifically, the understanding expert integrates a vision encoder (Zhai et al., 2023), a lightweight two-layer MLP, and a diffusion language model (Nie et al., 2025) sequentially. The image is encoded into semantic visual tokens by the encoder, projected into the language token space by the MLP, and jointly processed with prompt tokens by the language model. Given a training sample (v,p,r0), where v denotes the projected image tokens, p denotes the prompt,

To this end, we propose intra-modality bidirectional attention, a simple yet efficient attention scheme tailored for MoD. In particular, we partition an input sequence into modality blocks, apply full attention within each block, and enforce causal attention across blocks. This design preserves rich intra-modality context while enabling efficient inference: conditional blocks form a fixed prefix whose

IMG 1 IMG 2 PRM 1 RES 1

PRM 2 RES 2

- IMG 1

- IMG 2

- PRM 1

- RES 1

PRM 2

- RES 2

(a) Multimodal Understanding

PRM 1 IMG 1 IMG 2

- IMG 1

- IMG 2

PRM 2

- PRM 1

- PRM 2

IMG 3 IMG 4

- IMG 3

- IMG 4

(b) Multimodal Generation

IMG 1 IMG 2 Text 1

- IMG 1

- IMG 2

Text 2

- Text 1

- Text 2

IMG 3 IMG 4

- IMG 3

- IMG 4

Text 3 Text 4

- Text 3

- Text 4

(c) Interleaved Data

Text 1 Text 2

- Text 1

- Text 2

Text 3 Text 4

- Text 3

- Text 4

IMG 1 IMG 2

- IMG 1

- IMG 2

(d) Cross-Sample Isolation

- Figure 3. Implementation of intra-modality bidirectional attention. Yellow blocks indicate unmasked attention, while dashed white boxes denote masked attention. Text sequences are explicitly partitioned into Prompts (PRM) and Responses (RES) in cases (a–b).

Key-Value (KV) cache can be reused across denoising steps, avoiding redundant computation on long sequences.

The proposed attention scheme can be implemented by introducing appropriate attention masks between modality blocks. As illustrated in Fig. 3(a–c), this principle naturally accommodates different data types, including multimodal understanding, generation, and interleaved data. Besides, we use sample packing with strict isolation between samples (see Fig. 3d) to prevent cross-sample interference.

Compared to a representative baseline (You et al., 2025a) with globally bidirectional attention, our attention design achieves a 5.9 times speedup in practice (see Fig. 4).

- 3.3. Adaptive Length Augmentation

The understanding expert adopts masked diffusion for text modeling, typically assuming a fixed target length at inference time. To enable flexible generation without introducing architectural changes or violating sample isolation, we design a data-centric strategy called adaptive length augmentation that is fully compatible with our attention scheme.

- As summarized in Alg. 1, during training, the target response in each individual sample is stochastically perturbed in two

complementary ways. With probability pext, a bounded number of [EOS] tokens are appended to the original response, exposing the model to explicit termination at different positions. With probability ptrunc, the response is truncated to a random prefix, encouraging the model to learn proper continuation from partial targets. Importantly, both operations preserve strict sample isolation and do not require modifying the attention structure or sequence packing strategy.

- At inference (see Alg. 2 in Appendix. B.5), we perform block-wise generation (Arriola et al., 2025) under intramodality bidirectional attention: the fixed conditioning blocks (images and prompt) are cached once, and response tokens are generated iteratively by appending a length-L masked block and denoising it. If [EOS] appears with high confidence, decoding terminates; otherwise, the completed

block is cached and generation proceeds to the next block. This strategy enables efficient variable-length generation while fully reusing the KV cache of the fixed prefix.

Compared to prior approaches that rely on architectural modifications or multi-sample packing with globally bidirectional attention to handle variable-length outputs (Wu et al., 2025d; Kim et al., 2025; Yang et al., 2025b), our method remains lightweight, architecture-agnostic, and naturally compatible with sample isolation and efficient inference.

Algorithm 1 Training with Adaptive Length

- 1: Input: Model parameters θ; training data (v,p,r0); extension probability pext; truncation probability ptrunc.
- 2: repeat
- 3: r˜0 ← r0, u ∼ U(0,1)
- 4: if u < pext then
- 5: Sample an integer k uniformly between (1,|r0|) and update r˜0 by appending k EOS tokens.
- 6: else if u < pext + ptrunc and |r0| > 16 then
- 7: Sample an integer ℓ uniformly between (1,|r0|−1) and update r˜0 by truncating it to the first ℓ tokens.
- 8: end if
- 9: Train on (v,p,r˜0) with loss only on response tokens r˜0; update θ
- 10: until convergence

### 4. Experiments

We present experimental settings, results and analyses.

#### 4.1. Experimental Settings

Model. For the understanding expert, we use the representative LLaDA-8B-Instruct (Nie et al., 2025) to initialize the language model. As the vision encoder, we use SigLIP (Zhai et al., 2023), which has shown strong performance in many MLLMs, and adopt a randomly initialized two-layer MLP as the projector. For the generation expert, we use the same diffusion transformer architecture as the masked predictor ar-

- Table 1. Evaluation on multimodal understanding benchmarks. The symbol † denotes results from LaViDa-O (Li et al., 2025a), while ⋆ indicates results we evaluated using official checkpoints and inference scripts. “-” represents missing data, and “Diff.” refers to diffusion language models. Notably, LLaDA-o achieves state-of-the-art performance among diffusion-based unified multimodal models.

Model MMMU MME SeedB MMB MathVerse MathVista AI2D ChartQA DocVQA InfoVQA

val cog. perp. image en-dev mini-vis. testmini - - val val AR Based

Emu3 (Wang et al., 2024) 31.6 - - 68.2 58.5 - - 70.0 68.6 76.3 43.8 Janus-Pro (Chen et al., 2025c) 41.0 - 1567 72.1 79.2 - - - - - MetaMorph (Tong et al., 2025) 41.8 - - 71.8 75.2 - - - 37.1 - Show-o (Xie et al., 2024) 27.4 - 1232 - - - - - - - Show-o2 (Xie et al., 2025) 48.9 - 1620 - 79.3 - - 78.6 - - BAGEL (Deng et al., 2025) 55.3 - 1687 - 85.0 - 73.1 - - - -

Diff. Based

LaViDa-L (Li et al., 2025b) 43.3 341 1365 - 70.5 27.2 44.8 70.0 64.6 59.0 34.2 Dimple (Yu et al., 2025) 45.2 432 1514 - 74.6 - 42.3 74.4 63.4 - LLaDA-V (You et al., 2025a) 48.6 491 1507 74.8 82.9 28.5 59.7 77.8 78.3 83.9 66.3 MMaDA (Yang et al., 2025a) 30.2 242† 1410 64.2 68.5 13.5† 33.7† 66.6† 9.8† 10.9† 14.9† Lumina-DiMOO (Xin et al., 2025) 58.6 - 1534 83.1 84.5 10.3⋆ 30.3⋆ 43.2⋆ 8.3⋆ 7.2⋆ 6.2⋆ LaViDa-O (Li et al., 2025a) 45.1 488 1431 - 76.4 36.9 56.9 76.7 80.0 73.7 44.6 LLaDA-o 44.9 549 1412 75.3 71.1 37.1 66.1 79.3 87.9 91.5 54.7

chitecture in LLaDA (Nie et al., 2025) and is initialized from it, while additional conditional parameters for time embeddings are randomly initialized. We use VAE of FLUX (Labs, 2024) as the vision encoder for generation due to its strong reconstruction quality.

Training strategy. We train LLaDA-o in three stages to progressively scale both data difficulty and generation fidelity. In Stage 1, we use large-scale image understanding data together with image generation data, where generation is restricted to resolutions up to 512 to stabilize training. In Stage 2, we incorporate multimodal reasoning data and reuse a high-quality subset of the Stage 1 image generation data, while increasing the generation resolution to 1024 to improve high-resolution synthesis. Notably, we do not apply adaptive length augmentation for multimodal understanding in the first two stages. In Stage 3, we jointly apply adaptive length augmentation to activate variable-length generation for the understanding expert and add more high-quality image generation data, aligning the model with flexible text decoding and stronger visual generation. For more details on the training strategy and data, please refer to Appendix B.1.

Evaluation. We evaluate LLaDA-o on a broad set of benchmarks to reflect the main requirements of unified multimodal models: general knowledge understanding, reasoning, and fine-grained perception, as well as image generation. For multimodal understanding, we cover multidisciplinary knowledge (MMMU (Yue et al., 2024), MME (Fu et al., 2023), SEED-Bench (Li et al., 2023), and MMBench (Liu et al., 2024)), mathematical reasoning (MathVerse (Zhang et al., 2024) and MathVista (Lu et al., 2023)), and chart/document understanding (AI2D (Kembhavi et al., 2016), ChartQA (Masry et al., 2022), DocVQA (Mathew

et al., 2021), and InfoVQA (Mathew et al., 2022)). For text-to-image generation, we use two widely used benchmarks that test complementary aspects of generation: GenEval (Ghosh et al., 2023), which verifies fine-grained compositional attributes via an object-centric detection pipeline (e.g., object count, spatial relations, and color binding), and DPG-Bench (Hu et al., 2024), which evaluates faithful rendering of long, information-dense prompts with complex entity relationships and rich descriptions.

#### 4.2. Benchmark Results

Multimodal understanding. We compare LLaDA-o with unified multimodal models and multimodal large language models in Tab. 1. Notably, LLaDA-o achieves state-ofthe-art performance among omni-diffusion models (e.g., LaViDa-O (Li et al., 2025a)), demonstrating the effectiveness of the Mixture of Diffusion. This advantage is particularly evident on mathematical reasoning (e.g., MathVista (Lu et al., 2023)) and chart/document understanding (e.g., ChartQA (Masry et al., 2022)). These improvements support the effectiveness of our MoD framework. Compared with state-of-the-art autoregressive model BAGEL (Deng et al., 2025), LLaDA-o is generally weaker, which is expected given BAGEL’s stronger language backbone. BAGEL uses Qwen2.5-7B-Instruct (Team, 2024b) trained on 18T tokens, while our LLaDA-8B-Instruct (Nie et al., 2025) is trained on 2.3T tokens, and this gap is reflected in language capability (e.g., 84.8 vs. 49.4 on HumanEval). Despite this disadvantage, LLaDA-o narrows the gap; for example, on MathVista it achieves 66.1, improving over LLaDA-V (You et al., 2025a) (59.7) and approaching BAGEL (73.1). We believe MoD will further improve as masked diffusion backbones continue to improve.

- Table 2. Evaluation of text-to-image generation on the GenEval benchmark (Ghosh et al., 2023). “Gen.,” “Obj.,” and “Attr.” denote generation, object, and attribute, respectively, while “-” indicates missing data. Compared to state-of-the-art unified multimodal models, LLaDA-o demonstrates superior performance, particularly in two-object and color-specific generation tasks. Following the protocols of BAGEL (Deng et al., 2025) and Show-o2 (Xie et al., 2025), we evaluate the results using rewritten prompts.

Model # Params Single Obj. Two Obj. Counting Colors Position Color Attri. Overall↑ Gen. Only

PixArt-α (Chen et al., 2023) 0.6B 0.98 0.50 0.44 0.80 0.08 0.07 0.48 DALL-E 3 (Betker et al., 2023) - 0.96 0.87 0.47 0.83 0.43 0.45 0.67

SD3-Medium (Esser et al., 2024) 2B 0.99 0.94 0.72 0.89 0.33 0.60 0.74 FLUX.1-dev (Labs, 2024) 12B 0.98 0.81 0.74 0.79 0.22 0.45 0.66

Unified Emu3 (Wang et al., 2024) 8B - - - - - - 0.66

Janus (Wu et al., 2025a) 1.3B 0.97 0.68 0.30 0.84 0.46 0.42 0.61 Janus-Pro (Chen et al., 2025c) 7B 0.99 0.89 0.59 0.90 0.79 0.66 0.80

Mogao (Liao et al., 2025) 7B 1.00 0.97 0.83 0.93 0.84 0.80 0.89 MMaDA (Yang et al., 2025a) 8B 0.99 0.76 0.61 0.84 0.20 0.37 0.63

Show-o (Xie et al., 2024) 1.3B 0.98 0.80 0.66 0.84 0.31 0.50 0.68 BAGEL (Deng et al., 2025) 7B MoT 0.98 0.95 0.84 0.95 0.78 0.77 0.88 LaViDa-O (Li et al., 2025a) - - - - - - - 0.89

Lumina-DiMOO (Xin et al., 2025) 8B 1.00 0.96 0.87 0.95 0.85 0.82 0.91 Show-o2 (Xie et al., 2025) 7B 1.00 0.87 0.58 0.92 0.52 0.62 0.76 LLaDA-o (ours) 8B MoT 0.99 0.98 0.73 0.96 0.69 0.83 0.86

- Table 3. Evaluation of text-to-image generation on DPG-Bench (Hu et al., 2024). The symbol † denotes results from LuminaDiMOO (Xin et al., 2025), while “Gen.” stands for generation and “-” indicates missing data. Notably, LLaDA-o achieves state-of-the-art performance compared to previous generation-only and unified models.

Model # Params Global Entity Attribute Relation Other Overall↑ Gen. Only

PixArt-α (Chen et al., 2023) 0.6B 74.97 79.32 78.60 82.57 76.96 71.11 PixArt-Σ (Chen et al., 2024) 0.6B 86.89 82.89 88.94 86.59 87.68 80.54

DALL-E 3 (Betker et al., 2023) - 90.97 89.61 88.39 90.58 89.83 83.50 SD3-Medium (Esser et al., 2024) 2B 87.90 91.01 88.83 80.70 88.68 84.08

FLUX.1-dev (Labs, 2024) 12B 74.35 90.00 88.96 90.87 88.33 83.84

Unified Emu3-DPO (Wang et al., 2024) 8B - - - - - 81.60

Janus (Wu et al., 2025a) 1.3B 82.33 87.38 87.70 85.46 86.41 79.68 Janus-Pro (Chen et al., 2025c) 7B 86.90 88.90 89.40 89.32 89.48 84.19

Mogao (Liao et al., 2025) 7B 82.37 90.03 88.26 93.18 85.40 84.33 MMaDA† (Yang et al., 2025a) 8B 77.81 78.48 81.74 84.79 63.20 69.97

Show-o (Xie et al., 2024) 1.3B - - - - - 67.48 BAGEL (Deng et al., 2025) 7B MoT 88.94 90.37 91.29 90.82 88.67 85.07 LaViDa-O (Li et al., 2025a) - - - - - - 83.20

Lumina-DiMOO (Xin et al., 2025) 8B 81.46 92.08 88.98 94.31 82.00 86.04 Show-o2 (Xie et al., 2025) 7B 89.00 91.78 89.96 91.81 91.64 86.14 LLaDA-o (ours) 8B MoT 92.91 93.30 90.40 91.75 92.79 87.04

Text-to-image generation. We evaluate LLaDA-o against state-of-the-art generation-only models and unified multimodal models on GenEval (Ghosh et al., 2023) and DPGBench (Hu et al., 2024) (Tabs. 2 and 3). On GenEval, LLaDA-o outperforms most strong models such as Janus-

Pro (Chen et al., 2025c) and SD3-Medium (Esser et al., 2024). Although it is slightly behind Lumina-DiMOO (Xin et al., 2025) and Mogao (Liao et al., 2025) overall, it performs better on two-object generation and color binding. Notably, on DPG-Bench, LLaDA-o achieves state-of-the-art

- Table 4. Effect of confidence threshold on MathVista. We report the accuracy (%) and throughput (tokens/s) across varying confidence thresholds, with the block length fixed at 64.

Confidence Threshold (r) 0.2 0.4 0.6 0.8 0.9 1.0

Accuracy (%) 57.8 62.4 64.2 65.0 65.9 65.8 Throughput (tokens/s) 203.9 134.5 97.3 64.1 52.2 24.3

69

LLaDA-V

LLaDA-o (Ours)

66

MathVistaAccuracy(%)

63

60

5.9× Speedup

57

0 25 50 75 100 125 150 175 200

Throughput (Tokens/sec)

- Figure 4. Comparison of inference efficiency on MathVista. We visualize the throughput-accuracy trade-off by varying the confidence threshold for LLaDA-o and the refresh interval (n) of FastdLLM applied to LLaDA-V. Our approach outperforms LLaDA-V, achieving a 5.9× speedup with comparable performance.

performance (87.04), surpassing Show-o2 (Xie et al., 2025) and Lumina-DiMOO, indicating strong generation quality for long, information-dense prompts. These results support the effectiveness of our MoD framework in combining continuous diffusion model with a dLLM-based backbone for unified multimodal generation. Qualitatively, Fig. 1 shows that LLaDA-o produces more visually appealing images with richer fine-grained details than Lumina-DiMOO and LaViDa-O (Li et al., 2025a) while following the instructions.

We also provide additional qualitative text-to-image samples in Appendix. C.1.

- 4.3. Further Analysis

Tab. 4 and Fig. 4 analyze the inference efficiency and generation quality. Unlike autoregressive models, LLaDA-o offers the flexibility to regulate this trade-off via the confidence threshold: raising the threshold prioritizes accuracy by selecting only high-confidence tokens, while lowering it accelerates generation. Empirically, a threshold of 0.9 achieves the optimal balance. When comparing with state-of-the-art baselines on MathVista (Fig. 4), LLaDA-o demonstrates significant efficiency gains. Most notably, LLaDA-o delivers a

- 5.9× speedup compared to LLaDA-V. This substantial improvement validates the effectiveness of our intra-modality bidirectional attention, which reduces computational redun-

- Table 5. Effect of block length on MathVista. We report the average number of generated tokens under different block lengths, with the confidence threshold fixed at 0.95.

Block length 32 64 96 128 Average tokens 165 148 145 154 Accuracy (%) 63.6 66.1 66.2 65.3

- Table 6. Text-to-image generation performance across training stages. We report the evaluation results on GenEval and DPGBench to demonstrate the performance progression. In this table, we use the original prompts of GenEval.

###### Training Stage GenEval DPG-Bench

- Stage 1 0.74 86.1

- Stage 2 0.78 86.2

- Stage 3 0.82 87.0

dancy and enables efficient inference.

We provide comprehensive analysis of the variable-length behavior in Appendix C.2,C.3 and Tab. 5. Qualitatively, unlike LLaDA-V, which produces redundant or incomplete text depending on mismatched block settings, LLaDA-o generates content of appropriate length consistent across varying block sizes (L ∈ {16,...,128}). Quantitatively, Tab. 5 reveals that the generated length remains relatively stable: increasing the block length from 32 to 96 results in only a minor decrease in average token count (165 to 145) while improving accuracy (63.6% to 66.2%). These results demonstrate that the output length is mainly driven by the input content rather than the preset block size, confirming the effectiveness of our adaptive length augmentation.

Finally, Tab. 6 studies the effect of each training stage on text-to-image generation. Results on GenEval and DPGBench improve from Stage 1 to Stage 3, with Stage 3 performing best (0.82 and 87.0), supporting the effectiveness of our multi-stage training pipeline. For completeness, we report the computational cost of our three-stage training pipeline in Appendix B.4.

### 5. Conclusion

We presented LLaDA-o, a length-adaptive omni diffusion model for multimodal understanding and generation. Built on a Mixture of Diffusion framework with a shared efficient attention backbone and a data-centric adaptive length training strategy, LLaDA-o enables stable multimodal training and flexible-length generation. Experimental results demonstrate strong performance on multimodal understanding and text-to-image generation tasks. We believe that as masked diffusion models continue to advance in language modeling, LLaDA-o potentially provides a promising foundation for future omni diffusion approaches.

### References

An, X., Xie, Y., Yang, K., Zhang, W., Zhao, X., Cheng, Z., Wang, Y., Xu, S., Chen, C., Zhu, D., et al. Llavaonevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

Arriola, M., Gokaslan, A., Chiu, J. T., Yang, Z., Qi, Z., Han, J., Sahoo, S. S., and Kuleshov, V. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and Van Den Berg, R. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., and Zhu, J. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22669–22679, 2023.

Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3): 8, 2023.

Bie, T., Cao, M., Chen, K., Du, L., Gong, M., Gong, Z., Gu, Y., Hu, J., Huang, Z., Lan, Z., et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.

Chang, H., Zhang, H., Jiang, L., Liu, C., and Freeman, W. T. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11315–11325, 2022.

Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

Chen, J., Ge, C., Xie, E., Wu, Y., Yao, L., Ren, X., Wang, Z., Luo, P., Lu, H., and Li, Z. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pp. 74–91. Springer, 2024.

Chen, J., Cai, Z., Chen, P., Chen, S., Ji, K., Wang, X., Yang, Y., and Wang, B. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025a.

Chen, J., Xu, Z., Pan, X., Hu, Y., Qin, C., Goldstein, T., Huang, L., Zhou, T., Xie, S., Savarese, S., et al. Blip3-o: A family of fully open unified multimodal

models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025b.

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025c.

Chen, Z., Bai, X., Shi, Y., Fu, C., Zhang, H., Wang, H., Sun, X., Zhang, Z., Wang, L., Zhang, Y., et al. Opengpt4o-image: A comprehensive dataset for advanced image generation and editing. arXiv preprint arXiv:2509.24900, 2025d.

Cheng, S., Bian, Y., Liu, D., Zhang, L., Yao, Q., Tian, Z., Wang, W., Guo, Q., Chen, K., Qi, B., et al. Sdar: A synergistic diffusion-autoregression paradigm for scalable sequence generation. arXiv preprint arXiv:2510.06303, 2025.

Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Fang, R., Yu, A., Duan, C., Huang, L., Bai, S., Cai, Y., Wang, K., Liu, S., Liu, X., and Li, H. Flux-reason-6m & prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., and Ji, R. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Google. Gemini diffusion, 2025. URL https://deepmind.google/models/ gemini-diffusion.

Gu, S., Zhang, J., Zhou, S., Yu, K., Xing, Z., Wang, L., Cao, Z., Jia, J., Zhang, Z., Wang, Y., et al. Infinity-mm: Scaling multimodal performance with large-scale and highquality instruction data. arXiv preprint arXiv:2410.18558, 2024.

Guo, J., Zheng, T., Li, Y., Bai, Y., Li, B., Wang, Y., Zhu, K., Neubig, G., Chen, W., and Yue, X. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13869–13920, 2025.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Hoogeboom, E., Nielsen, D., Jaini, P., Forr´e, P., and Welling, M. Argmax flows and multinomial diffusion: Learning categorical distributions. NeurIPS, 34:12454–12465, 2021.

Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., and Yu, G. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

jackyhate. jackyhate/text-to-image-2m. https: //huggingface.co/datasets/jackyhate/ text-to-image-2M, 2024.

Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., and Farhadi, A. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pp. 235–251. Springer, 2016.

Khanna, S., Kharbanda, S., Li, S., Varma, H., Wang, E., Birnbaum, S., Luo, Z., Miraoui, Y., Palrecha, A., Ermon, S., et al. Mercury: Ultra-fast language models based on diffusion. arXiv preprint arXiv:2506.17298, 1, 2025.

Kim, J., Cheuk-Kit, L., Domingo-Enrich, C., Du, Y., Kakade, S., Ngotiaoco, T., Chen, S., and Albergo, M. Any-order flexible length masked diffusion. arXiv preprint arXiv:2509.01025, 2025.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Labs, B. F. Flux. https://github.com/ black-forest-labs/flux, 2024.

Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., and Shan, Y. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.

Li, S., Gu, J., Liu, K., Lin, Z., Wei, Z., Grover, A., and Kuen, J. Lavida-o: Elastic large masked diffusion models for unified multimodal understanding and generation. arXiv preprint arXiv:2509.19244, 2025a.

Li, S., Kallidromitis, K., Bansal, H., Gokul, A., Kato, Y., Kozuka, K., Kuen, J., Lin, Z., Chang, K.-W., and Grover, A. Lavida: A large diffusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839,

- 2025b.

Li, Z., Li, H., Shi, Y., Farimani, A. B., Kluger, Y., Yang, L., and Wang, P. Dual diffusion for unified image generation and understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2779–2790,

- 2025c.

Liang, W., Yu, L., Luo, L., Iyer, S., Dong, N., Zhou, C., Ghosh, G., Lewis, M., Yih, W.-t., Zettlemoyer, L., et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

Liang, Z., Li, Y., Yang, T., Wu, C., Mao, S., Nian, T., Pei, L., Zhou, S., Yang, X., Pang, J., et al. Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies. arXiv preprint arXiv:2508.20072, 2025.

Liao, C., Liu, L., Wang, X., Luo, Z., Zhang, X., Zhao, W., Wu, J., Li, L., Tian, Z., and Huang, W. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, Q. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024.

Lou, A., Meng, C., and Ermon, S. Discrete diffusion modeling by estimating the ratios of the data distribution. In Forty-first International Conference on Machine Learning, 2024.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775–5787, 2022.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. CoRR, 2023.

Ma, J., Zhu, X., Pan, Z., Peng, Q., Guo, X., Chen, C., and Lu, H. X2edit: Revisiting arbitrary-instruction image editing through self-constructed data and task-aware representation learning. arXiv preprint arXiv:2508.07607, 2025.

Masry, A., Long, D. X., Tan, J. Q., Joty, S., and Hoque, E. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

Mathew, M., Karatzas, D., and Jawahar, C. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., and Jawahar, C. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1697–1706, 2022.

Meyer, J., Padgett, N., Miller, C., and Exline, L. Public domain 12m: A highly aesthetic image-text dataset with novel governance mechanisms. arXiv preprint arXiv:2410.23144, 2024.

Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., and Li, C. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.

Ou, J., Han, J., Xu, M., Xu, S., Xie, J., Ermon, S., Wu, Y., and Li, C. Principled rl for diffusion llms emerges from a sequence-level perspective. arXiv preprint arXiv:2512.03759, 2025.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Pynadath, P., Shi, J., and Zhang, R. Candi: Hybrid discrete-continuous diffusion models. arXiv preprint arXiv:2510.22510, 2025.

Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., and Kuleshov, V. Simple and effective masked diffusion language models. arXiv preprint arXiv:2406.07524, 2024.

Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. K. Simplified and generalized masked diffusion for discrete data. arXiv preprint arXiv:2406.04329, 2024.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.

Song, Y., Zhang, Z., Luo, C., Gao, P., Xia, F., Luo, H., Li, Z., Yang, Y., Yu, H., Qu, X., et al. Seed diffusion: A large-scale diffusion language model with high-speed inference. arXiv preprint arXiv:2508.02193, 2025.

Swerdlow, A., Prabhudesai, M., Gandhi, S., Pathak, D., and Fragkiadaki, K. Unified multimodal discrete diffusion. arXiv preprint arXiv:2503.20853, 2025.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024a.

Team, Q. Qwen2.5: A party of foundation models, September 2024b. URL https://qwenlm.github.io/ blog/qwen2.5/.

Tong, S., Fan, D., Li, J., Xiong, Y., Chen, X., Sinha, K., Rabbat, M., LeCun, Y., Xie, S., and Liu, Z. Metamorph: Multimodal understanding and generation via instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17001–17012, 2025.

Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

Wen, J., Zhu, M., Liu, J., Liu, Z., Yang, Y., Zhang, L., Zhang, S., Zhu, Y., and Xu, Y. dvla: Diffusion visionlanguage-action model with multimodal chain-of-thought. arXiv preprint arXiv:2509.25681, 2025a.

Wen, Y., Li, H., Gu, K., Zhao, Y., Wang, T., and Sun, X. Llada-vla: Vision language diffusion action models. arXiv preprint arXiv:2509.06932, 2025b.

Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12966–12977, 2025a.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025b.

Wu, C., Zheng, P., Yan, R., Xiao, S., Luo, X., Wang, Y., Li, W., Jiang, X., Liu, Y., Zhou, J., et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025c.

Wu, Z., Zheng, L., Xie, Z., Ye, J., Gao, J., Feng, Y., Li, Z., W., V., Zhou, G., and Kong, L. Dreamon: Diffusion language models for code infilling beyond fixed-size canvas, 2025d. URL https://hkunlp.github.io/ blog/2025/dreamon.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Xie, J., Yang, Z., and Shou, M. Z. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

Xin, Y., Qin, Q., Luo, S., Zhu, K., Yan, J., Tai, Y., Lei, J., Cao, Y., Wang, K., Wang, Y., et al. Luminadimoo: An omni diffusion large language model for multimodal generation and understanding. arXiv preprint arXiv:2510.06308, 2025.

Yang, L., Tian, Y., Li, B., Zhang, X., Shen, K., Tong, Y., and Wang, M. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025a.

Yang, Y., Wang, C., Wang, S., Wen, Z., Qi, B., Xu, H., and Zhang, L. Diffusion llm with native variable generation lengths: Let [eos] lead the way. arXiv preprint arXiv:2510.24605, 2025b.

Ye, J., Xie, Z., Zheng, L., Gao, J., Wu, Z., Jiang, X., Li, Z., and Kong, L. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

You, Z., Nie, S., Zhang, X., Hu, J., Zhou, J., Lu, Z., Wen, J.-R., and Li, C. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025a.

You, Z., Ou, J., Zhang, X., Hu, J., Zhou, J., and Li, C. Effective and efficient masked image generation models. arXiv preprint arXiv:2503.07197, 2025b.

Yu, R., Ma, X., and Wang, X. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023.

Zhang, H., Duan, Z., Wang, X., Zhao, Y., Lu, W., Di, Z., Xu, Y., Chen, Y., and Zhang, Y. Nexus-gen: A unified model for image understanding, generation, and editing. arXiv preprint arXiv:2504.21356, 2025a.

Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.-W., Qiao, Y., et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024.

Zhang, Y., Ni, B., Chen, X.-S., Zhang, H.-R., Rao, Y., Peng, H., Lu, Q., Hu, H., Guo, M.-H., and Hu, S.M. Bee: A high-quality corpus and full-stack suite to unlock advanced fully open mllms. arXiv preprint arXiv:2510.13795, 2025b.

Zhao, S., Gupta, D., Zheng, Q., and Grover, A. d1: Scaling reasoning in diffusion large language models via reinforcement learning. arXiv preprint arXiv:2504.12216, 2025.

Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., and Levy, O. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

Zhou, J., Chen, H., Zhao, S., Kang, J., Li, J., Wang, E., Guo, Y., Sun, H., Wang, H., Kong, A., et al. Diffa: Large language diffusion models can listen and understand. arXiv preprint arXiv:2507.18452, 2025.

Zhu, F., Wang, R., Nie, S., Zhang, X., Wu, C., Hu, J., Zhou, J., Chen, J., Lin, Y., Wen, J.-R., et al. Llada 1.5: Variancereduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223, 2025a.

Zhu, F., You, Z., Xing, Y., Huang, Z., Liu, L., Zhuang, Y., Lu, G., Wang, K., Wang, X., Wei, L., et al. Llada-moe: A sparse moe diffusion language model. arXiv preprint arXiv:2509.24389, 2025b.

Zhuo, L., Zhao, L., Paul, S., Liao, Y., Zhang, R., Xin, Y., Gao, P., Elhoseiny, M., and Li, H. From reflection to perfection: Scaling inference-time optimization for text-to-image diffusion models via reflection tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15329–15339, 2025.

### A. Related Work

Diffusion large language models. Recently, diffusion large language models (dLLMs) (Nie et al., 2025; Ye et al., 2025; Zhu et al., 2025b) have emerged based on masked diffusion models (MDMs) (Ou et al., 2024; Lou et al., 2024; Shi et al.,

- 2024; Sahoo et al., 2024; You et al., 2025b), which are a special case of discrete diffusion models (Sohl-Dickstein et al., 2015; Hoogeboom et al., 2021). Through large-scale pretraining and supervised fine-tuning, these models have achieved performance comparable to strong autoregressive models such as LLaMA3. This demonstrates the practical applicability and strong potential of dLLMs as alternatives to ARMs. Beyond text generation, dLLMs have also made remarkable progress in various domains, including multimodal understanding (You et al., 2025a; Yu et al., 2025; Li et al., 2025b), audio understanding (Zhou et al., 2025), reinforcement Learning (Zhu et al., 2025a; Zhao et al., 2025; Ou et al., 2025) and vision-language-action tasks (Wen et al., 2025b;a; Liang et al., 2025). Most relevant to our work are unified multimodal models (Li et al., 2025a; Xin et al., 2025; Yang et al., 2025a). However, unlike these approaches that rely on masked diffusion models, we employ continuous diffusion models for image generation, thereby avoiding the information loss caused by image discretization.

Unified multimodal models are initially dominated by autoregressive architectures (Team, 2024a; Wang et al., 2024), which leverage structural consistency with LLMs via unified next-token prediction. Subsequently, hybrid approaches combine autoregressive text processing with diffusion-based image generation (Zhou et al., 2024; Xie et al., 2024; Tong et al., 2025; Liao et al., 2025; Deng et al., 2025). Concurrently, Diffusion Large Language Models (dLLMs) (Nie et al., 2025; Ye et al.,

- 2025; Zhu et al., 2025b) have achieved performance comparable to strong autoregressive models via large-scale pretraining and supervised fine-tuning. Building on this success, recent dLLM-based unified multimodal models (Li et al., 2025a; Xin et al., 2025; Yang et al., 2025a; Swerdlow et al., 2025) employ masked diffusion mechanisms for both multimodal understanding and generation.

### B. Experimental Details

- B.1. Training Stages and Data Setup The training of LLaDA-o proceeds in a preliminary alignment phase followed by three progressive stages.

Projector Alignment. Prior to the main training, we focus on aligning visual representations with the understanding expert. We train the MLP projector using the Stage 1 data from Infinity-MM (Gu et al., 2024), while keeping all other components frozen.

- Stage 1: Foundation Setup. In this stage, we establish the baseline capabilities using large-scale image understanding and generation data. We restrict image generation to a resolution of 512 and do not apply adaptive length augmentation for multimodal understanding. For text, we use an in-house 10M SFT dataset. For multimodal understanding, we utilize Stage 2-4 data from Infinity-MM (Gu et al., 2024), MAmmoTH-VL-Instruct-12M (Guo et al., 2025), LLaVA-OneVision1.5 (An et al., 2025), and FineVision (An et al., 2025). For text-to-image generation, we combine image captioning data (from Infinity-MM (Gu et al., 2024) and LLaVA-OneVision-1.5 (An et al., 2025)) with generation datasets including PD12M (Meyer et al., 2024), Text-to-Image-2M (jackyhate, 2024), BLIP3o-Pretrain (Chen et al., 2025b), Nexus-Gen (Zhang et al., 2025a), FLUX-Reason-6M (Fang et al., 2025), and synthetic data from Qwen-Image (Wu et al., 2025b).
- Stage 2: High-Resolution and Reasoning. We further incorporate multimodal reasoning data and increase the generation resolution to 1024. Adaptive length augmentation remains disabled for understanding tasks. In terms of data, we switch the understanding source to Honey-Data-15M (Zhang et al., 2025b). For generation, we refine the dataset by removing image captions and PD12M (Meyer et al., 2024), while increasing the sampling ratio of the remaining high-quality data. Additionally, we introduce interleaved multimodal data from X2Edit (Ma et al., 2025) and OmniGen2 (Wu et al., 2025c), along with editing data from ShareGPT-4o-Image (Chen et al., 2025a) and OpenGPT-4o-Image (Chen et al., 2025d). The text data remains unchanged.
- Stage 3: Variable-Length Refinement. In the final stage, we jointly apply adaptive length augmentation to activate variable-length generation for the understanding expert and fine-tune the model with high-quality data. For understanding, we incorporate Honey-Data-1M and retain Honey-Data-15M (Zhang et al., 2025b) with a reduced ratio. For generation, we add premium datasets including ShareGPT-4o-Image (Chen et al., 2025a), OpenGPT-4o-Image (Chen et al., 2025d), BLIP3o-60k (Chen et al., 2025b), and GenRef (Zhuo et al., 2025), while reducing the ratio of the Stage 2 generation data. For interleaved data, we remove X2Edit (Ma et al., 2025) and keep the rest consistent with Stage 2.

#### B.2. Multi-turn Dialogue Data and Interleaved Multimodal Data Setup

For multi-turn dialogues, LLaDA (Nie et al., 2025) randomly selects one turn for training. It concatenates the preceding dialogue history, including both prompts and responses, as the input context for that turn, and computes the loss only on the selected turn’s response. This setup is also used in LaViDa (Li et al., 2025b) and Dimple (Yu et al., 2025). In contrast, LLaDA-V (You et al., 2025a) computes the loss on the response of every turn in the dialogue. In LLaDA-o, we follow the LLaDA-V strategy, since it achieves strong performance and shows data scalability in LLaDA-V.

For interleaved multimodal data, we treat the text only as a condition (i.e., prompt) for image generation. Thus, we compute the continuous diffusion loss only on each turn of image tokens and mask out the loss on text tokens, preventing the image objective from interfering with text representations.

#### B.3. Prompts of Selected Generated Images

- Tab. 7 lists the text prompts used to generate the samples shown in Fig. 1. The Image IDs in the table correspond to the spatial arrangement of the images: IDs 1–5 represent the top row (from left to right), and IDs 6–10 represent the bottom row.

- Table 7. Prompts for the generated samples. The Image IDs correspond to the order of images in Fig. 1, arranged from left to right and top to bottom.

Image ID Prompt

- 1 Bright scene, aerial view, ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens.
- 2 A photorealistic corgi sitting calmly in a traditional Chinese tea house, steam from the teacup forming a tiny dragon, morning light, detailed fur.
- 3 A detective cat in a trench coat standing in a rainy alleyway, cyberpunk neon signs reflecting in puddles, film noir style, volumetric fog, dramatic shadows, cinematic shot.
- 4 An astronaut riding a horse on the moon, oil painting by Van Gogh.
- 5 A miniature library inside a vintage lightbulb, warm cozy light.
- 6 A raccoon wearing a tiny backpack and using a map to navigate a misty mountain trail, storybook illustration, soft colors.
- 7 A complex landscape of a futuristic city made entirely of folded origami paper, soft warm paper texture, subsurface scattering, dramatic lighting from inside the paper buildings, tilt-shift photography.
- 8 A transparent glass piano in a dark forest, glowing bioluminescent mushrooms growing inside the piano, magical dust floating in the air, fantasy illustration, intricate details.
- 9 A red sports car made entirely of knitted wool, soft studio lighting.
- 10 Portrait of a futuristic robot, pencil sketch by Leonardo da Vinci.

B.4. Computational Cost

- Tab. 8 details the computational resources and time required for each training stage of LLaDA-o. The primary training phases (Stage 1 and Stage 2) are conducted on 256 NVIDIA H800 GPUs, accounting for the majority of the computational cost. The final refinement stage (Stage 3) is performed using 64 NVIDIA A100 GPUs.

#### B.5. Text Generation Process of LLaDA-o

During inference, we adopt a blockwise sampling procedure (Arriola et al., 2025). As outlined in Algorithm 2, we first cache the fixed prefix (images and prompt). Subsequently, we extend the sequence by appending a block of length L, initialized with mask tokens. Within each block, we perform iterative denoising: tokens with prediction confidence exceeding a threshold τ are accepted, while others remain masked for the next step. If an [EOS] token is detected, we truncate the sequence and terminate; otherwise, we update the cache with the completed block and proceed to the next masked block.

- Table 8. Computational resources and training cost. The table details the GPU hardware and total GPU hours consumed in each training stage.

#### Training Stage GPUs GPU Hours

- Stage 1 256 H800 55,296

- Stage 2 256 H800 30,720

- Stage 3 64 A100 1,536

Algorithm 2 Text Generation Process of LLaDA-o.

- 1: Input: Parameters θ; Images V; Prompt P; Block size L; Threshold τ.
- 2: Output: Generated sequence y.
- 3: C ← ENCODE(V,P;θ); y ← ∅ {Initial Context Encoding}
- 4: loop
- 5: b ← [MASK]L; M ← {1,...,L} {Init Block & Mask indices}
- 6: while M ̸= ∅ do
- 7: p ← fθ(b,C) {Forward pass using parameters θ}
- 8: for i ∈ M do
- 9: if max(pi) > τ then
- 10: bi ← arg max(pi) {Keep high-confidence tokens}
- 11: M ← M \ {i}
- 12: end if
- 13: end for
- 14: end while
- 15: if EOS ∈ b then
- 16: return y ∥ TRUNC(b,EOS)
- 17: end if
- 18: y ← y ∥ b; C ← UPDATECACHE(C,b)
- 19: end loop

### C. Additional Results

#### C.1. Additional Generated Images

We provide additional qualitative examples generated by LLaDA-o in Fig. 5. These samples further demonstrate the model’s capability to produce high-quality images that are semantically aligned with user prompts, exhibiting both high fidelity and diversity.

#### C.2. Qualitative Comparison with LLaDA-V under Mismatched Block Lengths

In this section, we present qualitative samples demonstrating the variable-length text generation capability of LLaDA-o. As shown in Tab. 9, we compare LLaDA-o with the state-of-the-art LLaDA-V in scenarios where the semantic requirement of the prompt conflicts with the pre-defined generation block length.

First, as shown in the top row of Tab. 9, when the user requests simple text extraction but provides a long block length (i.e., L = 64), LLaDA-V tends to fill the entire window with redundant content. In contrast, LLaDA-o accurately extracts the text and correctly terminates, adhering to the user’s intent. Second, as shown in the bottom row, when the user requests a detailed image description but assigns a short block length (i.e., L = 16), LLaDA-V is constrained by the fixed window, resulting in an overly brief response. Conversely, LLaDA-o automatically extends the generation by appending additional blocks if no End-of-Sequence (EOS) token is detected within the current block, continuing until the generation is complete.

#### C.3. Effect of Block Length on Variable-Length Generation

In this section, we present qualitative samples to illustrate the variable-length generation behavior of LLaDA-o. As shown in Tab. 10, we maintain the same image and prompt while varying the pre-defined generation block length L ∈ {16,32,64,128}.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

A giant astronaut sitting on top of the Golden Gate Bridge fishing in the clouds, dreamlike atmosphere, vintage travel poster style, pastel colors, matte painting.

A cute tiny robot holding a red balloon, stop-motion claymation style, fingerprints visible on the clay, studio lighting, depth of field, plasticine texture.

The Great Wave off Kanagawa in a cyberpunk style, neon lights glowing on the waves, futuristic Tokyo skyline in the background, traditional woodblock print texture mixed with digital glitch art.

Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

A blooming rose made of colorful smoke, black background.

A snail with a disco ball as its shell, macro photography, realistic.

A running horse formed by splashing blue water, frozen motion photography.

A transparent glass hamburger with neon lights glowing inside, dark background.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

A typewriter made of clear glass, with tiny, living fireflies trapped inside as the "letters", on a writer's wooden desk, soft glow.

An octopus playing a drum set in a jazz club, cinematic lighting.

Macro photography of a dewdrop on a spider web, reflecting a whole galaxy inside the droplet, extreme closeup, high detail, bokeh background, magical atmosphere.

A majestic horse made of splashing liquid water, galloping on a sea of fire, high speed photography, frozen motion, crystal clear droplets, cinematic lighting, strong contrast between blue and orange.

Figure 5. Additional generated samples. We present 12 randomly selected images generated by LLaDA-o. For each sample, the prompt used for generation is shown below the corresponding image. All results are produced under the same setting as in the main paper.

Across these different settings, the generated outputs remain largely consistent, with only minor fluctuations in response length. This suggests that LLaDA-o is not rigidly constrained by the specific block length and can dynamically adapt the

##### effective output length based on the user prompt and image content.

- Table 9. Qualitative samples demonstrating variable-length generation. Top: Under a long block length setting (L = 64), LLaDA-V generates redundant content for a simple prompt, while LLaDA-o correctly terminates. Bottom: Under a short block length setting (L = 16), LLaDA-V produces an overly brief response for a detailed prompt, whereas LLaDA-o dynamically adapts the output length to the user prompt.

User What’s written on this image, just give me the answer.

[Figure 37]

LLaDA-V The image displays a lightboard sign with the phrase “TIME FOR CHANGE” written in bold, colorful letters. The sign is set against a backdrop of blurred lights, suggesting a festive or celebratory atmosphere. The text on the sign reads “TIME FOR CHANGE,” indicating a call to action or a message for transformation.

LLaDA-o TIME FOR CHANGE User Please describe this image in detail.

[Figure 38]

LLaDA-V The image captures a serene landscape featuring a tranquil lake surrounded by towering mountains.

LLaDA-o The image captures a serene mountain landscape, likely a popular tourist destination. Dominating the scene is a large lake, its surface a vibrant shade of green, reflecting the surrounding mountains. The lake’s edge is lined with a wooden fence, adding a rustic charm to the scene. Beyond the fence, a grassy area stretches out, dotted with trees and a small building, possibly a shelter or a visitor center. The building is nestled among the trees, blending seamlessly with the natural surroundings. The mountains rise majestically in the background, their peaks dusted with snow, suggesting a high-altitude location. The sky above is a clear blue, adding to the overall tranquility of the scene. The image does not contain any visible text.

- Table 10. Qualitative samples demonstrating variable-length generation. We show generations from LLaDA-o for the same image and the same prompt under different block length settings (L ∈ {16, 32, 64, 128}). Despite changing L, the output length remains largely stable, indicating that LLaDA-o can dynamically adapt the output length to the user prompt and image.

User Please describe this image in detail.

[Figure 39]

LLaDA-o (block:16)

The image captures a serene mountain landscape, likely a popular tourist destination. Dominating the scene is a large lake, its surface a vibrant shade of green, reflecting the surrounding mountains. The lake’s edge is lined with a wooden fence, adding a rustic charm to the scene. Beyond the fence, a grassy area stretches out, dotted with trees and a small building, possibly a shelter or a visitor center. The building is nestled among the trees, blending seamlessly with the natural surroundings. The mountains rise majestically in the background, their peaks dusted with snow, suggesting a high-altitude location. The sky above is a clear blue, adding to the overall tranquility of the scene. The image does not contain any visible text.

LLaDA-o (block:32)

The image depicts a serene, picturesque natural landscape featuring a tranquil lake surrounded by rugged mountains and lush greenery. The lake is a vibrant shade of turquoise, reflecting the clear blue sky above. A wooden fence runs along the edge of the lake, separating the grassy area from the water. The fence is simple and rustic, with vertical wooden posts and horizontal rails. In the foreground, there is a well-maintained grassy area with a wooden fence. The grass is lush and green, indicating a healthy, natural environment. A few yellow flowers are scattered throughout the grass, adding a touch of color to the scene. The overall atmosphere of the image is peaceful and inviting, suggesting a perfect spot for relaxation and enjoyment of the beauty of nature.

LLaDA-o (block:64)

The image depicts a serene natural landscape featuring a small, clear lake surrounded by lush green grass and tall evergreen trees. The lake is nestled in a valley, with rugged, rocky mountains rising in the background. The sky is clear and blue, indicating a sunny day. In the foreground, there is a wooden fence that runs along the edge of the grassy area, separating it from the lake. A small wooden structure, possibly a cabin or a small house, is visible on the right side of the image, partially obscured by the tall trees. The overall scene is peaceful and idyllic.

LLaDA-o (block:128)

The image captures a serene mountain landscape, featuring a small, clear lake nestled in a valley. The lake is surrounded by lush green grass and trees, creating a stark contrast with the rocky mountains in the background. The mountains, with their rugged peaks, rise majestically against the clear blue sky. The perspective of the image is from a low angle, looking up at the mountains, which adds a sense of grandeur and majesty to the scene. The overall mood of the image is peaceful and tranquil, inviting the viewer to immerse themselves in the natural beauty of this mountainous paradise.

