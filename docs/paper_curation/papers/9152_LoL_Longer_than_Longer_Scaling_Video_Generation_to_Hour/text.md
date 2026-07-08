# arXiv:2601.16914v1[cs.CV]23Jan2026

[Figure 1]

[Figure 2]

[Figure 3]

## LoL: Longer than Longer, Scaling Video Generation to Hour

### Justin Cui1,2 Jie Wu2,† Ming Li2,3 Tao Yang2 Xiaojie Li2 Rui Wang2 Andrew Bai1 Yuanhao Ban1 Cho-Jui Hsieh1,‡

1UCLA, 2ByteDance Seed, 3University of Central Florida ‡Corresponding author, †Project lead

### Abstract

Recent research in long-form video generation has shifted from bidirectional to autoregressive models, yet these methods commonly suffer from error accumulation and a loss of long-term coherence. While attention sink frames have been introduced to mitigate this performance decay, they often induce a critical failure mode we term sink-collapse: the generated content repeatedly reverts to the sink frame, resulting in abrupt scene resets and cyclic motion patterns. Our analysis reveals that sink-collapse originates from an inherent conflict between the periodic structure of Rotary Position Embedding (RoPE) and the multi-head attention mechanisms prevalent in current generative models. To address it, we propose a lightweight, training-free approach that effectively suppresses this behavior by introducing multi-head RoPE jitter that breaks inter-head attention homogenization and mitigates long-horizon collapse. Extensive experiments show that our method successfully alleviates sink-collapse while preserving generation quality. To the best of our knowledge, this work achieves the first demonstration of real-time, streaming, and infinitelength video generation with little quality decay. As an illustration of this robustness, we generate continuous videos up to 12 hours in length, which, to our knowledge, is among the longest publicly demonstrated results in streaming video generation.

### 1 Introduction

Video generation has undergone rapid evolution, propelled by advances in diffusion-based generative modeling. Breakthrough systems such as Sora [33], CogVideoX [49], Wan [44], Hunyuan-DiT [26] and Veo [14] have pushed the frontier of realism in synthesized motion and visual fidelity. These models exhibit remarkable capability in capturing complex spatiotemporal dynamics, producing results that often blur the boundary between generated and real footage. However, they all suffer from heavy computation cost when extending to long sequences.

To address this limitation, recent work shifts from bidirectional designs to autoregressive generation, predicting each new frame from previously generated ones to support much longer temporal modeling. Among them, Diffusion-Forcing [3, 23] applies frame-wise noise schedules but often suffers from instability [4, 16]. Simpler approaches predict the next frame from clean contexts using KV cache for efficient streaming. CausVid [52] distills a bidirectional teacher into a streaming student but faces overexposure from overlapping frames and train–test mismatch. Self Forcing [19] alleviates this via distribution alignment, yet remains constrained by the teacher’s temporal limit. Recent works such as LongLive [48], Self-Forcing++ [7], and Rolling-Forcing [31] extend high-quality generation to minutes through windowed DMD on ultra-long sequences.

t=0-1 hour t=1-2 hour t=2-3 hour t=3-4 hour t=4-5 hour t=5-6 hour

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

t=6-7 hour t=7-8 hour t=8-9 hour t=9-10 hour t=10-11 hour t=11-12 hour

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

- Figure 1 Streamingly generated ultra long video (12 hours) for prompt “A cinematic third-person shot of a wingsuit flyer racing through a narrow mountain valley. The flyer dives downwards, weaving smoothly between jagged cliffs as snow-capped peaks..”.

Using attention sink to stabilize autoregressive models was first proposed in StreamingLLM [47] which showed that retaining KV pairs of initial tokens can recover performance lost in windowed attention in Large Language Models. This concept has recently been adopted in autoregressive video generation [19, 39, 48] to enhance alignment and stability. Despite differing training paradigms, state-of-the-art methods like LongLive [48] and Self-Forcing++ [7] exhibit a shared failure when using attention sink: frames repeatedly regress toward the sink frames, a phenomenon we term sink-collapse. For example, both methods collapse at exact latent frame index 132 and 201 as shown in figure 4, with more collapses emerging regardless of input noise or prompts. Repetition has also been observed when extending bidirectional models as shown in RIFLEx [54], however it attributes the repetition to a single temporal dimension which does not generalize to autoregressive settings.

In this paper, we introduce Longer than Longer (LoL), the first attempt to tackle the sink-collapse problem in state-of-the-art autoregressive video generation models and propose a way to extend streaming generation indefinitely. First, by examining the repetitive patterns in Self-Forcing++ [7] and LongLive [48], we find that although both exhibit sink-collapse at identical frame indices, no clear periodicity emerges, unlike the periodic behavior observed in bidirectional models such as RIFLEx [54]. Summing phase alignment around sink frames from all temporal dimensions reveals that collapse points coincide with local maxima, indicating that sink-collapse arises from multiple temporal dimensions. Second, since modern transformers rely on multi-head attention across subspaces, we analyze their attention patterns and find that collapse occurs when multiple heads simultaneously assign significantly high weights to sink frames. Motivated by these observations, we propose a simple yet effective remedy by shifting the base frequencies of different heads around the original base θ which reduces inter-head homogenization and mitigates collapse. Finally, we extend video length from multiple minutes [7, 48] to infinity by integrating streaming RoPE generation and noise sampling, and a 3D causal VAE decoder at inference, enabling continuous video generation with sustained quality. In summary, our contributions are:

- • We systematically analyze the phenomenon of sink collapse in state-of-the-art ultral long video generation models and reveal the root cause of such phenomenon in autoregressive long video generation.
- • We propose a simple yet highly effective method to mitigate sink collapse by shifting the frequencies of different attention heads. Together with our implementation of streaming RoPE generation and noise sampling, we are able to extend real-time video generation to indefinitely long without collapsing.
- • Extensive experiments show that our method preserves generation quality while effectively mitigating the sink-collapse phenomenon compared to baseline methods.

To the best of our knowledge, this is the first time that real-time infinite streaming generation is achieved with little quality degradation, while solely relying on models of 1.3B size and KV cache.

### 2 Related Work

Long Video Generation To extend the generation to long sequences, a number of techniques have been introduced [8, 12, 15, 18, 21, 22, 25, 30, 32, 53]. SkyReels-V2 [4] and MAGI-1 [42] adopt diffusion-forcing [3] to

[Figure 16]

- Figure 2 The plot of intra-head phase concentration which shows the normalized L2 distance to the sink frames against the mean phase of RoPE embeddings (base 10000) with respect to the sink frames. The results reveal that sink-collapse emerges almost exactly where the phase concentration attains local maxima. Besides sink-collapse events, additional drops in L2 distance also occur around local maxima.

support potentially infinite rollouts. CausVid [52] employs asymmetric distillation and block causal attention in training and a KV cache at inference time to autoregressively extend sequences. Self-Forcing [19] further aligns training with inference by incorporating the KV cache directly during training, producing high-quality short videos. FIFO-Diffusion [23] utilizes a first-in-first-out queue to perform heterogeneous denoising using a regular bi-directional diffusion model. APT [30] distills the model into 1-step generator and utilizes adversarial training with KV cache to extend the generation up to 60 seconds.

Ultra Long Video Generation Recently, the landscape of ultra-long video generation has advanced rapidly, marked by several key innovative methods such as Rolling-Forcing [31], which establishes a diffusion-forcing training paradigm; LongLive [48], which employs attention sinks for temporal coherence and KV-recaching for seamless prompt transitions; and Self-Forcing++ [7], which extends the DMD formulation to long-horizon modeling, significantly increasing the scalability of video diffusion models. SVI [28] utilizes a hybrid approach to generate motion and content-rich long videos. For the first time, these approaches enable the synthesis of multi-minute videos while preserving high visual fidelity and temporal stability.

Position Embedding To capture sequential dependencies among tokens, positional embeddings encode order information into input representations. Early approaches such as sinusoidal embeddings [43] assign fixed position vectors to each token, while later methods introduce learnable embeddings to adapt positions during training [10, 11]. More advanced designs, including relative position encodings [38] and Rotary Position Embedding [40], model positional relationships through relative distances or rotations in feature space, improving generalization to longer sequences. Extensions like YaRN [35] further enhance extrapolation by rescaling frequency parameters. Unlike these studies that focus on interpolation and extrapolation, our work instead investigates a distinct issue, namely the sink-collapse phenomenon in autoregressive video generation [7, 48, 50].

- 3 Method

In this section, we first introduce the background for transforming bi-directional models into autoregressive ones for ultra-long video generation, then identify and address the cause of sink-collapse. We then show how to achieve infinite streaming generation under the current setting.

#### 3.1 Background

Autoregressive Long Video Generation Due to high computational cost, state-of-the-art bidirectional models [13, 14, 27, 44] are limited to generating only a few seconds of video. Recently, LongLive [48] and Self-Forcing++ [7] employ a short-horizon bidirectional teacher to extend generation to several minutes, utilizing asymmetric distillation based on CausVid [52] and Self-Forcing [19], reaching nearly 99% of the base model’s positional embedding capacity. The core methodology behind these methods extends regular Distribution Matching Distillation [51] which computes the reverse KL loss Et DKL pgent ∥pdatat on windowed self-generated long

Head 1 Head 2 Head 3 Head 5 Head 8 Head 9 Head 11

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

| |
|---|

normal framessink-collapse 1sink-collapse 2

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

| |
|---|

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

| |
|---|

- Figure 3 Visualization of inter-head attention homogenization. The first three latent frames serve as attention sinks, and the last three are being generated. Results are shown for the same DiT layer and diffusion step (KV cache size = 12, with 3 sink tokens) of different frames. The top row shows normal frames while the bottom two rows show two different sink-collapse frames where multiple attention heads simultaneously assign significantly higher weights to sink frames causing abrupt scene change back to these frames.

sequences with backward noise initialization [7, 48], which can be formulated as

≈ −Et Ez

∇θL DMD

i

extended

sT Φ(Gθ(zi),t),t − sS Φ(Gθ(zi),t),t

dGθ(zi) dθ

dzi . (1)

Rotary Position Embedding (RoPE) To enable the model to differentiate tokens across positions, positional embeddings are introduced to inject positional information into input representations [2, 35, 40, 43]. Among these approaches, Rotary Position Embedding (RoPE) [40] has become one of the most widely used methods in transformer-based architectures. RoPE encodes temporal or spatial positions by rotating the query and key vectors within a complex plane according to their token indices, effectively allowing the model to capture relative positional relationships. For a given position index t and channel dimension i, the rotation is defined as

cos(tθi) −sin(tθi) sin(tθi) cos(tθi)

2i

θi = θ−

, (2)

d , R(t,i) =

where d is the hidden dimension and θ (typically 10,000) controls the base frequency. The rotated query and key vectors are obtained by

q′m = R(m,i)qm, k′n = R(n,i)kn. (3) When computing attention between positions m and n, their rotated forms yield

⟨q′m,k′n⟩ = ⟨qm, R(n − m)kn⟩, (4)

where R(n − m) represents a rotation parameterized by the relative offset n − m indicating that the attention score depends on positional differences, enabling RoPE to naturally encode relative positional information.

#### 3.2 Sink Frames & Sink Collapse

Attention sink is first proposed in StreamingLLM [47] to extend LLM generation to significantly long sequences. Recent work such as LongLive [31, 48] extends it to autoregressive ultra long video generation for more stable generations.

In StreamingLLM, the initial tokens remain in the KV cache without being evicted. Similarly in autoregressive video generation, the initial frames get preserved when rolling the KV cache which are referred to as sink frames.

However, as shown in figure 2 that while sink frames can increase the overall alignment and stability, the generated videos are shown to constantly fallback to the initial frames abruptly, a phenomenon we refer to as sink collapse. For example, we plot the normalized L2 distance to the initial frames in figure 2 where significant drops at exactly the same positions such as frame index 132 and 201 can be seen across different prompts.

Upon close inspection, the key reason for sink collapse lies in the periodic nature of RoPE position embedding as describe before. While this rotation preserves relative phase differences in short contexts, the periodic trigonometric nature causes phase re-alignment at long horizons, effectively resetting positional distinctions. As generation proceeds autoregressively, this periodic aliasing leads multiple distant frames to share nearly identical embeddings, making the attention mechanism overemphasize these “sink” positions. Consequently, the model collapses into repetitive frames.

#### 3.3 Mitigating Sink Collapse

The phenomenon of duplication has also been observed when extending bi-directional models to generate longer sequences as discussed in [54] which proposes to identify the closest temporal dimension causing the repetition and alert its frequency. However, we find that it’s ineffective in autoregressive generation. We show that there are two major distinctions between the repetition of bi-directional models and autoregressive models which are summarized below.

Intra-head phase concentration We find that the repetition is not caused by a single temporal dimension but the results of all dimension together which is shown in figure 2 and section 4.3.1. E.g, for the observed repeat frame index 132, RIFLEx predicts the closest intrinsic frequency component index to be 8 and the period of it to be 118 which deviates from the actual repetition by a large margin. In section 4.3.1, we demonstrate that adjusting the closest intrinsic component alone does not resolve the issue, and that perturbing any individual RoPE component likewise yields no improvement, further justifying that the repetition is not caused by a single position embedding dimension. Instead, we visualize the phase concentration intensity in figure 2. It can be observed that sink-collapse emerges almost exactly when the phase concentrations reach their local

maxima. Formally, given the RoPE frequencies ωi = θ0−2i/d for i = 1,...,K, where K = d2, we define the phase coherence kernel for a relative displacement ∆ = m − n as

K

1 K

ej ω

i∆ . (5)

C(∆) =

i=1

Then the sink-relative phase concentration of a generated frame g can be formulated as Rsink(g) = C(g − s) where s is the sink frame. A large value indicates that multiple RoPE frequency components become phasealigned with the reference sink frame, leading to phase synchronization and consequently to the sink-collapse phenomenon.

Inter-head attention homogenization Prior work [39] has demonstrated that attending to sink frames constitutes a natural behavior that can enhance temporal alignment and stabilize the generation process. Nevertheless, given that modern transformer architectures rely on multi-head attention to capture diverse representational subspaces, our analysis reveals that sink-collapse is not attributed to any single attention head. Rather, it arises from a collective synchronization phenomenon, wherein multiple heads concurrently exhibit high phase concentration, resulting in a global degeneracy of attention diversity. We visualize the attention heatmap in figure 3 for regular frames (top row) and sink-collapse frames (bottom two rows) that’s being generated in a chunk size of 3 which is used by both LongLive [48] and Self-Forcing++ [7]. For regular frames, the model usually assign large self-attention weights to the tokens themselves that’s being generated (last 3 frames) and distribute the weights evenly to the rest of the tokens. However, for the frames where sink-collapse happen, nearly all attention heads in the same layer assign significant large weights to both the frames that are being generated and the sink frames, causing the model to copy these frames from all sub-spaces created by multi-head attention and results in abrupt scene transition back to sink frames.

Based on these two observation, we propose a simple yet effective approach to significantly mitigate it. Since sink-collapse happens when all attention heads exhibit strong similarities to the sink frames simultaneously as

shown in figure 3, we propose to shift the base frequency of different attention heads by a certain margin which we term as multi-head jitter which are described in the algorithm below. Owing to the inherent periodicity of the RoPE embedding, the introduced phase shift disrupts the global alignment among heads, thereby reducing the likelihood of simultaneous phase overlap across all heads and effectively mitigating the sink-collapse phenomenon.

Algorithm 1: LoL: Multi-Head RoPE Jitter Input: Q,K ∈RB×T×H×D, base θ0, jitter scale σθ Output: Rotated Q′,K′ with head-wise perturbed frequencies νi←−2i/dtime, i = 0:D/2−1 ; // frequency exponents for h = 1...H do

ϵh∼U[−1,1] θˆh←θ0(1 + σθϵh); ωh←[θˆν

h ,...,θˆhνD/2−1 ];

0

(Q′h,Kh′ )←RoPERotate(Qh,Kh,ωh); Q′←[Q′1,...,Q′H], K′←[K1′,...,KH′ ];

#### 3.4 Infinite Streaming Generation

Besides sink-collapse, ultra-long video generation is also constrained by the length of RoPE and the memory consumption of VAE decoding. For instance, the maximum generation length of both LongLive [48] and Self-Forcing++ [7] is 4 minutes and 15 seconds, limited by the 1024-frame latent sequence and VAE memory usage. We show that generation can be extended infinitely with little quality degradation owing to two inherent properties of current architectures: 1) Causal VAE. Both LongLive and Self-Forcing++ build upon Wan2.1 [44], which employs a 3D causal VAE that ensures temporal causality and allows a sliding-window decoding strategy which greatly reduces memory and computation. 2) Local Attention. Both models adopt local attention over the most recent N latent frames to limit computational complexity. As shown in equation (4), the dot product of two RoPE [40] embeddings mainly depends on their relative positional difference.

Thus, with sink-collapse mitigated, the model can generate videos of infinite length. During streaming generation, both initial noise and RoPE will be dynamically sampled, introducing minimal additional overhead compared to pre-generated methods.

### 4 Experiments

#### 4.1 Settings

Baseline methods Following RIFLEx [54], which tackles repetition in bidirectional models, we compare with several methods addressing position embedding limitations. PE directly extends the sequence length without modifying embeddings and serves as the baseline. PI [6] rescales RoPE frequencies as θPI = θ/s, where s = L′/L and L and L′ denote the training and inference sequence lengths, respectively. NTK [2] adjusts the base frequency b such that each dimension’s frequency is scaled as (λb)−2j/d

′

, where λ = sd′/(d′ − 2), d′ is the rotary dimension, and s = L′/L is the same as above. YARN [35] introduces fine-grained frequency grouping based on the number of cycles rj = (2π)−1Lθj over the training length L and interpolates between original and scaled frequencies using thresholds α and β via θjYaRN = γ(rj)θj + (1 − γ(rj))θj/s. Finally, RIFLEx [54] identifies the intrinsic repetition frequency k = arg minj |Nj − N| with Nj = 2π/θj, and updates it to θk = 2π/(Ls) to ensure Nk′ ≥ Ls, effectively mitigating repetition in bidirectional models such as HunyuanVideo [27] and CogVideoX [17].

Additionally, we compare with the following autoregressive methods on general video generation benchmarks such as NOVA [9], Pyramid Flow [22], SkyReels-V2-1.3B [4], MAGI-1-4.5B [42], CausVid [52] and SelfForcing [19].

Evaluation metrics In order to measure sink-collapse, we adopt No-Repeat score from RIFLEx [54] which computes the normalized L2 distance to the initial frames and was used to measure the repetition problem

|[Figure 38]|
|---|

[Figure 39]

|[Figure 40]|
|---|

[Figure 41]

[Figure 42]

… ….. …

PE

sink collapse

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

… ….. …

PI

reduced motion

[Figure 48]

|[Figure 49]|
|---|

|[Figure 50]|
|---|

[Figure 51]

[Figure 52]

NTK

… …

…..

sink collapse

[Figure 53]

|[Figure 54]|
|---|

|[Figure 55]|
|---|

[Figure 56]

[Figure 57]

RIFLEx

… ….. …

sink collapse

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Ours

… …..

…

all good

- Figure 4 Visualization of the results after applying different PE extension methods. Baseline approaches continue to exhibit sink-collapse or diminished motion, whereas our method effectively alleviates sink-collapse and best preserves motion dynamics, as shown in table 1.

in bidirectional models. Here, we apply a more strict measurement and compute the scores in the whole generation range and take the maximum value. Specifically, we compute two scores including Sink-Collapse Max which computes the maximum distance drop with respect to sink frames in any single test prompt, indicating the worst case scenario and Sink-Collapse Avg which computes the average distance drop across all test prompts, indicating the average scenario. Also following [7, 19, 28, 52], we adopt VBench [20] to evaluate the quality of generated videos [7, 52], consisting of 128 prompts from MovieGen [37]. We expect the proposed method to significantly improve Sink-Collapse score while maintaining similar generation quality.

Implementation Details We test our method on state-of-the-art ultra long video generation models including LongLive [48] and Self-Forcing++ [7] with a local attention window size of 12 and sink frames of 3 which uses 1.3B base model that enables generation at 20 fps on a single NVIDIA H100 as shown in [48]. For baseline position embedding extension methods, we use the last frame index before the first sink-collapse, e.g, 132 as the original length denoted as L in our equations and the expected generation length as the output length denoted as L′ in our equations. The frame indices in this paper all refer to latent frames, e.g, latent index 132 corresponds to around 528 video frames which is around 33 seconds at a frame rate of 16.

#### 4.2 Empirical Results

We present quantitative results of applying our method to LongLive [48] and Self-Forcing++ [7] in tables 1 and 2. As shown, naive position extrapolation (PE) leads to severe sink-collapse in both models. For instance, in LongLive, the maximum distance drop reaches 73.06 and the average drop 30.54; in Self-Forcing++, these values reach 68.07 and 34.11, respectively. This demonstrates that despite their distinct training paradigms, both methods suffer from pronounced sink-collapse when using attention sink frames. By contrast, position interpolation (PI) effectively alleviates the issue since positional encodings are interpolated before the first repetition occurs. However, this comes at the cost of a drastic reduction in dynamic degree where video motion becomes almost stagnant, consistent with observations in [54]. More advanced approaches such as NTK and YARN each make different trade-offs. NTK maintains higher motion dynamics with only a minor drop (about 6% for LongLive) but offers limited mitigation against sink-collapse. YARN, on the other hand, strongly suppresses sink-collapse but greatly hampers dynamics, similar to position interpolation. RIFLEx achieves state-of-the-art performance in reducing repetition for bidirectional models while largely preserving motion under the autoregressive setting. However, since it attributes repetition to a single temporal dimension, it fails to address sink-collapse effectively with collapse scores similar to position extrapolation. In contrast, our method (LoL) substantially mitigates sink-collapse in both maximum and average scores, reaching levels

- Table 1 Results of applying our methods to LongLive and Self-Forcing++ with attention sink frames on videos of 100 seconds. In both cases, our method can effectively mitigate the sink collapse problem while maintaining the overall quality of the video.

Method

Core Metrics General Metrics

Sink-Collapse Sink-Collapse Dynamic Temporal Text Clip Subject Background Motion Imaging Max ↓ Avg↓ Degree↑ Quality Alignment Score Consistency Consistency Smoothness Quality

Results on LongLive

PE 73.06 30.54 34.62 88.56 28.09 31.91 97.53 96.21 98.91 69.59 PI 4.97 2.27 00.35 85.25 24.62 29.11 99.65 99.00 99.00 56.47 NTK 41.11 11.64 28.72 87.95 27.98 31.79 97.88 96.57 99.03 69.83 YARN 11.17 5.08 2.67 85.17 28.02 31.93 99.35 97.93 99.49 68.89 RIFLEX 70.95 29.93 35.11 88.61 23.11 28.38 97.48 96.22 98.89 69.47 Ours (LoL) 16.67 3.93 35.27 88.69 27.80 31.68 97.62 96.25 98.91 69.45

Results on Self-Forcing++

PE 68.07 34.11 83.32 93.14 27.62 31.36 93.41 93.13 97.79 63.06 PI 17.07 2.62 1.95 84.66 26.51 31.05 98.94 97.19 99.19 69.80 NTK 49.65 14.96 82.90 93.04 27.26 30.95 93.41 93.10 97.68 62.86 YARN 33.04 6.69 36.71 88.29 27.19 31.26 96.77 95.28 95.28 67.50 RIFLEX 66.56 32.86 82.36 93.01 27.57 31.32 93.48 93.10 97.79 63.26 Ours (LoL) 22.70 6.12 81.20 92.91 27.38 31.14 93.74 93.11 97.72 62.92

Red color and Green color indicates whether there is severe repetition or significantly reduced motion. ↓ indicates lower is better, ↑ indicates higher is better.

- Table 2 Performance comparisons with other autoregressive video generation models on 75s and 100s videos. Results show that despite being training-free, our method effectively addresses the sink-collapse problem while maximally preserves video generation quality.

Results on 75s ↑ Results on 100s ↑

Model

Text Temporal Dynamic Framewise Text Temporal Dynamic Framewise Alignment Quality Degree Quality Alignment Quality Degree Quality

Autoregressive models

NOVA 23.37 86.32 31.24 31.53 22.89 86.24 31.09 31.03 MAGI-1 24.95 87.89 24.82 52.04 23.75 87.62 22.21 50.90 SkyReels-V2 22.70 88.99 39.89 51.55 22.05 88.80 38.75 50.48 CausVid 24.76 89.14 35.82 60.96 24.41 89.06 34.60 61.01 Self Forcing 23.39 87.79 29.15 60.02 22.00 87.39 26.41 58.25 Ultra-Long Autoregressive models

Self-Forcing++ 26.31 91.00 55.62 60.67 26.04 90.87 54.12 60.66 LongLive 28.08 88.64 35.14 64.16 28.09 88.56 34.62 64.05 Self-Forcing++(LoL) 27.39 93.08 81.30 60.59 27.38 92.91 81.20 60.25 LongLive(LoL) 27.85 88.78 35.77 63.90 27.80 88.69 35.27 63.76

comparable to position interpolation, while simultaneously preserving motion dynamics similar to position extrapolation. As a result, LoL enables indefinite, streaming video generation without collapsing, as illustrated

- in figure 4.

Additionally, we provide comparison results with other autoregressive baseline methods in table 2 which shows that our method can effectively address the sink-collapse problem without hindering the generation quality.

#### 4.3 Ablation Study

##### 4.3.1 Is repetition dominated by a single dimension?

Repetition phenomenon has been studied in bi-directional models as well. As mentioned in section 4.1 that state-of-the-art work such as RIFLEx [54] tries to identify the closest dimension that approximates the observed duplication frame and solve the problem by changing the frequency of that dimension. However,

- as we have shown before that sink-collapse in autoregressive generation is not caused by a single dimension but the results of all dimensions together in figure 2. Here, we further show that not only does adjusting the closest intrinsic dimension fail, but altering the frequency of any position-embedding dimension is also

ineffective. We plot the results of altering the dimension identified by RIFLEx and its neighboring dimensions

- in figure 5a.

1.0

L2dist.tosink

0.8

0.6

- k=6

- k=7

- k=8

- k=9

- k=10

0.4

0 50 100 150 200 250 300 350 400 Frame index

(a) Normalized L2 distance to sink frames of modifying the frequencies of the dimension identified by RIFLEx and its neighboring dimensions. None of the modifications alone can mitigate the sink-collapse phenomenon.

1.0

L2dist.tosink

0.8

0.6

= 6000 = 8000 = 12000 = 16000 = 20000

0.4

0 50 100 150 200 250 300 350 400 Frame index

(b) Normalized L2 distance to the first frame under different RoPE base values θ. Varying θ shifts the collapse index but fails to address the underlying problem.

Figure 5 Ablation studies on sink-collapse behavior under frequency and RoPE base modifications.

It can be seen that regardless of the dimensions whose frequency are altered, the sink-collapse cannot be effectively mitigated which further proves that the repetition is not caused by a single dimension.

##### 4.3.2 Can sink-collapse get fixed by different RoPE base?

In our main experiments, we follow the base mode [7, 45, 48] and use the standard RoPE base value of 10,000. While one might consider adjusting the base θ to alleviate the sink collapse issue, our analysis shows that this modification alone does not fundamentally resolve the problem. Changing θ only alters the phase progression rate of the positional embeddings, shifting the point at which sink collapse occurs rather than eliminating it. In summary, varying θ merely delays or advances the collapse schedule along the temporal axis. As shown in figure 5b, we set the base from 6000 to 20,000 which are significantly different from the original base value. However, the sink collapse phenomenon is not mitigated but shifted for different bases. In contrast, our method directly mitigates the root cause of sink collapse as described in section 3.3, maintaining stable and consistent generation quality across extended temporal ranges.

##### 4.3.3 What’s the impact of different jitter intensity?

Here we show the results of applying different jitter intensity σ. As shown in figure 6a that as we increase σ, the sink collapse phenomenon starts to alleviate. E.g, when the jitter intensity is 0.1 which is very small, the model still suffers from significant sink-collapse problem.

When we set σ to 0.5, the severity of sink-collapse has reduced, however, the problem can still be seen in extended generation length which can be seen at round 750 frame index. As we further increase the value to 0.8, the figure shows a smooth transition between frames and no significant drops are seen. Note that further increasing σ can also mitigate the problem but at the cost of reduced motion or quality. In general, we find 0.8 to strike a good balance between generation quality and mitigating sink-collapse which can be seen from the quantitative results in table 1 and table 2.

##### 4.3.4 How many attention heads do we need to jitter?

We randomly sample the number of heads to jitter based on the specified ratio using three different random seeds and report the aggregated results with standard deviation in figure 6b. As illustrated, increasing the number of jittered heads progressively alleviates the sink-collapse phenomenon, further confirming that the issue does not originate from any single attention head. Empirically, jittering all heads achieves the most significant mitigation while maintaining high generation quality, as shown in tables 1 and 2.

1.0

L2dist.tosink

0.8

0.6

= 0.1 = 0.5

- = 0.8

- = 0.9

0.4

100 200 300 400 500 600 700 800 Frame index

(a) Normalized L2 distance to sink frames for different RoPE jitter σ. Empirically, a value of 0.8 achieves a good balance between generation quality and sink-collapse mitigation.

1.0

L2dist.tosink

0.8

0.6

ratio=0.1 ratio=0.4 ratio=0.7 ratio=1.0

0.4

0 50 100 150 200 250 300 350 400 Frame index

(b) Normalized L2 distance to sink frames with varying numbers of jitter attention heads, averaged over three random seeds. Increasing the number of jitter heads gradually alleviates the sink-collapse phenomenon.

Figure 6 Ablation studies on RoPE jitter magnitude and the number of jitter attention heads.

### 5 Limitations and Further work

Although our method is training-free, fine-tuning or retraining may further improve overall performance. The generation quality is constrained by the underlying models, which rely on local attention and sink frames for improved alignment and stability. However maintaining long-term memory still remains challenging, especially for multi-hour videos which is an important direction of our future work. Given the inherent periodicity of RoPE position embeddings, future research may explore alternative embedding schemes or advanced training strategies. We also plan to improve controllability by integrating stronger control signals [39, 46], and enhance scalability by incorporating sparse or linear attention mechanisms [5, 41].

### 6 Limitations and Further work

As shown previously, the repetition doesn’t necessarily happen at global maximum of the RoPE phases. It happens at local maximums. Thus, our method can only significantly suppress the repetition behavior where sink collapse are rarely seen during hour-long video generation. Future work should investigate non-periodic position embeddings or advanced training strategies such as masking the sink frames should a repetition occurs. Although our method shows the potential for infinite video generation with little quality degradation, it is based on Wan2.1-T2V-1.3B, which has limited capacity. In addition, since the model has never been trained on real datasets, its performance is bounded by that of the teacher model. We plan to explore the use of stronger base models in the future. Currently, the model can generate videos in a streaming manner

- at about 16 FPS on a single GPU. We also hope to further improve its efficiency and extend it to support control signals in future work.

### References

- [1] Kling AI. Kling AI: Next-Gen AI Video &amp; AI Image Generator — app.klingai.com. https://app.klingai. com/, 2025.
- [2] bloc97. NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation., 2023. URL https://www.reddit.com/r/LocalLLaMA/comments/ 14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/.
- [3] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

- [4] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

- [5] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.

- [6] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.

- [7] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.

- [8] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.

- [9] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In ICLR, 2025.

- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

- [11] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

- [12] Xueji Fang, Liyuan Ma, Zhiyang Chen, Mingyuan Zhou, and Guo-jun Qi. Inflvg: Reinforce inference-time consistent long video generation with grpo. arXiv preprint arXiv:2505.17574, 2025.

- [13] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [14] Google DeepMind. Veo. https://deepmind.google/models/veo/, 2025.
- [15] Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, and Dahua Lin. End-to-end training for autoregressive video diffusion via self-resampling. arXiv preprint arXiv:2512.15702, 2025.

- [16] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

- [17] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

- [18] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.

- [19] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

- [20] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

- [21] Jiaxiu Jiang, Wenbo Li, Jingjing Ren, Yuping Qiu, Yong Guo, Xiaogang Xu, Han Wu, and Wangmeng Zuo. Lovic: Efficient long video generation with context compression. arXiv preprint arXiv:2507.12952, 2025.

- [22] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In ICLR, 2025.

- [23] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37:89834–89868, 2024.

- [24] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

- [25] Akio Kodaira, Tingbo Hou, Ji Hou, Masayoshi Tomizuka, and Yue Zhao. Streamdit: Real-time streaming text-to-video generation. arXiv preprint arXiv:2507.03745, 2025.

- [26] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [27] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025.

- [28] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025.

- [29] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.

- [30] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025.

- [31] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time, 2025. URL https://arxiv.org/abs/2509.25161.
- [32] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025.

- [33] OpenAI. Video generation models as world simulators. Technical report, OpenAI, February 2024. URL https://openai.com/index/video-generation-models-as-world-simulators/. Technical report.
- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [35] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023.

- [36] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in 200 k. arXiv preprint arXiv:2503.09642, 2025.

- [37] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

- [38] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. arXiv preprint arXiv:1803.02155, 2018.

- [39] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Schechtman, and Xun Huang. Motionstream: Real-time video generation with interactive motion controls, 2025. URL https://arxiv.org/abs/ 2511.01266.
- [40] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [41] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025.

- [42] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

- [43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

- [44] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [45] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [46] Angtian Wang, Haibin Huang, Jacob Zhiyuan Fang, Yiding Yang, and Chongyang Ma. Ati: Any trajectory instruction for controllable video generation. arXiv preprint arXiv:2505.22944, 2025.

- [47] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In ICLR, 2024.

- [48] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, and Song Hanand Yukang Chen. Longlive: Real-time interactive long video generation. 2025.
- [49] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

- [50] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout. arXiv preprint arXiv:2511.20649, 2025.

- [51] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.

- [52] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.

- [53] Lvmin Zhang, Shengqu Cai, Muyang Li, Chong Zeng, Beijia Lu, Anyi Rao, Song Han, Gordon Wetzstein, and Maneesh Agrawala. Pretraining frame preservation in autoregressive video memory compression. arXiv preprint arXiv:2512.23851, 2025.

- [54] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.

### More related work

Video Diffusion Models The landscape of video generation has been profoundly reshaped by the emergence of the Diffusion Transformer (DiT) [34], whose performance scales remarkably with computational capacity. Building upon DiT, a line of breakthroughs has followed. Sora [33] delivers highly realistic and temporally consistent videos with diverse, natural motion. Wan 2.1 [44] highlights the power of large-scale pretraining for high-resolution video synthesis, while CogVideoX [17, 49] enhances cross-modal fusion through expert transformers and adaptive LayerNorm, achieving superior text–motion alignment. Different from these architectures, Hunyuan Video [26] introduces a causal 3D VAE [24] for efficient spatio-temporal token compression and couples it with a large language model for text conditioning, yielding outstanding results. Commercial systems such as Seedance [13] and Kling [1] push quality and efficiency to new heights, while open initiatives like Open-Sora [29, 36] and Open-Sora-Plan [29] bring these advances to the open-source community, greatly accelerating progress in both realism and scalability.

### Phase alignment of other θ values

In this paper, our primary focus is on analyses conducted with the RoPE base value θ = 10000, which is the default configuration used in the pretrained Wan2.1-T2V-1.3B model [44]. This setting is also adopted by both LongLive [48] and Self-Forcing++ [7], where the same base value is retained in order to remain consistent with the original positional encoding spectrum of the backbone model. Here we further examine the behavior of sink-collapse under alternative choices of the RoPE base and analyze their patterns with respect to phase concentration. As illustrated in figure 7, experiments with a different θ value reveal a highly consistent pattern similar to that of θ = 10000. Sink-collapse phenomeon remains closely aligned with the emergence of phase concentration. In particular, the normalized L2 distance between consecutive representations shows a clear drop at the time step where the phase alignment reaches its local maxima. This provides additional empirical evidence for the problem discussed in section 3.3.

[Figure 63]

- Figure 7 The L2 distance to sink-frames compared to phase alignment at θ=20000. It can be seen that the sink-collapse location also highly correlates with the local maxima of phase concentration.

### The impact of sink frame number

In the main paper, our experiments primarily focus on the configuration that applies three sink tokens together with a local attention window of nine, resulting in a total window size of twelve. This setting follows the default configuration used in [48]. To further examine the robustness of the sink-collapse phenomenon, we additionally investigate how the number of sink frames affects the model’s behavior.

As shown in figure 8, we visualize the results obtained when increasing the number of sink frames from three to five. The results indicate that sink-collapse continues to occur even with a larger number of sink frames. This observation suggests that the instability is not sensitive to the specific choice of sink frame count and that increasing the number of sink frames alone does not resolve the underlying issue.

1.0

0.8

L2dist.tosink

0.6

- prompt1

- prompt2

- prompt3

- prompt4

0.4

0.2

0 50 100 150 200 250 300 350 400 Frame index

- Figure 8 Behavior of the model when configured with five attention-sink frames. Despite allocating a larger sink region, the generation still fall into repetitive patterns, illustrating that increasing the sink-frame budget does not alleviate the collapse.

Next, we reduce the number of sink frames to one, which represents the minimal possible sink-frame configuration. As shown in figure 9, even under this extreme setting, the model continues to exhibit sinkcollapse artifacts, indicating that the phenomenon persists despite minimizing the number of designated sink frames.

0 50 100 150 200 250 300 350 400 Frame index

0.4

0.5

0.6

0.7

0.8

0.9

1.0

L2dist.tosink

- prompt1

- prompt2

- prompt3

- prompt4

- Figure 9 Visualization of the sink-collapse phenomenon when only one attention-sink frame is used. The results indicate that reducing the sink-frame size to its minimum does not eliminate repetition artifacts, and the model still exhibits signs of sink collapse.

### Visualization of other model layers

Beyond the visualization provided in figure 3, we further include additional attention maps extracted from several different attention layers at the identified sink-collapse locations, as shown in figure 10. These visualizations demonstrate that the copying behavior is not confined to a single layer or a specific head. Instead, it consistently appears across multiple attention layers and is exhibited by a wide range of attention heads. This widespread pattern reinforces the observation that sink-collapse is a structural effect emerging throughout the model rather than a localized anomaly. Consequently, as described in the algorithm, shifting the phase alignment serves as an effective strategy to mitigate this behavior by disrupting the conditions under which such copying behavior becomes dominant.

Layer 6 Head 11 Layer 10 Head 9 Layer 12 Head 8 Layer 16 Head 6 Layer 23 Head 10 Layer 26 Head 4 Layer 27 Head 10

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 10 More visualization of attention heat map across different layers and different heads. It can be seen that the copying behavior is a prevalent issues throughout the model layers and attention heads.

Visualization of single prompt

Here, we present additional long-form visualizations of our generated videos, extending up to 12 hours in duration, as illustrated in figure 11. These extended results further demonstrate the robustness of our approach that the model sustains coherent structure, visual consistency, and motion stability throughout the entire generation window, with little quality degradation. This further proves that that video generation method using local attention and RoPE [40] can generate to much longer sequence as described in section 3.4.

t=0-1 hour t=1-2 hour t=2-3 hour t=3-4 hour t=4-5 hour t=5-6 hour

t=6-7 hour t=7-8 hour t=8-9 hour t=9-10 hour t=10-11 hour t=11-12 hour

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

- Figure 11 12-hour visualization for streamingly generated video for prompt "An enormous bloom of radiant jellyfish drifts through a breathtaking underwater world..."

Below is another streamingly generated 12-hour video featuring noticeably faster and more dynamic camera motion, as illustrated in figure 12. Despite the increased motion complexity, the generated sequence remains coherent and temporally consistent throughout the entire duration. This further demonstrates that our method maintains stable performance even under long-range, fast-motion scenarios.

t=0-1 hour t=1-2 hour t=2-3 hour t=3-4 hour t=4-5 hour t=5-6 hour

t=6-7 hour t=7-8 hour t=8-9 hour t=9-10 hour t=10-11 hour t=11-12 hour

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

- Figure 12 12-hour visualization for streamingly generated video for prompt "Cinematic FPV aerial shot flying forward over snow-capped mountains at golden hour, skimming along a razor ridgeline then dipping into a glacier valley..."

### Visualization of prompt switching

Our method is compatible with both single-prompt generation and scenarios that require prompt switching. To demonstrate this capability, we first present a one-hour video generated with several controlled prompt

transitions in figure 13, where the scene evolves as the prompts change over time. We further include another ultra-long video lasting 10-minutes but with a substantially larger number of prompt transitions in figure 14, highlighting a much more challenging setting with frequent semantic shifts. Across both examples, the generated sequences remain coherent and stable, indicating that our approach scales effectively even as the number and frequency of prompt switches increase.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

- Figure 13 1-hour visualization for streamingly generated video for prompts “The lens floats through a valley of multicolored trees” followed by “Leaves of emerald, amber, and rose swirl through the air like confetti.”, “Motion forward through a valley of yellow-colored grass”, etc and 12 hours for single prompt “A cinematic third-person shot of a wingsuit flyer racing through a mountain valley...”.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

- Figure 14 10-minute visualization with large scene transition for streamingly generated video for prompts “Nighttime streets blaze with neon on rain-slick asphalt as the camera hurtles into a high-speed chase.”, “The chase bursts onto the freeway. A fuel truck erupts, hurling flaming debris while the cars knife through shockwaves and falling metal.”, “A massive explosion tears through the street as the hero leaps through flame and shrapnel toward the finale.”, etc

### Discussion & Future Work

RoPE [40] position embedding has ben widely adopted by both Large Language Models and Video Diffusion models. However, due to the intrinsic modeling difference, the end results are different. Our method will greatly suppress the sink-collapse phenomenon without retraining. However, retraining should further improve the generation quality. Due to the periodic nature or RoPE, we plan to further investigate its impact beyond sink-collapse. Alternative position embedding could be tested as well. Also, during training, blocking the sink tokens at local phase concentration maxima is also possible which can be computed following equation (5). Although our method achieves infinite streaming generation, in order to further enhance the generation quality, we also plan to investigate models with larger size and integrating long-term memory into the model.

### Limitations

The primary objective of our work is to alleviate the sink-collapse phenomenon in ultra-long video generation frameworks such as LongLive [48] and Self-Forcing++ [7]. Because our method extends these models into

effectively infinite generation in a training-free manner, the overall generation quality is fundamentally bounded by the underlying LongLive and Self-Forcing++ architectures on which we build. First, the model has no mechanism for maintaining long-term memory. As a result, if an object leaves the frame and later reappears, or if it becomes occluded for an extended period, subject consistency may fail to hold. Second, the base model we rely on, a distilled 4-step Wan2.1-T2V-1.3B model, has limited representational capacity. When pushed to generate extremely long sequences for single prompts, such as videos lasting up to 12 hours, it can exhibit reduced visual diversity. As noted above, enhancing long-term memory and employing base models with enhanced capacity remain two important directions for our future work.

