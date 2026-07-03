# arXiv:2603.04379v1[cs.CV]4Mar2026

[Figure 1]

[Figure 2]

## Helios: Real Real-Time Long Video Generation Model

### Shenghai Yuan1,2,◦, Yuanyang Yin2,4,◦, Zongjian Li1, Xinwei Huang2, Xiao Yang3,§, Li Yuan1,†

1Peking University, 2ByteDance China, 3Canva, 4Chengdu Anu Intelligence ◦Work done during internship at ByteDance, §Project Leader, †Corresponding Author

### Abstract

We introduce Helios, the first 14B video generation model that runs at 19.5 FPS on a single NVIDIA H100 GPU and supports minute-scale generation while matching the quality of a strong baseline. We make breakthroughs along three key dimensions: (1) robustness to long-video drifting without commonly used anti-drifting heuristics such as self-forcing, error-banks, or keyframe sampling; (2) real-time generation without standard acceleration techniques such as KV-cache, sparse/linear attention, or quantization; and (3) training without parallelism or sharding frameworks, enabling image-diffusion-scale batch sizes while fitting up to four 14B models within 80 GB of GPU memory. Specifically, Helios is a 14B autoregressive diffusion model with a unified input representation that natively supports T2V, I2V, and V2V tasks. To mitigate drifting in long-video generation, we characterize typical failure modes and propose simple yet effective training strategies that explicitly simulate drifting during training, while eliminating repetitive motion at its source. For efficiency, we heavily compress the historical and noisy context and reduce the number of sampling steps, yielding computational costs comparable to—or lower than—those of 1.3B video generative models. Moreover, we introduce infrastructure-level optimizations that accelerate both inference and training while reducing memory consumption. Extensive experiments demonstrate that Helios consistently outperforms prior methods on both short- and long-video generation. We plan to release the code, base model, and distilled model to support further development by the community.

[Figure 3]

Project Page: https://pku-yuangroup.github.io/Helios-Page Email Address: yuanshenghai@stu.pku.edu.cn

[Figure 4]

### 1 Introduction

14B Real-Time Long Video Generation Model can be Cheaper, Faster but Keep Stronger than 1.3B — Helios Team

Over the past year, Diffusion Transformers have substantially advanced video generation [1, 8, 30, 31, 41, 43, 49, 63, 80, 90, 93, 122] and shown potential as world models [2–4, 20, 65, 66, 78, 83, 102, 112]. As video quality improves, demand for real-time generation has increased across applications, together with more requirements on video duration—especially for game engines [27, 44, 81, 104, 118, 127] and interactive generation [19, 51, 74, 108, 111]. However, mainstream models remain far from real time and infinite: they typically generate only 5–10 seconds, and even these short clips can require tens of minutes to synthesize.

Real-Time Infinity Video Generation aims to generate temporally coherent, high-quality long videos at interactive speeds, but this goal remains largely unsolved. Several community methods claim real-time infinite

[Figure 5]

- Figure 1 End-to-end throughput (FPS) of various video generation models on a single H100. The results are obtained at the same resolution with all official acceleration techniques, including FlashAttention, torch compile, and KV-cache. Helios is substantially faster than models at the same scale and matches the speed of smaller distilled ones.

[Figure 6]

- Figure 2 Benchmark performance of Helios and its counterparts. For both short- and long-video generation, Helios consistently outperforms existing distilled models while achieving performance comparable to that of base models.

generation; however, these approaches typically rely on 1.3B models [11, 26, 59, 60, 100]. The limited capacity of these models makes it difficult to represent complex motion and often leads to blurred high-frequency details. Krea-RealTime-14B [67] increases the model scale, but it largely follows the same paradigm and reaches only 6.7 FPS on a single H100 GPU. In addition, these methods often rely on train-as-infer rollouts (Self-Forcing [34]) to mitigate drifting, which substantially increases training cost and motivates step distillation [52, 61, 62, 105, 106]. More critically, robustness to drifting is tightly coupled to the rollout length used during training: when training is restricted to 5-second clips, severe drifting often emerges beyond the 5-second horizon at inference. Finally, these long-video generation methods based on causal masking [107] fundamentally change the inference regime of bidirectional pre-trained models and may limit the achievable quality.

To address these challenges, we propose Helios, a 14B recipe for real-time long-video generation that runs

The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it's tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The ...

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

A vibrant tropical fish glides gracefully through colorful ocean reefs, surrounded by swaying coral, shimmering schools of tiny fish, and beams of sunlight filtering down from the water’s surface. The scene feels alive with movement, as bubbles rise gently and the reef glows in vivid shades ...

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in thought pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down shirt ...

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

- Figure 3 Showcases of infinite videos generated by Helios. Despite overhead comparable to that of the 1.3B models [59, 60, 90, 100, 126], Helios still excels in visual quality, text alignment, and motion dynamics.

at up to 19.5 FPS on a single H100 GPU—even faster than some 1.3B models. Specifically, (1) For Infinity Generation, we cast long-video generation as infinite video continuation via Unified History Injection, and introduce Representation Control and Guidance Attention to efficiently inject historical context into the noisy context. This design avoids the limitations of causal masking [107] while preserving bidirectional inference, and it unifies T2V, I2V, and V2V within a single architecture. (2) For High-Quality Generation, we identify three canonical manifestations of drifting: position shift, color shift, and restoration shift. Based on this analysis, we propose simple yet effective strategies that explicitly simulate drifting during training, enabling long-video generation without drifting—without self-forcing [34] or error-banks [45]. In addition, we resolve the conflict between the periodic structure of rotary position embeddings (RoPE) [76] and multi-head attention

- [89], eliminating repetitive motion at its source [12]. (3) For Real-time Generation, to remove redundancy in both historical and noisy contexts, we propose Multi-Term Memory Patchification and Pyramid Unified Predictor Corrector that substantially reduce the number of tokens fed into the DiT. We further reformulate flow matching from a “full-resolution noise to full-resolution data” trajectory to multiple “low-resolution noise to multi-resolution data” trajectories, reducing compute to a level comparable to—or even lower than—that of image diffusion models [5, 21, 42, 47, 50, 94]. We introduce Adversarial Hierarchical Distillation, a purely

teacher-forced approach that uses only the autoregressive model as the teacher, reducing the number of sampling steps from 50 to 3. Together with infrastructure-level optimizations for memory efficiency and throughput, these advances push the system toward real-time video generation. To the best of our knowledge, Helios is the first 14B video generation model to reach 19.5 FPS on a single H100 GPU, delivering a 128× speedup while maintaining comparable quality. Finally, to address the lack of a comprehensive open-source benchmark for real-time long-video generation, we construct HeliosBench, which comprises 240 prompts spanning four duration regimes: very short (81 frames), short (240 frames), medium (720 frames), and long (1440 frames). Showcases, along with some benchmark results, are presented in Figures 1, 2, and 3.

Our contributions can be summarized as follows:

- • Without commonly used anti-drifting strategies (e.g., self-forcing, error-banks, keyframe sampling, or inverted sampling), Helios generates minute-scale videos with high quality and strong coherence.

- • Without standard acceleration techniques (e.g., KV-cache, causal masking, sparse/linear attention, TinyVAE, progressive noise schedules, hidden-state caching, or quantization), Helios achieves 19.5 FPS in end-to-end inference for a 14B video generation model on a single H100 GPU.

- • We introduce optimizations that improve both training and inference throughput while reducing memory consumption. These changes enable training a 14B video generation model without parallelism or sharding infrastructure, with batch sizes comparable to image models.

- • To address the lack of standardized benchmarks for real-time long-video generation, we release HeliosBench. Extensive experiments demonstrate that Helios significantly outperforms existing methods in quality while achieving inference speeds that surpass some 1.3B distilled models.

### 2 Related Work

#### 2.1 Long Video Generation

Most video generation models remain limited to short clips (typically 5–10 seconds), and scaling to longer durations without drifting remains challenging. Early methods such as FreeNoise [70] and FIFO-Diffusion [40] use training-free noise rescheduling. Subsequent approaches, including Diffusion Forcing [7] and Rolling Diffusion [71], inject frame-wise independent noise over the full sequence during training to mimic inferencetime context corruption, enabling long-video synthesis via autoregressive diffusion [77]. Later work [8, 84, 86] extends this paradigm to larger models. FramePack [116] trains a next-frame prediction model and introduces inverted sampling to reduce drifting. Self-Forcing [34] adopts causal attention [107] and proposes a train-asinfer rollout strategy to improve quality. Recent advances further explore error-bank mechanisms [28, 45, 69], GPT-like architectures [13, 18, 58], keyframe sampling [33, 96, 124], test-time training [14, 128], and multi-shot generation [6, 29, 37]. Despite this progress, they often exhibit pronounced drifting beyond their training horizon or rely on costly long-video fine-tuning, which limits their practicality for long-video generation.

#### 2.2 Real-Time Video Generation

Long-video generation demands efficient architectures and inference pipelines. For instance, using Wan2.1 14B [90], producing a 5-second video can take roughly 50 minutes on a single NVIDIA A100 GPU to reach acceptable quality. Common acceleration directions include parallelism, distillation [52, 61, 106], linear [9, 82, 101] or sparse attention [46, 99, 114], hidden-state caching [10, 55, 64], and quantization [95, 113, 115]. Existing real-time long-video systems are mostly distillation-based; e.g., [11, 26, 59, 60, 100] follow CausVid [107] and use DMD [105] to reduce sampling steps from 50 to 4, together with Self-Forcing-style rollouts [34] to narrow the train–inference gap. However, these methods are typically built on relatively small backbones (e.g., Wan2.1 1.3B [90]), which limits their ability to model complex motion and preserve high-frequency details. Moreover, although Krea [67] reports 11 FPS on a single NVIDIA B200 GPU, its speed drops to 6.7 FPS on an H100 GPU, and the results suffer from severe drifting, which remains problematic for real-time interactive generation. Additionally, some works claim to be real-time but actually require 8 GPUs [23, 78, 83].

[Figure 37]

DiT Block

Timestep (tHist, tNoisy)

“A majestic bald eagle glides ... ”

Text Token

VAE

Generated Video Chunk

[Figure 38]

[Figure 39]

[Figure 40]

KHist · amp KNoisy

###### Autoregressive Video Diffusion Transformer

VHist QHist

Guidance Self Attention

Pyramid Unified Predictor-Corrector

[Figure 41]

Long Term

Mid Term

Short Term

[Figure 42]

[Figure 43]

QNoisy

VNoisy

[Figure 44]

Text-to-Video

KText

Multi-Term Memory Patchification

RepresentationControl

Pad Pad Pad Pad

QHist

Image-to-Video

VText

Guidance Cross Attention

Pad Pad Pad

QNoisy

Video-to-Video

. . .

/ /

Historical Context

Noisy Context

Can be Padding

Zero Padding

DiT Block

/

- Figure 4 Architecture of Helios. Helios is an autoregressive video diffusion transformer built with Guidance Attention blocks. It reduces overhead by compressing historical and noisy context through Multi-Term Memory Patchification and Pyramid Unified Predictor Corrector, while unifying T2V, I2V, and V2V tasks via Representation Control.

### 3 Helios

- (1) For Infinity Generation, we introduce Unified History Injection to convert a bidirectional pre-trained model

- [90] into an autoregressive generator, enabling text-to-video (T2V), image-to-video (I2V), and video-to-video (V2V) within a unified framework. (2) For High-Quality Generation, we propose Easy Anti-Drifting to mitigate drifting, enabling high-quality minute-scale video generation without inefficient self-forcing [34] or error-banks [45]. (3) For Real-Time Generation, we further propose Deep Compression Flow to reduce both the number of visual tokens and sampling steps, enabling real-time generation on a single GPU with a 14B model.

#### 3.1 Unified History Injection

In this section, we describe how to extend a bidirectional model—originally limited to fixed-length generation—to synthesize videos of unbounded duration. The overall architecture is illustrated in Figure 4.

##### 3.1.1 Representation Control

Prior work typically turns a bidirectional model into an autoregressive generator by combining diffusion forcing [7, 75] with causal masking [107]. However, the resulting frame-wise noise space is extremely large, which slows down optimization and often necessitates step distillation [105, 106, 121]. This approach is undesirable for two reasons: (i) the inference procedure deviates substantially from the pre-trained model, limiting the achievable performance; (ii) distilled models hinder further development within the community.

We address these issues with Representation Control, which formulates long-video generation as video continuation. As shown in Figure 4, the input is the concatenation of a historical context XHist ∈ RB×C×T

Hist×H×W and a noisy context XNoisy ∈ RB×C×T

Noisy×H×W, where B, C, T, H, and W denote the batch size, number of channels, number of frames, height, and width, respectively. We keep THist and TNoisy fixed during both training and inference, with THist ≫ TNoisy. The model denoises XNoisy conditioned on XHist to generate a temporally coherent continuation, enabling the generation of arbitrarily long videos. Representation Control enables automatic task switching via the representation of XHist: if XHist is all zeros, the model performs T2V; if only the last frame is nonzero, it performs I2V; otherwise, it performs V2V.

(a) Position Shift (b) Color Shift

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

DegenerationDegeneration

DegenerationDegeneration

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

(c) Restoration Shift (Noise) (d) Restoration Shift (Blur)

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Figure 5 Visualization of three representative drifting patterns in long-video generation.

##### 3.1.2 Guidance Attention

The historical and noisy contexts exhibit different statistics and should therefore be treated differently. The historical context contains clean content that is already aligned with the text prompt; it should not be denoised and should remain insensitive to XNoisy. Instead, its role is to guide the denoising of XNoisy. We explicitly enforce this separation in two ways. First, we fix the timestep of XHist to 0 throughout the denoising process, indicating that it remains clean and noise-free. Second, inspired by [97, 123], we introduce Guidance Attention to strengthen the influence of the historical context on the generation of future frames:

In the self-attention layer, we compute the query, key, and value tensors for the noisy and historical contexts, denoted by QNoisy,KNoisy,VNoisy and QHist,KHist,VHist, respectively. To retain informative history while suppressing redundant or harmful signals, we introduce head-wise amplification tokens amp to modulate the historical keys. This design selectively amplifies or attenuates historical information per attention head, encouraging the model to focus on the most discriminative components:

XSelf = Attention([QNoisy,QHist],[KNoisy,KHist · amp],[VNoisy,VHist]) (1)

where [,] denotes concatenation, · means multiplication. In cross-attention, we inject semantic information from the text prompt into the model. Since XHist has already incorporated the semantics from previous steps, re-injecting the same semantics is redundant. We therefore apply cross-attention only to XNoisy:

XCross = Attention(QNoisy,KText,VText) (2) where KText and VText are the key and value tensors of the encoded text prompt.

#### 3.2 Easy Anti-Drifting

In this section, we summarize three common manifestations of drifting, as shown in Figure 5, and present simple yet effective techniques to mitigate both drifting and repetitive motion in long-video generation, without relying on self-forcing [34], error-banks [45], or other commonly used anti-drifting strategies.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

(a) Saturation Over Time (b) Aesthetic Over Time (c) RGB Mean Over Time (d) RGB Std Over Time

- Figure 6 Temporal trends of saturation, aesthetic, and RGB statistics for normal videos versus drifting videos. Normal videos are stable, while drifting videos initially follow a similar trajectory but shift abruptly and remain unstable.

- 3.2.1 Relative RoPE A major source of drifting is positional encoding, which we term Position Shift. In practice, diffusion models often perform best when the inference horizon matches the training horizon; changing the video length exposes the model to unseen temporal positions and can substantially degrade quality. Existing long-video methods typically use absolute RoPE along the time dimension. For instance, generating a 1440-frame video uses indices 0:1399, whereas training is often limited to short clips (e.g., 5 seconds), making drifting beyond the training horizon likely even with sophisticated mitigation. Training on longer videos is a direct but costly remedy [11, 59, 100]. Moreover, absolute temporal indices may cause the generation to repeatedly snap back to early positions, leading to abrupt scene resets and cyclic patterns, which we refer to as repetitive motion [12]. To address these issues, we propose Relative RoPE. Regardless of the target video length, we constrain

the temporal index range of XHist to 0:THist and assign XNoisy to THist:THist + TNoisy. This relative indexing enables stable generation at arbitrary lengths while alleviating the interaction between RoPE periodicity and multi-head attention, thereby reducing repetitive motion at its source.

- 3.2.2 First-Frame Anchor Drifting often appears as Color Shift, which becomes more severe as the generated video grows longer. To characterize this phenomenon, we analyze normal and drifting videos by tracking saturation, aesthetic scores [73], and RGB statistics (mean and variance) over time. As shown in Figure 6, normal videos exhibit relatively stable statistics, whereas drifting videos initially follow a similar trajectory but undergo a sharp shift after a certain point and remain unstable thereafter. Notably, drifting rarely occurs at the beginning of generation.

Motivated by this observation, we always retain the first frame in XHist during both training and inference. Serving as a global visual anchor, this frame constrains distribution shifts in later segments, stabilizes statistics over time, and effectively mitigates color shift under autoregressive extrapolation.

- 3.2.3 Frame-Aware Corrupt

Drifting is not limited to color shifts; it can also appear as image-restoration artifacts, such as blur and noise [45]. We refer to this phenomenon as Restoration Shift. This shift arises because the model is trained on clean videos but, at inference time, conditions on its own imperfect outputs as history; consequently, small errors can accumulate and amplify over time. To improve robustness to imperfect history, we propose Frame-Aware Corrupt, inspired by [7, 75], which simulates realistic history drift during training. Concretely, for each historical frame, we independently sample one of the following perturbations: (i) with probability pc, adjust the frame exposure by a magnitude uniformly sampled from [amin,amax]; (ii) with probability pa, add noise with a level uniformly sampled from [bmin,bmax]; (iii) with probability pb, downsample and then upsample using a downsampling factor uniformly sampled from [cmin,cmax]; or (iv) with probability pd, keep the latent clean, where pa + pb + pc + pd = 1. Perturbations are sampled independently per frame, so a history of THist frames yields THist independent corruption decisions, which is crucial for long-video stability.

#### 3.3 Deep Compression Flow - From Token View

In this section, we present a token-centric view of Deep Compression Flow. Our goal is to reduce the token-level computation of a 14B video generation model to a level comparable to that of a 1.3B model.

[Figure 69]

- Figure 7 Overhead reduction with Multi-Term Memory Patchification. A hierarchical history window uses progressively larger kernels, keeping the token budget constant while extending the context length.

##### 3.3.1 Multi-Term Memory Patchification

To enable real-time generation, we reduce redundancy in the historical context XHist via Multi-Term Memory Patchification. Inspired by prior work [25, 116, 125], we leverage a simple observation: in autoregressive video generation, predicting future frames depends mostly on temporally nearby history for local motion and short-range continuity, whereas distant history primarily contributes coarse global context.

Based on this observation, we adopt a hierarchical context window that partitions XHist into three parts—short, mid-, and long-term—containing T1, T2, and T3 frames, respectively, where 0 < T1 < T2 < T3. For each part, we apply an independent Conv kernel (p(ti),p(hi),p(wi)) to compress spatiotemporal tokens, where i ∈ {1,2,3} indexes the three parts. We increase the compression ratio with temporal distance; e.g., p(1)t < p(2)t < p(3)t , p(1)h < p(2)h < p(3)h , and p(1)w < p(2)w < p(3)w . After patchification, the number of tokens becomes:

T1HW p(1)t p(1)h p(1)w

T2HW p(2)t p(2)h p(2)w

Lshort =

, Lmid =

The total number of tokens in XHist is:

T3HW p(3)t p(3)h p(3)w

. (3)

, Llong =

T1 p(1)t p(1)h p(1)w

T2 p(2)t p(2)h p(2)w

T3 p(3)t p(3)h p(3)w

. (4)

Ltotal = HW

+

+

As illustrated in Figure 7, this design keeps Ltotal constant regardless of the target video length. Consequently, the model can retain substantially longer history under a fixed token budget, reducing both computational cost and memory footprint during training and inference. During training, we randomly zero out a certain proportion of the historical context to simulate T2V, I2V, and V2V during inference.

##### 3.3.2 Pyramid Unified Predictor Corrector

To reduce redundancy in the noisy context XNoisy, we propose Pyramid Unified Predictor Corrector, a multi-scale variant of the Unified Predictor Corrector (UniPC) sampler [119], as shown in Figure 8. Inspired by prior works [24, 38, 58, 87, 88], we observe that early sampling steps are dominated by strong noise and thus mainly determine global structure (e.g., layout and color), whereas later steps primarily refine fine-grained details (e.g., edges and textures). Accordingly, we adopt a coarse-to-fine schedule: we sample in a low-resolution latent space at early stages and progressively transition to full resolution. Concretely, Helios learn multi-scale velocity fields that define an ODE-based generative process. Starting from low-resolution Gaussian noise ϵ ∈ RB×C×T×h×w, we integrate the ODE to obtain a coarse-to-fine trajectory and progressively upsample it to get the full-resolution clean sample x0 ∈ RB×C×T×H×W, where h ≪ H and w ≪ W.

Training. We partition the generative process into K stages with increasing spatial resolutions, where stage k operates at resolution (hk,wk). To learn a direct transport direction from scale k−1 to scale k, we construct a linear interpolation path that serves as a continuous transition between the two scales:

xkt = (1 − λt)xk + λt Up(xk−1), (5)

σ = 0.99 0 0.99 0 0.99 0 0.99 σ = 0

stage change

stage change

stage change

###### ...

t = 999 T1

T2 T3 T4 T5 Tn t = 0

[Figure 70]

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

Upsample& ...

Upsample & Renoise

Upsample & Renoise

[Figure 81]

[Figure 82]

[Figure 83]

Renoise

Low Resolution Stage: Efficiency-Focused Denoising

Mid Resolution Stage: Balance-Focused Denoising High Resolution Stage: Quality-Focused Denoising

- Figure 8 Outline of Pyramid Unified Predictor Corrector. The process consists of three stages: (i) Low-stage focuses on efficiency, (ii) Mid-stage balances quality and efficiency, and (iii) High-stage prioritizes quality.

where k ∈ {1,2,...,K} and λt ∈ [0,1] controls the noise level. We use the same λt schedule across stages to keep flow matching consistent across scales. The timestep T ∈ [0,1000] associated with λt is partitioned into stage boundaries T0 = 1000 > T1 > ··· > TK = 0, so that stage k operates only on [Tk, Tk−1]. For the boundary conditions, when k = 1 we start from noise, i.e., Up(xk−1) = ϵ with ϵ ∼ N(0,I); when k = K, we recover the full-resolution sample, i.e., xk = x0. Along the linear path, the ground-truth velocity is constant:

vk = xk − Up(xk−1). (6)

We parameterize the velocity field as ukθ(·) and minimize the velocity-matching objective:

t,xkt,Up(xk−1),y ukθ(xkt ,y,λt,k) − vk 22 , (7) where y denotes the conditioning input. In practice, we set K = 3 to balance quality and efficiency. Inference. We similarly partition sampling into K stages and allocate (N1,N2,...,NK) steps to each stage, resulting in total steps N = Kk=1 Nk. At stage k, we sample at discrete timesteps {tnk}N

L = Ek,λ

n=0 and update:

k

, y, tnk−1 tnk − tnk−1 . (8)

xktn

= xktn−1

+ ukθ xktn−1

k

k

k

When transitioning from stage k−1 to k, naively upsampling the terminal state may introduce artifacts and break path continuity. Following PyramidFlow [38], we upsample the terminal state using nearest-neighbor interpolation and then correct the injected noise and its covariance to maintain distributional consistency across scales. From a computational perspective, single-scale inference with N steps costs O(HWN). In contrast, multi-scale sampling distributes steps across stages and processes fewer tokens in early stages. Under a standard pyramid (e.g., halving the resolution at each stage), the total number of processed tokens is:

H 2 ×

W 2

H × W +

H 4 ×

W 4

+

H 2K−1 ×

W 2K−1 ×

+ ··· +

N K

. (9)

Finally, UniPC [119] reuses predictions from previous steps to correct the current update. However, since prediction tensors change shape across different stages, cached predictions cannot be reused across transitions. We therefore reset the state buffer at each stage transition and re-accumulate the required state within the new stage; empirically, this preserves sampling stability while avoiding cross-scale correction artifacts.

#### 3.4 Deep Compression Flow - From Step View

##### 3.4.1 Problem Formulation

Step distillation is crucial for building real-time generative models. Among existing approaches [52, 61, 72, 121], Distribution Matching Distillation (DMD) [105] is widely adopted and well established. In DMD, we first sample noise ϵ and feed it to a few-step generator Gθ. Using x0 prediction and backward simulation, the

[Figure 84]

[Figure 85]

[Figure 86]

Dynamic Re-noise

[Figure 87]

[Figure 88]

Real-Score Estimator

[Figure 89]

[Figure 90]

[Figure 91]

Dynamic Re-noise

[Figure 92]

[Figure 93]

Few-Step Generator with EMA Update

[Figure 94]

[Figure 95]

[Figure 96]

Frame-Aware Corrupt

[Figure 97]

[Figure 98]

Fake-Score Estimator with Multi Heads

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

- Figure 9 Pipeline of Adversarial Hierarchical Distillation. The framework is based on DMD [105], with improvements such as Pure Teacher Forcing, Staged Backward Simulation, Coarse-to-Fine Learning and Adversarial Post-Training.

generator produces a clean sample x0. We then sample a noise level λτ ∼ U[0,1] and perturb x0 to obtain a noisy sample xτ. Next, we evaluate xτ with a real-score estimator preal and a fake-score estimator pfake, yielding scores sreal and sfake. The real score is computed via classifier-free guidance by combining conditional and unconditional predictions, i.e., CFG(scondreal ,suncondreal ), whereas the fake score uses only the conditional branch, i.e., scondfake . Their difference defines the distribution-matching gradient used to update Gθ. In addition, we train pfake with a flow-matching loss LFlow on xτ to improve stability. However, Helios changes the sampling procedure, so the standard pipeline is not directly applicable. We therefore propose Adversarial Hierarchical Distillation, a DMD-based framework (Figure 9) with the following improvements.

##### 3.4.2 Adversarial Hierarchical Distillation

Pure Teacher Forcing with Autoregressive Teacher. Existing approaches [11, 34, 59, 60, 100] that apply DMD to real-time long-video generation typically discard real data entirely during training. For instance, Self-Forcing [34] explicitly integrates the inference procedure of an autoregressive model into the training process: when generating the current section, previously generated sections are used as conditions to reduce the training–inference gap and mitigate drifting in video generation. However, we observe that the robustness of such methods against drifting is strongly dependent on the number of sections rolled out during training. Specifically, when training involves rollout of only five sections, the model frequently exhibits severe exposure bias during inference once the generated sequence exceeds this length. Motivated by this limitation, subsequent studies adopt a long self-rollout strategy [11, 59, 100], in which a large number of sections—corresponding to video durations of tens of seconds or even several minutes—are generated during training to enhance long-term stability. Nevertheless, this approach incurs substantial computational overhead, restricting existing methods to models with approximately 1.3B parameters [90]. To overcome this limitation, we employ real data exclusively as historical context during the distillation stage and require the generation of only a single section per training step, substantially improving training efficiency. Moreover, by incorporating the Easy Anti-Drifting mechanism proposed in Section 3.2, we achieve long-video anti-drifting performance comparable to that of long self-rollout strategies, without the need to perform such expensive rollouts. More importantly, we select Helios-Base1 as the teacher model because it is already capable of generating high-quality long videos, whereas existing methods typically rely on Wan [90], which is limited to synthesizing short videos.

Staged Backward Simulation. DMD performs backward simulation on a single flow trajectory to recover x0.

In contrast, we introduce Staged Backward Simulation, which decomposes the backward simulation into K stages, producing intermediate estimates {xk0}Kk=1; the final-stage output xK0 is used as x0. At stage k, given the current state xkt and the predicted velocity field ukθ(xkt ,y,λt,k), we estimate the terminal state as

xk0 = xkt − λt · ukθ(xkt ,y,λt,k), (10)

1We describe the details of Helios-Base in Section 5.1.

where the update follows directly from the linear interpolation path in Eq. 5. We then reconstruct xkt using Eq. 5 and re-estimate xk0 via Eq. 10, repeating this procedure until stage k converges. The resulting estimate xk0 initializes stage (k + 1). After K stages, we obtain x0 = xK0 .

Coarse-to-Fine Learning. Compared with DMD, Helios propagates gradients through K stages and multiple flow trajectories, which increases optimization difficulty and may slow convergence, especially early in training. We therefore adopt three curriculum-style strategies that progressively increase task difficulty. (1) Staged ODE Init. We construct a compact dataset of ODE solution pairs generated by Helios-Mid, which is then used for initialization following [107]. In contrast to prior work, our initialization is performed across K stages. At each stage, only a single section needs to be generated rather than multiple sections, and an autoregressive teacher is employed to guide the process. (2) Dynamic Re-noise. Uniformly sampling noise levels, as in standard DMD, is suboptimal in the hierarchical setting because different noise regimes contribute differently across training stages. Inspired by [5, 54], we sample timesteps from a Beta distribution whose parameters follow a cosine decay schedule: it concentrates on high-noise timesteps early to learn coarse structure and becomes increasingly uniform later to emphasize medium- and low-noise timesteps for detail refinement.

Adversarial Post-Training. DMD distills a multi-step teacher into a few-step student by matching the teacher-defined distribution; consequently, the student inherits the teacher’s biases and is bounded by the teacher’s expressive capacity. To relax this limitation, inspired by Spark-Wan [48] and DMD2 [106], we augment distillation with an additional GAN objective trained on real data. This auxiliary objective provides teacher-independent supervision and can further improve sample quality.

Concretely, we add multi-granularity classification branches D to pfake and distribute them across DiT layers. We train these branches with the non-saturated GAN objective:

LD = E log D(xrealτ ,τ) + E −log D(xKτ ,τ) . (11) To stabilize discriminator training, we incorporate an approximate R1 regularizer (following APT [53]):

LaR1 = D(xrealτ ,τ) − D(N(xrealτ ,σDI),τ) 22 . (12) Finally, the full adversarial objectives are defined as:

LD = E log D(xrealτ ,τ) + E −log D(xKτ ,τ) + λD · E D(xrealτ ,τ) − D(N(xrealτ ,σDI),τ) 22 , (13)

LG = E log D(xKτ ,τ) . (14) In practice, we set λD = 100 and σD = 0.1. To reduce memory usage, we feed the discriminator a random crop of size H′ × W′ from xKτ instead of the full-resolution sample, where H′ = H2 and W′ = W2 . Other Details. We initialize the few-step generator from Helios-Mid and inherit both the real-score and fake-score estimators from Helios-Base, which provides stable, high-fidelity supervision during training. Based on this design, the objectives for Adversarial Hierarchical Distillation are:

= LDMD + wG · LG, (15) Lpfake

LGθ

= LFlow + wD · LD. (16)

Here, we set wG = 5e − 2 and wD = 1e − 2 as the weight coefficients for the respective losses. We follow CausVid [107] and update Gθ only once every five updates of pfake.

- 3.5 Other Techniques In this section, we present several inference techniques that are training-free and parameter-free.

Adaptive Sampling. Figure 6 shows that drifting is accompanied by pronounced shifts in RGB statistics (mean and variance). Since the latent space is a compressed representation of RGB space, analogous distribution shifts also appear in latent statistics. This motivates an adaptive anti-drifting strategy. Let the RGB mean

and variance of the t-th generated latent section be µt and σt2. During inference, we maintain global statistics (global mean and variance), denoted by µ¯t and σ¯t2, updated via an exponential moving average (EMA):

µ¯t = ρµµ¯t−1 + (1 − ρµ)µt, (17) σ¯t2 = ρσσ¯t2−1 + (1 − ρσ)σt2, (18)

where ρµ,ρσ ∈ (0,1) are smoothing coefficients. When the statistics of the current latent section deviate from the global statistics beyond preset thresholds δµ and δσ:

∥µt − µ¯t∥2 > δµ and σt2 − σ¯t2 2 > δσ, (19)

we treat the current section as exhibiting significant drift. When generating the next section, we apply Frame-Aware Corrupt to the historical context, perturbing the drifting frames in a targeted, training-free manner. This implicitly reduces the model’s reliance on the biased history and encourages it to rely more on its intrinsic generative prior, thereby improving long-video quality and stability.

Interactive Interpolation. Long-video generation enables interactive editing, where users can revise the prompt on the fly, requiring the model to adapt rapidly without introducing temporal artifacts. A naive solution is to switch to the new prompt embedding abruptly; however, this induces an instantaneous conditional shift and often causes visible discontinuities (e.g., flicker or sudden semantic jumps) around the editing boundary. Following Krea [67], we instead adopt prompt interpolation, which gradually transitions the conditioning from the current to the target prompt over multiple steps. This yields a smoother handover between conditions and improves perceived temporal coherence. Specifically, let the current prompt embedding be e(1) ∈ Rℓ

Text×D and the target embedding be e(2) ∈ Rℓ

Text×D, where ℓText is the text length and D is the hidden dimension. We construct M intermediate conditions {e[j]}Mj=0−1 by linear interpolation:

j M − 1

e[j] = (1 − λj)e(1) + λje(2), λj =

, j = 0,1,...,M − 1, (20)

where λj ∈ [0,1] increases linearly with j, so that e[0] = e(1) and e[M−1] = e(2). During generation, we feed these embeddings sequentially to gradually move the conditioning from e(1) to e(2) over M steps. This gradual transition mitigates visual and semantic discontinuities caused by abrupt conditioning changes.

### 4 Infrastructure

#### 4.1 Workload Analysis

Scaling DiT-based video generators to 14B parameters yields prohibitive compute and memory costs, even with batch size 1. The primary bottleneck is the quadratic complexity of 3D attention over temporal and spatial tokens. In practice, training such models on a single GPU typically requires extensive parallelism (e.g., CP, TP, SP) and parameter/activation sharding (e.g., FSDP, DeepSpeed).

In contrast, Helios introduces Deep Compression Flow to compress both the historical and noisy contexts, enabling full forward and backward passes on a single GPU for the first two training stages, without parallelism

or sharding. In Multi-Term Memory Patchification, we set (p(1)t ,p(2)t ,p(3)t ) = (4,2,1), (p(1)h ,p(2)h ,p(3)h ) = (8,4,2), and (p(1)w ,p(2)w ,p(3)w ) = (8,4,2), with (T1,T2,T3) = (16,2,2). This reduces the historical-context token count from 5HW to 58HW (approximately 8×). With Pyramid Unified Predictor Corrector (K = 3), the noisycontext token count decreases from NHW to 167 NHW (approximately 2.29×).

For a standard DiT, the per-layer complexity is approximately O(αBℓD2 + βBℓ2D), where α and β are the costs of linear layers and attention, respectively, and ℓ is the sequence length (the ℓ2 term dominates self-attention). The overall complexity scales as O(L(αBℓD2 + βBℓ2D)), while activation memory scales as O(γLBℓD), where γ depends on the implementation. Therefore, the 8× and 2.29× token reductions translate to roughly 64× and 5.2× reductions in attention FLOPs for the historical and noisy contexts, respectively, and they linearly reduce activation and intermediate-state memory.

[Figure 105]

[Figure 106]

Update Weight

Few-Step Staged-Generator Cache Grad

Fake-Score Estimator with Multi Gan Heads

[Figure 107]

Free Memory

###### GAN - Generator Stage

[Figure 108]

Forward

Cache Grad Free Memory

Fake-Score Estimator Flow Grad with Multi Gan Heads

[Figure 109]

Backward

Update Weight

[Figure 110]

[Figure 111]

Trainable

Cache Grad Free Memory

Fake-Score Estimator Gan Grad with Multi Gan Heads

[Figure 112]

[Figure 113]

Frozen

GAN - Discriminator Stage

- Figure 10 Execution of Cache Grad for GAN. We cache discriminator gradients w.r.t. inputs to decouple backpropagation and free intermediate activations early, substantially reducing peak memory.

#### 4.2 Memory Optimization

In the first two training stages, the GPU needs to load only three components: the VAE, the text encoder, and the DiT. By offloading VAE latents and text embeddings to disk, the GPU effectively retains only the DiT, enabling single-GPU batch sizes comparable to those used for image diffusion models.

In the third stage, the memory demand increases substantially: the GPU must host four 14B models (the few-step generator, real-score estimator, fake-score estimator, and EMA model), as well as multiple GAN heads. Under an 80GB memory budget, this configuration can exceed capacity even for inference, and training further increases memory usage due to activations and intermediate states. We therefore adopt the following strategies to enable training under strict memory constraints.

Sharded EMA. Exponential moving average (EMA) stabilizes training by smoothing parameter updates and is typically stored in FP32 for numerical robustness. A naive implementation replicates the full FP32 EMA copy on every GPU, which incurs substantial memory overhead. Following OpenSora-Plan [49], we instead shard the EMA parameters across GPUs using ZeRO-3, so that each device stores only a fraction of the EMA states. For a 14B-parameter model sharded over Z GPUs, each device stores approximately 14×4

Z GiB of EMA parameters. This removes redundant replicas and improves memory efficiency.

Asynchronous VRAM Freeing. In stage-3 training, we sequentially execute multiple large models: we feed noise zt to the few-step staged generator to obtain xstaged0 , and then re-noise and evaluate the sample with the real-score estimator preal and fake-score estimator pfake to compute LDMD, LGAN, and LFlow. Because these computations are serialized, only one model needs to reside on the GPU at a time during the forward pass.

Moreover, under the two time-scale update rule (TTUR) [105], each iteration updates either the fake-score estimator pfake (via LFlow and LGAN) or the few-step generator Gθ (via LDMD). We exploit this structure to asynchronously offload unused models to host memory, limiting peak VRAM to roughly that of training a single 14B model. With pinned host memory, non-blocking transfers, and careful CPU–GPU scheduling, we maintain throughput close to GPU-only execution despite frequent transfers.

Cache Grad for GAN. (1) Updating the generator. Standard autodiff requires keeping all activations of the few-step generator Gθ and the fake-score estimator pfake until backpropagation finishes, which can make the subsequent computation of LDMD infeasible under memory limits. We therefore decouple the fake-score estimator pfake from the default backward pass by caching the discriminator gradient with respect to its input

###### Table 1 The detailed training hyperparameters of Stage-1 and Stage-2.

Configuration Stage-1-init Stage-1-post Stage-2-init Stage-2-post

Global Batch Size 128 128 256 192 Optimizer

AdamW, β1 = 0.9, β2 = 0.999 ϵ = 1e − 08, weight_decay=1e-04 Learning Rate 5e-5 3e-5 1e-4 3e-5 Learning Rate Schedule Constant Constant Constant Constant Training Steps 5.5k 7.5k 16k 20k Gradient Clipping 1.0 1.0 1.0 1.0 LoRA Rank 128 128 256 256 LoRA Alpha 128 128 256 256 Numerical Precision BFloat16 BFloat16 BFloat16 BFloat16 GPU Usage 64 NVIDIA H100 64 NVIDIA H100 64 NVIDIA H100 64 NVIDIA H100 History Corrupt Ratio pa = 0.0,pb = 0.8,pc = 0.1,pd = 0.1;bmin = 0,bmax = 0.33,cmin = 0,cmax = 0.1

AdamW, β1 = 0.9, β2 = 0.999 ϵ = 1e − 08, weight_decay=1e-04

AdamW, β1 = 0.9, β2 = 0.999 ϵ = 1e − 08, weight_decay=1e-04

AdamW, β1 = 0.9, β2 = 0.999 ϵ = 1e − 08, weight_decay=1e-04

###### Table 2 The detailed training hyperparameters of Stage-3.

Configuration Stage-3-ode Stage-3-post

Global Batch Size 128 128 Real-score initialization - Helios-Base Fake-score initialization - Helios-Base Real-score CFG weight - 3.0

AdamW, β1 = 0.0, β2 = 0.999 ϵ = 1e − 08, weight_decay=1e-03 Learning Rate (Gθ) 2.0e-6 2.0e-6 Learning Rate (pfake) - 4.0e-7 Learning Rate Schedule (Gθ & pfake) Constant Constant Learning Rate Warmup Step - -

AdamW, β1 = 0.0, β2 = 0.999 ϵ = 1e − 08, weight_decay=1e-03

Optimizer (Gθ & pfake)

Gradient Clipping (Gθ & pfake) 10.0 10.0 LoRA Rank (Gθ) 256 256 LoRA Rank (pfake) - 256 LoRA Alpha (Gθ) 256 256 LoRA Alpha (pfake) - 256 TTUR - 5 GAN Head Layers - 5, 15, 25, 35, 39 GAN Head Dim - 768 GAN Start Step - 1000 EMA Decay 0.99 0.99 EMA Start Step 250 750 Training Steps 3759 2250 Numerical Precision BFloat16 BFloat16 GPU Usage 128 NVIDIA H100 128 NVIDIA H100 History Corrupt Ratio pa = 0.4,pb = 0.4,pc = 0.0,pd = 0.2;amin = 0.3,amax = 1.7,bmin = 0,bmax = 0.33

during the forward pass. Specifically, we immediately free the estimator’s intermediate activations after its forward pass and reuse the cached input gradients during backpropagation, avoiding the need to retain the full computation graph. This reduces peak memory to that of a single 14B model.

- (2) Updating the fake-score estimator. We combine gradient accumulation with batched execution. We first compute LFlow in a separate forward/backward pass, accumulate its gradients into the discriminator, and immediately release its activations. We then concatenate the real, fake, and perturbed samples (for LaR1) into one batch and run a single forward/backward pass to compute the remaining loss terms. Compared with jointly computing all losses, this scheduling substantially reduces peak memory.

#### 4.3 Efficiency Optimization

To further accelerate training and inference, we replace multiple native PyTorch operations with custom implementations spanning both forward and backward propagation, thereby improving computational efficiency.

Flash Normalization. We implement kernel fusion for LayerNorm and RMSNorm using Triton, following [15, 32]. By consolidating mean and variance computation, normalization, and affine transformations into

a single kernel, we minimize memory traffic and leverage optimized primitives such as tl.math.rsqrt. To reduce memory footprint, we cache only scalar statistics (row-wise inv_var ∈ RB×ℓ and µ ∈ RB×ℓ) for the backward pass, avoiding storage of the full normalized tensor z ∈ RB×ℓ×D. This approach reduces the memory complexity of intermediate activations from O(BℓD) to O(Bℓ). Furthermore, we adopt a mixed-precision strategy where internal computations are performed in FP32 for numerical stability, while inputs and outputs retain their original data types (e.g., bfloat16). Finally, we maximize GPU bandwidth utilization through row-wise parallelism—mapping one program instance per token—and coalesced memory access patterns.

Flash Rotary Position Embedding. We implement kernel fusion optimization for Rotary Positional Embeddings (RoPE) using Triton. By consolidating complex number decomposition, rotation matrix multiplication, and result reconstruction into a single GPU kernel, we eliminate the memory fragmentation and data copying overheads inherent in PyTorch’s native unflattening, chunking, and indexing operations. Specifically, we flatten the input x ∈ RB×ℓ×H×D to R(B·ℓ·H)×D. We parallelize execution by mapping one program instance per attention head, using interleaved memory access to directly retrieve real and imaginary components. The rotation is applied using pre-computed cos and sin values—outreal = xreal · cos−ximag · sin and outimag = xreal · sin+ximag · cos—with results written back in-place. For backward propagation, we reuse the forward kernel to perform inverse rotation by simply negating the sine component (sinneg = −sin). This strategy obviates the need to store full intermediate tensors, requiring only cos ∈ RB×ℓ×(D/2) and sin ∈ RB×ℓ×(D/2). Consequently, we reduce the memory complexity of intermediate activations from O(BℓHD) to O(BℓD), where B, ℓ, H, and D denote batch size, sequence length, head count, and head dimension, respectively.

- 4.4 Other Techniques.

By eliminating the need for causal masking, Helios seamlessly integrates with high-efficiency attention backends such as FlashAttention [16, 17], thereby achieving superior throughput and reduced latency.

- 5 Experiments

- 5.1 Implementation Details

Training. We initialize from Wan-2.1-T2V-14B [90] and train on 0.8M clips of duration < 10 seconds using a three-stage progressive pipeline. Stage 1 (Base) performs architectural adaptation: we apply Unified History Injection, Easy Anti-Drifting, and Multi-Term Memory Patchification to convert the bidirectional pretrained model into an autoregressive generator. Stage 2 (Mid) targets token compression by introducing Pyramid Unified Predictor Corrector, which aggressively reduces the number of noisy tokens and thus the overall computation. Stage 3 (Distilled) applies Adversarial Hierarchical Distillation, reducing the sampling steps from 50 to 3 and eliminating the need for classifier-free guidance (CFG). Throughout training, we apply dynamic shifting to all timestep-dependent operations to match the noise schedule to the latent size. We cap the resolution at 384 × 640 and extract 109-frame clips from each video. More details are in Tables 1 and 2.

Inference. For Stages 1–2, we adopt UniPC [119] scheduler with 50 sampling steps, a classifier-free guidance (CFG) scale of 5.0, and v-prediction. For Stage 2, instead of standard CFG, we employ CFG-Zero-Star [22]. For Stage 3, we use x0-prediction with 3 sampling steps and a CFG scale of 1.0.

HeliosBench. Because no open-source benchmark targets real-time long-video generation, we build HeliosBench, a test set of 240 LLM-refined prompts from Self-Forcing [34]. We evaluate four duration tiers: very short (81 frames), short (240 frames), medium (720 frames), and long (1440 frames). For automated evaluation, existing benchmarks are only weakly aligned with human preference [35, 36, 109, 120]; however, they remain the best available options. Following [35, 109, 110], we report five dimensions: (1) Aesthetic, measured by the LAION aesthetic predictor [73]; (2) Dynamic, computed using the Farnebäck algorithm [110]; (3) Motion Smoothness, measured by RAFT [85]; (4) Semantic, measured by ViCLIP [92] for video–text alignment; and (5) Naturalness, measured by OpenS2V-Eval [110]. We additionally follow [116] to quantify drifting on Aesthetic, Motion Smoothness, Semantic, and Naturalness. Since these metrics are noisy and their raw scores may correlate poorly with human perception, we map each metric to a 10-point scale using its empirical score distribution for improved robustness. To measure throughput (FPS), we report end-to-end speed at 384 × 640 under default frame lengths, including the latency of both the VAE and the text encoder. For each model, we

- Table 3 Quantitative comparisons on 81-frames short videos. Helios achieves a superior speed-quality trade-off over existing methods, which are either computationally prohibitive or yield suboptimal results. “↑” higher is better.

Throughput (FPS) ↑

Total ↑

Aesthetic ↑

Dynamic ↑

Smoothness ↑

Semantic ↑

Naturalness ↑

Model #Params

Bidirectional Models

SANA Video [9] 2B 1.34 4.60 9 7 9 5 1 CogVideoX [103] 2B 1.95 5.55 7 10 7 4 5

CogVideoX 1.5 [103] 5B 1.47 5.15 7 5 8 5 4

Mochi-1 [80] 10B 0.53 5.35 6 7 9 5 4 HV Video [41] 13B 0.36 6.00 8 8 9 5 5

HV Video 1.5 [93] 8.3B 0.24 6.90 8 10 9 5 7

- Wan 2.1 1.3B [90] 1.3B 1.60 6.10 8 10 8 6 4

- Wan 2.2 5B [90] 5B 3.32 6.05 7 7 8 5 6

- Wan 2.1 14B [90] 14B 0.33 6.15 8 6 9 6 5

- Wan 2.2 14B [90] 14B 0.33 6.35 8 8 9 5 6 LTX Video [30] 1.9B 15.03 4.45 6 4 10 1 6

LTX Video 2 [31] 19B 3.84 6.00 8 8 9 6 4 Kandinsky 5 lite [1] 2B 1.03 6.25 8 6 10 5 6 Kandinsky 5 pro [1] 19B 0.49 7.35 8 10 10 6 7 StepVideo T2V [63] 30B 0.27 4.00 6 4 9 1 5

FastVideoWan 2.1 [117] 14B 5.37 5.25 9 3 9 5 4 TurboDiffusion 2.1 [115] 14B 10.15 5.10 8 6 9 4 4

TurboDiffusion 2.1-Quant [115] 14B 8.48 5.45 8 6 9 4 5 Autoregressive Models

NOVA [18] 0.6B 1.18 3.70 5 2 9 4 2 Pyramid Flow [38] 2B 3.11 4.55 8 3 10 4 3

MAGI-1 [86] 4.5B 0.37 5.25 8 3 10 6 3

InfinityStar [58] 8B 3.38 5.30 8 8 9 6 2 SkyReelsV2-DF [8] 1.3B 0.55 5.65 8 8 9 5 4 SkyReelsV2-DF [8] 14B 0.12 5.55 8 7 9 5 4

CausVid [107] 1.3B 24.41 4.50 8 7 9 4 2 Self Forcing [34] 1.3B 21.20 5.75 8 9 9 5 4

Rolling Forcing [59] 1.3B 19.47 5.25 8 4 9 5 4

LongLive [100] 1.3B 18.05 5.80 8 5 10 5 5 Infinite Forcing [39] 1.3B 22.19 5.10 8 6 9 4 4 Reward Forcing [60] 1.3B 22.13 5.55 8 7 9 5 4 Causal Forcing [126] 1.3B 20.98 5.40 8 10 8 5 3

Dummy Forcing Long [26] 1.3B 20.10 5.45 9 4 10 5 4 SANA Video Long [9] 2B 13.24 3.85 9 2 10 4 1

Krea [67] 14B 6.74 5.95 9 10 9 5 4 Autoregressive Video Continuation Models

LongCat-Video [84] 13.6B 0.33 6.30 9 10 9 6 4 Helios-Base 14B 0.54 6.35 8 8 9 5 6

Helios-Mid 14B 1.05 6.25 8 7 9 5 6 Helios-Distilled 14B 19.53 6.00 8 7 10 5 5

enable its officially supported acceleration techniques (e.g., FlashAttention, torch.compile, KV-cache, and warm-up) to achieve the best possible throughput. More details are provided in the Appendix A.

For baselines, we compare Helios with a broad set of open-source video generation models, including (1) base models: SANA Video [9], CogVideoX [103], Mochi [80], HV Video [41, 93], Wan [90], LTX Video [30, 31], Kandinsky [1], StepVideo [63], NOVA [18], Pyramid Flow [38], MAGI [86], InfinityStar [58], SkyReelsV2

- [8], and LongCat-Video [84]; and (2) distilled models: FastVideo [117], TurboDiffusion [115], CausVid [107], Self-Forcing [34], Rolling Forcing [59], LongLive [100], Infinite Forcing [39], Reward Forcing [60], Causal Forcing [126], Dummy Forcing [26], SANA Video Long [9], and Krea [67]. For models that only support short-video generation, we construct a subset by matching the 240 prompts to each model’s default length. For a fair comparison, we truncate outputs from long-video models to the first 81 frames.

A dramatic skydiving scene in a realistic photographic style, capturing a skydiver accelerating during free fall. The skydiver, a young man with a determined expression, is mid-air with arms outstretched and legs extended. His body is in dynamic motion, creating a sense of speed and ...

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

###### Helios

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

FastVideo Wan2.1

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

TurboDiffusion Wan2.1

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

StepVideo-T2V

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Kandinsky 5 pro

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Kandinsky 5 lite

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

LTX Video 2

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

LTX Video 1.9B

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

- CogVideoX 1.5
- CogVideoX 2B

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

- Figure 11 Qualitative comparisons on 81-frames short videos (Part-1). Even as a distilled model, Helios matches or even surpasses the base models in terms of visual quality, motion dynamics, and naturalness.

#### 5.2 Qualitative and Quantitative Comparison

Short Video Generation. First, we benchmark the ability of various models to generate 81-frame ultra-short video clips using Aesthetic, Dynamic, Smoothness, Semantic, and Naturalness scores, as well as their weighted sum. As shown in Table 3, our method achieves an overall score of 6.00, surpassing all distilled models and matching the performance of most base models of the same size. Notably, distilled models tend to produce videos with higher saturation and smaller motion amplitudes, which leads to higher Aesthetic and

A macro shot in realistic style of a man wearing an antique diving helmet with dark glass and a jetpack, standing on a molten lava surface. He strides confidently, his body slightly bent forward, with a determined expression. Behind him, a majestic dragon soars through the sky, its wings spreading ...

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

###### Helios

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

SANA Video

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

- Wan 2.1 14B
- Wan 2.2 14B

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

- Wan 2.1 1.3B

HV Video 1.5

HV Video

- Wan 2.2 5B

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Mochi-1

- Figure 12 Qualitative comparisons on 81-frames short videos (Part-2). Despite being a distilled model, Helios matches or surpasses the base models in visual fidelity, text alignment, and overall realism.

Smoothness scores compared to base models; however, this does not necessarily correlate with superior quality. Consequently, we focus primarily on Semantic and Naturalness. The experimental results indicate that Helios excels in these aspects, either matching or surpassing models such as Wan 14B [90] and HV Video [41, 93], thereby demonstrating strong video generation quality. In addition, Helios achieves a better balance between Dynamic and Motion Smoothness: it avoids the overly static motion patterns often seen in distilled models, while not introducing the temporal jitter or local inconsistency that can appear in aggressive acceleration settings. Furthermore, Helios achieves a real-time generation speed of 19.53 FPS on a single H100 GPU. In contrast, while SANA Video Long [9] is a distilled model and seven times smaller than Helios, its generation speed is 1.28 times slower. Compared with the same-sized FastVideo [117] and TurboDiffusion [115], Helios is

- Table 4 Quantitative comparisons on 120, 240, 720 and 1440-frames long videos. Helios consistently exceeds existing real-time long video generation methods. “↑” denotes higher is better.

Throughput (FPS) ↑

Total ↑

Throughput Score ↑

Total* ↑

Aesthetic ↑

Dynamic ↑

Smoothness ↑

Semantic ↑

Naturalness ↑

Drifting Aesthetic ↑

Drifting Smoothness ↑

Drifting Semantic ↑

Drifting Naturalness ↑

Model #Params

Autoregressive Models

NOVA [18] 0.6B 1.18 2.48 1 2.38 1 1 9 1 2 1 9 1 2 Pyramid Flow [38] 2B 3.11 2.85 1 2.75 6 3 9 3 1 1 8 1 2

MAGI-1 [86] 4.5B 0.37 4.92 1 4.82 7 3 10 6 2 1 10 6 5

InfinityStar [58] 8B 3.38 2.63 2 2.43 4 4 8 2 1 1 9 1 1 SkyReelsV2-DF [8] 1.3B 0.55 3.29 1 3.19 6 6 9 4 1 1 9 2 1 SkyReelsV2-DF [8] 14B 0.12 3.87 1 3.77 6 6 8 5 2 1 8 4 1

CausVid [107] 1.3B 24.41 5.38 8 4.58 8 6 9 3 1 6 10 7 6 Self Forcing [34] 1.3B 21.20 5.00 7 4.30 7 5 9 5 2 1 10 5 3

Rolling Forcing [59] 1.3B 19.47 6.86 7 6.16 8 3 9 5 4 7 10 9 7

LongLive [100] 1.3B 18.05 6.82 6 6.22 8 5 9 5 4 8 10 9 6 Infinite Forcing [39] 1.3B 22.19 6.50 7 5.80 8 6 9 4 4 7 10 9 5 Reward Forcing [60] 1.3B 22.13 6.88 7 6.18 8 7 9 5 4 7 10 9 6 Causal Forcing [126] 1.3B 20.98 3.86 7 3.61 7 10 8 4 1 1 9 3 3

Dummy Forcing Long [26] 1.3B 20.10 6.14 7 5.44 8 4 9 5 3 7 10 8 3 SANA Video Long [9] 2B 13.24 6.03 5 5.53 9 2 10 5 1 7 10 9 8

Krea [67] 14B 6.74 4.10 3 3.80 7 10 9 5 1 1 10 3 1 Autoregressive Video Continuation Models

LongCat-Video [84] 13.6B 0.33 6.54 1 6.44 8 7 9 6 4 7 10 8 7 Helios-Base 14B 0.54 6.57 1 6.47 8 6 9 6 5 7 10 8 5

Helios-Mid 14B 1.05 6.05 1 5.95 7 5 9 5 5 5 10 7 6 Helios-Distilled 14B 19.53 6.94 6 6.34 8 6 10 5 5 7 10 7 7

2–3× faster and outperforms Wan 14B [90] by a factor of 52. These results show that Helios not only advances video generation quality but also leads in generation speed among existing large-scale video generation models. A comprehensive qualitative analysis further supports the effectiveness of our method. Specifically, Figure 11 and Figure 12 illustrate that Helios generates high-quality videos that are more natural and better aligned with human perception than those from distilled models, while remaining comparable to base models.

Long Video Generation. Next, building on Table 3, we introduce Throughput Score and Drifting Score to evaluate long-video generation across different durations. As shown in Table 4, our method achieves a total score of 7.08, outperforming the strongest baseline, Reward Forcing [60] (6.88), while maintaining competitive runtime. In particular, we obtain a higher Naturalness score (6), avoiding the over-saturated appearance commonly observed in distilled models, and we simultaneously improve Dynamic and Motion Smoothness, yielding more vivid yet physically plausible motion. We also observe that Helios exhibits consistently lower drifting across multiple dimensions (Aesthetic, Semantic, and Naturalness), indicating that the model better preserves content identity and scene layout as the video extends to hundreds or even thousands of frames. Moreover, although Helios-Stage1 and Helios-Stage2 do not rely on Self-Forcing [34] or Error-Banks [45], they still effectively mitigate drifting over long horizons, suggesting a complementary route to improve long-term consistency. Figures 13 and 14 corroborate these quantitative results: Helios preserves visual quality over time, whereas baseline methods exhibit noticeable degradation and inconsistencies.

- 5.3 User Study We conduct a side-by-side user study with five representative models for real-time long-video generation

- [9, 26, 59, 60, 100] and five for short-video generation [9, 31, 84, 90, 93]. In each trial, participants view two videos and indicate whether one is better or whether they are comparable. Each questionnaire contains 40 pairwise comparisons, requiring participants to watch 80 clips in total. We randomize both the presentation order and the left–right placement to reduce bias. Each participant completes only one questionnaire to avoid information leakage and improve engagement. We collect 200 valid responses. As shown in Figure 15, Helios consistently outperforms prior methods on both long- and short-video generation.

- 5.4 Ablation Study

We evaluate key components through qualitative and quantitative comparisons. For simplicity, we omit the Throughput score from the quantitative results below.

##### 5.4.1 Impact of Guidance Attention

To study the effect of causal masking on autoregressive generation, we augment Guidance Attention with a causal mask in self-attention, following [34, 107]. This mask prevents the noisy context from interfering

A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her ...

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

###### Helios

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

NOVA

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Pyramid Flow

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

MAGI-1

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

InfinityStar

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

SkyReelsV2 DF- 1.3B

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

SkyReelsV2 DF- 14B

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

CausVid

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Self Forcing

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Rolling Forcing

frame 1 frame 241 frame 481 frame 721 frame 961 frame 1201 frame 1440

- Figure 13 Qualitative comparisons on 120, 240, 720 and 1440-frames long videos (Part-1). It is clear that Helios consistently outperforms the baseline models in terms of realism and naturalness.

- Table 5 Quantitative ablation of key components. Changing any component leads to a significant degradation in quality while exacerbating temporal drifting, thereby impairing Helios ’s ability to generate stable long videos. w Guidance Attention* indicates the additional use of a causal mask for self-attention. w Staged Backward Simulation* means incorporating multi-scale xk0 into real/fake-score estimators during training.

Drifting Naturalness ↑ Helios-Base 6.47 8 6 9 6 5 7 10 8 5

Total ↑

Aesthetic ↑↓

Dynamic ↑↓

Smoothness ↑↓

Semantic ↑

Naturalness ↑

Drifting Aesthetic ↑

Drifting Smoothness ↑

Drifting Semantic ↑

Model

w Guidance Attention* unstable training process

w/o Guidance Attention 6.23 9 4 7 7 5 6 10 8 2 w/o First Frame Anchor 5.51 8 5 8 7 4 3 10 6 2

w/o Frame-Aware Corrupt 4.70 7 4 8 6 4 2 10 3 1

Helios-Distilled 6.34 8 6 10 5 5 7 10 7 7 w Self-Forcing 6.11 9 6 10 4 5 8 10 7 6

w Bidirectional Teacher 4.75 8 6 9 3 4 5 10 4 4 w Staged Backward Simulation* unstable training process

w/o Coarse-to-Fine Learning 5.31 8 4 8 4 4 8 9 6 4 w/o Adversarial Post-Training 6.31 8 8 9 5 4 7 10 7 9

w Decouple DMD 5.21 7 9 7 4 4 7 8 6 4 w Reward-weighted Regression 6.23 8 10 9 5 5 8 10 7 4

A movie trailer in a classic cinematic style, featuring the adventurous journey of a 30-year-old space man wearing a vibrant red wool knitted motorcycle helmet. The scene unfolds against a vast blue sky and a desolate salt desert landscape. Shot on 35mm film, the trailer showcases ...

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

###### Helios

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

LongLive

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Infinite Forcing

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Reward Forcing

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Causal Forcing

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Dummy Forcing Long

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

SANA Video Long

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

Krea

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

LongCat-Video

frame 1 frame 241 frame 481 frame 721 frame 961 frame 1201 frame 1440

- Figure 14 Qualitative comparisons on 120, 240, 720 and 1440-frames long videos (Part-2). It is clear that Helios consistently outperforms the baseline models in terms of text alignment and dynamics.

[Figure 342]

[Figure 343]

- Figure 15 Side-by-side human evaluation of Helios versus counterparts. Left: Long Video; Right: Short Video

with the historical context. As shown in Table 5 and Figure 16, causal masking substantially reduces representational capacity and makes optimization harder. We attribute this to the fact that causal masking limits cross-section interactions, undermining temporal coherence across sections; as a result, each section tends to generate an independent new scene. To further evaluate Guidance Attention, we remove it from Helios-Base. Without Guidance Attention, the model progressively accumulates semantic content over time, leading to artifacts such as an abnormally enlarged bird crest or steadily increasing saturation.

A vibrant illustration in a whimsical cartoon style depicting a flock of paper airplanes fluttering through a dense jungle. The airplanes, resembling small birds, weave gracefully around towering trees, their wings fluttering gently. The jungle is lush and vibrant, with a variety of exotic plants ...

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

frame 1 frame 61 frame 121 frame 181 frame 240

w Guidance Attention*

A nature photography style photo capturing a family of orangutans along the Kinabatangan River in Borneo. The mother orangutan, with long reddishbrown fur and expressive brown eyes, is holding her baby tightly. The baby orangutan, with smaller size and lighter fur, is clinging to its mother’s ...

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

frame 1 frame 61 frame 121 frame 181 frame 240

A close-up shot of a Victoria crowned pigeon in a naturalistic wildlife photography style, showcasing its striking blue plumage and red chest. The bird’s crest is adorned with delicate, lacy feathers, and its eye is a striking red color, adding to its regal and majestic appearance. The pigeon’s ...

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

frame 1 frame 181 frame 361 frame 541 frame 720

###### Helios-Base

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

frame 1 frame 181 frame 361 frame 541 frame 720

w/o Guidance Attention

A vibrant anime illustration in a thick painting style featuring a young man in his 20s sitting on a fluffy white cloud in the sky, engrossed in reading a classic leather-bound book. He has short, messy black hair and expressive brown eyes, wearing a casual white t-shirt and blue jeans. His posture ...

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

frame 1 frame 361 frame 721 frame 1081 frame 1440

###### Helios-Base

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

frame 1 frame 361 frame 721 frame 1081 frame 1440

w/o Guidance Attention

- Figure 16 Qualitative ablation on Guidance Attention with Helios-Base. Introducing causal mask hinders the ability to learn temporal coherence, causing each generated section to appear independent. Conversely, removing Guidance Attention results in excessive semantic accumulation over time (e.g., a progressively enlarged bird crest). * indicates the additional usage of causal mask to self-attention to prevent noisy context from affecting historical context.

##### 5.4.2 Impact of First Frame Anchor

Building on Helios-Base, we study the effect of the First Frame Anchor on long-video generation. As shown in Table 5 and Figure 17, this component constrains the global appearance distribution and is crucial for maintaining color consistency. Removing it leads to noticeable degradation as early as frame 720, with errors compounding over longer sequences. The component also improves subject consistency: without it, the subject gradually deviates from the identity established in the first frame, causing cumulative identity drifting.

##### 5.4.3 Impact of Frame-Aware Corrupt

As shown in Table 5 and Figure 17, Frame-Aware Corrupt is essential for mitigating error accumulation in long sequences. Removing it causes severe drifting even at 240 frames, leading to a sharp drop in Aesthetic, Semantic, and Naturalness. The degradation becomes more pronounced for minute-scale generation.

A historical footage style photograph depicting a bustling gold rush town in California. The scene captures miners panning for gold in a stream, their faces weathered and determined. Behind them, makeshift wooden shacks and tents line the streets, with smoke rising from chimneys. A man in ...

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

frame 1 frame 361 frame 721 frame 1081 frame 1440

###### Helios-Base

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

frame 1 frame 361 frame 721 frame 1081 frame 1440

w/o First Frame Anchor

A 3D animation of a small, round, fluffy creature with big, expressive eyes exploring a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive ...

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

frame 1 frame 361 frame 721 frame 1081 frame 1440

###### Helios-Base

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

frame 1 frame 361 frame 721 frame 1081 frame 1440

w/o First Frame Anchor

A stunning mid-afternoon landscape photograph with a low camera angle, showcasing several giant wooly mammoths treading through a snowy meadow. Their long, wooly fur gently billows in the brisk wind as they move, creating a sense of natural movement. Snow-covered trees and ...

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

frame 1 frame 61 frame 121 frame 181 frame 240

###### Helios-Base

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

frame 1 frame 61 frame 121 frame 181 frame 240

w/o Frame-Aware Corrupt

A drone view of waves crashing against the rugged cliffs along Big Sur’s Garay Point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore, casting long shadows. In the distance, a small island with a lighthouse ...

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

frame 1 frame 61 frame 121 frame 181 frame 240

###### Helios-Base

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

frame 1 frame 61 frame 21 frame 181 frame 240

w/o Frame-Aware Corrupt

- Figure 17 Qualitative ablation on First Frame Anchor and Frame-Aware Corrupt with Helios-Base. Removing the First Frame Anchor not only introduces drifting but also causes the subject to deviate from the one in the initial frame. Removing Frame-Aware Corruption leads to noticeable drifting even in videos as short as 240 frames.

##### 5.4.4 Impact of Multi-Term Memory Patchification

As shown in Figure 7, Multi-Term Memory Patchification addresses the poor scalability of naive historicalcontext modeling. With naive designs, increasing the history length sharply increases the token count, GPU memory footprint, and inference time; in particular, out-of-memory (OOM) errors occur when the context length reaches 6. In contrast, Helios decomposes the memory into long-, mid-, and short-term scales and applies scale-specific compression ratios. This design extends the history length to 18 while keeping compute and memory costs stable, thereby avoiding OOM and improving inference speed.

A vibrant illustration in a whimsical cartoon style depicting a flock of paper airplanes fluttering through a dense jungle. The airplanes, resembling small birds, weave gracefully around towering trees, their wings fluttering gently. The jungle is lush and vibrant, with a variety of exotic plants ...

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

###### frame 1 frame 181 frame 361 frame 541 frame 720

w Staged Backward Simulation*

A nature photography style photo capturing a family of orangutans along the Kinabatangan River in Borneo. The mother orangutan, with long reddishbrown fur and expressive brown eyes, is holding her baby tightly. The baby orangutan, with smaller size and lighter fur, is clinging to its mother’s ...

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

frame 1 frame 181 frame 361 frame 541 frame 720

A close-up shot of a Victoria crowned pigeon in a naturalistic wildlife photography style, showcasing its striking blue plumage and red chest. The bird’s crest is adorned with delicate, lacy feathers, and its eye is a striking red color, adding to its regal and majestic appearance. The pigeon’s ...

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

frame 1 frame 361 frame 721 frame 1081 frame 1440

###### Helios-Distilled

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

frame 1 frame 361 frame 721 frame 1081 frame 1440

w Self-Forcing

A vibrant anime illustration in a thick painting style featuring a young man in his 20s sitting on a fluffy white cloud in the sky, engrossed in reading a classic leather-bound book. He has short, messy black hair and expressive brown eyes, wearing a casual white t-shirt and blue jeans. His posture ...

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

frame 1 frame 121 frame 181 frame 240

frame 61

Helios-Distilled

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

frame 1 frame 61 frame 121 frame 181 frame 240

w Bidrectional Teacher

###### Figure 18 Qualitative ablation on Stage Backward Simulation and Pure Teacher Forcing with Helios-Distilled.

Feeding multi-scale xk0 into the fake-score estimator pfake causes the model to converge toward incorrect directions. Pure Teacher Forcing achieves comparable robustness against long-video drifting as Self-Forcing [34]. * indicates that

multi-scale xk0, rather than only the full-resolution xK0 , is provided to the real/fake-score estimator.

##### 5.4.5 Impact of Pyramid Unified Predictor Corrector

As shown in Tables 3 and 4, extending a single flow trajectory into multiple multi-scale trajectories enables Pyramid Unified Predictor Corrector to nearly double throughput while incurring only a modest performance drop. This gap is further narrowed in Stage 3 via Adversarial Hierarchical Distillation with Helios-Base as the teacher; we refer to the resulting model as Helios-Mid.

##### 5.4.6 Impact of Pure Teacher Forcing with Autoregressive Teacher

To evaluate Pure Teacher Forcing, we compare against Self-Forcing [34] with long rollouts [11, 59, 60, 100]. As shown in Figure 18 and Table 5, Helios-Distilled achieves robustness to long-horizon drift comparable to SelfForcing. This result suggests that we can obtain similar benefits without long rollouts, substantially reducing training overhead. We further study the impact of teacher architecture by replacing the Autoregressive Teacher with a Bidirectional Teacher (Wan-2.1-T2V-14B [90]). For a fair comparison, we feed 21-frame segments to the real/fake score estimators at each step, following [11, 34, 59, 60, 100]. Overall, Helios-Distilled with an Autoregressive Teacher outperforms the variant that uses a Bidirectional Teacher.

A detailed digital painting in the style of a realistic Japanese manga, capturing reflections in the window of a train traveling through the Tokyo suburbs. The train moves smoothly, passing through lush green fields and dense forests. Outside the window, the scenery blurs into a series of vivid colors ...

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

frame 1 frame 181 frame 361 frame 541 frame 720

###### Helios-Distilled

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

frame 1 frame 181 frame 361 frame 541 frame 720

w/o Coarse-to-Fine Learning

A wide-angle underwater photograph captures a large orange octopus resting on the ocean floor, its tentacles spread out around its body and eyes closed. The octopus blends seamlessly with the sandy and rocky terrain. Behind a rock, a brown and spiny king crab is crawling towards it, its ...

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

frame 1 frame 181 frame 361 frame 541 frame 720

###### Helios-Distilled

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

frame 1 frame 181 frame 361 frame 541 frame 720

w/o Coarse-to-Fine Learning

A close-up view of a glass sphere containing a tranquil Zen garden. Inside, a small Eastern dwarf with weathered skin and a serene expression is raking the sand, meticulously creating intricate patterns with a bamboo rake. His movements are deliberate and meditative, enhancing the ...

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

frame 1 frame 181 frame 361 frame 541 frame 720

###### Helios-Distilled

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

frame 1 frame 181 frame 361 frame 541 frame 720

w/o Adversarial Post-Training

A rotating camera view inside a large New York museum gallery, showcasing a towering stack of vintage televisions, each displaying different programs from the 1950s and 1970s. The televisions show a mix of 1950s sci-fi movies, horror films, news broadcasts, static, and a 1970s ...

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

frame 1 frame 181 frame 361 frame 541 frame 720

###### Helios-Distilled

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

frame 1 frame 181 frame 361 frame 541 frame 720

w/o Adversarial Post-Training

- Figure 19 Qualitative ablation on Coarse-to-Fine Learning and Adversarial Post-Training with Helios-Distilled. Removing Coarse-to-Fine Learning prevents the model from converging, with particularly unacceptable quality in the first generated section. Removing Adversarial Post-Training leads to degradation in visual quality.

##### 5.4.7 Impact of Staged Backward Simulation

Staged Backward Simulation produces multi-scale estimates xk0. We can further interpolate these intermediate results to the t = 0 (i.e., noise-free) state using the following formula and feed them to preal and pfake:

′

xk

0 = (xk0 − λt · ϵ)/(1 − λt). (21)

In principle, incorporating multi-scale xk0 during training can provide richer supervision and improve robustness across resolutions. In practice, however, we feed only the full-scale xK0 to the real/fake score estimators. As shown in Figure 18, directly providing multi-scale xk0 to the fake-score estimator pfake causes the optimization to converge to an undesirable solution, resulting in a significant performance drop.

A cinematic film shot in 35mm capturing a dynamic step-printing scene of a person running. The runner is a young man with short, tousled brown hair and determined eyes, sprinting down a city street lined with tall buildings and neon signs. His arms are pumped vigorously, and he looks focused ...

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

frame 1 frame 181 frame 361 frame 541 frame 720

###### Helios-Distilled

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

frame 1 frame 181 frame 361 frame 541 frame 720

w Decouple DMD

A subtle and elegant photograph in a Japanese style, capturing a woman with gentle, contemplative eyes and flowing dark hair sitting by the window of a high-speed train. The train moves rapidly through a bustling cityscape, with blurred reflections of the city lights and buildings on the window ...

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

frame 1 frame 361 frame 721 frame 1081 frame 1440

###### Helios-Distilled

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

frame 1 frame 361 frame 721 frame 1081 frame 1440

w Decouple DMD

A cinematic scene from a classic western movie, featuring a rugged man riding a powerful horse through the vast Gobi Desert at sunset. The man, dressed in a dusty cowboy hat and a worn leather jacket, reins tightly on the horse's neck as he gallops across the golden sands. The sun sets ...

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

frame 1 frame 181 frame 361 frame 541 frame 720

###### Helios-Distilled

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

frame 1 frame 181 frame 361 frame 541 frame 720

w Reward-weighted Regression

A dynamic rally car speeding through a tight turn on a winding track, tires screeching as it navigates the curves with precision. The car is a sleek, racing machine with a vibrant red body and black accents, its headlights glowing brightly in the night. The driver, a determined-looking man ...

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

frame 1 frame 181 frame 361 frame 541 frame 720

###### Helios-Distilled

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

frame 1 frame 181 frame 361 frame 541 frame 720

w Reward-weighted Regression

- Figure 20 Qualitative ablation on Decouple DMD [54] and Reward-weighted Regression [60] with HeliosDistilled. The former may hinder convergence and cause grayish outputs, whereas the latter may intensify video flickering.

##### 5.4.8 Impact of Coarse-to-Fine Learning

As shown in Figure 19 and Table 5, Coarse-to-Fine Learning not only improves early-stage performance but also is critical for stable convergence. With the same training budget, Helios-Base already generates videos that better align with human perception. In contrast, removing Coarse-to-Fine Learning disrupts convergence and prevents quality improvements, especially for the first generated section.

##### 5.4.9 Impact of Adversarial Post-Training

Adversarial post-training leverages real data to mitigate the performance ceiling imposed by the teacher model during distillation. By introducing an adversarial objective in the distilled stage, the student model is

A romantic scene in a nighttime cityscape where a man and a woman walk hand in hand under a starry sky, their faces illuminated by the soft glow of streetlights. They are dressed in casual yet elegant attire, the man in a dark blue suit and the woman in a light green dress. A wooden bucket is ...

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

A cinematic scene from a classic western movie, featuring a rugged man riding a powerful horse through the vast Gobi Desert at sunset. The man, dressed in a dusty cowboy hat and a worn leather jacket, reins tightly on the horse's neck as he gallops across the golden sands. The sun sets ...

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

A vibrant concert stage scene in the style of a music video, featuring a woman in the spotlight, singing passionately. She stands confidently on the stage, microphone in hand, with a captivating expression on her face. The bright light behind her creates a dramatic silhouette, casting a warm ...

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

An American-style promotional poster featuring a woman in a green jacket and brown boots practicing her archery skills at an outdoor range. She stands with a focused expression, holding a recurve bow and a quiver of arrows on her back. Her hair flows naturally behind her as she aims at ...

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

A dynamic and chaotic scene in a dense forest during a heavy rainstorm, capturing a real girl frantically running through the foliage. Her wild hair flows behind her as she sprints, her arms flailing and her face contorted in fear and desperation. Behind her, various animals—rabbits, deer, and ...

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

###### Figure 21 Text-to-Video showcases. Part of the prompts are sourced from [34].

Bathed in the warm glow of the setting sun, the iconic Golden Gate Bridge stretches gracefully across the bay, its vibrant orange towers casting long shadows over the shimmering water. As the view slowly pans left, the bridge’s elegant lines and sweeping cables are illuminated against a backdrop ...

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

A high-resolution photograph captures an Airbus in flight against a clear sky, emphasizing its sleek design and iconic blue and white livery. The flights large wingspan and distinctive tail fin are clearly visible, showcasing its impressive size and engineering. The image is taken from a side ...

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

A sweeping aerial perspective reveals a striking, futuristic structure perched dramatically on a cliff’s edge, its bold, saucer-like form jutting out over shimmering waters. A vibrant, winding walkway leads visitors toward the building, guiding them along the cliffside and inviting exploration. The ...

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, emphasizing its dynamic movement. The road curves gently, with a ...

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

A breathtaking panoramic view of Mount Everest at sunrise, capturing the majestic peaks bathed in the warm golden light of dawn. The rugged, snowcovered terrain is dramatically highlighted, casting deep shadows that accentuate the mountain's intricate textures. The sky transitions beautifully ...

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

frame 321 frame 481 frame 641

frame 1 frame 161

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

frame 801 frame 961 frame 1121

frame 1281 frame 1440

###### Figure 22 Image-to-Video showcases. Part of the images are sourced from [35, 45].

- 0: A medium close-up of a serene, young East Asian woman with long pink hair, wearing a simple white gown that flows softly around her. She stands amidst gently falling sakura petals, which swirl around her, partially obscuring her figure in a dreamy, ethereal haze. Her delicate features are calm ...
- 1: In a serene, still environment, a woman gently lifts her hand towards a delicate flower petal, her fingertips barely touching the wisps of smoke floating nearby. She has a soft, contemplative expression on her face, and her hand moves slowly and gracefully. The scene is captured from a ...
- 2: A gentle breeze rustles through the air, causing cherry blossom petals to dance gracefully around a young woman's outstretched hand. She stands with a serene expression, wearing a traditional hanfu gown with flowing sleeves. The background showcases a tranquil garden with a winding path ...
- 3: A serene woman with closed eyes and soft, resting eyelashes, exuding a sense of tranquility and peace. She sits gracefully with her hands gently folded in her lap, wearing a flowing, pastel-colored dress that complements her calm demeanor. The lighting is soft and diffused, casting a gentle ...
- 4: A serene nature scene where a young woman with flowing hair and a gentle expression steps forward gracefully on a grassy meadow. In the background, a small bird flutters and lands on a nearby tree branch. The woman's soft movements contrast with the bird's quick flight, creating a ... section 0 section 1 section 2 section 3 section 4

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

- 0: Center frame, a sleek silver-gray house cat trots briskly across the leaf-strewn ground, its tail flicking energetically. As a cool breeze stirs, more leaves swirl and dance across the lens, adding a dynamic motion to the tranquil scene. Medium shot capturing the cat in motion against the backdrop ...
- 1: In a lush forest setting, a playful house cat bounds towards the treeline, leaping gracefully over a fallen log. Mid-stride, the cat transforms into a sleek, smoky brown wildcat with tufted ears and lean, powerful muscles. It continues to run swiftly through the drifting autumn leaves, propelled by ...
- 2: In the warm golden hour, a wildcat leaps gracefully across a sparkling stream. As it runs, its body broadens and its fur transforms from its original color to a deep russet tone, morphing seamlessly into a sleek red fox. The fox's bushy, vibrant plume tail catches the sunlight, illuminating its swift ...
- 3: In twilight, a red fox races across an old stone bridge, its chest heaving and coat gradually graying as it transforms into a majestic gray wolf midjump. The wolf then gallops through an open forest clearing, leaping over a babbling stream, and continues up the hillside. The sky transitions ...
- 4: In a serene moonlit forest, the transformation of a wolf into an antelope unfolds gracefully. Initially, the wolf's limbs elongate and refine, its fur shortening from its dense winter coat to a sleek, tawny color. As it shifts, small horns begin to emerge from its forehead. The antelope then swiftly ...

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

section 0 section 1 section 2 section 3 section 4

- 0: 90s VHS-style The Weather Channel scene, featuring a weatherman standing in front of a green screen with a large map of storm systems behind him. The weatherman, dressed in a casual but professional outfit, points emphatically at the rapidly moving storms on the map. His face shows ...
- 1: The weatherman now reaches into his pocket and pulls out a chunky black walkie-talkie, pressing it against his ear with a sudden look of intense focus. He nods sharply while listening to an urgent update, his brow furrowing deeper under the studio lights. The green screen map behind him ...
- 2: The weatherman suddenly unrolls a large, paper topographic map across a small stand that appears from the side, smoothing out the creases with frantic energy. He traces a specific mountain range with his finger, highlighting a dangerous path the storm is taking, his eyes wide with genuine ...
- 3: The weatherman abruptly grabs a bright red marker pen from the desk edge and begins circling a specific coastal city directly on the camera lens itself. He draws a jagged, erratic line across the glass to simulate the storm's unpredictable path, his hand shaking slightly with adrenaline. The ink ...
- 4: The weatherman suddenly dons a bright yellow rain slicker over his suit, struggling momentarily with the snaps as he prepares for a simulated outdoor report. He pulls the hood up over his head, framing his face tightly while the green screen behind him shifts to show footage of swaying ...

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

section 0 section 1 section 2 section 3 section 4

- 0: A young woman standing in the rain, looking up at the sky with a warm, inviting smile on her face. She is dressed in a light, flowy dress that clings to her form as droplets of water fall around her. Her hair is gently tousled from the rain, framing her delicate features. The background shows a ...
- 1: The young woman remains framed against the soft blur of city lights, the rain now glistening on her skin as she slowly extends her right hand palmup to catch the falling droplets. Her expression shifts slightly to one of quiet wonder as the water pools in her cupped fingers. The light fabric of her ...
- 2: The young woman in the light, flowy dress now closes her eyes, tilting her head back slightly to let the rain wash over her face. She reaches into her pocket and pulls out a bright red origami crane, holding it delicately between her fingers. The paper quickly darkens as it absorbs the moisture, ...
- 3: The young woman in the soaked flowy dress opens her eyes and suddenly unfurls a small, transparent umbrella with a floral pattern above her head. The rain drums rhythmically against the plastic canopy, creating a protective bubble around her upper body while water streams down the sides ...
- 4: The young woman stands beneath the transparent floral umbrella, the rain creating a rhythmic patter on its surface. She reaches up with her free hand and adjusts a pair of round, gold-rimmed glasses onto the bridge of her nose, blinking behind the lenses as they catch the ambient city light ...

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

section 0 section 1 section 2 section 3 section 4

###### Figure 23 Interactive-to-Video showcases. Part of the prompts are sourced from [91, 100].

- Table 6 Quantitative ablation on Flash Normalization and Flash RoPE. We report the total runtime of the DiT component in Helios, measured over 50 forward passes and 50 forward-backward passes on a single NVIDIA H100 GPU.

Inference Time (s) Training Time (s)

Wan-2.1-T2V-14B [90] 98.68 398.03 w Flash Normalization 89.91 360.77

w Flash RoPE 93.39 378.77 w Flash Normalization and Flash RoPE 84.41 340.38

encouraged not only to mimic the teacher’s outputs but also to better align with the distribution of real data, thereby enhancing realism beyond what pure distillation can achieve. To assess its impact, we remove it in an ablation study. As shown in Figure 19 and Table 5, disabling this object noticeably degrades visual quality, particularly in naturalness and realism, highlighting its importance for improving perceptual fidelity.

##### 5.4.10 Impact of Flash Normalization and Flash RoPE

To evaluate Flash Normalization and Flash RoPE, we measure the inference and training time of Wan-2.1T2V-14B [90] for 50 steps at 384 × 640 with 81-frame inputs. As reported in Table 6, Triton-optimized kernels substantially improve throughput, which is crucial for real-time long-video generation.

##### 5.4.11 Ablation on Decouple DMD

Decouple DMD [54] reformulates the original DMD objective [105, 106] as the weighted sum of two disentangled components: CFG Augmentation (CA) and Distribution Matching (DM). By assigning separate weights to these terms, it provides more flexible control over the optimization process and has been shown to yield improved performance in image generation settings. Following [54], we extend this formulation from image to video generation and integrate it into our framework. However, as illustrated in Figure 20 and Table 5, this variant exhibits noticeably slower convergence compared to our default setting. Moreover, the generated videos tend to suffer from grayish tones, occasional local jitter, and grid-like artifacts, indicating suboptimal temporal and spatial consistency. Given these limitations, we do not adopt Decouple DMD in our final model.

##### 5.4.12 Ablation on Reinforcement Post-Training

Beyond adversarial objectives that improve the upper bound, recent post-training methods attempt to raise the lower bound via reinforcement learning (RL) [56, 57, 98] Reward Forcing [60], we weight the standard DMD [105, 106], which can be interpreted as a form of reward-weighted regression [57, 68]:

pfake(x0 | y) preal(x0 | y)

r(x0,y) β · log

0 Rrl · LDMD . (22)

Ey,x

= Ey,x

0, exp

Here, pref denotes the reference distribution (e.g., the teacher), and β controls the trade-off between reward maximization and distribution shift. We use VideoAlign [57] as the reward model and its Motion Quality

- as the score. As shown in Figure 20 and Table 5, reinforcement learning consistently degrades performance: semantic and aesthetic scores drop, and the outputs exhibit severe flickering. We therefore exclude RL.

- 6 Application

Benefiting from the proposed Representation Control, Helios continues to adopt the conventional text-to-video pipeline for data preparation and model optimization during training. However, since the historical context is randomly zeroed out with a certain probability throughout training, the model can naturally generalize

- at inference time and seamlessly support T2V, I2V, and V2V tasks. The showcases are presented in Figure 22 and Figure 21, demonstrating satisfactory quality. Moreover, by incorporating Interactive Interpolation, Helios further enables, in a zero-shot manner, a key capability of world models – Interactive Generation. This mechanism allows users to dynamically modify the input prompt during the generation process, thereby providing real-time control over the generated content. Some representative showcases are shown in Figure 23.

### 7 Conclusion

In this paper, we presented Helios, the first 14B video generation model that runs at 19.5 FPS on a single NVIDIA H100 GPU and supports minute-scale generation, without relying on conventional anti-drifting strategies or acceleration techniques. Helios adopts a unified input representation that supports text-to-video, image-to-video, and video-to-video tasks within a single framework. To mitigate drifting in long-video generation, we analyze typical failure modes and develop a set of simple yet effective training strategies that explicitly simulate drifting. In terms of efficiency, we reduce computational cost to a level comparable to—or even lower than—that of image diffusion models by aggressively compressing the historical context, the noisy context, and the number of sampling steps. Extensive experiments show that Helios consistently outperforms existing methods across multiple dimensions for both short and long videos.

### 8 Limitations and Future Work

- (1) Existing metrics [35, 36, 120] are insufficient for accurately assessing the performance of video generation models. Although Helios can produce realistic and natural videos conditioned on text prompts, widely used metrics (e.g., Aesthetic and Smoothness) exhibit only marginal differences compared to prior approaches. A promising direction is to develop perceptually aligned metrics that better reflect human judgment.
- (2) Similar to existing autoregressive models [11, 59, 60, 100, 107, 116, 126], the generated segments may suffer from flickering artifacts at stitching boundaries. Nevertheless, our method consistently outperforms previous approaches. Future work could explore reinforcement learning techniques that explicitly optimize smoothness-related objectives to further enhance temporal consistency.
- (3) Due to resource constraints, our experiments are limited to a resolution of 384×640, leaving higher-resolution settings unexplored. This work prioritizes real-time long-video generation as a core capability for world models [2–4, 20, 65, 66, 78, 83, 102, 112] and video generation [1, 8, 30, 31, 41, 43, 49, 63, 79, 80, 90, 93, 122], without a dedicated design for long-video memory. Both remain important directions for future work.

### 9 Declaration of LLM Usage

We used advanced natural language processing models, such as GPT, to assist in preparing this manuscript. Specifically, GPT supported language-related tasks including grammar correction, spell-checking, and expression refinement, improving the paper’s precision and readability. GPT also helped with data organization, selection, and preliminary figure design, as well as drafting alternative phrasing for clarity. All academic content, analyses, and conclusions were independently developed, validated, and interpreted by the authors.

### 10 Ethics Statement

Helios is capable of generating high-quality, realistic videos. However, it also carries potential negative consequences, as the technology could be exploited to create deceptive content for fraudulent purposes. It is crucial to acknowledge that all technologies are vulnerable to misuse.

### 11 Acknowledgment

We thank all collaborators for their constructive comments and support throughout this project. This project is for research purposes only and is not integrated into any ByteDance products.

### References

- [1] Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko, Viacheslav Vasilev, Alexey Letunovskiy, Nikolai Vaulin, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, et al. Kandinsky 5.0: A family of foundation models for image and video generation. arXiv preprint arXiv:2511.14993, 2025.
- [2] Philip J Ball, Jakob Bauer, Frank Belletti, B Brownfield, A Ephrat, S Fruchter, A Gupta, K Holsheimer, A Holynski, J Hron, et al. Genie 3: A new frontier for world models. Google DeepMind Blog, pages 253–279, 2025.
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. In openai, 2024. URL https://openai.com/research/video-generation-models-as-world-simulators.
- [4] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.
- [5] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.
- [6] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025.
- [7] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [8] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.
- [9] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.
- [10] Huanpeng Chu, Wei Wu, Guanyu Feng, and Yutao Zhang. Omnicache: A trajectory-oriented global perspective on training-free cache reuse for diffusion transformer models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16302–16312, 2025.
- [11] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.
- [12] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Lol: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026.
- [13] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.
- [14] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17702–17711, 2025.
- [15] Michael Han Daniel Han and Unsloth team. Unsloth, 2023. URL http://github.com/unslothai/unsloth.
- [16] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.
- [17] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [18] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.

- [19] Yufan Deng, Xun Guo, Yuanyang Yin, Jacob Zhiyuan Fang, Yiding Yang, Yizhi Wang, Shenghai Yuan, Angtian Wang, Bo Liu, Haibin Huang, et al. Magref: Masked guidance for any-reference video generation. arXiv preprint arXiv:2505.23742, 2025.
- [20] Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, and Daquan Zhou. Rethinking video generation model for the embodied world. arXiv preprint arXiv:2601.15282, 2026.
- [21] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.
- [22] Weichen Fan, Amber Yijia Zheng, Raymond A. Yeh, and Ziwei Liu. Cfg-zero*: Improved classifier-free guidance for flow matching models, 2025. URL https://arxiv.org/abs/2503.18886.
- [23] Tianrui Feng, Zhi Li, Shuo Yang, Haocheng Xi, Muyang Li, Xiuyu Li, Lvmin Zhang, Keting Yang, Kelly Peng, Song Han, et al. Streamdiffusionv2: A streaming system for dynamic and interactive video generation. arXiv preprint arXiv:2511.07399, 2025.
- [24] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Miguel Angel Bautista, and Josh Susskind. f-dm: A multi-stage diffusion model via progressive signal transformation. arXiv preprint arXiv:2210.04955, 2022.
- [25] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.
- [26] Hang Guo, Zhaoyang Jia, Jiahao Li, Bin Li, Yuanhao Cai, Jiangshan Wang, Yawei Li, and Yan Lu. Efficient autoregressive video diffusion with dummy head. arXiv preprint arXiv:2601.20499, 2026.
- [27] Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, and Jiang Bian. Mineworld: a real-time and open-source interactive world model on minecraft. arXiv preprint arXiv:2504.08388, 2025.
- [28] Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, and Dahua Lin. End-to-end training for autoregressive video diffusion via self-resampling. arXiv preprint arXiv:2512.15702, 2025.
- [29] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025.
- [30] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.
- [31] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.
- [32] Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, Yanning Chen, and Zhipeng Wang. Liger-kernel: Efficient triton kernels for LLM training. In Championing Open-source DEvelopment in ML Workshop @ ICML25, 2025. URL https://openreview.net/ forum?id=36SjAIT42G.
- [33] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024.
- [34] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [35] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [36] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [37] Weinan Jia, Yuning Lu, Mengqi Huang, Hualiang Wang, Binyuan Huang, Nan Chen, Mu Liu, Jidong Jiang, and Zhendong Mao. Moga: Mixture-of-groups attention for end-to-end long video generation. arXiv preprint arXiv:2510.18692, 2025.

- [38] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. 2024.
- [39] Xianglong He Junyi Chen, Zhoujie Fu. Infinite-forcing: Towards infinite-long video generation, 2025. https://github.com/SOTAMak1r/Infinite-Forcing.
- [40] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37:89834–89868, 2024.
- [41] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [42] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.
- [43] Debang Li, Zhengcong Fei, Tuanhui Li, Yikun Dou, Zheng Chen, Jiangping Yang, Mingyuan Fan, Jingtao Xu, Jiahua Wang, Baoxuan Gu, et al. Skyreels-v3 technique report. arXiv preprint arXiv:2601.17323, 2026.
- [44] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025.
- [45] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025.
- [46] Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: Sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025.
- [47] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Feize Wu, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, et al. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025.
- [48] Zongjian Li, Shenghai Yuan, and Li Yuan. Sparkwan: Few-step video diffusion transformer via hybrid adversarial post-training, 2025. URL https://github.com/PKU-YuanGroup/Spark-Wan.
- [49] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [50] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.
- [51] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, Chao Liang, Yuan Zhang, and Jingtuo Liu. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13847–13858, 2025.
- [52] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.
- [53] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316, 2025.
- [54] Dongyang Liu, Peng Gao, David Liu, Ruoyi Du, Zhen Li, Qilong Wu, Xin Jin, Sihan Cao, Shifeng Zhang, Hongsheng Li, and Steven Hoi. Decoupled dmd: Cfg augmentation as the spear, distribution matching as the shield. arXiv preprint arXiv:2511.22677, 2025.
- [55] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7353–7363, 2025.
- [56] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.
- [57] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.

- [58] Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Infinitystar: Unified spacetime autoregressive modeling for visual generation. arXiv preprint arXiv:2511.04675, 2025.
- [59] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.
- [60] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025.
- [61] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [62] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.
- [63] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.
- [64] Zehong Ma, Longhui Wei, Feng Wang, Shiliang Zhang, and Qi Tian. Magcache: Fast video generation with magnitude-aware cache. arXiv preprint arXiv:2506.09045, 2025.
- [65] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025.
- [66] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025.
- [67] Erwann Millon. Krea realtime 14b: Real-time video generation, 2025. URL https://github.com/krea-ai/ realtime-video.
- [68] Jan Peters and Stefan Schaal. Reinforcement learning by reward-weighted regression for operational space control. In Proceedings of the 24th international conference on Machine learning, pages 745–750, 2007.
- [69] Ryan Po, Eric Ryan Chan, Changan Chen, and Gordon Wetzstein. Bagger: Backwards aggregation for mitigating drift in autoregressive video diffusion models. arXiv preprint arXiv:2512.12080, 2025.
- [70] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023.
- [71] David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. Rolling diffusion models. arXiv preprint arXiv:2402.09470, 2024.
- [72] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [73] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.
- [74] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, and Xun Huang. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.
- [75] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025.
- [76] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [77] Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7364–7373, 2025.

- [78] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.
- [79] FSVideo Team, Qingyu Chen, Zhiyuan Fang, Haibin Huang, Xinwei Huang, Tong Jin, Minxuan Lin, Bo Liu, Celong Liu, Chongyang Ma, et al. Fsvideo: Fast speed video diffusion model in a highly-compressed latent space. arXiv preprint arXiv:2602.02092, 2026.
- [80] Genmo Team. Mochi 1. https://github.com/genmoai/models, 2024.
- [81] HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, Wenhuan Li, Sheng Zhang, et al. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025.
- [82] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, et al. Kimi linear: An expressive, efficient attention architecture. arXiv preprint arXiv:2510.26692, 2025.
- [83] LingBot-World Team. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.
- [84] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025.
- [85] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020.
- [86] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [87] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.
- [88] Ye Tian, Xin Xia, Yuxi Ren, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Yunhai Tong, Ling Yang, and Bin Cui. Training-free diffusion acceleration with bottleneck sampling. arXiv preprint arXiv:2503.18940, 2025.
- [89] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [90] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [91] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, 37:65618–65642, 2024.
- [92] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.
- [93] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.
- [94] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [95] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.
- [96] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. arXiv preprint

- arXiv:2507.18634, 2025.

[97] Bowen Xue, Zheng-Peng Duan, Qixin Yan, Wenjing Wang, Hao Liu, Chun-Le Guo, Chongyi Li, Chen Li, and Jing Lyu. Stand-in: A lightweight and plug-and-play identity control for video generation. arXiv preprint

- arXiv:2508.07901, 2025.

- [98] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.
- [99] Ran Yan, Youhe Jiang, and Binhang Yuan. Flash sparse attention: More efficient natively trainable sparse attention. arXiv preprint arXiv:2508.18224, 2025.
- [100] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.
- [101] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2023.
- [102] Ying Yang, Zhengyao Lv, Tianlin Pan, Haofan Wang, Binxin Yang, Hubery Yin, Chen Li, Ziwei Liu, and Chenyang Si. Stableworld: Towards stable and consistent long interactive video generation. arXiv preprint arXiv:2601.15281, 2026.
- [103] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [104] Yang Ye, Tianyu He, Shuo Yang, and Jiang Bian. Reinforcement learning with inverse rewards for world model post-training. arXiv preprint arXiv:2509.23958, 2025.
- [105] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.
- [106] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024.
- [107] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025.
- [108] Jongmin Yu, Hyeontaek Oh, Zhongtian Sun, Angelica I Aviles-Rivero, Moongu Jeon, and Jinhong Yang. Alphaface: High fidelity and real-time face swapper robust to facial pose. arXiv preprint arXiv:2601.16429, 2026.
- [109] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Rui-Jie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024.
- [110] Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292, 2025.
- [111] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12978–12988, 2025.
- [112] Shenghai Yuan, Jinfa Huang, Yujun Shi, Yongqi Xu, Ruijie Zhu, Bin Lin, Xinhua Cheng, Li Yuan, and Jiebo Luo. Magictime: Time-lapse video generation models as metamorphic simulators. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [113] Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. arXiv preprint arXiv:2410.02367, 2024.
- [114] Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattn: Accurate sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025.
- [115] Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Turbodiffusion: Accelerating video diffusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025.
- [116] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.
- [117] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025.

- [118] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.
- [119] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36:49842–49869, 2023.
- [120] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.
- [121] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025.
- [122] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [123] Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Chongxuan Li. Concat-id: Towards universal identitypreserving video synthesis. arXiv preprint arXiv:2503.14151, 2025.
- [124] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. Advances in Neural Information Processing Systems, 37: 110315–110340, 2024.
- [125] Ziqin Zhou, Yifan Yang, Yuqing Yang, Tianyu He, Houwen Peng, Kai Qiu, Qi Dai, Lili Qiu, Chong Luo, and Lingqiao Liu. Hitvideo: Hierarchical tokenizers for enhancing text-to-video generation with autoregressive large language models. arXiv preprint arXiv:2503.11513, 2025.
- [126] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.
- [127] Yixuan Zhu, Jiaqi Feng, Wenzhao Zheng, Yuan Gao, Xin Tao, Pengfei Wan, Jie Zhou, and Jiwen Lu. Astra: General interactive world model with autoregressive denoising. arXiv preprint arXiv:2512.08931, 2025.
- [128] Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

## Appendix

### A Calculation of Helios-Bench

To provide a robust and human-aligned evaluation of generated videos, Helios-Bench employs a multi-stage scoring pipeline that normalizes, discretizes, and aggregates raw automated metrics to mitigate inherent noise and scale variations. Specifically, let sk denote the raw score of a generated video evaluated on metric k. We first bound and normalize each raw score to a standard range of [0,1] to compute the normalized score s¯k = max(0,min(1, s

k−smin,k

smax,k−smin,k )), where smin,k and smax,k are the predefined bounds for metric k. Because raw continuous scores often correlate poorly with human perception, we then map each normalized score s¯k to a discrete 10-point rating rk ∈ {1,2,...,10} using a step function parameterized by empirically derived thresholds Tk = [τ0,τ1,...,τ8], as shown in Table 7. For higher-is-better spatial and temporal quality metrics (i.e., Aesthetic, Motion Amplitude, Motion Smoothness, Semantic, and Naturalness), the rating is assigned by finding the first threshold the score meets or exceeds, formulated as rk = 10−i such that i = min{j | s¯k ≥ τj}, with a minimum rating of 1 if s¯k < τ8. Conversely, for lower-is-better metrics (specifically the drifting variants used in long-video evaluation), the rating is computed as rk = 10 − i such that i = min{j | s¯k ≤ τj}. Finally, the overall performance of a model is calculated as the duration-aware weighted sum of its discretized ratings, S = k wkrk, where wk denotes the metric-specific weight. For short videos, the evaluation heavily prioritizes Semantic alignment and Naturalness (each weighted at 0.35), with Aesthetic, Motion Amplitude, and Motion Smoothness each weighted at 0.10. For long videos, to adequately penalize spatial and semantic degradation over time, drifting metrics (Drifting Semantic, Drifting Naturalness, Drifting Aesthetic, and Drifting Motion Smoothness) are introduced with a weight of 0.099 each, while the base weights for Semantic and Naturalness are reduced to 0.255, and Aesthetic, Motion Amplitude, and Motion Smoothness are reduced to 0.03.

Table 7 Empirical Thresholds (Tk) for Metric Discretization. For the Aesthetic metric, the final two thresholds are marked with an asterisk (*) to indicate a potential anomaly in the empirical data or source configuration.

Metric (k) Type τ0 τ1 τ2 τ3 τ4 τ5 τ6 τ7 τ8

Aesthetic Higher Better 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 Motion Amplitude Higher Better 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05 Motion Smoothness Higher Better 0.99 0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 Semantic Higher Better 0.30 0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.22 Naturalness Higher Better 0.65 0.60 0.55 0.50 0.45 0.40 0.30 0.25 0.20

Drifting Aesthetic Lower Better 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 Drifting Motion Smoothness Lower Better 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 Drifting Semantic Lower Better 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 Drifting Naturalness Lower Better 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22

### B Latent Re-encode.

Since the ablation of self-forcing employs Wan-2.1-T2V [90] as the real/fake-score estimator—which requires 21 latent frames for optimal performance—Helios generates videos section-by-section in the latent space to satisfy this constraint. However, concatenating multiple sections introduces a distributional mismatch: within each section, the first frame exhibits statistics that differ substantially from those of subsequent frames. Consequently, the concatenated sequence contains multiple occurrences of a “first-frame” distribution, biasing the score estimators. To eliminate this artifact, we decode each latent section of xK0 via the VAE decoder and re-encode the resulting video as a single, continuous latent sequence, which contains only a single first-frame distribution. The standard re-noising operation is then applied. This re-encoding step improves distributional alignment with both preal and pfake, thereby reducing score-estimation bias and enhancing temporal coherence.

