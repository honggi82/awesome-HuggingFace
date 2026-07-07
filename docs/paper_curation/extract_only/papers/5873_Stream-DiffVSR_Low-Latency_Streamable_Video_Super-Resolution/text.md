## Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion

Hau-Shiang Shiu1, Chin-Yang Lin1, Zhixiang Wang2, Chi-Wei Hsiao3, Po-Fan Yu1, Yu-Chih Chen1, and Yu-Lun Liu1

1 National Yang Ming Chiao Tung University, 2 Shanda AI Research Tokyo 3 MediaTek Inc. xhs0964519.cs12@nycu.edu.tw, yulunliu@cs.nycu.edu.tw

# arXiv:2512.23709v2[cs.CV]4Apr2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

| |[Figure 6]<br><br>StableVSR<br><br>MGLD-VSR<br><br>MIA-VSR RVRT RealViformer<br><br>|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

Maximum Latency: 0.328 LPIPS: 0.099

Maximum Latency: 2.49 LPIPS: 0.130

Maximum Latency: 218 LPIPS: 0.151

Maximum Latency: 76.8 LPIPS: 0.123

100 Frames, REDS4

| |
|---|

Runtime/Frame(s)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

RVRT MGLD-VSR MIA-VSR Ours

Input \ Ours

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

7 Frames, Vimeo-90K-T

Maximum Latency: LPIPS: -

Maximum Latency: 40.24 LPIPS: 0.072

Maximum Latency: 0.084 LPIPS: 0.098

Maximum Latency: 0.041 LPIPS: 0.056

| |
|---|

RealBasicVSR

BasicVSR++ Ours

TMP

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Input \ Ours

Input \ Ours

Bicubic StableVSR BasicVSR++ Ours

LPIPS

Fig. 1: Comparison of visual quality and inference speed across various categories of VSR methods. Stream-DiffVSR achieves superior perceptual quality (lower LPIPS) and maintains comparable runtime to CNN- and Transformer-based online models, while also demonstrating significantly reduced inference latency compared to existing offline approaches. Best and second-best results are marked in red and green.

Abstract. Diffusion-based video super-resolution (VSR) methods deliver strong perceptual quality but are often unsuitable for latency-sensitive scenarios due to reliance on future frames and expensive multi-step denoising. We propose Stream-DiffVSR, a causally conditioned diffusion framework for efficient online VSR. Operating strictly on past frames, Stream-DiffVSR integrates a four-step distilled denoiser for fast inference, an Auto-regressive Temporal Guidance (ARTG) module that injects motion-aligned cues during latent denoising, and a lightweight temporalaware decoder with a Temporal Processor Module (TPM) to enhance detail and temporal coherence. Unlike chunk-wise streaming inference, our strictly frame-by-frame causal design avoids sequence-level waiting, substantially reducing time-to-first-frame and end-to-end latency. StreamDiffVSR processes 720p frames in 0.328 seconds on an RTX 4090 and consistently outperforms prior diffusion-based baselines. Compared with the online state-of-the-art TMP [131], it improves perceptual quality (LPIPS +0.095) while reducing latency by over 130×. Moreover, Stream-DiffVSR substantially lowers time-to-first-frame for diffusion-based VSR, reducing initial delay from over 4600 seconds to 0.328 seconds, making diffusionbased VSR markedly more practical for low-latency online and streaming deployment. Project page: https://jamichss.github.io/streamdiffvsr-project-page/

Keywords: Video super-resolution · Diffusion models · Online/Streaming inference

### 1 Introduction

Video super-resolution (VSR) aims to reconstruct high-resolution (HR) videos from low-resolution (LR) inputs and is vital in applications such as surveillance, live broadcasting, video conferencing, autonomous driving, and drone imaging. It is increasingly important in low-latency rendering workflows, including neural rendering and resolution upscaling in game engines and AR/VR systems, where latency-aware processing is crucial for visual continuity.

Specifically, latency-sensitive processing involves two key aspects: per-frame inference time (throughput) and end-to-end system latency (delay between receiving an input frame and producing its output). Existing VSR methods often struggle with this trade-off. While CNN- and Transformer-based models offer a balance between efficiency and quality, they fall short in perceptual detail. Diffusion-based models excel in perceptual quality due to strong generative priors, but suffer from high computational cost and reliance on future frames, making them impractical for time-sensitive video applications.

In this paper, we propose Stream-DiffVSR, a diffusion-based method specifically tailored to online video super-resolution, effectively bridging the gap between high-quality but slow diffusion methods and fast but lower quality CNNor Transformer-based methods. Unlike previous diffusion-based VSR approaches (e.g., StableVSR [75] and MGLD-VSR [114]) that typically require 50 or more denoising steps and bidirectional temporal information, our method leverages diffusion model distillation to significantly accelerate inference by reducing denoising steps to just four. Additionally, we introduce an Auto-regressive Temporal Guidance mechanism and an Auto-regressive Temporal-aware Decoder to effectively exploit temporal information from previous frames, significantly enhancing temporal consistency and perceptual fidelity.

Fig. 1 illustrates the core advantage of our approach by comparing visual quality and runtime across various categories of video super-resolution methods. Stream-DiffVSR achieves improved perceptual quality, as reflected by lower LPIPS [128] and temporal consistency, outperforming existing unidirectional CNN- and Transformer-based methods (e.g., MIA-VSR [141], RealViformer [130], TMP [131]). Notably, Stream-DiffVSR offers significantly faster per-frame inference than prior diffusion-based approaches (e.g., StableVSR [75], MGLDVSR [114]), attributed to our use of a distilled 4-step denoising process and a lightweight temporal-aware decoder.

In addition, existing diffusion-based methods, such as StableVSR [75] typically rely on bidirectional or future-frame information, resulting in prohibitively high processing latency that is not suitable for online scenarios. Specifically, for a 100-frame video, StableVSR (46.2 s/frame) would incur an initial latency exceeding 4600 seconds on an RTX 4090 GPU, as it requires processing the entire sequence before generating even the first output frame. In contrast, our Stream-

- Table 1: Comparison of diffusion-based VSR methods. We report online capability, inference steps, runtime (FPS on 720p, RTX 4090), maximum end-to-end latency (sec), and whether each method uses distillation, temporal modeling, or offline future frames. OOM denotes out-of-memory, and - indicates missing public inference results. Notably, Stream-DiffVSR runs in a strictly online, past-only setting and achieves one of the lowest end-to-end latencies among compared diffusion-based VSR methods under our measurement protocol.

# of FPS Max Temporal Temporal Method Online Steps @720p latency Distill Input Decoder

StableVSR [75] ✗ 50 0.02 4620 ✗ Future/Bi-dir ✗ MGLD-VSR [114] ✗ 50 0.02 218 ✗ Future/Bi-dir ✓ Upscale-A-Video [140] ✗ 30 OOM - ✗ Future/Bi-dir ✓ VEnhancer [30] ✗ 15 OOM - ✗ Future/Bi-dir ✓ Stream-DiffVSR (ours) ✓ 4 3.05 0.328 ✓ Past-only ✓

DiffVSR operates in a strictly causal, autoregressive manner, conditioning only on the immediately preceding frame. Consequently, the initial frame latency of Stream-DiffVSR corresponds to a single frame’s inference time (0.328 s/frame), reducing the latency by more than three orders of magnitude compared to StableVSR. This significant latency reduction demonstrates that Stream-DiffVSR effectively unlocks the potential of diffusion models for practical, low-latency online video super-resolution.

To summarize, the main contributions of this paper are:

- – We introduce Stream-DiffVSR, a diffusion-based framework explicitly designed for streamable, low-latency video super-resolution, enabling efficient inference by distilling diffusion sampling from 50 denoising steps down to 4 steps.
- – We propose a novel Auto-regressive Temporal Guidance mechanism and a Temporal-aware Decoder to effectively leverage temporal cues only from past frames, significantly enhancing perceptual quality and temporal consistency.
- – Extensive experiments demonstrate that our approach outperforms existing methods on key perceptual and temporal consistency metrics while maintaining low-latency inference, making diffusion-based VSR practical for latency-critical online/streaming and interactive video pipelines.

To contextualize our contributions, Table 1 compares recent diffusion-based VSR methods in terms of online inference capability, runtime efficiency, and temporal modeling. Stream-DiffVSR performs online, low-latency inference in a strictly causal (past-only), frame-by-frame manner, delivering diffusion-level perceptual quality while maintaining strong temporal stability. This substantial latency reduction of over three orders of magnitude compared to prior diffusion-based VSR models makes diffusion-based VSR markedly more practical for latency-critical online applications such as video conferencing and AR/VR.

### 2 Related Work

Video Super-resolution. VSR methods reconstruct high-resolution videos from low-resolution inputs through CNN-based approaches [6, 7, 88, 90, 99, 112], deformable convolutions [18,90,143], online processing [21,118,131], recurrent architectures [22, 36, 47, 78, 117], flow-guided methods [28, 60, 122], Transformerbased models [50,52,82,93,141], and implicit alignment techniques [109]. Causal and streaming paradigms have emerged for low-latency scenarios [26, 70, 138], while structured pruning [105] targets efficient deployment. Despite advances, low-latency online processing with high perceptual quality remains challenging.

VSR under Unknown Degradations. VSR under unknown degradations studies how to handle diverse and poorly specified degradation processes [8, 115] through pre-cleaning modules [8,27,59,100], online approaches [130], kernel estimation [39,69], synthetic degradations [9,38,85,129], new benchmarks [17,134], real-time systems [5, 40], advanced GANs [11, 14, 92], and Transformer restorers [4,51,124]. Recent efforts leverage text-to-video priors for VSR with complex, non-ideal degradations [107,135]. Warp error-aware consistency [45] emphasizes temporal error regularisation.

Diffusion-based Image and Video Restoration. Diffusion models provide powerful generative priors [10,20,74] for single-image SR [34,46,77,101,123], inpainting [56,63,91,102], and quality enhancement [24,33,98]. Video diffusion methods include StableVSR [75], MGLD-VSR [114], DC-VSR [29], DOVE [15], UltraVSR [58], Upscale-A-Video [140], DiffVSR [48], DiffIR2VR-Zero [116], VideoGigaGAN [111], VEnhancer [30], temporal coherence [96], AVID [132], and SeedVR2 [97]. Temporal consistency in video diffusion has also been addressed through opticalflow-guided approaches [16,49,113]. Auto-regressive video diffusion has emerged

- as a promising paradigm for streaming generation. CausVid [121] distills a bidirectional video diffusion transformer into a causal auto-regressive generator with 4-step inference. Self Forcing [35] addresses exposure bias in auto-regressive video diffusion by conditioning on self-generated outputs during training. Diffusion Forcing [12] unifies next-token prediction with full-sequence diffusion through independent per-token noise levels. Other auto-regressive approaches [55, 87, 106, 133] and causal architectures [23, 44] further demonstrate the potential of streaming diffusion. Acceleration techniques include consistency models [2, 25, 31,43,61,84], advanced solvers [1,62,137], flow-based methods [41,57], adversarial distillation [53,80,81,110], distribution matching distillation [119,120,139], other distillation approaches [13,65,71,79,108,136,142,144], video-specific distillation [125], and efficient architectures [3]. Theoretical advances [94, 95] and recent image/offline distillation methods [86, 103, 104, 126] exist. In contrast, Stream-DiffVSR combines diffusion distillation with strictly online (past-only) causal temporal modeling to enable low-latency VSR.

### 3 Method

We propose Stream-DiffVSR, a streamable auto-regressive diffusion framework for efficient video super-resolution (VSR). Its core innovation lies in an auto-

regressive formulation that improves both temporal consistency and inference speed. The framework comprises: (1) a distilled few-step U-Net for accelerated diffusion inference, (2) Auto-regressive Temporal Guidance that conditions latent denoising on previously warped high-quality frames, and (3) an Auto-regressive Temporal-aware Decoder that explicitly incorporates temporal cues. Together, these components enable Stream-DiffVSR to produce stable and perceptually coherent videos.

#### 3.1 Diffusion Models Preliminaries

Diffusion Models [32] transform complex data distributions into simpler Gaussian distributions via a forward diffusion process and reconstruct the original data using a learned reverse denoising process. The forward process gradually adds Gaussian noise to the initial data x0, forming a Markov chain: q(xt | xt−1) = N xt;√1 − βt xt−1, βtI for t = 1,...,T, where βt denotes a predefined noise schedule. At timestep t, the noised data xt can be directly sampled from the clean data x0 as: xt = √αt x0 + √1 − αt ϵ, where ϵ ∼ N(0,I) and αt = ti=1(1 − βi), where αt = ti=1(1 − βi). The reverse process progressively removes noise from xT, reconstructing the original data x0 through a learned denoising operation modeled as a Markov chain, i.e., pθ(x0,...,xT−1 | xT) =

- T t=1 pθ(xt−1 | xt). Each individual step is parameterized by a neural network-

based denoising function pθ(xt−1 | xt) = N xt−1;µθ(xt,t),Σθ(t)I . Typically, the network predicts the noise component ϵθ(xt,t), from which the denoising

xt − 1−α

mean is estimated as µθ(xt,t) = √1α

√1−αtt ϵθ(xt,t) . Latent Diffusion Models (LDMs) [72] further reduce computational complexity by projecting data into a lower-dimensional latent space using Variational Autoencoders (VAEs), significantly accelerating inference without sacrificing generative quality.

t

#### 3.2 U-Net Rollout Distillation

We distill a pre-trained Stable Diffusion (SD) ×4 Upscaler [72, 73], originally designed for 50-step inference, into a 4-step variant that balances speed and perceptual quality. To mitigate the training–inference gap of timestep-sampling distillation, we adopt rollout distillation, where the U-Net performs the full 4step denoising each iteration to obtain a clean latent. Detailed algorithms and implementation are provided in the supplementary material due to page limits.

Unlike conventional distillation that supervises random intermediate timesteps, our method applies loss only on the final denoised latent, ensuring the training trajectory mirrors inference and improving stability and alignment.

Our distillation requires no architectural changes. We train the U-Net by optimizing latent reconstruction with a loss that balances spatial accuracy, perceptual fidelity, and realism:

Ldistill = ∥zden − zgt∥22

+ λLPIPS · LPIPS (D(zden), xgt)

+ λGAN · LGAN (D(zden)) ,

(1)

Warped HQ(t-2)

Warped HQ(t-1)

Encode stage

Spatial convolution

[Figure 20]

[Figure 21]

layer features

output

[Figure 22]

[Figure 23]

Avg-pooling Avg-pooling

down-

down-

Concatenation

sampling

sampling

[Figure 24]

[Figure 25]

TemporalLayer

ConV. 1D

ReLu

|Encode stage<br><br>layer features|
|---|

|Encode stage<br><br>layer features|
|---|

|Denoised current<br><br>latent| |
|---|---|
|Latent(t)<br><br>| |

|Denoised current<br><br>latent| |
|---|---|
|Latent(t)<br><br>| |

ConV. 1D

Warping

Lat

Lat

chunking

Current frame feature

Warped previous frame feature

[Figure 26]

[Figure 27]

Spatial convolution

Spatial convolution

× (1− 𝛼) × 𝛼

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Weighted Sum

Temporal processor

Temporal processor

[Figure 32]

[Figure 33]

Interpolation

Spatial convolution

Spatial convolution

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

up-

Temporal processor

Temporal processor

sampling

Weighted Sum

[Figure 38]

[Figure 39]

Spatial convolution

Spatial convolution

Output to next spatial

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

convolution

Temporal processor

Temporal processor

Decoded HQ(t-1)

Decoded HQ(t)

- Fig. 2: Overview of Auto-regressive Temporal-aware Decoder. Given the denoised latent and warped previous frame, our decoder enhances temporal consistency using temporal processor modules. This module aligns and fuses these features via interpolation, convolution, and weighted fusion, effectively stabilizing detail reconstruction when decoding into the final RGB frame.

where zden and zgt are the denoised and ground-truth latent representations. The decoder D(·) maps latent features back to RGB space for perceptual (LPIPS) and adversarial (GAN) loss calculations, encouraging visually realistic outputs.

#### 3.3 Auto-regressive Temporal Guidance

Leveraging temporal information is essential for capturing dynamics and ensuring frame continuity in video super-resolution. However, extensive temporal reasoning often incurs significant computational overhead, increasing per-frame inference time and system latency. Thus, efficient online VSR requires carefully balancing temporal utilization and computational cost to support low-latency processing.

To this end, we propose Auto-regressive Temporal Guidance (ARTG), which enforces temporal coherence during latent denoising. At each timestep t, the U-Net takes both the current noised latent zt and the warped RGB frame from the previous output, xˆwarpt−1 = Warp(xSRt−1,ft←t−1), where ft←t−1 is the optical flow from frame t−1 to t. The denoising prediction is then formulated as:

ϵˆθ = UNet(zt, t, xˆwarpt−1 ), (2)

where the warped image xˆwarpt−1 serves as temporal conditioning input to guide the denoising process.

We train the ARTG module independently using consecutive pairs of lowquality and high-quality frames. The denoising U-Net and decoder are kept fixed during this stage, and the training objective focuses on reconstructing the target latent representation while preserving perceptual quality and visual realism. The

Auto-regressive Temporal Guidance

Auto-regressive Temporal Guidance

Auto-regressive Temporal Guidance

Encoder

Encoder

Encoder

[Figure 44]

❄

❄

[Figure 45]

[Figure 46]

❄

❄

[Figure 47]

❄❄

[Figure 48]

[Figure 49]

🔥

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

❄

[Figure 50]

| | |
|---|---|

| | |
|---|---|

HQ(t)

HQ(t)

HQ(t)

❄

[Figure 51]

❄

[Figure 52]

❄

[Figure 53]

❄

[Figure 54]

[Figure 55]

🔥 🔥 🔥

❄

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

🔥

Auto-regressive Temporal Decoder

Auto-regressive Temporal Decoder

Auto-regressive Temporal Decoder

Denoising U-Net

Denoising U-Net

Denoising U-Net

Stage 1: Denoising U-Net

Stage 2: Temporal Processor Module

Stage 3: Auto-regressive Temporal Guidance

- Fig. 3: Training pipeline of Stream-DiffVSR. The training process consists of three sequential stages: (1) Distilling the denoising U-Net to reduce diffusion steps while maintaining perceptual quality with training objective (1); (2) Training the Temporal Processor Module (TPM) within the decoder to enhance temporal consistency

- at the RGB level with training objective (3); (3) Training the Auto-Regressive Temporal Guidance (ARTG) module to leverage previously restored high-quality frames for improved temporal coherence with training objective (6). Each module is trained separately before integrating them into the final framework.

total loss function is defined as:

LARTG = ∥zden − zgt∥22

(3)

+ λLPIPS · LPIPS(D(zden), xgt)

+ λGAN · LGAN(D(zden)),

where zden denotes the denoised latent from DDIM updates with predicted noise ˆϵθ, and zgt is the ground-truth latent. The decoder D(·) maps latents to RGB, producing D(zden) for comparison with the ground-truth image xgt. The latent ℓ2 loss enforces alignment, the perceptual loss preserves visual fidelity, and the adversarial loss promotes realism. This design leverages only past frames to propagate temporal context, improving consistency without additional latency.

#### 3.4 Auto-regressive Temporal-aware Decoder

Although the Auto-regressive Temporal Guidance (ARTG) improves temporal consistency in the latent space, the features produced by the Stable Diffusion ×4 Upscaler remain at one-quarter of the target resolution. This mismatch may introduce decoding artifacts or misalignment in dynamic scenes.

To address this issue, we propose an Auto-regressive Temporal-aware Decoder that incorporates temporal context into decoding to enhance spatial fidelity and temporal consistency. At timestep t, the decoder takes the denoised latent zdent and the aligned feature ˆft−1 derived from the previous super-resolved frame. Specifically, we compute:

xˆwarpt−1 = Warp(xSRt−1, ft←t−1), ˆft−1 = Enc(xˆwarpt−1 ), (4)

where xSRt−1 is the previously generated RGB output, ft←t−1 is the optical flow from frame t − 1 to t, and Enc(·) is a frozen encoder that projects the warped image into the latent feature space.

|↑ LQ(t-1)|
|---|

HQ(t-1)

|[Figure 61]<br><br>HQ(1)|
|---|

LQ(1)

|[Figure 62]|
|---|

Stream-DiffVSR

Encoder

|Flow Warping|
|---|

Auto-regressive Temporal Guidance

| | |
|---|---|
|[Figure 63]<br><br>motion|map|
| | |
|Motion Est.| |

t-2frames

| |
|---|

| |
|---|

| | |
|---|---|
| | |

x4 times

|[Figure 64]<br><br>HQ(t-1)|
|---|

LQ(t-1)

|[Figure 65]|
|---|

Stream-DiffVSR

[Figure 66]

Efficient

|↑ LQ(t)| |
|---|---|
| | |

processing

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

Latent

| |
|---|

| |
|---|

| |
|---|

| |
|---|

HQ(t)

|[Figure 67]|
|---|

|[Figure 68]<br><br>HQ(t)|
|---|

LQ(t)

|[Figure 69]|
|---|

Stream-DiffVSR

[Figure 70]

Init latent

Data Stream

Denoising U-Net

Auto-regressive Temporal Decoder

- Fig. 4: Overview of our pipeline. Given a low-quality (LQ) input frame, we first initialize its latent representation and employ an autoregressive diffusion model composed of a distilled denoising U-Net, autoregressive temporal Guidance, and an autoregressive temporal Decoder. Temporal guidance utilizes flow-warped high-quality (HQ) results from the previous frame to condition the current frame’s latent denoising and decoding processes, significantly improving perceptual quality and temporal consistency in an efficient, online manner.

##### The decoder then synthesizes the current frame using:

xSRt = Decoder(zdent ,ˆft−1). (5)

We adopt a multi-scale fusion strategy inside the decoder to combine current spatial information and prior temporal features across multiple resolution levels, as illustrated in Fig. 2. This design helps reinforce temporal coherence while recovering fine spatial details.

Temporal Processor Module (TPM). We integrate TPM after each spatial convolutional layer in the decoder to explicitly inject temporal coherence, enhancing stability and continuity of reconstructed frames. These modules utilize latent features from the current frame and warped features from the previous frame, optimizing temporal consistency independently from spatial reconstruction. Our training objective for the TPM is defined as:

LTPM = Lrec(xrect , xGTt )

+ λflow OF(xrect , xrect−1) − OF(xGTt , xGTt−1)

+ λGANLGAN(xrect )

+ λLPIPSLPIPS(xrect , xGTt ),

2 2

(6)

where xSRt ∈ R3×H×W is the predicted frame at time t, and xGTt is the groundtruth frame. The reconstruction loss Lrec = SmoothL1(xrect ,xGTt ) enforces spatial fidelity, the adversarial loss LGAN improves realism, and the optical-flow term OF(·,·) reduces temporal discrepancies, yielding consistent and perceptually faithful outputs.

- Table 2: Quantitative comparison against bidirectional/offline methods on the REDS4 dataset. We compare CNN-, Transformer-, and diffusion-based methods on REDS4. Stream-DiffVSR achieves superior perceptual and temporal quality with high stability across sequences. ↑ indicates higher is better; ↓ indicates lower is better. Dir. denotes temporal direction: B for bidirectional/offline, U for unidirectional/online. Runtime is measured per 720p frame on an RTX 4090. Latency-max denotes the maximum end-to-end latency measured over 100-frame video sequences, providing a fair comparison with offline methods whose initial delay scales with sequence length. tLP and tOF are scaled by 100× and 10×. Best and second-best results are marked in red and blue. Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ NIQE↓ NRQM↑ BRISQUE↓ VMAF↑ tLP↓ tOF↓ Runtime (s)↓ latency-max (s)↓

CNN-based Methods

- Bicubic 25.501 0.712 0.460 0.187 7.360 3.459 60.256 46.132 21.603 4.241 - B BasicVSR++ 32.386 0.907 0.132 0.069 3.850 6.363 38.641 99.580 9.017 2.490 0.098 9.8 B RealBasicVSR 27.042 0.778 0.134 0.065 2.530 6.769 18.046 88.197 6.422 4.759 0.064 6.4

Transformer-based Methods

B RVRT 32.701 0.911 0.130 0.067 3.793 6.366 38.038 99.681 9.133 2.421 0.498 2.49 B MIA-VSR 32.790 0.912 0.123 0.064 3.742 6.451 37.099 99.669 8.870 2.354 0.768 76.8

Diffusion-based Methods

B StableVSR 27.928 0.793 0.102 0.047 2.713 6.960 16.249 95.303 5.755 2.742 46.2 4620 B MGLD-VSR 26.53 0.749 0.151 0.065 2.972 6.701 15.291 78.255 18.139 5.910 43.6 218 U Ours 27.256 0.766 0.099 0.062 3.114 7.055 17.717 88.751 4.198 3.638 0.328 0.328

- Table 3: Quantitative comparison against unidirectional/online methods on the REDS4 dataset.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ VMAF↑ tLP↓ tOF↓ Runtime (s)↓ latency-max (s)↓ CNN-based Methods

- Bicubic 25.501 0.712 0.460 0.187 27.362 7.360 3.459 60.256 46.132 21.603 4.241 - U TMP 30.672 0.871 0.194 0.090 63.818 4.378 5.796 43.394 98.586 10.424 2.480 0.041 0.041

###### Transformer-based Methods

U RealViformer 26.763 0.761 0.129 0.065 64.585 2.731 7.028 17.272 60.509 11.261 4.037 0.099 9.9

###### Diffusion-based Methods

U StableVSR* 27.174 0.763 0.111 0.051 66.428 2.572 6.944 15.805 88.675 11.107 3.925 46.2 4620 U Ours 27.256 0.766 0.099 0.062 65.595 3.114 7.055 17.117 88.751 4.198 3.638 0.328 0.328

#### 3.5 Training and Inference Stages

Our training pipeline consists of three independent stages (Fig. 3), while our inference process and the Auto-Regressive Diffusion-based VSR algorithm are illustrated in Fig. 4 and detailed in the appendix due to page constraints, respectively.

Distilling the Denoising U-Net. We first distill the denoising U-Net using pairs of low-quality (LQ) and high-quality (HQ) frames to optimize per-frame superresolution and latent-space consistency.

Training the Temporal Processor Module (TPM). In parallel, we train the Temporal Processor Module (TPM) in the decoder using ground-truth frames, keeping all other weights fixed. This enhances the decoder’s capability to incorporate temporal information into the final RGB reconstruction.

Training Auto-regressive Temporal Guidance. After training and freezing the

- U-Net and decoder, we train the ARTG, which leverages flow-aligned previous outputs to enhance temporal coherence without degrading spatial quality. This

###### Table 4: Quantitative comparison against DiT methods on the REDS4 dataset. Runtime is measured on an RTX Pro 6000.Gray-shaded entries are reported from FlashVSR; “–” indicates unavailable results.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ VMAF↑ tLP↓ tOF↓ Runtime (s)↓ latency-max (s)↓ Peak Mem (GB)

- B UAV 23.077 0.608 0.495 0.249 30.912 5.917 3.666 39.751 20.600 30.546 13.582 6.017 601.740 57.094

- B UAV 24.840 0.644 0.41 – 53.000 3.104 – – – – – – – – B SeedVR2 22.774 0.661 0.246 0.111 65.011 2.812 6.707 29.948 64.447 30.296 12.79 0.479 47.840 69.592 B SeedVR2 24.830 0.704 0.312 – 61.830 3.066 – – – – – – – – B DOVE 25.027 0.692 0.291 0.149 54.060 4.087 5.753 25.606 58.329 10.260 10.960 0.760 75.983 42.208 B VEnhancer 20.712 0.560 0.384 0.156 56.932 4.142 5.567 24.906 24.743 32.913 16.643 0.672 671.650 53.200 U FlashVSR 21.484 0.569 0.283 0.124 62.178 6.574 6.574 25.821 40.556 15.69 20.548 0.283 1.698 19.630 U FlashVSR 23.920 0.649 0.343 0.124 68.97 2.425 6.574 25.821 40.556 15.69 20.548 0.283 1.698 19.630 U FlashVSR-tiny 21.887 0.577 0.300 0.141 55.204 3.463 6.188 28.589 38.634 17.838 23.992 0.157 0.942 12.390 U FlashVSR-tiny 24.110 0.651 0.343 – 67.43 2.680 – – – – – – – – U Ours 27.256 0.766 0.098 0.062 65.595 3.111 7.056 17.667 88.751 4.198 3.638 0.243 0.243 19.580

###### Table 5: Quantitative comparison against bidirectional/offline methods on the Vimeo-90K-T dataset. Stream-DiffVSR surpasses other bidirectional methods in perceptual quality, temporal consistency, and runtime. Runtime is the average perframe inference time (seconds) on 448×256 videos using an RTX 4090. Best and secondbest results are shown in red and blue.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ VMAF↑ tLP↓ tOF↓ Runtime (s)↓ latency-max (s)↓

###### CNN-based Methods

- Bicubic 29.282 0.864 0.297 0.209 23.433 8.735 3.588 61.714 42.928 11.606 2.49 - B BasicVSR++ 37.479 0.956 0.098 0.117 51.940 7.077 5.509 47.792 92.905 4.691 1.57 0.012 0.084 B RealBasicVSR 29.388 0.857 0.156 0.149 56.986 5.069 7.413 23.822 79.781 10.947 3.46 0.008 0.056

Transformer-based Methods

B RVRT 37.815 0.955 0.093 0.105 49.937 7.205 5.393 48.352 94.660 4.873 1.429 0.061 0.305 B MIA-VSR 37.598 0.957 0.086 0.101 51.402 7.116 5.569 47.865 95.113 4.696 1.419 0.096 0.672

Diffusion-based Methods

B StableVSR 31.823 0.878 0.095 0.111 54.582 4.745 7.265 20.039 86.936 26.224 3.108 5.749 40.243 B MGLD-VSR 29.651 0.865 0.151 0.137 57.788 5.340 7.217 20.761 71.509 12.550 4.661 5.426 27.130 U Ours 32.593 0.900 0.056 0.105 52.755 4.403 7.672 29.297 88.311 4.307 2.689 0.041 0.041

staged training strategy progressively refines spatial fidelity, latent consistency, and temporal smoothness in a decoupled manner.

Inference. Given a sequence of low-quality (LQ) frames, our method autoregressively generates high-quality (HQ) outputs. For each frame t, denoising is conditioned on the previous output HQt−1, warped via optical flow to capture temporal motion. To balance quality and efficiency, we employ a 4-step DDIM scheme using a distilled U-Net. By combining motion alignment with reduced denoising steps, our inference pipeline achieves efficient and stable temporal consistency.

### 4 Experiment

Due to space limitations, we provide the experimental setup in the appendix.

We quantitatively compare Stream-DiffVSR with state-of-the-art VSR methods on REDS4 [67], Vimeo-90K-T [112], VideoLQ [8], and Vid4 [54], covering diverse scene content and motion characteristics. Tabs. 2, 5, 8 and 9 report results across CNN-, Transformer-, and diffusion-based approaches under both bidirectional (offline) and unidirectional (online) settings. In addition, Tabs. 4 and 7 provide supplementary comparisons on REDS4 and Vimeo-90K-T against several memory-intensive baselines, further highlighting Stream-DiffVSR’s quality–latency trade-off under practical memory budgets.

###### Table 6: Quantitative comparison against unidirectional/online methods on the Vimeo-90K-T dataset.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ VMAF↑ tLP↓ tOF↓ Runtime (s)↓ latency-max (s)↓

CNN-based Methods

- Bicubic 29.282 0.864 0.297 0.209 23.433 8.735 3.588 61.714 42.928 11.606 2.49 - U TMP 36.482 0.946 0.109 0.118 48.374 7.368 5.096 49.192 92.001 4.870 1.603 0.006 0.006

Transformer-based Methods

U RealViformer 30.291 0.877 0.130 0.140 53.107 5.515 6.711 24.628 54.689 8.232 2.769 0.013 0.091

Diffusion-based Methods

U StableVSR* 31.729 0.875 0.072 0.113 54.447 4.698 7.280 19.836 86.162 30.858 3.144 5.749 40.243 U Ours 32.593 0.900 0.056 0.105 52.755 4.403 7.672 29.297 88.311 4.307 2.689 0.041 0.041

###### Table 7: Quantitative comparison with memory-intensive baselines on Vimeo-90K-T under a single RTX Pro 6000.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ VMAF↑ tLP↓ tOF↓ Runtime (s)↓ latency-max (s)↓ Peak Mem (GB)

B UAV 25.263 0.72 0.226 0.177 55.562 5.602 7.247 24.766 46.489 9.546 7.576 0.499 3.490 37.987 B SeedVR2 24.585 0.719 0.177 0.152 58.809 4.204 8.107 20.447 41.430 21.191 12.460 0.039 0.270 41.366 B DOVE 28.611 0.834 0.12 0.116 57.749 5.734 7.182 23.388 77.997 8.214 4.387 0.223 1.560 31.934 B VEnhancer 23.712 0.712 0.234 0.189 57.822 6.175 7.910 34.906 60.920 32.913 16.643 1.834 12.840 35.855 U FlashVSR 26.114 0.789 0.136 0.118 59.451 5.249 7.317 29.288 78.575 19.018 9.351 0.025 0.150 5.340 U FlashVSR-tiny 26.154 0.791 0.136 0.121 57.845 5.249 7.317 29.288 76.376 19.881 9.793 0.019 0.114 4.280 U Ours 32.593 0.900 0.056 0.105 52.755 4.403 7.672 29.267 88.311 4.307 2.689 0.036 0.036 9.720

On REDS4, Stream-DiffVSR achieves strong perceptual quality (LPIPS=0.099)

compared with representative CNN (BasicVSR++ [7], RealBasicVSR [8]), Transformer (RVRT [52]), and diffusion-based baselines (StableVSR [75], MGLDVSR [114]), while maintaining competitive temporal consistency (tLP=4.198, tOF=3.638). Importantly, these results are obtained with substantially lower runtime (0.328s/frame vs. 43–46s/frame for diffusion-based baselines), matching the low-latency objective.

On Vimeo-90K-T, Stream-DiffVSR similarly attains favorable perceptual performance (LPIPS=0.056, DISTS=0.105) and improved temporal consistency (tLP=4.307, tOF=2.689) with a runtime of 0.041s/frame, supporting efficient causal inference for online/streaming usage.

In addition to speed, Stream-DiffVSR achieves a markedly lower memory footprint. We emphasize that Stream-DiffVSR performs strictly frame-by-frame causal inference, whereas FlashVSR [144] adopts chunk-wise streaming. Consequently, FlashVSR’s per-frame runtime can appear low, but its end-to-end latency is bounded by chunk buffering, while Stream-DiffVSR avoids sequencelevel waiting and provides immediate frame-level response. As shown in Tabs. 4, 7 and 9, prior diffusion-based VSR methods such as DOVE [15], SeedVR2 [97], and Upscale-A-Video(UAV) [140] often require large GPU memory budgets (e.g., exceeding 42GB on REDS4/VideoLQ under our settings) or run into OOM on single RTX 4090. In contrast, Stream-DiffVSR operates within 19.58GB while running more than 2.5× faster, underscoring its efficiency and deployability.

Results on VideoLQ and Vid4 further show consistent perceptual and temporal performance, indicating stable behavior across diverse content and motion patterns.

###### Table 8: Quantitative comparison on the VideoLQ dataset. (a) Bidirectional/Offline

###### (b) Unidirectional/Online

Dir. Method NIQE↓ NRQM↑ BRISQUE↓ CNN-based Methods

Dir. Method NIQE↓ NRQM↑ BRISQUE↓ CNN-based Methods

- Bicubic 7.945 3.151 57.944 B BasicVSR++ 5.909 3.745 56.800 B RealBasicVSR 3.973 6.095 30.158

- Bicubic 7.945 3.151 57.944 U TMP 6.751 3.511 59.841

Transformer-based Methods

###### Transformer-based Methods

B RVRT 6.939 3.493 60.557 B MIA-VSR 5.860 3.810 58.513

U RealViformer 4.070 6.066 28.266

Diffusion-based Methods

###### Diffusion-based Methods

B StableVSR 3.973 6.154 22.973 B MGLD-VSR 4.163 5.761 29.497

U StableVSR* 3.982 6.122 23.814 U Ours 3.929 6.140 23.176

U Ours 3.929 6.140 23.176

###### Table 9: Quantitative comparison with memory-intensive baselines on the VideoLQ dataset under a single RTX Pro 6000. Gray-shaded entries are reported from FlashVSR. – indicates unavailable results.

Dir. Method NIQE↓ NRQM↑ BRISQUE↓ Runtime (s) Latency-max (s) Peak Mem (GB) B VEhancer 6.221 3.85 48.1 9.544 477.237 47.985

- B SeedVR2 4.661 5.523 37.975 1.126 56.28 76.094

- B SeedVR2 5.205 – – – – – B UAV 6.299 3.652 44.139 8.081 404.07 55.897 B UAV 4.889 – – – – – B DOVE 5.090 5.214 36.631 1.735 86.774 46.344 U FlashVSR – – – – – OOM U FlashVSR 3.803 – – – – – U FlashVSR-tiny 4.569 5.164 42.514 0.204 1.224 44.180 U FlashVSR-tiny 4.070 – – – – – U Ours 3.929 6.140 27.176 0.454 0.454 22.800

#### 4.1 Qualitative Comparisons

We provide qualitative comparisons in Fig. 5, where Stream-DiffVSR generates sharper details and fewer artifacts than prior methods. Additional visualizations of temporal consistency and flow coherence are included in the supplemental material. A qualitative comparison with DOVE, SeedVR2, Upscale-A-Video (UAV) is included in the appendix and supplementary.

#### 4.2 Ablation Study

We ablate key components of Stream-DiffVSR including denoising-step reduction, ARTG, TPM, timestep selection, and training-stage combinations on REDS4 to ensure consistent evaluation of perceptual quality and temporal stability.

We perform ablation studies on training strategies in Tab. 13 and Tab. 11. For stage-wise training, partial or joint training yields inferior results, while our separate stage-wise scheme achieves the best trade-off across fidelity, perceptual, and temporal metrics. For distillation, rollout training outperforms random timestep selection in both quality and efficiency, reducing training cost from 60.5 to 21 GPU hours on 4×A6000 GPUs.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

UnidirectionalBidirectional

| |
|---|

StableVSR RVRT RealBasicVSR MGLD-VSR

BasicVSR++

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Input/Ours Ours GT

StableVSR* TMP RealViFromer

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

UnidirectionalBidirectional

StableVSR RVRT RealBasicVSR MGLD-VSR

BasicVSR++

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

| |
|---|

StableVSR* TMP RealViFromer

Input/Ours Ours GT

- Fig. 5: Qualitative comparison on REDS4 and Vimeo-90K-T datasets. Our method demonstrates superior visual quality with sharper details compared to unidirectional methods (TMP [131], RealViformer [130]) and competitive performance against bidirectional methods (StableVSR [75], MGLD-VSR [114], RVRT [52], BasicVSR++ [7], RealBasicVSR [8]). Improvements include reduced artifacts and enhanced temporal stability (see zoomed patches).

###### Table 10: Ablation study of temporal modules in Stream-DiffVSR.

Component LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ tLP↓ tOF↓ WarpErr↓

Per-frame 0.099 0.071 65.981 3.249 6.969 21.655 7.261 4.201 25.668 w/o ARTG 0.117 0.070 63.347 3.194 6.980 19.027 6.132 3.910 16.598 w/o TPM 0.116 0.078 67.110 3.197 7.007 20.279 12.847 4.639 21.990 TPM (unwarped) 0.122 0.082 63.849 3.201 7.159 14.063 12.846 5.689 17.143 Ours 0.099 0.062 65.586 3.111 7.256 17.667 4.265 3.620 14.909

We assess the runtime–quality trade-off by varying DDIM inference steps while keeping model weights fixed. As shown in Tab. 12 and Fig. 7, fewer steps increase efficiency but reduce perceptual quality, whereas more steps improve fidelity with higher latency. A 4-step setting provides the best balance.

Tab. 10 and Fig. 8 show the effectiveness of ARTG and TPM. The per-frame baseline uses only the distilled U-Net with both ARTG and TPM disabled. In the ablation labels, w/o indicates that a module is fully removed; for instance, TPM (unwarp) feeds TPM the previous HR frame without flow-based warping, removing motion alignment. ARTG improves perceptual quality (LPIPS 0.117→0.099) and temporal consistency (tLP100 6.132→4.265). TPM further enhances temporal coherence through temporal-feature warping and fusion, yielding additional gains in tLP100. These results highlight the complementary roles of latent-space guidance and decoder-side temporal modeling.

###### Table 11: Ablation study on training strategy.

Stage combination PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ tLP↓ tOF↓ WarpErr↓

- stage 1 and 2 25.442 0.702 0.156 0.100 67.528 21.781 6.37 27.307
- stage 1 and 3 26.307 0.753 0.121 0.077 64.902 13.094 4.09 21.689 stage 2 and 3 26.906 0.758 0.132 0.077 64.751 10.510 4.225 15.726 All stage jointly 26.135 0.736 0.124 0.073 67.35 17.816 4.596 24.298 Sperate (Ours) 27.256 0.766 0.099 0.062 65.586 4.265 3.620 14.909

- Table 12: Ablation study on denoising step count within Stream-DiffVSR. We evaluate 50, 10, 1, and 4 steps. Our 4-step design achieves a favorable balance between perceptual quality and runtime.

Step(s) LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ tLP↓ tOF↓ Runtime (s)↓

50 0.102 0.068 66.061 2.804 7.026 9.925 18.798 3.826 3.460 10 0.122 0.072 64.900 2.869 6.917 12.461 9.990 3.625 0.718 1 0.138 0.076 63.915 3.843 6.984 29.552 9.899 3.882 0.106

- 4 (Ours) 0.099 0.062 65.586 3.111 7.056 17.667 4.265 3.620 0.328

- 5 Conclusion

We propose Stream-DiffVSR, an efficient online video super-resolution framework using diffusion models. By integrating a distilled U-Net, Auto-Regressive Temporal Guidance, and Temporal-aware Decoder, Stream-DiffVSR achieves superior perceptual quality, temporal consistency, and practical inference speed for low-latency applications.

Limitations. Stream-DiffVSR remains heavier than CNN and Transformer models, and its use of optical flow can introduce artifacts under fast motion. Its autoregressive design may also degrade the earliest frames, suggesting the need for stronger initialization to reduce cold-start effects in streaming inference. Improving robustness to diverse and unknown degradations is an important direction for future work.

### Acknowledgements

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-E-A49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

### References

1. Lu et al., C.: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. NeurIPS 35, 5775–5787 (2022)

###### Table 13: Ablation study on Rollout Training. Comparison of random timestep distillation vs. rollout training across fidelity and perceptual metrics.

###### Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ GPU Hours↓ Random Timestep Selection 26.27 0.743 0.099 0.071 65.981 60.5 Rollout Distillation 26.36 0.753 0.095 0.075 66.391 21

[Figure 93]

[Figure 94]

[Figure 95]

W/OwarpingW/warpingPer-frame

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

- Fig. 6: Ablation study on the Temporal Processor Module (TPM). Integrating TPM improves motion stability and reduces temporal artifacts by leveraging warped previous-frame features, enhancing temporal consistency in video super-resolution.

| |
|---|

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

1-step 4-steps 10-steps

[Figure 106]

[Figure 107]

[Figure 108]

50-steps Finetuned 4-steps GT

Low resolution input

- Fig. 7: Ablation study on inference steps. The 4-step model yields the best quality–efficiency trade-off, validating our distillation strategy.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Low resolution input

[Figure 116]

[Figure 117]

W/O ART

With ART

- Fig. 8: Ablation study on Autoregressive Temporal Guidance (ARTG). ARTG enhances temporal consistency and perceptual quality by leveraging warped previous frames, reducing flickering, and improving structural coherence.

- 2. Luo et al., S.: Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378 (2023)
- 3. Bai, W., Xu, S., Ren, Y., Hao, J., Sun, M., Chen, W., Sun, H.: Instantvir: Realtime video inverse problem solver with distilled diffusion prior. arXiv preprint arXiv:2511.14208 (2025)
- 4. Blau, Y., Michaeli, T.: The perception-distortion tradeoff. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 6228–6237

(2018)

- 5. Cao, Y., Wang, C., Song, C., Tang, Y., Li, H.: Real-time super-resolution system of 4k-video based on deep learning. In: 2021 IEEE 32nd International Conference on Application-specific Systems, Architectures and Processors (ASAP). pp. 69–

76. IEEE (2021)

- 6. Chan, K.C., Wang, X., Yu, K., Dong, C., Loy, C.C.: Basicvsr: The search for essential components in video super-resolution and beyond. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4947–4956 (2021)
- 7. Chan, K.C., Zhou, S., Xu, X., Loy, C.C.: Basicvsr++: Improving video superresolution with enhanced propagation and alignment. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5972– 5981 (2022)

- 8. Chan, K.C., Zhou, S., Xu, X., Loy, C.C.: Investigating tradeoffs in real-world video super-resolution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5962–5971 (2022)
- 9. Chang, K.C., Wang, R., Lin, H.J., Liu, Y.L., Chen, C.P., Chang, Y.L., Chen, H.T.: Learning camera-aware noise models. In: European Conference on Computer Vision. pp. 343–358. Springer (2020)
- 10. Chao, C.H., Sun, W.F., Cheng, B.W., Lo, Y.C., Chang, C.C., Liu, Y.L., Chang, Y.L., Chen, C.P., Lee, C.Y.: Denoising likelihood score matching for conditional score-based data generation. arXiv preprint arXiv:2203.14206 (2022)
- 11. Chen, B., Li, G., Wu, R., Zhang, X., Chen, J., Zhang, J., Zhang, L.: Adversarial diffusion compression for real-world image super-resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 28208– 28220 (2025)
- 12. Chen, B., Martí Monsó, D., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems 37, 24081–24125 (2024)
- 13. Chen, J., Xue, S., Zhao, Y., Yu, J., Paul, S., Chen, J., Cai, H., Han, S., Xie, E.: Sana-sprint: One-step diffusion with continuous-time consistency distillation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16185–16195 (2025)
- 14. Chen, R., Mu, Y., Zhang, Y.: High-order relational generative adversarial network for video super-resolution. Pattern Recognition 146, 110059 (2024)
- 15. Chen, Z., Zou, Z., Zhang, K., Su, X., Yuan, X., Guo, Y., Zhang, Y.: Dove: Efficient one-step diffusion model for real-world video super-resolution. arXiv preprint arXiv:2505.16239 (2025)
- 16. Chu, E., Huang, T., Lin, S.Y., Chen, J.C.: Medm: Mediating image diffusion models for video-to-video translation with temporal correspondence guidance. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 1353– 1361 (2024)
- 17. Conde, M.V., Lei, Z., Li, W., Bampis, C., Katsavounidis, I., Timofte, R., Luo, Q., Song, J., Jiang, L., Lei, H., et al.: Aim 2024 challenge on efficient video superresolution for av1 compressed content. In: European Conference on Computer Vision. pp. 304–325. Springer (2024)
- 18. Dai, J., Qi, H., Xiong, Y., Li, Y., Zhang, G., Hu, H., Wei, Y.: Deformable convolutional networks. In: Proceedings of the IEEE international conference on computer vision (ICCV). pp. 764–773 (2017)
- 19. Ding, K., Ma, K., Wang, S., Simoncelli, E.P.: Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence 44(5), 2567–2581 (2020)
- 20. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12873–12883 (2021)
- 21. Fuoli, D., Danelljan, M., Timofte, R., Van Gool, L.: Fast online video superresolution with deformable attention pyramid. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 1735–1744 (2023)
- 22. Fuoli, D., Gu, S., Timofte, R.: Efficient video super-resolution through recurrent latent space propagation. In: 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW). pp. 3476–3485. IEEE (2019)
- 23. Gao, K., Shi, J., Zhang, H., Wang, C., Xiao, J., Chen, L.: Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375 (2024)

- 24. Gao, S., Liu, X., Zeng, B., Xu, S., Li, Y., Luo, X., Liu, J., Zhen, X., Zhang, B.: Implicit diffusion models for continuous super-resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10021– 10030 (2023)
- 25. Geng, Z., Pokle, A., Luo, W., Lin, J., Kolter, J.Z.: Consistency models made easy. arXiv preprint arXiv:2406.14548 (2024)
- 26. Ghasemabadi, A., Janjua, M.K., Salameh, M., Niu, D.: Learning truncated causal history model for video restoration. Advances in Neural Information Processing Systems 37, 27584–27615 (2024)
- 27. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Communications of the ACM 63(11), 139–144 (2020)
- 28. Guo, Z., Li, W., Loy, C.C.: Generalizable implicit motion modeling for video frame interpolation. Advances in Neural Information Processing Systems 37, 63747– 63770 (2024)
- 29. Han, J., Sim, G., Kim, G., Lee, H.S., Choi, K., Han, Y., Cho, S.: Dc-vsr: Spatially and temporally consistent video super-resolution with video diffusion prior. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. pp. 1–11 (2025)
- 30. He, J., Xue, T., Liu, D., Lin, X., Gao, P., Lin, D., Qiao, Y., Ouyang, W., Liu, Z.: Venhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667 (2024)
- 31. Heek, J., Hoogeboom, E., Salimans, T.: Multistep consistency models. arXiv preprint arXiv:2403.06807 (2024)
- 32. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
- 33. Ho, J., Saharia, C., Chan, W., Fleet, D.J., Norouzi, M., Salimans, T.: Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research 23(47), 1–33 (2022)
- 34. Hsiao, C.W., Liu, Y.L., Yang, C.K., Kuo, S.P., Jou, K., Chen, C.P.: Ref-ldm: A latent diffusion model for reference-based face image restoration. Advances in Neural Information Processing Systems 37, 74840–74867 (2024)
- 35. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009

(2025)

- 36. Isobe, T., Jia, X., Gu, S., Li, S., Wang, S., Tian, Q.: Video super-resolution with recurrent structure-detail network. arXiv preprint arXiv:2008.00455 (2020)
- 37. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1125–1134 (2017)
- 38. Jeelani, M., Cheema, N., Illgner-Fehns, K., Slusallek, P., Jaiswal, S., et al.: Expanding synthetic real-world degradations for blind video super resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1199–1208 (2023)
- 39. Ji, X., Cao, Y., Tai, Y., Wang, C., Li, J., Huang, F.: Real-world super-resolution via kernel estimation and noise injection. In: proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. pp. 466–467 (2020)
- 40. Jiang, Y., Nawała, J., Feng, C., Zhang, F., Zhu, X., Sole, J., Bull, D.: Rtsr: A real-time super-resolution model for av1 compressed content. In: 2025 IEEE International Symposium on Circuits and Systems (ISCAS). pp. 1–5. IEEE (2025)

- 41. Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., Lin, Z.: Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954 (2024)
- 42. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5148–5157 (2021)
- 43. Kim, D., Lai, C.H., Liao, W.H., Murata, N., Takida, Y., Uesaka, T., He, Y., Mitsufuji, Y., Ermon, S.: Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279 (2023)
- 44. Kim, J., Kang, J., Choi, J., Han, B.: Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems 37, 89834–89868 (2024)
- 45. Lei, C., Xing, Y., Chen, Q.: Blind video temporal consistency via deep video prior. Advances in Neural Information Processing Systems 33, 1083–1093 (2020)
- 46. Li, H., Yang, Y., Chang, M., Chen, S., Feng, H., Xu, Z., Li, Q., Chen, Y.: Srdiff: Single image super-resolution with diffusion probabilistic models. Neurocomputing 479, 47–59 (2022)
- 47. Li, W., Tao, X., Guo, T., Qi, L., Lu, J., Jia, J.: Mucan: Multi-correspondence aggregation network for video super-resolution. arXiv preprint arXiv:2007.11803

(2020)

- 48. Li, X., Liu, Y., Cao, S., Chen, Z., Zhuang, S., Chen, X., He, Y., Wang, Y., Qiao, Y.: Diffvsr: Enhancing real-world video super-resolution with diffusion models for advanced visual quality and temporal consistency. arXiv preprint arXiv:2501.10110

(2025)

- 49. Liang, F., Wu, B., Wang, J., Yu, L., Li, K., Zhao, Y., Misra, I., Huang, J.B., Zhang, P., Vajda, P., et al.: Flowvid: Taming imperfect optical flows for consistent videoto-video synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8207–8216 (2024)
- 50. Liang, J., Cao, J., Fan, Y., Zhang, K., Ranjan, R., Li, Y., Timofte, R., Van Gool, L.: Vrt: A video restoration transformer. arXiv preprint arXiv:2201.12288 (2022)
- 51. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1833–1844 (2021)
- 52. Liang, J., Fan, Y., Xiang, X., Ranjan, R., Ilg, E., Green, S., Cao, J., Zhang, K., Timofte, R., Gool, L.V.: Recurrent video restoration transformer with guided deformable attention. Advances in Neural Information Processing Systems 35, 378–393 (2022)
- 53. Lin, S., Wang, A., Yang, X.: Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929 (2024)
- 54. Liu, C., Sun, D.: On bayesian adaptive video super resolution. IEEE transactions on pattern analysis and machine intelligence 36(2), 346–360 (2013)
- 55. Liu, H., Liu, S., Zhou, Z., Xu, M., Xie, Y., Han, X., Pérez, J.C., Liu, D., Kahatapitiya, K., Jia, M., et al.: Mardini: Masked autoregressive diffusion for video generation at scale. arXiv preprint arXiv:2410.20280 (2024)
- 56. Liu, K.H., Yang, C.K., Chen, M.H., Liu, Y.L., Lin, Y.Y.: Corrfill: Enhancing faithfulness in reference-based inpainting with correspondence guidance in diffusion models. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1618–1627. IEEE (2025)
- 57. Liu, X., Zhang, X., Ma, J., Peng, J., et al.: Instaflow: One step is enough for highquality diffusion-based text-to-image generation. In: The Twelfth International Conference on Learning Representations (2023)

- 58. Liu, Y., Pan, J., Li, Y., Dong, Q., Zhu, C., Guo, Y., Wang, F.: Ultravsr: Achieving ultra-realistic video super-resolution with efficient one-step diffusion space. arXiv preprint arXiv:2505.19958 (2025)
- 59. Liu, Y.L., Lai, W.S., Yang, M.H., Chuang, Y.Y., Huang, J.B.: Learning to see through obstructions with layered decomposition. IEEE transactions on pattern analysis and machine intelligence 44(11), 8387–8402 (2021)
- 60. Liu, Y.L., Liao, Y.T., Lin, Y.Y., Chuang, Y.Y.: Deep video frame interpolation using cyclic frame generation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 33, pp. 8794–8802 (2019)
- 61. Lu, C., Song, Y.: Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081 (2024)
- 62. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research pp. 1–22 (2025)
- 63. Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L.: Repaint: Inpainting using denoising diffusion probabilistic models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11461–11471 (2022)
- 64. Ma, C., Yang, C.Y., Yang, X., Yang, M.H.: Learning a no-reference quality metric for single-image super-resolution. Computer Vision and Image Understanding 158, 1–16 (2017)
- 65. Meng, C., Rombach, R., Gao, R., Kingma, D., Ermon, S., Ho, J., Salimans, T.: On distillation of guided diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14297–14306 (2023)
- 66. Mittal, A., Moorthy, A.K., Bovik, A.C.: No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing 21(12), 4695–4708

(2012)

- 67. Nah, S., Baik, S., Hong, S., Moon, G., Son, S., Timofte, R., Lee, K.M.: Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study. In: CVPR Workshops (June 2019)
- 68. Nah, S., Baik, S., Hong, S., Moon, G., Son, S., Timofte, R., Mu Lee, K.: Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study. In: CVPRW (2019)
- 69. Pan, J., Bai, H., Dong, J., Zhang, J., Tang, J.: Deep blind video super-resolution. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4811–4820 (2021)
- 70. Qu, Z., Jiang, X., Yang, Y., Li, D., Zhao, C.: Online video quality enhancement with spatial-temporal look-up tables. In: European Conference on Computer Vision. pp. 449–465. Springer (2024)
- 71. Ren, Y., Xia, X., Lu, Y., Zhang, J., Wu, J., Xie, P., Wang, X., Xiao, X.: Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. Advances in neural information processing systems 37, 117340–117362 (2024)
- 72. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 73. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10684– 10695 (June 2022)
- 74. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models (2021)

- 75. Rota, C., Buzzelli, M., van de Weijer, J.: Enhancing perceptual quality in video super-resolution through temporally-consistent detail synthesis using diffusion models. In: European Conference on Computer Vision. pp. 36–53. Springer (2024)
- 76. Saad, M.A., Bovik, A.C.: Blind quality assessment of videos using a model of natural scene statistics and motion coherency. In: 2012 Conference Record of the Forty Sixth Asilomar Conference on Signals, Systems and Computers (ASILOMAR). pp. 332–336. IEEE (2012)
- 77. Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M.: Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence 45(4), 4713–4726 (2022)
- 78. Sajjadi, M.S., Vemulapalli, R., Brown, M.: Frame-recurrent video superresolution. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6626–6634 (2018)
- 79. Salimans, T., Ho, J.: Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512 (2022)
- 80. Sauer, A., Boesel, F., Dockhorn, T., Blattmann, A., Esser, P., Rombach, R.: Fast high-resolution image synthesis with latent adversarial diffusion distillation. In: SIGGRAPH Asia 2024 Conference Papers. pp. 1–11 (2024)
- 81. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial diffusion distillation. In: European Conference on Computer Vision. pp. 87–103. Springer (2024)
- 82. Shi, S., Gu, J., Xie, L., Wang, X., Yang, Y., Dong, C.: Rethinking alignment in video super-resolution transformers. arXiv preprint arXiv:2207.08494 (2022)
- 83. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)
- 84. Song, Y., Dhariwal, P.: Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189 (2023)
- 85. Song, Y., Wang, M., Yang, Z., Xian, X., Shi, Y.: Negvsr: Augmenting negatives for generalized noise modeling in real-world video super-resolution. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 10705–10713 (2024)
- 86. Sun, L., Wu, R., Ma, Z., Liu, S., Yi, Q., Zhang, L.: Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach (2025)
- 87. Sun, M., Wang, W., Li, G., Liu, J., Sun, J., Feng, W., Lao, S., Zhou, S., He, Q., Liu, J.: Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 7364–7373 (2025)
- 88. Sun, Y.C., Yeo, C.Y., Chu, E., Chen, J.C., Liu, Y.L.: Fiper: Factorized features for robust image super-resolution and compression. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)
- 89. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. pp. 402–419. Springer (2020)
- 90. Tian, Y., Zhang, Y., Fu, Y., Xu, C.: Tdan: Temporally-deformable alignment network for video super-resolution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3360–3369 (2020)
- 91. Tsai, S.R., Chang, W.C., Lee, J.Y., Su, C.H., Liu, Y.L.: Lightsout: Diffusion-based outpainting for enhanced lens flare removal. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6353–6363 (2025)
- 92. Tsai, Y.J., Liu, Y.L., Qi, L., Chan, K.C., Yang, M.H.: Dual associated encoder for face restoration. arXiv preprint arXiv:2308.07314 (2023)

- 93. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)
- 94. Wang, F.Y., Huang, Z., Bergman, A., Shen, D., Gao, P., Lingelbach, M., Sun, K., Bian, W., Song, G., Liu, Y., et al.: Phased consistency models. Advances in neural information processing systems 37, 83951–84009 (2024)
- 95. Wang, F.Y., Yang, L., Huang, Z., Wang, M., Li, H.: Rectified diffusion: Straightness is not your need in rectified flow. arXiv preprint arXiv:2410.07303 (2024)
- 96. Wang, H., Liu, Y., Liu, H., Wang, C.C., Guo, Y., Li, H., Wang, B., Sun, J.: Temporal-consistent video restoration with pre-trained diffusion models. arXiv preprint arXiv:2503.14863 (2025)
- 97. Wang, J., Lin, S., Lin, Z., Ren, Y., Wei, M., Yue, Z., Zhou, S., Chen, H., Zhao, Y., Yang, C., et al.: Seedvr2: One-step video restoration via diffusion adversarial post-training. arXiv preprint arXiv:2506.05301 (2025)
- 98. Wang, J., Yue, Z., Zhou, S., Chan, K.C., Loy, C.C.: Exploiting diffusion prior for real-world image super-resolution. International Journal of Computer Vision 132(12), 5929–5949 (2024)
- 99. Wang, X., Chan, K.C., Yu, K., Dong, C., Change Loy, C.: Edvr: Video restoration with enhanced deformable convolutional networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 0–0 (2019)
- 100. Wang, X., Xie, L., Dong, C., Shan, Y.: Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In: International Conference on Computer Vision Workshops (ICCVW) (2021)
- 101. Wang, Y., Yang, W., Chen, X., Wang, Y., Guo, L., Chau, L.P., Liu, Z., Qiao, Y., Kot, A.C., Wen, B.: Sinsr: diffusion-based image super-resolution in a single step. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 25796–25805 (2024)
- 102. Weng, S., Zheng, H., Zhan, P., Hong, Y., Jiang, H., Li, S., Shi, B.: Vires: Video instance repainting with sketch and text guidance. arXiv preprint arXiv:2411.16199

(2024)

- 103. Wu, R., Sun, L., Ma, Z., Zhang, L.: One-step effective diffusion network for realworld image super-resolution. arXiv preprint arXiv:2406.08177 (2024)
- 104. Wu, R., Yang, T., Sun, L., Zhang, Z., Li, S., Zhang, L.: Seesr: Towards semanticsaware real-world image super-resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 25456–25467 (2024)
- 105. Xia, B., He, J., Zhang, Y., Wang, Y., Tian, Y., Yang, W., Van Gool, L.: Structured sparsity learning for efficient video super-resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22638– 22647 (2023)
- 106. Xie, D., Xu, Z., Hong, Y., Tan, H., Liu, D., Liu, F., Kaufman, A., Zhou, Y.: Progressive autoregressive video diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6322–6332 (2025)
- 107. Xie, R., Liu, Y., Zhou, P., Zhao, C., Zhou, J., Zhang, K., Zhang, Z., Yang, J., Yang, Z., Tai, Y.: Star: Spatial-temporal augmentation with text-to-video models for real-world video super-resolution. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17108–17118 (2025)
- 108. Xie, S., Xiao, Z., Kingma, D., Hou, T., Wu, Y.N., Murphy, K.P., Salimans, T., Poole, B., Gao, R.: Em distillation for one-step diffusion models. Advances in Neural Information Processing Systems 37, 45073–45104 (2024)

- 109. Xu, K., Yu, Z., Wang, X., Mi, M.B., Yao, A.: Enhancing video super-resolution via implicit resampling-based alignment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2546–2555 (2024)
- 110. Xu, Y., Zhao, Y., Xiao, Z., Hou, T.: Ufogen: You forward once large scale text-toimage generation via diffusion gans. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8196–8206 (2024)
- 111. Xu, Y., Park, T., Zhang, R., Zhou, Y., Shechtman, E., Liu, F., Huang, J.B., Liu, D.: Videogigagan: Towards detail-rich video super-resolution (2024)
- 112. Xue, T., Chen, B., Wu, J., Wei, D., Freeman, W.T.: Video enhancement with task-oriented flow. International Journal of Computer Vision 127(8), 1106–1125

(2019)

- 113. Yang, S., Zhou, Y., Liu, Z., Loy, C.C.: Fresco: Spatial-temporal correspondence for zero-shot video translation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8703–8712 (2024)
- 114. Yang, X., He, C., Ma, J., Zhang, L.: Motion-guided latent diffusion for temporally consistent real-world video super-resolution. In: European Conference on Computer Vision. pp. 224–242. Springer (2024)
- 115. Yang, X., Xiang, W., Zeng, H., Zhang, L.: Real-world video super-resolution: A benchmark dataset and a decomposition based learning scheme. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4781–4790

(2021)

- 116. Yeh, C.H., Lin, C.Y., Wang, Z., Hsiao, C.W., Chen, T.H., Shiu, H.S., Liu, Y.L.: Diffir2vr-zero: Zero-shot video restoration with diffusion-based image restoration models. arXiv preprint arXiv:2407.01519 (2024)
- 117. Yi, P., Wang, Z., Jiang, K., Jiang, J., Ma, J.: Progressive fusion video superresolution network via exploiting non-local spatio-temporal correlations. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3106– 3115 (2019)
- 118. Yin, G., Qu, Z., Jiang, X., Jiang, S., Han, Z., Zheng, N., Yang, H., Liu, X., Yang, Y., Li, D., et al.: Online streaming video super-resolution with convolutional lookup table. IEEE Transactions on Image Processing 33, 2305–2317 (2024)
- 119. Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, B.: Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems 37, 47455–47487 (2024)
- 120. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6613– 6623 (2024)
- 121. Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., Shechtman, E., Huang, X.: From slow bidirectional to fast autoregressive video diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22963–22974 (2025)
- 122. Youk, G., Oh, J., Kim, M.: Fma-net: Flow-guided dynamic filtering and iterative feature refinement with multi-attention for joint video super-resolution and deblurring. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 44–55 (2024)
- 123. Yue, Z., Wang, J., Loy, C.C.: Resshift: Efficient diffusion model for image superresolution by residual shifting. Advances in neural information processing systems 36, 13294–13307 (2023)

- 124. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Efficient transformer for high-resolution image restoration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5728– 5739 (2022)
- 125. Zhai, Y., Lin, K., Yang, Z., Li, L., Wang, J., Lin, C.C., Doermann, D., Yuan, J., Wang, L.: Motion consistency model: Accelerating video diffusion with disentangled motion-appearance distillation. Advances in Neural Information Processing Systems 37, 111000–111021 (2024)
- 126. Zhang, A., Yue, Z., Pei, R., Ren, W., Cao, X.: Degradation-guided one-step image super-resolution with diffusion priors (2024), https://arxiv.org/abs/2409. 17058
- 127. Zhang, K., Liang, J., Van Gool, L., Timofte, R.: Designing a practical degradation model for deep blind image super-resolution. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4791–4800 (2021)
- 128. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)
- 129. Zhang, R., Gu, J., Chen, H., Dong, C., Zhang, Y., Yang, W.: Crafting training degradation distribution for the accuracy-generalization trade-off in real-world super-resolution (2023)
- 130. Zhang, Y., Yao, A.: Realviformer: Investigating attention for real-world video super-resolution. In: European Conference on Computer Vision. pp. 412–428. Springer (2024)
- 131. Zhang, Z., Li, R., Guo, S., Cao, Y., Zhang, L.: Tmp: Temporal motion propagation for online video super-resolution. IEEE Transactions on Image Processing (2024)
- 132. Zhang, Z., Wu, B., Wang, X., Luo, Y., Zhang, L., Zhao, Y., Vajda, P., Metaxas, D., Yu, L.: Avid: Any-length video inpainting with diffusion model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7162–7172 (2024)
- 133. Zhang, Z., Liu, K., Chen, Z., Li, X., Chen, Y., Duan, B., Kong, L., Zhang, Y.: Infvsr: Breaking length limits of generic video super-resolution. arXiv preprint arXiv:2510.00948 (2025)
- 134. Zhao, W., Zhou, J., Zhu, X., Chen, W., Zhang, X.Y., Lei, Z., Wang, F.: Realisvsr: Detail-enhanced diffusion for real-world 4k video super-resolution. arXiv preprint arXiv:2507.19138 (2025)
- 135. Zhao, W., Zhou, J., Zhu, X., Chen, W., Zhang, X.Y., Lei, Z., Wang, F.: Realisvsr: Detail-enhanced diffusion for real-world 4k video super-resolution. arXiv preprint arXiv:2507.19138 (2025)
- 136. Zheng, J., Hu, M., Fan, Z., Wang, C., Ding, C., Tao, D., Cham, T.J.: Trajectory consistency distillation: Improved latent consistency distillation by semi-linear consistency function with trajectory mapping. arXiv preprint arXiv:2402.19159

(2024)

- 137. Zheng, K., Lu, C., Chen, J., Zhu, J.: Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems 36, 55502–55542 (2023)
- 138. Zheng, M., Sun, L., Dong, J., Pan, J.: Efficient video super-resolution for realtime rendering with decoupled g-buffer guidance. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 11328–11337 (2025)
- 139. Zhou, M., Zheng, H., Wang, Z., Yin, M., Huang, H.: Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In: Forty-first International Conference on Machine Learning (2024)

- 140. Zhou, S., Yang, P., Wang, J., Luo, Y., Loy, C.C.: Upscale-a-video: Temporalconsistent diffusion model for real-world video super-resolution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2535–2545 (2024)
- 141. Zhou, X., Zhang, L., Zhao, X., Wang, K., Li, L., Gu, S.: Video super-resolution transformer with masked inter&intra-frame attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 25399– 25408 (2024)
- 142. Zhou, Z., Chen, D., Wang, C., Chen, C., Lyu, S.: Simple and fast distillation of diffusion models. Advances in Neural Information Processing Systems 37, 40831– 40860 (2024)
- 143. Zhu, X., Hu, H., Lin, S., Dai, J.: Deformable convnets v2: More deformable, better results. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9308–9316 (2019)
- 144. Zhuang, J., Guo, S., Cai, X., Li, X., Liu, Y., Yuan, C., Xue, T.: Flashvsr: Towards real-time diffusion-based streaming video super-resolution. arXiv preprint arXiv:2510.12747 (2025)

### Overview

This supplementary material provides additional details and results to support the main paper. We first describe the complete experimental setup in Sec. A, including training procedures, datasets, evaluation metrics, and baseline configurations. We then present extended implementation details and a three-stage breakdown of our training pipeline in Secs. B and C, covering U-Net distillation, temporal-aware decoder training, and the Auto-regressive Temporal Guidance module. Next, we report additional quantitative and visual comparisons on multiple benchmarks under both bidirectional and unidirectional settings in Secs. D and E, followed by extensive qualitative visualizations illustrating perceptual quality and temporal consistency. We also include representative failure cases to highlight current limitations in Sec. F. Video results. We provide an accompanying HTML index for interactive playback of all supplementary videos and zoom-in comparisons.

### A Experimental Setup

#### A.1 Training and Evaluation Setup

Stream-DiffVSR is trained in three sequential stages to ensure stable optimization and modular control over temporal components. All evaluation experiments are conducted on an NVIDIA RTX 4090 GPU with TensorRT acceleration, unless otherwise specified. In particular, the resource-focused comparisons (runtime/latency/peak memory) against memory-intensive baselines are measured on a single NVIDIA RTX Pro 6000 GPU, as reported in the corresponding tables. Details of the stage-wise training procedure and configurations are provided in the supplementary.

Runtime and latency measurement. All reported runtime and latency numbers are measured end-to-end on the GPU, including optical-flow estimation and all model components used at inference time (temporal modules and postprocessing). We use a warm-up run and report the average per-frame runtime over the evaluation sequence; latency-max denotes the maximum end-to-end output delay over the sequence.

Resolution alignment for evaluation. Some diffusion Transformer (DiT)-based baselines produce outputs at a fixed resolution that may not match the groundtruth (GT) frame size required by our benchmarks. To ensure a fair and consistent evaluation across all methods, we align the output resolution to the GT resolution before computing metrics. Specifically, when a method’s native output resolution differs from the GT, we apply bicubic downsampling to resize the restored frames to the GT resolution; otherwise, no resizing is performed. All full-reference (e.g., PSNR/SSIM/LPIPS/DISTS/VMAF/tLP/tOF) metrics are computed on these resolution-aligned outputs.

#### A.2 Datasets

We evaluate our method using widely-recognized benchmarks: REDS [68] and Vimeo-90K [112]. REDS consists of 300 video sequences (1280×720 resolution, 100 frames each); sequences 000, 011, 015, and 020 (REDS4) are used for testing. Vimeo-90K-T contains 91,701 clips (448×256 resolution), with 64,612 for training and 7,824 (Vimeo-90K) for evaluation, offering diverse content for training.

For testing under complex degradation, we also evaluate on two additional benchmarks: VideoLQ [127], a no-reference video quality dataset curated from real Internet content, and Vid4 [54], a classical benchmark with 4 videos commonly used for VSR evaluation. The evaluation results are provided in supplementary.

#### A.3 Evaluation metrics

We assess the effectiveness of our approach using a comprehensive set of perceptual and temporal metrics across multiple aspects. Reference-based Perceptual Quality: LPIPS [128] and DISTS [19]. No-reference Perceptual Quality: MUSIQ [42], NIQE [76], NRQM [64], BRISQUE [66]. Temporal Consistency: Temporal Learned Perceptual Similarity (tLP), and Temporal Optical Flow difference (tOF). Inference Speed: Per-frame runtime, latency measured on an NVIDIA RTX 4090 GPU to evaluate low-latency applicability. Note that while we report PSNR and SSIM results (REDS4: 27.256 / 0.768) for completeness, we do not rely on these distortion-based metrics in our main analysis, as they often fail to reflect perceptual quality and temporal coherence, especially in generative VSR settings. This has also been observed in prior work [128]. Our qualitative results demonstrate superior perceptual and temporal quality, as we prioritize low-latency stability and consistency over overfitting to any single metric.

#### A.4 Baseline methods

We evaluate our method against leading CNN-based, Transformer-based, and Diffusion-based models. Specifically, we include bidirectional (offline) methods such as BasicVSR++ [7], RealBasicVSR [8], RVRT [52], StableVSR [75], MGLDVSR [114], and unidirectional (online) methods including MIA-VSR [141], TMP [131], RealViformer [130], and StableVSR∗ [75], comprehensively comparing runtime, perceptual quality, and temporal consistency. In addition, we consider several memory-intensive diffusion/DiT-style and long-video enhancement baselines, including VEnhancer [30], Upscale-A-Video [140], SeedVR2 [97], DOVE [15], and FlashVSR [144] (as well as its lightweight variant FlashVSR-tiny [144]), and report their runtime/latency and peak GPU memory usage where applicable. For FlashVSR, we follow the official implementation and evaluation setting, including the temporal chunk size of 6; we report latency and peak GPU memory

under this configuration. Unless otherwise noted, all baselines are evaluated using official implementations with the authors’ recommended settings.

### B Additional Implementation Details

- B.1 Implementation Details

Our UNet backbone is initialized from the StableVSR [75] released UNet checkpoint, which is trained for image-based super-resolution from Stable Diffusion (SD) x4 Upscaler [72,73]. We then perform 4-step distillation to adapt this UNet for efficient video SR. ARTG, in contrast, is built upon our distilled UNet encoder and computes temporal residuals from previous high-resolution outputs using convolutional and transformer blocks. These residuals are injected into the decoder during upsampling, enhancing temporal consistency without modifying the encoder or increasing diffusion steps. Our decoder is initialized from AutoEncoderTiny and extended with a Temporal Processor Module (TPM) to incorporate multi-scale temporal fusion during final reconstruction.

C Additional Training Detials

- C.1 Stage 1: U-Net Distillation

We initialize the denoising U-Net from the 50-step diffusion model released by StableVSR [75], which was trained on REDS [67] dataset. To accelerate inference, we distill the 50-step U-Net into a 4-step variant using a deterministic DDIM [83] scheduler. During training, our rollout distillation always starts from the noisiest latent at timestep 999 and executes the full sequence of four denoising steps {999,749,499,249}. Supervision is applied only to the final denoised latent at t = 0, ensuring that training strictly mirrors the inference trajectory and reducing the gap between training and inference. We use a batch size of 16, learning rate of 5e-5 with constant, and AdamW optimizer (β1 = 0.9, β2 = 0.999, weight decay 0.01). Training is conducted for 600K iterations with a patch size of 512× 512.The distillation loss consists of MSE loss in latent space, LPIPS [128] loss, and adversarial loss using a PatchGAN discriminator [37] in pixel level, with weights of 1.0, 0.5, and 0.025 respectively. Adversarial loss are envolved after 20k iteration for training stabilization.

#### C.2 Stage 2: Temporal-aware Decoder Training

The decoder receives both the encoded ground truth latent features and temporally aligned context features (via flow-warped previous frames). The encoder used to extract temporal features is frozen.We use a batch size of 16, learning rate

0∗StableVSR [75] is originally a bidirectional model. We implement a unidirectional variant (StableVSR∗) that only uses forward optical flow for fair comparison under the online setting.

of 5e-5 with constant, and AdamW optimizer (β1 = 0.9, β2 = 0.999, weight decay 0.01). Training is conducted for 600K iterations with a patch size of 512 × 512. Loss consists of smooth L1 reconstruction loss, LPIPS [128] loss, flow loss using RAFT [89] and adversarial loss using a PatchGAN discriminator [37] in pixel level for training, with weights of 1.0, 0.3, 0.1 and 0.025 respectively. Flow loss and adversarial loss are envolved after 20k iteration for training stabilization.

#### C.3 Stage 3: Auto-regressive Temporal Guidance

We train the ARTG module while freezing both the U-Net and decoder. Optical flow is computed between adjacent frames using RAFT [89], and the warped previous super-resolved frame is injected into the denoising U-Net and decoder. The loss formulation is identical to Stage 1, conducted with 60K iterations. This guides ARTG to enhance temporal coherence while maintaining alignment with the original perceptual objectives.

Algorithm 1: Training procedure for U-Net rollout distillation.

Input: Dataset D = {(I, I˜ )}; pre-trained VAE; 4-step noise scheduler; student

U-Net with parameters θ; discriminator D(·). for epoch = 1 to N do

for each batch (I, I˜ ) ∈ D do z0 ← VAE.encode(I); Sample ϵ ∼ N(0, I);

zT ← αTz0 + √1 − αT ϵ ; // Add noise at maximum timestep T // –- Rollout 4-step denoising –-

zˆT ← [ zT, I˜]; for step s = T, . . . , 1 do

ϵˆ ← U−Net(zˆs, s); zˆs−1 ← Scheduler.step(ˆϵ, s, zˆs);

Iˆ ← VAE.decode(zˆ0); LL2 ← ∥Iˆ− I∥22; LLPIPS ← LPIPS(I, Iˆ ); LGAN ← softplus −D(Iˆ) ; L ← λL2 LL2 + λLPIPS LLPIPS + λGAN LGAN; Update parameters: θ ← θ − η∇θL;

### D Additional Quantitative comparison.

We provide extended quantitative results across multiple datasets and settings. Specifically, we report both bidirectional and unidirectional performance with mean and standard deviation on REDS4 (Tabs. 14 and 15), Vimeo-90K (Tabs. 16

Algorithm 2: Auto-Regressive Diffusion VSR.

Notation: {I˜i}: Input LR frames, {Iˆi}: Enhanced frames, FlowWarp: Warping w.r.t. flow, VAE: Auto-regressive VAE, UNet: Distilled diffusion U-Net, ARTG: Auto-Regressive Temporal guidance, PrepareLatents: Create latent input, timesteps: {t1, . . . , t4}

Input: {I˜i}Ni=1, flows {fi−1}Ni=2, VAE, UNet, ARTG. Output: {Iˆi}Ni=1. for i = 1 to N do

LQi ← I˜i zi ← PrepareLatents(LQi, t) if i > 1 then

Iˆiw−1 ← FlowWarp(Iˆi−1, fi−1) Ei−1 ← VAE.encode(Iˆiw−1)

for t ∈ timesteps do if i > 1 then

zi ← ARTG(zi, Iˆiw−1) ϵˆ ← UNet(zi, t) zi ← DiffusionUpdate(ˆϵ, t, zi)

###### if i > 1 then

Iˆi ← VAE.Decode(z, Ei−1) else

Iˆi ← VAE.Decode(z) return {Iˆi}

and 17) and VideoLQ (Tab. 18) while additional results are provided on Vid4 (Tabs. 19 and 20). These supplementary results further validate the robustness of our approach under diverse benchmarks and temporal settings.

- Table 14: Quantitative comparison against bidirectional/offline methods on the REDS4 dataset with mean and standard deviation. We compare CNN-, Transformer-, and diffusion-based approaches. Stream-DiffVSR shows superior perceptual quality, temporal consistency, and stability. All values are reported as mean ± std over 4 videos. ↑ / ↓ denote higher/lower is better. Dir.: B = bidirectional/offline, U

= unidirectional/online. Runtime is reported as the average per-frame inference time across all test sequences on an RTX 4090. Latency-first and Latency-avg measure first-frame and average latency; tLP and tOF are scaled by 100× and 10×. Best and second-best values are marked in red and blue. For space reasons, the main paper reports the mean-only version; the full mean±std statistics are shown here.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ tLP↓ tOF↓ Runtime (s)↓ latency-first (s)↓ latency-avg (s)↓ CNN-based Methods

- Bicubic 25.501 ± 1.516 0.712 ± 0.062 0.460 ± 0.042 0.187 ± 0.013 27.362 ± 2.239 7.360 ± 0.120 3.459 ± 0.177 60.256 ± 1.828 21.603 ± 5.817 4.241 ± 5.765 - - B BasicVSR++ 32.386 ± 2.415 0.907 ± 0.029 0.132 ± 0.023 0.069 ± 0.012 67.002 ± 4.291 3.850 ± 0.439 6.363 ± 0.330 38.641 ± 5.224 9.017 ± 4.384 2.490 ± 4.440 0.098 9.8 4.9 B RealBasicVSR 27.042 ± 1.865 0.778 ± 0.059 0.134 ± 0.016 0.060 ± 0.006 67.033 ± 4.283 2.530 ± 0.452 6.769 ± 0.242 18.046 ± 4.185 6.422 ± 4.726 4.759 ± 7.722 0.064 6.4 3.2

Transformer-based Methods

B RVRT 32.701 ± 2.487 0.911 ± 0.027 0.130 ± 0.022 0.067 ± 0.011 67.251 ± 4.372 3.793 ± 0.463 6.366 ± 0.339 38.038 ± 5.779 9.133 ± 4.408 2.421 ± 4.316 0.498 49.8 24.9 B MIA-VSR 32.790 ± 2.535 0.912 ± 0.028 0.123 ± 0.022 0.064 ± 0.011 68.140 ± 3.964 3.742 ± 0.472 6.451 ± 0.304 37.099 ± 5.668 8.870 ± 4.606 2.354 ± 4.026 0.768 0.768 0.768

Diffusion-based Methods

B StableVSR 27.928 ± 2.411 0.793 ± 0.063 0.102 ± 0.015 0.047 ± 0.006 67.058 ± 3.797 2.713 ± 0.456 6.960 ± 0.211 16.249 ± 4.133 5.755 ± 4.618 2.742 ± 4.741 46.2 4620 2310 B MGLD-VSR 26.53 ± 1.939 0.749 ± 0.062 0.151 ± 0.019 0.065 ± 0.006 66.081 ± 4.027 2.972 ± 0.386 6.701 ± 0.202 15.291 ± 4.463 18.139 ± 8.772 5.910 ± 6.888 43.6 218 109 U Ours 27.256 ± 2.134 0.766 ± 0.062 0.099 ± 0.013 0.062 ± 0.007 65.595 ± 3.982 3.114 ± 0.186 7.055 ± 0.257 17.117 ± 1.836 4.198 ± 3.795 3.638 ± 4.855 0.328 0.328 0.328

- Table 15: Quantitative comparison against unidirectional/online methods on the REDS4 dataset with mean and standard deviation.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ tLP↓ tOF↓ Runtime (s)↓ latency-first (s)↓ latency-avg (s)↓ CNN-based Methods

- Bicubic 25.501 ± 1.516 0.712 ± 0.062 0.460 ± 0.042 0.187 ± 0.013 27.362 ± 2.239 7.360 ± 0.120 3.459 ± 0.177 60.256 ± 1.828 21.603 ± 5.817 4.241 ± 5.765 - - U TMP 30.672 ± 2.317 0.871 ± 0.039 0.194 ± 0.039 0.090 ± 0.010 63.818 ± 4.129 4.378 ± 0.333 5.796 ± 0.312 43.394 ± 4.442 10.424 ± 5.654 2.480 ± 3.852 0.041 0.041 0.041

Transformer-based Methods

U RealViformer 26.763 ± 1.898 0.761 ± 0.062 0.129 ± 0.062 0.065 ± 0.004 64.585 ± 5.117 2.731 ± 0.454 6.356 ± 0.079 17.272 ± 4.546 11.261 ± 5.613 11.782 ± 3.762 0.099 9.9 4.95

Diffusion-based Methods

U StableVSR* 27.174 ± 2.449 0.763 ± 0.069 0.111 ± 0.017 0.051 ± 0.006 66.428 ± 4.040 2.572 ± 0.356 6.944 ± 0.211 15.805 ± 4.626 11.107 ± 8.293 3.925 ± 4.561 46.2 4620 2310 U Ours 27.256 ± 2.134 0.766 ± 0.062 0.099 ± 0.013 0.062 ± 0.007 65.595 ± 3.982 3.114 ± 0.186 7.055 ± 0.257 17.117 ± 1.836 4.198 ± 3.795 3.638 ± 4.855 0.328 0.328 0.328

- Table 16: Quantitative comparison on the Vimeo-90K-T dataset with mean and standard deviation(bidirectional/offline). Our Stream-DiffVSR achieves superior perceptual quality, temporal consistency, and substantially lower runtime. Results are reported as mean ± std across the dataset, with runtime measured on 448×256 videos using an RTX 4090 GPU. Best and second-best results are shown in red and blue. For space reasons, the main paper presents the mean-only version; the full mean±std statistics are provided here.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ tLP↓ tOF↓ Runtime (s)↓ latency-first (s)↓ latency-avg (s)↓ CNN-based Methods

- Bicubic 29.282 ± 3.647 0.864 ± 0.061 0.297 ± 0.105 0.209 ± 0.044 23.433 ± 5.633 8.735 ± 0.397 3.588 ± 0.43 61.714 ± 4.599 11.606 ± 7.674 2.49 ± 1.645 - - B BasicVSR++ 37.479 ± 4.724 0.956 ± 0.033 0.098 ± 0.04 0.117 ± 0.024 51.940 ± 6.169 7.077 ± 1.111 5.509 ± 3.514 47.792 ± 12.514 4.691 ± 5.013 1.57 ± 0.974 0.012 0.084 0.042 B RealBasicVSR 29.388 ± 2.692 0.857 ± 0.059 0.156 ± 0.113 0.149 ± 0.06 56.986 ± 4.418 5.069 ± 0.464 7.413 ± 0.66 23.822 ± 10.19 10.947 ± 14.292 3.46 ± 2.446 0.008 0.056 0.028

Transformer-based Methods

B RVRT 37.815 ± 5.049 0.955 ± 0.033 0.093 ± 0.05 0.105 ± 0.023 49.937 ± 6.509 7.205 ± 1.005 5.393 ± 0.992 48.352 ± 12.147 4.873 ± 6.486 1.429 ± 1.079 0.061 0.427 0.213 B MIA-VSR 37.598 ± 4.724 0.957 ± 0.032 0.086 ± 0.039 0.101 ± 0.025 51.402 ± 6.522 7.116 ± 1.158 5.569 ± 1.249 47.865 ± 13.17 4.696 ± 5.874 1.419 ± 0.997 0.096 0.096 0.096

Diffusion-based Methods

B StableVSR 31.823 ± 3.686 0.878 ± 0.058 0.095 ± 0.044 0.111 ± 0.025 54.582 ± 6.111 4.745 ± 0.857 7.265 ± 1.427 20.039 ± 6.398 26.224 ± 9.042 3.108 ± 2.794 5.749 40.243 20.121 B MGLD-VSR 29.651 ± 2.354 0.865 ± 0.057 0.151 ± 0.076 0.137 ± 0.032 57.788 ± 3.876 5.340 ± 0.798 7.217 ± 0.814 20.761 ± 8.394 12.550 ± 10.504 4.661 ± 3.449 5.426 27.130 13.560 U Ours 32.593 ± 3.82 0.900 ± 0.060 0.056 ± 0.035 0.105 ± 0.017 52.755 ± 6.017 4.403 ± 1.02 7.672 ± 1.476 29.297 ± 10.007 4.307 ± 4.359 2.689 ± 1.619 0.041 0.041 0.041

- Table 17: Quantitative comparison on the Vimeo-90K-T dataset with mean

###### and standard deviation(unidirectional/online).

Dir. Method PSNR↑ SSIM↑ LPIPS↓ DISTS↓ MUSIQ↑ NIQE↓ NRQM↑ BRISQUE↓ tLP↓ tOF↓ Runtime (s)↓ latency-first (s)↓ latency-avg (s)↓ CNN-based Methods

- Bicubic 29.282 ± 3.647 0.864 ± 0.061 0.297 ± 0.105 0.209 ± 0.044 23.433 ± 5.633 8.735 ± 0.397 3.588 ± 0.43 61.714 ± 4.599 11.606 ± 7.674 2.49 ± 1.645 - - U TMP 36.482 ± 4.672 0.946 ± 0.039 0.109 ± 0.057 0.118 ± 0.027 48.374 ± 6.31 7.368 ± 0.909 5.096 ± 0.891 49.192 ± 11.55 4.870 ± 5.177 1.603 ± 1.011 0.006 0.006 0.006

###### Transformer-based Methods

U RealViformer 30.291 ± 2.518 0.877 ± 0.055 0.130 ± 0.061 0.140 ± 0.03 53.107 ± 3.65 5.515 ± 0.486 6.711 ± 0.889 24.628 ± 7.933 8.232 ± 6.864 2.769 ± 1.909 0.013 0.091 0.045

###### Diffusion-based Methods

U StableVSR* 31.729 ± 3.698 0.875 ± 0.061 0.098 ± 0.049 0.113 ± 0.026 54.447 ± 6.008 4.698 ± 0.853 7.280 ± 1.444 19.836 ± 6.131 30.858 ± 13.166 3.144 ± 2.845 5.749 40.243 20.121 U Ours 32.593 ± 3.82 0.900 ± 0.060 0.056 ± 0.035 0.105 ± 0.017 52.755 ± 6.017 4.403 ± 1.02 7.672 ± 1.476 29.297 ± 10.007 4.307 ± 4.359 2.689 ± 1.619 0.041 0.041 0.041

- Table 18: Quantitative comparison with memory-intensive baselines on the VideoLQ dataset under a single RTX Pro 6000. Runtime is reported as the average per-frame inference time across all test sequences. Latency-max denotes the maximum end-to-end latency. Peak-Mem denotes the peak memory usage across all test sequences. Gray-shaded entries are reported from FlashVSR. – indicates unavailable results. Dir. Method NIQE↓ NRQM↑ BRISQUE↓ Runtime (s) Latency-max (s) Peak Mem (GB) B VEhancer 6.221 ± 1.673 3.85 ± 1.107 48.1 ± 15.362 9.544 477.237 47.985

- B SeedVR2 4.661 ± 0.803 5.523 ± 0.844 37.975 ± 8.257 1.126 56.28 76.094

- B SeedVR2 5.205 – – – – – B UAV 6.299 ± 0.723 3.652 ± 0.851 44.139 ± 8.898 8.081 404.07 55.897 B UAV 4.889 – – – – – B DOVE 5.090 ± 0.961 5.214 ± 0.911 36.631 ± 11.667 1.735 86.774 46.344 U FlashVSR – – – – – OOM U FlashVSR 3.803 – – – – – U FlashVSR-tiny 4.569 ± 0.756 5.164 ± 0.888 42.514 ± 8.846 0.204 1.224 44.180 U FlashVSR-tiny 4.070 – – – – – U Ours 3.929 ± 0.64 6.140 ± 1.086 27.176 ± 5.664 0.454 0.454 22.800

### E Additional Visual Result

Figs. 9 to 12 presents qualitative results on challenging sequences with diverse content and motion. We also provide qualitative comparisons with UpscaleA-Video [140] on AIGC video frame in Fig. 15. Compared with CNN-based (TMP [131], BasicVSR++ [7]) and Transformer-based (RealViformer [130]) approaches, as well as diffusion-based methods (e.g., MGLD-VSR) and several memory-intensive baselines (e.g., VEnhancer [30], SeedVR2 [97], DOVE [15], Upscale-A-Video and FlashVSR [144] where available), our method produces sharper structures and more faithful textures. These visual comparisons further demonstrate the effectiveness of our design in maintaining perceptual quality and temporal consistency across diverse scenes.

Temporal consistency comparison. As shown in the consecutive-frame comparisons Fig. 13, Stream-DiffVSR alleviates flickering artifacts and preserves stable textures over time, yielding noticeably stronger temporal coherence than prior VSR methods.

Optical flow visualization comparison. The optical flow consistency visualizations Fig. 14 further highlight our advantages: Stream-DiffVSR generates smoother and more temporally coherent flow fields, reflecting improved motion stability and reduced temporal artifacts.

### F Failure cases

Fig. 16 illustrates a limitation of our approach on the first frame of a video sequence. Since no past frames are available for temporal guidance, the model may produce blurrier details or less stable structures compared to subsequent frames. This issue is inherent to all online VSR settings, where temporal information cannot be exploited at the sequence start. As shown in later frames,

###### Table 19: Quantitative comparison against bidirectional/offline methods on the Vid4 dataset. Dir. Method PSNR↑ SSIM↑ LPIPS↓ NRQM↑ BRISQUE↓ tLP↓ tOF↓ latency-max (s)↓

CNN-based Methods

- Bicubic 21.719 0.582 0.512 3.429 58.680 27.819 1.145 B BasicVSR++ 26.230 0.828 0.193 6.481 38.409 15.029 0.507 6.86 B RealBasicVSR 21.963 0.597 0.210 7.122 21.804 6.630 0.9 4.48

Transformer-based Methods

B RVRT 26.377 0.826 0.229 6.006 44.667 17.146 0.507 1.743 B MIA-VSR 26.175 0.826 0.174 6.619 38.509 14.297 0.505 53.76

Diffusion-based Methods

B StableVSR 22.541 0.644 0.194 7.224 13.254 48.585 0.957 3234 B MGLD-VSR 21.983 0.605 0.243 7.129 16.525 31.744 3.152 152.6

U Ours 22.725 0.652 0.191 7.346 15.260 8.985 0.962 0.229

###### Table 20: Quantitative comparison against unidirectional/online methods on the Vid4 dataset.

Dir. Method PSNR↑ SSIM↑ LPIPS↓ NRQM↑ BRISQUE↓ tLP↓ tOF↓ latency-max (s)↓ CNN-based Methods

- Bicubic 21.719 0.582 0.512 3.429 58.680 27.819 1.145 U TMP 25.579 0.797 0.256 5.698 46.257 14.199 0.566 0.029

###### Transformer-based Methods

U RealViformer 21.963 0.597 0.257 7.604 21.804 11.633 1.107 6.93

###### Diffusion-based Methods

U StableVSR* 22.213 0.623 0.203 7.233 11.966 59.594 1.036 3234 U Ours 22.725 0.652 0.191 7.346 15.260 8.985 0.962 0.229

once temporal context becomes available, our method quickly stabilizes and reconstructs high-fidelity details.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

| |
|---|

VEnhancer UAV

SeedVR2

[Figure 122]

[Figure 123]

[Figure 124]

Input frame FlashVSR-tiny Ours

DOVE

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

VEnhancer UAV

SeedVR2

[Figure 129]

[Figure 130]

[Figure 131]

| |
|---|

Input frame FlashVSR-tiny Ours

DOVE

###### Fig. 9: Additional visual results on VideoLQ dataset.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

| |
|---|

TMP BasicVSR++

MGLD-VSR

[Figure 136]

[Figure 137]

[Figure 138]

Video frame Ours GT

RealViFromer

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

| |
|---|

TMP BasicVSR++

MGLD-VSR

[Figure 143]

[Figure 144]

[Figure 145]

Video frame Ours GT

RealViFromer

###### Fig. 10: Additional visual results on Vimeo-90K-T dataset.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

| |
|---|

TMP BasicVSR++

MGLD-VSR

[Figure 150]

[Figure 151]

[Figure 152]

Video frame Ours GT

RealViFromer

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

| |
|---|

TMP BasicVSR++

MGLD-VSR

[Figure 157]

[Figure 158]

[Figure 159]

Video frame Ours GT

RealViFromer

###### Fig. 11: Additional visual results on Vimeo-90K-T dataset.

| |
|---|

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

TMP BasicVSR++

MGLD-VSR

[Figure 164]

[Figure 165]

[Figure 166]

Video frame Ours GT

RealViFromer

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

| |
|---|

TMP BasicVSR++

MGLD-VSR

[Figure 171]

[Figure 172]

[Figure 173]

Video frame Ours GT

RealViFromer

###### Fig. 12: Additional visual results on Vimeo-90K-T dataset.

|[Figure 174]<br><br>Bicubic|
|---|
|[Figure 175]<br><br>MIA-VSR|
|[Figure 176]<br><br>MGLD-VSR|
|[Figure 177]<br><br>RealViFormer|
|[Figure 178]<br><br>X4-Upscaler|
|[Figure 179]<br><br>Ours|
|[Figure 180]<br><br>GT|

|[Figure 181]<br><br>Bicubic|
|---|
|[Figure 182]<br><br>MIA-VSR|
|[Figure 183]<br><br>MGLD-VSR|
|[Figure 184]<br><br>RealViFormer|
|[Figure 185]<br><br>X4-Upscaler|
|[Figure 186]<br><br>Ours|
|[Figure 187]<br><br>GT|

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Time

Time

###### Fig. 13: Temporal consistency comparison. Qualitative comparison of temporal consistency across consecutive frames. Our proposed Stream-DiffVSR effectively mitigates flickering artifacts and maintains stable texture reconstruction, demonstrating superior temporal coherence compared to existing VSR methods.

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

RealViformer X4-Upscaler MGLD-VSR Ours GT

###### Fig. 14: Optical flow visualization comparison. Visualization of optical flow consistency across different VSR methods. Our proposed Stream-DiffVSR produces smoother and more temporally coherent flow fields, indicating improved motion consistency and reduced temporal artifacts compared to competing approaches.

| |
|---|

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

UAV

[Figure 212]

[Figure 213]

[Figure 214]

Ours

###### Fig. 15: Qualitative comparison with Upscale-A-Video (UAV) on AIGC video frames.

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

OursGT

| |
|---|

[Figure 219]

[Figure 220]

[Figure 221]

Video Frame Frame 1 Frame 2 Frame 3

###### Fig. 16: Limitation on the first frame without temporal context. Our method may underperform on the first frame of a video sequence due to the absence of prior temporal information. This limitation is inherent to online VSR settings, where no past frames are available for guidance.

