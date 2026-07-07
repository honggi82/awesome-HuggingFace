# arXiv:2511.19365v2[cs.CV]8Apr2026

## DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation

Zehong Ma1,3,†, Longhui Wei3,‡,∗, Shuai Wang2, Shiliang Zhang1,∗, Qi Tian3 1 State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University 2Nanjing University 3Huawei Inc.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Figure 1. Qualitative results of text-to-image generation of DeCo. All images are 512×512 resolution.

### Abstract

https://github.com/Zehong-Ma/DeCo.

Pixel diffusion aims to generate images directly in pixel space in an end-to-end fashion. This approach avoids the limitations of VAE in the two-stage latent diffusion, offering higher model capacity. Existing pixel diffusion models suffer from slow training and inference, as they usually model both high-frequency signals and low-frequency semantics within a single diffusion transformer (DiT). To pursue a more efficient pixel diffusion paradigm, we propose the frequency-DeCoupled pixel diffusion framework. With the intuition to decouple the generation of high and low frequency components, we leverage a lightweight pixel decoder to generate high-frequency details conditioned on semantic guidance from the DiT. This thus frees the DiT to specialize in modeling low-frequency semantics. In addition, we introduce a frequency-aware flow-matching loss that emphasizes visually salient frequencies while suppressing insignificant ones. Extensive experiments show that DeCo achieves superior performance among pixel diffusion models, attaining FID of 1.62 (256×256) and 2.22 (512×512) on ImageNet, closing the gap with latent diffusion methods. Furthermore, our pretrained text-to-image model achieves a leading overall score of 0.86 on GenEval in system-level comparison. Codes are publicly available at

*Corresponding authors. ‡ Project leader. † Work was done during internship at Huawei Inc.

### 1. Introduction

Diffusion models [9, 19, 51, 71, 74] have achieved remarkable success in high-fidelity image generation, offering exceptional quality and diversity. Research in this field generally follows two main directions: latent diffusion and pixel diffusion. Latent diffusion models [29, 38, 44, 46, 72] split generation into two stages. A VAE first compresses images into a compact latent space, and a diffusion model operates within this space. However, their performance is largely constrained by the VAE’s reconstruction quality and latent distribution [30, 76]. Training a VAE often requires adversarial or additional supervision [66], which can be unstable and cause inevitable low-level artifacts.

Pixel diffusion avoids these VAE-dependent limitations by directly modeling raw pixels in an end-to-end manner. This enables more optimal distribution learning and eliminates artifacts from imperfect VAE compression. However, it is challenging for pixel diffusion to jointly model complex high-frequency signals and low-frequency semantics within the enormous pixel space. As illustrated in Fig. 2 (a), traditional methods [6, 9, 20, 26, 54] typically rely on a single model like diffusion transformer (DiT) to learn these two components from a single-scale

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

| |Patchify| |
|---|---|---|
| | | |

|Unpatchify| |
|---|---|
| | |

Unpatchify

DiT Output (Baseline)

Patchify

|[Figure 14]|
|---|

Diffusion

Pixel Velocity

Transformer

（Low&High）

[Figure 15]

Generated Image (Baseline)

Single-Scale

- (a) Baseline
- (b) DeCo (Ours) (c) Visualization

Small-Scale

t=1 t=34 t=67 t=100

[Figure 16]

Diffusion Transformer

|[Figure 17]|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

DiT Output (Ours)

Freq-aware FMLoss

(Low-Freq Semantics)

Downsample

Condition

[Figure 22]

[Figure 23]

[Figure 24]

Pixel Velocity

|[Figure 25]|
|---|

Generated Image (Ours)

Pixel Decoder

(High-Freq Signals)

Large-Scale

- Figure 2. Illustration of our frequency-decoupled (DeCo) framework. In (a), traditional baseline models rely on a single DiT to jointly model both low-frequency semantics and high-frequency signals. (b) shows our DeCo framework, where a lightweight pixel decoder focuses on the high-frequency reconstruction, and the DiT models low-frequency semantics. As shown in (c), decoupling DiT from modeling high-frequency signals leads to better low-frequency semantic features in DiT Output, and higher image quality.

input for each timestep. The complex high-frequency signals, particularly high-frequency noise, could be hard to learn [31, 50, 68]. They could also distract the DiT from learning low-frequency semantics [31, 33, 62]. As illustrated in Fig. 2 (c), this paradigm leads to noisy DiT outputs and degraded image quality.

Previous works [6, 8, 24, 54] demonstrate that it’s more effective to reconstruct high-frequency signals from highresolution input and model low-frequency semantics from low-resolution input. Other studies [7, 43, 49] show that transformers tend to capture low-frequency semantics well but struggle with high-frequency signals. We thus propose to decouple the generation of high and low frequency components. As illustrated in Fig. 2 (b), DeCo utilizes the DiT to specialize in low-frequency semantic modeling with downsampled inputs. Semantic cues are hence incorporated with a lightweight pixel decoder to reconstruct highfrequency signals. In other words, the pixel decoder takes the low-frequency semantics from DiT as condition and predicts pixel velocities with a high-resolution input. This new paradigm hence frees the DiT to specialize in modeling semantics, and allows for more specialized details generation.

To further emphasize visually salient frequencies and suppress perceptually insignificant high-frequency components, we introduce a frequency-aware Flow-Matching (FM) loss inspired by JPEG [23]. Unlike the standard FM loss, which treats all frequencies equally, our frequencyaware variant transforms the pixel velocity into the frequency domain using a discrete cosine transform and assigns adaptive weights to each frequency band. The adaptive weights are derived from JPEG quantization tables, which encode robust priors about the visual importance of different frequencies [23]. By emphasizing visually salient

frequencies and suppressing high-frequency noise, this loss simplifies the optimization landscape and enhances the visual fidelity. It is worth noting that our motivation largely aligns with the concurrent work JiT [31]. JiT provides an explicit formula to decouple the high-frequency noise and clean image, while our DeCo provides a high-freq shortcut via the pixel decoder, enabling an implicit frequency decoupling. More discussions are included in Appendix A.

We have conducted extensive experiments to test the performance of DeCo. It achieves superior results among pixel diffusion models, with FID scores of 1.62 (256×256) and 2.22 (512×512) on ImageNet, closing the gap with two-stage latent diffusion methods. Our pretrained textto-image model also achieves leading results on GenEval (0.86) and DPG-Bench (81.4) in system-level evaluation. In summary, our contributions can be summarized as two aspects: i) we introduce a novel frequency-decoupled framework DeCo for pixel diffusion, where a lightweight pixel decoder is proposed to model high-frequency signals, freeing the DiT to specialize in low-frequency semantic modeling, and ii) a novel frequency-aware FM loss is proposed to prioritize perceptually important frequencies, simplifying the training and improving visual quality. The strong performance verifies the effectiveness of decoupling the modeling of high and low frequency components in pixel diffusion.

### 2. Related Work

This work is closely related to latent diffusion, pixel diffusion, and frequency-decoupled image generation.

Latent Diffusion. Latent diffusion trains diffusion models in a compact latent space learned by a VAE [46]. Compared to raw pixel space, the latent space significantly reduces spatial dimensionality, easing learning difficulty and com-

putational cost [4, 46, 76]. Consequently, VAEs have become a fundamental component in modern diffusion models [25, 38, 39, 44, 52, 55, 58, 63–65, 67, 73, 78]. However, training VAEs often involves adversarial objectives and perceptual supervision, which complicate the overall pipeline [66]. Poorly trained VAEs can produce decoding artifacts [6, 80], limiting the generalization of latent diffusion models. Early latent diffusion models mainly used UNet-based architectures [56, 57]. The pioneering DiT [44] introduced transformers into diffusion models, replacing the U-Net [1, 9]. SiT [38] further validated the DiT with linear flow diffusion. Subsequent works enhanced latent diffusion through representation alignment and joint optimization. REPA [77] aligns intermediate features with a pretrained DINOv2 [42] model to learn better low-frequency semantics, which is compatible with our framework and is applied in both our baseline and final DeCo. REPA-E [30] attempts to jointly optimize the VAE and DiT in an end-toend fashion. However, this approach may suffer from training collapse with diffusion loss, as the continually changing latent space leads to unstable denoising targets. In contrast, pixel diffusion denoises in a fixed space, ensuring consistent targets and stable training.

Pixel Diffusion. Pixel diffusion has progressed much more slowly than its latent counterparts due to the vast dimensionality of pixel space [9, 20, 26]. Early pixel diffusion models typically rely on long residual connections [9, 54], which may hinder scalability [66]. Recent attempts split the diffusion process into chunks at different resolution scales to reduce computational costs [6, 54]. Early methods split the diffusion process into multiple resolution stages [6, 54]. Relay Diffusion [54] trains separate models for each scale, leading to higher cost and two-stage optimization. Pixelflow [6] uses one model across all scales and needs a complex denoising schedule that slows down inference. Alternative approaches explore different model architectures. FractalGen [32] builds fractal generative models by recursively applying atomic modules, achieving selfsimilar pixel-level generation. TarFlow [79] introduces a transformer-based normalizing flow to directly model and generate pixels. Recent PixNerd [66] employs a DiT to predict neural field parameters for each patch, rendering pixel velocities akin to test-time training. JiT [31] predicts the clean image to anchor generation to the low-dimensional data manifold. PixelGen [40] introduces perceptual supervision to simplify the training of pixel diffusion.

Frequency-Decoupled Image Generation. Multi-scale cascaded methods [6, 54] can be approximately regarded as a form of temporal frequency decoupling, i.e., early steps generate low-frequency semantics, and later steps refine high-frequency details. However, these methods still use a single model or architecture to learn all frequencies for each timestep and the high-frequency signals still exist. High-

frequency noise may interfere with low-frequency semantic learning. They also rely on complex denoising schedules and small patch sizes, which reduce training or sampling efficiency. Our DeCo introduces an explicit frequency decoupling in architecture. Instead of separating distinct frequencies across timesteps, DeCo simultaneously processes them within each timestep in an end-to-end manner. Recent two-stage work DDT [67] explores single-scale frequency decoupling in a compressed latent space, showing that frequency decoupling remains important even in compressed space. Unlike DDT, our DeCo is a multi-scale design for pixel diffusion. Our decoder uses attention-free linear layers instead of DDT’s attention-based DiT blocks, making it more efficient in large-scale inputs.

### 3. Method

#### 3.1. Overview

This part first reviews the conditional flow matching in baseline pixel diffusion, then proceeds to introduce our frequency-decoupled pixel diffusion framework.

Conditional Flow Matching. The conditional flow matching [35, 37] provides a continuous-time generative modeling framework that learns a velocity field vθ(x,t,y) to transport samples from a simple prior distribution (e.g., Gaussian) to a data distribution conditioned on the label y and time t. Given a forward trajectory xt by an interpolation between a clean image x0 and noise x1, the objective of conditional flow matching is to match the model-predicted velocity vθ(xt,t,y) to the ground-truth velocity vt:

t,t,y ∥vθ(xt,t,y) − vt∥2 , (1) where the linear interpolation of trajectory xt is defined as:

LFM = Ex

xt = (1 − t)x0 + tx1. (2)

The ground-truth velocity vt can be derived from x′t, i.e., the time derivative of xt:

vt = x′t = x1 − x0. (3)

In the pixel diffusion baseline, the trajectory xt is usually first patchified into tokens by a patch embedding layer [6, 44] instead of a VAE to downsample the image. In our baseline and DeCo experiments, we use the same patch size of 16 for the DiT’s input. In baseline, the patchified trajectory x¯t is then fed into the DiT to predict the pixel velocity with an unpatchify layer. The DiT is required to simultaneously model both the high-frequency signals and low-frequency semantics. The high-frequency signals, particularly the high-frequency noise, are hard to model, which can distract the DiT from learning low-frequency semantics. DeCo. To separate high-frequency generation from lowfrequency semantic modeling, we propose a frequencydecoupled framework DeCo. As illustrated in Fig. 3, the

xt

Patchify

vt

FM Loss

v(xt,t, y)

v(xt,t, y)

xlow

Diffusion Transformer

Pixel Decoder

xt Frequency-aware

RGB2YCbCr

t

t

FM Loss

DCT

y

MSE

LinearReshape

Adaptive

Linear

Linear&Reshape

xlow

Weights

MLP

,  

Weights Generation

v(xt,t, y)

xt

Layer Norm Scale, Shift MLP Scale

Quality Quant. Tables

pos

Decoder Block × N

- Figure 3. Overview of the proposed frequency-decoupled (DeCo) framework. The DiT operates on downsampled inputs to model lowfrequency semantics, while the lightweight pixel decoder generates high-frequency details under the DiT’s semantic guidance.

Dense Query Construction. The pixel decoder directly takes the full-resolution noised image as input, without downsampling. All noised pixels are concatenated with their corresponding positional embeddings pos and linearly projected by Win to form dense query vectors h0:

DiT is utilized to generate low-frequency semantics c from downsampled low-resolution inputs x¯t as follows:

##### xlow = DiT(¯xt,t,y), (4)

where t is time and y is the label or textual prompt. As depicted in Sec. 3.2, a lightweight pixel decoder then takes the low-frequency semantics c from DiT as a condition to generate additional high-frequency details with a fullresolution dense input xt, predicting the final pixel velocity as follows:

##### h0 = Win(Concat(xt,pos)), (7)

where h0 ∈ RB×H×W×d, with H and W denoting the original image height and width (e.g., 256), and d representing the hidden dimension of pixel decoder (e.g., 32). See Tab. 4 (c) and (d) for related ablation studies.

##### vθ(xt,t,y) = Dec(xt,t,xlow). (5)

Decoder Block. For each decoder block, the DiT output xlow is linearly upsampled and reshaped to match the spatial resolution of xt, yielding xuplow. A MLP then generates modulation parameters α,β,γ for AdaLN:

This new paradigm leverages the pixel decoder to generate high-frequency details, freeing the DiT to specialize in modeling semantics. The decoupling disentangles the modeling of different frequencies into different modules, leading to faster training and improved visual fidelity.

α,β,γ = MLP(σ(xuplow + t)), (8) where the σ is the SiLU activation function. We utilize the AdaLN-Zero [44] to modulate the dense decoder queries in each block as follows:

To further emphasize visually salient frequencies and ignore insignificant high-frequency components, we introduce a frequency-aware Flow-Matching (FM) Loss LFreqFM as depicted in Sec. 3.3. This loss reweights different frequency components with the adaptive weights derived from JPEG perceptual priors [23]. Combined with the standard pixel-level flow-matching loss and the REPA [77] alignment loss from the baseline, the final objective can be represented as:

hN = hN-1 + α ∗ (MLP((1 + γ) ∗ hN-1 + β)), (9) where the MLP contains two linear layers with SiLU [10].

Velocity Prediction. Finally, a linear projection followed by a rearrangement operation maps the decoder output to the pixel space, yielding the predicted velocity vθ(xt,t,y). The velocity encompasses the high-frequency details generated by the pixel decoder and the semantic cues from DiT.

L = LFM + LFreqFM + LREPA (6)

#### 3.2. Pixel Decoder

As illustrated in Fig. 3, the pixel decoder is a lightweight attention-free network composed of N linear decoder blocks and several linear projection layers. All operations are local and linear, enabling efficient high-frequency modeling without the computational overhead of self-attention.

#### 3.3. Frequency-aware FM Loss

To further encourage the pixel decoder to focus on perceptually important frequencies and suppress insignificant noise, we introduce a frequency-aware flow-matching (FM) loss.

1.0

DiT Output (Ours)

DiT Output (Baseline)

Pixel Velocity (Ours)

Pixel Velocity (Baseline) Suppressed High Frequency

0.8

NormalizedLogEnergy

| |
|---|

Pixel Decoder Focuses on High-Freq Signals, Freeing DiT for Low-Freq Semantics.

0.6

0.4

0.2

0 10 20 30 40 50 60

Frequency (Low High)

- Figure 4. DCT energy distribution of DiT outputs and predicted pixel velocities. Compared with baseline, DeCo suppresses highfrequency signals in DiT outputs while preserving strong highfrequency energy in pixel velocity, confirming effective frequency decoupling. The distribution is computed on 10K images across all diffusion steps using DCT transform with 8×8 block size.

Spatial–Frequency Transformation. We first transform both the predicted and ground-truth pixel velocities from the spatial domain to the frequency domain. This is done by converting the color space to YCbCr and applying a block-wise 8×8 discrete cosine transform (DCT), following JPEG [23]. Denoting this transformation as T , we have:

Vθ = T (vθ(xt,t,y)), Vt = T (vt). (10) Perceptual Weighting. To emphasize visually salient frequencies while suppressing insignificant ones, we employ the JPEG quantization tables [23] as visual priors to generate adaptive weights. Frequencies with smaller quantization intervals are more perceptually important. Thus, we use the normalized reciprocal of the scaled quantization tables Qcur in quality q as adaptive weights w, i.e., w = Q1

). When the quality q is between 50 and 100, the scaled quantization tables Qcur in quality q can be acquired following JPEG’s predefined rules [23]:

##### /E(Q1

cur

cur

Qcur = max 1,

Qbase · (100 − q) + 25 50

, (11)

where Qbase denotes the standard base quantization tables defined in the JPEG specification [23]. With the adaptive weights w, the frequency-aware FM loss is defined as:

t,t,y w ∥Vθ − Vt∥2 (12)

LFreqFM = Ex

#### 3.4. Empirical Analysis

To verify that DeCo effectively decouples frequencies, we analyze the DCT energy spectra of the DiT outputs and the pixel velocity, as shown in Fig. 4. Compared to the

- 2

- 3

- 4

- 5

- 6

- 7

Baseline

DeCo (Ours)

###### FID-50K

10x faster 2.59

2.57

| |
|---|

200K 400K 800K 1.6M 4.0M

Training Iteration

Figure 5. FID comparison between our DeCo and baseline. DeCo reaches 2.57 FID in 400k iterations, 10× faster than the baseline.

baseline, our pixel decoder successfully maintains all frequency components in pixel velocity. Meanwhile, the DiT outputs in DeCo exhibit significantly lower high-frequency energy than those of the baseline, indicating high-frequency components have been shifted away from the DiT and into the pixel decoder. These observations confirm that DeCo performs effective frequency decoupling. Results from Tab. 4 (c) and (d) further show that this successful decoupling benefits from two key architectural designs.

Multi-scale Input Strategy. The multi-scale input strategy is critically important. With this strategy, the pixel decoder can easily model high-frequency signals on high-resolution original inputs, freeing the DiT modeling the low-frequency semantics from low-resolution inputs where high-frequency signals have been partly suppressed.

AdaLN-based Interaction. AdaLN provides a powerful interaction mechanism between DiT and pixel decoder. In our framework, the DiT provides stable, low-frequency semantic conditioning. The AdaLN layer then uses the DiT output as condition to modulate the dense query features in the pixel decoder. Our experiments confirm that this modulation is more effective than the simpler method, such as upsampling and adding the low-frequency features to their high-frequency counterparts like UNet.

### 4. Experiments

We conduct ablation studies and baseline comparisons on ImageNet 256 × 256. For class-to-image generation, we provide detailed comparisons on ImageNet 256 × 256 and 512 × 512, and report FID [17], sFID [41], IS [48], Precision, and Recall [27]. For text-to-image generation, we report results on GenEval [14] and DPG-Bench [21].

#### 4.1. Comparison with Baselines

Setup. In the baseline comparisons, all diffusion models are trained on ImageNet at a 256×256 resolution for 200k iterations, using a large DiT variant. The key architectural modification from the baseline is the replacement of the final two DiT blocks with our proposed pixel decoder. For in-

- Table 1. Comparison with the baseline and other recent methods. Text in gray: latent diffusion models that require VAE. †: use 100 steps.

|Method<br><br>|Params<br><br>|Train Mem (GB) Speed (s/it)|Inference Mem (GB) 1 image 1 iter<br><br>|Generation Metrics FID↓ sFID↓ IS↑ Rec.l↑<br><br>|
|---|---|---|---|---|
|DiT-L/2 [44] PixDDT [67] PixNerd [66] PixelFlow [6] JiT+REPA [31]<br><br>|458M+ 86M 434M<br><br>458M<br>459M 459M<br>|28.5 0.43<br><br>23.6 0.22<br><br>29.2 0.25 73.2 1.61<br><br>24.8 0.23<br><br><br>|3.5 0.63s 0.013s<br><br>2.4 0.49s 0.010s<br><br>2.6 0.48s 0.010s<br><br>3.9 6.61s† 0.066s<br><br><br>2.5 0.46s 0.009s<br><br><br>|41.93 13.76 36.52 0.59 46.37 17.14 36.24 0.63 37.49 10.65 43.01 0.62 54.33 9.71 24.67 0.58 39.06 11.45 39.57 0.63|
|Baseline DeCo w/o REPA DeCo w/o LFreqFM DeCo|459M 426M 426M 426M<br><br>|24.8 0.22<br><br>26.4 0.22<br>27.5 0.24 27.5 0.24<br>|2.5 0.46s 0.009s 2.4 0.46s 0.009s 2.4 0.46s 0.009s 2.4 0.46s 0.009s<br><br>|61.10 15.86 16.81 0.60 67.55 10.58 19.10 0.56 34.12 10.41 46.44 0.64 31.35 9.34 48.35 0.65<br><br>|

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Figure 6. Qualitative results of class-to-image generation of DeCo. All images are 256×256 resolution.

ference, we use 50 Euler steps without classifier-free guidance [18] (CFG). We compare against the two-stage DiTL/2 [44] that requires a VAE, and other recent pixel diffusion models such as PixelFlow [6] and PixNerd [66]. We also adapt DDT [67] into pixel diffusion to create a PixDDT baseline. Besides, we intergrate recent JiT [31] in our baseline with REPA [77] for a fair comparison. Please refer to Appendix B.1 for more details.

Detailed Comparisons. As detailed in Tab. 1, our DeCo framework, despite having fewer parameters, significantly outperforms the baseline across all metrics while maintaining comparable training and inference costs. Notably, with the frequency-decoupled architecture alone, DeCo w/o LFreqFM lowers FID from 61.10 to 34.12 and raises the IS from 16.81 to 46.44 compared to the baseline. By additionally incorporating a frequency-aware FM loss, our DeCo further reduces the FID to 31.35 and achieves consistent gains on other metrics. Compared to the two-stage DiT-L/2, our VAE-free DeCo model demonstrates substantially lower training and inference overhead while achieving comparable performance. Against other pixel diffusion methods, DeCo is more efficient and effective than the multi-scale cascade model PixelFlow [6], which suffers from high computational costs. DeCo also shows superior performance compared to the single-scale attention-based PixDDT [67].

Compared to the recent PixNerd [66], our method achieves a better FID with lower training and inference costs.

JiT identifies that the high-dimensional noise may distract the model with limited capacity from learning lowdimensional data [31]. To address this, it predicts the clean image and anchors the generation process to the lowdimensional data manifold, successfully reducing the FID from 61.10 to 39.06, as shown in Tab. 1. Our DeCo shares the similar motivation, i.e., preventing high-frequency signals with high-dimensional noise from interfering with the DiT’s ability to learn low-frequency semantics. However, our DeCo proposes an alternative architectural solution. We introduce a lightweight pixel decoder to focus on modeling high-frequency signals and free the DiT to learn lowfrequency semantics. Our DeCo can also alleviate the negative impact of the high-frequency noise in the clean image, such as camera noise. Consequently, our DeCo achieves a superior FID of 31.35 compared to the 39.06 of JiT.

#### 4.2. Class-to-Image Generation

Setup. Please refer to Appendix B.2 for detailed setups. Main Results. Our DeCo achieves leading FID of 1.62 on ImageNet 256×256 and 2.22 on ImageNet 512×512. At the 256×256 resolution, DeCo demonstrates remarkable inference efficiency. It generates an image in just 1.05s with

- Table 2. Class-to-image generation on ImageNet 256×256 and 512×512 with CFG. DeCo achieves superior performance in end-to-end pixel diffusion and is competitive with two-stage latent diffusion models. Text in gray: latent diffusion models that require VAE. Text in blue background: inference with Heun sampler [16] and 50 sampling steps.

###### Params Epochs NFE Latency(s) FID↓ sFID↓ IS↑ Pre.↑ Rec.↑

|DiT-XL/2 [44] SiT-XL/2 [38] REPA-XL/2 [77]<br><br>|675M + 86M 1400 250×2 3.44 675M + 86M 1400 250×2 3.44 675M + 86M 800 250×2 3.44|2.27 4.60 278.2 0.83 0.57 2.06 4.50 284.0 0.83 0.59 1.42 4.70 305.7 0.80 0.64<br><br>|
|---|---|---|
|ADM [9] RDM [54] JetFormer [61] FractalMAR-H [32] PixelFlow-XL/4 [6] PixNerd-XL/16 [66] Baseline-XL/16<br><br>DeCo-XL/16 DeCo-XL/16 DeCo-XL/16<br><br>JiT-H/16 (Heun) [31] DeCo-XL/16 (Heun)<br><br>|554M 400 250 15.2 553M + 553M 400 250 38.4<br><br>2.8B - - 848M 600 - 155 677M 320 120×2 9.78 700M 320 100×2 1.18 700M 320 100×2 1.03 682M 320 100×2 1.05 682M 800 100×2 1.05 682M 800 250×2 2.63 953M 600 100×2 682M 600 100×2 1.05<br><br>|4.59 5.25 186.7 0.82 0.52 1.99 3.99 260.4 0.81 0.58 6.64 - - 0.69 0.56 6.15 - 348.9 0.81 0.46 1.98 5.83 282.1 0.81 0.60<br><br>1.95 4.54 300 0.80 0.60<br>2.79 4.90 296.0 0.79 0.60 1.90 4.47 303 0.80 0.61 1.71 4.54 304 0.80 0.61 1.62 4.41 301 0.80 0.62 1.86 - 303 - 1.69 4.59 304 0.79 0.63<br><br><br>|

256256×

|DiT-XL/2 [44] SiT-XL/2 [38]|675M + 86M 600 250×2 11.1 675M + 86M 600 250×2 11.1<br><br>|3.04 5.02 240.8 0.84 0.54 2.62 4.18 252.2 0.84 0.57|
|---|---|---|
|ADM-G [9] RIN [22] SimpleDiffusion [80] VDM++ [26] PixNerd-XL/16 [66] DeCo-XL/16|554M 400 250 21.2<br><br>320M - 250 2B 800 250×2 2B 800 250×2 700M 340 100×2 2.47 682M 340 100×2 2.25<br><br>|7.72 6.57 172.7 0.87 0.53<br><br>3.95 - 210.0 - 3.54 - 205.0 - 2.65 - 278.1 - -<br><br>2.84 5.95 245.6 0.80 0.59 2.22 4.67 290.0 0.80 0.60<br><br>|

512512×

Table 3. Text-to-image generation on GenEval [14] and DPG-Bench [21] at a 512×512 resolution.

Diffusion GenEval DPG-Bench Method Params Sin.Obj. Two.Obj Counting Colors Pos Color.Attr. Overall↑ Average

|PixArt-α [3] SD3 [11] FLUX.1-dev [29] DALL-E 3 [2] BLIP3o [5] OmniGen2 [70]|0.6B 8B 12B 4B 4B<br><br>|0.98 0.50 0.44 0.80 0.08 0.07 0.48<br><br>0.98 0.84 0.66 0.74 0.40 0.43 0.68<br>0.99 0.81 0.79 0.74 0.20 0.47 0.67 0.96 0.87 0.47 0.83 0.43 0.45 0.67<br><br><br>- - - - - - 0.81 1 0.95 0.64 0.88 0.55 0.76 0.80|71.1 84.0 83.5 79.4 83.6<br><br>|
|---|---|---|---|
|PixelFlow-XL/4 [6] PixNerd-XXL/16 [66] DeCo-XXL/16|882M 1.2B 1.1B<br><br>|- - - - - - 0.60 0.97 0.86 0.44 0.83 0.71 0.53 0.73<br><br>1 0.92 0.72 0.91 0.80 0.79 0.86|80.9 81.4<br><br>|

100 inference steps, whereas RDM [54] requires 38.4s and PixelFlow [6] needs 9.78s. In terms of training efficiency, as shown in Tab. 1, a single iteration for our model takes only 0.24s, far less than PixelFlow’s 1.61s. When trained for the same 320 epochs, our model’s FID (1.90) is substantially lower than the baseline’s 2.79 and surpasses the recent PixelFlow and PixNerd. As illustrated in Fig. 5, DeCo achieves a FID of 2.57 in just 80 epochs (400k iterations), which exceeds the baseline’s FID at 800 epochs, marking

a 10× improvement in training efficiency. After 800 training epochs, our DeCo achieves a superior FID of 1.62 with 250 sampling steps across pixel diffusion models, which is even comparable to the two-stage latent diffusion models. Using the same heun [16] sampler and 50-step inference at 600 epochs, DeCo reaches an FID of 1.69, outperforming JiT’s FID of 1.86 with fewer parameters and FLOPs. At the 512×512 resolution, our DeCo model substantially outperforms existing pixel-based diffusion methods, setting

Table 4. Ablation experiments on architecture design and hyper-parameters.

(a) Hidden Size d of Pixel Decoder.

|Channel<br><br>|FID↓ sFID IS↑|
|---|---|
|16 32 64<br><br>|37.63 10.64 41.54<br><br>34.12 10.41 46.44<br>35.88 10.79 44.07<br>|

(d) Interaction of DiT and Pixel Decoder.

|Interaction<br><br>|FID↓ sFID↓ IS↑|
|---|---|
|AdaLN Add|31.35 9.34 48.35 36.02 9.99 41.74<br><br>|

(b) Depth N of Pixel Decoder.

|Depth<br><br>|FID↓ sFID↓ IS↑|
|---|---|
|1 3 6<br><br>|37.10 10.73 41.06<br><br>34.12 10.41 46.44<br>35.46 10.82 44.60<br>|

(e) Loss Weight of LFreqFM

|Weight|FID↓ sFID↓ IS↑<br><br>|
|---|---|
|0.5<br><br>1<br><br>2<br><br><br>|33.54 10.27 46.38<br><br>31.35 9.34 48.35<br>32.97 9.42 46.55<br>|

(c) Patch Size of Pixel Decoder

|Patch Size<br><br>|FID↓ sFID↓ IS↑|
|---|---|
|1 4 16<br><br>|31.35 9.34 48.35 34.39 11.15 45.53 55.59 44.16 34.44|

(f) JPEG Quality in LFreqFM.

|Quality|FID↓ sFID↓ IS↑<br><br>|
|---|---|
|50 85 100|31.54 9.45 47.70 31.35 9.34 48.35 33.84 10.74 46.14<br><br>|

a leading FID of 2.22. Moreover, by fine-tuning our ImageNet 256×256 model at 320 epochs for 20 additional epochs following PixNerd [66], our FID and IS are comparable to those of DiT-XL/2 [44] and SiT-XL/2 [38] after 600 training epochs.

#### 4.3. Text-to-Image Generation

Setup. Please refer to Appendix B.3 for training details. Main Results. Compared to two-stage latent diffusion methods, our DeCo achieves an overall score of 0.86 on the GenEval benchmark [14]. This result outperforms prominent text-to-image models such as SD3 [11] and FLUX.1dev [29], as well as unified models including BLIP3o [5] and OmniGen2 [70]. Notably, our model achieves superior performance despite using the same training data as BLIP3o. On DPG-Bench [21], DeCo delivers a competitive average score comparable to two-stage latent diffusion methods. When compared to other end-to-end pixel diffusion methods, DeCo achieves a significant performance advantage over PixelFlow and PixNerd. These results show that end-to-end pixel diffusion, as implemented in our DeCo, can achieve performance comparable to that of twostage methods with limited training and inference costs. Visualizations of images generated by our text-to-image DeCo can be found in Fig. 1. Please see Appendix C for prompts.

#### 4.4. More Ablations

This section presents ablation studies on the pixel decoder design, the interaction mechanism between the DiT and the pixel decoder, and the hyperparameters of the frequencyaware FM loss. All experiments follow the setup in Sec. 4.1. Hidden Size of Pixel Decoder. As shown in Tab. 4 (a), DeCo achieves the best performance when the hidden size d is set to 32. Smaller sizes limit model capacity, while larger sizes offer no further gains. Thus, we use a hidden size of 32 by default.

Depth of Pixel Decoder. In Tab. 4 (b), a 3-layer decoder achieves the best results. A single layer lacks capacity, whereas a 6-layer design may introduce optimization difficulties. With a hidden size of 32 and 3 layers, our attention-

free decoder is lightweight (8.5M parameters) and efficient for high-resolution inputs.

Patch Size of Pixel Decoder. As shown in Tab. 4 (c), DeCo performs best when the decoder’s patch size is set to 1, enabling direct processing of the full-resolution input. Patchifying the decoder’s input degrades results, with the worst performance observed when using a large patch size of 16 like DiT. This demonstrates the effectiveness of our multiscale input strategy. All comparisons use similar parameter counts and computational costs.

Interaction between DiT and Pixel Decoder. Tab. 4 (d) shows that simply upsampling DiT outputs and adding them to dense decoder features, as done in UNet [47], underperforms compared to AdaLN-based interaction. AdaLN [44] provides a more effective interaction mechanism, using the DiT output as a semantic condition for velocity prediction.

Loss Weight. In Tab. 4 (e), the loss weight of 1 for LFreqFM gives the best results, which we adopt as the default setting. JPEG Quality in LFreqFM. In Tab. 4 (f), we study the effect of the JPEG quality factor in LFreqFM. With a quality of 100 (lossless compression), all frequency components are equally weighted, yielding an FID of 33.84, close to 34.12 without LFreqFM as expected. The commonly used quality of 85 performs best, emphasizing the important frequencies while slightly downweighting insignificant ones for optimal balance. Reducing quality to 50 overly suppresses highfrequency signals, slightly harming performance. Therefore, we use a JPEG quality of 85 in all experiments.

### 5. Conclusions

We introduced DeCo, a novel frequency-decoupled framework for pixel diffusion. By separating the modeling of low-frequency semantics with a DiT and high-frequency signals with a lightweight pixel decoder, DeCo significantly improves generation quality and efficiency. Our proposed frequency-aware FM loss further enhances visual quality by prioritizing perceptually important frequencies. DeCo achieves leading performance in pixel diffusion on both class-to-image and text-to-image generation benchmarks, closing the gap with two-stage latent diffusion methods.

### Acknowledgments

This work was supported in part by the Grant 2023-JCJQLA-001-088, in part by the Grant 2025ZD1601300, in part by the Natural Science Foundation of China under Grant U20B2052, in part by the Okawa Foundation Research Award, in part by the Ant Group Research Fund, and in part by the Kunpeng&Ascend Center of Excellence, Peking University.

### References

- [1] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023. 3
- [2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. OpenAI Technical Report, 2023. 7
- [3] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,

2023. 7

- [4] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024. 3
- [5] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 7, 8, 1, 4
- [6] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025. 1, 2, 3, 6, 7
- [7] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions. In The Eleventh International Conference on Learning Representations, 2023. 2
- [8] Emily L Denton, Soumith Chintala, Rob Fergus, et al. Deep generative image models using a laplacian pyramid of adversarial networks. Advances in neural information processing systems, 28, 2015. 2
- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 1, 3, 7
- [10] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoidweighted linear units for neural network function approximation in reinforcement learning. Neural networks, 107:3–11,

2018. 4

- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik

- Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. 7, 8
- [12] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024. 1
- [13] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025. 1
- [14] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 5, 7, 8
- [15] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025. 1
- [16] Karl Heun et al. Neue methoden zur approximativen integration der differentialgleichungen einer unabh¨angigen ver¨anderlichen. Z. Math. Phys, 45:23–38, 1900. 7
- [17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 5
- [18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6, 1
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [20] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 1, 3
- [21] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 5, 7, 8
- [22] Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. arXiv preprint arXiv:2212.11972, 2022. 7
- [23] Joint Photographic Experts Group. Information technology

— digital compression and coding of continuous-tone still images: Requirements and guidelines. Technical Report ITU-T T.81, International Telecommunication Union (ITUT), 1992. 2, 4, 5, 3

- [24] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. In International Conference on Learning Representations, 2018. 2
- [25] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24174–24184, 2024. 3
- [26] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36: 65484–65516, 2023. 1, 3, 7
- [27] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 5
- [28] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. arXiv preprint arXiv:2404.07724, 2024. 1, 2
- [29] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 1, 7, 8
- [30] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025. 1, 3
- [31] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise, 2025. 2, 3, 6, 7
- [32] Tianhong Li, Qinyi Sun, Lijie Fan, and Kaiming He. Fractal generative models. arXiv preprint arXiv:2502.17437, 2025. 3, 7
- [33] Ziqiang Li, Pengfei Xia, Xue Rui, and Bin Li. Exploring the effect of high-frequency components in gans training. ACM Trans. Multimedia Comput. Commun. Appl., 19(5), 2023. 2
- [34] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025. 1
- [35] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. 3
- [36] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 2
- [37] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference, 2024. 3
- [38] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740,

2024. 1, 3, 7, 8

- [39] Zehong Ma, Longhui Wei, Feng Wang, Shiliang Zhang, and Qi Tian. Magcache: Fast video generation with magnitudeaware cache. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 3
- [40] Zehong Ma, Ruihan Xu, and Shiliang Zhang. Pixelgen: Pixel diffusion beats latent diffusion with perceptual loss. arXiv preprint arXiv:2602.02493, 2026. 3

- [41] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021. 5
- [42] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3
- [43] Namuk Park and Songkuk Kim. How do vision transformers work? In International Conference on Learning Representations, 2022. 2
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 1, 3, 4, 6, 7, 8

- [45] William B Pennebaker and Joan L Mitchell. JPEG: Still image data compression standard. Springer Science & Business Media, 1992. 2
- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 3
- [47] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 8
- [48] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 5
- [49] Chenyang Si, Weihao Yu, Pan Zhou, Yichen Zhou, Xinchao Wang, and Shuicheng Yan. Inception transformer. Advances in Neural Information Processing Systems, 35:23495–23509,

2022. 2

- [50] Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the diffusability of autoencoders. In Forty-second International Conference on Machine Learning, 2025. 2
- [51] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020. 1
- [52] Tianhui Song, Weixin Feng, Shuai Wang, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Dmm: Building a versatile image generation model via distillation-based model merging. arXiv preprint arXiv:2504.12364, 2025. 3
- [53] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 1

- [54] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023. 1, 2, 3, 7
- [55] Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Dim: Diffusion mamba

- for efficient high-resolution image synthesis. arXiv preprint arXiv:2405.14224, 2024. 3
- [56] Yuchuan Tian, Zhijun Tu, Hanting Chen, Jie Hu, Chao Xu, and Yunhe Wang. U-dits: Downsample tokens in u-shaped diffusion transformers, 2024. 3
- [57] Yuchuan Tian, Hanting Chen, Mengyu Zheng, Yuchen Liang, Chao Xu, and Yunhe Wang. U-repa: Aligning diffusion u-nets to vits. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 3
- [58] Yuchuan Tian, Jing Han, Chengcheng Wang, Yuchen Liang, Chao Xu, and Hanting Chen. Dic: Rethinking conv3x3 designs in diffusion models. CoRR, abs/2501.00603, 2025. 3
- [59] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [60] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1
- [61] Michael Tschannen, Andr´e Susano Pinto, and Alexander Kolesnikov. Jetformer: An autoregressive generative model of raw images and text. arXiv preprint arXiv:2411.19722,

2024. 7

- [62] Haohan Wang, Xindi Wu, Zeyi Huang, and Eric P Xing. High-frequency component helps explain the generalization of convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8684–8694, 2020. 2
- [63] Haoyu Wang, Hao Tang, Donglin Di, Zhilu Zhang, Wangmeng Zuo, Feng Gao, Siwei Ma, and Shiliang Zhang. Mosa: Motion-coherent human video generation via structureappearance decoupling. arXiv preprint arXiv:2508.17404,

2025. 3

- [64] Haoyu Wang, Zhilu Zhang, Donglin Di, Shiliang Zhang, and Wangmeng Zuo. Mv-vton: Multi-view virtual try-on with diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7682–7690, 2025.
- [65] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Exploring dcn-like architecture for fast image generation with arbitrary resolution. Advances in Neural Information Processing Systems, 37:87959–87977, 2024. 3
- [66] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv preprint arXiv:2507.23268, 2025. 1, 3, 6, 7, 8
- [67] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025. 3, 6, 1
- [68] Zhe Wang, Ziqiu Chi, Yanbing Zhang, et al. Fregan: Exploiting frequency components for training gans under limited data. Advances in Neural Information Processing Systems, 35:33387–33399, 2022. 2
- [69] Zidong Wang, Lei Bai, Xiangyu Yue, Wanli Ouyang, and Yiyuan Zhang. Native-resolution image synthesis. arXiv preprint arXiv:2506.03131, 2025. 1

- [70] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 7, 8
- [71] Tao Wu, Xuewei Li, Zhongang Qi, Di Hu, Xintao Wang, Ying Shan, and Xi Li. Spherediffusion: Spherical geometryaware distortion resilient diffusion model. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6126– 6134, 2024. 1
- [72] Tao Wu, Yong Zhang, Xiaodong Cun, Zhongang Qi, Junfu Pu, Huanzhang Dou, Guangcong Zheng, Ying Shan, and Xi Li. Videomaker: Zero-shot customized video generation with the inherent force of video diffusion models. arXiv preprint arXiv:2412.19645, 2024. 1
- [73] Tao Wu, Yibo Jiang, Yehao Lu, Zhizhong Wang, Zeyi Huang, Zequn Qin, and Xi Li. Multicrafter: Highfidelity multi-subject generation via disentangled attention and identity-aware preference alignment. arXiv preprint arXiv:2509.21953, 2025. 3
- [74] Tao Wu, Yong Zhang, Xintao Wang, Xianpan Zhou, Guangcong Zheng, Zhongang Qi, Ying Shan, and Xi Li. Customcrafter: Customized video generation with preserving motion and concept composition abilities. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 8469– 8477, 2025. 1
- [75] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 1
- [76] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025. 1, 3
- [77] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 3, 4, 6, 7, 1
- [78] Xiaoyu Yue, Zidong Wang, Zeyu Lu, Shuyang Sun, Meng Wei, Wanli Ouyang, Lei Bai, and Luping Zhou. Diffusion models need visual priors for image generation. arXiv preprint arXiv:2410.08531, 2024. 3
- [79] Shuangfei Zhai, Ruixiang Zhang, Preetum Nakkiran, David Berthelot, Jiatao Gu, Huangjie Zheng, Tianrong Chen, Miguel Angel Bautista, Navdeep Jaitly, and Josh Susskind. Normalizing flows are capable generative models. arXiv preprint arXiv:2412.06329, 2024. 3
- [80] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024. 3, 7

## DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation Supplementary Material

### A. Comparison with JiT

lar parameter counts and computation FLOPs to our DeCo. Training memory and speed are measured with batch size of 256 on 8×A800 GPUs, while inference memory and time are measured on a single A800 with batch size of 1.

DiT Pixel Decoder (High-Freq Shortcut)

[Figure 35]

[Figure 36]

[Figure 37]

xlow  xt

#### B.2. Class-to-Image Generation

Low-Freq Components High-Freq Details Noisy Input

For class-to-image generation experiments on ImageNet, we first train the model at a 256×256 resolution for 320 epochs. Subsequently, we fine-tune the model for additional 20 epochs at a 512×512 resolution. During inference, we use 100 Euler steps with CFG [18] and guidance interval [28]. Inference latency is measured on a single A800 GPU. The batch size and learning rate follow the default settings previously described. We use a global batch size of 256 and the AdamW optimizer with a constant learning rate of 1e-4. We set the CFG scale to 3.2 for the 256 × 256 resolution (320 epochs) and 5.0 for the 512×512 resolution (340 epochs). The CFG scale is set to 3.0 for the model of 800 epochs at a 256 × 256 resolution. The guidance interval [28] is set to 0.1 following previous work [66].

Figure 7. Comparison with JiT.

The key idea of DeCo is to provide a high-freq shortcut via the pixel decoder, enabling an implicit frequency decoupling. High-freq components include noisy input xt and high-freq details ∆. In standard DiTs, high-freq signals tend to be suppressed after the patchification and deep transformer layers. Since pixel decoder directly models each pixel of the raw input, DeCo can automatically use this high-freq shortcut and offload high-freq modeling to pixel decoder, enabling DiT to focus on low-freq components xlow. Unlike JiT, which explicitly predicts x0 = xlow + ∆, our DiT implicitly models xlow, excluding hard high-freq details ∆. The pixel decoder, guided by xlow, only needs to fit a simple JiT-style formulation vθ = x

low+(∆−xt)

#### B.3. Text-to-Image Generation

t . This separation is learned end-to-end. Visualization in Fig. 2(c), spectrum analysis in Fig. 4, and superior performance over “JiT+REPA” in Tab. 1 validate our implicit decoupling.

For text-to-image generation, we trained our model on the BLIP3o [5] dataset, which contains approximately 36M pretraining images and 60k high-quality instruction-tuning data. We adopt Qwen3-1.7B [75] as the text encoder. The entire training takes about 6 days on 8× H800 GPUs. We adopt Qwen3-1.7B [75] as the text encoder. To improve the alignment of frozen text features [12], we jointly train several transformer layers on the frozen text features similar to Fluid [12]. The total batch size is 1536 for 256 × 256 resolution pretraining and 512 for 512 × 512 resolution pretraining. Following PixNerd [66], we pretrain DeCo on 256 × 256 resolution for 200K steps and pretrain on 512 × 512 resolution for 80K steps. We further fine-tune the pretrained DeCo on BLIP3o-60k with 40k steps at the 512 × 512 resolution following PixNerd. We adopt the gradient clip to stabilize training. The whole training only takes about 6 days on 8× H800 GPUs. We use the Adams-2nd solver with 25 steps as the default choice for sampling. The cfg scale is set to 4.0. We leave the native resolution [69] or native aspect training [13, 15, 34] as future works.

### B. Implementary Details

#### B.1. Baseline Comparisons

In this subsection, we summarize the settings used for all baseline comparisons. In the baseline comparisons, all diffusion models are trained on ImageNet at 256×256 resolution for 200k iterations using a large DiT variant. Following previous works [44, 66], we use a global batch size of 256 and the AdamW optimizer with a constant learning rate of 1e-4. Both baseline and DeCo adopt SwiGLU [59, 60], RoPE2d [53], and RMSNorm, and are trained with lognorm sampling and REPA [77]. The patch size of DiT’s input is set to 16 for both baseline and our DeCo. The patch size of pixel decoder is set to 1. Our main architectural change on the baseline is to replace the final two DiT blocks of the baseline with our proposed pixel decoder.

For inference, we use 50 Euler steps without classifierfree guidance [18] (CFG) for all models except PixelFlow [6], which requires 100 steps. We also report results for the two-stage DiT-L/2 that requires a VAE and the recent pixel diffusion models PixelFlow [6] and PixNerd [66]. For a fair comparison, we further integrate DDT [67] into the pixel diffusion to form PixDDT, which has the simi-

#### B.4. Experiment Configurations

Table 5 summarizes the experiment configurations for DeCo-L/16, DeCo-XL/16, and DeCo-XXL/16. In practice, we follow the training setups from previous works such as DiT [44], SiT [38], and PixNerd [66]. Besides, we sweep

###### DeCo-L DeCo-XL DeCo-XXL

architecture DiT depth 22 28 16 hidden dim 1024 1152 1536 heads 16 16 24 params 426M 682M 1.1B decoder depth 3 decoder hidden dim 32 patch size 16 image size 256 (other settings: 512) training

optimizer AdamW [36], β1, β2 = 0.9, 0.999 batch size 256 learning rate 1e-4 lr schedule constant weight decay 0 ema decay 0.9999 time sampler logit(t)∼N(µ, σ2), µ = 0, σ = 1 noise scale 1.0 sampling ODE solver Euler ODE steps 100 time steps linear in [0.0, 1.0] CFG scale range [3.0-3.2] (256×256), [4.5-5.0] (512×512) CFG interval [28] [0.1, 1]

Table 5. Configurations of experiments.

the CFG scale within the given ranges using an interval of 0.1.

### C. Text-to-Image Prompts

Below, we list the prompts used for text-to-image generation in Fig. 1. These prompts cover a mix of animals, people, and scenes to evaluate semantic understanding and visual detail generation.

- • A lovely horse stands in the bedroom.
- • A baby cat stands on two legs, wearing a chothes.
- • A cyberpunk woman with glowing tattoos and a mechanical arm beneath a holographic sky.
- • A man sipping coffee on a sunny balcony filled with potted plants, wearing linen clothes and sunglasses, basking in the morning light.
- • A beautiful woman.
- • A cute panda is wielding a sword in realistic style.
- • An extremely happy American Cocker Spaniel is smiling and looking up at the camera with his head tilted to one side.
- • A raccoon wearing a detective’s hat, observing something with a magnifying glass.
- • Close-up of an aged man with weathered features and sharp blue eyes peering wisely from beneath a tweed flat cap.

### D. Quantization Tables

In our DeCo, we use the normalized reciprocal of scaled JPEG quantization tables as adaptive weights to emphasize different frequency components. These tables are a core component of the JPEG compression standard and are designed based on properties of the human visual system (HVS) [45].

As shown in Sec. 3.3, a quantization table is an 8×8 matrix that determines the compression level for each frequency coefficient after the Discrete Cosine Transform (DCT). The JPEG standard uses two separate tables: one for the luminance (Y) component and another for the chrominance (Cb/Cr) components. This design is based on key characteristics of human perception. The core principle is that the human eye is not equally sensitive to all visual information. Specifically, two HVS properties are crucial. Firstly, the human eye is much more sensitive to lowfrequency components than to high-frequency components. Secondly, the eye is more sensitive to changes in brightness (luminance) than in color (chrominance) [45].

Based on extensive experiments, the standard base quantization tables Qbase in Fig. 8 were developed to reflect these properties [45]. These tables have smaller values (finer quantization intervals) for low-frequency coefficients, which are perceptually more important. Conversely, these tables have larger values (coarser quantization intervals) for high-frequency coefficients, as the resulting information loss is less noticeable to the human eye. Similarly, the luminance table generally has smaller values than the chrominance table. These base tables can be scaled using a quality factor q to create new scaled quantization tables Qcur for different compression levels.

Since a smaller quantization step implies that a frequency component is more significant to human perception, we use the normalized reciprocal of the scaled quantization tables as adaptive weights, i.e., Q1

with normalization. This allows us to assign a higher weight to the frequency components that are visually more important in our frequency-aware flow-matching loss LFreqFM.

cur

### E. Pseudocodes for DeCo

#### E.1. Training Step of DeCo

In Algorithm 1, we provide the pseudocodes for the training step of DeCo. DeCo utilizes the DiT to specialize in low-frequency semantic modeling with downsampled small-scale inputs x¯t. Semantic cues c are hence incorporated with a lightweight pixel decoder to reconstruct highfrequency signals. In other words, the pixel decoder takes the low-frequency semantics c from DiT as condition and predicts pixel velocities vθ with a high-resolution input xt. This new paradigm hence frees the DiT to specialize in

- Algorithm 2 K-Means Visualization in Fig. 2 (c) # Feats: DiT outputs (T, H, W, C) # I: generated images (T, H, W, 3) # T: Sampling steps for t in T:

# Flatten spatial dimensions f = Feats[t].reshape(-1, C)

# Cluster pixel-wise features labels = kmeans(f, n_clusters=8)

# Map clusters to visualization vis = colormap(labels).reshape(H, W)

plot(vis, I[t])

- Algorithm 3 DCT Spectral Analysis

Base Luminance (Y)

Base Chrominance (Cb/Cr)

|17|18|24|47|99|99|99|99|
|---|---|---|---|---|---|---|---|
|18|21|26|66|99|99|99|99|
|24|26|56|99|99|99|99|99|
|47|66|99|99|99|99|99|99|
|99|99|99|99|99|99|99|99|
|99|99|99|99|99|99|99|99|
|99|99|99|99|99|99|99|99|
|99|99|99|99|99|99|99|99|

|16|11|10|16|24|40|51|61|
|---|---|---|---|---|---|---|---|
|12|12|14|19|26|58|60|55|
|14|13|16|24|40|57|69|56|
|14|17|22|29|51|87|80|62|
|18|22|37|56|68|109|103|77|
|24|35|55|64|81|104|113|92|
|49|64|78|87|103|121|120|101|
|72|92|95|98|112|100|103|99|

Qbase

Base Quantization Tables

Scaled Luminance (Y)

Scaled Chrominance (Cb/Cr)

|5|3|3|5|7|12|15|18|
|---|---|---|---|---|---|---|---|
|4|4|4|6|8|17|18|16|
|4|4|5|7|12|17|20|17|
|4|5|7|9|15|26|24|18|
|5|7|11|17|20|32|31|23|
|7|10|16|19|24|31|34|27|
|15|19|23|26|31|36|36|30|
|21|27|28|29|33|30|31|29|

|5|5|7|14|30|30|30|30|
|---|---|---|---|---|---|---|---|
|5|6|8|20|30|30|30|30|
|7|8|17|30|30|30|30|30|
|14|20|30|30|30|30|30|30|
|30|30|30|30|30|30|30|30|
|30|30|30|30|30|30|30|30|
|30|30|30|30|30|30|30|30|
|30|30|30|30|30|30|30|30|

Qcur

Scaled Quantization Tables in Quality 85

Figure 8. Base and Scaled Quantization Tables.

# V: Predicted velocity (B * T, ori_H, ori_W, C) # Feats: DiT outputs (B * T, H, W, C)

Algorithm 1 Training step

# Pre-compute frequency scan order # The block size of DCT is set to 8 idx = ZigZagIndices(block_size=8) F_energy, V_energy = 0, 0 F_num_blocks, V_num_blocks =0, 0 for f, v in (Feats, V):

# θDiT: DiT network # θDec: Pixel Decoder network

- # x0: training batch

- # y: class label or textual prompt # Qcur: scaled quantization tables in quality 85.

# Split input into patches

# Prepare inputs t = sample t()

f_patches = Unfold(f, kernel_size=8) v_patches = Unfold(v, kernel_size=8)

x1 = randn like(x0) xt = (1-t)x0 + tx1 # original scale x¯t = patchify(xt, patch_size=16) # small-scale

# 2D Discrete Cosine Transform f_freq = DCT2D(f_patches) f_energy = f_freq ** 2 v_freq = DCT2D(v_patches) v_energy = v_freq ** 2

# Prepare ground-truth velocities vt = x1 - x0 Vt = DCT2D(RGB2YCbCr(vt))

# Accumulate energy following ZigZag order # Maps 2D (u,v) to 1D frequency index F_energy += Sum(f_energy.reorder(idx)) F_num_blocks += len(f_patches) V_energy += Sum(v_energy.reorder(idx)) V_num_blocks += len(v_patches)

# Generate low-frequency semantic condition c = θDiT(x¯t, t, y) # Predict velocity conditioned on c

- vθ = θDec(xt, t, c) Vθ = DCT2D(RGB2YCbCr(vθ)) # Compute Loss FM_loss = mean(∥vθ − vt∥2)

- w = 1/Qcur w = w / w.mean() # normalized adaptive weights

# Log-scale Normalization Feats_S = log(1 + F_energy / F_num_blocks) Feats_S = Feats_S / max(Feats_S) V_S = log(1 + V_energy / V_num_blocks) V_S = V_S / max(V_S)

FreqFM_loss = mean(w * ∥Vθ − Vt∥2) loss = FM_loss + FreqFM_loss + REPA_loss

plot(Feats_S) plot(V_S)

modeling semantics, and allows for more specialized details generation. To emphasize visually salient frequencies and suppress perceptually insignificant high-frequency components, we further introduce a frequency-aware FlowMatching loss LFreqFM inspired by the JPEG [23]. A REPA [77] loss is used in both our Baseline and DeCo.

#### E.2. K-Means Visualization in Fig. 2 (c)

In Algorithm 2, we provide the pseudocodes for the KMeans visualization in Fig. 2 (c). The number of clusters in K-Means is set to 8. We uniformly select 4 timesteps from the sampling process and visualize the clustering results at

these timesteps.

#### E.3. DCT energy distribution in Fig. 4

In Algorithm 3, we provide the pseudocodes for DCT spectral analysis of Fig. 4. We apply an 8×8 DCT to transfrom the DiT outputs and pixel velocities into frequency domain. Each 8×8 patch yields 64 frequency coefficients, which are then converted into energy via a square operation. We reorder these energies from low to high frequency using standard zigzag indexing, where lower indices correspond to lower-frequency components. Finally, we apply log-scale normalization to rescale all energies to the range [0, 1] for comparison. As demonstrated in Fig. 4, compared with baseline, DeCo suppresses high-frequency signals in DiT outputs while preserving strong high-frequency energy in pixel velocity, confirming effective frequency decoupling. The distribution is computed on 10K images across all diffusion steps, i.e., B=10,000 and T=100.

### F. More Visualizations

In this section, we provide more visualizations, including text-to-image generation in Fig. 9, class-to-image generation at a 256×256 resolution in Fig. 10, and class-to-image generation at a 512 × 512 resolution in Fig. 11. Our DeCo supports multiple languages with the Qwen3 text encoder after pretraining on the BLIP3o dataset [5], such as Chinese, Japanese, and English.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

蒸汽朋克风格的长城，巨大齿 轮与空中飞艇，黄昏史诗光影 (Steampunk Great Wall with huge gears and airships, epic dusk light)

破碎的青花瓷风少女面孔，细 腻纹理，超现实主义 (Shattered blue-and-white porcelain girl’s face, fine texture, surreal)

赛博朋克大熊猫，佩戴机械义 肢，在霓虹街道漫步 (Cyberpunk panda with mechanical prosthetics walking through neon streets)

透明冰晶材质的龙，盘踞雪山 之巅 (A transparent ice-crystal dragon coiled atop a snowy peak)

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

海辺に置かれた木箱

雨の草原を歩く馬

朝日を浴びる海辺の少女 (A girl on the seaside bathed in morning light)

悲しいロボット (A sad robot)

(A wooden box placed on the

(A horse walking on a rainy

seaside)

grassland)

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Old fisherman holding a

Viking warrior wearing a red

A beautiful girl with hair flowing like a cascading waterfall.

Portrait of a woman made entirely of porcelain and gold.

galaxy inside a glass jar.

sweater

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Pixel art sunset over a retro synthwave city.

A highway leading straight into a giant purple moon with a person.

A fox sleeping inside a large transparent lightbulb

A lion made of burning charcoal and glowing embers.

Figure 9. More Qualitative results of text-to-image generation at a 512×512 resolution. Our DeCo supports multiple languages with the Qwen3 text encoder, such as Chinese, Japanese, and English.

[Figure 54]

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

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

###### Figure 10. More qualitative results of class-to-image generation at a 256×256 resolution.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

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

###### Figure 11. Qualitative results of class-to-image generation at a 512×512 resolution.

