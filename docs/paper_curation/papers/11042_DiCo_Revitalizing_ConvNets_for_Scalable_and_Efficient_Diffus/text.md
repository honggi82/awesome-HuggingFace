## DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling

# arXiv:2505.11196v2[cs.CV]22Sep2025

#### Yuang Ai1,2 Qihang Fan1,2 Xuefeng Hu3 Zhenheng Yang3 Ran He1,2 Huaibo Huang1,2∗

1CASIA 2UCAS 3ByteDance

Code and models: https://github.com/shallowdream204/DiCo

[Figure 1]

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

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Figure 1: Diffusion ConvNet achieves superior image quality with high efficiency. We show samples from two of our DiCo-XL models trained on ImageNet at 512×512 and 256×256 resolution.

### Abstract

Diffusion Transformer (DiT), a promising diffusion model for visual generation, demonstrates impressive performance but incurs significant computational overhead. Intriguingly, analysis of pre-trained DiT models reveals that global selfattention is often redundant, predominantly capturing local patterns—highlighting the potential for more efficient alternatives. In this paper, we revisit convolution as an alternative building block for constructing efficient and expressive diffusion models. However, naively replacing self-attention with convolution typically results in degraded performance. Our investigations attribute this performance gap to the higher channel redundancy in ConvNets compared to Transformers. To resolve this, we introduce a compact channel attention mechanism that promotes the activation of more diverse channels, thereby enhancing feature diversity. This leads to Diffusion ConvNet (DiCo), a family of diffusion models built entirely from

∗Corresponding author: Huaibo Huang <huaibo.huang@cripac.ia.ac.cn>

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

standard ConvNet modules, offering strong generative performance with significant efficiency gains. On class-conditional ImageNet generation benchmarks, DiCo-XL achieves an FID of 2.05 at 256×256 resolution and 2.53 at 512×512, with a 2.7× and 3.1× speedup over DiT-XL/2, respectively. Furthermore, experimental results on MS-COCO demonstrate that the purely convolutional DiCo exhibits strong potential for text-to-image generation.

### 1 Introduction

Diffusion models [73, 75, 33, 74, 76] have sparked a transformative advancement in generative learning, demonstrating remarkable capabilities in synthesizing highly photorealistic visual content. Their versatility and effectiveness have led to widespread adoption across a broad spectrum of realworld applications, including text-to-image generation [66, 69, 67], image editing [59, 46, 10], image restoration [45, 3, 4], video generation [36, 88, 7], and 3D content creation [64, 87, 84].

Early diffusion models (e.g., ADM [14] and Stable Diffusion [67]) primarily employed hybrid U-Net [68] architectures that integrate convolutional layers with self-attention. More recently, Transformers [83] have emerged as a more powerful and scalable backbone [62, 6], prompting a shift toward fully Transformer-based designs. As a result, Diffusion Transformers (DiTs) are gradually supplanting traditional U-Nets, as seen in leading diffusion models such as Stable Diffusion 3 [20], FLUX [49], and Sora [9]. However, the quadratic computational complexity of self-attention presents substantial challenges, especially for high-resolution image synthesis. Recent efforts [100, 79, 25, 63, 91, 2] have explored more efficient alternatives, focusing on linear-complexity RNNlike architectures, such as Mamba [26] and Gated Linear Attention [92]. While these models improve efficiency, their causal design inherently conflicts with the bidirectional nature of visual generation [30, 55], limiting their effectiveness. Furthermore, as illustrated in Fig. 3, even with highly optimized CUDA implementations, their runtime advantage over conventional DiTs remains modest in high-resolution settings. This leads us to a key question: Is it possible to design a hardware-efficient diffusion backbone that also preserves strong generative capabilities like DiTs?

To approach this question, we begin by examining the characteristics that underlie the generative power of DiTs. In visual recognition tasks, the success of Vision Transformers [18] is often credited to the self-attention’s ability to capture long-range dependencies [42, 23, 22]. However, in generative tasks, we observe a different dynamic. As depicted in Fig. 4, for both pre-trained class-conditional (DiT-XL/2 [62]) and text-to-image (PixArt-α [12] and FLUX [49]) DiT models, when queried with an anchor token, attention predominantly concentrates on nearby spatial tokens, largely disregarding distant ones. This finding suggests that computing global attention may be redundant for generation, underscoring the significance of local spatial modeling. Unlike recognition tasks, where long-range interactions are critical for global semantic reasoning, generative tasks appear to emphasize finegrained texture and local structural fidelity. These observations reveal the inherently localized nature of attention in DiTs and motivate the pursuit of more efficient architectures.

In this work, we revisit convolutional neural networks (ConvNets) and propose Diffusion ConvNet (DiCo), a simple yet highly efficient convolutional backbone tailored for diffusion models. Compared to self-attention, convolutional operations are more hardware-friendly, offering significant advantages for large-scale and resource-constrained deployment. While substituting self-attention with convolution substantially improves efficiency, it typically results in degraded performance. As illustrated

- in Fig. 5, this naive replacement introduces pronounced channel redundancy, with many channels remaining inactive during generation. We hypothesize that this stems from the inherently stronger representational capacity of self-attention compared to convolution. To address this, we introduce a compact channel attention (CCA) mechanism, which dynamically activates informative channels with lightweight linear projections. As a channel-wise global modeling approach, CCA enhances the model’s representational capacity and feature diversity while maintaining low computational overhead. Unlike modern recognition ConvNets that rely on large, costly kernels [15, 28], DiCo adopts a streamlined design based entirely on efficient 1×1 pointwise convolutions and 3×3 depthwise convolutions. Despite its architectural simplicity, DiCo delivers strong generative performance.

As shown in Fig. 2 and Fig. 3, DiCo models outperform recent diffusion models on both the ImageNet 256×256 and 512×512 benchmarks. Notably, our DiCo-XL models achieve impressive FID scores of 2.05 and 2.53 at 256×256 and 512×512 resolution, respectively. In addition to performance gains,

###### (a) FID vs. FLOPs w/img_size = 256 (b) FID vs. Runtime w/ img_size = 256

DiT (ICCV23) DiC (CVPR25) DiG (CVPR25) DiCo (ours)

DiT (ICCV23) DiT-FlashAttn DiG (CVPR25)

65

65

FID-50K(400Kiter.)

###### FID-50K(400Kiter.)

DiCo (ours)

45

45

25

25

2.9xfaster,1.6xgain 0 125

5

5

0

3 6 9 12

15

25 50 75

100

FLOPs (G)

Runtime (ms)

- Figure 2: Comparison of performance and efficiency with recent diffusion models (DiT [62], DiC [81], and DiG [100]) on ImageNet 256×256. Our proposed DiCo achieves the best performance while maintaining high efficiency. Compared to DiG-XL/2 with CUDA-optimized Flash Linear Attention [92], DiCo-XL runs 2.9× faster and achieves a 1.6× improvement in FID.

|U-ViT-L|/4| | |
|---|---|---|---|
|U-ViT|-H/4| |DiM-H|
| |DiT-<br><br>7.8xfa|XL/2 ster<br><br>,1.5xg|ain<br><br>|
|DiC|o-XL (ours|)|DiS-H/2|

Model Variants

0

30

60

90

Small Base Large XLarge

Runtime(ms)

DiT (ICCV23) DiS (arXiv24) DiG (CVPR25)

DiCo (ours)

6.8xfaster

2

4.4

2.8

0 35 70 105 140 Runtime (ms)

FID-50K(w/cfg)

3.6

(a) Runtime Comparison w/ img_size = 512 (b) FID vs. Runtime w/ img_size = 512

- Figure 3: (a) Runtime comparison between DiT [62], DiS [25] (with Mamba [26]), DiG [100] (with Gated Linear Attention [92]), and our DiCo at 512×512 resolution. DiCo is 3.3× faster than DiS at the small model scale and 6.8× faster at the XL scale. (b) FID vs. runtime of various methods on ImageNet 512×512. DiCo-XL achieves an FID of 2.53 while maintaining high efficiency.

DiCo models exhibit considerable efficiency advantages over attention-based [83], Mamba-based [26], and linear attention-based [44] diffusion models. Specifically, at 256×256 resolution, DiCo-XL achieves a 26.4% reduction in Gflops and is 2.7× faster than DiT-XL/2 [62]. At 512×512 resolution, DiCo-XL operates 7.8× and 6.7× faster than the Mamba-based DiM-H [79] and DiS-H/2 [25] models, respectively. Our largest model, DiCo-H with 1 billion parameters, further reduces the FID on ImageNet 256×256 to 1.90. In addition, we validate the applicability of DiCo for text-to-image generation on the MS-COCO dataset. These results collectively highlight the strong potential of DiCo in diffusion-based generative modeling.

Overall, the main contributions of this work can be summarized as follows:

- • We analyze pre-trained DiT models and reveal significant redundancy and locality within their global attention mechanisms. These findings may inspire researchers to develop more efficient strategies for constructing high-performing diffusion models.
- • We propose DiCo, a simple, efficient, and powerful ConvNet backbone for diffusion models. By incorporating compact channel attention, DiCo significantly improves representational capacity and feature diversity without sacrificing efficiency.
- • We conduct extensive experiments on class-conditional ImageNet benchmarks. DiCo outperforms recent diffusion models in both generation quality and speed. Furthermore, the purely convolutional DiCo demonstrates strong potential in text-to-image generation.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- (a) Attention maps from DiT blocks of DiT-XL/2-512 [62].

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

- (b) Attention maps from DiT blocks of PixArt-α-512 [12].

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

(c) Attention maps from MM-DiT [20] blocks of FLUX.1-dev [49].

- Figure 4: Visualization of attention maps from well-known DiT models. The intensity of the blue color indicates the magnitude of attention scores. For self-attention across different layers in these models, only a few neighboring tokens contribute significantly to the attention distribution of a given anchor token (red box), resulting in highly redundant and localized representations.

0 20 40 60 80 100 120

Channel Index

0.00

0.05

0.10

0.15

0.20

0.25

0.30

ActivationScores

(a) DiT [62]

0 20 40 60 80 100 120

Channel Index

0.00

0.05

0.10

0.15

0.20

0.25

ActivationScores

(b) DiCo w/o channel activation

0 20 40 60 80 100 120

Channel Index

0.00

0.02

0.04

0.06

0.08

0.10

0.12

0.14

0.16

ActivationScores

(c) DiCo (ours)

- Figure 5: Comparison of channel activation scores across different diffusion models. Channel activation scores are computed using ReLU followed by global average pooling on the final layer’s self-attention or convolution outputs [102]. Directly replacing self-attention in DiT with convolution introduces significant channel redundancy, as most channel activation scores remain at low levels.

### 2 Related Work

Architecture of Diffusion Models. Early diffusion models commonly employ U-Net [68] as the foundational architecture [14, 34, 67]. More recently, a growing body of research has explored Vision Transformers (ViTs) [18] as alternative backbones for diffusion models, yielding remarkable results [62, 6, 58, 96, 65, 54]. Notably, DiT [62] has demonstrated the excellent performance of transformer-based architectures, achieving SOTA performance on ImageNet generation. However, the quadratic computational complexity inherent in ViTs presents significant challenges in terms of efficiency for long sequence modeling. To mitigate this, recent studies have explored the use of RNN-like architectures with linear complexity, such as Mamba [26] and linear attention [44], as backbones for diffusion models [25, 100, 79, 91, 63]. DiS [25] and DiM [79] employ Mamba to reduce computational overhead, while DiG [100] leverages Gated Linear Attention [92] to achieve competitive performance with improved efficiency. In this work, we revisit ConvNets as backbones for diffusion models. We show that, with proper design, pure convolutional architectures can achieve superior generative performance, providing an efficient and powerful alternative to DiTs.

ConvNet Designs. Over the past decade, convolutional neural networks (ConvNets) have achieved remarkable success in computer vision [31, 41, 89, 5, 19]. Numerous lightweight ConvNets have been developed for real-world deployment [39, 71, 38, 16]. Although Transformers have gradually become the dominant architecture across a wide range of tasks, their substantial computational overhead remains a significant challenge. Many modern ConvNet designs achieve competitive performance

while maintaining high efficiency. ConvNeXt [57] explores the modernization of standard ConvNets and achieves superior results compared to transformer-based models. RepLKNet [15] investigates the use of large-kernel convolutions, expanding kernel sizes up to 31×31. UniRepLKNet [17] further generalizes large-kernel ConvNets to domains such as audio, point clouds, and time-series forecasting. In this work, we explore the potential of pure ConvNets for diffusion-based image generation, and show that simple, efficient ConvNet designs can also deliver excellent performance.

### 3 Method

#### 3.1 Preliminaries

Diffusion formulation. We first revisit essential concepts underpinning diffusion models [33, 76]. Diffusion models are characterized by a forward noising procedure that progressively injects noise into a data sample x0. Specifically, this forward process can be expressed as:

T

q(xt|xt−1),q(xt|x0) = N(xt;√α¯tx0,(1 − α¯t)I), (1)

q(x1:T|x0) =

t=1

where α¯t are predefined hyperparameters. The objective of a diffusion model is to learn the reverse process: pθ(xt−1|xt) = N(µθ(xt),Σθ(xt)), where neural networks parameterize the mean and covariance of the process. The training involves optimizing a variational lower bound on the loglikelihood of x0, which simplifies to:

DKL(q∗(xt−1|xt,x0)||pθ(xt−1|xt)). (2)

L(θ) = −p(x0|x1) +

t

To simplify training, the model’s predicted mean µθ can be reparameterized as a noise predictor ϵθ. The objective then reduces to a straightforward mean-squared error between the predicted noise and

the true noise ϵt: Lsimple(θ) = ||ϵθ(xt) − ϵt||22. Following DiT [62], we train the noise predictor ϵθ using the simplified loss Lsimple, while the covariance Σθ is optimized using the full loss L.

Classifier-free guidance. Classifier-free guidance (CFG) [35] is an effective method to enhance sample quality in conditional diffusion models. It achieves such enhancement by guiding the sampling process toward outputs strongly associated with a given condition c. Specifically, it modifies the predicted noise to obtain high p(x|c) as: ϵˆθ(xt,c) = ϵθ(xt,∅) + s · ∇x log p(x|c) ∝ ϵθ(xt,∅) + s · (ϵθ(xt,c) − ϵθ(xt,∅)). where s ≥ 1 controls the guidance strength, and ϵθ(xt,∅) is an unconditional prediction obtained by randomly omitting the conditioning information during training. Following prior works [62, 100], we adopt this technique to enhance the quality of generated samples.

#### 3.2 Network Architecture

Currently, diffusion models are primarily categorized into three architectural types: (1) Isotropic architectures without any downsampling layers, as seen in DiT [62]; (2) Isotropic architectures with long skip connections, exemplified by U-ViT [6]; and (3) U-shaped architectures, such as U-DiT [82]. Motivated by the crucial role of multi-scale features in image denoising [97, 1], we adopt a U-shaped design to construct a hierarchical model. We also conduct an extensive ablation study to systematically compare the performance of these different architectural choices in Table 4.

As illustrated in Fig. 6 (a), DiCo employs a three-stage U-shaped architecture composed of stacked DiCo blocks. The model takes the spatial representation z generated by the VAE encoder as input. For an image of size 256 × 256 × 3, the corresponding z has dimensions 32 × 32 × 4. To process this input, DiCo applies a 3 × 3 convolution that transforms z into an initial feature map z0 with D channels. For conditional information—specifically, the timestep t and class label y—we employ a multi-layer perceptron (MLP) and an embedding layer, serving as the timestep and label embedders, respectively. At each block l within DiCo, the feature map zl−1 is passed through the l-th DiCo block to produce the output zl.

Within each stage, skip connections between the encoder and decoder facilitate efficient information flow across intermediate features. After concatenation, a 1 × 1 convolution is applied to reduce the channel dimensionality. To enable multi-scale processing across stages, we utilize pixel-unshuffle operations for downsampling and pixel-shuffle operations for upsampling. Finally, the output feature zL is normalized and passed through a 3 × 3 convolutional head to predict both noise and covariance.

(c) Conv Module

(b) DiCo Block

(a) Latent Diffusion ConvNet

| | |
|---|---|
| | |

Timestep t

1x1 Conv

Noised Latent

Noise

U-Shaped 32 x 32 x 4 32 x 32 x 4 Architecture

Label y

Scale

32 x 32 x 4

Compact Channel Attn

Feedforward

Conv Embed

Layer Norm & Conv

GELU

Scale & Shift

- Stage 1
- Stage 2
- Stage 3

DiCo Blocks

DiCo Blocks

3x3 DConv

Layer Norm

Concat & Linear

| | |
|---|---|
| | |

1x1 Conv

2x Downsample

2x Upsample

Scale

(d) CCA

DiCo Blocks

DiCo Blocks

Conv Module

Concat & Linear

Sigmoid

Scale & Shift

2x Downsample

2x Upsample

1x1 Conv

Layer Norm

MLP

DiCo Blocks

Global AvgPool

Input Features Cond.

- Figure 6: Architecture of DiCo, which consists of (b) DiCo Block, (c) Conv Module, and (d) Compact Channel Attention (CCA). DConv denotes depthwise convolution.

#### 3.3 DiCo Block

Motivation. As shown in Fig. 4, the self-attention computation in DiT models—whether for class-conditional or text-to-image generation—exhibits a distinctly local structure and significant redundancy. This observation motivates us to replace the global self-attention in DiT with more hardware-efficient operations. A natural alternative is convolution, which is well-known for its ability to efficiently model local patterns. We first attempt to substitute self-attention with a combination of 1 × 1 pointwise convolutions and 3 × 3 depthwise convolutions.

However, the direct replacement leads to a degradation in generation performance. As shown in Fig. 5, compared to DiT, many channels in the modified model remain inactive, indicating substantial channel redundancy. We hypothesize that this performance drop stems from the fact that self-attention, being dynamic and content-dependent, provides greater representational power than convolution, which relies on static weights. To address this limitation, we introduce a compact channel attention mechanism to dynamically activate informative channels. We describe the full design in detail below.

Block designs. The core design of DiCo is centered around the Conv Module, as shown in Fig. 6 (c). We first apply a 1 × 1 convolution to aggregate pixel-wise cross-channel information, followed by a 3 × 3 depthwise convolution to capture channel-wise spatial context. A GELU activation is employed for non-linear transformation. To further address channel redundancy, we introduce a compact channel attention (CCA) mechanism to activate more informative channels. As illustrated

- in Fig. 6 (d), CCA first aggregates features via global average pooling (GAP) across the spatial dimensions, then applies a learnable 1 × 1 convolution followed by a sigmoid activation to generate channel-wise attention weights. Generally, the whole process of Conv Module can be described as:

X)),CCA(X) = X ⊙ Sigmoid(WpGAP(X)), (3) where Wp

Y = Wp

CCA(GELU(WdWp

2

1

is the 1 × 1 point-wise conv, Wd is the depthwise conv, and ⊙ denotes the channel-wise multiplication. As shown in Fig. 5 (c), this simple and efficient design effectively reduces feature redundancy and enhances the representational capacity of the model. To incorporate conditional information from the timestep and label, we follow DiT by adding the input timestep embedding t and label embedding y, and using them to predict the scale parameters α,γ and the shift parameter β.

(·)

Modification for text-to-image. We investigate two different approaches for incorporating textual features into DiCo. The first uses the widely adopted cross-attention [12] mechanism, integrated into the DiCo architecture to fuse text and visual features. The second transforms CLIP text embeddings into dynamic depthwise convolution (DWC) kernels. We pad the text embeddings to a length of 81, feed them through a learnable MLP, and reshape the output into a 9 × 9 kernel. This kernel dynamically modulates DiCo’s features via depthwise convolution. In this way, we can construct a

Algorithm 1 PyTorch code of text conditional depthwise convolution

import torch import torch.nn.functional as F

def text_conditional_dwconv(x, context): # x: (B, C, H, W) input feature maps # context: (B, 77, C) CLIP text embeddings after an MLP # output: (B, C, H, W) output after depthwise convolution B, C, H, W = x.shape context_pad = torch.cat([context, context[:, -1:].expand(-1, 4, -1)], dim=1) # (B, 81, C) kernels = context_pad.reshape(B, 9, 9, C).permute(0, 3, 1, 2).reshape(B * C, 1, 9, 9) x_flat = x.view(1, B * C, H, W) output = F.conv2d(x_flat, kernels, padding=4, groups=B * C).view(B, C, H, W) return output

fully convolutional text-to-image DiCo model without relying on any self-attention or cross-attention operations. We provide its detailed PyTorch implementation in Algorithm 1. Both feature fusion modules are inserted after the Conv Module within each DiCo block.

#### 3.4 Architecture Variants

We establish four model variants—DiCo-S, DiCo-B, DiCo-L, and DiCo-XL—whose parameter counts are aligned with those of DiT-S/2, DiT-B/2, DiT-L/2, and DiT-XL/2, respectively. Compared to their DiT counterparts, our DiCo models achieve a significant reduction in computational cost, with Gflops ranging from only 70.1% to 74.6% of those of DiT. Furthermore, to explore the potential of our design, we scale up DiCo to 1 billion parameters, resulting in DiCo-H. The architectural configurations of these models are detailed in Appendix Table 6.

### 4 Experiments

#### 4.1 Experimental Setup

Datasets and Metrics. Following previous works [62, 100, 81], we conduct experiments on classconditional ImageNet-1K [13] generation benchmark at 256×256 and 512×512 resolutions. We use the Fréchet Inception Distance (FID) [32] as the primary metric to evaluate model performance. In addition, we report the Inception Score (IS) [70], Precision, and Recall [48] as secondary metrics. All these metrics are computed using OpenAI’s TensorFlow evaluation toolkit [14].

Implementation Details. For DiCo-S/B/L/XL, we adopt exactly the same experimental settings as used for DiT. Specifically, we employ a constant learning rate of 1 × 10−4, no weight decay, and a batch size of 256. The only data augmentation applied is random horizontal flipping. We maintain an exponential moving average (EMA) of the DiCo weights during training, with a decay rate of 0.9999. The pre-trained VAE [67] is used to extract latent features. For our largest model, DiCo-H, we follow the training settings of U-ViT [6], increasing the learning rate to 2 × 10−4 and scaling the batch size to 1024 to accelerate training. Additional details are provided in Appendix Sec. B.

#### 4.2 Main Results

Comparison under the DiT Setting. In addition to DiT [62], we also select recent diffusion models, DiG [100] and DiC [81], as baselines, since they similarly follow the experimental setup of DiT.Table 1 presents the comparison results on ImageNet 256×256. Across different model scales trained for 400K iterations, our DiCo consistently achieves the best or second-best performance across all metrics. Furthermore, when using classifier-free guidance (CFG), our DiCo-XL achieves an FID of 2.05 and an IS of 282.17. Beyond performance improvements, DiCo also demonstrates significant efficiency gains compared to both the baselines and Mamba-based models.

Table 2 presents the results on ImageNet 512×512. At higher resolutions, our model demonstrates greater improvements in both performance and efficiency. Specifically, DiCo-XL achieves an FID of 2.53 and an IS of 275.74, while reducing Gflops by 33.3% and achieving a 3.1× speedup compared to DiT-XL/2. These results highlight that our convolutional architecture remains highly efficient and effective for high-resolution image generation.

- Table 1: Comparison under the DiT setting on ImageNet 256×256. The performance at 400K training steps is reported without CFG for early-stage comparison. We mark the best results for each model scale in bold. Throughput (image/s) is measured on A100 with batch size 64 at fp32 precision. DiT and DiG are optimized with FlashAttention-2 and Flash Linear Attention, respectively.

Model Token Mixing Type Gflops Throughput FID↓ IS↑ Pre.↑ Rec.↑

ADM-U [14] Conv + Attn 742 - 3.94 215.84 0.83 0.53 LDM-4 [67] Conv + Attn - - 3.95 178.22 0.81 0.55 U-ViT-H/2 [6] Attn 133.25 73.45 2.29 263.88 0.82 0.57

###### Mamba-based diffusion models.

DiM-H [79] Conv + SSM 210 25.06 2.21 - - DiS-H/2 [25] Conv + SSM - 33.95 2.10 271.32 0.82 0.58 DiffuSSM-XL [91] SSM 280.3 - 2.28 259.13 0.86 0.56 DiMSUM-L/2 [63] Attn + SSM 84.49 59.13 2.11 - - 0.59

###### Baselines and Ours (w/ the same hyperparameters).

DiT-S/2 (400K) [62] Attn 6.06 1234.01 68.40 - - DiC-S (400K) [81] Conv 5.9 - 58.68 25.82 - DiG-S/2 (400K) [100] Conv + Attn 4.30 961.24 62.06 22.81 0.39 0.56 DiCo-S (400K) Conv 4.25 1695.73 49.97 31.38 0.48 0.58

DiT-B/2 (400K) Attn 23.02 380.11 43.47 - - DiC-B (400K) Conv 23.5 - 32.33 48.72 - DiG-B/2 (400K) Conv + Attn 17.07 345.89 39.50 37.21 0.51 0.63 DiCo-B (400K) Conv 16.88 822.97 27.20 56.52 0.60 0.61

DiT-L/2 (400K) Attn 80.73 114.63 23.33 - - DiG-L/2 (400K) Conv + Attn 61.66 109.01 22.90 59.87 0.60 0.64 DiCo-L (400K) Conv 60.24 288.32 13.66 91.37 0.69 0.61

DiT-XL/2 (400K) Attn 118.66 76.90 19.47 - - DiC-XL (400K) Conv 116.1 - 13.11 100.15 - DiG-XL/2 (400K) Conv + Attn 89.40 71.74 18.53 68.53 0.63 0.64 DiCo-XL (400K) Conv 87.30 208.47 11.67 100.42 0.71 0.61

DiT-XL/2 (w/ CFG) Attn 118.66 76.90 2.27 278.24 0.83 0.57 DiG-XL/2 (w/ CFG) Conv + Attn 89.40 71.74 2.07 278.95 0.82 0.60 DiCo-XL (w/ CFG) Conv 87.30 208.47 2.05 282.17 0.83 0.59

DiC-H (w/ CFG) Conv 204.4 - 2.25 - - DiCo-H (w/ CFG) Conv 194.15 117.57 1.90 284.31 0.83 0.61

Scaling the model up. To further explore the potential of our model, we scale it up to 1 billion parameters. As shown in Table 1, compared to DiCo-XL, the larger DiCo-H achieves further improvements in FID (1.90 vs. 2.05), demonstrating the great scalability of our architecture. More comparison results and generated images can be found in the Appendix Sec. D and Sec. E.

Text-to-Image Generation. We follow [6] to conduct small-scale text-to-image generation experiments. Specifically, we adopt the same experimental setup as [96]: training and evaluating models from scratch on MS-COCO [53], using CLIP as the text encoder with a token length of 77.

As shown in Table 3, our DiCo achieves superior generation quality for text-to-image generation. Notably, using text conditional DWC in place of cross-attention further improves throughput while maintaining competitive performance. This suggests that the fully convolutional DiCo has the potential to serve as the backbone for large-scale text-to-image diffusion models.

#### 4.3 Ablation Study

For the ablation study, we use the small-scale model and evaluate performance on the ImageNet 256×256 benchmark to enable fast training speed. All models are trained for 400K iterations and evaluated without CFG. Notably, in this section, self-attention in DiT is not accelerated using FlashAttention-2 to ensure a fair speed comparison with other efficient attention mechanisms.

- Table 2: Comparison under the DiT setting on ImageNet 512×512. We mark the best performance with CFG in bold. The performance at 1.3M/3M training steps is reported without using CFG.

Model Token Mixing Type Gflops Throughput FID↓ IS↑ Pre.↑ Rec.↑

ADM-U [14] Conv + Attn 2813 - 3.85 221.72 0.84 0.53 U-ViT-L/4 [6] Attn 76.52 128.49 4.67 213.28 0.87 0.45 U-ViT-H/4 [6] Attn 133.27 73.42 4.05 263.79 0.84 0.48

###### Mamba-based diffusion models.

DiM-H [79] Conv + SSM 708 7.39 3.78 - - DiS-H/2 [25] Conv + SSM - 8.59 2.88 272.33 0.84 0.56 DiffuSSM-XL [91] Attn + SSM 1066.2 - 3.41 255.06 0.85 0.49

Baselines and Ours (w/ the same hyperparameters). DiT-XL/2 (1.3M) [62] Attn 524.70 18.58 13.78 - - -

DiCo-XL (1.3M) Conv 349.78 57.45 8.10 132.85 0.78 0.62 DiT-XL/2 (3M) Attn 524.70 18.58 12.03 105.25 0.75 0.64 DiCo-XL (3M) Conv 349.78 57.45 7.48 146.35 0.78 0.63

DiT-XL/2 (w/ CFG) Attn 524.70 18.58 3.04 240.82 0.84 0.54 DiCo-XL (w/ CFG) Conv 349.78 57.45 2.53 275.74 0.83 0.56

We analyze both the overall architecture and the contributions of individual components within DiCo to better understand their impact on model performance.

Architecture Ablation. We evaluate the performance of DiCo under various architectural designs and conduct a fair comparison with DiT. As shown in Table 4, DiCo consistently outperforms DiT across all structures while also delivering significant efficiency gains. These results highlight the potential of DiCo as a strong and efficient alternative to DiT.

Component-wise Ablation. We conduct a component-wise analysis of DiCo, examining the effects of the activation function, convolutional kernel size, compact channel attention (CCA), and the conv module (CM). The overall ablation results are summarized in Table 5. Increasing the convolutional kernel size leads to further perfor-

Table 3: Comparison for text-to-image generation on MS-COCO. We follow the setup in [96].

Model Type FID

AttnGAN [90] GAN 35.49 DM-GAN [101] GAN 32.64 VQ-Diffusion [27] Discrete Diffusion 19.75 DF-GAN [78] GAN 19.32 XMC-GAN [98] GAN 9.33 Frido [24] Diffusion 8.97 LAFITE [99] GAN 8.12

U-Net [6] Diffusion 7.32 U-ViT-S/2 [6] Diffusion 5.95 U-ViT-S/2 (Deep) [6] Diffusion 5.48

MMDiT [96] Diffusion 5.30 MMDiT+REPA [96] Diffusion 4.14

DiCo-CrossAttn Diffusion 4.87 DiCo-DWC Diffusion 4.93

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

- 0
- 1

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

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

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

- Figure 7: CCA effectively reduces channel redundancy and enhances feature diversity. Left: Features from the first stage of DiCo without CCA. Right: Features from the first stage of DiCo.

- Table 4: Ablation study on architectural design. We compare different architectural variants, including isotropic, isotropic with skip connections, and U-shape. For all these architectural designs, our DiCo consistently outperforms DiT.

Model Skip Hierarchy #Params Gflops Throughput FID↓ IS↑ Pre.↑ Rec.↑

DiT (iso.) [62] ✗ ✗ 32.9M 6.06 1086.81 67.16 20.41 0.37 0.57 DiCo (iso.) ✗ ✗ 33.7M 5.67 1901.39 60.58 25.44 0.44 0.55

DiT (iso.&skip) ✓ ✗ 34.3M 6.44 1037.55 62.63 22.08 0.39 0.56 DiCo (iso.&skip) ✓ ✗ 35.1M 6.04 1807.22 56.95 26.84 0.46 0.56

DiT (U-shape) ✓ ✓ 33.0M 4.23 1140.93 54.00 28.52 0.42 0.59 DiCo (U-shape) ✓ ✓ 33.1M 4.25 1695.73 49.97 31.38 0.48 0.58

- Table 5: Ablation study on the components of DiCo. ‡ indicates that the model depth and width are adjusted for a fair comparison. We analyze the effects of the activation function, DWC kernel size, compact channel attention (CCA), and the conv module (CM). For the CM, we compare it against several advanced efficient attention mechanisms to validate its effectiveness and efficiency.

Model #Params Gflops Throughput FID↓ IS↑ Pre.↑ Rec.↑

DiCo-S 33.1M 4.25 1695.73 49.97 31.38 0.48 0.58 GELU→ReLU 33.1M 4.25 1695.68 51.26 30.23 0.47 0.57 3×3→5×5 DWC 33.2M 4.29 1628.59 48.03 32.51 0.49 0.58 3×3→7×7 DWC 33.4M 4.34 1469.45 47.49 32.93 0.49 0.59

Compact Channel Attn (CCA) 33.1M 4.25 1695.73 49.97 31.38 0.48 0.58 w/o CCA ‡ 33.0M 4.24 1731.13 54.78 28.40 0.48 0.57 CCA→SE module ‡ [40] 32.9M 4.25 1657.10 50.89 30.49 0.48 0.57 CCA→Channel Self-Attn ‡ [97] 33.2M 4.26 1569.51 50.24 30.85 0.45 0.59

Conv Module (CM) 33.1M 4.25 1695.73 49.97 31.38 0.48 0.58 CM→Self-Attn ‡ [83] 33.0M 4.23 1140.93 54.00 28.52 0.42 0.59 CM→Window Attn ‡ [56] 33.0M 4.34 1165.22 53.23 28.33 0.43 0.59 CM→Focused Linear Attn ‡ [29] 32.9M 4.33 971.49 50.60 30.85 0.46 0.60 CM→Agent Attn ‡ [65] 33.3M 4.24 1160.89 52.07 28.82 0.43 0.60

mance gains but at the expense of reduced efficiency, highlighting a trade-off between performance and computational cost.

The introduction of CCA results in a 4.81-point improvement in FID. As illustrated in Fig. 7, CCA significantly enhances feature diversity, demonstrating its effectiveness in improving the model’s representational capacity. We also compare CCA with SE module [40] and Channel-wise SelfAttention [97]; despite its simplicity, CCA achieves superior performance and higher efficiency.

For the Conv Module, we benchmark it against several advanced efficient attention mechanisms (Window Attention [56], Focused Linear Attention [29], Agent Attention [65]). The results show that our CM offers both better performance and computational efficiency.

### 5 Conclusion

We propose a new backbone for diffusion models, Diffusion ConvNet (DiCo), as a compelling alternative to the Diffusion Transformer (DiT). DiCo replaces self-attention with a combination of 1 × 1 pointwise convolutions and 3 × 3 depthwise convolutions, and incorporates a compact channel attention mechanism to reduce channel redundancy and enhance feature diversity. As a fully convolutional network, DiCo surpasses recent diffusion models on the ImageNet 256×256 and 512×512 benchmarks, while achieving significant efficiency gains. Furthermore, the purely convolutional DiCo demonstrates strong potential in text-to-image generation. We look forward to further scaling up DiCo and extending it to broader generative tasks.

### Acknowledgements

This research is partially funded by National Natural Science Foundation of China (Grant No. 62576342), Beijing Natural Science Foundation (4252054), Youth Innovation Promotion Association CAS(Grant No.2022132), Beijing Nova Program(20230484276).

### References

- [1] Yuang Ai, Huaibo Huang, and Ran He. Lora-ir: taming low-rank experts for efficient all-in-one image restoration. arXiv preprint arXiv:2410.15385, 2024. 5
- [2] Yuang Ai, Huaibo Huang, Tao Wu, Qihang Fan, and Ran He. Breaking complexity barriers: Highresolution image restoration with rank enhanced linear attention. arXiv preprint arXiv:2505.16157, 2025. 2
- [3] Yuang Ai, Huaibo Huang, Xiaoqiang Zhou, Jiexiang Wang, and Ran He. Multimodal prompt perceiver: Empower adaptiveness generalizability and fidelity for all-in-one image restoration. In CVPR, 2024. 2
- [4] Yuang Ai, Xiaoqiang Zhou, Huaibo Huang, Xiaotian Han, Zhengyu Chen, Quanzeng You, and Hongxia Yang. Dreamclear: High-capacity real-world image restoration with privacy-safe dataset curation. In NeurIPS, 2024. 2
- [5] Yuang Ai, Xiaoqiang Zhou, Huaibo Huang, Lei Zhang, and Ran He. Uncertainty-aware source-free adaptive image super-resolution with wavelet augmentation transformer. In CVPR, 2024. 4
- [6] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 2, 4, 5, 7, 8, 9, 15, 18, 19
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [8] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018. 18, 19
- [9] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. 2
- [10] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, 2023. 2
- [11] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022. 18, 19
- [12] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Zhongdao Wang, James T Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024. 2, 4, 6
- [13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 7
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS,

2021. 2, 4, 7, 8, 9, 18, 19

- [15] Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. Scaling up your kernels to 31x31: Revisiting large kernel design in cnns. In CVPR, 2022. 2, 5
- [16] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In CVPR, 2021. 4
- [17] Xiaohan Ding, Yiyuan Zhang, Yixiao Ge, Sijie Zhao, Lin Song, Xiangyu Yue, and Ying Shan. Unireplknet: A universal perception large-kernel convnet for audio video point cloud time-series and image recognition. In CVPR, 2024. 5
- [18] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2, 4
- [19] Junxian Duan, Yuang Ai, Jipeng Liu, Shenyuan Huang, Huaibo Huang, Jie Cao, and Ran He. Test-time forgery detection with spatial-frequency prompt learning. IJCV, 2025. 4
- [20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2, 4
- [21] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, 2021. 18, 19
- [22] Qihang Fan, Huaibo Huang, Yuang Ai, and Ran He. Rectifying magnitude neglect in linear attention. In ICCV, 2025. 2
- [23] Qihang Fan, Huaibo Huang, and Ran He. Breaking the low-rank dilemma of linear attention. In CVPR,

2025. 2

- [24] Wan-Cyuan Fan, Yen-Chun Chen, DongDong Chen, Yu Cheng, Lu Yuan, and Yu-Chiang Frank Wang. Frido: Feature pyramid diffusion for complex scene image synthesis. In AAAI, 2023. 9

- [25] Zhengcong Fei, Mingyuan Fan, Changqian Yu, and Junshi Huang. Scalable diffusion models with state space backbone. arXiv preprint arXiv:2402.05608, 2024. 2, 3, 4, 8, 9, 18, 19
- [26] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 2, 3, 4
- [27] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, 2022. 9
- [28] Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zhengning Liu, Ming-Ming Cheng, and Shi-Min Hu. Segnext: Rethinking convolutional attention design for semantic segmentation. In NeurIPS, 2022. 2
- [29] Dongchen Han, Xuran Pan, Yizeng Han, Shiji Song, and Gao Huang. Flatten transformer: Vision transformer using focused linear attention. In ICCV, 2023. 10
- [30] Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, and Gao Huang. Demystify mamba in vision: A linear attention perspective. In NeurIPS, 2024. 2
- [31] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 4
- [32] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 7
- [33] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2, 5
- [34] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. JMLR, 2022. 4
- [35] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598,

2022. 5

- [36] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022. 2
- [37] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In ICML, 2023. 18, 19
- [38] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for mobilenetv3. In ICCV, 2019. 4
- [39] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017. 4
- [40] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In CVPR, 2018. 10
- [41] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In CVPR, 2017. 4
- [42] Huaibo Huang, Xiaoqiang Zhou, Jie Cao, Ran He, and Tieniu Tan. Vision transformer with super token sampling. In CVPR, 2023. 2
- [43] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, 2023. 18
- [44] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In ICML, 2020. 3, 4
- [45] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In NeurIPS, 2022. 2
- [46] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 2
- [47] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. In NeurIPS, 2023. 18, 19
- [48] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. In NeurIPS, 2019. 7
- [49] Black Forest Labs. Flux: Official inference repository for flux.1 models, 2024. Accessed: 2024-11-12. 2, 4
- [50] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, 2022. 18
- [51] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation generation method. In NeurIPS, 2024. 18
- [52] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In NeurIPS, 2024. 18
- [53] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 8
- [54] Qihao Liu, Zhanpeng Zeng, Ju He, Qihang Yu, Xiaohui Shen, and Liang-Chieh Chen. Alleviating distortion in image generation via multi-resolution diffusion models and time-dependent layer normalization. In NeurIPS, 2024. 4
- [55] Songhua Liu, Weihao Yu, Zhenxiong Tan, and Xinchao Wang. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097, 2024. 2
- [56] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021. 10
- [57] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, 2022. 5

- [58] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In ECCV, 2024. 4, 18, 19, 20
- [59] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2
- [60] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021. 16
- [61] Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T Freeman, and Yu-Xiong Wang. Randar: Decoder-only autoregressive visual generation in random orders. In CVPR,

2025. 18

- [62] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2, 3, 4, 5, 7, 8, 9, 10, 18, 19
- [63] Hao Phung, Quan Dao, Trung Tuan Dao, Hoang Phan, Dimitris N. Metaxas, and Anh Tuan Tran. DiMSUM: Diffusion mamba - a scalable and unified spatial-frequency method for image generation. In NeurIPS, 2024. 2, 4, 8, 18
- [64] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2
- [65] Yifan Pu, Zhuofan Xia, Jiayi Guo, Dongchen Han, Qixiu Li, Duo Li, Yuhui Yuan, Ji Li, Yizeng Han, Shiji Song, et al. Efficient diffusion transformer with step-wise dynamic attention mediators. In ECCV, 2024. 4, 10
- [66] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 2
- [67] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 4, 7, 8
- [68] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 2, 4
- [69] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [70] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NeurIPS, 2016. 7
- [71] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In CVPR, 2018. 4
- [72] Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-xl: Scaling stylegan to large diverse datasets. In SIGGRAPH, 2022. 18, 19
- [73] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [74] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 2
- [75] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. 2
- [76] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 2, 5
- [77] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525,

2024. 18

- [78] Ming Tao, Hao Tang, Fei Wu, Xiao-Yuan Jing, Bing-Kun Bao, and Changsheng Xu. Df-gan: A simple and effective baseline for text-to-image synthesis. In CVPR, 2022. 9
- [79] Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Dim: Diffusion mamba for efficient high-resolution image synthesis. arXiv preprint arXiv:2405.14224, 2024. 2, 3, 4, 8, 9, 18, 19
- [80] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NeurIPS, 2024. 18, 19
- [81] Yuchuan Tian, Jing Han, Chengcheng Wang, Yuchen Liang, Chao Xu, and Hanting Chen. Dic: Rethinking conv3x3 designs in diffusion models. In CVPR, 2025. 3, 7, 8, 18
- [82] Yuchuan Tian, Zhijun Tu, Hanting Chen, Jie Hu, Chao Xu, and Yunhe Wang. U-dits: Downsample tokens in u-shaped diffusion transformers. In NeurIPS, 2024. 5
- [83] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 2, 3, 10
- [84] Hongbo Wang, Jie Cao, Jin Liu, Xiaoqiang Zhou, Huaibo Huang, and Ran He. Hallo3d: Multi-modal hallucination detection and mitigation for consistent 3d content generation. In NeurIPS, 2024. 2
- [85] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Flowdcn: Exploring dcn-like architectures for fast image generation with arbitrary resolution. In NeurIPS, 2024. 18, 20
- [86] Yuqing Wang, Shuhuai Ren, Zhijie Lin, Yujin Han, Haoyuan Guo, Zhenheng Yang, Difan Zou, Jiashi Feng, and Xihui Liu. Parallelized autoregressive visual generation. In CVPR, 2025. 18

- [87] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS, 2023. 2
- [88] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023. 2
- [89] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In CVPR, 2017. 4
- [90] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In CVPR,

2018. 9

- [91] Jing Nathan Yan, Jiatao Gu, and Alexander M Rush. Diffusion models without attention. In CVPR, 2024. 2, 4, 8, 9, 18, 19
- [92] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In ICML, 2024. 2, 3, 4
- [93] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In CVPR, 2025. 18, 20
- [94] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 18
- [95] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. In NeurIPS, 2024. 18
- [96] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025. 4, 8, 9, 20
- [97] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In CVPR, 2022. 5, 10
- [98] Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-modal contrastive learning for text-to-image generation. In CVPR, 2021. 9
- [99] Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun. Towards language-free training for text-to-image generation. In CVPR, 2022. 9
- [100] Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. In CVPR, 2025. 2, 3, 4, 5, 7, 8, 18
- [101] Minfeng Zhu, Pingbo Pan, Wei Chen, and Yi Yang. Dm-gan: Dynamic memory generative adversarial networks for text-to-image synthesis. In CVPR, 2019. 9
- [102] Zhuangwei Zhuang, Mingkui Tan, Bohan Zhuang, Jing Liu, Yong Guo, Qingyao Wu, Junzhou Huang, and Jinhui Zhu. Discrimination-aware channel pruning for deep neural networks. In NeurIPS, 2018. 4

### Appendix

We provide the following supplementary information in the Appendix:

- • Sec. A. detailed configurations of DiCo model variants.
- • Sec. B. additional implementation details of DiCo.
- • Sec. C. scalability analysis of DiCo models.
- • Sec. D. additional comparison results with generative model family.
- • Sec. E. additional samples generated by DiCo-XL models.
- • Sec. F. discussion of limitations of this work.
- • Sec. G. discussion of broader impacts of this work.

### A DiCo Model Variants

We introduce several variants of DiCo, each scaled to different model sizes. Specifically, we present five variants: Small (33M parameters), Base (130M), Large (460M), XLarge (700M), and Huge (1B). These variants are created by adjusting the hidden size D, the number of layers (L1,L2,L3,L4,L5), and the FFN ratio. They span a wide range of parameter counts and FLOPs, from 33M to 1B parameters and from 4.25 Gflops to 194.15 Gflops, offering a comprehensive foundation for evaluating scalability and efficiency. Notably, when compared to their corresponding DiT counterparts, our DiCo models require only 70.1% to 74.6% of the FLOPs, demonstrating the computational efficiency of our design. The detailed configurations for these variants are provided in Table 6.

- Table 6: Architecture variants of DiCo. Gflops are measured with an input size of 32 × 32 × 4. Compared to DiT, our DiCo models are more computationally efficient.

Model #Params (M) Gflops GflopsGflopsDiCo

DiT

Hidden Size D #Layers FFN Ratio

DiCo-S 33.1 4.25 70.1% 128 [5, 4, 4, 4, 4] 2 DiCo-B 130.0 16.88 73.3% 256 [5, 4, 4, 4, 4] 2 DiCo-L 463.9 60.24 74.6% 352 [9, 8, 9, 8, 9] 2 DiCo-XL 701.2 87.30 73.6% 416 [9, 9, 10, 9, 9] 2 DiCo-H 1037.4 194.15 - 416 [14, 12, 10, 12, 14] 4

- Table 7: Implementation details of DiCo variants. For DiCo-S/B/L/XL, we follow the same experimental settings as DiT [6]. For the largest variant DiCo-H, we adopt the training hyperparameters from U-ViT [6], increasing the batch size and learning rate to accelerate training.

Model DiCo-S DiCo-B DiCo-L DiCo-XL DiCo-XL DiCo-H Resolution 256 × 256 256 × 256 256 × 256 256 × 256 512 × 512 256 × 256

Optimizer AdamW AdamW AdamW AdamW AdamW AdamW Betas (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.99, 0.99) Weight decay 0 0 0 0 0 0 Peak learning rate 1 × 10−4 1 × 10−4 1 × 10−4 1 × 10−4 1 × 10−4 2 × 10−4 Learning rate schedule constant constant constant constant constant constant Warmup steps 0 0 0 0 0 5K Global batch size 256 256 256 256 256 1024 Numerical precision fp32 fp32 fp32 fp32 fp32 fp32 Training steps 400K 400K 400K 3750K 3000K 1000K Computational resources 8 A100 8 A100 16 A100 32 A100 64 A100 64 A100 Training Time 11 hours 16 hours 29 hours 266 hours 256 hours 113 hours Data Augmentation random flip random flip random flip random flip random flip random flip

VAE sd-ft-ema sd-ft-ema sd-ft-ema sd-ft-ema sd-ft-ema sd-ft-ema Sampler iDDPM iDDPM iDDPM iDDPM iDDPM iDDPM Sampling steps 250 250 250 250 250 250

100

DiCo-S DiCo-B DiCo-L DiCo-XL

DiCo-S DiCo-B DiCo-L DiCo-XL

100

80

InceptionScore

80

FID-50K

60

60

40

40

20

20

100K 200K 300K 400K Training Steps

100K 200K 300K 400K Training Steps

- Figure 8: Scaling the DiCo models consistently improves performance throughout training. We report FID-50K and Inception Score across training iterations for four DiCo variants.

0 100K 200K 300K 400K 500K 600K 700K 800K 900K 1M

Training Iterations

0.13

0.14

0.15

0.16

0.17

0.18

0.19

0.20

0.21

TrainingLoss

DiCo-S DiCo-B DiCo-L DiCo-H

10K 20K 30K 40K 50K 60K 70K 80K 90K 100K

0.14

0.15

0.16

0.17

0.18

0.19

0.20

0 0.25M 0.50M 0.75M 1.00M 1.25M 1.50M 1.75M 2.00M 2.25M 2.50M 2.75M 3.00M 3.25M 3.50M 3.75M

Training Iterations

0.13

0.14

0.15

0.16

0.17

0.18

0.19

0.20

0.21

TrainingLoss

DiCo-XL (256x256) DiCo-XL (512x512)

10K 20K 30K 40K 50K 60K 70K 80K 90K 100K

0.14

0.15

0.16

0.17

0.18

0.19

0.20

- Figure 9: Training loss curves for DiCo models. We also highlight early training dynamics during the first 100K iterations. Larger DiCo variants exhibit lower training losses, reflecting improved optimization with scale.

### B Additional Implementation Details

For DiCo-S/B/L/XL models, we adopt the same experimental settings as those used for DiT. For DiCo-H, the largest variant, we use the hyperparameters from U-ViT, increasing the batch size and learning rate to expedite training. All experiments are conducted on NVIDIA A100 (80G) GPUs. During inference, all models use the iDDPM [60] sampler with 250 sampling steps. The whole implementation details are summarized in Table 7.

### C Scalability Analysis

Impact of scaling on metrics. Table 8 and Fig. 8 illustrate the impact of DiCo model scaling across various evaluation metrics. Our results show that scaling DiCo consistently enhances performance across all metrics throughout training, highlighting its potential as a strong candidate for a large-scale foundational diffusion model.

- Table 8: Performance of DiCo models without CFG at different training steps on ImageNet 256×256. Scaling the ConvNet backbone consistently leads to improved generative performance.

Model Gflops Training Steps FID ↓ sFID ↓ IS ↑ Precision ↑ Recall ↑

50K 109.50 18.28 12.00 0.283 0.322 100K 79.79 15.60 16.39 0.382 0.440 150K 68.93 13.98 19.93 0.411 0.498 200K 62.61 13.18 22.87 0.433 0.531 250K 57.95 12.47 25.58 0.450 0.546 300K 54.58 11.99 27.74 0.467 0.565 350K 51.77 11.69 29.68 0.479 0.576 400K 49.97 11.41 31.38 0.481 0.582

DiCo-S 4.25

50K 84.78 14.24 14.85 0.392 0.423 100K 53.66 9.13 25.01 0.494 0.544 150K 43.09 8.21 33.02 0.536 0.571 200K 37.24 8.03 39.10 0.561 0.596 250K 33.31 7.72 44.89 0.577 0.598 300K 30.68 7.59 49.34 0.584 0.597 350K 28.72 7.55 52.95 0.594 0.599 400K 27.20 7.43 56.52 0.603 0.617

DiCo-B 16.88

50K 68.11 11.50 18.62 0.480 0.465 100K 35.42 6.52 38.12 0.607 0.561 150K 25.69 6.04 53.91 0.643 0.579 200K 20.81 5.80 65.78 0.665 0.589 250K 18.19 5.70 74.77 0.676 0.588 300K 16.11 5.60 81.47 0.685 0.594 350K 14.92 5.58 86.01 0.689 0.602 400K 13.66 5.50 91.37 0.694 0.604

DiCo-L 60.24

50K 66.53 11.35 19.41 0.501 0.472 100K 31.78 6.39 42.81 0.637 0.563 150K 22.61 5.94 60.49 0.672 0.572 200K 17.95 5.67 73.21 0.693 0.582 250K 15.60 5.58 82.96 0.697 0.591 300K 13.70 5.52 89.44 0.707 0.599 350K 12.59 5.48 95.49 0.710 0.605 400K 11.67 5.38 100.42 0.711 0.606

DiCo-XL 87.30

Impact of scaling on training loss. We further analyze the effect of model scale on training loss. As shown in Fig. 9, larger DiCo models consistently achieve lower training losses, indicating more effective optimization with increased scale.

### D Additional Comparison Results

ImageNet 256×256. Table 9 presents a comparison across generative model family on ImageNet 256×256. Among diffusion models, our DiCo-XL achieves a strong FID of 2.05 with only 87.3 Gflops. Our largest variant, DiCo-H, attains an FID of 1.90. When compared to other generative model types, DiCo also demonstrates competitive performance. Notably, DiCo-H, with just 1B parameters, outperforms VAR-d30—which has 2B parameters—in terms of FID.

ImageNet 512×512. Table 10 presents the results on ImageNet 512×512. Among diffusion models, our DiCo-XL achieves an FID of 2.53 with only 349.8 GFLOPs. Compared to other generative models, DiCo continues to demonstrate strong performance. Specifically, DiCo-XL, with only 701M parameters, outperforms VAR-d36-s, which has 2.3B parameters, achieving superior FID performance with significantly fewer parameters.

- Table 9: Comparison with generative model family on ImageNet 256×256. We report the performance of state-of-the-art generative models across different paradigms, including GAN-based, masked prediction (Mask.)-based, autoregressive (AR), visual-autoregressive (VAR), and diffusion (Diff.)-based models.

Type Model #Params Gflops FID ↓ IS ↑ Precision ↑ Recall ↑

GAN BigGAN-deep [8] 160M - 6.95 171.4 0.87 0.28 GAN GigaGAN [43] 569M - 3.45 225.5 0.84 0.61 GAN StyleGAN-XL [72] 166M - 2.30 265.1 0.78 0.53

Mask. MaskGIT [11] 227M - 6.18 182.1 0.80 0.51 Mask. RCG [51] 502M - 3.49 215.5 - Mask. TiTok-S-128 [95] 287M - 1.97 281.8 - -

AR VQ-GAN-re [21] 1.4B - 5.20 280.3 - AR ViTVQ-re [94] 1.7B - 3.04 227.4 - AR RQTran.-re [50] 3.8B - 3.80 323.7 - AR LLamaGen-3B [77] 3.1B - 2.18 263.3 0.81 0.58 AR MAR-H [52] 943M - 1.55 303.7 0.81 0.62 AR PAR-3B [86] 3.1B - 2.29 255.5 0.82 0.58 AR RandAR-XXL [61] 1.4B - 2.15 322.0 0.79 0.62

VAR VAR-d16 [80] 310M - 3.30 274.4 0.84 0.51 VAR VAR-d20 600M - 2.57 302.6 0.83 0.56 VAR VAR-d24 1.0B - 2.09 312.9 0.82 0.59 VAR VAR-d30 2.0B - 1.92 323.1 0.82 0.59

Diff. ADM-U [14] 608M 742 3.94 215.8 0.83 0.53 Diff. U-ViT-L/2 [6] 287M 77 3.40 219.9 0.83 0.52 Diff. U-ViT-H/2 [6] 501M 133.3 2.29 263.9 0.82 0.57 Diff. Simple Diffusion [37] 2.0B - 2.77 211.8 - Diff. VDM++ [47] 2.0B - 2.12 267.7 - Diff. DiT-XL/2 [62] 675M 118.7 2.27 278.2 0.83 0.57 Diff. SiT-XL [58] 675M 118.7 2.06 277.5 0.83 0.59 Diff. DiM-H [79] 860M 210 2.21 - - Diff. DiS-H/2 [25] 901M - 2.10 271.3 0.82 0.58 Diff. DiffuSSM-XL [91] 673M 280.3 2.28 259.1 0.86 0.56 Diff. DiMSUM-L/2 [63] 460M 84.49 2.11 - - 0.59 Diff. DiG-XL/2 [100] 676M 89.40 2.07 279.0 0.82 0.60 Diff. DiC-H [81] 1.0B 204.4 2.25 - - Diff. DiCo-XL 701M 87.30 2.05 282.2 0.83 0.59 Diff. DiCo-H 1.0B 194.15 1.90 284.3 0.83 0.61

ImageNet 1024×1024. We conduct experiments on ImageNet 1024×1024. Since the original DiT paper does not report results at this resolution, we train both DiT and DiCo from scratch under the same settings for a fair comparison. We use small-scale models and train them for 400K iterations. Table 11 clearly shows that DiCo scales much more efficiently to high resolutions—achieving a 5× speedup with better generation quality.

Comparison with FlowDCN. Deformable convolution is a strong and widely used convolutional variant, and FlowDCN [85] represents an important baseline in this space. To fairly compare with FlowDCN, we retain its original isotropic architecture and training setup, and replace its MultiScale DCN with our proposed Conv Module. Table 12 shows that our Conv Module achieves better generative performance and 2.1× higher throughput compared to MultiScale DCN, despite being structurally simpler and easier to implement.

Comparison with Modern DiTs. Given the rapid advancements in modern DiTs, it is important to compare our method with recent DiTs such as LightningDiT [93]. We find that after aligning with their experimental settings, our method achieves competitive performance along with a clear efficiency advantage. Table 13 shows that our model achieves an FID score of 1.33 and offers a 2.8×

- Table 10: Comparison with generative model family on ImageNet 512×512. We report the performance of state-of-the-art generative models across different paradigms, including GAN-based, masked prediction (Mask.)-based, autoregressive (AR), visual-autoregressive (VAR), and diffusion (Diff.)-based models.

Type Model #Params Gflops FID ↓ IS ↑ Precision ↑ Recall ↑

GAN BigGAN-deep [8] 160M - 8.43 177.9 0.88 0.29 GAN StyleGAN-XL [72] 166M - 2.41 267.8 0.77 0.52

Mask. MaskGIT [11] 227M - 7.32 156.0 0.78 0.50

AR VQ-GAN [21] 227M - 26.52 66.8 0.73 0.31 VAR VAR-d36-s [80] 2.3B - 2.63 303.2 - -

Diff. ADM-U [14] 731M 2813 3.85 221.7 0.84 0.53 Diff. U-ViT-L/4 [6] 287M 76.5 4.67 213.3 0.87 0.45 Diff. U-ViT-H/4 [6] 501M 133.3 4.05 263.8 0.84 0.48 Diff. Simple Diffusion [37] 2.0B - 4.53 205.3 - Diff. VDM++ [47] 2.0B - 2.65 278.1 - Diff. DiT-XL/2 [62] 675M 524.7 3.04 240.8 0.84 0.54 Diff. SiT-XL [58] 675M 524.7 2.62 252.2 0.84 0.57 Diff. DiM-H [79] 860M 708 3.78 - - Diff. DiS-H/2 [25] 901M - 2.88 272.3 0.84 0.56 Diff. DiffuSSM-XL [91] 673M 1066.2 3.41 255.1 0.85 0.49 Diff. DiCo-XL 701M 349.8 2.53 275.7 0.83 0.56

- Table 11: Comparison with DiT on ImageNet 1024×1024. The performance is reported at 400K iterations without CFG.

Model #Params FLOPs Throughput ↑ FID ↓ IS ↑ Precision ↑ Recall ↑

DiT-S/2 33M 241.8G 28 it/s 113.06 12.60 0.25 0.37 DiCo-S 33M 67.8G 143 it/s 102.60 16.45 0.33 0.38

throughput gain over modern DiTs. This demonstrates that our convolutional backbone can match or exceed the generative quality of modern DiTs while being significantly more efficient.

### E Model Samples

We present samples generated by the DiCo-XL models at resolutions of 256×256 and 512×512. Fig. 10 through 29 display uncurated samples across various classifier-free guidance scales and input class labels. As illustrated, our DiCo models demonstrate the ability to generate high-quality, detail-rich images.

### F Limitations

While this work demonstrates the strong performance and scalability of DiCo through extensive experiments, there are some limitations to note. Due to limited computational resources, our model is scaled to 1B parameters, while some advanced generative models have been scaled to even larger sizes. We aim to investigate the broader generative potential of DiCo in future research.

### G Broader Impacts

Our work on the generative model DiCo contributes to advances in controllable image generation. This has potential positive applications in data augmentation, scientific visualization, and accessibility technologies. However, such models may also be misused to generate misleading or harmful content,

- Table 12: Comparison with FlowDCN on ImageNet 256×256. The performance is reported at 400K iterations without CFG.

Model #Params FLOPs Throughput ↑ FID ↓ IS ↑ SiT-S/2 [58] 32.9M 6.1G 1234 it/s 57.6 24.8 FlowDCN-S/2 [85] 30.3M 4.4G 1194 it/s 54.6 26.4 Ours-S/2 31.3M 4.6G 2489 it/s 52.1 29.2

- Table 13: Comparison with modern DiTs on ImageNet 256×256. We follow the setup in [93].

Model Epochs #Params FLOPs Throughput ↑ FID ↓ IS ↑

REPA [96] 800 675M 118.7G 77 it/s 1.42 305.7 LightningDiT [93] 800 675M 118.8G 73 it/s 1.35 295.3 Ours 800 679M 118.3G 201 it/s 1.33 300.2

especially in contexts involving deepfakes or biased representations of specific classes. To mitigate these risks, we encourage responsible usage aligned with ethical AI guidelines and emphasize the importance of transparency when deploying generative models in real-world applications.

[Figure 138]

- Figure 10: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “arctic wolf” (270)

[Figure 139]

Figure 11: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “volcano” (980)

[Figure 140]

Figure 12: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “husky” (250)

[Figure 141]

Figure 13: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “ulphur-crested cockatoo” (89)

[Figure 142]

#### Figure 14: Uncurated 512×512 DiCo-XL samples.

Classifier-free guidance scale = 4.0 Class label = “cliff drop-off” (972)

[Figure 143]

Figure 15: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “balloon” (417)

[Figure 144]

Figure 16: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “lion” (291)

[Figure 145]

Figure 17: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “otter” (360)

[Figure 146]

Figure 18: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 2.0 Class label = “red panda” (387)

[Figure 147]

Figure 19: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 2.0 Class label = “panda” (388)

[Figure 148]

#### Figure 20: Uncurated 512×512 DiCo-XL samples.

- Classifier-free guidance scale = 1.5 Class label = “coral reef” (973)

[Figure 149]

Figure 21: Uncurated 512×512 DiCo-XL samples. Classifier-free guidance scale = 1.5 Class label = “macaw” (88)

[Figure 150]

Figure 22: Uncurated 256×256 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “macaw” (88)

[Figure 151]

Figure 23: Uncurated 256×256 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “dog sled” (537)

[Figure 152]

Figure 24: Uncurated 256×256 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “arctic fox” (279)

[Figure 153]

Figure 25: Uncurated 256×256 DiCo-XL samples. Classifier-free guidance scale = 4.0 Class label = “loggerhead sea turtle” (33)

[Figure 154]

#### Figure 26: Uncurated 256×256 DiCo-XL samples.

- Classifier-free guidance scale = 2.0 Class label = “golden retriever” (207)

[Figure 155]

Figure 27: Uncurated 256×256 DiCo-XL samples. Classifier-free guidance scale = 2.0 Class label = “lake shore” (975)

[Figure 156]

Figure 28: Uncurated 256×256 DiCo-XL samples. Classifier-free guidance scale = 1.5 Class label = “space shuttle” (812)

[Figure 157]

Figure 29: Uncurated 256×256 DiCo-XL samples. Classifier-free guidance scale = 1.5 Class label = “ice cream” (928)

