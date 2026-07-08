# arXiv:2601.05966v2[cs.CV]14Jan2026

## VideoAR: Autoregressive Video Generation via Next-Frame & Scale Prediction

Longbin Ji∗ Xiaoxiong Liu∗ Junyuan Shang† Shuohuan Wang Yu Sun Hua Wu Haifeng Wang ERNIE Team, Baidu

{jilongbin, liuxiaoxiong, shangjunyuan}@baidu.com {wangshuohuan, sunyu02, wu hua, wanghaifeng}@baidu.com ∗Equal contribution †Project lead https://ernie-research.github.io/VideoAR/

[Figure 1]

Figure 1. VideoAR generates high-fidelity and temporally consistent videos from text prompts.

### Abstract

Recent advances in video generation have been dominated by diffusion and flow-matching models, which produce high-quality results but remain computationally intensive and difficult to scale. In this work, we introduce VideoAR, the first large-scale Visual Autoregressive (VAR) framework for video generation that combines multiscale next-frame prediction with autoregressive modeling. VideoAR disentangles spatial and temporal dependencies by integrating intra-frame VAR modeling with causal nextframe prediction, supported by a 3D multi-scale tokenizer that efficiently encodes spatio-temporal dynamics. To improve long-term consistency, we propose Multi-scale Temporal RoPE, Cross-Frame Error Correction, and Random Frame Mask, which collectively mitigate error propagation and stabilize temporal coherence. Our multi-stage pre-

training pipeline progressively aligns spatial and temporal learning across increasing resolutions and durations. Empirically, VideoAR achieves new state-of-the-art results among autoregressive models, improving FVD on UCF101 from 99.5 to 88.6 while reducing inference steps by over 10×, and reaching a VBench score of 81.74—competitive with diffusion-based models an order of magnitude larger. These results demonstrate that VideoAR narrows the performance gap between autoregressive and diffusion paradigms, offering a scalable, efficient, and temporally consistent foundation for future video generation research.

### 1. Introduction

Video generation has achieved remarkable progress in recent years, enabling high-quality synthesis that captures

both spatial structures and temporal dynamics. Most commercial and open-source systems are built upon diffusion and flow-matching frameworks [11], which typically leverage well pre-trained image diffusion models and are further adapted for temporal consistency. However, large-scale video diffusion models are computationally expensive and difficult to scale, as they rely on bi-directional denoising of entire temporal sequences simultaneously.

In contrast, autoregressive (AR) models have gained traction in image generation due to their scalability, the ease of integrating with optimized LLM infrastructures, and their unified modeling of diverse modalities. A critical step in adapting LLMs for vision is to design effective visual tokenizers that discretize visual inputs. Early pixel-based or VQ-VAE[25] tokenization approaches adopted the rasterscan next-token prediction paradigm, but they proved inefficient for complex visual signals. More recently, Visual AutoRegressive (VAR) [21] modeling reformulated autoregression as a coarse-to-fine “next-scale prediction,” achieving superior generalization and scaling compared to diffusion models while requiring fewer inference steps.

Despite this progress, autoregressive (AR) approaches for video generation remain underexplored and face several fundamental challenges when applied to high-dimensional spatio-temporal data: 1) Mismatch between spatial and temporal modeling - naive next-token prediction [1, 28] modeling is poorly aligned with the intrinsic structure of video data, as causal temporal progression and 2D spatial synthesis follow different modeling principles [32]. 2) Error propagation - autoregressive over long video token sequences leads to severe error propagation [30], resulting in substantial quality degradation. 3) Limited temporal–spatial controllability - existing temporal–spatial sampling strategies lack fine-grained controllability regarding video dynamics and duration.

To address these challenges, we propose VideoAR, the first VAR-based framework for large-scale video generation pretraining. VideoAR disentangles spatial and temporal modeling by leveraging the intra-frame modeling strength of VAR while adopting a next-frame prediction paradigm for inter-frame dependencies. For tokenization, we extend the 2D encoder–decoder of VAR into a 3D architecture that captures temporal dynamics, and initialize it with pre-trained 2D-VAR weights to efficiently transfer spatial knowledge. The Transformer backbone autoregressively predicts frames conditioned on text prompts and preceding frames, with each frame represented by multi-scale tokens produced by our 3D-VAR encoder. To better capture spatio-temporal relations, we introduce Multi-scale Temporal RoPE, which enhances temporal awareness and improves bit-level prediction accuracy. VideoAR further adopt two training strategies to address the error propagation problem: (1) Cross-Frame Error Correction where

the flip ratio is progressively increased across frames and inherated between cross-frame transitions, and (2) Random Frame Mask, which weakens excessive reliance on previous frames and alleviates over-memorization. Finally, we incorporate temporally-adjustable classifier guidance to flexibly control video dynamics, and employ frame reencoding to enable duration extension.

VideoAR establishes a new state-of-the-art among autoregressive models on video generation tasks. Notably, VideoAR-L achieves a gFVD score of 90.3 on the widely used UCF-101 benchmark, representing a substantial improvement over the previous best autoregressive model, PAR-4×, which reported 99.5 gFVD. This significant reduction in gFVD highlights the strong generative capability of our base model.

Furthermore, we train a large-scale variant on real-world video data, which achieves comparable performance to leading diffusion-based methods such as CogVideo [8] and Step-Video [13] on VBench [9]. VideoAR can generate 4second videos at a resolution of 384×672 with high fidelity and strong temporal consistency. In addition, we observe clear scaling behavior—enlarging the transformer backbone consistently improves the quality of generated videos.

In summary, our main contributions are threefold:

- • We introduce VideoAR, the first video generation framework that integrates the VAR paradigm with next-frame prediction, enabling multi-scale and temporally consistent synthesis.
- • We propose Multi-scale Temporal RoPE and two effective training strategies—Cross-Frame Error Correction and Random Frame Mask —to enhance spatio-temporal modeling. These techniques are particularly effective in long-form generation, mitigating frame drift and context collapse.
- • Our efficiently pre-trained video tokenizer and transformer backbone achieve state-of-the-art results on standard generation benchmarks, while significantly outperforming prior methods in inference speed and computational efficiency.

With these designs, VideoAR not only narrows the gap between autoregressive and diffusion-based models, but also establishes a foundation for future large-scale video generation research.

### 2. Related Work

##### 2.1. Diffusion-based Video Generation

Recent video generation models such as Veo3 [6], Sora [14], and Wanx [26] have achieved remarkably realistic visual quality and strong temporal consistency by applying large-scale latent diffusion models [15]. These models progressively synthesize visual content through iterative noise injection and denoising in the latent space. In paral-

lel, several works [2, 4] explore AR-Diffusion, which integrates autoregressive modeling with diffusion processes by introducing a non-decreasing corruption schedule and temporally causal attention, aligning training and inference dynamics for stable image-to-image translation and longterm video generation. While diffusion-based approaches offer superior fidelity and temporal smoothness, they remain computationally expensive during inference and lack flexibility in controlling generation length or ensuring finegrained temporal coherence.

##### 2.2. Autoregressive Visual Generation

Inspired by the success of AR modeling in natural language processing, recent studies have extended this paradigm to visual domains. In contrast to diffusion models, autoregressive (AR) generation predicts visual elements sequentially, making it naturally well-suited for structured sequence modeling tasks such as visual-token and temporal framewise generation. LlamaGen [19] and MAGVIT-v2 [34] adopt a next-token prediction framework over visual tokens obtained from VQ-VAE-style tokenizers, which quantize latent representations into discrete symbols. These methods achieve comparable visual quality to diffusion models while being significantly more efficient at inference time. PAR [31] further parallelizes the next-token prediction process to improve inference efficiency, and Loong [30] pioneers AR-based video generation by introducing temporalbalanced losses and a multi-stage training strategy. However, token-based AR methods still suffer from low spatial resolution due to excessive token lengths, and their generation quality often degrades because of error accumulation across spatial-temporal dimensions and weak modeling of spatial correlations among flattened pixel tokens. To address these limitations, we propose VideoAR, which combines temporal causal modeling with inter-frame multiscale spatial generation and cross-frame self-correction, enabling efficient and coherent autoregressive video synthesis.

### 3. Preliminary

Visual autoregressive (VAR) model generally consist of a multi-scale visual tokenizer and a Transformer-based generator. An image I ∈ RH×W×3 is tokenized with an encoder E into a feature map F ∈ RH×W×d = E(I), where H,W,d are the height, width, and channels. A quantizer Q then decomposes F into K multi-scale residual maps R1:K = Q(F) with gradually increasing resolution, where each residual Rk ∈ RH

k×Wk×|V | and |V | is the vocabulary size. The Transformer autoregressively predicts residuals at the next scale conditioned on all previous scales and the text prompt Ψ:

p(Rk|R1:k−1,Ψ) (1) Concretely, to predict Rk, the model take the feature representaion Fk−1 as the input by aggregating all previously

generated residuals R1:k−1, upsamples them to the base resolution (H,W), and then downsamples to match the target resolution (Hk,Wk):

k−1

F˜k−1 = down(

i=1

up(Ri,(H,W)),(Hk,Wk)) (2)

where up and down denotes bilinear up- and down-sampling respectively.

During training, to mitigate train–test discrepancies and improve robustness to prediction errors, the transformer takes as input the partially corrupted version of Fflipk . Specifically, with probability pflip, some labels in Rk are randomly flipped:

Rflipk = pflip(Rk) (3)

This enables the model to account for potential prediction errors and better generalize to autoregressive inference.

### 4. Method

In this section, we present our proposed framework VideoAR, which integrates the strengths of visual autoregressive (VAR) modeling with next-frame prediction for efficient and high-quality video generation. The pipeline is composed of two main components. First, we introduce a 3D video tokenizer in Sec. 4.1 that compresses raw video into compact discrete representations while preserving both spatial and temporal structures. This tokenizer serves as the foundation for scalable and efficient modeling. Second, we design an autoregressive video model in Sec. 4.2 built upon multi-scale residual prediction, where temporal consistency is further enhanced by our proposed training strategies.

##### 4.1. Visual Tokenizer

3D Architecture. To better capture spatial–temporal correlations, we adopt a causal 3D convolutional architecture [33], which allows the tokenizer to process both images and videos within a unified framework. Concretely, the 3D convolutional encoder with temporal subsampling compresses the input video V ∈ R(1+T)×H×W×3 into a compact spatio-temporal latent representation F ∈ R(1+T/τ)×H

′×W′×d, where τ denotes the temporal compression factor. This design leverages the inherent redundancy across adjacent frames, enabling efficient video modeling while maintaining fidelity.

To further scale to long-form video generation, we eliminate all non-causal temporal operations (e.g., temporal normalization) from both the encoder and decoder, ensuring that each latent feature only depends on past frames. This causal design enables inference on extremely long videos in a chunk-by-chunk manner, without performance loss compared to full-sequence inference.

Text Tokens

| |
|---|
| |
| |
| |
| |

|(0,0,0)|
|---|
|(1,1,1)|
|(2,2,2)|
|(3,3,3)|
|(4,4,4)|

(

|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Block-wise Causal Mask

Text Prompt "Generate a video of a man applying eyeliner"

|0|
|---|
|1|
|2|
|3|
|4|
|5|
|6|
|7|
|8|
|9|
|10|
|11|
|12|
|13|
|14|
|15|

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | |R|an|do|m| | | | | | |
| | | | | | | |Fra|m|e| | | | | | |
| | | | | | | |M|as|k| | | | | | |
| | | | | | | | | | | | | | | | |

(

Text Tokenizer

( ( ( ( ( ( ( ( ( ( ( ( (

#### TransformerLayers(causal)

###### Scale Ids

|SOS|
|---|

|(5,5,5)|
|---|

|1|
|---|

F

###### Video Frames

Video Tokenizer

R

[Figure 2]

[Figure 3]

[Figure 4]

|(5,5,5)|
|---|
|(5,5,6)|
|(5,6,5)|
|(5,6,6)|

|2|
|---|
|2|
|2|
|2|

F

[Figure 5]

| |
|---|
| |
| |
| |

R

[Figure 6]

R

Multi-Scale Video Encoder

R

Frame 1

Spatio: f16 Temporal:f4

Multi-Scale Residual Labels

| |
|---|

F

|(6,5,5)|
|---|

|1|
|---|

T*H*W=(5, 32, 32)

R

R

| |
|---|

Multi-Scale Prediction

|(6,5,5)|
|---|
|(6,5,6)|
|(6,6,5)|
|(6,6,6)|

|2|
|---|
|2|
|2|
|2|

F

R

| |
|---|
| |
| |
| |

Multi-Scale Video Decoder

R

Accumulated Feature Input

Frame 2~5 Video Tokens

|1|
|---|

Text Tokens

Scale Ids

Inference

(𝑡,ℎ,𝑤) Multi-Scale Temporal RoPE

Video Features

Figure 2. Overall framework of VideoAR. Given a text prompt, the video frames are first compressed into a sequence of spatio-temporal tokens via a multi-scale causal 3D tokenizer. Each frame is represented by residual maps at multiple scales, which are autoregressively predicted by a Transformer with block-wise causal masking. The input embeddings combine text tokens, accumulated video features, and scale embeddings, while the proposed Multi-Scale Temporal RoPE encodes temporal, spatial, and scale-aware positional information. Random frame masking is applied during training to mitigate exposure bias and improve long-term consistency. Finally, the multi-scale video decoder reconstructs the video frames from the predicted residuals.

Quanization Considering our temporal-causal modeling, we leverage temporal-independent quantization where each frame is passed through isolated the multi-scale quantizer.

toregressive video generation.

##### 4.2. Autoregressive Video Modeling

Training. To enable efficient and stable training of our video tokenizer, we adopt a 3D inflation strategy by initializing the model from a well-trained image tokenizer [7]. This initialization provides a strong spatial prior, substantially stabilizing optimization and accelerating convergence. Concretely, following the inflation procedure in [34], we populate the temporally last slice of the 3D CNN using the weights from the image tokenizer, while the remaining temporal parameters and the discriminator are randomly initialized.

Extension to 3D Architecture. Building upon the spatiotemporal features extracted by our 3D tokenizer, we extend the visual autoregressive (VAR) paradigm from images to videos. Specifically, the Transformer autoregressively predicts the residuals of the t-th frame conditioned on all previously generated frames, the coarser scales of the current frame, and the text prompt:

p(Rtk|R1:1:tK−1,Rt1:k−1,Ψ) (5)

where R1:1:tK−1 denotes the multi-scale residual maps of all past frames, and Rt1:k−1 denotes the residuals of the already generated coarser scales of the t-th frame. The input feature for the t-th frame at scale k is constructed as:

The tokenizer is trained with a standard combination of complementary objectives. We apply the reconstruction, perceptual, commitment losses on each frame. Following [33], we use LeCAM regularization [23] for improved stability and entropy penalty to encourage codebook utilization.

k−1

F˜tk−1 = down(

up(Rti,(H,W)),(Hk,Wk)) (6)

The overall training objective can be formulated as:

i=1

where up(·) and down(·) denote spatial up- and downsampling.

L = λrecLrec + λpercLperc + λGANLGAN + λcommitLcommit + λentropyLentropy,

(4)

To initialize generation, the feature of the first scale in

the first frame (F˜10 in Fig. 2) is set to a special < SOS > token embedding, enabling text-conditioned generation.

where λ’s are balancing weights for different objectives.

This training scheme ensures that the tokenizer learns compact yet expressive spatio-temporal representations, benefiting both reconstruction fidelity and downstream au-

For subsequent frames (t > 1), the first-scale feature (F˜t0) is initialized from the accumulated features of the previous

frame, injecting temporal context into the next-frame generation.

the first scale of the subsequent frame, we propose a crossframe error inheritance mechanism. Specifically, the flip ratio of each frame’s first scale is initialized within a range above the final scale’s flip ratio of the preceding frame. By compelling the model to correct these inherited perturbations at the very first scale, our training procedure enhances temporal robustness and substantially mitigates the influence of preceding-frame errors on subsequent generations.

Multi-scale Temporal RoPE. To better capture spatiotemporal dependencies, we introduce Multi-scale Temporal RoPE, an extension of Rotary Position Embeddings (RoPE) [18] by factorizing the embedding space into three axes—time, height, and width. The design principles of Multi-scale Temporal RoPE are threefold: (1) compatibility with the native RoPE formulation for text tokens, (2) explicit temporal awareness, and (3) spatial consistency across frames with multi-scale inputs.

pflip(t) ∼ Uniform(pmin + δt,pmax + δt) pflip(tinit−f) ∼ Uniform(pprev−f,pmax + δt)

Given a multimodal input consisting of a text prompt Ψ and video tokens, we assign the same temporal, height, and width indices to text tokens to maintain compatibility with RoPE. Let xtk(h,w) denote the token at scale k of the t-th frame at spatial location (h,w), where h < Hk and w < Wk. The positional encoding is defined as:

˜bt,kh,w,j = bt,kh,w,j ⊕ ξh,w,jt,k , ξh,w,jt,k ∼ Bernoulli pflip(t) ,

(8)

s where ⊕ denotes XOR, δ denotes the factor for increasing the flipping range. The model conditions on corrupted history and is supervised by self-corrected targets with requantized errors [7], improving robustness to compounding mistakes.

Position: (t,h,w) = (t + |Ψ|, h + |Ψ|, w + |Ψ|), Embedding: xtk(h,w) = F˜tk(h,w) + sk,

Random Frame Mask Let the attention window size be w. For each step t, we form a stochastic causal context St = {t′ |t′ ∈ [max(1,t−w), t−1], mt′ = 1} with i.i.d. mt′ ∼ Bernoulli(1 − pmask). Denote text keys/values by (Ktext,Vtext) and video keys/values from frames in St by (KS

(7)

where the spatial indices (h,w) are consistent across frames, while the temporal index increases with t to preserve ordering. Additionally, a learnable scale embedding sk is added to differentiate coarse-to-fine scales during autoregressive generation.

). The attention output for frame t is

###### ,VS

t

t

]⊤ √

###### Qt [Ktext, KS

t

Ot = Softmax

###### [Vtext, VS

], (9)

t

Temporal-Consistency Enhancement. Autoregressive video generation suffers from error accumulation: quality degrades as t grows due to train–test discrepancy. We adopt two complementary strategies to mitigate this: CrossFrame Error Correction with a time-ramped schedule, and Random Frame Mask with a causal sliding window.

d

which discourages over-reliance on distant frames while preserving necessary temporal context.

Multi-Stage Training Pipeline. Following Infinity [7], our training objective is defined as a bit-wise cross-entropy loss between the predicted residual maps Rˆ 1:1:TK and the ground truth R1:1:TK. To achieve robust temporal consistency and high-quality synthesis in long-form, high-resolution videos, we adopt a progressive multi-stage training strategy. In Stage I, we jointly pretrain on large-scale image and low-resolution video datasets, enabling the model to acquire fundamental spatial-temporal representations while benefiting from efficient convergence. In Stage II, we continue training on higher-resolution image and video data to enhance fine-grained visual fidelity and temporal coherence. Finally, in Stage III, we perform long-form video fine-tuning using only high-resolution video datasets, allowing the model to capture extended motion dynamics and long-range temporal dependencies. This hierarchical training scheme effectively balances training stability, scalability, and generation quality across diverse video domains.

frame1 frame2

𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠

Time-dependent Corruption

###### flip 0 ~ 0.25 flip 0 ~ 0.28

𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠 𝑠

f = 0.2 f = 0.2 ~ 0.28

Cross-Frame Error Inheritance

Figure 3. Our proposed Cross-Frame Error Correction.

Cross-Frame Error Correction. Following the bitwise formulation in Infinity, we represent each token in Rtk by d bits bt,kh,w,j ∈ {0,1}, j = 1,...,d.

To account for the accumulation of error propagation along extended frame sequences, we introduce time-dependent corruption by injecting perturbations with progressively increasing flip ratios, thereby simulating inference-time situation, see Fig. 3. Furthermore, since errors at the final scale of each frame inevitably propagate into

[Figure 7]

Figure 4. Generation results from our VideoAR-4B on VBench and UCF-101 datasets. Zoom in for clearer visualization.

Temporal-Spatial Adaptive Classifier-free Guidance. At test time, we perform causal decoding over (t,k) with cached states to ensure efficiency. To balance semantic fidelity and temporal consistency, we introduce a temporalspatial adaptive classifier-free guidance (CFG) applied to the logits to enable flexible control of text alignment and temporal dynamics under different model settings.

Empirically, we observe that larger guidance coefficients lead to improved visual quality and stronger dynamics across frames, whereas smaller coefficients yield more stable temporal transitions and greater sampling diversity. Therefore, we adapts spatial-CFG along scales and also set the temporal starting point of CFG for first scale in a preselected scheduler.

### 5. Experiments

##### 5.1. Experimental Setup

Datasets. We conduct experiments on a diverse set of benchmarks encompassing both low-resolution toy datasets

and high-resolution real-world long-form video generation. For short videos generation, we use UCF-101 [17] (8K clips, 101 action categories) as a standard benchmark for human-action modeling. For long-form and open-domain scenarios, we conduct large-scale pretraining and evaluation on proprietary in-house datasets. All videos are uniformly resized to pixelsqrt ∈ (128,256,512) and temporally sampled to T ∈ [17,65] frames depending on the dataset.

Evaluation metrics. We evaluate our model along two axes: reconstruction quality and generation quality. For reconstruction, we report the Fr´echet Video Distance (rFVD) [24], which directly reflects the fidelity of the learned video tokenizer. For generation quality, we measure gFVD on the held-out human-centric test set of UCF-101. Moreover, to assess real-world generation performance, we evaluate on the standard VBench [9], which provides a comprehensive suite of perceptual and temporal metrics specifically designed for video generation models.

[Figure 8]

Figure 5. Visualization for VideoAR’s Image-to-Video and Video-to-Video generation performance. I2V refers to Image-to-Video, purple boxes refers to the given image. V2V shotN refers to N times video-to-video Extension of 4 seconds window.

##### 5.2. Experimental results

Video Reconstruction. The efficacy of an autoregressive video generation model is largely depends on the quality and compactness of its underlying video tokenizer. We

assess this aspect by reporting the reconstruction Fr´echet Video Distance (rFVD). Tab. 2 presents a comparative analysis on the UCF-101 dataset, demonstrating our model’s superior trade-off between compression efficiency and reconstruction fidelity.

- Table 1. Performance comparison on VBench. The best results are in bold, and the second-best are underlined. Our 4B model achieves a competitive overall score and sets a new state-of-the-art on the Semantic Score, Aesthetic Qualiy, Object Class and Multiple Objects, surpassing models with substantially larger parameters.

Methods #Size Total Quality

Score

Semantic Score

Subject Consis

Background Consis

Temp Flicker

Motion Smooth

Dynamic Degree

Aesthetic Quality

Image Quality

Object Class

Multiple Objects

Human Action

Color

Spatial Relation

Scene

Appearance Style

Temporal Style

Overall Consis

LaVie[29] 3B 77.08 78.78 70.31 91.41 97.47 98.30 96.38 49.72 54.94 61.90 91.82 33.32 96.80 86.39 34.09 52.69 23.56 25.93 26.41 VideoCrafter-2.0[3] 1B 80.44 82.20 73.42 96.85 98.22 98.41 97.73 42.50 63.13 67.22 92.55 40.66 95.00 92.92 35.86 55.29 25.13 25.84 28.23 CogVideoX[8] 5B 81.61 82.75 77.04 96.23 96.52 98.66 96.92 70.97 61.98 62.90 85.23 62.11 99.40 82.81 66.35 53.20 24.91 25.38 27.59 Kling[20] - 81.85 83.39 75.68 98.33 97.60 99.30 99.40 46.94 61.21 65.62 87.24 68.05 93.40 89.90 73.03 50.86 19.62 24.17 26.42 Step-Video-T2V[13] 30B 81.83 84.46 71.28 98.05 97.67 99.44 99.08 53.06 61.23 70.63 80.56 50.55 94.00 88.25 71.47 24.38 23.17 26.01 27.12 Gen-3[16] - 82.32 84.11 75.17 97.10 96.62 98.61 99.23 60.14 63.34 66.82 87.81 53.64 96.40 80.90 65.09 54.57 24.31 24.71 26.69 Hunyuan-Video[10] 13B 83.24 85.09 75.82 97.37 97.76 99.44 98.99 70.83 60.36 67.56 86.10 68.55 94.40 91.60 68.68 53.88 19.80 23.89 26.44

VideoAR (Ours) 4B 81.74 82.88 77.15 95.51 97.85 98.83 98.37 61.39 63.42 60.71 94.98 72.88 94.40 90.73 56.14 50.52 22.68 25.11 27.44

- Table 2. Reconstruction performance of our video tokenizer and other methods on the UCF-101 dataset.

Methods Tokens (T×H×W) Ratio rFVD ↓ TATS [5] 4 × 16 × 16 8 162 MAGVIT [33] 5 × 16 × 16 8 58 OmniTokenizer [27] 5 × 16 × 16 8 42 VideoAR-L (Ours) 5 × 8 × 8 16 61

- Table 3. Video generation results: class-conditional generation on UCF-101 dataset. VideoAR achieves the best gFVD score with significantly fewer steps and lower latency.

however, lies in inference speed: with only 30 decoding steps—a more than 10× reduction—VideoAR-L generates a video in just 0.86 seconds, achieving over 13× faster inference compared to PAR-4x.

This dual advancement stems directly from our architectural innovations. High-fidelity spatial details are preserved through intra-frame visual autoregression, while robust temporal consistency.

Real-World Video Generation. To further validate the effectiveness and scalability of our approach, we pre-train a 4B-parameter VideoAR model on the challenging task of real-world video generation. As presented in Tab. 1, our model attains an overall VBench score of 81.74, achieving performance comparable to, or even surpassing, current state-of-the-art models that are significantly larger in scale, such as Step-Video-T2V (30B) and Hunyuan-Video (13B).

Methods #Params gFVD ↓ Steps ↓ Time (s) ↓ CogVideo [8] 9.4B 626 - TATS [5] 312M 332 - OmniTokenizer [27] 650M 191 5120 336.70 MAGVIT-v2-AR [34] 840M 109 1280 PAR-4x [31] 792M 99.5 323 11.27

VideoAR-L (Ours) 926M 90.3 30 0.86 VideoAR-XL (Ours) 2.0B 88.6 30 1.12

Our model’s primary strengths are revealed through a fine-grained analysis of the VBench metrics. In particular, VideoAR achieves a new state-of-the-art Semantic Score (SS) of 77.15, surpassing all competitors. This result highlights its remarkable ability to maintain precise text-tovideo alignment. While maintaining competitive results on general visual quality metrics such as Aesthetic Quality (AQ) and Overall Consistency (OC), these superior performances in semantics and motion clearly showcase the distinctive strengths of our model.

Our VideoAR-L tokenizer employs an aggressive 16× spatial compression, encoding video clips into a compact 5 × 8 × 8 latent token grid. This design yields a 4× reduction in sequence length compared to recent state-of-theart video tokenizers such as MAGVIT [33] and OmniTokenizer [27], both of which operate at only 8× compression ratio. Despite the substantially lower token density, our tokenizer maintains excellent reconstruction quality, achieving an rFVD score of 61—on par with MAGVIT (58). This result highlights the effectiveness of our tokenizer in retaining fine-grained spatial and temporal structure, establishing a strong and efficient representation for downstream autoregressive video generation.

Qualitative results (Fig. 4 and supplementary material) further corroborate the quantitative improvements. VideoAR consistently produces visually compelling and semantically coherent videos, spanning imaginative artistic stylizations, high-fidelity natural scenes, and dynamic human actions with strong temporal consistency.

Video Generation on UCF-101. Our VideoAR framework establishes a new state-of-the-art on the UCF-101 dataset, marking a paradigm shift in achieving both superior generation quality and unprecedented inference efficiency. As shown in Tab. 3, our 2B parameter model, VideoAR-XL, achieves a new best FVD of 88.6, surpassing the previous leading autoregressive model, PAR-4x, by a notable 11%. Even our smaller 926M model, VideoAR-L, outperforms it with an FVD of 90.3. The most remarkable advancement,

Crucially, these results confirm that our VideoAR strategy offers a compelling alternative to diffusion-based paradigms. It achieves SOTA-level performance, particularly in semantic control and motion depiction, while providing strong potential for improved scalability and significantly higher inference efficiency.

###### Image-to-Video and Video-Continuation.

As an autoregressive video generation model, our pro-

- Table 4. Ablation study for Multi-scale Temporal RoPE and Cross-Frame Error Correction on the UCF-101 dataset. The checkmark (✓) indicates the component is enabled.

Methods

Multi-scale Temp. RoPE

Time-Dependent Corruption

Error Inheritance gFVD ↓

VideoAR-L (Baseline) 96.04 VideoAR-L ✓ 94.95 VideoAR-L ✓ ✓ 93.57 VideoAR-L (Full) ✓ ✓ ✓ 92.50

posed VideoAR can directly extend future frames from preceding content (including an initial image and sequential frames) without requiring external fine-tuning. For evaluation, we sample several test cases from VBench-I2V. We present multiple Image-to-Video (I2V) and Video-to-Video (V2V) examples where VideoAR enables single or multishot continuous video generation. As shown in Figure 5, VideoAR-4B accurately follows semantic prompts aligned with input images across various settings, including object motion control and camera trajectory adjustments. For the video continuity task, VideoAR can generate natural and consistent content over multiple iterations, ultimately producing long-form videos exceeding 20 seconds in duration.

- 5.3. Ablation Studies

We conduct a comprehensive ablation study on the UCF101 dataset. All models are trained for a fixed 1,000 steps, which is sufficient to reveal clear trends in model performance.

Effect of Multi-scale Temporal RoPE. Our first enhancement replaces the standard positional encoding with Multiscale Temporal RoPE. As shown in the second row of Tab. 4, this single modification reduces the FVD from 96.04 to 94.95. This result highlights the importance of rotational relative positional encoding for modeling the complex spatio-temporal dynamics of video data, thereby improving frame-to-frame consistency.

Effect of Temporal-Consistency Enhancement.Next, we evaluate our proposed Cross-Frame Error Correction mechanism, which consists of two synergistic components. (1) We first activate Time-dependent Corruption, a data augmentation strategy that simulates inference-time conditions during training. This addition further reduces the FVD to 93.57. (2) Building on this, we incorporate Error Inheritance Initialization, which encourages the model to correct inherited perturbations for improved future predictions. This final step yields our full model, achieving a state-ofthe-art FVD of 92.50. Further ablation for Random Frame Mask is performed on our large-scale real-world dataset, as strong augmentation to small datasets UCF-101 can impede model convergence. As shown in Tab. 5, incorporating this technique during the 256px training stage improves the overall VBench score from 76.22 to 77.00.

Table 5. Ablation study for Random Frame Mask on the VBench benchmark during 256px stage-I training.

Methods Overall Score Quality Semantic

VideoAR-L 76.22 78.63 66.64 VideoAR-L w/ Rand. Mask 77.00 79.78 65.89

### 6. Discussion

##### 6.1. Comparsion to Concurrent work: InfinityStar

We highlight several key differences compared to InfinityStar [12].

- (1) Spatio-temporal Modeling Paradigm. InfinityStar

adopts a 3D-VAR formulation, where each generation block operates over a temporal window of frames. In contrast, our VideoAR employs a next-frame prediction paradigm combined with multi-scale modeling within each frame. This design enables fine-grained spatial modeling through structured coarse-to-fine generation, while maintaining temporal consistency via explicit frame-wise prediction.

- (2) Training Strategy. InfinityStar is fine-tuned from

a well-established 8B-scale image generation foundation model, benefiting from strong pre-trained priors. In contrast, our VideoAR is trained from scratch using joint lowresolution image–video data, which focuses on learning unified spatio-temporal representations from the ground up.

(2) Training Scale and Sequence Length. Also, VideoAR is trained with relatively modest sequence lengths, primarily due to practical training considerations at this stage. Consequently, long-horizon temporal coherence has not yet been exhaustively explored. Nevertheless, the proposed framework imposes no inherent limitation on sequence length, and is fully compatible with longer-context training. We expect further gains in long-term consistency as training scale and sequence length are increased.

### 7. Conclusion

We present VideoAR, a new paradigm for scalable autoregressive video generation built upon the next-scale prediction principle. By extending VAR framework to videos, VideoAR unifies spatial and temporal modeling through a causal 3D tokenizer and a Transformer-based generator. The proposed Multi-scale Temporal RoPE enhances spatio-temporal representation learning, while Cross-Frame Error Correction and Random Frame Mask effectively mitigate cumulative errors and improve long-form stability. Extensive experiments demonstrate that VideoAR not only achieves state-of-the-art gFVD (88.6) and VBench (81.7) scores but also enables 13× faster inference compared to existing AR baselines. These findings highlight autoregressive modeling as a practical and powerful alternative to diffusion-based approaches, paving the way toward efficient, large-scale video generation.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2
- [2] Sand. ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, and Yuqiao Li. Magi-1: Autoregressive video generation at scale, 2025. 3
- [3] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024. 8
- [4] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 3
- [5] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022. 8
- [6] Google DeepMind. Veo 3, 2025. Closed-source video generation model. 2
- [7] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15733–15744, 2025. 4, 5
- [8] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2, 8
- [9] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 2, 6
- [10] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 8
- [11] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [12] Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Infini-

- tystar: Unified spacetime autoregressive modeling for visual generation. arXiv preprint arXiv:2511.04675, 2025. 9
- [13] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Sheng-Siang Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, Yu Zhou, Deshan Sun, Deyu Zhou, Jian Zhou, Kaijun Tan, Kang An, Mei Chen, Wei Ji, Qiling Wu, Wenzheng Sun, Xin Han, Yana Wei, Zheng Ge, Aojie Li, Bin Wang, Bizhu Huang, Bo Wang, Brian Li, Changxing Miao, Chen Xu, Chenfei Wu, Chenguang Yu, Da Shi, Dingyuan Hu, Enle Liu, Gang Yu, Gege Yang, Guanzhe Huang, Gulin Yan, Hai bo Feng, Hao Nie, Hao Jia, Hanpeng Hu, Hanqi Chen, Haolong Yan, Heng Wang, Hong-Wei Guo, Huilin Xiong, Hui Xiong, Jiahao Gong, Jianchang Wu, Jiao Wu, Jie Wu, Jie Yang, Jiashuai Liu, Jiashuo Li, Jingyang Zhang, Jun-Nan Guo, Junzhe Lin, Kai hua Li, Lei Liu, Lei Xia, Liang Zhao, Liguo Tan, Liwen Huang, Li-Li Shi, Ming Li, Mingliang Li, Muhua Cheng, Na Wang, Qiao-Li Chen, Qi He, Qi Liang, Quan Sun, Ran Sun, Rui Wang, Shaoliang Pang, Shi kui Yang, Si-Ye Liu, Siqi Liu, Shu-Guang Gao, Tiancheng Cao, Tianyu Wang, Weipeng Ming, Wenqing He, Xuefeng Zhao, Xuelin Zhang, Xi Zeng, Xiaojian Liu, Xuan Yang, Ya-Nan Dai, Yanbo Yu, Yang Li, Yin-Yong Deng, Yingming Wang, Yilei Wang, Yuanwei Lu, Yu Chen, Yu Luo, and Yu Luo. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. ArXiv, abs/2502.10248, 2025. 2, 8
- [14] OpenAI. Sora 2, 2025. Closed-source video generation model. 2
- [15] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [16] Runway. Runway gen-3, 2024. Closed-source video generation model. 8
- [17] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 6
- [18] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 5

- [19] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 3
- [20] Kuaishou Technology. Kling, 2024. 8
- [21] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. 2024. 2
- [22] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothy Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [23] Hung-Yu Tseng, Lu Jiang, Ce Liu, Ming-Hsuan Yang, and Weilong Yang. Regularizing generative adversarial networks

- under limited data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7921–7931, 2021. 4
- [24] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019. 6
- [25] Aaron van den Oord, Oriol Vinyals, and koray kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems. Curran Associates, Inc.,

2017. 2

- [26] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2
- [27] Junke Wang, Yi Jiang, Zehuan Yuan, Bingyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint imagevideo tokenizer for visual generation. Advances in Neural Information Processing Systems, 37:28281–28295, 2024. 8
- [28] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2
- [29] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. IJCV, 2024. 8
- [30] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 2, 3
- [31] Yuqing Wang, Shuhuai Ren, Zhijie Lin, Yujin Han, Haoyuan Guo, Zhenheng Yang, Difan Zou, Jiashi Feng, and Xihui Liu. Parallelized autoregressive visual generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12955–12965, 2025. 3, 8
- [32] Hu Yu, Biao Gong, Hangjie Yuan, DanDan Zheng, Weilong Chai, Jingdong Chen, Kecheng Zheng, and Feng Zhao. Videomar: Autoregressive video generatio with continuous tokens. arXiv preprint arXiv:2506.14168, 2025. 2
- [33] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 3, 4, 8

[34] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 3, 4, 8

## VideoAR: Autoregressive Video Generation via Next-Frame & Scale Prediction Supplementary Material

### A. Implementation Details.

Model Architecture. Our video generation framework comprises a video tokenizer and an autoregressive Transformer. The tokenizer, which maps video clips into a discrete latent space, is spatially initialized with the pre-trained weights of the publicly available Infinity image model. It operates with a latent dimension of d = 48 and a temporal subsampling ratio of τ = 4. The core generative module is an autoregressive Transformer whose scale follows the architectural principles of the LLaMA family [22]. VideoARL/XL employs 24/32 layers Transformer with a hidden dimension of 1536/2048 and 12/16 heads. For large-scale experiments, we further introduce VideoAR-4B with 36 layers, 2560-dim and 32 heads.

Training Configuration. We first fine-tune the video tokenizer on the UCF-101 dataset for 2000 epochs with a batch size of 128. Subsequently, the autoregressive Transformer is trained using the AdamW optimizer (β1 = 0.9,β2 = 0.95) with a cosine learning rate schedule of 1×10−4 and a weight decay of 0.05. We employ several key strategies during this phase: (i) Cross-Frame Error Correction (Eq. (8)) with parameters pmin = 0, pmax = 0.25, and δ = 0.01; and (ii) Random Frame Mask (Eq. (9)). For experiments on our large-scale internal dataset, we use the same hyperparameter settings while extending the training duration to account for the increased data complexity. To optimize computational efficiency, we utilize mixedprecision training and gradient checkpointing.

Inference. During inference, we adopt different temporal–spatial CFG schedules for class-conditional UCF-101 and real-world text-to-video generation. For UCF-101, to balance sampling diversity and visual quality, we gradually increase the initial CFG strength in each frame’s first scale from γ = 1 → 5, followed by a linear increase within each frame up to γ = 10. For real-world text-to-video generation, we reduce the CFG from γ = 5 → 3 along each temporal-spatial dimension together to maintain stronger spatial consistency in the generated videos.

### B. Visualization of Generated Videos

In this section, we present additional generated samples of our VideoAR-4B on VBench benchmarks. As illustrated in Figure A1, VideoAR-4B demonstrates the capability to generate high-fidelity and temporally consistent videos across diverse domains. Specifically, it can maintain object consistency in high-dynamic scenes (e.g., fireworks displays, drum-playing performances), synthesize high-aesthetic natural scenery (e.g., sunsets, the Pacific Coast), precisely ad-

here to semantic guidance for multi-object scenarios, and generate stylized content such as Cyberpunk-themed videos and imaginative visuals.

### C. Limitations and Future Work

Through extensive experiments, we identify three primary limitations of the current model:

- • Limited Resolution and FPS. Our current VideoAR4B generates videos at a resolution of 384×672 and 8 frames per second (FPS), which is insufficient for commercial applications—where standard specifications typically include 24 FPS and 720P resolution. This constraint stems from limited computational resources during training, which restrict the maximum sequence length. Additionally, the adoption of a full autoregressive (VAR) attention mask results in high computational overhead. In future work, we will extend the training sequence length and explore sparser attention mechanisms to enable highresolution, fluid video generation.
- • Drifting Issues in High-Dynamic Scenarios. During experimentation, we observe that VideoAR-4B tends to produce drifted motions for high-dynamic scenes (e.g., complex human movements). This phenomenon arises from the error-propagation inherent to autoregressive models. To address this, future research will enhance the model’s performance by integrating iterative inference-time rollouts and reinforcement learning algorithms.

[Figure 9]

###### Figure A1. Visualization of more generated videos of our VideoAR-4B model.

