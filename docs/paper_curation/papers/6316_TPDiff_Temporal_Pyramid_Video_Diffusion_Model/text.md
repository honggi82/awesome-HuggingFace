# arXiv:2503.09566v1[cs.CV]12Mar2025

## TPDiff: Temporal Pyramid Video Diffusion Model

Lingmin Ran1 Mike Zheng Shou1,* 1Show Lab, National University of Singapore

### Abstract

t=0

Denoising step:

| | | |
|---|---|---|
| | | |
| | | |

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

The development of video diffusion models unveils a significant challenge: the substantial computational demands. To mitigate this challenge, we note that the reverse process of diffusion exhibits an inherent entropy-reducing nature. Given the inter-frame redundancy in video modality, maintaining full frame rates in high-entropy stages is unnecessary. Based on this insight, we propose TPDiff, a unified framework to enhance training and inference efficiency. By dividing diffusion into several stages, our framework progressively increases frame rate along the diffusion process with only the last stage operating on full frame rate, thereby optimizing computational efficiency. To train the multi-stage diffusion model, we introduce a dedicated training framework: stage-wise diffusion. By solving the partitioned probability flow ordinary differential equations (ODE) of diffusion under aligned data and noise, our training strategy is applicable to various diffusion forms and further enhances training efficiency. Comprehensive experimental evaluations validate the generality of our method, demonstrating 50% reduction in training cost and 1.5x improvement in inference efficiency. Our project page is: https://showlab.github.io/TPDiff/

Same frame rate

Frame rate increases

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

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

(a) Vanilla Video (b) TPDiff Diffusion Model

t=T

[Figure 25]

(c) Comparison of our method and vanilla diffusion

Figure 1. Overview of our method. Our method employs progressive frame rates, which utilizes full frame rate only in the final stage as shown in (a) and (b), thereby largely optimizing computational efficiency in both training and inference shown in (c).

of approaches to increase training and inference efficiency. Show-1 [41] and Lavie [32] adopts cascaded framework to model temporal relations at low resolution and apply superresolution to improve the final video resolution. However, the cascaded structure leads to error accumulation and significantly increases the inference time. SimDA [36] proposes a lightweight model which replaces Attention [31] with 3D convolution [29] to model temporal relationship. Although convolution is computationally efficient, DiT [22] demonstrates that attention-based model is scalable and achieves better performance as the volume of data and model parameters increases. Recently, [8] introduces an interesting work: pyramid flow. This method proposes spatial pyramid: it employs low resolution during the early diffusion steps and gradually increases the resolution as the diffusion process proceeds. It avoids the need to always maintain full resolution, significantly reducing computational costs.

### 1. Introduction

With the development of diffusion models, video generation has achieved significant breakthroughs. The most advanced video diffusion models [10, 19, 23] not only enable individuals to engage in artistic creation but also demonstrate immense potential in other fields like robotics [12] and virtual reality [28]. Despite the powerful performance of video diffusion models, the complexity of jointly modeling spatial and temporal distribution makes their training costs prohibitively high [9, 20, 40]. Moreover, as the demand for long videos increases, the training and inference costs will continue to scale accordingly.

To alleviate this problem, researchers propose a series

However, pyramid flow has several problems: 1) It only

*Corresponding Author.

demonstrates its effectiveness under flow matching [13] and does not explore its applicability to other diffusion forms like denoising diffusion implicit models (DDIM) [26]. 2) It formulates video generation in an auto-regressive manner which significantly reduces inference speed. 3) The feasibility of modeling temporal relationship in a pyramid-like structure remains unexplored.

To solve the problems, we propose TPDiff, a general framework to accelerate training and inference speed. Our method is inspired by the fact that video is a highly redundant modality [16], as consecutive frames often contain minimal variations. Additionally, in a typical diffusion process, latents in the early timesteps contain limited informative content and the temporal relations between frames are weak, which makes maintaining full frame rate throughout this process unnecessary. Based on this insight, we propose temporal pyramid: 1) In our method, the frame rate progressively increases as the diffusion process proceeds as shown in Fig. 1. Unlike previous works [32, 41] require an additional temporal interpolation network, we adopt a single model to handle different frame rates. To achieve this, we divide the diffusion process into multiple stages, with each stage operating at different frame rate. 2) To train the temporal pyramid model, we solve the partitioned probability flow ordinary differential equations (ODE) [14, 27] by leveraging data-noise alignment and reach a unified solution for various types of diffusion. 3) Our experiments show our method is generalizable to different diffusion forms, including flow matching and DDIM, achieving 2x faster training and 1.5x faster inference compared to vanilla diffusion models. The core contributions of this paper are summarized as follows:

- • We introduce temporal pyramid video diffusion model, a generalizable framework aiming at enhancing the efficiency of training and inference for video diffusion models. By employing progressive frame rates across different stages of the diffusion process, the framework achieves substantial reductions in computational cost.
- • We design a dedicated training framework: stage-wise diffusion. We solve the decomposed probability flow ODE by aligning noise and data. The solution is applicable to different diffusion forms, enabling flexible and seamless extension to various video generation frameworks.
- • Our experiments demonstrate that the proposed method can be applied across various diffusion frameworks, achieving performance improvement, 2x faster training and 1.5x faster inference.

### 2. Related works

Generative Video Models. The field of video generation has witnessed significant progress recently due to the advancement of diffusion models [3, 25] These models

generate videos from text descriptions or images. Most methods develop video models based on powerful text-toimage models like Stable Diffusion [24], adding extra layers to capture cross-frame motion and ensure consistency. Among these, Tune-A-Video [35] employs a causal attention module and limits training module to reduce computational costs. AnimateDiff [5] utilizes a plug-and-play temporal module to enable video generation on personalized image models [1]. Recently, DiT models [17, 20] pushes the boundaries of video generation. Commercial products [10, 15, 19] and open-source works [9, 20, 40] demonstrate remarkable performance by scaling up DiT pretraining. Although DiT achieves significant performance improvements, its training cost escalates to an unaffordable level, hindering the development of video generation.

Temporal Pyramid. The complex temporal structure of videos raises a challenge for generation and understanding. SlowFast [4] simplifies video understanding by utilizing an input-level frame pyramid, where frames at different levels are sampled at varying rates. Each level is independently processed by a separate network, with their mid-level features interactively fused. This combination of the frame pyramid enables SlowFast to efficiently manage the variability of visual tempos. Similarly, DTPN [2] employs different frame-per-second (FPS) sampling to construct a pyramidal representation for videos of arbitrary length. Temporal pyramid network [39] leverages the feature hierarchy to handle the variance of temporal information. It avoids to learn visual tempos inside a single network, and only need frames sampled at a single rate at the input-level. Although the effectiveness of temporal pyramid have been validated in video understanding, its application in generation remains under-explored.

### 3. Method

#### 3.1. Preliminary

Denoising Diffusion Implicit Models DDIM [26] extends DDPMs [6] by operating in the latent space. Similar to DDPM, in the forward process, DDIM transforms real data x0 into a series of intermediate sample xt, and eventually the Gaussian noise ϵ ∼ N(0,I) according to noise schedule αt:

xt = √αtx0 + √1 − αtϵ,ϵ ∼ N(0,I), (1)

where t ∼ [1,T] and T denotes the total timesteps. After adding noise to the latent, we usually train a neural network ϵθ to predict the added noise. Formally, ϵθ is trained using following objecive:

t,ϵ∼N(0,I),t∼ Uniform (1,T) ∥ϵ − ϵθ (xt,t)∥22 . (2)

min

Ex

θ

Given a pretrained diffusion model ϵθ, one can generate new data by solving the corresponding probability flow

Vanilla Diffusion Ours

Diffusion Process

t=0

t=0

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

t=0

t=t1

- t=t1
- t=t2

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

t=t2

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

t=T t=T

t=T

Temporal interpolation Noise direction

Diffusion trajectory

(a) Temporal pyramid video diffusion model (b) Training strategy

Figure 2. Methodology. a) Pipeline of temporal pyramid video diffusion model. We divide diffusion process into multiple stages with increasing frame rate. In each stage, new frames are initially temporally interpolated from existing frames. b) Our training strategy: stagewise diffusion. In vanilla diffusion models, the noise direction along the ODE path points toward the real data distribution. In stage-wise diffusion, the noise direction is oriented to the end point of the current stage.

ODE [27]. DDIM is essentially a first-order ODE solver, which formulates a denoising process to generate xt−1 from a sample xt via:

√1 − αtϵθ(xt,t) αt

+√1 − αtϵθ(xt,t),

xt −

xt−1 = αt−1

(3) where αt = α

αt−1.

t

Flow Matching Flow-based generative models aim to learn a velocity field vθ that transports Gaussian noise ϵ ∼ N(0,I) to the distribution of real data x0. Flow matching [13] adopts linear interpolation between noise ϵ and data x0:

xt = (1 − t)x0 + tϵ,ϵ ∼ N(0,I). (4) It trains a neural network ϵθ to match the velocity field and then solves the ODE for a given boundary condition ϵ to obtain the flow. The flow matching loss function is as follows:

Ex,ϵ∼N(0,I),t∼ Uniform (1,T) ∥(ϵ − x0) − vθ (xt,t)∥22 .

min

θ

(5)

#### 3.2. Temporal pyramid diffusion

The core module of existing video diffusion models, attention [31], exhibits quadratic complexity with respect to sequence length. Our goal is to reduce the sequence length in video generation and decrease the computational cost. Our method is based on two key insights: 1) There is considerable redundancy between consecutive video frames. 2) the early stages of the diffusion process remain at low signalto-noise ratio (SNR), resulting in minimal information content. It suggests that operating at full frame rate during these

initial timesteps is unnecessary. Based on these insights, we propose temporal pyramid video diffusion as shown in Fig. 2. Compared to traditional video diffusion model using fixed frame rate, our framework progressively increases the frame rate as the denoising proceeds.

In detail, we divide the diffusion process into multiple stages, each characterized by a distinct frame rate, and employ a single model to learn data distributions across all stages. We create K stages {[tk,tk−1)}1k=K where 0 < t1 < t2... < tK−1 < T, T denotes the total timesteps. The frame rate at the kth stage is reduced to 2 1

of the original one. It ensures that only the last stage operates at full frame rate, thereby optimizing computational efficiency. Despite efficiency, the vanilla diffusion model does not support multi-stage training and inference. Therefore, the remaining challenges are: 1) How to train the multistage diffsion model in a unified way, which will be introduced in Section 3.3 and Section 3.4, 2) How to perform inference, which will be discussed in Section 3.5.

k−1

#### 3.3. Training strategy

In stage k, we denote (sk,ek) as the start and end timestep, xˆs

as start and end point. The objective of training is to transport distribution of xˆs

and xˆe

k

k

at every stage. To achieve the objective, the key is to obtain stage-wise 1) target, i.e. ϵ in DDIM and dxt

to xˆe

k

k

dt in flow matching, and 2) intermediate latents xt where t ∈ [sk,ek) [33, 38]. In the following, we will introduce a unified training framework named stage-wise diffusion.

Stage-wise Diffusion To ensure generality, recognizing different diffusion frameworks share a similar formulation as shown in Equation 1 and Equation 4, we present a unified diffusion form:

xt = γtx0 + σtϵ, (6) where the form of γt and σt depend on diffusion framework selected. Our derivation is based on Equation 6, without constraining the parameterization of γt and σt. Considering continuity between stages with distinct frame rates, we obtain xˆs

by: xˆs

and xˆe

k

k

Up(Down(x0,2k+1),2) + σs

ϵ, (7) xˆe

= γs

k

k

k

Down(x0,2k) + σe

ϵ, (8)

= γe

k

k

k

where ϵ ∼ N(0,I), Down(·,2k) and Up(·,2k) are downsampling and upsampling 2k times along temporal axis. We derive the start point of current stage from the end point of preceding stage in Equation 7 to bridge adjacent stages, which is crucial for inference and will be introduced in Section 3.5. However, this design also leads to boundary distribution shift and we cannot directly obtain training target from Equation 7 and Equation 8. Instead, we should compute added noise in every stage with boundary condition xˆs

and xˆe

. Fortunately, DPM-Solver [14] derives the relationship between any two points, xs and xe on diffusion ODE path and this relationship can also be applied to any stage in our method. Accordingly, in stage k, by replacing xs with xˆs

k

k

and xe with xt, we can express intermediate latent xt as a function of xˆs

k

:

k

γt γs

k − γt

xˆs

xt =

k

λt

e−λϵ(xt

λ

λsk

,tλ)dλ, (9)

where ek < t < sk, λt = lnγ

σt, and tλ is the inverse function of λt. Equation 9 consists of two components: a deterministic scaling factor, given by γ

t

γsk , and the exponentially weighted integral of the noise ϵ(xt

t

,tλ). If ϵ(xt

,tλ) is a constant in stage k, denoted as ϵk, the above integral is equivalent to:

λ

λ

γt γs

k − γtϵk

xˆs

xt =

k

γt γs

xˆs

=

k

k

+ γtϵk

λt

e−λdλ

λsk

σt γt −

σs

k

γs

k

.

(10)

While enforcing a constant value for ϵk at any stage is challenging, we can leverage data-noise alignment [11] to constrain its value within a narrow range. In detail, before adding noise to video, we pre-determine the target noise distribution for each video by minimizing the aggregate distance between video-noise pairs as shown in Fig. 3, thereby ensuring data-noise alignment and Equation 9 are approximately equivalent to Equation 10. The alignment process

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

Data Distribution

Noise Distribution

(a) Random sampling (b) Data-noise alignment

Figure 3. Data-Noise Alignment. For every training sample, (a) vanilla diffusion training randomly samples noises across the entire noise distribution, resulting in stochastic ODE path during training. (b) In contrast, our method samples noises in the closest range, making the ODE path approximately deterministic during training.

can be implemented using Scipy [21] in one line of code as shown in Algorithm 1.

Algorithm 1 Data-Noise Alignment Require: Video batch x, random noise ϵ

- 1: assign mat ← scipy.optimize. linear sum assignment(dist(x, ϵ))

- 2: ϵ′ ← ϵ[assign mat]

##### Output: ϵ′

Our experiments demonstrate that this approximation is valid and does not compromise the model’s performance.

Through data-noise alignment, we can apply Equation 10 to any point in the stage, including the end point xˆe

. By substituting t = ek and xt = xˆe

into Equation 10, through simple transformation, we arrive at the expression for noise ϵk of stage k :

k

k

xˆek γek − xˆs

k γsk

. (11)

ϵk =

σek γek − σs

k γsk

Then we can easily get any intermediate point xt in stage k by substituting ϵk into Equation 10. Consequently, we can compute the corresponding loss using xt and ϵk obtained in our method and optimize model parameters in the same way as vanilla diffusion training. Note that the above derivation does not constrain the expressions of γt and σt, making our method applicable to different diffusion frameworks. We also note that the direction of ϵk points towards the end point of the current stage rather than the final target in vanilla diffusion models as shown in Fig. 2. By reducing the distance between intermediate points and their target points, our method facilitates the training process and further accelerates model convergence.

#### 3.4. Practical implementation

In practice, for diffusion framework whose ODE path is curved, like DDIM, we can substitute γt = √α¯t and

σt = √1 − α¯t into Equation 10 and Equation 11 to obtain xt and ϵk. For flow matching, since it can transport any prior distribution to other distributions, we can model each stage as a complete flow matching process [8], resulting in a simpler expression:

xt = (1 − t′)ˆxe

+ t′xˆs

, (12) where t′ = t−e

k

k

sk−ek . And The objective of stage k is:

k

dxt dt′ = xˆs

k − xˆe

k

(13)

One aspect pyramid flow overlooks is the noise-data aligment, leading to increased variance in the prior distribution, thereby hindering model convergence. Notably, if we model each stage as a complete DDIM process, the model fails to converge. This is because it is exceedingly challenging for a single model to fit multiple curved ODE trajectories.

In conclusion, we visualize the training process of our method in Algorithm 2.

Algorithm 2 Stage-wise Diffusion Require: Training dataset D, Number of stages K, Dif-

fusion type DDIM or Flow Matching, Model ϵθ or vθ, Create K stages {[sk,ek)}Kk=1

- 1: repeat
- 2: Sample x0 ∼ D;
- 3: Sample stage k from {1,...K}, then sample timestep t ∈ [sk,ek)
- 4: Sample noise ϵ′ ∈ N(0,I) aligned with x0
- 5: Add ϵ′ to x0 and get xˆe

k

and xˆs

k

- 6: if Flow Matching then
- 7: xt = (1 − t′)ˆxe

k

+ t′xˆs

k

where t′ = t−e

k

sk−ek

- 8: vk = xˆs

k − xˆe

k

- 9: Compute loss: ℓ = ||vθ(xt) − v||2
- 10: else
- 11: ϵk =

xˆek αek

− αxˆssk

k

σek αek

− ασssk

k

- 12: xt = α

t

αsk xˆs

k

+ αtϵk σ

t

αt − σs

k αsk

- 13: Compute loss: ℓ = ||ϵθ(xt) − ϵk||2
- 14: end if
- 15: Update θ with gradient-based optimizer using ∇θℓ
- 16: until Convergence

#### 3.5. Inference Strategy

After training, we can use standard sampling algorithm [14] to solve the reverse ODE in every stage. However, carefully handling stage continuity is necessary.

Upon completion of a stage, we first upsample ek in temporal dimension to double its frame rate via interpolation. Subsequently, we scale Up(ek) and inject additional ran-

dom noise to match the distribution of xˆs

during training:

k−1

γs

) + δn′,n′ ∼ N(0,Σ′). (14)

k

xˆs

=

Up(ˆxe

k−1

k

γe

k

Scaling factor γsk

γek ensures mean continuity while δ is utilized to compensate for the discrepancy in variance [8]. Considering the simplest scenario using nearest temporal upsampling and lowering the effect of noise, we derive Equation 14 as (see Appendix A.1 for detailed derivations):

√2σt 2

√2γs

n′, (15)

+ √2γs

k

Up(ˆxe

) +

xˆs

=

k−1

k

σs

k

k

where,

1 −1 −1 1

n′ ∼ N(0,

). (16)

### 4. Experiments

#### 4.1. Experimental setting

We implement our method in both DDIM and flow matching. Since most video diffusion models are bulit upon pretrained image models, our experiments are based on two image models: MiniFlux [8] and SD1.5 [24]. These two models are trained under flow matching and DDIM, respectively. We extend MiniFlux to MiniFlux-vid by finetuning all its parameters on video data and we adopt AnimateDiff [5] to extend SD1.5 to video model. The number of stages is set to 3 and each stage is uniformly partitioned in all experiments. Our experiments are conducted on NVIDIA H100 GPU.

Dataset We construct our dataset by selecting approximately 100k high-quality text-video pairs from OpenVID1M [18]. This dataset comprises videos with both motion and aesthetic scores in the top 20%, or at least one of the scores in the top 3%. The resolution for MiniFlux-vid and AnimateDiff is 384p and 256x256.

Baselines We compare our method with the video diffusion models trained in vanilla diffusion framework. To demonstrate our approach does not lead to performance degradation, we train two baselines: MiniFlux-vid and Animatediff under vanilla flow matching and DDIM framework, without using temporal pyramid. We train these baselines from scratch on our curated dataset using the same hyperparameters as our method. To demonstrate the effectiveness of our approach compared to existing methods, we also train modelscope [34] and OpenSora [20] from scratch on our dataset.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

A man is talking on Mars.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Thunderstorm at night.

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

A fish is swimming in the water.

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

Fireworks bloom in the sky.

Figure 4. Qualitative comparison. In each pair of videos, the first row presents the results of models trained using vanilla diffusion and the second row shows the results of our method. The first two video pairs are generated by MiniFlux-vid and the remaining are generated by animatediff.

Evaluation We evaluate our model from two perspectives: generation quality and efficiency. To evaluate the generation quality, we adopt quantitative metrics from VBench [7] to compare our method’s performance with existing models. For efficiency, we visualize the convergence curve to intuitively demonstrate training efficiency. In detail, to evaluate the model’s generative capability during training, we follow common practice [34] to use validation videos from MSRVTT [37] for zero-shot generation evaluation. We systematically compute the FVD [30] value during training and present the FVD-GPU hours curve to demonstrate the training efficiency of our method. We also report the average inference time to validate the inference efficiency.

#### 4.2. Quantitative results

Model Method Latency(s)↓ Speed up MiniFlux-vid

Vanilla 20.79 -

Ours 12.18 1.71x AnimateDiff

Vanilla 6.01 -

##### Ours 4.04 1.49x

Table 1. Inference efficiency of baselines and our method. The total denoising step is set to 30 for all models.

Tab. 2 shows quantitative comparison between our method and baselines. Compared to existing method, our model achieves better results with a higher total score. Compared to the vanilla diffusion models, our approach demonstrates improvements in most aspects, demonstrating that it enhances efficiency without compromising performance. This further indicates that the vanilla video diffusion model contains substantial redundancy in temporal modeling, whereas our approach effectively eliminates such redundancies.

Figure. 5 shows that our method achieves speedup of 2x and 2.13x in training compared to vanilla diffusion models. This acceleration primarily stems from two factors: 1) Noise-data pairing: By aligning noise with data, we reduce the randomness in training. The model learns a nearly deterministic ODE path rather than the expectation of multiple intersecting ODE paths. 2) Shorter averaged sequence length. Since the computational complexity of attention mechanism scales quadratically with sequence length, our method requires significantly less computational complexity on average. For example, to generate a video of length T, the averaged computational cost of attention modules in our method is halved, reducing to 13(T2 + (T2 )2 + (T4 )2) ≈ 0.44T2 compared to T2 in vanilla diffusion model. This advantage is also reflected in faster inference speed as shown in Tab. 1.

2x Speed Up

- (a) DDIM

(b) Flow Matching

[Figure 88]

[Figure 89]

- (b) Flow Matching

2.13x Speed Up

- Figure 5. Convergence curve of vanilla diffusion models and our method on (a) DDIM, (b) Flow Matching. We illustrate the FVD of two methods with different GPU hours consumed. Our method achieves higher training efficiency compared to vanilla approachs.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

w. Renoising

w.o. Renoising

- Figure 6. Ablation study of inference strategy. Our method generates smooth, high-quality videos, whereas the baseline without inference renoising exhibits significant flickers

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

w. Data-noise Pair

w.o. Data-noise Pair

- Figure 7. Ablation study of data-noise alignment. Our method can produce clearer videos compared to the baseline.

Method Total Score Motion Smoothness Object Class Multiple Objects Spatial Relationship

ModelScope [34] 73.12% 95.83% 67.06% 33.91% 27.55% OpenSora [20] 74.08% 91.97% 78.30% 30.69% 41.11%

AnimateDiff - Vanilla 73.65% 96.28% 81.34% 39.42% 43.23% AnimateDiff - Ours 74.96% 97.68% 86.77% 31.15% 56.62%

MiniFlux-vid - Vanilla 77.69% 98.34% 77.32% 36.26% 49.79% MiniFlux-vid - Ours 78.52% 98.96% 83.21% 43.18% 42.23%

Table 2. Comparison of video generation quality of baselines and our method.

#### 4.3. Qualitative result

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Vanilla Diffusion

As shown in Fig. 4, we show qualitative comparison between our method and vanilla video diffusion models. The results generated by our method are presented in the second column and the outputs of our baseline are displayed in the first column. Evidently, our approach is able to generate videos with better semantic accuracy and larger motion. For instance, under prompt ”A man is talking on Mars”, the baseline generates a person merely shaking their head without speaking, failing to fully adhere to the prompt. In contrast, our approach accurately generates the specified actions, demonstrating superior alignment with the given prompt. Moreover, for AnimateDiff, the baseline generates videos that are nearly static, whereas our approach achieves motion with a more natural and reasonable amplitude.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Ours

Figure 8. Comparison between vanilla diffusion and our method after 5000 training steps. Our method can generate temporally stable videos even at very early training steps while vanilla method cannot. The prompt is ”A serene scene of a sunflower field.”

Ablation on renoising inference strategy We study the effect of this strategy by comparing inference with and without renoising as shown in Fig. 6. The results indicate that while not using corrective noises can still produce generally coherent videos, it inevitably leads to flickers and blurred result.

#### 4.4. Ablation study

We conduct ablation study on two key designs: data-noise alignment and renoising inference strategy.

Ablation on data-noise alignment To demonstrate the effectiveness of data-noise alignment, we curate a baseline that trains without alignment. Fig. 7 and Tab. 3 present comparison between our method and this variant. Our method is capable of generating high-quality and smooth videos, whereas the baseline produces blurred results. This is because, without alignment, the approximation from Equation 9 to Equation 10 incurs increased error. Consequently, xt and ϵk computed via Equation 10 and Equation 11 deviate from the true values, leading to blurred results.

Metrics FVD↓ CLIPSIM↑ MiniFlux-vid

Model Method

w.o. alignment 602.1 29.34

w. alignment 562.6 29.76 AnimateDiff

w.o. alignment 834.5 28.21 w. alignment 782.6 28.91

Table 3. Ablation on data-noise alignmenmt.

### 5. Discussion

In our experiments, we observe that our approach is capable of generating temporally stable videos even at very early training steps as shown in Fig. 8. For results of the vanilla diffusion model, the sunflower appears abruptly, whereas our method achieves much smoother camera movement. This is attributed to the temporal pyramid, which alleviates the need to learn the temporal relation of all frames under low SNR timesteps where inter-frame connections are actually absent. Consequently, our method achieves better visual quality and motion dynamics.

### 6. Conclusion

In this paper, we propose a general acceleration framework for video diffusion models. We introduce TPDiff, a framework that progressively increases the frame rate along the diffusion process. Moreover, we design a dedicated training framework named stage-wise diffusion, which is applicable to any form of diffusion. Our experiments demonstrate that our method accelerates both training and inference on different frame-

works.

### References

- [1] Stability AI. https : / / huggingface . co / runwayml/stable-diffusion-v1-5. 2
- [2] et al. Da Zhang, Xiyang Dai. Dynamic temporal pyramid network: A closer look at multi-scale modeling for activity detection. Asian Conference on Computer Vision, 2018. 2
- [3] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. arXiv preprint arxiv:2105.05233,

2021. 2

- [4] et al. Feichtenhofer, Christoph. Slowfast networks for video recognition. Proceedings of the IEEE/CVF international conference on computer vision, 2019. 2
- [5] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. 2024. 2, 5
- [6] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [7] et al. Huang Z, He Y. Vbench: Comprehensive benchmark suite for video generative models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 7
- [8] et al Jin, Yang. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 1, 5, 11

- [9] et al. Kong W, Tian Q. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 2
- [10] KuaiShou. Kling. https://klingai.com/,2024,

2024. 1, 2

- [11] Yiheng Li, Heyang Jiang, Akio Kodaira, Masayoshi Tomizuka, Kurt Keutzer, and Chenfeng Xu. Immiscible diffusion: Accelerating diffusion training with noise assignment. arXiv preprint arXiv:2406.12303, 2024. 4
- [12] et al. Liu S, Wu L. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864,

2024. 1

- [13] et al. Liu Xingchao, Chengyue Gong. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2, 3
- [14] et al. Lu C, Zhou Y. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 2022. 2, 4, 5
- [15] Luma. dream-machine. https://lumalabs.ai/ dream-machine,2024, 2024. 2
- [16] et al. Ma Siwei. Image and video compression with neural networks: A review. IEEE Transactions on Circuits and Systems for Video Technology, 2019. 2
- [17] et al. Ma X, Wang Y. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2

- [18] et al. Nan K, Xie R. Openvid-1m: A large-scale highquality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024. 5
- [19] OpenAI. Sora. https://openai.com/index/sora, 2024, 2024. 1, 2
- [20] OpenSora. opensora. https://github.com/ hpcaitech/Open-Sora, 2023. 1, 2, 5, 8
- [21] et al. Pauli Virtanen, Ralf Gommers. Scipy 1.0: fundamental algorithms for scientific computing in python. Nature Methods, 2020. 4
- [22] Saining Xie Peebles William. Scalable diffusion models with transformers. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1
- [23] et al. Polyak, Adam. Movie gen: A cast of media foundation models. arXiv:2410.13720, 2024. 1
- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. 2021. 2, 5
- [25] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. arXiv preprint arxiv:1503.03585, 2015. 2
- [26] et al. Song, Jiaming. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [27] et al. Song Y, Sohl-Dickstein J. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2021. 2, 3
- [28] et al. Stan G B M, Wofk D. Ldm3d-vr: Latent diffusion model for 3d vr. arXiv preprint arXiv:2311.03226, 2023. 1
- [29] et al. Tran D, Bourdev L. Learning spatiotemporal features with 3d convolutional networks. Proceedings of the IEEE international conference on computer vision, 2015. 1
- [30] et al. Unterthiner T, Van Steenkiste S. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv preprint arxiv:1706.03762, 2017. 1, 3
- [32] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, Yuwei Guo, Tianxing Wu, Chenyang Si, Yuming Jiang, Cunjian Chen, Chen Change Loy, Bo Dai, Dahua Lin, Yu Qiao, and Ziwei Liu. Lavie: High-quality video generation with cascaded latent diffusion models. 2023. 1, 2
- [33] et al. Wang F Y, Yang L. Rectified diffusion: Straightness is not your need in rectified flow. arXiv preprint arXiv:2410.07303, 2024. 3
- [34] et al. Wang J, Yuan H. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 5, 7, 8
- [35] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. 2023. 2
- [36] et al. Xing Z, Dai Q. Simda: Simple diffusion adapter for efficient video generation. Proceedings of the IEEE/CVF Con-

- ference on Computer Vision and Pattern Recognition, 2024. 1
- [37] et al. Xu J, Mei T. Msr-vtt: A large video description dataset for bridging video and language. Proceedings of the IEEE conference on computer vision and pattern recognition, 2016. 7
- [38] et al. Yan, Hanshu. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. arXiv preprint arXiv:2405.07510, 2024. 3
- [39] et al. Yang, Ceyuan. Temporal pyramid network for action recognition. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020. 2
- [40] et al. Yang Z, Teng J. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 2
- [41] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arxiv:2309.15818, 2023. 1, 2

### A. Appendix.

#### A.1. DERIVATION

This section provides derivation for Equation. 23. Our derivation primarily follows pyramid flow [8], and we extend it to the temporal dimension. According to Equation. 8 and Equation. 7:

Up(Down(x0,2k+1)),σs2

k ∼ N(γs

I) Up(ˆxe

xˆs

k

(17)

k

Up(Down(x0,2k+1)),σe

) ∼ N(γe

I)

k+1

k+1

k+1

Spatial pyramid has demonstrate that stages can be smoothly connected by renoising the endpoint of the last stage. Renoising process can be expressed as:

γs

) + αn′,n ∼ N(0,Σ′) (18) where the rescaling coefficient sk

k

xˆs

=

Up(ˆxe

k+1

k

γe

k+1

ek+1 allows the means of these distributions to be matched, and α is the noise weight. Additionally, we need to match the covariance matrices:

γs2

σk2+1Σ + α2Σ′ = σs2

I. (19)

k

γe2

k

k+1

we consider the simplest interpolation: nearest neighbor temporal upsampling. Then we can get upsampling Σ and noise’s covariance matrix Σ′ has the same structure as Σ:

Σblock =

1 1 1 1

=⇒ Σ′ =

1 γ γ 1

(20)

To ensure Σ′ is semidefinite, γ ∈ [−1,0]. Then we solve Equation. 19 and Equation. 20 by considering the equality of their diagonal and non-diagonal elements and get the solution:

√1 − γ σs

σs

γs

(21)

√1 − γ

√1 − γ

k

k

√

, α =

γe

=

k+1

−γ + γs

k

k

To reduce the affect of noise, let γ = −1 and substitute it into Equation. 21, we can get:

√2γs

√2σt 2

(22)

+ √2γs

k

γe

=

,α =

k+1

σs

k

k

We can finally obtain Equation. 23:

√2γs

√2σt 2

n′ (23) where:

+ √2γs

k

xˆs

=

Up(ˆxe

) +

k−1

k

σs

k

k

1 −1 −1 1

n′ ∼ N(0,

) (24)

