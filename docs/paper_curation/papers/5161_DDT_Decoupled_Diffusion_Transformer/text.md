## DDT: Decoupled Diffusion Transformer

Shuai Wang1 Zhi Tian2 Weilin Huang2 Limin Wang 1,

### 1Nanjing University 2ByteDance Seed Vision https://github.com/MCG-NJU/DDT

𝑥

𝑥 𝑡 𝑦

𝑥

𝑥 𝑡 𝑦

# arXiv:2504.05741v2[cs.CV]9Apr2025

Pretrained Visual Encoder

Pretrained Visual Encoder

Condition Encoder

Diffusion Transfomer

REPAlign

REPAlign

𝑥

Velocity Decoder

𝑣

𝑡

𝑣

(a) Our Decoupled Diffusion Transformer (b) Conventional Diffusion Transformer (c) FID compared with Other Diffusion Models

Figure 1. Our deoupled diffusion transformer (DDT-XL/2) achieves a SoTA 1.31 FID under 256 epochs. Our decoupled diffusion transformer models incorporate a condition encoder to extract semantic self-conditions and a velocity decoder to decode velocity.

### Abstract

condition between adjacent denoising steps. To minimize performance degradation, we propose a novel statistical dynamic programming approach to identify optimal sharing strategies.

Diffusion transformers have demonstrated remarkable generation quality, albeit requiring longer training iterations and numerous inference steps. In each denoising step, diffusion transformers encode the noisy inputs to extract the lower-frequency semantic component and then decode the higher frequency with identical modules. This scheme creates an inherent optimization dilemma: encoding lowfrequency semantics necessitates reducing high-frequency components, creating tension between semantic encoding and high-frequency decoding. To resolve this challenge, we propose a new Decoupled Diffusion Transformer (DDT), with a decoupled design of a dedicated condition encoder for semantic extraction alongside a specialized velocity decoder. Our experiments reveal that a more substantial encoder yields performance improvements as model size increases. For ImageNet 256 × 256, Our DDT-XL/2 achieves a new state-of-the-art performance of 1.31 FID (nearly 4× faster training convergence compared to previous diffusion transformers). For ImageNet 512 × 512, Our DDTXL/2 achieves a new state-of-the-art FID of 1.28. Additionally, as a beneficial by-product, our decoupled architecture enhances inference speed by enabling the sharing self-

### 1. Introduction

Image generation is a fundamental task in computer vision research, which aims at capturing the inherent data distribution of original image datasets and generating high-quality synthetic images through distribution sampling. Diffusion models [19, 21, 29, 30, 41] have recently emerged as highly promising solutions to learn the underlying data distribution in image generation, outperforming the GAN-based models [3, 40] and Auto-Regressive models [5, 43, 51].

The diffusion forward process gradually adds Gaussian noise to the pristine data following an SDE forward schedule [19, 21, 41]. The denoising process learns the score estimation from this corruption process. Once the score function is accurately learned, data samples can be synthesized by numerically solving the reverse SDE [21, 29, 30, 41].

: Corresponding author (lmwang@nju.edu.cn).

Diffusion Transformers [32, 36] introduce the transformer architecture into diffusion models to replace the traditionally dominant UNet-based model [2, 10]. Empirical evidence suggests that, given sufficient training iterations, diffusion transformers outperform conventional approaches even without relying on long residual connections [36]. Nevertheless, their slow convergence rate still poses great challenge for developing new models due to the high cost.

In this paper, we want to tackle the aforementioned major disadvantages from a model design perspective. Classic computer vision algorithms [4, 17, 23] strategically employ encoder-decoder architectures, prioritizing large encoders for rich feature extraction and lightweight decoders for efficient inference, while contemporary diffusion models predominantly rely on conventional decoder-only structures. We systematically investigate the underexplored potential of decoupled encoder-decoder designs in diffusion transformers, by answering the question of can decoupled encoder-decoder transformer unlock the capability of accelerated convergence and enhanced sample quality?

Through investigation experiments, we conclude that the plain diffusion transformer has an optimization dilemma between abstract structure information extraction and detailed appearance information recovery. Further, the diffusion transformer is limited in extracting semantic representation due to the raw pixel supervision [28, 52, 53]. To address this issue, we propose a new architecture to explicitly decouple low-frequency semantic encoding and high-frequency detailed decoding through a customized encoder-decoder design. We call this encoder-decoder diffusion transformer model as DDT (Decoupled Diffusion Transformer). DDT incorporates a condition encoder to extract semantic selfcondition features. The extracted self-condition is fed into a velocity decoder along with the noisy latent to regress the velocity field. To maintain the local consistency of selfcondition features of adjacent steps, we employ direct supervision of representation alignment and indirect supervision from the velocity regression loss of the decoder.

In the ImageNet256 × 256 dataset, using the traditional off-shelf VAE [38], our decoupled diffusion transformer (DDT-XL/2) model achieves the state-of-the-art performance of 1.31 FID with interval guidance under only 256 epochs, approximately 4× training acceleration compared to REPA [52]. In the ImageNet512×512 dataset, our DDT-XL/2 model achieves 1.28 FID within 500K finetuning steps.

Furthermore, our DDT achieves strong local consistency on its self-condition feature from the encoder. This property can significantly boost the inference speed by sharing the self-condition between adjacent steps. We formulate the optimal encoder sharing strategy solving as a classic minimal sum path problem by minimizing the performance drop of sharing self-condition among adjacent steps. We propose

a statistic dynamic programming approach to find the optimal encoder sharing strategy with negligible second-level time cost. Compared with the naive uniform sharing, our dynamic programming delivers a minimal FID drop. Our contributions are summarized as follows.

- • We propose a new decoupled diffusion transformer model, which consists of a condition encoder and a velocity decoder.
- • We propose statistic dynamic programming to find the optimal self-condition sharing strategy to boost inference speed while keeping minimal performance downgradation.
- • In the ImageNet256×256 dataset, using tradition SDf8d4 VAE, our decoupled diffusion transformer (DDT-XL/2) model achieves the SoTA 1.31 FID with interval guidance under only 256 epochs, approximately 4× training acceleration compared to REPA [52].
- • In the ImageNet512 × 512 dataset, our DDT-XL/2 model achieves the SoTA 1.28 FID, outperforming all previous methods with a significant margin.

### 2. Related Work

Diffusion Transformers. The pioneering work of DiT [36] introduced transformers into diffusion models to replace the traditionally dominant UNet architecture [2, 10]. Empirical evidence demonstrates that given sufficient training iterations, diffusion transformers outperform conventional approaches even without relying on long residual connections. SiT [32] further validated the transformer architecture with linear flow diffusion. Following the simplicity and scalability of the diffusion transformer [32, 36], SD3 [12], Lumina [54], and PixArt [6, 7] introduced the diffusion transformer to more advanced text-to-image areas. Moreover, recently, diffusion transformers have dominated the text-to-video area with substantiated visual and motion quality [1, 20, 24]. Our decoupled diffusion transformer (DDT) presents a new variant within the diffusion transformer family. It achieves faster convergence by decoupling the low-frequency encoding and the high-frequency decoding.

Fast Diffusion Training. To accelerate the training efficiency of diffusion transformers, recent advances have pursued multi-faceted optimizations. Operator-centric approaches [13, 45, 48, 49] leverage efficient attention mechanisms: linear-attention variants [13, 45, 49] reduced quadratic complexity to speed up training, while sparseattention architectures [48] prioritized sparsely relevant token interactions. Resampling approaches [12, 16] proposed lognorm sampling [12] or loss reweighting [16] techniques to stabilize training dynamics. Representation learning enhancement approaches integrate external inductive biases:

|[Figure 1]| |[Figure 2]| |[Figure 3]| |[Figure 4]| |
|---|---|---|---|---|---|---|---|
|[Figure 5]|[Figure 6]|[Figure 7]|[Figure 8]|[Figure 9]|[Figure 10]|[Figure 11]|[Figure 12]|
|[Figure 13]|[Figure 14]|[Figure 15]|[Figure 16]|[Figure 17]|[Figure 18]|[Figure 19]|[Figure 20]|

- Figure 2. Selected 256 × 256 and 512 × 512 resolution samples. Generated from DDT-XL/2 trained on ImageNet 256 × 256 resolution and ImageNet 512 × 512 resolution with CFG = 4.0.

|[Figure 21]|[Figure 22]|[Figure 23]|[Figure 24]|[Figure 25]|
|---|---|---|---|---|
|[Figure 26]|[Figure 27]|[Figure 28]|[Figure 29]|[Figure 30]|

Diffusion Process in 𝒙 space 𝒕 = 𝟎.𝟐 𝒕 = 𝟎.𝟒 𝒕 = 𝟎.𝟔 𝒕 = 𝟎.𝟖 𝒕 = 𝟏.𝟎

- Figure 3. The reverse-SDE process (generation) of SiT-XL/2 in x space. There is a clear generation process from low frequency to high frequency. Most of the time is spent on generating highfrequency details (from t = 0.4 to t = 1.0).

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

1.0

shift 1.0

- shift 1.5

- shift 2

- shift 3

0.8

RespacedTimesteps

0.6

###### FID50K

0.4

0.2

0.0

5 6 7 8 9 10 11 12 Num of inference steps

0.0 0.2 0.4 0.6 0.8 1.0 Uniform Timesteps

Figure 4. The FID50K metric of SiT-XL/2 for different timeshift values. We employ a 2-nd order Adams-like solver to collect the performance. Allocating more computation at noisy steps significantly improves the performance.

ably. In this framework, t = 0 corresponds to the pure noise timestep.

As illustrated in Fig. 3, diffusion models perform autoregressive refinement on spectral components [11, 37]. The diffusion transformer encodes the noisy latent to capture lower-frequency semantics before decoding higherfrequency details. However, this semantics encoding process inevitably attenuates high-frequency information, creating an optimization dilemma. This observation motivates our proposal to decouple the conventional decode-only diffusion transformer into an explicit encoder-decoder architecture.

REPA [52], RCG [27] and DoD [53] borrowed visionspecific priors into diffusion training, while masked modeling techniques [14, 15] strengthened spatial reasoning by enforcing structured feature completion during denoising. Collectively, these strategies address computational, sampling, and representational bottlenecks.

### 3. Preliminary Analysis

Linear-based flow matching [29, 30, 32] represents a specialized family of diffusion models that we focus on as our primary analytical subject due to its simplicity and efficiency. For the convenience of discussion, in certain situations, diffusion and flow-matching will be used interchange-

Lemma 1. For a linear flow-matching noise scheduler at timestep t, let us denote Kfreq as the maximum frequency of the clean data xdata. The maximum retained frequency

in the noisy latent satisfies:

fmax(t) > min

t 1 − t

2

,Kfreq . (1)

Lemma 1 is directly borrowed from [11, 37], we place the proof of Lemma 1 in Appendix. According to Lemma 1, as t increases to less noisy timesteps, semantic encoding becomes easier (due to noise reduction) while decoding complexity increases (as residual frequencies grow). Consider the worst-case scenario at denoising step t, the diffusion transformer encodes frequencies up to fmax(t), to progress to step s, it must decode a residual frequency of at least fmax(s) − fmax(t). Failure to decode these residual frequencies at step t creates a critical bottleneck for progression to subsequent steps. From this perspective, if allocating more of the calculations to more noisy timesteps can lead to an improvement, it means that diffusion transformers struggle with encoding lower frequency to provide semantics. Otherwise, if allocating more of the calculations to less noisy timesteps can lead to an improvement, it means that flow-matching transformers struggle with decoding higher frequency to provide fine details.

To figure out the bottom-necks of current diffusion models, we conducted a targeted experiment using SiT-XL/2 with a second-order Adams-like linear multistep solver. As shown in Fig. 4, by varying the time-shift values, we demonstrate that allocating more computation to early timesteps improves final performance compared to uniform scheduling. This reveals that diffusion models face challenges in more noisy steps. This leads to a key conclusion: Current diffusion transformers are fundamentally constrained by their low-frequency semantic encoding capacity. This insight motivates the exploration of encoderdecoder architectures with strategic encoder parameter allocation.

Prior researches further support this perspective. While lightweight diffusion MLP heads demonstrate limited decoding capacity, MAR [28] overcomes this limitation through semantic latents produced by its masked backbones, enabling high-quality image generation. Similarly, REPA [52] enhances low-frequency encoding through alignment with pre-trained vision foundations [35].

### 4. Method

Our decoupled diffusion transformer architecture comprises a condition encoder and a velocity decoder. The condition encoder extracted the low-frequency component from noisy input, class label, and timestep to serve as a selfcondition for the velocity decoder; the velocity decoder processed the noisy latent with the self-condition to regress the high-frequency velocity. We train this model using the established linear flow diffusion framework. For brevity,

we designate our model as DDT (Decoupled Diffusion Transformer).

##### 4.1. Condition Encoder

The condition encoder mirrors the architectural design and input structure of DiT/SiT with improved micro-design. It is built with interleaved Attention and FFN blocks, without long residual connections. The encoder processes three inputs, the noisy latent xt, timestep t, and class label y, to extract the self-condition feature zt through a series of stacked Attention and FFN blocks:

zt = Encoder (xt,t,y). (2)

Specifically, the noisy latent xt are patchfied into continuous tokens and then fed to extract the self-condition zt with aforementioned encoder blocks. The timestep t and class label y serve as external-conditioning information projected into embedding. These external-condition embeddings are progressively injected into the encoded features of xt using AdaLN-Zero[36] within each encoder block.

To maintain local consistency of zt across adjacent timesteps, we adopt the representation alignment technique from REPA [52]. Shown in Eq. (3), this method aligns the intermediate feature hi from the i-th layer in the selfmapping encoder with the DINOv2 representation r∗. Consistent to REPA [52], the hϕ is the learnable projection MLP:

Lenc = 1 − cos(r∗,hϕ(hi)). (3) This simple regularization accelerates training convergence, as shown in REPA [52], and facilitates local consistency of zt between adjacent steps. It allows sharing the selfcondition zt produced by the encoder between adjacent steps. Our experiments demonstrate that this encodersharing strategy significantly enhances inference efficiency with only negligible performance degradation.

Additionally, the encoder also receives indirect supervision from the decoder, which we elaborate on later.

##### 4.2. Velocity Decoder

The velocity decoder adopts the same architectural design as the condition encoder and consists of several stacked interleaved Attention and FFN blocks, akin to DiT/SiT. It takes the noisy latent xt, timestep t, and self-conditioning zt as inputs to estimate the velocity vt. Unlike the encoder, we assume that class label information is already embedded within zt. Thus, only the external-condition timestep t and self-condition feature zt are used as condition inputs for the decoder blocks:

###### vt = Decoder (xt,t,zt). (4)

As demonstrated previously, to further improve consistency of self-condition zt between adjacent steps, we employ

AdaLN-Zero [36] to inject zt into the decoder feature. The decoder is trained with the flow matching loss as shown in Eq. (5):

Ldec = E[

1

||(xdata − ϵ) − vt||2dt]. (5)

0

##### 4.3. Sampling acceleration

By incorporating explicit representation alignment into the encoder and implicit self-conditioning injection into the decoder, we achieve local consistency of zt across adjacent steps during training (shown in Fig. 5). This enables us to share zt within a suitable local range, reducing the computational burden on the self-mapping encoder.

Formally, given total inference steps N and encoder

computation bugets K, thus the sharing ratio is 1 − KN , we define Φ with |Φ| = K as the set of timesteps where the

self-condition is recalculated, as shown in Equation 6. If the current timestep t is not in Φ, we reuse the previously computed zt−∆t as zt. Otherwise, we recompute zt using the encoder and the current noisy latent xt:

zt =

zt−∆t, if t ∈/ Φ Encoder (xt,t,y), if t ∈ Φ

(6)

Uniform Encoder Sharing. This naive approach recaluculate self-condition zt every KN steps. Previous work, such as DeepCache [33], uses this naive handcrafted uniform Φ set to accelerate UNet models. However, UNet models, trained solely with a denoising loss and lacking robust representation alignment, exhibit less regularized local consistency in deeper features across adjacent steps compared to our DDT model. Also, we will propose a simple and elegant statistic dynamic programming algorithm to construct Φ. Our statistic dynamic programming can exploit the optimal Φ set optimally compared to the naive approaches [33].

Statistic Dynamic Programming. We construct the statistic similarity matrix of zt among different steps S ∈ RN×N using cosine distance. The optimal Φ set would guarantee the total similarity cost − Kk

Φk+1

i=Φk S[Φk,i] achieves global minimal. This question is a well-formed classic minimal sum path problem, it can be solved by dynamic programming. As shown in Eq. (8), we donate Cki as cost and Pki as traced path when Φk = i. the state transition function from Ckj−1 to Cki follows:

- i

min

- j=0

{Ckj−1 − Σil=jS[j,l]}. (7)

Cki =

Pki = argminij=0{Cki−1 − Σil=jS[j,l]}. (8) After obtaining the cost matrix C and tracked path P, the optimal Φ can be solved by backtracking P from PKN.

### 5. Experiment

We conduct experiments on 256x256 ImageNet datasets. The total training batch size is set to 256. Consistent with methodological approaches such as SiT [32], DiT [36], and REPA [52], we employed the Adam optimizer with a constant learning rate of 0.0001 throughout the entire training process. To ensure a fair comparative analysis, we did not use gradient clipping and learning rate warm-up techniques. Our default training infrastructure consisted of 16× or 8× A100 GPUs. For sampling, we take the Euler solver with 250 steps as the default choice. As for the VAE, we take the off-shelf VAE-ft-EMA with a downsample factor of 8 from Huggingface1. We report FID [18], sFID [34], IS [39], Precision and Recall [25].

##### 5.1. Improved baselines

Recent architectural improvements such as SwiGLU [46, 47], RoPE [42], and RMSNorm [46, 47] have been extensively validated in the research community [8, 31, 50]. Additionally, lognorm sampling [12] has demonstrated significant benefits for training convergence. Consequently, we developed improved baseline models by incorporating these advanced techniques, drawing inspiration from recent works in the field. The performance of these improved baselines is comprehensively provided in Tab. 2. To validate the reliability of our implementation, we also reproduced the results for REPA-B/2, achieving metrics that marginally exceed those originally reported in the REPA[52]. These reproduction results provide additional confidence in the robustness of our approach.

The improved baselines in our Tab. 2 consistently outperform their predecessors without REPA. However, upon implementing REPA, performance rapidly approaches a saturation point. This is particularly evident in the XL model size, where incremental technique improvements yield diminishingly small gains.

##### 5.2. Metric comparison with baselines

We present the performances of different-size models at 400K training steps in Tab. 2. Our diffusion encoderdecoder transformer(DDT) family demonstrates consistent and significant improvements across various model sizes. Our DDT-B/2(8En4De) model exceeds Improved-REPAB/2 by 2.8 FID gains. Our DDT-XL/2(22En6De) exceeds REPA-XL/2 by 1.3 FID gains. While the decoder-only diffusion transformers approach performance saturation with REPA[52], our DDT models continue to deliver superior results. The incremental technique improvements show diminishing gains, particularly in larger model sizes. However, our DDT models maintain a significant performance advantage, underscoring the effectiveness of our approach.

1https://huggingface.co/stabilityai/sd-vae-ftema

256×256, w/o CFG 256×256, w/ CFG

Params Epochs FID↓ IS↑ Pre.↑ Rec.↑ FID↓ IS↑ Pre.↑ Rec.↑

|MAR-B [28] CausalFusion [9] LDM-4 [38] DDT-L (Ours)<br><br>|208M 368M 400M 458M|800 800 170 80<br><br>|3.48 192.4 0.78 0.58 5.12 166.1 0.73 0.66<br><br>10.56 103.5 0.71 0.62 7.98 128.1 0.68 0.67<br><br>|2.31 281.7 0.82 0.57 1.94 264.4 0.82 0.59<br>3.6 247.7 0.87 0.48 1.64 310.5 0.81 0.61<br>|
|---|---|---|---|---|
|MAR-L [28] VAVAE [50] CausalFusion [9] ADM [10] DiT-XL [36] SiT-XL [32] ViT-XL [16] U-ViT-H/2 [2] MaskDiT [14] FlowDCN [48] RDM [44] REPA [52] DDT-XL (Ours) DDT-XL (Ours) DDT-XL (Ours)|479M<br><br>675M<br><br>676M<br><br><br>554M 675M 675M 451M 501M 675M 618M 553M 675M 675M 675M 675M<br><br>|800 800 800 400 1400 1400 400 400 1600 400 / 800 80 256 400<br><br>|2.6 221.4 0.79 0.60<br><br>2.17 205.6 0.77 0.65<br>3.61 180.9 0.75 0.66 10.94 - 0.69 0.63<br><br><br>9.62 121.5 0.67 0.67<br><br>8.3 - - 8.10 - - 6.58 - - -<br><br>5.69 178.0 0.74 0.60 8.36 122.5 0.69 0.65<br><br>5.27 153.4 0.75 0.62<br><br>5.9 157.8 0.70 0.69<br>6.62 135.2 0.69 0.67<br><br><br>6.30 146.7 0.68 0.68<br><br><br>6.27 154.7 0.68 0.69<br>|1.78 296.0 0.81 0.60<br><br>1.35 295.3 0.79 0.65<br><br>1.77 282.3 0.82 0.61<br><br>4.59 186.7 0.82 0.52<br><br>2.27 278.2 0.83 0.57<br><br><br>2.06 270.3 0.82 0.59<br><br><br>2.06 - - -<br><br><br>2.29 263.9 0.82 0.57 2.28 276.6 0.80 0.61 2.00 263.1 0.82 0.58 1.99 260.4 0.81 0.58 1.42 305.7 0.80 0.64 1.52 263.7 0.78 0.63 1.31 308.1 0.78 0.62 1.26 310.6 0.79 0.65<br><br>|

- Table 1. System performance comparison on ImageNet 256 × 256 class-conditioned generation. Gray blocks mean the algorithm uses VAE trained or fine-tuned on ImageNet instead of the off-shelf SD-VAE-f8d4-ft-ema.

Model FID↓ sFID↓ IS↑ Prec.↑ Rec.↑

SiT-B/2 [32] 33.0 6.46 43.7 0.53 0.63 REPA-B/2 [52] 24.4 6.40 59.9 0.59 0.65 REPA-B/2(Reproduced) 22.2 7.50 69.1 0.59 0.65 DDT-B/2† (8En4De) 21.1 7.81 73.0 0.60 0.65

Improved-SiT-B/2 25.1 6.54 58.8 0.57 0.64 Improved-REPA-B/2 19.1 6.88 76.49 0.60 0.66 DDT-B/2 (8En4De) 16.32 6.63 86.0 0.62 0.66

SiT-L/2 [32] 18.8 5.29 72.0 0.64 0.64 REPA-L/2 [52] 10.0 5.20 109.2 0.69 0.65 Improved-SiT-L/2 12.7 5.48 95.7 0.65 0.65 Improved-REPA-L/2 9.3 5.44 116.6 0.67 0.66 DDT-L/2 (20En4De) 7.98 5.50 128.1 0.68 0.67

SiT-XL/2 [32] 17.2 5.07 76.52 0.65 0.63 REPA-XL/2 [52] 7.9 5.06 122.6 0.70 0.65 Improved-SiT-XL/2 10.9 5.3 103.4 0.66 0.65 Improved-REPA-XL/2 8.14 5.34 124.9 0.68 0.67 DDT-XL/2 (22En6De) 6.62 4.86 135.1 0.69 0.67

- Table 2. Metrics of 400K training steps with different model sizes. All results are reported without classifier-free guidance. gray means metrics are copied from the original paper, otherwise it is produced by our codebase. By default, our DDT models are built on improved baselines. DDT† means model built on naive baseline without architecture improvement and lognorm sampling, consistent to REPA. Our DDT models consistently outperformed their counterparts.

##### 5.3. System level comparision

ImageNet 256×256. We report the final metrics of DDTXL/2 (22En6De) and DDT-L/2 (20En4De) at Tab. 1. Our DDT models demonstrate exceptional efficiency, achieving convergence in approximately 14 of the total epochs compared to REPA [52] and other diffusion transformer models. In order to maintain methodological consistency with REPA, we employed the classifier-free guidance with 2.0 in the interval [0.3,1], Our models delivered impressive results: DDT-L/2 achieved 1.64 FID, and DDT-XL/2 got 1.52 FID within just 80 epochs. By extending training to 256 epochs—still significantly more efficient than traditional 800-epoch approaches—our DDT-XL/2 established a new state-of-the-art benchmark of 1.31 FID on ImageNet 256×256, decisively outperforming previous diffusion transformer methodologies. To extend training to 400 epochs, our DDT-XL/2(22En6De) achieves 1.26 FID, nearly reaching the upper limit of SD-VAE-ft-EMA-f8d4, which has a 1.20 rFID on ImageNet256.

ImageNet 512×512 We provide the final metrics of DDTXL/2 at Tab. 3. To validate the superiority of our DDT model, we take our DDT-XL/2 trained on ImageNet 256 × 256 under 256 epochs as the initialization, fine-tune out DDT-XL/2 on ImageNet 512 × 512 for 100K steps. We adopt the aforementioned interval guidance [26] and we achieved a remarkable state-of-the-art performance of 1.90 FID, decisively outperforming REPA by a significant 0.28

###### ImageNet 512 × 512

Model FID↓ sFID↓ IS↑ Pre.↑ Rec.↑

BigGAN-deep [3] 8.43 8.13 177.90 0.88 0.29 StyleGAN-XL [40] 2.41 4.06 267.75 0.77 0.52

ADM-G [10] 7.72 6.57 172.71 0.87 0.42 ADM-G, ADM-U 3.85 5.86 221.72 0.84 0.53 DiT-XL/2 [36] 3.04 5.02 240.82 0.84 0.54 SiT-XL/2 [32] 2.62 4.18 252.21 0.84 0.57 REPA-XL/2 [52] 2.08 4.19 274.6 0.83 0.58 FlowDCN-XL/2 [48] 2.44 4.53 252.8 0.84 0.54 DDT-XL/2 (500K) 1.28 4.22 305.1 0.80 0.63

- Table 3. Benchmarking class-conditional image generation on ImageNet 512×512. Our DDT-XL/2(512 × 512) is fine-tuned from the same model trained on 256 × 256 resolution setting of 1.28M steps. We adopt the interval guidance with interval [0.3, 1] and CFG of 3.0

performance margin. In Tab. 3, some metrics exhibit subtle degradation, we attribute this to potentially insufficient fine-tuning. When allocating more training iterations to DDT-XL/2, it achieves 1.28 FID at 500K steps with CFG3.0 within the time interval [0.3,1.0].

- 5.4. Acceleration by Encoder sharing

As illustrated in Fig. 5, there is a strong local consistency of the self-condition in our condition encoder. Even zt=0 has a strong similarity above 0.8 with zt=1. This consistency provides an opportunity to speed up inference by sharing the encoder between adjacent steps.

We employed the simple uniform encoder sharing strategy and the new novel statistics dynamic programming strategy. Specifically, for the uniform strategy, we only recalculate the self-condition zt every K steps. For statistics dynamic programming, we solve the aforementioned minimal sum path on the similarity matrix by dynamic programming and recalculate zt according to the solved strategy. As shown in Fig. 6, there is a significant inference speedup nearly without visual quality loss when K is smaller than

- 6. As shown in Tab. 4, the metrics loss is still marginal, while the inference speedup is significant. The novel statistics dynamic programming slightly outperformed the naive uniform strategy with less FID drop.

- 5.5. Ablations

We conduct ablation studies on ImageNet 256 × 256 with DDT-B/2 and DDT-L/2. For sampling, we take the Euler solver with 250 steps as the default choice without classifier-free guidance. For training, we train each model with 80 epochs(400k steps), and the batch size is set to 256.

Encoder-Decoder Ratio we systematically explored ratios ranging from 2 : 1 to 5 : 1 across different model sizes.

SharRatio Acc Φ FID↓ sFID↓ IS↑ Prec.↑ Rec.↑

0.00 1.0× Uniform 1.31 4.62 308.1 0.78 0.66 0.50 1.6× Uniform 1.31 4.48 300.5 0.78 0.65 0.66 1.9× Uniform 1.32 4.46 301.2 0.78 0.65 0.75 2.3× Uniform 1.34 4.43 302.7 0.78 0.65

Uniform 1.36 4.40 303.3 0.78 0.64

0.80 2.6×

StatisticDP 1.33 4.37 301.7 0.78 0.64 0.83 2.7×

Uniform 1.37 4.41 302.8 0.78 0.64

StatisticDP 1.36 4.35 300.3 0.78 0.64 0.87 3.0×

Uniform 1.42 4.43 302.8 0.78 0.64 StatisticDP 1.40 4.35 302.4 0.78 0.64

Table 4. Metrics of 400K training steps with different model sizes. All results are reported without classifier-free guidance. gray means metrics are copied from the original paper, otherwise it is produced by our codebase. Our DDT models consistently outperformed its counterparts

Timsteps: 0 1

[Figure 31]

Timsteps:01

- Figure 5. The cosine similarity of self-condition feature zt from encoder between different timesteps. There is a strong correlation between adjacent steps, indicating the redundancy.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

SharRatio 50% (Acc 1.6x) 75% (Acc 2.3x) 83% (Acc 2.7x) 87% (Acc 3.0x)

- Figure 6. Sharing the self-condition zt in adjacent steps significant speedup the inference.We tried various sharing frequency configurations. There is marginal visual quality down-gradation when the sharing frequency is reasonable.

in Fig. 7 and Fig. 8. Our notation mEnnDe represents models with m encoder layers and n decoder layers. The inves-

32

4En8De 6En6De 8En4De 10En2De

4En8De 6En6De 8En4De 10En2De

85

4En8De 6En6De 8En4De 10En2De

30

0.660

80

28

75

0.655

26

InceptionScore

70

FID50K

0.650

24

Recall

65

22

0.645

60

20

55

0.640

18

50

0.635

16

45

100 150 200 250 300 350 400 Training iterations(K steps)

100 150 200 250 300 350 400 Training iterations(K steps)

100 150 200 250 300 350 400 Training iterations(K steps)

- Figure 7. The DDT-B/2 built upon Improved-baselines under various Encoder and Decoder layer ratio. DDT-B/2(8En4De) achieves much faster convergence speed and better performance.

100 150 200 250 300 350 400 Training iterations(K steps)

- 8

- 9

- 10

- 11

- 12

- 13

- 14

- 15

FID50K

10En14De 12En12De

16En8De 20En4De

100 150 200 250 300 350 400 Training iterations(K steps)

80

90

100

110

120

130

InceptionScore

10En14De 12En12De

16En8De 20En4De

100 150 200 250 300 350 400 Training iterations(K steps)

0.645

0.650

0.655

0.660

0.665

0.670

0.675

Recall

10En14De 12En12De

16En8De 20En4De

- Figure 8. The DDT-L/2 built upon Improved-baselines under various Encoder and Decoder layer ratio. DDT-L/2 prefers an unexpected aggressive encoder-deocder ratio DDT-L/2(20En4De) achieves much faster convergence speed and better performance.

tigation experiments in Fig. 7 and Fig. 8 revealed critical insights into architectural optimization. We observed that a larger encoder is beneficial for further improving the performance as the model size increases. For the Base model in Fig. 7, the optimal configuration emerged as 8 encoder layers and 4 decoder layers, delivering superior performance and convergence speed. Notably, the Large model in Fig. 8 exhibited a distinct preference, achieving peak performance with 20 encoder layers and 4 decoder layers, an unexpectedly aggressive encoder-decoder ratio. This unexpected discovery motivates us to scale the layer ratio in DDT-XL/2 to 22 encoder layers and 6 decoders to explore the performance upper limits of diffusion transformers.

Decoder Block types. In our investigation of decoder block types and their impact on high-frequency decoding performance, we systematically evaluated multiple architectural configurations. Our comprehensive assessment included alternative approaches such as simple 3×3 convolution blocks and naive MLP blocks. As shown in Tab. 5, the default (Attention with the MLP) setting achieves better results. Thanks to the encoder-decoder design, naive Conv blocks even achieve comparable results.

### 6. Conclusion

In this paper, we have introduced a novel Decoupled Diffusion Transformer, which rethinks the optimization dilemma

DecoderBlock FID↓ sFID↓ IS↑ Prec.↑ Rec.↑

Conv+MLP 16.96 7.33 85.1 0.62 0.65 MLP+MLP 24.13 7.89 65.0 0.57 0.65 Attn+MLP 16.32 6.63 86.0 0.62 0.66

Table 5. Metrics of 400K training steps on DDT-B/2(8En4De) with different decoder blocks. All results are reported without classifier-free guidance. The Default Attention + MLP configuration achieves best performance.

of the traditional diffusion transformer. By decoupling the low-frequency encoding and high-frequency decoding into dedicated components, we effectively resolved the optimization dilemma that has constrained diffusion transformer. Furthermore, we discovered that increasing the encoder capacity relative to the decoder yields increasingly beneficial results as the overall model scale grows. This insight provides valuable guidance for future model scaling efforts. Our experiments demonstrate that our DDT-XL/2 (22En6De) with an unexpected aggressive encoder-decoder layer ratio achieves great performance while requiring only 256 training epochs. This significant improvement in efficiency addresses one of the primary limitations of diffusion models: their lengthy training requirements. The decoupled architecture also presents opportunities for inference optimization through our proposed encoder result sharing mechanism. Our statistical dynamic programming approach for determining optimal sharing strategies enables faster inference while minimizing quality

degradation, demonstrating that architectural innovations can yield benefits beyond their primary design objectives.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2
- [2] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023. 2, 6
- [3] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018. 1, 7
- [4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 2
- [5] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022. 1
- [6] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-\alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2
- [7] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024. 2
- [8] Xiangxiang Chu, Jianlin Su, Bo Zhang, and Chunhua Shen. Visionllama: A unified llama interface for vision tasks. arXiv preprint arXiv:2403.00522, 2024. 5
- [9] Chaorui Deng, Deyao Zh, Kunchang Li, Shi Guan, and Haoqi Fan. Causal diffusion transformers for generative modeling. arXiv preprint arXiv:2412.12095, 2024. 6, 12
- [10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2, 6, 7, 11
- [11] Sander Dieleman. Diffusion is spectral autoregression, 2024. 3, 4
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. 2, 5
- [13] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Diffusion-rwkv: Scaling rwkvlike architectures for diffusion models. arXiv preprint arXiv:2404.04478, 2024. 2

- [14] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23164–23173,

2023. 3, 6

- [15] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23164–23173,

2023. 3

- [16] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7441–7451, 2023. 2, 6
- [17] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 2
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 5
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [20] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2
- [21] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022. 1
- [22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 11

- [23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 2
- [24] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2
- [25] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 5
- [26] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. arXiv preprint arXiv:2404.07724, 2024. 6

- [27] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation generation method. Advances in Neural Information Processing Systems, 37:125441–125468, 2024. 3
- [28] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2025. 2, 4, 6
- [29] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 1, 3
- [30] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 1, 3
- [31] Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, and Lei Bai. Fit: Flexible vision transformer for diffusion model. arXiv preprint arXiv:2402.12376, 2024. 5
- [32] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740,

2024. 2, 3, 5, 6, 7

- [33] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15762–15772, 2024. 5
- [34] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021. 5
- [35] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4
- [36] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2, 4, 5, 6, 7

- [37] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. arXiv preprint arXiv:2206.13397, 2022. 3, 4
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 6
- [39] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 5
- [40] Axel Sauer, Katja Schwarz, and Andreas Geiger. Styleganxl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022. 1, 7
- [41] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based

- generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1
- [42] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 5

- [43] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1
- [44] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023. 6
- [45] Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Dim: Diffusion mamba for efficient high-resolution image synthesis. arXiv preprint arXiv:2405.14224, 2024. 2
- [46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 5
- [47] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 5
- [48] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Flowdcn: Exploring dcnlike architectures for fast image generation with arbitrary resolution. arXiv preprint arXiv:2410.22655, 2024. 2, 6, 7
- [49] Jing Nathan Yan, Jiatao Gu, and Alexander M Rush. Diffusion models without attention. arXiv preprint arXiv:2311.18257, 2023. 2
- [50] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025. 5, 6
- [51] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Randomized autoregressive visual generation. arXiv preprint arXiv:2411.00776, 2024. 1
- [52] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 2, 3, 4, 5, 6, 7, 12
- [53] Xiaoyu Yue, Zidong Wang, Zeyu Lu, Shuyang Sun, Meng Wei, Wanli Ouyang, Lei Bai, and Luping Zhou. Diffusion models need visual priors for image generation. arXiv preprint arXiv:2410.08531, 2024. 2, 3
- [54] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583,

2024. 2

### A. Model Specs

Config #Layers Hidden dim #Heads

B/2 12 768 12 L/2 24 1024 16 XL/2 28 1152 16

### B. Hyper-parameters

VAE SD-VAE-f8d4-ft-ema VAE donwsample 8

latent channel 4

optimizer AdamW [22] base learning rate 1e-4

weight decay 0.0 batch size 256

learning rate schedule constant

augmentation center crop diffusion sampler Euler-ODE

diffusion steps 250 evaluation suite ADM [10]

#### C. Linear flow and Diffusion Given the SDE forward and reverse process:

dxt = f(t)xtdt + g(t)dw (9) dxt = [f(t)xt − g(t)2∇x log p(xt)]dt + g(t)dw (10)

A corresponding deterministic process exists with trajectories sharing the same marginal probability densities of reverse SDE.

- 1

- 2

g(t)2∇xt

log p(xt)]dt (11)

dxt = [f(t)xt −

Given xt = αtxdata + σϵ. The traditional diffusion model learns:

ϵ σ(t)

(12)

∇xt

log p(xt) = −

The flow-matching framework actually learns the following:

vt = αx˙ + σϵ˙ (13) = x − ϵ (14)

Here we will demonstrate in flow-matching, the vt prediction is actually as same as the reverse ode:

αx˙ + σϵ˙ (15)

- 1

- 2

g(t)2∇xt

log p(xt) (16)

=f(t)xt −

Let us start by expanding the reverse ode first.

- 1

- 2

g(t)2∇xt

log p(xt) (17)

f(t)xt −

- 1

- 2

ϵ σ(t)

g(t)2[−

] (18)

=f(t)(α(t)xdata + σ(t)ϵ) −

g(t)2 σ(t)

- 1

- 2

)ϵ (19)

=f(t)α(t)xdata + (f(t)σ(t) +

To prove Eq. (16), we needs to demonstrate that: α˙(t) = ftα(t) (20) σ˙(t) = ftσ(t) +

gt2 σ(t)

- 1

- 2

. (21)

Here, let us derive the relation between ft and α(t),α˙(t). We donate xdata(t) = α(t)xdata is the remain component of xdata in xt, it is easy to find that:

dxdata(t) = ftxdata(t)dt (22) d(α(t)xdata) = ftα(t)xdatadt (23) dα(t) = ftα(t)dt (24)

So, Eq. (20) is right.

Based on the above equation, we will demonstrate the relation of gt,ft with σ(t). Note that Gaussian noise has nice additive properties.

aϵ1 + bϵ2 ∈ N(0, a2 + b2) (25)

Let us start with the gaussian noise component ϵ(t) calculation, reaching at t, every noise addition at s ∈ [0,t] while

been decayed by a factor of αα((st)). Thus, the mixed Gaussian noise will have a std variance σ(t) of:

t

α(t) α(s)

)2gs2]ds) (26)

σ(t) = (

[(

0

t

gs α(s)

)2]ds) (27)

σ(t) = α(t) (

[(

0

After obtaining the relation of ft,gt and α(t),σ(t), we derive α˙(t) and σ˙(t) with above conditions:

t

fsds] (28) α˙(t) = ftα(t) (29)

α˙(t) = ft exp[

0

As for σ˙(t), it is quit complex but not hard:

gt2 α(t) ( 0 t[( g

- 1

- 2

t

gt α(s)

)2]ds) + α(t)

σ˙(t) = α˙(t) (

[(

α(s))2gs2]ds)

0

t

(30)

gt2 α2(t) ( 0 t[( g

- 1

- 2

t

gt α(s)

)2]ds) + α(t)

σ˙(t) = (ftα(t)) (

[(

α(s))2]ds) (31)

0

t

- 1

- 2gt2

t

gt α(s)

)2]ds) +

σ˙(t) = ftα(t) (

[(

α(t) ( 0 t[( g

α(s))2]ds)

0

t

(32) σ˙(t) = ftσ(t) +

- 1

- 2

gt σ(t)

(33) So, Eq. (21) is right.

### D. Proof of Spectrum Autoregressive

Given the noise scheduler{αt,σt}, the clean data xdata and Gaussian noise ϵ. Denote Kfreq as the maximum frequency of the clean data xdata The noisy latent xt at timestep t has been defined as:

xt = αtxdata + σtϵ (34)

The spectrum magnitude ciof xt on DCT basics ui follows:

ci = Eϵ[uTi xt]2 ci = Eϵ[uTi (αtxdata + σtϵ)]2

Recall that the spectrum magnitude of Gaussian noise ϵ is uniformly distributed.

ci = [αtuTi xdata]2 + 2αtσtEϵ[uTi xdatauTi ϵ] + σt2Eϵ[uTi ϵ]2 ci = [αtuTi xdata]2 + σt2Eϵ[uTi ϵ]2 ci = αt2[uTi xdata]2 + σt2λ

if σt2λ has bigger value than [αtuTi xdata]2, the spectrum magnitude ci on DCT basics ui will be canceled, thus the maximal remaining frequency fmax(t) of original data in xt follows:

2

αtuTi xdata σtλ

,Kfreq (35)

fmax(t) > min

2

T i xdata

Though αtu

depends on the dataset. Here, we directly suppose it as a constant 1. And replace α = t and σ = 1 − t in above equation:

σtλ

2

t 1 − t

,Kfreq (36)

fmax(t) > min

### E. Linear multisteps method

We conduct targeted experiment on SiT-XL/2 with Adams–Bashforth like linear multistep solver; To clarify, we did not employ this powerful solver for our DDT models in all tables across the main paper.

The reverse ode of the diffusion models tackles the following integral:

xi+1 = xi +

ti+1

vθ(xt,t)dt (37)

ti

The classic Euler method employs vθ(xi,ti) as an estimate of vθ(xt,t) throughout the interval [ti,ti+1]

xi+1 = xi + (ti+1 − ti)vθ(xi,ti). (38)

The most classic multi-step solver Adams–Bashforth method (deemed as Adams for brevity) incorporates the Lagrange polynomial to improve the estimation accuracy with previous predictions.

- i
- j=0

i

t − tk tj − tk

)vθ(xj,tj)

vθ(xt,t) =

(

k=0,k̸=j

- i
- j=0

i

ti+1

t − tk tj − tk

xi+1 ≈ xi +

)vθ(xj,tj)dt

(

ti

k=0,k̸=j

i

- i
- j=0

ti+1

t − tk tj − tk

xi+1 ≈ xi +

vθ(xj,tj)

(

)dt

ti

k=0,k̸=j

Note that t ti+1

( ik=0,k̸=j t−t

tj−tk )dt of the Lagrange polynomial can be pre-integrated into a constant coefficient, resulting in only naive summation being required for ODE solving.

k

i

### F. Classifier free guidance.

As classifier-free guidance significantly impacts the performance of diffusion models. Traditional classifier-free guidance improves performance at the cost of decreased diversity. Interval guidance is recently been adopted by REPA[52] and Causalfusion[9], It applies classifier-free guidance only to the high-frequency generation phase to preserve the diversity. We sweep different classifier-free guidance strength with selected intervals. Our DDT-XL/2 achieves the best performance with interval [0.3,1] with a classifer-free guidance of 2. Recall that we donate t = 0 as the pure noise timestep while REPA[52] use t = 1, thus this exactly correspond to the [0,0.7] interval in REPA[52]

###### Classifer-free guidance with intervals

Interval: [0, 1] Interval: [0.2, 1] Interval: [0.3, 1] Interval: [0.35, 1]

- 4

- 5

- 6

- 7

- 8

- 9

FID10K

1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 CFG values

Figure 9. FID10K of DDT-XL/2 with different Classifer free guidance strength and guidance intervals. We sweep different classifier-free guidance strength with selected intervals. Our DDTXL/2 achieves the best performance with interval [0.3, 1] with a classifer-free guidance of 2.

