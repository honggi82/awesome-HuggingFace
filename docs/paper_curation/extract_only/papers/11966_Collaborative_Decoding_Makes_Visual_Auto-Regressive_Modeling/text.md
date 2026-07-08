# arXiv:2411.17787v1[cs.CV]26Nov2024

## Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient

Zigeng Chen , Xinyin Ma , Gongfan Fang , Xinchao Wang* National University of Singapore

{zigeng99, maxinyin, gongfan}@u.nus.edu, xinchao@nus.edu.sg

https://github.com/czg1225/CoDe

### Abstract

Memory: 39GB FID: 1.95

- (a) Original VAR
- (b) VAR with CoDe

All Foward Steps: 3.62s

[Figure 1]

[Figure 2]

In the rapidly advancing field of image generation, Visual Auto-Regressive (VAR) modeling has garnered considerable attention for its innovative next-scale prediction approach. This paradigm offers substantial improvements in efficiency, scalability, and zero-shot generalization. Yet, the inherently coarse-to-fine nature of VAR introduces a prolonged token sequence, leading to prohibitive memory consumption and computational redundancies. To address these bottlenecks, we propose Collaborative Decoding (CoDe), a novel efficient decoding strategy tailored for the VAR framework. CoDe capitalizes on two critical observations: the substantially reduced parameter demands at larger scales and the exclusive generation patterns across different scales. Based on these insights, we partition the multi-scale inference process into a seamless collaboration between a large model and a small model. The large model serves as the ’drafter’, specializing in generating low-frequency content at smaller scales, while the smaller model serves as the ’refiner’, solely focusing on predicting high-frequency details at larger scales. This collaboration yields remarkable efficiency with minimal impact on quality: CoDe achieves a 1.7x speedup, slashes memory usage by around 50%, and preserves image quality with only a negligible FID increase from 1.95 to 1.98. When drafting steps are further decreased, CoDe can achieve an impressive 2.9x acceleration ratio, reaching 41 images/s at 256x256 resolution on a single NVIDIA 4090 GPU, while preserving a commendable FID of 2.27.

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

Refining Steps: 0.71s

Memory: 20GB FID: 1.98

[Figure 13]

Drafting Steps: 1.40s

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

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Figure 1. We partition the next-scale prediction process into the efficient collaboration between large and small VAR models.

markable image quality and promising multi-modal potential [34, 61, 75]. However, the next-token prediction process inherent to conventional AR models requires numerous decoding steps, leading to considerable generation latency. Visual Auto-Regressive (VAR) Modeling [55] replaces the GPT-style next-token prediction with a next-scale prediction strategy. It generates content in a multi-level, coarseto-fine progression, enabling the model to decode multiple tokens in parallel [31, 45, 53, 68, 70], thus considerably reducing the inference steps.

Despite less decoding steps, VAR’s progressive scaling approach significantly increases the overall sequence length. To generate a 16x16 token image, VAR hierarchically decodes up to 680 tokens across 10 scales—2.7 times the sequence length required by conventional AR models. As the VAR model must store the KV cache accumulated from all previous tokens, this prolonged sequence results in a substantial memory overhead, especially in the final scales. During inference, the KV cache consumes approximately 12 times more memory than the forward computa-

### 1. Introduction

The past year has witnessed significant advancements in Auto-Regressive (AR) models for image generation [18, 24, 28, 33, 52, 64], driven by their proven scalability and strong generalization capabilities [1, 9, 16, 56]. Leveraging these strengths, AR approaches have demonstrated re-

*Corresponding author

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

1.7x Speedup

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

0.5x Memory

- Figure 2. Comparison of generation results between original VAR-d30 (up) and our VAR-CoDE (bottom) for ImageNet 256×256. Our method achieves 1.7x speedup (3.62s to 2.11s), and needs only 0.5x memory space (40GB to 20GB), with negligible quality degradation.

method. As shown in Figure 2, compared to the original VAR-d30 model, CoDe achieves a 1.7x speedup with merely 0.5x GPU memory consumption and negligible quality loss (FID [19] slightly increases from 1.95 to 1.98). Notably, our approach can reach up to a 2.9x speedup, generating 41 images (256x256) per second on a single NVIDIA 4090 GPU, while still maintaining an FID of 2.27.

tion itself. For instance, generating images using VAR-d30 with a batch size of 128 demands 70 GB of memory, with 57 GB dedicated solely to KV caching, becoming the primary bottleneck. Moreover, the extended sequence length exacerbates the computational cost of self-attention, given the quadratic growth in attention map calculations.

To address these inefficiencies, we begin by analyzing the specific properties of VAR’s next-scale prediction paradigm. First, we observe that as generated scales increase, the parameter demands for high-quality token generation drop substantially, leading to considerable computational redundancy at most of the tokens in the long sequence. Next, we find that the generation patterns at small and large scales are exclusive, resulting in mutual interference between learning low- and high-frequency modeling capability, thus hindering efficient parameter utilization.

In conclusion, we introduce CoDe, a novel decoding framework for visual auto-regressive modeling that significantly enhances speed and reduces memory overhead with a negligible impact on quality. CoDe divides the hierarchical sequence modeling of VAR into a collaborative process, utilizing a large VAR model and a small VAR model in a progressive partnership. Additionally, we propose specialized fine-tuning to optimize each model for its specific role, effectively mitigating training interference and maximizing parameter utilization. To the best of our knowledge, CoDe is the fastest method available to achieve an FID below 2, making it a clear advancement in efficient image generation.

Our Approach. Inspired by the above observations, we propose Collaborative Decoding (CoDe), a simple yet highly effective method that significantly boosts the inference efficiency of VAR models while preserving generation quality comparable to the original. Our core idea is to decompose the long-sequence scaling-up process into a collaboration between two VAR models with different sizes and specialized roles to enhance efficiency. Figure 1 presents the overview of our method. We use a large VAR model as a drafter for the initial small scales, where model capacity demands are high but computations and KV cache are sparse. Conversely, we use a small VAR model as a refiner at the remaining large scales, where computations and KV cache are intensive but fewer parameters are needed. Next, we apply specialized fine-tuning to both models to address the optimization interference between scales encountered during the pre-training stage. Each model is fine-tuned exclusively on the specific scales it handles, leading to a notable performance boost with limited additional cost.

### 2. Related Works

Auto-regressive image generation. Early works [4, 57] pioneered image generation by generating pixels in rasterscan order. Later, VQVAE [58] and VQGAN [10] improved this approach by quantizing image patches into discrete tokens, using a transformer in a decoder-only setup to generate these tokens in a raster-scan manner. Building on these foundations, recent efforts have focused on enhancing autoregressive (AR) models for image generation. LlamaGEN [52] and Lumina-mGPT [34], for example, use a GPT-style next-token prediction strategy to achieve high-quality image generation with good scalability. AiM [24] and MARS [18] further improve this paradigm by introducing mixtureof-experts and linear attention mechanisms [14]. Methods like SHOW-O [61], Transfusion [75], and DART [15] combine diffusion processes with autoregressive modeling,

Extensive experiments validate the effectiveness of our

[Figure 42]

[Figure 43]

[Figure 44]

(a) Quality vs. Params at different scales (b) Spectrum analysis (c) Training-free model collaboration

- Figure 3. (a) Effectiveness of increasing parameters at the k-th scale is evaluated by predicting token map rk using four VAR models with different parameter sizes (2B, 1B, 0.6B, and 0.3B), while other scales (r1, r2, . . . , rk−1, rk+1, . . . , r10) are generated using the largest VAR-d30 model. (b) Fourier spectrum analysis is conducted on generated content at the first 3 scales and the last 3 scales. (c) Training-free performance comparison of model collaboration decoding across various settings of draft tokens M and refiner tokens 680 − M.

### 3. Method

while [3, 28, 38] introduces masked autoregression to generate images. However, all these approaches suffer from high latency due to the large number of forward steps. VAR [55] addresses this by using hierarchical parallel decoding, which greatly reduces the number of steps, leading to significantly lower inference latency without sacrificing quality. Building on VAR’s progress, many recent works have aimed to improve next-scale prediction across multiple tasks, including text-to-image [30, 41, 53, 70], controllable generation [31, 33], audio generation [45], and 3D generation [68].

#### 3.1. Prelinimary

Visual auto-regressive modeling [55] redefines the traditional AR by shifting from a “next-token” prediction to a “next-scale” prediction. In this framework, each autoregressive unit is a token map at varying scales, rather than a single token. For a given image feature map f ∈ Rh×w×C, VAR quantizes it into K multi-scale token maps R = (r1,r2,...,rK) at progressively finer resolutions, with the final token map rK matching the original feature map’s resolution. The probability distribution is reformulated as:

Efficient Image Generation Models. For diffusion models, acceleration techniques are already well-developed. [37, 47, 48, 65, 66] focuses on reducing sampling steps while [12, 32, 63, 67, 73] optimize model architectures through pruning [5, 11] or knowledge distillation [20]. To avoid the high costs of training, some training-free methods are proposed. Some approaches develop fast solvers for stochastic differential equations (SDE) or ordinary differential equations (ODE) to enhance sampling efficiency [2, 35, 36, 69, 74]. Other works [27, 39, 40, 50, 60, 62, 71, 72] exploit specific characteristics of diffusion models to skip redundant computations during the denoising process. [6, 13, 25] reduce latency to another level through distributed computing. Quantization methods [26, 29, 49] have also shown great potential. In contrast, research into accelerating AR image generation is still in its early stages. LANTERN [21] and SJD [54] employ speculative decoding to speed up next-token prediction, achieving notable speedups but not suitable for the innovative next-scale prediction paradigm introduced by VAR.

K

p(rk | r1,r2,...,rk−1), (1)

p(r1,r2,...,rK) =

k=1

k×wk consists of hk × wk tokens at scale k, and the sequence (r1,r2,...,rk−1) serves as the ”prefix” for rk. In this paradigm, during each autoregressive step k, the model predicts all hk × wk tokens in rk in parallel, conditioned on prior scales and position embeddings. This approach aligns with a coarse-to-fine generation pattern, enabling parallel decoding within each scale. VAR effectively improves inference speed and generation quality, but it also considerably increases sequence length.

where each token map rk ∈ [V ]h

#### 3.2. Key Observations

VAR represents an innovative paradigm specific for visual generation, distinct from traditional autoregressive approaches and introducing many yet unexplored characteristics. In this work, we revisit the entire next-scale prediction process to uncover specific properties that can be optimized to reduce computational redundancy and improve inference efficiency.

In this paper, we introduce CoDe, a novel and efficient decoding method tailored for the next-scale prediction paradigm. CoDe effectively addresses the inefficiencies associated with the long sequence structure of VAR models while maintaining high generation quality. Notably, CoDe stands out as the fastest method capable of achieving an FID below 2, making it a clear advancement.

Observation 1: As the predicted scale becomes larger, the need for parameters reduces significantly. While VAR models exhibit strong scalability, the effectiveness of increasing parameter counts varies greatly in predicting var-

Refine Steps: 589 Tokens

[Figure 45]

[Figure 46]

Draft Steps: 91 Tokens

[Figure 47]

[Figure 48]

Ground Truth Labels rk* (Drafing Steps)

Teacher Logits Pteacher(rk) (Refining Steps)

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

... ...

[Figure 67]

[Figure 68]

CSE loss KL loss

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

... ...

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

Drafter Model Refiner Model

r1

r2 r3 r4 r5 r6 r7 r8 r9 r10

r1 r2 r3 r4 r5 r6 r7 r8 r9 r10

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Refiner VAR Model (0.3B Params)

Drafter VAR Model (2B Params)

Ground Truth Labels rk*

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

C r1 r2 r3 r4 r5 r6 r7 r8 r9

C r1 r2 r3 r4 r5 r6 r7 r8 r9

...

[Figure 135]

[Figure 136]

[Figure 137]

- Figure 4. Overview of the collaborative decoding process, we use a drafting step N = 6 for instance. CoDe uses a large VAR model as

the drafter ϵθd to generate the token maps RL = (r1, r2, . . . , rN) at smaller scales. The small refiner model ϵθr then uses RL as an initial prefix to efficiently predict the remaining token maps RH = (rN+1, rN+2, . . . , rK) at larger scales. Both models are fine-tuned on their designated predictive scales using ground truth labels rk∗ and teacher logits pteacher(rk), respectively.

ious scales. In Figure 3. (a), we present the differences in class-conditional generation (ImageNet-256 [8]) quality when using VAR models of different sizes at each scale. At the initial small scales, increasing model parameters leads to noticeable improvements in generation quality. However, as the predicted scale gradually progresses, the impact of additional parameters becomes minimal. At the final scale—which accounts for 38% tokens of the entire sequence—we observe that the performance of the 2B model is even nearly equivalent to that of the 0.3B model. These findings suggest that as the predicted scale increases, the parameter demand for accurate token predictions declines markedly, revealing substantial computational redundancy in the current VAR inference process at large scales.

all scales results in significant mutual interference between learning low-frequency and high-frequency modeling capabilities, making parameter optimization challenging.

In sum, the key insights from our observations can be concluded as follows: (1) At large scales, VAR models require significantly fewer parameters for accurate token prediction, leading to considerable computational redundancy. (2) Training a single model to generate across all scales causes mutual interference between small and large scales, hindering parameter optimization due to conflicting lowfrequency and high-frequency learning tendencies.

#### 3.3. Collaborative Decoding

Based on the above insights, we propose a simple yet pretty efficient decoding method for next-scale prediction called collaborative decoding. As presented in Figure 4, CoDe decomposes the next-scale prediction process into a collaboration between a large drafter model and a small refiner model. The drafter is the original large VAR model with 2B parameters, while the refiner is a lightweight 0.3B VAR model, sharing the same transformer-based architecture but with reduced width and depth. The drafter is responsible for generating coarse, low-frequency global structures at smaller scales to draft the image, while the refiner continually predicts the high-frequency details at larger scales to refine the image.

Observation 2: The generative patterns are exclusive between small and large scales. For VAR models, the content generated at small and large scales varies significantly. As demonstrated by the Fourier analysis in Figure 3 (b), the feature maps produced by the first three scales contain primarily low-frequency components, while the last three scales focus on high-frequency components. To further validate these distinct generative patterns, we conducted a perturbation fine-tuning experiment. Using a pre-trained VARd16 model, we applied CSE loss only to tokens in the largest three scales and fine-tuned for just 1% of the original training epochs. This minor fine-tuning at large scales led to a complete collapse of the model’s global modeling capacity at small scales, with the FID increasing from 3.30 to 21.93 and the IS score dropping from 277 to 88. These results demonstrate that VAR models perform entirely distinct generative tasks at small and large scales, with minimal overlap. Training a single VAR model to predict across

Model Collaboration. To generate an image im, original VAR model needs to auto-regressively generate K multiscale token maps R = (r1,r2,...,rK) with progressively finer resolutions, where each token map rk ∈ [V ]h

k×wk

represents the token map at scale k. In CoDe, we decompose the auto-regressive steps into N drafting steps and K − N refining steps. In the initial drafting phase where

computation is sparse but additional parameters provide significant benefits, we serve a large VAR model as the drafter ϵθ

, generating the initial set of low-frequency token maps RL = (r1,r2,...,rN) up to scale N, where N < K. These token maps predict the global structure of the image, serving as a prefix for the remaining finer-scale token maps. The drafting process can be represented as:

d

N

pθ

(r1,r2,...,rN) =

d

k=1

(rk | r1,r2,...,rk−1). (2)

pθ

d

After drafting, the KV cache of the large model is released, significantly optimizing memory usage.

In the refining stage where the computation is intensive but the parameter demand is low, we serve a small VAR model as the refiner ϵθ

, employing the drafter’s predictions RL as a prefix and focus on refining the details by generating the remaining high-frequency token maps RH = (rN+1,rN+2,...,rK) up to scale K, where rK matches the resolution of the original feature map. Notably, predicting rN+1 requires an attention mask, as the refiner model lacks the KV cache for the first N scales. The probability distribution of refining steps is formulated as:

r

p(rN+1,rN+2,...,rK | RL)

K

p(rk | RL,rN+1,rN+2,...,rk−1).

=

k=N+1

(3)

Finally, the generated images im are reconstructed from both drafting maps RL and refining maps RH as follows:

##### im = D(Q(RL,RH), (4)

where Q(.) is a residual-style quantization function, and D(.) is a multi-scale VQVAE decoder. Figure 3. (c) demonstrates the remarkable effectiveness of model collaboration under training-free conditions. With a large model as the drafter, the small refiner only results in a slight and gentle increase in FID. Conversely, with a small model as the drafter, even assigning the last 80% tokens to a large refiner fails to improve performance. This also verifies Observation 1: small scales require more model capacity, whereas the parameter demands for large scales are low.

The model collaboration method effectively reduces the computational redundancy of VAR and maintains generation quality comparable to the original model, while significantly offering faster speed and reduced memory usage.

Specialized Fine-Tuning. Given the exclusive generation patterns between drafting scales and refining scales, we propose the specialized fine-tuning to further specialize drafter and refiner models by fine-tuning them solely on their respective predictive scales, thereby avoiding training interference and enhancing generation quality.

, we fine-tune the drafting steps to improve the generation of the initial low-frequency token maps RL. The drafter model is trained using a CSE loss CSE(·,·) between its generated token distribution pθ

For the drafter model ϵθ

d

(rk) and the ground truth labels rk∗, defined as:

d

Ldrafter =

N

(rk),rk∗), (5)

CSE(pθ

d

k=1

which encourages the drafter to closely match the target global structure at each drafting step.

, we adopt a Knowledge Distillation (KD) [20] approach to partially transfer the knowledge from a larger pre-trained VAR model. The KD loss function for epoch ep can be expressed as:

For the refiner model ϵθ

r

K

L(refinerep) =

λep·1[k≤N]+1[k>N] KL(pθr(rk) ∥ pteacher(rk)),

k=1

(6)

where pteacher(·) is the distribution predicted by the larger teacher VAR model, KL(·,·) is the standard KL-divergence loss aligns refiner’s output distribution with that of the teacher model, and λep is a dynamic weighting factor. λt linearly decreases from one to zero from the initial to the end of finetuning, gradually shifting the KD emphasis from all tokens to the specific refining tokens. This dynamic weighting allows the refiner to specialize in refining the details with a smoother and more stable training process.

The specialized fine-tuning of the drafter and refiner models ensures that each model becomes highly proficient in its respective task. By minimizing interference between small and large scales, this approach allows for more sufficient parameter optimization, resulting in enhanced image generation quality with limited training costs.

### 4. Experimental Results

#### 4.1. Experimental Setup.

Implementation Details. We evaluated our method’s effectiveness on the ImageNet1K [8] class-conditional generation benchmarks. The proposed CoDe framework involves collaboration between a large and a small VAR model. Specifically, we use the pre-trained VAR-d30 model as the drafter and the pre-trained VAR-d16 model as the refiner. Each model undergoes specialized fine-tuning focused exclusively on its respective predictive scales. For the drafter, we fine-tune it for 5% of its original training epochs with a base learning rate of 1e-6 and a weight decay of 0.08. The refiner model, on the other hand, is fine-tuned for 25% of its original training epochs with a base learning rate of 1e-5 and no weight decay. Both models are optimized using the AdamW [23] optimizer with a batch size of 1024, achieved through gradient accumulation. The fine-tuning was conducted on 4 NVIDIA L20 GPUs.

[Figure 138]

[Figure 139]

[Figure 140]

2.9x Speedup

4.6x Speedup

(a) Efficiency & Quality trade-off (b) Time cost with different batchsizes (c) Time cost at each scale

- Figure 5. (a) Our CoDe demonstrates the optimal efficiency-quality trade-off among all evaluated methods. (b) Inference latency is measured across varying batch sizes for the original VAR-d30, our CoDe (N=6), and the VQVAE decoder. (c) We analyze the time cost associated with parallel decoding at each scale, showing that the refiner model is significantly more efficient than the drafter at larger scales.

Table 1. Quantitative assessment of the efficiency-quality trade-off across various methods. Inference efficiency is evaluated with a batch size of 64 on NVIDIA L20 GPU, with latency measured excluding VQVAE or VQGAN as it incurs a shared time cost across all methods.

Inference Efficiency Generation Quality #Steps Speedup↑ Latency↓ Throughput↑ #Param Memory↓ FID ↓ IS ↑ Precision↑ Recall↑

Method

DiT-XL/2 50 0.2x 19.20s 3.33it/s 675M 11369MB 2.26 239 0.80 0.60 MAR-B 100 0.1x 29.80s 2.15it/s 208M 8725MB 2.31 282 0.82 0.57 AiM-XL 256 0.4x 9.32s 6.87it/s 763M 20983MB 2.56 257 0.82 0.57

LlamaGen-XXL 384 <0.1x 73.97s 0.87it/s 1.4B 42632MB 2.34 254 0.80 0.59

VAR-d30 10 1.0x 3.62s 17.71it/s 2.0B 39228MB 1.95 301 0.81 0.59 VAR-d24 10 1.7x 2.07s 30.92it/s 1.0B 25093MB 2.11 311 0.82 0.59 VAR-d20 10 2.8x 1.29s 49.62it/s 600M 17814MB 2.61 301 0.83 0.56

VAR-CoDe N=9 9+1 1.2x 2.97s 21.54it/s 2.0+0.3B 28803MB 1.94 296 0.80 0.61 VAR-CoDe N=8 8+2 1.7x 2.11s 30.33it/s 2.0+0.3B 21019MB 1.98 302 0.81 0.60 VAR-CoDe N=7 7+3 2.3x 1.60s 40.00it/s 2.0+0.3B 19943MB 2.11 303 0.82 0.59 VAR-CoDe N=6 6+4 2.9x 1.27s 50.39it/s 2.0+0.3B 19943MB 2.27 297 0.82 0.58

Table 2. Effect of specialized fine-tuning

###### Discription N=6 N=7 N=8 N=9

CoDe Training-free 2.42 2.26 2.10 1.99 CoDe Fine-tuning 2.27 2.11 1.98 1.94

Evaluations. We evaluated our method on ImageNet1K256 generation, focusing on both quality and efficiency metrics. For quality assessment, we used standard metrics such as FID, Inception Score (IS), Precision, and Recall. For other baselines, we use their default sampling methods. For CoDe, we reduced the sampled top-k from 900 to 600 due to improved token prediction accuracy after fine-tuning, while keeping the default top-p at 0.96. To compensate for reduced diversity, we introduced a temperature coefficient of t = 1.1 for sampling on the smallest 7 scales. Efficiency was measured through inference latency, throughput, memory consumption, and parameter count to provide a comprehensive comparison. Notably, the speed measurements

exclude the VQGAN decoder, which contributes minimally to overall runtime and is a shared component across all methods. All efficiency tests were conducted on a single NVIDIA L20 GPU without additional optimizations such as FlashAttention [7], using PyTorch 2.1 [43] and FP16.

#### 4.2. Main Results.

Quality-Efficiency Trade-off. We evaluated the qualityefficiency trade-off of our proposed method (CoDe) against original VAR models [55], state-of-the-art AR models, and diffusion models. The AR models included GPT-style LlamaGEN [52], Mamba-based Aim [24], and MAE-style MAR [28]. For diffusion models, we employed the widely used DiT models [44]. To measure the trade-offs, we used series models with different parameter counts for VAR, LlamaGEN, Aim, and MAR, while for DiT we varied the DDIM [51] steps to generate the efficiency-quality curve.

As illustrated in Table 1, We analyzed CoDe’s performance under different drafting steps N. Compared to traditional AR models, our proposed VAR-CoDe (N=6) achieves

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Original VAR

- 1.0x Speedup

100% Memory

CoDe N=7

- 2.3x Speedup 51% Memory

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

CoDe N=8 1.7x Speedup 54% Memory

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

CoDe N=6 2.9x Speedup 51% Memory

Figure 6. Qualitative comparison between the original VAR-d30 model and our proposed CoDe model, with different drafting steps.

sults demonstrate the superiority of our approach over the conventional VAR paradigm and highlight the gains from specialized fine-tuning for optimal parameter utilization

60 times faster inference than LlamaGEN-XXL, while also surpassing its generation quality. Additionally, in comparison to diffusion models, VAR-CoDe (N=6) is 15 times faster than DiT-XL/2, while maintaining the same level of quality. Compared to the original VAR-d30, CoDe (N=8) achieves a 1.7x speedup and reduces memory consumption by 50%, with only a negligible FID increase from 1.95 to 1.98. When the drafting stage is reduced to just 6 steps, CoDe (N=6) achieves a 2.9x speedup, reaching a throughput of over 50it/s, while maintaining a low FID of 2.27. This is a speed unmatched by any other existing methods. Compared to other VAR models with fewer parameters (VARd24 and VAR-d20), our method achieves significantly better quality while maintaining the same speedup ratio.

Qualitative Results. We provide an extensive qualitative comparison between the original VAR-d30 model and our proposed CoDe, with varying drafting steps N = {6,7,8}. As illustrated in Figure 6, our approach achieves significant speedup and substantial memory optimization, with only minimal quality degradation that is nearly imperceptible to the human eye. Even at a speedup rate of 2.9 times, the generated images maintain exceptionally high quality and accurate semantic information. It is essential to note that the objective of CoDe is to enhance the efficiency of VAR’s inference process while preserving generation quality, rather than mirroring the exact outputs of the original model. Through specialized fine-tuning, CoDe’s draft model exhibits higher predictive accuracy compared to the original model. This sometimes results in a different global structure from the original output, yet the image quality remains consistently high or even better.

As shown in Figure 5 (a), VAR-CoDe achieves the best efficiency-quality trade-off compared to all other methods, effectively solving the inefficiencies introduced by the extended sequence length in the original VAR paradigm.

Training-Free vs Specialized Finetuning. Our CoDe framework utilizes a large drafter model alongside a smaller refiner model for progressive inference. It can operate in a training-free manner by directly using pre-trained VAR-d30 and VAR-d16 models as the drafter and refiner, respectively. Alternatively, we can perform specialized fine-tuning to further enhance the models’ performance at their respective scales, with limited training cost. Table 2 compares the results of training-free CoDe with the specialized fine-tuned version. Even without additional training, CoDe demonstrates competitive performance, outperforming VAR-d24 and VAR-d20 models with the same speedup ratio. With specialized fine-tuning, CoDe’s performance improves significantly, even achieving a slight quality enhancement over the original model at a 1.2x acceleration ratio. These re-

Zero-shot Task Generalization. To evaluate the zero-shot generalization capability of CoDe, we conducted additional experiments on zero-shot class-conditional inpainting and image editing. During image inpainting, we applied teacher forcing by providing the ground truth tokens outside the masked area, allowing the model to generate tokens solely within the mask. Notably, class-conditional information was introduced to the model. In the image editing task, CoDe was restricted to generating tokens within a given bounding box based on a specific class label. Figure 7 illustrates the qualitative results of CoDe’s zero-shot performance (N=8). Our approach demonstrates strong zeroshot generalization without any additional training on these

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

painting

painting

###### Inpainting

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

painting painting

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Editing

Editing

Image

Editing

Figure 7. Qualitative results of CoDe’s zero-shot generalization on image inpainting and image editing.

Table 3. Memory usage comparison across different batch sizes

across different batch sizes. As shown in Figure 5.(b), CoDe achieves a speedup of 1.6x for batch size 1 but reaches up to 2.9x for batch size 64. The reduced speedup at smaller batch sizes is due to low GPU utilization when generating 256x256 images, which limits the acceleration potential. However, as GPU utilization increases with larger batch sizes, the speedup of CoDe also improves significantly. This indicates that CoDe is well-suited for computationally intensive image generation tasks, and holds great potential for enhancing efficiency in high-resolution image generation under the VAR paradigm. Additionally, we report the time cost of the VQVAE decoder in the figure, which is a constant shared cost across all methods.

Memory Consumption↓ Running KV Cache Params Total

Method

VAR (bs=8) 314MB 3595MB 8089MB 12002MB

+CoDe 284MB 1023MB 9275MB 10619MB VAR (bs=16) 615MB 7191MB 8089MB 15901MB

+CoDe 557MB 2056MB 9275MB 11951MB VAR (bs=32) 1216MB 14345MB 8089MB 23662MB

+CoDe 1103MB 4083MB 9275MB 14614MB VAR (bs=64) 2420MB 28707MB 8089MB 39228MB

###### +CoDe 2195MB 8160MB 9275MB 19943MB

VAR (bs=128) OOM(0.48GB) OOM(57GB) OOM(0.80GB) OOM(70GB)

###### +CoDe 4380MB 16320MB 9275MB 30598MB

Memory Consumption Analysis. In Table 3, we provide a detailed analysis of memory usage during the VAR inference process. The KV cache of the VAR model is the largest memory consumer, requiring 12 times the memory needed for the model’s decoding operation due to the significantly extended sequence length in the VAR paradigm. Our proposed CoDe effectively addresses this KV cache memory challenge. Specifically, we use the large VAR model only for predicting the first 91 tokens, which represent just 13% of the total sequence. After this, the KV cache in the drafter is released, and the remaining computation is handled by a smaller model. This approach drastically reduces the KV cache memory requirements, compressing it to approximately 28% of the original VAR-d30. Although the refiner model adds a small number of additional parameters, this overhead is minimal compared to the major optimization achieved in KV cache storage. As a result, the total memory usage of CoDe is significantly lower than the original model. Moreover, as batch size increases, the memory savings with CoDe become even more pronounced, underscoring its potential to support efficient high-resolution image generation within the VAR paradigm.

downstream tasks.

#### 4.3. Efficiency Analysis.

Time Cost Analysis. We first present the time cost (bs=64) for each decoding step of the VAR-d30 model in Figure 5 (c). Notably, the computational complexity across the ten decoding steps of VAR is highly non-uniform. The last three decoding steps alone account for 64% of the total inference time, with latency increasing quadratically as token map resolution grows. This highlights that addressing the efficiency bottleneck of large-scale predictions is crucial to enhance inference efficiency. Our solution is to replace the large VAR model with a smaller one for the last few steps, as fewer parameters are needed for large-scale token maps. This small refiner is substantially faster than the original large drafter at these scales, achieving a 4.6x speed improvement in the final and most computationally demanding step. This change dramatically accelerates the entire inference process compared to the original VAR model.

Next, we present the speedup effect of CoDe (N=6)

### 5. Conclusion

This work presents CoDe, a novel method designed for efficient decoding in visual auto-regressive modeling. CoDe effectively mitigates the significant memory overhead and computational redundancy typically associated with the prolonged sequences of next-scale predictions. Through extensive experimentation, our method demonstrates a superior efficiency-quality trade-off, establishing a new benchmark for efficient, high-quality image generation.

### References

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [2] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analyticdpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. arXiv preprint arXiv:2201.06503, 2022. 3
- [3] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022. 3
- [4] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020. 2
- [5] Zigeng Chen, Gongfan Fang, Xinyin Ma, and Xinchao Wang. 0.1% data makes segment anything slim. arXiv preprint arXiv:2312.05284, 2023. 3
- [6] Zigeng Chen, Xinyin Ma, Gongfan Fang, Zhenxiong Tan, and Xinchao Wang. Asyncdiff: Parallelizing diffusion models by asynchronous denoising. arXiv preprint arXiv:2406.06911, 2024. 3
- [7] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022. 6
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 4, 5, 1
- [9] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1

- [10] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 2
- [11] Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, and Xinchao Wang. Depgraph: Towards any structural pruning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16091–16101, 2023. 3

- [12] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models. Advances in neural information processing systems, 36, 2024. 3
- [13] Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, and Jiannan Wang. xdit: an inference engine for diffusion transformers (dits) with massive parallelism. arXiv preprint arXiv:2411.01738, 2024. 3
- [14] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 2
- [15] Jiatao Gu, Yuyang Wang, Yizhe Zhang, Qihang Zhang, Dinghuai Zhang, Navdeep Jaitly, Josh Susskind, and Shuangfei Zhai. Dart: Denoising autoregressive transformer for scalable text-to-image generation. arXiv preprint arXiv:2410.08159, 2024. 2
- [16] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024. 1
- [17] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. A survey on vision transformer. IEEE transactions on pattern analysis and machine intelligence, 45(1):87–110, 2022. 2
- [18] Wanggui He, Siming Fu, Mushui Liu, Xierui Wang, Wenyi Xiao, Fangxun Shu, Yi Wang, Lei Zhang, Zhelun Yu, Haoyuan Li, et al. Mars: Mixture of auto-regressive models for fine-grained text-to-image synthesis. arXiv preprint arXiv:2407.07614, 2024. 1, 2
- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 2, 1
- [20] Geoffrey Hinton. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 3, 5
- [21] Doohyuk Jang, Sihwan Park, June Yong Yang, Yeonsung Jung, Jihun Yun, Souvik Kundu, Sung-Yub Kim, and Eunho Yang. Lantern: Accelerating visual autoregressive models with relaxed speculative decoding. arXiv preprint arXiv:2410.03355, 2024. 3
- [22] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 1
- [23] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 5
- [24] Haopeng Li, Jinyue Yang, Kexin Wang, Xuerui Qiu, Yuhong Chou, Xin Li, and Guoqi Li. Scalable autoregressive image generation with mamba. arXiv preprint arXiv:2408.12245,

2024. 1, 2, 6

- [25] Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Kai Li, and Song Han. Distrifusion: Distributed parallel inference for high-resolution diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7183– 7193, 2024. 3

- [26] Muyang Li*, Yujun Lin*, Zhekai Zhang*, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by lowrank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024. 3
- [27] Senmao Li, Taihang Hu, Fahad Shahbaz Khan, Linxuan Li, Shiqi Yang, Yaxing Wang, Ming-Ming Cheng, and Jian Yang. Faster diffusion: Rethinking the role of unet encoder in diffusion models. arXiv preprint arXiv:2312.09608, 2023. 3
- [28] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024. 1, 3, 6
- [29] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17535–17545, 2023. 3
- [30] Xiang Li, Hao Chen, Kai Qiu, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024. 3
- [31] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Zhe Lin, Rita Singh, and Bhiksha Raj. Controlvar: Exploring controllable visual autoregressive modeling. arXiv preprint arXiv:2406.09750, 2024. 1, 3
- [32] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36, 2024. 3
- [33] Zongming Li, Tianheng Cheng, Shoufa Chen, Peize Sun, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Controlar: Controllable image generation with autoregressive models. arXiv preprint arXiv:2410.02705, 2024. 1, 3
- [34] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 1, 2
- [35] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. arXiv preprint arXiv:2202.09778, 2022. 3
- [36] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787,

2022. 3

- [37] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 3
- [38] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual gener-

ation. arXiv preprint arXiv:2409.04410, 2024. 3

- [39] Zhaoyang Lyu, Xudong Xu, Ceyuan Yang, Dahua Lin, and Bo Dai. Accelerating diffusion models via early stop of the diffusion process. arXiv preprint arXiv:2205.12524, 2022. 3
- [40] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. arXiv preprint arXiv:2312.00858, 2023. 3
- [41] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, and Yi Jin. Star: Scale-wise text-toimage generation via auto-regressive representations. arXiv preprint arXiv:2406.10797, 2024. 3
- [42] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012. 1
- [43] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 6
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 6

- [45] Kai Qiu, Xiang Li, Hao Chen, Jie Sun, Jinglu Wang, Zhe Lin, Marios Savvides, and Bhiksha Raj. Efficient autoregressive audio modeling via next-scale prediction. arXiv preprint arXiv:2408.09027, 2024. 1, 3
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2
- [47] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 3
- [48] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023. 3
- [49] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1972–1981, 2023. 3
- [50] Junhyuk So, Jungwon Lee, and Eunhyeok Park. Frdiff: Feature reuse for exquisite zero-shot acceleration of diffusion models. arXiv preprint arXiv:2312.03517, 2023. 3
- [51] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 6
- [52] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1, 2, 6
- [53] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint arXiv:2410.10812,

2024. 1, 3

- [54] Yao Teng, Han Shi, Xian Liu, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Accelerating autoregressive text-to-image generation with training-free speculative jacobi decoding. arXiv preprint arXiv:2410.01699,

2024. 3

- [55] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. 1, 3, 6
- [56] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1
- [57] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016. 2
- [58] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2
- [59] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023. 1
- [60] Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al. Cache me if you can: Accelerating diffusion models through block caching. arXiv preprint arXiv:2312.03209, 2023. 3
- [61] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 1, 2
- [62] Xingyi Yang and Xinchao Wang. Hash3d: Trainingfree acceleration for 3d generation. arXiv preprint arXiv:2404.06091, 2024. 3
- [63] Xingyi Yang, Daquan Zhou, Jiashi Feng, and Xinchao Wang. Diffusion probabilistic model made slim. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22552–22562, 2023. 3
- [64] Ziyu Yao, Jialin Li, Yifeng Zhou, Yong Liu, Xi Jiang, Chengjie Wang, Feng Zheng, Yuexian Zou, and Lei Li. Car: Controllable autoregressive modeling for visual generation. arXiv preprint arXiv:2410.04671, 2024. 1
- [65] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. arXiv preprint arXiv:2311.18828, 2023. 3
- [66] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image superresolution by residual shifting. Advances in Neural Information Processing Systems, 36, 2024. 3
- [67] Dingkun Zhang, Sijia Li, Chen Chen, Qingsong Xie, and Haonan Lu. Laptop-diff: Layer pruning and normalized distillation for compressing diffusion models. arXiv preprint arXiv:2404.11098, 2024. 3

- [68] Jinzhi Zhang, Feng Xiong, and Mu Xu. G3pt: Unleash the power of autoregressive modeling in 3d generation via cross-scale querying transformer. arXiv preprint arXiv:2409.06322, 2024. 1, 3
- [69] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022. 3
- [70] Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, and Xingyu Ren. Var-clip: Text-to-image generator with visual auto-regressive modeling. arXiv preprint arXiv:2408.01181, 2024. 1, 3
- [71] Wentian Zhang, Haozhe Liu, Jinheng Xie, Francesco Faccio, Mike Zheng Shou, and J¨urgen Schmidhuber. Crossattention makes inference cumbersome in text-to-image diffusion models. arXiv preprint arXiv:2404.02747, 2024. 3
- [72] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024. 3
- [73] Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-image generation on mobile devices. arXiv preprint arXiv:2311.16567, 2023. 3
- [74] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpmsolver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36, 2024. 3
- [75] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 1, 2

## Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient Supplementary Material

In this document, we provide supplementary materials that extend beyond the scope of the main manuscript, constrained by space limitations. These additional materials include:

- • We provide more quantitative analysis results to further illustrate our approach;
- • We offer more qualitative comparisons for visualization;
- • We discuss the limitations of our approach and look into future work.

### A. Additional Quantitative Results

In this section, we present additional quantitative analyses to further substantiate our approach.

Impact of Increasing Model Parameters. To validate Observation 1, we analyze the effect of varying model sizes on class-conditional image generation quality using ImageNet-256 [8]. Specifically, we evaluate the impact of model size at the k-th scale by predicting the token map rk with four Visual Autoregressive (VAR) models [55] of different parameter sizes (2B, 1B, 0.6B, and 0.3B). For all other scales (r1,r2,...,rk−1,rk+1,...,r10), the largest VAR-d30 model is used for generation. Detailed quantitative results are summarized in Table 4. Our results reveal that increasing model parameters at the earlier scales yields significant improvements in generation quality. However, as the scales progress, the marginal benefits of larger models diminish. At the final scale—responsible for 38% of the sequence tokens—we observe that the performance of the 2B model is nearly identical to that of the 0.3B model. This indicates that as the predicted scale increases, the demand for model parameters to ensure accurate token predictions decreases substantially. These findings highlight significant computational redundancy in the current VAR inference process at larger scales.

Training-Free Performance of CoDe. The proposed CoDe framework employs a large drafter model in conjunction with a smaller refiner model for progressive inference. Notably, it can operate in a training-free manner by leveraging pre-trained VAR-d30 and VAR-d16 models as the drafter and refiner, respectively. Table 6 presents the performance of training-free CoDe across various drafting step settings N = {1,2,3,4,5,6,7,8,9}. Even without additional training, CoDe achieves competitive performance, surpassing the VAR-d24 and VAR-d20 models while maintaining the same speedup ratio.

Image Quality Assessment. In our paper, we use standard metrics such as FID [19], Inception Score (IS), Precision, and Recall to evaluate the generation quality. In or-

Table 4. Impact of increasing parameters across scales

Scale Params FID ↓ IS ↑ Precision↑ Recall↑

- 2 0.3B 2.23 291 0.8122 0.5895

- 2 0.6B 2.13 292 0.8078 0.5947

- 2 1.0B 2.04 295 0.8107 0.6027

- 2 2.0B 1.95 301 0.8107 0.5945

- 3 0.3B 2.35 283 0.8064 0.5864

- 3 0.6B 2.21 290 0.8047 0.5967

- 3 1.0B 2.09 295 0.8074 0.5940

- 3 2.0B 1.95 301 0.8107 0.5945

- 4 0.3B 2.27 290 0.8086 0.5953

- 4 0.6B 2.18 293 0.8068 0.5924

- 4 1.0B 2.13 296 0.8061 0.5983

- 4 2.0B 1.95 301 0.8107 0.5945

- 5 0.3B 2.17 296 0.8119 0.5936

- 5 0.6B 2.13 298 0.8087 0.5948

- 5 1.0B 2.10 301 0.8087 0.6025

- 5 2.0B 1.95 301 0.8107 0.5945

- 6 0.3B 2.09 301 0.8119 0.5984

- 6 0.6B 2.05 304 0.8100 0.5976

- 6 1.0B 2.05 305 0.8089 0.5999

- 6 2.0B 1.95 301 0.8107 0.5945

- 7 0.3B 2.09 302 0.8067 0.6010

- 7 0.6B 2.05 305 0.5095 0.6061

- 7 1.0B 2.04 307 0.8077 0.6008

- 7 2.0B 1.95 301 0.8107 0.5945

- 8 0.3B 2.08 304 0.8135 0.5978

- 8 0.6B 2.04 308 0.8110 0.6024

- 8 1.0B 2.02 307 0.8094 0.6038

- 8 2.0B 1.95 301 0.8107 0.5945

- 9 0.3B 2.02 304 0.8133 0.6059

- 9 0.6B 2.01 307 0.8121 0.5948

- 9 1.0B 2.00 307 0.8097 0.6011

- 9 2.0B 1.95 301 0.8107 0.5945

- 10 0.3B 1.99 306 0.8120 0.5978

- 10 0.6B 1.97 305 0.8102 0.6053

- 10 1.0B 1.98 303 0.8102 0.6053

- 10 2.0B 1.95 301 0.8107 0.5945

der to more comprehensively evaluate the quality of generated images, we introduced three image quality assessment (IQA) metrics, including MUSIQ [22], CLIPIQA [59], and NIQE [42]. MUSIQ, CLIPIQA, and NIQE are three distinct

[Figure 201]

[Figure 202]

Figure 8. Up: images generated by the original VAR-d16 models. Down: images generated by the perturbation fine-tuned VAR-d16.

Table 5. No reference metrics for additional image quality assessments.

Inference Efficiency Image Quality Assessment

Method

#Steps Speedup↑ Latency↓ Throughput↑ #Param Memory↓ MUSIQ ↑ CLIPIQA ↑ NIQE↓ VAR-d30 10 1.0x 3.62s 17.71it/s 2.0B 40414MB 60.72 0.6813 6.1739

VAR-CoDe N=9 9+1 1.2x 2.97s 21.54it/s 2.0+0.3B 28803MB 60.78 0.6818 6.1024 VAR-CoDe N=8 8+2 1.7x 2.11s 30.33it/s 2.0+0.3B 21019MB 60.79 0.6812 6.0849 VAR-CoDe N=7 7+3 2.3x 1.60s 40.00it/s 2.0+0.3B 19943MB 60.82 0.6800 6.1247 VAR-CoDe N=6 6+4 2.9x 1.27s 50.39it/s 2.0+0.3B 19943MB 60.76 0.6808 6.1490

Table 6. The training-free performance of CoDe

natural image properties. While MUSIQ and CLIPIQA excel in leveraging learned features for state-of-the-art performance, NIQE stands out for its simplicity, computational efficiency, and independence from reference images, though it may struggle with unnatural or heavily edited content. Together, these metrics cater to diverse IQA needs, from deeplearning-based evaluations to lightweight statistical assessments. As shown in Table 5, our CoDe method achieves comparable or even superior generation quality compared to the original VAR-d30. This result further demonstrates the effectiveness of our approach.

###### Configuration FID ↓ IS ↑ Precision↑ Recall↑

CoDe N=9 1.99 306 0.8120 0.5978 CoDe N=8 2.10 308 0.8155 0.5915 CoDe N=7 2.25 309 0.8204 0.5781 CoDe N=6 2.42 306 0.8283 0.5721 CoDe N=5 2.56 303 0.8313 0.5660 CoDe N=4 2.75 295 0.8342 0.5427 CoDe N=3 2.99 288 0.8410 0.5327 CoDe N=2 3.19 283 0.8433 0.5179 CoDe N=1 3.39 268 0.8132 0.5382

### B. More Qualitative Results.

IQA metrics, each with unique approaches and strengths. MUSIQ (Multi-Scale Image Quality) leverages a vision transformer (ViT) [17] and a multi-scale representation to evaluate global aesthetics and local distortions, making it effective for diverse image types, including high-resolution and non-standard aspect ratios. CLIPIQA utilizes the pretrained CLIP [46] model, which combines semantic understanding from large-scale image-text training to assess image quality in a context-aware manner, excelling in tasks aligned with human perception. In contrast, NIQE (Natural Image Quality Evaluator) is a no-reference metric that models natural scene statistics (NSS) using a multivariate Gaussian distribution to measure deviations from high-quality

Additional Qualitative Comparisons. We provide additional qualitative comparisons between the original VARd30 model and our proposed CoDe framework, evaluated with varying drafting steps N = {6,7,8,9}. As shown in Figures 9 and 10, CoDe achieves significant speedup and substantial memory optimization, with only minimal quality degradation that is nearly imperceptible to the human eye. Even at a speedup rate of 2.9×, the generated images maintain exceptionally high quality and preserve accurate semantic information. It is important to emphasize that the primary goal of CoDe is to enhance the efficiency of the VAR inference process while maintaining high generation quality, rather than reproducing the exact outputs of the original model. Through specialized fine-tuning,

CoDe’s drafter model demonstrates superior predictive accuracy compared to the original model, sometimes resulting in a different global structure. Nevertheless, the image quality remains consistently high and, in some cases, even improves over the original outputs.

Qualitative Results of Perturbation Fine-Tuning. In our study, we conducted a perturbation fine-tuning experiment to examine the distinct generative roles of small and large scales. Using a pre-trained VAR-d16 model, we applied the CSE loss exclusively to tokens in the largest three scales and fine-tuned the model for just 1% of the original training epochs. This minimal fine-tuning at large scales caused a complete collapse of the model’s global modeling capacity at small scales, with the FID increasing from 3.30 to 21.93 and the IS score dropping from 277 to 88. Figure 8 illustrates the qualitative results of perturbation fine-tuning. After slight fine-tuning, the VAR-d16 model nearly loses its ability to model global structures. These findings underscore that VAR models undertake entirely distinct generative tasks at small and large scales, with minimal overlap in functionality.

### C. Limitations and Future Work

Limitations. The core concept of CoDe involves decomposing the next-scale prediction process into a collaboration between a large model and a small model. This approach necessitates the availability of two models with different sizes. If only a single large VAR model is available and faster inference is desired, it becomes necessary to retrain a smaller refiner model. However, since the refiner model can be extremely compact, techniques such as model pruning and knowledge distillation can be applied to limit the additional training cost.

Future Work. This study demonstrates that CoDe significantly reduces inference latency and memory consumption for VAR models. Furthermore, the efficiency gains from CoDe become even more pronounced in computationally intensive scenarios. As a result, CoDe is particularly wellsuited for high-resolution image generation tasks based on next-scale prediction. In future work, we aim to explore the application of CoDe in building an efficient VAR model specifically optimized for high-resolution image generation.

Original VAR-d30 1.0x Speedup Throughput: 17.71it/s Memory: 40414MB FID: 1.95

[Figure 203]

VAR-CoDe N=9 1.2x Speedup Throughput: 21.54it/s Memory: 28803MB FID: 1.94

[Figure 204]

VAR-CoDe N=8 1.7x Speedup Throughput: 30.33it/s Memory: 21019MB FID: 1.98

[Figure 205]

VAR-CoDe N=7 2.3x Speedup Throughput: 40.00it/s Memory: 19943MB FID: 2.11

[Figure 206]

- VAR-CoDe N=6 2.9x Speedup Throughput: 50.39it/s Memory: 19943MB FID: 2.27

[Figure 207]

Original VAR-d30 1.0x Speedup Throughput: 17.71it/s Memory: 40414MB FID: 1.95

[Figure 208]

VAR-CoDe N=9 1.2x Speedup Throughput: 21.54it/s Memory: 28803MB FID: 1.94

[Figure 209]

VAR-CoDe N=8 1.7x Speedup Throughput: 30.33it/s Memory: 21019MB FID: 1.98

[Figure 210]

- VAR-CoDe N=7 2.3x Speedup Throughput: 40.00it/s Memory: 19943MB FID: 2.11

[Figure 211]

VAR-CoDe N=6 2.9x Speedup Throughput: 50.39it/s Memory: 19943MB FID: 2.27

[Figure 212]

