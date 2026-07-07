# arXiv:2605.18739v2[cs.CV]19May2026

[Figure 1]

2026-5-20

## LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation

Yukang Chen*† Luozhou Wang* Wei Huang* Shuai Yang* Bohan Zhang Yicheng Xiao Ruihang Chu Weian Mao Qixin Hu Shaoteng Liu Yuyang Zhao Huizi Mao Ying-Cong Chen Enze Xie Xiaojuan Qi Song Han NVIDIA github.com/NVlabs/LongLive

Abstract: We present LongLive-2.0, an NVFP4-based parallel infrastructure throughout the full training and inference workflow of long video generation, addressing speed and memory bottlenecks. (1) For training, we introduce sequence-parallel autoregressive (AR) training, instantiated as Balanced SP, which co-designs the efficient teacher-forcing layout with SP execution by pairing clean-history and noisy-target temporal chunks on each rank, enabling a natural teacher-forcing mask with SP-aware chunked VAE encoding. Combined with NVFP4 precision, it reduces GPU memory cost and accelerates GEMM computation during training, the proportion of which increases as video length grows. Moreover, we show that a high-quality infrastructure and dataset enable a remarkably clean training pipeline. Unlike existing Self-Forcing series methods that rely on ODE initialization and subsequent distribution matching distillation (DMD), LongLive-2.0 directly tunes a diffusion model into a long, multi-shot, interactive auto-regressive (AR) diffusion model. It can be further converted to real-time generation (4 to 2 denoising steps) with standalone LoRA weights. (2) For inference on Blackwell GPUs, we enable W4A4 NVFP4 inference, quantize KV cache into NVFP4 for memory savings, and boost end-to-end throughput with asynchronous streaming VAE decoding. On non-Blackwell GPU architectures, we deploy SP inference to match the speed on Blackwell GPUs, while the quantized KV cache can lower inter-GPU communication of SP. Experiments show up to 2.15× speedup in training, and 1.84× in inference. LongLive-2.0-5B achieves 45.7 FPS inference while attaining strong performance on benchmarks. To our knowledge, LongLive-2.0 is the first NVFP4 training and inference system for long video generation.

[Figure 2]

[Figure 3]

[Figure 4]

|[Figure 5]<br><br>[Figure 6]|
|---|

|[Figure 7]<br><br>[Figure 8]|
|---|

|[Figure 9]<br><br>[Figure 10]|
|---|

|[Figure 11]<br><br>[Figure 12]|
|---|

|[Figure 13]<br><br>[Figure 14]|
|---|

BF16

|[Figure 15]<br><br>[Figure 16]|
|---|

|[Figure 17]<br><br>[Figure 18]|
|---|

|[Figure 19]<br><br>[Figure 20]|
|---|

|[Figure 21]<br><br>[Figure 22]|
|---|

|[Figure 23]<br><br>[Figure 24]|
|---|

NVFP4

Figure 1 | LongLive 2.0 supports NVFP4-based multi-shot long-video generation for both training and inference. Representative frames from five shots generated with BF16 and NVFP4 (Left): NVFP4 preserves the overall scene composition, subject structure, and shot-level semantics of the BF16 baseline. Note that LongLive 2.0 allows flexible customization of the duration for each shot. Efficiency and memory comparison (Right): NVFP4 achieves 2.15× faster training and 1.84× faster inference, reducing training latency from 1372.9 ms per iteration to 639.5 ms per iteration and inference latency from 40.3 to 21.9 ms/frame, i.e., 45.7 FPS, while reducing memory usage from 35.4 GB to 19.4 GB.

### 1. Introduction

Long video generation suffers from excessive GPU memory consumption and low computational efficiency in both training and inference. For training, a high-quality long video model requires extensive training over massive longvideo datasets, leading to prohibitively high computational costs. For inference, long video models are commonly required in interactive and real-time applications that demand strict low latency; yet, the video length poses severe challenges to deployment. Existing works on long video generation primarily focus on algorithmic designs, while largely neglecting infrastructure optimizations for training, inference, and real-world deployment.

Existing works on long video generation still have notable limitations. At the infrastructure level, few works explore joint co-design between training and inference. For inference, quantization-based methods only adopt post-training quantization (PTQ) [73, 74, 75], leading to misalignment between training and inference with suboptimal performance. At the algorithm level, prevailing training pipelines such as Self-Forcing [26] and Causal-Forcing [82] are overly complicated. Long-video diffusion training typically requires ODE initialization, distribution matching distillation (DMD), and subsequent long tuning in a multistage manner.

In this work, we present LongLive-2.0, an NVFP4-based parallel infrastructure for long video generation training

© 2026 NVIDIA. All rights reserved. * Equal contribution † Project Lead

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

+ Model

AR Training

[Figure 29]

[Figure 30]

✅ AR ✅ Long ✅ Multi-shot

✅ AR ✅ Long ✅ Multi-shot ✅ Real-time

NVFP4 training Balanced SP

W4 GPU

W4 GPU

NVFP4 W4A4 & KV Parallel Dequant

[Figure 31]

[Figure 32]

⫸⫸

⫸⫸

Diffusion Model

[Figure 33]

[Figure 34]

[Figure 35]

Async Decode

LoRA

[Figure 36]

! Step-distill

[Figure 37]

!

VAE

Few-step

Training Infra

Inference Infra

- Figure 2 | Overview of the LongLive-2.0 Framework. Training Infra (Left): The diffusion model is fine-tuned via AR training on long videos, where Balanced SP and NVFP4 quantization improve training efficiency. In parallel, we derive standalone LoRA weights via DMD training. Inference Infra (Right): Full NVFP4 enables low-precision inference (W4A4) and KV-cache compression. Furthermore, asynchronous decoding eliminates idle time to maximize generation throughput.

and inference, as shown in Figure 2. On the training side, we introduce sequence-parallel AR training to scale AR training for long videos, with Balanced SP as the current instantiation. Unlike traditional SP, which treats the clean-context and noisy-target latent streams as an ordinary concatenated sequence, Balanced SP assigns each GPU the clean and noisy latents from the same temporal chunk. This paired layout balances loss-bearing tokens across GPUs and enables a natural teacher-forcing [81] attention mask after Ulysses All-to-All communication. Balanced SP also allows SP-aware chunked VAE encoding so that latent preparation is partitioned consistently with the DiT sequence. Combined with NVFP4 quantization, the training process becomes more memory- and compute-efficient. This efficiency gain becomes increasingly important as input videos grow longer, since both latent preparation and GEMM-heavy DiT computation become increasingly costly.

In our case, high-quality training infrastructure enables training models on long videos directly and efficiently, leading to a cleaner pipeline. As shown in Figure 4, existing methods [26, 82] rely on complex multi-stage processes, involving ODE initialization and DMD, but still have limitations in long, interactive, or multi-shot generation. The original LongLive [65] adds a long tuning stage to support long and interactive generation, but this further complicates the training pipeline. In contrast, LongLive-

- 2.0 directly achieves a long, interactive, multi-shot AR model via long-video fine-tuning. The model can then be converted to real-time generation (from 4 to 2 denoising steps) with standalone LoRA weights. Through algorithm–infrastructure co-design, LongLive-2.0 achieves strong performance on video generation benchmarks, including VBench [27] and VBench-Long [28].

2. Training Infrastructure

LongLive-2.0 supports a clean training pipeline. We directly fine-tune a bidirectional diffusion model into a long, interactive, multi-shot AR model with long-video data. Meanwhile, we derive standalone LoRA weights via DMD training directly on the original diffusion model. With LoRA weights integrated, our AR model seamlessly gains few-step denoising ability and enables real-time inference.

- 2.1. Sequence-Parallel AR Training

On the inference side, Blackwell GPUs allow full NVFP4 alignment between training and inference for highly efficient W4A4 inference, and we further quantize the KV cache into NVFP4 for substantial memory savings. On other GPU architectures (non-Blackwell), SP inference also enables real-time generation; we defer the details to Appendix D, where the quantized KV cache also lowers inter-GPU communication. Moreover, LongLive-2.0 targets end-to-end generation speed, a more practical metric than diffusion-model FPS alone. While existing reports often exclude VAE decoding, we reduce this gap with two system-level optimizations: customized parallel dequantization in the NVFP4 KV-cache kernel minimizes the overhead of low-bit KV computation, and asynchronous streaming decoding overlaps VAE decoding with model denoising. As video length increases, decoding overhead is increasingly amortized, allowing end-to-end FPS to approach model-only FPS.

LongLive-2.0 trains a chunk-level AR diffusion model that denoises the current noisy chunk conditioned on clean generated history. We use clean-context teacher forcing [30, 37, 76, 77, 81] rather than diffusion forcing [5] to avoid the train-test gap, but a literal teacher-forcing pass supervises only one target suffix at a time. Following the efficient parallel teacher-forcing formulation summarized in Self-Forcing [26], for an 𝑁-chunk raw video window

Strong infrastructure can further improve algorithm design.

###### AR Training + Traditional SP

###### AR Training + Balanced SP + NVFP4

###### GPU 0

Video Seq.

Halo Clean Latent Seq. Noisy Latent Seq.

W4 NVFP4 Model Weights (W4)

X0, X1, X2, X3

###### W4 GPU 0

W4 GPU 1

###### W4 GPU 2

###### W4 GPU 3

VAE

Layer i-1

NVFP4 GEMM

Layer i-1

| |
|---|

Speedup 2-4×

BF16

X0

h X1

h X2

h X3

z0, z1, z2, z3

Activation

| | | | |
|---|---|---|---|
| | | | |

VAE

VAE

VAE

VAE

DGRAD

|NVFP4|
|---|

GPU 0

###### GPU 1

GPU 2 GPU 3

(GEMM)

Clean z3

Noisy z3

Clean

Noisy

Clean z1

Noisy z1

Clean

Noisy

z2

z2

z0

z0

Clean

Clean

Clean

Noisy

+noise

z0

z1

z2

z3

|NVFP4|
|---|

|Dequantization|
|---|

+noise

| | | | | | | |Key|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | |Query| | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

- Clean z0

- Noisy z0

Clean z1

- Noisy z1

Clean z2

- Noisy z2

Clean z3

- Noisy z3

Model Weights

|RHT NVFP4|
|---|

- GPU 0

- GPU 1

- GPU 2

- GPU 3

W4

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

- Clean z0
- Clean z1
- Clean z2 Noisy z3

- GPU 0
- GPU 1
- GPU 2
- GPU 3

Query

Model Weights

|FP32<br><br>Weight| |
|---|---|
| | |

W4

Model

W4

|2D NVFP4| |
|---|---|
| | |

Weights Model

update

W4

Weights

Causal Attention Mask

FPROP

WGRAD

Our Attention Mask

(GEMM)

(GEMM)

W4

###### GPU 0

###### GPU 1

###### GPU 2

###### GPU 3

W4

W4

W4

###### GPU 0

###### GPU 1

GPU 2 GPU 3

BF16

Clean

Noisy

Clean

Noisy

Clean

Noisy

Clean

Noisy

Gradient

Clean

Clean

Clean

Noisy

z0

z0

z1

z1

z2

z2

z3

z3

z0

z1

z2

z3

Layer i+1

Layer i+1

Video longer, the proportion of GEMM ↑

Loss

Loss

Loss

Loss

Loss

Training Speedup

Load Imbalance

Load Balance

- Figure 3 | Overview of the Training Infrastructure. Traditional SP (Left): In the efficient teacher-forcing layout, clean history tokens and noisy target tokens are concatenated into one sequence. Naive SP treats this as a general sequence, causing loss computation workload imbalance and leaving VAE encoding replicated across sequence-parallel ranks. Balanced SP (Middle): The same temporal chunk ownership is reused across clean/noisy latent streams, SP attention, VAE encoding, and loss computation. This chunk-aligned layout balances loss-bearing tokens across GPUs while avoiding replicated VAE preparation. NVFP4 (Right): NVFP4 training orthogonally accelerates GEMMs and reduces memory footprint.

X we encode the raw video into VAE latents Z and form paired streams [z𝑐𝑙𝑒𝑎𝑛; z𝑛𝑜𝑖𝑠𝑦]. A block-sparse AR mask lets each noisy chunk attend to preceding clean chunks and its own noisy tokens, so one forward pass supervises all 𝑁 noisy chunks.

This efficient formulation makes the AR objective practical, but it also creates a structured long sequence that quickly exceeds the memory capacity of a single GPU. Naively applying SP to AR video training leaves two inefficiencies. First, slicing the concatenated DiT sequence [z𝑐𝑙𝑒𝑎𝑛; z𝑛𝑜𝑖𝑠𝑦] can create clean-heavy and noisy-heavy ranks, which imbalances the loss-bearing workload. Second, the VAE stage still encodes the full video on every SP rank (or on one root rank followed by broadcast), so latent preparation does not benefit from sequence sharding. We therefore co-design the AR training layout with the sequence-parallel data layout and instantiate it as Balanced SP on top of DeepSpeed-Ulysses [29]. Balanced SP shares the same temporal partition across VAE preparation, local clean/noisy latent construction, DiT attention, and loss computation; under this layout, the block-sparse AR attention mask is generated directly on the SP-native token order. Balanced SP constructs the paired clean/noisy streams locally on each rank. Rather than materializing a full [z𝑐𝑙𝑒𝑎𝑛; z𝑛𝑜𝑖𝑠𝑦] sequence on one rank and then slicing it, rank 𝑝 prepares its own clean latent chunk and applies

the noise schedule locally to obtain the matched noisy chunk. Using z to denote the DiT sequence after patch embedding, let 𝑃 be SP group size, 𝐿 be the total cleanplus-noisy token length, 𝐻 be the number of attention heads, and 𝑑 be the head dimension. Rank 𝑝 owns

z(𝑝) = [︁z(𝑐𝑙𝑒𝑎𝑛𝑝) , z(𝑛𝑜𝑖𝑠𝑦𝑝) ]︁ ∈ R

𝐿

𝑃 ×𝐻×𝑑. (1)

This paired layout gives every rank both context and target tokens from the same temporal range, making the loss computation uniform across ranks.

The same chunk ownership is also applied before the DiT. Each rank VAE-encodes only its local raw-video chunk X(𝑝) plus a left halo that covers the VAE temporal receptive field, then discards the halo latents and keeps the exact local latent chunk Z(𝑝). If 𝐹 is the number of latent frames and ℎ is the halo size, replicated VAE encoding costs 𝑂(𝐹) per rank, while Balanced SP reduces the per-rank VAE cost to 𝑂(𝐹/𝑃 + ℎ) without changing the DiT training objective. After Ulysses All-to-All, the paired layout naturally produces an interleaved global token order. Rather than materializing a permutation back to [all clean; all noisy] at every attention layer, we construct the AR mask directly on this communication-native order and compile it with flex_attention [18]. Appendix C gives the exact halo construction, natural-mask index mapping, globalcoordinate handling, and SP-sharded error-buffer design.

###### Self-Forcing LongLive

###### Causal-Forcing

###### LongLive 2.0

|Bidirectional Diffusion| |
|---|---|
|ODE Init| |

|Bidirectional Diffusion| |
|---|---|
|AR Training| |

|Bidirectional Diffusion| |
|---|---|
|ODE Init| |

|Bidirectional Diffusion|
|---|

DMD

AR Training

|AR Initial (Few-step)| |
|---|---|
|DMD| |

|AR Initial| |
|---|---|
|ODE Init| |

|AR Initial (Few-step)| |
|---|---|
|DMD| |

|AR Model (Few-step)|
|---|

LoRA Weights

|AR Model (Few-step)|
|---|

|AR Initial (Few-step)| |
|---|---|
|DMD| |

|AR Model (Few-step)| |
|---|---|
|Long Tuning| |

|AR Model (Few-step)|
|---|

|AR Model (Few-step)|
|---|

[Figure 38]

[Figure 39]

✅ AR ✅ Real-time ✅ Long ✅ Multi-shot

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

✅ AR ✅ Real-time ✅ AR ✅ Real-time ✅ AR ✅ Real-time ✅ Long

[Figure 47]

[Figure 48]

- Figure 4 | Clean Pipeline for AR Video Generation. LongLive-2.0 bypasses the complex, multi-stage processes (e.g., ODE initialization, intermediate DMD) required by previous methods. Instead, our first stage directly performs AR training on the base bidirectional model using long-video data. By simply injecting standalone LoRA weights to enable few-step inference, we achieve a streamlined pipeline that uniquely supports long, interactive, multi-shot, and real-time generation all at once.

##### 2.2. NVFP4 Training

quantization. At the 5B scale, this requires custom quantization and dequantization kernels together with dedicated CUDA kernels for NVFP4 GEMMs; for the RHT-enabled branch, we additionally use Triton kernels for the transformed quantization and dequantization path. As shown in Figure 3, we apply the standard NVFP4 recipe to the linear layers: 2D block scaling for weights, 1D block scaling for activations and gradients, and higher precision for numerically sensitive operations such as reductions, normalization statistics, and optimizer states. This follows prior NVFP4 training practice and preserves consistency across forward and backward GEMMs [1, 4]. For the most gradient-sensitive path, we use prior stabilization techniques, notably Random Hadamard Transform (RHT) before quantization on the operands of the weight-gradient GEMM. In our 64s training setting, this NVFP4 stack provides an approximately 1.8× training speedup.

NVFP4 [52] is attractive for long-video generation, because it reduces memory cost and accelerates lowprecision GEMMs, whose share grows as video length increases. We therefore use NVFP4 for both AR training and DMD step distillation. To the best of our knowledge, this is the first end-to-end NVFP4 recipe for long video generation.

NVFP4 Preliminaries. NVFP4 represents each tensor element using a 4-bit floating-point value in the E2M1 [54] format together with hierarchical scaling [1, 12]. For a tensor X, the dequantized tensor can be written as:

X^ = X^ FP4 · 𝛼FP8 · 𝛼FP32, X^ FP4 ∈ FE2M1, (2)

where 𝛼FP8 is a block-wise (16 elements) scale stored in FP8 E4M3 and 𝛼FP32 is a tensor-wise global scale stored in FP32. For a tensor X, we set:

Few-step Distillation in NVFP4. In few-step distillation, both teacher and student operate in W4A4 NVFP4, keeping distillation tightly aligned with inference. As shown in Figure 5, the Real-Score model is quantized to W4A4 for NVFP4 inference. We use adaptive block scaling via scale search [12] to quantize NVFP4 weights and activations: besides the standard target magnitude 6, the quantizer also evaluates 4 and selects the lower-error encoding for each block (Appendix F). This adaptive search reduces weight quantization error under W4A4 inference. The trainable Fake-Score model and Generator use the same W4A4 NVFP4 backbone, freeze the quantized backbone, and optimize only LoRA adapters:

X^ = X^ FP4 · 𝛼FP8 · 𝛼FP32, X^ FP4 ∈ FE2M1.

(3)

where 𝐵𝑖 denotes the 𝑖-th 16-element quantization block, 𝑀FP8 = 448 is the maximum representable magnitude of E4M3, and 𝑀FP4 = 6 is the maximum representable magnitude of E2M1. Unlike uniform integer quantization, FP4 uses non-uniform dynamic step sizes, providing finer resolution for small values and coarser spacing for large ones. In addition, NVFP4 is natively supported on NVIDIA Blackwell GPUs, enabling more efficient hardware acceleration for low-precision computation.

W ≃ Dequant(𝑄search(W0)) + ΔW, ΔW =

(4)

𝛼LoRA 𝑟

BA.

Multi-Shot AR NVFP4 Training. In AR training, we train the AR long-video generator on real multi-shot data with the AR objective described in § 2.1 and the multishot prompting interface in § 4.1, using end-to-end NVFP4

where W0 is the pretrained backbone weight, 𝑄search denotes scale-search-based NVFP4 quantization, A and

#### Few-step distillation

(NVFP4 +LoRA)

NVFP4 KV Cache Window (e.g., 5 chunks)

update

###### Fake Score

###### LoRA

BF16🔥

NVFP4

[Figure 49]

||Chunk|Chunk|Chunk|Chunk|Chunk|
|---|---|---|---|---|
|
|---|

KV

Diffusion Loss

| | |
|---|---|
|…| |
| | |

NVFP4 Model GPU 0

RolloutVideoChunks

Memory 10GB↓

Parallel Dequantization

###### DMD Loss

|0|1|2|3|4|
|---|---|---|---|---|

Latent

###### Real Score

Speedup 1.5×

VAE GPU 𝟏

Async Decode

NVFP4

|Wait / Idle|0|1|2|3|4|
|---|---|---|---|---|---|

Video

update

###### Generator

###### LoRA

NVFP4

BF16🔥

[Figure 50]

##### Figure 6 | NVFP4 inference infrastructure. LongLive-

- 2.0 combines W4A4 NVFP4 inference, quantized KV cache, and asynchronous VAE decoding to improve throughput and reduce memory for long-video generation. a simple 𝐾-smoothing:

K¯ ℓ,𝑐[𝑡,ℎ,:] = Kℓ,𝑐[𝑡,ℎ,:] −

1 𝑑

∑︁𝑑

𝑢=1

Kℓ,𝑐[𝑡,ℎ,𝑢]. (6)

We then apply the same adaptive scale selection described in Equation 13, without repeating the notation here. The storage cost changes from 4𝑇𝑐𝐻𝑑 bytes to 9 8𝑇𝑐𝐻𝑑 bytes, ignoring the amortized tensor-wise scale and padding overhead, which is close to a 3.6× KV-cache

compression ratio in practice. This chunkwise NVFP4 cache preserves generation quality while substantially reducing memory footprint. Since LongLive-2.0 uses sinktoken sliding windows, each attention step may access multiple cached chunks; we therefore implement a customized parallel CUDA dequantization kernel for efficient in-window reconstruction (Figure 6). This keeps the overall KV-cache quantization/dequantization overhead below 2% in practice.

- 3.3. Asynchronous Streaming Decoding

- Figure 5 | NVFP4 DMD training infrastructure. The generator, real-score model, and fake-score model are colocated under a low-precision NVFP4 setup.

B are trainable low-rank matrices of rank 𝑟, and 𝛼LoRA is the LoRA scaling factor. Restricting updates to a LoRA subspace follows recent low-bit adapter tuning in LLMs [17, 23] and is more stable in our DMD setting than updating the full quantized backbone [26, 65, 82]. The DMD objective is unchanged (§ 4.1); only the LoRA weights are trainable.

### 3. Inference Infrastructure

##### 3.1. NVFP4 Inference

At deployment time, we execute the generator in W4A4 NVFP4, either as a quantized backbone with a separate LoRA branch or as a merged W4A4+LoRA model with fused low-rank kernels. Since AR long-video generation is dominated by repeated linear layers and attention GEMMs, replacing BF16 GEMMs with FP4 GEMMs reduces memory traffic and offers an ideal theoretical throughput speedup of up to 4×. We additionally materialize quantized weights and drop BF16 master weights after LoRA wrapping, further reducing resident memory. Unlike post-training quantization (PTQ) methods [34, 72, 78], our backbone is trained with NVFP4-aware training, which better preserves generation quality under W4A4 inference.

The final variational autoencoder (VAE) decoding step is often a major bottleneck in video generation. The centralized decoding scheme used in the baseline LongLive model accumulates all latent chunks before sequential decoding, leading to a VAE-side GPU memory cost of 𝒪(𝐶 ·𝑇𝑐) for 𝐶 chunks and a long end-to-end latency. We instead design a heterogeneous asynchronous pipeline. We first re-engineer the 3D VAE to support chunk-by-chunk streaming decoding with immediate CPU offloading, reducing the VAE GPU memory footprint to 𝒪(𝑇𝑐). We then dedicate one GPU to VAE decoding and run it asynchronously alongside the 𝑃-GPU DiT SP cluster. Let 𝑡DiT and 𝑡VAE denote the per-chunk latencies of denoising and decoding, respectively. While the DiT cluster denoises chunk 𝑐 + 1, the VAE node decodes chunk 𝑐. Since the DiT loop is dominant in practice (𝑡DiT ≥ 𝑡VAE), decoding is largely hidden behind denoising, reducing the end-to-end latency from 𝐶(𝑡DiT + 𝑡VAE) to approximately 𝐶 · 𝑡DiT + 𝑡VAE and enabling memory-efficient streaming generation.

##### 3.2. Parallel KV Quantization

In AR long video generation, KV cache memory grows linearly with history and quickly becomes a bottleneck [60]. We therefore quantize the cache at the frame-chunk level, aligned with our blockwise pipeline. Each chunk contains 𝐹𝑐 = 8 frames and 𝑇𝑐 = 𝐹𝑐𝐿𝑓 latent tokens. For layer ℓ, the cached KV chunk 𝑐 is

Kℓ,𝑐,Vℓ,𝑐 ∈ R𝑇

𝑐×𝐻×𝑑, (5) which we reshape to R(𝑇

𝑐𝐻)×𝑑 and quantize independently with NVFP4 micro-block scaling. For keys, we first apply

- Table 1 | AR training efficiency of LongLive-2.0. We compare end-to-end iteration time (seconds); red subscripts denote speedup over BF16+SP.

Input Length

BF16 w/o SP

BF16 w/ SP

BF16 Balanced SP

NVFP4 Balanced SP

16s 75.3 52.2 45.8 40.11.3× 32s 202.7 162.7 136.8 119.31.4× 64s OOM 1372.9 1196.5 639.52.1×

- Table 2 | Progressively quantizing the generator, real-score, and fake-score models in DMD training. We report peak per-GPU memory.

Key

Shot 1

[Figure 51]

[Figure 52]

[Figure 53]

Query

[Figure 54]

Shot 2

[Figure 55]

[Figure 56]

[Figure 57]

Shot-level

Sink

Shot 3

[Figure 58]

[Figure 59]

[Figure 60]

Global-level Sink

Figure 7 | Multi-shot Attention Sink for streaming multi-shot inference.

Generator Real Fake Peak Memory Ratio ↓ BF16 BF16 BF16 70.5 GB -

### 4. Algorithm-level Designs

NVFP4 BF16 BF16 63.3 GB 0.90× NVFP4+LoRA NVFP4 BF16 57.2 GB 0.81× NVFP4+LoRA NVFP4 NVFP4+LoRA 49.0 GB 0.69×

- 4.1. Training in Clean Pipeline Multi-Shot Interactive AR Training.

The AR objective and efficient teacher-forcing layout are described in § 2.1; here we focus on the algorithmic interface enabled by chunk-level generation. We employ Wan2.2-TI2V-5B [59] as our base model. We treat each temporal latent chunk Z𝑖 as an editable generation unit and bind it to an individual text prompt T𝑖. Cross-attention is factorized per chunk as CrossAttn(Z𝑖,T𝑖), rather than conditioning the whole video on a single global prompt. This decoupling lets different shots carry different prompts, supports prompt switches at chunk boundaries, and preserves previously generated history when the user edits future chunks.

##### 4.2. Inference with Multi-Shot Attention Sink

To deploy our model for multi-shot streaming, we adopt sliding-window self-attention with KV caching to cap the per-step compute footprint at 𝒪(𝑊 ·𝐿𝑐), where 𝑊 is the attention-window length in chunks and 𝐿𝑐 is the token length of each chunk. However, naively discarding tokens outside the window causes appearance drift. While standard attention sinks [63] mitigate this by pinning the first few video frames, they fail in multi-shot settings: a single global sink cannot preserve intra-shot coherence, while a moving shot-level sink loses global identity.

Few-step Distillation. Our few-step distillation framework is derived from LongLive, but with several important simplifications. First, because the AR-trained model already supports long-video generation, we avoid the original multi-stage strategy with ODE initialization, shortvideo DMD, and streaming long-tuning DMD. We instead perform one-stage DMD distillation on top of the ARtrained model, yielding a cleaner formulation without separate initialization or progressive long-tuning stages. Second, we do not fully fine-tune the DiT backbone; instead, we optimize LoRA modules only during the entire distillation process. This choice leads to more stable optimization and makes the resulting few-step capability easily transferable to any Wan2.2-TI2V-5B-based AR model. Specifically, we initialize the student, critic, and teacher from the original Wan2.2-TI2V-5B model. Similar to LCMLoRA [45], we find that the trained LoRA can be directly plugged into the AR model to reduce inference steps without further tuning. In the end, the distilled model reduces generation to two steps, while preserving the long-video generation ability of the original framework. We discuss the differences between our strategy and straightforward DMD fine-tuning in Appendix (§ H).

Multi-Shot Attention Sink. To resolve this, we introduce a multi-shot attention sink with two cooperating anchor sets (Figure 7): Global Sink (𝒜𝑔): the first 𝑆𝑔 frames of the video, permanently fixed to preserve global identity. Shot-Level Sink (𝒜𝑠): the first 𝑆𝑠 frames of the current shot, re-bound at every scene cut to maintain local temporal coherence.

At any chunk generation step 𝑡, the effective key/value set is 𝒦eff(𝑡) = 𝒜𝑔 ∪ 𝒜𝑠 ∪ KV[𝑡−𝑊,𝑡), with overlapping tokens deduplicated. 𝒜𝑠 incurs zero memory overhead: it is tracked merely via two scalar pointers (START, LEN). It is virtually prepended to the sliding window only after the window rolls past it, avoiding data copying.

Interaction with Chunk-wise Prompting. Crucially, this mechanism integrates seamlessly with our chunk-wise interactive prompting (§ 4.1). A prompt switch 𝑝𝑘 →𝑝′𝑘 inherently defines a scene cut. This simply triggers the local re-binding of 𝒜𝑠 to the new chunk and re-initializes the subsequent cross-attention cache, leaving the global sink 𝒜𝑔 and preceding history untouched. This strict decoupling enables minute-scale interactive generation without redundant recomputation.

Table 3 | Inference efficiency of LongLive-2.0 under progressively enabled optimizations. The experiments are handled on NVIDIA GB200 180GB GPU, another GPU is used under Async Decoding; end-to-end (E2E) latency and peak memory are reported at different target video lengths.

16 s 32 s 64 s E2E Gen.↓ (s)

Inference Settings

FPS↑

Total Mem.↓ (GB)

E2E Gen.↓ (s)

Total Mem.↓ (GB)

E2E Gen.↓ (s)

Total Mem.↓ (GB)

BF16 24.8 26.6 36.4 53.2 36.4 112.9 36.4 NVFP4 32.0 22.9 29.7 46.6 29.7 96.0 29.7 + NVFP4 KV Cache 29.7 23.8 19.4 48.9 19.4 99.5 19.4 + Async Decoding 29.7 15.9 19.4 29.1 19.4 57.6 19.4 3 Steps 35.2 12.7 19.4 23.2 19.4 46.0 19.4 2 Steps 45.7 11.2 19.4 19.2 19.4 36.3 19.4

Table 4 | Comparison on VBench among LongLive-2.0 and baselines. #Steps means denoising steps.

Throughput Evaluation Scores ↑ (FPS) ↑ Total Quality Semantic

Model Precision #Steps #Params Resolution

Self-Forcing [26] BF16 4 1.3B 832×480 21.2 84.31 85.07 81.28 Causal-Forcing [82] BF16 4 1.3B 832×480 21.0 84.04 84.59 81.84 Rolling-Forcing [43] BF16 4 1.3B 832×480 19.5 81.22 84.08 69.78 Context-Forcing [8] BF16 4 1.3B 832×480 17.0 83.44 84.98 77.29 CausVid [69] BF16 4 1.3B 832×480 21.2 81.20 84.05 69.80 SANA Video-480P [7] BF16 4 2B 832×480 13.2 84.17 84.85 81.46 SANA Video-720P [7] BF16 4 2B 1280×720 – 84.05 84.63 81.73

- Wan2.1-T2V-1.3B [59] BF16 50 1.3B 832×480 1.6 84.26 85.30 80.09
- Wan2.2-TI2V-5B [59] BF16 50 5B 1280×720 3.3 83.32 84.95 76.81 LongLive [65] BF16 4 1.3B 832×480 20.7 84.87 86.97 76.47

BF16 4 5B 1280×720 24.8 85.06 86.67 78.63 NVFP4 4 5B 1280×720 29.7 84.51 86.43 76.81 NVFP4 2 5B 1280×720 45.7 83.14 85.40 74.12

LongLive-2.0

### 5. Experimental Results

##### 5.1. Training Efficiency

AR Training Efficiency. Table 1 reports the end-toend AR training iteration time under BF16, BF16+SP, BF16+Balanced SP, and NVFP4+Balanced SP. Plain BF16 is efficient only at shorter video lengths, taking 75.3s and 202.7s at 16s and 32s but running out of memory (OOM) at 64s. Adding sequence parallelism makes long-video training feasible and reduces the 16s/32s iteration time to 52.2s and 162.7s, respectively, while our Balanced SP further improves the BF16 path to 45.8s, 136.8s, and 1196.5s across the three lengths.

Combining Balanced SP with NVFP4 gives the fastest training configuration. It reduces the iteration time to 40.1s, 119.3s, and 639.5s for 16s, 32s, and 64s videos, corresponding to 1.3×, 1.4×, and 2.1× speedups over the BF16+SP baseline. The gain becomes most pronounced at the longest sequence length, where NVFP4+Balanced SP nearly halves the iteration time compared with

BF16+Balanced SP and more than doubles throughput over BF16+SP.

NVFP4 DMD Training. We next study few-step DMD training, where the generator, real-score model, and fakescore model are co-located on each GPU. Table 2 shows a progressive NVFP4 conversion path, with NVFP4 for the frozen real-score branch and NVFP4+LoRA for the trainable branches. Peak per-GPU memory decreases monotonically from 70.5 GB to 49.0 GB, corresponding to a 21.5 GB reduction per GPU, or 0.69× of the BF16 baseline.

##### 5.2. Inference Efficiency

Tables 4 and 3 show that LongLive-2.0 achieves strong throughput with a favorable latency–memory trade-off. The 4-step 5B model reaches 29.7 FPS, surpassing all listed baselines, while the 2-step variant further improves throughput to 45.7 FPS. In the progressive ablation, NVFP4 first improves both speed and memory over BF16, and KV-cache quantization further reduces peak memory from 29.7 GB to 19.4 GB with only a modest latency cost.

Table 5 | Comparison on VBench-Long for 60s video generation. Scores are reported in percentage. Avg. Rank is computed over the six metrics. The best is in bold, and the second-best is underlined.

Avg. Rank ↓

Subject Consistency ↑

Background Consistency ↑

Motion Smoothness ↑

Dynamic Degree ↑

Aesthetic Quality ↑

Imaging Quality ↑

Method

NOVA [16] 8.50 77.50 88.06 98.94 12.00 47.53 44.97 MAGI-1 [57] 6.67 79.46 87.76 99.26 56.00 52.10 54.54 Causal-Forcing [82] 6.50 93.52 94.12 95.74 72.32 51.24 62.30 SkyReels-V2 [6] 6.00 84.99 89.95 98.67 44.00 57.64 66.67 Self-Forcing [26] 5.83 95.84 95.27 98.20 51.72 56.05 62.22 CausVid [69] 5.33 86.75 89.85 98.47 52.00 62.88 67.47 Rolling-Forcing [43] 4.50 94.09 94.47 98.65 36.00 63.50 72.42

LongLive [65] 4.17 97.13 95.89 98.61 44.56 58.17 67.56 LongLive-2.0 3.67 97.48 97.00 98.86 60.62 53.68 65.51

→ NVFP4 3.83 97.62 96.97 98.94 45.88 53.72 66.24

Asynchronous decoding then lowers E2E latency by overlapping denoising and VAE decoding, and the final 2-step system reaches 36.3s E2E latency for 64s videos while maintaining the 19.4 GB memory footprint.

##### 5.3. Performance

Short-video generation. We first evaluate LongLive2.0 on short-video generation using the official VBench prompts with our prompt augmentation, as shown in Table 4. LongLive-2.0 achieves the strongest performance at the higher 1280×720 resolution. We evaluate LongLive2.0 under NVFP4 quantization. Reducing denoising steps further increases speed to 35.2 FPS (3 steps) and 45.7 FPS (2 steps). This shows that NVFP4 with fewer denoising steps enables efficient real-time 720p video generation, achieving up to 2× speedup over prior methods.

We note that higher resolution does not always yield higher VBench scores. Since VBench resizes videos and samples frames, results depend on the evaluation protocol. Similar trends appear in Table 4, where 720p models (e.g., Wan2.2TI2V-5B, SANA) do not consistently outperform 480p. Thus, slightly lower scores at 720p are expected and do not indicate worse quality.

Long Video Generation. We evaluate LongLive-2.0 with 4 denoising steps on long video generation using MovieGenBench prompts and VBench-Long. As shown in Table 5, LongLive-2.0 achieves the best average rank among all compared methods on 60s generation, demonstrating strong overall long-range generation ability. Its advantage is most pronounced in subject and background consistency: the NVFP4 model obtains the best subject consistency score of 97.62, while the BF16 model obtains the best background consistency score of 97.00.

### 6. Conclusion

In this work, we present LongLive-2.0, a algorithm–infrastructure co-design system for efficient long video generation across training and inference. For training, we propose Balanced SP along with NVFP4 quantization. For inference, we quantize both the model (W4A4) and KV cache to NVFP4 and accelerate execution with parallel dequantization. Benefiting from this strong infrastructure, LongLive-2.0 enables a remarkably clean training pipeline that directly fine-tunes diffusion models to long, multi-shot AR without complex ODE initialization or additional long tuning stages. Real-time generation can be further achieved with lightweight LoRA weights. LongLive-2.0 achieves up to 2.1× training speedup and 1.8× inference speedup, with LongLive-2.0-5B supporting 45.7 FPS while maintaining strong benchmark performance. To our knowledge, LongLive-2.0 is the first end-to-end NVFP4 training and inference system tailored for long video generation.

Limitations. The acceleration gain from low-bit quantization is hardware-dependent. NVFP4 inference delivers acceleration only on Blackwell GPUs (e.g., GB200), which are equipped with the latest-generation Tensor Cores and optimized kernels. In contrast, non-Blackwell GPUs, like A100 (Ampere architecture) and H100 (Hopper architecture), lack native hardware support for these optimized kernels. To compensate this limitation, we use SP inference as an alternative solution to boost inference efficiency on non-Blackwell platforms (Section D in the appendix).

Broader Impacts. LongLive-2.0 reduces computational costs and lowers the resource threshold for related research and deployment. It shares the ethical impacts with existing video generation models. The NVFP4 and parallelism infrastructure itself involves no negative social implications.

### References

- [1] Felix Abecassis, Anjulie Agrusa, Dong Ahn, Jonah Alben, Stefania Alborghetti, Michael Andersch, Sivakumar Arayandi, Alexis Bjorlin, Aaron Blakeman, Evan Briones, et al. Pretraining large language models with nvfp4. arXiv preprint arXiv:2509.25149, 2025.
- [2] Eduardo Alvarez. Introducing nvfp4 for efficient and accurate low-precision inference, 2025. NVIDIA Technical Blog.
- [3] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated llms. NeurIPS, 37:100213–100240, 2024.
- [4] Roberto L Castro, Andrei Panferov, Soroush Tabesh, Oliver Sieberling, Jiale Chen, Mahdi Nikdan, Saleh Ashkboos, and Dan Alistarh. Quartet: Native fp4 training can be optimal for large language models. arXiv preprint arXiv:2505.14669, 2025.
- [5] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets fullsequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [6] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. SkyReels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.
- [7] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, and Enze Xie. Sana-video: Efficient video generation with block linear diffusion transformer. In ICLR, 2026.
- [8] Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028, 2026.
- [9] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu,

- Hongxu Yin, Yao Lu, and Song Han. Scaling RL to long videos. In NeurIPS, 2025.
- [10] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Yihui He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, and Song Han. Longvila: Scaling longcontext visual language models for long videos. In ICLR, 2025.
- [11] Brian Chmiel, Maxim Fishman, Ron Banner, and Daniel Soudry. Fp4 all the way: Fully quantized training of llms. arXiv preprint arXiv:2505.19115, 2025.
- [12] Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, and Song Han. Four over six: More accurate nvfp4 quantization with adaptive block scaling. arXiv preprint arXiv:2512.02010, 2025.
- [13] Hanshuai Cui, Zhiqing Tang, Zhi Yao, Fanshuai Meng, Weijia Jia, and Wei Zhao. Not all frames deserve full computation: Accelerating autoregressive video generation via selective computation and predictive extrapolation. arXiv preprint arXiv:2604.02979, 2026.
- [14] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minutescale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.
- [15] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and ChoJui Hsieh. LoL: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026.
- [16] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In International Conference on Learning Representations, 2025.
- [17] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. NeurIPS, 36:10088–10115, 2023.
- [18] Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496, 2(3):4, 2024.
- [19] Jiarui Fang and Shangchun Zhao. Usp: A unified sequence parallelism approach for long context generative ai. arXiv preprint arXiv:2405.07719, 2024.

- [20] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.
- [21] Diandian Gu, Peng Sun, Qinghao Hu, Ting Huang, Xun Chen, Yingtong Xiong, Guoteng Wang, Qiaoling Chen, Shangchun Zhao, Jiarui Fang, et al. Loongtrain: Efficient training of long-sequence llms with head-context parallelism. arXiv preprint arXiv:2406.18485, 2024.
- [22] Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. Trans. Mach. Learn. Res., 2026, 2026.
- [23] Wei Huang, Yi Ge, Shuai Yang, Yicheng Xiao, Huizi Mao, Yujun Lin, Hanrong Ye, Sifei Liu, Ka Chun Cheung, Hongxu Yin, et al. Qerl: Beyond efficiency– quantization-enhanced reinforcement learning for llms. arXiv preprint arXiv:2510.11696, 2025.
- [24] Wei Huang, Yue Liao, Yukang Chen, Jianhui Liu, Haoru Tan, Si Liu, Shiming Zhang, Shuicheng Yan, and Xiaojuan Qi. Mc#: Mixture compressor for mixture-of-experts large models. T-PAMI, 2026.
- [25] Wei Huang, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, and Xiaojuan Qi. Mixture compressor for mixtureof-experts llms gains more. arXiv preprint arXiv:2410.06270, 2024.
- [26] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [27] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024.
- [28] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench++: Comprehensive and versatile benchmark suite for video generative models. T-PAMI, 48(3):3268–3285, 2026.
- [29] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song,

- Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.
- [30] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.
- [31] Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M. Rehg, and Tobias Hinz. Shotadapter: Text-to-multi-shot video generation with diffusion models. In CVPR, pages 28405– 28415, 2025.
- [32] Youngrae Kim, Qixin Hu, C.-C. Jay Kuo, and Peter A. Beerel. MemRoPE: Training-free infinite video generation via evolving memory tokens. arXiv preprint arXiv:2603.12513, 2026.
- [33] Jia Li, Xiaomeng Fu, Xurui Peng, Weifeng Chen, Youwei Zheng, Tianyu Zhao, Jiexi Wang, Fangmin Chen, Xing Wang, and Hayden Kwok-Hay So. Train short, inference long: Training-free horizon extension for autoregressive video generation. arXiv preprint arXiv:2602.14027, 2026.
- [34] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024.
- [35] Ruibin Li, Tao Yang, Fangzhou Ai, Tianhe Wu, Shilei Wen, Bingyue Peng, and Lei Zhang. Longhorizon streaming video generation via hybrid attention with decoupled distillation. arXiv preprint arXiv:2604.10103, 2026.
- [36] Shenggui Li, Fuzhao Xue, Chaitanya Baranwal, Yongbin Li, and Yang You. Sequence parallelism: Long sequence training from system perspective. In ACL, pages 2391–2404, 2023.
- [37] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. NeurIPS, 37:56424– 56445, 2024.
- [38] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinitelength video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025.
- [39] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device

- llm compression and acceleration. MLSys, 6:87–100, 2024.
- [40] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial posttraining for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025.
- [41] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889, 2023.
- [42] Jinxiu Liu, Xuanming Liu, Kangfu Mei, Yandong Wen, Ming-Hsuan Yang, and Weiyang Liu. Streaming autoregressive video generation via diagonal distillation. arXiv preprint arXiv:2603.09488, 2026.
- [43] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.
- [44] Ziming Liu, Shaoyu Wang, Shenggan Cheng, Zhongkai Zhao, Kai Wang, Xuanlei Zhao, James Demmel, and Yang You. Startrail: Concentric ring sequence parallelism for efficient near-infinitecontext transformer model training. arXiv preprint arXiv:2407.00611, 2024.
- [45] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick Von Platen, ApolinÃA˛rio Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.
- [46] Yawen Luo, Xiaoyu Shi, Junhao Zhuang, Yutian Chen, Quande Liu, Xintao Wang, Pengfei Wan, and Tianfan Xue. ShotStream: Streaming multi-shot video generation for interactive storytelling. arXiv preprint arXiv:2603.25746, 2026.
- [47] Xin Ma, Yaohui Wang, Xinyuan Chen, Gengyun Jia, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [48] Yuexiao Ma, Xuzhe Zheng, Jing Xu, Xiwei Xu, Feng Ling, Xiawu Zheng, Huafeng Kuang, Huixia Li, Xing Wang, Xuefeng Xiao, Fei Chao, and Rongrong Ji. Flow caching for autoregressive video generation. arXiv preprint arXiv:2602.10825, 2026.
- [49] Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, and Yukang Chen. Triattention: Efficient long reasoning with trigonometric kv compression. arXiv preprint arXiv:2604.04921, 2026.

- [50] Xiaofeng Mao, Shaohao Rui, Kaining Ying, Bo Zheng, Chuanhao Li, Mingmin Chi, and Kaipeng Zhang. PackForcing: Short video training suffices for long video sampling and long context inference. arXiv preprint arXiv:2603.25730, 2026.
- [51] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. Fp8 formats for deep learning. arXiv preprint arXiv:2209.05433, 2022.
- [52] NVIDIA. Nvidia blackwell architecture technical brief, 2024. Accessed: 2025-05-13.
- [53] NVIDIA. Speeding up variable-length training with dynamic context parallelism and nvidia megatron core, 2026.
- [54] Open Compute Project. OCP Microscaling Formats (MX) Specification. Open Compute Project, version 1.0 edition, 2023.
- [55] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195– 4205, 2023.
- [56] Bita Darvish Rouhani et al. Microscaling data formats for deep learning. arXiv preprint arXiv:2310.10537, 2023.
- [57] Sand.ai. MAGI-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [58] Jiahao Tian, Chenxi Song, Wei Cheng, and Chi Zhang. Free-lunch long video generation via layer-adaptive o.o.d correction. arXiv preprint arXiv:2603.25209, 2026.
- [59] Team Wan. Wan: Open and advanced largescale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [60] Haocheng Xi, Shuo Yang, Yilong Zhao, Muyang Li, Han Cai, Xingyang Li, Yujun Lin, Zhuoyang Zhang, Jintao Zhang, Xiuyu Li, et al. Quant videogen: Autoregressive long video generation via 2-bit kv-cache quantization. arXiv preprint arXiv:2602.02958, 2026.
- [61] Xunzhi Xiang, Zixuan Duan, Guiyu Zhang, Haiyu Zhang, Zhe Gao, Junta Wu, Shaofeng Zhang, Tengfei Wang, Qi Fan, and Chunchao Guo. Pathwise testtime correction for autoregressive long video generation. arXiv preprint arXiv:2602.05871, 2026.
- [62] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large

- language models. In ICML, pages 38087–38099. PMLR, 2023.
- [63] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.
- [64] Jiacheng Yang, Jun Wu, Yaoyao Ding, Zhiying Xu, Yida Wang, and Gennady Pekhimenko. Streamfusion: Scalable sequence parallelism for distributed inference of diffusion transformers on gpus. arXiv preprint arXiv:2601.20273, 2026.
- [65] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. In ICLR, 2026.
- [66] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. MANIQA: multi-dimension attention network for no-reference image quality assessment. In CVPR Workshops, pages 1190–1199, 2022.
- [67] Yang Yang, Tianyi Zhang, Wei Huang, Jinwei Chen, Boxi Wu, Xiaofei He, Deng Cai, Bo Li, and PengTao Jiang. Anchor forcing: Anchor memory and triregion rope for interactive streaming video diffusion. arXiv preprint arXiv:2603.13405, 2026.
- [68] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.
- [69] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. arXiv preprint arXiv:2412.07772, 2024.
- [70] Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, and Xiaojuan Qi. VideoSSM: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519, 2025.
- [71] Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. Helios: Real realtime long video generation model. arXiv preprint arXiv:2603.04379, 2026.
- [72] Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. Turboquant: Online vector quantization with near-optimal distortion rate. arXiv preprint arXiv:2504.19874, 2025.

- [73] Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread INT4 quantization. In ICML, 2025.
- [74] Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, and Jianfei Chen. Sageattention3: Microscaling FP4 attention for inference and an exploration of 8-bit training. arXiv preprint arXiv:2505.11594, 2025.
- [75] Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In ICLR, 2025.
- [76] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025.
- [77] Yuan Zhang, Jiacheng Jiang, Guoqing Ma, Zhiying Lu, Haoyang Huang, Jianlong Yuan, Nan Duan, and Daxin Jiang. Generative pre-trained autoregressive diffusion transformer. arXiv preprint arXiv:2505.07344, 2025.
- [78] Tianchen Zhao, Tongcheng Fang, Haofeng Huang, Enshu Liu, Rui Wan, Widyadewi Soedarmadji, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, et al. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation. arXiv preprint arXiv:2406.02540, 2024.
- [79] Xuanlei Zhao, Shenggan Cheng, Chang Chen, Zangwei Zheng, Ziming Liu, Zheming Yang, and Yang You. Dsp: Dynamic sequence parallelism for multi-dimensional transformers. arXiv preprint arXiv:2403.10266, 2024.
- [80] Zengqun Zhao, Yanzuo Lu, Ziquan Liu, Jifei Song, Jiankang Deng, and Ioannis Patras. Relax forcing: Relaxed kv-memory for consistent long video generation. arXiv preprint arXiv:2603.21366, 2026.
- [81] Deyu Zhou, Quan Sun, Yuang Peng, Kun Yan, Runpei Dong, Duomin Wang, Zheng Ge, Nan Duan, and Xiangyu Zhang. Taming teacher forcing for masked autoregressive video generation. In CVPR, pages 7374–7384, 2025.
- [82] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for highquality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.
- [83] Kai Zou, Dian Zheng, Hongbo Liu, Tiankai Hang, Bin Liu, and Nenghai Yu. HiAR: Efficient autoregressive long video generation via hierarchical denoising. arXiv preprint arXiv:2603.08703, 2026.

### A. Related Work

##### A.1. Long Video Generation

Recent video generation research has shifted from short-clip bidirectional diffusion transformers to causal autoregressive (AR) synthesis, where videos are generated frame-by-frame or chunk-by-chunk. CausVid [69] converts a pretrained bidirectional video diffusion model into a causal AR generator and distills it into a few-step streaming model. MAGI1 [57] scales chunk-level AR generation with nearly constant peak inference cost, while AAPT [40] explores one-step real-time interactive generation. These works establish AR video generation as a promising formulation for streaming synthesis, but also expose challenges such as exposure bias, error accumulation, memory growth, and long-range temporal drift.

- A major line of work addresses the train–test mismatch in AR video diffusion. Self-Forcing [26] trains models under their own rollout distribution rather than only teacher-forced ground-truth contexts [22, 81]. LongLive [65], SelfForcing++ [14], and Rolling Forcing [43] extend this idea to real-time long-video generation with causal attention, KV re-cache, attention sinks, long-context tuning, joint denoising, and few-step distillation. More recent forcing-based methods analyze finer-grained mismatch: Causal Forcing [82] studies the architectural gap between bidirectional teachers and causal students; Context Forcing [8] uses long-context teachers and Slow-Fast Memory to supervise long-context students; HiAR [83] performs hierarchical denoising so that future blocks are conditioned on contexts at matched noise levels; and Diagonal Distillation [42] exploits both temporal chunks and denoising steps to improve streaming distillation.

Another important direction studies long-range memory and efficient cache management. Since dense attention over all generated frames is infeasible, existing methods rely on sliding windows, KV caches, attention sinks, or compressed memory. However, naive memory reuse can cause identity drift, temporal repetition, or motion stagnation. LoL [15], Deep Forcing [68], Relax Forcing [80], MemRoPE [32], VideoSSM [70], and Hybrid Forcing [35] improve long-horizon stability through RoPE stabilization, deep attention sinks, structured KV memory, evolving memory tokens, state-space memory, and hybrid linear/sparse attention. Complementary system-oriented methods reduce the deployment cost of AR video generation: Quant VideoGen [60] compresses KV cache memory, FlowCache [48] introduces chunk-wise caching, SCOPE [13] applies selective computation, and Helios [71] designs a large AR model for real-time long-video generation.

Finally, training-free horizon extension and interactive generation have also emerged. FLEX [33], Test-Time Correction [61], FreeLOC [58], and PackForcing [50] extend pretrained or short-trained models to longer horizons through positional correction, test-time trajectory calibration, or structured cache partitioning. Anchor Forcing [67] targets prompt-switching in streaming diffusion [31], while ShotStream [46] extends AR generation to multi-shot interactive storytelling. Overall, AR long video generation has evolved from simply causalizing diffusion models into a broader problem involving rollout alignment, memory design, positional extrapolation, distillation, and efficient deployment.

##### A.2. FP4 Quantization

Low-bit quantization has become a central tool for reducing the cost of large generative models. A substantial body of work studies post-training quantization (PTQ) and quantization-aware training (QAT) for LLMs and diffusion models [34, 72, 78]. Representative techniques improve robustness by correcting outlier channels, smoothing activation ranges, reconstructing layer outputs, or using low-rank compensation [20, 24, 25, 34, 39, 62]. These methods are highly effective for deployment compression, but most of them still assume integer-style quantization or focus on PTQ, leaving a mismatch between low-precision inference and the precision regime used during training.

Recent work has therefore moved beyond FP8 [51] toward FP4 floating-point training and inference. FP4 is attractive because it can reduce memory traffic and matrix-multiplication cost more aggressively than FP8, but the E2M1 value set is extremely coarse and requires careful scaling. Block-scaled formats such as MXFP4 [54, 56] and NVFP4 [2, 52] address this issue through microscaling factors shared by small groups of values. Compared with MXFP4, NVFP4 uses finer 16-element blocks, FP8 E4M3 block scales, and a tensor-level global scale, which improves local dynamic-range tracking and has been shown to offer a favorable accuracy-efficiency trade-off in large-scale studies [1, 11].

Stable end-to-end FP4 training also depends on algorithmic choices beyond the numeric format. Prior studies show that weights, activations, and gradients must be quantized consistently, while numerically sensitive operations such as reductions, normalization statistics, and optimizer states often remain in higher precision [1, 11]. Random Hadamard transforms and rotation-based quantization help disperse block-level outliers, stochastic rounding reduces bias in low-

precision gradient updates, and adaptive block-scale selection such as Four Over Six further lowers NVFP4 quantization error [1, 3, 11, 12]. Complementary low-bit adapter methods show that quantized backbones can be paired with trainable low-rank updates for efficient finetuning and reinforcement learning [17, 23].

However, existing FP4 studies are primarily centered on LLM pretraining, LLM finetuning, or general low-bit inference. Autoregressive long-video generation introduces different system pressures: spatio-temporal sequences are much longer, denoising repeatedly stresses the same GEMM and attention paths, KV caches grow with generated history, and quality is sensitive to any mismatch between training, distillation, and deployment precision. Our work studies NVFP4 in this setting, aiming to jointly align stable training, W4A4 inference, KV-cache compression, and long-video deployment.

##### A.3. Sequence Parallelism

To overcome single-device memory limits, sequence parallelism (SP) distributes long sequences across multiple devices. Existing SP techniques primarily follow two paradigms. Ring-style systems [9, 10, 21, 36, 41, 44] partition sequences into chunks, overlapping point-to-point communication with attention computation. Conversely, DeepSpeed-Ulysses [29] partitions along the attention head dimension, utilizing All-to-All communication to gather full sequences. This strictly decouples communication from the core attention arithmetic. At extreme scales, hybrid approaches like USP [19] integrate both methods, using Ulysses intra-node and Ring inter-node.

As generative modeling expands from LLMs to Diffusion Transformers (DiTs) [47, 55], computational bottlenecks shift toward the massive spatio-temporal sequences inherent in video generation. Consequently, recent infrastructures customize fundamental SP paradigms for multi-dimensional data. For instance, StreamFusion [64] tailors hybrid SP communication to the unique memory profiles of DiTs. At the lowest infrastructure level, Dynamic Sequence Parallelism (DSP) [79] re-engineers 1D SP by dynamically switching communication across spatial and temporal axes, reducing All-to-All overhead. Furthermore, systems like Megatron Core introduce Dynamic Context Parallelism [53] to optimize sequence sharding and activation memory specifically for variable-length video pre-training.

In AR video training, efficient mask-based teacher forcing introduces a structure that is absent from ordinary longcontext modeling: the same temporal chunk appears once as clean context and once as a noisy prediction target, and training relies on complex spatio-temporal masks compiled via FlexAttention [18]. Simply combining this clean/noisy teacher-forcing layout with an existing SP backend is therefore insufficient. Ring-style methods are difficult to apply directly because their load-balancing assumptions do not align with these irregular block-sparse masks, while a naive Ulysses partition can separate clean-only and noisy-only ranks and leave VAE latent preparation replicated across SP ranks. We therefore build on DeepSpeed-Ulysses but co-design the AR training layout with sequence-parallel execution. In the current instantiation, Balanced SP lets each rank locally construct paired clean/noisy latents from its temporal chunk, builds a natural teacher-forcing mask on the post-All-to-All order, and shards VAE encoding with an exact left-halo scheme. This distinguishes our method from prior SP systems, which mainly optimize communication schedules or activation memory for generic long sequences rather than the clean/noisy pairing and latent-preparation bottleneck specific to teacher-forced video DiT training.

### B. Multi-shot Long-video Dataset

We curate a large-scale long-video dataset for training LongLive-2.0. We split raw long videos into independent shots and annotate each with structured captions spanning visual, scene, character, action, and cinematography aspects. After completing shot-level captioning, we merge the captions of all segmented shots from the same full-length video and further refine the integrated descriptions to ensure temporal coherence and logical consistency across consecutive frames and scenes.

Subsequently, we conduct rigorous data filtering and quality cleaning to remove low-quality and invalid samples. We remove videos with excessively short shot duration, content containing logos, watermarks or prominent text, footage with severe camera shake, abnormal playback speed such as fast-forward and slow-motion, overexposed or underexposed frames, blurry and out-of-focus visuals, and low-motion clips with frozen frames or only trivial zoom effects. For further quality control, we adopt the MANIQA [66] metric to evaluate the visual quality of sampled video frames, and the average score is adopted as the overall quality score for each video. Only top-ranked high-quality videos are retained. In the final version, our dataset contains 120K long videos with abundant segmented shots. The videos are evenly distributed in three duration groups: 16–32 seconds, 32–64 seconds, and over 64 seconds, each accounting for one-third of the total data volume.

### C. Balanced SP Details

Hybrid parallelism and global coordinates. We use a hybrid scheme with world_size = dp_size×sp_size. Ranks in the same SP group share the same sample and prompt, while only the temporal token dimension is partitioned. To match non-parallel training, Rotary Position Embedding (RoPE) uses global frame indices and sequence offsets rather than local rank indices. The attention mask, supervision mask, and loss mask are also evaluated in global coordinates, and the loss is normalized by the global number of valid tokens. Consequently, the SP implementation preserves the same training objective as the non-parallel formulation while reducing per-rank activation memory. Inside each DiT attention block, we use z to denote the hidden sequence corresponding to the clean/noisy latent streams. Let 𝑃 be the SP group size, 𝐿 be the total clean-plus-noisy token length, 𝐻 be the number of attention heads, and 𝑑 be the head dimension. The Ulysses backend exchanges the sequence and head dimensions:

𝑃 ×𝐻×𝑑 −−−−−→All-to-All ̃︀z(𝑝) ∈ R𝐿×

z(𝑝) ∈ R

𝐿

𝐻

𝑃 ×𝑑, (7)

so that each device computes full-sequence attention over its assigned 𝐻/𝑃 heads. A second All-to-All restores the original sequence-sharded layout before the following FFN.

Exact SP-aware VAE encoding. In a naive SP pipeline, each rank either encodes the full video independently or waits for a root rank to encode and broadcast the complete latent sequence. This makes VAE latent preparation scale with the full video length on every rank, even though the following DiT sequence is already sharded. Balanced SP keeps the pre-DiT clean/noisy construction local to each rank: each rank VAE-encodes only its local raw-video chunk X(𝑝) plus a left halo that covers the temporal receptive field of the VAE encoder. After encoding, the rank discards halo latents, keeps only the local latent chunk Z(𝑝), and locally forms the matched clean/noisy latent streams. As long as the halo covers the encoder’s left temporal dependency, these local latents are identical to those obtained from full-video encoding, while the per-rank VAE cost is reduced from 𝑂(𝐹) to 𝑂(𝐹/𝑃 + ℎ) for 𝐹 latent frames, SP size 𝑃, and halo size ℎ.

Natural teacher-forcing mask. The standard teacher-forcing mask is defined over the logical DiT sequence layout [z𝑐𝑙𝑒𝑎𝑛; z𝑛𝑜𝑖𝑠𝑦], where all clean chunks are placed before all noisy chunks. However, after local paired clean/noisy construction and the Ulysses All-to-All exchange, the communication-native global order becomes

[z(0)𝑐𝑙𝑒𝑎𝑛,z(0)𝑛𝑜𝑖𝑠𝑦,...,z(𝑃−1)

𝑐𝑙𝑒𝑎𝑛 ,z(𝑃−1)

𝑛𝑜𝑖𝑠𝑦 ]. (8)

A conventional mask would therefore require an explicit permutation that gathers all clean chunks before all noisy chunks, applies attention in that logical order, and scatters the output back to the SP layout at every attention layer. We instead evaluate the original teacher-forcing visibility rule directly on the interleaved Ulysses order. Let 𝐿loc = 𝐿/(2𝑃) be the number of clean or noisy tokens contributed by each rank. For a token index 𝑖 in the interleaved order,

𝑝(𝑖) = ⌊︂

⌋︂, 𝑟(𝑖) = 𝑖 mod 2𝐿loc, 𝑡(𝑖) = 𝑝(𝑖)𝐿loc + (𝑟(𝑖) mod 𝐿loc). (9)

𝑖 2𝐿loc

Here 𝑝(𝑖) is the rank block, 𝑟(𝑖) is the within-rank offset, and 𝑡(𝑖) is the original temporal position. The condition 𝑟(𝑖) < 𝐿loc identifies clean tokens, while 𝑟(𝑖) ≥ 𝐿loc identifies noisy tokens. Thus each interleaved token has a deterministic logical identity, allowing us to define

𝑀nat(𝑖,𝑗) = 𝑀TF(𝜋(𝑖),𝜋(𝑗)), (10)

where 𝜋(·) denotes the recovered clean/noisy identity and temporal position. 𝜋 is never materialized on Q/K/V tensors; the block-sparse mask predicate computes it from token indices and flex_attention compiles the predicate into the fused attention kernel. This preserves the conventional teacher-forcing visibility while keeping attention in the communication-native SP order.

SP-aware error recycling. A pure teacher-forcing setup still leaves residual exposure bias: during training, the clean prefix is drawn from ground truth, while at inference it consists of the model’s own rollout. We therefore maintain an error-recycling buffer of past latent prediction errors and stochastically inject them into z𝑐𝑙𝑒𝑎𝑛 during training [38]. Under Balanced SP, this buffer must follow the same temporal partition as the DiT sequence; otherwise errors from one SP rank would be replayed at positions that are not reachable by another rank.

Concretely, we use a two-dimensional bucket layout indexed by local block position and diffusion timestep. The position dimension is sharded by SP: if the full sequence has 𝑁blk temporal blocks and SP size 𝑃, each rank stores only 𝑁blk/𝑃

###### Training Time per Iteration

###### Training GPU Memory Cost

102.3

142.7

100

140

###### SP TP DP

###### SP TP DP

×𝟐.3

120

###### Per-GPUMemory(GB)

80

101.7

97.8

###### IterationTime(s)

###### ×3.9

100

60

80

70.3

50.2

62.9

60

52.8 51.2

43.9

40

40.4

29.8

38.8

40

26.5

30.4

23.1

15.5

16.2 13.2

20

20

6.0 4.4 5.7

1.8 2.5

0

128

256 512 768

0

128

256 512 768

Training Video Frames

Training Video Frames

Figure 8 | Iteration speed and peak memory for sequence parallelism (SP), tensor parallelism (TP), and data parallelism (DP) in interactive AR video generation training on 4 NVIDIA GB200 GPUs. Left: iteration speed. Right: peak memory. SP is fastest at all tested sequence lengths and becomes the most memory-efficient method at long contexts.

local block positions together with its global block offset. This preserves the position-dependent nature of rollout errors while reducing per-rank buffer memory. For context corruption, the clean prefix error is sampled by matching the local position and marginalizing over timestep, since rollout errors accumulate across the denoising trajectory. For latent and noise corruption, both local position and timestep are matched.

During warming-up, we gather buffer entries across data-parallel ranks with the same SP rank, rather than across the full world group. This fills each local position bucket faster using different batch samples, while avoiding cross-SP communication whose positions would be invalid for the current rank. We shard timestep buckets by SP rank and save one buffer checkpoint per SP rank, which keeps checkpoint size bounded and prevents position-bucket misalignment when resuming training.

Figure 8 compares iteration speed and peak memory for sequence parallelism (SP), tensor parallelism (TP), and data parallelism (DP) in interactive AR training with 4 NVIDIA GB200 GPUs. SP is consistently the fastest, yielding 1.12×–1.41× speedup over TP and 3.40×–3.86× over DP. TP is slightly more memory-efficient at short contexts, but SP becomes the most memory-efficient at long contexts, reducing peak memory to 51.24/62.85 GB at sequence lengths 128/192, compared with 70.26/101.70 GB for TP and 97.75/142.69 GB for DP. Overall, SP provides the best throughput and memory scaling for long-context interactive AR training.

### D. Sequence Parallelism Inference

To reduce memory and latency during extreme long-video generation, we extend the DeepSpeed-Ulysses [29] sequence parallelism (SP) strategy from training (§ 2.1) to inference, as illustrated in Figure 9. Let 𝑃 denote the SP group size, 𝐿 the total token sequence length, 𝐻 the number of attention heads, and 𝑑 the head dimension. Although SP reduces the per-device memory footprint to 𝒪(𝐿/𝑃), its efficiency is bottlenecked by the All-to-All communication required to transpose the sequence and head dimensions before attention. In a standard BF16 pipeline, exchanging the Query (Q), Key (K), and Value (V) tensors incurs a payload of 𝒪(𝐿 × 𝐻 × 𝑑 × 2 bytes) per layer, which heavily stresses interconnect bandwidth.

To mitigate this bottleneck, we combine SP with NVFP4 communication. Since the historical K and V tensors are already retrieved from the chunkwise NVFP4 KV cache (§ 3.1) in compressed form, we also cast the runtime Q to NVFP4 immediately before the pre-attention All-to-All. Thus, for any tensor M ∈ {Q,K,V}, communication is performed entirely in the low-precision space:

𝐿 𝑃 ×𝐻×𝑑 (NVFP4)

𝐻 𝑃 ×𝑑

−−−−−→All-to-All M̃︁(𝑝) ∈ R𝐿×

M(𝑝) ∈ R

(NVFP4) , (11)

where 𝑝 ∈ {0,...,𝑃 − 1} is the device index. Executing this transposition natively on NVFP4 data reduces the effective payload from 16 bits to roughly 4.5 bits per element. After accounting for micro-block scaling overhead, the empirical communication volume is reduced by roughly 3.6×. This NVFP4-accelerated collective alleviates the bandwidth bottleneck and improves the scalability of SP for long-context AR inference.

- Table 6 | SP inference latency and communication overhead on NVIDIA H100 GPUs. We report end-to-end generation latency and total communication time for BF16 and 4-bit KV-cache settings across different SP group sizes and video lengths. The 64s numbers are estimated from measured shorter-length runs.

16 s 32 s 64 s SP Size

E2E Gen.↓ (s)

Comm.↓ (s)

E2E Gen.↓ (s)

Comm.↓ (s)

E2E Gen.↓ (s)

Comm.↓ (s)

KV Precision

- 1 BF16 31.0 – 50.2 – 85.0 –

- 2 BF16 19.3 1.8 38.1 3.2 62.5 5.4 2 4-bit KV Cache 18.3 1.1 36.0 2.3 53.3 3.6 4 BF16 26.2 12.8 38.6 12.2 65.4 20.6 4 4-bit KV Cache 21.1 7.8 32.3 9.7 54.8 16.4

###### NVFP4 Inference & Sequence Parallelism

More generally, SP is compatible with a broad range of compression techniques beyond NVFP4 KV-cache quantization, including other low-bit KV compression schemes [60] and attention-pruning methods such as TriAttention [49]. These methods are complementary to SP: by reducing the tensors exchanged around the pre-attention communication path, they can further lower communication overhead and accelerate inference on non-Blackwell GPUs. We leave a systematic comparison of these alternatives to future work.

| |[B, L, N, D]| |
|---|---|---|
|GPU 0|GPU 1 GPU 2|GPU 3<br><br>[B, L/P, N, D]|

Transformer Layer x𝑁

|Linear Layer|
|---|

|Linear Layer|
|---|

|Linear Layer|
|---|

|Linear Layer|
|---|

KV Cache (Quantized)

Quantized Communication

All-to-All Communication

[B, L, N/P, D]

GPU 0 GPU 1 GPU 2 GPU 3

|Attention|
|---|

|Attention|
|---|

|Attention|
|---|

|Attention|
|---|

All-to-All 2x Speedup

All-to-All Communication

[B, L/P, N, D]

|NVFP4 GEMM|
|---|

2-4x Speedup

GPU 0 GPU 1 GPU 2 GPU 3

|Linear Layer|
|---|

|Linear Layer|
|---|

|Linear Layer|
|---|

|Linear Layer|
|---|

Figure 9 | Sequence Parallelism (SP) Inference. Inference uses a W4A4 model with SP. KV-cache quantization significantly reduces communication overhead during the All-to-All exchange.

- Table 6 verifies that SP inference also provides a practical acceleration path on nonBlackwell GPUs. On H100, moving from single-GPU inference to SP=2 reduces BF16 end-to-end latency from 31.0s/50.2s/85.0s to 19.3s/38.1s/62.5s for 16s/32s/64s videos, respectively. Quantizing the KV cache further reduces the tensors exchanged by SP collectives, cutting communication time from 1.8s to 1.1s for 16s videos at SP=2 and from 12.8s to 7.8s at SP=4. This translates into lower end-to-end latency across the reported lengths, showing that low-bit KV cache compression is an effective way to mitigate the communication overhead introduced by multi-GPU SP inference.

### E. Visual Ablation of Multi-Shot Attention Sink

- Figure 10 provides a qualitative ablation of the multi-shot attention sink introduced in § 4.2. Without the multi-shot sink, sliding-window generation can lose shot-local anchors once earlier frames leave the active KV window, causing the later part of a shot to drift in subject appearance and scene layout. With the proposed multi-shot attention sink, the global sink preserves video-level identity while the shot-level sink keeps the current shot anchored, producing a more stable continuation from the start to the end of the second shot.

### F. Scale Search NVFP4 Quantization

The DMD teacher is quantized for W4A4 NVFP4 inference. For teacher weights, we adopt Four Over Six (4/6) adaptive block scaling [12]. NVFP4 stores each value in the E2M1 FP4 set {0,±0.5,±1,±1.5,±2,±3,±4,±6}, together with an E4M3 FP8 scale for every 16-value block and a tensor-level FP32 scale. The standard NVFP4 rule maps the largest absolute value in each block to the largest FP4 magnitude, 6, which avoids saturation and maximizes dynamic range. However, because FP4 has non-uniform spacing, this choice creates a large representational gap near the block maximum: when the maximum maps to 6, values between roughly 4/6 and 1 of the maximum can only be rounded to 4 or 6. As a result, near-maximal values, especially those around 75% of the block maximum, can dominate the quantization error.

Four-Over-Six addresses this issue by also considering a second encoding in which the block maximum is mapped

Shot 1 Shot 2 Start Shot 2 End

[Figure 61]

[Figure 62]

[Figure 63]

w/o Multi-Shot

Attention Sink

……

[Figure 64]

[Figure 65]

[Figure 66]

with Multi-Shot

Attention Sink

……

- Figure 10 | Visual ablation of the multi-shot attention sink. Without the multi-shot attention sink, the generated content drifts. With the multi-shot attention sink stabilizes shot-level appearance.

Table 7 | LongLive-2.0 Precision Settings. We compare BF16 and W4A4 NVFP4 precision under different quantization methods on VBench. #Step means the number of denoising steps.

Precision Quant. #Step #Params Resolution Total↑ Quality↑ Semantic↑

BF16 – 4 5B 1280×720 85.06 86.67 78.63 NVFP4 PTQ 4 5B 1280×720 84.04 85.76 77.15 NVFP4 Pre-trained 4 5B 1280×720 84.51 86.43 76.81

to 4 rather than 6. This sacrifices the ability to use the FP4 values ±6 for that block, but makes the high-magnitude region more evenly represented; for example, the FP4 value 3 then corresponds to 75% of the block maximum. Since this is beneficial only for some blocks and harmful for others, the scale is selected adaptively by explicitly comparing reconstruction error.

Let U¯ 𝐵

𝑖

= U𝐵

𝑖

/𝛼FP32 denote the globally normalized values in block 𝐵𝑖. We define two candidate FP8 block scales:

𝛼𝑖FP8(6) = castE4M3(︂

max |U¯ 𝐵

𝑖| 6

)︂, 𝛼𝑖FP8(4) = castE4M3(︂

max |U¯ 𝐵

𝑖| 4

)︂. (12)

For each candidate scale, the block is quantized to E2M1 FP4 and dequantized back to the original scale. We then select the candidate with lower mean-squared reconstruction error:

𝛼𝑖⋆ = arg min

𝛼∈{𝛼FP8𝑖(6),𝛼FP8𝑖(4)}

⃦U𝐵

𝑖 − U^ 𝐵

𝑖

(𝛼)⃦

2 2

, (13)

where U^ 𝐵

𝑖

(𝛼) denotes the dequantized block under block scale 𝛼. This per-block scale search keeps the standard 6-based encoding for blocks that require larger dynamic range, while switching to the 4-based encoding for blocks whose error is dominated by near-maximal values. Because NVFP4 uses E4M3 block scales, the 4 and 6 choices can be represented with sufficient fractional precision, enabling this adaptive selection with small quantization-kernel overhead on Blackwell GPUs.

G. Ablation of NVFP4 Quantization

- Figure 11 provides a qualitative comparison between PTQ and pre-trained NVFP4. The top row shows PTQ results and the bottom row shows pre-trained NVFP4 results; the first column gives the initial frame, and the following frames compare temporal visual quality. PTQ introduces visible degradation, especially blurred eye regions, while pre-trained NVFP4 preserves sharper details.

- Table 7 further isolates the effect of W4A4 NVFP4 quantization on LongLive-2.0 under the same short-video evaluation protocol. The BF16 model serves as the full-precision reference, while direct PTQ converts the trained model to W4A4 NVFP4 only at deployment time. The results show that this direct PTQ path introduces a clear quality drop, indicating a non-negligible mismatch between BF16 training and low-precision W4A4 inference. In contrast, the pre-trained

Shot 1

Shot 3

Shot 2

[Figure 67]

[Figure 68]

[Figure 69]

Blur!

###### Blur!

……

……

PTQ

|[Figure 70]<br><br>[Figure 71]|
|---|

|[Figure 72]<br><br>[Figure 73]|
|---|

Shot 1

Shot 2

Shot 3

[Figure 74]

[Figure 75]

[Figure 76]

Distinct!

###### Distinct!

…… ……

Ours

|[Figure 77]<br><br>[Figure 78]|
|---|

|[Figure 79]<br><br>[Figure 80]|
|---|

- Figure 11 | Comparison of PTQ and Pre-trained NVFP4. Top: PTQ. Bottom: pre-trained NVFP4. The first column shows the initial frame, while the following frames compare temporal visual quality. PTQ leads to blurred eyes, whereas pre-trained NVFP4 preserves much clearer facial details.

W4A4 NVFP4 setting keeps the model aligned with the target deployment precision and remains close to the BF16 baseline, supporting our design choice of using NVFP4 as a training- and inference-aligned precision rather than only a post-training compression.

### H. DMD Training Strategies

We investigate two strategies for DMD fine-tuning to our AR video generation framework, as illustrated in Figure 12. The first strategy is to directly perform DMD fine-tuning on the AR model. Specifically, the student, teacher, and critic are all initialized from the multi-step AR DiT obtained after AR training. This design is straightforward and follows a similar spirit to Self-Forcing [26]. The second strategy is standalone LoRA injection, where the student, teacher, and critic are initialized from the original diffusion model, e.g., Wan2.2-TI2V-5B, and the AR mask is applied to the teacher during DMD training. We then train a LoRA module for DMD and inject the LoRA weights into the AR model.

Compared with direct DMD fine-tuning, standalone LoRA injection is more flexible and convenient in practice. Since the LoRA module is trained independently from a specific AR checkpoint, it can be inserted into different AR models trained on various types of video data. Moreover, this strategy allows DMD fine-tuning to be conducted in parallel with AR training, without waiting for the AR training stage to finish.

Empirically, we also observe different visual characteristics between the two strategies. For a fair comparison, we also apply LoRA with the same configuration in the direct DMD fine-tuning setting. Direct DMD fine-tuning tends to produce videos with higher contrast and a more synthetic appearance, while standalone LoRA injection yields more natural visual quality. We note that visual preference can be subjective: the higher-contrast results produced by direct DMD fine-tuning may be appealing in some cases, while we prefer the more natural visual style of standalone LoRA injection and therefore adopt it as our default strategy.

### I. Implementation Details

We build LongLive-2.0 on Wan2.2-TI2V-5B [59]. The text encoder and VAE are kept frozen throughout training. Unless otherwise stated, we use BF16 mixed precision, gradient checkpointing, and AdamW with weight decay 0.01. For the NVFP4 setting, the GEMM operands in the forward, backward, and weight-gradient paths are quantized to NVFP4, while numerically sensitive operations and optimizer states remain in higher precision.

AR training. This stage performs AR training with sequence parallelism. We train on 32 NVIDIA GB200 GPUs with

[Figure 81]

- Figure 12 | Comparison of two DMD fine-tuning strategies. (1) Direct DMD fine-tuning of the AR model. In this strategy, the student, teacher, and critic are all initialized from the multi-step AR DiT obtained after AR training. This is the most straightforward way to perform DMD fine-tuning, similar to Self-Forcing [26]. (2) Standalone LoRA injection. In this strategy, the student, critic, and teacher are initialized from the original diffusion model, e.g., Wan2.2-TI2V-5B, while the AR mask is applied to the teacher. DMD fine-tuning is then performed with LoRA. This strategy is more flexible and convenient: the resulting LoRA weights can be injected into different AR models trained on various types of video data. It also allows DMD fine-tuning to be conducted in parallel with AR training, without waiting for AR training to finish. As shown in the qualitative comparison, the two strategies lead to different visual characteristics. Direct DMD fine-tuning tends to produce videos with higher contrast and a more synthetic appearance, while standalone LoRA injection yields more natural visual quality. Therefore, we adopt the standalone LoRA injection strategy in our framework. For a fair comparison, we also use LoRA with the same configuration in the direct DMD fine-tuning setting.

SP size 4 and hybrid-full FSDP. The local batch size is 1 per SP group, with gradient accumulation 2, giving a global batch size of 16. We train for 600 iterations. The generator is optimized with learning rate 1.0×10−5 and AdamW betas (0.0,0.999). We maintain an EMA with decay 0.99 starting from step 100. NVFP4 AR training uses 1920 NVIDIA GB200 GPU hours.

DMD LoRA distillation. This stage distills the AR model with DMD while keeping the pretrained backbone frozen and training LoRA adapters. We train on 16 NVIDIA GB200 GPUs with local batch size 2 and gradient accumulation 1, giving a global batch size of 32. We train for 5000 iterations. The generator learning rate is 1.0×10−5 and the critic learning rate is 2.0×10−6, both with AdamW betas (0.0,0.999). The critic is updated every step, and the generator is updated every five steps. We use LoRA rank 128, alpha 128, dropout 0, and BF16 adapter weights. LoRA is applied to both the generator and fake-score critic, targeting Linear layers inside the causal Wan attention blocks. NVFP4 DMD LoRA distillation uses 60 NVIDIA GB200 GPU hours.

