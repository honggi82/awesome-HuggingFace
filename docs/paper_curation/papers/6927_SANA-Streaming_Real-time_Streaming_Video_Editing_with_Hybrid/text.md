[Figure 1]

2026-6-1

## SANA-Streaming: Real-time Streaming Video Editing with Hybrid Diffusion Transformer

Yuyang Zhao1*, Yicheng Pan3*, Qiyuan He1,4*, Jincheng Yu1*, Junsong Chen1,5*, Tian Ye1, Haozhe Liu1, Enze Xie1, Song Han1,2 1NVIDIA 2MIT 3THU 4NUS 5HKU *Equal contribution Project Page: https://nvlabs.github.io/Sana/Streaming

Real-time streaming video-to-video editing (V2V) is critical for interactive applications such as live broadcasting and gaming, yet it remains a formidable challenge due to the stringent requirements for temporal consistency and inference throughput. In this paper, we present SANA-Streaming, a system-algorithm co-designed framework for high-resolution, real-time streaming video editing on consumer GPUs, with the following three core designs: (1) Hybrid Diffusion Transformer architecture introduces softmax attention in part of the blocks to improve local modeling capabilities while preserving the efficiency of linear layers. (2) Cycle-Reverse Regularization is a novel training strategy that enforces semantic consistency by predicting source frames from generated content via flow matching, improving temporal consistency without requiring paired long edited videos. (3) Efficient System Co-design combines fused GDN kernels and Mixed-Precision Quantization (MPQ) optimized for the NVIDIA Blackwell (RTX 5090) architecture. By profiling real-world throughput, our MPQ maximizes Tensor Core utilization while maintaining generation quality. The resulting system achieves real-time 1280×704 resolution editing at 24 end-to-end FPS on a single RTX 5090 GPU, with the DiT core running at 58 FPS. Experimental results demonstrate that our co-design approach significantly outperforms existing SOTA methods in both temporal coherence and system throughput.

# arXiv:2605.30409v1[cs.CV]28May2026

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

[Figure 2]

###### 0s 20s 50s

[Figure 3]

[Figure 4]

[Figure 5]

RTX 5090

Source Original Video

minute-level streaming

×

1280 704 resolution

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

(c) DiT latency w.r.t. video length

Style Impressionist oil painting

RTX 5090

4 steps distilled

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Background futuristic observatory

RTX 5090 single GPU

(d) GPU memory w.r.t. video length

46.2

Baseline

[Figure 15]

[Figure 16]

[Figure 17]

+Hybrid Attn

20.0

Local tactical jacket

+GDN Kernel

16.0

24 FPS end-to-end real-time

###### +MPQ 12.6 3.7×

0 13 25 38 50

(b) Properties

(a) Generation results of di!erent editing tasks

(e) Latency breakdown of 45s videos

Figure 1 | Overview of SANA-Streaming. Our method supports temporally consistent minute-length video editing at 1280×704 resolution while maintaining bounded memory and real-time throughput on a single RTX 5090.

Working

© 2026 NVIDIA. All rights reserved.

4

#### 1. Introduction

Instruction-guided video editing aims to transform an input video according to a natural-language instruction while preserving the motion and content that should remain unchanged. Recent video editing systems have made rapid progress on short clips, driven by stronger video diffusion transformers and large-scale synthetic editing data [17, 29, 2, 14, 19, 25]. However, real-time streaming V2V editing remains substantially harder. The model must generate frames in chronological order, maintain long-range appearance consistency across hundreds of frames, and fit within the latency and memory budget of a consumer GPU. These requirements rule out many accurate but bidirectional or full-attention designs that work well for short clips. To this end, we present SANA-Streaming, a system-algorithm co-designed framework for real-time, high-resolution streaming video editing.

A natural starting point is SANA-Video [6], the efficient video generation backbone on which our system is built. SANA-Video replaces vanilla attention with linear attention and introduces block linear attention with a constant-memory state cache, making long high-resolution video generation practical on edge GPUs. Its long-video extension, LongSANA, further adapts this all-linear backbone to streaming generation through causal chunking, recurrent state reuse, and LongLive-style [34] streaming long tuning. This design is attractive for throughput, but it exposes a key limitation for V2V editing: linear attention compresses history into finite recurrent states and is less localized than softmax attention, making it harder to preserve fine-grained source correspondence across chunk boundaries. In practice, this manifests as visible chunk-to-chunk appearance jumps and temporal flicker in long causal generation (Figure 12). Conversely, replacing all blocks with softmax attention improves local modeling but becomes memory-prohibitive for long high-resolution streams (Figure 1(d)). This tension motivates a hybrid design that allocates expressive softmax attention only where it is most useful, while preserving recurrent linear-state caching in the majority of layers.

Hybrid Diffusion Transformer. The DiT model of SANA-Streaming is a hybrid diffusion transformer derived from the SANA-Video’s all-linear backbone, where we insert softmax-attention blocks evenly in the all linear backbones and replace the vanilla linear attention with an efficient variant of Gated DeltaNet (GDN) [35]. In this hybrid design, SANA-Streaming leverage GDN blocks to maintain the important global information with dynamic gate and decay. Softmax attention blocks are adopted to efficiently explore the locality and first block consistency with window attention and attention sink. Such design keeps memory constant and generation speed for arbitrary generation length while still keeping long range consistency. As shown in Figure 1, compared with all softmax attention variant, the hybrid design only requires 5.56 GB VRAM for long video generation and 3.7 times faster generation speed, enabling real-time execution on the consumer GPUs.

Cycle Consistent Streaming Training. Due to the lack of paired long video editing data, we adopt LongLive [34] as our basic streaming training framework, which uses short video generation teacher to supervise the rollout long video clips with Distribution Matching Distillation (DMD) [39, 38] loss. However, the short teacher supervision lacks the long-range information and therefore may lead drifting when supervise long rollout clips. In view of this limitation, we introduce Cycle-Reverse Regularization. Starting from the LongLive training strategy, our method adds a reverse editing objective: after generating an edited chunk, the model is required to reconstruct the corresponding source chunk conditioned on a reverse prompt. Since the long video itself is a real-world temporal consistent video, the reverse loss can force the model to learn how to maintain the long range temporal consistency. In addition to the DiT training, we also distill our VAE decoder from bidirectional convolution to causal convolution, supporting streaming decoding.

Efficient System Co-design. To enhance the generation efficiency on the consumer GPUs, we explore the inference system design. On the one hand, we improve the efficiency of frame-wise GDN kernel with partition strategy, so that the state matrices can be stored on the GPU’s SRAM, leading to 1.5-2.2x speed up on different GPU architectures. In addition, since the NVFP4 and FP8 Tensor Core is super efficient on Blackwell architecture while the BF16 precision has the best quality, we design a precision search algorithm inspired from AutoML. By evaluating the trade-off between generation quality and efficiency, we find the best mixed precision strategy for different layers and different blocks, leading to 59% speed up over BF16 on the DiT model with marginal quantization error.

Data Pipeline. The success of SANA-Streaming relies on the strong data pipeline for both short videos and long videos. For short videos, we leverage the image editing model to edit the first frame and I2V generation

###### 3x 1x

| | | |
|---|---|---|
| | | |

###### FP4 + FP8 + BF16 FP4 + FP8 FP4 + FP8 + BF16 FP4 + FP8 + BF16 FP4 + FP8 FP4 + FP8 + BF16

Linear Attention

Cross Attention

Cross Attention

Mix FFN

###### Softmax Attention

Mix FFN

Gated DeltaNet

Window + Sink

3x1x1 Temporal Conv

Output Layer

Output Layer

⨂

⨂

MatMul

1x1 Conv

MatMul

Delta Rule Update

⨂

Softmax

3D

###### 3D

SiLU

MatMul

SiLU

Decay Gate

Write Gate

SiLU

ReLU ReLU

3D

3D

1x3x3 Conv

α β g

Q g

Q

K V

K V

1x1 Conv

Linear Linear Linear

Linear

Linear Linear

Linear Linear Linear

Linear

Si = αiSi−1Δ +βiKiVi

X

X

X

Cost : O(n2d)

Cost : O(nd2)

Layer FP4 Layer Layer FP8 Layer Decayed State Mask Cached State Compute

Layer BF16 Layer

Figure 2 | Hybrid streaming diffusion transformer. SANA-Streaming interleaves GDN blocks and softmax attention blocks. The two types of blocks focus on different aspect of information with a fixed computational cost. For the detailed block design, each layer will use the most suitable precision for efficiency and generation quality.

model to obtain the pairs, as well as data verification to filter out low quality pairs. For the long video, since generating consistent pairs is difficult, we use VLM to generate forward and reverse editing prompts for streaming long training and our cycle-reverse regularization.

[Figure 18]

1

Our contributions are summarized as follows.

- • We propose a hybrid diffusion transformer for streaming video editing, combining global information from GDN blocks with strong locality information from softmax-attention blocks for efficient and high quality generation.
- • We introduce Cycle-Reverse Regularization, which leverages long source videos through a reverse flow-matching objective and improves long-range consistency without requiring paired long edited videos.
- • We propose a comprehensive editing data generation pipeline for both short video pairs and long video prompts, which is the basic of streaming video editing.
- • We develop an efficient system design with fused GDN kernels and Mixed-Precision Quantization, achieving 24 end-to-end FPS and 58 DiT FPS on an RTX 5090 GPU.

#### 2. Methodology

###### 2.1. Hybrid Architecture

This section introduces the key architectural design of SANA-Streaming for minute-length streaming video editing. Motivated by the complementary limitations of softmax attention and linear attention, we develop a hybrid diffusion transformer enabling long video editing with global consistency and local granularity under fixed GPU memory.

Limitations of Softmax Attention and Linear Attention. Causal softmax attention provides strong token-level interaction and precise source correspondence, which is crucial for high-quality video-to-video editing. However, attending to all historical high-resolution video tokens is memory-prohibitive, as the active KV cache grows with video length. In addition, as shown in Figure 3(a), softmax attention commonly focuses more on local information. Therefore, restricting softmax attention to a local window with a persistent sink token is applicable while significantly reducing memory cost in long video generation. However, global information is excluded in such attention operation. In contrast, linear attention is naturally suitable for streaming because it compresses the generated prefix into a compact recurrent state whose size is independent of the number of chunks. Nevertheless, the dense attention maps (Figure 3(b)) demonstrate that linear attention does not focus enough on the neighborhood, leading to chunk-boundary flickering (Figure 12).

We therefore combine the two mechanisms in an interleaved hybrid design as Figure 2 shows, following Qwen-Next [33]. In SANAStreaming, most blocks use Gated DeltaNet (GDN) linear attention to maintain compact global memory for consistency, while a small number of blocks use sliding window softmax attention with a persistent sink chunk and a short recent window. This preserves the efficiency of an all-linear streaming DiT while enabling fine-grained attention to local context.

- (a) Softmax Attention Maps

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

- (b) Linear Attention Maps

###### Frame-wise Gated DeltaNet for Global Accumulated Memory.

The linear-attention blocks in SANA-Streaming serve as the global memory pathway for causal streaming. Instead of token-wise updates or caching raw tokens from all previous chunks, each block maintains a compact recurrent state that is updated frame-by-frame and carried across chunks. This state summarizes the generated prefix into a fixed-size memory, enabling long-range information propagation under fixed GPU memory.

Figure 3 | Attention map visualization of different heads.

Given the latent 𝑥 ∈ R𝐹×𝑁×𝐶 in each head, we obtain the query 𝑞 ∈ R𝐹×𝑁×𝐶, key 𝑘 ∈ R𝐹×𝑁×𝐶, value 𝑣 ∈ R𝐹×𝑁×𝐶, decay gate 𝛼 ∈ R𝐹, write gate 𝛽 ∈ R𝐹×𝑁 and output gate 𝑔 ∈ R𝐹×𝑁×𝐶, where 𝐹, 𝑁, and 𝐶 denote the number of frames, number of tokens in each frame and head dimension respectively. The numerator state of the 𝑓th frame 𝑆𝑓𝑘𝑣 and denominator normalizer state 𝑆𝑓𝑧 can be updated with the delta-rule correction [35]:

𝑆𝑓𝑘𝑣 = 𝛼𝑓𝑆𝑓𝑘𝑣−1(𝐼 − 𝛽𝑓𝑘^𝑓𝑘^𝑓⊤) + 𝛽𝑓𝑣𝑓𝑘^𝑓⊤, 𝑆𝑓𝑧 = 𝛼𝑓𝑆𝑓𝑧−1(𝐼 − 𝛽𝑓𝑘𝑓𝑘𝑓⊤) + 𝛽𝑓𝑘𝑓⊤. (1)

where 𝑘𝑓 and 𝑘^𝑓 is the key before and after RoPE [22]. This update differs from simple accumulation:

1

rather than directly writing 𝑣𝑓𝑘𝑓⊤ into memory, the model only writes the residual. As a result, the recurrent state behaves as a correction-based memory, which is more robust for long streaming sequences. More details

are available in Appendix. G. The final output is then read from the updated state through a normalized readout followed by an output gate and a linear projection:

𝑜𝑓 = 𝑊𝑜 (︃𝑔𝑓 ⊙

)︃, (2)

𝑆𝑓𝑘𝑣𝑞^𝑓 (𝑆𝑓𝑧)⊤𝑞𝑓 + 𝜖

where 𝑞𝑓 and 𝑞^𝑓 is the query before and after RoPE [22]. During streaming inference, each Gated DeltaNet block only caches the terminal recurrent states (𝑆𝑘𝑣,𝑆𝑧) of the previous chunk and uses them to initialize the next chunk. Therefore, the memory cost of the linear-attention pathway remains independent of the number of streamed chunks. In our hybrid architecture, these Gated DeltaNet blocks provide a compact global accumulated memory.

Softmax blocks for local window and sink refinement. GDN blocks provide efficient global memory and softmax attention blocks help for more locality information. Instead of using all the frames for attention, each chunk attends to a constrained context consisting of itself, a persistent sink, and a local window. The sink chunk provides a stable visual anchor shared across the stream, while the recent window preserves high-resolution local evidence from neighboring chunks. This design restores precise local matching with fixed GPU memory.

###### 2.2. Long-term Consistency Training

Streaming long training. We train SANA-Streaming under the same causal streaming protocol used during inference. Given a long source video and an editing prompt, the model generates edited latent chunks autoregressively. Upon generating each chunk, the streaming cache is updated with the generated chunk, and a LongLive-style distribution matching distillation (DMD) self-forcing loss [34, 15] is applied to the current chunk. mitigates the discrepancy between short-clip training and minute-length streaming inference, enabling the model to learn from its own generated history.

[Figure 24]

Transfer the background to a modern medical bay.

|[Figure 25]| | |
|---|---|---|
| | |Noisy Edited Video|
| | | |

[Figure 26]

Self-Forcing DMD Loss

[Figure 27]

| | |
|---|---|
| | |

Fake Score Model

Edited Video

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Source Video

⊕

Student Model

[Figure 34]

[Figure 35]

[Figure 36]

Real Score Model

[Figure 37]

[Figure 38]

[Figure 39]

Streaming Long Training

[Figure 40]

Noisy Source Video Velocity to Source Video

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

###### Student Model

[Figure 48]

Flow Matching Loss

⊕

Source Video

[Figure 49]

[Figure 50]

[Figure 51]

Transfer the background to a traditional hospital.

Cycle-reverse Regularization

Figure 4 | Streaming Long Training and Cycle-Reverse Regularization. The forward streaming branch uses streaming long training with DMD loss. The reverse branch reuses the generated edited chunk as visual condition and applies a reverse instruction to reconstruct the original source chunk with a flow-matching loss, which improves temporal consistency without paired long edited targets.

Cycle-reverse regularization. While streaming long training improves long-horizon generation, videoto-video editing further requires the edited stream to preserve the source structure and motion over time. Although minute-length source videos are relatively easy to collect, paired target videos that faithfully follow the editing instruction are rarely available. We therefore introduce a cycle-reverse regularization strategy that uses only long source videos without paired edited videos.

[Figure 52]

1

Specifically, the model first performs a forward streaming edit from the source video to the target domain according to the editing prompt. Then, the generated edited chunk is used as the visual condition for a reverse edit, guided by a reverse prompt that describes how to recover the original source domain, as shown in Figure 4. The reverse pass is trained with a flow-matching objective toward the original source chunk. Overall, streaming long training teaches the model to follow editing instructions under causal rollout and cycle-reverse regularization encourages the edited video to preserve long-term structure and motion consistency learned from real-world source videos.

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

Bidirectional Tokens

Causal VAE Distillation. The video VAE decoder is another bottleneck for streaming video editing. The original LTX2 VAE decoder uses bidirectional temporal context, while streaming inference cannot access future latent frames. We convert it into a causal decoder by replacing symmetric temporal padding with left-only padding, so each output frame only depends on current and previous latent frames. For stable initialization, we reuse the pretrained LTX2 VAE weights and remap each temporal 3-tap convolution from bidirectional weights [𝑎,𝑏,𝑐] to causal weights [0,𝑎,𝑏+𝑐], preserving the previous-frame and current-frame roles while folding the unavailable future-frame contribution into the current frame, as shown in Figure 5. We then distill the causal decoder from the original bidirectional decoder. The training objective combines Charbonnier reconstruction loss [3], perceptual loss [18], Haar wavelet high-frequency loss [37, 41], and intermediate decoder feature distillation [45]:

### 1 2 3 4

Bidirectional Tokens

Causal Tokens

### 1 2 3 4 1 2 3 4 a b c 0 a

Remap

### a b c

### b +

Remap

### c

Bidirectional Conv Weight

Causal Tokens

Bidirectional Conv Weight

Causal Conv Weight

### 1 2 3 4

Figure 5 | Remapping weights in VAE.

### b +

### 0 a

### c

Causal Conv Weight

Done

ℒvae = 𝜆charbℒcharb + 𝜆percℒperc + 𝜆haarℒhaar + 𝜆distillℒdistill. (3)

###### 2.3. Efficient System Co-design

Efficient GDN Kernel. The recurrent GDN update is algorithmically efficient, but a literal implementation is not hardware efficient. For both H100 and RTX 5090, the recurrent state (𝑆𝑓,𝑧𝑓) is small enough to keep in on-chip storage, while the per-frame 𝑄𝑓,𝐾𝑓,𝑉𝑓 activations span hundreds of spatial tokens and must be streamed from HBM. Following the IO-aware principle of FlashAttention [10, 9], we tile the spatial dimension so that each block of activations is loaded once, reduced into compact frame-level summaries, and then

- FB1,-2
- FB2,1

FB1,-1

- Latent Frame in Block 1
- Latent Frame in Block 2

5

Zero Padding

Cached FB1,-1 FB2,0

KV Cache

[Figure 54]

[Figure 55]

(a) Per-layer quantization policy search (b) Per-block and mixed quantization policy search

Figure 6 | Mixed-Precision Quantization Policy Search on Relative RMSE. For each subfigure, the x-axis is the estimated speed up and the y-axis is the quantization error over the BF16 reference. The selected mixed-precision policy achieves the best trade-off between efficiency and quality.

discarded. We rewrite the update by separating the spatial reduction from the temporal recurrence:

𝑃𝑓 = 𝐼 − 𝐾𝑓⊤ diag(𝛽𝑓)𝐾𝑓, 𝐴𝑓 = 𝐾𝑓⊤ diag(𝛽𝑓)𝑉𝑓, (4) 𝑆𝑓 = 𝛼𝑓𝑃𝑓𝑆𝑓−1 + 𝐴𝑓. (5)

The vector path computes 𝑃𝑓𝑧 = 𝐼 − 𝐾𝑓⊤ diag(𝛽𝑓)𝐾𝑓 and 𝑏𝑓 = 𝐾𝑓⊤𝛽𝑓, then 𝑧𝑓 = 𝛼𝑓𝑃𝑓𝑧𝑧𝑓−1 + 𝑏𝑓. This decomposition exposes most of the work as frame-parallel: 𝑃𝑓,𝐴𝑓,𝑃𝑓𝑧,𝑏𝑓 are computed independently per frame and head, while only the compact scan over (𝑆𝑓,𝑧𝑓) remains sequential.

Our Triton implementation uses a three-phase chunkwise pipeline. Phase A computes the frame summaries with blocked reductions over the spatial dimension. Phase B performs the short recurrent scan over frames while keeping the state in on-chip storage. Phase C streams 𝑄𝑓 blocks and applies the compact states to produce the output. For bidirectional GDN, we exploit the linearity of Phase C and combine the forward and reverse histories before the output pass, avoiding a second output kernel and a second read of 𝑄𝑓. This chunkwise design consistently gives large layer-level speedups and roughly 1.5×–2.2× end-to-end sampling speedups over the PyTorch reference.

Mixed-Precision Quantization Policy Search. Quantization is further explored in this paper to support the efficiency on consumer GPUs. NVFP4 quantization is fast on Blackwell architecture but not uniformly reliable, for two main reasons: (1) some layers are more sensitive and should remain in higher precision, such as the patch embedding layer and output layer; (2) some layers has very few parameters with small computational cost, so the quantization overhead exceeds the kernel speedup, such as gate layer. We therefore search at layer-type and block-position granularity instead of assigning the same precision to the whole DiT. We first set the most sensitive or unsupported layers to BF16, including the patch embedding layer, output layer, timestep embedding layer, the gates in attention blocks and depth-wise convolution in MixFFN. For the remaining layers, we start from an FP8 base and apply group wise NVFP4 on it. The searchable FP4 groups include per-layer groups: self-attention 𝑄/𝐾/𝑉/𝑂, cross-attention 𝑄/𝐾𝑉/𝑂, FFN input and output projections, and temporal FFN projections; block-wise groups: shallow (0-5), middle (6-13) and deep blocks (14-19). For each candidate 𝜋, we run a 30-second streaming video generation with the same noise on a calibration set. Let 𝑥^ref0 be the BF16 output latent and 𝑥^𝜋0 be the output latent under policy 𝜋. We score the candidate by latent relative RMSE and LPIPS [18]:

√︁

⃦𝑥^𝜋0 − 𝑥^ref0 ⃦2

1 𝑁

2

RelRMSE(𝜋) =

. (6)

(︀

)︀

𝑥^ref0

Std

+ 𝜖

- Figure 6 and Figure 11 illustrate how each group influences efficiency and quality. We make the following observations. First, the key and query of self-attention (SA-K, SA-Q) and the output projection of cross-

- 1. Edit Prompt and First Frame Generation
- 2. In-Context Video Generation
- 3. Long Video Edit Instruction

4. VLM Verification

Qwen3VL

|Instruction Alignment|9|
|---|---|
|Non-edit Consistency & Temporal Consistency|8|
|Physical Plausibility|8|
|Video Quality|9|
|Overall|8.5|

Source Video

Edit Instruction

Video Caption

Edited Video

Local Edit

Background

Taxonomy of Editing

Style Transfer

Composite

Edited First Frame

| | |
|---|---|
| | |

Video Caption

A cinematic video in a luxurious, photo-

realisticstyle… ControlNet

Video Generation

[Figure 56]

First Frame

[Figure 57]

[Figure 58]

Source Video

Replace the traditional Chinese attire with a tailored midnight-blue tuxedo featuring satin lapels and a matching pocket square…

Edit Instruction

Qwen3VL

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Inverse Edit Instruction

Transfer the background to a traditional hospital…

Edited First Frame

[Figure 63]

Pose Video

[Figure 64]

[Figure 65]

[Figure 66]

Edited Video

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

VLM Verifier

5. Long Video Verification

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

|Black_Ratio| |0.988|
|---|---|---|
|Mean_STD| |18.551|
| | | |

Long Source Video

Figure 7 | Data Pipeline. For short videos, we sample taxonomy-guided edit instructions, edit the first frame as a visual anchor, and generate edited videos by conditioning on the source video, edited first frame, caption, and pose video. For long videos, we generate paired forward and inverse instructions from source videos to support streaming long training and cycle-reverse regularization. Both short and long samples are filtered by VLM-based verification, and long videos are additionally screened for invalid black-frame segments.

attention (CA-O) sit nearly on top of the FP8 baseline while still delivering a ∼2.06× speedup, so they are the safest layers to demote and form our Basic Mix, joined by FFN-Temporal – the only FFN component whose quantization error is negligible relative to its efficiency gain. Second, the remaining attention projections (SA-V, SA-O, CA-Q) move noticeably above the FP8 horizontal, indicating that the value and post-mixing pathway is sensitive to FP4 noise; we keep them in FP8. Third, CA-KV sits closest to FP8 on both axes but covers only 1% of the DiT FLOPs, so demoting it brings little speedup and we exclude it from the policy. Fourth, the FFN input/output point-wise convolutions are the heaviest modules in the network (∼49% of FLOPs combined); demoting them globally collapses quality. However, the block-range sweep reveals a clear depth ordering: shallow blocks are the most fragile, while deep and middle blocks tolerate FP4 well, opening a depth-restricted demotion path. To this end, the resulting policy (red star in Figure 6) uses FP4 for the most robust groups, including cross-attention output, self-attention query and key, temporal FFN projection in all blocks as well as FFN input and output projections in middle and deep blocks. The unsupported and most sensitive layers listed at the start of this section (patch embedding, output projection, timestep embedding, attention gates, MixFFN depth-wise convolution) remain in BF16, and all other linear layers not assigned to FP4 stay in FP8. The mixed-precision policy gains 3x efficiency on the quantized layers and 1.59x efficiency on DiT latency over BF16 baseline. The detailed analysis of each layer and block are available in Appendix A.

3. Data Pipeline

Since one of the main downstream tasks of SANA-Streaming is live broadcasting, we introduce our data pipeline to further generate high-quality human-centric video editing pairs as well as the editing prompts for long videos. Figure 7 summarizes the complete pipeline, including short-video pair construction, long-video instruction construction, and VLM-based filtering. The details of our data generation pipeline are available at Appendix E.

4. Experiments

- 4.1. Implementation Details

###### Edit Instruction

Transfer the background to a modern medical bay…

[Figure 77]

1

SANA-Streaming is a 2B hybrid diffusion transformer with LTX2 VAE [13] (compression ratio 32×32×8). The DiT contains 20 transformer blocks with evenly inserted 5 softmax attention blocks and 15 GDN blocks. The condition video latent is concatenated to the latent on the channel dimension. The model is trained by two stages on 1280×704 resolution: bidirectional short training and streaming long training. In the bidirectional short training stage, we use bidirectional GDN with forward and backward scan. In the streaming long training stage, 3 latent frames are regarded as a chunk. The forward scan with previous cache and backward scan within the chunk is adopted by GDN. The streaming model is distilled to 4 steps. With the algorithm and system co-design, SANA-Streaming achieves 24 end-to-end FPS and 58 DiT FPS on an RTX 5090 GPU. For

Table 1 | OpenVE-Bench comparison on the five spatially aligned edit categories used in our evaluation. The color and color highlights indicate the best and second-best quality scores, respectively. Latency is measured

- with batch size 1, while throughput is measured with batch inference on eight H100 GPUs for 81-frame 1280×704 videos. † indicates the step distilled model.

Latency Throughput

Global BG Local Local Local (s) (FPS) Style Change Change Remove Add

Methods #Params. #Reso.

Avg.

VACE [17] 14B 1280x720 1991 0.3 1.57 1.49 1.55 2.07 1.46 1.26 OmniVideo [24] 1.3B 640x352 — — 1.19 1.11 1.18 1.14 1.14 1.36 InsViE [29] 2B 720x480 750 0.9 1.45 2.20 1.06 1.48 1.36 1.17 Lucy-Edit [25] 5B 1280x704 97 6.7 2.22 2.27 1.57 3.20 1.75 2.30 ICVE [19] 13B 384x240 6051 0.2 2.18 2.22 1.62 2.57 2.51 1.97 DITTO [2] 14B 832x480 1971 0.3 2.13 4.01 1.68 2.03 1.53 1.41 OpenVE-Edit [14] 5B 1280x704 97 6.7 2.50 3.16 2.36 2.98 1.85 2.15

SANA-Streaming 2B 1280x704 20 32.4 2.62 3.48 2.29 3.20 2.27 1.88 SANA-Streaming † 2B 1280x704 1 762.8 2.42 3.60 1.82 3.10 1.78 1.82

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Source Video

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

LucyEdit

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Ditto

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

###### Ours

Transform the entire video to resemble a vivid oil painting, enhancing the textures and brush strokes for a painterly e!ect

Change the background to grass

- Figure 8 | Qualitative comparison with prior video editors. SANA-Streaming preserves source motion and non-edited content while following the edit instructions.

quantitative evaluation, we use the five OpenVE-Bench [14] pixel-aligned categories: global style, background change, local change, local remove, and local add.

###### 4.2. Performance Comparison and Analysis

- Table 1 compares SANA-Streaming with representative instruction-guided video editing systems on the five OpenVE-Bench categories used in our evaluation. SANA-Streaming first stage model is a bidirectional model on short videos, and it achieves the state-of-the-art performance on this benchmark, with nearly 2.5x smaller model size and 5x faster throughput than the previous best method (OpenVE [14]). The streaming and distilled version of our model targets deployment on consumer GPUs. However, even with distillation, SANA-Streaming can still achieve comparable performance with previous methods and has more than 100x higher throughput. Qualitative comparisons are shown in Figure 8 and more qualitative long video results of SANA-Streaming are available in Figure 13.

2

###### 4.3. Ablation Studies

Efficient System Co-design. Efficient system co-design is crucial to achieve real-time end-to-end generation on consumer GPUs. In Table 2, we break down the deployment optimizations over the BF16 baseline. Specifically, the lossless efficient GDN kernel improves the DiT latency by 22%, and Mixed-Precision Quantization (MPQ) further reduces the DiT latency to 16.8s, yielding a 1.59× DiT speedup over the BF16 baseline and achieving 24 FPS for end-to-end generation. These results demonstrate the importance and effectiveness of

- Table 2 | Latency (s) analysis of the efficient system codesign on an RTX 5090 for 1-minute video.

###### Table 3 | Causal VAE Distillation Comparison.

Methods PSNR LPIPS SSIM Bidirectional 32.98 0.0274 0.923 Causal (Before Training) 24.66 0.132 0.785 Causal (After Training) 32.14 0.0336 0.911

System VAE VAE DiT End-to-end DiT Setting Encode Decode Latency Latency Speedup

BF16

26.8 50.7 1.00× + GDN Kernel 21.9 45.8 1.22× + MPQ 16.8 40.7 1.59×

15.4 8.5

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Source Video

- Case 1, 1, 15,25
- Case 2, 5, 25, 50

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

w/o Reg.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

w/ Reg.

Change the turtle's shell material to crystal… Change the tree bark to look like it's made of crystal…

- Figure 9 | Effect of Cycle-Reverse Regularization. Adding cycle-reverse regularization improves the preservation of source motion and non-edited regions while keeping the video temporally stable.

our algorithm-system co-design.

Cycle-Reverse Regularization. Cycle-Reverse Regularization aims to improve temporal consistency when the paired long videos are not available. As shown in the first case of Figure 9, cycle-reverse regularization can keep the non-edited region consistent with the source video. In the second case, the edited “crystal tree” changes in the later frames when regularization is not applied but our model keeps it consistent throughout the whole long video.

Causal VAE. We compare the fine-tuned causal VAE decoder against the original decoder with causal and non-causal decoding in Figure 10. Direct causal conversion introduces visible blur and loss of fine detail because the decoder no longer has access to future latent frames. After causal decoder distillation, the streaming VAE recovers sharper textures and object boundaries, achieving comparable performance with the bidirectional teacher (Figure 10 and Table 3).

1

#### 5. Related Work

Hybrid Linear-Softmax Architectures. Diffusion Transformers (DiTs) have become the dominant paradigm for high-quality image [31, 32, 5, 37, 28] and video generation [27, 12, 13]. However, full attention incurs quadratic complexity, leading to prohibitive memory and latency costs for long sequences. To improve efficiency, prior work such as SANA-Video [6, 43] replaces full attention with linear attention and SLA [42] replaces softmax attention with sparse attention and linear attention. While effective for scaling, pure linear attention often degrades fine-grained modeling, especially for short-range dependencies. Recent LLM studies suggest that neither full nor linear attention alone achieves the best efficiency–quality tradeoff. Hybrid designs therefore interleave linear and softmax attention to combine scalable context modeling with periodic refinement, as seen in Kimi Linear [26] and Qwen3-Next/3.5 [33, 20]. Inspired by the success of hybrid architectures in LLMs, we introduce a hybrid diffusion transformer that combines the local modeling strength of softmax attention with the global recurrent memory of linear attention for efficient, high-quality video generation.

Video Editing. Instruction-guided video editing has recently shifted from tuning-free attention manipulation and mask-based control toward large supervised video editing models. VACE [17] unifies video creation and editing through a video condition interface and context adapter. InsViE [29] and Ditto [2] emphasize scalable synthetic paired-data construction for instruction-based editing. OpenVE [14] further introduces

[Figure 122]

SANA-Streaming : Real-time Streaming Video Editing with Hybrid Diffusion Transformer

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

| |
|---|

| |
|---|

| |
|---|

Ground Truth

| |
|---|

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

| |
|---|

| |
|---|

| |
|---|

Teacher Bidirectional

| |
|---|

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Before Causal Training

| |
|---|

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

| |
|---|

| |
|---|

| |
|---|

Causal Trained

| |
|---|

- Figure 10 | Causal VAE distillation. Our decoder is comparable with the bidirectional teacher model.

OpenVE-3M and OpenVE-Bench, a unified benchmark with diverse spatially aligned and non-spatially aligned editing categories. Other recent systems, including OmniVideo [24], ICVE [19], and Lucy-Edit [25], improve instruction following and local edit fidelity through stronger multimodal conditioning, in-context learning, or open-weight video editing backbones. These methods mainly target offline short-clip editing. In contrast, SANA-Streaming focuses on real-time streaming V2V editing and achieves real-time video editing on a consumer GPU.

Long Video Generation. With the development of stronger video generation models, long video generation has attracted increasing attention. Diffusion Forcing [4], Self-Forcing [15], and Causal-Forcing [44] explore causal and autoregressive generation paradigms for producing long videos beyond the context length of pretrained short-video generation models. Furthermore, LoL [8] studies the misbehavior of RoPE in long-video generation and proposes scaling strategies for longer temporal contexts. LongLive [34, 7] shows that causal frame-level generation, KV recaching, streaming long tuning, and short-window attention with a frame sink can enable real-time interactive long video generation. SANA-Streaming uses LongLive as the base streaming training framework and proposes cycle-reverse regularization to further improve temporal consistency.

#### 6. Conclusion

1

We present SANA-Streaming, a system–algorithm co-designed framework for real-time, high-resolution streaming video-to-video editing on consumer GPUs. By combining a hybrid diffusion transformer with Cycle-Reverse Regularization and an efficient system design, our approach achieves a favorable balance between temporal consistency, visual quality, and inference efficiency under streaming constraints. Empirical results demonstrate that SANA-Streaming enables minute-length video editing at real-time speed while maintaining competitive editing quality. This work provides a practical step toward interactive video editing systems and highlights the importance of jointly addressing modeling, training, and system challenges in long-form generative tasks.

Limitations. While Cycle-Reverse Regularization mitigates the scarcity of paired data, the shortage of high-quality long video editing samples still hinders temporal consistency in complex scenarios. In addition, like other instruction-guided generative models, SANA-Streaming may produce inconsistent or incorrect edits under ambiguous or underspecified instructions. The model lacks explicit mechanisms to resolve ambiguity or guarantee faithful execution of user intent, which can lead to unpredictable outputs in challenging cases.

#### References

- [1] aigc apps. Videox-fun: A video generation pipeline for diffusion transformer, 2026.
- [2] Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, et al. Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742, 2025.
- [3] Pierre Charbonnier, Laure Blanc-Feraud, Gilles Aubert, and Michel Barlaud. Two deterministic halfquadratic regularization algorithms for computed imaging. In Proceedings of 1st international conference on image processing, volume 2, pages 168–172. IEEE, 1994.
- [4] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [5] Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. Sana-sprint: One-step diffusion with continuous-time consistency distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16185–16195, 2025.
- [6] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.
- [7] Yukang Chen, Luozhou Wang, Wei Huang, Shuai Yang, Bohan Zhang, Yicheng Xiao, Ruihang Chu, Weian Mao, Qixin Hu, Shaoteng Liu, Yuyang Zhao, Huizi Mao, Ying-Cong Chen, Enze Xie, Xiaojuan Qi, and Song Han. Longlive-2.0: An nvfp4 parallel infrastructure for long video generation, 2026.
- [8] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Lol: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026.
- [9] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.
- [10] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344– 16359, 2022.
- [11] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.
- [12] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [13] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.
- [14] Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie. Openve-3m: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025.
- [15] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [16] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674, 2023.
- [17] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17191–17202, 2025.

- [18] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European conference on computer vision, pages 694–711. Springer, 2016.
- [19] Xinyao Liao, Xianfang Zeng, Ziye Song, Zhoujie Fu, Gang Yu, and Guosheng Lin. In-context learning with unpaired clips for instruction-based video editing. arXiv preprint arXiv:2510.14648, 2025.
- [20] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.
- [21] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366. PMLR, 2021.
- [22] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.
- [23] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.
- [24] Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-video: Democratizing unified video understanding and generation. arXiv preprint arXiv:2507.06119, 2025.
- [25] DecartAI Team. Lucy edit: Open-weight text-guided video editing. 2025.
- [26] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, et al. Kimi linear: An expressive, efficient attention architecture. arXiv preprint arXiv:2510.26692, 2025.
- [27] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [28] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [29] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16692–16701, 2025.
- [30] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.
- [31] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.
- [32] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.
- [33] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [34] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.
- [35] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024.
- [36] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2023.

- [37] Tian Ye, Song Fei, and Lei Zhu. Ultraflux: Data-model co-design for high-quality native 4k text-to-image generation across diverse aspect ratios. arXiv preprint arXiv:2511.18050, 2025.
- [38] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.
- [39] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.
- [40] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.
- [41] Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, and Di Huang. Diffusion-4k: Ultra-high-resolution image synthesis with latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23464–23473, 2025.
- [42] Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, et al. Sla: Beyond sparsity in diffusion transformers via fine-tunable sparse-linear attention. arXiv preprint arXiv:2509.24006, 2025.
- [43] Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. Sana-wm: Efficient minute-scale world modeling with hybrid linear diffusion transformer. arXiv preprint arXiv:2605.15178, 2026.
- [44] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.
- [45] Ya Zou, Jingfeng Yao, Siyuan Yu, Shuai Zhang, Wenyu Liu, and Xinggang Wang. Turbo-vaed: Fast and stable transfer of video-vaes to mobile devices. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

#### A. Mixed-Precision Quantization

Search setup and metrics. Quality. For each candidate policy, we run the full 30-second streaming generation under the same input noise as the BF16 reference on 45 calibration prompts, and compare the produced output latent against the BF16 output latent. We report the relative RMSE and LPIPS averaged over the calibration set (lower is better), where RelRMSE normalizes the latent RMSE by the BF16 reference’s standard deviation so that errors are comparable across prompts of different scale. Compute accounting. The DiT processes the 30-second video in chunks of 3 latent frames, so a single forward pass operates on a chunk of 3 × 22 × 40 latent tokens. The FLOP totals in Table 4 are for this single per-chunk forward, computed as

∑︀

ℓ 2 · inℓ · outℓ · 𝑁ℓ over every quantizable linear / point-wise convolution, with 𝑁ℓ the tokens seen by layer ℓ (𝑁 = 2640 for self-attention and FFN, 𝑁 = 300 for cross-attention key/value over text tokens). The full DiT then executes ∼9.84T FLOPs per chunk; reporting per-chunk FLOPs (rather than per-video) is sufficient because the FP4 / FP8 fractions and the resulting speedup factors are invariant to the chunk count. Columns. #Params FP4 and FLOPs are the parameters (M) and per-chunk FLOPs (G) demoted to FP4 in each configuration; the remaining linear layers stay in FP8. %Param / %FLOPs are the corresponding shares of the full DiT. Speedup is the idealized roofline factor over BF16, so all-FP8 gives 2x and all-FP4 gives 4x. To collapse quality and efficiency into a single scalar, we report Cost over Speedup = RMSE/Speedup: the FP8 ratio of 0.23 is the reference that guides us to identify the best mixed-precision policy.

- Table 4 | Mixed-precision search summary on the SANA hybrid DiT (2B parameters, 9.84T FLOPs in a chunk generation of 3 × 22 × 40 latent).

#Params FLOPs

Cost over FP4 (M) (G) Speedup ↓

Config / module

%Param %FLOPs RMSE ↓ LPIPS ↓ Speedup ↑

Baselines

FP8 0 0 0% 0% 0.45 0.10 2.00× 0.23 FP4 2040 9840 100% 100% 1.04 0.35 4.00× 0.26

Single-group FP4 (only this group → FP4, rest FP8)

FFN-Input 602 3179 29% 32% 0.65 0.17 2.39× 0.27 FFN-Output 310 1635 15% 17% 0.57 0.14 2.18× 0.26 FFN-Temporal 310 1635 15% 17% 0.52 0.12 2.18× 0.24 SA-O 103 545 5% 6% 0.61 0.15 2.06× 0.30 SA-Q 103 545 5% 6% 0.47 0.10 2.06× 0.23 SA-K 103 545 5% 6% 0.49 0.11 2.06× 0.24 SA-V 103 545 5% 6% 0.56 0.13 2.06× 0.27 CA-Q 103 545 5% 6% 0.54 0.12 2.06× 0.26 CA-O 103 545 5% 6% 0.46 0.10 2.06× 0.22 CA-KV 201 120 10% 1% 0.49 0.11 2.01× 0.24

Block-range FP4 (all modules in this block range → FP4)

Shallow Blocks (0–5) 612 2950 30% 30% 0.79 0.22 2.35× 0.34 Middle Blocks (6–13) 816 3940 40% 40% 0.71 0.19 2.50× 0.29 Deep Blocks (14–19) 612 2950 30% 30% 0.64 0.16 2.35× 0.27

Mixed-Precision

Basic Mix (CA-O + SA-Q + SA-K + FFN-Temporal) 619 3270 30% 33% 0.56 0.14 2.40× 0.23 Basic Mix + FFN-Input/Output (Deep blocks 14–19) 893 4710 44% 48% 0.60 0.15 2.63× 0.23 Basic Mix + FFN-Input/Output (Deep+Mid 6–19, Ours) 1258 6640 62% 67% 0.64 0.17 3.02× 0.21

Trade-off analysis. We make the following observations from Table 4. First, the baselines reveal the central tension: pure FP4 has a higher cost over speedup than FP8 (0.26 vs. 0.23), so doubling the raw compute speedup does not pay back the quality penalty. The goal of the search is therefore not to push as much as possible to FP4, but to find a mixed policy whose ratio is strictly below the FP8 reference of 0.23. Second, the single-group sweep spreads from 0.22 (CA-O) to 0.30 (SA-O), and only CA-O dips below the FP8 reference on its own; SA-Q, SA-K and FFN-Temporal sit just above, while the value/post-mixing projections (SA-V, SA-O, CA-Q) and FFN input/output project clearly above FP8, providing direct evidence for the per-layer decisions made above. Third, holding the FP4 FLOP budget fixed at 30% across the three block ranges flips the cost-over-speedup ratio from 0.34 (shallow) through 0.29 (middle) to 0.27 (deep): the same FP4 budget hurts least when spent on the deeper blocks, motivating the depth-restricted demotion of FFN input/output. Finally, the build-up in the mixed-precision section makes the trade-off concrete: Basic Mix matches the FP8 ratio (0.23 at 2.40×); adding FFN input/output in the deep blocks holds the ratio at 0.23 while lifting the speedup to 2.63×; and only extending those FFN demotions into the middle blocks (Ours) drops the ratio below FP8, reaching 0.21 at an idealized 3.02× speedup over BF16, dominating both baselines and every intermediate configuration we evaluated.

[Figure 155]

[Figure 156]

(a) Per-layer quantization policy search (b) Per-block and mixed quantization policy search

- Figure 11 | Mixed-Precision Quantization Policy Search on LPIPS. For each subfigure, the x-axis is the estimated speed up and the y-axis is the quantization error over the BF16 reference. The selected mixed-precision policy achieves the best trade-off between efficiency and quality.

1

[Figure 158]

[Figure 159]

- Case 1, 1, 15,25
- Case 2, 5, 25, 50

[Figure 160]

| |
|---|

[Figure 161]

| |
|---|

[Figure 162]

| |
|---|

[Figure 163]

| |
|---|

| |
|---|

| |
|---|

[Figure 164]

| |
|---|

[Figure 165]

| |
|---|

Chunk 1 Chunk 2 Chunk 1 Chunk 2

- Figure 12 | Limitations of Linear Attention. Purely linear attention [6] suffers from visible chunk-to-chunk appearance jumps and temporal flicker.

#### B. Implementation Details

Model Design. SANA-Streaming is a 2B hybrid diffusion transformer with the LTX2 VAE [13] (compression ratio 32×32×8). The DiT contains 20 transformer blocks, including 5 evenly inserted softmax-attention blocks and 15 GDN blocks. The hidden size is 2240 with 20 attention heads.

Dataset. We use the open-source Ditto [2] and OpenVE [14] datasets, as well as the human-centric editing data built by our data pipeline (Appendix E). The total number of short video clips is about 10M. For the pretraining stage, we filter the dataset based on our VLM verification results, with dataset-specific filtering criteria, obtaining 7M clips after filtering. In the SFT stage, we use stricter filtering criteria to obtain 1M clips. For the long-video dataset, we use an internal dataset with 10K one-minute videos and annotate the edit instruction and reverse instruction using our data pipeline.

Model Training. The bidirectional model is trained with 32 NVIDIA H100 GPUs for about 100K iterations

- with batch size 2 per GPU. The learning rate is set to 5e-5 with the AdamW optimizer. For the long training stage, following LongLive [34], we train the model in three stages: ODE initialization, self-forcing training, and streaming long training. The proposed cycle-reverse regularization is adopted in the long training stage.

15

Algorithm 1 Bidirectional GDN forward Require: 𝑞𝑘𝑣 ∈ R𝐵×𝐹𝑆×3×𝐻×𝐷, 𝛽, decay 𝛼, normalization weights, RoPE tables Ensure: output 𝑂 ∈ R𝐵×𝐹𝑆×𝐻×𝐷

- 1: Pad head dimension to BLOCK_D = 128 ◁ Phase A: frame-parallel summaries
- 2: for all (𝑏,ℎ,𝑓) in parallel do
- 3: 𝑃𝑓 ← 𝐼 − 𝐾rot⊤ ,𝑓 diag(𝛽𝑓)𝐾rot,𝑓
- 4: 𝐴𝑓 ← 𝐾rot⊤ ,𝑓 diag(𝛽𝑓)𝑉𝑓
- 5: 𝑃𝑓𝑧 ← 𝐼 − 𝐾𝑓⊤ diag(𝛽𝑓)𝐾𝑓
- 6: 𝑏𝑓 ← 𝐾𝑓⊤𝛽𝑓
- 7: end for ◁ Phase B: compact recurrent scans
- 8: for all (𝑏,ℎ) in parallel do
- 9: 𝑆−→1,𝑧−→1 ← cached state or zero
- 10: for 𝑓 = 0 to 𝐹 − 1 do
- 11: 𝑆𝑓→ ← 𝛼𝑓𝑃𝑓𝑆𝑓→−1 + 𝐴𝑓
- 12: 𝑧𝑓→ ← 𝛼𝑓𝑃𝑓𝑧𝑧𝑓→−1 + 𝑏𝑓
- 13: end for
- 14: 𝑆𝐹←−1,𝑧𝐹←−1 ← 0
- 15: for 𝑓 = 𝐹 − 1 down to 1 do
- 16: 𝑆𝑓←−1 ← 𝛼𝑓𝑃𝑓𝑆𝑓← + 𝐴𝑓
- 17: 𝑧𝑓←−1 ← 𝛼𝑓𝑃𝑓𝑧𝑧𝑓← + 𝑏𝑓
- 18: end for
- 19: for 𝑓 = 0 to 𝐹 − 1 do
- 20: 𝑆𝑓hist ← 𝑆𝑓→ + 𝑆𝑓←
- 21: 𝑧𝑓hist ← 𝑧𝑓→ + 𝑧𝑓←
- 22: end for
- 23: end for ◁ Phase C: output streaming over spatial tiles
- 24: for all (𝑏,ℎ,𝑓) in parallel do
- 25: for 𝑠0 = 0 to 𝑆 − 1 step BLOCK_S do
- 26: Load and normalize 𝑄𝑓,𝑠

0:𝑠0+BLOCK_S

- 27: Apply RoPE to obtain 𝑄rot
- 28: 𝑁 ← 𝑄rot𝑆𝑓hist
- 29: 𝑑 ← 𝑄𝑧𝑓hist
- 30: 𝑂𝑓,𝑠

0:𝑠0+BLOCK_S ← 𝑁/(𝑑 + 𝜖)

- 31: end for
- 32: end for

#### C. More Results

We present additional video editing results in Figure 13. For each group, the first row shows the original video as the source reference. The subsequent rows demonstrate our model’s capability in three distinct editing tasks: style transfer (e.g., low poly or mosaic), background replacement (e.g., concrete terrace or boxing gym), and local object addition (e.g., adding a lantern or dragonfly). The results show that our method can precisely apply these modifications while maintaining high temporal consistency and structural fidelity to the original motion and character identity. Figure 12 further illustrates the chunk-boundary artifacts of pure linear attention, motivating the hybrid architecture used in SANA-Streaming.

#### D. GDN Kernel Details

Tensor shapes and layout. This appendix describes the Triton GDN kernel used in our experiments. The input layout is 𝑞𝑘𝑣 ∈ R𝐵×𝑁×3×𝐻×𝐷 with 𝑁 = 𝐹𝑆. In the production configuration, 𝐷 = 112 is padded to BLOCK_D=128 inside the kernels. The recurrent matrix state therefore has shape 128 × 128 per (𝐵,𝐻) stream, and the vector state has length 128. Intermediate frame summaries use shape (𝐵𝐻,𝐹,128,128) for matrix terms and (𝐵𝐻,𝐹,128) for vector terms.

GPU precision Phase A (𝑛𝑤,𝐵𝑆) Phase B (𝑑,𝑛𝑤,𝑛𝑠,acc) Phase C (𝑛𝑤,𝐵𝑆)

A100 fp32 (16,32) (4,32,1,+) (16,32) A100 TF32 (16,32) (4,8,2,−) (16,32) A100 bf16 (8,32) (4,8,2,−) (4,32)

H100 fp32 (8,32) (4,32,1,+) (16,32) H100 TF32 (8,32) (4,8,2,−) (16,32) H100 bf16 (8,64) (4,8,2,−) (8,32) GB200 fp32 (8,128) (8,4,1,−) (4,64) GB200 TF32 (8,128) (4,8,2,−) (4,64) GB200 bf16 (4,64) (4,8,2,−) (8,64) 5090 fp32 (8,16) (8,4,1,−) (8,16) 5090 TF32 (8,16) (8,8,1,−) (8,16) 5090 bf16 (8,32) (8,8,1,−) (4,32) GB10 fp32 (8,16) (8,4,1,−) (8,16) GB10 TF32 (8,16) (1,8,1,−) (8,16) GB10 bf16 (8,32) (1,8,1,−) (4,32)

- Table 5 | Final Triton launch parameters. Across phases, 𝑛𝑤 is the number of warps, 𝑛𝑠 is the number of pipeline stages, and 𝐵𝑆 is the spatial block size. For Phase B, 𝑑 is the number of output-column splits (𝑑 = 1 means no D-tiling), and acc indicates whether accumulator fusion is enabled.

Implementation details. Phase A is implemented as two kernels: a matrix stream for (𝑃𝑓,𝐴𝑓) and a lighter vector stream for (𝑃𝑓𝑧,𝑏𝑓). The vector stream does not need 𝑉 or RoPE. Both streams accumulate frame summaries in fp32 and store 𝐼 − 𝑃 directly, so Phase B can compute (𝐼 − 𝑃)𝑆 with one MMA. Phase B keeps the scan state in fp32. In bidirectional mode it writes the reverse contribution directly into the forward history buffer, so Phase C is launched once on the combined history. Phase C then streams the spatial dimension of 𝑄𝑓 in BLOCK_S tiles.

Precision modes. The kernel supports three dot-product modes, inherited from the global GDN precision setting:

- • fp32: Triton dot operands are fp32 with input_precision="ieee". The Phase-A to Phase-B HBM bridge is fp32, and Phase-C numerator/denominator buffers are fp32.
- • TF32: the Phase-A to Phase-B bridge remains fp32, but matrix products use TF32 tensor cores.
- • bf16: dot operands use bf16 tensor cores with fp32 accumulation. Phase-A summaries and Phase-C output buffers are stored in bf16 to reduce HBM traffic, while the recurrent Phase-B scan state remains fp32.

For fp32 and TF32 modes, the frame summaries communicated between Phase A and Phase B are stored in fp32. For bf16 mode, they are stored in bf16 to reduce HBM traffic.

Architecture-specific launch parameters. We tune Phase A, B, and C separately because they stress different resources. Phase A and C are spatial streaming kernels controlled mainly by BLOCK_S. Phase B is a compact recurrent scan controlled mainly by the number of warps and whether D-tiling is enabled. fp32 and TF32 share the same Phase-A and Phase-C launch shapes, but Phase B is dispatched by the exact dot-product mode. Table 5 shows the final effective launch configuration used by the kernel.

D-tiled Phase B. For Phase B, the output columns of the matrix recurrence are independent:

𝑆𝑓[:,𝐽] = 𝛼𝑓𝑃𝑓𝑆𝑓−1[:,𝐽] + 𝐴𝑓[:,𝐽].

We exploit this by splitting the output-column dimension into 𝑑splits tiles. Each Triton program owns one column tile 𝐽, which lowers live state size and increases grid parallelism. The vector state 𝑧𝑓 is not columnsplittable, so only the leading D-tile computes it. Because D-tiling also duplicates reads of 𝑃𝑓, the selected policy depends on GPU and precision, as shown in Table 5.

fixed sink rolling GPU precision impl. total (s) ch0 (ms) ch1+ (ms) total (s) ch0 (ms) ch1+ (ms) total (s) ch0 (ms) ch1+ (ms) A100 fp32 PyTorch 232.69 72.30 67.15 237.52 75.77 69.98 252.31 72.78 79.45

14.37 (5.53×) bf16 PyTorch 57.50 12.07 7.85 60.63 12.08 8.85 69.51 12.32 13.90

87.72 (2.65×)

7.20 (10.04×)

4.87 (13.78×)

94.26 (2.52×)

13.04 (5.81×)

10.47 (6.68×)

99.30 (2.54×)

12.96 (5.61×)

Kernel

4.99 (2.78×) H100 fp32 PyTorch 32.35 9.35 5.99 35.02 9.33 7.53 43.88 9.61 13.20

43.03 (1.34×)

3.45 (3.49×)

2.47 (3.18×)

46.07 (1.32×)

4.02 (3.01×)

3.74 (2.36×)

48.40 (1.44×)

3.99 (3.08×)

Kernel

5.58 (2.37×) bf16 PyTorch 28.86 7.23 4.73 30.89 7.18 5.26 36.36 7.40 8.43

21.69 (1.49×)

6.06 (1.54×)

2.88 (2.08×)

24.62 (1.42×)

5.78 (1.61×)

5.24 (1.44×)

27.72 (1.58×)

3.84 (2.50×)

Kernel

2.61 (3.23×) GB200 fp32 PyTorch 21.65 6.44 4.54 23.30 6.34 5.27 30.46 6.70 9.96

18.88 (1.53×)

1.58 (4.56×)

1.05 (4.49×)

20.86 (1.48×)

1.94 (3.71×)

1.71 (3.09×)

22.19 (1.64×)

1.92 (3.85×)

Kernel

3.51 (2.84×) bf16 PyTorch 22.32 6.21 4.55 23.51 6.19 5.37 30.84 6.23 9.12

13.71 (1.58×)

2.40 (2.68×)

1.55 (2.93×)

15.40 (1.51×)

2.50 (2.54×)

2.53 (2.08×)

16.94 (1.80×)

2.53 (2.65×)

Kernel

2.22 (4.10×) 5090 fp32 PyTorch 89.28 11.40 24.74 92.59 11.43 26.86 103.43 11.72 33.21

13.65 (1.64×)

1.41 (4.42×)

1.51 (3.02×)

15.52 (1.51×)

1.54 (4.02×)

2.33 (2.30×)

15.43 (2.00×)

1.48 (4.21×)

Kernel

4.98 (6.67×) bf16 PyTorch 46.03 9.39 5.72 47.17 9.08 6.55 55.17 9.56 10.51

49.16 (1.82×)

5.08 (2.24×)

2.95 (8.39×)

51.73 (1.79×)

5.30 (2.16×)

4.54 (5.92×)

53.13 (1.95×)

4.23 (2.77×)

Kernel

3.74 (2.81×) GB10 fp32 PyTorch 397.58 117.57 93.70 425.10 117.94 107.12 491.71 120.16 149.10

35.83 (1.28×)

3.20 (2.94×)

2.11 (2.71×)

36.59 (1.29×)

3.48 (2.61×)

2.60 (2.52×)

39.72 (1.39×)

3.59 (2.66×)

Kernel

26.87 (5.55×) bf16 PyTorch 368.64 101.09 83.62 393.57 101.44 91.01 434.69 102.39 116.46

193.76 (2.05×)

16.55 (7.10×)

10.02 (9.35×)

212.71 (2.00×)

18.25 (6.46×)

18.11 (5.92×)

229.11 (2.15×)

18.34 (6.55×)

Kernel

186.15 (1.98×)

12.48 (8.10×)

7.88 (10.62×)

206.15 (1.91×)

15.83 (6.41×)

13.47 (6.75×)

216.68 (2.01×)

15.77 (6.49×)

19.81 (5.88×)

Kernel

- Table 6 | GDN-kernel runtimes and speedups over PyTorch. PyTorch rows report baseline runtimes; Kernel rows report Triton-kernel runtimes with speedup in parentheses. Column headers give units: total is end-to-end sampling time in seconds, while ch0 and ch1+ are per-call GDN times in milliseconds for the first chunk and later chunks, respectively. The three RoPE/cache strategies are fixed-RoPE, rolling-RoPE with sink, and rolling-RoPE.

Benchmark. We evaluate three RoPE/cache strategies: fixed-RoPE, rolling-RoPE with sink, and rollingRoPE. All runs use the same output chunk pattern, with an initial five-frame chunk followed by three-frame chunks. In Table 6, total is the end-to-end sampling time, ch0 is the first GDN call that emits five frames, and ch1+ is the median subsequent GDN call that emits three frames. Depending on the cache strategy, a GDN call may also consume cached recurrent state or replay cached pre-RoPE context internally. We report two measurements. First, the end-to-end sampling time is the wall-clock time of the diffusion sampling phase; video preparation, VAE decoding, encoding, and saving are excluded. Second, per-call GDN time is measured inside the model with CUDA events around each GDN forward call. Calls are bucketed by strategy and chunk type, and we report the median time for the initial chunk (ch0) and the subsequent chunks (ch1+). For each configuration we run PyTorch and the Triton GDN kernel under the same precision setting and compute speedup as the ratio of PyTorch time to Triton-kernel time. The reported sampling time uses the second generated sample to avoid one-time warmup effects. Table 6 reports PyTorch and Triton-kernel runtimes, with speedups over PyTorch in parentheses on the Triton rows. In fp32, per-call GDN speedups are 1.4×–13.8× across five GPU classes and three rope strategies; end-to-end sampling speedups are 1.42×–2.65× because non-GDN work remains in the pipeline. At bf16, per-call speedups are 2.3×–10.6× and sampling speedups are 1.28×–2.01×.

#### E. Data Pipeline

Our data pipeline aims to construct high-quality human-centric video editing pairs, enabling controllable semantic edits and temporally coherent motion. In the following subsections, we present the construction process for both short and long videos, along with the corresponding quality control procedures. Figure 7 summarizes the complete pipeline, including short-video pair construction, long-video instruction construction, and VLM-based filtering.

###### E.1. Short Video

Edit Instruction Generation and First Frame Edit. We first generate edit instructions with a taxonomyguided prompting strategy. For each source video, we sample an editing task from four high-level categories: local human edits, background edits, artistic style transfer, and composite edits. Local edits include outfit replacement, outfit color or pattern changes, hairstyle changes, hair color changes, and accessory edits. Background edits either replace the scene while preserving the subject, or jointly adapt the background and outfit to form a coherent contextual transformation. Style-transfer instructions are sampled from a diverse style dictionary, while composite edits combine multiple attributes such as fashion, hair, makeup, setting, season, decade, or character archetype. Given the sampled task, we prompt a vision-language model, Qwen3VL [33], to analyze the source video and produce a single self-contained editing instruction with concrete visual details such as material, color, silhouette, texture, lighting interaction, and style coherence. We then apply this edit to the first frame of the video. Specifically, we use Qwen-Image-Edit [28] conditioned on the edit instruction, the first frame, and the pose extracted from the first frame to obtain an edited first frame, which serves as a visual anchor for later edit-video generation.

In-Context Generation for Edit Video. To enable fine-grained and controllable video editing, we provide the edited first frame and the source video to Qwen3VL [33] to generate a caption describing the desired target video content. Meanwhile, we extract a pose video from the source video to serve as a motion anchor, preserving temporal dynamics. Conditioned on these signals, we generate the edited video using a controllable text-image-video-to-video pipeline, Wan2.2-Fun-Control [1]. For each sample, the generated caption serves as the text prompt, the pose video provides motion guidance, and the edited first frame is used as both the initial frame and a visual reference.

###### E.2. Long Video

To support streaming long training and cycle-reverse regularization, we construct an additional long-video subset with paired forward and backward edit instructions. For each long source video, we first sample a representative anchor frame, typically the first frame, and use it as the visual evidence for instruction generation. Conditioned on the anchor frame, we use Gemini-3-Flash to generate the forward edit instruction 𝑝+. The edit prompt is sampled from five edit families: background replacement, local addition, local removal, local replacement, and style transfer, following the taxonomy-based approach described above.

We construct the backward instruction 𝑝− using the source video and its forward instruction 𝑝+. The VLM is asked to infer the inverse operation that would recover the observed source content after the forward edit has been applied.

###### E.3. Data Verification

VLM Verification for Short Video. After generating candidate edited videos, we apply VLM-based verification to filter low-quality samples. The verifier compares each edited video against the original source video, the edit instruction, and the predicted edited-video caption. This comparison is necessary because a generated video can be visually plausible while still failing to follow the intended edit, altering non-edit regions, or introducing temporal artifacts.

We evaluate each candidate along four dimensions. Instruction alignment measures whether the edited video faithfully implements the requested semantic change. Non-edit consistency and temporal stability measures whether regions not targeted by the edit are preserved from the source video and whether the output remains stable across frames. Physical plausibility evaluates lighting consistency, geometry, motion, and other real-world constraints after editing. Video quality captures sharpness, color stability, compression artifacts,

and overall visual coherence. Each dimension is scored on a 0–10 scale, and the verifier also returns brief comments describing strengths and failure cases. The resulting scores and textual comments are saved for each sample and used to select the final training data.

Data Verification and VLM Verification for Long Videos. Long videos may contain title cards or black-screen segments that provide little useful editing content. We sample the first and last 10 seconds at approximately 2 FPS, and discard videos with excessive near-black pixel ratio. Moreover, for long-video self-forcing distillation, the student is bounded by the teacher’s editing capability; if the teacher cannot execute a forward edit, the corresponding long sample provides little useful supervision. We therefore verify long-video instructions by taking the first 81 frames of each source video and asking the teacher model to generate the edited 81-frame clip conditioned on the forward instruction. A VLM then compares the source clip, the teacher-edited clip, and the instruction, deciding whether the specified edit is successfully performed. If not, we discard the sample.

#### F. Causal VAE Training

We train only the decoder of the causal VAE, while keeping the pretrained VAE encoder fixed. Given a training video 𝑥 ∈ R𝐵×𝐶×𝑇×𝐻×𝑊, the frozen encoder produces a latent code 𝑧, and the causal decoder reconstructs 𝑥^ = 𝐷𝜃causal(𝑧). The bidirectional LTX2 decoder is used as a frozen teacher for intermediate feature supervision. Below we expand the loss terms in Eq. 3. For compact notation, 𝑥𝑏,𝑡 and 𝑥^𝑏,𝑡 denote the 𝑡-th frame of the target and reconstruction in video 𝑏.

Charbonnier reconstruction loss. For robust pixel-level reconstruction, we use a Charbonnier penalty on the decoded video:

√︁(^𝑥𝑏,𝑐,𝑡,ℎ,𝑤 − 𝑥𝑏,𝑐,𝑡,ℎ,𝑤)2 + 𝜖2, 𝜖 = 10−6. (7)

1 𝐵𝐶𝑇𝐻𝑊 ∑︁

ℒcharb =

𝑏,𝑐,𝑡,ℎ,𝑤

This term behaves similarly to an ℓ1 reconstruction loss for large residuals while remaining smooth around zero, which stabilizes decoder-only fine-tuning.

Perceptual loss. To preserve perceptual similarity beyond per-pixel accuracy, we apply the AlexNet-based LPIPS loss independently to each video frame:

1 𝐵𝑇 ∑︁

𝒟LPIPSAlex (^𝑥𝑏,𝑡,𝑥𝑏,𝑡). (8)

ℒperc =

𝑏,𝑡

Haar wavelet loss. The Haar wavelet term explicitly supervises high-frequency details. Let 𝒲H(·) denote a single-level 2D Haar transform applied frame by frame, and let 𝒲Hhigh(·) concatenate the three high-frequency subbands LH, HL, and HH, omitting the low-frequency LL band. We minimize

1 𝐵𝑇 ∑︁

1 𝑀H

⃦𝒲Hhigh (^𝑥𝑏,𝑡) − 𝒲Hhigh (𝑥𝑏,𝑡)⃦

ℒhaar =

. (9)

1

𝑏,𝑡

Here 𝑀H is the number of high-frequency Haar coefficients per frame. This loss encourages the causal decoder to recover sharp edges and fine texture that are often smoothed by purely pixel-level objectives.

Intermediate feature distillation. We further align the causal student decoder with the bidirectional teacher decoder using intermediate activations. Let 𝐹ℓ𝑆(𝑧) and 𝐹ℓ𝑇(𝑧) be the student and teacher decoder features at layer ℓ, where ℓ is selected from the decoder mid-block and upsampling blocks. The teacher is frozen and gradients are stopped through its features:

1 𝑁ℓ ⃦𝐹ℓ𝑆(𝑧) − 𝐹ℓ𝑇(𝑧)⃦2 2 , (10)

∑︁

ℒdistill =

ℓ∈𝒮

where 𝒮 is the set of distilled decoder blocks and 𝑁ℓ is the number of elements in the feature tensor.

#### G. Frame-wise Gated DeltaNet

This section provides the detailed formulation of the frame-wise Gated DeltaNet (GDN) block used in SANA-Streaming, which extends the Gated Delta Network [35] to the frame-wise streaming video setting. The design goal of this block is to provide a compact, finite recurrent memory for causal streaming generation: instead of caching raw key/value tokens from all previous chunks, each GDN block only carries forward the terminal recurrent states from the previous chunk, so the cache size is independent of the number of streamed chunks.

###### G.1. Notation

All equations below are presented for a single attention head; the full multi-head computation runs 𝐻 heads in parallel and concatenates their outputs along the channel dimension. For a chunk of 𝐹 frames with 𝑁 spatial tokens per frame, the per-head latent feature is

𝑥 ∈ R𝐹×𝑁×𝐷, where 𝐷 is the per-head channel dimension. For the 𝑓-th frame, the per-head input tokens are 𝑥𝑓 ∈ R𝑁×𝐷. For convenience we use the column-token convention in the following derivation, 𝑋𝑓 = 𝑥⊤𝑓 ∈ R𝐷×𝑁,

and we omit the head index ℎ throughout, with the understanding that all weight matrices are head-specific. Each head maintains two recurrent states: a KV memory

𝑆𝑓𝑘𝑣 ∈ R𝐷×𝐷 and a normalizer state

𝑆𝑓𝑧 ∈ R𝐷.

###### G.2. Feature parameterization

Given 𝑋𝑓, the block computes per-head query, key, and value features by pointwise (kernel size 1) linear projections:

𝑄𝑓 = 𝑊𝑞𝑋𝑓, 𝐾𝑓 = 𝑊𝑘𝑋𝑓, 𝑉𝑓 = 𝑊𝑣𝑋𝑓, (11)

where 𝑊𝑞,𝑊𝑘,𝑊𝑣 ∈ R𝐷×𝐷 and 𝑄𝑓,𝐾𝑓,𝑉𝑓 ∈ R𝐷×𝑁. We then apply RMSNorm [40], a ReLU kernel feature map following Sana’s linear attention design [31], and a fixed 3D RoPE [22] to the query and key:

𝑄^𝑓 = 𝜌3D(︁ReLU

)︀)︁, 𝐾^𝑓 = 𝜌3D(︁ReLU

)︀)︁. (12)

(︀

(︀

RMSNorm(𝑄𝑓)

RMSNorm(𝐾𝑓)

To keep the magnitude of the normalizer projection 𝐾𝑓⊤𝑆𝑓𝑧 stable for large spatial token counts, both 𝐾𝑓 and 𝐾^𝑓 are additionally rescaled by 1/

√

𝐷𝑁 before entering the recurrence. The unrotated key 𝐾𝑓 is used for the normalizer state, while the rotated key 𝐾^𝑓 is used for the KV memory update and readout; this “RoPE-on-numerator-only” choice preserves mass conservation in the linear-attention denominator, since rotated 𝐾^𝑓 does not sum to a positive scalar after frame-wise accumulation.

###### G.3. Gate parameterization

The block predicts three gates: a frame-wise decay gate 𝛼𝑓, a token-wise write gate 𝛽𝑓, and an output gate 𝐺𝑓.

Decay gate 𝛼𝑓. We adopt a Mamba-style discretized state-space parameterization [11] for the decay gate, which empirically yields substantially more stable long-horizon recurrence than a plain sigmoid:

1 𝑁

∑︁𝑁

(︀

)︀

𝑥¯𝑓 =

:,𝑛 ∈ R𝐷, (13)

𝑋𝑓

𝑛=1

)︀ ∈ (0,1], (14)

(︀

)︀ ∈ R>0, 𝛼𝑓 = exp

(︀−𝑒𝐴

· Δ𝑓

Δ𝑓 = softplus

𝑊𝛼 𝑥¯𝑓 + 𝑏𝛼

log

where 𝑊𝛼 ∈ R1×𝐷 projects the spatially pooled frame feature 𝑥¯𝑓 to a scalar, 𝑏𝛼 ∈ R is a bias, and 𝐴log ∈ R is a learnable per-head log-rate parameter that ensures 𝐴 = 𝑒𝐴

> 0. The decay 𝛼𝑓 is therefore a per-frame per-head scalar shared across all 𝑁 spatial positions and all 𝐷 channels of the head. This parameterization is equivalent to the discretization 𝛼 = 𝑒−𝐴Δ of a continuous-time linear ODE with data-dependent step size Δ, and matches the selective-scan gating of Mamba-2.

log

Write gate 𝛽𝑓. The write gate is a token-wise sigmoid:

)︀ ∈ (0,1)1×𝑁, (15) where 𝑊𝛽 ∈ R1×𝐷 produces a single scalar per token (per head). For convenience we write

(︀

𝛽𝑓 = sigmoid

𝑊𝛽𝑋𝑓 + 𝑏𝛽

𝐵𝑓 = Diag(𝛽𝑓) ∈ R𝑁×𝑁

for its diagonal-matrix form, used in the matrix update equations below. The write gate controls how aggressively each token’s delta-rule correction is written into memory.

Output gate 𝐺𝑓. The output gate is a per-token, per-channel SiLU-modulated linear projection:

(︀

)︀ ∈ R𝐷×𝑁, (16)

𝐺𝑓 = SiLU

𝑊𝑔𝑋𝑓 + 𝑏𝑔

where 𝑊𝑔 ∈ R𝐷×𝐷. This is analogous to the output gate of Gated Linear Attention [36] and the gated MLP of Mamba and serves as a fine-grained channel-wise selector on the attention output.

- G.4. Frame-wise delta-rule update At frame 𝑓, the previous recurrent states are first decayed:

𝑆̃︀𝑓𝑘𝑣 = 𝛼𝑓 𝑆𝑓𝑘𝑣−1, 𝑆̃︀𝑓𝑧 = 𝛼𝑓 𝑆𝑓𝑧−1. (17) The KV memory is then updated by a delta-rule correction in the spirit of fast-weight programmers [21, 35]:

(︀

)︀

𝑉𝑓 − 𝑆̃︀𝑓𝑘𝑣𝐾^𝑓

𝐵𝑓𝐾^⊤𝑓 . (18) Equivalently, this can be written in the compact delta-rule form

𝑆𝑓𝑘𝑣 = 𝑆̃︀𝑓𝑘𝑣 +

(︀

)︀

𝐼 − 𝐾^𝑓𝐵𝑓𝐾^⊤𝑓

+ 𝑉𝑓𝐵𝑓𝐾^⊤𝑓 . (19)

𝑆𝑓𝑘𝑣 = 𝛼𝑓 𝑆𝑓𝑘𝑣−1

The normalizer state is updated in the same correction-based manner, using the unrotated key 𝐾𝑓 to preserve mass conservation:

(︀

)︀

1 − 𝐾⊤𝑓 𝑆̃︀𝑓𝑧

𝑆𝑓𝑧 = 𝑆̃︀𝑓𝑧 + 𝐾𝑓𝐵𝑓

, (20) where 1 ∈ R𝑁 is an all-one vector. Equivalently,

(︀

)︀

𝑆𝑓𝑧−1 + 𝐾𝑓𝐵𝑓1. (21) The key difference from simple gated accumulation [36, 23] lies in the residual term

𝑆𝑓𝑧 = 𝛼𝑓

𝐼 − 𝐾𝑓𝐵𝑓𝐾⊤𝑓

𝑉𝑓 − 𝑆̃︀𝑓𝑘𝑣𝐾^𝑓.

Instead of directly writing 𝑉𝑓𝐾^⊤𝑓 into memory, GDN first predicts the current values from the decayed recurrent state and only writes back the remaining error, scaled by 𝛽𝑓. This correction-based update reduces interference between stored key/value associations [21], providing a more controlled finite-state memory for long streaming sequences. It is critical for minute-long streaming video generation where the same recurrent state must accumulate consistent identity and scene information across thousands of frames.

- G.5. Normalized readout and output projection After the recurrent states are updated, the output is obtained by a normalized linear-attention readout:

𝑌𝑓 =

𝑆𝑓𝑘𝑣𝑄^𝑓 (𝑆𝑓𝑧)⊤𝑄𝑓 + 𝜖

, (22)

where the numerator 𝑆𝑓𝑘𝑣𝑄^𝑓 ∈ R𝐷×𝑁 uses the rotated query, the denominator (𝑆𝑓𝑧)⊤𝑄𝑓 ∈ R1×𝑁 uses the unrotated query and is broadcast over the 𝐷 channel dimension, and 𝜖 = 10−6 is a small constant for numerical stability. The final per-head output is produced by an element-wise multiplication with the output gate followed by a linear projection,

𝑂𝑓 = 𝑊𝑜

(︀

𝐺𝑓 ⊙ 𝑌𝑓

)︀

, (23)

where 𝑊𝑜 ∈ R𝐷×𝐷 and ⊙ denotes the Hadamard product. Per-head outputs {𝑂𝑓(ℎ)}𝐻ℎ=1 are then concatenated along the channel dimension and passed through a final shared linear projection.

- G.6. Streaming cache

During streaming inference, each GDN block only caches the terminal states (︀

)︀ ∈ R𝐷×𝐷 × R𝐷

𝑆𝑘𝑣, 𝑆𝑧

per head from the previous chunk, and uses them to initialize the next chunk. The total cache size is therefore 𝒪(︀

)︀

𝐻𝐷2

per layer and is independent of the number of streamed chunks or the total streaming length. This allows the GDN blocks in SANA-Streaming to serve as a compact global accumulated memory that compresses the entire streaming history into a fixed-size matrix state, while the softmax blocks complement them by performing local window-and-sink refinement [30] over recent tokens.

#### H. Safeguards

For deployment, the model is paired with safeguards outside the generative model itself. We use a layered design in which policy checks are applied before generation, during generation when needed, and after generation before content is returned.

Input screening. User prompts and conditioning inputs is first checked by a policy classifier, such as Llama Guard [16] or an equivalent safety classifier. The classifier flags requests involving disallowed sexual content, minors, graphic violence, self-harm, instructions for wrongdoing, privacy violations, and other categories defined by the deployment policy. Prompts that clearly violate policy are rejected. Ambiguous prompts can be routed to a stricter policy, rewritten into a safe form, or sent for human review.

Generation-time controls. The generation service should preserve the original safety decision throughout sampling. For example, the system can attach a policy state to each request and avoid later prompt expansions that introduce disallowed entities or actions. Rate limits, authentication, and abuse monitoring should be applied to reduce automated misuse. For high-risk categories, the service can use conservative decoding settings or disable generation entirely rather than relying only on post-hoc filtering.

Output screening. Generated videos are checked before release. A practical implementation can sample frames from the beginning, middle, and end of each video, run an image or vision-language safety classifier on those frames, and optionally run OCR to detect unsafe or private text rendered in the video. If any sampled frame is flagged, the system blocks the video or route it to manual review. For applications with stricter requirements, all frames or short temporal clips can be screened instead of sparse frame samples.

Provenance and traceability. Generated content includes provenance metadata or a watermark when the deployment setting supports it. The service retains minimal audit logs needed for abuse investigation, including the safety decision, model version, and policy version, while avoiding unnecessary storage of sensitive user data.

Evaluation. Safeguards will be evaluated with a held-out red-team set covering direct policy violations, paraphrases, multilingual prompts, prompt-injection attempts, and benign prompts that are easy to over-block.

We report both block rates on unsafe prompts and false-positive rates on benign prompts. Because safety classifiers are imperfect, the safeguard layer should be treated as a risk reduction mechanism rather than a proof that unsafe outputs cannot occur.

#### I. Broader Impacts

SANA-Streaming enables real-time, high-quality video editing, which may benefit applications such as content creation, live broadcasting, and human–computer interaction. At the same time, such technology could be misused to generate misleading or manipulated video content, including real-time deepfakes or deceptive visual edits.

To mitigate these risks, our pipeline incorporates several safeguards, including VLM-based data verification to filter low-quality or misaligned samples, as well as instruction consistency and temporal stability checks during data construction. These measures aim to improve the faithfulness and reliability of generated content. However, we acknowledge that such safeguards are not sufficient to fully prevent misuse or unintended outcomes, especially in open or real-world deployment scenarios. Careful consideration of responsible deployment will be important to ensure that the technology is used in a socially beneficial manner.

##### 50s

0s 20s 30s

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Source Original Video

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

| | | | | |
|---|---|---|---|---|
| |E6F0E 7|F7E8E 6|E6ED F2| |
| | | | | |
| | | | | |
| | | | | |

Style Impressionist oil painting

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Background futuristic observatory

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Local tactical jacket

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Source Original Video

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Style Impressionist oil painting

Worki ng

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Background futuristic observatory

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Local tactical jacket

Figure 13 | More visualizations. SANA-Streaming follows editing instructions while preserving consistency with the original input videos.

6

