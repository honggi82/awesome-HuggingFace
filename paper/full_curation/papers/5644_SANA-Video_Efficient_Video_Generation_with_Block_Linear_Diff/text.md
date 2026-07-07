###### arXiv:2509.24695v2[cs.CV]13Oct2025

[Figure 1]

2025-10-14

###### SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

Junsong Chen1,2*, Yuyang Zhao1*, Jincheng Yu1*, Ruihang Chu4, Junyu Chen1, Shuai Yang1 Xianbang Wang3, Yicheng Pan4, Daquan Zhou5, Huan Ling1, Haozhe Liu6, Hongwei Yi1 Hao Zhang1, Muyang Li3, Yukang Chen1, Han Cai1, Sanja Fidler1, Ping Luo2 Song Han1,3, Enze Xie1

1NVIDIA 2HKU 3MIT 4THU 5PKU 6KAUST *Equal contribution. Project Page: https://nvlabs.github.io/Sana/Video

Abstract: We introduce SANA-Video, a small diffusion model that can efficiently generate videos up to 720×1280 resolution and minute-length duration. SANA-Video synthesizes high-resolution, high-quality and long videos with strong text-video alignment at a remarkably fast speed, deployable on RTX 5090 GPU. Two core designs ensure our efficient, effective and long video generation: (1) Linear DiT: We leverage linear attention as the core operation, which is more efficient than vanilla attention given the large number of tokens processed in video generation. (2) Constant-Memory KV Cache for Block Linear Attention: we design block-wise autoregressive approach for long video generation by employing a constant-memory state, derived from the cumulative properties of linear attention. This KV cache provides the Linear DiT with global context at a fixed memory cost, eliminating the need for a traditional KV cache and enabling efficient, minute-long video generation. In addition, we explore effective data filters and model training strategies, narrowing the training cost to 12 days on 64 H100 GPUs, which is only 1% of the cost of MovieGen. Given its low cost, SANA-Video achieves competitive performance compared to modern state-of-the-art small diffusion models (e.g., Wan 2.1-1.3B and SkyReel-V2-1.3B) while being 16× faster in measured latency. Moreover, SANA-Video can be deployed on RTX 5090 GPUs with NVFP4 precision, accelerating the inference speed of generating a 5-second 720p video from 71s to 29s (2.4× speedup). In summary, SANA-Video enables low-cost, high-quality video generation.

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

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Physical AI World Simulation

(a). Generation results from our 2B models

Block Linear Attention

SANA-Video-2B

Causal Full Attention

Wan2.1-1.3B

|116|16×<br><br>40 40|3 0|
|---|---|---|
| | | |

568

100

1.3B

SkyReelv2

|[Figure 23]<br><br>46<br><br>76<br><br>OOM|
|---|
|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>7.2<br><br>15<br><br>7.2<br><br>26<br><br>7.2 7.2 7.2|

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

|[Figure 33]<br><br>93|
|---|
|[Figure 34]<br><br>[Figure 35]|
|[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>35<br><br>20|
|[Figure 41]<br><br>[Figure 42]<br><br>2<br><br>5<br><br>11|

(min)

[Figure 43]

(GB)

- Wan2.1
- Wan2.2

1.3B

75

CogVideoX

5B 5B

50

25

[Figure 44]

[Figure 45]

SANA-Video

36

2B

[Figure 46]

[Figure 47]

0

- 0

40

80

- 1 10 30 60 65

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

0 148 295 443 590

5 10 30 50

(b). Generation latency (s) on 720x1280x81 video with default inference settings

(c). GPU VRAM v.s. video length (s) on 480x832 resolution

(d). Generation latency v.s. video length (s) on 480x832 resolution

Figure 1 | An overview of generated videos and inference latency and memory of SANA-Video. The generation latency is measured under 50 denoising steps. Linear attention is more efficient for video generation and our block linear attention maintains a fixed memory requirement for long videos.

© 2025 NVIDIA. All rights reserved.

###### 1. Introduction

Video generation is currently a highly active field, fueling applications that range from creative content production and digital live streaming to virtual product displays. Recent large-scale models from industry labs, such as Veo3 [1], Kling [2], Wan [3] and Seedance [4], have demonstrated remarkable performance in generating high-fidelity video content. However, this quality comes at the cost of immense computational complexity. Video generation is an exceptionally token-extensive task; for instance, producing a single 5-second video at 720p resolution with a model like Wan 14B [3] requires to process over 75,000 tokens, taking 32 minutes on a H100 GPU. This sheer volume of data leads to prohibitive training costs and extremely slow generation speeds, rendering these powerful models impractical for widespread research and application. Even with large cost, generating long video (>10 s) is hard to realize with these large models due to the full-sequence processing operation. Recent works (e.g., MAGI-1 [5] and SkyReelv2 [6]) explores the long video generation but the efficiency is strictly constrained by the vanilla attention and KV cache. Given these challenges, a pivotal question arises: Can we develop a high-quality and high-resolution video generator that is computationally efficient and runs very fast on both cloud and edge devices?

This paper proposes SANA-Video, a small diffusion model designed for both efficient training and rapid inference without compromising output quality. In stark contrast to the massive resource requirements of contemporary models, SANA-Video’s training is remarkably cost-effective, requiring only 64 NVIDIA H100 GPUs for 12 days, which represents as little as 1% of the training cost of MovieGen [7] and 10% of that of OpenSora [8]. This efficiency extends to inference, where SANA-Video can generate a 5-second, 720p video in just 36 seconds on a NVIDIA H100 GPU. By drastically reducing the computational barrier, SANA-Video makes high-quality video generation more accessible and practical for a broader range of users and systems. The improvements mainly lie in three key components.

Linear DiT. We extend SANA [9] linear DiT design to the video domain, addressing the significant computational bottleneck of traditional self-attention (𝑂(𝑁2)), as shown in Fig. 1(d). By replacing all attention modules with our efficient linear attention, we reduce complexity to 𝑂(𝑁), which is crucial for high-resolution video generation and leads to a 4× acceleration on 720p video. To enhance our model for video, we make two key improvements. We first integrate Rotary Position Embeddings (RoPE) [10] to improve long-context modeling. In Sec. 3.2, we detail our exploration of the optimal placement for RoPE and how we address the training instability it can introduce. Additionally, we introduce a 1D temporal convolution to the Mix-FFN via a shortcut connection. This design allows us to effectively leverage pre-trained image models and efficiently adapt them for video generation by aggregating temporal features.

Block Linear Attention with KV Cache. The success of SANA-Video in long video generation, a.k.a, LongSANA, is mainly inspired by the attribute of causal linear attention [11]. Based on our reformulation of the causal linear attention operation, we reduce the KV cache to a small and fixed memory, along with a fixed computational cost for each new token. This natively supports long-context operations. Based on the block linear attention module, we introduce a two-stage autoregressive model continue-training paradigm, including autoregressive block training with monotonically increasing SNR sampler and the improved self-forcing specially for our long context attention operation, leading to efficient, long, and high quality video generation.

Efficient Data Filter and Training. The low training cost is mainly attribute to three aspects: the powerful pre-trained text-to-image (T2I) model, efficient data filtering, and the efficient training strategy. First, SANA-Video is continue pre-trained from SANA [9, 12]-1.6B T2I model with the modification for spatio-temporal modeling (Sec. 3.2). Second, we collect data from diverse data source and design specific data filtering criterion for each data source. In addition, a strong VLM [13] serves as our video captioner, producing highly detailed captions (80-100 words), including subject category, color, appearance, actions, expressions, surrounding environment, camera angles, etc. Third, with the highquality video-text pairs, we train SANA-Video in multiple stages from low resolution to high resolution and finally leverage human preferred data for SFT, ensuring the model can efficiently learn the motion and aesthetic appearance.

In conclusion, our model achieves a latency that is over 13× faster than the state-of-the-art Wan2.1 for 720p video generation (Fig. 1(b)), while delivering competitive results across many benchmarks. Additionally, we quantize and deploy our SANA-Video on RTX 5090 GPUs with the NVFP4 precision, where it takes just 29 seconds to generate a 5s 720p video. We hope our model can be efficiently used by everyday users, providing a powerful foundation model for fast video generation.

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
|[Figure 53]| | |
| | | |

###### Linear Attention

###### Mix-FFN

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

n x d

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

Scale

Temporal 3 x 1 x 1 ConvLayer

n x d

MatMul

Block0t=0 Block1t=0.2 Block2t=0.7 Block3t=1 Causal Linear Attention

1 x 1 ConvLayer

n x d

d x d

(a). Block-wise autoregressive training pipeline

MatMul

[Figure 75]

[Figure 76]

d x n

n x d

[Figure 77]

[Figure 78]

SiLU

3D RoPE

[Figure 79]

3D RoPE

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

LinearAttn

CrossAttn

[Figure 84]

MixFFN

ReLU

ReLU

1 x 3 x 3 ConvLayer

AutoEncoder

n x d d x n

| | |
|---|---|
| | |

Linear Linear Linear

###### N x

[1+T, H, W, 3]

n x d

1 x 1 ConvLayer

Scale

Q K V

Asigncyberpunkthat sayscat"SANA"with a Rewriter Time Emb

Cost : O(n)

Small LLM

(b). Architecture overview of our SANA-Video.

(c). Linear DiT module

n x d

- Figure 2 | Overview of SANA-Video. Fig.(a) A high-level block-wise autoregressive training pipeline based on our block causal KV cache. (Details in Sec. 3.3). Fig.(b) Our model pipeline, containing an Autoencoder, Re-writer, Linear DiT, and a text encoder. Fig.(c) The detailed design of the added 3D RoPE in linear attention and the temporal convolution in our Linear DiT’s Mix-FFN.

MatMul

d x d

n x d

MatMul

[Figure 85]

[Figure 86]

| | |
|---|---|
| |F|

d x n

n x d

###### 2. Preliminaries

Mix

|Attn<br><br>Attn<br><br>[14] an| |
|---|---|
| |with d 𝑢(|
|paper, our| |

Time Emb

Relu Relu

Cross

###### 2.1. Video Diffusion Model

n x d d x n

Following SANA [9], we use Rectified Flows (RFs) SNR sampler as the training objective in Eq. 1. Here, 𝑐 is the conditional embedding, 𝜃 is the model weights, 𝑥𝑡 | 𝑡,𝑐;𝜃) denotes the output velocity predicted by the diffusion model. 𝑣(𝑥) is the target velocity. In this SANA-Video is a unified framework for Text-to-Image (T2I), Text-to-Video (T2V), and Image-to-Video (I2V) generation by varying condition embeddings. Specifically, for T2I and T2V, 𝑐 is the text prompt and 𝑥 is the image or video. For I2V, we use first frame and text prompt as condition 𝑐. By setting the noise of the first frame to zero, SANA-Video can realize I2V without any model modification. Therefore, the joint training of T2I, T2V, and I2V makes SANA-Video a unified framework that can perform all tasks with a single model.

Linear

Linea Linea Linea

Linear DiT (N x Blocks)

Working

[Figure 87]

Q K V

(︀

)︀ − 𝑣(𝑥)⃦2 . (1)

𝑥𝑡 | 𝑡,𝑐;𝜃

⃦𝑢

E𝑐,𝑡,𝑥0

###### 2.2. Autoregressive Long Video Generation

Autoregressive diffusion models combine a token/block-wise autoregressive chain-rule decomposition with denoising diffusion models, emerging as a promising direction for long sequence generation like language [15] and video generation [6, 16, 17]. Specifically, for a sequence of 𝑁 blocks 𝑥1:𝑁 = (𝑥1,𝑥2,...,𝑥𝑁), the generation process is a product of block distribution using the chain rule 𝑝(𝑥1:𝑁) =

∏︀𝑁

𝑖=1 𝑝(𝑥𝑖|𝑥𝑗<𝑖), with each block distribution 𝑝(𝑥𝑖|𝑥𝑗<𝑖) modeled using a diffusion process (Eq. 1). This approach leverages the strengths of both autoregressive models and diffusion models to capture sequential dependencies and enable block-wise, high-quality generation.

###### 3. SANA-Video

Scaling video generation to higher resolutions and longer sequences dramatically increases the number of tokens, making the 𝑂(𝑁2) complexity of self-attention a major bottleneck in computation, speed, and memory. This underscores the need for efficient linear attention in video generation. Building upon SANA Linear DiT [9], we introduce Linear Video DiT (Fig. 2(a)) for video generation by integrating two key components: Rotary Position Embeddings (RoPE) and a temporal 1D convolution within the Mix-FFN. These designs keep SANA’s macro architecture as well as additional temporal modeling (Fig. 2(b)), allowing us to leverage a pre-trained image model and efficiently adapting it into a powerful video model through continuous pre-training. In addition to the short video generation, we introduce block linear attention module for efficient long video generation. With the re-formulation of linear attention, the block linear attention module and causal Mix-FFN keep a constant-memory KV cache and linear computational cost for long video. Based on this KV cache, we design two stage post-training paradigm to unlock the infinite length generation ability, leading to a high-quality and efficient long video generation model.

###### 3.1. Training strategy

- Stage1: VAE Adaptation on Text-to-Image (T2I). Training video DiT models from scratch is resource-intensive due to the mismatch between image and video VAEs. We address this by first efficiently adapting existing T2I models to new video VAEs. Specifically, We leverage different video VAEs in generating videos of different resolution. For 480P videos, VAEs with high-compression ratio limits the overall performance, and thus we adopt Wan-VAE [3]. For 720P high-resolution video, we introduce our video VAE, DCAE-V, which provides a higher compression ratio for more efficient generation (details in Sec. 3.4). The adaptation of both VAEs is highly efficient, converging within 5-10k training steps, further demonstrating the strong generalization ability of our Linear DiT.
- Stage2: Continue Pre-Training from T2I Model. Initializing video Linear DiT from a pre-trained T2I model [9] is an efficient and effective way to leverage the well-learned visual and textual semantic knowledge. Therefore, we initialize our SANA-Video with a model adapted from the first stage and introduce additional designs to model long-context and motion information (Sec. 3.2). The additional temporal designs are tailor-made for linear attention, improving the locality of attention operation. The newly added layers are zero-initialized with skip connection, which minimizes their influence on the pre-trained weights during early training. After this identity initialization, SANA-Video is trained in a coarse-to-fine manner. It first trains on low-resolution, short videos (e.g., 192P 2.5 seconds) before moving to higher resolution, longer videos (e.g., 480P 5 seconds) with different data filtering criteria (Appendix D). This coarse-to-fine approach efficiently encourages SANA-Video to fast learn dynamic information with abundant data and then refine details using less, but higher-quality, data.
- Stage3: Autoregressive Block Training. The continued pre-training makes SANA-Video an efficient small diffusion model, primarily for high-resolution 5-second video generation. To enable the generation of much longer videos, we analyze the attributes of linear attention in Sec. 3.3 and propose a constant-memory block KV cache for autoregressive generation. Building on this design, we conduct autoregressive block training in two steps: we first train the autoregressive module and then address exposure bias with our improved self-forcing block training (Sec. 3.3.1). This process results in a high-quality, efficient model for long video generation.

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

###### 3.2. Efficient Linear DiT Pre-Training

SANA-Video adopts the SANA [9] as the base architecture and innovatively tailors the Linear Diffusion Transformer blocks to handle the unique challenges of T2V tasks, as depicted in Fig. 2. Several dedicated designs are proposed as follows:

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

TrainingLoss

Linear Attention with

Linear Attention with

Linear Attention with RoPE(ϕ(x)) (Ours)

Vanilla Attention from Wan

No PE

ϕ(RoPE(x))

Training Steps

(a) Attention map comparison (b) Pre/Post-RoPE QK Sum

- Figure 3 | Analysis of Linear Attention with RoPE. (a) Visual comparison of attention maps. First two plots compare vanilla softmax attention (Wan) to our linear attention without positional encoding. The latter two plots show our method’s effect: applying RoPE after the ReLU kernel results in a sparser, more localized attention pattern. (b) Training loss for the QK sum (Eq. 2 denominator). Removing RoPE from the denominator (green line) ensures training stability, as discussed in Sec. 3.2.

15K iterations visualizations

Done

Linear Attention in Video DiT. Our work extends the SANA [9] architecture by integrating Rotary Position Embeddings (RoPE) [10] into its efficient ReLU (𝜑) linear attention blocks. This integration is crucial for enhancing the model’s ability to handle the sequential and spatial relationships in high-quality video generation. The core of our design lies in applying RoPE after the ReLU activation, specifically as RoPE(ReLU(𝑥)), as shown in Fig. 2. This order is critical because it prevents the ReLU kernel from filtering out the positional information encoded by RoPE. As Fig. 3 shows, this design results in attention maps with a clear focus on local regions, which is essential for capturing fine-grained video details. However, applying RoPE directly to queries and keys (as in vanilla attention) can make the linear attention mechanism numerically unstable [18] due to the difference between softmax and ReLU similarity functions. The RoPE transformation can change the non-negative nature of the ReLU output, potentially causing the denominator in the

[Figure 94]

RoPE(ReLU(x)) ReLU(RoPE(x))

Training Loss

[Figure 95]

[Figure 96]

Training Steps

###### Ablation expo

(a) Sequence of ReLU and Rope (b) Pre/Post-Rope QK Sum

standard linear attention formula (Eq. 2) to become zero. To solve this, we modify the calculation: while the numerator includes RoPE on the queries and keys, we remove RoPE from either the key or the query in the denominator. This ensures the denominator remains positive, guaranteeing training stability (Fig. 3 (b)) while still benefiting from positional encoding.

(︀∑︀

)︀

(︀∑︀

)︀

𝑁 𝑗=1 RoPE(𝜑(𝐾𝑗))𝑇 𝑉𝑗

𝑁 𝑗=1 RoPE(𝜑(𝐾𝑗))𝑇 𝑉𝑗

𝑂𝑖 = RoPE(𝜑(𝑄

𝑖))

)︀ =⇒ RoPE(𝜑(𝑄

𝑖))

)︀ , (2)

(︀∑︀

(︀∑︀

𝑁 𝑗=1 RoPE(𝜑(𝐾𝑗))𝑇

𝑁 𝑗=1 𝜑(𝐾𝑗)𝑇

RoPE(𝜑(𝑄𝑖))

𝜑(𝑄𝑖)

where 𝑂𝑖, 𝑄𝑖, 𝐾𝑖 and 𝑉𝑖 denote the output, query, key and value of the 𝑖th token.

Mix-FFN with Spatial-Temporal Mixture. As shown in Fig. 3, we compare the linear attention map in SANA-Video with the softmax attention map in Wan2.1 [3]. We observe that linear attention is much denser and less focused on local details compared to softmax attention. SANA [9] ameliorates the locality problem in image generation with the convolution in Mix-FFN. Building upon the Mix-FFN, we enhance it with a temporal 1D convolution. The temporal convolution with a shortcut connection is appended to the end of the block (Fig. 2(b)), enabling seamless temporal feature aggregation while preserving initialization. The module helps capture local relationships along the temporal axis, resulting in better motion continuity and consistency in generated videos. As evidenced in our ablation study (Fig. 6(a)), this addition leads to a significantly lower training loss and improved motion performance.

###### 3.3. LongSANA with Block Linear Attention

This section outlines key components enabling efficient long-video generation. Inspired by the inherent attribute of causal linear attention [11], we explore the constant-memory global KV cache in our block linear attention module, which supports long-context attention with small, fixed GPU memory. Based on this module, we introduce a two-stage autoregressive model continue training paradigm: autoregressive block training with a monotonically increasing SNR sampler and an improved self-forcing method for our long-context attention.

###### 3.3.1. Block Linear Attention with KV Cache

- Table 1 | For a sequence with 𝑁 tokens ∈ R1×𝐷, memory and compute costs are compared among three attention types. Causal linear attention shows best efficiency while maintains global memory.

Metric Causal Full Attention Causal Local Attention Causal Linear Attention

Memory 𝑂(𝑁 × 𝐷) 𝑂(𝑊 × 𝐷) 𝑂(𝐷2) Comp. Cost (𝑁-th token) 𝑂(𝑁 × 𝐷) 𝑂(𝑊 × 𝐷) 𝑂(𝐷2) Comp. Cost (𝑁 tokens) 𝑂(𝑁2 × 𝐷) 𝑂(𝑁 × 𝑊 × 𝐷) 𝑂(𝑁 × 𝐷2)

Limitation of Causal Vanilla Attention. In view of the training objective (Eq. 1), block-wise causal attention is required to implement autoregressive generation. Recent works [5, 6, 17] use a combination of full attention within a block and causal attention to previous blocks. To reduce computational costs, they leverage KV cache, which is effective but comes with memory overhead. For each new token ∈ R1×𝐷 with 𝑁 cached conditional tokens, it requires 𝑂(𝑁 ×𝐷) memory to store the cache and 𝑂(𝑁 × 𝐷) FLOPs for the attention computation. However, since the computational and memory costs grow linearly, these methods [5, 6, 17] often restrict the attention window to a local scope during long video generation. While this maintains a stable cost, it comes at the expense of losing global-context information.

KV Cache in Block Linear Attention. In contrast to the dramatically increased computational and memory cost in causal vanilla attention, linear attention [11] has significant efficiency advantage, naturally supporting long video generation with global attention while maintaining constant memory. Consider the causal attention setting, linear attention (Eq. 2) output for the 𝑖th token can be re-formulated as:

𝜑(𝑄𝑖)(︁∑︀𝑖

𝑗=1 𝜑(𝐾𝑗)𝑇𝑉𝑗)︁ 𝜑(𝑄𝑖)(︁∑︀𝑖

𝜑(𝑄𝑖)(︁∑︀𝑖−1

𝑗=1 𝜑(𝐾𝑗)𝑇𝑉𝑗 + 𝜑(𝐾𝑖)𝑇𝑉𝑖)︁ 𝜑(𝑄𝑖)(︁∑︀𝑖−1

𝑗=1 𝑆𝑗 + 𝑆𝑖)︁ 𝜑(𝑄𝑖)(︁∑︀𝑖−1

𝜑(𝑄𝑖)(︁∑︀𝑖−1

𝑗=1 𝜑(𝐾𝑗)𝑇 + 𝜑(𝐾𝑖)𝑇)︁, (3)

𝑂𝑖 =

𝑗=1 𝜑(𝐾𝑗)𝑇)︁ =

𝑗=1 𝜑(𝐾𝑗)𝑇 + 𝜑(𝐾𝑖)𝑇)︁ =

where 𝑆𝑗 = 𝜑(𝐾𝑗)𝑇𝑉𝑗 denotes the attention state for the 𝑗th token. We omit RoPE here for simplicity. Obviously, as long as the cumulative sum of state

∑︀𝑖−1 𝑗=1 𝑆𝑗 and the cumulative sum of keys

∑︀𝑖−1

𝑗=1 𝜑(𝐾𝑗)𝑇 are stored, only the attention state for the 𝑖th token 𝑆𝑖 ∈ R𝐷×𝐷 is required to compute. Therefore, the memory cost is only

∑︀𝑖−1

𝑗=1 𝑆𝑗 ∈ R𝐷×𝐷 and ∑︀𝑖−1

𝑗=1 𝜑(𝐾𝑗)𝑇 ∈ R𝐷×1, taking 𝑂(𝐷2) in total, and the computational cost is only 𝑂(𝐷2). In Table 1 and Fig. 4(a), we compare the memory and computational cost among causal full attention, causal local attention and our causal

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

[Figure 97]

BlockBlock12

Token-1

[Figure 98]

Zero Padding

Cached Token-1

- Token0

Token-2

- Token1

# Done

Vanilla Attention Linear Attention Causal Linear Attention

(a) Block Causal Linear Attention

(b) Block Causal Mix-FFN

- Figure 4 | Overview of Block Linear Attention. (a) We compare the attention compute mechanism among vanilla attention, linear attention and causal linear attention. illustration of block causal Mix-FFN in processing the adjacent blocks.

|FB1,-1<br><br>Zero Padding<br><br>FB1,-2<br><br><br>(b) The<br><br>attention|
|---|
|Cached FB1,-1<br><br>FB2,0<br>FB2,1<br><br><br>attention, our<br><br>generation, this two operations,|

- Latent Frame in Block 1
- Latent Frame in Block 2

linear attention. Since 𝑁 > 𝑊 >> 𝐷, causal linear achieves the best efficiency and can still maintain global memory in long video generation.

qknorm curve

Block Causal Mix-FFN. In addition to linear at proposed temporal-spatial Mix-FFN enhances locality using convolutional layers. To support long video g module must also operate causally. We ensure causal processing during both training and inference with t as illustrated in Fig. 4(b). First, to prevent information leakage from subsequent blocks during training, we append an all-zero token (‘Zero Padding’ ∈ R1×𝐻𝑊×𝐷) to the end of each block ∈ R𝑇×𝐻𝑊×𝐷. Second, our causal temporal convolution (kernel size 3) requires the last frame of the preceding block. We address this by caching the last token of each block (‘𝑇𝑜𝑘𝑒𝑛−1’ ∈ R1×𝐻𝑊×𝐷) and prepending it to the next. Overall, our causal linear DiT module keeps a fixed memory cache, containing cumulative sum of attention states and keys from all previous frames for attention, along with the last frame of the previous block for Mix-FFN.

###### 3.3.2. Autoregressive Block Training

The continue training of the autoregressive SANA-Video variant, i.e. LongSANA, begins with the pre-trained 5s SANA-Video model. To align with the pre-trained model’s distribution, we propose a monotonically increasing SNR sampler. Specifically, we randomly select a block and sample a timestep for it with the SNR sampler [14]. Then the timesteps for the remaining blocks are sampled via propagated probability [19], ensuring all the timesteps are monotonically increasing, i.e., later blocks have larger timestep than early blocks. This proposed timestep sampler offers two key advantages. First, the monotonically increasing timesteps have a much smaller sampling space than random timesteps, which results in faster convergence and better performance. Second, applying the SNR sampler to a randomly selected block guarantees that every block is trained with sufficient information.

However, monotonically increasing SNR sampler cannot address a severe problem in autoregressive generation, i.e., exposure bias, where condition blocks are ground truth during training but are generated content during inference, leading to error accumulation and limiting performance in long video generation. Self-Forcing [17] aims to address this issue in a vanilla attention DiT model with autoregressive rollout. Limited by the increasing VRAM requirement of causal vanilla attention (Fig. 1(c) and Table 1), Self-Forcing uses local attention within a designed window size. Consequently, it sets the length of self-generated content to be the same as the pre-trained model (i.e., 5s). Later on, LongLive [20] explores streaming long training on 1 minute video, but it still limits to the local attention with sink due to the complexity of full attention. In contrast to the full attention, the block linear attention in LongSANA supports a long-context global KV cache with a small and constant GPU memory. This allows us to further extend LongLive with global attention when self-generating a much longer video (e.g., 1 min), which better aligns the conditioning signals between training and inference and keeps better temporal consistency.

###### 3.3.3. Real-time Long Video Generation with Block Linear Attention

We follows Self-Forcing [17] for autoregressive inference, with the KV cache update based on our design (Alg. 1). Specifically, we first initialize KV cache as empty and start to denoise the first block. After it is fully denoised, the attention state

∑︀0 0 𝑆, cumulative sum of keys

∑︀0

0 𝜑(𝐾)𝑇 and cache for convolustion in Spatial-Temporal Mix-FFN (conv cache 𝑓) will be stored. For the remaining blocks (e.g., 𝑛-th block), they will use the existing KV cache to denoise the latent until clean and then update the cumulative attention state

∑︀𝑛 0 𝑆 and cumulative sum of keys

∑︀𝑛

0 𝜑(𝐾)𝑇. Also, conv cache 𝑓 will be replaced with the new cache. Such update leverages the global while keeping the memory constant

Algorithm 1 Block Linear Diffusion Inference with Linear KV Cache

Require: KV cache Require: Denoise timesteps {𝑡1, . . . , 𝑡𝑇}, noise scheduler Ψ Require: Number of blocks 𝑀 Require: Block-wise diffusion model 𝐺𝜃 (𝐺KV𝜃 returns cumulative sum of state

∑︀

∑︀

𝜑(𝐾)𝑇 and conv cache 𝑓)

𝑆, cumulative sum of key

- 1: Initialize model output X𝜃 ← []
- 2: Initialize KV cache KV ← [None, None, None]
- 3: for 𝑖 = 1, . . . , 𝑀 do
- 4: Initialize 𝑥𝑖𝑡𝑇 ∼ 𝒩(0, 𝐼)
- 5: for 𝑗 = 𝑇, . . . , 1 do
- 6: Set 𝑥^𝑖0 ← 𝐺𝜃(𝑥𝑖𝑡𝑗; 𝑡𝑗, KV)
- 7: if 𝑗 = 1 then
- 8: X𝜃.append(^𝑥𝑖0)
- 9: Update Cache KV ← 𝐺KV𝜃 (^𝑥𝑖0; 0, KV)
- 10: else
- 11: Sample 𝜖 ∼ 𝒩(0, 𝐼)
- 12: Set 𝑥𝑖𝑡𝑗−1 ← Ψ(^𝑥𝑖0, 𝜖, 𝑡𝑗−1)
- 13: end if
- 14: end for
- 15: end for
- 16: return X𝜃

and small, making the long video generation efficient and effective. Attribute to the efficient block linear attention, our 4-step LongSANA is able to generate 1-min and 16 FPS 480P video within 35 seconds on NVIDIA H100 GPU, achieving real-time, 27 FPS generation speed.

###### 3.4. Deep Compression Video Autoencoder

SANA-Video achieves high efficiency and quality for 480P video generation using Wan-VAE. However, even with our efficient linear attention, the generation speed for 720P videos is 2.3× slower. This efficiency drop is even more severe for full attention DiT models (4× for Wan 2.1 1.3B), inspiring us to explore a more efficient VAE that can compress more tokens. We fine-tune DCAE [21] into DCAE-V, with a spatial down-sampling factor of 𝐹 = 32, a temporal factor of 𝑇 = 4, and channels 𝐶 = 32. The number of latent channels aligns with our pre-trained T2I model, enabling fast adaptation from an image to a video model in the same latent space.

The concurrent Wan2.2-5B model also achieves 32 times spatial compression, by combining a VAE with a spatial down-sampling factor of 16 and a patch embedding compression of 2. The advantages of DCAE-V over Wan2.2-VAE are twofold. First, DCAE-V’s 32 latent channels align with our pre-trained T2I model, which improves convergence speed. Second, to achieve the same compression ratio, Wan2.2-VAE would require the model to predict a much larger latent dimension (192 vs. 32 in DCAE-V), a task that is difficult for a small diffusion model (Details in Appendix C.1). As shown in Table 3, DCAE-V exhibits reconstruction performance comparable to other state-of-the-art VAEs like Wan2.1 [3], Wan2.2 [3], and LTX-Video [22]. This high compression allows our model to achieve performance on par with much larger models (e.g., Wan2.1-14B and Wan2.2-5B) while demonstrating significant acceleration, as shown in

- Table 2. Specifically, SANA-Video can generate a 720P 5s video within just 36 seconds, which is a 53× acceleration over Wan2.1-14B. When compared to Wan2.2-5B, which shares the same compression ratio as ours, SANA-Video achieves a 3.2× acceleration.

Table 2 | Latency on H100 GPU and VBench evaluation on 720 × 1280 × 81 resolution videos.

Table 3 | Reconstruction capability of different Autoencoders on Panda-70M 192p resolution.

Models Latency(s) Total ↑ Quality ↑ Semantic ↑ Wan-2.1-14B 1897 83.73 85.77 75.58

Autoencoder Ratio PSNR ↑ SSIM ↑ LPIPS ↓

F8T4C16 (Wan2.1-VAE) 16 34.41 0.95 0.01 F16T4C48 (Wan2.2-VAE) 21 35.61 0.96 0.01 F32T8C128 (LTX-VAE) 64 32.26 0.93 0.04

- Wan-2.1-1.3B 400 83.38 85.67 74.22
- Wan-2.2-5B 116 83.28 85.03 76.28 SANA-Video-2B 36 84.05 84.63 81.73

F32T4C32 (Our DCAE-V) 128 33.25 0.94 0.03

- 3.5. Data Filtering Pipeline

To curate our training dataset, we collect public real and synthetic data and implement a multi-stage filtering paradigm. First, we use PySceneDetect [23] and FFMPEG to cut raw videos into single-scene short clips. For each video clip, we analyze its aesthetic and motion quality, as well as providing detailed captions. Specifically, the motion quality is measured by Unimatch [24] (optical flow) and VMAF [25] (pixel difference), and only clips with moderate and clear motion are kept. Furthermore, the average optical flow is used as a representation of motion magnitude, injecting into prompt for better motion controllability. Aesthetic quality is measured by a pre-trained video aesthetic model (DOVER [26]) and key frame saturation obtained with OpenCV [27], where low aesthetic score and over-saturated videos are removed. Finally, we collect approximately 5,000 human preferred high-quality videos based on stringent motion and aesthetic criteria. The SFT data is collected with diverse but balanced motion and style categories, which can further improve the overall performance. More details are in Appendix D.

Synthetic Data

Public Data

Data Acquisition Raw Data

Scene Cut

TextClips Pairs

Captioning

Filters

Motion

Aesthetic

Saturation

Pre-Training

SFT

Human Selection

Text Embeds

VAE Embeds

Figure 5 | Data filtering paradigm of SANA-Video.

- 4. Experiments

- 4.1. Implementation Details.

Pipeline Settings. For DiT model, to best utilize the pre-trained text-to-image model SANA [9], our SANA-Video-2B is almost identical to those of the original SANA [9], including the diffusion transformer model and small decoder-only text encoder. For 480P videos, we leverage a Wan2.1-VAE [3] autoencoder. For 720P high-resolution video generation, we fine-tune the DCAE [21] into a video deep compression autoencoder (DCAE-V) to facilitate more efficient training and inference. Our final model is trained on 64 H100 GPUs for approximately 12 days. Details are in Appendix B.1.

###### 4.2. Performance Comparison and Analysis

The comprehensive efficiency and performance comparison among SANA-Video with state-of-the-art is illustrated in Table 4. We adopt VBench [31] as the performance evaluation metric and the generation latency of a 480P 81-frame video as efficiency metric. As shown in Table 4, SANA-Video exhibits remarkable latency of 60 seconds, marking it the fastest model compared. This translates to a throughput that is 7.2× faster than MAGI-1 and over 4× faster than Step-Video. In terms of comparison, SANA-Video achieves a Total Score of 83.71 on text-to-video generation, comparable with large model Open-Sora-2.0 (14B) and outperforming Wan2.1 (1.3B). In addition, SANA-Video achieves 88.02 Total Score on image-to-video generation, outperformance large DiT models Wan2.1 (14B) and HunyuanVideo-I2V (11B). Furthermore, SANA-Video achieves the best semantic / I2V score across all the methods, demonstrating strong vision-text semantic alignment.

###### 4.3. Ablation Studies

We then conduct ablation studies on the crucial architectural modifications discussed in Sec. 3.2. As shown in Fig. 6, we provide training loss curves and latency profiles on H100 GPUs.

Linear Attention Module. We incorporate three key designs to enhance our linear attention model. First, we integrate 3D RoPE to focus linear attention on local features (Fig. 3). This improves performance, as evidenced by a significantly lower training loss (Fig. 6(a)). Second, to address differences between linear and vanilla attention, we introduce a Spatial-Temporal Mix-FFN module. Its training loss curve (Fig. 6(b)) demonstrates that a 1D temporal convolution layer significantly enhances performance. Finally, our linear attention design provides a significant efficiency advantage.

Table 4 | Comprehensive comparison of our method with SOTA approaches in efficiency and performance on VBench. The speed is tested on one H100 GPU with BF16 Precision. Latency: Measured with a batch size of 1, on a 480×832×81 video, using the model’s default inference steps for a fair comparison. We highlight the best, second best, and third best entries.

Evaluation scores ↑

Methods Latency Speedup #Params

(s) (B) Total Quality Semantic / I2V Text-to-Video

MAGI-1 [5] 435 1.1× 4.5B 79.18 82.04 67.74 Step-Video [28] 246 2.0× 30B 81.83 84.46 71.28 CogVideoX1.5 [29] 111 4.4× 5B 82.17 82.78 79.76 SkyReels-V2 [6] 132 3.7× 1.3B 82.67 84.70 74.53 Open-Sora-2.0 [25] 465 1.0× 14B 84.34 85.4 80.12 Wan2.1-14B [3] 484 1.0× 14B 83.69 85.59 76.11 Wan2.1-1.3B [3] 103 4.7× 1.3B 83.31 85.23 75.65

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

SANA-Video 60 8.0× 2B 83.71 84.35 81.35 Image-to-Video

MAGI-1 [5] 435 1.1× 4.5B 89.28 82.44 96.12 Step-Video-TI2V [28] 246 2.0× 30B 88.36 81.22 95.50 CogVideoX-5b-I2V [29] 111 4.4× 5B 86.70 78.61 94.79 HunyuanVideo-I2V [30] 210 2.3× 13B 86.82 78.54 95.10 Wan2.1-14B [3] 493 1.0× 14B 86.86 80.82 92.90

SANA-Video 60 8.2× 2B 88.02 79.65 96.40

Block: 1 Block: 3

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

IncreasingRandom

| |
|---|

| |
|---|

TrainingLoss

TrainingLoss

[Figure 105]

[Figure 106]

| |
|---|

| |
|---|

Training Steps

Training Steps

(d) Timestep Settings

(a) 3D RoPE (c) Attention Type Profile

(b) Temporal Conv

15K iterations visualizations

Figure 6 | SANA-Video configuration ablation studies. (a) Training loss curves with and without 3D RoPE. (b) Training loss curves with and without temporal 2D Convolution. (c) Latency comparison of SANA-Video between linear and full attention. (d) Comparison of monotonically increasing versus random timestep sampling in autoregressive block training. Note that monotonically increasing sampling improves consistency across blocks.

Done

As Fig. 6(c) shows, our model’s latency becomes lower at higher resolutions, achieving a 2× speedup at 480P and 4× at 720P, proving its superior efficiency for high-resolution video generation.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

TrainingLoss

TrainingLoss

Monotonically Increasing SNR Sampler. We compare the proposed monotonically increasing SNR sampler with random timestep sampling in the autoregressive block training. As shown in Fig. 6(d) (two columns are from different blocks), monotonically increasing SNR sampler achieves better quality and more consistency across blocks.

Random

Long Video Generation. We compare SANA-Video with previous autoregressive video generation methods on VBench, as shown in Table 5. SANA-Video achieves comparable performance with Self-Forcing [17] while outperforming SkyReel-V2 [6] and CausVid [16].

##### Ablation expo

Training Steps

Training Steps

Increasing

(a) 3D RoPE (c) Attention Type profile

(b) Temporal Conv

(d) timestep settings

###### 5. Applications and Deployment

As a pre-training model, SANA-Video can be easily extended to multiple applications of video generation. First, we adapt SANA-Video to several world model applications (Fig. 1 and Appendix E): embodied AI, autonomous driving and game generation. (Details are in Appendix E). Second, we quantize our model to NVFP4 for efficient inference.

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

BF16

NVFP4

| |
|---|

| |
|---|

Table 5 | Comparison of autoregressive video generation methods on VBench.

130

80

|120<br><br>50<br><br>2.4×|
|---|

|71<br><br>29<br><br>2.4×|
|---|

87

53

Model Total Score ↑ Quality Score ↑ Semantic Score ↑

43

27

CausVid 81.20 84.05 69.80 SkyReels-V2 82.67 84.70 74.53 Self-Forcing 84.31 85.07 81.28

0

0

480×832×81 720×1280×81

50-Step Latency (s) on RTX 5090

Figure 7 | Latency comparison of our model on BF16 and NVFP4 precision.

SANA-Video 83.70 84.43 80.78

Done

On-Device Deployment with 4-Bit Quantization. To facilitate efficient edge deployment, we quantize SANA-Video from BF16 to NVFP4 format using SVDQuant [32]. To balance efficiency and fidelity, we selectively quantize the following layers: the QKV and output projections in self-attention, the query and output projections in cross-attention, and the 1x1 convolutions in feed-forward layers. Other components (normalization layers, temporal convolutions, and KV projections in cross-attention) are kept at higher precision to preserve semantic quality and prevent compounding errors. As shown in Fig. 7, this strategy reduces the end-to-end generation time for a 720p 5-second video from 71s to 29s on a single RTX 5090 GPU, achieving a 2.4× latency speedup while maintaining a quality indistinguishable from the BF16 baseline.

###### quant

###### 6. Related Work

###### 6.1. Video Diffusion Model

Video generation has become a rapidly growing focus in generative AI. Modern approaches typically use a VAE to compress videos into a latent space, where a diffusion model—conditioned on text, images, or both—learns to generate content. Early studies, such as Make-A-Video [33], PYoCo [34] and Tune-A-Video [35], adapted text-to-image models with additional temporal layers to enable video generation. Works like, MagicVideo [36], SVD [37] and Latent Video Diffusion [38] played pioneering roles in scaling latent diffusion approaches. However, the limited compression rate of VAEs has hindered their ability to generalize to long video sequences. A major breakthrough came with Sora [39], which introduced a temporal VAE to compress temporal dimensions alongside spatial ones, while adopting a transformer-based backbone [40] at scale. Recent efforts have pushed this framework further. For instance, Wan 2.2 [3] incorporated a sparse MoE architecture that routes different diffusion steps to specialized experts, while VEO3 [1] extended the paradigm by integrating audio, achieving state-of-the-art performance. The success of MovieGen [7], Seaweed [41], Goku [42], and Waiver [43] further demonstrates the potential of video generation and its broad impact on practical applications. These developments underscore video generation as one of the most dynamic and competitive frontiers in generative AI community.

###### 6.2. Autoregressive Diffusion Model

Autoregressive generation dominates the text domain, while diffusion models have become the standard for visual generation. Recent research explores how to combine these paradigms to duplicate the long-term planning capacity of large language models in vision generation. A straightforward solution [44] is to jointly train an autoregressive (AR) model for text and a diffusion model for vision, but this leaves the visual side reliant solely on diffusion without benefiting from AR modeling. Inspired by block diffusion [15], several works [45, 46, 47, 48, 49] explore AR–diffusion hybrids: MAR [45] disentangles the two, letting AR predict conditions and diffusion reconstruct tokens; ACDiT [46] integrates them via block-wise diffusion with autoregression across blocks, while CausalDiffusion [47] extends this to token-level autoregression. Extending these ideas to video is natural since frames form temporal chunks. FAR [50] generates each frame autoregressively; MarDini [51] employs an AR planner to provide frame-level conditions, with diffusion recovering pixels for tasks such as video interpolation, video extension, and image-to-video generation. Beyond this, MAGI [5] and Skyreel [6] remove the dual-model design, training under the strategy of diffusion forcing [52], where later frames are assigned higher noise levels, thereby enabling infinite autoregressive inter-chunk prediction and high-quality inner-chunk diffusion generation. More recently, self-forcing [17] highlights a gap between training (real data diffused with noise) and inference (model-generated conditions), and proposes rollout-based training to align the two, leading to more robust long-term prediction.

###### 6.3. Efficient Attention for Multimodal Generation.

Diffusion Transformers (DiT) have emerged as the mainstream architecture for visual content generation. Representative models include PixArt-α [53], Stable Diffusion 3 (SD3) [14], and Flux [54], the latter demonstrating the potential of scaling DiT to 12B parameters for high-resolution image synthesis.. To address the computational challenges of vanilla attention (𝑂(𝑛2))), various methods have replaced it with linear-complexity mechanisms. For instance, DiG [55] uses gated linear attention, PixArt-Σ [56] designs key-value token compression, while LinFusion [57] involves Mamba-based structure and SANA [9] employs ReLU linear attention approaches to reduce computational overhead. With the rise of video generation, the computational demands of standard quadratic attention have become a major bottleneck. To address the high computational cost of 3D video attention, many existing works employ factorized spatial and temporal attention to reduce complexity [33, 58, 59, 60]. Other methods reduce attention complexity by selectively skipping certain token interactions [61, 62, 63, 64, 65, 66]. Simultaneously, other models, such as Mamba-based architectures [67, 68], have explored state-space models and linear-complexity designs for efficient video generation. However, these methods either retain some quadratic complexity due to global self-attention layers or are limited to local attention. In contrast, our model maintains a constant-memory KV cache with global attention mechanism, enabling the generation of high-quality, minute-length videos.

###### 7. Conclusion

In this paper, we introduce SANA-Video, a small diffusion model that can efficiently generate high resolution, highquality and long videos at a remarkably fast speed and a low hardward requirement. The significance of SANA-Video lies in the following improvements: linear attention as the core operation, leading to remarkable efficiency improvement in token-extensive video generation task; block linear attention with costant-memory KV cache, supporting minute-long video generation with a fixed memory cost; effective data filters and model training strategies, narrowing the training cost to 12 days on 64 H100 GPUs. With such a small cost, SANA-Video showcases 16× faster speed but competitive performance with modern state-of-the-art small diffusion models.

Acknowledgements. We would like to express our heartfelt gratitude to Shuchen Xue from UCAS, Haocheng Xi from UCB, Songlin Yang, Xingyang Li and Wenkun He from MIT for their invaluable insightful discussions on efficient attention designs, as well as Tian Ye from HKUST(GZ) for his expertise on data curation. Their collaborative efforts and constructive discussions have been instrumental in shaping this work.

###### A. LLM Usage

Our use of large language models (LLMs) was limited to editorial assistance to improve the clarity and readability of this manuscript. Specifically, these tools were used to refine grammar and phrasing, enhance the logical flow between sections, and condense overly verbose passages for conciseness. Crucially, all original research ideas, experimental designs, and data analyses were conceived and executed by the authors; the LLM did not contribute to any scientific or methodological content.

###### B. More Implementation Details

- B.1. Pipeline Configuration

As detailed in Table 6, our SANA-Video-2B model supersedes the original SANA [9] architecture, including the diffusion transformer and a small decoder-only text encoder, to best utilize the pre-trained text-to-image model’s weights. However, we introduce several key modifications to support video generation. We increase the FFN dimension from 5600 to 6720 and the head dimension from 32 to 112 to accommodate 3D RoPE, and we add a temporal convolution in the Mix-FFN module to enhance motion performance. To effectively capture latent features from both images and videos, our approach uses different VAEs based on resolution. For 480P videos, we leverage a Wan2.1-VAE [3] to prioritize reconstruction quality with a lower compression rate (F8T4C16). In contrast, for high-resolution 720P videos, we fine-tune the DCAE [21] into a more aggressive deep compression autoencoder, DCAE-V (F32T4C32), to facilitate more efficient training and inference. For conditional feature extraction, we follow SANA by using a small decoder-only LLM for efficient text processing. For our training strategy, we also employ multi-aspect augmentation to enable arbitrary aspect ratio generation and facilitate image-video joint training, allowing the model to generate both images and videos from a single architecture. The AdamW optimizer [69] is utilized with a weight decay of 0.03 and a constant learning rate of 5e-5. We use Accelerate FSDP [70] for efficient sharded data parallel training. Our final model is trained on 64 H100 GPUs for approximately 12 days.

Table 6 | Architecture details of the proposed SANA-Video.

Model Width Depth FFN #Heads #Param (M)

| | | | | | |
|---|---|---|---|---|---|
|SANA-Video-2B|2240|20|6720|20|2056|

- C. More Results

Please refer to our project link (https://nvlabs.github.io/Sana/Video/), for the qualitative comparison and our generation results.

###### C.1. VAE Comparison

In Sec. 3.4, we analyze the differences and performance of various video VAEs. To select the VAE that best suits our small diffusion model, we conducted a generalization experiment. We hypothesize that a VAE with better reconstruction ability under perturbation will be a better fit, as the diffusion model’s output during inference may be slightly different from the clean latent distribution seen during VAE training. Specifically, we add Gaussian noise to the encoded latent before decoding it, setting 𝑥′𝑡 = 𝑥𝑡 + 𝜖𝑧, where 𝑧 ∼ 𝒩(0,𝐼). As the results in Table 7 show, our DCAE-V performs much more robustly under different noise levels. This demonstrates its superior reconstruction generalization, making it the ideal choice for our small diffusion model.

###### C.2. Qualitative Comparison

Text-to-Video Generation. We compare the text-to-video generation results with current state-of-the-art small diffusion models Wan2.1-1.3B [3] and Wan2.2-5B [3]. As shown in Fig. 8, SANA-Video has comparable semantic understanding, great motion control, and high aesthetic quality.

Image-to-Video Generation. We compare the image-to-video generation results with small diffusion models LTXVideo [22] (2B) and SkyReelv2-I2V [6] (1.3B). As shown in Fig. 9, SANA-Video has the best semantic understanding

Table 7 | Performance comparison of different VAE models on 1000 samples from Panda-70M with different noise perturbation levels.

Model latent shape psnr↑ ssim↑ lpips↓

- Wan2.1VAE (𝜖 = 0) 16, T/4, H/8, W/8 34.41 0.95 0.01
- Wan2.2VAE (𝜖 = 0) 48, T/4, H/16, W/16 35.61 0.96 0.01 DCAE-V (𝜖 = 0) 32, T/4, H/32, W/32 33.25 0.94 0.03

- Wan2.1VAE (𝜖 = 0.1) 16, T/4, H/8, W/8 28.61 0.89 0.06
- Wan2.2VAE (𝜖 = 0.1) 48, T/4, H/16, W/16 30.12 0.92 0.04

- DCAE-V (𝜖 = 0.1) 32, T/4, H/32, W/32 31.91 0.93 0.04

- Wan2.1VAE (𝜖 = 0.2) 16, T/4, H/8, W/8 24.25 0.78 0.16
- Wan2.2VAE (𝜖 = 0.2) 48, T/4, H/16, W/16 25.94 0.84 0.10

- DCAE-V (𝜖 = 0.2) 32, T/4, H/32, W/32 29.34 0.90 0.05

ability (“camera remains steady” instruction in the first case) as well as the best motion control (“slow-motion effect” instruction in the second case) and moderate motion magnitude (first case).

###### C.3. More I2V Results

Our SANA-Video is a unified framework that can perform T2I, T2V and I2V with a single model. We visualize the I2V generation results in Fig. 10. The first column is the reference image and the remaining columns are the generated video. Our SANA-Video can generate semantic consistent and temporal smooth videos based on the first frame.

###### C.4. Influence of Motion Score

As mentioned in our data pipeline (Sec. D), we use the average optical flow value to represent the motion magitude, which is called motion score in our paper. The motion score is added to the text prompt to better control the motion. In Fig. 11, we compare the impact of motion score in the I2V task, which is more clear with the same reference image. By increasing the motion, SANA-Video can generate videos with larger but still consistent motion.

###### C.5. LongSANA Visualization

In Fig. 12, we provide an example of our 1-minute long video generation. LongSANA is able to generate motion consistent and semantically aligned long videos.

###### D. Data Processing Pipeline

To curate our training dataset, we collect a mix of public real and synthetic data, which is then refined through a multistage filtering paradigm, as shown in Fig. 5. We use PySceneDetect [23] to cut raw videos into single-scene, 5-second clips, and Qwen-2.5-VL-7B [13] rewrites prompts to ensure prompt-clip alignment. The data is further filtered based on multiple criteria, including motion, aesthetics, and saturation. We use Unimatch [24] and VMAF [25] for motion, DOVER [26] for aesthetics, and OpenCV [27] for saturation. Finally, for the SFT stage, a subset of approximately 5,000 high-quality videos is selected based on stringent motion and aesthetic criteria and then classified to ensure a balanced and diverse dataset. The details of this data curation process are discussed as follows.

Scene Detection and Shot Cut. In the pre-training stage, we focus on generating 5-second short videos with 16 FPS on a specific scene. However, the raw videos are commonly long and contains more than one scene. Therefore, we cut the raw videos to small video shots with two steps: PySceneDet [23] to split the scenes and FFmpeg [71] to split videos into short clips.

Motion Filtering. Our pre-training dataset comes from multiple sources, and each source of data differs not only in style but also in motion. Motion that is too fast or too slow degrades the motion performance of SANA-Video. Following Open-Sora [8], we apply Unimatch [24] and Vmaf to score the motion of each video. Unimatch can evaluate the optical flow of two given images of the same shape. We select frames from each video every 0.5 seconds, reshape them into 320x576, and calculate the average optical flow over all selected frames. Vmaf, on the other hand, simply computes the

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

|E6F0E7|F7E8E6|E6EDF2|
|---|---|---|
| | | |
| | | |

SANA-VideoWan2.1-1.3BWan2.2-5B

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

| |
|---|

| |
|---|

Prompt: Photorealistic movie scene still, medium shot: A flamingo with iridescent pink feathers and elegant ballet shoes performs a flawless pirouette on a grand, ornate stage. Its neck is gracefully curved, one leg extended, highlighting intricate body motion. Intense spotlight illuminates iridescent feathers and polished wood floor. Background: photorealistic, opulent opera house interior, slightly blurred. Dramatic, theatrical lighting enhances elegance and drama.

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

SANA-VideoWan2.1-1.3BWan2.2-5B

Done

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

#### quant

Prompt: A lone painter in a dark suit and vintage cap sits in the center of an expansive foggy wilderness, calmly painting on an easel. Suddenly, flames erupt on his canvas, black smoke rising into a gloomy sky. The background is a vast green field stretching to a blurred horizon, symbolizing creation and destruction. Medium shot, artist in focus amidst the chaos.

- Figure 8 | Qualitative comparison among T2V methods. SANA-Video has comparable motion control and video-text semantic alignment with state-of-the-art small diffusion models.

pixel difference of two images; we use FFmpeg [71] to compute the Vmaf over all consecutive frames and normalize them. Due to the variance of different video sources, we analyze the motion scale and set the appropriate motion range individually, ensuring our data has moderate and clear motion. During pre-training, we also append Motion score: {unimatch value} to the text prompt to help control the motion magnitude of the generated videos (Fig. 11).

Aesthetics Many Text-to-Image works have proven that high aesthetic data can improve the training efficiency of an image generation model [53]. We believe that this also applies to the video generation model. We use Dover [26] to score each video for its aesthetics. Dover produces three different scores: aesthetic score, technical score, and overall score, among which we use the overall score as the filter metric.

Saturation We also observe that some of our data, especially synthetic data and real data converted from HDR to SDR, has unnatural color, appearing in high saturation. To prevent these data from damaging the output quality of SANA-Video, we use OpenCV [27] to compute a saturation score of each video. We select frames from each video every 0.5 seconds, convert their color representation from RGB to HSV, where the “S” channel in HSV color representation stands for saturation. By averaging the “S” channel over all pixels and frames, we obtain the saturation score of a video. We keep only videos with a saturation score lower than a threshold set to a reasonable value for each data source.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

SANA-VideoSANA-VideoLTX-VideoSkyReel-V2LTX-VideoSkyReel-V2

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Prompt: The individual's expression shifts from

###### neutral to intense ... The camera remains steady, focusing on the

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

person's face and upper body.

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Prompt: A majestic brown cow across a dusty field under a clear blue sky. The camera captures the animal from a low angle ... A slowmotion effect enhances the fluidity of the cow's movement...

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

I2V compare Done

- Figure 9 | Qualitative comparison among I2V methods. SANA-Video has better motion control and video-text semantic alignment.

Captioning [3] shows that LLM rewritten prompts can produce more accurate and detailed prompts within the same distribution, and thus make the model easier to learn, and consequently enhance the model’s performance. Moreover, for synthetic data with existing prompts, replacing their prompts with LLM rewritten ones helps reduce the misalignment between the original prompts and their synthetic output. Following [3], we use Qwen-2.5-VL [72] to caption our data as shown in Fig. 13.

SFT Data For our final stage of SFT training, we selected approximately 5,000 high-quality videos based on stringent criteria for motion and aesthetics. The motion requirement is fulfilled by the presence of either distinct object motion, camera motion, or both. Ideal object motion is characterized by a moderate magnitude and a clearly focused action that is free from occlusions. Similarly, any camera movement must be stable and smooth, without jittering, to maintain 3D consistency. The aesthetic criteria are equally comprehensive. Beyond technical qualities like balanced brightness and natural color, we prioritize videos with appealing overall content and layout, demonstrated by thoughtful composition and engaging subject matter. Following this filtering process, the videos were classified into four motion categories (human activities, animal activities, other objects, natural or urban scenes) and three aesthetic styles (realistic, cartoon, cinematic). This strategic sampling across diverse categories is crucial for ensuring both the model’s high performance and the breadth of its capabilities. The influence of fine-tuning on the SFT data is illustrated in Fig. 14, where both the aesthetic details (the eyes in the first example), and the motion realism (the pipe of the second example) will be improved.

###### E. World Model

We fine-tune SANA-Video on several downstream tasks to demonstrate the potential of applying SANA-Video to world model related generation: embodied AI, autonomous driving and game generation.

World Model for Embodied AI. The first important downstream task for video generation is embodied AI, where SANA-Video can be used to generate simulation data for robot training. In this task, we leverage AgiBot [73] as the

###### Reference Frame

###### Generation Frames

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

A fluffy Ragdoll cat is seen stirring a pot on the stove with a wooden spoon…

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Two young men stand together on a bustling city street at night, taking a selfie…

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

A man is skydiving, suspended mid-air against a backdrop of fluffy…

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

A serene valley unfolds beneath a cloudy sky, with a river winding…

Figure 10 | Visualization of image-to-video generation. SANA-Video can keep consistent with the first frame while generating realistic motion.

I2V Example Done

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Motion30Motion10

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Figure 11 | The impact of motion score on I2V task. Higher motion score can lead to larger motion.

training data, which contains synchronized views of different camera views. The head-front view is adopted as the target videos and filtered with our data pipeline. The generation results are shown in the first row of Fig. 15.

World Model for Autonomous Driving. Video generation model is also a good simulator for autonomous vehicle scenarios, and SANA-Video can be used to generate diverse and realistic driving scenes. In this task, we fine-tune SANA-Video on internal driving data, using the front camera with 30 FOV. The generation results are shown in the second row of Fig. 15.

## Motion Score Done

World Model for Game Generation. We explore downstream game generation to create interactive video games. Specifically, we use VPT [74] as the training data, containing screen recording videos of players playing Minecraft. The raw videos are cut and processed following our data pipeline in Appendix D. In addition, we train a small classifier to identify low-quality data in the scenario. The generation results are shown in the third row of Fig. 15.

|[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>LongSANA<br><br>Done<br><br>0-10s<br><br>10-20s<br><br>20-30s<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>A white Arctic fox runs gracefully across a fallen log in a dense forest. The fox's fur is pristine white, and it moves with agility and purpose, its tail held high. The camera follows the fox closely, capturing its fluid movements and the serene beauty of the woodland setting. The fox's ears are perked up, and its eyes are focused ahead, embodying the spirit of freedom and wildness.<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]<br><br>30-40s<br><br>40-50s<br><br>50-60s<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]|
|---|

###### Figure 12 | Long video visualization of LongSANA.

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

User

You are a video captioning specialist whose goal is to generate high-quality English prompts by referring to the details of the

user's input videos. Your task is to carefully analyze the content, context, and actions within the video, and produce a

User

complete, expressive, and natural-sounding caption that accurately conveys the scene. The caption should preserve the

original intent and meaning of the video while enhancing its clarity and descriptive richness. Strictly adhere to the formatting of the examples provided.

Task Requirements:

- 1. You need to describe the main subject of the video in detail, including their appearance, actions, expressions, and the

surrounding environment.

- 2. You need to emphasize movement information in the input and different camera angles.
- 3. Your output should convey natural movement attributes, incorporating natural actions related to the described subject

category, using simple and direct verbs as much as possible.

- 4. You should reference the detailed information in the video, such as character actions, clothing, backgrounds, and

emphasize the details in the photo.

- 5. Control the output prompt to around 80-100 words.
- 6. No matter what language the user inputs, you must always output in English. Example of the English prompt:

- 1. A Japanese fresh film-style photo of a young East Asian girl with double braids sitting by the boat. The girl wears a white

square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly

melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests

her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The photo has a vintage film texture. A medium shot of a seated portrait.

- 2. An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly

dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white

waist tie and a name tag in bold Chinese characters that says "紫阳" (Ziyang). The background has a light yellow indoor tone,

with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A

close-up shot from a slightly elevated perspective.

- 3. CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its

back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs,

and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.

- 4. In the style of an American drama promotional poster, Walter White sits in a metal folding chair wearing a yellow

protective suit, with the words "Breaking Bad" written in sans-serif English above him, surrounded by piles of dollar bills and

blue plastic storage boxes. He wears glasses, staring forward, dressed in a yellow jumpsuit, with his hands resting on his

knees, exuding a calm and confident demeanor. The background shows an abandoned, dim factory with light filtering through

the windows. There's a noticeable grainy texture. A medium shot with a straight-on close-up of the character.

Directly output the English text.

A cat leaps off a balcony, mid-air jump with its legs extended and tail flowing behind, from a low-angle view, capturing the moment just as it leaves the edge.

Qwen-2.5-VL

Figure 13 | An overview of the captioning pipeline.

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

| |
|---|

w.SFTw.o.SFT

| |
|---|

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

| |
|---|

| |
|---|

Figure 14 | Analysis of the influence of SFT. Fine-tuning on the human preferred SFT data can improve the video details and ad e to the laws of physics.

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

|adherence|
|---|

w.SFTw.o.SFT

| |
|---|

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

RoboticsDrivingMineCraft

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

| |
|---|

| |
|---|

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

SFT Inﬂuence Done

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Figure 15 | Visualization of world model task generation.

18

### World Model Done

###### References

- [1] Google DeepMind. Veo 3. https://https://deepmind.google/models/veo/, 2025.
- [2] Kuaishou. Kling ai. https://klingai.kuaishou.com/, 2024.
- [3] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [4] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [5] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [6] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Juncheng Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengchen Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.
- [7] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.
- [8] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [9] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han. SANA: Efficient high-resolution text-to-image synthesis with linear diffusion transformers. In ICLR, 2025.
- [10] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [11] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In ICML, 2020.
- [12] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. In ICML, 2025.
- [13] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.
- [15] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.
- [16] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.
- [17] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [18] Jianlin Su. Transformer upgrade path: 2. rotary position encoding with comprehensive advantages. https://kexue.fm/ archives/8265, Mar 2021.
- [19] Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In CVPR, 2025.
- [20] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.
- [21] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.

- [22] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.
- [23] Brandon Castellano. PySceneDetect.
- [24] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11):13941–13958, 2023.
- [25] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in $200 k. arXiv preprint arXiv:2503.09642, 2025.
- [26] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20144–20154, 2023.
- [27] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal of Software Tools, 2000.
- [28] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.
- [29] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [30] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [31] Fan Zhang, Shulin Tian, Ziqi Huang, Yu Qiao, and Ziwei Liu. Evaluation agent: Efficient and promptable evaluation framework for visual generative models. arXiv preprint arXiv:2412.09645, 2024.
- [32] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024.
- [33] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.
- [34] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22930–22941, October 2023.
- [35] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7623–7633, 2023.
- [36] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.
- [37] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [38] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023.
- [39] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [41] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025.

- [42] Shoufa Chen, Chongjian Ge, Yuqi Zhang, Yida Zhang, Fengda Zhu, Hao Yang, Hongxiang Hao, Hui Wu, Zhichao Lai, Yifei Hu, et al. Goku: Flow based video generative foundation models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23516–23527, 2025.
- [43] Yifu Zhang, Hao Yang, Yuqi Zhang, Yifei Hu, Fengda Zhu, Chuang Lin, Xiaofeng Mei, Yi Jiang, Zehuan Yuan, and Bingyue Peng. Waver: Wave your way to lifelike video generation. arXiv preprint arXiv:2508.15761, 2025.
- [44] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.
- [45] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024.
- [46] Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720, 2024.
- [47] Chaorui Deng, Deyao Zhu, Kunchang Li, Shi Guang, and Haoqi Fan. Causal diffusion transformers for generative modeling. arXiv preprint arXiv:2412.12095, 2024.
- [48] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388, 2025.
- [49] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Flowar: Scale-wise autoregressive image generation meets flow matching. arXiv preprint arXiv:2412.15205, 2024.
- [50] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.
- [51] Haozhe Liu, Shikun Liu, Zijian Zhou, Mengmeng Xu, Yanping Xie, Xiao Han, Juan C Pérez, Ding Liu, Kumara Kahatapitiya, Menglin Jia, et al. Mardini: Masked autoregressive diffusion for video generation at scale. arXiv preprint arXiv:2410.20280, 2024.
- [52] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [53] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-𝛼: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [54] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [55] Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7664–7674, 2025.
- [56] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-𝜎: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024.
- [57] Songhua Liu, Weihao Yu, Zhenxiong Tan, and Xinchao Wang. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097, 2024.
- [58] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [59] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [60] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, 133(5):3059–3078, 2025.

- [61] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. In ICML, 2025.
- [62] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. NeurIPS, 2025.
- [63] Xingyang Li*, Muyang Li*, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, Maneesh Agrawala, Ion Stoica, Kurt Keutzer, and Song Han. Radial attention: 𝒪(𝑛 log 𝑛) sparse attention with energy decay for long video generation. NeurIPS, 2025.
- [64] Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattn: Accurate sparse attention accelerating any model inference. In ICML, 2025.
- [65] Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Vsa: Faster video diffusion with trainable sparse attention. NeurIPS, 2025.
- [66] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhenghong Liu, and Hao Zhang. Fast video generation with sliding tile attention. ICML, 2025.
- [67] Hongjie Wang, Chih-Yao Ma, Yen-Cheng Liu, Ji Hou, Tao Xu, Jialiang Wang, Felix Juefei-Xu, Yaqiao Luo, Peizhao Zhang, Tingbo Hou, et al. Lingen: Towards high-resolution minute-length text-to-video generation with linear computational complexity. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2578–2588, 2025.
- [68] Yu Gao, Jiancheng Huang, Xiaopeng Sun, Zequn Jie, Yujie Zhong, and Lin Ma. Matten: Video generation with mamba-attention. arXiv preprint arXiv:2405.03025, 2024.
- [69] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [70] Accelerate: An Extension of PyTorch for enabling easy deployment of models for large-scale training, 2022.
- [71] FFmpeg Developers. FFmpeg. http://ffmpeg.org, 2025.
- [72] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [73] AgiBot World Colosseum contributors. Agibot world colosseum. https://github.com/OpenDriveLab/AgiBotWorld, 2024.
- [74] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. Advances in Neural Information Processing Systems, 35:24639–24654, 2022.

