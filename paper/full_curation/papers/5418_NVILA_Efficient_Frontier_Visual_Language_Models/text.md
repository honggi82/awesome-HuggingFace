[Figure 1]

2026-4-28

## NVILA: Efficient Frontier Visual Language Models

Zhijian Liu1,† Ligeng Zhu1,† Baifeng Shi3 Zhuoyang Zhang2 Yuming Lou6 Shang Yang2 Haocheng Xi3 Shiyi Cao3 Yuxian Gu2,6 Dacheng Li3 Xiuyu Li3 Yunhao Fang4 Yukang Chen1 Cheng-Yu Hsieh5 De-An Huang1 An-Chieh Cheng4 Vishwesh Nath1 Andriy Myronenko1 Jinyi Hu2,6 Sifei Liu1 Ranjay Krishna5 Daguang Xu1 Xiaolong Wang1,4 Pavlo Molchanov1 Jan Kautz1 Hongxu Yin1,‡ Song Han1,2,‡ Yao Lu1,†,‡

1NVIDIA 2MIT 3UC Berkeley 4UC San Diego 5University of Washington 6Tsinghua University †Equal contribution ‡Equal advisory

Abstract: Visual language models (VLMs) have made significant advances in accuracy in recent years. However, their efficiency has received much less attention. This paper introduces NVILA, a family of open VLMs designed to jointly optimize efficiency and accuracy. Building on top of VILA, we improve its model architecture by first scaling up the spatial and temporal resolutions, and then compressing visual tokens. This “scale-then-compress” approach enables NVILA to efficiently process high-resolution images and long videos. We further conduct a systematic investigation that enhances NVILA’s efficiency throughout its entire lifecycle, from training and fine-tuning to deployment. NVILA matches or surpasses the accuracy of leading open and proprietary VLMs across a wide range of image and video benchmarks. At the same time, it reduces training cost by 1.9–5.1×, prefilling latency by 1.6–2.2×, and decoding latency by 1.2–2.8×. We release our code and models to facilitate reproducibility.

# arXiv:2412.04468v3[cs.CV]25Apr2026

Links: Code (on GitHub) | Models (on Hugging Face)

VILA-1.5 LLaVA-OV Qwen2-VL NVILA (Ours)

InternVL2

Training Time (GPU days)

Pre-filling Latency (ms)

Decoding Speed (token/s)

LongVideoBench

AI2D

ChartQA

VQAv2

405

55

163 131

ImageVideo

5.1X Faster

35

Video-MME (w/ sub.)

TextVQA

DocVQA

MLVU

80

LLaVA-OV NVILA

Qwen2-VL NVILA

Qwen2-VL NVILA

405

1446

145

InfoVQA

1.9X Faster

(image)SEED

2.2X Faster 2.8X Faster

208

652

51

RealWorldQA

Video-MME (w/osub.)

MVBench

MathVista

LLaVA-OV NVILA

Qwen2-VL NVILA

Qwen2-VL NVILA

MMMU

(a) Training

(b) Deployment

(c) Accuracy on image and video benchmarks

(1.9-5.1X Faster) (Pre-fill: 1.6-2.2X Faster / Decode: 1.2-2.8X Faster)

(On-par or superior accuracy on all benchmarks)

Figure 1 | NVILA – Efficient Frontier VLMs. (a) NVILA trains image and video models 5.1× and 1.9× faster, respectively, than LLaVA-OneVision (OV), the only baseline with publicly reported training costs. (b) Against Qwen2-VL, NVILA achieves a 1.6–2.2× speedup in the prefilling stage and a 1.2–2.8× speedup in the decoding stage. (c) This efficiency comes at no cost to accuracy; in fact, NVILA delivers comparable or superior accuracy across image and video benchmarks. All models in this figure have 8B parameters. Training time in (a) is measured on NVIDIA H100 GPUs; inference speed in (b) is measured on a single NVIDIA RTX 4090 GPU. Accuracy in (c) is normalized to the highest score on each benchmark.

### 1. Introduction

community has made substantial progress in improving their accuracy [1, 2, 3, 4, 5] and broadening their applications across diverse domains, including robotics [6, 7, 8], autonomous driving [9], and medical applications [10, 11]. However, much less attention

Visual language models (VLMs) integrate visual and textual inputs, enabling a wide range of visionlanguage interactions. In recent years, the research

© 2026 NVIDIA. All rights reserved.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

##### 1979 2007

Question: What is the diﬀerence of export share in High income Asia and Paciﬁc between two periods? Write out the calculation process?

Question: What is the implication of temperature based on this image?

[Figure 6]

Answer: The implication of temperature based on this image is that the Earth's temperature has increased over the years, leading to the melting of ice caps and glaciers.

Answer: The chart shows the share of U.S. agricultural exports by destination for two periods: 1995-99 and 2011-15. To calculate the diﬀerence in export share for High income Asia and Paciﬁc between these periods, we need to subtract the share for 1995-99 from the share for 2011-15. From the chart, we can see that:

|the player who is dunking?|
|---|

[Figure 7]

- * In 1995-99, the share for High income Asia and Paciﬁc was 33%.
- * In 2011-15, the share for High income Asia and Paciﬁc was 20%. The diﬀerence in export share for High income Asia and Paciﬁc between the two periods is calculated as follows: Diﬀerence = Share in 2011-15 - Share in 1995-99 Diﬀerence = 20% - 33% Diﬀerence = -13% Therefore, the diﬀerence in export share for High income Asia and Paciﬁc between the two periods is -13%.

Question: What is the number of

th

Question: Which city does this player's team belong to? Answer: The player's team belongs to Denver. This can be seen from the "Denver" logo on the athlete's helmet.

Answer: The player who is dunking is wearing number 4. This player is in mid-air, executing a powerful dunk over the rim of the basketball hoop.

[Figure 8]

[Figure 9]

Question: Summarize the information in the boarding pass with json format.

Answer: {"Departure": "Boston", "Arrival": "Charlotte", "Date": "Tuesday, September 07, 2021", "Flight": "AA653", "Gate": "B4", "Terminal": "B", "Seat": "17B", "Boarding Time": "8:50 AM", "Departure Time": "9:25 AM", "Arrival Time": "11:55 AM", "Inﬂight Services": "WiFi", "Priority": "1", "Boarding Ends": "15 minutes before departure"}

Question: Which drug should use for young female? Answer: Drug Y.

Question: Which drug should use for young male? Answer: Drug C.

Question: Which drug should use for middle age? Answer: Drug X.

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

Question: How many times does the character jump in this video? Answer: The character jumps twice in this video.

Question: How many purple coins does he new collect during this process? Answer: He collects a total of 10 purple coins.

[Figure 22]

[Figure 23]

Question: Analyze the meaning of the sign. Answer: The sign in the image is a warning sign for a winding road. It is a yellow diamond-shaped sign with a black squiggly line in the middle, indicating a winding road ahead. The sign is placed on the side of the road to alert drivers to the upcoming winding road. The sign is also accompanied by a smaller sign below it that reads "30 M.P.H.", indicating the recommended speed limit for the winding road.

Question: I want to go Rochester, which lane should I stay? Answer: To reach Rochester, you should stay in the middle lane. The highway sign indicates that Rochester is located in the middle lane, as shown by the downward arrow pointing to the center of the road. This lane will lead you directly to Rochester.

- Figure 2 | Qualitative examples. NVILA produces accurate responses across a diverse set of tasks, including OCR on text-heavy documents, multi-step visual reasoning, and multi-image comparison.

temporal resolutions, and then compressing visual tokens. Scaling preserves more visual detail, raising the accuracy ceiling, while compression reduces the visual token count, lowering computational cost. This “scale-then-compress” strategy enables NVILA to process high-resolution images and long videos both accurately and efficiently. We further systematically optimize efficiency across NVILA’s full lifecycle: training, fine-tuning, and deployment.

Image Video

| |[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]| | | |
|---|---|---|---|---|

|[Figure 27]|
|---|

[Figure 28]

More tiles

More frames …

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

| |
|---|

| |
|---|

| |
|---|

…

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|
| |Visual Encoder (VIT)| |
| | | |
| |Projector (MLP)| |
| | | |

These innovations make NVILA both efficient and accurate. It reduces training cost by 1.9–5.1×, prefilling latency by 1.6–2.2×, and decoding latency by 1.2–2.8×. It also matches or surpasses the accuracy of leading open VLMs [5, 3, 2] and proprietary VLMs [12, 13] across a wide range of image and video benchmarks. NVILA further opens up new application domains, including temporal localization, robotic navigation, and medical imaging. We release our code and models to support reproducibility and to inspire further work on efficient VLMs.

Spatial token compression

Temporal token compression

What’s in this image/video?

Compressed visual tokens Textural tokens

### 2. Approach

Token Processor (LLM)

We first describe NVILA’s efficient model architecture, which follows a “scale-then-compress” design: scaling up spatial and temporal resolutions to raise the accuracy ceiling, then compressing visual tokens to recover efficiency. We then present strategies to improve efficiency across NVILA’s full lifecycle: training, fine-tuning, and deployment. Unless otherwise specified, all analysis uses the 8B model.

A cat.

- Figure 3 | NVILA architecture. NVILA consists of three components: a visual encoder (SigLIP), a projector (two-layer MLP), and a token processor (Qwen2 LLM). Input images are processed by Dynamic-S2 tiling and spatially compressed before being fed to the LLM alongside text tokens.

has been paid to their efficiency.

VLMs are expensive across multiple dimensions. First, training a VLM is time-consuming: training a state-of-the-art 7B VLM [4] can take up to 400 GPU days, with even higher costs for larger models. This creates a significant entry barrier for researchers. Second, VLMs often require adaptation to specialized domains (e.g., medical imaging), but fine-tuning a VLM is memory-intensive: fully fine-tuning a 7B VLM can require over 64GB of GPU memory, well beyond what is available on most consumer-grade GPUs. Finally, VLMs are increasingly deployed in edge applications with a tight compute budget (e.g., laptops, robots), so deploying a VLM is latency-sensitive. Addressing all three together requires a comprehensive approach to VLM efficiency.

In this paper, we introduce NVILA, a family of open VLMs designed to jointly optimize efficiency and accuracy. Building on VILA [2], we improve its model architecture by first scaling up the spatial and

#### 2.1. Efficient Model Architecture

We build NVILA on top of VILA [2], an autoregressive VLM. As shown in Figure 3, it consists of three components: a visual encoder that extracts features from visual inputs (e.g., images, videos); a projector that aligns embeddings across visual and language modalities; and a token processor, typically instantiated with an LLM, which takes both visual and language tokens as input and produces language tokens. Specifically, NVILA uses SigLIP [14] as its visual encoder, a two-layer MLP as its projector, and Qwen2 [15] (in various parameter sizes) as its token processor.

VILA processes images at a fixed 448×448 resolution regardless of aspect ratio, and samples at most 14 frames per video∗. Both choices introduce significant information loss and limit the model’s ability to handle high-resolution images and long videos. As a result, VILA lags behind leading VLMs, especially on

∗This is the configuration for VILA-1.5 40B; other variants, such as VILA-1.5 3B, use 384×384 resolution and 8 frames.

text-heavy image and long-video benchmarks (Table 9 and Table 10).

We therefore advocate for the “scale-then-compress” paradigm: first scale up the spatial and temporal resolutions to raise the accuracy ceiling, then compress the visual tokens to recover efficiency. Scaling alone, however, significantly increases compute: doubling the resolution quadruples the number of visual tokens, and self-attention’s quadratic complexity in the LLM further amplifies the cost. Compression then offsets this overhead, since higher information density lets the model retain (or even surpass) the detail captured at lower resolution while using fewer total tokens.

#### 2.1.1. Spatial “Scale-Then-Compress”

For spatial scaling, naively increasing the encoder’s input resolution (e.g., to 896×896) imposes uniform overhead on all images regardless of their content. Instead, we adopt S2 [16], which extracts multi-scale high-resolution features via adaptive image tiling. Given a vision encoder pre-trained at 4482 resolution and an input image of arbitrary size, S2 first resizes the image to multiple scales (e.g., 4482, 8962, 13442); at each scale, it splits the image into 4482 tiles that are independently processed by the encoder. The per-tile feature maps from the same scale are stitched back into a feature map for the whole image at that scale. Finally, feature maps from different scales are interpolated to a common spatial size and concatenated along the channel dimension.

S2 always resizes images to a square regardless of their original aspect ratio, which can cause distortion for images with extreme aspect ratios. To address this, we propose Dynamic-S2, which adaptively processes images of varying aspect ratios. Dynamic-S2 follows S2, but at the largest scale, instead of resizing to a square, it picks the closest size that preserves the input’s aspect ratio and is divisible into 4482 tiles. This design is inspired by the dynamic resolution strategy in InternVL [17].

With Dynamic-S2, NVILA captures richer highresolution detail, yielding up to 30 points of absolute accuracy gain on text-heavy benchmarks (Table 1). The remaining challenge is to compress these spatial tokens. VILA [2] shows that a simple 2×2 spatialto-channel (STC) reshape reduces the token count by 4× without sacrificing accuracy. Pushing further, however, hurts accuracy: increasing the STC ratio to 3×3 leads to a nearly 10-point accuracy drop on DocVQA. We hypothesize that more aggressive token reductions make the projector substantially harder to train. To address this, we introduce an additional visual encoder pre-training stage that jointly tunes the vision encoder and projector. This stage recovers

most of the accuracy loss from 3×3 compression while preserving its 2.4× speedup over the 2×2 baseline in both training and inference.

We also explored alternative designs for spatial token compression, such as TokenLearner from RT1 [6] and Perceiver Resampler from MiniCPM-V [18]. At the same token-reduction ratio, these learnable methods perform no better than the simple spatialto-channel design, even with the additional visual encoder pre-training stage. We attribute this to optimization difficulty rather than representational capacity, and leave a deeper investigation to future work.

#### 2.1.2. Temporal “Scale-Then-Compress”

For temporal scaling, we uniformly sample more frames per video. Following prior work [19], we include additional video SFT to enable the model to process longer sequences. As shown in Table 2, increasing from 8 to 32 frames improves Video-MME accuracy by more than 5 points, but it also increases the number of visual tokens by 4×.

As with spatial tokens, we compress the temporal representations to recover efficiency. Since consecutive frames often contain similar information, we adopt temporal averaging [20], which partitions frames into groups and pools visual tokens within each group. This reduces temporal redundancy while retaining important spatiotemporal detail. Empirically, 4× compression yields only a modest accuracy drop. Compared with the original baseline at the same token budget, the scale-then-compress model has nearly the same cost† and substantially higher accuracy. Pushing further to 256 frames with an 8× compression ratio yields state-of-the-art results among 7–8B opensource models on Video-MME (Table 10).

#### 2.2. Efficient Training

Training a state-of-the-art VLM is costly and compute-intensive. This section explores systemalgorithm co-design for efficient VLM training. On the algorithm side, we examine an unsupervised dataset pruning method that streamlines the training set. On the system side, we investigate FP8 mixed-precision training for acceleration.

#### 2.2.1. Dataset Pruning

Prior work [21, 4, 22] has aggregated ever-larger SFT datasets from diverse sources, yielding consistent benchmark improvements. However, not all data contributes equally, and unchecked dataset growth

†Running the visual encoder on more frames adds overhead, but this is not the runtime bottleneck.

- Table 1 | Spatial “scale-then-compress”. Scaling the spatial resolution with Dynamic-S2 greatly improves accuracy, particularly on text-heavy benchmarks. Compressing visual tokens via spatial pooling reduces both tile count and tokens-per-tile at a moderate accuracy cost. Adding a visual encoder pre-training (VEP) stage recovers most of this loss. IM-10 denotes the average score across the 10 image benchmarks in Table 9.

Spatial Pooling

#Tokens/Tile #Tiles/Image AI2D DocVQA TextVQA IM-10

Baseline (VILA-1.5) 2×2 256 (=16×16) 1 87.0 61.3 67.5 61.2 Scale (Dynamic-S2) 2×2 256 (=16×16) 9–12 90.1 91.1 77.0 71.5 Scale + Compress 3×3 121 (=11×11) 1–12 87.4 82.3 74.1 67.1 Scale + Compress + VEP 3×3 121 (=11×11) 1–12 89.8 88.8 76.1 70.8 Alternative Designs

TokenLearner – 121 1–12 90.0 86.5 75.6 69.8 Perceiver Resampler – 121 1–12 76.8 71.8 65.3 59.4

- Table 2 | Temporal “scale-then-compress”. Scaling up the number of frames consistently improves video understanding. Compressing visual tokens via temporal averaging reduces token count substantially with only a marginal accuracy drop, allowing NVILA to process up to 256 frames at a similar token budget.

Video-MME (w/o sub.) Short Medium Long Overall

Temporal Pooling

#Frames

#Tokens/Video

Baseline (VILA-1.5) 8 1× 2048 (=162×8) 65.4 53.8 47.7 55.7 Scale 32 1× 8192 (=162×32) 73.2 58.9 50.9 61.0 Scale + Compress 32 4× 2048 (=162×32/4) 73.7 56.7 50.0 60.1 Scale + Compress 256 8× 8192 (=162×256/8) 75.0 62.2 54.8 64.0

leads to significant redundancy. In NVILA, we apply the “scale-then-compress” principle to data: we first expand our SFT dataset mixture and then compress it via pruning. While prior work has explored data selection for vision inputs [23, 24, 25] and text-only inputs [26, 27, 28], few studies address the mixed image-text setting of VLM training. With tens of millions of training samples, pruning the dataset without sacrificing accuracy is essential.

Inspired by recent work in knowledge distillation [29], we score each training example by the DeltaLoss

𝑝large(𝑥) 𝑝small(𝑥)

, (1)

Δ(𝑥) = log

where 𝑝large(𝑥) and 𝑝small(𝑥) are the probabilities a large and a small reference VLM, respectively, assign to the answer tokens of 𝑥. Given the full SFT mixture 𝐷 =

⋃︀𝑁

𝑖=1 𝐷𝑖 partitioned into 𝑁 source-level subsets and a target keep-ratio 𝜌 ∈ (0,1], we form the pruned training set

⋃︁𝑁

𝑖|⌉{︀

}︀

𝐷′ =

top⌈𝜌|𝐷

Δ(𝑥) ⃒ 𝑥 ∈ 𝐷𝑖

, (2)

𝑖=1

which keeps the top-scoring ⌈𝜌|𝐷𝑖|⌉ examples within each subset. The main motivation is to filter out examples that are either too easy or too distracting:

- • If both models answer correctly, or both fail, Δ(𝑥) is near zero; the example offers little discriminative signal.
- • When the small model answers correctly but the large model fails, Δ(𝑥) is negative, suggesting the example tends to distract learning and will eventually be forgotten by a more capable model.
- • When the small model fails but the large model succeeds, Δ(𝑥) is positive, indicating strong supervision: challenging for small models but learnable by larger ones.

We apply DeltaLoss to each subset 𝐷𝑖 and ablate over a range of keep-ratios 𝜌.

We evaluate DeltaLoss against cluster pruning and random pruning (Table 3). For random pruning, we sample examples uniformly at random and report the mean over three runs. For cluster pruning, we apply 𝑘-means clustering on SigLIP features and prune uniformly across clusters. We sweep three keep-ratios (𝜌 ∈ {10%,30%,50%}) and report average performance across 10 image benchmarks (IM-10). DeltaLoss consistently outperforms both baselines; on DocVQA in particular, random pruning degrades sharply while DeltaLoss retains accuracy. At a 50% keep-ratio, the average score remains competitive while training time is halved. We therefore adopt 𝜌 = 50% for all subsequent experiments.

[Figure 34]

[Figure 35]

[Figure 36]

Question: <image>What is the weather in this photo like? Answer the question using a single word or phrase. Answer: Snowy DeltaLoss: 0.0343 (too easy ❌ )

Question: <image>\nWhat color is the canopy? A. white/yellow B. green/white C. blue/white D. red/white

Question: <image> Which action depicted is a sign of respect? Answer the question using a single word or phrase. Answer: Hat over heart DeltaLoss: 4.1605 (helpful ✅ )

Answer with the option's letter from the given choices directly. Answer: D DeltaLoss: -1.916 (wrong answer ❌ )

- Figure 4 | DeltaLoss visualization. Left, Middle, and Right panels show training examples scored as too easy, distracting, and informative by DeltaLoss, respectively.

- Table 3 | Dataset pruning on the NVILA recipe. DeltaLoss consistently outperforms other data selection methods, with minimal accuracy loss at a 50% keep-ratio.

Table 4 | Dataset pruning on the Pixmo recipe [30]. DeltaLoss outperforms other data selection methods in most settings, with negligible accuracy drop at a 50% keep-ratio.

|E6F0E 7|F7E8E 6|E6EDF 2|
|---|---|---|
| | | |
| | | |

Method IM-10 MMMU DocVQA TextVQA 100% (baseline) 75.6 48.0 90.1 78.8 50% DeltaLoss [29] 75.5 48.1 89.7 78.4 Cluster Pruning 74.5 47.8 88.3 77.0 Random Pruning 74.0 47.6 87.1 76.6 30% DeltaLoss [29] 74.0 47.8 87.9 76.4 Cluster Pruning 73.5 47.7 84.1 76.0 Random Pruning 73.1 47.7 82.9 75.6 10% DeltaLoss [29] 72.4 47.1 84.4 74.5 Cluster Pruning 72.2 47.4 79.6 73.2 Random Pruning 72.0 47.0 77.3 72.6

Method IM-10 MMMU DocVQA TextVQA 100% (baseline) 74.9 45.8 90.0 76.4 50% DeltaLoss [29] 75.2 47.2 89.2 76.5 Cluster Pruning 75.0 47.1 89.0 76.3 Random Pruning 74.6 45.1 89.6 76.2 30% DeltaLoss [29] 74.3 46.8 88.7 74.7 Cluster Pruning 73.4 46.4 87.2 74.6 Random Pruning 72.7 45.4 88.8 74.1 10% DeltaLoss [29] 72.6 46.5 88.1 74.2 Cluster Pruning 72.2 46.2 87.5 73.9 Random Pruning 71.4 45.4 87.7 72.8

Question: Can you explain the meme? Answer: This meme is a humorous take on the high cost of modern gaming hardware. It features a split image of a man's face, showing two contrasting reactions to the price of an RTX 2080 Ti graphics card. The top half of the image shows the man with a wide, excited smile, representing his initial enthusiasm for the powerful graphics card. The bottom half, however, depicts him with a shocked expression, mouth open in disbelief at the price tag of $1,200. The meme cleverly captures the emotional rollercoaster many gamers experience when considering high-end hardware purchases. It highlights the excitement of owning cutting-edge technology, followed by the sticker shock of its hefty price. This format is a popular way to express the contrast between initial excitement and subsequent disappointment or surprise, particularly in the context of expensive tech products.

We further examine the impact of data pruning on newly added datasets. We incorporate varying portions of Pixmo data [30] into the NVILA training set. As shown in Table 4, naively combining Pixmo with the NVILA training set degrades DocVQA and TextVQA while only improving MMMU, suggesting that indiscriminate dataset growth can hurt performance. Applying DeltaLoss to prune the Pixmo data yields consistent improvements across benchmarks, even at small keep-ratios.

The NVIDIA Hopper and Blackwell architectures (e.g., H100 and B200) now also support FP8 natively, promising greater compute and memory efficiency than BF16.

FP8 has been widely applied to LLM training. NVIDIA’s Transformer Engine performs matrix multiplications (GEMM) in FP8, accelerating training. FP8-LM [33] additionally quantizes gradients, the weight master copy, and the first-order momentum to FP8, reducing communication overhead and memory footprint. COAT [34] further compresses activations and the optimizer’s second-order momentum, improving memory efficiency while maintaining accuracy.

#### 2.2.2. FP8 Training

FP16 [31] and BF16 [32] are the standard training precisions, trading off some of FP32’s numerical stability for faster compute on NVIDIA GPUs.

We adopt the FP8 implementation from COAT [34]

Table 5 | FP8 training. FP8 accelerates NVILA training without degrading accuracy, with the largest gains when gradient checkpointing (GC) is disabled. Throughput is measured at the largest batch size (BS) that fits on 64 H100 GPUs. Video-MME uses an 8frame setting with subtitles.

GC BS Throughput MMMU Video-MME

BF16 ✗ 4 199.2 (1.0×) 47.9 52.9 FP8 ✗ 16 390.1 (2.0×) 47.0 53.0

BF16 ✓ 30 491.7 (2.5×) 47.8 53.1 FP8 ✓ 36 579.9 (2.9×) 47.7 53.0

to accelerate NVILA training. A key difference between LLM and VLM training is sequence-length variability. After packing, LLM training samples tend to be similar in length, so throughput is relatively insensitive to batch size. VLM training samples, in contrast, vary widely in length: video samples may require tens of thousands of tokens, image samples a few hundred to a few thousand, and text-only samples far fewer. As a result, batches dominated by short samples underutilize the GPU and benefit substantially from a larger batch size. As shown in Table 5, applying FP8 to both weights and activations allows NVILA to increase the batch size from 4 to 16, producing a 2× speedup. When gradient checkpointing is enabled, quantizing activations is less essential; instead, we integrate the cross-entropy kernel from Liger [35] to reduce peak memory due to Qwen’s large vocabulary. In this setting, FP8 training still provides

- a 1.2× speedup over BF16.

#### 2.3. Efficient Fine-Tuning

Once a foundation VLM is trained, domain-specific fine-tuning adapts it to specialized tasks. Conventional parameter-efficient fine-tuning (PEFT) methods are largely studied in the context of LLMs, and how to best fine-tune a VLM remains underexplored. In NVILA, we find that (i) the learning rate should be set differently for the ViT and the LLM, and (ii) tuning only a small subset of ViT parameters can match full PEFT.

When jointly fine-tuning the vision encoder (ViT) and the language model (LLM) with PEFT methods, we observe that the ViT’s learning rate should be 5–50× smaller than the LLM’s. We also find that finetuning only the ViT’s LayerNorm layers matches the accuracy of LoRA (Table 6) while reducing training time by 25% relative to applying LoRA to the vision encoder. With this configuration, NVILA can be finetuned to various downstream tasks within 24GB of GPU memory while maintaining on-par performance.

Table 6 | Fine-tuning recipe. Our recommendation is to tune the LLM with either LoRA or QLoRA and to tune ViT’s layer normalization (LN) layers with a much smaller learning rate. This setup achieves competitive accuracy and is also the most memory- and compute-efficient. All experiments use a batch size of 1 with gradient checkpointing disabled, and throughput is measured on a single NVIDIA A100 80GB GPU. For settings with {1,5,10,50}, we select the learning rate ratio from this set that gives the best results for each benchmark. “FT-5” refers to the average accuracy across AITZ [36], ALFRED [37], nuScenes [38], PathVQA [39], and Widget Caption [40].

Memory (GB)

Throughput (iter/s) LRLLM/LRViT

Accuracy (FT-5)

ViT LLM

1 69.2 {1,5,10,50} 71.8

LoRA LoRA 20.1 3.4

1 63.5 {1,5,10,50} 71.4

LN LoRA 19.2 4.5

1 64.0 {1,5,10,50} 70.1

FT LoRA 21.9 4.2

1 63.0

LoRA QLoRA 11.1 2.6

- {1,5,10,50} 70.8

LN QLoRA 10.2 3.1

1 62.7

- {1,5,10,50} 70.9

FT FT 63.5 6.1 1 77.7

#### 2.4. Efficient Deployment

VLMs are increasingly deployed in resourceconstrained edge settings such as robotics, where both latency and memory are limited. To deploy NVILA efficiently, we build a specialized inference engine that targets the two phases of inference, prefilling and decoding, with phase-specific quantization.

In the compute-bound prefilling stage, we first apply token compression (Section 2.1) to reduce the LLM backbone’s inference workload. After compression, the vision tower becomes the primary bottleneck, accounting for over 90% of the prefilling latency. We therefore apply W8A8 quantization to the vision tower to reduce NVILA’s time-to-first-token (TTFT). For the memory-bound decoding stage, we follow AWQ [41] and apply W4A16 quantization to the LLM backbone. We further optimize the original AWQ implementation by introducing FP16 accumulation in the W4A16 GEMM kernels, yielding a 1.7× kernel speedup without compromising accuracy. A detailed comparison is provided in Figure 5.

- Table 7 | Quantization recipe. W4A16 quantization on the LLM backbone introduces a small accuracy drop, while W8A8 on the ViT is nearly lossless. Together, they reduce TTFT by 28%. TTFT: timeto-first-token.

ViT LLM AI2D MMMU VideoMME TTFT (s)

FP16 FP16 91.0 50.7 63.9 0.90 FP16 W4A16 90.9 49.2 62.0 0.77

W8A8 W4A16 90.9 49.3 62.1 0.65

- Table 8 | Training recipe. NVILA adds two stages to VILA’s pipeline: Stage 2 pre-trains the visual encoder to recover the accuracy lost to spatial token compression, and Stage 5 fine-tunes the model on video data to extend long-video understanding. “LR” denotes the initial learning rate of the trainable components in each stage.

Visual Encoder Projector Token Processor

(ViT) (MLP) (LLM) LR Initial from [14] random from [15] –

- Stage 1 frozen trainable frozen 1×10-3
- Stage 2 trainable trainable frozen 5×10-5
- Stage 3 frozen trainable trainable 5×10-5
- Stage 4 trainable trainable trainable 2×10-5
- Stage 5 trainable trainable trainable 2×10-5

### 3. Experiments

#### 3.1. Training Details

We train NVILA in a five-stage pipeline: (1) projector initialization, (2) visual encoder pre-training, (3) token processor pre-training, (4) image instructiontuning, and (5) video instruction-tuning. Stages 1, 3, and 4 are also part of VILA training. Stage 2 recovers the accuracy loss caused by spatial token compression (Table 1), and Stage 5 extends the model’s long-video understanding capability. The detailed training recipe is given in Table 8, and the data recipe in Table A1.

Our implementation is built on PyTorch 2.3.0 [42, 43] and Transformers 4.46.0 [44]. We use DeepSpeed 0.9.5 [45] to shard large models across devices and gradient checkpointing to reduce memory usage. We adopt FlashAttention-2 [46] to accelerate training in both the LLM and the visual encoder. We also implement function-preserving, on-the-fly sequence packing to fuse samples of different lengths, yielding roughly a 30% speedup. All models are trained on 128 NVIDIA H100 GPUs with a global batch size of 2048 across all stages, using AdamW without weight decay. We adopt a cosine learning-rate schedule with a linear warm-up over the first 3% of training. Per-stage initial

learning rates are listed in Table 8.

We release two model variants: NVILA-Lite maximizes efficiency by applying all the techniques described in this paper, while NVILA trades a small amount of efficiency for higher accuracy. Both variants share the same training pipeline.

3.2. Accuracy Results

- 3.2.1. Image Benchmarks

We conduct comprehensive evaluations on a diverse set of image benchmarks (Table 9): AI2D [47], ChartQA [48], DocVQA [49], InfographicVQA [50], MathVista [51], MMMU [52] (with zero-shot CoT), RealWorldQA [53], SEED-Bench [54], TextVQA [55], and VQAv2 [56].

NVILA is competitive with the strongest opensource models in each size category, including Qwen2VL [5], InternVL2 [3], and Pixtral [57]. On general VQA tasks (ChartQA, DocVQA, InfoVQA, TextVQA, VQAv2, SEED), NVILA (8B and 15B) is on par with or surpasses proprietary models such as GPT4o and Gemini 1.5 Pro. On the science benchmark AI2D, NVILA leads its size group at both 8B and 15B, with NVILA-15B (94.1) approaching the proprietary frontier (GPT-4o 94.2, Gemini 1.5 Pro 94.4, Claude 3.5 Sonnet 94.7). On reasoning and knowledge benchmarks (MMMU, RealWorldQA, MathVista), NVILA-15B leads among open-source models in its size group, with the largest gain over the 8B model coming on MMMU. On OCR-heavy benchmarks (ChartQA, DocVQA, TextVQA), the 8B model is also highly competitive. We provide qualitative examples in Figure 2 to illustrate NVILA’s OCR, reasoning, and multi-image capabilities.

- 3.2.2. Video Benchmarks

We evaluate our models on a range of video understanding benchmarks [58, 59, 60, 61], spanning short clips of a few seconds to videos up to an hour in duration. Table 10 compares NVILA against strong baselines [62, 63, 5, 4, 64, 19]. With the “scale-thencompress” design, NVILA supports long-context video understanding with up to 256 frames. NVILA-8B achieves state-of-the-art results on all evaluated benchmarks among open-source 7–8B models and matches the proprietary GPT-4o mini.

- 3.3. Efficiency Results

NVILA achieves competitive accuracy on image and video benchmarks while remaining efficient through the “scale-then-compress” design. Architecturally, we scale up to native resolution (using 1–12 tiles per

- Table 9 | Image benchmarks. Best result among open-source models within each size group is in bold and the second-best is underlined; proprietary and grayed-out larger models (>15B parameters) are shown for reference and are not included in bolding.

AI2D ChartQA DocVQA InfoVQA MathVista MMMU Real WorldQA

SEED TextVQA VQAv2 test test test test testmini val test pro image val testdev

GPT-4o – 94.2 85.7 92.8 79.2 63.8 69.1 64.7 51.9 75.4 76.2 77.4 78.7 Claude 3.5 Sonnet – 94.7 90.8 85.2 74.3 67.7 68.3 63.7 51.5 60.1 – 74.1 70.7 Gemini 1.5 Pro – 94.4 87.2 93.1 81.0 63.9 62.2 57.6 43.5 70.4 – 78.7 80.2

LLaVA-1.5 7B 55.5 17.8 28.1 25.8 25.6 35.7 – – 54.8 66.1 58.2 78.5 VILA-1.5 8B 76.6 52.7 40.6 25.9 36.7 38.6 32.7 – 52.7 73.8 68.5 83.0 Cambrian-1 8B 73.0 73.3 77.8 41.6 49.0 42.7 – – 64.2 74.7 71.7 81.2 Florence-VL 8B 74.2 74.7 84.9 51.7 55.5 43.7 – – 64.2 74.9 74.2 84.7 LLaVA-OneVision 8B 81.4 80.0 87.5 68.8 63.2 48.8 42.8 24.1 66.3 75.4 78.3 84.0 Llama 3.2 11B 91.9 83.4 88.4 – 51.5 50.7 – – – – – 75.2 InternVL2 8B 83.8 83.3 91.6 74.8 58.3 51.2 42.6 29.0 64.2 76.2 77.4 76.7 Qwen2-VL 8B 83.0 83.0 94.5 76.5 58.2 54.1 46.6 30.5 70.1 76.0 84.3 82.9 NVILA-Lite 8B 91.0 84.8 91.7 67.9 64.5 50.7 45.7 26.5 65.6 76.3 78.1 85.0 NVILA 8B 92.3 86.1 93.7 70.7 65.4 49.9 44.4 27.8 68.6 76.5 80.1 85.4

LLaVA-1.5 13B 61.1 18.2 30.3 29.4 27.7 37.0 – – 55.3 68.2 61.3 80.0 VILA-1.5 13B 79.9 59.5 58.6 30.4 42.7 37.9 33.6 – 57.5 72.6 65.0 82.8 Cambrian-1 13B 73.6 73.8 76.8 – 48.0 40.0 – – 63.0 74.4 72.8 – Pixtral 12B 79.0 81.8 90.7 50.8 58.0 52.5 – – 65.4 – 75.7 80.2 NVILA-Lite 15B 92.0 81.8 90.6 69.3 61.7 58.7 51.8 33.7 67.1 75.6 77.3 83.7 NVILA 15B 94.1 86.9 94.0 73.5 66.1 56.7 51.8 33.8 69.5 76.6 80.0 84.8

LLaVA-NeXT 34B – – – – 46.5 48.1 44.5 22.9 – 75.9 69.5 83.7 Cambrian-1 34B 79.7 75.6 75.5 46.0 53.2 49.7 – – 67.8 75.3 76.7 83.8 VILA-1.5 40B 88.9 67.8 58.6 38.4 49.3 51.9 46.9 25.0 60.8 69.1 73.6 84.3 InternVL2 40B 87.1 86.2 93.9 78.7 63.7 55.2 47.4 34.2 71.8 78.2 83.0 – LLaVA-OneVision 72B 85.6 83.7 91.3 74.9 67.5 56.8 52.3 31.0 71.9 75.4 80.5 85.2 NVLM-D-1.0 78B 94.2 86.0 92.6 – 65.2 59.7 54.6 – 69.7 – 82.1 85.4 Llama 3.2 90B 92.3 85.5 90.1 – 57.3 60.3 – 39.5 – – – –

image) and then compress visual tokens with a 3×3 spatial-to-channel reshape, yielding a 2.4× speedup with little accuracy loss. On the data side, we curate a diverse 10M-sample SFT dataset, prune it to a 5M high-quality subset with DeltaLoss, and still consistently outperform LLaVA-OneVision, which trains on more than 8M samples. We further integrate FP8 training for higher throughput, jointly tune ViT/LLM learning rates for fine-tuning efficiency, and apply W8A8 quantization to the vision tower to reduce latency. Together, these full-stack optimizations let NVILA train faster, fine-tune with less memory, and infer with lower latency, all while improving accuracy.

We compare NVILA’s inference performance against Qwen2-VL [5] in Figure 5. For a fair comparison, both models sample 64 frames per video, with all experiments conducted on a single NVIDIA RTX 4090 GPU. Qwen2-VL is quantized to W4A16 and served with vLLM [65], a state-of-the-art LLM/VLM serving engine. For NVILA, we quantize the LLM backbone to W4A16 and the vision tower to W8A8. With our specialized inference engine, NVILA achieves up to

- a 2.2× speedup in prefilling and up to 2.8× higher

decoding throughput over Qwen2-VL.

### 4. More Capabilities

#### 4.1. Temporal Localization

Following LITA [66], we add support for temporal localization in NVILA. We introduce discrete time tokens to represent video timestamps and train the model with a smoothed cross-entropy loss. As shown in Table 11, NVILA substantially outperforms all baselines across all metrics.

#### 4.2. Robotic Navigation

NVILA can serve as a strong foundation for robotic agents in Vision-Language Navigation [67] and supports real-time deployment on resource-constrained edge devices. At each time step 𝑡, the agent receives a language instruction and a video observation, plans the next action, and transitions to the next state 𝑡+1, where it receives a new observation. NVILA’s efficient and flexible handling of multi-frame inputs

- Table 10 | Video benchmarks. Best result in bold. #F: number of frames sampled. acc.: accuracy. m-avg: M-Avg metric (multiple-choice average) used by MLVU. mc: multiple-choice. w/o sub. / w/ sub.: without / with subtitles.

ActivityNet-QA LongVideoBench MLVU MVBench NExT-QA Video-MME #F acc. score val test m-avg test mc w/o sub. w/ sub.

GPT-4o mini – – – – 56.5 58.8 – – – 64.8 68.9 GPT-4o – – 61.9 – 66.7 66.7 64.6 – – 71.9 77.2

LLaVA-NeXT-Video 7B 32 53.5 3.2 43.5 43.5 – 33.7 – 46.5 – Video-XL 7B 2048 – – 49.5 51.3 64.9 55.3 77.2 55.5 61.0 InternVL2 8B 64 – – 54.6 – 64.0 65.8 – 56.3 59.3 LLaVA-OneVision 8B 32 56.6 – 56.5 – 64.7 56.7 79.4 58.2 61.5 Oryx-1.5 8B 128 – – 56.3 – 67.5 67.6 81.8 58.8 64.2 LongVILA 7B 256 59.5 – 57.1 – – 67.1 80.7 60.1 65.1 LongVU 7B 1fps – – – – 65.4 66.9 – 60.6 – Qwen2-VL 8B 2fps – – 55.6 56.8 65.5 67.0 – 63.3 69.0 NVILA 8B 256 60.9 3.7 57.7 58.7 70.1 68.1 82.2 64.2 70.0

- Table 11 | Temporal localization. LITA results are from the original paper; VILA-1.5 results are from our reproduction. NVILA uses the same data mixture

- as VILA-1.5, with only the base VLM replaced.

ActivityNet-RTL #Frames Mean IoU Precision@0.5

LITA [66] 7B 100 24.1 21.1 LITA [66] 13B 100 28.6 25.9 VILA-1.5 8B 256 32.1 29.3 NVILA 8B 256 34.8 32.1

- Table 12 | Robotic navigation. All results except NVILA’s are from NaVILA [8]. All models use only RGB inputs. NE: Navigation Error (↓); OS: Oracle Success; SR: Success Rate; SPL: Success Rate weighted by Path Length.

Table 13 | Medical application. Results of NVILAM3 on medical benchmarks; task-specific baselines and datasets are described in [11]. Metrics: accuracy for VQA; BLEU-4 and ROUGE for report generation; F1 for classification.

VQA Report Gen. Classif. Rad Path CXR CheXpert

Med-Gemini – 78.8 83.3 20.5 28.3 48.3 VILA-M3 8B 84.7 91.0 21.1 32.0 61.6 NVILA 8B 85.5 92.9 22.8 32.8 61.1

Task-spfc. SOTA 84.2 91.7 15.4 30.6 51.5

#### 4.3. Medical Application

NVILA also offers strong potential in the medical domain. The NVILA-M3 framework [11] integrates multiple domain-expert models tailored to specific medical tasks, such as image segmentation and classification. These expert models extract and interpret intricate features that are difficult for general VLMs to discern. By coupling specialized models with a vision-language learning paradigm, NVILAM3 learns nuanced relationships between visual inputs and textual annotations, improving both task-specific accuracy and robustness. As reported in Table 13, NVILA-M3 achieves an overall 9% improvement over task-specific state-of-the-art baselines through the use of expert models. This demonstrates the value of combining domain expertise with general-purpose VLMs in precision-critical fields.

R2R Val-Unseen Obs. NE ↓ OS ↑ SR ↑ SPL ↑

Seq2Seq – RGB 10.10 8.0 0.0 0.0 CMA – RGB 9.55 10.0 5.0 4.0 NaVid 7B RGB 5.47 49.0 37.0 35.0 NVILA 8B RGB 5.43 60.4 53.3 48.8

enables seamless integration of historical and current observations. The NaVILA framework [8] introduces a tailored navigation prompt and fine-tunes NVILA using navigation-specific SFT data curated from a simulator [68]. As shown in Table 12, NVILA’s straightforward design achieves competitive results on VLN-CE. Real-world deployment results are shown in Figure 6: the full camera→GPU→action pipeline runs at 1Hz.

Decoding Throughput

Vision Tower LLM Backbone

[Figure 37]

[Figure 38]

|137<br><br>212<br><br>268|995<br><br>2.22x| |
|---|---|---|
| |871<br><br>966| |

| |144.6<br><br>144.6<br><br>144.6<br><br>|2.84x| |
|---|---|---|---|
| | | | |

38.5

630

NVILA- FP16 +Token Compression +W4A16 LLM +FP16 Accumulation +W8A8 ViT Qwen2-VL-FP16 +W4A16 LLM

51.9

630

[Figure 39]

[Figure 40]

630

[Figure 41]

[Figure 42]

630

1

515

137

31.1

572

Baseline

50.8

575

0 30 60 90 120 150

0 300 600 900 1200 1500

Throughput for Video Input (tokens/sec)

TTFT for Video Input (ms)

| | |
|---|---|
|34.5<br><br>56<br><br>1.55x|.8<br><br>|

|1.2|4x|
|---|---|
| | |

14.0

44.0

55.2

NVILA- FP16 +Token Compression +W4A16 LLM +FP16 Accumulation +W8A8 ViT Qwen2-VL-FP16 +W4A16 LLM

[Figure 43]

[Figure 44]

14.0

44.0

55.2

14.0

40.6

162.8

14.0

25.1

162.8

10.3

25.1

162.8

20.0

33.9

Baseline

20.5

3

130.5

0 16 32 48 64 80

0 36 72 108 144 180

Throughput for Image Input (tokens/sec)

TTFT for Image Input (ms)

(a) Time-To-First-Token (TTFT) breakdown (b) Throughput of decoding stage

- Figure 5 | Inference efficiency breakdown. We benchmark NVILA-8B against Qwen2-VL-7B [5] on image and video tasks, attributing the incremental gain to each optimization. Qwen2-VL is served via vLLM [65] with W4A16 quantization; NVILA is deployed with our specialized inference engine. NVILA achieves 1.6–2.2× faster prefilling and up to 2.8× higher decoding throughput. All measurements are taken on a single NVIDIA RTX 4090 GPU.

### 5. Related Work

#### 5.1. Visual Language Models

VLMs, especially proprietary ones, have advanced rapidly over the past two years. OpenAI upgraded from GPT-4V [69] to GPT-4o [12], achieving 5–10% gains across image and video QA benchmarks. Google extended the context length to 1M tokens in Gemini 1.5 Pro [70], a significant step up from Gemini 1.0 [71]; Gemini 1.5 Pro tops the Video-MME leaderboard [61] for long-video understanding. Anthropic released Claude 3.5 [13], which is competitive with GPT-4o and improves notably over Claude 3 [72]. Other proprietary models have followed similar trajectories, including Apple’s MM1 → MM1.5 [73] and xAI’s Grok-1.5 [53] → Grok-2 [74].

Meanwhile, open-source VLMs continue to evolve

- at both the system/framework level [75] and the algorithm/recipe level [2], progressively narrowing the gap to proprietary models [19, 76, 77, 78, 5]. Many recent open VLMs now match or exceed GPT-4V and GPT-4o on key benchmarks. Representative examples include InternVL2 [3], Qwen2-VL [5], LLaVAOneVision [4], Llama 3.2 Vision [79], Molmo [30], NVLM [78], and MiniCPM-V [18].

Despite this progress in accuracy, much less attention has been paid to the efficiency of training, fine-tuning, and deployment. This paper explores how to build VLMs that are both highly accurate and

optimized for end-to-end efficiency.

#### 5.2. Efficiency

Token reduction. Prior work [80, 81, 82, 63, 83, 84, 85, 86] has studied spatial and temporal token reduction, but none focuses on reducing tokens for a frontier VLM, where preserving accuracy at scale is the primary challenge.

Data selection. For LLM pre-training, promising approaches include domain mixing [87], samplewise data selection [27, 88], and theory-driven optimal selection [28]. Comparatively few studies tackle the mixed image-text setting of VLM training; in this work we specifically target supervised fine-tuning (SFT) data for VLMs.

Low-precision training. FP8 training [89, 90] has gained traction for LLMs, but to our knowledge no prior work has demonstrated its feasibility for VLMs without sacrificing accuracy.

Pruning, distillation, and quantization. Pruning and distillation [91, 92] are commonly applied to LLMs, but their best practice for VLMs remains open: should the LLM be pruned or distilled before adding a vision encoder, or should the assembled VLM be pruned or distilled? Similarly, quantization methods such as AWQ [41] and GPTQ [93] are well-studied for LLMs, and VILA [2] has shown that AWQ transfers directly to VLMs. However, little attention has been

[Figure 45]

Goal Position Oracle Path Prediction

Instruction: Exit the living room and turn right into the kitchen. Turn left at the end of the counter and wait in the room across the hallway slightly to the left. Agent: The next action is turn left 15 degrees.

[Figure 46]

[Figure 47]

Instruction: Walk forward out of the room. Turn right and enter the other room and stop in front of the table. Agent: The next action is move forward 75 cm.

- Figure 6 | Robotic navigation. NVILA serves as a Vision-Language Navigation agent that takes language instructions and visual observations as input (top: simulation; bottom: real-world). The real-world platform is a Unitree Go2 robot equipped with a LiDAR sensor and an Intel RealSense camera. An RTX 4090 GPU runs NVILA-8B, processing 8 frames per action step.

paid to quantizing the vision encoder, which becomes critical at higher input resolutions and longer videos.

Parameter-efficient fine-tuning. Methods such as LoRA [94], DoRA [95], QLoRA [96], and GaLore [97] are widely used to reduce LLM fine-tuning memory. For VLMs, which combine a vision encoder with an LLM, efficient fine-tuning techniques are still underexplored, a gap we address in this work.

### 6. Conclusion

This paper introduced NVILA, a family of open VLMs designed to strike a strong balance between efficiency and accuracy. By adopting the “scalethen-compress” paradigm, NVILA processes highresolution images and long videos efficiently while maintaining high accuracy. We further systematically optimize its efficiency across the entire lifecycle, from training and fine-tuning to deployment. NVILA matches or surpasses current leading VLMs in accuracy while being substantially more resource-efficient, and it opens up new possibilities for applications such as temporal localization, robotic navigation, and medical imaging. We hope NVILA will empower researchers and developers across a wide range of applications and research directions.

### References

- [1] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In Conference on Neural Information Processing Systems (NeurIPS), 2024.
- [2] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. VILA: On Pretraining for Visual Language Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [3] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [4] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy Visual Task Transfer. arXiv:2408.03326, 2024.
- [5] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv:2409.12191, 2024.
- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-1: Robotics Transformer for Real-World Control at Scale. In Robotics: Science and Systems (RSS), 2022.
- [7] Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and Wang He. NaVid: Videobased VLM Plans the Next Step for Vision-andLanguage Navigation. In Robotics: Science and Systems (RSS), 2024.
- [8] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Xueyan Zou, Jan Kautz, Erdem Biyik, Hongxu Yin,

- Sifei Liu, and Xiaolong Wang. NaVILA: Legged Robot Vision-Language-Action Model for Navigation. arXiv:2412.04453, 2024.
- [9] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. DriveVLM: The Convergence of Autonomous Driving and Large VisionLanguage Models. In Conference on Robot Learning (CoRL), 2024.
- [10] Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, et al. Capabilities of Gemini Models in Medicine. arXiv:2404.18416, 2024.
- [11] Vishwesh Nath, Wenqi Li, Dong Yang, Andriy Myronenko, Mingxin Zheng, Yao Lu, Zhijian Liu, Hongxu Yin, Yee Man Law, Yucheng Tang, Pengfei Guo, Can Zhao, Ziyue Xu, Yufan He, Greg Heinrich, Stephen Aylward, Marc Edgar, Michael Zephyr, Pavlo Molchanov, Baris Turkbey, Holger Roth, and Daguang Xu. VILA-M3: Enhancing VisionLanguage Models with Medical Expert Knowledge. arXiv:2411.12915, 2024.
- [12] OpenAI. GPT-4o, 2024.
- [13] Anthropic. Claude 3.5, 2024.
- [14] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid Loss for Language Image Pre-Training. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
- [15] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 Technical Report. arXiv:2407.10671, 2024.
- [16] Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, and Trevor Darrell. When Do We Not Need Larger Vision Models? In European Conference on Computer Vision (ECCV), 2024.
- [17] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min

- Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. arXiv:2404.16821, 2024.
- [18] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. MiniCPM-V: A GPT-4V Level MLLM on Your Phone. arXiv:2408.01800, 2024.
- [19] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, and Song Han. LongVILA: Scaling Long-Context Visual Language Models for Long Videos. arXiv:2408.10188, 2024.
- [20] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal Segment Networks: Towards Good Practices for Deep Action Recognition. In European Conference on Computer Vision (ECCV), 2016.
- [21] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A Fully Open, VisionCentric Exploration of Multimodal LLMs. In Conference on Neural Information Processing Systems (NeurIPS), 2024.
- [22] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What Matters When Building Vision-Language Models? In Conference on Neural Information Processing Systems (NeurIPS), 2024.
- [23] Cody Coleman, Christopher Yeh, Stephen Mussmann, Baharan Mirzasoleiman, Peter Bailis, Percy Liang, Jure Leskovec, and Matei Zaharia. Selection via Proxy: Efficient Data Selection for Deep Learning. In International Conference on Learning Representations (ICLR), 2020.
- [24] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying CLIP Data. In International Conference on Learning Representations (ICLR), 2024.
- [25] Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. SemDeDup: DataEfficient Learning at Web-Scale through Semantic Deduplication. arXiv:2303.09540, 2023.
- [26] Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari Morcos. D4: Improving LLM Pretraining via Document De-Duplication and Diversification. In Conference on Neural Information Processing Systems (NeurIPS), 2023.

- [27] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: Selecting Influential Data for Targeted Instruction Tuning. In International Conference on Machine Learning (ICML), 2024.
- [28] Yuxian Gu, Li Dong, Hongning Wang, Yaru Hao, Qingxiu Dong, Furu Wei, and Minlie Huang. Data Selection via Optimal Control for Language Models. arXiv:2410.07064, 2024.
- [29] Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. MiniPLM: Knowledge Distillation for Pre-Training Language Models. arXiv:2410.17215, 2024.
- [30] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, KuoHao Zeng, Jon Borchardt, Dirk Groeneveld, Jen Dumas, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models. arXiv:2409.17146, 2024.
- [31] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed Precision Training. In International Conference on Learning Representations (ICLR), 2018.
- [32] Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan, Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. A Study of BFLOAT16 for Deep Learning Training. arXiv:1905.12322, 2019.
- [33] Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Zhang, Chen Li, Jia Ning, Ruizhe Wang, Zheng Zhang, Shuguang Liu, Joe Chau, Han Hu, and Peng Cheng. FP8-LM: Training FP8 Large Language Models. arXiv:2310.18313, 2023.
- [34] Haocheng Xi, Han Cai, Ligeng Zhu, Yao Lu, Kurt Keutzer, Jianfei Chen, and Song Han. COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training. arXiv:2410.19313, 2024.

- [35] Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, and Yanning Chen. Liger Kernel: Efficient Triton Kernels for LLM Training. arXiv:2410.10989, 2024.
- [36] Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the Zoo: Chain-of-Action-Thought for GUI Agents. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024.
- [37] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
- [38] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
- [39] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. PathVQA: 30000+ Questions for Medical Visual Question Answering. arXiv:2003.10286, 2020.
- [40] Yang Li, Gang Li, Luheng He, Jingjie Zheng, Hong Li, and Zhiwei Guan. Widget Captioning: Generating Natural Language Description for Mobile User Interface Elements. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.
- [41] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration. In Conference on Machine Learning and Systems (MLSys), 2024.
- [42] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Conference on Neural Information Processing Systems (NeurIPS), 2019.
- [43] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo

- Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark-Albert Saroufim, Marcos Yukio, Helen Suk, Shunting Zhang, Michael Suo, Phil Tillet, Xu Zhao, Eikan Wang, Keren Zhou, Richard Zou, Xiaodong Wang, Ajit Mathews, William Wen, Gregory Chanan, Peng Wu, and Soumith Chintala. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2024.
- [44] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-Art Natural Language Processing. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.
- [45] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2020.
- [46] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In International Conference on Learning Representations (ICLR), 2024.
- [47] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A Diagram is Worth a Dozen Images. In European Conference on Computer Vision (ECCV), 2016.
- [48] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In Annual Meeting of the Association for Computational Linguistics (ACL), 2022.
- [49] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. DocVQA: A Dataset for VQA on Document Images. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021.
- [50] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. InfographicVQA. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022.
- [51] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. In International Conference on Learning Representations (ICLR), 2024.

- [52] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [53] xAI. Grok-1.5, 2024.
- [54] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [55] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA Models That Can Read. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- [56] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [57] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12B. arXiv:2410.07073, 2024.
- [58] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. ActivityNet-QA: A Dataset for Understanding Complex Web Videos via Question Answering. In AAAI Conference on Artificial Intelligence (AAAI), 2019.
- [59] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding. arXiv:2406.04264, 2024.
- [60] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. MVBench: A Comprehensive Multi-modal Video Understanding Benchmark. In IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR), 2024.
- [61] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-Modal LLMs in Video Analysis. arXiv:2405.21075, 2024.
- [62] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang, and Bo Zhao. Video-XL:

- Extra-Long Vision Language Model for Hour-Scale Video Understanding. arXiv:2409.14485, 2024.
- [63] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding. arXiv:2410.17434, 2024.
- [64] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx MLLM: OnDemand Spatial-Temporal Understanding at Arbitrary Resolution. arXiv:2409.12961, 2024.
- [65] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In ACM Symposium on Operating Systems Principles (SOSP), 2023.
- [66] De-An Huang, Shijia Liao, Subhashree Radhakrishnan, Hongxu Yin, Pavlo Molchanov, Zhiding Yu, and Jan Kautz. LITA: Language Instructed Temporal-Localization Assistant. arXiv preprint arXiv:2403.19046, 2024.
- [67] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the NavGraph: Vision-and-Language Navigation in Continuous Environments. In European Conference on Computer Vision (ECCV), 2020.
- [68] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Visionand-Language Navigation: Interpreting visuallygrounded navigation instructions in real environments. In IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR), 2018.
- [69] OpenAI. GPT-4V, 2023.
- [70] Google. Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context. arXiv:2403.05530, 2024.
- [71] Google. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805, 2023.
- [72] Anthropic. Claude 3, 2024.
- [73] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, Sam Dodge, Keen You, Zhen Yang, Aleksei Timofeev, Mingze Xu, Hong-You Chen, Jean-Philippe Fauconnier, Zhengfeng Lai, Haoxuan You, Zirui Wang, Afshin Dehghan, Peter Grasch, and Yinfei Yang. MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-Tuning. arXiv:2409.20566, 2024.
- [74] xAI. Grok-2, 2024.
- [75] Oleksii Kuchaiev, Jason Li, Huyen Nguyen, Oleksii Hrinchuk, Ryan Leary, Boris Ginsburg, Samuel Kriman, Stanislav Beliaev, Vitaly Lavrukhin, Jack

- Cook, Patrice Castonguay, Mariya Popova, Jocelyn Huang, and Jonathan M Cohen. NeMo: A Toolkit for Building AI Applications using Neural Modules. arXiv:1909.09577, 2019.
- [76] Yunhao Fang, Ligeng Zhu, Yao Lu, Yan Wang, Pavlo Molchanov, Jan Kautz, Jang Hyun Cho, Marco Pavone, Song Han, and Hongxu Yin. VILA2: VILA Augmented VILA. arXiv:2407.17453, 2024.
- [77] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, and Guilin Liu. Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders. arXiv:2408.15998, 2024.
- [78] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuoling Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. NVLM: Open Frontier-Class Multimodal LLMs. arXiv:2409.11402, 2024.
- [79] Meta. Llama 3, 2024.
- [80] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token Merging: Your ViT But Faster. In International Conference on Learning Representations (ICLR), 2023.
- [81] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An Image is Worth 1/2 Tokens After Layer 2: Plugand-Play Inference Acceleration for Large VisionLanguage Models. In European Conference on Computer Vision (ECCV), 2024.
- [82] Yizhe Xiong, Hui Chen, Tianxiang Hao, Zijia Lin, Jungong Han, Yuesong Zhang, Guoxin Wang, Yongjun Bao, and Guiguang Ding. PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation. In European Conference on Computer Vision (ECCV), 2024.
- [83] Joonmyung Choi, Sanghyeok Lee, Jaewon Chu, Minhyuk Choi, and Hyunwoo J Kim. vid-TLDR: Training Free Token Merging for Light-Weight Video Transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [84] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [85] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. SlowFast-LLaVA: A Strong TrainingFree Baseline for Video Large Language Models. arXiv:2407.15841, 2024.
- [86] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and

- Sifei Liu. SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models. In Conference on Neural Information Processing Systems (NeurIPS), 2024.
- [87] Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. In Conference on Neural Information Processing Systems (NeurIPS), 2023.
- [88] Qianlong Du, Chengqing Zong, and Jiajun Zhang. MoDS: Model-Oriented Data Selection for Instruction Tuning. arXiv:2311.15653, 2023.
- [89] Maxim Fishman, Brian Chmiel, Ron Banner, and Daniel Soudry. Scaling FP8 Training to TrillionToken LLMs. arXiv:2409.12517, 2024.
- [90] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. FP8 Formats for Deep Learning. arXiv:2209.05433, 2022.
- [91] Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Bhuminand Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact Language Models via Pruning and Knowledge Distillation. In Conference on Neural Information Processing Systems (NeurIPS), 2024.
- [92] Lucio Dery, Steven Kolawole, Jean-François Kagy, Virginia Smith, Graham Neubig, and Ameet Talwalkar. Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes. arXiv:2402.05406, 2024.
- [93] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. In International Conference on Learning Representations (ICLR), 2023.
- [94] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR), 2021.
- [95] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. DoRA: WeightDecomposed Low-Rank Adaptation. In International Conference on Machine Learning (ICML), 2024.
- [96] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient Finetuning of Quantized LLMs. In Conference on Neural Information Processing Systems (NeurIPS), 2024.
- [97] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian.

- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. In International Conference on Machine Learning (ICML), 2024.
- [98] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. ALLaVA: Harnessing GPT4V-Synthesized Data for Lite Vision-Language Models. arXiv:2402.11684, 2024.
- [99] Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and Better Understanding Vision-Language Models: Insights and Future Directions. arXiv:2408.12637, 2024.
- [100] Pablo Montalvo and Ross Wightman. PDF Association Dataset (PDFA), 2024.
- [101] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, Chee Seng Chan, and Lianwen Jin. ICDAR 2019 Competition on Large-Scale Street View Text with Partial Labeling. In International Conference on Document Analysis and Recognition (ICDAR), 2019.
- [102] Chee Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, Chee Seng Chan, and Lianwen Jin. ICDAR 2019 Robust Reading Challenge on Arbitrary-Shaped Text. In International Conference on Document Analysis and Recognition (ICDAR), 2019.
- [103] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. COYO-700M: Image-Text Pair Dataset, 2022.
- [104] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. ShareGPT4V: Improving Large Multi-Modal Models with Better Captions. In European Conference on Computer Vision (ECCV), 2024.
- [105] Ahmed Masry, Parsa Kavehzadeh, Xuan Long Do, Enamul Hoque, and Shafiq Joty. UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.
- [106] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal C4: An Open, BillionScale Corpus of Images Interleaved with Text. In Conference on Neural Information Processing Systems (NeurIPS), 2024.
- [107] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. MSRVTT: A Large Video Description Dataset for Bridging Video and Language. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

- [108] Jonathan Krause, Justin Johnson, Ranjay Krishna, and Li Fei-Fei. A Hierarchical Approach for Generating Descriptive Image. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [109] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [110] Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A Corpus for Reasoning about Natural Language Grounded in Photographs. In Annual Meeting of the Association for Computational Linguistics (ACL), 2019.
- [111] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. VisualMRC: Machine Reading Comprehension on Document Images. In AAAI Conference on Artificial Intelligence (AAAI), 2021.
- [112] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. TextCaps: A Dataset for Image Captioning with Reading Comprehension. In European Conference on Computer Vision (ECCV), 2020.
- [113] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: Visual Question Answering by Reading Text in Images. In International Conference on Document Analysis and Recognition (ICDAR), 2019.
- [114] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene Text Visual Question Answering. In IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
- [115] Jianfeng Kuang, Wei Hua, Dingkang Liang, Mingkun Yang, Deqiang Jiang, Bo Ren, and Xiang Bai. Visual Information Extraction in the Wild: Practical Dataset and End-to-end Solution. In International Conference on Document Analysis and Recognition (ICDAR), 2023.
- [116] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction. In International Conference on Document Analysis and Recognition (ICDAR), 2019.
- [117] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. OCR-Free Document Understanding Transformer. In European Conference on Computer Vision (ECCV), 2022.
- [118] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal ArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models. In Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

- [119] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding. arXiv:2306.17107, 2023.
- [120] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In Conference on Neural Information Processing Systems (NeurIPS), 2022.
- [121] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual Question Answering. In IEEE/CVF International Conference on Computer Vision (ICCV), 2015.
- [122] Paul Lerner, Olivier Ferret, Camille Guinaudeau, Hervé Le Borgne, Romaric Besançon, José G Moreno, and Jesús Lovón Melgarejo. ViQuAE, A Dataset for Knowledge-based Visual Question Answering about Named Entities. In International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2022.
- [123] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José MF Moura, Devi Parikh, and Dhruv Batra. Visual Dialog. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [124] Drew A Hudson and Christopher D Manning. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- [125] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-LLaVA: Solving Geometric Problem with MultiModal Large Language Model. arXiv:2312.11370, 2023.
- [126] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning. In International Conference on Learning Representations (ICLR), 2024.
- [127] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling Context in Referring Expressions. In European Conference on Computer Vision (ECCV), 2016.
- [128] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. GeoQA: A Geometric Question Answering Benchmark Towards Multimodal Numerical Reasoning. In Annual Meeting of the Association for Computational Linguistics (ACL), 2021.
- [129] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge. In IEEE/CVF Conference on

- Computer Vision and Pattern Recognition (CVPR), 2019.
- [130] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic Prompt Learning via Policy Gradient for Semi-Structured Mathematical Reasoning. In International Conference on Learning Representations (ICLR), 2023.
- [131] Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, and Liangwei Wang. On the General Value of Evidence, and Bilingual Scene-Text Visual Question Answering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
- [132] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. DVQA: Understanding Data Visualizations via Question Answering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- [133] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic. arXiv:2306.15195, 2023.
- [134] Tianyu Yu, Jinyi Hu, Yuan Yao, Haoye Zhang, Yue Zhao, Chongyi Wang, Shan Wang, Yinxv Pan, Jiao Xue, Dahai Li, Zhiyuan Liu, Hai-Tao Zheng, and Maosong Sun. Reformulating Vision-Language Foundation Models and Datasets Towards Universal Multimodal Assistants. arXiv:2310.00653, 2023.
- [135] Bo Zhao, Boya Wu, Muyang He, and Tiejun Huang. SVIT: Scaling Up Visual Instruction Tuning. arXiv:2307.04087, 2023.
- [136] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 2024.
- [137] Yangzhou Liu, Yue Cao, Zhangwei Gao, Weiyun Wang, Zhe Chen, Wenhai Wang, Hao Tian, Lewei Lu, Xizhou Zhu, Tong Lu, Yu Qiao, and Jifeng Dai. MMInstruct: A High-Quality Multi-Modal Instruction Tuning Dataset with Extensive Diversity. arXiv:2407.15838, 2024.
- [138] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned Language Models are Zero-Shot Learners. In International Conference on Learning Representations (ICLR), 2021.
- [139] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning. In International Conference on Learning Representations (ICLR), 2024.
- [140] Databricks. Dolly, 2023.

- [141] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825, 2023.
- [142] Xudong Xie, Ling Fu, Zhifei Zhang, Zhaowen Wang, and Xiang Bai. Toward Understanding WordArt: Corner-Guided Transformer for Scene Text Recognition. In European Conference on Computer Vision (ECCV), 2022.
- [143] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning. In International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2021.
- [144] Jianhao Shen, Ye Yuan, Srbuhi Mirzoyan, Ming Zhang, and Chenguang Wang. Measuring VisionLanguage STEM Skills of Neural Models. In International Conference on Learning Representations (ICLR), 2024.
- [145] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. SLAKE: A SemanticallyLabeled Knowledge-Enhanced Dataset for Medical Visual Question Answering. In IEEE International Symposium on Biomedical Imaging (ISBI), 2021.
- [146] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just Ask: Learning to Answer Questions from Millions of Narrated Videos. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021.
- [147] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards Automatic Learning of Procedures from Web Instructional Videos. In AAAI Conference on Artificial Intelligence (AAAI), 2018.
- [148] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, YuanFang Wang, and William Yang Wang. VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research. In IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
- [149] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, and Yiming Yang. Direct Preference Optimization of Video Large Multimodal Models from Language Model Reward. arXiv:2404.01258, 2024.
- [150] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video Instruction Tuning With Synthetic Data. arXiv:2410.02713, 2024.

Table A1 | Data recipe. Training data for each stage, grouped by category.

- Stage 1: Projector Initialization Feature Align LLaVA-CC3M-Pretrain [1]

- Stage 2: Visual Encoder Pre-Training Recaptioned Data ALLAVA [98]

Document Docmatix [99], PDFA [100] OCR LSVT [101], ArT [102]

- Stage 3: Token Processor Pre-Training Recaptioned Data COYO [103] (25M Subset and recaptioned by VILA2 [76]), ShareGPT4v-Pretrain [104]

Document Docmatix [99], UniChart-Pretrain [105] Interleaved Data MMC4 [106]

- Stage 4: Image Instruction-Tuning

Hybrid ShareGPT4V-SFT [104], Molmo(subset) [30], The Cauldron(subset) [22], Cambrian(subset) [21], LLaVA-OneVision(subset) [4]

Captioning MSR-VTT [107], Image Paragraph Captioning [108], ShareGPT4V-100K [104] Reasoning CLEVR [109], NLVR2 [110], VisualMRC [111] Document DocVQA [49], UniChart-SFT [105], ChartQA [48]

OCR TextCaps [112], OCRVQA [113], ST-VQA [114], POIE [115], SROIE [116], SynthDoG-en [117], TextOCR-GPT4V, ArxivQA [118], LLaVAR [119]

General VQA ScienceQA [120], VQAv2 [121], ViQuAE [122], Visual Dialog [123], GQA [124], Geo170K [125], LRV-

Instruction [126], RefCOCO [127], GeoQA [128], OK-VQA [129], TabMWP [130], EstVQA [131] Diagram & Dialogue DVQA [132], AI2D [47], Shikra [133], UniMM-Chat [134]

Instruction LRV-Instruction [126], SVIT [135], MMC-Instruction [136], MM-Instruction [137]

Text-only FLAN-1M [138], MathInstruct [139], Dolly [140], GSM8K-ScRel-SFT [141] Knowledge WordART [142], WIT [143], STEM-QA [144]

Medical PathVQA [39], SLAKE [145], MedVQA Video ActivityNet-QA [58], MSRVTT-QA [107], iVQA [146], Youcook2 [147], VaTeX [148], ShareGPTVideo [149]

- Stage 5: Video Instruction-Tuning Video LLaVA-Video-178K [150] Image LLaVA-OneVision(subset) [4]

