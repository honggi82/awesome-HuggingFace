# arXiv:2601.08303v2[cs.CV]11Feb2026

## SnapGen++: Unleashing Diffusion Transformers for Efficient High-Fidelity Image Generation on Edge Devices

Dongting Hu1,2 Aarush Gupta1 Magzhan Gabidolla1 Arpit Sahni1 Huseyin Coskun1 Yanyu Li1 Yerlan Idelbayev1 Ahsan Mahmood1 Aleksei Lebedev1 Dishani Lahiri1 Anujraaj Goyal1 Ju Hu1 Mingming Gong2,3 Sergey Tulyakov1 Anil Kag1

1 Snap Inc. 2 The University of Melbourne 3 MBZUAI Project Page: https://snap-research.github.io/snapgenplusplus/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

HiDream-I1

Model Param.

Ours-small 0.4B

Ours-full 1.6B

SANA 1.6B

SD3-M 2B

SD3.5-L 8.1B

Flux.1-dev 12B

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

17B

Handsome man, serious, white hair, fantasy, looks away, book cover

[Figure 17]

[Figure 18]

[Figure 19]

A dog that has been meditating all the time

[Figure 20]

[Figure 21]

[Figure 22]

Haunted mansion made out of pink flowers

[Figure 23]

[Figure 24]

[Figure 25]

Figure 1. Top: Our text-to-image Diffusion Transformer (0.4B parameters) generates diverse, high-fidelity 1K images in just 1.8s on a mobile device. All examples are produced by this on-device model at a resolution of approximately 10242. Bottom: Comparison across various text-to-image models. Both our on-device (small) and server-side (full) versions achieve competitive visual quality.

###### Abstract

Recent advances in diffusion transformers (DiTs) have set new standards in image generation, yet remain impractical for on-device deployment due to their high computational and memory costs. In this work, we present an efficient DiT framework tailored for mobile and edge devices that achieves transformer-level generation quality under strict resource constraints. Our design combines three key components. First, we propose a compact DiT architecture with an adaptive global–local sparse attention mechanism that balances global context modeling and local detail preservation. Second, we propose an elastic training framework that jointly optimizes sub-DiTs of varying capacities within a unified supernetwork, allowing a single model to dynamically adjust for efficient inference across different hardware. Finally, we develop K-DMD (Knowledge-Guided Distribution Matching Distillation), a step-distillation pipeline that integrates the DMD objective with knowledge transfer from few-step teacher models, producing high-fidelity and lowlatency generation (e.g., 4-step) suitable for real-time ondevice use. Together, these contributions enable scalable, efficient, and high-quality diffusion models for deployment on diverse hardware.

###### 1. Introduction

Image generation models [18, 36, 54, 56, 70] have made remarkable progress, enabling a wide range of creative applications. Recent advances [11, 53] show a clear shift toward diffusion transformer (DiT) architectures, with large-scale models such as Flux [36] and Qwen-Image [70] achieving state-of-the-art image quality, editing flexibility, and personalization. However, these transformer-based models are extremely large—often containing tens of billions of parameters—requiring server-grade GPUs and custom CUDA kernels [40] for inference, which introduces high computational cost and dependence on cloud infrastructure. To improve accessibility, recent works [28, 41, 85] have explored deploying compact diffusion models directly on mobile devices. Systems such as SnapFusion [41], Mobile Diffusion [85], and SnapGen [28] demonstrate efficient on-device text-to-image (T2I) generation using lightweight U-Net backbones that achieve favorable quality–efficiency trade-offs.

While these on-device models alleviate latency and cloud dependence, their U-Net-based architectures lag far behind recent DiT models in scalability and generative performance. To bridge this architectural gap, we propose an Efficient Diffusion Transformer tailored for mobile and edge deployment, achieving server-level generation quality under strict resource constraints. To address the quadratic complexity of attention especially at

high resolutions (e.g., 1K), we introduce a three-stage DiT with an adaptive global–local sparse attention mechanism that effectively combines coarse-grained Key–Value (KV) Compression for global context modeling with fine-grained Blockwise Neighborhood Attention for spatial relation modeling. By dynamically allocating attention based on content, the model achieves flexible computation and high representational fidelity, outperforming U-Net–based systems such as SnapGen [28] in generation quality while maintaining comparable inference speed.

Deploying such models on real-world devices presents another key challenge: the heterogeneity of deployment hardware. On-device generation must meet stringent compute, memory, and power constraints, while devices vary widely—from entry-level smartphones to high-end flagships and lightweight edge servers. A single static model cannot perform efficiently across this spectrum, leading to fragmented development and suboptimal deployment. To address this, we introduce an Elastic Training Framework that jointly optimizes sub-DiTs of varying capacities within a unified DiT supernetwork. This elastic framework enables a single model to encompass multiple sub-networks, each tailored to different hardware. At inference, the appropriate sub-network is selected dynamically, enabling seamless adaptation across heterogeneous devices without retraining. This design ensures scalability, efficiency, and consistent generation quality across diverse deployment scenarios.

To close the performance gap between large-scale and compact diffusion models, we employ knowledge distillation to transfer the generative capability of the full-step teacher to the student. We further propose KnowledgeGuided Distribution Matching Distillation (K-DMD), a step-distillation framework that integrates the DMD objective [78, 79] with knowledge transfer from a few-step (i.e., 4-step) teacher, enabling efficient distillation while preserving high fidelity and supporting on-device generation.

###### Our main contributions are as follows:

- 1. Efficient DiT-based architecture. We design a compact yet expressive diffusion transformer optimized for on-device generation, achieving strong performance under strict computational and memory constraints.
- 2. Elastic training framework. We propose an elastic training paradigm to jointly optimize sub-DiTs of varying capacities within a unified supernetwork, enabling adaptive inference across heterogeneous hardware with stable convergence and robust generalization.
- 3. Knowledge-guided distillation pipeline. We introduce K-DMD, a step-distillation framework that integrates the DMD objective with knowledge transfer from fewstep teacher models, achieving high-fidelity image synthesis with substantially reduced inference latency and supporting efficient on-device generation.

###### 2. Related Work

T2I Diffusion Models. Diffusion models [24, 36, 60, 70] have become the state of the art in text-to-image (T2I) generation, surpassing earlier GAN-based approaches [6, 20] in fidelity and diversity. Early latent diffusion models [10, 34, 37–39, 41, 54, 56, 57] employed U-Net backbones for iterative denoising in latent space, balancing image quality and memory efficiency. Recent advances replace U-Nets with Diffusion Transformers (DiTs) [36, 53, 67, 70], achieving improved scalability, quality, and generalization across generation and editing tasks [33, 46, 50, 67, 70]. However, their billion-scale parameters and high computational cost make them impractical for on-device deployment.

Efficient Diffusion Transformers. Recent efforts [10, 13, 40, 47, 52, 76] aim to improve DiT efficiency. PixArtΣ [10] introduces key–value compression for 4K image generation, while SANA [76] employs linear self-attention to enable efficient synthesis on consumer GPUs. LinFusion [47] replaces quadratic attention in Stable Diffusion [56] with Mamba-based [14, 21] attention for ultrahigh-resolution (16K) generation. Hybrid designs such as Simple Diffusion [25, 26], HourGlass-DiT [13], and UDiTs [65] combine convolutional and transformer blocks in U-Net–style hierarchies. U-ViT [4] introduces long-skip connections for faster convergence, and Playgroundv3 [45] reduces key/value dimensions to mimic single-level U-Nets. Despite these advances, DiTs still depend on quadratic attention and large memory footprints, limiting efficient highresolution (e.g., 1024×1024) generation on mobile devices. On-Device Generative Models. To enable on-device deployment, prior works have explored quantization [40, 62], pruning [28, 41, 72, 73], and knowledge distillation [28, 35] to reduce model size and latency. Early on-device systems [9, 41, 85] pruned and distilled U-Net architectures to generate 512-pixel images within seconds. SnapGen [28] demonstrated 1024-pixel image generation with a compact U-Net, though with trade-offs in quality and editing flexibility. To our knowledge, no prior work has deployed an efficient DiT for high-fidelity on-device generation.

Model Scalability and Elastic Networks. Once-for-All [7] and Slimmable Networks [81] pioneered supernetworks adaptable to varying computational budgets for recognition and detection tasks. Follow-up studies [17, 27, 66, 69] extended this idea to transformers and large language models. However, elastic architectures remain underexplored in generative models. We build our model in this direction by introducing an Elastic DiT framework that enables flexible diffusion transformer deployment across heterogeneous devices without retraining separate models.

Sparse Attention. Yuan et al. [83] and Hassani et al. [22] propose hardware-efficient sparse attention designs using block- and neighborhood-based formulations optimized for GPUs. For video generation, Zhang et al. [84] and Xi et al.

[74] exploit local and spatiotemporal sparsity for efficient attention, while Xia et al. [75] introduce adaptive sparse attention with online sparsity discovery without retraining.

Step Distillation. Step distillation accelerates diffusion inference by compressing multi-step sampling into a few denoising iterations [3, 42, 58, 61, 77, 78]. Progressive Distillation [49, 58] first showed that student models can learn from intermediate teacher trajectories, while Consistency and Phase Consistency Models [61, 68] enhance stability by enforcing cross-step prediction consistency. Adversarial Diffusion Distillation (ADD) [42] introduces GANstyle objectives for few-step, high-fidelity synthesis, and Distribution Matching Distillation (DMD) [78, 79] aligns teacher–student noise distributions for improved perceptual quality. Recent works such as UFOGen [77], SANASprint [12], and SD3.5-Flash [3] combine distillation with architectural optimizations for near real-time generation.

###### 3. Method

We introduce an efficient Diffusion Transformer (DiT) architecture, an elastic training framework, and a multi-stage distillation pipeline. Together, these components enable efficient high-fidelity image generation on edge devices.

###### 3.1. Efficient Three-Stage DiT Architecture

We develop the efficient DiT through a series of key architectural design ablations. All variants are trained on the ImageNet-1K dataset [16] for conditional image generation at 256 × 256 resolution and evaluated using the validation loss (Val Loss) following the protocol of [67]. This metric shows stronger correlation with perceptual quality and human preference than conventional image metrics such as FID [23], aligned with the findings in [18]. Model efficiency is measured by parameter count and inference latency on iPhone 16 Pro Max. For consistency, all models employ the Flux VAE [36] and the CLIP-L [55] text encoder, and are trained for 200K iterations using the flowmatching [44, 48]. As a reference, we implement the SnapGen [28] (Fig. 3, rightmost column) as our baseline, which achieves a latency of 274ms and a Val Loss of 0.5131.

(A) Baseline Architecture. Our design builds on the PixArt-α [11] DiT backbone, chosen for its strong balance between parameter efficiency and computational cost. To adapt it for edge deployment, we incorporate multi-query attention (MQA) [59] and reduce the feed-forward expansion ratio to 3, yielding a compact 424M-parameter DiT. This baseline attains a validation loss of 0.506 with an inference latency of 2000ms (Fig. 3, first column).

Computation Analysis. The main computational bottleneck arises from self-attention (SA) at high resolutions. For a 10242 image, the VAE encoder yields a 1282 latent map. After patchification, this corresponds to 642 tokens (4096 in total), substantially increasing SA cost and often causing

xt

h

Patchify

LN

Scale, Shift

DiT w/ ASSA

Linear

GQA Linear

GQA Linear

Down Blocks

Linear, Sigmoid

×6

#### q k v

Time

###### SA ASSA

Downsample

LayerNorm RoPE

LayerNorm RoPE

LayerNorm

MLP

Scale

| |
|---|
|DiT w/ SA ×20|
| |

⨁

Middle Blocks CA

×8

⨁

Global

Text Embed

KV Compression

LN

Upsample

Local

##### q kc vc q v k

Scale, Shift

Grouped Query Attn

Blockwise Neighborhood Attn

FFN

DiT w/ ASSA

Up Blocks

×10

Scale

hglobal hlocal

⨁

Per-Head Interpolate

Unpatchify

DiT Block

h→

vˆt

(a) Three-Stage Transformer

(b) Adaptive Sparse Self-Attention (ASSA)

Figure 2. Efficient DiT Overview. Left: Our model consists of three stages: Down, Middle and Up. Down and Up blocks operate on high-resolution latent while using our novel Adaptive Sparse Self-Attention (ASSA) layers. Middle blocks operate at latents downsampled by 2 × 2 window and use standard Self-Attention (SA) layers. Other layers in the blocks are Cross-Attention (CA) for modulating with input text conditioning and Feed-Forward (FFN) layer. Right: We delve deeper into our ASSA layer. It consists of two parallel attention processing branches: (i) coarse-grained key-value compression for overall structure, and (ii) fine-grained blockwise neighborhood attention features. Finally, the layers to weight these two features are adaptively per head through an input-dependent weighting function.

(truncated) Param.(M) Latency(ms) ValLoss(×10 4) Figure 3. Efficient DiT Ablations. We plot the performance (validation loss) and model footprint (parameters & latency on iPhone 16 Pro Max) for various stages in our ablations. Using a baseline DiT yields extremely high latency. Our multi-stage design with ASSA layers and additional enhancements results in an Efficient DiT with comparable latency and better performance than the state-of-the-art on-device model SnapGen [28].

800

| |424 406 408 429<br><br>2,000<br><br>550<br><br>293<br><br>360<br><br>5060<br><br>5130 5130<br><br>5090| | | | |372<br><br>274<br><br>5131| |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

700

Param.(M)/Latency(ms)

600

500

400

300

200

100

0

SnapGen (U-Net)

(A) Baseline Architecture

(B) Three-Stage Transformer

(C) ASSA at High-Resolution

(D) Additional Enhancements

(C) Adaptive Sparse Self-Attention (ASSA) at HighResolution Stages. Although token downsampling in the middle stage reduces the overall computational cost, the bottleneck remains in the SA operations of the down and up stages. To alleviate this, we introduce an adaptive sparse self-attention (ASSA) (Fig. 2 (b)) that replaces full SA over 4096 tokens with two complementary components:

out-of-memory (OOM) errors on edge hardware. To address this, we introduce several architectural modifications that improve efficiency while preserving generation fidelity.

(B) Three-Stage Diffusion Transformer. Inspired by recent efficient architectures such as Hourglass-DiT [13] and U-DiT [65], we extend the baseline into a three-stage design (Fig. 2 (a), left). The three stages are denoted as Down, Middle, and Up. A single downsample layer is applied after the down stage and an upsample layer before the up stage, producing a compact latent representation of 1024 tokens (32 × 32) in the middle stage. Half of the transformer layers are assigned to the middle, while the remaining layers are divided between the down and up—with slightly more layers in the up blocks, following SiD2 [26]. This design cuts latency from 2000ms to 550ms, while increasing the validation loss to 0.513 (Fig. 3, second column).

(i) Global Attention. We apply Key-Value (KV) compression by performing a 2×2 convolution with stride 2 on the k and v feature maps. Given the key and value tensors k,v ∈ RH×W×d, we compute the compressed tensors

kc = Conv2×2,s=2(k), vc = Conv2×2,s=2(v), (1)

resulting in kc,vc ∈ RH2 ×W2 ×d. This reduces the key/value token length by a factor of four, enabling each query to attend to a compressed global context with substantially lower memory and computational overhead.

(ii) Local Attention. To preserve fine-grained spatial details, we introduce Blockwise Neighborhood Attention (BNA), which restricts attention computation to a local region around each token. As shown in Fig. 4(a), naive local attention restricts each token to attending only to its spatial neighbors within a fixed window (e.g., 3 × 3), analogous to a convolutional receptive field. When visualized in the attention matrix, this local interaction pattern forms a banddiagonal structure, as shown in Fig. 4(b). While such localized attention is more efficient than full self-attention, it is not natively supported on mobile hardware and still incurs nontrivial overhead when applied per token. To further optimize for edge deployment, we adopt a blockwise formulation (Fig. 4(c)), where the token grid is divided into B (a hyperparameter) non-overlapping spatial blocks, and attention is computed independently within each block. Formally, we partition the query, key, and value matrices q,k,v ∈ R(HW)×d along the sequence dimension into B non-overlapping blocks:

q = [q1;...;qB],k = [k1;...;kB],v = [v1;...;vB], (2) where each block qb,kb,vb ∈ RN

b×d and block size Nb = HW/B. For each query block qb, attention is computed only within a limited neighborhood of key–value blocks Nr(b) = {b − r,...,b,...,b + r }, where r denotes the block neighborhood radius (bandwidth). The blockwise neighborhood attention is defined as

r(b)]⊤

qb[kN

√

[vN

r(b)], b = 1,...,B,

Ab = Softmax

d

(3) where [kN

r(b)] represent the concatenation of key and value blocks within the neighborhood Nr(b). This formulation enforces spatial locality, produces a blocksparse attention pattern that scales efficiently as O(N2/B), and preserves strong local contextual modeling for highresolution features. It is worth noting that different hyperparameter combinations of the block number B and neighborhood radius r can be used, effectively controlling the token-level spatial neighborhood size (see the supplementary material for a detailed illustration).

r(b)] and [vN

The final attention score is a linear interpolation between glocal attention and local attention, conditional on the input hidden states. This nove sparse attention design substantially reduces the overall attention overhead while preserving generation quality. As shown in Fig. 3 (third column), our sparse attention model achieves a latency of 293ms without loss of generation quality (val loss of 0.513).

(D) Additional Enhancements. To further improve performance, we introduce several enhancements:

• Dense long-range skip connections: Following [5], we add dense skip connections in the middle stage to increase the capacity of the bottleneck representation.

Key/Value Tokens

Key/Value Tokens

| | | | | |[Figure 26]| | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

[Figure 27]

[Figure 28]

QueryTokens

QueryTokens

(a) Neighborhood Attention with 3 neighbors

(b) Self Attention Mask with 3 neighbors

(c) Blockwise Neighborhood Attention with 8 blocks

Figure 4. Illustration of Blockwise Neighborhood Attention. (a) Naive Neighborhood Attention, where each query attends to its local window of 3 neighboring tokens. (b) Corresponding selfattention mask showing the limited receptive field for each query. (c) Blockwise Neighborhood Attention extends this concept by grouping tokens into 8 local blocks, enabling efficient attention computation while preserving locality.

- • Grouped Query Attention (GQA): We employ GQA [2] by increasing the number of key/value heads to eight, improving multi-head diversity and reducing query–key bottlenecks with minimal additional parameter overhead.
- • Expanded FFN capacity: The FFN expansion ratio increases to four in down and up stages, yielding higher representation power without excessive computational cost.
- • Layer redistribution: Four transformer layers are reassigned from the middle stage—two each to the down and up—to achieve a more balanced depth and better information hierarchy. Thanks to the efficiency of the proposed sparse self-attention, we can afford a slight increase in computational load to gain capacity and performance.

As shown in Fig. 3 (fourth column), this configuration achieves a latency of 360ms and a validation loss of 0.509, offering a strong trade-off between efficiency and accuracy. With all components combined, our efficient DiT architecture attains conv-level latency while surpassing it in both visual quality and scalability in image generation, outperforming SnapGen by a large margin in validation loss. Some qualitative results are in the supplementary material.

###### 3.2. Elastic DiT Framework

Recent works such as Matformer [17] and Gemma-3n [63] demonstrate the importance of building unified yet adaptable LLM architectures that can be deployed efficiently across heterogeneous platforms (e.g., high-end smartphones, low-power devices, and server-side environments). Motivated by this, we design an Elastic DiT framework that enables a single diffusion transformer to flexibly scale its capacity according to available computational resources.

Framework Design. To enable this flexibility, we identify a structural decomposition that allows parameter sharing across subnetworks of different widths [81], slicing the projection matrices in the attention and FFN layers along the hidden dimension to sample subnetworks of varying sizes from a single supernetwork. In cross-attention layers, the

Ldi!

seamlessly across heterogeneous platforms while maintaining strong performance and visual fidelity. As shown in Tab. 1, the elastic training recipe achieves comparable validation loss and DINO-FID to standalone training while reducing the overall model-state footprint through parameter sharing. Note that these results are obtained from relatively small-scale experiments on ImageNet, where the overhead from data loading and embedding computation is limited. When scaling to large-scale text-to-image (T2I) training and distillation, this overhead becomes significantly more pronounced, as the data pipeline and larger teacher components dominate the total training cost.

[Figure 29]

Super Network

|½ hidden dim| |
|---|---|
| | |

###### v!(xt,t)

[Figure 30]

Ldist

→v!

Same weights

Sub Network

[Figure 31]

½ hidden dim v!

s→!(xt,t)

xt ω

Ldi!

x0 ω → x0

- Figure 5. Elastic Training Framework. Given a supernetwork, we define sub-networks as different granularities of the hidden dimension. During training, we sample sub-networks uniformly and supervise them using the output from the supernetwork. In addition, we use standard diffusion loss on all granularities. This leads to more stable training and imparts knowledge to sub-networks.

Training Recipe

DINO FID

Training Footprint

Model Val Loss

- 0.4B 0.5090 128 6.6 GB
- 1.6B 0.5073 109 18.8 GB

Standalone

- 0.4B 0.5093 125 –
- 1.6B 0.5071 110 18.8 GB

Elastic

Table 1. Comparison between Standalone and Elastic training for 0.4B and 2B models. Elastic training reuses parameters between model scales, reducing memory allocation while maintaining similar validation loss and DINO-FID.

key and value projections are not sliced, as they are independent of the model width (hidden dimension). Parameters strictly tied to the hidden-state length—such as those in layer normalization and modulation layers—are isolated, since they are lightweight and dimension-specific. This design produces three model variants: a tiny 0.3B model

###### 3.3. Distillation Pipelines

- (0.375× width) for low-end Android devices, a small 0.4B model (0.5× width) for high-end smartphones, and a full 1.6B supernetwork (1× width) that can be quantized for ondevice deployment or server-side inference. Training Recipe. Naively optimizing multiple subnetworks with shared weights often leads to unstable gradient updates, even under low learning rates. To mitigate this issue, we propose a unified elastic training strategy that stabilizes joint optimization across subnetworks of different widths (Fig. 5). During training, subnetworks parameter-

We apply both the flow matching loss (Eq. (4)) and the distillation loss (Eq. (5)) during the pretraining stage. Following the SnapGen [28] pipeline, we then perform large-scale knowledge distillation to substantially enhance the performance of small student models, followed by step distillation enabling efficient inference and real-time generation on edge devices.

Knowledge Distillation. A large cloud-scale teacher [70] (denoted as ξ) supervises the training of the elastic DiT models through both output- and feature-level distillation. The student θ ∈ {Θ, Θs} is first encouraged to match the teacher’s velocity predictions:

ized by Θs ⊆ Θ are sampled jointly with the full supernetwork Θ in each iteration and optimized under a unified flow-matching objective:

Lξout(θ) = vξ(xt,t) − vθ(xt,t) 22, (6) and further aligns its internal representations via feature

Ldiff(θ) = Eϵ∼N(0,I), t ∥(ϵ − x0) − vθ(xt,t)∥22 , (4)

where θ ∈ {Θ, Θs}. Their gradients are then aggregated using adaptive scaling to ensure balanced updates across subnetworks. Additionally, a lightweight distillation loss is applied between each subnetwork and the full-capacity (supernetwork) model to further improve training stability and ensure consistent convergence behavior:

distillation on the final transformer layer:

Lξfeat(θ,ϕ) = fξ(xt,t) − ϕ fθ(xt,t) 22, (7)

where ϕ is the projector. The overall distillation objective combines both levels of supervision with timestep-aware scaling [28]:

(xt,t) − ∇vΘ(xt,t) 22, (5) where ∇ denotes the stop-gradient operator. This elas-

Ldist(Θs) = vΘ

s

LKD(θ,ϕ) = S Ldiff, Lξout + Lξfeat, (8) where S(·) the timestep-aware scaling operator.

tic training framework enables DiT models to be deployed

###### 4. Experiments

[Figure 32]

Few-Step Teacher

[Figure 33]

fω→ vω→(xt,t)

###### 4.1. Experimental Setup

Lfeat Lout

T2I Configuration. We use the 1.6B parameter efficient DiT (Sec. 3.1) as the supernetwork for our elastic training (Sec. 3.2) which embeds two sub-networks of 0.3B and 0.4B parameters. We employ TinyCLIP [71] and Gemma34b-it [64] as text encoders with token-wise concatenation for rich semantic embeddings. Following [18, 28], we drop these independently to enable inference even in the absence of other encoder. Since we use Qwen-Image [70] as our teacher, we use their VAE to align the latent space. We also train a tiny decoder similar to [28] for on-device generation. On-Device Runtime. The VAE decoder takes 120 ms, and the per-step latency of the DiT (0.4B) is 360 ms, yielding a nominal runtime of about 1.6 s for a 4-step generation. Including additional system overhead, the total on-device runtime is around 1.7 s. Further implementation details are provided in the supplementary material.

[Figure 34]

ω

Student

xt fω vω(xt,t)

[Figure 35]

Critic

###### xˆ0

fc

x0

→ωLDMD

[Figure 36]

Teacher

fω

[Figure 37]

: Few-Step LoRA

- Figure 6. Knowledge-guided Distribution Matching Distillation (K-DMD). Our step distillation method combines distribution matching with knowledge transfer from a few-step teacher.

Step Distillation. Following recent one-step distillation methods [78, 79], we adopt Distribution Matching Distillation (DMD) for step distillation. However, DMD requires careful tuning of hyperparameters such as teacher guidance scale and auxiliary loss weight. We observe that optimal settings vary across model capacities, and applying DMD to smaller models with only millions of parameters often causes unstable convergence.

Training Recipe. Inspired by recent works [70], we use multi-aspect ratio data to pre-train the elastic model using flow-matching loss [18, 48] at 256 resolution, followed by 1024 base resolution. In the next stage, we use knowledge distillation from Qwen-Image [70] and K-DMD stepdistillation training with Qwen-Image-Lightening [51]. We provide additional details in supplementary.

To address these issues, we propose Knowledge-guided DMD (K-DMD), which extends DMD-based step distillation by incorporating knowledge distillation from a fewstep teacher [51] (Fig. 6). Following [79], we compute the KL divergence between the real score from the teacher ξ and the student output distribution estimated by a critic model c (initialized with the same weights as the student θ):

###### 4.2. Evaluations

Quantitative Results. We evaluate our T2I model against standard baselines on DPG-Bench [30], GenEval [19], and T2I-CompBench [31] to assess key T2I generation attributes. Following [28], we also report CLIP-Score [55] on a subset of MS-COCO [43]. Results for the tiny (0.3B), small (0.4B), and full (1.6B) variants of our elastic model are shown in Tab. 2, with main findings summarized below.

dxˆ0 dθ

∇θLξDMD(θ) = fc F(ˆx0,τ),τ − fξ F(ˆx0,τ),τ

, with xˆ0 = xt − σt vθ(xt,t),

(9) where τ is randomly sampled to diffuse (via F) the input xˆ0 before passing it to the teacher ξ and critic c.

- • Our models achieve competitive or superior performance across all major benchmarks—including DPG, GenEval, T2I-CompBench, and CLIP—compared to much larger models such as Flux.1-dev [36] and SD3.5-Large [1].
- • The small variant (0.4B) surpasses models up to 20× larger while retaining on-device efficiency comparable to SnapGen, and the tiny variant (0.3B) achieves the highest throughput among all evaluated models.
- • The elastic design enables a smooth trade-off between visual quality and computational cost, achieving a strong balance of fidelity, scalability, and on-device efficiency.

To further leverage the power of the large-scale fewstep teacher [51] (denoted as ξ′), we feed the same input xt as the student and incorporate Lξ

′

′

out (Eq. (6)) and Lξ

feat

(Eq. (7)) into the training objective. The final step distillation objective is defined as:

′

′

LK-DMD(θ,ϕ) = LξDMD + Lξ

out + Lξ

feat. (10) This objective enables stable convergence across models of varying capacities without requiring additional hyperparameter tuning. Furthermore, the few-step teacher can be activated by enabling the few-step LoRA [29], introducing no extra memory overhead, as illustrated in Fig. 6. The critic model c is updated alternatively with flow-matching (Eq. (4)) on student’s distribution xˆ0 aligned with previous works [3, 32, 80].

Qualitative Results. To visually assess image–text alignment and overall aesthetics, we compare images generated by different T2I models in Fig. 1. We observe that many existing models tend to produce overly stylized or less realistic images, and often fail to capture the full prompt and omit important visual elements.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- Figure 7. Human Evaluation. We conduct a user study comparing our small (0.4B) and full (1.6B) variants with three baselines—SANA

- (1.6B), SD3-Medium (2B), and Flux.1-dev (12B)—across three key attributes: realism, visual fidelity, and text–image alignment.

Table 2. Quantitative Evaluation. Scores are reported on DPGBench, GenEval, T2I-CompBench, and CLIP (COCO). Throughput/FPS (samples/s) is measured on a single 80GB A100 GPU using the largest batch size that fits for 10242 images. Latency (ms) is measured on iPhone 16 Pro Max with one forward pass.

Model Arch. Param. FPS ↑ Latency↓ DPG ↑ GenEval ↑ T2I-C.B. ↑ CLIP ↑

SnapGen [28] U-Net 0.4B 0.51 274 81.1 0.66 – 0.332 PixArt-α [11] DiT 0.6B 0.42 † 71.1 0.48 0.351 0.316 PixArt-Σ [10] DiT 0.6B 0.46 † 80.5 0.53 0.427 0.317 SANA [76] Hybrid 1.6B 0.91 † 84.8 0.66 0.476 0.327 LUMINA-Next [86] DiT 2.0B 0.06 † 74.6 0.46 0.353 0.309 SD3-Medium [18] DiT 2.0B 0.28 † 84.1 0.68 0.522 0.323 SDXL [54] U-Net 2.6B 0.18 † 74.7 0.55 0.402 0.301 Playgroundv2.5 [39] DiT 2.6B 0.18 † 75.5 0.56 0.237 0.319 IF-XL [15] U-Net 5.5B 0.06 † 75.6 0.61 0.421 0.311 SD3.5-Large [1] DiT 8.1B 0.08 † 85.6 0.71 0.507 0.326 Flux.1-dev [36] DiT 12B 0.04 † 83.8 0.66 0.471 0.316

Ours-tiny DiT 0.3B 0.81 280 84.6 0.69 0.502 0.330 Ours-small DiT 0.4B 0.62 360 85.2 0.70 0.506 0.332 Ours-full DiT 1.6B 0.28 1580 87.2 0.76 0.536 0.338

Note. “†” indicates out-of-memory (OOM) at 1024 × 1024 resolution.

Human Preference Study. For a thorough comparison between baselines, we conduct a user study following the widely used Parti prompts [82]. We include SANA (1.6B), SD3-M (2B), and Flux.1-dev (12B) as the baselines and ask participants to select images with better attributes between the baselines and our models. The evaluation considers three key aspects: realism, fidelity, and text alignment. As shown in Fig. 7, our full variant surpasses all baselines in both fidelity and realism, while remaining highly competitive in image–text alignment, particularly against SD3-M. The small variant also demonstrates robust performance, outperforming larger baselines such as Flux.1-dev and SANA on most attributes.

Few-Step Generation. After applying Knowledge-guided Distribution Matching Distillation (K-DMD), our models are capable of generating high-quality images in only four steps. As shown in Fig. 8, we compare the performance of the 28-step base models with the 4-step distilled models using DPG and GenEval scores. The results indicate that the distilled 4-step models achieve performance comparable to the 28-step baselines, despite the significant reduction in sampling steps. While there is a slight drop in scores, the

quality remains nearly lossless, demonstrating the effectiveness of our step-distillation approach.

Ours-tiny (0.3B) Ours-small (0.4B) Ours-full (1.6B)

[Figure 38]

[Figure 39]

[Figure 40]

28 Steps w/o K-DMD

[Figure 41]

[Figure 42]

[Figure 43]

- 4 Steps w/

K-DMD

84.6 / 0.69 85.2 / 0.70 87.2 / 0.76

81.6 / 0.66 82.7 / 0.69 85.6 / 0.76

Figure 8. Few-step Generation. Comparison of images produced by the tiny (0.3B), small (0.4B), and full (1.6B) models under 28step (w/o K-DMD) and 4-step (w/ K-DMD) settings. Numbers in the corners denote DPG / GenEval scores.

- 5. Conclusion

In this work, we presented an Efficient Diffusion Transformer that brings transformer-based image generation to mobile and edge devices. Through adaptive global–local sparse attention, our model achieves strong quality–efficiency trade-offs under strict resource limits. An Elastic Training Framework enables dynamic scalability across heterogeneous hardware, while K-DMD distills high-fidelity knowledge from few-step teachers for fast, high-quality generation. Extensive experiments demonstrate that our models achieve near server-level generation quality while operating efficiently on mobile devices. Together, these advances make diffusion transformers practical for real-world on-device deployment, paving the way for scalable generative intelligence on edge devices.

###### References

- [1] Stability AI. Stable diffusion 3.5. https://github.com/Stability-AI/sd3.5, 2024. 7, 8, 15, 16
- [2] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, Singapore, 2023. Association for Computational Linguistics. 5
- [3] Hmrishav Bandyopadhyay, Rahim Entezari, Jim Scott, Reshinth Adithyan, Yi-Zhe Song, and Varun Jampani. Sd3.5flash: Distribution-guided distillation of generative flows,

2025. 3, 7

- [4] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 3
- [5] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 5
- [6] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In International Conference on Learning Representations (ICLR), 2019. 3
- [7] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. In International Conference on Learning Representations (ICLR), 2020. 3
- [8] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, Yimeng Wang, Kai Yu, Wenxuan Chen, Ziwei Feng, Zijian Gong, Jianzhuang Pan, Yi Peng, Rui Tian, Siyu Wang, Bo Zhao, Ting Yao, and Tao Mei. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025. 15, 16
- [9] Thibault Castells, Hyoung-Kyu Song, Tairen Piao, Shinkook Choi, Bo-Kyeong Kim, Hanyoung Yim, Changgwun Lee, Jae Gon Kim, and Tae-Ho Kim. EdgeFusion: On-Device Text-to-Image Generation. arXiv preprint arXiv:2404.11925, 2024. 3
- [10] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024. 3, 8, 15, 16
- [11] Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 2, 3, 8, 15, 16
- [12] Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. Sana-sprint: One-step diffusion with continuous-time consistency distillation, 2025. 3

- [13] Katherine Crowson, Stefan Andreas Baumann, Alex Birch, Tanishq Mathew Abraham, Daniel Z Kaplan, and Enrico Shippole. Scalable high-resolution pixel-space image synthesis with hourglass diffusion transformers. In Proceedings of the 41st International Conference on Machine Learning, pages 9550–9575. PMLR, 2024. 3, 4
- [14] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML), 2024. 3
- [15] DeepFloyd. Deepfloyd. https://github.com/deep-floyd/IF,

2023. 8, 15, 16

- [16] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 3
- [17] Khatri Devvrit, Sneha Kudugunta, Aditya Kusupati, Tim Dettmers, Kaifeng Chen, Inderjit S Dhillon, Yulia Tsvetkov, Hannaneh Hajishirzi, Sham M. Kakade, Ali Farhadi, and Prateek Jain. Matformer: Nested transformer for elastic inference. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3, 5
- [18] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 2, 3, 7, 8, 15, 16
- [19] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36, 2024. 7
- [20] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems (NeurIPS), 2014. 3
- [21] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 3
- [22] Ali Hassani, Steven Walton, Humphrey Shi, et al. Generalized neighborhood attention: Multi-dimensional sparse attention at the speed of light. arXiv preprint arXiv:2504.16922, 2025. 3
- [23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 3
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 2020. 3
- [25] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In Proceedings of the 40th International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 3

- [26] Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion: 1.5 fid on imagenet512 with pixel-space diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18062–18071, 2025. 3, 4
- [27] Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, and Qun Liu. Dynabert: Dynamic bert with adaptive width and depth. In Advances in Neural Information Processing Systems, pages 9782–9793. Curran Associates, Inc., 2020. 3
- [28] Dongting Hu, Jierun Chen, Xijie Huang, Huseyin Coskun, Arpit Sahni, Aarush Gupta, Anujraaj Goyal, Dishani Lahiri, Rajesh Singh, Yerlan Idelbayev, Junli Cao, Yanyu Li, Kwang-Ting Cheng, S.-H. Chan, Mingming Gong, Sergey Tulyakov, Anil Kag, Yanwu Xu, and Jian Ren. Snapgen: Taming high-resolution text-to-image models for mobile devices with efficient architectures and training. arXiv:2412.09619 [cs.CV], 2024. 2, 3, 4, 6, 7, 8, 13, 14, 15, 16, 17
- [29] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 7
- [30] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 7
- [31] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-toImage Generation . IEEE Transactions on Pattern Analysis Machine Intelligence, (01):1–17, 5555. 7
- [32] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 7
- [33] Xin Jin, Yichuan Zhong, and Yapeng Tian. TP-blend: Textual-prompt attention pairing for precise object-style blending in diffusion models. Transactions on Machine Learning Research, 2025. 3
- [34] Anil Kag, Huseyin Coskun, Jierun Chen, Junli Cao, Willi Menapace, Aliaksandr Siarohin, Sergey Tulyakov, and Jian Ren. Ascan: Asymmetric convolution-attention networks for efficient recognition and generation. arXiv preprint arXiv:2411.04967, 2024. 3
- [35] Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. Bk-sdm: Architecturally Compressed Stable Diffusion for Efficient Text-to-Image Generation. In Workshop on Efficient Systems for Foundation Models@ ICML2023, 2023. 3
- [36] Black Forest Labs. Flux: A generative model by black forest labs. https://github.com/black-forestlabs/flux, 2024. Accessed: 2025-05-14. 2, 3, 7, 8, 15, 16
- [37] Daiqing Li, Aleks Kamko, Ali Sabet, Ehsan Akhgari, Linmiao Xu, and Suhail Doshi. Playground v1, . 3
- [38] Daiqing Li, Aleks Kamko, Ali Sabet, Ehsan Akhgari, Linmiao Xu, and Suhail Doshi. Playground v2, .

- [39] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground V2. 5: Three Insights towards Enhancing Aesthetic Quality in Text-toImage Generation. arXiv preprint arXiv:2402.17245, 2024. 3, 8, 15, 16
- [40] Muyang Li*, Yujun Lin*, Zhekai Zhang*, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3
- [41] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 13
- [42] Shanchuan Lin, Anran Wang, and Xiao Yang. SDXLLightning: Progressive Adversarial Diffusion Distillation,

2024. 3

- [43] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 7
- [44] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [45] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-toimage alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 3
- [46] Renjing Liu, Jiatao Li, William Peebles, and Saining Xie. Magicedit: High-fidelity and temporally coherent video editing. arXiv preprint arXiv:2303.08354, 2023. 3
- [47] Songhua Liu, Weihao Yu, Zhenxiong Tan, and Xinchao Wang. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097, 2024. 3
- [48] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 3, 7
- [49] Chenlin Meng, Ruiqi Gao, Diederik P Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In NeurIPS 2022 Workshop on Score-Based Methods, 2022. 3
- [50] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. 3
- [51] ModelTC. Qwen-image-lightning: Distilled qwen-image models for fast, high-fidelity text-to-image generation. https://github.com/ModelTC/Qwen-ImageLightning, 2025. Version V1.x/ V2.x available; Apache2.0 license. 7, 15
- [52] Dogyun Park, Moayed Haji-Ali, Yanyu Li, Willi Menapace, Sergey Tulyakov, Hyunwoo J. Kim, Aliaksandr Siarohin,

- and Anil Kag. Sprint: Sparse-dense residual fusion for efficient diffusion transformers, 2025. 3
- [53] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2023. 2, 3
- [54] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3, 8, 15, 16
- [55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 7
- [56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3
- [57] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, pages 36479–36494. Curran Associates, Inc., 2022. 3
- [58] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. 3
- [59] Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019. 3
- [60] Yang Song, Jascha Sohl-Dickstein, Durk P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations (ICLR), 2021. 3
- [61] Yang Song, Chenlin Meng, and Stefano Ermon. Consistency models. International Conference on Machine Learning (ICML), 2023. 3
- [62] Yang Sui, Yanyu Li, Anil Kag, Yerlan Idelbayev, Junli Cao, Ju Hu, Dhritiman Sagar, Bo Yuan, Sergey Tulyakov, and Jian Ren. Bitsfusion: 1.99 bits weight quantization of diffusion model. In Advances in Neural Information Processing Systems, pages 76775–76818. Curran Associates, Inc., 2024. 3
- [63] Gemma Team. Gemma 3n. 2025. 5
- [64] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´e, Morgane Rivi`ere, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Ga¨el Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr,

- Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, Andr´as Gy¨orgy, Andr´e Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Pluci´nska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim P˜oder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and L´eonard Hussenot. Gemma 3 technical report, 2025. 7
- [65] Yuchuan Tian, Zhijun Tu, Hanting Chen, Jie Hu, Chao Xu, and Yunhe Wang. U-dits: Downsample tokens in u-shaped diffusion transformers, 2024. 3, 4
- [66] Mojtaba Valipour, Mehdi Rezagholizadeh, Hossein Rajabzadeh, Parsa Kavehzadeh, Marzieh Tahaei, Boxing Chen, and Ali Ghodsi. Sortednet: A scalable and generalized framework for training modular deep neural networks, 2024.

- 3
- [67] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3
- [68] Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency model. arXiv preprint arXiv:2405.18407, 2024. 3
- [69] Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, and Song Han. Hat: Hardware-aware transformers for efficient natural language processing. In Annual Conference of the Association for Computational Linguistics, 2020. 3
- [70] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 2, 3, 6, 7, 15, 16

- [71] Kan Wu, Houwen Peng, Zhenghong Zhou, Bin Xiao, Mengchen Liu, Lu Yuan, Hong Xuan, Michael Valenzuela, Xi (Stephen) Chen, Xinggang Wang, Hongyang Chao, and Han Hu. Tinyclip: Clip distillation via affinity mimicking and weight inheritance. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 21970–21980, 2023. 7
- [72] Yushu Wu, Yanyu Li, Anil Kag, Ivan Skorokhodov, Willi Menapace, Ke Ma, Arpit Sahni, Ju Hu, Aliaksandr Siarohin, Dhritiman Sagar, Yanzhi Wang, and Sergey Tulyakov. Taming diffusion transformer for efficient mobile video generation in seconds, 2025. 3
- [73] Yushu Wu, Zhixing Zhang, Yanyu Li, Yanwu Xu, Anil Kag, Yang Sui, Huseyin Coskun, Ke Ma, Aleksei Lebedev, Ju Hu, Dimitris N. Metaxas, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapgen-v: Generating a five-second video within five seconds on a mobile device. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 2479–2490, 2025. 3
- [74] Ruijie Xi, Qingxiong Zhang, Hongyu Gao, et al. Sparse videogen: Accelerating video diffusion transformers with

spatial-temporal sparsity. arXiv preprint arXiv:2502.01776,

2025. 3

- [75] Yifei Xia, Suhan Ling, Fangcheng Fu, Yujie Wang, Huixia Li, Xuefeng Xiao, and Bin Cui. Training-free and adaptive sparse attention for efficient long video generation. arXiv preprint arXiv:2502.21079, 2025. 3
- [76] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han. SANA: Efficient high-resolution text-to-image synthesis with linear diffusion transformers. In The Thirteenth International Conference on Learning Representations, 2025. 3, 8, 15, 16
- [77] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8196–8206, 2024. 3
- [78] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024. 2, 3, 7
- [79] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fr´edo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024. 2, 3, 7
- [80] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 7
- [81] Jiahui Yu, Linjie Huang, Shixiang Wang, Aviv Efrat, Jaehoon Cho, Jonathan Brandt, Tong Gao, Wei Chen, and Thomas Han. Slimmable neural networks. In International Conference on Learning Representations (ICLR), 2019. 3, 5
- [82] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022. Featured Certification. 8
- [83] Yifan Yuan, Jiayi Zhang, Peng Sun, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025. 3
- [84] Zeyu Zhang, Weihao Xu, Yujie Wang, et al. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025. 3
- [85] Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-image generation on mobile devices. arXiv preprint arXiv:2311.16567, 2023. 2, 3
- [86] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, Xu Luo, Zehan Wang, Kaipeng Zhang, Lirui Zhao, Si Liu, Xiangyu Yue, Wanli Ouyang, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-next : Making lumina-t2x stronger and faster with next-dit. In Advances in Neural Information Processing Systems, pages 131278–

131315. Curran Associates, Inc., 2024. 8, 15, 16

## SnapGen++: Unleashing Diffusion Transformers for Efficient High-Fidelity Image Generation on Edge Devices

### Supplementary Material

###### A. Discussion of On-Device Latency

We report the per-step latency and total generation time in Tab. 1. Note that the VAE decoder requires approximately 120ms [28], and additional components such as latent scaling, scheduler stepping, and CLIP embedding introduce negligible latency, similar to observations in [28, 41]. Thanks to our proposed Adaptive Sparse Self-Attention, the quantized full model can still run on mobile devices without encountering out-of-memory issues.

Table 1. Latency and Generation Time of Our Models

Per-step Latency

4-step Generation

Model Parameters

Ours-tiny 0.3B 280ms 1.2s Ours-small 0.4B 360ms 1.8s Ours-full* 1.6B 1580ms 6.7s

* Model is 4-bit quantized.

###### B. Demo on Mobile Device

We include an on-device demonstration on the project page, showcasing our small model (0.4B). It achieves a generation time of 1.8s per image and produces high-quality outputs at 1024×1024 resolution on an iPhone 16 Pro Max. The application is implemented using the open-source Swift Core ML Diffusers framework. Upon launching the app, users can input textual prompts and generate corresponding images by simply tapping the “Generate” button.

Two screenshots of on-device generation on an iPhone 16 Pro Max are shown in Fig. 1, featuring results from both our small and full variant with 4-bit quantization.

###### C. On-device Deployment Details

To enable mobile-friendly deployment, we optimize the model to minimize computational overhead by reducing operations such as transpose and reshape. We structure the model in a convolutional fashion, where the channel dimension is placed as the third-to-last dimension (i.e., (B,C,H,W)), rather than following the conventional transformer layout of (B,L,D). We reimplement the attention mechanism using split einsum operations to improve on-device efficiency. For Blockwise Neighborhood Attention (BNA), computations for each block are executed in parallel through a for-loop, enabling efficient execution

[Figure 44]

[Figure 45]

Ours-full (1.6B) w/ 4-bit quantization

Ours-small (0.4B)

Figure 1. On-device Image Generation Demo. Screenshots from our on-device application running on an iPhone 16 Pro Max. The left panel shows results from the small (0.4B) model, and the right panel shows results from the full variant with 4-bit quantization.

on mobile hardware. Finally, the model is exported via CoreML to generate a computation graph for deployment.

To deploy the full model (1.6B) on deivce, we quantize all linear and convolutional layer weights using k-means clustering over their values. Most layers are quantized to 4 bits (16 clusters), while more sensitive layers are assigned 8 bits. Sensitivity is determined with a simple heuristic: for each layer, we measure the mean-squared error (MSE) between the layer’s quantized output and the corresponding output from the unquantized model, when quantizing that layer in isolation. Layers with the largest degradation in MSE are designated as sensitive and quantized at 8 bits, resulting in an overall average quantization of 4.3 bits. After quantization, we freeze the weights and fine-tune the remaining parameters, such as biases and normalization layers, using self-distillation for several thousand iterations.

[Figure 46]

[Figure 47]

[Figure 48]

(a) Self Attention Mask with 3 spatial neighbors

(b) Blockwise Neighborhood Attention with (b=8, r=1)

(c) Blockwise Neighborhood Attention with (b=16, r=3)

[Figure 49]

[Figure 50]

[Figure 51]

(d) Self Attention Mask with 5 spatial neighbors

(e) Blockwise Neighborhood Attention with (b=8, r=2)

(f) Blockwise Neighborhood Attention with (b=16, r=5)

- Figure 2. Illustration of Blockwise Neighborhood Attention (BNA). Visualization of BNA under different hyperparameter settings of block number (b) and neighborhood radius (r), showing the corresponding spatial neighbor coverage and attention sparsity.

###### D. Additional Illustration of Blockwise Neighborhood Attention

In Fig. 2, we illustrate BNA under different configurations. Specifically, configurations (b) and (c) in BNA produce spatial neighbor coverage similar to the standard self-attention mask with three spatial neighbors in (a), while configurations (e) and (f) correspond closely to the five-neighbor case in (d). By adjusting the block number b and neighborhood radius r, one can flexibly control the sparsity of BNA to balance computational efficiency and representational fidelity. In our experiments setting we set b to 16 and r to 1, essentially yields 9 spatial neighbor tokens at 10242 resolution.

###### E. Detailed Results on T2I Benchmarks

We present detailed results for DPG-Bench in Tab. 2, GenEval in Tab. 3 and T2I-CompBench in Tab. 4.

###### F. Qualitative Comparison on ImageNet

We present some visual results of ImageNet-1K between our 0.4B small model (Validation Loss = 0.5090) and SnapGen U-Net [28] (0.4B, Validation Loss = 0.5131) in Fig. 3.

###### G. Additional Qualitative Comparison on T2I

To further demonstrate the visual fidelity and prompt adherence of our model, we provide additional qualitative comparisons on text-to-image (T2I) generation tasks. Our models are evaluated across diverse prompts spanning objects, scenes, and artistic compositions, highlighting their ability to produce high-quality, semantically accurate, and visually consistent outputs. As shown in Fig. 4 and Fig. 5, our approach delivers competitive visual quality and superior alignment with textual descriptions, outperforming baseline methods with significantly larger parameter counts.

###### Table 2. Detailed Results of DPG-Bench Comparisons.

Model Param Global Entity Attribute Relation Other Overall ↑

SnapGen [28] 0.4B 88.3 85.1 87.0 87.3 87.6 81.1 PixArt-α [11] 0.6B 75.0 79.3 78.6 82.6 77.0 71.1 PixArt-Σ [10] 0.6B 86.9 82.9 88.9 86.6 87.7 80.5 SANA [76] 1.6B 86.0 91.5 88.9 91.9 90.7 84.8 LUMINA-Next [86] 2.0B 82.8 88.7 86.4 80.5 81.8 74.6 SD3-Medium [18] 2.0B 83.5 89.6 86.7 93.2 92.5 85.1 SDXL [54] 2.6B 83.3 82.4 80.9 86.8 80.4 74.7 Playgroundv2.5[39] 2.6B 83.1 82.6 81.2 84.1 83.5 75.5 IF-XL [15] 5.5B 77.7 81.2 83.3 81.8 82.9 75.6 SD3.5-Large [1] 8.1B 87.4 92.1 90.0 88.2 88.1 85.6 Flux.1-dev [36] 12B 74.4 90.0 89.9 90.9 88.3 83.8 HiDream-I1-Full [8] 17B 76.4 90.2 89.5 93.7 91.8 85.9 Qwen-Image [70] 20B 91.3 91.6 92.0 94.3 92.7 88.3

Ours-tiny 0.3B 88.5 90.2 88.8 92.6 78.8 84.6 Ours-small 0.4B 84.2 90.9 89.0 93.1 79.6 85.2 Ours-full 1.6B 85.7 91.5 89.6 94.5 80.4 87.2

###### H. Training Implementation Details

We adopt FSDP2 for distributed training across 32 nodes, each equipped with 8 A100 GPUs (80GB). The model is initially trained at a resolution of 2562 with a global batch size of 8192 using the Adam optimizer and a learning rate of 1 × 10−4 for 400K iterations under elastic training. Subsequently, the resolution is increased to 10242 with a global batch size of 2048 and gradient checkpointing enabled. This stage incorporates knowledge distillation (KD) and continues under elastic training for an additional 100K iterations.

For the step-distillation stage (K-DMD), we set the time shift to 3, following the few-step teacher configuration in [51]. The teacher in the DMD objective employs cfg = 4, consistent with the default setting of Qwen-Image [70]. We apply LoRA to both the student network and the critic, using a rank of 64 and α = 128. The student is updated every 5 iterations. Training is conducted for 10K iterations across 4 nodes (global batch size 512) using the Adam optimizer with a learning rate of 1 × 10−4 and β = (0, 0.99).

###### Table 3. Detailed Results of GenEval Bench Comparisons.

Single Object

Two Objects

Color Attribution

Counting Colors Position

Model Param.

Overall ↑

SnapGen [28] 0.4B 1.00 0.84 0.60 0.88 0.18 0.45 0.66 PixArt-α [11] 0.6B 0.98 0.50 0.44 0.80 0.08 0.07 0.48 PixArt-Σ [10] 0.6B 0.99 0.65 0.46 0.82 0.12 0.12 0.53 SANA [76] 1.6B 0.99 0.77 0.62 0.88 0.21 0.47 0.66 LUMINA-Next [86] 2.0B 0.92 0.46 0.48 0.70 0.09 0.13 0.46 SD3-Medium [18] 2.0B 0.98 0.74 0.63 0.67 0.34 0.36 0.62 SDXL [54] 2.6B 0.98 0.74 0.39 0.85 0.15 0.23 0.55 Playgroundv2.5 [39] 2.6B 0.98 0.77 0.52 0.84 0.11 0.17 0.56 IF-XL [15] 5.5B 0.97 0.74 0.66 0.81 0.13 0.35 0.61 SD3.5-Large [1] 8.1B 0.98 0.89 0.73 0.83 0.34 0.47 0.71 FLUX.1-dev [36] 12B 0.98 0.81 0.74 0.79 0.22 0.45 0.66 HiDream-I1-Full [8] 17B 1.00 0.98 0.79 0.91 0.60 0.72 0.83 Qwen-Image [70] 20B 0.99 0.92 0.89 0.88 0.76 0.77 0.87

Ours-tiny 0.3B 1.00 0.91 0.62 0.85 0.26 0.56 0.69 Ours-small 0.4B 1.00 0.91 0.64 0.89 0.22 0.55 0.70 Ours-full 1.6B 1.00 0.97 0.66 0.90 0.32 0.70 0.76

###### Table 4. Detailed Results of T2I CompBench Comparisons.

Model Param. Color Complex Nonspatial Shape Spatial Texture Overall↑

PixArt-α [11] 0.6B 0.416 0.334 0.308 0.389 0.197 0.461 0.351 PixArt-Σ [10] 0.6B 0.585 0.380 0.309 0.479 0.244 0.566 0.427 SANA [76] 1.6B 0.660 0.377 0.312 0.529 0.322 0.652 0.476 LUMINA-Next [86] 2.0B 0.511 0.350 0.303 0.333 0.185 0.438 0.353 SD3-Medium [18] 2.0B 0.794 0.384 0.315 0.582 0.324 0.731 0.522 SDXL [54] 2.6B 0.570 0.331 0.311 0.481 0.199 0.520 0.402 Playgroundv2.5 [39] 2.6B 0.644 0.364 0.308 0.486 0.217 0.607 0.437 IF-XL [15] 5.5B 0.591 0.354 0.311 0.512 0.182 0.577 0.421 SD3.5-Large [1] 8.1B 0.768 0.382 0.316 0.591 0.275 0.712 0.507 FLUX.1-dev [36] 12B 0.764 0.374 0.307 0.501 0.253 0.627 0.471 HiDream-I1-Full [8] 17B 0.749 0.401 0.314 0.592 0.399 0.696 0.525 Qwen-Image [70] 20B 0.836 0.399 0.317 0.605 0.443 0.743 0.557

Ours-tiny 0.3B 0.765 0.372 0.316 0.545 0.331 0.680 0.502 Ours-small 0.4B 0.770 0.370 0.316 0.551 0.350 0.679 0.506 Ours-full 1.6B 0.794 0.375 0.316 0.600 0.419 0.712 0.536

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

###### Figure 3. Qualitative comparison on ImageNet-1K. Visual comparison between on-device models SnapGen [28] (0.4B, left in each pair, validation loss = 0.5131) and our small model (0.4B, right in each pair, validation loss = 0.5090). Our model produces sharper textures, more consistent colors, and improved structural fidelity across diverse categories.

a woman … realism with fantasy elements, … , emphasizes emotion over realism, …

[Figure 60]

Epic long distance cityscape photo of New York City …in rainforest, at sunset, …, golden light

[Figure 61]

a bloody mary cocktail next to a napkin

[Figure 62]

…, of a two women, skin to skin touch face, …, skin texture, Rembrandt light, soft shadows

[Figure 63]

A smiling banana wearing a bandana

[Figure 64]

realist portrait …, sensual bengali girl, full body green water algae …, very detailed faces

[Figure 65]

A dolphin is smoking a cigarette on a beach

[Figure 66]

beautiful lady, …, dark makeup, wearing a floral blue vest top, soft light, dark grey background

[Figure 67]

Photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, …

[Figure 68]

California santa monica, sunsetsparkling, ..., ultra photoreal, full color, …, high contrast

[Figure 69]

…, lit by the soft glow of a neon sign in a dimly lit, …, hinting at a narrative of longing and nostalgia. …

[Figure 70]

outdoor white bed with a view

[Figure 71]

female 1950s detective in futuristic city, fedora, realistic colour …

[Figure 72]

hyperealistic pirate skull

[Figure 73]

