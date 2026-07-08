# arXiv:2605.31336v1[cs.CV]29May2026

## DecMem: Towards Minute-Long Consistent World Generation with Decoupled Memory

Zhenhao Yang1∗ , Xiaoshi Wu2, Zhengyao Lv1, Xiaoyu Shi2† Xintao Wang2, Pengfei Wan2, Kun Gai2, Kwan-Yee K. Wong1† 1The University of Hong Kong, 2Kling Team, Kuaishou Technology

### Abstract

Recent advances in video generative models have promoted rapid progress in controllable world models. However, maintaining fine-grained spatio-temporal consistency under long-horizon reasoning remains a key challenge. In this work, we move beyond explicit 3D memory and coarse frame-level implicit modeling, and propose a fine-grained, learnable, and scalable memory for consistent world generation. We first identify two fundamental limitations of naïve learnable memory architectures in long-horizon extrapolation, namely computational inefficiency and attention dispersion. Through a systematic analysis of attention dispersion, we propose DecMem, a decoupled memory architecture that employs Sparse Global Memory for efficient fine-grained access to global history and Anchored Local Memory for stable and high-quality extrapolation. Extensive experiments demonstrate that DecMem significantly outperforms current state-ofthe-art methods. By ensuring precise and efficient long-term memory and achieving superior extrapolation capabilities, DecMem enables minute-level controllable long video generation with high fidelity and consistency. Project page is available at https://jeffreyyzh.github.io/DecMem-Page

### 1 Introduction

With the rapid evolution of generative video modeling, leveraging powerful pretrained video generation backbones [39, 19] to construct world models has become a pivotal research frontier. While recent works have successfully achieved controllable generation through injecting action information [21, 35, 11, 51, 38, 27, 54, 43, 47], generating high-quality and consistent long videos remains a formidable challenge. This issue is particularly pronounced in “revisit” scenarios, where existing models frequently fail to recall previously generated scenes as inference extends, leading to significant temporal inconsistencies. Fundamentally, building a temporally consistent world model demands flexible and efficient exploitation of long-term memory, rather than being confined to local context mechanisms such as sliding windows [12, 37, 2, 16, 3, 49] or their extensions that incorporate attention sinks [45, 24, 6, 32].

Existing memory mechanisms can be broadly classified into two categories, namely explicit memory and implicit memory. Explicit Memory approaches rely on explicitly constructed 3D representations [41, 15, 23, 55, 8, 40, 36]. While geometric priors naturally favor spatial consistency, their performance is bounded by the underlying 3D estimator. Maintaining 3D representations incurs additional overheads, and estimation errors accumulated over time can erode long-range consistency. Early implicit memory approaches [44, 50, 34] leverage camera poses and field-of-view (FOV) to retrieve relevant frames from a memory bank, thereby expanding their effective context window (see Fig. 1(c)). Attention-based implicit memory approaches [4, 42, 5], on the other hand, model

*Work done during an internship at Kling Team, Kuaishou Tech. †Corresponding Author.

Preprint.

###### (a)

Ground Truth Ours

Dense Attn

Dense Attn+Decay

(b)

[Figure 1]

| |
|---|

| |
|---|

| |
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Frame688Frame810

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

| |
|---|

| |
|---|

(c)

Prediction Result

Ground Truth

Prediction Result

|[Figure 10]|
|---|

|[Figure 11]|
|---|

[Figure 12]

Learnable Block Retrieval

FOV-based Frame Retrieval

###### Ours

WorldMem

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

[Figure 17]

| |
|---|

| |
|---|
| |

| | |
|---|---|

[Figure 18]

[Figure 19]

[Figure 20]

Fine-grained Match Learnable Retrieval

[Figure 21]

Retrieval Context

Blocks as Context Frames as Context

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

| |
|---|

| |
|---|

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

| |
|---|
| |

| |
|---|

[Figure 30]

Noise Injection Heuristic Search

[Figure 31]

- Figure 1: (a) Visual quality and spatio-temporal consistency of different long-horizon extrapolation methods (memory bank initialized with 221 frames). Prior methods fail to jointly preserve fidelity and consistency, while ours breaks this trade-off and sustains fine-grained memory under long rollouts.

###### (b) Generation latency of our method and naïve Dense Attention (memory bank initialized with 221 frames). Our sparse block retrieval substantially reduces cost without sacrificing quality. (c) Comparison of our learnable block retrieval against FOV-based frame retrieval (e.g., WorldMem).

inter-frame dependencies implicitly in the attention mechanism. While all these implicit memory approaches advance toward learnable memory, they remain bounded by a frame-level granularity bottleneck. FOV-based approaches rely on heuristic policies that cannot be jointly optimized with the generative objective, whereas attention-based approaches, though end-to-end learnable, treat each frame as an indivisible unit and fail to capture sub-frame spatio-temporal correspondences.

To overcome the granularity bottleneck of implicit memory while avoiding the fragility of explicit 3D representations, a straightforward design is to let every token perform dense attention over all historical features, thereby achieving the finest-grained and fully learnable long-term memory. However, this simple design suffers from two fundamental limitations, namely attention dispersion and computational inefficiency. As the context grows, we observe a flood of weakly-relevant historical features which dilutes the attention weights allocated to the critical ones. Such an attention dispersion causes severe quality degradation and structural collapse (Fig. 1(a)). The per-latent generation latency scales linearly with the sequence length, and the overall generation cost grows rapidly. Such a computational inefficiency severely constrains the scalability towards minute-long video synthesis (Fig. 1(b)). While some training-free extrapolation methods [57] alleviate the attention dispersion problem by mechanically down-weighting distant tokens, they do so at the cost of longrange memory loss (Fig. 1(a)), exposing a fundamental dilemma between short-range fidelity and long-range consistency.

To address the aforementioned limitations, we propose a fine-grained, learnable, and scalable decoupled memory architecture, named DecMem, consisting of two complementary modules. The first module is the Sparse Global Memory (SGM), which performs block-level sparse retrieval over the full history to achieve efficient yet fine-grained long-term memory access. The second module is the Anchored Local Memory (ALM), which anchors attention in recent frames to stabilize the attention distribution. By decoupling global retrieval from local anchoring, DecMem resolves the attention dispersion problem and enables scaling to minute-long video synthesis with strong spatio-temporal consistency.

Our main contributions can be summarized as follows:

- • We systematically reveal the root cause of the limited long-horizon extrapolation capability of naïve dense-attention designs and pinpoint the intrinsic limitations of training-free strategies in preserving long-range memory. We then propose a fine-grained, learnable, and scalable memory mechanism for long-video world model.

- • We introduce a novel decoupled memory architecture named DecMem, with Sparse Global Memory for global, efficient, and fine-grained memory access, and Anchored Local Memory for explicit mitigation of attention dispersion under long-horizon inference.
- • Our method consistently surpasses current state-of-the-art baselines, achieving minute-long controllable video generation with strong spatio-temporal consistency and visual quality.

### 2 Related Works

Interactive World Model. Driven by the remarkable success of diffusion methods [31, 29] in highfidelity video generation [39, 19], leveraging these generative priors to construct controllable world models has emerged as a pivotal research direction. Early explorations, such as GameNGen [38] and Matrix [9], primarily utilized discrete keyboard information as control signals, while subsequent works [1, 51, 54, 11, 21] incorporated mouse trajectories to enable precise view-dependent interactions. More recently, Hunyuan-GameCraft2 [35] and Yume-1.5 [27] have integrated prompt-based instructions to trigger new events. However, maintaining robust long-term consistency in world simulation remains a key challenge.

Memory Retrieval. To achieve spatio-temporal consistency in long-video generation, one line of work [41, 15, 23, 55, 8, 40, 36] explicitly constructs geometric representations to establish spatial correspondences between a target frame and the historic frames stored in a memory bank. While such explicit memory mechanisms enable precise spatial association, their performance is bounded by the accuracy of the underlying 3D estimator, with estimation errors accumulate as generation extends. To circumvent the fragility of explicit 3D representations, an alternative line of work resorts to implicit memory, for instance, by leveraging camera poses and field-of-view (FOV) [44, 34, 50] to retrieve relevant frames. Despite their efficacy, these explicit retrieval mechanisms often ignore the potential of learning-based optimization.

Existing learnable approaches [4, 42, 5] primarily model the relation between memory features and the current frame being generated with an attention mechanism. They perform frame-level retrieval based on attention similarity. Their memory representations remain at frame granularity and are insufficient for achieving fine-grained spatio-temporal consistency. Hong et al. [14] introduce a learnable retrieval mechanism, but it fails to scale video generation to minute-level.

Long Video Extrapolation. Constrained by the context length seen during training, long-video generation inevitably faces extrapolation beyond the training horizon at inference. Existing works fall into three main categories. First, some full sequence diffusion approaches apply training-free strategy [30, 26, 17] to decompose long-video synthesis into overlapping clips generation. This addresses inter-clip smoothness but fails to model long-range dependencies across clips. Second, recent autoregressive methods [2, 16, 3, 37, 48, 24, 20, 45, 6, 49] adopt sliding-window inference to limit the computational cost. However, this bounded window attention still discards substantial finegrained history. Third, another line of work directly extends the context of pretrained full-sequence diffusion models [39, 19] to generate a full sequence in a single pass. For instance, RIFLEx [56] adjusts the frequency parameters of RoPE to alleviate content repetition. UltraViCo [57] introduces weight decay to improve visual quality. However, they do not consider long-term memory when inference length scales. This leads to a pronounced degradation of global spatio-temporal consistency under long-horizon extrapolation. In contrast, our proposed method focuses on efficient information extraction from the long historical context, while mitigating attention dispersion for quality retention.

### 3 Method

In this section, we first present the preliminaries of autoregressive video generation, followed by action-conditioned world modeling (Section 3.1). Section 3.2 analyzes the attention dispersion phenomenon that emerges as world models extrapolate over long horizons. To address this limitation and efficiency problem, we introduce a novel decoupled memory architecture, named DecMem, consisting of a Sparse Global Memory (SGM) for efficient long-context modeling and an Anchored Local Memory (ALM) for stable attention distribution. Section 3.3 and Section 3.4 provide the details of SGM and ALM respectively. Finally in Section 3.5, we present a multimodal position embedding for encoding camera pose with spatio-temporal information.

#### 3.1 Preliminaries

Autoregressive Video Generation. Modern video generative frameworks typically operate in a compressed latent space. A pretrained Variational Autoencoder (VAE) encodes the raw video sequence into a latent representation z1:0 T ∈ RC×T×H×W. The objective of an autoregressive video generative model is to predict the subsequent latent zT0 +1 conditioned on the denoised history z1:0 T. During the training phase, following Rectified Flow [25], we sample noise ϵT+1 ∼ N(0,I) and construct the noisy latent zTt +1 via linear interpolation between the clean latent and the noise. We apply teacher forcing [18] paradigm and provide clean history z1:0 T during training. The model vθ is optimized to predict the flow velocity vT+1 = ϵT+1 − zT0 +1 by minimizing the following objective:

L = vθ z1:0 T,zTt +1,t − vT+1 22 (1)

Action-Conditioned World Modeling. To transform a video generator into a world model, we incorporate action condition as control signals. Following the spirit of Hunyuan-Gamecraft [21], the action embedding a is mapped with a light-weight fusion module ψ(·) and added with video latents:

x = Patchify(z) ⊕ ψ(a) (2)

where ⊕ denotes the element-wise addition. The feature x is then sent to Transformer blocks for further fusion. This ensures deep multimodal fusion between visual features and action controls, while keeping negligible computational overhead.

#### 3.2 Attention Dispersion in Long World Simulation

[Figure 32]

- Figure 2: Attention maps of different long world modeling approaches during long-horizon video inference.

[Figure 33]

Figure 3: Attention distribution in the generation of the 810th frame (sampled every 80 frames).

In this section, we first analyze the attention mechanism and identify the root cause of failure in naïve long-video inference. This analysis naturally motivates our new architectural design. As shown in Fig. 1(a), the naïve dense attention architecture exhibits pronounced quality degradation in long extrapolation. Conversely, training-free decay strategies [57] suppress out-of-window attention to mitigate short-range distortion, but doing so at the expense of long-term spatio-temporal consistency.

To analyze the attention mechanism, we visualize the attention maps throughout a long-horizon inference in Fig. 2. As the extrapolation length grows, it can be observed that the query’s attention is progressively diluted by a pool of historical tokens. A vast number of historical features acquire small but non-zero weights. This phenomenon becomes particularly pronounced in the generation of the 810th frame (see Fig. 3). The resulting long-tail distribution inevitably lowers the effect weights allocated to those semantically critical historical frames (see Section B for more details).

Mechanically down-weighting distant tokens [57] indiscriminately suppresses all out-of-window attention (Fig. 2) to emphasize the within-window attention, eliminating genuine long-range dependencies, and thereby cutting off the model’s access to distant critical information. This dilemma surfaces a key insight: what is required is not a more carefully engineered attention prior, but a learnable architecture that adaptively suppresses redundancy to preserve attention concentration, while explicitly extracting and exploiting history features that helps long-term memory retention.

Building on this analysis, we propose a novel decoupled memory architecture for efficient long-range memory and resistance to attention dispersion. To this end, we design a Sparse Global Memory (SGM) module (Section 3.3) for efficient and fine-grained memory access and an Anchored Local

|VAE<br><br>Decoder<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>|top-!<br><br>Compress<br><br>SGM<br><br>|Sparse Retrieval (top-"=2) #&!,# #&!,$<br><br>| | |
|---|---|
| | |
<br><br>'(&)&|
|---|
<br><br>|Context-aware Attention<br><br>#!,#<br><br>!!,#&%!,# !!,$&%!,$<br><br>#!,$| |
|---|---|
| | |
|
|---|
<br><br>Gated Output<br><br>Compress<br><br>|#! ALM (win=2)<br><br>!!&%!|
|---|
<br><br>DecMem<br><br>Frame t<br><br>QKV Projection<br><br>*!%&'<br><br>…<br><br>*!()'<br><br>Token Feature<br><br>Pooled Block Feature<br><br>Retrieved Block Feature|
|---|

Decoder

VAE

Compress

Figure 4: DecMem pipeline comprises decoupled memory for long-term consistency and extrapolation generalization while keeping the computational cost low. Sparse Global Memory (SGM) combines a block-level sparse retrieval module and a context-aware attention module for long-term memory fine-grained retrieval in an end-to-end manner, whereas Anchored Local Memory (ALM) keeps short-term transition smooth. For clearer visualization, we display 3 frame latents as key & value and the last frame indexed by t as query. Each frame contains 2 blocks with 2 tokens per block.

Memory (ALM) module (Section 3.4) to relieve the attention dispersion problem. The outputs of these two modules are fused through a learnable gating mechanism, adaptively preserving short-range fidelity and long-range memory.

#### 3.3 Sparse Global Memory

To overcome the scaling bottleneck of dense attention in long-video synthesis, we introduce the Sparse Global Memory (SGM) module. By executing retrieval at a fine-grained block level, SGM enables precise recall of long-term dependencies without the heavy computational overhead of global modeling.

Specifically, SGM carries out a two-stage process, namely block-level sparse retrieval and contextaware attention (see Fig. 4). In block-level sparse retrieval, we first split the latent frame into M non-overlapping blocks and aggregate features within each block by pooling. These pooled features are then used to identify the most relevant historical blocks to represent the fine-grained memory context. Let q¯t,i denote the pooled feature of the i-th block qt,i in the current frame t. We evaluate the relevance between this block and the historical blocks by computing their attention scores using the pooled features. Blocks with the top-k scores are chosen to represent the fine-grained memory context Ct,i for qt,i.

After block-level sparse retrieval, we next perform context-aware attention using the retrieved blocks in Ct,i. This helps to reduce attention computation from the full sequence to only the top-k most relevant blocks, preventing the per-step computation from growing linearly. Specifically, we perform a dense attention computation for each token in the query block qt,i with every token in the retrieved blocks in Ct,i. Once this block-level attention computation is completed for every query block, we assemble the block outputs into a frame output osgmt for the current frame t.

Through SGM’s sparse block-level computations, we substantially reduce the attention cost while achieving fine-grained retrieval over long-range global history.

#### 3.4 Anchored Local Memory

To counteract quality degradation caused by attention dispersion during extended inference, we introduce Anchored Local Memory (ALM) as a complementary branch to stabilize the attention distribution. Given that temporally adjacent frames inherently exhibit the strongest visual and semantic correlation with the current frame, ALM strictly confines its attention to a local window of

the most recent frames, thereby providing a high-confidence attention anchor to mitigate temporal drift and reinforce the model’s long-range extrapolation capability.

Specifically, we implement ALM with a sliding window attention mechanism, with the context window constrained to the immediate past w frames (see Fig. 4). This formulation explicitly models the interaction between the current frame and its immediate history, stabilizing the attention distribution during variable-length extrapolation.

Finally, we adaptively fuse the outputs of the two branches through a learnable gating mechanism, with the ALM output oalmt serving as a stable baseline that prevents the fused attention from being diluted by long-tail distractors from distant frames, and the SGM output osgmt endowing the model with fine-grained memory and retrieval capability over previously visited scenes:

ot = oalmt + Gt ⊙ osgmt (3) where Gt is the learnable gate derived from the current frame features. Modulated by the gating, the two branches jointly yield an adaptive trade-off between global consistency and extrapolation robustness, sustaining spatio-temporal consistency in minute-long video generation without collapse.

#### 3.5 Multimodal Position Embedding

To inject geometric and spatio-temporal priors into the attention computation, we extend video RoPE [39, 19, 46] by incorporating camera geometry embeddings. To avoid modality interference in the feature subspace, we partition the channels and apply position embeddings separately to each group. We follow PRoPE [22] to inject the camera geometry P for encoding the relative geometric relationship. For a token at position (ti,xi,yi) (i.e., the i-th token, located in frame ti at patch coordinates (xi,yi)), the full transformation matrix R(fulli) can be written as:

Rcam(Pt

) 0 0 0 Rsp(xi,yi) 0 0 0 Rtem(ti)

i

R(fulli) = diag(Rcam,Rsp,Rtem) =

(4)

where Rcam, Rsp, and Rtem denote the transformation matrices derived from camera parameters, patch coordinates, and frame index respectively. Then the position-encoded query and key can be computed as:

qi = (R(fulli) )⊤ Projq(hi), ki = (R(fulli) )−1 Projk(hi). (5) where hi denotes i-th token of input feature in each spatio-temporal attention layer, Projq(.) and Projk(.) are the respective projection transformations of the input features. More details can be found in Section A.3. This multimodal position embedding jointly encodes camera geometry, patch location and frame index, bringing precise geometric and spatio-temporal perception.

### 4 Experiment

Implementation details. Our pipeline is implemented on a 1B pretrained video generation model in chunk-wise auto-regressive manner, with each chunk containing 4 latents for faster generation. The latents within a chunk can attend to each other, and we keep the causality between chunks. We train our model on 64 NVIDIA H200 GPUs with a global batch size of 64. For long-term memory retrieval in our SGM module, we divide each frame into 6 blocks of the same size with padding. We set k in top-k historical blocks retrieval to 80 unless otherwise specified. Our ALM module employs a context window of 8 frame latents. We train DecMem on the WorldMem [44] datasets and apply FID [13], PSNR, and LPIPS [53] to evaluate the distribution-level, pixel-level, and perceptual-level similarity between generated results and the ground truth. More details can be found in Section A.

Baselines. We compare our DecMem with Oasis [7], MineWorld [10], and WorldMem [44] to demonstrate the effectiveness of our method. These methods are trained fully on the MineCraft datasets and hence have abundant domain knowledge. Oasis and MineWorld both use sliding windows to handle their memory, whereas WorldMem employs FOV-based memory retrieval.

#### 4.1 Quantitative Experiments

Evaluation Settings. We evaluate our method and the baseline models in handling controllable video generation within training context and beyond. All the models are provided with 221 ground-truth

frames as memory bank initialization, and tasked to generate the subsequent 120 frames. For Oasis and MineWorld with a context window of w frames (with w being 8 and 32 respectively), we initialize their memory frames with the w − 1 most recent frames. For WorldMem, which keeps 8 frames in its sliding window, we additionally keep other previous frames in its memory bank following its original setting. For our method with end-to-end memory retrieval, all the frames are fed into the model for fine-grained block retrieval. More details can be found in Section A.1.

Within Training Window. Here, we use the first 8 generated frames (i.e., 222nd–229th frames) to assess the proficiency of each model in retrieving and leveraging immediate historical context. This comparison ensures that the inference remains within the respective training window of each model. As shown in Table 1, our method outperforms all other baselines under all the metrics being considered, demonstrating the effectiveness of our precise memory in short term.

Extrapolation Generalization. Here, we use the last 8 generated frames (i.e., 334th–341st frames) to evaluate the extrapolation capability of the world models beyond the training length. As illustrated in Table 1, our method demonstrates superior robustness during extrapolation, effectively preserving spatial consistency. In contrast, competing baselines such as WorldMem exhibit rapid performance degradation after crossing the training-length threshold.

User Study. To validate the effectiveness of our method from a perceptual perspective, we conducted a user study with 58 participants, who were asked to rate the generated videos along three dimensions: Visual Quality (VQ), Action Controllability (AC), and Spatio-temporal Consistency (STC). As reported in Table 1, our method outperforms all baselines across three dimensions, validating DecMem’s overall superiority in visual quality, controllability, and spatio-temporal consistency.

Inference Latency. We also compare the inference speed by computing frame rates (in FPS) from the average generation time for 120 frames. The results in Table 1 show that our method outperforms all the baselines in efficiency, achieving nearly 2x speedup compared with the most competitive baseline.

Table 1: Quantitative comparison and user study results.

Within Training Window Extrapolation Generalization User Study Method PSNR↑ LPIPS↓ FID↓ PSNR↑ LPIPS↓ FID↓ VQ↑ AC↑ STC↑

FPS↑

MineWorld [10] 20.2989 0.1789 41.9661 14.6109 0.3736 74.2075 15.06% 14.49% 14.29% 0.1588 Oasis [7] 24.1293 0.1196 15.9163 13.4232 0.4475 63.8851 25.87% 22.26% 19.41% 1.9806 WorldMem [44] 26.5414 0.0797 11.7379 19.1401 0.2673 38.4689 19.31% 25.33% 24.16% 0.5424 Ours 30.0785 0.0494 9.8904 25.2294 0.1006 16.2667 39.77% 37.81% 42.12% 3.6496

#### 4.2 Qualitative Experiments

Following the setting of Section 4.1, we initialize the environment with 221 frames and let the models generate the subsequent 120 frames to show their spatio-temporal consistency with the initial environment. Fig. 5 demonstrates that our method achieves superior fidelity in short-term generation by precisely reconstructing local details, while other methods (including FOV-based WorldMem) struggle to maintain the accuracy of detailed memory. More importantly, in long-term scenarios, our method effectively preserves fine-grained details and overall video quality, ensuring robust spatial-temporal consistency that surpasses existing baselines.

Notably, our method supports ultra-long video synthesis of minute-long duration (see Fig. 6) while maintaining rigorous consistency in revisiting scenes, effectively overcoming temporal degradation common in long-horizon generation.

#### 4.3 Ablation Study

Component ablation. To validate the contribution of each module in DecMem, we conduct component ablation by removing SGM and ALM separately. Each variant is initialized with 221 memory frames and tasked to generate over 600 frames. We additionally compare against Dense Attention and Dense Attention with a training-free temporal decay strategy [57]. The following points are observed from the results reported in Fig. 7. (1) Dense Attention exhibits linearly growing latency (Fig. 7, left) and quality collapse in long extrapolation, revealing its fundamental inability to scale to long-horizon generation. (2) Dense Attention + Decay alleviates late-stage degradation (after 700 frames) but introduces a regression in the middle extrapolation range (around 300th-700th frames) as reflected in worse LPIPS scores relative to the dense baseline. This shows uniform temporal decay in-

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

MineWorldOasisWorldMemOursGT

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

222n" 341#$

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

MineWorldOasisWorldMemOursGT

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

Figure 5: Qualitative comparison on the Minecraft Datasets.

discriminately suppresses both redundant and informative historical features, eroding memory fidelity. (3) w/o SGM yields the worst generation quality across the entire extrapolation horizon. Without global memory retrieval, the model degenerates into a local-context-only generator and rapidly loses long-range consistency. (4) w/o ALM preserves reasonable quality in the early extrapolation stage but suffers from severe degradation beyond 600 frames, with FID and LPIPS both worst than that of vanilla Dense Attention. This confirms that, without the local anchoring mechanism, attention dispersion over the growing global context becomes the dominant failure mode, corroborating our analysis in Section 1. (5) Full DecMem matches Dense Attention in the early stage and consistently surpasses all variants in the later stage, while maintaining a near-constant computational cost (thanks to the sparse block retrieval). Only the full model, which combines sparse global retrieval with local anchoring, maintains stable quality throughout the entire rollout.

Number of retrieval blocks (top-k). In this section, we compare different numbers of memory retrieval blocks. We initialize the model with 221 memory frames and evaluate it under three settings: (a) within training window (222nd–229th frames), (b) mid-range extrapolation (334th– 341st frames), and (c) long-range extrapolation (798th–805th frames), with k set to 20, 50, 80, and 100. As shown in Table 2, increasing k does not yield consistent improvements across all metrics.

0-10s

10-20s 20-30s

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

OursGT

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

30-40s 40-50s 50-60s

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

OursGT

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

Rollout with Precise Memory

Figure 6: Minute-long video generation results with precise memory.

[Figure 121]

Figure 7: Quantitative comparison of efficiency and quality between different design. (Left) Time to generate one chunk at the current frame index. (Middle, Right) LPIPS and FID computed using 8 neighboring frames (t→t+8) at each position.

Notably, increasing it from 80 to 100 degrades both PSNR and FID under long-range extrapolation. Since k governs both the recall coverage of SGM and the consistency quality after fusion with the short-range anchored signals from ALM, an overly large value dilutes the retrieved context and weakens this complementarity. We therefore set k to 80 to balance long- and short-range quality.

Table 2: Ablation on Number of retrieval blocks

Within Training Context Mid-range Extrapolation Long-range Extrapolation k PSNR↑ LPIPS↓ FID↓ PSNR↑ LPIPS↓ FID↓ PSNR↑ LPIPS↓ FID↓

20 29.6230 0.0522 10.2337 24.3749 0.1105 16.9071 19.6497 0.2217 27.7425 50 29.9877 0.0500 9.9851 25.0605 0.1023 16.2642 19.9425 0.2100 27.0902 80 30.0785 0.0494 9.8904 25.2294 0.1006 16.2667 20.5896 0.2019 25.2748 100 30.1529 0.0490 9.9653 25.4616 0.0962 15.5984 20.5535 0.1994 25.8790

### 5 Conclusion

This paper proposes a fine-grained, learnable and scalable memory architecture for world models. We first analyze two intrinsic limitations of the naïve dense attention design under long-horizon inference, namely computational inefficiency and attention dispersion. Building upon a systematic analysis of the attention dispersion, we propose a decoupled memory architecture, consisting of a Sparse Global Memory (SGM) branch which performs fine-grained, learnable sparse memory retrieval for efficient long-range memory preservation, and an Anchored Local Memory (ALM) branch which supplies stable attention anchors that effectively counteract dispersion from distant noise. Extensive experiments validate the effectiveness of this architecture, ultimately enabling minute-long, efficient, and highly consistent controllable video generation.

### References

- [1] Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownfield, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Oliveira, Jack Parker-Holder, Frank Perbet, Guy Scully, Jeremy Shar, Stephen Spencer, Omer Tov, Ruben Villegas, Emma Wang, Jessica Yung, Cip Baetu, Jordi Berbel, David Bridson, Jake Bruce, Gavin Buttimore, Sarah Chakera, Bilva Chandra, Paul Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin, Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson, Dimple Vijaykumar, Luyu Wang, Piers Wingfield, Nat Wong, Keyang Xu, Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, Aäron van den Oord, Inbar Mosseri, Adrian Bolton, Satinder Singh, and Tim Rocktäschel. Genie 3: A new frontier for world models. 2025.
- [2] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [3] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.
- [4] Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, and Xiang Bai. Out of sight but not out of mind: Hybrid memory for dynamic video world models. arXiv preprint arXiv:2603.25716, 2026.
- [5] Taiye Chen, Xun Hu, Zihan Ding, and Chi Jin. Learning world models for interactive video generation. arXiv preprint arXiv:2505.21996, 2025.
- [6] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Lol: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026.
- [7] Decart, Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. 2024. URL https://oasis-model.github.io/. Project website.
- [8] Zicheng Duan, Jiatong Xia, Zeyu Zhang, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen, Xinyu Zhang, and Lingqiao Liu. Liveworld: Simulating out-of-sight dynamics in generative video world models. arXiv preprint arXiv:2603.07145, 2026.
- [9] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568, 2024.
- [10] Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, and Jiang Bian. Mineworld: a real-time and open-source interactive world model on minecraft. arXiv preprint arXiv:2504.08388, 2025.
- [11] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.
- [12] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2568–2577, 2025.
- [13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

- [14] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.
- [15] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198, 2025.
- [16] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [17] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37: 89834–89868, 2024.
- [18] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.
- [19] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [20] Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling sink: Bridging limited-horizon training and open-ended testing in autoregressive video diffusion. arXiv preprint arXiv:2602.07775, 2026.
- [21] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025.
- [22] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. arXiv preprint arXiv:2507.10496, 2025.
- [23] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903, 2025.
- [24] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.
- [25] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [26] Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, 37: 131434–131455, 2024.
- [27] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025.
- [28] Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger. Gta: A geometry-aware attention mechanism for multi-view transformers. arXiv preprint arXiv:2310.10375, 2023.
- [29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [30] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023.
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [32] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, and Xun Huang. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.
- [33] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [34] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.
- [35] Junshu Tang, Jiacheng Liu, Jiaqi Li, Longhuang Wu, Haoyu Yang, Penghao Zhao, Siruis Gong, Xiang Yuan, Shuai Shao, and Qinglin Lu. Hunyuan-gamecraft-2: Instruction-following interactive game world model. arXiv preprint arXiv:2511.23429, 2025.
- [36] HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, Wenhuan Li, Sheng Zhang, et al. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025.
- [37] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [38] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.
- [39] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [40] Zun Wang, Han Lin, Jaehong Yoon, Jaemin Cho, Yue Zhang, and Mohit Bansal. Anchorweave: World-consistent video generation with retrieved local spatial memories. arXiv preprint arXiv:2602.14941, 2026.
- [41] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. arXiv preprint arXiv:2506.05284, 2025.
- [42] Chendong Xiang, Jiajun Liu, Jintao Zhang, Xiao Yang, Zhengwei Fang, Shizun Wang, Zijun Wang, Yingtian Zou, Hang Su, and Jun Zhu. Geometry-aware rotary position embedding for consistent video world model. arXiv preprint arXiv:2602.07854, 2026.
- [43] Jiannan Xiang, Yi Gu, Zihan Liu, Zeyu Feng, Qiyue Gao, Yiyan Hu, Benhao Huang, Guangyi Liu, Yichi Yang, Kun Zhou, et al. Pan: A world model for general, interactable, and long-horizon world simulation. arXiv preprint arXiv:2511.09057, 2025.
- [44] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025.
- [45] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.
- [46] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [47] Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, Jun Zhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu Yang, Qiang Fu, et al. Yan: Foundational interactive video generation. arXiv preprint arXiv:2508.08601, 2025.
- [48] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.

- [49] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025.
- [50] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025.
- [51] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325, 2025.
- [52] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.
- [53] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [54] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.
- [55] Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory. arXiv preprint arXiv:2512.15716, 2025.
- [56] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.
- [57] Min Zhao, Hongzhou Zhu, Yingze Wang, Bokai Yan, Jintao Zhang, Guande He, Ling Yang, Chongxuan Li, and Jun Zhu. Ultravico: Breaking extrapolation limits in video diffusion transformers. arXiv preprint arXiv:2511.20123, 2025.

### A Implementation Details

#### A.1 Experiment Settings.

Training and Evaluation Details. We train our DecMem on WorldMem [44] datasets, which contains 11 k videos with 1500 frames at 360x640 resolution and 10 FPS. We randomly sample 237 frames and resize them to 352x640 for training and evaluation. We adopt a two-stage training strategy. In the first stage, we initialize from a pre-trained full-sequence video generation checkpoint and adapt its architecture into a causal generation paradigm, training for 25K steps so that the model robustly establishes autoregressive generation as a reliable backbone for subsequent memory injection. In the second stage, we integrate the proposed Sparse Global Memory (SGM) and Anchored Local Memory (ALM) modules on top of this causal backbone, and jointly train for an additional 25K steps so that the model learns to retrieve sparse global context and exploit anchored local memory in a coordinated manner, ultimately delivering fine-grained long-horizon spatiotemporal consistency. We apply the AdamW optimizer with a learning rate of 2e-5 and adopts teacher forcing strategy. The training process lasts for approximately 7 days. For evaluation, we apply 300 videos from the WorldMem [44] datasets ensuring no overlap with the training data. For diffusion-based methods, we keep all of them to denoise 20 steps for fair comparison.

User Study Details. To assess generation quality from a perceptual standpoint, we conducted a user study with 58 participants. Each trial was presented in a unified layout (see Fig. 8): the left panel simultaneously showed the ground-truth reference clip together with its corresponding action control signals, while the right panel played, side by side, the candidate videos generated by different methods, which were randomly shuffled and anonymized (labeled A, B, C, and D, respectively) to eliminate positional bias and method-identification cues. Participants were asked to select the indices of the videos they deemed best under each predefined evaluation criterion, i.e, visual quality, action controllability, and spatio-temporal consistency. We aggregated all responses and computed, for every metric, the preference rate of each method; the final results are summarized in Table 1.

[Figure 122]

Figure 8: User study demo.

#### A.2 Base Model Architecture

For our pretrained video generation models, we apply the latent diffusion transformer as our base model as illustrated in Fig. 9. Since we rely on actions and poses to control scene generation rather than using prompts for guidance, we discard the cross-attention module designed for the T2V task, employ spatial self-attention to fuse information within frames, and use spatiotemporal self-attention to capture the relationships among latents across frames. Before each attention or feed-forward network (FFN) module, the timestep is mapped to a scale, which is then used to apply RMSNorm [52] to the features.

[Figure 123]

[Figure 124]

[Figure 125]

|Timestep|
|---|

Skip

VAE Encoder

Addition

| |RMSNorm&Scale<br><br>Spatiotemporal|
|---|---|
|Self-Attn| |

SpatiotemporalSelf-Attn

RMSNorm&Scale

RMSNorm&Scale

RMSNorm&Scale

Feed-ForwardNN

SpatialSelf-Attn

[Figure 126]

[Figure 127]

[Figure 128]

VAE Decoder

zzz

𝑧zz

Transformer Block

###### x N

Figure 9: Base Model Architecture.

#### A.3 Details of Multimodal Position Embedding

In Section 3.5, we introduce a multimodal position embedding that injects camera geometry, patch coordinates, and frame indices into the attention computation. Concretely, the per-head dimension of 72 is evenly partitioned into three groups of 24 channels, with each group encoding one modality through its corresponding transformation.

Camera Embedding. For the camera-pose channel, we follow PRoPE [22] for projective positional encoding. Let Kt ∈ R3×3 denote the camera intrinsics of the t-th frame and Ttcw = (Rtcw, tcwt ) ∈ SE(3) denote its world-to-camera extrinsics. The standard 3×4 projection matrix that maps a 3D world point to the image plane of camera t is:

Pt = Kt 03×1 Ttcw. (6)

To make Pt invertible, the standard basis vector e4 = (0,0,0,1)⊤ is appended to Pt as its last row, yielding a 4×4 matrix:

Pt e⊤4 ∈ R4×4. (7)

P˜t =

The obtained P˜t captures the full viewing frustum and hence it can be applied for encoding the complete geometric relationship between camera views. This can be computed as follows:

−1 Kt−1

Kt

#### 0 0 1

0 0 1

P˜t

P˜t−1

Ttcw

Ttcw

, (8)

=

1

2

1

2

1

2

which simultaneously models pose and intrinsics differences between two views. We apply P˜t as a block-diagonal transformation on the camera-pose channels:

cam/4 ⊗ P˜t ∈ Rd

cam×dcam, (9)

Rcam(Pt) = Id

where dcam is the number of feature channels assigned to the camera modality and ⊗ denotes the Kronecker product. Together with Eq. (5) in the main text, the resulting query–key inner product

is modulated by P˜t

P˜t−1

as in Eq. (8), so attention is conditioned on the relative camera frustum geometry.

1

2

Spatial Embedding. For the spatial channels with dimension dsp = 24, we apply the standard 2D axial RoPE [33] on patch coordinates (x,y). The channels are split evenly into two halves encoding the horizontal and vertical axes respectively:

##### Rsp(x,y) = diag R1d x; d2sp , R1d y; d2sp , (10)

where R1d(p;d) denotes the canonical 1D rotary matrix of dimension d at position p, built from the frequency basis θi = θbase−2i/d, i = 0,...,d/2 − 1. The resulting query–key inner product depends only on the relative offset (x1−x2, y1−y2), yielding translation-equivariant intra-frame spatial perception.

Temporal Embedding. For the temporal channels with dimension dtem = 24, we apply 1D RoPE along the frame index t:

Rtem(t) = R1d(t; dtem), (11) which modulates attention by the relative frame distance t1−t2.

Apart from modulating the inner product between query and key with position embedding (Eq. (5)), we follow previous work [28] to inject a relative transformation to the value and the final output for more aligned feature aggregation. However, we do not apply such a transformation on the temporal channels to values or outputs. This process is denoted as:

R(csi) = diag(Rcam,Rsp,I) =

Rcam(Pt

#### ) 0 0

i

0 Rsp(xi,yi) 0 0 0 I

(12)

vi = (R(csi))−1Projv(hi), o′i = R(csi)oi (13)

where I represents the identity matrix, Projv is the value projection transformations of the hidden states, o′i is the position-encoded output features of i-th token.

By explicitly modeling spatiotemporal and geometric relationships, this multimodal RoPE design strengthens the model’s spatiotemporal awareness and establishes a reliable prior that underpins the fine-grained memory.

### B More Analysis about Attention Dispersion

In Section 3.2, we analyze the issue of attention dispersion in long-horizon world modeling. In this section, we further provide a quantitative analysis by examining how the proportions of critical weights and negligible attention weights evolve during inference. As illustrated in Fig. 10, for dense attention, the proportion of tail weights gradually increases as inference progresses, while the proportion of key weights correspondingly decreases. This opposing trend leads to the dilution of critical attention.

Although a training-free decay strategy can mitigate the growth of negligible weights and thus help maintain short-term quality, it still exhibits a similar trend to dense attention. Moreover, as analyzed in Section 3.2, it degrades the long-term memory capability of the world model.

In contrast, our method maintains the proportion of irrelevant weights at a relatively constant level throughout inference, thereby reducing the influence of unimportant tail features and preserving a stable share of critical attention weights. This demonstrates the advantage of our decoupled memory design. By introducing ALM as an attention anchor, the model is encouraged to focus most of its attention on important regions, preventing severe quality degradation caused by attention dispersion. Meanwhile, the SGM architecture effectively leverages global memory to explore and utilize long-term temporal features.

[Figure 129]

Figure 10: (Left) Sum of negligible attention weights (<0.02) against inference frame index. (Right) Sum of critical attention weights (>0.05) against inference frame index.

### C Comparison with Industrial-scale Model

To further demonstrate the effectiveness of our method, we compare it against two industrial world models, Matrix-Game 2.0 [11] and WorldPlay [34]. Both baselines are trained on multi-domain datasets and thus exhibit stronger cross-scene generalization. Besides, they follow Image-to-Video (I2V) or Text-to-Video (T2V) paradigm and do not support video-clip-based memory banks initialization. To guarantee a fair comparison, we deliberately forgo DecMem’s advantage of video-conditioned environment initialization and align our input interface with the single-image protocol of the baselines: specifically, we replicate the VAE latent of a single reference frame along the temporal axis to populate the initial chunk that serves as the model’s contextual condition.

Each model is tasked to generate 30-second interactive videos conditioned with a single initial image. Owing to insufficient initialization information, ground-truth videos naturally diverge with even identical action sequences, rendering direct comparisons with ground-truth videos meaningless. We therefore adopt the user-study protocol following Section 4.1, performing perceptual evaluations along three perceptual axes: Visual Quality, Action Controllability, and Spatio-temporal Consistency.

As shown in Table 3, our method achieves visual fidelity and action controllability on par with advanced industrial models, while advancing in long-horizon spatio-temporal consistency (+5.14%). These results demonstrate the effectiveness of our approach for long-term, consistent, and controllable world generation.

Table 3: Results of user study for comparison with industrial world models.

Evaluation Criteria Matrix-Game 2.0 [11] WorldPlay [34] Ours

Visual Quality↑ 28.74% 35.04% 36.22% Action Controllability↑ 33.07% 29.96% 36.96% Spatio-temporal Consistency↑ 26.07% 34.39% 39.53%

### D More Ablation study

Visualization of the effectiveness of each Module To further validate the efficacy of each component, we compare DecMem against a series of ablated variants and qualitatively analyze the generated samples. Following the protocol in Section 4.1, we initialize the memory bank with 221 frames and let each model auto-regressively roll out the subsequent 500 frames. As shown in Fig. 11, (1) w/o SGM, deprived of the sparse global retrieval mechanism, can only attend to nearby frames, and once generation exceeds the local context window, long-range memory collapses entirely, causing the output to drift markedly away from the ground truth (GT) and manifest as pronounced sceneidentity drift. (2) w/o ALM initially preserves short-range fidelity, however, the generation quality deteriorates sharply beyond roughly 600 frames, with high-frequency details lost and the underlying scene geometry collapsing, revealing that without anchored local memory the model is prone to drift in long-term because of the attention dispersion. In contrast, (3) Full DecMem simultaneously preserves fidelity in short-range and supports stable long-horizon extrapolation, ultimately delivering fine-grained, spatiotemporally consistent minute-long video generation. These observations directly corroborate our central claim that the global consistency and long-range extrapolation fidelity can be addressed through a decoupled memory architecture.

Action classifier free guidance Inspired by text-conditioned visual generation models that use classifier-free guidance (CFG) to adjust generation diversity and adherence to the text, we explored the impact of applying CFG to actions in world models on image quality. Specifically, during training, we randomly set the conditional action embeddings to zero and added them to the original latent, simulating the approach of training with dropped actions. During inference, the model predicts the flow velocity vθ(zt,at,t) based on action at for each latent zt at step t. When CFG is applied, we first obtain the conditioned and unconditioned predictions, vθ(zt,at,t) and vθ(zt,∅,t). The final guided velocity is computed as a weighted combination:

vˆθ(zt,at,t) = vθ(zt,∅,t) + s · (vθ(zt,at,t) − vθ(zt,∅,t)) (14)

where s denotes the guidance scale. Notably, for the sake of brevity, we omit other conditional information and illustrate the auto-regressive denoising process for a single frame only.

300th

400th 500th 600th 700th

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

(w/oSGM) Ours

OursGT Ours

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

(w/oALM)

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Figure 11: The visualization results to show the effectiveness of each components.

Following the experimental setup in Section 4.1, we compare the quantitative performance of the two methods. For CFG branch, we apply a guidance scale of 7.5 in the denoising process. As shown in Table 4, disabling CFG yields higher pixel fidelity (PSNR) within the training horizon and during the early stage of extrapolation. However, as the extrapolation length grows, the generation quality of the CFG-free model degrades rapidly and eventually suffers from large distribution difference from ground truth, whereas enabling CFG substantially keeps both the stability and the generation quality under long-horizon extrapolation. This observation suggests that CFG trades a marginal loss in short-range fidelity for a pronounced gain in long-range quality.

Table 4: Ablation on action classifier-free guidance.

Within Training Context Middle Extrapolation Long Extrapolation Method PSNR↑ LPIPS↓ FID↓ PSNR↑ LPIPS↓ FID↓ PSNR↑ LPIPS↓ FID↓

Ours (w/o CFG) 30.7952 0.0471 10.0613 26.1088 0.1021 17.7244 21.4909 0.2380 42.5515 Ours (w/ CFG) 30.0785 0.0494 9.8904 25.2294 0.1006 16.2667 20.5896 0.2019 25.2748

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

Turn Left

Frame 0 Frame 39 Frame 382 Frame 609 Frame 675 Frame 1100

Turn Right

|[Figure 156]|
|---|

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

Pred Frame

Frame 60 Frame 200 Frame 376 Frame 453 Frame 632 Frame 1493

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

Memory Frame

Frame 41 Frame 90 Frame 1188 Frame 1250 Frame 1295 Frame 1444

Figure 12: Long video generation on Context as Memory [50] dataset.

### E More Visualization Results

In Fig. 13 and Fig. 14, following the setting in Section 4.2, we present additional comparisons with baseline methods. Across diverse scenarios, our model consistently demonstrates superior memory performance. Furthermore, as demonstrated in Fig. 15, it is capable of inference for up to one minute while maintaining high fidelity.

To demonstrate the effectiveness of our method across diverse datasets, we adopt the Context-asMemory [50] dataset for both training and evaluation. This dataset contains abundant revisiting scenarios and can be used to assess the model’s memory capability. We drive the camera through a revisitation trajectory—repeated leftward and rightward pans—in three stylistically distinct environments: an island, a city, and a chemical plant, systematically probing the model’s fine-grained spatiotemporal consistency upon re-entering previously visited regions. As shown in the Fig. 12, our model faithfully reproduces previously observed structural layouts and local details across all three settings, demonstrating that the proposed memory mechanism sustains robust long-term consistency across diverse environments.

### F Licenses

The WorldMem [44] datasets and code used in the main experiments is released under the S-Lab License 1.0. The baseline code bases including oasis [7], MineWorld [10] and Matrix-Game [11] are all released under the MIT License. HY-WorldPlay [34] is released under the TENCENT HYWORLDPLAY COMMUNITY LICENSE AGREEMENT. We have strictly adhered to the terms and usage conditions of all the aforementioned licenses throughout our experiments.

### G Broader Impacts and Limitations

Broader Impacts The method proposed in this work aims to improve the spatiotemporal consistency of world models and to enhance their long-horizon extrapolation capability. It can be applied to some applications including gaming, virtual simulation, embodied AI, and film creation. At the same time, as a controllable approach capable of synthesizing long-duration, highly consistent videos, our work may inadvertently amplify the risks of technological misuse. Specifically, the ability to generate temporally extended and spatiotemporally coherent video could be exploited for fraudulent forgery and may substantially lower the barrier to producing disinformation at scale. We therefore call upon the community to strengthen defensive research directions—such as forgery detection and content provenance tracing—as essential mitigation measures against these risks.

Limitations Our research focuses primarily on solving the precise memory and extrapolation generalization rather than inference acceleration via distillation, so real-time performance has not yet been achieved. In the future work, we will focus on developing an efficient real-time world model with hybrid memory mechanisms combining compressed global memory and fine-grained object-level memory, further improving the long-term consistency.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

MineWorldOasisWorldMemOursGTMineWorldOasisWorldMemOursGT

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

222n" 341#$

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Figure 13: More qualitative comparison between our methods and other baselines.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

MineWorldOasisWorldMemOursGTMineWorldOasisWorldMemOursGT

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

222n" 341#$

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

Figure 14: More qualitative comparison between our methods and other baselines.

0-10s

10-20s 20-30s

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

OursGT

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

30-40s 40-50s 50-60s

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

OursGT

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

0-10s 10-20s 20-30s

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

OursGT

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

30-40s 40-50s 50-60s

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

OursGT

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Rollout with Precise Memory

Figure 15: More visualization results of minute-long video generation.

