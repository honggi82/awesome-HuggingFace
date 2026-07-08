# arXiv:2605.28691v1[cs.CV]27May2026

[Figure 1]

## OSP-Next: Efficient High-Quality Video Generation with Sparse Sequence Parallelism, HiF8 Quantization, and Reinforcement Learning

Yunyang Ge1,∗ Xianyi He1,∗ Zezhong Zhang2 Bin Lin1 Bin Zhu1,3 Xinhua Cheng1 Li Yuan1,†

Open-Sora Plan Team

∗Equal contribution, †Corresponding author {yunyang, HeXianyi, linbin.ece, binzhu, chengxinhua}@stu.pku.edu.cn zezhong002@e.ntu.edu.sg yuanli-ece@pku.edu.cn

1Peking University, 2Nanyang Technological University, Singapore, 3Rabbitpre AI

#### Abstract

Diffusion Transformers achieve strong video generation quality, but the quadratic cost of full attention limits efficiency. We introduce OSP-Next, an efficient text-to-video generation model that integrates sparse attention, parallelism, quantization, and reinforcement learning. OSP-Next uses a hybrid full-sparse attention architecture, where the sparse component is implemented with Skiparse-2D Attention. This fixed-pattern mechanism applies token-wise and group-wise sparse attention along spatial dimensions, leveraging locality while maintaining native compatibility with FlashAttention kernels. Based on the local equivalence of rearrangement in Skiparse-2D Attention, we further propose Sparse Sequence Parallelism (SSP), which partitions subsequences across ranks and switches sparse patterns through a single All-to-All communication. Compared with Ulysses Sequence Parallelism (SP), SSP provides a native parallel strategy for sparse attention and reduces communication volume by 75%. OSP-Next also incorporates HiF8 quantization to enable stable joint training with 8-bit quantization and sparse fine-tuning, and applies Mix-GRPO post-training to improve the performance of the sparse model. Experiments show that OSP-Next achieves a VBench total score of 83.73%, surpassing the Wan2.1 baseline. Under the 5-second 720P and 5-second 768P settings, OSP-Next achieves up to 1.64× single-GPU speedup and over 1.52× eight-GPU speedup on NVIDIA H200 GPUs. In addition, with only a 0.4% drop in VBench total score, OSP-Next-HiF8 achieves 1.69× and 2.27× speedups under the two settings on a single Ascend 950PR, demonstrating the efficiency and performance of OSP-Next across hardware platforms.

[Figure 2]

GitHub Repo: https://github.com/PKU-YuanGroup/OSP-Next HuggingFace Model: https://huggingface.co/yunyangge/OSP-Next

[Figure 3]

#### 1 Introduction

In recent years, Diffusion Transformers (DiTs) [22, 35] have increasingly replaced UNets [24, 25] as denoisers for image and video generation [6, 8, 11, 14, 19, 28, 29, 34, 45, 46], demonstrating strong generative capacity.

However, the computational inefficiency of DiTs and the long token sequences required for video modeling limits speed and achievable resolution. For example, models such as HunyuanVideo [11] and Wan2.1 [28] require approximately 30 minutes to one hour to generate a 5-second 720p video on a single NVIDIA A100 GPU.

To mitigate the challenges introduced by quadratic computational complexity, prior works have explored various techniques, including sparse attention, linear attention, and quantization. Training-free sparse attention methods [30, 31, 33, 37] typically select tokens dynamically according to token similarity scores. This strategy relies on a pretrained full attention model and keeps the model weights fixed, preventing the model from adapting to the distribution shift caused by changes in the attention pattern. Training-based sparse attention methods [2, 41] also commonly select tokens according to predefined similarity rules. However, these methods require dynamically constructed attention masks and are typically implemented with FlexAttention [5] or custom kernels specifically optimized for sparse attention, making them difficult to natively integrate with efficient FlashAttention [3, 4, 26] kernels. Moreover, irregular attention masks are difficult to parallelize and can lead to load imbalance across different ranks. Linear attention [1, 36, 40] reduces computational complexity to linear in the sequence length by design, but this simplification also limits the expressive capacity of the model. Quantization methods, including FP8 and INT8, provide a substantially narrower representational range than BF16 and FP16. Consequently, models trained with 8-bit precision [38, 39] often exhibit limited performance.

We introduce OSP-Next, a text-to-video model that integrates sparse attention, parallelism, quantization, and reinforcement learning. Through a combination of optimization strategies, OSP-Next mitigates the performance and efficiency issues observed in prior methods while satisfying the requirements of parallel computation and native compatibility with FlashAttention kernels. Our main contributions are as follows.

- 1. We propose Skiparse-2D Attention. In contrast to sparse patterns based on dynamic token selection, Open-Sora Plan v1.3 introduces Skiparse-1D Attention, a fixed-rule attention pattern that alternates between token-wise skip sparse attention and group-wise skip sparse attention to approximate 3D full attention. In OSP-Next, Skiparse Attention is applied separately along the height and width dimensions to form Skiparse-2D Attention, which better aligns with the spatial locality of image and video modalities. Experimental results indicate that Skiparse-2D Attention captures an interaction pattern closer to that of 3D full attention than Skiparse-1D Attention, thereby achieving better performance. Moreover, by avoiding dynamic construction of a 2D attention mask over the sequence dimension, Skiparse-2D Attention remains natively compatible with efficient FlashAttention kernels without custom kernels.
- 2. We introduce Sparse Sequence Parallelism (SSP). Skiparse Attention first performs Skiparse Rearrange, which partitions the original sequence into multiple subsequences concatenated along the batch dimension, and then computes attention in parallel. Skiparse Rearrange exhibits local equivalence: applying Skiparse Rearrange to the original sequence is equivalent to applying it independently to each minimal repeatable unit. Therefore, different subsequences can be assigned to different ranks, with attention computation performed independently on each rank. When the sparse pattern of Skiparse Attention switches between token-wise sparse attention and group-wise sparse attention, the local equivalence property of Skiparse Rearrange allows data exchange to be completed by one local rearrangement on each rank followed by one All-to-All communication step within the communication group. In addition, since the subsequences produced by Skiparse Rearrange have equal lengths, load balance is maintained across ranks. Compared with Ulysses Sequence Parallelism (SP), SSP reduces the communication volume by 75% and decreases the number of communication steps per block from four to one.
- 3. We introduce joint training of sparse-model fine-tuning and HiF8 quantization. HiF8 is an 8-bit precision format that dynamically adjusts the numbers of exponent and mantissa bits, enabling both high precision and a large dynamic range. Benefiting from the utilization of locality in image and video modalities by Skiparse-2D Attention and the training stability of the hybrid architecture in OSP-Next, which combines full attention Blocks and Skiparse Attention Blocks, we can perform sparse-model fine-tuning and 8-bit fine-tuning simultaneously. The loss curve of OSP-Next trained with HiF8 almost overlaps with that of OSP-Next trained with BF16, and the gap in VBench [9] score remains within 0.5%.

- 4. We introduce reinforcement learning for sparse models. Although the attention pattern of OSP-Next closely approximates that of the pretrained full-attention model, fine-tuning from the pretrained model to the sparse model still degrades generation quality. To mitigate this quality degradation, we further optimize OSP-Next with Mix-GRPO [12] during the post-training stage. Experimental results show that reinforcement learning substantially improves the generation quality of sparse models.

OSP-Next achieves a VBench total score of 83.73%, outperforming the Wan2.1 baseline based on full attention. Under the 5-second 720P with padding and 5-second 768P without padding settings, OSP-Next achieves

- 1.53× and 1.64× speedups on a single NVIDIA H200, and 1.42× and 1.52× speedups on eight NVIDIA H200 GPUs, respectively. In addition, OSP-Next-HiF8 achieves 1.69× and 2.27× speedups on a single Ascend 950PR, with only a 0.4% VBench total score drop relative to the baseline. These results show that OSP-Next provides high-quality generation and acceleration across different hardware platforms, offering a new pipeline for trainable and parallelizable sparse video generation models.

- 2 Related Work

- 2.1 Sparse Video Generation Model

Sparse video generation models have recently attracted increasing attention as a practical solution to the high computational cost of video Diffusion Transformers. Open-Sora Plan [14] introduces Skiparse Attention for video generation, which sparsifies 3D attention by reorganizing tokens into sparse subsequences while preserving spatio-temporal modeling ability. Sparse VideoGen [30] accelerates pretrained video DiTs in a training-free manner by identifying spatial and temporal attention heads and applying head-wise sparse computation during inference. Sparse VideoGen2 [33] further improves training-free sparse inference by using semantic-aware token permutation, which clusters related tokens to obtain more efficient sparse attention layouts. Sparse-vDiT [2] analyzes the attention maps of video DiTs and exploits recurring sparse patterns, such as diagonal and stripe structures, with hardware-aware sparse kernel selection. VSA [42] adopts a trainable coarse-to-fine sparsification strategy, where video tokens are first grouped into tiles and only high-importance tiles are selected for fine-grained token-level attention. Currently, trainable sparse methods for VideoDiT remain limited and cannot provide speed gains during pretraining. Moreover, existing trainable sparse methods are mostly based on complex dynamic token selection strategies. Such methods are difficult to combine with parallelization strategies and usually require complex masks implemented with FlexAttention, making them incompatible with flash attention.

##### 2.2 Parallel Strategy

Scaling video generation models requires efficient parallel training and inference strategies. Data parallelism, such as DDP, replicates the full model on each device and splits the input batch, but does not reduce the memory cost of model parameters or activations on each GPU. Tensor parallelism [27] partitions matrix operations or attention heads across devices, reducing per-device computation but introducing communication within Transformer layers. FSDP [43] further shards model parameters, gradients, and optimizer states across devices, making it effective for training large models under limited GPU memory. However, these strategies mainly parallelize the batch, channel, or parameter dimensions, while video generation is often bottlenecked by the extremely long spatio-temporal sequence length.

Sequence parallelism is therefore particularly important for video Diffusion Transformers. Ulysses-style sequence parallelism [10] partitions the sequence dimension and uses All-to-All communication to gather the required tokens for attention computation. Ring Attention [15] instead computes attention in a blockwise manner by circulating key-value blocks among devices, enabling long-sequence attention while overlapping communication with computation. Recent unified sequence-parallel frameworks [7] further combine these two paradigms to improve scalability under different sequence lengths and hardware settings.

Despite these advances, the interaction between sequence parallelism and sparse video generation remains underexplored. Most sparse video generation methods focus on designing sparse attention patterns or accelerating pretrained models, but do not explicitly consider how the sparse token layout should be distributed across devices. Directly combining sparse attention with sequence parallelism may lead to workload imbalance

and irregular communication, since different devices can receive different numbers of active tokens or attention blocks. Therefore, designing sparse video generation models that are naturally compatible with sequenceparallel execution remains an important and open direction.

##### 2.3 Fine-Grained Quantization

A central challenge in FP8 [20] mixed-precision training is controlling quantization error caused by the limited dynamic range of standard FP8 formats, such as E4M3 and E5M2. To mitigate this issue, existing methods commonly adopt fine-grained quantization, where independent scaling factors are assigned to smaller subsets of tensors to better match local numerical distributions.

For activations, per-token quantization assigns an individual scale to each token or tile. This is effective for DiT-based models and LLMs, where activation magnitudes vary significantly across tokens and are often affected by outliers. Compared with per-tensor scaling, per-token scaling prevents a few outlier tokens from dominating the global scale and thus reduces quantization error. However, it requires computing token-wise maximum values before quantization, introducing reduction overhead and creating a serial dependency before MatMul operations.

For weights, per-channel and per-block quantization are widely used to capture distribution differences across output channels or local matrix regions. Per-channel quantization assigns one scale to each output channel, while per-block quantization further divides the weight matrix into fixed-size tiles, such as 128 × 128, to balance accuracy and metadata cost. More fine-grained schemes, such as MXFP8 [21] microscaling, assign a shared microscale to every 32 consecutive elements, offering stronger robustness to local outliers but requiring denser scale metadata and more complex hardware support.

Overall, fine-grained quantization improves numerical accuracy by using more adaptive scaling, but this comes with non-negligible system overhead. Frequent scale computation, storage, loading, and broadcasting increase memory bandwidth pressure and complicate hardware implementation, partially offsetting the efficiency benefits of 8-bit computation. In contrast, HiF8 [18] provides a wider dynamic range and can therefore rely on coarse per-tensor quantization while maintaining training accuracy close to the BF16 baseline. This avoids much of the metadata and reduction overhead required by fine-grained FP8 quantization, leading to a simpler and more efficient training pipeline.

##### 2.4 Reinforcement Learning for Video Generation

Reinforcement Learning (RL) post-training has recently been explored to improve the preference alignment of diffusion-based video generation models by directly optimizing reward signals related to visual quality, motion consistency, and text-video alignment. FlowGRPO [16] formulates the denoising process as a Markov decision process and applies GRPO-style optimization over the full trajectory, but full-step optimization is costly for video generation due to long spatio-temporal sequences and expensive video-level reward evaluation. Mix-GRPO [12] improves efficiency by combining stochastic SDE sampling with deterministic ODE sampling, restricting policy-gradient updates to selected denoising timesteps. Following this trajectory-level RL paradigm, DanceGRPO [32] adapts GRPO to dance video generation with rewards that emphasize pose dynamics, motion naturalness, and rhythm consistency, while BranchGRPO [13] introduces branched denoising rollouts, allowing multiple candidates to share early denoising computation before diverging for reward-based optimization.

Different from these GRPO-style methods that optimize sampled denoising trajectories with policy-gradient updates, DiffusionNFT [44] follows a different reward fine-tuning paradigm for diffusion models. It avoids relying on approximate diffusion transition probabilities for importance sampling or policy-ratio estimation, which are commonly used when casting the denoising process as an MDP. By reducing the dependence on such transition approximation, DiffusionNFT mitigates potential bias in diffusion RL post-training and provides a more direct optimization route. For video generation, this is especially relevant because trajectory-level rollout, policy-ratio computation, and reward evaluation become particularly expensive for long spatio-temporal sequences. Despite these advances, existing RL post-training methods mainly target standard diffusion or video generation models, while their compatibility with sparse attention, sequence parallelism, and low-precision training remains less studied.

###### Skiparse-1D Attention

- 1D: Flatten to sequence and skip attn

Skiparse-2D Attention

- 2D: Skiparse-1D rearrange along H and W

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

H

Rearrange Alternately

T W

###### Full Attn Token-wise Skip Attn Group-wise Skip Attn

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

H

Rearrange Alternately

T W

Full Attn

Token-wise Skip Attn Group-wise Skip Attn

- Figure 1 Skiparse-2D Attention. Skiparse-2D Attention alternates between Token-wise Sparse Attention and Group-wise Sparse Attention, enabling any two tokens to interact within at most two attention operations. Unlike

- Skiparse-1D Attention, which treats a video as a flattened one-dimensional sequence, Skiparse-2D Attention applies the Skiparse-1D pattern orthogonally along the height and width dimensions, thereby better leveraging the locality of image and video modalities.

#### 3 Method

##### 3.1 Skiparse-2D Attention

Open-Sora Plan v1.3 [14] first introduces Skiparse Attention, a trainable sparse attention mechanism for DiTs. In this work, video latents are flattened into a one-dimensional sequence, and token-wise skip sparse attention and group-wise skip sparse attention are applied alternately to construct a sparse attention mechanism whose computational cost lies between spatial-temporal attention and 3D full attention. Open-Sora Plan v1.3 adopts a hybrid DiT architecture, applying full attention in the first and last several layers while using Skiparse Attention in the middle layers, which provides substantial acceleration while maintaining sufficient generation quality. In the subsequent Open-Sora Plan v1.5, the model is extended to a hybrid architecture with a U-shaped varying sparsity pattern. This design achieves a VBench score above 83% and becomes the first model trained from scratch to approach the performance of open-source 3D full attention models like HunyuanVideo.

However, simply flattening a video into a one-dimensional sequence introduces several issues. First, it is inconsistent with the 2D locality of image and video modalities. Under the above Skiparse-1D Attention, interactions among local tokens are restricted: some spatially adjacent tokens cannot interact within a single attention operation and instead require intermediate tokens. Moreover, the interaction pattern among global tokens fails to match the 2D structure of image and video modalities. These limitations weaken the modeling capability of Skiparse-1D Attention. This limitation becomes more pronounced at higher sparsity levels. Furthermore, in the any resolution training setting, Skiparse-1D Attention can produce inconsistent token interaction patterns. It constructs subsequences from the flattened one-dimensional sequence and pads only the end when necessary, while ignoring the spatial positions of pixels. Consequently, tokens at the same spatial position across different videos may be assigned to different subsequences and thus follow different interaction patterns, further making the model more difficult to optimize.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Video A No Padding

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

[Figure 4]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- Video B

Need Padding

- Video C

Padding to Canonical Shape

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Need Padding

Skiparse Rearrange

- Figure 2 Any Resolution Strategy. The minimum repeating unit of Skiparse-2D Attention is a k2 ∗ k2 subfigure, where k denotes the sparse ratio. When the height or width of a video shape is not a multiple of k2, we pad it to the nearest multiple of k2 and generate the corresponding attention mask. This any resolution strategy ensures that tokens at the same spatial position across videos of different shapes are assigned to the same subfigure.

Therefore, we extend Skiparse-2D Attention to better support image and video modalities. By viewing

- Skiparse-2D Attention as applying Skiparse-1D Attention separately along the width and height dimensions, we obtain the interactive pattern of Skiparse-2D Attention, as shown in Fig. 1. The token-wise interaction pattern resembles pixel unshuffle, but the resulting subfigures are concatenated along the batch dimension as different subsequences rather than along the channel dimension. The group-wise interaction resembles patch unshuffle, where the resulting subfigures are likewise treated as different subsequences and concatenated along the batch dimension. During attention computation, interactions occur only within each subfigure. Since the complexity of attention is O(n2), both token-wise sparse attention and group-wise sparse attention reduce

the sequence dimension to k1 of the original size and increase the batch dimension k times. As a result, the total computation of the attention operation is reduced to k1 of the original cost, where k denotes the sparse ratio and corresponds to the skip interval in both operations. By alternating token-wise sparse attention and group-wise sparse attention in the model, any two tokens can interact through at most two attention operations.

- 3.2 Any Resolution Strategy The Skiparse Rearrange satisfies local equivalence: applying token-wise or group-wise rearrange to an H × W

figure is equivalent to independently applying the rearrange to kH2 × Wk2 subfigures, each of size k2 × k2. In other words, the smallest repeatable unit of the Skiparse Rearrange is a k2 × k2 subfigure.

When H or W is not divisible by k2, the naive Skiparse Rearrange becomes invalid because some subfigures cannot form the smallest repeatable unit. To support Skiparse Rearrange at arbitrary resolutions, we pad the original figure. The padding operation should preserve the local equivalence of Skiparse Rearrange. Therefore, we apply padding to the end of both the H and W dimensions and construct the corresponding attention mask. As shown in the fig. 2, this padding strategy preserves the local equivalence of Skiparse Rearrange and keeps the interaction pattern among non-padding tokens unchanged, thereby enabling training at arbitrary resolutions.

According to the any resolution strategy, the attention mask in Skiparse Attention is set to None when both H and W are exact multiples of k2. Otherwise, it is only a 1D mask of the flattened sequence dimension,

Algorithm 1 Pseudocode of Skiparse Rerrange

###### Token-wise Pattern to Group-wise Pattern

[Figure 5]

[Figure 6]

def orig_to_tsa(x, T, H, W, P): return rearrange(x, "b (t h p w q) c -> (p q b) (t h w) c",

[Figure 7]

[Figure 8]

p=P, q=P, h=H // P, w=W // P) def tsa_to_orig(x, T, H, W, P):

- Rank 0 Shard
- Rank 1 Shard
- Rank 2 Shard
- Rank 3 Shard

- Rank 0 Shard
- Rank 1 Shard
- Rank 2 Shard
- Rank 3 Shard

[Figure 9]

[Figure 10]

return rearrange(x, "(p q b) (t h w) c -> b (t h p w q) c",

p=P, q=P, h=H // P, w=W // P) def orig_to_gsa(x, T, H, W, P):

[Figure 11]

[Figure 12]

LocalRearrange

All-to-All

return rearrange( x, "b (txh p1 p2 w q1 q2) c -> (p1 q1 b) (txh p2 w q2) c", p1=P, q1=P, p2=P, q2=P, w=W // (P ** 2)

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

) def gsa_to_orig(x, T, H, W, P):

return rearrange( x, "(p1 q1 b) (txh p2 w q2) c -> b (txh p1 p2 w q1 q2) c", p1=P, q1=P, p2=P, q2=P, w=W // (P ** 2)

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

) def tsa_to_gsa(x, T, H, W, P): return rearrange(

Group-wise Pattern to Token-wise Pattern

x, "(p2 q2 b) (txh_p1 p1 w_q1 q1) c -> (p1 q1 b) (txh_p1 p2 w_q1 q2) c",

[Figure 21]

[Figure 22]

p1=P, q1=P, p2=P, q2=P, w_q1=W // (P ** 2) )

[Figure 23]

[Figure 24]

def gsa_to_tsa(x, T, H, W, P): return rearrange(

- Rank 0 Shard
- Rank 1 Shard
- Rank 2 Shard
- Rank 3 Shard

- Rank 0 Shard
- Rank 1 Shard
- Rank 2 Shard
- Rank 3 Shard

[Figure 25]

[Figure 26]

x, "(p1 q1 b) (txh_p1 p2 w_q1 q2) c -> (p2 q2 b) (txh_p1 p1 w_q1 q1) c",

[Figure 27]

[Figure 28]

LocalRearrange

p1=P, q1=P, p2=P, q2=P, w_q1=W // (P ** 2) )

All-to-All

[Figure 29]

[Figure 30]

def sparse_sequence_parallel(x, grid_sizes, sparse_ratio, group): group_size = get_size(group) T, H, W = grid_sizes P = sparse_ratio P2 = P * P G = P2 // group_size B_local = x.shape[0] b = B_local // G

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- # 1. Local rearrangement x = orig_to_tsa(x, T, H // P, W // P, P) base, C = x.shape[1], x.shape[2]

- # 2. Distributed all-to-all transpose x = contiguous(x).view(group_size, G, G * b, base, C) recv = empty_like(x) all_to_all_single(recv, x, group=group)

- # 3. Local permutation recv = recv.view(group_size, G, G, b, base, C) recv = recv.permute(0, 2, 1, 3, 4, 5) x = contiguous(recv).view(P2 * G * b, base, C)

- # 4. Reverse local rearrangement x = tsa_to_orig(x, T, H // P, W // P, P) return x

- Figure 3 Sparse Sequence Parallel (SSP). Benefiting from the local equivalence of Skiparse-2D Attention, we can locally rearrange the subfigures sharded to each rank to identify the elements that remain on the same local rank after pattern switching. The target layout can then be obtained through All-toAll communication within the sparse parallel group, without requiring global All-Gather. Compared with Ulysses sequence parallelism (SP), SSP reduces the communication volume by 75% and requires only one communication operation per block.

without requiring a complex 2D mask. This means that Skiparse Attention natively supports FlashAttention kernels and avoids the compilation overhead of FlexAttention required by other sparse attention methods.

- 3.3 Sparse Sequence Parallel (SSP) Skiparse Attention is implemented by alternating Token-wise Sparse Attention (TSA) and Group-wise Sparse

Attention (GSA). In both TSA and GSA, each subsequence has a length of k1 of the original sequence and performs global attention. In the non-parallel setting, the transformation from the original pattern to either

sparse pattern, as well as the conversion between the two sparse patterns, can be implemented with a single rearrange operation, as shown in Alg. 1.

[Figure 37]

[Figure 38]

V-Tokens

[Figure 39]

[Figure 40]

[Figure 41]

Full Attn Block Full Attn Block Full Attn Block Full Attn Block

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Skiparse Attn Block

[Figure 51]

- T-Tokens

[Figure 52]

[Figure 53]

Timestep

[Figure 54]

[Figure 55]

MLP

OSP-Next Model Architecture

[Figure 56]

| | |
|---|---|
|Ulysses SP|[Figure 57]<br><br>[Figure 58]<br><br>All-to-All|
| | |
|Group-wise Sk|[Figure 59]<br><br>[Figure 60]<br><br>iparse Self Attn|
| | |
|[Figure 61]<br><br>[Figure 62]<br><br>Ulysses SP|All-to-All|
| | |
|[Figure 63]<br><br>[Figure 64]<br><br>Layer|Norm|
| | |
|Group-wise Skip|[Figure 65]<br><br>[Figure 66]<br><br>arse Cross Attn|
| | |
|Layer|[Figure 67]<br><br>[Figure 68]<br><br>Norm|
| | |
|[Figure 69]<br><br>[Figure 70]<br><br>FF|N|
| | |

[Figure 71]

[Figure 72]

Sparse SP All-to-All From Token-wise Skiparse Pattern To Group-wise Skiparse Pattern

[Figure 73]

[Figure 74]

Layer Norm

[Figure 75]

[Figure 76]

Ulysses SP All-to-All

[Figure 77]

[Figure 78]

Token-wise Skiparse Self Attn

[Figure 79]

[Figure 80]

Ulysses SP All-to-All

[Figure 81]

[Figure 82]

Layer Norm

[Figure 83]

[Figure 84]

Token-wise Skiparse Cross Attn

[Figure 85]

[Figure 86]

Layer Norm

[Figure 87]

[Figure 88]

FFN

[Figure 89]

[Figure 90]

Sparse SP All-to-All From Group-wise Skiparse Pattern To Token-wise Skiparse Pattern

[Figure 91]

[Figure 92]

Layer Norm

[Figure 93]

[Figure 94]

V-Tokens

Skiparse Attn Block Parallel Skiparse Attn

[Figure 95]

[Figure 96]

Parallel Token-wise Skiparse Self Attn Token-wise Skiparse Pattern

[Figure 97]

- … Rank 0

[Figure 98]

- … Rank 1

[Figure 99]

… Rank 3

[Figure 100]

- … Rank 2

[Figure 101]

- … Rank 0

[Figure 102]

- … Rank 1

[Figure 103]

… Rank 3

[Figure 104]

- … Rank 2

[Figure 105]

- … Rank 0

[Figure 106]

- … Rank 1

[Figure 107]

… Rank 3

[Figure 108]

- … Rank 2

Parallel Token-wise Skiparse Cross Attn

[Figure 109]

[Figure 110]

T-Tokens

Parallel Group-wise Skiparse Self Attn Group-wise Skiparse Pattern

[Figure 111]

- … Rank 0

[Figure 112]

- … Rank 1

[Figure 113]

… Rank 3

[Figure 114]

- … Rank 2

Parallel Group-wise Skiparse Cross Attn

[Figure 115]

[Figure 116]

T-Tokens

Figure 4 Model Architecture. OSP-Next adopts CrossDiT with a hybrid attention strategy. Specifically, we use full attention blocks in the first 2n layers and the last 2n layers to better capture fine-grained features, while using Skiparse Attention blocks in the remaining middle layers to maximize runtime efficiency. When computing self attention, the model processes all subsequences in parallel, reducing the computational complexity to k1 of full attention, where k denotes the sparse ratio. When SSP is enabled, each rank holds a subset of the subsequences. When Ulysses SP and SSP are enabled together, each rank obtains the local heads of the corresponding subsequences through an All-to-All communication before attention. Element-wise operations, such as LayerNorm and FFN, are unaffected by the parallelization strategy. During block forward, SSP changes the token pattern through a single All-to-All communication.

For the parallel implementation of Skiparse Rearrange, we employ Sparse Sequence Parallel (SSP), as specified by the sparse_sequence_parallel function in Alg. 1 and Fig. 3. Because Skiparse Rearrange generates subsequences of identical length, these subsequences can be sharded across different ranks for parallel computation, naturally maintaining load balance. More importantly, the local-equivalence property of Skiparse Rearrange enables switching between the two sparse patterns without an All-Gather operation. Instead, each rank applies a local rearrange and then uses All-to-All synchronization to obtain the portion of data assigned to that rank after the sparse-pattern transition. Compared with the naive All-Gather–Rearrange–Reshard implementation, this approach reduces the global communication volume from O(N2S) to O(NS), where N is the size of the communication group and S is the amount of data maintained by each rank. Furthermore, element-wise operations such as LayerNorm and FFN are independent of the sequence dimension, so they require no additional adaptation. Under this design, each block requires only one All-to-All operation during pattern switching. In contrast, Ulysses Sequence Parallel requires All-to-All communication for the query, key, value, and attention output; Sparse Sequence Parallel therefore reduces the communication volume by 75%.

Sparse Sequence Parallel is naturally compatible with Ulysses Sequence Parallel, because it only repartitions the enlarged batch dimension obtained after applying Skiparse Rearrange. When both parallelization strategies are enabled, the hidden states are partitioned according to the boundaries of the smallest repeatable units before they enter the main blocks, thereby preserving the properties of Skiparse Rearrange.

3.4 Model Structure

In Open-Sora Plan v1.3, we adopt a spindle-shaped architecture, where the first and last 2n layers use full attention Layers and the middle layers use Skiparse Attention Layers. In Open-Sora Plan v1.5, we use a

- U-shaped sparsity schedule, where the sparsity level increases from full attention Layers in the outer layers to large sparse attention layers in the inner layers. To balance performance and efficiency, OSP-Next adopts a spindle-shaped structure similar to that of Open-Sora Plan v1.3, as shown in Fig. 4.

…

[Figure 117]

[Figure 118]

Skiparse Attn Block

[Figure 119]

[Figure 120]

Full Attn Block Full Attn Block Full Attn Block Full Attn Block

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

##### 3.5 HiF8 Quantization

###### 3.5.1 HiF8 Data Format

HiF8 is a proprietary 8-bit floating-point format developed by Huawei. Unlike conventional FP8 variants, such as E4M3 and E5M2, which use a fixed exponent bit width, HiF8 introduces a variable-width Dot field that dynamically allocates the available bits between the exponent and mantissa. This design mitigates the

### HiFloat8：锥形精度详解

[Figure 127]

[Figure 128]

根据HiF8编码原理，可以画出左图所示的有效位-阶码分布图：

- • 有效位包含1.M的隐藏位1，因此比Mantissa位宽多1-bit
- • HiF8在数值绝对值靠近1的时候，精度高；远离1，精度逐渐降低。精度不存在跳变， 都是比特数为1的渐变
- • HiF8的综合二进制指数范围达到[-22, 15]，和FP16的[-24, 15]接近
- • HiF8编码了4个特殊值，不区分正负0，是一个表达完备的8-bit单数据浮点格式

###### 特殊值支持：

- Figure 5 Effective mantissa–exponent allocation of 8-bit floating-point formats. HiF8 adopts a tapered precision design tailored to the statistical distribution of neural network data. Unlike conventional FP8 formats such as E4M3 and E5M2, whose fixed exponent allocation imposes a trade-off between dynamic range and numerical precision, HiF8 concentrates higher precision in frequently occurring value regions, which typically follow a Gaussian-like distribution in model weights and activations. This enables a better match between numerical representation and deep learning data characteristics.

HiF8精度区间划分：

- 1. 高精度区间：𝐸 = [−3,3]，7个二进制指数，4-bit有效位
- 2. 中精度区间：𝐸 = ±[4,7]，8个二进制指数，3-bit有效位
- 3. 低精度区间：𝐸 = ±[8,15]，16个二进制指数，2-bit有效位
- 4. 极小值区间：𝐸 = [−22,−16]，7个二进制指数，1-bit有效位

- 1. S = 0, D = DML, M = 3’b0，X = 𝑍ero（HiF8：8’b00000000)
- 2. S = 1, D = DML, M = 3’b0，X = NaN（HiF8：8’b10000000)
- 3. D = 4, Es = 4’b0111 = 15, M = 1’b1时，X = ± Inf（HiF8：8’bX1101111)

https://gitcode.com/cann

如何用好HiF8的锥形精度，是能否发挥其价值的关键！

inherent trade-off between dynamic range and representational precision. Its three core properties are as follows.

Wide Dynamic Range. Conventional FP8-E4M3 fixes the exponent to 4 bits with a bias of 7, resulting in a limited exponent span and frequent overflow or underflow in the deep attention layers of DiT-based generative models. By unifying normal and denormal encodings within a coherent scheme, HiF8 extends the effective exponent range to [−22,,15], covering 38 distinct exponent values and approaching the 40-exponent coverage of FP16, [−24,,15]. This dynamic range allows HiF8 to quantize activations and gradients that span several orders of magnitude without fine-grained per-token or per-block rescaling, thereby reducing the engineering complexity and memory overhead of low-precision training pipelines.

Tapered Precision. Conventional FP8 formats face a fundamental trade-off between precision and dynamic range: E4M3 sacrifices dynamic range for higher mantissa precision, whereas E5M2 makes the opposite compromise. Neither configuration fully satisfies both requirements of neural-network training. As shown in Fig. 5, HiF8 addresses this trade-off by progressively reducing the mantissa bit width as the magnitude of a value increases: it preserves a 3-bit mantissa within the central exponent range [−3,,3] and gradually reduces it to a 1-bit mantissa at the extremes of the representable range. This tapered precision profile aligns well with the approximately Gaussian distribution of neural-network weights and activations, concentrating encoding capacity in the high-density region of the numerical distribution while reducing precision only for low-frequency values.

Redundancy-Free Encoding. The exponent field of HiF8 uses sign-magnitude encoding with an implicit leading 1 bit in the magnitude. This design ensures that the representable exponent ranges of fields with different widths are strictly non-overlapping and that the entire exponent space [−22,,15] is covered without gaps or duplicates. In contrast, conventional FP8 formats inherited from IEEE 754 include redundant code points, such as dual representations of ±0 and multiple NaN encodings, which waste code points within the already constrained 8-bit budget. In HiF8, every code point maps to a unique numerical value, achieving full encoding utilization and providing a stronger foundation for quantization accuracy under tight bit-width constraints.

Rollout Sampling Advantage Estimate

###### Model Input

[Figure 130]

Stochastic SDE Update

Deterministic ODE Update

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

| | | |
|---|---|---|
|ideo (OSP|Po<br><br>-N|[Figure 137]<br><br>[Figure 138]<br><br>licy ext)|
| | | |

[Figure 139]

[Figure 140]

Visual Quality

Motion Quality

Text Prompt

[Figure 141]

…

[Figure 142]

###### V

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

RLGradientFlow

[Figure 147]

[Figure 148]

[Figure 149]

Text Alignment

Overall Score

Noise Sample

Generated Videos

Early Optimization Steps

Late Deteministic Steps

Denoising Trajectory (Timesteps T → 0)

Policy Gradient Loss / Update Signal

- Figure 6 Overview of the pipeline for OSP-Next RL post-training. The policy performs denoising with a hybrid sampling strategy, using stochastic SDE updates in the early steps and deterministic ODE updates in the late steps. The generated videos are then evaluated by VideoAlign along three dimensions—visual quality, motion quality, and text alignment—to produce the training signal for policy optimization.

###### 3.5.2 Quantization Scheme

Mixed-precision training requires a scaling factor to map high-precision tensors, such as BF16 and FP32 tensors, into the representable range of HiF8. Let Amax = max|X| denote the maximum absolute value of a data block X, and let HiF8max denote the largest representable value in the HiF8 format. In this work, HiF8max is set to 15 for forward computation and 224 for backward computation, respectively, let ϵ denote a small positive constant for numerical stability. The scaling factor and quantization step are defined as follows:

HiF8max Amax + ϵ

(1)

Scale =

Xscaled = X × Scale. (2)

Current Scaling. We adopt a current scaling strategy. At each training iteration, the full input tensor X is traversed to compute the real-time Amax, from which Scale is derived and immediately applied before downstream computation, such as MatMul. This strategy ensures that the scaling factor remains consistent with the actual data distribution of the current step, thereby eliminating the risk of overflow caused by stale statistics.

Per-Tensor Quantization Granularity. Thanks to the wide dynamic range of HiF8, a single per-tensor scale is sufficient for the entire tensor, without requiring the per-block granularity, such as 128!×!128 tiles, that is typically needed for FP8 formats. Per-tensor quantization substantially reduces the storage and memorybandwidth overhead associated with scale metadata, simplifies the control logic, and improves end-to-end training throughput.

##### 3.6 Mix-GRPO For the Sparse-Quant Model

###### 3.6.1 Mix-GRPO

In the post-training stage of the sparse-quant model, we employ a hybrid ODE-SDE denoising strategy following Mix-GRPO [12], as shown in Fig. 6. Although Skiparse-2D Attention and HiF8 quantization substantially improve efficiency, converting a full attention model into a sparse attention model can still introduce a quality gap. We use Mix-GRPO to further align the fine-tuned sparse model with video-generation preferences while preserving the efficiency advantages of OSP-Next.

Conventional flow-based GRPO methods, such as FlowGRPO [16], formulate the full denoising trajectory

- as a Markov decision process (MDP). During rollout, SDE sampling is applied at every denoising step to

introduce stochastic exploration, followed by policy optimization over the complete state-action sequence. This design is effective for preference alignment but expensive for video generation, because each update requires trajectory-level sampling and policy-ratio computation across all denoising steps. It also mixes gradients from different stages of the generation process: early steps mainly determine global layout and motion structure, whereas late steps focus on local texture and visual details. Jointly optimizing all steps can therefore lead to unfocused updates and unnecessary computation.

Mix-GRPO addresses this issue by combining ODE and SDE sampling. This design leverages an inherent property of diffusion models, namely the ability to introduce more randomness during the early denoising stages. Accordingly, we apply SDE sampling to the initial denoising steps and switch to deterministic ODE sampling in the later stages. In this way, Mix-GRPO shortens the effective MDP horizon, confines stochastic exploration to the optimized timesteps, reduces the number of denoising steps requiring policy-gradient updates, and improves the training efficiency of the overall RL pipeline.

For a deterministic reverse probability flow ODE, the denoising dynamics can be written as:

dxt dt

= f(xt,t) −

- 1

- 2

g2(t)∇xt

log qt(xt). (3)

According to the Fokker-Planck equation, this ODE has an equivalent probability flow SDE that maintains the same marginal distribution at each time step:

dxt dt

dw dt

= f(xt,t) − g2(t)∇xt

log qt(xt) + g(t)

. (4)

Mix-GRPO mixes the two forms by using SDE sampling on the optimized timesteps and ODE sampling otherwise:

f(xt,t) − g2(t)st(xt) dt + g(t)dw, if t ∈ S, f(xt,t) − 12g2(t)st(xt) dt, otherwise.

(5)

dxt =

where we define the score function as

st ≜ ∇xt

log qt(xt). (6)

Compared with FlowGRPO, this mixed SDE-ODE strategy provides two practical advantages for OSP-Next. First, it reduces optimization overhead because only a subset of timesteps contributes to the GRPO loss. Second, it excludes the ODE portions from the RL objective, thereby reducing unnecessary stochastic sampling and lowering rollout cost. These properties are especially important for video models, where each trajectory contains long spatio-temporal sequences and reward evaluation is substantially more expensive than in image generation.

For OSP-Next, Mix-GRPO therefore serves as an efficient preference-alignment mechanism for the sparse-quant model. It helps recover the performance drop caused by sparse fine-tuning from the pretrained full attention model and by quantization, while keeping the training pipeline compatible with Skiparse-2D Attention, Sparse Sequence Parallelism, and HiF8 simulation. Empirically, this post-training stage provides stable gains for sparse video generation and brings OSP-Next closer to the full attention baseline.

###### 3.6.2 Reward Model

We adopt VideoAlign as the reward model. Since VideoAlign [17] is natively trained on videos, it reduces the evaluation gap in video scoring compared with image-trained models such as CLIP [23]. In our implementation, VideoAlign provides reward dimensions including visual quality, motion quality, text alignment, and an overall score. We use the overall VideoAlign score as the scalar reward for Mix-GRPO optimization.

- Table 1 VBench results. Compared with the Wan2.1 baseline, OSP-Next achieves a higher total score, demonstrating the effectiveness of the OSP-Next architecture. All metrics reported below are measured in percentages (%). The abbreviations are defined as follows: SC: Subject Consistency; BC: Background Consistency; TF: Temporal Flickering; MS: Motion Smoothness; DD: Dynamic Degree; AQ: Aesthetic Quality; IQ: Imaging Quality; ObjC: Object Class; MO: Multiple Objects; HA: Human Action; SR: Spatial Relationship; AS: Appearance Style; TS: Temporal Style; OverC: Overall Consistency.

Model Total Quality Semantic SC BC TF MS DD AQ IQ ObjC MO HA Color SR Scene AS TS OverC

Wan2.1-T2V-14B 83.69 85.59 76.11 97.52 98.09 99.46 98.30 65.46 66.07 69.43 86.28 69.58 95.40 88.59 75.39 45.75 22.64 23.19 25.91 OSP-Next 83.73 85.19 77.86 96.30 96.95 98.49 98.09 70.83 66.91 69.64 90.10 77.48 97.20 87.05 73.65 49.39 21.93 24.05 26.34 OSP-Next-HiF8 83.29 84.57 78.21 96.61 97.10 98.40 98.26 59.44 66.44 70.85 90.87 77.70 96.80 90.11 78.36 48.20 20.96 24.01 26.26

#### 4 Experiment

##### 4.1 Training Setup

Dataset. We use an internally collected dataset of high-quality videos and apply the same filtering strategy as described in Open-Sora Plan. We discard videos that are overly static or have excessively low resolution, and train with a fixed duration of 81 frames and a fixed resolution of 720 × 1280.

Training. The weights of OSP-Next are initialized from the Wan2.1-T2V-14B [28] model. We use a configuration with eight full attention blocks, with four placed at the beginning and four at the end, while the middle 32 layers use Skiparse Attention blocks. Skiparse-2D Attention is adopted with a sparse ratio of 2. During training, we use SSP with a parallelism degree of 4. We adopt Fully Sharded Data Parallel (FSDP) as the training strategy and use FSDP-EMA during training, with the EMA decay set to 0.999. We train the model with the AdamW optimizer, using a learning rate of 0.00002, a weight decay of 0.001, and a global batch size of 24. The SFT weights are obtained after 6000 training steps. For the Mix-GRPO post-training stage, we adopt an SDE-ODE hybrid sampling strategy with a total of 25 sampling steps, where the first 10 steps use SDE and the remaining 15 steps use ODE. We apply LoRA with a rank of 32 and an alpha of 64 for parameter-efficient fine-tuning. The rollout group size is set to 4. The learning rate is set to 0.0001, and the KL regularization coefficient β is set to 0.004. OSP-Next-HiF8 is implemented with HiF8 simulation operators and is also initialized from the Wan2.1-T2V-14B model. Its training configuration remains the same as that of OSP-Next.

Inference. During inference, we use a Discrete Euler sampler with 50 sampling steps, a shift value of 7.0, and classifier-free guidance set to 5.0.

##### 4.2 Main Results

###### 4.2.1 Vbench Scores

We adopt VBench[9] as the quantitative metric for assessing model generation quality. As reported in Tab. 1, OSP-Next obtains a total VBench score of 83.73%, surpassing the Wan2.1 baseline and demonstrating the effectiveness of the OSP-Next architecture. Moreover, OSP-Next-HiF8, which uses 8-bit precision, achieves a total score of 83.29%, keeping the gap from the baseline below 0.4%. These results suggest that HiF8 precision effectively preserves the capability of the pretrained model while validating the feasibility of joint training between HiF8 and the sparse model.

###### 4.2.2 Visual Results

We compare the visual generation quality of the OSP-Next series with the Wan2.1 baseline, as shown in Fig. 7. Although OSP-Next is a sparse Attention model, its visual quality shows no noticeable difference from the baseline. In addition, owing to the dynamic precision of HiF8 quantization, OSP-Next-HiF8 shows no clear degradation in generation quality compared with OSP-Next. These results demonstrate the effectiveness of joint training with sparse-model fine-tuning and 8-bit quantization.

[Figure 150]

###### Wan2.1OSP-NextOSP-Next-HiF8Wan2.1OSP-NextOSP-Next-HiF8

[Figure 151]

[Figure 152]

A handheld 35mm camera holds an extreme close-up on a gray-haired, bearded man in his sixties, seated at a weathered café table in Paris. His professorial appearance is defined by a textured wool coat, a button-down shirt, a brown beret, and wire-rimmed glasses that catch the warm, golden light of the late afternoon. The shallow depth of field renders the bustling Parisian street behind him a soft, painterly blur of limestone buildings and passing figures. Volumetric light filters through the sycamore leaves, casting a gentle chiaroscuro across his deeply lined face. He sits almost motionless, his eyes subtly tracking the out-of-focus pedestrians strolling just beyond the frame, his stillness contrasting with the city’s quiet rhythm. The atmosphere is one of profound, melancholic contemplation, as he ponders the history of the universe. Finally, the faintest closed-mouth smile breaks the stillness, a quiet, intimate revelation as if he has just deciphered the mystery of life itself.

[Figure 153]

[Figure 154]

[Figure 155]

A sun-dappled white and orange tabby cat darts joyfully through an overgrown garden labyrinth, chasing an unseen flicker of movement. The handheld camera hugs the ground at the cat’s eye level, creating an intimate, immersive tracking shot that weaves through the narrow dirt path. Wide, luminous amber eyes sparkle with playful excitement as the cat scans a chaotic tapestry of ferns, wildflowers, and tangled branches. Golden hour daylight fractures through the dense canopy of leaves above, casting a warm chiaroscuro of scattered beams and soft shadows that set the cat’s orange fur ablaze with honeyed contrast. The atmosphere is serene yet adventurous, alive with the rustle of foliage. Rendered with a sharp, shallow depth of field that melts the chaotic greenery into a smooth, painterly bokeh, the image is steeped in cinematic warm earth tones and a delicate, nostalgic film grain texture that enhances the raw, organic beauty of the scene.

- Figure 7 Generation quality of the OSP-Next series with the Wan2.1 baseline. By exploiting video locality through Skiparse-2D Attention and benefiting from the hybrid attention architecture, OSP-Next achieves generation quality that is highly comparable to the Wan2.1 baseline. In addition, due to the dynamic precision of the HiF8 quantization format, OSP-Next-HiF8 shows almost no quality drop compared with OSP-Next.

###### 4.2.3 Inference Speed

To evaluate the acceleration effect of OSP-Next on different hardware platforms, we compare its inference speed on NVIDIA H200, Ascend 910C, and Ascend 950PR, as shown in Tab. 2.

On a single GPU/NPU, OSP-Next achieves more than 1.53× speedup over Wan2.1 under both with padding and without padding settings. In the multi-GPU/NPU evaluation, the Wan2.1 baseline consistently uses Ulysses SP. For OSP-Next, the Skiparse Attention blocks use only SSP under the 2-GPU/NPU and 4-GPU/NPU settings, and use both SSP and SP under the 8-GPU/NPU setting, while the full attention blocks consistently use Ulysses SP. OSP-Next also achieves stable acceleration in the multi-GPU/NPU evaluation. Since the without padding setting requires no attention mask, it yields larger speedups. In addition, OSP-Next-HiF8 achieves up to 2.27× speedup on Ascend 950PR, demonstrating the efficiency advantage of combining HiF8 precision with Skiparse-2D Attention.

##### 4.3 Ablation Study

- Table 2 Inference speed. Since OSP-Next adopts an any resolution strategy, we evaluate two settings with 81 frames: 720P, where padding and attention masks are present, and 768P, where padding and attention masks are absent. OSP-Next achieves more than 1.42× speedup on different hardware models under both settings, with better performance when padding is absent. The OSP-Next-HiF8 model achieves a 2.27× speedup on Ascend 950PR, demonstrating the speed advantage of HiF8 quantization. Here, SDPA denotes torch.scaled_dot_product_attention. Since Ascend 950PR resources are currently in limited supply, the results for this hardware are restricted to one-NPU only.

(b) (T, H, W) = (81, 768, 1280). Without padding. Model 1 GPU/NPU 2 GPU/NPU 4 GPU/NPU 8 GPU/NPU Nvidia H200, torch.compile enabled, FlashAttention3

(a) (T, H, W) = (81, 720, 1280). With padding. Model 1 GPU/NPU 2 GPU/NPU 4 GPU/NPU 8 GPU/NPU Nvidia H200, torch.compile enabled, FlashAttention3

Wan2.1 Baseline 946s 501s 254s 129s OSP-Next 620s(1.53×) 335s(1.50×) 171s(1.49×) 91s(1.42×)

Wan2.1 Baseline 1061s 561s 283s 140s OSP-Next 645s(1.64×) 342s(1.64×) 173s(1.64×) 92s(1.52×)

Ascend 910C, torch.compile disabled, SDPA

Ascend 910C, torch.compile disabled, SDPA

Wan2.1 Baseline 2353s 1197s 523s / OSP-Next 1362s(1.73×)683s(1.75×) 367s(1.43×) /

Wan2.1 Baseline 2372s 1168s 587s / OSP-Next 1353s(1.75×)673s(1.74×) 354s(1.66×) /

Ascend 950PR, torch.compile disabled, SDPA

Ascend 950PR, torch.compile disabled, SDPA

Wan2.1 Baseline 1219s / / / OSP-Next 957s(1.27×) / / / OSP-Next-HiF8 723s(1.69×) / / /

Wan2.1 Baseline 1338s / / / OSP-Next 762s(1.76×) / / / OSP-Next-HiF8 590s(2.27×) / / /

All Skiparse Attn Blocks

Full Attn Blocks = 4

Full Attn Blocks = 12

Full Attn Blocks = 16

Full Attn Blocks = 8

Wan2.1 Baseline

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Skiparse-2DSkiparse-1D

Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach.The crashing blue waters create white-tipped waves,while the golden light of the setting sun illuminates the rocky shore.

- PSNR = 9.82dB
- PSNR = 10.15dB

PSNR = 13.79dB

PSNR = 17.32dB

PSNR = 17.90dB

RSNR = 16.66dB

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

PSNR = 17.46dB

PSNR = 17.88dB

PSNR = 18.07dB

PSNR = 17.86dB

- Figure 8 Initialization Gap. OSP-Next adopts Wan2.1-T2V-14B as the initialization weights, whereas an initialization gap remains between the pretrained full attention model and the sparse attention model. To estimate the magnitude of this gap, we compare the zero-shot inference results of the initialized sparse attention model against the baseline. With an increasing number of full attention blocks, the zero-shot inference results of the sparse attention model progressively approach the baseline. Across all settings, Skiparse-2D Attention achieves generation quality that is consistently closer to the baseline than Skiparse-1D Attention. Finally, to balance generation efficiency and generation quality, we choose 8 full attention blocks as the configuration of OSP-Next.

###### 4.3.1 Skiparse-2D Initialization

OSP-Next is initialized from Wan2.1, which adopts full attention, and therefore faces a transfer gap when adapted to a sparse Attention architecture. For the spindle-shaped architecture used in OSP-Next, we vary the number of full attention blocks and evaluate the corresponding zero-shot generation quality, thereby quantifying the transfer gap between full attention and sparse Attention.

As shown in Fig. 8, when all blocks are replaced with Skiparse Attention blocks, the zero-shot inference results of the model fail to generate valid videos. This indicates that sparse Attention with predefined rules, such as Skiparse-2D Attention, has difficulty fully inheriting the capability of a pretrained full attention model in the zero-shot setting. OSP-Next alleviates this issue by adopting a spindle-shaped architecture with full attention blocks on both sides. As the number of full attention blocks increases, the zero-shot inference results of the sparse model gradually approach those of the baseline. In addition, the generation quality of the model based on Skiparse-1D Attention lags behind that of the model based on Skiparse-2D Attention, demonstrating the advantage of Skiparse-2D Attention in handling image and video modalities. To achieve a balance between performance and efficiency, OSP-Next uses 8 full attention blocks in its training configuration, with 4 placed

- at the beginning and 4 placed at the end.

[Figure 167]

BF16SFTHiF8SFTBF16RLHiF8RL

[Figure 168]

a bench on the left of a truck, front view

[Figure 169]

[Figure 170]

In a still frame, a stop sign

- Figure 9 Impact of reinforcement learning on the Sparse Attention model. Adapting a pretrained full attention model to a sparse attention model often results in a degradation of model performance. To mitigate this issue, we use Mix-GRPO for post-training OSP-Next. After post-training, OSP-Next exhibits notable gains in spatial relation understanding, video-text alignment, and related dimensions, indicating that reinforcement learning is effective for the sparse attention model.

###### 4.3.2 Reinforce Learning for Sparse Model

Although Skiparse-2D Attention better aligns with the locality of the visual modality and adopts a hybrid attention architecture to better match the capability of the full attention model, the model supervised fine-tuned only on text-video pairs still shows degraded performance compared with the pretrained model. Some prior studies attempt to use the pretrained model as a teacher to align the outputs of the sparse model. However, this approach assumes that the output pattern of the sparse model should be fully consistent with that of the pretrained model, which limits the performance upper bound of the sparse model. Moreover, output-level alignment remains insufficient for capturing the differences between the sparse model and the pretrained model in text-video alignment, aesthetics, and other visual quality-related aspects.

In OSP-Next, we instead use reinforcement learning to improve the capability of the sparse model. As shown in Fig. 9, the model optimized with the Mix-GRPO strategy achieves substantial improvements in spatial relationships, text-video alignment, aesthetics. This demonstrates the effectiveness of reinforcement learning in the post-training stage of the sparse model.

#### 5 Conclusion

In this paper, we introduce OSP-Next, a sparse model that employs Skiparse-2D Attention to better leverage the locality inherent in image and video modalities. Motivated by the properties of Skiparse-2D Attention, we further develop Sparse Sequence Parallelism (SSP), a sequence parallelism strategy that is natively aligned with this sparse attention pattern. Compared with Ulysses Sequence Parallelism (SP), SSP reduces communication volume by 75% and can be used jointly with Ulysses SP. We additionally study the feasibility of joint training with 8-bit quantization for the sparse model, and the effectiveness of reinforcement learning during sparse-model post-training. Experiments demonstrate that OSP-Next maintains comparable VBench scores and generation quality to the pretrained full attention model, while consistently providing over 1.42× speedup across various settings. Moreover, the 8-bit quantized OSP-Next-HiF8 achieves a 2.27× speedup over the pretrained model.

#### References

- [1] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.
- [2] Pengtao Chen, Xianfang Zeng, Maosen Zhao, Mingzhu Shen, Wei Cheng, Gang Yu, and Tao Chen. Sparse-vdit: Unleashing the power of sparse attention to accelerate video diffusion transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 2957–2965, 2026.
- [3] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, volume 2024, pages 35549–35562, 2024.
- [4] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.
- [5] Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496, 2(3):4, 2024.
- [6] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [7] Jiarui Fang and Shangchun Zhao. Usp: A unified sequence parallelism approach for long context generative ai. arXiv preprint arXiv:2405.07719, 2024.
- [8] Yunyang Ge, Xinhua Cheng, Chengshu Zhao, Xianyi He, Shenghai Yuan, Bin Lin, Bin Zhu, and Li Yuan. Flashi2v: Fourier-guided latent shifting prevents conditional image leakage in image-to-video generation. arXiv preprint arXiv:2509.25187, 2025.
- [9] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [10] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.
- [11] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [12] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.
- [13] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025.
- [14] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [15] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ringattention with blockwise transformers for near-infinite context. In International Conference on Learning Representations, volume 2024, pages 3992–4008, 2024.
- [16] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026.
- [17] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. Advances in Neural Information Processing Systems, 38:82155–82192, 2026.
- [18] Yuanyong Luo, Zhongxing Zhang, Richard Wu, Hu Liu, Ying Jin, Kai Zheng, Minmin Wang, Zhanying He, Guipeng Hu, Luyao Chen, et al. Ascend hifloat8 format for deep learning. arXiv preprint arXiv:2409.16626, 2024.

- [19] Xin Ma, Yaohui Wang, Xinyuan Chen, Gengyun Jia, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [20] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, et al. Fp8 formats for deep learning. arXiv preprint arXiv:2209.05433, 2022.
- [21] Asit Mishra, Dusan Stosic, Simon Layton, and Paulius Micikevicius. Recipes for pre-training llms with mxfp8. arXiv preprint arXiv:2506.08027, 2025.
- [22] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [25] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
- [26] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024.
- [27] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.
- [28] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [29] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.
- [30] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.
- [31] Yifei Xia, Suhan Ling, Fangcheng Fu, Yujie Wang, Huixia Li, Xuefeng Xiao, and Bin Cui. Training-free and adaptive sparse attention for efficient long video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15982–15993, 2025.
- [32] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.
- [33] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. Advances in Neural Information Processing Systems, 38:96965–96991, 2026.
- [34] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pages 83048–83077, 2025.
- [35] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.
- [36] Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, et al. Sla: Beyond sparsity in diffusion transformers via fine-tunable sparse-linear attention. arXiv preprint arXiv:2509.24006, 2025.

- [37] Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025.
- [38] Jintao Zhang, Pengle Zhang, Jun Zhu, Jianfei Chen, et al. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In International Conference on Learning Representations, volume 2025, pages 71566–71585, 2025.
- [39] Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Turbodiffusion: Accelerating video diffusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025.
- [40] Jintao Zhang, Haoxu Wang, Kai Jiang, Kaiwen Zheng, Youhe Jiang, Ion Stoica, Jianfei Chen, Jun Zhu, and Joseph E Gonzalez. Sla2: Sparse-linear attention with learnable routing and qat. arXiv preprint arXiv:2602.12675, 2026.
- [41] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025.
- [42] Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Faster video diffusion with trainable sparse attention. Advances in Neural Information Processing Systems, 38: 152509–152534, 2026.
- [43] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.
- [44] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.
- [45] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [46] Zangwei Zheng, Xiangyu Peng, Yuxuan Lou, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, et al. Open-sora 2.0: Training a commercial-level video generation model in $200 k. arXiv preprint arXiv:2503.09642, 2025.

[Figure 171]

###### Wan2.1OSP-NextOSP-Next-HiF8Wan2.1OSP-NextOSP-Next-HiF8

[Figure 172]

[Figure 173]

A low-angle tracking shot glides through knee-deep powder, capturing several giant woolly mammoths as they tread majestically across a snowy meadow. Their immense, long woolly fur ripples and lifts gently in the biting alpine wind, each strand textured with crystalline frost. In the distance, a forest of snow-covered evergreens frames dramatic, snow-capped peaks that pierce a pale winter sky. Mid-afternoon sunlight filters through wispy, drifting clouds, casting a warm, honeyed glow that ignites the beasts’ dark silhouettes with a soft rim light. The shallow depth of field isolates their colossal tusks and heavy, breathing forms against a dreamy, bokeh-filled background of sparkling snow. Fine ice particles dance in the crisp air, catching the luminous backlight and creating an ethereal atmosphere. The stunning, intimate photography emphasizes the rough texture of matted fur and the gentle steam of warm breath, rendering a serene yet powerful portrait of ancient giants in a pristine, glacial wilderness.

[Figure 174]

[Figure 175]

[Figure 176]

A lone 30-year-old space man strides across an endless salt desert under a vast, electric-blue sky, his red wool knitted motorcycle helmet a striking anomaly against the stark white crust. Shot on rich 35mm film with vivid, saturated colors, the trailer unfolds with a shallow depth of field that isolates our unconventional hero. A low-angle dolly shot glides alongside his weathered boots, capturing the crisp texture of salt crystals crunching beneath each step. Golden hour sunlight bathes the scene in a warm, honeyed glow, casting long, distorted shadows that stretch toward the horizon. The camera cranes upward in a sweeping reveal, emphasizing his profound isolation within the shimmering, mirror-like salt flats. Soft lens flares kiss the edge of the frame, while the wool helmet’s intricate knit pattern catches the backlight, glowing like a crimson beacon. A tangible atmosphere of serene solitude and surreal adventure permeates every grainy, tactile frame, evoking a nostalgic, otherworldly odyssey.

###### Figure 10 More comparisons between OSP-Next and the Wan2.1 baseline.

[Figure 177]

###### Wan2.1OSP-NextOSP-Next-HiF8Wan2.1OSP-NextOSP-Next-HiF8

[Figure 178]

[Figure 179]

A litter of golden retriever puppies bursts from a pristine snowbank in a sun-drenched alpine meadow. A low-angle tracking shot glides across the glittering surface as fluffy heads pop up one by one, muzzles and whiskers dusted with crystalline powder. Their fur, a warm honey-gold, contrasts vividly against the cool, blue-tinged white of the snow. Soft backlighting from the golden hour sun creates luminous halos around their tiny, wriggling bodies, making the airborne snowflakes sparkle like diamond dust. The atmosphere is pure, joyful chaos—tails wagging furiously beneath the surface, sending up little explosions of powder. Shallow depth of field isolates a single puppy in the foreground, its dark, curious eyes and snow-caked nose rendered in exquisite detail, while others playfully tumble out of focus behind. The texture is palpable: the cold, feathery snow, the downy softness of puppy fur, the warm breath condensing in quick, happy puffs. The color palette is a harmonious blend of creamy golds, crisp whites, and subtle icy blues, evoking an irresistible feeling of winter wonder and innocent delight.

[Figure 180]

[Figure 181]

[Figure 182]

A cream and sable corgi, sporting sleek jet-black sunglasses, trots confidently along a pristine tropical beach. A low-angle tracking shot follows close beside its sandy paws, the camera gliding smoothly on a dolly. Crystal-clear turquoise water laps gently at the shore, its surface glinting with a million dancing highlights under the radiant golden hour sun. The light backlights the dog's fluffy silhouette, creating a luminous rim of fire around its fur. In the background, lush emerald palm fronds sway lazily against a sky of soft pastel blues and warm corals. The atmosphere is pure, carefree bliss and serene island tranquility. The color palette is rich and saturated, dominated by the vivid blue of the sea, the warm gold of the sun, and the bright green of the tropical foliage. The shallow depth of field keeps the corgi and its immediate sandy path in sharp, textured focus, while the distant palms and sky melt into a dreamy, soft-focus bokeh, enhancing the idyllic, cinematic texture of this perfect island moment.

###### Figure 11 More comparisons between OSP-Next and the Wan2.1 baseline.

[Figure 183]

###### Wan2.1OSP-NextOSP-Next-HiF8Wan2.1OSP-NextOSP-Next-HiF8

[Figure 184]

[Figure 185]

A solitary panda sits on a moss-covered boulder deep in a misty bamboo forest, cradling a worn acoustic guitar in its massive paws. Soft golden-hour light filters through the dense canopy, casting dappled beams that catch drifting dust motes and illuminate the panda’s glossy black-and-white fur texture. A slow, gentle tracking shot circles the scene at close range, using a shallow depth of field to blur the towering emerald bamboo stalks in the background while keeping the panda’s focused, serene expression razor-sharp. The atmosphere is tranquil and intimate, filled with earthy forest scents and a hushed reverence. One claw gently plucks a bass string, sending a low, resonant vibration through the humid air. The color palette is rich and organic, dominated by deep jade greens, creamy whites, warm amber highlights, and the dark honeyed wood of the instrument, with subtle volumetric light rays piercing the shadows. The overall mood is meditative and soulful, capturing a fleeting moment of pure harmony between creature and nature.

[Figure 186]

[Figure 187]

[Figure 188]

A bespectacled rabbit sits in a worn leather armchair, its wire-rimmed glasses perched low on its twitching nose. The rabbit’s paws, surprisingly dexterous, hold a broadsheet newspaper wide open, the inky headlines sharp against the cream paper. It’s a close-up, shallow depth-of-field shot, with the background melting into a soft, warm blur of a book-lined study. A low, raking key light from a single desk lamp casts a pool of golden illumination across the scene, catching the fine texture of the rabbit’s fur and the crisp edges of the newsprint. Dust motes dance lazily in the volumetric beam of light, suspended in the quiet, contemplative atmosphere. The color palette is rich and nostalgic, dominated by deep mahogany browns, brass highlights, and the warm amber of the lamplight, evoking a feeling of hushed, intellectual solitude. The camera executes an almost imperceptible, creeping dolly zoom toward the rabbit’s concentrated eye, magnified through the spectacle lens, as it scans the text with profound, scholarly focus.

###### Figure 12 More comparisons between OSP-Next and the Wan2.1 baseline.

[Figure 189]

###### Wan2.1OSP-NextOSP-Next-HiF8Wan2.1OSP-NextOSP-Next-HiF8

[Figure 190]

[Figure 191]

A towering humanoid figure of craggy, moss-flecked stone strides through an ancient forest, his every step a low seismic rumble. A full-body shot tracks him laterally with a slow dolly move, keeping his immense form centered against the towering redwoods. Golden hour sunlight pierces the canopy in slanted, volumetric beams, wrapping his granite shoulders in a molten amber rim light. Dust motes and drifting pollen dance in the shafts of light, catching on the rough, porous texture of his basalt skin and the emerald velvet of the moss. The forest floor, carpeted in damp ferns and fallen pine needles, sinks slightly beneath his weight. A shallow depth of field blurs the distant trunks into soft, dark verticals, while the air holds a cool, earthy stillness tinged with petrichor. Shadows pool deep and blue in the crevices of his jointed stone limbs, contrasting with the warm, glowing highlights that trace his mineral veins. The mood is one of solemn, mythic pilgrimage, a living monolith moving with deliberate, ancient purpose through the cathedral-like silence.

[Figure 192]

[Figure 193]

[Figure 194]

A young woman stands bathed in golden hour sunlight, her face softly illuminated as she gazes directly into the lens. A gentle breeze lifts strands of her hair, adding a tactile, airy texture to the scene. With a playful, tender smile, she brings her fingertips to her lips and blows a slow, deliberate kiss toward the camera. The shallow depth of field isolates her from a dreamy, out-of-focus garden background, where dappled light filters through leafy trees. Warm amber and honey tones dominate the palette, accented by the creamy blush of her skin and the subtle pastel of her dress. The atmosphere is nostalgic and intimate, captured with a subtle film grain and a slight halation glow on the highlights. Shot with a static medium close-up, the moment feels suspended in time, as floating dust motes catch the backlight and drift lazily through the frame, enhancing the romantic, cinematic mood.

###### Figure 13 More comparisons between OSP-Next and the Wan2.1 baseline.

[Figure 195]

###### Wan2.1OSP-NextOSP-Next-HiF8Wan2.1OSP-NextOSP-Next-HiF8

[Figure 196]

[Figure 197]

A majestic brown bear dons a flowing scarlet superhero cape that ripples with heroic energy as it soars through a vast cerulean sky. The camera follows in a dynamic FPV tracking shot, weaving between towering art deco skyscrapers that gleam with polished chrome and glass. Warm golden hour sunlight pierces through the urban canyon, casting dramatic volumetric rays and a honeyed backlight that illuminates every tuft of the bear’s thick, textured fur. The bustling city below is a miniature tapestry of moving traffic and glinting windows, captured with a shallow depth of field that keeps the ursine hero in razor-sharp focus against a softly blurred metropolitan backdrop. Wispy cirrus clouds streak past, their edges kissed by a peach and lavender gradient as the sun dips toward the horizon. The atmosphere is one of triumphant wonder and playful majesty, with a rich, saturated color palette of deep indigos, burnished golds, and vibrant crimson. The cape snaps crisply in the wind, its velvety fabric catching the light, while the bear’s expression is one of pure, joyful determination.

[Figure 198]

[Figure 199]

[Figure 200]

A lone astronaut in a weathered spacesuit kneels beside a crackling campfire on the lunar surface, tending a sizzling skillet. A slow dolly shot circles the scene, the camera gliding at ground level to capture the surreal contrast of glowing embers against the ashen, cratered regolith. The lighting is a dramatic chiaroscuro, with the fire’s warm orange and amber glow carving sharp, dancing shadows across the astronaut’s visor, which reflects the star-filled void of space. Beyond the intimate pool of light, the landscape stretches into an endless, cold monochrome of silver-grey dust and sharp basaltic rocks under a pitch-black sky. The atmosphere is one of profound solitude and quiet wonder, a lone campfire serving as a fragile beacon of life. Smoke rises in strange, curling plumes in the low gravity, catching the firelight. The texture is hyper-realistic, with the rough, porous moon dust, the smooth, heat-blistered metal of the skillet, and the bubbling, savory food rendered in crisp detail, all within a shallow depth of field that keeps the focus tight on this intimate act of survival.

###### Figure 14 More comparisons between OSP-Next and the Wan2.1 baseline.

[Figure 201]

###### Wan2.1OSP-NextOSP-Next-HiF8Wan2.1OSP-NextOSP-Next-HiF8

[Figure 202]

[Figure 203]

A slow, smooth pan right glides across an expansive field of tall, golden grass, each blade undulating in hypnotic, rhythmic waves under a gentle breeze. The setting sun dominates the background, a molten orb of deep amber and soft crimson bleeding into the horizon, casting long, dramatic shadows that stretch across the textured sea of grass. Warm, golden-hour backlighting filters through the feathery seed heads, creating a luminous, ethereal rim light that sets the field aglow with a honeyed translucence. The atmosphere is serene and contemplative, imbued with the quiet melancholy of a day ending. A shallow depth of field keeps the immediate foreground grasses sharply defined, their silken textures and fine bristles catching the light, while the distant treeline and the sun soften into a painterly, impressionistic blur. The color palette is a rich tapestry of warm ochre, burnt orange, and soft lavender, accented by the crisp, earthy scent of dry grass. The motion feels like a gentle exhale, a final, sweeping glance across the tranquil, sun-drenched landscape.

[Figure 204]

[Figure 205]

[Figure 206]

A woman in a flowing emerald green dress and a wide-brimmed straw sun hat takes a leisurely, pleasant stroll along a quiet, tree-lined avenue in Johannesburg. The scene unfolds during a breathtaking sunset, where the sky is a canvas of deep oranges, soft pinks, and pale purples. A gentle tracking shot follows her from a low angle, the camera smoothly gliding alongside her relaxed pace. Golden hour sunlight streams through the leaves of Jacaranda trees, creating a dappled, chiaroscuro effect on the pavement and casting a warm, honeyed backlight that makes her dress fabric shimmer with a subtle silken texture. The atmosphere is serene and contemplative, filled with the soft, dusty glow of a South African evening. In the shallow depth of field, the background melts into a soft bokeh of distant city lights and lush foliage, while the air itself seems thick with golden, volumetric light. The color palette is rich and warm, dominated by amber, olive, and burnt sienna tones, evoking a nostalgic, cinematic romance.

###### Figure 15 More comparisons between OSP-Next and the Wan2.1 baseline.

