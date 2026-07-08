## MMInference: Accelerating Pre-filling for Long-Context Visual Language Models via Modality-Aware Permutation Sparse Attention

# arXiv:2504.16083v2[cs.CV]23May2025

Yucheng Li1* Huiqiang Jiang2§ Chengruidong Zhang2 Qianhui Wu2 Xufang Luo2 Surin Ahn2 Amir H. Abdi2 Dongsheng Li2 Jianfeng Gao2 Yuqing Yang2 Lili Qiu2

### Abstract

The integration of long-context capabilities with visual understanding unlocks unprecedented potential for Vision Language Models (VLMs). However, the quadratic attention complexity during the pre-filling phase remains a significant obstacle to real-world deployment. To overcome this limitation, we introduce MMInference (Multimodality Million tokens Inference), a dynamic sparse attention method that accelerates the prefilling stage for long-context multi-modal inputs. First, our analysis reveals that the temporal and spatial locality of video input leads to a unique sparse pattern, the Grid pattern. Simultaneously, VLMs exhibit markedly different sparse distributions across different modalities. We introduce a permutation-based method to leverage the unique Grid pattern and handle modality boundary issues. By offline search the optimal sparse patterns for each head, MMInference constructs the sparse distribution dynamically based on the input. We also provide optimized GPU kernels for efficient sparse computations. Notably, MMInference integrates seamlessly into existing VLM pipelines without any model modifications or fine-tuning. Experiments on multi-modal benchmarks—including Video QA, Captioning, VisionNIAH, and Mixed-Modality-NIAH—with stateof-the-art long-context VLMs (LongVila, LlavaVideo, VideoChat-Flash, Qwen2.5-VL) show that MMInference accelerates the pre-filling stage by up to 8.3× at 1M tokens while maintaining accuracy. Our code is available at https://aka.

Dynamic Sparse Attention

AShape VS Grid

Block-

Sparse Pattern

Sparse Last

Antidia gonal

Pooling/

Estimation

Q

Compress

(e.g. O(n), O(n^2))

Dynamic Sparse Index

###### Permutation

Column /Row Grid (back) slash

Block

(Sparse Load)

###### Tile-wise

Block-sparse Tensor Core

(Dense Compute)

(e.g. WGMMA)

In kernel

Figure 1: Dynamic sparse attention pipelines leverage sparse loading with dense computation (Zheng et al., 2023) to enable hardware-efficient acceleration. MMInference adopts a bottom-up system-algorithm co-design that accounting for both the mathematical equivalence constraints of sparse loading and the locality properties of real-world attention patterns.

### 1. Introduction

Scaling the context size of Vision Language Models (VLMs) allows them to handle extended temporal information from long video and text inputs, which is crucial for various applications including robotics (Black et al., 2024; Prasad et al., 2024; Cheang et al., 2024), autonomous driving (Hu et al., 2023; Wang et al., 2024c; Gao et al., 2024), and healthcare (Liu et al., 2024b). In addition, Zhang et al. (2024b) and Chen et al. (2025) show that scaling the context size of VLMs can improve the resolution in the temporal dimension and lead to better performance in video understanding tasks.

ms/MMInference.

*Work during internship at Microsoft. 1University of Surrey 2Microsoft Corporation. Correspondence to: Huiqiang Jiang <hjiang@microsoft.com>.

Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

However, due to the quadratic complexity of attention, processing long multi-modal inputs (i.e., the pre-fill stage) can take minutes prior to auto-regressive decoding. As shown in Fig. 2a, this leads to significant Time-to-First-Token latency, which hinders the wide adoption of long-context VLMs in real-world applications. Previous work (Child et al., 2019; Liu et al., 2022; 2024a; Yuan et al., 2025; Lu et al., 2025) re-

25

ViT

Attention

20

FFN

Latency(min)

Total

10

1

| | | |
|---|---|---|
| | | |

512 1k 2k 4k Num. of Frames

(a) VLMs’ attention incurs heavy cost.

Attention Recall is 71.3%

1.0

1 4

VLM LLM

| |[Figure 1]| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

[Figure 2]

25%

0.8

8 12 16 20 24 28

20%

TopKRatio

0.6

15%

Layer

0.4

10%

0.2

5%

1%

0.0

1 4 8 12 16 20 24 28 Head

1 5 10 15 20 25 28 Layer Index

(b) VLMs’ attention is sparse.

(c) Sparsity of VLMs’ attention is dynamic.

Figure 2: (a) Latency breakdown of the pre-filling stage, with 256 tokens per frame. (b) How much element in attention needs to be computed to achieve 95% recall in a 128k context. (c) Low attention recall when reusing the top-k indices from a different request. Visualizations are based on LongVILA-7B-1M (Chen et al., 2025) with a single A100.

veals that attention matrices are typically sparse, prompting the development of sparse attention methods such as Sparse Transformer (Child et al., 2019), Swin Transformer (Liu et al., 2021), and StreamingLLM (Xiao et al., 2024). More recently, MInference (Jiang et al., 2024) proposes to use dynamic sparse attention that estimates the sparse index online, and leverages optimized GPU kernels for end-to-end acceleration. However, these methods fail to exploit the unique sparse patterns in long-context VLMs, and struggle with mixed or interleaved modalities, limiting their applicability without compromising performance.

Unlike long-text contexts, video and image inputs in VLMs exhibit spatiotemporal locality, forming grid-like attention patterns with evenly spaced vertical and horizontal lines (Fig. 3a). In mixed-modality inputs, clear modality boundaries emerge: attention across modalities diverges significantly from intra-modality attention (Fig. 3b). These factors pose unique challenges for exploiting sparsity to accelerate the pre-fill stage.

In this paper, we present MMInference, a permutation-based dynamic sparse attention method that significantly reduces attention FLOPs, accelerating the pre-fill stage of longcontext VLMs. First, MMInference identifies the grid heads and leverages a row- and column-wise permutation to gather the sparse grid for efficient hardware computation. Next, we detect Query-boundary and 2D-boundary patterns to address inter-modality boundary issues, and apply a modality-wise permutation to isolate intra-modality regions. This results in a consecutive sparse index within each modality, permitting efficient hardware implementation of sparse computing. Finally, a Modality-Aware Sparse Attention Search Algorithm is devised to fine-tune both inter- and intra-modality patterns offline, to optimize performance with minimal overhead.

We conduct extensive experiments using four state-of-theart long-context VLMs, Llava-Video (Zhang et al., 2024b), LongVila (Chen et al., 2025), VideoChat-Flash (Li et al.,

2025) and Qwen2.5-VL (Bai et al., 2025), across diverse video understanding tasks such as video captioning (Maaz et al., 2024), video question answering (Yu et al., 2019; Xiao et al., 2021; Mangalam et al., 2023; Fu et al., 2024), and video information retrieval (Zhang et al., 2024a). Additionally, we propose the Mixed-Modality Needle in a Hackathon task to assess multi-modal input performance. Our method effectively addresses modality boundaries, significantly accelerates the prefilling stage, and maintains high accuracy. With a 1M-length context, it achieves speedups of up to 8.3× and 1.7× over FlashAttention-2 and MInference, respectively.

### 2. Attention Heads in VLMs

The sparsity of the attention operation in pre-trained textonly LLMs, particularly in long-context scenarios, has been extensively studied (Wu et al., 2025; Ribar et al., 2024; Jiang et al., 2024; Li et al., 2024), showing that only 3% of attention weights are activated while achieving a recall rate of 96.8%. Similarly, VLMs also demonstrate notable dynamic sparsity in long-context scenarios. This section examines the shared and distinct properties of text-only and multi-modal LLMs in long-context scenarios, focusing on attention sparsity, sparse patterns, and modality boundaries.

#### 2.1. Multi-modality Attention is Dynamically Sparse

As illustrated in Fig. 2a, for a 128k × 128k attention matrix in VLMs, retaining only the top 5.78% of attention weights on average suffices to recall 95% of total attention, indicating that each token attends only to a limited subset of tokens, even in long sequences. However, VLMs exhibit lower sparsity than text-only LLMs, where only 1.79% of weights achieve a 95% recall rate. Notably, the bottom layers in VLMs (e.g., the first four layers in LongVila) show reduced sparsity. Yet, due to variability across attention heads, 52.3% of heads in VLMs require less than 2% of

[Figure 3]

Vision

1 frame language

[Figure 4]

[Figure 5]

K-wise boundary

Language

Fixed stride size

Vision

Patch size 14 * 14

Q-wise boundary

Language

Modality boundary

(a) Grid pattern.

(b) Q-Boundary pattern.

(c) 2D-Boundary pattern.

[Figure 6]

[Figure 7]

[Figure 8]

V2V V2T

Vlines after shuffle

Consecutive Modality

Consecutive pattern

Hlines after shuffle

T2V T2T

(d) Permuted Grid pattern.

(f) Permuted 2D-Boundary pattern. Figure 3: Visualization of pre- vs. post-permutation sparsity attention patterns in VLMs.

(e) Permuted Q-Boundary pattern.

attention to be recalled. This highlights substantial computational redundancy in VLMs, especially in long-context scenarios.

predictable patterns.

We observe that certain VLM attention heads exhibit a grid pattern. While the grid’s stride and starting position vary with context, the horizontal and vertical lines are evenly spaced and often symmetrical—a distinct behavior compared to text-only LLMs (Jiang et al., 2024; Lai et al., 2025).

Similarly to LLMs, while the sparse nature of attention matrices remains consistent across inputs, the specific distributions of sparse attention are highly dynamic. As shown

- in Fig. 2c, reusing top-k indices for 95% attention recall (derived from Fig. 2b) across different contexts leads to a significant drop in performance.

- Fig. 3a visualizes a grid head, demonstrating how local tokens in temporal and spatial dimensions are evenly distributed within the attention map, with attention focused primarily on these local tokens.

2.3. Modality Boundaries in Multi-Modal Input

The input format of VLMs differs significantly from textonly LLMs. A dedicated vision encoder generates visual representations, which are processed alongside text embeddings by the LLM. Despite pretraining on large-scale datasets, the interactions and processing patterns between modalities vary considerably, leading to distinct modality boundaries in attention (Tu et al., 2025), as illustrated in

- Fig. 3b and 3c.

#### 2.2. The Grid Head in VLMs

In long-context language modeling, efficient attention mechanisms like sliding window attention (Jiang et al., 2023) and StreamingLLM (Xiao et al., 2024) exploit the locality property of text sequences. However, multi-modal inputs introduce unique geometric structures that redefine locality. As shown in Child et al. (2019), image patches exhibit locality along both vertical and horizontal directions, forming local window and slash-like patterns. Similarly, video inputs maintain locality across temporal and spatial dimensions, with frame-based sampling yielding more regular and

Inter-modality Attention Pattern

| |
|---|

| |
|---|

| |
|---|

| |
|---|

No-Boundary head K-Boundary head Q-Boundary head 2D-Boundary head

Intra-modality Attention Pattern

| |
|---|

Approximate

by last q

Permutation

Λ-shape head vertical-slash head grid head

Figure 4: The framework of MMInference, encompassing both inter- and intra-modality sparse attention patterns.

Specifically, we observe two key characteristics: 1) Intramodality consistency: Attention within each modality follows a consistent pattern. For instance, the vision region in Fig. 3b exhibits a clear slash pattern, where critical elements are effectively clustered. 2) Modality-separated continuity: Patterns within a modality can be interrupted by boundaries from other modalities. As shown in Fig. 3b, vision slashes are segmented by the boundary introduced by the language region.

We categorize the modality boundary patterns of VLMs into four distinct types: No-Boundary, K-Boundary, QBoundary, and 2D-Boundary, as illustrated in Figs. 3 and 4. 1) No Boundary and K-Boundary exhibit either no clear modality boundary or a boundary only along the key dimension, as shown in Fig. 9. Since continuity is maintained along the query dimension, these heads can be efficiently handled using intra-modality sparse patterns. 2) Q-Boundary refers to attention modality boundaries across the query dimension. For example, in Fig. 3b, sparse patterns like Text-to-Video and Video-to-Video appear interconnected, forming a trapezoidal structure, while a clear boundary separates Visual-to-Text and Text-to-Visual attention. 3) 2D-Boundary occurs when modality boundaries are present in both query and key dimensions. As shown

- in Fig. 3c, the 2D modality boundary segments attention weights into distinct blocks. Additionally, our analysis of Audio LMs (Chu et al., 2024) and end-to-end multimodal LMs (Xu et al., 2025a; Li et al., 2025) reveals that the cross-modality boundary phenomenon persists across these architectures. These boundaries pose unique challenges and hinder direct application of existing sparse attention methods to multi-modal inputs.

#### 2.4. Sparse Distributions Continuity Across Boundaries

Although sparsity patterns in VLMs are often discontinuous across modalities due to modality boundaries, we find that sparsity distributions can remain continuous across these boundaries and extrapolate to other regions of the same modality. For example, in Fig. 3b, the slash lines maintain the same relative position across different areas of the vision modality. In a more complex case, Fig. 3c shows interleaved vision and text modalities forming a mixed structure. However, by spatially aggregating regions of the same modality, we observe that sparsity patterns can extend beyond local regions and often exhibit global extrapolation potential. The upper-left region in Fig. 3c exemplifies this, where the grid pattern, initially separated by textual boundaries, becomes consecutive after spatial clustering in both row and column dimensions. To validate this observation, we conducted a quantitative attention recall experiment on mixed-modality inputs, as detailed in §4.6.

### 3. MMInference

Following the analysis in §2, we propose MMInference to accelerate the pre-filling stage of long-context VLMs as shown in Fig. 4. The framework consists of three modules, covering both inter- and intra-modality sparse patterns: 1) the novel Grid sparse attention, together with the A-shape and Vertical-Slash patterns (Jiang et al., 2024) forms the intra-modality attention; 2) Q-Boundary and 2D-Boundary mix-modality patterns; 3) Modality-aware sparse attention search algorithm. We first perform offline pattern search to identify different patterns for each attention head. Then we

use online dynamic sparse approximation to build the sparse index, and finally we perform dynamic sparse computation using optimized GPU kernels.

#### 3.1. Grid Head in Multi-Modality

To better leverage the inductive bias in visual modalities (e.g., images, videos) and the vertical and horizontal structures in attention patterns, we propose a permutation-based dynamic sparse attention for grid head, as shown in Algo. 1.

Algorithm 1 Grid Head Input: Q,K,V ∈ RS×d

h, stride space sg ∈ ϕg

# Approximate stride and phase (last q = 64) A ← softmax Q[−last q:]K⊤/

√

d + mcasual

# Online search grid stride and phase br, ← 0 for i ← 1 to |ϕg| do

if max(view( A, sg,i)) > br then sg ← sg,i, pg ← argmax(view( A, sg,i)) br ← max(view( A, sg,i))

end end for # Permute Q, K, V tensors Q, K, V ← permute (Q) , permute (K) , permute (V )

# Dynamic block sparse attention w/ FlashAttention (only the last and rightmost block)

√

A ← softmax sparse(QK⊤, sg, pg)/

d

# Sparse mixed scores and values y ← sparse(AV , sg, pg) return y

Specifically, we first perform an online search to determine the stride and phase of grid pattern. Since only a view operation is applied to the approximate attention matrix A, the actual latency overhead remains minimal. Next, we use the identified grid stride and phase to permute the Q, K, and V tensors to compute sparse attention efficiently (see Fig. 3d). In our implementation, instead of explicitly permuting Q, K, and V , we optimize computational efficiency by dynamically loading and writing these tensors within the kernel, minimizing the overhead associated with tensor transpositions. In addition to Grid sparse attention, we also employ A-shape and Vertical-Slash attention for intra-modality operation, see Appendix C.3 for more details.

#### 3.2. Hybrid Modality Sparse Attention

As analyzed in §2 and illustrated in Fig. 3, modality boundaries exist in multi-modal LLMs. We classify these boundaries into four patterns: No-Boundary, K-Boundary, QBoundary, and 2D-Boundary. As the sparse index is continuous along the query dimension for both the No-Boundary

and K-Boundary heads, we can directly apply the three intra-modality attention globally. However, for Q-Boundary and 2D-Boundary, MMInference uses a permutation-based approach to efficiently handle these modality boundaries.

Q-Boundary Head As shown in Fig.3b, Fig.3e, and §2.4, the Q-Boundary pattern shows a clear separation across modality, but the sparse distribution remains continuous within each modality. Building on this insight, we propose a row-wise permutation (Algorithm 2) that groups tokens of the same modality by permuting Q, and then applies offline-optimized sparse attention (A-shape, Vertical-Slash, and Grid Head) for intra-modality processing. Note that we leverage the final segment of each modality’s queries to dynamically approximate the sparse indices and extrapolate to the entire modality. This method enables flexibility in handling fragmented multi-modality inputs. Additionally, instead of explicitly permuting tensors, our implementation performs dynamic loading and writing inside the kernel for optimized efficiency.

Algorithm 2 Q-Boundary Head Input: Q,K,V ∈ RS×d

h, modality type index im, modality type set m ∈ ϕm

# Permute Q tensors based on modality Q ← permute (Q, im) # Looping over the modalities in query dimension y ← 0 for i ← 1 to |ϕm| do

# Intra-modality sparse attention for each modality w/ FlashAttention

√

Ami ← softmax sparse(QmiK⊤, imi)/

d

ymi ← sparse(AmiV ) # Update the modality output to the final output y ← ymi ∪ y

end for return y

2D-Boundary Head Beyond Query-Boundary, there are attention heads that exhibit modality boundaries in both query and key dimensions, as shown in Fig. 3c. Given a query token, attention to key tokens from different modalities varies significantly, and queries from different modalities focus on keys in highly diverse patterns. To address 2D modality boundaries, we design a 2D permutation approach that groups Q, K, and V according to their modalities. This allows us to leverage intra-modality continuity to handle each part of 2D boundary pattern separately and efficiently. We further illustrate this approach in Fig. 3f and it detailed in Algorithm 3. Specifically, we perform permutation on both row- and column-wise for Q, K, and V , and then iteratively traverse each modality pair to compute dynamic

Table 1: Performance (%) of different models and different methods on video understanding tasks evaluated at frames from 110 to 256.

VideoDC ActNet-QA EgoSchema Next-QA PerceptionTest VideoMME

Model FLOPs

Avg. test test test mc val w/o sub. w/ sub.

Llava-Video-7B # Frames: 110; Total # tokens: 20,240

Full Attention 100% 3.66 59.6 57.0 81.2 66.1 64.7 71.0 57.6 SF-fixed 4.8% 3.26 57.3 53.3 79.8 62.9 59.9 67.1 54.8 SF-strided 41.4% 3.45 58.5 56.1 80.6 64.4 61.4 68.5 56.1 A-shape 48.2% 3.56 56.0 51.6 79.8 65.7 54.4 65.6 53.8 Tri-shape 49.0% 3.58 59.3 54.5 80.3 66.1 63.6 70.1 56.7 VisionZip 35.2% 1.35 42.1 40.5 69.5 41.4 44.9 62.1 43.1 MInference 78.8% 3.64 59.6 57.0 80.6 66.1 64.6 71.0 57.5 Ours 47.3% 3.58 59.8 57.1 80.1 66.2 64.5 71.8 57.6

LongVILA-7B # Frames: 256; Total # tokens: 65,800

- Full Attention 100% 2.76 59.5 61.9 80.7 58.1 60.1 65.1 55.5 SF-fixed 2.2% 1.99 51.3 59.6 76.5 55.5 57.1 63.0 52.1 SF-strided 26.6% 2.58 56.0 61.4 76.7 55.5 53.6 59.2 52.2 A-shape 29.1% 2.75 56.6 60.9 75.0 55.3 49.1 59.6 51.3 Tri-shape 29.3% 2.63 58.1 62.0 77.8 56.2 59.3 63.3 54.2 VisionZip OOM MInference 47.0% 2.77 59.7 62.2 79.1 57.8 60.0 65.2 55.2 Ours 31.8% 2.84 60.2 62.2 79.4 57.8 60.0 65.5 55.4

Qwen2.5-VL-7B-Instruct # Frames: 256; Total # tokens: 33,950

- Full Attention 100% 3.71 58.3 64.3 85.4 68.7 64.7 71.3 59.5 Ours 41.3% 3.75 58.0 63.9 84.9 68.9 65.1 70.9 59.4

#### 3.3. Modality-Aware Sparse Attention Search Algorithm

Algorithm 3 2D-Boundary Head Input: Q,K,V ∈ RS×d

h, modality type index im, modality type set m ∈ ϕm

Due to modality boundaries in VLMs, we propose a modality-aware sparse attention pattern search algorithm (see Algorithm 4). The process unfolds in three steps: 1) intra-modality search within each modality following (Jiang et al., 2024), 2) cross-modality search across all modality pairs, and 3) inter-modality search informed by the results of the first two steps.

# Permute Q, K, V tensors based on modality Q ← permute (Q, im) , K ← permute (K, im) V ← permute (V , im) # Looping over the modalities in pairs y ← 0

- for i ← 1 to |ϕm| do
- for j ← 1 to |ϕm| do

### 4. Experiments

# Dynamic sparse attention for each modality pair w/ FlashAttention

In this section, we address two key questions: (i) How effective MMInference is? We evaluate our method on three general long-video tasks: long-video understanding, Video Needle in a Haystack, and Video-Text Needle in a Haystack. These benchmarks cover long-video captioning, open-ended QA, multiple-choice QA, mixed-modality tasks, and retrieval tasks, providing a comprehensive assessment of MMInference’s effectiveness across diverse long-video scenarios. (ii) How efficient MMInference is? We analyze end-to-end latency and its breakdown to thoroughly evaluate the efficiency of MMInference.

mmi,mj ← buildmask(imi, imj) Ami,mj ← softmax(

√

sparse(QmiK⊤mj, imi, imj)/

d + mmi,mj) ymi,mj ← sparse(Ami,mjV mj) # Update the modality output to the final output y ← ymi,mj ∪ y

end for end for return y

sparse attention. The 2D-Boundary requires constructing an attention mask and searching for sparse patterns in crossmodality regions. For example, in Fig. 3f, we build modality boundary indices for Vision-to-Text (bottom-left) and Text-to-Vision (upper-right) attention. This mask index construction is implemented in Triton (Tillet et al., 2019).

#### 4.1. Dataset and Baselines

Implementation Details Our experiments are conducted on two state-of-the-art long-video VLMs: Llava-

LongVILA-1M w/ MMInference in VNIAH Recall 97.7%

LongVILA-1M in VNIAH Recall 98.3%

1.0

1.0

[Figure 9]

[Figure 10]

0

0

0.8

0.8

DepthPercent(%)

DepthPercent(%)

20

20

0.6

0.6

40

40

Score

Score

60

60

0.4

0.4

80

80

0.2

0.2

100

100

0.0

0.0

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K4.8K5.1K5.4K5.7K6.0K

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K4.8K5.1K5.4K5.7K6.0K

Num. of Frames

Num. of Frames

(a) MMInference in V-NIAH

(b) FullAttention in V-NIAH

LongVILA-1M w/ MMInference in MMNIAH Recall 91.3%

LongVILA-1M in MMNIAH Recall 90.9%

1.0

1.0

[Figure 11]

[Figure 12]

0

0

0.8

0.8

DepthPercent(%)

DepthPercent(%)

20

20

0.6

0.6

40

40

Score

Score

60

60

0.4

0.4

80

80

0.2

0.2

100

100

0.0

0.0

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K

Num. of Frames

Num. of Frames

(d) FullAttention in MM-NIAH Figure 5: V-NIAH (Zhang et al., 2024a) and MM-NIAH results using LongVila-Qwen2-7B-1M (Chen et al., 2025).

(c) MMInference in MM-NIAH

Video (Zhang et al., 2024b) and LongVILA (Chen et al., 2025). We follow the MInference experimental setup, configuring the corresponding search space while adopting optimal configurations from prior work for other methods. We adjust the local window sizes of A-shape and tri-shape patterns to align FLOPs with our method. For MInference, we adopt its optimal configuration, which results with FLOPs approximately twice as high as our method’s in VLMs. Our implementation leverages Triton (Tillet et al., 2019), FlashAttention (Dao, 2024), and dynamic sparse compiler PIT (Zheng et al., 2023). For the Vertical-Slash and Grid Head patterns, we set lastq = 64. Latency experiments are performed on a single NVIDIA A100 using bfloat16, with greedy decoding to ensure stable results. Additional implementation details are provided in Appendix C.

Dataset Our evaluation uses the official metrics and scripts provided by these tasks. Additionally, we introduce a Mixed-Modality Needle in a Haystack (MM-NIAH) task to assess VLMs’ retrieval capabilities on mixed-modality inputs. Dataset details are provided in Appendix D.

(i) Video Understanding Tasks: These include ActNetQA (Yu et al., 2019), EgoSchema (Mangalam et al., 2023), Next-QA (Xiao et al., 2021), PerceptionTest (Patraucean et al., 2024), VideoDC (Lab, 2024), and VideoMME (Fu et al., 2024). These benchmarks span five categories, covering tasks such as captioning and video question answering. Input lengths range from 110 frames (e.g., 20k) to 256

frames (e.g., 66k) in Llava-Video (Zhang et al., 2024b) and LongVILA (Chen et al., 2025).

- (ii) Video Needle in a Haystack (V-NIAH) (Zhang et al., 2024a): A long-video retrieval task testing VLMs’ performance with tokens of up to 6k frames (e.g., 1.1M tokens), where inserted images are placed at various positions.
- (iii) Mixed-Modality Needle in a Haystack (MM-NIAH): To evaluate VLMs in mixed-modality scenarios, we construct a mix-modality version of NIAH. Specifically, 25% of the input consists of text segments inserted at the document level across different frames in long-video inputs, forming a mix-modality haystack. All other settings align with VNIAH, including the multi-choice VQA task with randomly inserted images. This benchmark tests input lengths of up to 4.5k frames (e.g., 1.1M tokens).

Baselines We include five training-free sparse attention approaches, one visual token compression method, and also incorporate FlashAttention-2 (Dao, 2024) as a baseline. 1) SparseTransformer (Fixed) (Child et al., 2019): Retains attention within each segment and allows all tokens to attend to the segment’s initial tokens. 2) SparseTransformer (Strided) (Child et al., 2019): Employs local windows with dilated attention. 3) A-Shape (Xiao et al., 2024): Preserves only the sink token with local attention. 4) Tri-Shape (LI et al., 2025; Acharya et al., 2024): Extends A-Shape by enabling full attention for all tokens to the last window’s

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

(a) All Textual Context (b) Visual Context Inserted (c) More Visual Context (d) All Visual Context

Figure 6: Transition of sparse patterns from textual context to visual context. (a) The vertical-slash pattern for all textual context. (b) Grid pattern appears when visual modality is appended. (c) Grid pattern dominates.

#### 4.4. Mixed-Modality Needle In A Haystack

queries. 5) Vertical-Slash Pattern (Jiang et al., 2024): Focuses on specific tokens (vertical lines) and tokens at fixed intervals (slash lines). 6) VisionZip (Yang et al., 2024): A visual token compression method that reduces the number of visual tokens per frame by evaluating tokens based on their attention scores and discarding less important ones. Full details on implementation, hyperparameters, and illustrations for our baselines can be found in Appendix C.

Beyond V-NIAH, we introduce a mixed-modality NIAH test to evaluate the performance of different sparse methods on video-text inputs, in Fig. 5c, 5d, and 14. Mixedmodality inputs lead to more pronounced performance degradation across all methods. However, by incorporating inter-modality sparse patterns, our method maintains performance close to full attention, especially when compared to MInference and ours w/o inter-modality. Notably, Tri-shape and MInference show significant drops at 1.8k frames (i.g. 440K tokens) and 2.7k frames (i.g. 660K tokens).

#### 4.2. Long Video Understanding

- Table 1 presents the performance of different methods on video understanding tasks. The results show that: 1) Our method and MInference closely approximate full attention across all tasks while requiring only half the FLOPs of MInference. 2) Static sparse patterns, such as A-shape and Tri-shape, perform reasonably well on most tasks but experience a notable performance drop in multi-choice VQA tasks like EgoSchema. Additionally, the slight increase in query full attention in Tri-shape effectively improves performance. 3) Among SF patterns, the slash pattern better preserves performance. Even when using SF-fixed with only 2%-5% of FLOPs, it still maintains strong performance on most tasks.

#### 4.5. Latency

25

FlashAttention-2

MInference

18

Latency(min)

MMInference

8.3x

10

6x 1.6x 1.5x

3

1.5x 3.3x

| |
|---|

1

| |
|---|

| |
|---|

| |
|---|

| |
|---|

300/120K 900/360K 1800/720K 2700/1M Num. of Frames / Context Windows

#### 4.3. Video Needle In A Haystack

Figure 7: End-to-End Latency.

Fig. 5a, 5b, and 13 show the performance of different models on V-NIAH, revealing notable differences in handling long-context video retrieval as the number of processed frames increases: 1) Our method achieves results nearly identical to full attention. 2) A-shape struggles with midcontext information even at 300 frames, while Tri-shape maintains full performance until 3.9k frames (i.g. 700K tokens) before a sharp decline. 3) SF-fixed degrades at 2.1k frames (i.g. 350K tokens), while SF-strided surpasses Trishape, holding performance until 4.5k frames (i.g. 825K tokens). 4) MInference preserves VLM retrieval well, with only slight degradation beyond 4.8K frames.

Fig. 7 and 16 present end-to-end and kernel-level latency across different context sizes. The grid pattern significantly outperforms the vertical-slash pattern in sparsity, achieving a 2–3× speedup even at 1M tokens. Additionally, the grid pattern achieves an end-to-end speedup of 8.3× and a kernellevel speedup of 12×.

#### 4.6. Analysis

Transition of Sparse Patterns Across Modalities Since LLMs and VLMs exhibit different sparse patterns, we examine the interplay between the Grid and Vertical-Slash pattern. As shown in Fig. 6, Llava-Video-7B primarily uses VerticalSlash pattern for purely textual inputs. However, once a visual input is appended, it transitions to a Grid pattern to

Table 2: Performance (%) on video understanding tasks based on VideoChat-Flash (Li et al., 2025) at frames 512 with 8k tokens.

VideoDC ActNet-QA EgoSchema Next-QA PerceptionTest VideoMME

Model

Avg. test test test mc val w/o sub. w/ sub.

VideoChat-Flash 3.21 53.6 57.0 81.2 69.1 63.2 70.5 56.8 w/ MMInference 3.19 54.3 57.3 79.8 69.1 63.0 70.2 56.7

capture the geometric structure of the visual content. This shift occurs at the modality boundary, creating a more structured arrangement of vertical and horizontal intervals. Such behavior highlights the need for distinct sparsity strategies in visual and mixed-modality contexts, rather than simply reusing sparse patterns from LLMs for VLMs.

1.0

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Indexing and Modality Region<br><br>Text Index on Text<br><br>Text Index on Vision<br><br>Vision Index on Vision| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.9

0.8

AttnRecall

0.7

0.6

0.5

0.4

0 2 4 6 8 10 12 Layer

Figure 8: The sparse index does not effectively extrapolate from text to the visual modality. However, an index built within the same modality can generalize across modality boundaries.

Sparse Index Across Modalities In Fig. 8, the sparse index achieves high recall for textual regions but fails to generalize to visual ones. To address this, we construct a sparse index from the visual modality and evaluate it on separate visual segments, each separated by modality boundaries. Remarkably, this approach extrapolates effectively across all visual segments, even when interspersed with textual boundaries. As shown in Fig. 8, the sparse index achieves high recall in the textual but fails to generalize to the visual. To address this, we construct a sparse index using the visual modality and evaluate it across distinct regions of the visual modality, separated by modality boundaries. Remarkably, this approach successfully extrapolates to all visual regions even when interrupted by text-induced boundaries.

#### Integrate with token compression methods As shown in

- Table 2, our method integrates seamlessly with token compression techniques, enabling near-lossless performance while supporting longer or higher-resolution video inputs. Specifically, VideoChat-Flash reduces tokens per frame from 196 to 16 at the ViT stage, while our method further applies sparse attention in the LLM decoder. Results demonstrate strong performance retention across benchmarks.

### 5. Related Work

Long-Context Vision Language Models Recent VLMs have extended their context length to support long multimodal inputs (Zhang et al., 2024a; Chen et al., 2025; Wang et al., 2024b; Team et al., 2024), enabling applications such as long-video understanding (Fu et al., 2024; Xiao et al., 2021; Wang et al., 2024a; Bai et al., 2025), multi-modal retrieval (Zhang et al., 2024a), and multi-modal chain-ofthought reasoning (Qwen, 2024). For instance, Zhang et al. (2024a) transfer long-context capabilities from base LLMs to vision tasks, Chen et al. (2025) introduce multi-modal sequence parallelism to accelerate video fine-tuning, and Zhang et al. (2024b) emphasize the role of data calibration and synthetic data in boosting VLM performance.

Efficiency Optimization for VLMs While long-context VLMs achieve high accuracy, their high inference cost limits practical use in long-video scenarios. A common strategy is vision token compression—reducing video feature resolution by dropping or merging less important visual tokens (Bolya et al., 2023; Chen et al., 2024; Shen et al., 2024; He et al., 2024; Tu et al., 2025; Weng et al., 2024; Wen et al., 2024). RNN-Transformer hybrids are also used (Wang et al., 2024b) to balance efficiency and context length. However, these methods often assume inputs are long videos paired with short text, focusing solely on visual token optimization, while overlooking mixed-modality inputs critical for multiturn interactions (Huang et al., 2024). Recently, Xu et al. (2025b) applied dynamic sparse attention to long-context VLMs, but their approach ignores modality-specific inductive biases and is limited to single-modality video tasks.

### 6. Conclusion

We propose MMInference, a modality-aware permutation sparse attention method that accelerates long-context VLMs. It features permutation-based grid sparse attention, Qboundary/2D-boundary patterns for mixed-modality boundaries, and a Modality-Aware Sparse Attention Search Algorithm. Our optimized GPU kernels enable end-to-end acceleration. Experiments on video understanding tasks, V-NIAH and MM-NIAH using Llava-Video and LongVila demonstrate that MMInference preserves full-attention performance while achieving up to 8.3× speedup at 1M tokens.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Acharya, S., Jia, F., and Ginsburg, B. Star attention: Efficient llm inference over long sequences. ArXiv preprint, abs/2411.17116, 2024. URL https://arxiv.org/ abs/2411.17116.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. ArXiv preprint, abs/2502.13923, 2025. URL https://arxiv.org/abs/2502.13923.

Black, K., Nakamoto, M., Atreya, P., Walke, H. R., Finn, C., Kumar, A., and Levine, S. Zero-shot robotic manipulation with pre-trained image-editing diffusion models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.

net/forum?id=c0chJTSbci.

Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., and Hoffman, J. Token merging: Your vit but faster. ICLR, 2023.

Cheang, C.-L., Chen, G., Jing, Y., Kong, T., Li, H., Li, Y., Liu, Y., Wu, H., Xu, J., Yang, Y., et al. Gr-2: A generative video-language-action model with webscale knowledge for robot manipulation. ArXiv preprint, abs/2410.06158, 2024. URL https://arxiv.org/ abs/2410.06158.

Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. ECCV, pp. 19–35, 2024. doi: 10.1007/ 978-3-031-73004-7 2.

Chen, Y., Xue, F., Li, D., Hu, Q., Zhu, L., Li, X., Fang, Y., Tang, H., Yang, S., Liu, Z., He, Y., Yin, H., Molchanov, P., Kautz, J., Fan, L., Zhu, Y., Lu, Y., and Han, S. LongVILA: Scaling long-context visual language models for long videos. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=wCXAlfvCy6.

Child, R., Gray, S., Radford, A., and Sutskever, I. Generating long sequences with sparse transformers. ArXiv preprint, abs/1904.10509, 2019. URL https:// arxiv.org/abs/1904.10509.

Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., et al. Qwen2-audio technical

report. ArXiv preprint, abs/2407.10759, 2024. URL https://arxiv.org/abs/2407.10759.

Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. ICLR, 2024.

Ding, H., Li, D., Su, R., Zhang, P., Deng, Z., Stoica, I., and Zhang, H. Efficient-vdit: Efficient video diffusion transformers with attention tile. arXiv preprint arXiv:2502.06155, 2025.

Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. ArXiv preprint, abs/2405.21075, 2024. URL https://arxiv.org/ abs/2405.21075.

Gao, S., Yang, J., Chen, L., Chitta, K., Qiu, Y., Geiger, A., Zhang, J., and Li, H. Vista: A generalizable driving world model with high fidelity and versatile controllability. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https:

//openreview.net/forum?id=Tw9nfNyOMy.

Hassani, A., Walton, S., Li, J., Li, S., and Shi, H. Neighborhood attention transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6185–6194, 2023.

He, Y., Chen, F., Liu, J., Shao, W., Zhou, H., Zhang, K., and Zhuang, B. Zipvl: Efficient large vision-language models with dynamic token sparsification and kv cache compression. ArXiv preprint, abs/2410.08584, 2024. URL https://arxiv.org/abs/2410.08584.

Hu, A., Russell, L., Yeo, H., Murez, Z., Fedoseev, G., Kendall, A., Shotton, J., and Corrado, G. Gaia-1: A generative world model for autonomous driving. ArXiv preprint, abs/2309.17080, 2023. URL https:// arxiv.org/abs/2309.17080.

Huang, M., Long, Y., Deng, X., Chu, R., Xiong, J., Liang, X., Cheng, H., Lu, Q., and Liu, W. Dialoggen: Multimodal interactive dialogue system for multi-turn text-toimage generation. ArXiv preprint, abs/2403.08857, 2024. URL https://arxiv.org/abs/2403.08857.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. ArXiv preprint, abs/2310.06825, 2023. URL https://arxiv.org/ abs/2310.06825.

Jiang, H., Li, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han, Z., Abdi, A. H., Li, D., Lin, C.-Y., Yang, Y., and Qiu, L. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In The Thirty-eighth

Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/ forum?id=fPBACAbqSN.

Lab, L. Video detail caption, 2024. URL https://huggingface.co/datasets/ lmms-lab/VideoDetailCaption.

Lai, X., Lu, J., Luo, Y., Ma, Y., and Zhou, X. Flexprefill: A context-aware sparse attention mechanism for efficient long-sequence inference. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=OfjIlbelrT.

- Li, X., Wang, Y., Yu, J., Zeng, X., Zhu, Y., Huang, H., Gao, J., Li, K., He, Y., Wang, C., et al. Videochatflash: Hierarchical compression for long-context video modeling. ArXiv preprint, abs/2501.00574, 2025. URL https://arxiv.org/abs/2501.00574.
- Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., and Chen, D. SnapKV: LLM knows what you are looking for before generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https: //openreview.net/forum?id=poE54GOq2l.

LI, Y., Jiang, H., Wu, Q., Luo, X., Ahn, S., Zhang, C., Abdi,

- A. H., Li, D., Gao, J., Yang, Y., and Qiu, L. SCBench: A KV cache-centric analysis of long-context methods. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=gkUyYcY1W9.

Li, Y., Liu, J., Zhang, T., Chen, S., Li, T., Li, Z., Liu, L., Ming, L., Dong, G., Pan, D., et al. Baichuan-omni-1.5 technical report. ArXiv preprint, abs/2501.15368, 2025. URL https://arxiv.org/abs/2501.15368.

Liu, D., Chen, M., Lu, B., Jiang, H., Han, Z., Zhang, Q., Chen, Q., Zhang, C., Ding, B., Zhang, K., et al. Retrievalattention: Accelerating long-context llm inference via vector retrieval. ArXiv preprint, abs/2409.10516, 2024a. URL https://arxiv.org/abs/2409.

10516.

Liu, L., Qu, Z., Chen, Z., Tu, F., Ding, Y., and Xie, Y. Dynamic sparse attention for scalable transformer acceleration. IEEE Trans. Computers, pp. 3165–3178, 2022. doi: 10.1109/TC.2022.3208206.

Liu, L., Yang, X., Lei, J., Liu, X., Shen, Y., Zhang, Z., Wei, P., Gu, J., Chu, Z., Qin, Z., et al. A survey on medical large language models: Technology, application, trustworthiness, and future directions. ArXiv preprint, abs/2406.03712, 2024b. URL https://arxiv.org/ abs/2406.03712.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. ICCV, pp. 9992– 10002, 2021. doi: 10.1109/ICCV48922.2021.00986.

Lu, E., Jiang, Z., Liu, J., Du, Y., Jiang, T., Hong, C., Liu, S., He, W., Yuan, E., Wang, Y., et al. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025.

Maaz, M., Rasheed, H. A., Khan, S., and Khan, F. Videochatgpt: Towards detailed video understanding via large vision and language models. ACL, pp. 12585–12602, 2024. doi: 10.18653/V1/2024.ACL-LONG.679.

Mangalam, K., Akshulakov, R., and Malik, J. Egoschema: A diagnostic benchmark for very long-form video language understanding. NeurIPS, 2023.

Patraucean, V., Smaira, L., Gupta, A., Recasens, A., Markeeva, L., Banarse, D., Koppula, S., Heyward, J., Malinowski, M., Yang, Y., Doersch, C., Matejovicova, T., Sulsky, Y., Miech, A., Fr´echette, A., Klimczak, H., Koster, R., Zhang, J., Winkler, S., Aytar, Y., Osindero, S., Damen, D., Zisserman, A., and Carreira, J. Perception test: A diagnostic benchmark for multimodal video models. NeurIPS, 2023.

Patraucean, V., Smaira, L., Gupta, A., Recasens, A., Markeeva, L., Banarse, D., Koppula, S., Malinowski, M., Yang, Y., Doersch, C., et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36, 2024.

Prasad, A., Lin, K., Wu, J., Zhou, L., and Bohg, J. Consistency policy: Accelerated visuomotor policies via consistency distillation. ArXiv preprint, abs/2405.07503, 2024. URL https://arxiv.org/abs/2405.07503.

Qwen, T. Dao, tri and haziza, daniel and massa, francisco and sizov, grigory, 2023. URL https://crfm.stanford.edu/2023/10/ 12/flashdecoding.html.

Qwen, T. Qvq: To see the world with wisdom,

2024. URL https://qwenlm.github.io/blog/ qvq-72b-preview/.

Ribar, L., Chelombiev, I., Hudlass-Galley, L., Blake, C., Luschi, C., and Orr, D. Sparq attention: Bandwidthefficient llm inference. ICML, 2024.

Shen, X., Xiong, Y., Zhao, C., Wu, L., Chen, J., Zhu, C., Liu, Z., Xiao, F., Varadarajan, B., Bordes, F., et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. ArXiv preprint, abs/2410.17434, 2024. URL https://arxiv.org/ abs/2410.17434.

Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. ArXiv preprint, abs/2403.05530, 2024. URL https://arxiv.org/ abs/2403.05530.

Tillet, P., Kung, H.-T., and Cox, D. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pp. 10–19, 2019.

Tu, D., Vashchilenko, D., Lu, Y., and Xu, P. VL-cache: Sparsity and modality-aware KV cache compression for vision-language model inference acceleration. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.

net/forum?id=HMrcv7Q4Ub.

Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. ArXiv preprint, abs/2409.12191, 2024a. URL https://arxiv.org/abs/2409.12191.

Wang, X., Song, D., Chen, S., Zhang, C., and Wang, B. Longllava: Scaling multi-modal llms to 1000 images efficiently via a hybrid architecture. ArXiv preprint, abs/2409.02889, 2024b. URL https://arxiv.org/ abs/2409.02889.

Wang, X., Zhu, Z., Huang, G., Chen, X., Zhu, J., and Lu, J. Drivedreamer: Towards real-world-drive world models for autonomous driving. In European Conference on Computer Vision, pp. 55–72. Springer, 2024c.

Wen, Y., Cao, Q., Fu, Q., Mehta, S., and Najibi, M. Efficient vision-language models by summarizing visual tokens into compact registers. ArXiv preprint, abs/2410.14072, 2024. URL https://arxiv.org/ abs/2410.14072.

Weng, Y., Han, M., He, H., Chang, X., and Zhuang, B. Longvlm: Efficient long video understanding via large language models. ECCV, pp. 453–470, 2024. doi: 10. 1007/978-3-031-73414-4 26.

Wu, W., Wang, Y., Xiao, G., Peng, H., and Fu, Y. Retrieval head mechanistically explains long-context factuality. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.

net/forum?id=EytBpUGB1Z.

Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. ICLR, 2024.

Xiao, J., Shang, X., Yao, A., and Chua, T.-S. Next-qa: Next phase of question-answering to explaining temporal actions. CVPR, pp. 9777–9786, 2021. doi: 10.1109/ CVPR46437.2021.00965.

Xu, J., Guo, Z., He, J., Hu, H., He, T., Bai, S., Chen, K., Wang, J., Fan, Y., Dang, K., et al. Qwen2. 5-omni technical report. ArXiv preprint, abs/2503.20215, 2025a. URL https://arxiv.org/abs/2503.20215.

Xu, R., Xiao, G., Huang, H., Guo, J., and Han, S. Xattention: Block sparse attention with antidiagonal scoring. ArXiv preprint, abs/2503.16428, 2025b. URL https://arxiv.org/abs/2503.16428.

Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., and Jia, J. Visionzip: Longer is better but not necessary in vision language models. ArXiv preprint, abs/2412.04467, 2024. URL https://arxiv.org/abs/2412.04467.

Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y., and Tao, D. Activitynet-qa: A dataset for understanding complex web videos via question answering. AAAI, pp. 9127–9134, 2019. doi: 10.1609/AAAI.V33I01.33019127.

Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z., Xie, Z., Wei, Y., Wang, L., Xiao, Z., et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025.

Zhang, P., Zhang, K., Li, B., Zeng, G., Yang, J., Zhang, Y., Wang, Z., Tan, H., Li, C., and Liu, Z. Long context transfer from language to vision. ArXiv preprint, abs/2406.16852, 2024a. URL https://arxiv.org/ abs/2406.16852.

Zhang, P., Chen, Y., Su, R., Ding, H., Stoica, I., Liu, Z., and Zhang, H. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025.

Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., and Li, C. Video instruction tuning with synthetic data. ArXiv preprint, abs/2410.02713, 2024b. URL https: //arxiv.org/abs/2410.02713.

Zheng, N., Jiang, H., Zhang, Q., Han, Z., Ma, L., Yang, Y., Yang, F., Zhang, C., Qiu, L., Yang, M., et al. Pit: Optimization of dynamic sparse deep learning models via permutation invariant transformation. In Proceedings of the 29th Symposium on Operating Systems Principles, pp. 331–347, 2023.

### A. Modality-Aware Sparse Attention Search Algorithm

In Algorithm 4, we detail the procedure for selecting the optimal sparse attention pattern for each attention head under a constrained FLOPs budget. The algorithm jointly determines the best pattern and its configuration (e.g., stride size in grid attention, number of vertical/slash lines in VS pattern) to maximize accuracy. We first construct a kernel-aware search space, where all candidate patterns have comparable real-world FLOPs based on GPU kernel measurements—rather than theoretical estimates—to ensure practical efficiency.

We then evaluate each candidate using a reference example and select the configuration that maximizes attention recall, using the actual attention output as the objective. This recall-based scoring incorporates the V matrix and builds on FlashAttention (Dao, 2024), enabling end-to-end pattern selection with minimal memory overhead and improved performance.

Algorithm 4 Modality-aware Sparse Attention Pattern Search Input: Q,K,V ∈ RS×d

h, inter-modality search space ρinter, intramodality search space ρintra, modality type set m ∈ ϕm, optimized sparse pattern P

# Intra-modality sparse attention pattern search for i ← 1 to |ϕm| do

pmi ← KernelAwareSearch (Q, K, V , mi) P ← P ∪ pmi

end for # Cross-modality sparse attention pattern search

- for i ← 1 to |ϕm| do
- for j ← 1 to |ϕm| do pmi,mj ← KernelAwareSearch (Q, K, V , mi, mj) P ← P ∪ pmi,mj

end for end for # Inter-modality sparse attention pattern search for i ← 1 to |ρinter| do

pi ← argmin (|sparse(Q, K, V , i) − attention(Q, K, V )| P ← P ∪ pi

end for return P

[Figure 18]

[Figure 19]

(a) K-Boundary pattern. (b) No-Boundary pattern. Figure 9: Additional inter-modality sparse pattern.

[Figure 20]

[Figure 21]

[Figure 22]

(a) A-shape (b) SF-fixed (c) SF-strided

[Figure 23]

[Figure 24]

(d) Tri-shape (e) Vertical-Slash (MInference) Figure 10: The baselines of sparse attention in our experiments.

### B. Pattern Analysis

#### B.1. Additional Mix-modality Pattern

In §2, we explain how the grid pattern naturally arises from the geometric structure of vision inputs. Fig. 9 further illustrates two additional patterns in the mixed-modality search space: the K-Boundary and No-Boundary patterns. Notably, both patterns incur no additional cost compared to pure intra-modality attention, as their sparse indices can be computed across all rows without extra computation.

- B.2. Additional Sparse Attention Pattern Visualization

We further analyze the sparse patterns in Qwen2.5-VL (Wang et al., 2024a) with dynamic resolution inputs and in VideoChatFlash (Li et al., 2025) under visual token compression, across both video benchmark and mixed-modality inputs, as shown in Fig.17 and Fig.18.

- C. Experiment Details

#### C.1. Vision Language Models

We use two state-of-the-art VLMs in our experiments: LongVILA (Chen et al., 2025) and Llava-Video (Zhang et al., 2024b). Llava-Video supports varying numbers of frames (32, 64, 110) for video understanding, and as reported, performance improves with more frames. Thus, we adopt the 110-frame variant for benchmarking. For LongVILA, we use the 256-frame version (LongVILA-256Frame) with a 128K context length for video understanding benchmarks, and the 1M-token version (LongVILA-1M), designed for retrieval tasks, for the V-NIAH evaluation.

Table 3: Hyperparameters detail of baselines.

Method Hyperparameters A-shape Sink = 128,Local = 4096 SF-fixed Local = token per frame,vline stride = token per frame SF-strided Local = token per frame,vline stride = token per frame Tri-shape Sink = 128,Local = 4096,Bottom = 128 MInference Vertical size ∈ {1000,2000,4000},Slash size ∈ {1024,2048,4096,6144} VisionZip dominant = 54,contextual = 10

#### C.2. Baselines

We include five sparse attention baselines in our experiments: A-shape (Xiao et al., 2024), SF-fixed (Child et al., 2019), SF-strided (Child et al., 2019), Tri-shape (LI et al., 2025), MInference (Jiang et al., 2024), and VisionZip (Yang et al., 2024). Fig. 10 illustrates the attention patterns of these baselines.

While VisionZip (Yang et al., 2024) is primarily a visual token compression method—compressing vision tokens using attention scores from the vision encoder before passing them to the LLM—it is included for comparison as it reduces FLOPs in the pre-filling stage and offers insight into token compression approaches.

- C.3. A-shape and Vertical-Slash A-shape and Vertical-Slash are used for intra-modality attention, alongside our newly proposed Grid pattern.

At inference time, we estimate the attention matrix online to dynamically determine the spatial layout of sparse indices, conditioned on the assigned pattern and actual input. Sparse attention is then computed using our optimized GPU kernels. Note that while the masks for Vertical-Slash and Grid patterns are dynamically generated, A-shape uses a static mask, incurring no additional overhead beyond sparse computation.

A-shape head. A-shape is a static sparse pattern that includes the first seven initial tokens along with a local attention window.

Vertical-Slash head. Due to the continuity of vertical and slash lines, we matmul the last query vector Q[−last q:] and key vector K to produce the estimated attention matrix A, which, in turn, is used to determine the indices for the vertical iv and slash is lines. After obtaining the sparse indices for the vertical and slash lines, we convert them into a sparse format ivs. Using these sparse indices, we perform block-sparse calculations of the attention weights and attention output.

- C.4. Permutation for the Grid Pattern and Across Modality

We illustrate how the permutation is applied to the Grid pattern and the Q-boundary and 2D-boundary patterns in Fig. 11 and Fig. 11.

[Figure 25]

[Figure 26]

[Figure 27]

(a) Before Permutation (b) Row-wise Permutation (c) Column-wise Permutation Figure 11: Permutation for the Grid Pattern. (a) Before permutation. (b) Row-wise permutation. (c) Column-wise permutation.

[Figure 28]

[Figure 29]

[Figure 30]

(a) Mix-modality (b) Q-wise Permutation (c) K-wise Permutation Figure 12: Permutation for mix-modality context. (a) Mix-modality. (b) Q-wise permutation. (c) K-wise permutation.

#### C.5. Search Space

Following (Jiang et al., 2024), we set the target FLOPs t to be the same as 1k global tokens and 4k local window tokens in the A-shape pattern. Additionally, we use only one sample as our calibration set from the egoschema task with no more than 25K tokens, which exhibits strong generalization and stability across different lengths and domains. The search time is approximately 15 minutes on a single A100. This pattern search is individually conducted for each model: Llava-Video-7B, LongVila-256Frame, and LongVila-1M. The search space is shown in Table 4.

- D. Benchmark Details We evaluate our method on several video understanding benchmarks that test different aspects of video comprehension:

EgoSchema EgoSchema (Mangalam et al., 2023) is a diagnostic benchmark for very long-form video language understanding, structured as a multiple-choice question answering task. The benchmark requires models to answer questions about egocentric videos by selecting from given options (labeled A through E). The evaluation can be performed either on the full set via submission to an evaluation server, or on a released subset of 500 questions for direct scoring.

Video-MME Video-MME (Fu et al., 2024) is a comprehensive multi-modal evaluation benchmark that tests MLLMs across diverse video types and temporal dimensions. It spans 6 primary visual domains with 30 subfields and includes videos ranging from 11 seconds to 1 hour in duration. The benchmark comprises 900 videos totaling 254 hours, with 2,700 manually annotated question-answer pairs. It evaluates models’ ability to process not just video frames but also integrated multi-modal inputs like subtitles and audio.

NExT-QA NExT-QA (Xiao et al., 2021) focuses on advancing video understanding from basic description to explaining temporal actions. It features both multiple-choice and open-ended QA tasks that target three key aspects: causal action reasoning, temporal action reasoning, and common scene comprehension. The benchmark is specifically designed to evaluate models’ ability to reason about actions beyond superficial scene descriptions.

Perception Test The Perception Test (Patraucean et al., 2023) perce evaluates perception and reasoning skills across video, audio, and text modalities. It contains 11.6k real-world videos with an average length of 23 seconds, featuring perceptually interesting situations. The benchmark tests four key skills (Memory, Abstraction, Physics, Semantics) and various types of reasoning (descriptive, explanatory, predictive, counterfactual). Videos are densely annotated with six types of labels: multiple-choice QA, grounded video QA, object tracks, point tracks, temporal action segments, and sound segments.

ActivityNet-QA ActivityNet-QA (Yu et al., 2019) is a large-scale VideoQA dataset consisting of 58,000 QA pairs on 5,800 complex web videos derived from the ActivityNet dataset. The benchmark is fully annotated and designed to test models’ understanding of complex web videos through question answering. Unlike automatically generated datasets, ActivityNet-QA features human-annotated questions and answers, making it particularly valuable for evaluating real-world video understanding capabilities.

#### Attention Type Parameters

(frame stride, True, False, False, 1024) (frame stride, False, True, False, 1024) (frame stride, False, False, True, 1024) (frame stride, True, True, False, 1024) (frame stride, False, True, True, 1024) (frame stride, True, True, True, 1024) (stride, True, False, False, 1024) (stride, False, True, False, 1024) (stride, False, False, True, 1024) (stride, True, True, False, 1024) (stride, False, True, True, 1024) (stride, True, True, True, 1024)

Grid Attention

(128, 1024) (128, 2048) (128, 4096)

A-shape

(1000, 1024) (1000, 2048) (2000, 2048) (1000, 3096) (2000, 3096) (1000, 4096) (2000, 4096) (3500, 200) (1000, 2500)

Vertical-Slash

Table 4: The search space for each attention pattern: 1) Grid Attention: (stride, use hline, use vline, use slash, max stride); 2) A-shape: (sink, local); 3) Vertical-Slash: (vertical size, slash size)

Video Detail Description (VideoDC) VideoDC (Lab, 2024) focuses on comprehensive video understanding through detailed descriptions. The benchmark consists of question-answer pairs generated with GPT-3.5, where questions prompt for detailed descriptions focusing on main subjects, their actions, and background scenes. The evaluation assesses the quality and completeness of video descriptions generated by models.

### E. Additional Experiments Results

#### E.1. Additional Video Needle In A Haystack Results

we further present the results of the Video Needle In A Haystack task with our baselines. The results of our method and full atttenton is shown in Fig. 5.

#### E.2. Additional Mixed-Modality Needle In A Haystack Results

We further present the results of the Mixed-Modality Needle In A Haystack task with our baselines and the inter-modality variant of our method. The results of full atttenton and MMInference are shown in Fig. 5.

#### E.3. Latency Breakdown

As shown in Fig. 16, we present the micro-benchmark results of various sparse attention methods across different context lengths.

LongVILA-1M w/ A-shape in VNIAH Recall 44.0%

LongVILA-1M w/ Tri-shape in VNIAH Recall 85.7%

1.0

1.0

[Figure 31]

[Figure 32]

0

0

0.8

0.8

DepthPercent(%)

DepthPercent(%)

20

20

0.6

0.6

40

40

Score

Score

60

60

0.4

0.4

80

80

0.2

0.2

100

100

0.0

0.0

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K4.8K5.1K5.4K5.7K6.0K

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K4.8K5.1K5.4K5.7K6.0K

Num. of Frames

Num. of Frames

(a) A-shape

(b) Tri-shape

LongVILA-1M w/ SF-fixed in VNIAH Recall 59.3%

LongVILA-1M w/ SF-strided in VNIAH Recall 90.7%

1.0

1.0

[Figure 33]

[Figure 34]

0

0

0.8

0.8

DepthPercent(%)

DepthPercent(%)

20

20

0.6

0.6

40

40

Score

Score

60

60

0.4

0.4

80

80

0.2

0.2

100

100

0.0

0.0

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K4.8K5.1K5.4K5.7K6.0K

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K4.8K5.1K5.4K5.7K6.0K

Num. of Frames

Num. of Frames

(c) SF-fixed

(d) SF-strided

LongVILA-1M w/ MInference in VNIAH Recall 96.7%

1.0

[Figure 35]

0

0.8

DepthPercent(%)

20

0.6

40

Score

60

0.4

80

0.2

100

0.0

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K4.8K5.1K5.4K5.7K6.0K

Num. of Frames

(e) MInference Figure 13: Video Needle In A Haystack (Zhang et al., 2024a) results using LongVila-Qwen2-7B-1M (Chen et al., 2025).

#### E.4. VS Pattern vs. Grid Pattern

Both VS pattern and Grid pattern achieve strong performance on video understanding and V-NIAH tasks. However, due to the grid attention pattern observed in VLMs, the overlap between blocks covered by diagonal lines in the VS pattern is minimal, reducing sparsity within the kernel. This explains why VS pattern exhibits significantly higher latency compared to Grid pattern. Additionally, leveraging permutation-based optimization effectively reduces the number of blocks involved in kernel computation, thereby lowering latency while maintaining comparable performance.

### F. Sparse Attention in DiT

Recently, many efficient DiT methods (Hassani et al., 2023; Xi et al., 2025; Zhang et al., 2025; Xu et al., 2025b; Ding et al., 2025) have adopted sparse attention to accelerate long video generation. We note that these methods can also benefit from permutation-based transformations to achieve kernel-efficient implementations. For example, the 2D/3D sliding window attention in NATTEN can be converted into dense tensor core computation via permutation, as illustrated in Fig. 15. Similarly, the temporal head in Sparse VideoGen (Xi et al., 2025) and the anti-diagonal structure in xAttention (Xu et al.,

LongVILA-1M w/ Tri-shape in MMNIAH Recall 73.8%

LongVILA-1M w/ A-shape in MMNIAH Recall 43.3%

1.0

1.0

[Figure 36]

[Figure 37]

0

0

0.8

0.8

DepthPercent(%)

DepthPercent(%)

20

20

0.6

0.6

40

40

Score

Score

60

60

0.4

0.4

80

80

0.2

0.2

100

100

0.0

0.0

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K

Num. of Frames

Num. of Frames

(a) A-shape

(b) Tri-shape

LongVILA-1M w/ MInference in MMNIAH Recall 88.0%

LongVILA-1M MMInference w/o Inter in MMNIAH Recall 88.0%

1.0

1.0

[Figure 38]

0

[Figure 39]

0

0.8

DepthPercent(%)

0.8

DepthPercent(%)

20

20

0.6

0.6

40

40

Score

Score

60

0.4

60

0.4

80

80

0.2

0.2

100

100

0.0

0.0

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K

3006009001.2K1.5K1.8K2.1K2.4K2.7K3.0K3.3K3.6K3.9K4.2K4.5K

Num. of Frames

Num. of Frames

(d) MMInference w/o Inter-modality Figure 14: Mixed-Modality Needle In A Haystack results using LongVila-Qwen2-7B-1M (Chen et al., 2025).

(c) MInference

2025b) can be restructured through permutation to enable sparse loading with dense computation, significantly speeding up DiT inference, especially in long-context scenarios.

### G. Kernel Implementation

As shown in Algorithms 5, 6, and 7, we provide implementation details of the FlashAttention-based kernels. The Grid-shape kernel in Algorithm 5 integrates block-sparse FlashDecoding (Qwen, 2023), which sparsifies the query loading, with block-sparse FlashAttention-2, which sparsifies the key loading. The Q-Boundary kernel in Algorithm 6 introduces sparsity

0

| |[Figure 40]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

200

400

600

800

1000

0 250 500 750 1000

0

| |[Figure 41]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

200

400

600

800

1000

0 250 500 750 1000

(a) Natten

(b) Permutated Natten

- Figure 15: Permutation-based implementation of 2D/3D sliding window attention (Hassani et al., 2023) enables efficient sparse attention optimization for DiT architectures.

A-shape

Tri-shape

Vertical-Slash FlashAttention-2 Grid

| |
|---|

| |
|---|

4000

2000

80

60

Latency(ms)

40

20

0

16K 32K 64K 128K 256K 512K 1M Context Windows

- Figure 16: The latency breakdown of a single attention kernel for four sparse attention patterns and FlashAttention (Dao, 2024) across different context windows in a single A100, including the index time for dynamic sparse approximation and building dynamic sparsity. At 1M tokens, the latency for Grid is 358ms.

along the query dimension using FlashAttention-2 (Dao, 2024), while the 2D-Boundary kernel in Algorithm 7 applies sparsity along both the query and key dimensions.

- Algorithm 5 Grid-Shape Flash Attention

# Sparse load in K using FlashAttention

#### Input: Q,K,V ∈ RS×d

h, block size B, stride size σ, query start index sq, key start index sk Scale τ ← d1

- for i ← 1 to N do

Load Qchip ← Q[i×B:(i+1)×B] ∈ RB×d

h

Initialize Ochip ← (0)B×d

h

∈ RB×d

h

Initialize m ← (−inf)B ∈ RB Initialize l ← (0)B ∈ RB

# Loop in K

- for j ← 1 to Mσ do Load Kchip ← K[j×B:(j+1)×B]×σ+σ×sk ∈ RB×dh Load Vchip ← V [j×B:(j+1)×B]×σ+σ×sk ∈ RB×dh

h

Initialize O ← (0)S×d

∈ RS×d

h

h

# Parallelized in GPU # Sparse load in Q using FlashDecoding

- for i ← 1 to Nσ do Load Qchip ← Q[i×B:(i+1)×B]×σ+sq ∈ RB×dh Initialize Ochip ← (0)B×dh ∈ RB×dh Initialize m ← (− inf)B ∈ RB Initialize l ← (0)B ∈ RB # Loop in K
- for j ← 1 to M do Load Kchip ← K[j×B:(j+1)×B] ∈ RB×dh Load Vchip ← V [j×B:(j+1)×B] ∈ RB×dh

S ← τQchipKchipT S ← mask(S)

minew ← max(mi, rowmax(S)) ∈ RB S ← S − minew P ← exp(S) lnewi ← rowsum(S)) α ← exp(mi − minew) li ← αli + lnewi Ochip ← αOchip + PVchip

S ← τQchipKchipT S ← mask(S)

minew ← max(mi, rowmax(S)) ∈ RB S ← S − minew P ← exp(S) lnewi ← rowsum(S)) α ← exp(mi − minew) li ← αli + lnewi Ochip ← αOchip + PVchip

end for # Write outputs Ochip ← diag(li)−1Ochip Save Oi ← Ochip

#### end for

end for # Write outputs Ochip ← diag(li)−1Ochip Save Oi ← Ochip

end for

- Algorithm 6 Q-Boundary Flash Attention

Algorithm 7 2D-Boundary Flash Attention Input: Q,K,V ∈ RS×d

#### Input: Q,K,V ∈ RS×d

h, block size B, modality index Im, sparse attention kernel Opm Scale τ ← d1

h, block size B, modality index Im, sparse attention kernel Opm Scale τ ← d1

h

h

Initialize O ← (0)S×d

Initialize O ← (0)S×d

∈ RS×d

∈ RS×d

h

h

h

h

# Loop modality and parallelized in GPU for m ∈ {text, vision, ..., } do

# Loop modality and parallelized in GPU for mq ∈ {text, vision, ..., } do

for i ← 1 to Nm do Load index Ichip ← Im[i×B:(i+1)×B] ∈ RB Load Qchip ← QIchip ∈ RB×dh Initialize Ochip ← (0)B×dh ∈ RB×dh Initialize m ← (− inf)B ∈ RB Initialize l ← (0)B ∈ RB # Loop in K using modality sparse attention Ochip, m, l ← Opm(Qchip, K, V , Ochip, m, l) # Write outputs w/ modality index Ochip ← diag(li)−1Ochip Save OiIchip ← Ochip

for i ← 1 to Nm,q do Load index Ichip,q ← Im,q[i×B:(i+1)×B] ∈ RB Load Qchip ← QIchip,q ∈ RB×dh Initialize Ochip ← (0)B×dh ∈ RB×dh Initialize m ← (− inf)B ∈ RB Initialize l ← (0)B ∈ RB # Loop in K and modality for mk ∈ {text, vision, ..., } do

for j ← 1 to Mm,k do Load index Ichip,k ← Im,k[j×B:(j+1)×B] ∈ RB Load Kchip ← KIchip,k ∈ RB×dh Load Vchip ← V Ichip,k ∈ RB×dh Ochip, m, l ← Opm(Qchip, Kchip, Vchip, Ochip, m, l)

end for end for

end for end for # Write outputs w/ modality index Ochip ← diag(li)−1Ochip Save OiIchip,q ← Ochip

end for end for

[Figure 42]

- (a) Qwen2.5-VL on EgoSchema

[Figure 43]

- (b) VideoChat on EgoSchema

[Figure 44]

- (c) Qwen2.5-VL on VideoMME

[Figure 45]

- (d) VideoChat on VideoMME

- Figure 17: Visualization of sparse attention patterns in Qwen2.5-VL with dynamic resolution input and VideoChat-Flash with visual token compression across different benchmarks.

[Figure 46]

- (a) Qwen2.5-VL on Mix-modality

[Figure 47]

- (b) VideoChat on Mix-modality

- Figure 18: Visualization of sparse attention patterns in Qwen2.5-VL with dynamic resolution input and VideoChat-Flash with visual token compression with mix-modality inputs.

