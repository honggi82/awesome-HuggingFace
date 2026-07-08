# arXiv:2512.07806v2[cs.CV]31May2026

## Multi-view Pyramid Transformer: Look Coarser to See Broader

Gyeongjin Kang1* Seungkwon Yang2* Seungtae Nam2 Younggeun Lee1 Jungwoo Kim2 Eunbyung Park2†

1Sungkyunkwan University 2Yonsei University

https://gynjn.github.io/MVP/

[Figure 1]

Figure 1. Our method efficiently processes a wide range of input views, reconstructing diverse large-scale scenes in under 0.1–2.0 seconds. We utilize Viser [73] for 3D scene visualization. Each marker in the plot represents performance under different numbers of input views: 16, 32, 64, 128, and 256 views for Ours, and 16, 32, and 64 views for both iLRM and Long-LRM. See Tab. 9 and 10 for further details.

### Abstract

We propose Multi-view Pyramid Transformer (MVP), a scalable multi-view transformer architecture that directly reconstructs large 3D scenes from tens to hundreds of images in a single forward pass. Drawing on the idea of “looking broader to see the whole, looking finer to see the details,” MVP is built on two core design principles: 1) a local-to-global inter-view hierarchy that gradually broadens the model’s perspective from local views to groups and ultimately the full scene, and 2) a fine-to-coarse intraview hierarchy that starts from detailed spatial represen-

*Equal contribution †Corresponding author

tations and progressively aggregates them into compact, information-dense tokens. This dual hierarchy achieves both computational efficiency and representational richness, enabling fast reconstruction of large and complex scenes. We validate MVP on diverse datasets and show that, when coupled with 3D Gaussian Splatting as the 3D representation, it achieves state-of-the-art generalizable reconstruction quality while maintaining high efficiency and scalability across a wide range of view configurations.

### 1. Introduction

Transformer architectures [59] have profoundly reshaped the landscape of modern artificial intelligence. Its selfattention mechanism enables effective modeling of long-

range dependencies, leading to significant advances across diverse domains such as natural language processing [8, 32], computer vision [9, 16], and multimodal learning [35, 49]. This architectural paradigm has further demonstrated remarkable success in 3D vision tasks, including 3D reconstruction [25, 60, 61] and novel view synthesis [17, 30].

Recent methods utilizing transformer-based architectures for 3D vision tasks [57, 66, 77] effectively learn complex spatial relationships and 3D representations in a data-driven manner, in contrast to traditional geometrybased pipelines [19, 51]. These approaches often reformulate 3D reconstruction as a multi-view 2D reasoning problem, where a set of multi-view input images is tokenized and processed as a sequence of tokens. This sequenceto-sequence formulation allows transformer architectures to model global geometric relationships across views and infer consistent scene representations. However, as each highresolution image contributes a large number of tokens, the sequence length grows rapidly with the number of input views, leading to substantial computational and memory overhead due to the quadratic complexity of self-attention. This scalability issue poses a major challenge in applying transformer-based architectures to large-scale 3D reconstruction tasks involving many high-resolution images.

Motivated by these limitations, a growing body of work has introduced architectural designs to improve the efficiency and scalability of multi-view transformer models. Long-LRM [81] proposed to integrate the transformer and bidirectional Mamba [14] blocks. While improving efficiency through Mamba’s linear-complexity design, its expressive capacity remains limited compared to the transformer’s self-attention. iLRM [31] employs a compact scene representation, which enables full attention across all input views. Although it achieves promising reconstruction quality and improved scalability, its reliance on global attention introduces a computational bottleneck as the number of input views increases. LVT [28] adopts a local-view attention mechanism to improve scalability, where each input view attends only to nearby views. However, because attention is restricted to neighboring views, global 3D consistency is achieved only indirectly through multiple layers of local interactions. Moreover, defining neighborhood relationships between input views is itself a challenging problem, and the reliance on known camera poses further limits the flexibility and applicability of the architecture.

In this work, we introduce Multi-view Pyramid Transformer (MVP), a scalable multi-view pyramid transformer architecture that substantially improves both scalability and overall performance compared to existing approaches. The key intuition behind the proposed architecture is ‘looking broader to see the whole, looking finer to see the details’, which has been successfully adopted in many contemporary architectures. Early convolutional networks [22, 39, 56]

evolve feature representations from fine-grained spatial feature maps in the initial layers to coarse, semantically rich maps in the later layers, capturing global context while reducing computational cost. A comparable architectural design also emerges in modern transformer architectures, such as the Swin Transformer [42, 43], which progressively reduces the spatial and temporal resolutions of image tokens across layers. Building on this fine-to-coarse design philosophy, we introduce Dual Attention Hierarchy, which enables scalable multi-view reasoning by structuring attention hierarchically along two complementary dimensions: interview and intra-view hierarchy.

(1) In the inter-view hierarchy, the attention window progressively expands to cover a broader range of input views, enabling the model to reason from neighboring inter-view relationships to global scene context. (2) In the intra-view hierarchy, the spatial image tokens are gradually merged across network layers to enlarge the receptive field within each input view, allowing the model to capture visual cues across multiple scales from fine details to coarse structures. In the early layers, the model operates on narrower view windows and fine-grained image tokens to efficiently extract local geometric details, while the later layers adopt wider view windows and coarse-grained image tokens to integrate broader contextual information. The two hierarchies operate in complementary directions, local-to-global in the inter-view hierarchy and fine-to-coarse in the intra-view hierarchy, achieving a balanced trade-off between computational efficiency and representational richness.

Beyond computational efficiency, there is another fundamental reason motivating our architectural design. Although full global attention theoretically enables all-to-all interactions across input views, it often fails to generalize in long-context settings [27, 41, 70]. As the number of input views increases, the attention distribution becomes diluted and unstable, leading to degraded correspondence learning and inconsistent feature alignment. In multi-view 3D reconstruction, this limitation manifests as diminishing performance gains when scaling to more input views, reflecting the difficulty of maintaining consistent geometric correspondence under long-range attention. In contrast, our Dual Attention Hierarchy prevents the number of tokens involved in attention from growing excessively across layers, thereby avoiding attention dilution, stabilizing optimization in long-context regimes, and ensuring consistent multi-view reasoning as the number of input views increases.

We validate the effectiveness of the proposed architecture within the feed-forward 3D Gaussian Splatting (3DGS) framework. The proposed model efficiently processes up to 128 input views at a resolution of 960×540 in under one second on a single H100 GPU, significantly outperforming the previous approaches in both inference speed and reconstruction quality (Fig. 1). Remarkably, it often achieves

superior reconstruction quality to optimization-based methods such as 3D-GS in both sparse and dense-view settings, even without any additional post-processing. These results demonstrate that our method not only ensures high scalability but also delivers enhanced reconstruction performance compared to existing feed-forward 3DGS models. To summarize, our main contributions are as follows,

- • We propose MVP, a novel scalable multi-view pyramid transformer architecture that significantly improves both efficiency and performance.
- • We introduce a dual hierarchical attention mechanism that jointly operates inter-view and intra-view attentions, enabling efficient multi-view reasoning.
- • We conduct extensive experiments demonstrating that our method achieves state-of-the-art performance while maintaining remarkable efficiency and scalability across diverse datasets and view configurations.
- • We analyze the proposed architecture from multiple perspectives, revealing how our design effectively achieves both global and local 3D consistency.

### 2. Related Work

#### 2.1. Multi-view transformers

Recent progress in multi-view transformers, with DUSt3R [61] serving as a notable milestone, has established a powerful paradigm for learning 3D geometry directly from multi-view imagery. This progress has spurred a range of methods that tackle different facets of 3D reconstruction. The first category comprises 3D geometry reconstruction approaches [60–62, 67] that jointly estimate camera poses and scene structure in a feed-forward manner, often predicting dense point maps or depth fields in a single pass. The second category includes large view synthesis models [17, 29, 30] that prioritize photorealistic novel view generation, typically emphasizing visual fidelity and scalability rather than explicitly recovering 3D geometry. The third category encompasses large reconstruction models [28, 31, 64, 77, 81], which leverage high-capacity neural networks and explicit or implicit 3D representations [34, 45] to reconstruct large-scale scenes while also enabling photorealistic novel view synthesis. Although the aforementioned works differ in goals and architectures, they largely incorporate multi-view attention, a mechanism that naturally facilitates correspondence reasoning across input views, an essential component of 3D reconstruction. However, multi-view attention remains computationally expensive and struggles to scale to a large number of input views. In this work, we propose multi-view pyramid attention architecture that advances the field toward more scalable large-scale 3D reconstruction.

- 2.2. Efficient sequence models To model long-range dependencies, transformer-based architectures have become the dominant paradigm across diverse domains, including natural language processing [8, 59], computer vision [16, 42], and video understanding [1, 6]. However, the quadratic complexity of global selfattention poses significant computational and memory challenges, particularly for high-resolution inputs or long sequences. Efforts to improve efficiency generally follow two directions: (1) modifying the attention mechanism, by restricting attention to sparse patterns or local windows [5, 12, 42, 74, 78] or linearizing to sub-quadratic complexity [14, 18, 33]; and (2) optimizing the token representation, by compressing or merging tokens [7, 65, 69, 81] to reduce sequence length without losing essential information. Previous work [20, 26, 76] adopts hierarchical architectures to enhance efficiency, but their hierarchies are confined to individual spatial domains. Moreover, temporal downsampling [43] is not straightforward in the multi-view setting, so we instead reduce the spatial token resolution to accommodate more views. Meanwhile, Long-LRM [81] and iLRM [31] introduce efficient multi-view transformers for 3D reconstruction, but they still rely on the same full global attention mechanism across all input views, leading to high computational cost and limited scalability.

The proposed approach explores a different design philosophy by introducing a Dual Attention Hierarchy that operates jointly along view and spatial dimensions. This hierarchical formulation enables the model to capture both local geometric cues and global multi-view relationships in a scalable and computationally efficient manner. Different from [43], which operates in a temporal video setting, our model works in a multi view setting and progressively increases the number of views, analogous to temporal length in videos, whereas they do not progressively extend the temporal dimension in different stages. We also employ Alternative Attention [60] to further improve efficiency when processing more than two views simultaneously.

- 3. Method

We introduce MVP, a scalable multi-view dense prediction transformer architecture capable of processing more than one hundred input-view images in under one second. The core of our model is the Dual Attention Hierarchy, which progressively integrates contextual and spatial information through inter-view and intra-view hierarchies. While our formulation is general and applicable to a wide range of multi-view tasks, in this work, we instantiate it within the feed-forward 3D Gaussian Splatting (3DGS) framework, enabling efficient and high-quality 3D reconstruction.

#### 3.1. Overall Architecture

Our MVP transformer architecture (Fig. 2) comprises an input tokenizer (Sec. 3.2) followed by a three-stage hier-

[Figure 2]

- Figure 2. Architecture Overview. Given tokenized inputs, our model applies a three stage hierarchy of alternating attention blocks, varying in both self-attention coverage and token resolution. A Pyramidal Feature Aggregation module fuses the outputs from all stages, which are then passed to a final head for dense prediction.

archy of attention blocks (Sec. 3.3.1), with token reduction modules interleaved between each stage (Sec. 3.3.2). A pyramidal feature aggregation module then collects the outputs from each stage, fusing them into a unified feature (Sec.3.4). Finally, a decoder processes the fused representation to produce the dense multi-view predictions (Sec. 3.5).

#### 3.2. Input Encoding

Given a set of N input images {Ii}Ni=1, where Ii ∈ RH×W×3, we begin by tokenizing the input views, which are subsequently processed by a series of transformer layers. To incorporate camera geometry, we encode the known camera poses as 9D Pl¨ucker ray map, Pi ∈ RH×W×9, where each ray is represented by concatenating its origin, direction, and their cross product [79]. Then, each input image Ii is concatenated with its corresponding ray map Pi along the channel dimension, forming a 12-channel posedimage tensor I˜i ∈ RH×W×12. This posed image tensor is then patchified into non-overlapping patches of size p through a linear projection layer. For each input view, we further append four register tokens [15, 55, 60], resulting in a total of N(HW/p2 + 4) tokens that serve as input to the subsequent transformer stages.

- 3.3. Dual Attention Hierarchy MVP employs a dual-level attention hierarchy consisting of inter-view and intra-view attention mechanisms. In our current implementation, this hierarchy comprises three stages, progressively integrating multi-view contextual information from local to global and visual cues from fine- to coarsegrained scales in a computationally efficient manner.

##### 3.3.1. Inter-view Attention Hierarchy

We introduce a group-wise self-attention mechanism that attends to tokens within predefined groups of views, serving as an efficient intermediate stage between purely local (frame-wise) and fully global attention. This formulation enables scalable multi-view interaction while pre-

serving global consistency. Our framework also generalizes the Alternating-Attention (AA) module [60] as a special case, encompassing frame-wise, group-wise, and global self-attention within a unified framework.

- Stage 1: The first stage focuses on extracting spatial features and local geometric details from each frame independently, employing only frame-wise attention.
- Stage 2: The second stage begins to reason about interview relationships utilizing group-wise self-attention. First, we apply a grouping operator, group(·) : RNhw×d → RMN ×Mhw×d, where M is the number of views within a group (h,w is the downsampled token resolution, see Sec. 3.3.2). In this work, the grouping operator simply partitions the N views into MN consecutive groups by the locality of frame indices. Then, we perform frame-wise selfattention, followed by a group-wise self-attention.

G ← group(T), Gi,j ← self-att(Gi,j) ∀i,j (frame-wise attention), Ti ← self-att(Gi) ∀i (group-wise attention),

(1)

where T ∈ RNhw×d represents all input tokens, Gi,j ∈ Rhw×d denotes the tokens that correspond to the j-th image in the i-th group, and Gi ∈ RMhw×d is every token in the i-th group (we omit the register tokens for brevity).

- Stage 3: At the final stage, the model integrates information across all views to form a global understanding of the scene and construct a fully coherent 3D representation. In this stage, the architecture simplifies to the AA module, where all views belong to the same single group (M == N).

##### 3.3.2. Intra-view Attention Hierarchy

The intra-view fine-to-coarse hierarchy operates along the spatial dimensions of each input frame. This scaling strategy enables the model to transition from fine-grained local reasoning in early layers to coarse-grained global reasoning

in later layers within each image. We implement this using a single convolution layer between stages that performs spatial downsampling and channel up-projection simultaneously. At each stage, the number of image tokens is reduced by a factor of four (h → h2 and w → w2 ), enlarging the effective receptive field of each token. Meanwhile, the token embedding dimension is doubled, increasing feature capacity to encode information from this expanded spatial region. This progressive token reduction not only forms a feature pyramid, but also allows larger group-wise attention as the spatial resolution decreases.

- 3.4. Pyramidal Feature Aggregation To perform dense multi-view prediction, we introduce a Pyramidal Feature Aggregation (PFA) module that integrates multi-scale and multi-stage features produced by our dual attention hierarchy. While conceptually related to the Dense Prediction Transformer [50], the proposed PFA is specifically tailored to the hierarchical structure of our architecture, enabling effective fusion across the three stages.

Concretely, multi-view feature tokens from all stages are collected and fused through a top-down refinement process. First, we reshape the tokens from all stages into spatial feature maps, then a convolutional layer is applied to project them into a shared high-dimensional space while preserving spatial resolution. The resulting feature maps are progressively upsampled and fused with those from preceding stages through residual convolutional fusion blocks, allowing the model to combine coarse global context with fine local details. Finally, the aggregated feature tensor is rearranged back into a token sequence and passed to the decoder for dense 3D Gaussian prediction. More formally, the aggregated feature F can be obtained as follows,

F = fuse(up(fuse(up(F(3)) + F(2))) + F(1)), (2)

where F(1),F(2),F(3) are the reshaped output feature maps from each stage, up(·) and fuse(·) denote an upsampling layer and feature fusion, respectively.

- 3.5. Output Decoding Given the aggregated tokens, we apply a single linear layer for dense prediction. In our feed-forward 3DGS setting, each output pixel parameterizes a 3D Gaussian primitive

with attributes (µj,sj,qj,αj,cj) for position, scale, rotation (quaternion), opacity, and color, respectively. Following prior works [28, 47], our model predicts spherical harmonic coefficients for both view-dependent color and opacity, enabling accurate modeling of appearance variations across viewing directions.

- 3.6. Training Objective By rendering novel view images from the predicted Gaussian primitives using the target camera poses, we compare the rendered target view images Iˆi with ground-truth target

view images Ii to compute and minimize the loss. We use a weighted sum of MSE loss and perceptual loss.

1 |T | i∈T

(LMSE(Iˆi,Ii) + λLpercept(Iˆi,Ii)), (3)

Limg =

where T is a set of target view indices, and we set λ = 0.2 as the weighting factor throughout the paper. We also regularize view-dependent splat opacity to mitigate undesirable opacity artifacts across different viewpoints [28]. Let NG (NG = NHW for per-pixel Gaussians) be the number of predicted Gaussian primitives, and we compute the regularizing factor Rα as

NG

1 NG

|σ(αj · ωj)|, (4)

Rα =

j=1

where σ(·) is the Sigmoid function and ωj the spherical harmonic basis from the randomly sampled per-pixel view direction. Our overall loss function is L = Limg+γRα, where we set γ = 0.001 as the weighting factor.

### 4. Experiments

Dataset. We conduct experiments on the DL3DV dataset [40], which consists of large-scale multi-view scenes spanning diverse indoor and outdoor environments. Each scene is processed using COLMAP [51, 52] to obtain accurate camera poses, containing approximately 250–350 frames per scene. Following prior works [31, 81], we adopt the official training and benchmark splits provided by the dataset. We also evaluate the zero-shot generalization performance on the Tanks&Temples [36] and MipNeRF360 [4] datasets, which contain high-quality realworld scenes captured under challenging lighting and geometry conditions. For all datasets, we use images at a resolution of 960×540 and evaluate with different numbers of input views, ranging from 16 to 256. We further evaluate under a low resolution setting (256×256) on the RealEstate10K (RE10K) dataset [80], following the same train-test split and indices protocol in [10, 31].

Implementation details. We utilize 2 frame-wise, 4 groupwise, and 8 global attention blocks for each stage. Tokens initially correspond to 8×8 patches and are progressively merged to 16×16 and 32×32 across stages, with embedding sizes 256, 512, and 1024. We also apply PRoPE [37] to encode cameras as relative positional signals, providing geometry aware conditioning for our attention layers.

Training details. We adopt a three stage training schedule, starting at low resolution with a fixed number of views and progressing to high resolution with mixed input view counts. In the first stage, we train the model at 480×256 with 32 input views and 12 target views. In the second stage, we switch to 960×540 while keeping 32 input views and reducing to 6 target views. In the final stage, we continue to train at 960×540 with varying numbers of input

[Figure 3]

- Figure 3. Qualitative results on the DL3DV (top two rows), Tanks&Temples (third row), and Mip-NeRF360 (bottom row). For a fair and reliable comparison, we evaluate all methods with 32 input views, matching the training setup used for other feed-forward baselines.

16 views 32 views 64 views 128 views PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓

Method

3D-GS30k [34] 21.48 0.753 0.252 8min 24.43 0.827 0.191 8min 27.34 0.883 0.146 8min 29.43 0.914 0.123 8min Long-LRM [81] 21.05 (22.66) 0.708 (0.740) 0.297 (0.292) 0.50 23.97 0.778 0.267 0.84 23.60 0.789 0.260 2.08 21.24 0.739 0.308 6.39 iLRM [31] 21.92 (22.91) 0.748 (0.766) 0.316 (0.295) 0.19 24.30 0.803 0.256 0.53 24.44 0.819 0.240 1.66 22.98 0.807 0.249 5.61 Ours 23.76 0.798 0.239 0.09 25.96 0.847 0.187 0.17 27.73 0.881 0.154 0.36 29.02 0.903 0.134 0.77

- Table 1. Quantitative comparisons on the DL3DV dataset with varying numbers of input views. The values in parentheses (·), for the 16-view setting are taken from the original papers, where the models are trained with a fixed 16-view input. For all metrics, we report the results by re-evaluating the models from their 32-view checkpoints.

Method

192 views 256 views PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ 3D-GS30k [34] 30.14 0.924 0.116 8min 30.39 0.926 0.114 8min Long-LRM [81] OOM (GPU memory limit exceeded, 80GB) iLRM [31] 21.68 0.788 0.264 11.91 20.63 0.767 0.281 20.92 Ours 29.54 0.912 0.127 1.23 29.67 0.915 0.125 1.84

- Table 2. Quantitative comparisons on the DL3DV dataset with varying numbers of input views.

Method Views

Tanks & Temples Mip-NeRF360

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Long-LRM [81]

32

18.59 0.614 0.366 21.08 0.484 0.445 iLRM [31] 18.58 0.631 0.385 21.09 0.495 0.466 Ours 19.54 0.708 0.277 22.21 0.587 0.355

Long-LRM

64

19.44 0.651 0.334 21.30 0.499 0.431 iLRM 19.82 0.692 0.318 21.60 0.522 0.444

- Ours 21.24 0.761 0.221 23.72 0.656 0.302

Long-LRM

128

18.47 0.613 0.375 19.82 0.484 0.457 iLRM 19.22 0.696 0.319 21.32 0.551 0.424

- Ours 22.36 0.804 0.184 25.12 0.736 0.248

- Table 3. Quantitative comparisons on the Tanks&Temples and Mip-NeRF360 under a zero-shot setting. For the 128 view setting on Mip-NeRF360, the stump and treehill scenes are excluded due to their limited number of frames.

views, selecting the number of target views according to GPU memory constraints. In this stage, we freeze the parameters of the first two architectural stages (frame- and group-wise blocks) and update only the global modules. Each training stage takes approximately 4, 3, and 2 days, respectively, using 32 H100 GPUs.

#### 4.1. Results

Throughout the manuscript, inference times are measured on a single NVIDIA H100 GPU with FlashAttention3 [54]. Comparison. Tab. 9 and 10 report quantitative comparisons on the DL3DV benchmark dataset. We compare our method with the optimization based 3D Gaussian Splatting [34] with 30K optimization steps and two recent feed-forward reconstruction methods, Long-LRM [81] and iLRM [31]. Across all evaluated settings from 16 to 256 input views, our method consistently outperforms previous single-pass models by a large margin in both reconstruction quality and inference efficiency. Note that, unlike our training scheme, the baselines are trained only with 32 input views, and do not exhibit scalability as the number of views increases. Also, in the very dense 256-view setting, our approach re-

mains within 0.7 dB PSNR of the optimization-based baseline, but over 250× faster, while Long-LRM fails to run due to out-of-memory (OOM) issues. In Tab. 3, we report generalization performance with Tanks&Temples and MipNeRF360 dataset. For all tested view counts (32, 64, and 128), our method consistently exceeds the baselines on all the evaluation metrics and the gap increases as the number of input views increases.

Tab. 4 summarizes the results on the low resolution (256 × 256) RE10K dataset with 4 and 8 input views. We adopt prior methods [31, 63] as baselines. For [63], we select the model that performs best among the 4-view configurations. Our coarse variant of patch sizes (8, 16, 32) already outperforms the baselines, while our fine variant of patch sizes (4, 8, 16) further improves the rendering quality. Note that our models are trained with varying numbers of input views, unlike the baselines.

4 views 8 views PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Method

CLiFT [63] 30.13 0.916 0.095 29.68 0.910 0.092 iLRM [31] 30.37 0.923 0.095 28.90 (31.57) 0.920 (0.935) 0.102 (0.082) Ours-coarse 30.56 0.925 0.087 31.78 0.939 0.077 Ours-fine 32.12 0.941 0.077 33.40 0.952 0.066

- Table 4. Quantitative comparisons on the RE10K dataset. Baselines are evaluated using 4-view models, and values in parentheses (·) are from the models trained for that specific view setting.

Attention visualization. To investigate how our method achieves both local (group) and global multi-view consistency, we visualize the learned attention patterns across views and stages, highlighting how the model adaptively attends to geometrically consistent regions within group and across the entire views. Using the first frame as the reference view, we first select three query patches. For each query patch, we then visualize its top-3 attended tokens from other views in different stages. At stage 2, we show tokens within the same group, while at stage 3, we show tokens from both within and outside the group. As shown in Fig. 4, the group-wise attention over fine tokens effectively captures local spatial correspondence, while the global attention over coarse tokens still focuses on semantically and geometrically consistent regions across distant views.

Longer context generalization. Tab. 5 evaluates how well each model generalizes to longer input contexts. All baselines, including our method, are only trained with 32 input views and then tested on 32 (seen), 40, and 48 views. Our approach not only achieves the best evaluation metrics at the training length, but also continues to improve most when moving from 32 to 40 and 48 views, whereas LongLRM and iLRM show much smaller gains and tend to saturate. At the same time, the inference time of our model increases only slightly with more views and remains substantially lower than the baselines. These results support our claim that the proposed hierarchical design alleviates attention dilution in long context regimes and maintains sta-

ble multi-view reasoning when extrapolating to longer input contexts beyond the training range.

### 5. Ablation and Analysis

We conduct ablation experiments in Tab. 6 to validate the proposed architectural components, with a particular focus on reducing computational cost and improving scalability while maintaining comparable reconstruction quality.

- 1) Feature aggregation. To validate our Pyramidal Feature Aggregation (PFA), we tested a variant that decodes only from the final, coarse feature map. This variant showed degraded reconstruction quality, highlighting the importance of the PFA’s multi-scale fusion for generating fine-grained representations, especially comparing LPIPS metric.
- 2) Group-wise attention. We ablate the group-wise attention by constructing two variants: one where all groupwise blocks are replaced with frame-wise blocks, and another where they are replaced with global blocks. The results exhibit a clear computation–performance trade-off: while global attention attains the best accuracy at the cost of higher complexity, our group-wise attention recovers most of the performance gains with lower computational overhead. Moreover, as our group size is fixed to four views, the computational cost of the global variant grows much more rapidly as the number of input views increases (Fig. 5a).
- 3) Dual hierarchy. We further analyze our dual-attention hierarchy by comparing variants without the intra-view hierarchy, without the inter-view hierarchy, and without both components. Note that our variant without both components is analogous to a modified version of VGGT [60], using a linear patch-embedding layer instead of a DINO [48] encoder and omitting camera tokens. As shown in Fig. 5b, the hierarchies are critical for scalability. In particular, for 256 input images, removing the inter-view hierarchy results in more than 6× higher latency. Moreover, the variants without the intra-view hierarchy and without both intraand inter-view hierarchies encounter out-of-memory errors at 256 views; even at 64 input images, they are approximately 50× and 80× slower than our method, respectively. Notably, compared to the variant that uses a patch size of 16 without dual hierarchy, our method achieves a lower computational cost while permitting a finer, higher-performance patch size compared to non-hierarchical alternatives.
- 4) Reversed hierarchy. For the reversed variant, we change the intra-view hierarchy to apply attention in the order of global, group, and frame, and modify the inter-view branch to start from the lowest-resolution features and progressively upsample. This design yields clearly lower performance, confirming the effectiveness of our design.
- 5) Redundancy pruning. We prune Gaussian primitives whose view-dependent opacity falls below 0.01 at the target view during rendering. As shown in Tab. 7, this substantially reduces the number of active primitives.

[Figure 4]

- Figure 4. Attention visualization. For colored query patches (red, yellow, green) in the reference view, we highlight top-3 attended tokens: on the left, tokens attended within the group (blue overlay), and on the right, tokens attended within and outside the group (green overlay).

Method

32 views (Trained) 40 views (Unseen) 48 views (Unseen) PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓

Long-LRM [81] 23.97 0.778 0.267 0.84 24.18 (0.21) 0.787 0.260 1.05 (0.21) 24.30 (0.33) 0.797 0.252 1.38 (0.54) iLRM [31] 24.30 0.803 0.257 0.53 24.54 (0.24) 0.811 0.248 0.76 (0.23) 24.78 (0.48) 0.820 0.240 1.04 (0.51) Ours 25.88 0.845 0.188 0.17 26.36 (0.48) 0.855 0.178 0.22 (0.05) 27.06 (1.18) 0.870 0.164 0.26 (0.09)

Table 5. Longer context generalization. values in parentheses (·) denote the improvements over 32 views.

Method Stage 1 Stage 2 Stage 3 Patch Size Hidden Dim. Feature Agg. PSNR ↑ SSIM ↑ LPIPS ↓ Baseline Frame(2) Group(4) Global(8) (8, 16, 32) (128, 256, 512) o 22.79 0.733 0.235 w/o Feature Agg. Frame(2) Group(4) Global(8) (8, 16, 32) (128, 256, 512) x 21.58 0.646 0.340

w/o Group-attn.

Frame(2) Frame(4) Global(8)

(8, 16, 32) (128, 256, 512) o

22.53 0.720 0.247 Frame(2) Global(4) Global(8) 22.94 0.739 0.235

w/o Inter-view Hierarchy Global(2) Global(4) Global(8) (8, 16, 32) (128, 256, 512) o 22.94 0.739 0.236 w/o Intra-view Hierarchy Frame(2) Group(4) Global(8) (8, 8, 8) (512, 512, 512) o 22.83 0.732 0.249 w/o Dual Hierarchy (p=8) Global(2) Global(4) Global(8) (8, 8, 8) (512, 512, 512) o 23.20 0.747 0.241 w/o Dual Hierarchy (p=16) Global(2) Global(4) Global(8) (16, 16, 16) (512, 512, 512) o 21.80 0.651 0.341

Reversed Hierarchy Global(8) Group(4) Frame(2) (32, 16, 8) (512, 256, 128) o 18.95 0.442 0.555

Table 6. Ablations. We train all variants for 100K iterations at a reduced resolution (256×256) on the DL3DV dataset.

16 32 64 128 192 256

# Input Views

- 0

- 1

- 2

- 3

Time(sec)

Baseline

replaced w/ Frame replaced w/ Global

(a)

16 32 64 128 192 256

# Input Views

0

20

40

60

Time(sec)

Baseline w/o Inter w/o Intra w/o Dual (p=8)

w/o Dual (p=16)

(b)

- Figure 5. Ablation and inference time analysis of MVP’s components. (a) Inference time analysis of group-attention mechanism. (b) Inference time analysis of the dual hierarchy components.
- 6. Conclusion and discussions We introduce MVP, a highly scalable and efficient transformer architecture for multi-view reasoning. The core of our approach is a novel dual attention hierarchy on two complementary axes: an inter-view hierarchy that simultaneously expands its attention scope from local per-frame reasoning, through intermediate group-wise attention, to a

# views PSNR diff. SSIM diff. LPIPS diff. Avg. # GS Avg. pruned ratio Render time (s)

32 -0.02 -0.01 -0.01 3,392,471 79.7% 0.003 64 -0.06 0 -0.01 5,347,737 84.0% 0.005 128 -0.15 0 0 8,556,380 87.2% 0.008 256 -0.15 0 -0.01 14,572,584 89.1% 0.012

Table 7. Rendering analysis after redundancy pruning. Rendering times are measured on a RTX 4090 GPU and correspond to the time required to render a single image.

full global context, and an intra-view hierarchy that progressively aggregates spatial features. By fusing features from all stages via a pyramidal feature aggregation module, our model produces high-fidelity results with remarkable inference speed, enabling high-quality multi-view 3D reconstruction from a large number of input views in a single pass. While our current experiments focus on the posed, feed-forward static 3D reconstruction, we believe MVP can be extended to dynamic scenes or visual geometry related tasks with minimal architectural changes.

### Acknowledgements

This research was supported by the Ministry of Science and ICT (MSIT) of Korea, under the National Research Foundation (NRF) grant (RS-2024-00337548); the Korea-US Collaborative Research Fund (KUCRF), funded by the Ministry of Science and ICT and Ministry of Health & Welfare, Republic of Korea (RS-2024-00468417); Samsung Research Funding & Incubation Center of Samsung Electronics under Project Number SRFC-IT2401-01; and the Artificial Intelligence Industrial Convergence Cluster Development Project funded by the Ministry of Science and ICT (MSIT, Korea) & Gwangju Metropolitan City. This work was also supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT): No. RS-2024-00457882, AI Research Hub Project; No. RS-2025-25441838, Development of a Human Foundation Model for Human-Centric Universal Artificial Intelligence and Training of Personnel; and No. RS-2020-II201361, Artificial Intelligence Graduate School Program (Yonsei University).

### References

- [1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luˇci´c, and Cordelia Schmid. Vivit: A video vision transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6836–6846, 2021.
- [2] Dejan Azinovi´c, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. Neural rgb-d surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6290– 6301, 2022.
- [3] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
- [4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022.
- [5] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.
- [6] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Icml, page 4, 2021.
- [7] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.
- [8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [9] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Pro-

- ceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [10] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19457–19467, 2024.
- [11] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.
- [12] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
- [13] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning, 2023. arXiv preprint arXiv:2307.08691, 2023.
- [14] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML), 2024.
- [15] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023.
- [16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
- [17] John Flynn, Michael Broxton, Lukas Murmann, Lucy Chai, Matthew DuVall, Cl´ement Godard, Kathryn Heal, Srinivas Kaza, Stephen Lombardi, Xuan Luo, et al. Quark: Real-time, high-resolution, and general neural view synthesis. ACM Transactions on Graphics (TOG), 43(6):1–20, 2024.
- [18] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First conference on language modeling, 2024.
- [19] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003.
- [20] Ali Hatamizadeh, Vishwesh Nath, Yucheng Tang, Dong Yang, Holger R Roth, and Daguang Xu. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In International MICCAI brainlesion workshop, pages 272–284. Springer, 2021.
- [21] David Haynes, Steven Corns, and Ganesh Kumar Venayagamoorthy. An exponential moving average algorithm. In 2012 IEEE Congress on Evolutionary Computation, pages 1–8, 2012.
- [22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- [23] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
- [24] Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers. arXiv preprint arXiv:2010.04245, 2020.

- [25] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. LRM: Large reconstruction model for single image to 3d. In ICLR, 2024.
- [26] Teng Hu, Jiangning Zhang, Zihan Su, and Ran Yi. Ultragen: High-resolution video generation with hierarchical attention. arXiv preprint arXiv:2510.18775, 2025.
- [27] Nam Hyeon-Woo, Kim Yu-Ji, Byeongho Heo, Dongyoon Han, Seong Joon Oh, and Tae-Hyun Oh. Scratching visual transformer’s back with uniform attention. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5807–5818, 2023.
- [28] Tooba Imtiaz, Lucy Chai, Kathryn Heal, Xuan Luo, Jungyeon Park, Jennifer Dy, and John Flynn. Lvt: Largescale scene reconstruction via local view transformers. arXiv preprint arXiv:2509.25001, 2025.
- [29] Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, and Georgios Pavlakos. Rayzer: A selfsupervised large view synthesis model. arXiv preprint arXiv:2505.00702, 2025.
- [30] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. In ICLR, 2025.
- [31] Gyeongjin Kang, Seungtae Nam, Seungkwon Yang, Xiangyu Sun, Sameh Khamis, Abdelrahman Mohamed, and Eunbyung Park. ilrm: An iterative large 3d reconstruction model. arXiv preprint arXiv:2507.23277, 2025.
- [32] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [33] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020.
- [34] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
- [35] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.
- [36] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017.
- [37] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. Advances in Neural Information Processing Systems, 2025.
- [38] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

- [39] Tsung-Yi Lin, Piotr Doll´ar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017.
- [40] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.
- [41] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12: 157–173, 2024.
- [42] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.
- [43] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202–3211, 2022.
- [44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [45] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.
- [46] Nithin Gopalakrishnan Nair, Srinivas Kaza, Xuan Luo, Vishal M Patel, Stephen Lombardi, and Jungyeon Park. Scaling transformer-based novel view synthesis with models token disentanglement and synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 28567–28576, 2025.
- [47] Mateusz Nowak, Wojciech Jarosz, and Peter Chin. Vod3dgs: View-opacity-dependent 3d gaussian splatting. arXiv preprint arXiv:2501.17978, 2025.
- [48] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [50] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of

- the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.
- [51] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [52] Johannes Lutz Sch¨onberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV), 2016.
- [53] Thomas Sch¨ops, Johannes L. Sch¨onberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2538–2547, 2017.
- [54] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37: 68658–68685, 2024.
- [55] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.
- [56] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
- [57] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1–18. Springer, 2024.
- [58] S. Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(4):376–380, 1991.
- [59] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [60] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.
- [61] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024.
- [62] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning, 2025.
- [63] Zhengqing Wang, Yuefan Wu, Jiacheng Chen, Fuyang Zhang, and Yasutaka Furukawa. Clift: Compressive lightfield tokens for compute-efficient and adaptive neural rendering. arXiv preprint arXiv:2507.08776, 2025.

- [64] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for highquality mesh. arXiv preprint arXiv:2404.12385, 2024.
- [65] Heming Xia, Chak Tou Leong, Wenjie Wang, Yongqi Li, and Wenjie Li. Tokenskip: Controllable chain-of-thought compression in llms. arXiv preprint arXiv:2502.12067, 2025.
- [66] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. In European Conference on Computer Vision, pages 1–20. Springer, 2024.
- [67] Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [68] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [69] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19792–19802, 2025.
- [70] Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, and Furu Wei. Differential transformer. arXiv preprint arXiv:2410.05258, 2024.
- [71] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1–17, 2025.
- [72] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the International Conference on Computer Vision (ICCV), 2023.
- [73] Brent Yi, Chung Min Kim, Justin Kerr, Gina Wu, Rebecca Feng, Anthony Zhang, Jonas Kulhanek, Hongsuk Choi, Yi Ma, Matthew Tancik, and Angjoo Kanazawa. Viser: Imperative, web-based 3d visualization in python, 2025.
- [74] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020.
- [75] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 2019.
- [76] Dong Zhang, Hanwang Zhang, Jinhui Tang, Meng Wang, Xiansheng Hua, and Qianru Sun. Feature pyramid transformer. In European conference on computer vision, pages 323–339. Springer, 2020.

- [77] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, pages 1–19. Springer, 2024.
- [78] Qingru Zhang, Dhananjay Ram, Cole Hawkins, Sheng Zha, and Tuo Zhao. Efficient long-range transformers: You need to attend more, but not necessarily at every layer. arXiv preprint arXiv:2310.12442, 2023.
- [79] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025.
- [80] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018.
- [81] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Longsequence large reconstruction model for wide-coverage gaussian splats. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4349–4359, 2025.

## Multi-view Pyramid Transformer: Look Coarser to See Broader Supplementary Material

### A. Additional Details.

Datasets. We adopt the official DL3DV [40] split for benchmark dataset. In total, we use 9,995 scenes for training and 140 scenes for benchmarking, with no overlap between the two splits. For zero-shot inference, we use the train and truck scenes in Tanks&Temples [36], following [34, 81], and 9 scenes in Mip-NeRF360 [4] dataset. Considering our evaluation resolution of 960×540, we use downsampled images (from the original images) whose resolution is closest to, but not smaller than, this target resolution (Tab. 8).

scene name downsample scene name downsample bicycle 4 room 2

bonsai 2 stump 4 counter 2 flower 4

garden 4 treehill 4 kitchen 2

Table 8. Resolution of Mip-NeRF360 evaluation dataset.

During evaluation, we select every eighth frame in each sequence as target views:

- • DL3DV: For input views, we follow Long-LRM [81] and adopt their released indices for 16 to 128 input images. For additional settings with 192 and 256 input images, we uniformly sample input views from the remaining frames after excluding all target indices.
- • Tanks&Temples: For the 32-view setting, we use the input-view indices provided by Long-LRM, while for the 64- and 128-view settings, input views are uniformly sampled from non-target frames.
- • Mip-NeRF360: For all view configurations (32, 64, 128), input views are uniformly sampled from the frames excluding the target indices.

Importantly, target view indices are kept fixed across all input-view configurations for each scene.

Implementation details. The proposed MVP architecture consists of 14 transformer blocks. Within each attention [59] block, we first apply LayerNorm [3] to the input, and then perform QK-Norm [24] on the query and key projections using an RMSNorm [75] layer. Each block comprises a multi head attention module with head dimension 64, followed by a two layer feed forward network with GELU [23] activation. For the spherical harmonics representation, we set the degree to 1 for color and 2 for opacity.

For the post-activation parameterization of 3D Gaussians [34], we follow the implementations from Long-LRM and iLRM [31]. For rendering, we employ the gsplat [71] library for efficient differentiable rasterization.

We employ PRoPE [37], which represents camera poses

as relative positional signals within the attention mechanism. Unlike the original implementation, we use a “Pl¨ucker (extrinsics and intrinsics) rays + PRoPE” formulation instead of “Cam (intrinsics only) rays + PRoPE”, as we found this variant to yield better empirical performance.

[Figure 5]

Figure 6. Architecture overview for our final MVP model. MVP employs a multi-stage hierarchy performing self-attention across increasing frame coverage: frame-wise, group-wise, and global. Each stage is linked by an Intra-view Transition layer using convolution for spatial downsampling and channel up-projection. Finally, a Pyramidal Feature Aggregation module fuses multi-level representations into a unified feature map.

Training details. We train MVP transformer using a threestage training schedule:

- • First stage: We train at 480×256 resolution for 100k iterations with a learning rate of 2e−4. Each training sample uses 32 input views and 12 target views, with a batch size of 8 per GPU, resulting in a total batch size of 256. The frame interval is uniformly sampled between 64 and 128.
- • Second stage: We increase the resolution to 960×540 and train for 50k iterations with a learning rate of 2e−5. In this stage, we use 32 input views and 6 target views, with a batch size of 2 per GPU for a total batch size of 64. We sample input views from the full frame range and apply the intrinsic augmentation strategy proposed in [81].
- • Third stage: We keep the resolution at 960×540 and further train for 30k iterations with a learning rate of 2e−5, using various numbers of input and target views to improve robustness across different view configurations. We again draw input views from the entire frame range, while continuing to use the intrinsic augmentation strategy.

For all stages, we use a cosine learning rate scheduler (with a 3k-step warmup in the first stage only) and the AdamW [44] optimizer with β1 = 0.9 and β2 = 0.95, and a weight decay of 0.05. Weight decay is not applied to normalization and bias parameters. We also employ EMA [21] and do not use gradient clipping, following

16 views 32 views 64 views 128 views PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓

Method

3D-GS30k [34] 21.96 0.766 0.237 8min 25.09 0.838 0.175 8min 28.02 0.890 0.134 8min 29.88 0.917 0.115 8min Long-LRM [81] 20.65 0.707 0.328 0.50 23.54 0.776 0.270 0.84 23.15 0.787 0.263 2.08 20.78 0.741 0.307 6.39 iLRM [31] 21.62 0.746 0.316 0.19 23.93 0.800 0.259 0.53 24.11 0.816 0.243 1.66 22.72 0.804 0.251 5.61 Ours 23.49 0.799 0.238 0.09 25.75 0.848 0.186 0.17 27.62 0.883 0.154 0.36 28.80 0.903 0.136 0.77

- Table 9. Quantitative comparisons on the DL3DV evaluation dataset with varying numbers of input views. For all metrics, we report the results by re-evaluating the models from their 32-view checkpoints.

Method

192 views 256 views PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s) ↓ 3D-GS30k [34] 30.53 0.926 0.109 8min 30.75 0.929 0.107 8min Long-LRM [81] OOM (GPU memory limit exceeded, 80GB) iLRM [31] 21.57 0.787 0.267 11.91 20.62 0.768 0.283 20.92 Ours 29.32 0.911 0.130 1.23 29.42 0.913 0.128 1.84

- Table 10. Quantitative comparisons on the DL3DV evaluation dataset with varying numbers of input views.

16 views 32 views 64 views PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS

Method

Long-LRM 24.42 0.873 0.224 28.31 0.909 0.195 27.20 0.905 0.207 iLRM 26.78 0.896 0.240 28.45 0.913 0.214 28.07 0.914 0.213 Ours 27.41 0.900 0.198 28.63 0.910 0.187 29.11 0.916 0.185

Table 12. Quantitative comparisons on the ScanNet++ dataset under sparse input view setting.

[Figure 6]

[46]. To improve training efficiency, we employ FlashAttention2 [13] and gradient checkpointing [11]. We also use mixed-precision training with bfloat16 to speed up optimization while preserving numerical stability.

Tab. 11 provides our training configurations with different numbers of input views. We adjust the batch size and the number of target renderings to balance iteration time and memory consumption. As described in the main manuscript, we freeze the frame- and group-wise attention blocks and train only the global attention blocks, which enables efficient fine-tuning while preserving the learned local and group representations.

Figure 7. Qualitative performance on the ScanNet++ dataset.

patches. For each query patch, we visualize its top-3 attended tokens from other viewpoints across different stages. At stage 2, we display the attended tokens restricted to the same group, whereas at stage 3, we additionally include tokens attended from views outside the group (Fig. 9).

Additional design choice. We ablate the number of views per group (2, 4, and 8) in Tab. 13. In addition to our primary objective of novel view synthesis, we also evaluate the model on a spatial cognition task [37]. In this task, one camera pose in a set of image–camera pairs is corrupted and the model must identify the mismatched pair, which probes its multi-view awareness. We replace the 3D-GS linear decoder with a head that produces a single scalar per token. For each input view, these scalars are averaged across all tokens to yield a single score, and a softmax over all views then produces a probability vector indicating which image–camera pair is most likely to be corrupted. A group size of 4 (baseline) offers consistently strong accuracy across different numbers of views, while groups of 2 provide too little context and groups of 8 bring only marginal gains with higher computational cost. Therefore, we adopt a group size of four as our default setting.

# Input views 16 32 64 128 Batch size (per GPU) 4 2 1 1

# Target views 6 6 6 1

Table 11. Training configurations for varying input views.

### B. Additional Results

We additionally evaluate our method on the recently released DL3DV [40] evaluation split, which comprises 51 scenes in our experiments. Tab. 9 and 10 present quantitative results comparing our approach with 3D Gaussian Splatting [34] (3D-GS), Long-LRM [81], and iLRM [31]. Our method surpasses existing feed-forward approaches by a large margin in both reconstruction quality and inference efficiency. We evaluate 16-view metrics of Long-LRM and iLRM using checkpoints trained with 32 input views.

Additional zero-shot comparisons We also evaluate our method on the real-world ScanNet++ [72] dataset in Tab. 12 under sparse input-view settings, which naturally reflects robustness to larger viewpoint variations (Fig. 7).

We also ablate the usage of register tokens in Tab. 14. Register token [15] was introduced to mitigate abnormally high token norm values and artifacts in the attention maps of large Vision Transformer models, which often lead to attention artifacts and degraded dense prediction performance. Consistent with this observation, we find that the variant

Additional attention visualization. Using the first frame as the reference view, we begin by selecting three query

Novel view synthesis Spatial cognition PSNR ↑ SSIM ↑ LPIPS ↓ 8 views 16 views 32 views

Method

Group 2 22.94 0.711 0.299 51.3% 75.7% 83.6% Group 4 (baseline) 23.18 0.716 0.300 77.1% 91.4% 96.4% Group 8 23.18 0.717 0.297 84.3% 90.7% 97.1%

Table 13. Ablation studies on the number of views per group.

without register tokens exhibits higher L2 norms across frames, while incorporating register tokens yields modest performance gains. The configuration for this ablation is the same as that of the ablation studies in the main manuscript.

Avg. intra-frame feature norm Stage 1 Stage 2 Stage 3

Method PSNR SSIM LPIPS

Baseline 22.79 0.733 0.235 14.01 84.49 113.36 w/o Register 22.52 0.723 0.242 15.98 286.61 229.82

Table 14. Ablation study on register tokens.

Point map estimation. In addition to novel view synthesis, we also evaluate our method on point map estimation, which quantifies the geometric accuracy of the reconstructed 3D scenes represented with explicit primitives. We evaluate point map accuracy on NRGBD [2] and ETH3D [53], and report the Chamfer distance and the F1score. Specifically, we first back-project the ground-truth depth maps into 3D point clouds using the camera poses. We then rigidly align the predicted point clouds to these reference points with the Umeyama algorithm [58], and finally apply the masks to discard points in invalid regions. We use an image resolution of 960×540, which matches the training resolution for both our method and the baselines. It is important to note that both our approach and the baselines are trained using only a photometric rendering loss from 3D-GS, in contrast to geometry-supervised methods that exploit ground-truth depth or point clouds [60–62, 67]. Consequently, the numbers in Tab. 15 should be read primarily as evidence of relative improvements within this setting, rather than as a direct comparison to fully geometrysupervised models. Overall, our method outperforms the baseline, even though the baseline is additionally regularized using pretrained DepthAnything [68] model during training.

Inference time comparison. Fig. 8 reports inference time as a function of the number of input views. We measure all timings at an input resolution of 960×540 using the official implementations released by the respective authors. Our method exhibits substantially better scalability than existing feed-forward baselines, achieving consistently lower latency across all view counts. Long-LRM encounters an out-of-memory issue on a 80GB GPU once the number of input views exceeds 192. Note that the reported inference time only accounts for the generation of 3D Gaussians. For novel view rendering, Long-LRM encounters a memory er-

NRGBD [2] ETH3D [53] CD ↓ F1-score ↑ CD ↓ F1-score ↑ Long-LRM [81]

Method Views

0.53 0.52 2.75 0.32 Ours 0.18 0.54 1.74 0.34 Long-LRM

16

0.43 0.59 2.69 0.39 Ours 0.14 0.56 2.22 0.42

32

Table 15. Quantitative comparison of point map estimation. We set the threshold value for f1-score to 0.1. We exclude iLRM [31] from this evaluation, as it predicts low-resolution Gaussians that would require additional upsampling, potentially degrading its performance and hurt a fair comparison.

ror when using more than 128 input views.

[Figure 7]

Figure 8. Inference time comparison.

FLOPs comparison. We report theoretical FLOPs comparisons or our model with global- and alternating-attention variants in Tab. 16, assuming 14 transformer layers and identical input tokenizers. Group size for our model is set to 4. V, L, and D denote the number of views, the number of tokens per input view, and the hidden dimension, respectively.

PFLOPs with (V, L, D) input (32, 1920, 1024) (32, 8160, 1024) (128, 8160, 1024)

Method Theoretical FLOPs

Global. Att. V LD(336D + 56V L) 0.24 4.0 62.93 Alter. Att. V LD(336D + 28(1 + V )L) 0.13 2.11 31.89

Ours V LD(21D + (3.3125 + 16V )L + 1.5) 0.01 0.1 0.81

Table 16. Theoretical FLOPs comparison.

Robustness to camera pose. We evaluate the sensitivity of our method to inaccurate camera poses on DL3DV benchmark dataset (32-view, 960×540) by adding random Gaussian noise with varying standard deviations to the rotation and translation components. As shown in Tab. 17, performance degrades noticeably as noise increases, particularly for translation perturbations. This indicates that our method relies on reasonably accurate camera poses, which remains a limitation to be addressed in future work.

##### Remarks on posed setting. We agree that known cam-

0 0.001 0.005 PSNR SSIM LPIPS PSNR SSIM LPIPS PSNR SSIM LPIPS

std.

Rotation 25.96 0.847 0.187 24.65 0.839 0.190 23.34 0.746 0.237 Translation - - - 24.75 0.804 0.204 20.85 0.645 0.297

Table 17. Robustness evaluations on camera pose.

era poses may limit applicability in some cases. Nevertheless, posed settings still often remain highly relevant in practice, including production-ready volumetric multiview video systems, autonomous driving with calibrated cameras, and robotics applications where reasonably accurate poses can be estimated from inertial sensors. We also view posed multi-view modeling as an important step toward unposed settings, as evidenced by recent transformerbased approaches (e.g., GS-LRM [77], LVSM [30]) and their follow-ups (e.g., VGGT [60], Depth Anything 3 [38]), which adopt multi-view transformers as a core component. We hope our work aligns with this evolution and provides a solid foundation for future pose-free extensions.

Additional qualitative results. We provide further qualitative comparisons on the RE10K [80], DL3DV, Tanks&Temples [36], and Mip-NeRF360 [4] datasets in the remainder of this manuscript (Fig. 10, 11, 12, 13, 14 and 15).

[Figure 8]

###### Figure 9. Attention visualization. For colored query patches (red, yellow, green) in the reference view, we highlight top-3 attended tokens: on the left, tokens attended within the group (blue overlay), and on the right, tokens attended within and outside the group (green overlay).

[Figure 9]

###### Figure 10. Qualitative results on the 4-view RE10K dataset. Per-pixel error maps are shown in the bottom-right corner of each image.

[Figure 10]

###### Figure 11. Qualitative results on the DL3DV-Benchmark across varying input view counts (128, 64, 32, and 16). The rows are arranged in descending order of view count, with two rows displayed for each setting.

[Figure 11]

###### Figure 12. Qualitative results on the DL3DV-Evaluation across varying input view counts (128, 64, 32, and 16). The rows are arranged in descending order of view count, with two rows displayed for each setting.

[Figure 12]

###### Figure 13. Qualitative results on the Mip-NeRF360 (top three rows), and truck scene from Tanks&Temples (bottom row). We visualize our rendering results with 32 input views, showing that our method demonstrates clear improvements on generalization and multi-view consistency under sparse inputs.

[Figure 13]

###### Figure 14. Qualitative results on the Mip-NeRF360 (top two rows), and truck scene from Tanks&Temples (bottom row). We visualize our rendering results as the number of input views increases, revealing progressively improved image quality and demonstrating that our method scales effectively with the number of views.

[Figure 14]

###### Figure 15. Qualitative visualization of rendered color and depth maps from novel viewpoints using 32 input images. Scenes from DL3DV (top four rows), Mip-NeRF360 (fifth and sixth row), and Tanks&Temples (bottom two rows) are shown.

