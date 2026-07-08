# arXiv:2603.16869v3[cs.CV]10May2026

## SegviGen: Repurposing 3D Generative Model for Part Segmentation

LIN LI∗, Renmin University of China, China HAORAN FENG∗, Tsinghua University, China ZEHUAN HUANG†, Beihang University, China HAOHUA CHEN, Beihang University, China WENBO NIE, Beijing Jiaotong University, China SHAOHUA HOU, Beihang University, China KEQING FAN, Beihang University, China PAN HU, Beihang University, China SHENG WANG, Bambu Lab, China BUYU LI, Bambu Lab, China LU SHENG‡, Beihang University, China

[Figure 1]

Fig. 1. SegviGen enables diverse and accurate 3D part segmentation by leveraging priors from large-scale 3D generative models. With substantially less training data, it produces high-fidelity segmentation results with sharp part boundaries and strong generalization across object categories.

∗Equal contribution †Project lead ‡Corresponding author

Authors’ Contact Information: Lin Li, Renmin University of China, Beijing, China, ruclilin@163.com; Haoran Feng, Tsinghua University, Shenzhen, China; Zehuan Huang, Beihang University, Beijing, China; Haohua Chen, Beihang University, Beijing, China; Wenbo Nie, Beijing Jiaotong University, Beijing, China; Shaohua Hou, Beihang University, Beijing, China; Keqing Fan, Beihang University, Beijing, China; Pan Hu, Beihang University, Beijing, China; Sheng Wang, Bambu Lab, Shenzhen, China; Buyu Li, Bambu Lab, Shenzhen, China; Lu Sheng, Beihang University, Beijing, China, lsheng@buaa.edu.cn.

This work is licensed under a Creative Commons Attribution 4.0 International License.

We introduce SegviGen, a framework that repurposes native 3D generative models for 3D part segmentation. Existing pipelines either lift strong 2D priors into 3D via distillation or multi-view mask aggregation, often suffering from cross-view inconsistency and blurred boundaries, or explore native 3D discriminative segmentation, which typically requires large-scale annotated 3D data and substantial training resources. In contrast, SegviGen leverages the structured priors encoded in pretrained 3D generative model to induce segmentation through distinctive part colorization, establishing a novel and efficient framework for part segmentation. Specifically, SegviGen encodes

© 2026 Copyright held by the owner/author(s). ACM 1557-7368/2026/7-ART68 https://doi.org/10.1145/3811399

an input 3D asset and predicts part-indicative colors on active voxels of a geometry-aligned reconstruction. It supports interactive part segmentation, full segmentation, and full segmentation with 2D guidance in a unified framework. Extensive experiments show that SegviGen improves over the prior state of the art by 40% on interactive part segmentation and by 15% on full segmentation, while using only 0.32% of the training data. This undoubtedly demonstrates that pretrained 3D generative priors transfer effectively to 3D part segmentation, enabling strong performance with limited supervision. Code and pretrained weights are publicly available at https://github.com/Nelipot-Lee/SegviGen.

CCS Concepts: • Computing methodologies → Computer vision; Computer graphics.

### ACM Reference Format:

Lin Li, Haoran Feng, Zehuan Huang, Haohua Chen, Wenbo Nie, Shaohua Hou, Keqing Fan, Pan Hu, Sheng Wang, Buyu Li, and Lu Sheng. 2026. SegviGen: Repurposing 3D Generative Model for Part Segmentation. ACM Trans. Graph. 45, 4, Article 68 (July 2026), 12 pages. https://doi.org/10.1145/3811399

- 1 INTRODUCTION

Part segmentation provides explicit part-level structures of 3D assets, serving as a core primitive for 3D content creation pipelines and offering fundamental 3D perception capabilities for spatial intelligence. It enables a wide range of downstream applications, including part-level editing, animation rigging, and industrial uses such as 3D printing. However, existing methods often fall short in segmentation quality, producing erroneous regions and imprecise boundaries that limit their practical usability.

To this end, one line of work attempts to transfer the comprehensive 2D segmentation priors to 3D via 2D-to-3D lifting. Methods such as SAMPart3D [Yang et al. 2024a] optimize 3D segmentation via 2D-to-3D distillation, but incur substantial computational and time overhead, and often yield blurry boundaries. In parallel, another set of methods [Deng et al. 2025; Huang et al. 2024b; Zhou et al. 2025] applies SAM [Carion et al. 2025; Kirillov et al. 2023; Ravi et al.

- 2024] to obtain 2D masks of multi-view projected images, which are then back-projected and fused into 3D masks. However, these multi-view pipelines incur substantial runtime overhead, are sensitive to view coverage, and the back-projection and fusion step often introduces cross-view inconsistencies and imprecise boundaries.

Recently, another line of work [Ma et al. 2025a; Zhu et al. 2025] moves toward native 3D part segmentation so as to remedy the inherent shortcomings of the aforementioned methods that leverage

- 2D segmentation priors. However, it is a typical requirement to collect large-scale training datasets with curated 3D part annotations, where fine-grained annotations are costly and inconsistent across sources in granularity, hierarchy, and boundary definitions.

Therefore, a more promising approach is to leverage a prior model that encodes both 3D structure and texture to perform segmentation. In particular, 3D generative models trained on large-scale unannotated 3D textured assets internalize rich part-level structure and texture patterns, providing a strong 3D prior over geometry and appearance. Such priors encourage part segmentation with sharper boundaries, while reducing reliance on dense part annotations and extensive task-specific training. This motivates us to ask: How can

- 3D generative priors be effectively transferred to part-level 3D segmentation to improve quality and data efficiency?

Motivated by this perspective, we propose SegviGen, a generative framework for 3D part segmentation that leverages the rich 3D structural and textural knowledge encoded in large-scale 3D generative models. Specifically, we formulate part segmentation as a colorization task that leverages the full capacity of 3D generative models. The model is trained to predict part-indicative colors, along with reconstructing the underlying geometry. This formulation naturally accommodates additional conditioning signals, enabling SegviGen to flexibly support interactive part segmentation, full segmentation, and 2D segmentation map–guided full segmentation under a unified architecture. Notably, 2D segmentation map-guided full segmentation allows users to customize the decomposition through a 2D segmentation map.

Qualitative and quantitative results show that SegviGen consistently surpasses the prior state-of-the-art, P3-SAM [Ma et al. 2025a], while using only 0.32% of the training data. On interactive part segmentation, it achieves the best performance across all metrics on PartObjaverse-Tiny [Yang et al. 2024b] and PartNeXT [Wang et al. 2025], with a 40% gain in IoU@1, an important metric that reflects the model’s single-click accuracy. On full segmentation without guidance, SegviGen outperforms the best baseline by 15% in overall IoU, averaged across datasets. Our main contributions are summarized as follows:

- • We propose SegviGen, a unified multi-task framework for 3D part segmentation that effectively exploits the structural and textural priors encoded in pretrained 3D generative models, enabling accurate and efficient segmentation.
- • We reformulate 3D segmentation as part-wise colorization, where SegviGen predicts the colors of actiave voxel as part labels in a single generative process.
- • Extensive experiments show that SegviGen outperforms the prior state of the art by 40% on interactive part segmentation and 15% on full segmentation, using only 0.32% of the training data, highlighting the effectiveness of transferring 3D generative priors to part segmentation.

2 Related Work 2.1 3D Part Segmentation

Traditional 3D part segmentation is typically cast as supervised semantic labeling on points or faces, using fixed part taxonomies provided by curated 3D segmentation datasets [Chen et al. 2009; Dai et al. 2017; Mo et al. 2019; Qi et al. 2017]. Concretely, these methods [Hanocka et al. 2019; Lin et al. 2021; Qi et al. 2017; Wu et al. 2024a, 2022, 2024e] typically combine a 3D feature encoder with a segmentation head to predict dataset-specific part IDs. However, the closed-world nature of both the label space and the training data limits generalization, making it difficult to transfer to unseen object categories or arbitrary, non-canonical part decompositions.

To alleviate this generalization bottleneck, recent works exploit 2D foundation models as transferable priors [Caron et al. 2021; Kirillov et al. 2023; Li et al. 2022; Oquab et al. 2023; Radford et al. 2021; Ravi et al. 2024] for 3D part segmentation. A common strategy adopts a render-and-lift pipeline: it segments multi-view renderings with promptable 2D models and then projects and fuses the masks back onto the 3D surface [Tang et al. 2025c; Xu et al. 2025; Xue

et al. 2025; Yang et al. 2023; Zhou et al. 2023]. Another line leverages distillation or feature projection to supervise 3D predictors with transferred 2D representations or pseudo-labels [Garosi et al. 2025; Umam et al. 2024]. However, thest pipelines inherit the 2D–3D domain gap and multi-view alignment issues, and typically entails longer optimization and training cycles.

Recognizing the scalability and reliability issues of 2D-to-3D lifting, recent studies have shifted toward native feed-forward 3D segmentation that predicts masks directly on 3D representations at inference time. Representative efforts for open-world part segmentation include training queryable 3D predictors with automatically curated supervision [Ma et al. 2025b], learning continuous partaware 3D feature fields for direct decomposition [Liu et al. 2025], and prompt-guided 3D mask prediction models [Zhou et al. 2025], with more recent large-scale native 3D part segmentation models such as P3-SAM [Ma et al. 2025a] and PartSAM [Zhu et al. 2025] further scaling training on millions of shape–part pairs. Despite encouraging progress, these native 3D approaches are fundamentally bottlenecked by the availability of large-scale, high-quality 3D part annotations, and the inconsistency of part taxonomies and granularity across datasets often introduces supervision mismatch, ultimately weakening cross-domain generalization.

- 2.2 3D Generative Model

The rapid progress of diffusion-based generative modeling [Ho et al. 2020; Song et al. 2020], together with the emergence of largescale, high-quality 3D data collections [Deitke et al. 2024, 2023], has catalyzed a wave of 3D generative methods [Chen et al. 2024a,b; Dong et al. 2025c; Gao et al. 2025; Hao et al. 2024; He et al. 2024; Hong et al. 2023; Huang et al. 2024c; Li et al. 2024, 2025c; Liu et al.

- 2024a,b,c, 2023; Long et al. 2024; Meng et al. 2024; Roessle et al.

- 2024; Tang et al. 2025a; Voleti et al. 2025; Wang et al. 2024a,b; Wei et al. 2025; Wen et al. 2024; Wu et al. 2024d,c,b; Xu et al. 2024; Ye et al. 2025; Zhang et al. 2024; Zhao et al. 2025b, 2024]. A prevalent route builds 3D assets through a 2D-to-3D pipeline: models first synthesize multi-view imagery and subsequently reconstruct the underlying 3D geometry and appearance from these views [Huang et al. 2025, 2024a; Liu et al. 2023; Long et al. 2024; Qu et al. 2025; Tang et al. 2025a; Voleti et al. 2025; Wang et al. 2024b; Wen et al. 2024; Xu et al. 2024], yet view-to-view discrepancies in the synthesized images can propagate and degrade the final 3D quality.

In contrast, a growing family of native 3D generative models learns directly in 3D latent spaces, typically pairing a variational autoencoder [Kingma 2013] with a diffusion transformer (DiT) [Peebles and Xie 2023] to perform denoising over compact latents [Chen et al. 2025; Dong et al. 2025a; Li et al. 2024, 2025b,e,d; Lin et al.

- 2025; Tang et al. 2025b; Wu et al. 2025b, 2024c, 2025a; Xiang et al.

- 2025a,b; Zhang et al. 2024; Zhao et al. 2025a, 2024]. By learning to generate in a compact yet expressive 3D latent space, these models encode rich structural and texture knowledge across large-scale 3D assets, providing a strong transferable prior for downstream 3D part segmentation. In particular, TRELLIS2 [Xiang et al. 2025a] introduces a field-free structured latent via an omni-voxel sparse voxel representation (O-Voxel) that jointly models geometry and appearance, enabling efficient generation with sharp, high-frequency

textures that better preserve fine-grained part boundaries for 3D segmentation.

3 METHOLODOGY

We propose SegviGen, a unified multi-task framework for 3D part segmentation that supports three practical settings: interactive partsegmentation, full segmentation, and full segmentation with 2D guidance. To leverage the prior knowledge encoded in a pretrained 3D generative model, we cast 3D segmentation as a colorization problem. Conditioned on these inputs, the model reconstructs the 3D asset while predicting colors for active voxels in the structured 3D representation, where each color corresponds to an individual part, yielding the final segmentation. Below, we begin by describing the underlying 3D generative model (Sec. 3.1), followed by our task reformulation (Sec. 3.2), and then detail the overall pipeline (Sec. 3.3).

- 3.1 Preliminary: Structured-Latent 3D Generative Model

Recent work [Xiang et al. 2025a] organizes each textured 3D asset into a sparse set of active voxels on a regular grid, where every active voxel stores geometry and texture features aligned in 3D. Given the sparse omni-voxel representation, a Sparse Compression VAE (SC-VAE) maps each voxelized asset feature tensor x to a compact structured latent z1 = 𝐸𝜙(x) and reconstructs it via xˆ = 𝐷𝜃 (z1), yielding an expressive yet highly compressed 3D latent space. On top of these latents, a conditional flow-matching generator learns a time-dependent vector field v𝜓 (z𝑡,𝑡, c) under conditioning c by matching the constant velocity along linear interpolants:

z0 ∼ N(0, I), 𝑡 ∼ U(0, 1), z𝑡 = (1 − 𝑡)z0 + 𝑡z1, (1)

Lcfm = E v𝜓 (z𝑡,𝑡, c) − (z1 − z0) 22 . (2)

Thislatentgenerativepipelineenables efficient synthesis of geometryand texture-consistent 3D assets, and the resulting structured latents capture rich joint statistics of shape and appearance, providing a strong transferable prior for fine-grained 3D part segmentation.

- 3.2 Task Reformulation and I/O Representation

Interactive part-segmentation is formulated as binary part extraction: given user-provided 3D points indicating a target part, we supervise the model to color the selected part in white and the remaining regions in black. Full segmentation targets multipart decomposition: we assign each part a distinct color from a randomly sampled color palette and supervise voxel colors accordingly. To reduce sensitivity to particular color choices, we use 𝐾=10 independently sampled palettes per shape, providing multiple colorizations for the same underlying partition. Full segmentation with 2D guidance additionally conditions the model on a rendered 2D segmentation map: we first colorize the 3D parts and render the corresponding 2D segmentation map, and we then train the model to generate 3D voxel colors that are consistent with the color assignments in the 2D guidance. Overall, this formulation preserves a unified model interface across settings, enabling a consistent architecture and training pipeline.

Input Mesh Noisy Latents Point Embedding

Point Embedding

[Figure 2]

[Figure 3]

[Figure 4]

Token Value:

(u,v,w)

Token Coordinate:

(i,j,k), 0<i,j,k<S (u,v,w) (0,0,0)

Self-Attention (ROPE)

[Figure 5]

[Figure 6]

VAE Learnable Embed

Add Noise

Valid Point Embeds (Learnable) Padding Point Embeds (Zero)

Task ID Embedding

Token-wise Concatenation

[Figure 7]

[Figure 8]

[Figure 9]

Positional

Encoding

MLP

Multi-Task Flow Transformer

0: Interactive Seg 1: Full Seg 2: Full Seg w. 2D map

Flow Matching Loss

- Fig. 2. Pipeline of SegviGen. We reformulate 3D part segmentation as a conditional colorization task. During training, given a 3D mesh and its part-color ground truth, we encode both with a pretrained 3D VAE, add noise to the ground-truth latent, and then concatenate the geometry latent, noisy color latent, and point-condition tokens to form the final latent input. Conditioned on the sampled timestep and a task embedding, the multi-task flow transformer predicts the noise residual for flow-matching training.

both training and inference. Given point coordinates u = {𝑢𝑖}𝑚𝑖=1 with 𝑢𝑖 ∈ R3, we form point-condition tokens

- 3.3 Unified Multi-Task 3D Part Segmentation

- 3.3.1 Overall framework. To fully leverage pretrained 3D generative models, we cast 3D part segmentation as a conditional part-wise colorization task in 3D latent space. Given an input asset 𝑋, a pretrained 3D VAE encoder 𝐸(·) produces a encoded latent 𝑧 = 𝐸(𝑋), which helps specify the active voxel support and anchors generation to the underlying shape. For each task, we construct a part-wise colorized target and encode it into the same latent space to obtain 𝑦, following the task-specific scheme in Sec. 3.2. We then sample 𝜖 ∼ N(0,𝐼) and 𝑡 ∼ U(0, 1) to form a noisy interpolation

𝑦𝑡 = (1 − 𝑡)𝑦 + 𝑡 𝜖. (3) A pretrained DiT-based backbone is fine-tuned to predict the noise residual conditioned on the noisy input 𝑦𝑡, the geometry latent 𝑧, the task condition 𝐶, and a learned task embedding 𝑒𝜏:

𝑣ˆ𝜃 = 𝑓𝜃 (𝑦𝑡, 𝑧, 𝐶, 𝑒𝜏, 𝑡) . (4) Training follows the conditional flow-matching objective

L(𝜃) = E𝑋,𝜏,𝑡,𝜖 𝑤(𝑡) 𝑣 ˆ𝜃 − (𝜖 −𝑦) 22 , (5) where 𝑤(𝑡) is an optional timestep weighting.

- 3.3.2 Condition Injection. We adopt task-specific conditioning designs while maintaining a unified interface across settings. For interactive segmentation, user clicks in the UI provide an efficient and intuitive form of guidance. In our framework, each click is encoded as a sparse point token comprising its 3D coordinates and an associated feature vector. Since the 3D coordinates are already effectively encoded by RoPE within the attention layers, we omit the additional learnable input-level positional embedding used in prior designs [Ma et al. 2025a]. Instead, all points share the same learnable feature vector 𝑄 , which serves as the point token during

𝑄 = q(𝑢1), . . ., q(𝑢𝑚) , q(𝑢𝑖) = 𝑢𝑖 ; e𝑝 , (6)

where e𝑝 is a shared learnable feature appended to every point token. Conditioned on 𝑄, the denoising model is instantiated as

𝑣ˆ𝜃 = 𝑓𝜃 (𝑦𝑡, 𝑧, 𝑄, 𝑒𝜏, 𝑡) . (7)

When the number of points is fewer than 10, we pad the point tokens to a length of 10 using zero coordinates and zero features. To preserve a single unified model, we keep this interface for full segmentation and 2D-guided full segmentation by providing 10 padded tokens with all-zero coordinates and features.

For full segmentation with 2D guidance, we additionally provide a user-specified 2D segmentation colorization as guidance. In this setting, the guidance specifies the desired part decomposition in image space, which is then transferred to 3D through our generative framework. This provides an explicit way to obtain finer or coarser 3D segmentations when such decomposition is indicated by the input 2D map, while interactive segmentation further supports practical refinement by extracting and merging local regions through additional user clicks. The guidance image is encoded into a sequence of conditioning tokens injected via cross-attention:

𝑝 = 𝑔𝜙(𝐼guide), (8)

where 𝑔𝜙(·) denotes an image encoder. In this setting, denoising is conditioned on both the padded point-token interface 𝑄0 and the image guidance tokens 𝑝:

𝑣ˆ𝜃 = 𝑓𝜃 (𝑦𝑡, 𝑧, (𝑄0, 𝑝), 𝑒𝜏, 𝑡) . (9)

Table 1. Comparison of interactive part segmentation performance. We report IoU at different numbers of clicks, compared with Point-SAM [Zhou et al. 2025] and P3-SAM [Ma et al. 2025a] on PartObjaverse-Tiny [Yang et al. 2024b] and PartNeXT [Wang et al. 2025].

PartObjaverse-Tiny PartNeXT IoU@1 IoU@3 IoU@5 IoU@7 IoU@10 IoU@1 IoU@3 IoU@5 IoU@7 IoU@10

Method

Point-SAM [Zhou et al. 2025] 24.87 48.99 59.67 64.33 67.99 23.90 47.50 56.71 61.23 65.04 P3-SAM [Ma et al. 2025a] 33.04 50.57 53.78 54.74 55.51 35.61 51.26 52.03 52.61 53.81 SegviGen 42.49 61.14 67.53 71.50 75.02 54.86 71.15 78.11 79.96 82.73

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Point-SAMP3-SAMOurs

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

- Fig. 3. Interactive part-segmentation results. We compare SegviGen with existing representative baselines, including Point-SAM [Zhou et al. 2025] and P3-SAM [Ma et al. 2025a]. In the figure, yellow points denote user clicks, while the predicted target part is highlighted in red. Leveraging priors from pretrained

- 3D generative models, SegviGen achieves more accurate results with sharper boundaries than prior methods, while requiring substantially less training data.

- 3.3.3 TaskEmbedding. To improvemulti-taskgeneralization within a single model, task identity is encoded as a continuous embedding and injected alongside the timestep signal. Let 𝜏 ∈ {1, . . .,𝑇} denote the task index. A sinusoidal encoding is first computed from 𝜏,

𝑠𝜏 = PE(𝜏) ∈ R𝑑𝑓 , (10) where PE(·) follows the standard sinusoidal scheme. A lightweight MLP then maps 𝑠𝜏 to the task embedding

𝑒𝜏 = MLP𝜓 (𝑠𝜏) ∈ R𝑑. (11)

In parallel, the timestep 𝑡 is embedded as 𝑒𝑡 ∈ R𝑑. The final modulation vector used by DiT backbone is obtained by additive fusion,

𝑚 = 𝑒𝑡 + 𝑒𝜏, 𝑣ˆ𝜃 = 𝑓𝜃 (𝑦𝑡, 𝑧, 𝐶, 𝑚) , (12)

where 𝑚 conditions the adaptive layers to jointly encode diffusion progress and task semantics. During training, samples from different tasks are interleaved and supervised with their corresponding 𝜏, encouraging the shared backbone to learn task-discriminative behaviors while preserving a unified parameterization.

4 EXPERIMENTS

- 4.1 Setting

Implementation Details. We adopt Trellis.2 [Xiang et al. 2025a] as our base model, which is a 3D generative framework with a native and compact structured latent representation. For all experiments,

the Tex-SLAT flow model is trainable, while the remaining SC-VAE is kept frozen. We adopt the AdamW optimizer [Loshchilov and Hutter 2019] with a learning rate of 1 × 10−4. All experiments are conducted on 8 NVIDIA A800 GPUs, and the model is trained for 8 hours. Unless otherwise specified, the segmentation results shown in this paper are produced with 12-step inference.

Datasets. For training, we use the PartVerse dataset [Dong et al. 2025b], which contains 12k objects with a total of approximately 91k annotated parts. For evaluation, we use PartObjaverse-Tiny [Yang et al. 2024a], which contains 200 textured mesh objects, and a 300object textured-mesh subset of PartNeXT [Wang et al. 2025].

Baselines. We compared our model’s performance on full segmentation between P3-SAM [Ma et al. 2025a], Find3D [Ma et al. 2025b], SAMPart3D [Yang et al. 2024a], Partfield [Liu et al. 2025]. P3-SAM is a native 3D point-promptable part segmenter with multiple mask heads and an IoU predictor. Find3D targets open-world, languagequeryable parts by auto-labeling rendered multi-view images with SAM and a VLM. SAMPart3D and PartField both learn part-aware 3D features from multi-view SAM masks and obtain parts via feature clustering.

For interactivepartsegmentation,we compared our model against P3-SAM [Ma et al. 2025a] and Point-SAM [Zhou et al. 2025], where Point-SAM adapts the SAM prompt-and-mask paradigm to point clouds and is trained with SAM-generated pseudo masks.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Partobjaverse-TinyPartNeXT

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

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

Input Find3D SAMPart3D PartField P3-SAM Ours Ours w. 2D Map

- Fig. 4. Full segmentation results. We compare SegviGen against a broad set of prior methods, where different colors indicate different segmented parts. From the results, SegviGen achieves high-accuracy full segmentation with sharp part boundaries using only 3D input.

Metrics. To evaluate the interactive segmentation, we sample 10 positive points for each part, then measure the average IOU between the predicted masks for all clicks of all parts and their corresponding ground truth masks. IoU@N stands for IoU score in N foreground clicks. The evaluation metric for full segmentation is the same method in previous work [Liu et al. 2025; Ma et al. 2025a], using IoU to measure the accuracy of overall mask predictions.

Voxel-to-Mesh Color Transfer. SegviGen predicts part-indicative colors on active O-Voxels. Since the mesh decoded by Trellis.2 may differ from the input in tessellation and local topology, we transfer the predicted voxel colors back to the original input mesh. Specifically, each mesh vertex is assigned the color of its nearest active voxel, and each face label is determined by majority voting over its vertices. This preserves the original mesh structure and is more suitable for mesh-level segmentation than directly using the decoded mesh. We further apply lightweight mesh-level smoothing to remove isolated spikes introduced by projection near part boundaries.

4.2 Main Results

4.2.1 Interactive Part-Segmentation. We evaluate interactive part segmentation on two benchmarks: PartObjaverse-Tiny [Yang et al. 2024b] and PartNeXT [Wang et al. 2025]. We benchmark against two state-of-the-art native 3D methods: Point-SAM [Zhou et al. 2025], which is specialized for point cloud segmentation, and P3-SAM [Ma et al. 2025a]. The quantitative results are summarized in Table 1.

As shown in Table 1, SegviGen consistently outperforms all baselines by a significant margin across all interaction rounds. Notably, our method demonstrates exceptional efficiency in the few-shot interaction setting. In the most challenging 1-click scenario (IoU@1), SegviGen achieves 42.49% on PartObjaverse-Tiny and 54.86% on PartNext, surpassing the Point-SAM by approximately 17.6% and 31.0%, respectively. This indicates that our generative framework possesses a much stronger initial understanding of 3D part structures compared to discriminative approaches, allowing it to infer complete part geometries from minimal user guidance.

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

- Fig. 5. Results with different view 2D guidance. Visible regions are consistently segmented, while occluded regions may receive different color assignments under different views.

Furthermore, as the number of user clicks increases from 1 to 10, SegviGen exhibits a steady and robust performance gain. On the PartNext dataset, our method reaches an IoU of 82.73% at 10 clicks, significantly higher than Point-SAM (65.04%) and P3-SAM (53.81%). This demonstrates that our model effectively incorporates user feedback to refine boundaries and resolve ambiguities.

- 4.2.2 Full Segmentation. We evaluate the full segmentation capability of SegviGen in two distinct settings: (1) Using purely native 3D representation. (2) Incorporating with 2D guidance. Quantitative comparisons with state-of-the-art methods, including Find3D [Ma et al. 2025b], SAMPart3D [Yang et al. 2024a], PartField [Liu et al.

- 2025], and P3-SAM [Ma et al. 2025a], are presented in Table 2. Qualitative results are shown in 4.

Without 2D Guidance. In this setting, SegviGen performs segmentation solely based on the structural and appearance priors learned during pretraining, without access to any external 2D segmentation maps. The model is prompted to generate part-indicative colors directly from the latent 3D representation. As shown in Table 2, our method demonstrates superior generalization, particularly on PartNext. SegviGen achieves an IoU of 55.40%, significantly outperforming PartField (41.50%) and SAMPart3D (29.62%) While SAMPart3D performs well on the smaller PartObjaverse-Tiny dataset (59.05%), its performance collapses on PartNext. In contrast, SegviGen maintains robust performance (50.64% on PartObjaverse-Tiny).

With 2D Guidance. To further unleash the potential of SegviGen, we introduce a 2D-guided mode where the model is conditioned on a single-view 2D segmentation map (rendered via nvdiffrast or derived from a 2D segmenter). This setting combines the rich semantic cues of 2D foundation models with the geometric consistency of our 3D generative framework. Incorporating this lightweight 2D prior yields substantial performance gains. As shown in Table 2, SegviGen (w. 2D Map) achieves new state-of-the-art results on both datasets, reaching 62.98% on PartObjaverse-Tiny and 71.53% on PartNext.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Input Interactive Full

Fig. 6. Generalization results on AI-generated meshes. The results show that SegviGen can produce plausible interactive and full part segmentations on AI-generated 3D assets without additional training.

Table 2. Quantitative results (IoU) for full segmentation. “w. 2D Map” denotes the setting with 2D segmentation-map guidance.

Method PartObjaverse-Tiny PartNext

|Find3D [Ma et al. 2025b] SAMPart3D [Yang et al. 2024a] PartField [Liu et al. 2025] P3-SAM [Ma et al. 2025a]<br><br>|15.62 19.04 59.05 29.62 51.72 41.50 45.36 31.94|
|---|---|
|SegviGen SegviGen (w. 2D Map)|50.64 55.40 62.98 71.53<br><br>|

- 4.2.3 Effect of 2D Segmentation Guidance. We further evaluate SegviGen with 2D segmentation maps rendered from different viewpoints. As shown in 5, visible regions in the guidance map are reliably transferred to 3D. For regions invisible from the guided view, different viewpoints may lead to different color assignments. However, these differences mainly reflect label-assignment ambiguity rather than incorrect decomposition, as the resulting parts remain consistent and plausible. This suggests that SegviGen can effectively absorb different 2D segmentation results as guidance, while the exact color assignment of occluded regions may vary across views.
- 4.2.4 Generalization to AI-Generated Meshes. To evaluate generalization beyond artist-created meshes, we further test SegviGen on meshes generated by Hunyuan3D 2.1. These meshes differ from the training and benchmark assets in geometry quality, topology, and part composition, and do not have ground-truth part annotations. We therefore provide qualitative results in 6. The results show that SegviGen can produce plausible part decompositions on AI-generated 3D assets, demonstrating its potential applicability to automatically generated 3D content.

4.3 Ablation Studies and Analysis

4.3.1 Point Embedding Mechanism. To investigate the optimal representation point prompt within our framework, we conducted an

[Figure 81]

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

Interactive Full Full w. 2D Map

Fig. 7. Failure cases. Due to semantic ambiguity, SegviGen may produce slightly more or fewer parts than expected in interactive and full segmentation. In the 2D-guided setting, extremely fine structures may still be difficult to segment clearly and consistently from the guidance map.

Table 3. Ablation study on point embedding mechanisms. We compare the performance of Explicit Coordinate Encoding and Label-based Semantic Embedding under varying numbers of clicks on PartObjaverse [Yang et al. 2024b].

Table 4. Effect of sampling steps on segmentation performance. We choose 12 steps as a practical trade-off between accuracy and efficiency.

Steps IoU@1 IoU@3 IoU@5 IoU@7 IoU@10 Time 1 42.90 59.98 65.86 69.50 72.85 0.44s 4 44.51 60.40 66.65 70.64 73.58 1.02s 8 44.21 61.14 67.64 71.14 74.49 1.81s 12 42.49 61.14 67.53 71.50 75.02 2.63s 25 43.82 61.67 68.30 71.87 74.99 5.12s

Method IoU@1 IoU@3 IoU@5 IoU@7 IoU@10

Explicit Coord 41.75 60.19 67.43 71.61 75.40 Label-based 42.49 61.14 67.53 71.50 75.02

ablation study on the point embedding mechanism, comparing two distinct strategies:

4.4 Failure Cases and Limitations

SegviGen still has two main limitations. First, in the interactive and full segmentation settings, semantic ambiguity may cause the model to produce more or fewer parts than expected, since multiple valid part decompositions may exist for the same object. Second, although 2D guidance improves controllability, the model cannot always reproduce highly detailed part decompositions specified by the input 2D segmentation map. When the guidance becomes overly fine-grained, the resulting 3D segmentation may exhibit reduced boundary precision and smoothness. Addressing these issues with more explicit semantic control and fine-structure-aware segmentation remains an important direction for future work.

Explicit Coordinate Encoding. In this setting, spatial coordinates are explicitly injected into the feature space. We utilize a frequencybased positional encoding scheme to map continuous 3D coordinates into high-dimensional embeddings, which will fuse with learnable semantic vectors. Consequently, the input features explicitly encapsulate both absolute spatial information and semantic category.

Label-based Semantic Embedding. In this setting, the feature vectors serve solely as semantic indicators without explicitly encoding geometric values. A shared learnable embedding vector is assigned to all foreground points. The spatial information is preserved implicitly via the coordinate indices of the SparseTensor, relying on the sparse backbone’s intrinsic ability to process spatial locality. As shown in Table 3, as the number of interactions increases, the Explicit Coordinate Encoding method outperforms the Label-based approach, particularly in the later stages.

5 CONCLUSION

This paper introduces SegviGen, a framework that repurposes pretrained 3D generative models for 3D part segmentation. In contrast to 2D-to-3D lifting methods that often suffer from cross-view inconsistency and blurred boundaries, and native 3D discriminative approaches that require large-scale part annotations and heavy training, SegviGen transfers generative priors to deliver accurate and globally coherent segmentations with limited supervision. It reformulates segmentation as part-wise colorization, jointly reconstructing geometry and predicting part-indicative colors, and supports multiple task settings via flexible conditioning. Experiments on interactive and full segmentation benchmarks show consistent

4.3.2 Number of denoising steps at inference. We analyze the impact of sampling steps on segmentation performance in Table 4. Due to the trajectory property of the flow model, we observe a great performance even with one step. Performance improves as steps increase, but gains begin to saturate over 8 steps. Although 25 steps offer marginal improvements, the inference latency nearly doubles compared to 12 steps. Thus we adopt 12 steps as the optimal balance between high-quality results and computational efficiency.

improvements over prior methods, underscoring the effectiveness and data efficiency of 3D generative priors for 3D part segmentation.

Acknowledgment

This work was supported by National Natural Science Foundation of China (62132001), Beijing Natural Science Foundation (L252218), and the Fundamental Research Funds for the Central Universities.

References

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. 2025. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025).

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 2021. Emerging Properties in Self-Supervised Vision Transformers. In ICCV.

Sijin Chen, Xin Chen, Anqi Pang, Xianfang Zeng, Wei Cheng, Yijun Fu, Fukun Yin, Yanru Wang, Zhibin Wang, Chi Zhang, Jingyi Yu, Gang Yu, Bin Fu, and Tao Chen. 2024a. MeshXL: Neural Coordinate Field for Generative 3D Foundation Models. arXiv:2405.20853 [cs.CV] https://arxiv.org/abs/2405.20853

Xiaobai Chen, Aleksey Golovinskiy, and Thomas Funkhouser. 2009. A Benchmark for 3D Mesh Segmentation. In SIGGRAPH.

Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, Guosheng Lin, and Chi Zhang. 2024b. MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers. arXiv:2406.10163 [cs.CV] https://arxiv.org/abs/2406.10163

Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, and Guosheng Lin. 2025. Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention. arXiv:2507.17745 [cs.CV] https://arxiv.org/abs/2507.17745

Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. 2017. ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. In arXiv.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. 2024. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems 36 (2024).

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13142–13153.

Ken Deng, Yunhan Yang, Jingxiang Sun, Xihui Liu, Yebin Liu, Ding Liang, and Yan-Pei Cao. 2025. GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation. In arXiv.

Junting Dong, Qi Fang, Zehuan Huang, Xudong Xu, Jingbo Wang, Sida Peng, and Bo Dai. 2025c. Tela: Text to layer-wise 3d clothed human generation. In European Conference on Computer Vision. Springer, 19–36.

Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jaehyeok Kim, Chenjian Gao, Zhanpeng Huang, Zibin Wang, Tianfan Xue, and Dan Xu. 2025a. From One to More: Contextual Part Latents for 3D Generation. arXiv:2507.08772 [cs.CV] https://arxiv.org/abs/2507.08772

Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jaehyeok Kim, Chenjian Gao, Zhanpeng Huang, Zibin Wang, Tianfan Xue, and Dan Xu. 2025b. From One to More: Contextual Part Latents for 3D Generation. arXiv:2507.08772 [cs.CV] https://arxiv.org/abs/2507.08772

Daoyi Gao, Yawar Siddiqui, Lei Li, and Angela Dai. 2025. MeshArt: Generating Articulated Meshes with Structure-Guided Transformers. arXiv:2412.11596 [cs.CV] https://arxiv.org/abs/2412.11596

Marco Garosi, Riccardo Tedoldi, Davide Boscaini, Massimiliano Mancini, Nicu Sebe, and Fabio Poiesi. 2025. 3D Part Segmentation via Geometric Aggregation of 2D Visual Features. In arXiv.

Rana Hanocka, Amir Hertz, Noa Fish, Raja Giryes, Shachar Fleishman, and Daniel Cohen-Or. 2019. MeshCNN: a network with an edge. In ACM.

Zekun Hao, David W. Romero, Tsung-Yi Lin, and Ming-Yu Liu. 2024. Meshtron: HighFidelity, Artist-Like 3D Mesh Generation at Scale. arXiv:2412.09548 [cs.GR] https: //arxiv.org/abs/2412.09548

Zexin He, Tengfei Wang, Xin Huang, Xingang Pan, and Ziwei Liu. 2024. Neural LightRig: Unlocking Accurate Object Normal and Material Estimation with MultiLight Diffusion. arXiv:2412.09593 [cs.CV] https://arxiv.org/abs/2412.09593

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).

Rui Huang, Songyou Peng, Ayca Takmaz, Federico Tombari, Marc Pollefeys, Shiji Song, Gao Huang, and Francis Engelmann. 2024b. Segment3d: Learning fine-grained class-agnostic 3d segmentation without manual labels. In European Conference on Computer Vision. Springer, 278–295.

Xiufeng Huang, Ka Chun Cheung, Runmin Cong, Simon See, and Renjie Wan. 2025. Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction. arXiv:2507.14921 [cs.CV] https://arxiv.org/abs/2507.14921

Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. 2024a. Mv-adapter: Multi-view consistent image generation made easy. arXiv preprint arXiv:2412.03632 (2024).

Zehuan Huang, Hao Wen, Junting Dong, Yaohui Wang, Yangguang Li, Xinyuan Chen, Yan-Pei Cao, Ding Liang, Yu Qiao, Bo Dai, et al. 2024c. Epidiff: Enhancing multi-view synthesis via localized epipolar-constrained diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9784–9794.

Diederik P Kingma. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013).

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. 2023. Segment Anything. In arXiv.

Lin Li, Zehuan Huang, Haoran Feng, Gengxiong Zhuang, Rui Chen, Chunchao Guo, and Lu Sheng. 2025a. Voxhammer: Training-free precise and coherent 3d editing in native 3d space. arXiv preprint arXiv:2508.19247 (2025).

Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. 2022. Grounded Language-Image Pre-training. In arXiv.

Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long.

2024. CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner. arXiv preprint arXiv:2405.14979 (2024).

Weiyu Li, Jiarui Liu, Hongyu Yan, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. 2025b. CraftsMan3D: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner. arXiv:2405.14979 [cs.GR] https://arxiv.org/abs/2405.14979

Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui Liu, Zihao Wang, Xiao Chen, Feipeng Tian, Jianxiong Pan, Zeming Li, Gang Yu, Xiangyu Zhang, Daxin Jiang, and Ping Tan. 2025c. Step1X3D: Towards High-Fidelity and Controllable Generation of Textured 3D Assets. arXiv:2505.07747 [cs.CV] https://arxiv.org/abs/2505.07747

Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. 2025e. TripoSG: HighFidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models. arXiv preprint arXiv:2502.06608 (2025).

Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, and Yan-Pei Cao. 2025d. TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models. arXiv:2502.06608 [cs.CV] https://arxiv.org/abs/2502.06608

Kevin Lin, Lijuan Wang, and Zicheng Liu. 2021. End-to-End Human Pose and Mesh Reconstruction with Transformers. In CVPR.

Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Yiqiang Feng, Yadong Mu, and Katerina Fragkiadaki. 2025. PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers. arXiv:2506.05573 [cs.CV] https: //arxiv.org/abs/2506.05573

Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. 2024a. Part123: part-aware 3d reconstruction from a single-view image. In ACM SIGGRAPH 2024 Conference Papers. 1–12.

Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. 2024b. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10072–10083.

Minghua Liu, Mikaela Angelina Uy, Donglai Xiang, Hao Su, Sanja Fidler, Nicholas Sharp, and Jun Gao. 2025. PartField: Learning 3D Feature Fields for Part Segmentation and Beyond. In ICCV.

Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. 2024c. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems 36 (2024).

Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2023. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453 (2023).

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2024. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9970–9980.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. arXiv:1711.05101 [cs.LG] https://arxiv.org/abs/1711.05101

Changfeng Ma, Yang Li, Xinhao Yan, Jiachen Xu, Yunhan Yang, Chunshi Wang, Zibo Zhao, Yanwen Guo, Zhuo Chen, and Chunchao Guo. 2025a. P3-sam: Native 3d part segmentation. In arXiv.

Ziqi Ma, Yisong Yue, and Georgia Gkioxari. 2025b. Find Any Part in 3D. In arXiv. Quan Meng, Lei Li, Matthias Nießner, and Angela Dai. 2024. LT3SD: Latent Trees for

3D Scene Diffusion. arXiv preprint arXiv:2409.08215 (2024).

Kaichun Mo, Shilin Zhu, Angel X. Chang, Li Yi, Subarna Tripathi, Leonidas J. Guibas, and Hao Su. 2019. PartNet: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level 3D Object Understanding. In CVPR.

Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2023. DINOv2: Learning Robust Visual Features without Supervision. In arXiv.

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4195–4205.

Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. 2017. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. In arXiv preprint arXiv:1612.00593.

Yansong Qu, Shaohui Dai, Xinyang Li, Yuze Wang, You Shen, Liujuan Cao, and Rongrong Ji. 2025. DeOcc-1-to-3: 3D De-Occlusion from a Single Image via Self-Supervised Multi-View Diffusion. arXiv:2506.21544 [cs.CV] https://arxiv.org/abs/2506.21544

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In arXiv.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2024. SAM 2: Segment Anything in Images and Videos. In arXiv.

Barbara Roessle, Norman Müller, Lorenzo Porzi, Samuel Rota Bulø, Peter Kontschieder, Angela Dai, and Matthias Nießner. 2024. L3DG: Latent 3D Gaussian Diffusion. arXiv preprint arXiv:2410.13530 (2024).

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020). George Tang, William Zhao, Logan Ford, David Benhaim, and Paul Zhang. 2025c. Segment Any Mesh. In arXiv.

Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. 2025a. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision. Springer, 1–18.

Jiaxiang Tang, Ruijie Lu, Zhaoshuo Li, Zekun Hao, Xuan Li, Fangyin Wei, Shuran Song, Gang Zeng, Ming-Yu Liu, and Tsung-Yi Lin. 2025b. Efficient Part-level 3D Object Generation via Dual Volume Packing. arXiv:2506.09980 [cs.CV] https: //arxiv.org/abs/2506.09980

Ardian Umam, Cheng-Kun Yang, Min-Hung Chen, Jen-Hui Chuang, and Yen-Yu Lin.

2024. PartDistill: 3D Shape Part Segmentation by Vision-Language Model Distillation. In arXiv.

Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. 2025. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision. Springer, 439–457. Penghao Wang, Yiyang He, Xin Lv, Yukai Zhou, Lan Xu, Jingyi Yu, and Jiayuan Gu.

2025. PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding. arXiv:2510.20155 [cs.CV] https://arxiv.org/abs/2510.20155 Zhengyi Wang, Jonathan Lorraine, Yikai Wang, Hang Su, Jun Zhu, Sanja Fidler, and Xiaohui Zeng. 2024a. LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models. arXiv:2411.09595 [cs.LG] https://arxiv.org/abs/2411.09595

Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. 2024b. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034 (2024).

Si-Tong Wei, Rui-Huan Wang, Chuan-Zhi Zhou, Baoquan Chen, and Peng-Shuai Wang.

2025. OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation. arXiv:2504.09975 [cs.GR] https://arxiv.org/abs/2504.09975

Hao Wen, Zehuan Huang, Yaohui Wang, Xinyuan Chen, Yu Qiao, and Lu Sheng. 2024. Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion. arXiv preprint arXiv:2406.03184 (2024).

Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. 2024d. Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image. arXiv preprint arXiv:2405.20343 (2024).

Ruiqi Wu, Xinjie Wang, Liu Liu, Chunle Guo, Jiaxiong Qiu, Chongyi Li, Lichao Huang, Zhizhong Su, and Ming-Ming Cheng. 2025b. DIPO: Dual-State Images Controlled Articulated Object Generation Powered by Diverse Data. arXiv:2505.20460 [cs.CV] https://arxiv.org/abs/2505.20460

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. 2024c. Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer. arXiv preprint arXiv:2405.14832 (2024).

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, and Yao Yao. 2025a. Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention. arXiv:2505.17412 [cs.CV] https://arxiv.org/abs/2505.17412

Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. 2024a. Point Transformer V3: Simpler, Faster, Stronger. In CVPR.

Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. 2022. Point transformer V2: Grouped Vector Attention and Partition-based Pooling. In NeurIPS.

Xiaoyang Wu, Zhuotao Tian, Xin Wen, Bohao Peng, Xihui Liu, Kaicheng Yu, and Hengshuang Zhao. 2024e. Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training. In CVPR.

Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu, Hiroyuki Sato, Hongdong Li, et al. 2024b. Blockfusion: Expandable 3d scene generation using latent tri-plane extrapolation. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–17.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, and Jiaolong Yang. 2025a. Native and Compact Structured Latents for 3D Generation. Tech report (2025). Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2025b. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 21469–21480.

Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. 2024. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191 (2024).

Mutian Xu, Xingyilang Yin, Lingteng Qiu, Yang Liu, Xin Tong, and Xiaoguang Han. 2025. SAMPro3D: Locating SAM Prompts in 3D for Zero-Shot Instance Segmentation. In arXiv.

Yuheng Xue, Nenglun Chen, Jun Liu, and Wenyun Sun. 2025. ZeroPS: High-quality Cross-modal Knowledge Transfer for Zero-Shot 3D Part Segmentation. In arXiv.

Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y. Lam, Yan-Pei Cao, and Xihui Liu. 2024a. SAMPart3D: Segment Any Part in 3D Objects. In arXiv.

Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y. Lam, Yan-Pei Cao, and Xihui Liu. 2024b. SAMPart3D: Segment Any Part in 3D Objects. arXiv:2411.07184 [cs.CV] https://arxiv.org/abs/2411.07184

Yunhan Yang, Xiaoyang Wu, Tong He, Hengshuang Zhao, and Xihui Liu. 2023. SAM3D: Segment Anything in 3D Scenes. In arXiv.

Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, and Jun Zhu. 2025. ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding. arXiv:2506.01853 [cs.CV] https://arxiv.org/abs/2506.01853

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. 2024. CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–20.

Ruowen Zhao, Junliang Ye, Zhengyi Wang, Guangce Liu, Yiwen Chen, Yikai Wang, and Jun Zhu. 2025b. DeepMesh: Auto-Regressive Artist-mesh Creation with Reinforcement Learning. arXiv:2503.15265 [cs.CV] https://arxiv.org/abs/2503.15265

Wang Zhao, Yan-Pei Cao, Jiale Xu, Yuejiang Dong, and Ying Shan. 2025a. Assembler: Scalable 3D Part Assembly via Anchor Point Diffusion. arXiv:2506.17074 [cs.CV] https://arxiv.org/abs/2506.17074

Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. 2024. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. Advances in Neural Information Processing Systems 36 (2024).

Yuchen Zhou, Jiayuan Gu, Tung Yen Chiang, Fanbo Xiang, and Hao Su. 2025. Point-SAM: Promptable 3D Segmentation Model for Point Clouds. In The Thirteenth International Conference on Learning Representations.

Yuchen Zhou, Jiayuan Gu, Xuanlin Li, Minghua Liu, Yunhao Fang, and Hao Su. 2023. PartSLIP++: Enhancing Low-Shot 3D Part Segmentation via Multi-View Instance Segmentation and Maximum Likelihood Estimation. In arXiv.

Zhe Zhu, Le Wan, Rui Xu, Yiheng Zhang, Honghua Chen, Zhiyang Dou, Cheng Lin, Yuan Liu, and Mingqiang Wei. 2025. PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data. In arXiv.

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Fig. 8. Interactive demo. Users specify clicks on the 3D asset to perform interactive part segmentation, and can adjust visualization settings.

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

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

Input Find3D SAMPart3D PartField P3-SAM Ours Ours w. 2D Map

- Fig. 9. More qualitative comparisons for full segmentation.

[Figure 125]

- Fig. 10. More qualitative segmentation results of our SegviGen.

[Figure 126]

Input Segmentation and Edit Region Edited 3D Mesh Input Segmentation and Edit Region Edited 3D Mesh

Fig. 11. Interactive 3D editing results with SegviGen and VoxHammer [Li et al. 2025a]. SegviGen provides precise part segmentation to facilitate downstream editing models. The target region to be edited is indicated by green points. It demonstrates the practical utility of SegviGen in downstream 3D editing pipelines.

