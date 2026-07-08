# arXiv:2512.15715v1[cs.CV]17Dec2025

## In Pursuit of Pixel Supervision for Visual Pre-training

Lihe Yang2,1* Shang-Wen Li1 Yang Li1 Xinjie Lei1 Dong Wang1 Abdelrahman Mohamed1 Hengshuang Zhao2 Hu Xu1 1FAIR, Meta 2HKU

https://github.com/facebookresearch/pixio

### Abstract

At the most basic level, pixels are the source of the visual information through which we perceive the world. Pixels contain information at all levels, ranging from lowlevel attributes to high-level concepts. Autoencoders represent a classical and long-standing paradigm for learning representations from pixels or other raw inputs. In this work, we demonstrate that autoencoder-based selfsupervised learning remains competitive today and can produce strong representations for downstream tasks, while remaining simple, stable, and efficient. Our model, codenamed “Pixio”, is an enhanced masked autoencoder (MAE) with more challenging pre-training tasks and more capable architectures. The model is trained on 2B web-crawled images with a self-curation strategy with minimal human curation. Pixio performs competitively across a wide range of downstream tasks in the wild, including monocular depth estimation (e.g., Depth Anything), feed-forward 3D reconstruction (i.e., MapAnything), semantic segmentation, and robot learning, outperforming or matching DINOv3 trained at similar scales. Our results suggest that pixel-space selfsupervised learning can serve as a promising alternative and a complement to latent-space approaches.

### 1. Introduction

Over the past decade, progress in computer vision has consistently been driven by corresponding advances in representation learning. Starting from supervised representation learning [40] based on human annotations (e.g., ImageNet [18]), the vision community has since transitioned to self-supervised representation learning (e.g., [31, 11, 32, 9]) using unannotated data. The rich information contained in the data itself can serve as a powerful source of supervision for learning general-purpose representations.

Modern methods on self-supervised learning generally fall into two categories, depending on where the objec-

*This work was done during an internship at Meta.

tive is formulated: the raw input space (i.e., pixels in the case of vision data) or a latent space produced by models. The first category is represented by denoising autoencoders (DAE) [65], now commonly realized as masked autoencoders (MAE) [32] which learn by predicting unknown pixels under structural corruptions. The second category stems from contrastive learning [28, 31, 11] and has evolved to incorporate various forms of latent-space objectives (e.g., DINO [9], JEPA [2]). Today, the go-to solution for off-theshelf self-supervised learning models is typically DINO and its extensions, whereas MAEs often serve as pre-trained initializations for fine-tuning foundation models [39, 51, 79].

In this paper, we aim to push the frontier of pixel-based self-supervised learning, without relying on any objective defined in latent spaces. We suggest that pixels are ultimately the origin of the visual information through which we perceive the physical world. They inherently contain the desired information at all levels, ranging from low-level attributes (e.g., colors, textures, materials, geometry) to highlevel concepts (e.g., semantics, relations, entities, events). Rather than focus on higher-level abstractions that may treat lower-level signals as “nuance”, we pursue generic visual representations that compress and re-organize information across all levels (see Figure 1).

Our study builds on the MAE paradigm [32], with improvements introduced on both the algorithm and data sides. Algorithmically, we observe that the original MAE design is suboptimal in the large-data, large-model regime we investigate. To bring the models into this regime, we increase the pre-training difficulty by adopting larger masking blocks, and strengthen the model’s capability by deepening the AE’s decoder and enlarging the set of class tokens (see Figure 2). We empirically show that these simple designs facilitate representation learning by providing a sufficiently challenging task for a capable model to solve.

On the data side, we largely close the gap between the original MAE [32], trained on ImageNet [18], and the DINO family [9, 47], which benefit from training on carefully curated web-scale data. We build training data at similar scale with less manual curation, avoiding bias toward

[Figure 1]

[Figure 2]

(a) model learns object co-occurrence patterns (left and right doors) (b) model infers hidden camera pose and 3D spatial layout

[Figure 3]

[Figure 4]

(c) model reasons about symmetric color patterns (left red color is masked) (d) model understands reflection and predicts the mirrored person

- Figure 1. Pixel supervision compels the model to compress and re-organize visual knowledge across all levels. To accurately predict pixels, the model must understand geometry, texture, semantics, materials, lighting, etc. By masking and pixel reconstruction, MAE learns these desirable visual properties and even exhibits early reasoning capabilities [71]. From left to right in each group: masked input, reconstructed image (visible patches are kept), ground truth image (unseen during training).

specific benchmark distributions. Specifically, we collect a diverse pool of 2 billion web-crawled images [74] and employ a soft self-curation strategy, where each image’s sampling probability is determined by its reconstruction loss. We demonstrate that our MAE-driven algorithm significantly benefits from such diverse, large-scale data, greatly outperforming its original version trained on the limited ImageNet-1K dataset.

We evaluate our system, codenamed “Pixio”, across a diverse range of vision tasks. We observe that in tasks where preserving lower-level details is important, such as monocular depth estimation (e.g., Depth Anything [75]), feed-forward 3D reconstruction (i.e., MapAnything [38]), semantic segmentation, Pixio outperforms or matches the state-of-the-art DINOv2/v3 counterparts. It also delivers very promising results on robot learning tasks. Collectively, our results suggest that pixel-space self-supervised learning can serve as a competitive alternative and a complement to latent-space approaches.

### 2. Related Work

This section reviews the evolution of supervision signals in visual representation learning, discussing how the definition of ground truth has progressed.

Early deep visual representation learning [40, 58, 29] relies on explicit human annotations to learn transferable features. In this stage, ImageNet [18] class labels are typically treated as the ground truth, providing the primary supervision signal. However, a single categorical label is insufficient to describe an image. Such a pre-training paradigm is restricted to human-defined concepts and faces scalabil-

ity challenges, yielding marginal gains when transferring to downstream tasks [30].

For more scalable and richer supervision, CLIP [49] resorts to web-crawled image-text pairs, scaling the pretraining data to 400M and facilitating open-vocabulary applications [41, 26]. However, alt-text remains an imperfect proxy for ground truth. Captions are constrained by human knowledge and linguistic expression, often providing only partial descriptions [78] or even irrelevant information. Recent efforts [62, 15] continue to refine this paradigm, but fundamental limitations persist.

Both class labels and web captions represent projections of the physical world through human cognition and natural language. Such high-level abstractions can produce promising results in the short term [77, 74], as they are directly distilling human knowledge. Nevertheless, they will ultimately hinder the upper bound of visual intelligence due to inherent human bias. For instance, humans naturally focus on object existence and attributes, rather than complex interentity relationships [70]. Many visual phenomena, e.g., subtle lighting changes, intricate spatial arrangements, or abstract aesthetic qualities, are difficult or impossible to verbalize. Moreover, such paired data is not infinitely scalable and can become a new bottleneck [64].

Self-supervised learning methods [72, 31, 32] avoid using explicit human annotations. However, human priors remain embedded through carefully designed pretext tasks. The inductive bias from humans implicitly serves as the ground truth guiding models to learn the physical world. These methods typically first perform some artificial distortions on the raw input and then train the model to be in-

##### MAE

[Figure 5]

[Figure 6]

|CLS|
|---|

[Figure 7]

encoder

decoder

Where have we been? Where are we going?

larger mask block

more [CLS] tokens

LI F E I – F E I & J I A D E N G

web-scale training data

deeper decoder

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

CLS

encoder decoder

Pixio

- Figure 2. Pixio introduces four simple yet critical updates to MAE, with following motivations. Deeper decoder: MAE’s shallow decoder lacks capacity for pixel regression, forcing the encoder to sacrifice representation quality for reconstruction. Larger mask block: singlepatch masking causes reconstruction shortcuts and provides insufficient context. More [CLS] tokens: a single class token cannot capture diverse global properties. Web-scale training data: IN-1K lacks the visual diversity needed for learning transferable representations.

variant to them. The distortions can be simple color perturbations [11, 12], spatial shuffling [46], spatial masking [4, 73, 83], token removal [32], Gaussian noise [14], etc. The supervision can be applied in either latent space [33, 60, 8, 3, 4] or directly in pixel space [65, 48, 10, 73, 32]. For latent-space practices, inductive bias (e.g., multi-view consistency [11, 9]) is necessary to prevent model collapse and drive the model to learn towards the direction as human hopes or benchmark [18] prefers.

This work advocates for pixel as a promising supervision for visual representation learning. It introduces less human bias than other artificially constructed learning targets [72, 11, 9]. Humans only decide what images or videos to capture, not what features to emphasize or what invariances to enforce. Although pixel remains an imperfect representation of our complex physical world, it is fundamentally grounded in observable reality with minimal abstraction and human intervention, especially compared to human-defined categories, human-written texts, or humandesigned visual priors. We demonstrate highly capable vision models can be learned through direct pixel prediction. It offers a scalable and less biased path toward advancing visual intelligence [71].

- 3. Pixio

#### 3.1. Preliminary of MAE

MAE adapts BERT [19] to visual domain. The core idea is to mask partial input signals and train the model to recover them. Three key designs distinguish MAE: 1) direct pixel supervision, 2) disentangling visible and masked tokens with an asymmetric encoder-decoder architecture, and 3) a high masking ratio to mitigate (also leverage) visual redundancy. We explain these three aspects below.

Unlike masked prediction in discrete latent space [4, 3], MAE shows pixel is simple, feasible, and more effective

- as direct supervision [10]. Notably, it does not require stabilization mechanisms such as negative samples [11], stop gradient [31], or careful centering and sharpening [9].

Before MAE, masked modeling approaches [19, 20, 4] partially replace input tokens with special [MASK] tokens, resulting in train-test distribution shift. Processing all tokens also imposes heavier computational burden on the encoder. MAE elegantly addresses both limitations by removing masked tokens at the encoder and appending them back

- at the lightweight decoder. Compared to discrete and semantically rich text tokens,

visual signals (e.g., image, video) are contiguous, low-level, and redundant [61, 24]. MAE demonstrates that a high masking ratio (e.g., 75%) is necessary to avoid ground truth leakage and construct a meaningful pretext task [80].

#### 3.2. MAE Redesign

Pixio builds upon MAE with four critical modifications. So we first review MAE’s core designs in Section 3.1. Then, we introduce our algorithm improvements in Section 3.2 and pre-training data update in Section 3.3.

We preserve the three core designs of MAE, but revisit several overlooked components that are critical for performance. Our redesigns are summarized in Figure 2.

Deeper decoder. As shown in Figure 3, the best generic feature (e.g., for classification, depth estimation, semantic segmentation) in MAE does not reside in the final encoder block [1, 5]. For a 32-block ViT-H encoder, optimal feature emerges as early as the 20th block.

We conjecture that the decoder lacks sufficient capacity for pixel regression. To minimize reconstruction loss, the encoder has to sacrifice some capacity (e.g., later blocks) in modeling low-level details (i.e., serve as the “decoder”), rather than prioritizing semantic understanding as expected. Under this hypothesis, our solution becomes straightforward: increasing the decoder depth. Simply adding more decoder blocks (while still maintaining lightweight overall overhead) yields substantial improvements (Figure 4). Note that over-parameterized decoder is also not appropriate, as models may learn to prioritize memorizing visual details over learning transferable representations. Additionally, an overly powerful decoder can cause encoder laziness, with the encoder relying on the decoder for representation learning and yielding suboptimal representations.

Larger mask block. MAE randomly drops individual patch tokens at the encoder input. However, as illustrated in Figure 2, single-patch masking disrupts local context. More critically, the model can achieve plausible reconstruction by simply copying from nearby visible patches, without genuine visual understanding. Therefore, we mask at a larger granularity, e.g., 4×4 local patch blocks. This provides richer local context for learning and mitigates ground truth leakage. Nevertheless, excessively large masking units (e.g., 8×8 patches) are not recommended, as masked regions will become unpredictable.

Large-context masking is indeed not new in masked image modeling [73, 34, 2, 47]. In this work, we conduct comprehensive ablation studies to provide more insights on the correlation between masking ratio, masking granularity, and downstream performance (Figure 5).

More [CLS] tokens. MAE appends a class token [19] alongside patch tokens. Unlike image-level contrastive learning [13, 9], this token receives no explicit loss supervision, yet works effectively in classification tasks. This token implicitly captures global information, enabling patch tokens to perform local-global interaction. A single class token is insufficient to capture diverse global visual properties, such as scene type, image style, object concepts, camera pose, etc. Therefore, we append multiple class tokens. For downstream tasks requiring a global representation, we can either average or concatenate them.

These tokens relate to ViT register tokens [16, 17, 57], but serve different roles. Our class tokens function as global representations used directly in downstream tasks (e.g., image classification, robot learning), rather than being discarded during evaluation [16].

#### 3.3. Web-Scale Data and Curation

Earlier visual self-supervised learning works were mostly trained on small-scale IN-1K. Later, DINOv2 [47] demonstrates large-scale data with diverse concepts [66] is essential for learning robust and transferrable representations. However, DINOv2 and DINOv3 [57] employ excessive benchmark-centric data curation. For example, they use benchmark images as queries to retrieve similar training images from a large pool. They also directly inject benchmark training images (e.g., IN-1K, Mapillary [45]) with excessive repetitive sampling (can be 100×).

Undoubtedly, such careful curation can yield strong benchmark results in the short run. However, it may cause the model to be fragile to future unknown scenarios with different data distributions. We thus advocate pre-training on large-scale, conceptually diverse, and minimally-curated data to avoid benchmark bias.

We first follow MetaCLIP [74] to collect 2 billion webcrawled images, covering substantially more diverse scenes than curated benchmarks [18, 59]. However, the raw distribution is dominated by product images and text-heavy images (e.g., documents) [23], which should not be overemphasized in generic representation learning. We thus apply two complementary curation strategies. First, we employ loss-based soft sampling. We pre-compute reconstruction loss l for each image using a Pixio model trained on the raw data. We then probabilistically sample training images: image i is accepted if li ≥ u, where u ∼ U(0,1). This strategy downsamples easy-to-reconstruct images (e.g., product images), while highlighting challenging, visually rich content. Second, we filter images with low histogram entropy in colors to reduce text-heavy images, which may exhibit high reconstruction loss but limited scene diversity. Together, these two strategies preserve rich and diverse visual content with minimal curation bias.

### 4. Experiments

Earlier self-supervised learning works [32, 9] primarily focused on image classification tasks. Such scenarios are now more appropriate to be addressed by open-vocabulary classifiers [49, 74] or MLLMs [36]. Therefore, We primarily evaluate tasks that require dense visual representations [47], such as depth estimation, 3D reconstruction, and segmentation, which remain challenging to MLLMs.

Brief introduction of our setup. Our largest ViT-5.4B/16 model is pre-trained on 2B curated web-crawled images for 20B seen samples over 1.3M iterations with a batch size of 16384. The input resolution is 256×256. We employ a ViT decoder with 512 dimensions and 32 blocks, apply masking at 4×4 patch granularity, and append 8 class tokens. In the main paper, we mainly compare our distilled Pixio-H encoder (631M params) against the state-of-the-

DPT Head Linear Regression Head

Method ViT #Params

NYUv2 KITTI NYUv2 KITTI RMSE ↓ δ1 ↑ RMSE ↓ δ1 ↑ RMSE ↓ δ1 ↑ RMSE ↓ δ1 ↑ MAE H/14 631M 0.465 80.8 2.740 90.9 0.595 70.3 4.419 79.4 DINOv2 L/14 304M 0.384 88.0 2.510 94.7 0.599 72.0 4.928 73.7 DINOv2 g/14 1137M 0.355 90.1 2.424 94.6 0.560 75.3 4.425 78.1 DINOv3 H+/16 841M 0.320 93.2 2.386 95.6 0.559 76.3 4.944 73.2

###### Pixio H/16 631M 0.268 95.5 2.210 96.7 0.366 90.8 3.494 90.3

- Table 1. Domain-specific monocular metric depth estimation with frozen encoder and a trainable DPT (Dense Prediction Transformer) head or a linear regression head.

Method ViT #Params

NYUv2 KITTI DIODE-In DIODE-Out Sintel DA-2K

rel ↓ δ1 ↑ rel ↓ δ1 ↑ rel ↓ δ1 ↑ rel ↓ δ1 ↑ rel ↓ δ1 ↑ acc ↑ MAE H/14 631M 0.054 97.0 0.090 91.8 0.047 97.0 0.087 92.7 0.650 71.2 94.1 DINOv2 L/14 304M 0.050 97.6 0.077 94.0 0.046 97.6 0.088 92.8 0.505 73.4 95.5 DINOv2 g/14 1137M 0.046 97.9 0.074 94.6 0.041 98.1 0.085 93.0 0.450 75.3 96.9 DINOv3 H+/16 841M 0.044 98.2 0.081 94.0 0.038 98.4 0.081 93.4 0.393 75.6 97.5

Pixio H/16 631M 0.041 98.0 0.083 93.4 0.034 98.6 0.075 94.3 0.535 75.8 97.3

- Table 2. Depth Anything V2’s scenario for zero-shot monocular relative depth estimation. The model is trained on a combination of five synthetic datasets, and then transferred to unseen datasets. The entire model (encoder + DPT head) is trainable.

art DINOv3-H+ encoder (841M params) [57], which is the second-largest version distilled from their largest 7B model. Additional details are provided in the appendix.

#### 4.1. Monocular Depth Estimation

Domain-specific monocular depth estimation. We freeze the pre-trained encoder and add a trainable DPT head [50] or linear regression head on it. The metric depth estimation model is trained and evaluated on the same domain (NYUv2 [56] or KITTI [27]). As compared in Table 1, our Pixio clearly outperforms the most capable DINOv3 model under both DPT and linear heads, e.g., reducing RMSE from 0.320 → 0.268 and improving δ1 from 93.2 → 95.5 on NYUv2. Compared to the initial MAE model, the improvement is substantial (RMSE: 0.465 → 0.268, δ1: 80.8 → 95.5), demonstrating the necessity of our modifications to both algorithm and data.

Depth Anything. In addition to domain-specific depth estimation, we further follow Depth Anything V2 [75] to evalute zero-shot monocular relative depth estimation, where the model is trained on five synthetic datasets and tested on unseen distributions [6, 63]. As shown in Table 2, Pixio outperforms or matches DINOv3 on most benchmarks (e.g., NYUv2, DIODE, DA-2K), and is inferior to DINOv2/v3 on the autonomous driving benchmark KITTI. This is expected, since we do not inject abundant driving-related im-

ages as DINOv2 does, which directly uses over 1 million Mapillary images [45].

#### 4.2. Feed-Forward 3D Reconstruction

Feed-forward 3D reconstruction [68] requires strong visual capability in understanding spatial layout and capturing multi-view dense correspondence. We strictly follow MapAnything’s [38] training framework and compare the performance of different pre-trained encoders. As shown in Table 3, evaluated across indoor [76], outdoor [53], and synthetic scenes [69, 81], Pixio consistently delivers better reconstruction and pose estimation results than MAE, DINOv2, and DINOv3. This demonstrates that although trained with a single view, Pixio exhibits stronger multiview capability than pre-training frameworks [47, 57] that explicitly use multiple views (e.g., 8 views in DINOv3).

#### 4.3. Semantic Segmentation

Semantic segmentation requires dense classification of each pixel. We evaluate on two natural image benchmarks (ADE20K [82] and Pascal VOC [21]) and one satellite image benchmark LoveDA [67]. As shown in Table 4, Pixio achieves competitive results outperforming or on par with the state-of-the-art DINOv3 model, despite employing a simpler pre-training objective and requiring no benchmarkspecific data curation. Note that our model uses 200M fewer

ScanNet++ v2

Method ViT

Scale Points Pose Depth rel ↓ rel ↓ τ ↑ auc5 ↑ rel ↓ τ ↑

MAE H/14 0.050 0.057 63.3 65.6 0.058 55.4 DINOv2† L/14 0.041 0.052 67.6 73.2 0.052 60.6 DINOv3 H+/16 0.035 0.051 69.0 68.5 0.051 61.2

Pixio H/16 0.029 0.041 78.8 80.5 0.042 72.4 ETH3D

MAE H/14 0.161 0.138 36.1 20.2 0.100 19.2 DINOv2† L/14 0.204 0.130 37.6 23.5 0.095 24.8 DINOv3 H+/16 0.156 0.146 39.8 26.9 0.096 18.6

Pixio H/16 0.160 0.120 51.3 37.8 0.080 34.6 TartanAirV2-WB

MAE H/14 0.217 0.178 28.1 22.0 0.148 19.7 DINOv2† L/14 0.291 0.222 30.2 23.6 0.154 20.6 DINOv3 H+/16 0.342 0.189 29.9 22.4 0.133 24.5

Pixio H/16 0.224 0.185 34.1 38.4 0.111 26.7

- Table 3. MapAnything [38] for feed-forward 3D reconstruction at two-view input. The input is images only. All models are trained on six Apache datasets. †: official MapAnything model.

parameters and is distilled from a model with 1.3B fewer parameters.

- 4.4. Robot Learning

We evaluate Pixio on CortexBench [43], using four benchmarks: Adroit, DMC, MetaWorld, and Trifinger. We compare against three specialized models (VC1 [43], R3M [44], Theia [54]) and two generic models (DINOv2, DINOv3). For fair comparison, we select the best-performed embedding configuration (either global embedding or spatial embeddings) for each model. For models using spatial embeddings, we follow Theia [54] to employ a three-layer CNN followed by an MLP for policy prediction. We average the class tokens of Pixio as its global embedding, which we find more effective than its spatial embeddings in this scenario. It also eliminates the computational overhead of the CNNbased policy network. As shown in Table 5, Pixio is 1.2% better than R3M and 3.1% better than DINOv3, indicating its strong capability in robot learning.

- 4.5. Ablation Study

Unless otherwise specified, we train a ViT-H encoder on ImageNet-21K for 400 epochs with batch size 8192 (∼5B seen images). We use IN-21K because it is publicly accessible and makes our results reproducible. For evaluation, by default, we use a DPT head for depth estimation and a linear head for semantic segmentation.

DPT Head Linear Head ADE VOC LoveDA ADE VOC LoveDA

Model

MAE-H 37.6 76.0 50.2 35.2 70.8 47.6 DINOv2-L 50.1 84.6 55.2 47.4 80.2 52.2 DINOv2-g 51.5 85.2 55.0 49.0 81.8 51.9 DINOv3-H+ 52.3 85.6 55.3 50.3 82.1 52.7

###### Pixio-H 53.6 85.9 54.7 50.2 82.2 53.9

- Table 4. Semantic segmentation with frozen encoder and trainable DPT head or linear classification head.

Method Model Adroit DMC MW Trifinger Avg VC1 ViT-L 52.0 81.3 88.3 67.5 72.3 R3M RN-50 74.7 72.4 94.1 67.4 77.2 Theia ViT-B 60.0 79.9 87.7 65.8 73.4 DINOv2 ViT-L 64.0 70.4 90.9 66.9 73.1 DINOv3 ViT-H+ 63.3 78.5 89.3 70.0 75.3 Pixio ViT-H 70.7 77.5 92.8 72.8 78.4

- Table 5. CortexBench for robot learning. DMC: DeepMind Control Suite. MW: MetaWorld.

Decoder. As shown in Figure 3, the best generic feature of the MAE-H encoder is far before the last encoder block. Based on our previous analysis, increasing the decoder capacity is an intuitive approach to address this issue. In Figure 4, we conduct a comprehensive ablation study on the decoder’s embedding dimension (width) and number of attention blocks (depth). By simply increasing the decoder depth from 8 (default) to 32, all downstream performance is tremendously enhanced, e.g., IN-1K k-NN accuracy 35.3 → 55.8, NYUv2 depth error 0.431 → 0.410, and ADE20K mIoU 35.8 → 40.4. Similar improvements are achieved by increasing the decoder width from 512 (default) to 768 and depth from 8 (default) to 16. However, the decoder cannot be too heavy, as it may replace the role of the encoder. Further increasing the decoder capacity (e.g., 768×32) produces unsatisfactory results.

Masking granularity. We compare performance under different masking granularity in Figure 5. Since optimal masking granularity may correlate with masking ratio, we evaluate two masking ratios: 75% and 62.5%. First, under MAE’s default decoder configuration (512×8) and masking ratio (75%), merely changing the masking granularity from a single patch to 2×2 patches improves IN-1K k-NN accuracy by 19.0, reduces NYUv2 depth error from 0.431 → 0.362, and improves ADE20K mIoU by 6.0. Similar trends are observed under the better 512×32 decoder configuration. We also tried to further enlarge the masking context to 8×8 patches, but the results are poor, as overly large masking context will cause unpredictability.

Pascal SemSeg mIoU

IN-1K k-NN Acc

ADE20K SemSeg mIoU

NYUv2 Depth RMSE

KITTI Depth RMSE

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

74

38

60

0.70

4.8

72

55

0.68

36

4.7

0.66

50

70

4.6

34

0.64

45

68

4.5

0.62

32

40

66

4.4

0.60

35

30

64

4.3

0.58

30

62

0.56

4.2

28

0.5 0.6 0.7 0.8 0.9 1.0

0.5 0.6 0.7 0.8 0.9 1.0

0.5 0.6 0.7 0.8 0.9 1.0

0.5 0.6 0.7 0.8 0.9 1.0

0.5 0.6 0.7 0.8 0.9 1.0

Relative Block Depth

Relative Block Depth

Relative Block Depth

Relative Block Depth

Relative Block Depth

ViT-H/14 ViT-L/16

- Figure 3. Probing frozen features in different blocks of the original MAE encoder, which is trained on ImageNet-1K. The relative block depth is computed as the ratio of the block index to the total number of blocks, for easy comparison across architectures (ViT-H: 32 blocks, ViT-L: 24 blocks). We use a linear head for both monocular depth estimation (regression) and semantic segmentation (classification).

8 (MAE) 16 32 48

Decoder Depth

35

40

45

50

55

| |
|---|

| |
|---|

| |
|---|

IN-1K k-NN Acc

8 (MAE) 16 32 48

Decoder Depth

0.400

0.425

0.450

0.475

0.500

0.525

0.550

0.575

| |
|---|

| |
|---|

NYUv2 Depth RMSE

8 (MAE) 16 32 48

Decoder Depth

2.8

2.9

3.0

3.1

3.2

3.3

3.4

| |
|---|

| |
|---|

KITTI Depth RMSE

| || |
|---|
<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| || |
|---|
<br><br>| | | | |
| || |
|---|
<br><br>| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| || |
|---|
| | | | |
| | | | | | |

8 (MAE) 16 32 48

Decoder Depth

26

28

30

32

34

36

38

40

42

ADE20K SemSeg mIoU

8 (MAE) 16 32 48

Decoder Depth

64

66

68

70

72

74

76

| |
|---|

| |
|---|

| |
|---|

Pascal SemSeg mIoU

Decoder Width 768 Decoder Width 512 (MAE) Decoder Width 384

- Figure 4. Ablation study of using decoders of different depth (#attention blocks) or width (feature dimension) to train MAE on IN-21K. The encoder is ViT-H (1280-d × 32-blocks). Here, we use a DPT head for depth estimation and a linear head for semantic segmentation.

IN-1K NYUv2 KITTI ADE20K Pascal k-NN ↑ RMSE ↓ RMSE ↓ mIoU ↑ mIoU ↑

Method

MAE 37.9 0.392 2.899 37.2 67.4 Pixio 59.5 0.321 2.581 46.8 80.2

Table 6. Ablation study on the combination of our framework updates. Both methods are pre-trained on the curated 2B images. MAE: decoder 512×8, masking at 1×1 patch, 1 class token. Pixio: decoder 512×32, masking at 2×2 patches, 4 class tokens.

Class tokens. MAE uses a single class token for global information. In Figure 6, we examine the performance with more class tokens. The IN-1K classification is steadily improved by increasing the number of class tokens, e.g., k-NN accuracy 63.3 → 75.1 when increased from 1 → 4. Minor gains are also obtained on dense prediction tasks.

All three framework updates. We further check the performance gain from incorporating all three proposed modifications into MAE. As shown in Table 6, Pixio substantially enhances MAE across image classification (37.9 → 59.5 on IN-1K), depth estimation (0.392 → 0.321 on NYUv2 ↓) and semantic segmentation (37.2 → 46.8 on ADE20K). Notably, even trained on large-scale curated 2B images, the original MAE framework fails to deliver promising results, highlighting the importance of first establishing a solid pre-training framework.

IN-1K NYUv2 ADE20K Pascal k-NN ↑ RMSE ↓ mIoU ↑ mIoU ↑

Data Curated

IN-1K ✓ 77.2 0.395 42.9 77.1 IN-21K ✓ 75.2 0.360 44.8 80.7 YFCC ✓ 59.2 0.345 44.6 80.3

✗ 54.2 0.351 44.7 78.0 ✓ 59.5 0.321 46.8 80.2

Ours: 2B

Table 7. Comparison of different training data sources. IN-1K: 1.3M images. IN-21K: 13M images. YFCC: 99M images.

Data sources and data curation. Data is fundamental to self-supervised pre-training. We compare different data sources, including IN-1K, IN-21K, YFCC100M [59], and our collected 2B web images [74]. All models are trained for 5B seen samples. The first three sources are delicately curated with substantial human efforts [42]. They feature balanced yet limited visual concepts and narrow image distributions. In contrast, the web-crawled data source is diverse across all dimensions but uncurated, with dominant product images and text-heavy images (e.g., documents) that are suboptimal for generic visual pre-training.

As shown in Table 7, for dense prediction tasks, IN-21K and YFCC100M substantially outperform IN-1K, demonstrating the necessity of larger-scale pre-training data. Meanwhile, although the uncurated 2B images yield infe-

- 65

| |
|---|

| |
|---|

| |
|---|

IN-1K k-NN Acc

1×1 (MAE) 2×2 4×4

Masking Granularity (#patches)

0.36

0.38

0.40

0.42

0.44

0.46

| |
|---|

| |
|---|

| |
|---|

NYUv2 Depth RMSE

1×1 (MAE) 2×2 4×4

Masking Granularity (#patches)

2.65

2.70

2.75

2.80

2.85

2.90

2.95

3.00

| |
|---|

| |
|---|

KITTI Depth RMSE

1×1 (MAE) 2×2 4×4

Masking Granularity (#patches)

32

34

36

38

40

42

44

| |
|---|

| |
|---|

ADE20K SemSeg mIoU

1×1 (MAE) 2×2 4×4

Masking Granularity (#patches)

66

68

70

72

74

76

78

80

| |
|---|

| |
|---|

Pascal SemSeg mIoU

Decoder 512×32, Mask 75% Decoder 512×32, Mask 62.5%

| | | |
|---|---|---|
| | | |

Decoder 512×8, Mask 75% (MAE)

| | | |
|---|---|---|
| | | |

Decoder 512×8, Mask 62.5%

Figure 5. Ablation study on masking granularity (measured in #patches). MAE uses single-patch (1×1) masking granularity.

1 (MAE) 4 8 16

#Class Tokens

62

64

- 66

60

55

50

45

40

35

1×1 (MAE) 2×2 4×4

Masking Granularity (#patches)

Pascal SemSeg mIoU

IN-1K k-NN Acc

ADE20K SemSeg mIoU

NYUv2 Depth RMSE

KITTI Depth RMSE

80.75

45.0

2.78

0.360

74

80.50

44.8

2.76

0.358

72

80.25

44.6

2.74

0.356

70

80.00

44.4

0.354

2.72

68

79.75

0.352

44.2

2.70

79.50

0.350

44.0

2.68

79.25

0.348

43.8

2.66

0.346

79.00

1 (MAE) 4 8 16

1 (MAE) 4 8 16

1 (MAE) 4 8 16

1 (MAE) 4 8 16

#Class Tokens

#Class Tokens

#Class Tokens

#Class Tokens

Decoder 512×32 Decoder 384×32

Figure 6. Ablation study on the number of class tokens. MAE uses a single class token.

IN-1K NYUv2 KITTI ADE20K Pascal k-NN ↑ RMSE ↓ RMSE ↓ mIoU ↑ mIoU ↑

Model

Pixio-5B 68.4 0.238 0.213 50.2 82.0 Pixio-1B 66.4 0.247 0.217 50.8 82.5 Pixio-H 63.2 0.268 0.221 50.2 82.2 Pixio-L 67.7 0.286 0.233 49.3 81.7 Pixio-B 64.0 0.373 0.278 45.5 79.9

Table 8. Performance of various Pixio models. Depth estimation uses a DPT head, while semantic segmentation uses a linear head.

rior results compared to the highly curated IN-21K, performance is greatly improved after our simple curation procedure. We advocate for pre-training on such webcrawled sources as they are up-to-date and scalable, providing greater potential for future model scaling.

#### 4.6. Performance of Distilled Models

Results of our distilled more efficient student models and the largest teacher model (5B) are shown in Table 8. Although requiring significantly fewer parameters, most students can match the performance of the teacher.

#### 4.7. Fine-tuning for ImageNet

In addition to k-NN evaluation results on ImageNet-1K, we further examine results under fine-tuning [32]. As compared in Figure 7, while Pixio lags behind DINOv3 in linear probing (0 block), the gap narrows substantially when further fine-tuning the last encoder block. We want to em-

89.2 89.1

| |78.2<br><br>86.9<br><br>87.4 87.7 87.4 87.3 87.3 87.3 87.4<br><br>77.0<br><br>79.0<br><br>79.7<br><br>82.4<br><br>83.6<br><br>85.2<br><br>86.1<br><br>86.6 86.9<br><br>87.4<br>88.9 88.9 88.8 88.9 88.9 88.9<br><br><br>MAE ViT-H<br><br>DINOv3 ViT-H+<br><br>Pixio ViT-H| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

89

87

ImageNetAcc

85

83

81

79

77

75

0 1 2 4 8 12 18 24 32

#Blocks Fine-tuned

Figure 7. Fine-tuning encoder blocks for ImageNet. “0 block” corresponds to linear probing. Pixio does not intentionally involve any ImageNet data for pre-training, while DINOv3 adds ImageNet images explicitly into its training set with repetitive sampling.

phasize that, as a generic visual encoder, Pixio does not intentionally involve any ImageNet data for pre-training, while DINOv3 explicitly incorporates ImageNet-1K images (1.3M) into its training set, which constitute 10% of its total training data (1689M) due to repetitive sampling.

### 5. Conclusion

This work demonstrates pixel can serve as a very promising supervision for large-scale visual pre-training, meantime enjoying simplicity, stability, and efficiency. We introduce three minimal algorithm updates to MAE and train our Pixio on 2B web-crawled images with a simple selfcuration strategy. Pixio delivers results better than or comparable to the state-of-the-art DINOv3 model.

### Acknowledgments

We thank Kaiming He, Saining Xie, Lifei Huang, Ramya Raghavendra, Stephane Kasriel, and Rob Fergus for their helpful discussions and valuable feedback.

### References

- [1] Benedikt Alkin, Lukas Miklautz, Sepp Hochreiter, and Johannes Brandstetter. Mim-refiner: A contrastive learning boost from intermediate pre-trained representations. In ICLR, 2025. 4
- [2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In CVPR, 2023. 1, 4
- [3] Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In ICML, 2022. 3
- [4] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In ICLR, 2022. 3
- [5] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, et al. Perception encoder: The best visual embeddings are not at the output of the network. In NeurIPS, 2025. 4
- [6] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In ECCV, 2012. 5
- [7] John Canny. A computational approach to edge detection. TPAMI, 2009. 2
- [8] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020. 3
- [9] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 1, 3, 4
- [10] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020. 3
- [11] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 1, 3
- [12] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv:2003.04297, 2020. 3
- [13] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, 2021. 4
- [14] Xinlei Chen, Zhuang Liu, Saining Xie, and Kaiming He. Deconstructing denoising diffusion models for self-supervised learning. In ICLR, 2025. 3
- [15] Yung-Sung Chuang, Yang Li, Dong Wang, Ching-Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang,

Jason Weston, Luke Zettlemoyer, et al. Meta clip 2: A worldwide scaling recipe. In NeurIPS, 2025. 2

- [16] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In ICLR,

- 2024. 4

[17] Timoth´ee Darcet, Federico Baldassarre, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Cluster and predict latent patches for improved masked image modeling. TMLR,

- 2025. 4, 1

- [18] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 1, 2, 3, 4
- [19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. 3, 4
- [20] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3
- [21] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. IJCV, 2010. 5
- [22] David Fan, Jue Wang, Shuai Liao, Yi Zhu, Vimal Bhat, Hector Santos-Villalobos, Rohith MV, and Xinyu Li. Motionguided masking for spatiotemporal representation learning. In ICCV, 2023. 1
- [23] David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, et al. Scaling language-free visual representation learning. In ICCV, 2025. 4
- [24] Christoph Feichtenhofer, Yanghao Li, Kaiming He, et al. Masked autoencoders as spatiotemporal learners. In NeurIPS, 2022. 3
- [25] Letian Fu, Long Lian, Renhao Wang, Baifeng Shi, Xudong Wang, Adam Yala, Trevor Darrell, Alexei A Efros, and Ken Goldberg. Rethinking patch dependence for masked autoencoders. arXiv:2401.14391, 2024. 1, 2
- [26] Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. IJCV, 2024. 2
- [27] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In CVPR, 2012. 5
- [28] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR,

2006. 1

- [29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR,

2016. 2

- [30] Kaiming He, Ross Girshick, and Piotr Doll´ar. Rethinking imagenet pre-training. In ICCV, 2019. 2
- [31] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 1, 2, 3
- [32] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 1, 2, 3, 4, 8

- [33] Olivier Henaff. Data-efficient image recognition with contrastive predictive coding. In ICML, 2020. 3
- [34] Ronghang Hu, Shoubhik Debnath, Saining Xie, and Xinlei Chen. Exploring long-sequence masked autoencoders. arXiv:2210.07224, 2022. 4
- [35] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In ECCV,

2016. 4

- [36] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv:2410.21276, 2024. 4
- [37] Ioannis Kakogeorgiou, Spyros Gidaris, Bill Psomas, Yannis Avrithis, Andrei Bursuc, Konstantinos Karantzalos, and Nikos Komodakis. What to hide from your students: Attention-guided masked image modeling. In ECCV, 2022. 1
- [38] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, et al. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv:2509.13414, 2025. 2, 5, 6, 4
- [39] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. In CVPR, 2023. 1
- [40] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012. 1, 2
- [41] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In CVPR, 2023. 2
- [42] Zhuang Liu and Kaiming He. A decade’s battle on dataset bias: Are we there yet? In ICLR, 2025. 7
- [43] Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, et al. Where are we in the search for an artificial visual cortex for embodied intelligence? In NeurIPS, 2023. 6, 4
- [44] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. In CoRL, 2022. 6
- [45] Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulo, and Peter Kontschieder. The mapillary vistas dataset for semantic understanding of street scenes. In ICCV, 2017. 4, 5
- [46] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV,

2016. 3

- [47] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. TMLR, 2024. 1, 4, 5
- [48] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016. 3

- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 4
- [50] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, 2021. 5, 4
- [51] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In ICLR, 2025. 1
- [52] Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, and Herv´e J´egou. Spreading vectors for similarity search. In ICLR, 2019. 1
- [53] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In CVPR, 2017.

- 5

[54] Jinghuan Shang, Karl Schmeckpeper, Brandon B May, Maria Vittoria Minniti, Tarik Kelestemur, David Watkins, and Laura Herlant. Theia: Distilling diverse vision foundation models for robot learning. arXiv:2407.20179, 2024.

- 6

- [55] Jeongwoo Shin, Inseo Lee, Junho Lee, and Joonseok Lee. Self-guided masked autoencoder. In NeurIPS, 2024. 1
- [56] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 5
- [57] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv:2508.10104, 2025. 4, 5
- [58] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR,

2015. 2

- [59] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 2016. 4, 7
- [60] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In ECCV, 2020. 3
- [61] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In NeurIPS, 2022. 3
- [62] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv:2502.14786, 2025. 2
- [63] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv:1908.00463, 2019. 5

- [64] Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. Position: Will we run out of data? limits of llm scaling based on humangenerated data. In ICML, 2024. 2
- [65] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008. 1, 3
- [66] Huy V Vo, Vasil Khalidov, Timoth´ee Darcet, Th´eo Moutakanni, Nikita Smetanin, Marc Szafraniec, Hugo Touvron, Camille Couprie, Maxime Oquab, Armand Joulin, et al. Automatic data curation for self-supervised learning: A clustering-based approach. TMLR, 2024. 4
- [67] Junjue Wang, Zhuo Zheng, Ailong Ma, Xiaoyan Lu, and Yanfei Zhong. Loveda: A remote sensing land-cover dataset for domain adaptive semantic segmentation. In NeurIPS,

2021. 5

- [68] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 5
- [69] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In IROS, 2020. 5
- [70] Zehan Wang, Sashuai Zhou, Shaoxuan He, Haifeng Huang, Lihe Yang, Ziang Zhang, Xize Cheng, Shengpeng Ji, Tao Jin, Hengshuang Zhao, et al. Spatialclip: Learning 3d-aware image representations from spatially discriminative language. In CVPR, 2025. 2
- [71] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv:2509.20328, 2025. 2, 3
- [72] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In CVPR, 2018. 2, 3
- [73] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In CVPR, 2022. 3, 4
- [74] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. In ICLR, 2024. 2, 4, 7
- [75] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In NeurIPS, 2024. 2, 5, 4
- [76] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In ICCV, 2023. 5
- [77] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 2
- [78] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. In ECCV, 2024. 2
- [79] Chuhan Zhang, Guillaume Le Moing, Skanda Koppula, Ignacio Rocco, Liliane Momeni, Junyu Xie, Shuyang Sun,

- Rahul Sukthankar, Jo¨elle K. Barral, Raia Hadsell, Zoubin Ghahramani, Andrew Zisserman, Junlin Zhang, and Mehdi S. M. Sajjadi. Efficiently reconstructing dynamic scenes one d4rt at a time. arXiv:2512.08924, 2025. 1
- [80] Qi Zhang, Yifei Wang, and Yisen Wang. How mask matters: Towards theoretical understandings of masked autoencoders. In NeurIPS, 2022. 3
- [81] Yuchen Zhang, Nikhil Keetha, Chenwei Lyu, Bhuvan Jhamb, Yutian Chen, Yuheng Qiu, Jay Karhade, Shreyas Jha, Yaoyu Hu, Deva Ramanan, et al. Ufm: A simple path towards unified dense correspondence with flow. arXiv:2506.09278,

2025. 5

- [82] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017. 5
- [83] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. In ICLR, 2022. 3

## In Pursuit of Pixel Supervision for Visual Pre-training Supplementary Material

### 6. Failure Attempts, Limitations, and Future Directions

#### 6.1. Failure Attempts

In addition to the three aforementioned modifications to MAE, we explored several other approaches that ultimately did not yield performance improvements, including but not limited to:

- • Multi-block masking: We experimented with both inpainting (predicting center regions given surrounding context) [2] and outpainting variants (predicting surrounding context given center regions) [17]. Compared to our adopted masking strategy that is based on n × n local patches, these approaches introduce additional hyper-parameter complexity, requiring careful tuning of the number of blocks, block scale ranges, and block aspect ratios. Furthermore, they constrain the diversity of masking patterns available during training. For instance, the outpainting variant consistently provides a large contiguous region of visible context, which limits the model’s ability to learn long-range dependencies across spatially distant patches. Empirically, neither variant delivered performance gains over our simpler patch-block masking approach.
- • Hybrid masking ratios: MAE, Pixio, as well as many other masked image modeling works employ a fixed masking ratio across training. However, different images may benefit from different masking ratios depending on their complexity. For simple images with high redundancy, aggressive masking ratios are necessary to create a sufficiently challenging pretext task. Conversely, for complex images with rich, non-redundant content, excessively high masking ratios can make reconstruction unpredictable, causing the model to converge to trivial solutions rather than learning meaningful representations. To address this, several works [37, 22, 55] have proposed adaptive mechanisms that dynamically determine optimal masks based on motion cues or attention maps. However, these approaches introduce additional complexity and may exhibit bias toward specific image distributions (e.g., object-centric datasets). We explored a simpler alternative: hybrid masking ratios. For each training image, we randomly sample a masking ratio from a pre-defined set (e.g., [62.5%, 75%, 87.5%]), allowing both simple and complex images to be trained with more appropriate difficulty levels. While this design seems conceptually reasonable, we did not observe clear improvements.
- • Hybrid masking granularity: As shown in Table 14, we observe that different downstream tasks benefit from different masking granularity during pre-training. Mid-level, geometry-focused tasks (e.g., depth estimation) perform better with finer masking granularity (e.g., 2×2 patch blocks), while high-level, semantics-oriented tasks (e.g., semantic segmentation) favor coarser masking granularity (e.g., 4×4 patch blocks). This is expected, as easier, smaller masking units encourage the model to capture fine-grained spatial relationships, whereas harder, larger masking units promote learning of broader semantic context. Given these complementary benefits, a natural strategy is to employ hybrid masking granularity during training. We randomly vary the masking block size across batches to help the model adapt to different contextual scales and develop multi-level visual understanding. However, despite extensive attempts, we found that using a single, fixed masking granularity throughout training consistently yields the best performance.
- • Koleo loss on class tokens: DINOv2 [47] employs Koleo loss [52] to enforce uniformity in the feature distribution across samples, encouraging informative representations. However, this constitutes a strong manually-imposed inductive bias, as semantically similar samples may naturally have similar representations and should not be artificially repelled. We explored an alternative application: applying Koleo loss to our multiple class tokens rather than to individual samples. The motivation is to encourage each class token to capture distinct aspects of the image (e.g., semantics, scene layout, lighting, style), thereby promoting functional specialization among tokens. In preliminary experiments, this regularization yielded minor improvements on dense prediction tasks. However, it severely degraded ImageNet classification performance with these class tokens. More critically, we observed training instability even with very small loss weights (e.g., 0.1). Given these issues, we ultimately excluded this regularization from our framework.
- • Cross-attention in decoder: We observed that in both the original MAE and our Pixio, reconstructed masked regions exhibit higher visual quality than reconstructed visible regions. This occurs because the reconstruction loss is only computed on masked tokens, leading to optimization bias toward these tokens. This phenomenon also implies that the appended learnable [MASK] tokens and the encoder-extracted visible tokens reside in different feature spaces. Given this observation, we hypothesized that employing cross-attention between visible and masked tokens [25], rather than full self-attention,

- might better model their distinct representations while facilitating information transfer. Although this modification provided marginal computational speedup by reducing the attention complexity, it did not yield improvements in downstream performance. We therefore retained the standard self-attention mechanism in our final design.
- • Predicting both masked and visible patches. Following the aforementioned observation of misaligned feature spaces between mask tokens and visible tokens, we explored applying reconstruction loss to both masked and visible patches. In practice, this substantially degraded model performance across downstream tasks. This finding demonstrates that plain autoencoding on all image patches is suboptimal for learning transferable visual representations, and that the asymmetric reconstruction objective is crucial to MAE’s effectiveness.
- • Predicting partial masked patches: MAE reconstructs all masked patches at the decoder. However, masked patches themselves contain redundancy. Neighboring masked patches often have similar content. Also considering the increased computational cost of our deeper decoder, we attempted to reconstruct only a randomly sampled subset of masked patches [25], thereby reducing decoder overhead while maintaining the pretext task. However, while this provided marginal training speedup, it consistently degraded downstream performance.
- • Feeding multi-stage features to the decoder: MAE uses only the encoder’s final block output for decoding. Since pixel reconstruction requires both high-level semantic understanding and low-level textural details, relying solely on the final features may place excessive burden on the last encoder block, potentially compromising its ability to learn high-level abstractions. Ideally, different encoder stages should naturally capture different levels of visual information. Motivated by this, we extracted intermediate features from four encoder stages, concatenated them along the channel dimension, and fed this fused representation to the decoder. Our hypothesis was that combining early-stage and late-stage features would enable natural complementarity under pixel-level supervision. However, improvements were marginal and inconsistent across tasks. Therefore, we retained the simpler single-stage design in our final framework.

In summary, our three presented modifications (i.e., deeper decoder, larger masking blocks, and additional class tokens) represent minimal yet critical improvements to MAE. We highly value such simplicity in design. While some above explored alternatives may indeed be viable, we were unable to identify optimal configurations that consistently improved performance. We hope these empirical insights will inform future research in masked image modeling.

In addition to pre-training framework, we have also tried other data curation strategies, including but not limited to:

- • Online hard example mining: Rather than pre-computing image difficulty using a pre-trained MAE model on uncurated data, we explored selecting informative samples dynamically during training. Specifically, at each training iteration, we performed a forward pass on a batch of N candidate images and computed reconstruction loss for each. We then backpropagated only through the k images with the highest reconstruction loss, where k = α·N and α ∈ (0,1] controls the selection ratio. However, this approach proved problematic in practice. Early in training, when the model has not yet learned meaningful representations, the loss-based difficulty estimation is unreliable and noisy. This instability can lead to suboptimal convergence, as the training distribution shifts unpredictably. Therefore, we adopted the offline pre-computation strategy instead, which provides more stable difficulty estimates.
- • Canny edge density as a proxy for sample difficulty: In addition to reconstruction loss, we explored using Canny edge density [7], which is measured as the proportion of edge pixels detected in an image, as a heuristic proxy for image complexity. The intuition is that images with richer edge structures may contain more informative visual content. However, we found that such hand-crafted edge detectors are overly sensitive to low-level patterns and repetitive textures (e.g., grass, fabric patterns, brick walls), which produce high edge responses but offer limited semantic diversity.

#### 6.2. Limitations and Future Directions

As the name conveys, MAE’s core principles are masking and autoencoding. This work advocates for pixel supervision in visual pre-training. Pixel supervision shares philosophical similarities with autoencoding: both use models to compress and reconstruct input signals. We believe pixels provide the most comprehensive and unbiased supervision for pre-training, capturing rich visual information with minimal human intervention.

However, we recognize fundamental limitations in applying masking to static images. On the positive side, masking constructs a meaningful pretext task. High masking ratios encourage models to learn both high-level semantics and low-level details. Nevertheless, random masking remains an artificial distortion, which introduces undesirable biases. In practice, masking presents unavoidable trade-offs. Low masking ratios cause ground truth leakage, making reconstruction trivial. High masking ratios provide insufficient context for learning and create distribution shift between training and inference.

Critically, masked image modeling never exposes the model to natural, complete images during training. Despite these limitations, masking (or other artificial distortions) appears necessary for image-based pre-training.

The fundamental reason why so many artificial distortions and human inductive biases are necessary is, static images have inherent limitations as a medium for learning visual intelligence. Images are not the natural format in which visual information exists in the physical world. Humans do not learn from isolated snapshots. Instead, we learn through continuous temporal experiences in a causal manner, observing how the world evolves over time. From this perspective, video deserves greater emphasis, particularly long videos that capture the natural progression of events and their causal relationships. Videos offer a crucial advantage: the temporal dimension enables natural predictive objectives without artificial masking. Models can predict future frames from current observations—a task grounded in the causal structure of the physical world. This eliminates the need for artificial spatial masking [61] or noise injection [14].

This work serves as a pioneering validation that pixel supervision alone can produce strong visual representations competitive with more complex pre-training paradigms. Looking forward, we will scale this supervision approach to web-scale video data. By leveraging the temporal richness of videos and natural predictive objectives, we aim to develop more powerful and less biased visual foundation models for both videos and images.

### 7. Implementation Details

#### 7.1. Pre-training

The basic hyperparameters for our pre-training closely follow the original MAE framework, with several adaptations for large-scale training. Given our web-scale training data, we extend the training iterations from 500K to 1.3M and increase the batch size from 4,096 to 16,384. Importantly, we find that reducing the peak learning rate (2.4e-3 → 8e-4) is essential for stable convergence on less curated, more diverse web data. We also increase the input resolution from 224×224 to 256×256 with a patch size of 16×16. Comprehensive pre-training configurations are detailed in Table 9, and the complete architecture details of our largest Pixio-5B model are provided in Table 10.

config value data 2B web-crawled images iterations 1,284,000 batch size 16,384 input resolution 256×256 precision bfloat16 optimizer AdamW learning rate 8e-4 learning rate schedule cosine decay warmup steps 128,400 weight decay 0.05 optimizer momentum β1,β2 = 0.9, 0.95 data augmentation RandomResizedCrop crop scale 0.2-1.0 crop ratio 0.75-1.33 drop path 0.4 masking ratio 75% masking granularity 4×4 patches

Table 9. Pre-training details.

config value encoder

#params 5.4B patch size 16×16 #blocks 48 embedding dimension 3072 hidden dimension 12288 attention heads 32 positional embedding learnable #class tokens 8

decoder

#params 103M #blocks 32 embedding dimension 512 hidden dimension 2048 attention heads 16 positional embedding learnable

Table 10. Teacher model details.

#### 7.2. Distillation

Using our pre-trained Pixio-5B encoder as the teacher, we distill a series of smaller, more efficient student encoders: Pixio1B (1.4B parameters), Pixio-H (631M), Pixio-L (303M), and Pixio-B (86M). Specifically, the teacher encoder processes unmasked images while the student encoder receives either masked or the same unmasked inputs. For capable students (e.g., Pixio-1B, Pixio-H), we use masked inputs. This encourages the student to learn robust representations despite partial information. We align student features to teacher features through a lightweight MLP projection head, optimizing cosine

similarity loss at both patch-token and class-token levels. The two losses are equally weighted and averaged for optimization. We use 50% masking ratio with 4×4 patch masking granularity. All students are trained for 500K iterations with batch size 8,192 and learning rate 1e-3. We use drop path [35] 0.4 for students Pixio-1B and Pixio-H, while using drop path 0.1 for less capable students Pixio-L and Pixio-B. All other hyper-parameters remain identical to the pre-training stage.

- 7.3. Downstream Evaluation We have open-sourced the downstream evaluation code to facilitate reproducibility. We highlight some details below.

ImageNet-1K classification. For k-NN protocol, we follow DINO’s [9] official implementation. We report (k-NN) accuracy with k = 10. For fine-tuning protocol, we follow MAE’s official implementation. In both cases, images are resized to 256 pixels on the shorter side and then center-cropped to 256 × 256 for inference. For Pixio, we average all the class tokens to obtain the global representation.

Monocular depth estimation and semantic segmentation. We evaluate under two settings: a trainable DPT head [50] or a linear regression/classification head, with the encoder frozen in both cases. Following DINOv2 [47], we find that for certain encoders (e.g., DINOv2, MAE, Pixio), concatenating patch tokens with (averaged) class tokens along the channel dimension yields better performance than using patch tokens alone. We therefore report results using the optimal configuration for each encoder. For the DPT head, we extract intermediate features evenly from four encoder stages. All models are trained for 60 epochs. To ensure fair comparison across architectures, we use training resolution 256×256 for encoders with patch size 16 and 224×224 for those with patch size 14, maintaining consistent effective sequence length. During inference, we apply a sliding window approach with overlap and ensemble the predictions from overlapping regions.

For Depth Anything [75], MapAnything [38], and CortexBench [43], we follow the official implementations, replacing the encoder with our pre-trained models while keeping all other components unchanged.

### 8. Ablation Studies

Limited by space in the main paper, we primarily presented ablation results through figures. Here, we provide detailed numerical results and include additional ablation configurations for completeness.

#### 8.1. Block-Wise Performance of MAE and Pixio

Table 11 shows the block-wise feature quality of the official MAE models. Notably, the best generic features reside far before the final encoder block. For instance, on ADE20K semantic segmentation with MAE-H, there is a substantial 3.3 mIoU performance gap between the last encoder block and the optimal intermediate block, indicating that the final layers sacrifice representation quality for reconstruction. However, with our deeper decoder design, this issue is largely resolved. As shown in Table 12, Pixio’s final encoder block produces competitive features, with only a negligible 0.006 RMSE gap on NYUv2 compared to the best intermediate block. This validates our hypothesis that insufficient decoder capacity forces MAE’s late encoder blocks to assume decoding responsibilities.

ViT Block Index IN-1K KNN ↑ NYUv2 RMSE ↓ KITTI RMSE ↓ ADE20K mIoU ↑ Pascal mIoU ↑

32 55.0 0.593 4.411 35.1 70.7 28 60.1 0.583 4.247 36.3 71.9 24 62.1 0.574 4.298 37.4 73.6 20 61.9 0.564 4.218 38.4 74.4 16 51.1 0.612 4.465 37.3 73.6

H/14

24 57.7 0.585 4.607 34.2 70.7 21 61.1 0.585 4.451 35.4 72.1 18 60.0 0.582 4.285 36.2 73.0 15 40.7 0.622 4.433 34.3 70.8 12 28.4 0.711 4.864 28.4 62.2

L/16

- Table 11. Probing officially released MAE-H/14 (1280×32) and MAE-L/16 (1024 ×24) encoders [32], which are trained on ImageNet-1K. Decoder: 512×8. We use a linear head for both monocular depth estimation (regression) and semantic segmentation (classification).

Block Index IN-1K KNN ↑ NYUv2 RMSE ↓ KITTI RMSE ↓ ADE20K mIoU ↑ Pascal mIoU ↑

48 68.4 0.360 3.603 50.2 82.0 42 69.7 0.354 3.583 50.3 82.4 36 70.4 0.361 3.570 50.7 81.9 30 70.8 0.370 3.579 50.3 82.0 24 70.2 0.390 3.575 49.8 81.6

- Table 12. Probing our Pixio-5.4B encoder (3072×48), which is trained from scratch on our curated 2B images. Decoder: 512×32. We use a linear head for both monocular depth estimation (regression) and semantic segmentation (classification).

8.2. Decoder Design

Beyond the decoder widths (768, 512, 384 dimensions) presented in the main paper, Table 13 additionally reports results with an even shallower decoder (256 dimensions). The results confirm that excessively shallow decoders are suboptimal, as they lack sufficient capacity for the challenging pixel reconstruction objective.

Decoder

IN-1K KNN ↑ NYUv2 RMSE ↓ KITTI RMSE ↓ ADE20K mIoU ↑ Pascal mIoU ↑ Width Depth

768

8 44.2 0.480 3.156 35.4 71.9 16 58.3 0.408 2.828 41.3 77.1 32 49.0 0.458 3.007 36.7 75.1 48 32.0 0.574 3.418 26.7 62.8

512

8 (MAE) 35.3 0.431 2.986 35.8 71.6 16 55.1 0.409 2.789 39.5 76.1 32 55.8 0.410 2.749 40.4 76.9 48 57.6 0.422 2.832 40.5 77.1

384

8 35.2 0.469 3.047 32.1 68.3 16 48.6 0.425 2.825 36.6 73.3 32 56.2 0.410 2.821 39.7 75.2 48 55.6 0.412 2.940 39.8 76.8

256

8 32.6 0.499 2.995 29.1 64.8 16 38.6 0.473 3.001 32.4 68.5 32 47.2 0.451 2.923 35.7 71.2 48 43.7 0.437 2.898 37.4 74.4

- Table 13. Ablation study on the decoder on a ViT-H encoder (1280-dim×32-blocks). Here we mask at a single patch and use 1 class token.

#### 8.3. Masking Design

Extending the analysis in the main paper, Table 14 provides a comprehensive evaluation of different masking configurations (varying both masking ratio and granularity) under the 384×32 decoder setting. These results further validate the importance of larger masking granularity for learning better representations.

#### 8.4. Number of Class Tokens

In addition to the comparisons in the main paper, Table 15 ablates whether class tokens should be included in the decoder input. We observe that feeding class tokens to the decoder yields slightly better performance, suggesting that allowing them to participate in reconstruction helps learn more informative global representations.

Masking

Decoder

IN-1K KNN ↑ NYUv2 RMSE ↓ KITTI RMSE ↓ ADE20K mIoU ↑ Pascal mIoU ↑ Ratio Granularity

1×1 (MAE) 35.3 0.431 2.986 35.8 71.6 2×2 54.3 0.362 2.653 41.8 77.1 4×4 43.3 0.373 2.895 42.7 78.3

75%

512×8

- 1×1 32.6 0.468 2.944 31.2 66.6
- 2×2 49.5 0.378 2.715 38.6 74.2 4×4 53.2 0.356 2.654 41.8 78.1

62.5%

- 1×1 55.8 0.410 2.749 40.4 76.9
- 2×2 63.3 0.358 2.782 44.5 80.2 4×4 63.5 0.387 2.932 43.5 79.9

75%

512×32

- 1×1 52.2 0.444 2.896 37.6 75.2
- 2×2 62.8 0.360 2.650 43.7 79.2 4×4 52.8 0.360 2.741 44.3 79.5

62.5%

- 1×1 56.2 0.410 2.821 39.7 75.2
- 2×2 61.1 0.351 2.697 43.7 79.0 4×4 61.8 0.366 2.909 44.4 80.2

75%

384×32

- 1×1 46.6 0.450 2.907 35.3 72.1
- 2×2 57.0 0.359 2.675 42.4 78.2 4×4 57.9 0.357 2.725 44.5 79.4

62.5%

Table 14. Ablation study on masking ratio and masking granularity (measured in #patches). Here we use 1 class token.

Decoder #[CLS] In Decoder IN-1K KNN ↑ NYUv2 RMSE ↓ KITTI RMSE ↓ ADE20K mIoU ↑ Pascal mIoU ↑

1 (MAE) ✓ 63.3 0.358 2.782 44.5 80.2 4 ✓ 75.1 0.360 2.746 44.8 80.7 8 ✓ 75.0 0.361 2.654 44.8 80.5

16 ✓ 74.0 0.360 2.775 45.0 80.7

512×32

1 ✗ 64.1 0.373 2.787 44.3 80.1 4 ✗ 68.9 0.364 2.663 44.8 80.2 8 ✗ 70.6 0.376 2.794 44.2 80.0

16 ✗ 71.9 0.373 2.728 44.2 80.4

1 (MAE) ✓ 61.1 0.351 2.697 43.7 79.0 4 ✓ 68.9 0.350 2.683 43.9 80.2 8 ✓ 70.6 0.346 2.687 44.5 80.2

16 ✓ 71.0 0.352 2.736 44.4 80.0

384×32

1 ✗ 62.6 0.362 2.688 43.9 79.3 4 ✗ 66.0 0.369 2.784 43.3 79.7 8 ✗ 68.7 0.361 2.762 44.1 79.7

16 ✗ 70.6 0.356 2.715 44.3 80.0

Table 15. Ablation study on the number of class tokens and whether to include them in the decoder. Here we mask at 2×2 patch blocks.

