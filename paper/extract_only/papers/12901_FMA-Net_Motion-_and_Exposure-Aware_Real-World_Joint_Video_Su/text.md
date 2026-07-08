# arXiv:2512.04390v2[cs.CV]30Jun2026

## FMA-Net++: Motion- and Exposure-Aware Joint Video Super-Resolution and Deblurring

##### Geunhyuk Youk1 , Jihyong Oh2† , and Munchurl Kim1†

1 KAIST, Republic of Korea {rmsgurkjg, mkimee}@kaist.ac.kr 2 CMLab, Chung-Ang University, Republic of Korea jihyongoh@cau.ac.kr

##### https://kaist-viclab.github.io/fmanetpp_site/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

FMA-Net++ (Ours)

# Params

·

30.0

· · · · ·

Restormer* (6.05 / 40.84)

RVRT* (7.09 / 41.93)

BSSTNet* (6.69 / 49.15)

FMA-Net* (6.38 / 52.23)

FMA-Net* (CVPR 2024)

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

5M 10M 20M 30M 40M

29.0

PSNR(dB)

·

BSSTNet* (CVPR 2024)

·

IART* (CVPR 2024)

Restormer* (CVPR 2022)

DBVSR* (7.01 / 38.97)

BasicVSR++* (6.88 / 47.34)

IART* (6.79 / 45.18)

Blurry LR input (NIQE↓ / MUSIQ↑)

FMA-Net++ (5.83 / 58.89)

28.0

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

·

·

·

·

RVRT* (NeurIPS 2022)

·

VRT* (TIP 2024)

27.0

Ev-DeblurVSR* (AAAI 2025)

Restormer* (6.21 / 29.87)

RVRT* (7.31 / 32.69)

BSSTNet* (6.26 / 43.99)

FMA-Net* (6.72 / 49.09)

|[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

DBVSR* (ICCV 2021)

26.0

·

0 103

102

DBVSR* (7.17 / 25.85)

BasicVSR++* (7.26 / 36.74)

IART* (6.76 / 46.41)

Blurry LR input (NIQE↓ / MUSIQ↑)

FMA-Net++ (5.80 / 56.39)

Runtime (ms)

(a) Qualitative comparison on challenging real-world videos.

(b) Quantitative comparison on the GoPro [34] dataset.

Fig. 1: FMA-Net++ outperforms state-of-the-art methods in real-world qualitative results and quantitative benchmarks for joint video super-resolution and deblurring.

Abstract. Joint video super-resolution and deblurring (VSRDB) requires both efficient long-range temporal modeling and robustness to frame-wise exposure-duration variation, which changes the extent of motion blur across video frames. We propose FMA-Net++, a non-recurrent, sequence-level framework built from Hierarchical Refinement with Bidirectional Aggregation (HRBA) blocks. By stacking HRBA blocks, FMANet++ processes video frames in parallel while hierarchically expanding the temporal receptive field, avoiding the limited temporal receptive field of sliding-window designs and the sequential bottleneck of recurrent ones. To handle exposure-duration-dependent blur, we introduce an Exposure Time-aware Modulation (ETM) layer that conditions HRBA features on exposure embeddings from an Exposure Time-aware Feature Extractor (ETE). The conditioned features guide an exposure-aware flow-guided dynamic filtering module to predict motion- and exposureaware degradation kernels. FMA-Net++ decouples degradation learning from restoration: the former predicts degradation priors and the latter exploits them for efficient high-resolution restoration. To evaluate VSRDB under controlled exposure-duration variation, we introduce the

† Co-corresponding authors.

REDS-ME (multi-exposure) and REDS-RE (random-exposure) benchmarks. Trained solely on synthetic data, FMA-Net++ achieves stateof-the-art accuracy and temporal consistency on these benchmarks. It further shows strong out-of-distribution performance on GoPro and challenging real-world videos, while outperforming recent methods in both restoration quality and inference speed.

Keywords: Joint Video Super-Resolution Deblurring · Temporal Modeling · Dynamic Exposure

### 1 Introduction

Joint video super-resolution and deblurring (VSRDB) [11,17,49] aims to restore sharp high-resolution (HR) videos from blurry low-resolution (LR) inputs. In practice, blurry LR videos are common, and treating SR or deblurring separately is inadequate: SR cannot remove motion blur, while deblurring cannot recover high-frequency details, motivating a joint VSRDB approach [35,49]. The physical degradation process underlying these blurry LR videos is driven by two deeply intertwined factors: the motion field determines the spatial patterns of blur, and the exposure time controls its temporal extent and intensity [33,34,46]. Compounding this, camera auto-exposure mechanisms vary the exposure dynamically across frames [20,46]. This continuous fluctuation causes the severity of motion blur to change drastically within a single video sequence, resulting in complex, spatio-temporally variant degradations that standard restoration methods struggle to model.

While significant progress has been made in various video restoration tasks [7, 27, 36, 49], most existing methods assume a fixed exposure time. This assumption severely limits their robustness, as they struggle to handle the dynamically changing blur severity arising from continuous exposure variations. For instance, VSR [5,7,16,24,29] and video deblurring [23,51–53,55] approaches may produce artifacts or temporally inconsistent results when faced with exposure shifts. Even methods designed for unknown degradations, such as Blind VSR [3,22,36], typically assume spatially-invariant kernels and fail to account for the coupled effect of motion and varying exposure. Furthermore, recent joint VSRDB approaches like FMA-Net [49], despite handling motion-dependent degradation, remain constrained by this fixed-exposure assumption. Thus, VSRDB methods that explicitly address frame-wise exposure-duration variation are critically needed.

Moreover, beyond the exposure issue, prevailing temporal modeling strategies face inherent limitations: Sliding-window architectures [16, 24, 40, 41] tend to suffer from limited temporal receptive fields, while recurrent architectures [5, 7,12,28,29] lack parallelizability due to sequential bottlenecks, as conceptually compared in Fig. 2(a). Although recent transformer-based models like VRT [25] enable parallel processing, they incur heavy computational and memory burdens. To overcome these limits and address the aforementioned exposure variation, we introduce FMA-Net++, an efficient sequence-level framework that couples longrange temporal modeling with exposure-aware degradation estimation.

The core architectural unit of FMA-Net++ is the Hierarchical Refinement with Bidirectional Aggregation (HRBA) block. Instead of relying on restrictive sliding windows, such as those in FMA-Net [49], or inherently sequential recurrent structures [5,7,29], stacking HRBA blocks enables sequencelevel parallelization and hierarchically expands the temporal receptive field to capture long-range dependencies. To handle exposure-duration-dependent blur, each HRBA block includes an Exposure Time-aware Modulation (ETM) layer that conditions features on per-frame exposure embeddings, producing representations rich in both temporal context and exposure information. Leveraging these representations, an exposure-aware Flow-Guided Dynamic Filtering (FGDF) module estimates motion- and exposure-aware degradation kernels. Architecturally, we decouple degradation learning from restoration: the former predicts these rich priors, and the latter utilizes them to restore sharp HR frames efficiently, as illustrated in Fig. 2(b).

To systematically evaluate VSRDB under controlled exposure-duration variation, we construct two new benchmarks, REDS-ME (multi-exposure) and REDSRE (random-exposure). Trained solely on synthetic data, FMA-Net++ achieves state-of-the-art accuracy and temporal consistency on our new benchmarks and the GoPro [34] dataset. It outperforms recent methods in both restoration quality and inference speed, and we further validate robustness on real-world videos using qualitative results and no-reference metrics (see Fig. 1).

The main contributions of this work are as follows:

- – We design an efficient, sequence-level architecture based on Hierarchical Refinement with Bidirectional Aggregation (HRBA) blocks. By hierarchically expanding temporal receptive fields without sequential bottlenecks, HRBA effectively captures long-range temporal dependencies while enabling sequence-level parallel processing.
- – We formulate the physical coupling of motion blur and frame-wise exposureduration variation in VSRDB. To address this, we propose an Exposure Time-aware Modulation (ETM) layer that conditions features on perframe exposure embeddings, driving the estimation of motion- and exposureaware degradation kernels.
- – We introduce two new benchmarks, REDS-ME and REDS-RE, for systematic evaluation under dynamic exposure. Extensive experiments demonstrate that FMA-Net++ achieves state-of-the-art performance, high computational efficiency, and strong generalization to challenging real-world videos.

### 2 Related Work

#### 2.1 Joint Video Super-Resolution and Deblurring

VSRDB tackles the challenging task of jointly restoring sharp HR videos from blurry LR inputs, where blur degradations can be shaped by the coupled effects of motion and exposure. While single-task approaches for VSR [5,7,16,24,27,42] or video deblurring [23,27,52,53,56] have advanced, applying them sequentially

Net𝐷𝐷

Sliding window-based Recurrent-based

Hierarchical-based

Stacked HRBA

Sharp HR (GT)

|| |
|---|
<br><br>| | |
|---|---|
| | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Sliding window|
|---|

|𝒦𝒦𝐷𝐷|
|---|

⊛𝑠𝑠

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

⋯ ⋯

⋯ ⋯

⋯ ⋯

Reconstruction

[Figure 21]

[Figure 22]

Blurry LR

ETE

𝒖𝒖 Exposure

guidance Degradation guidance

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

⋯ ⋯

⋯ ⋯

Restored Sharp HR

Stacked HRBA Exposure-aware

 Long-range temporal context

✘ Limited temporal

⊛𝑠𝑠

✘ Lacks parallelizability

FGDF

 Sequence-level parallelization

✘ receptive field

Net𝑅𝑅

(a) Conceptual comparison of temporal modeling strategies

(b) Overview of FMA-Net++

Fig. 2: Conceptual illustration and overview of the FMA-Net++ framework.

often amplifies artifacts [35, 49]. However, specific methods tackling this joint VSRDB challenge remain scarce. Early approaches like HOFFR [11] struggle with spatially variant blur due to standard CNN limitations. Although FMANet [49] handles motion-dependent degradation via Flow-Guided Dynamic Filtering, it is constrained by a sliding-window design and a fixed-exposure assumption. FMA-Net++ departs from FMA-Net in two key aspects: it replaces sliding-window restoration with a sequence-level HRBA backbone, and it generalizes degradation estimation to jointly model motion and exposure. Recently, Ev-DeblurVSR [17] utilized auxiliary event streams for VSRDB, but it requires non-standard event data and still assumes a fixed exposure time (a limitation explicitly discussed in [17]), and therefore does not address RGB-only VSRDB under frame-wise exposure variation. These gaps motivate our sequence-level, exposure-aware approach for robust VSRDB using only standard RGB inputs.

#### 2.2 Temporal Modeling in Video Restoration

Effectively modeling long-range temporal dependencies is crucial for video restoration tasks like VSR. However, prevailing strategies face inherent architectural trade-offs. Sliding-window approaches [16, 24, 38, 42, 49] operate on fixed local neighborhoods, constraining input flexibility and limiting the capture of longrange context. Conversely, recurrent methods [5,7,27,29] propagate information sequentially. While they enable longer temporal aggregation, they are inherently constrained by sequential processing, which restricts parallelization and makes them prone to vanishing gradients over long sequences [9,29]. Recently, transformer-based models [4,25] have been explored. Although VRT [25] achieves sequence-level parallelization via a shifted-window mechanism [30], its heavy architecture incurs massive computational and memory costs. Furthermore, most of these works target sharp inputs, lacking robustness to complex degradations. This landscape motivates the need for sequence-level backbones that hierarchically expand temporal receptive fields while enabling efficient parallel processing.

#### 2.3 Exposure Time-Aware Restoration

In modern camera systems, auto-exposure mechanisms dynamically vary exposure across frames, yielding spatio-temporally variant blur that fixed-exposure models cannot faithfully capture [20,37,46]. While recent efforts in related tasks (e.g., video deblurring [20,37] and frame interpolation [37,46,54]) estimate exposure or exploit auxiliary sensing (events) to guide restoration, they do not

explicitly model the joint effects of motion and exposure within the VSRDB setting, and event-dependent designs limit practicality for standard RGB videos. To address these limitations, we propose an Exposure Time-aware Modulation (ETM) layer that injects per-frame exposure embeddings into temporal features. These conditioned features then drive our exposure-aware Flow-Guided Dynamic Filtering (FGDF), estimating motion- and exposure-aware degradation kernels. This design achieves robust VSRDB using only standard RGB inputs, integrating seamlessly with our sequence-level backbone.

### 3 Method

#### 3.1 Problem Formulation

We address joint VSRDB under frame-wise varying exposure, where the perframe exposure time ∆te,i is unknown at test time. Given a blurry LR video X = {Xi}Ti=1 ∈ RT×H×W×3, our goal is to restore the sharp HR video Yˆ = {Yˆi}Ti=1 ∈ RT×sH×sW×3 with an upscaling factor s.

To motivate an exposure-aware degradation model, we generalize the conventional fixed-exposure blur model [34]. The blurry LR frame Xi at a spatial position p is physically formed by spatially downsampling and temporally integrating the continuous latent sharp signal S over the exposure interval ∆te,i under the continuous motion field M:

i·∆t+∆te,i

1 ∆te,i

S (q + M(q,τ),τ)dτ , (1)

Xi(p) = Ds

i·∆t

where Ds is the spatial downsampling operator, ∆t is the frame interval, and q is the HR coordinate corresponding to p. Since directly inverting this continuous physical process is intractable, we approximate it with a discrete, learnable formulation using a spatio-temporally variant degradation kernel Ki:

Xi ≈ Ki ∗s Yi′, (2)

where ∗s denotes a filtering operation with stride s, and Yi′ = {Yi−k,...,Yi+k} is a short temporal neighborhood of sharp HR frames for a small k. Conceptually,

the ideal degradation kernel Ki at p depends jointly on the exposure time and the motion field through a complex mapping F:

Ki(p) = F (∆te,i,{M(q,τ) | q ∈ Ω(p);τ ∈ [i · ∆t,i · ∆t + ∆te,i]}), (3)

where Ω(p) denotes the spatial neighborhood of HR coordinates corresponding to p. This formulation captures how dynamic exposure and motion couple to create complex, spatio-temporally variant blur.

Our framework solves this inverse problem of restoring the sharp HR sequence by architecturally decoupling degradation learning from restoration. First, it uses an Exposure Time-aware Feature Extractor and learned optical flow to approximate the properties of ∆te,i and M, thereby estimating a learnable approximation of Ki. Second, these estimated priors guide the restoration network to reconstruct Yˆ .

Exposure Time-Aware Feature Representation 𝒖𝒖𝑖𝑖

W Occlusion-AwareBackward Warping

Flow-Guided Dynamic Downsampling

↑ Bilinear Upsampling ∗↓ C Channel-wise Concatenation

Predicted Degradation Kernels 𝒦𝒦𝑖𝑖𝐷𝐷

Bidirectional Aggregation

Element-wise Addition

𝑿𝑿𝑖𝑖−1

𝑿𝑿𝑖𝑖+1

𝑿𝑿𝑖𝑖−1

𝑿𝑿𝑖𝑖+1

𝑿𝑿𝑖𝑖

𝑿𝑿𝑖𝑖

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

𝑭𝑭𝑖𝑖−1𝐷𝐷,𝑀𝑀 C 𝑭𝑭𝑖𝑖𝐷𝐷,𝑀𝑀 C 𝑭𝑭𝑖𝑖+1𝐷𝐷,𝑀𝑀 C

RDB

RDB

RDB

𝑭𝑭𝑖𝑖−1𝐷𝐷,0

- 𝑭𝑭𝑖𝑖𝐷𝐷,0
- 𝑭𝑭𝑖𝑖𝐷𝐷,1

- 𝑭𝑭𝑖𝑖+1𝐷𝐷,0
- 𝑭𝑭𝑖𝑖+1𝐷𝐷,1

RDB

RDB

RDB

𝑭𝑭𝑖𝑖−1𝑅𝑅,0

- 𝑭𝑭𝑖𝑖𝑅𝑅,0
- 𝑭𝑭𝑖𝑖𝑅𝑅,1

- 𝑭𝑭𝑖𝑖+1𝑅𝑅,0
- 𝑭𝑭𝑖𝑖+1𝑅𝑅,1

HRBA𝐷𝐷,1

HRBA𝐷𝐷,1

HRBA𝐷𝐷,1

𝑭𝑭𝑖𝑖−1𝐷𝐷,1

HRBA𝑅𝑅,1

HRBA𝑅𝑅,1

HRBA𝑅𝑅,1

⋮

⋮

⋮

𝑭𝑭𝑖𝑖−1𝑅𝑅,1

𝑭𝑭𝑖𝑖−1𝐷𝐷,𝑀𝑀−1

𝑭𝑭𝑖𝑖𝐷𝐷,𝑀𝑀−1

𝑭𝑭𝑖𝑖+1𝐷𝐷,𝑀𝑀−1

⋮

⋮

⋮

𝑭𝑭𝑖𝑖−1𝑅𝑅,𝑀𝑀−1

𝑭𝑭𝑖𝑖𝑅𝑅,𝑀𝑀−1

𝑭𝑭𝑖𝑖+1𝑅𝑅,𝑀𝑀−1

HRBA𝐷𝐷,𝑀𝑀

HRBA𝐷𝐷,𝑀𝑀

HRBA𝐷𝐷,𝑀𝑀

𝐟𝐟𝑖𝑖−1𝐷𝐷,𝑀𝑀 𝐟𝐟𝑖𝑖+1𝐷𝐷,𝑀𝑀

𝐟𝐟𝑖𝑖𝐷𝐷,𝑀𝑀

HRBA𝑅𝑅,𝑀𝑀

HRBA𝑅𝑅,𝑀𝑀

HRBA𝑅𝑅,𝑀𝑀

𝐹𝐹𝑖𝑖+1𝐷𝐷,𝑀𝑀 Conv2d Conv2d Conv2d

𝐹𝐹𝑖𝑖−1𝐷𝐷,𝑀𝑀

𝐹𝐹𝑖𝑖𝐷𝐷,𝑀𝑀

|Conv2d| |
|---|---|
| | |

𝑭𝑭𝑖𝑖+1𝑅𝑅,𝑀𝑀 Upsample Upsample Upsample

𝑭𝑭𝑖𝑖−1𝑅𝑅,𝑀𝑀

𝑭𝑭𝑖𝑖𝑅𝑅,𝑀𝑀

Conv2d

Conv2d

Conv2d

↑ ↑ ↑

𝐟𝐟𝑖𝑖−1𝒀𝒀 𝐟𝐟𝑖𝑖𝒀𝒀 𝐟𝐟𝑖𝑖+1𝒀𝒀

|𝑖𝑖−1<br><br>𝐷𝐷|
|---|

|𝒦𝒦𝑖𝑖𝐷𝐷|
|---|

|𝑖𝑖+1<br><br>𝐷𝐷|
|---|

𝒦𝒦

𝒦𝒦

↑ ↑

↑

[Figure 29]

[Figure 30]

[Figure 31]

- W ∗↓ ∗↓ ∗↓

###### W W

𝒀𝒀𝑖𝑖−1 𝒀𝒀𝑖𝑖 𝒀𝒀𝑖𝑖+1

{𝒀𝒀𝑡𝑡}𝑡𝑡=𝑖𝑖−2𝑖𝑖 {𝒀𝒀𝑡𝑡}𝑡𝑡=𝑖𝑖𝑖𝑖+2

𝑿𝑿𝑖𝑖−1 {𝒀𝒀𝑡𝑡}𝑡𝑡=𝑖𝑖−1𝑖𝑖+1 𝑿𝑿𝑖𝑖+1

𝑿𝑿𝑖𝑖

(a) Degradation Learning Network (Net𝐷𝐷) (b) Restoration Network (Net𝑅𝑅)

Fig. 3: Architecture of FMA-Net++ for joint VSRDB.

#### 3.2 Overall Architecture of FMA-Net++

- Fig. 3 illustrates the overall architecture of FMA-Net++, which consists of two main networks: a Degradation Learning Network (NetD) and a Restoration Network (NetR). Both networks are built upon stacks of Hierarchical Refinement with Bidirectional Aggregation (HRBA) blocks. This sequence-level backbone processes all frames in parallel, effectively avoiding sequential bottlenecks while hierarchically expanding the temporal receptive field.

Following the degradation formulation in Eq. 2, our architecture decouples degradation learning from restoration. Given an input blurry LR sequence X and exposure embeddings from a pretrained Exposure Time-aware Feature Extractor (ETE), NetD first leverages HRBA and an Exposure Time-aware Modulation (ETM) layer to estimate degradation priors. These priors then drive the exposure-aware Flow-Guided Dynamic Filtering (FGDF) to model spatiotemporally variant degradations. Finally, NetR restores the sharp HR video Yˆ guided by these priors, improving both accuracy and efficiency.

#### 3.3 Hierarchical Refinement with Bidirectional Aggregation

As the core architectural unit shared by both NetD and NetR, the HRBA block overcomes the fundamental trade-offs faced by prior temporal modeling strategies (Sec. 2.2): namely, limited temporal receptive fields in slidingwindow methods [42,49] and the lack of parallelizability in sequential recurrent approaches [7,27]. By stacking HRBA blocks, our architecture enables sequencelevel parallel processing. At each refinement level, information from increasingly distant past and future frames is aggregated bidirectionally, thus hierarchically expanding the temporal receptive field to effectively capture long-range dependencies.

𝑭𝑭𝑖𝑖𝑗𝑗

| |Conv2d| |
|---|---|---|
| | | |

Conv2d

Only for Net𝑅𝑅

C W

𝐟𝐟𝑖𝑖𝑗𝑗 𝐟𝐟𝑖𝑖𝑗𝑗+1

FFN

𝑭𝑭𝑖𝑖±1𝑗𝑗 𝑭𝑭𝑖𝑖±1→𝑖𝑖𝑗𝑗

𝒌𝒌𝑖𝑖𝑗𝑗

Conv2d

DA-Attn

ETM

C

𝑭𝑭𝑖𝑖𝑗𝑗 𝑭𝑭𝑖𝑖𝑗𝑗+1

Multi-Attn

𝑭𝑭𝑖𝑖𝑗𝑗

𝑭𝑭𝑖𝑖𝑗𝑗 Conv2d

𝒌𝒌𝑖𝑖𝑗𝑗

FFN

𝒖𝒖𝑖𝑖

Only for Net𝑅𝑅

𝒦𝒦𝑖𝑖𝐷𝐷

Self-Attn

Exposure time-aware feature extractor (ETE)

𝒖𝒖

𝑿𝑿

𝑭𝑭𝑖𝑖𝑗𝑗

(a) HRBA Block (b) Multi-Attention

- Fig. 4: Details of an HRBA block. (a) Structure of the HRBA block at (j+1)-th refinement step for i-th frame (Sec. 3.3). (b) Structure of Multi-Attention. FFN refers to the feed-forward network of the transformer [1,10].

As shown in Fig. 4(a), each HRBA block iteratively refines the feature map Fij ∈ RH×W×C and a set of multi-flow-mask pairs fij ∈ R2×H×W×(2+1)n for a given frame i at refinement step j + 1. Specifically, fij is defined as:

fij ≡ (fik→i+1, oki→i+1),(fik→i−1, oki→i−1)

n k=1

(4)

where n is the number of multi-flow-mask pairs, each containing an optical flow f and corresponding occlusion mask o representing motion towards neighbors i±1. Keeping multiple motion hypotheses (n > 1) enhances robustness under severe blur by providing one-to-many correspondences [6,14]. The refinement process first computes intermediate features F˜ij via occlusion-aware warping [15,35] of neighboring features Fij±1 using fij, followed by fusion using concatenation and convolution. The multi-flow-mask pairs are then updated residually, fij+1 = fij + ∆fij, where the residual ∆fij is predicted based on F˜ij and fij. The intermediate feature F˜ij is further enhanced through two crucial modules before producing the final output Fij+1.

Multi-Attention. As shown in Fig. 4(b), the multi-attention module employs self-attention [1] to capture spatial dependencies and integrate the aggregated hierarchical temporal context. Within NetR, it subsequently applies DegradationAware (DA) attention. This cross-attention mechanism uses query Q derived from the estimated exposure- and motion-aware degradation kernel KiD (predicted by NetD), while key K and value V are projected from the self-attention output. This allows NetR features to adapt specifically to the estimated degradation characteristics of each frame.

Exposure Time-aware Modulation (ETM). To handle frame-wise exposure variation, every HRBA block applies ETM via a lightweight SFT layer [43]. The exposure embedding ui is provided by the ETE, a ResNet-18 [13] backbone pretrained via supervised contrastive learning [19] to separate exposure anchors in a latent space. Conditioned on ui, a shallow network M predicts affine parameters (α,β) = M(ui) and modulates the attention output Fˆij as Fij+1 = (1 + α) ⊙ Fˆij + β. This injects exposure information into all refinement stages with negligible overhead, enabling dynamic exposure adaptation.

In summary, compared to sliding windows [16, 24, 38, 42, 49], HRBA accesses long-range context via hierarchical aggregation. Compared to recurrent schemes [5,7,27,29], it avoids sequential dependencies, enabling efficient parallelization and stable training on long sequences.

#### 3.4 Degradation Learning and Restoration Networks

As outlined in Sec. 3.2, our framework comprises two main networks leveraging the HRBA backbone to solve the inverse problem defined in Sec. 3.1.

Degradation Learning Network (NetD). As shown in the left part of Fig. 3, NetD aims to estimate degradation priors from the input blurry LR sequence

- X. It processes X through a stack of HRBA blocks with integrated ETM layers, producing refined features FD,M and multi-flow-mask pairs fD,M. From these outputs, NetD predicts two key priors for each frame Xi: (i) image flow-mask pairs fiY = fYi→i±1, oYi→i±1 , representing motion between the sharp HR frame

2

Yi and its neighbors, and (ii) degradation kernels KiD ∈ R3×H×W×k

d.

To apply these predicted kernels, we utilize the Flow-Guided Dynamic Filtering (FGDF) [49] module. Crucially, because KiD is predicted from features already infused with exposure information via the ETM layer, this operation naturally becomes an exposure-aware FGDF. This kernel formulation, representing the degradation from three consecutive sharp HR frames {Yi−1,Yi,Yi+1} to the blurry LR frame Xi, follows the design principle of [49] as it offers a robust trade-off between performance and computational cost. The shape of KiD reflects its spatio-temporally variant and position-dependent nature, providing a structured learnable parameterization for modeling complex varying degradations.

To ensure accurate prior estimation, NetD is trained with a reconstruction objective: the predicted priors must reconstruct the blurry LR frame Xˆi from the ground-truth (GT) sharp HR frames Y as:

Xˆi = KiD ⊛s {Yt→i}it+1=i−1 , (5) where ⊛s denotes the FGDF operation [49] with stride s, defined as:

kd2

i+1

KiD ⊛s {Yt→i}it+1=i−1 (p) =

Ki,tD (p,pk) Yt→i(sp + pk), (6)

t=i−1

k=1

where pk denotes the k-th spatial offset within the kd × kd grid. Unlike conventional dynamic filtering that uses fixed spatial neighborhoods, FGDF explicitly performs filtering along motion trajectories by dynamically sampling features guided by the estimated optical flow. By utilizing our exposure-conditioned kernel KiD, this operation naturally achieves jointly motion- and exposure-aware filtering, making Eq. 5 a practical, learnable approximation of the conceptual degradation model in Eq. 2. The warped HR frame Yt→i is defined as:

Yi, if t = i W(Yt,fiY→t,oYi→t), if t = i ± 1

(7)

Yt→i =

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

5 ∶ 1 5 ∶ 2 5 ∶ 3 5 ∶ 4 5 ∶ 5

- Fig. 5: Example frames from our REDS-ME dataset across five exposure levels and two scenes. Each column corresponds to an exposure ratio from 5:1 (shortest exposure) to 5 : 5 (longest exposure). Longer exposures lead to increasingly severe motion blur, illustrating the controlled effect of exposure variation on blur extent.

where W denotes the occlusion-aware backward warping [47].

Restoration Network (NetR). As shown in the right part of Fig. 3, NetR performs the final restoration, taking the blurry LR sequence X along with the rich priors predicted by NetD (FD,M,fD,M, and KD) as input. It first generates initial features by combining X and the context feature FD,M using concatenation and an RDB [44]. These features are then refined through another stack of HRBA blocks, initializing the multi-flow-mask pairs with fD,M from NetD to leverage the motion prior. Crucially, within each HRBA block in NetR, the DA attention utilizes the estimated kernel KiD as its query, after which the ETM layer continues to provide exposure conditioning, enabling degradation- and exposure-adaptive restoration. Finally, the refined features FR,M pass through an upsampling block to predict a high-frequency residual Yˆires. The final sharp HR frame is obtained by adding this residual to the bilinearly upsampled blurry LR input:

Yˆi = Yˆires + Xi ↑s, (8) where ↑s denotes the ×s bilinear upsampling.

#### 3.5 Training Strategy

We adopt a three-stage training strategy to effectively optimize FMA-Net++. First, the ETE is pretrained using a supervised contrastive loss [19] on exposure labels derived from our data synthesis process (Sec. 4.1; see Suppl. for detailed contrastive loss formulations). Crucially, we freeze the ETE afterward to provide a stable exposure reference space, preventing representation drift during subsequent training. Second, guided by the frozen ETE, NetD is trained to predict reliable degradation priors, supervised by a composite loss LD:

T

T

l1(Xˆi,Xi) + λ1

l1(Yi±1→i,Yi) + λ2l1(fY ,fRAFTY ), (9)

LD =

i=1

i=1

where the first term is the reconstruction loss of the blurry LR input via Eq. 5, the second term is a warping loss, and the third term explicitly supervises the optical flow guided by a pretrained RAFT [39]. RAFT is used only for trainingtime pseudo-supervision and is not required during inference. Finally, the entire

framework is jointly trained end-to-end, fine-tuning NetD alongside NetR with the total loss:

Ltotal = l1(Yˆ ,Y ) + λ3LD, (10) where the first term is the primary restoration loss on the final sharp HR output.

### 4 Experiments

#### 4.1 Experimental Setup

Datasets and Benchmarks. To systematically evaluate VSRDB under controlled exposure-duration variation, we construct two new benchmarks derived from the REDS dataset [33]: REDS-ME (Multi-Exposure) and REDS-RE (Random-Exposure).

For REDS-ME, we synthesize five distinct exposure levels corresponding to duty cycles from 5:1 (shortest exposure) to 5:5 (longest exposure), as illustrated in Fig. 5. Crucially, these discrete levels serve as pseudo-labels for ETE pretraining. We train FMA-Net++ using all five levels from the REDS-ME training set and evaluate on the most challenging levels (5 : 4, 5 : 5) of the REDS4-ME test set (derived from the standard REDS4 partition). To explicitly assess robustness to dynamic exposure variations, we construct REDS-RE by temporally mixing frames across the five exposure levels within the REDS4-ME test scenes, yielding controlled exposure trajectories with varying blur extents. Furthermore, we use the standard GoPro dataset [34] to evaluate generalization to different scene and motion statistics, alongside qualitative assessments on real-world blurry videos. Details of the data synthesis pipeline are provided in the Suppl.

Evaluation Metrics. We evaluate restoration quality using PSNR and SSIM [45].

Temporal consistency is measured by tOF [2]. For real-world videos where GT is unavailable, we report no-reference metrics such as NIQE [32] and MUSIQ [18]. We also compare model efficiency in terms of parameter count and runtime.

Implementation Details. All implementation details, including network configurations and hyperparameter settings, are provided in the Suppl.

#### 4.2 Comparisons with State-of-the-Art Methods

We compare FMA-Net++ against SOTA methods across relevant categories: single-image SR (SwinIR [26], HAT [8]), single-image deblurring (Restormer [50], FFTformer [21]), VSR (BasicVSR++ [7], IART [48]), video deblurring (VRT [25], RVRT [27], BSSTNet [52]), Blind VSR (DBVSR [36]), and joint VSRDB (FMANet [49], Ev-DeblurVSR [17]). For a fair comparison in the joint VSRDB setting under varying exposure conditions, relevant SOTA methods were adapted and retrained on our REDS-ME training set, denoted by ∗ in Tables 1 and 2.

Quantitative Results. Table 1 presents the performance on REDS4-ME across two challenging exposure levels (5:4 and 5:5), representing severe motion blur. FMA-Net++ consistently outperforms all baselines across PSNR, SSIM, and tOF. For instance, FMA-Net++ achieves significant gains of 0.62 dB and 0.73

Table 1: Quantitative comparison of ×4 VSRDB on REDS4-ME for two challenging exposure levels (5:4 and 5:5). All metrics are computed on the RGB channels. Red and blue indicate the best and second-best performance, respectively. Runtime is measured per LR frame of resolution 180 × 320. The superscript ∗ denotes models retrained on our proposed REDS-ME training set.

|Methods<br><br>|# Params (M)|Runtime (s)<br><br>|REDS4-ME-5:4 PSNR SSIM tOF|REDS4-ME-5:5 PSNR SSIM tOF|
|---|---|---|---|---|
| | | |↑ ↑ ↓<br><br>|↑ ↑ ↓|

Super-Resolution + Deblurring SwinIR [26] + Restormer [50] 11.9 + 26.1 0.221 + 0.753 26.23 0.7464 3.775 25.53 0.7229 4.558

HAT [8] + FFTformer [21] 20.8 + 16.6 0.352 + 1.414 26.66 0.7634 3.207 25.92 0.7400 3.995 BasicVSR++ [7] + RVRT [27] 7.3 + 13.6 0.048 + 0.349 27.28 0.7901 2.887 26.98 0.7621 3.164

IART [48] + BSSTNet [52] 13.4 + 52.0 1.041 + 0.482 27.50 0.8006 2.578 27.26 0.7888 2.721

Deblurring + Super-Resolution Restormer [50] + SwinIR [26] 26.1 + 11.9 0.043 + 0.221 26.36 0.7499 3.464 25.84 0.7316 3.948

FFTformer [21] + HAT [8] 16.6 + 20.8 0.066 + 0.352 26.36 0.7534 3.256 25.87 0.7356 3.739 RVRT [27] + BasicVSR++ [7] 13.6 + 7.3 0.019 + 0.048 26.35 0.7492 3.314 25.95 0.7424 3.610

BSSTNet [52] + IART [48] 52.0 + 13.4 0.025 + 1.041 26.51 0.7711 3.103 26.33 0.7564 3.313 Blind Video Super-Resolution DBVSR [36] 14.1 0.096 24.50 0.7208 3.449 22.19 0.6122 4.554 Joint Video Super-Resolution and Deblurring

|Restormer∗ [50] DBVSR∗ [36] BasicVSR++∗ [7] IART∗ [48] VRT∗ [25] RVRT∗ [27] BSSTNet∗ [52] Ev-DeblurVSR [17] Ev-DeblurVSR∗ [17] FMA-Net [49] FMA-Net∗ [49]|26.5 14.1 7.3 13.4 35.6 12.9 52.0 8.3 8.3 9.6 9.6<br><br>|0.045 0.096 0.048 1.041 0.684 0.385 0.548 0.062 0.062 0.318 0.318<br><br>|27.45 0.7851 2.161<br><br>26.77 0.7629 3.021<br><br>27.70 0.7922 2.302<br><br>28.23 0.8153 2.143<br><br><br>27.93 0.8045 2.366<br><br>28.11 0.8093 2.136<br><br><br>28.75 0.8342 1.893<br><br>24.51 0.7154 3.602 27.40 0.7839 2.521 26.42 0.7958 2.503<br><br>29.04 0.8275 1.891<br><br><br>|27.12 0.7750 2.516<br><br>26.07 0.7405 3.765<br>27.14 0.7770 2.746 27.64 0.7972 2.590<br><br><br>27.41 0.7887 2.839<br><br>27.58 0.7944 2.558<br>28.11 0.8119 2.298 24.38 0.7047 4.094 26.82 0.7672 3.059 26.67 0.8005 2.443<br><br><br>28.51 0.8136 2.269<br>|
|---|---|---|---|---|
|FMA-Net++ (Ours)|12.8<br><br>|0.074|29.66 0.8546 1.688<br><br>|29.24 0.8453 1.956|

dB over the second-best model, FMA-Net∗, on levels 5 : 4 and 5 : 5, respectively. Furthermore, FMA-Net++ demonstrates superior efficiency compared to methods with similar complexity like RVRT∗ [27]. It achieves remarkably higher performance while being significantly faster (over 5.2× speedup), an efficiency that primarily arises from our parallelizable HRBA architecture. While VRT [25] also enables sequence-level parallelization, it suffers from massive memory consumption and a heavy computational burden, requiring 20.5 GB for a 10-frame sequence at 180×320 resolution. In contrast, our FMA-Net++ requires only 6.2 GB, demonstrating that HRBA achieves parallel temporal modeling far more efficiently while maintaining state-of-the-art accuracy.

Table 2 evaluates robustness to dynamic exposure (REDS-RE) and generalization ability to an unseen dataset (GoPro [34]). On REDS-RE, featuring dynamic exposure transitions within sequences, the performance advantage of FMA-Net++ over other methods widens considerably compared to REDS-ME. This result validates the effectiveness of our explicit exposure-aware modeling (ETM) in adapting to temporally varying exposure conditions where fixedexposure assumptions struggle. On the unseen GoPro dataset, which has different scene and motion statistics from REDS-ME, FMA-Net++ again achieves the best performance across all metrics, indicating strong generalization beyond the training domain.

Qualitative Results. Fig. 6 presents visual comparisons on synthetic benchmarks (REDS4-ME-5 : 5 and GoPro) that contain severe motion blur, while

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Restormer*

RVRT* FMA-Net*

BSSTNet*

GT

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

REDS4-ME-5:5 Blurry LR input BasicVSR++*

DBVSR*

IART* Ev-DeblurVSR*

FMA-Net++

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Restormer*

RVRT* FMA-Net*

BSSTNet*

GT

|[Figure 60]|
|---|

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

GoPro Blurry LR input BasicVSR++*

DBVSR*

IART* Ev-DeblurVSR*

FMA-Net++

- Fig. 6: Qualitative comparisons of ×4 VSRDB on REDS4-ME-5 : 5 and GoPro [34]. Each scene contains severe motion blur with different characteristics.

Fig. 1(a) shows results on challenging real-world videos captured with a smartphone. On both synthetic and real-world data, FMA-Net++ consistently restores sharper details, cleaner edges, and more legible text with fewer artifacts, achieving the best perceptual quality (NIQE/MUSIQ). We omit multi-modal methods such as Ev-DeblurVSR [17] from the real-world comparison, as they are fundamentally not applicable to standard RGB videos that lack the required event data. This highlights the strong practicality and generalization of our approach, which achieves these results using only conventional RGB inputs despite being trained solely on synthetic data. Further qualitative results are provided in the Suppl.

### 5 Ablation Study

We present ablation studies validating our key design choices. Further ablation studies and detailed analyses can be found in the Suppl.

#### 5.1 Effectiveness of Hierarchical Architecture

To validate the advantages of our proposed hierarchical temporal architecture (conceptually compared with other strategies in Fig. 2(a)), we compare the full FMA-Net++ against two variants built upon its core components but employing

- Table 2: Quantitative comparison of ×4 VSRDB on REDS-RE and GoPro [34].

|Methods<br><br>|REDS-RE|GoPro|
|---|---|---|
| |PSNR ↑ SSIM ↑ tOF ↓|PSNR ↑ SSIM ↑ tOF ↓|
|Restormer∗ [50] DBVSR∗ [36] BasicVSR++∗ [7] IART∗ [48] VRT∗ [25] RVRT∗ [27] BSSTNet∗ [52] Ev-DeblurVSR∗ [17] FMA-Net∗ [49]<br><br>|27.79 0.7953 1.775<br><br>27.30 0.7742 2.398<br><br>28.14 0.8044 1.904<br><br><br>28.68 0.8248 1.852<br><br><br>28.24 0.8124 2.071<br><br>28.56 0.8208 1.926<br><br>29.33 0.8427 1.602 27.94 0.7987 2.039<br><br><br>29.29 0.8413 1.614<br><br><br>|27.54 0.8350 3.302<br><br>26.05 0.7815 4.730<br>27.40 0.8282 3.285 27.76 0.8394 3.302 27.39 0.8304 3.616<br><br><br>27.64 0.8364 3.223<br>28.57 0.8650 2.753<br><br><br>27.25 0.8247 3.536<br>28.83 0.8655 2.727<br>|
|FMA-Net++ (Ours)|30.13 0.8643 1.360|30.49 0.9018 2.091|

different temporal modeling strategies: (i) a sliding-window variant processing three frames at a time, similar to [49], and (ii) a recurrent variant where the hierarchical refinement is adapted for sequential propagation. All variants maintain the same number of HRBA blocks and utilize the same ETM and multi-attention mechanisms, forming an identical backbone for a fair comparison.

- Table 3 presents the comparison results on REDS4-ME-5:5. Our hierarchical

FMA-Net++ demonstrates substantial improvements over both variants. Compared to the sliding-window variant, it yields markedly better results across all metrics, effectively overcoming the limitations imposed by a fixed temporal receptive field. Compared to the recurrent variant, it achieves superior performance in both PSNR and tOF. This noticeable gain in temporal consistency likely stems from its non-recurrent hierarchical structure, which mitigates gradient-vanishing issues that can affect sequential propagation over long sequences. We also empirically observe that this design achieves the most stable training dynamics among the compared variants. Furthermore, in terms of efficiency, the hierarchical design demonstrates a modest speed advantage over the recurrent approach. Overall, these results validate that our hierarchical strategy serves as a highly effective backbone for high-quality and temporally consistent video restoration.

Predicted 𝒦𝒦𝑖𝑖𝐷𝐷

Predicted 𝒦𝒦𝑖𝑖𝐷𝐷

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

w/ correct 5:1 𝑢𝑢𝑖𝑖 spread: 5.29

w/ incorrect 5:1 𝑢𝑢𝑖𝑖 spread: 12.28

[Figure 71]

[Figure 72]

REDS4-ME-5:1 input REDS4-ME-5:5 input

w/ incorrect 5:5 𝑢𝑢𝑖𝑖 spread: 18.17

w/ correct 5:5 𝑢𝑢𝑖𝑖 spread: 37.89

(a) t-SNE visualization of ETE features

(b) Effect of guidance on kernels

- Fig. 7: Visual analysis of exposure-aware modeling. (a) t-SNE [31] visualization of ETE features (u) shows clear clustering across exposure levels. (b) Bidirectional guidance test: for both a 5:5 and a 5:1 input, the predicted degradation kernel adapts its spatial spread to the supplied exposure guidance.

#### 5.2 Effectiveness of Exposure-Aware Modeling

Table 4 evaluates our exposure-aware modeling by comparing FMA-Net++ with variants without ETE, including a capacity-matched baseline (13.1M), slightly larger than the full model (12.8M). Increasing the capacity of the w/o-ETE model absorbs most of the in-distribution PSNR gain on REDS4-ME, indicating that part of the improvement under fixed exposure comes from backbone capacity. However, the matched baseline does not close the gap on the exposurevarying REDS-RE split, where FMA-Net++ w/ ETE still improves PSNR from 29.88 to 30.13 and reduces tOF from 1.399 to 1.360. This indicates that the ETE/ETM pipeline contributes to exposure-conditioned robustness beyond pa-

- Table 3: Comparison of temporal modeling variants within FMA-Net++ on REDS4ME-5:5. Our HRBA design achieves the best accuracy and temporal consistency.

|Temporal Modeling Strategy|Runtime (s)<br><br>|PSNR ↑<br><br>|tOF ↓|
|---|---|---|---|
|Sliding-window variant Recurrent propagation variant HRBA (Ours)<br><br>|0.314 0.086 0.074|28.57 29.11 29.24<br><br>|2.231 1.989 1.956|

- Table 4: Ablation study on the Exposure Time-aware Feature Extractor (ETE) for multiple datasets.

|Methods<br><br>|# Params (M)|Runtime (s)|In-distribution| |Out-of-distribution| |
|---|---|---|---|---|---|---|
| | | |REDS4-ME-5:4<br><br>|REDS4-ME-5:5|REDS-RE<br><br>|GoPro|
| | | |PSNR↑ tOF↓<br><br>|PSNR↑ tOF↓|PSNR↑ tOF↓<br><br>|PSNR↑ tOF↓|
|FMA-Net++ w/o ETE FMA-Net++ w/o ETE (matched) FMA-Net++ w/ ETE|9.8 13.1 12.8<br><br>|0.071 0.088 0.074|29.55 1.764 29.67 1.713 29.66 1.688<br><br>|29.12 2.054 29.20 2.011 29.24 1.956<br><br>|29.72 1.436<br><br>29.88 1.399<br>30.13 1.360<br>|29.78 2.267<br><br>29.85 2.251<br><br>30.49 2.091<br>|

rameter count. The consistent gain on the unseen GoPro dataset (+0.64 dB) further reflects improved generalization beyond the training domain.

Choice of ETE Design. To justify the contrastive ETE design, we compare it with three deployable exposure-conditioning alternatives that compute guidance from the input: frame-difference features, a classification-pretrained ETE, and an ordinal-regression ETE. As shown in Table 5, all alternatives underperform our contrastive ETE on REDS-RE, indicating that the gain does not come from merely adding an arbitrary exposure-related signal. The contrastive objective provides a more effective exposure embedding for the ETM/FGDF pipeline.

Visual Analysis of Learned Priors. To further validate this explicit conditioning, we visualize the learned representations in Fig. 7. The t-SNE [31] visualization of ETE features (u) in Fig. 7(a) shows clear clustering across exposure levels, confirming that ETE successfully extracts exposure-dependent characteristics. Furthermore, Fig. 7(b) demonstrates the impact of this guidance through a bidirectional test. For a severely blurred 5:5 input, correct 5:5 guidance yields a spatially diffuse kernel, whereas incorrect 5:1 guidance contracts it into a concentrated one. Conversely, for a mildly blurred 5 : 1 input, 5 : 5 guidance drives the kernel to become more diffuse. These bidirectional responses show that the predicted kernel KiD adapts its spatial shape to the supplied exposure guidance. Frame-wise Guidance Analysis. We further analyze the role of ETE guidance on REDS-RE by keeping the input sequence fixed and corrupting only the frame-wise guidance u, as shown in Table 6. Correct guidance performs best, while fixed, sequence-shuffled, and random wrong-level guidance degrade both PSNR and tOF, with larger drops on transition frames where the exposure level changes. Notably, random wrong-level guidance performs worse than removing guidance entirely, showing that u is actively used rather than ignored.

Fixed-Input Sensitivity. Complementary to the REDS-RE corruption test, Table 7 reports the effect of replacing u with exposure embeddings from different levels on fixed REDS4-ME-5:5 inputs. Performance decreases gradually as the guidance deviates from the correct level, yet remains stable even under the farthest 5 : 1 guidance. This shows that, when the input exposure is fixed and

- Table 5: Comparison of exposureconditioning designs on REDS-RE.

Table 6: Effect of corrupting frame-wise guidance u on REDS-RE.

|Conditioning design<br><br>|PSNR↑ / tOF↓|
|---|---|
|Frame-difference Classification-pretrained ETE Ordinal-regression ETE Contrastive ETE (Ours)<br><br>|29.75 / 1.429 29.92 / 1.398<br><br>29.97 / 1.382<br>30.13 / 1.360<br>|

|Guidance u|Average<br><br>|Transition Frames|
|---|---|---|
| |PSNR↑ / tOF↓<br><br>|PSNR↑ / tOF↓|
|Correct Same-frame fixed 5:5 Sequence-shuffled Random wrong-level w/o ETE (no guidance)<br><br>|30.13 / 1.360 29.88 / 1.387 29.59 / 1.525 29.47 / 1.605 29.72 / 1.436<br><br>|29.98 / 1.395 29.40 / 1.631 29.18 / 1.713 29.05 / 1.739 29.48 / 1.633|

Table 7: Sensitivity to ETE guidance on fixed REDS4-ME-5:5 inputs.

|Input Frame|Guidance u|PSNR ↑ / tOF ↓<br><br>|
|---|---|---|
|5:5<br><br>|from 5:5 (Correct) from 5:4 from 5:3 from 5:2 from 5:1|29.24 / 1.956 29.20 / 1.972 29.13 / 2.012 29.11 / 2.027 29.07 / 2.041<br><br>|
|Baseline w/o ETE<br><br>| |29.12 / 2.054|

Table 8: Ablation on key components of FMA-Net++.

Methods # Params Time (s) PSNR↑ SSIM↑ tOF↓ The number of HRBA blocks M

- (a) M = 1 7.7M 0.035 28.29 0.8174 2.461

- (b) M = 2 9.4M 0.048 28.74 0.8339 2.151 Multi-Attention

- (c) self-attn + SFT 13.2M 0.066 28.86 0.8378 2.132 Proposed

- (d) FMA-Net++ 12.8M 0.074 29.24 0.8453 1.956

strong spatio-temporal evidence is available, FMA-Net++ leverages its HRBA backbone rather than relying exclusively on ETE.

#### 5.3 Effectiveness of HRBA and Core Components

We conduct ablation studies to validate the key components and design choices of FMA-Net++, summarizing the main results in Table 8.

First, we investigate the impact of our hierarchical refinement strategy by varying the number of stacked HRBA blocks (M). As shown in Table 8(a, b, d), increasing M from 1 to 2 and finally to our full configuration (M = 4, row d) progressively improves both PSNR and tOF. This demonstrates the effectiveness of hierarchically expanding the temporal receptive field. As visualized in the Suppl., features become progressively sharper and more structurally aligned through the stacked blocks, further validating our hierarchical design.

Next, we validate the effectiveness of the Degradation-Aware (DA) attention within NetR’s multi-attention module. Replacing DA attention with a standard SFT layer [43] for modulation significantly degrades performance, confirming that explicitly leveraging the estimated degradation priors via DA attention is crucial for targeted restoration.

### 6 Conclusion

In this paper, we addressed the challenging problem of joint VSRDB under unknown and dynamically varying exposure conditions. To tackle this challenge, we introduced FMA-Net++, a novel framework built upon HRBA blocks that enables effective sequence-level temporal modeling with efficient parallel processing. Crucially, our proposed ETM layer injects per-frame exposure conditioning into the features. This allows our exposure-aware FGDF module to predict degradation kernels that capture the coupled effects of motion and exposure. Extensive experiments on the proposed REDS-ME and REDS-RE benchmarks, as well as GoPro and real-world videos, demonstrate that FMA-Net++ achieves state-ofthe-art results, showcasing superior performance, efficiency, and robustness while generalizing effectively despite being trained on synthetic data.

Acknowledgements. This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korean Government [Ministry of Science and ICT (Information and Communications Technology)] (Project Number: RS-2022-00144444, Project Title: Deep Learning Based Visual Representational Learning and Rendering of Static and Dynamic Scenes).

### References

- 1. Attention is all you need. Advances in Neural Information Processing Systems 30

(2017) 7

- 2. Learning temporal coherence via self-supervision for gan-based video generation. ACM Transactions on Graphics 39(4), 75–1 (2020) 10
- 3. Bai, H., Pan, J.: Self-supervised deep blind video super-resolution. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(7), 4641–4653 (2024) 2
- 4. Cao, J., Li, Y., Zhang, K., Van Gool, L.: Video super-resolution transformer. arXiv preprint arXiv:2106.06847 (2021) 4
- 5. Chan, K.C., Wang, X., Yu, K., Dong, C., Loy, C.C.: Basicvsr: The search for essential components in video super-resolution and beyond. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4947– 4956 (2021) 2, 3, 4, 8
- 6. Chan, K.C., Wang, X., Yu, K., Dong, C., Loy, C.C.: Understanding deformable alignment in video super-resolution. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 35, pp. 973–981 (2021) 7
- 7. Chan, K.C., Zhou, S., Xu, X., Loy, C.C.: Basicvsr++: Improving video superresolution with enhanced propagation and alignment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5972– 5981 (2022) 2, 3, 4, 6, 8, 10, 11, 12
- 8. Chen, X., Wang, X., Zhou, J., Qiao, Y., Dong, C.: Activating more pixels in image super-resolution transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22367–22377 (2023) 10, 11
- 9. Chiche, B.N., Woiselle, A., Frontera-Pons, J., Starck, J.L.: Stable long-term recurrent video super-resolution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 837–846 (2022) 4
- 10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020) 7
- 11. Fang, N., Zhan, Z.: High-resolution optical flow and frame-recurrent network for video super-resolution and deblurring. Neurocomputing 489, 128–138 (2022) 2, 4
- 12. Haris, M., Shakhnarovich, G., Ukita, N.: Recurrent back-projection network for video super-resolution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3897–3906 (2019) 2
- 13. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 770–778 (2016) 7
- 14. Hu, P., Niklaus, S., Sclaroff, S., Saenko, K.: Many-to-many splatting for efficient video frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3553–3562 (2022) 7
- 15. Jaderberg, M., Simonyan, K., Zisserman, A., et al.: Spatial transformer networks. Advances in Neural Information Processing Systems 28 (2015) 7
- 16. Jo, Y., Oh, S.W., Kang, J., Kim, S.J.: Deep video super-resolution network using dynamic upsampling filters without explicit motion compensation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3224–3232 (2018) 2, 3, 4, 8
- 17. Kai, D., Zhang, Y., Wang, J., Xiao, Z., Xiong, Z., Sun, X.: Event-enhanced blurry video super-resolution. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 4175–4183 (2025) 2, 4, 10, 11, 12

- 18. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5148–5157 (2021) 10
- 19. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D.: Supervised contrastive learning. Advances in Neural Information Processing Systems 33, 18661–18673 (2020) 7, 9, 20
- 20. Kim, T., Lee, J., Wang, L., Yoon, K.J.: Event-guided deblurring of unknown exposure time videos. In: European Conference on Computer Vision. pp. 519–538. Springer (2022) 2, 4
- 21. Kong, L., Dong, J., Ge, J., Li, M., Pan, J.: Efficient frequency domain-based transformers for high-quality image deblurring. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5886–5895 (2023) 10, 11
- 22. Lee, S., Choi, M., Lee, K.M.: Dynavsr: Dynamic adaptive blind video superresolution. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 2093–2102 (2021) 2
- 23. Li, D., Shi, X., Zhang, Y., Cheung, K.C., See, S., Wang, X., Qin, H., Li, H.: A simple baseline for video restoration with grouped spatial-temporal shift. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9822–9832 (2023) 2, 3
- 24. Li, W., Tao, X., Guo, T., Qi, L., Lu, J., Jia, J.: Mucan: Multi-correspondence aggregation network for video super-resolution. In: European Conference on Computer Vision. pp. 335–351. Springer (2020) 2, 3, 4, 8
- 25. Liang, J., Cao, J., Fan, Y., Zhang, K., Ranjan, R., Li, Y., Timofte, R., Van Gool, L.: Vrt: A video restoration transformer. IEEE Transactions on Image Processing 33, 2171–2182 (2024) 2, 4, 10, 11, 12
- 26. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1833–1844 (2021) 10, 11
- 27. Liang, J., Fan, Y., Xiang, X., Ranjan, R., Ilg, E., Green, S., Cao, J., Zhang, K., Timofte, R., Gool, L.V.: Recurrent video restoration transformer with guided deformable attention. Advances in Neural Information Processing Systems 35, 378– 393 (2022) 2, 3, 4, 6, 8, 10, 11, 12
- 28. Lin, J., Huang, Y., Wang, L.: Fdan: Flow-guided deformable alignment network for video super-resolution. arXiv preprint arXiv:2105.05640 (2021) 2
- 29. Liu, C., Yang, H., Fu, J., Qian, X.: Learning trajectory-aware transformer for video super-resolution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5687–5696 (2022) 2, 3, 4, 8
- 30. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10012–10022

(2021) 4

- 31. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of Machine Learning Research 9(11) (2008) 13, 14
- 32. Mittal, A., Soundararajan, R., Bovik, A.C.: Making a “completely blind” image quality analyzer. IEEE Signal Processing Letters 20(3), 209–212 (2012) 10
- 33. Nah, S., Baik, S., Hong, S., Moon, G., Son, S., Timofte, R., Mu Lee, K.: Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 1996–2005 (2019) 2, 10, 21

- 34. Nah, S., Hyun Kim, T., Mu Lee, K.: Deep multi-scale convolutional neural network for dynamic scene deblurring. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3883–3891 (2017) 1, 2, 3, 5, 10, 11, 12, 21, 29
- 35. Oh, J., Kim, M.: Demfi: deep joint deblurring and multi-frame interpolation with flow-guided attentive correlation and recursive boosting. In: European Conference on Computer Vision. pp. 198–215. Springer (2022) 2, 4, 7
- 36. Pan, J., Bai, H., Dong, J., Zhang, J., Tang, J.: Deep blind video super-resolution. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4811–4820 (2021) 2, 10, 11, 12
- 37. Shang, W., Ren, D., Yang, Y., Zhang, H., Ma, K., Zuo, W.: Joint video multiframe interpolation and deblurring under unknown exposure time. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13935–13944 (2023) 4
- 38. Tao, X., Gao, H., Liao, R., Wang, J., Jia, J.: Detail-revealing deep video superresolution. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4472–4480 (2017) 4, 8
- 39. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: European Conference on Computer Vision. pp. 402–419. Springer (2020) 9
- 40. Tian, Y., Zhang, Y., Fu, Y., Xu, C.: Tdan: Temporally-deformable alignment network for video super-resolution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3360–3369 (2020) 2
- 41. Wang, L., Guo, Y., Liu, L., Lin, Z., Deng, X., An, W.: Deep video super-resolution using hr optical flow estimation. IEEE Transactions on Image Processing 29, 4323– 4336 (2020) 2
- 42. Wang, X., Chan, K.C., Yu, K., Dong, C., Change Loy, C.: Edvr: Video restoration with enhanced deformable convolutional networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 0–0 (2019) 3, 4, 6, 8
- 43. Wang, X., Yu, K., Dong, C., Loy, C.C.: Recovering realistic texture in image superresolution by deep spatial feature transform. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 606–615 (2018) 7, 15
- 44. Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., Loy, C.C.: Esrgan: Enhanced super-resolution generative adversarial networks. In: European Conference on Computer Vision Workshops (September 2018) 9
- 45. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004) 10
- 46. Weng, W., Zhang, Y., Xiong, Z.: Event-based blurry frame interpolation under blind exposure. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1588–1598 (2023) 2, 4
- 47. Wolberg, G.: Digital image warping, vol. 10662. IEEE computer society press Los Alamitos, CA (1990) 9
- 48. Xu, K., Yu, Z., Wang, X., Mi, M.B., Yao, A.: Enhancing video super-resolution via implicit resampling-based alignment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2546–2555 (2024) 10, 11, 12
- 49. Youk, G., Oh, J., Kim, M.: Fma-net: Flow-guided dynamic filtering and iterative feature refinement with multi-attention for joint video super-resolution and deblurring. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 44–55 (June 2024) 2, 3, 4, 6, 8, 10, 11, 12, 13, 25, 26, 27

- 50. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer: Efficient transformer for high-resolution image restoration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5728– 5739 (2022) 10, 11, 12, 21
- 51. Zhang, H., Xie, H., Yao, H.: Spatio-temporal deformable attention network for video deblurring. In: European Conference on Computer Vision. pp. 581–596. Springer (2022) 2
- 52. Zhang, H., Xie, H., Yao, H.: Blur-aware spatio-temporal sparse transformer for video deblurring. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2673–2681 (2024) 2, 3, 10, 11, 12
- 53. Zhang, K., Luo, W., Zhong, Y., Ma, L., Liu, W., Li, H.: Adversarial spatio-temporal learning for video deblurring. IEEE Transactions on Image Processing 28(1), 291– 301 (2018) 2, 3
- 54. Zhang, Y., Wang, C., Tao, D.: Video frame interpolation without temporal priors. Advances in Neural Information Processing Systems 33, 13308–13318 (2020) 4
- 55. Zhong, Z., Gao, Y., Zheng, Y., Zheng, B.: Efficient spatio-temporal recurrent neural network for video deblurring. In: European Conference on Computer Vision. pp. 191–207. Springer (2020) 2
- 56. Zhu, C., Dong, H., Pan, J., Liang, B., Huang, Y., Fu, L., Wang, F.: Deep recurrent neural network with multi-scale bi-directional propagation for video deblurring. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 3598– 3607 (2022) 3
- 57. Jia, X., De Brabandere, B., Tuytelaars, T., Gool, L.V.: Dynamic filter networks. Advances in Neural Information Processing Systems 29 (2016) 25, 26
- 58. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014) 20
- 59. Zhang, G., Zhu, Y., Wang, H., Chen, Y., Wu, G., Wang, L.: Extracting motion and appearance via inter-frame attention for efficient video frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5682–5692 (2023) 21

## FMA-Net++: Supplementary Material

In this Supplementary Material, we provide additional details and results to support the main paper. First, we present the detailed training strategy (Sec. S1) and comprehensive implementation details (Sec. S2). Next, we detail the synthesis process for our proposed REDS-ME and REDS-RE benchmarks (Sec. S3). Furthermore, we present further ablation studies and analyses (Sec. S4). Finally, we provide additional qualitative comparisons (Sec. S5), and discuss the limitations and future directions of our FMA-Net++ (Sec. S6).

### S1 Detailed Training Strategy

As described in Sec. 3.5 of the main paper, we adopt a three-stage training strategy where the Exposure Time-aware Feature Extractor (ETE) is first pretrained and subsequently frozen. In this first stage, the ETE is trained to learn an exposure-discriminative representation space. To achieve this, we apply supervised contrastive learning [19], utilizing the discrete exposure levels from our synthetic dataset (detailed in Sec. 4.1 of the main paper and Sec. S3) as pseudolabels. The ETE is optimized to minimize the following contrastive loss Le:

exp q⊤p/α

1 |P| p∈P

, (S1)

Le =

−

log

exp(q⊤p′/α)

q∈B

p′∈B\{q}

where q denotes the anchor feature extracted by the ETE, P contains positive samples within the mini-batch B that share the same exposure label as the anchor, and α is a temperature scaling parameter.

By freezing the ETE after this pretraining stage, we establish a stable exposure reference space. This design effectively prevents representation drift when subsequently optimizing NetD and NetR with LD and Ltotal, which are defined in Eqs. 9 and 10 of the main paper.

### S2 Implementation Details

We train FMA-Net++ using the Adam optimizer [58] with default settings on 4 NVIDIA A6000 GPUs. In the first training stage, the ETE is trained with a minibatch size of 128, a learning rate of 0.01, and a temperature scaling parameter α = 0.5 (as defined in Eq. S1). In the second stage, NetD is trained with a mini-batch size of 8, using an initial learning rate of 2 × 10−4 that is reduced by half at 70%, 85%, and 95% of the total 280K iterations. The third stage jointly trains both NetD and NetR with the same batch size and learning rate schedule as in the second stage.

FMA-Net++ is trained on 10-frame input sequences with a spatial patch size of 64 × 64 and evaluated on full-length videos. The SR scale factor is set to

s = 4 throughout all experiments. The number of HRBA blocks is M = 4 for both NetD and NetR, and the number of multi-flow-mask pairs is n = 9. For the input to the first HRBA block in NetD, these pairs are initialized with no initial motion and full visibility (i.e., f = 0 and o = 1). The degradation kernel size is set to kd = 20. The loss coefficients (defined in Eqs. 9 and 10 of the main paper) are set to λ1 = 10−4, λ2 = 10−4, and λ3 = 0.1. Additionally, we adopt the multi-Dconv head transposed attention (MDTA) and Gated-Dconv feedforward network (GDFN) modules proposed in Restormer [50] for the attention and feed-forward networks in our multi-attention block.

### S3 Details of REDS-ME and REDS-RE Benchmarks

One significant challenge in VSRDB under dynamic exposures is the lack of benchmarks for systematic performance evaluation. To address this, we construct two new benchmarks, REDS-ME (Multi-Exposure) and REDS-RE (RandomExposure), derived from the REDS dataset [33].

#### S3.1 Synthesis of Multi-Exposure Sequences (REDS-ME)

Synthesis Pipeline. The data generation process approximates the continuous degradation formulation defined in Eq. 1 of the main paper. Following the widely adopted methodology for realistic motion blur synthesis [33,34], we implement a blur-then-downsample pipeline. First, the original 120fps REDS videos are interpolated to 1920fps using EMA-VFI [59] to obtain sufficient intermediate high-framerate frames, denoted as H.

To simulate the temporal integration of light over a specific exposure time ∆te, we average a variable number of these consecutive high-framerate frames. The resulting blurry LR frame Xi is formally generated as:

Xi ≈ Ds

Me−1

1 Me

H[i · Mint + k] , (S2)

k=0

where Ds is the spatial bicubic downsampling operator. In the discrete approximation, Mint represents the frame interval of the target 24fps sequences mapped into the 1920fps domain (i.e., Mint = 80), and Me is the variable number of accumulated high-framerate frames determined by the exposure time ∆te. This pipeline isolates exposure duration as the controlled variable governing motion-blur extent. Other camera factors, such as ISO/gain changes, white balance shifts, clipping, and sensor noise, are intentionally outside the scope of REDS-ME/RE; our synthesis follows the standard blur-then-downsample assumption [33,34].

Definition of Exposure Levels. To construct REDS-ME, we synthesize five variants of blurry videos representing different exposure levels by varying Me in Eq. S2. Motivated by the original REDS dataset’s temporal sampling strategy, we mathematically define the exposure levels based on the duty cycle r = N/5,

Algorithm S1 Generation of Exposure Trajectory for a Video in REDS-RE

Require: Total frames T, Initial exposure level E0 ∈ {1, 2, 3, 4, 5}, Update interval

I ∈ {5, 7}

- 1: E ← E0
- 2: for i = 0 to T − 1 do
- 3: if i == 0 or i mod I == 0 then
- 4: ∆E ← RandomChoice({−1, 0, +1}) ▷ Sample exposure change
- 5: E ← Clip(E + ∆E, 1, 5) ▷ Update and keep within bounds
- 6: end if
- 7: Exposure[i] ← E ▷ Assign exposure level to current frame
- 8: end for
- 9: return Exposure

where N ∈ {1,2,3,4,5}. Given the 24fps frame interval ∆t, the per-frame exposure time ∆te is precisely defined as:

N 5

∆t. (S3)

∆te = r∆t =

Consequently, the ratio 5 : 1 represents the shortest exposure (20% duty cycle, minimal motion blur), while 5 : 5 represents the longest exposure (100% duty cycle, severe motion blur). These precise discrete levels also serve as pseudolabels for pretraining our ETE module.

For training, we utilize all five exposure variations from the REDS-ME training set. For evaluation, we adopt the two most challenging exposure levels, 5:4 and 5:5, from the REDS4-ME test set.

#### S3.2 Synthesis of Random-Exposure Sequences (REDS-RE)

To explicitly evaluate robustness under dynamically varying exposure conditions, we construct the REDS-RE benchmark by temporally mixing frames from all five exposure levels within each REDS4-ME test scene. To impose temporal inertia rather than frame-wise randomness, we employ a structured, intervalbased random walk strategy rather than simple frame-wise randomization.

Specifically, the exposure level is updated only at fixed intervals. Depending on the test scene, this update interval is set to either every 5 or 7 frames to maintain consistent temporal inertia within a sequence. At each update step, the exposure level is uniformly sampled to increment (+1), decrement (−1), or remain constant (0), constrained within the valid bounds (5 : 1 to 5 : 5). This generation protocol is detailed in Algorithm S1. As visualized in Fig. S1, this process yields diverse, step-wise exposure trajectories.

### S4 Further Ablation Studies

In this section, we provide further ablation studies and detailed analyses that were omitted from the main paper due to space constraints.

[Figure 73]

[Figure 74]

- Fig. S1: Visualization of synthesized exposure trajectories in the REDS-RE benchmark. Each colored line represents the evolution of the exposure level for a different test scene. The trajectories follow a step-wise random walk with varying update intervals, producing temporally coherent exposure-duration variation.

#### S4.1 Quantitative Evaluation of ETE Embeddings

To demonstrate that our Exposure Time-aware Feature Extractor (ETE) learns a discriminative exposure representation space, we perform a quantitative evaluation of the extracted embeddings ui. Note that ETE is optimized purely via a contrastive loss without an explicit classification head. Therefore, we evaluate its inherent separability using a k-Nearest Neighbors classifier (k = 5) with cosine similarity and 5-fold cross-validation on the extracted features.

As shown in Fig. S2, the ETE embeddings achieve a high Top-1 classification accuracy of 92.0%. The confusion matrix reveals a dominantly diagonal structure, confirming that the embeddings are well-separated according to their true exposure levels. Notably, almost all misclassifications are bounded to adjacent exposure levels without any extreme outliers. For instance, the most frequent confusion occurs between the two longest exposures, with 14 samples of 5 : 4 predicted as 5 : 5, and exactly 14 samples of 5 : 5 predicted as 5 : 4. This is expected, as the visual distinction between 80% and 100% duty-cycle motion blur is inherently ambiguous (see Fig. 5 of the main paper). Conversely, short exposures (e.g., 5 : 1), which exhibit minimal blur, are perfectly isolated with 100% accuracy.

These quantitative results indicate that the ETE successfully captures exposure-dependent structural cues, preserving an ordinal structure consistent with the continuous nature of motion blur. Overall, the ETE provides a reliable exposure-dependent prior for the subsequent restoration networks.

#### S4.2 Progressive Feature Refinement in HRBA

We visualize the intermediate representations of the refined feature FiR,j across four refinement stages in Fig. S3 to illustrate how the HRBA blocks progres-

Confusion Matrix of ETE Embeddings

100 0 0 0 0

5:15:25:35:45:5

GroundTruthExposureLevel

0 94 5 1 0

0 3 96 1 0

0 0 2 84 14

0 0 0 14 86

5:1 5:2 5:3 5:4 5:5

Predicted Exposure Level

- Fig. S2: Confusion matrix of the learned ETE embeddings, evaluated using a k-NN classifier (k = 5) with cosine similarity. The embeddings achieve a high Top-1 accuracy of 92.0% with a strongly diagonal distribution. Notably, almost all misclassifications occur between adjacent exposure levels (e.g., between 5 : 4 and 5 : 5), reflecting the inherent visual ambiguity of severe motion blur. Conversely, short exposures (e.g., 5:1) are perfectly isolated.

sively operate. As shown in the figure, the initial stage exhibits noisy and spatially diffuse activations, while later stages produce increasingly sharper and more structurally aligned features, with high-frequency details (e.g., building edges) becoming more prominent. This progressive sharpening indicates that our hierarchical refinement strategy iteratively enhances feature quality, leading to sharper and more temporally consistent outputs.

#### S4.3 Effect of the Number of Multi-Flow-Mask Pairs

We analyze how the number of multi-flow-mask pairs n within our HRBA backbone affects performance and stability in motion estimation. As shown in Table S1, increasing n consistently improves restoration quality with negligible computational overhead (only a 0.001s increase in runtime from n = 1 to n = 9). A larger n enables the model to establish more robust one-to-many correspondences, effectively leveraging multiple motion hypotheses. This is especially critical under severe motion blur, where a single flow estimation is highly susceptible to localized errors.

Fig. S4 visualizes this effect. With only one pair (n = 1), the predicted optical flow is noisy and spatially distorted, failing to capture accurate motion boundaries. In contrast, using nine pairs (n = 9) produces much cleaner and sharper flow fields that align well with actual object motion. This confirms that integrating the multi-flow mechanism remains effective for robust motion modeling under challenging degradation conditions. We thus retain this component and set n = 9 in our final configuration.

[Figure 75]

[Figure 76]

[Figure 77]

Average of 𝑭𝑭𝑖𝑖𝑅𝑅,1 (1st refinement) Average of 𝑭𝑭𝑖𝑖𝑅𝑅,2 (2nd refinement)

Input (𝑿𝑿𝑖𝑖)

[Figure 78]

[Figure 79]

[Figure 80]

Average of 𝑭𝑭𝑖𝑖𝑅𝑅,3 (3rd refinement) Average of 𝑭𝑭𝑖𝑖𝑅𝑅,4 (4th refinement)

Output ( 𝒀𝒀𝑖𝑖)

- Fig. S3: Visualization of the progressive feature refinement through HRBA blocks across four iterations.

- Table S1: Ablation study for the number of multi-flow-mask pairs (n) on REDS4ME-5:5.

|# n<br><br>|# Params (M)|Runtime (s)<br><br>|REDS4-ME-5:5 PSNR↑ / SSIM↑ / tOF↓|
|---|---|---|---|
|n = 1 n = 5 n = 9|11.9 12.3 12.8<br><br>|0.073 0.074 0.074<br><br>|28.52 / 0.8248 / 2.357<br><br>28.97 / 0.8387 / 2.106<br>29.24 / 0.8453 / 1.956<br>|

#### S4.4 Effect of Exposure-Aware FGDF

FGDF was originally introduced in FMA-Net [49] to perform motion-aware filtering along optical-flow trajectories. In FMA-Net++, we enhance FGDF by explicitly conditioning the filtering weights on the exposure-aware features u (Sec. 3.4 of the main paper). To validate the contribution of this exposure conditioning, we compare three variations of degradation modeling on the severely blurred REDS4-ME-5:5 test set: (1) conventional dynamic filtering (CDF) [57], (2) pure FGDF [49] (without ETM guidance), and (3) our full exposure-aware FGDF.

As shown in Table S2, while transitioning from CDF to pure FGDF improves degradation modeling by tracking motion trajectories, incorporating the exposure-aware guidance yields a further performance leap across all motion magnitudes. This confirms that our exposure-aware conditioning effectively strengthens the underlying motion-aware modeling, providing accurate degradation kernels even in challenging high-motion scenarios.

#### S4.5 Analysis of Loss Functions

We validate the design of our composite loss function LD (Eq. 9 of the main paper), which guides the training of NetD. Specifically, we analyze the impact

[Figure 81]

[Figure 82]

[Figure 83]

Input (𝑿𝑿𝑖𝑖) 𝒇𝒇𝑖𝑖→𝑖𝑖−1𝒀𝒀 (FMA-Net++ w/ 𝑛𝑛 = 1) 𝒇𝒇𝑖𝑖→𝑖𝑖−1𝒀𝒀 (FMA-Net++ w/ 𝑛𝑛 = 9)

###### Fig. S4: Effect of the number of multi-flow-mask pairs (n) on the predicted optical flow for a severely blurred scene.

- Table S2: Comparison of different degradation modeling mechanisms on REDS4-ME-

5:5, reporting the NetD prediction performance. Each cell reports PSNR/tOF values averaged within each motion magnitude interval.

Network: NetD [0,20) [20,40) ≥ 40

CDF [57] 47.67 / 0.046 42.92 / 0.228 34.99 / 0.688 FGDF [49] (w/o ETM) 48.42 / 0.042 43.89 / 0.206 36.87 / 0.651 Exposure-aware FGDF (Ours) 48.57 / 0.040 44.21 / 0.197 37.38 / 0.637

of the coefficients for the warping loss (λ1) and the RAFT supervision loss (λ2) on the REDS4-ME-5:5 test set.

As summarized in Table S3, both loss terms are essential for achieving optimal performance. First, adjusting the weight (λ1) of the warping loss term significantly affects the final restoration quality: an overly large weight interferes with the primary reconstruction objective, while a weight that is too small fails to enforce accurate alignment in the sharp HR space. Second, removing the RAFT supervision (λ2 = 0) causes a notable drop in performance, confirming that pseudo-GT flow supervision is crucial for learning accurate motion priors. Our chosen coefficients (λ1 = λ2 = 10−4) provide the best trade-off, yielding the highest performance across all metrics.

- Table S3: Ablation on the loss coefficients (λ1 and λ2) used in LD evaluated on the REDS4-ME-5:5 test set.

Hyperparameters REDS4-ME-5:5

PSNR ↑ / SSIM ↑ tOF ↓ Analysis on Warping Loss (λ1)

- λ1 = 10−3 29.13 / 0.8395 2.013

- λ1 = 5 × 10−5 29.20 / 0.8437 1.971 Analysis on RAFT Supervision (λ2)

- λ2 = 0 (w/o RAFT) 29.07 / 0.8347 2.143

- λ2 = 10−3 29.12 / 0.8391 2.022 λ2 = 5 × 10−5 29.16 / 0.8409 1.998 λ1 = 10−4, λ2 = 10−4 (Final) 29.24 / 0.8453 1.956

### S5 Additional Qualitative Results

In this section, we provide extended visual comparisons and analyses to complement the quantitative results presented in the main paper.

#### S5.1 Qualitative Comparison with FMA-Net

While Tables 1 and 2 of the main paper demonstrate that our FMA-Net++ quantitatively outperforms the retrained FMA-Net∗ [49] (which utilizes a slidingwindow approach), we further provide a direct visual comparison to highlight their structural restoration capabilities. We specifically focus on challenging scenes containing severe motion blur and low spatial redundancy (e.g., human faces), where leveraging rich, long-range temporal context is critical to compensate for the heavily degraded spatial information.

As shown in Fig. S5, FMA-Net∗, constrained by its narrow temporal window (T = 3), fails to gather sufficient temporal redundancy from adjacent frames and consequently produces over-smoothed textures and distorted facial structures. In contrast, our FMA-Net++ utilizes the HRBA backbone to hierarchically aggregate and refine features, effectively leveraging a broader temporal context. This enhanced temporal modeling allows our model to restore sharper edges and more temporally consistent, high-frequency details. These visual results show that our hierarchical design overcomes the temporal limitations of the baseline FMA-Net framework.

[Figure 84]

[Figure 85]

[Figure 86]

Blurry LR FMA-Net*

[Figure 87]

[Figure 88]

Blurry LR input FMA-Net++ GT

- Fig. S5: Qualitative comparison between FMA-Net∗ [49] and FMA-Net++ (Ours) in a challenging scene featuring complex facial details and severe motion blur. Our model successfully reconstructs sharp and temporally consistent structures without distortion.

#### S5.2 Additional Visual Comparisons and Real-World Generalization

We provide additional qualitative comparisons to complement the results shown in the main paper. Further visual results on challenging scenes from the REDS4ME-5:5 and GoPro test sets are shown in Fig. S6.

Furthermore, examples of real-world video restoration are presented in Fig. S7. These real-world smartphone videos include continuous exposure changes and non-uniform motion blur that are not restricted to the discrete synthetic anchors (5 : 1 to 5 : 5) used during training. Although the ETE is optimized only on discrete exposure anchors, FMA-Net++ still restores these real-world videos well. The ordinal structure analyzed in Sec. S4.1 suggests that the learned exposure-aware feature space can interpolate between nearby exposure conditions, helping the model handle intermediate, unseen exposure states.

### S6 Limitations and Future Work

While FMA-Net++ addresses the coupled degradation of motion blur and dynamic exposure, our current formulation and REDS-ME/RE benchmarks deliberately focus on blur-extent variation induced by temporal integration. They do not explicitly model other entangled camera factors, such as ISO/gain changes, white-balance shifts, clipping, dynamic-range changes, heavy sensor noise, or spatially varying lighting. Extending our framework to jointly handle these complex, concurrent factors presents an important direction for future research.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Restormer* BSSTNet* EV-DeblurVSR*

RVRT*

FMA-Net*

|[Figure 95]|
|---|

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Blurry LR input DBVSR*

BasicVSR++*

IART*

FMA-Net++ GT

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Restormer* BSSTNet* EV-DeblurVSR*

RVRT*

FMA-Net*

|[Figure 107]|
|---|

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Blurry LR input DBVSR*

BasicVSR++*

IART*

FMA-Net++ GT

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Restormer* BSSTNet* EV-DeblurVSR*

RVRT*

FMA-Net*

|[Figure 119]|
|---|

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Blurry LR input DBVSR*

BasicVSR++*

IART*

FMA-Net++ GT

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Restormer* BSSTNet* EV-DeblurVSR*

RVRT*

FMA-Net*

|[Figure 131]|
|---|

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Blurry LR input DBVSR*

BasicVSR++*

IART*

FMA-Net++ GT

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Restormer* BSSTNet* EV-DeblurVSR*

RVRT*

FMA-Net*

|[Figure 143]|
|---|

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Blurry LR input DBVSR*

BasicVSR++*

IART*

FMA-Net++ GT

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Restormer* BSSTNet* EV-DeblurVSR*

RVRT*

FMA-Net*

|[Figure 155]|
|---|

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Blurry LR input DBVSR*

BasicVSR++*

IART*

FMA-Net++ GT

###### Fig. S6: Additional qualitative comparisons on the REDS4-ME-5 : 5 and GoPro [34] test sets. These scenes feature severe motion blur and complex textures, representing highly challenging degradation scenarios. Compared to existing state-of-the-art methods, FMA-Net++ consistently reconstructs sharper structural details and cleaner edges while effectively suppressing severe motion artifacts. Best viewed in zoom.

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

|[Figure 166]|
|---|

Restormer* (5.58 / 34.61)

RVRT* (6.01 / 46.58)

BSSTNet* (5.07 / 53.00)

FMA-Net* (5.66 / 51.94)

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

DBVSR* (5.95 / 45.90)

BasicVSR++* (5.67 / 45.34)

IART* (5.54 / 51.46)

Blurry LR input (NIQE↓ / MUSIQ↑)

FMA-Net++ (4.27 / 60.96)

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

|[Figure 176]|
|---|

Restormer* (5.75 / 40.46)

RVRT* (6.14 / 49.49)

BSSTNet* (5.56 / 51.09)

FMA-Net* (5.45 / 52.33)

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

DBVSR* (5.93 / 46.89)

BasicVSR++* (5.74 / 50.41)

IART* (6.25 / 47.63)

Blurry LR input (NIQE↓ / MUSIQ↑)

FMA-Net++ (4.86 / 55.80)

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

Restormer* (6.34 / 33.59)

RVRT* (6.73 / 44.94)

BSSTNet* (6.12 / 48.15)

FMA-Net* (6.11 / 51.54)

|[Figure 186]|
|---|

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

DBVSR* (6.76 / 39.47)

BasicVSR++* (6.70 / 48.66)

IART* (6.58 / 38.07)

Blurry LR input (NIQE↓ / MUSIQ↑)

FMA-Net++ (5.62 / 56.72)

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Restormer* (5.54 / 28.70)

RVRT*

BSSTNet* (5.41 / 44.69)

FMA-Net* (5.31 / 47.24)

- (5.76 / 42.72)

BasicVSR++*

- (6.20 / 43.88)

|[Figure 196]|
|---|

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

DBVSR* (6.51 / 40.43)

IART* (5.25 / 47.63)

FMA-Net++ (4.75/ 50.09)

Blurry LR input (NIQE↓ / MUSIQ↑)

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Restormer* (6.08 / 30.53)

RVRT* (6.92 / 24.39)

BSSTNet* (6.17 / 36.82)

FMA-Net* (6.27 / 36.94)

|[Figure 206]|
|---|

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

DBVSR* (6.83 / 24.96)

BasicVSR++* (6.67 / 29.92)

IART* (6.92 / 28.93)

FMA-Net++ (5.69 / 40.19)

Blurry LR input (NIQE↓ / MUSIQ↑)

- Fig. S7: Qualitative comparisons on challenging real-world videos captured with smartphones. These videos contain continuous exposure changes and non-uniform motion blur, deviating from the discrete synthetic anchors used during training. Despite this domain gap, FMA-Net++ recovers legible text and fine textures, and achieves favorable no-reference scores (NIQE↓ / MUSIQ↑) in these examples. Best viewed in zoom.

