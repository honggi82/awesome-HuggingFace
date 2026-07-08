# arXiv:2509.26618v5[cs.CV]8Nov2025

## DA2: DEPTH ANYTHING IN ANY DIRECTION

Haodong Li123§, Wangguandong Zheng1, Jing He3, Yuhao Liu1, Xin Lin2, Xin Yang34, Ying-Cong Chen34‡, Chunchao Guo1‡

1Tencent Hunyuan 2UC San Diego 3 HKUST(GZ) 4 HKUST hal211@ucsd.edu; yingcongchen@ust.hk; chunchaoguo@gmail.com

[Figure 1]

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

- A
- B

A B

D C

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

A B

D C

A

B

Figure 1: Teaser of DA2. Powered by large-scale training data from our panoramic data curation engine, and the distortion-aware SphereViT, DA2 predicts dense distance from a single 360◦ panorama, with remarkable geometric fidelity. The reconstructed 3D structures exhibit sharp geometric details and robust performance across diverse scenes, highlighting DA2’s strong zero-shot generalization.

ABSTRACT

Panorama has a full FoV (360◦×180◦), offering a more complete visual description than perspective images. Thanks to this characteristic, panoramic depth estimation is gaining increasing traction in 3D vision. However, due to the scarcity of panoramic data, previous methods are often restricted to in-domain settings, leading to poor zero-shot generalization. Furthermore, due to the spherical distortions inherent in panoramas, many approaches rely on perspective splitting (e.g., cubemaps), which leads to suboptimal efficiency. To address these challenges, we propose DA2: Depth Anything in Any Direction, an accurate, zero-shot generalizable, and fully end-to-end panoramic depth estimator. Specifically, for scaling up panoramic data, we introduce a data curation engine for generating high-quality panoramic depth data from perspective, and create ∼543K panoramic RGB-depth pairs, bringing the total to ∼607K. To further mitigate the spherical distortions, we present SphereViT, which explicitly leverages spherical coordinates to enforce the spherical geometric consistency in panoramic image features, yielding improved performance. A comprehensive benchmark on multiple datasets clearly demonstrates DA2’s SoTA performance, with an average 38% improvement on AbsRel

§Work primarily done during an internship at Tencent Hunyuan. ‡Corresponding author.

over the strongest zero-shot baseline. Surprisingly, DA2 even outperforms prior in-domain methods, highlighting its superior zero-shot generalization. Moreover, as an end-to-end solution, DA2 exhibits much higher efficiency over fusion-based approaches. Both the code and the curated panoramic data has be released. Project page: depth-any-in-any-dir.github.io.

- 1 INTRODUCTION

Unlike the commonly used perspective images, panorama offers an immersive 360◦×180◦ view, capturing visual content from any direction. This wide FoV makes panorama an essential visual representation in computer vision, empowering a variety of exciting applications, such as AR/VR (Chen

- et al., 2023) and immersive visual generation (Yang et al., 2025a; Kalischek et al., 2025). However, immersive visual (2D) experiences alone are not enough. To push the new frontier of panoramic application scenarios, high-quality depth (3D) information from panoramas is crucially needed for 3D reconstruction and more advanced features such as 3D scene generation (Skywork AI, 2025; Li et al., 2025; Lu et al., 2025), physical simulation (Shah et al., 2025), etc. Inspired by this, we focuses on estimating scale-invariant1 distance2 from each panorama pixel to the sphere center (i.e., the 360◦ camera) in an end-to-end manner, with high-fidelity and strong zero-shot generalization.

Panoramic depth estimation is particularly valuable for applications requiring comprehensive spatial awareness. However, capturing or rendering panoramas is much more challenging than perspective images, panoramic depth data is much more limited in both quantity and diversity. Consequently, early methods were largely trained and tested in in-domain settings, with highly limited zero-shot generalization. Given the wealth of high-quality perspective depth data, is it possible to transform them into panoramic? Motivated by this, we propose a data curation engine, transforming perspective samples into high-quality panoramic data. Concretely, given a perspective RGB image with known horizontal and vertical FoVs, we first apply Perspective-to-Equirectangular (P2E) projection to map the image onto the spherical space. However, due to the limited FoV of perspective images (with a typical horizontal range of 70◦−90◦), only a small portion of the spherical space can be covered (as highlighted in Fig. 3’s left sphere). Thus, such a P2E projected image can be viewed as an “incomplete” panorama. Then, panoramic out-painting will be performed to generate a “complete” panorama to match the input of our model, using an image-to-panorama out-painter: FLUX-I2P (BFL, 2024; Tencent, 2025). For the associated GT depth, we apply only the P2E projection without out-painting, due to concerns on the absolute accuracy of out-painted depth. Overall, this data curation engine substantially boosts the quantity and diversity of panoramic data, and significantly strengthens the zero-shot performance of DA2, as shown in Fig. 2 and Tab. 2.

Panoramas typically use equirectangular projection (ERP)3 to represent the 360◦×180◦ visual space. However, a 3D spherical space cannot be “losslessly” projected onto a 2D plane. During the sphereto-plane projection, distortions and stretching are inevitable, particularly near the poles. This spherical distortion is analogous to the challenge in world map projection, where you can never accurately express both the areas and shapes of each land. To mitigate the impact of spherical distortion, inspired by the positional embeddings in Vision Transformers (ViTs), we propose SphereViT—the main backbone of DA2. Specifically, from the layout of ERP, we first compute the spherical angles (azimuth and polar) of each pixel in the camera-centric spherical coordinates. After that, we expand this two-channel angle field into the image feature dimension using sine-cosine basis embedding, forming the Spherical Embedding. Since all panoramas have the same full FoV, this spherical embedding can be fixed and reusable. Therefore, to inject spherical awareness, it’s only necessary to let the image feature “attend” to the spherical embedding, but not vice versa—the spherical embedding doesn’t need to be further refined. Consequently, rather than adding positional embeddings onto the image features before self-attention, as in standard ViTs (Vaswani et al., 2017; Dosovitskiy et al., 2020), SphereViT uses cross-attention: image features are regarded as queries and the spherical em-

1Please see Supp’s Sec. D for discussions on: metric, scale-invariant (biased), and affine-invariant (relative). 2We acknowledge the distinction between distance (d = x2 + y2 + z2) and depth (d = z). We focus on

scale-invariant distance prediction. Please allow us to use “depth” occasionally for readability and fluency.

3ERP can represent a full vertical FoV (i.e., 180◦). If smaller than 180◦, cylindrical projection can be used, such as the panoramic camera mode in mobile phones. Both can present a full horizontal FoV (i.e., 360◦).

[Figure 21]

[Figure 22]

- Figure 2: Scaling-law curves of model performance vs data size. Native, high-quality panoramic data is scarce, constraining the zero-shot generalization of panoramic depth estimators. With our data curation engine, DA2 achieves steadily and clearly higher performance as more perspective depth data are converted to panoramic form. Detailed numerical results are provided in Tab. 2.

beddings as keys and values. This design lets the image feature explicitly attend to the panorama’s spherical geometry, yielding distortion-aware representations and improved performance.

To validate DA2, we conduct a comprehensive benchmark on scale-invariant distance combining multiple well-recognized evaluation datasets. However, due to the scarcity of panoramic data, existing zero-shot approaches in panoramic depth estimation are limited, whereas in perspective, there exist many powerful zero-shot methods. Therefore, to ensure a more fair and comprehensive comparison, following the panoramic depth estimation pipeline proposed by Wang et al. (2025c;d), we also benchmark DA2 against prior zero-shot perspective depth estimators (Hu et al., 2024; Yin et al.,

- 2023; Piccinelli et al., 2024; 2025b; Wang et al., 2025a;c;d; Bhat et al., 2023; Yang et al., 2024a;b; He et al., 2024b), The results in Tab. 1 clearly demonstrate DA2’s SoTA performance, with an average 38% improvement on AbsRel over the strongest zero-shot baseline. Notably, it even surpasses prior in-domain methods, further underscoring its superior generalization ability. Beyond that, DA2 seamlessly supports various applications, such as panoramic multi-view reconstruction, home decoration, and robotics simulation (please see our Supp’s Sec. A). Our key contributions are:

- • Panoramic data curation engine. We introduce a data curation engine that generates highquality panoramic depth data from perspective data, greatly scaling up the panoramic depth training data and substantially improving the zero-shot generalization ability of DA2.
- • SphereViT. We propose SphereViT—the primary backbone of DA2. By directly leveraging the spherical coordinates of panoramas, SphereViT effectively mitigates the impact of spherical distortions and enhances the spherical geometry awareness of image features.
- • Comprehensive benchmark. Both zero-shot / in-domain, panoramic / perspective methods are compared to build a comprehensive benchmark for panoramic depth estimation.
- • SoTA performance. Experimental results clearly demonstrate DA2’s SoTA performance. DA2 even beats prior in-domain methods. It also enables many downstream applications.

2 RELATED WORKS

- 2.1 PERSPECTIVE DEPTH ESTIMATION

Perspective depth estimation is being advanced very rapidly. Metric and scale-invariant depth models, driven by large-scale training data, have achieved strong results, like UniDepth (Piccinelli et al.,

- 2024; 2025b), Metric3D (Hu et al., 2024; Yin et al., 2023), DepthPro (Bochkovskiy et al., 2025), and MoGe (Wang et al., 2025c;d). Relative depth models also benefit greatly from scaling up the training data, like DepthAnything (Yang et al., 2024a;b). Another line of work fine-tunes massively pre-trained generative models, e.g., Stable Diffusion (Rombach et al., 2022; Ho et al., 2020; He et al., 2024a; Li et al., 2024b; Liang et al., 2024; Gu et al., 2024), FLUX (BFL, 2024; Yang et al., 2025b), with limited high-quality data, also yielding impressive results (Ke et al., 2024; He et al., 2024b; Wang et al., 2025b; Li et al., 2024a). Despite these remarkable advances, perspective methods remain constrained by the limited FoV and cannot estimate depth in all directions simultaneously. In contrast, DA2 targets full FoV depth estimation with strong zero-shot generalization.

Partial Panorama Full Panorama

RGB

[Figure 23]

|× 𝑁|
|---|

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

P2E Proj. FLUX-I2P

Panoramic Out-painting

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

P2E Proj.

Vertical FoV (YFoV) Horizontal FoV (XFoV)

Perspective Projection Equirectangular Projection (ERP) Equirectangular Projection (ERP)

GT Depth

- Figure 3: Panoramic data curation engine. This module converts large-scale, high-quality perspective RGB–depth pairs into full panoramas through P2E projection and panoramic out-painting using FLUX-I2P. It dramatically scales up the panoramic depth training data, forming a solid training data foundation for DA2. The highlighted area on the spheres indicate the FoV coverage.

- 2.2 PANORAMIC DEPTH ESTIMATION

In-domain. Due to the scarcity of panoramic data, most existing methods are constrained to indomain settings. Network designs have evolved from CNNs (Zioulis et al., 2018; Zhuang et al.,

- 2022)) to ViTs (Shen et al., 2022; Yun et al., 2023). Pipeline designs are mainly aimed to mitigate the spherical distortions inherent in panoramas. Many approaches fuse features from both the ERP (1 panorama) and cubemap (6 perspectives) projections (Wang et al., 2020; Jiang et al., 2021; Wang et al., 2022; Li et al., 2022; Ai et al., 2023; Wang & Liu, 2024). For alternative solutions, SliceNet (Pintore et al., 2021) and HoHoNet (Sun et al., 2021) use RNNs or LSTMs along longitudes. SphereDepth (Yan et al., 2022), Elite360D (Ai & Wang, 2024), HUSH (Lee et al., 2025) introduce spherical icosahedral meshes and spherical harmonics. While effective, these strategies still require additional modules, making them less streamlined and efficient. DA2 introduces SphereViT to handle the spherical distortions in an end-to-end manner, without extra modules.

Zero-shot. With the rise of zero-shot perspective depth estimators, there has been a trend toward developing zero-shot depth estimators for panoramas. 360MonoDepth (Rey et al., 2022) blends tangent perspective depths predicted by MiDaS (Ranftl et al., 2020) on an icosahedral mesh, but suffers from multi-view inconsistencies. PanDA (Cao et al., 2025) leverages M¨obius transformation-based data augmentation for self-supervision. UniK3D Piccinelli et al. (2025a) separately predicts camera rays and distance maps, can generalize on various cameras. But their performance remains suboptimal, due to limited panoramic data: ∼20K labeled and ∼92K unlabeled in PanDA, ∼29K in UniK3D. DepthAnyCamera (Guo et al., 2025) projects perspective images with various horizontal FoVs (20◦−124◦, ≪360◦) into spherical space, can also generalize on various cameras. But its performance still remains constrained by the incomplete FoVs. In contrast, DA2 introduces a panoramic data curation engine, significantly boosting the quantity and diversity of panoramic data from available perspective data, yielding a clearly enhanced zero-shot generalization performance.

- 3 METHODOLOGY

This section presents the methodology of DA2 in detail, covering the panoramic data curation engine (Sec. 3.1) and SphereViT with its training loss functions (Sec. 3.2).

- 3.1 PANORAMIC DATA CURATION ENGINE

“The quality of your data determines the ceiling of your ambitions.” (Surge AI, 2020)

Due to the scarcity of high-quality panoramic data, existing panoramic depth estimators are often trained and evaluated within specific domains, greatly restricting their zero-shot generalization ability and real-world applicability. Thus, the very first goal of this work is to scale up the panoramic data and build a strong data foundation for DA2. Motivated by this, we propose a perspective-topanoramic data curation engine that generates high-quality panoramic data from perspective data.

- As illustrated in Fig. 3, the inputs of the panoramic data curation engine are a perspective image sized (Wper,Hper) and its FoVs, i.e., XFoV and YFoV. XFoV represents the coverage of this perspective image in the azimuth field |ϕl−ϕr| and YFoV denotes the coverage in the polar angle field |θu−θd|.
- At first, P2E projection will be performed to map the perspective image onto the spherical space.

ℒdis ℒnor

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

ViT 𝒘/ 𝐸sphere

[Figure 44]

ViT (DINO)

Distance

Normal

SphereViT

[Figure 45]

Distance-to-3DPC

3DPC-to-Normal

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Query

CrossAttn

Key Value

ViT Image Feature 𝑍

[Figure 51]

[Figure 52]

|2π|
|---|

[Figure 53]

0

[Figure 54]

Azimuth

[Figure 55]

Polar

π

0

Sphere Angle Field 𝐴 Spherical Emb. 𝐸sphere

3D Point Cloud (3DPC)

- Figure 4: The architecture of SphereViT and training losses. By leveraging the spherical embedding

Esphere, which is explicitly derived from the spherical coordinates of panoramas, SphereViT produces distortion-aware image features, yielding more accurate geometrical estimation for panoramas. The

training supervision combines a distance loss Ldis for globally accurate distance values and a normal loss Lnor for locally smooth and sharp surfaces. The effect of Lnor is ablated in Fig. 6 (b) and Tab. 3.

Specifically, we start by obtaining the focal lengths from both FoVs:

Wper 2 × tan FoV

Hper 2 × tan FoV2 y

. (1)

fx =

, fy =

x

2

Then, the 3D vector d and its unit vector dˆ from the perspective camera to each 2D pixel (x,y) of the perspective image (x ∈ [0,Wper − 1],y ∈ [0,Hper − 1]) are given by:

(x − Wper2−1) fx

(y − Hper2−1) fy

d |d|

,1], dˆ =

. (2) Then, in the spherical space, the azimuth ϕ (longitude) and polar θ (colatitude) angles of dˆ are:

d = [

,

ϕ = atan2(dˆx,dˆz) + ϕc, θ = arccos(dˆy) + θc, (3)

where (ϕc,θc) denote the spherical coordinates of the perspective image’s optical center, used as offsets to obtain the absolute longitude and colatitude of each pixel. After that, the mapped pixel

position (u,v) on the ERP image (i.e., panorama) sized (Wpano,Hpano) is given by:

θ π

ϕ 2π

Hpano, (4)

Wpano, v =

u =

where ϕ ∈ [0,2π),θ ∈ [0,π]. After P2E projection, due to the limited FoV of perspective images, only a small portion of the sphere can be covered, as highlighted in Fig. 3’s left sphere. This incompleteness leads to suboptimal performance: 1) the model lacks global context since it never observes the full views of panoramic images, particularly near the poles; and 2) spherical distortions vary significantly between the equator and poles, with severe stretching occurring at high latitudes.

Thus, following (Tencent, 2025), the second step of our data curation engine adopts a LoRA (Hu et al., 2022) fine-tuned FLUX model named FLUX-I2P for panoramic out-painting, generating “full” panoramas from the “partial” panoramas. Earlier panoramic out-painting methods (Gao et al., 2024; Feng et al., 2023) often exhibited spatial inconsistencies, especially near the poles and the left–right seam. To address this, FLUX-I2P concatenates image features with the spherical coordinates (azimuth ϕ and polar θ) along the channel dimension before feeding them into the Diffusion Transformer (DiT) (Peebles & Xie, 2022), to improve the spatial coherence. For the GT depth associated with the perspective image, we apply only the P2E projection without panoramic out-painting, because the absolute accuracy of out-painted depth is hard to guarantee. As ablated in Tab. 3, although the panoramic out-painting on the P2E projected GT depth is not performed, FLUX-I2P’s panoramic out-painting on the RGB images clearly improves the panoramic depth estimation performance by a large margin, demonstrating its significance in our panoramic data curation engine.

- 3.2 SPHEREVIT & TRAINING LOSSES

This data curation engine creates ∼543K panoramic samples, scales the total from ∼63K to ∼607K (∼10 times), significantly addressing the data scarcity issue that causes poor generalization. Here we focus on DA2’s model structure and training, to effectively learn from the greatly scaled-up data.

Recently, ViT-based depth models have achieved great success (Wang et al., 2025c;d; Yang et al., 2024a;b; Piccinelli et al., 2025a), where positional embeddings (PE) are crucial for encoding spatial information. For perspectives, PE is typically derived from the 2D (x,y) pixel coordinates. However, for panoramas, pixel coordinates (u,v) correspond to spherical coordinates (longitude ϕ and latitude θ). The spherical nature introduces non-uniformity: high-latitude regions (near the poles) are stretched, while low-latitude regions (near the equator) are compressed. Conventional 2D PE cannot account for this spherical distortion, limiting the model’s spherical spatial understanding. To address this, many approaches fuse features from both the ERP (1 panorama) and cubemap (6 perspectives) projections or employ auxiliary modules, introducing inefficiencies and complexity. In contrast, DA2 aims to handle the distortions more simply and efficiently, without extra modules.

To this end, DA2 proposes SphereViT, as illustrated in Fig. 4. SphereViT leverages the spherical coordinates of panoramas to efficiently and explicitly inject spherical-awareness into the ViT image features, yielding distortion-aware representations and improved performance. Specifically, we first compute the azimuth and polar angles (ϕ,θ) of each pixel (u,v) in an ERP image sized (W,H):

u W

v H

. (5) Then, given the image feature Z ∈ R(H

ϕ = 2π ×

, θ = π ×

′×W′)×D, where W′ = WP ,H′ = HP and P is patch size, we resize and flatten this two-channel angle field A ∈ RH×W×2 (Eq. 5) into A′ ∈ R(H

′×W′)×2. Motivated by the PE mechanism of ViT, sine-cosine embedding is utilized to expand A′’s channel from 2 to the image feature dimension D. Concretely, we first define a series of coefficients {2d

′

n=1, where D′ = D4 ,dn = (n−1) log

}D

n

2(H′)

D′ . Then, for each two-channel unit of A′: A′i,j = [ϕi,θj] ∈ R1×2, where i ∈ [0,W′ − 1],j ∈ [0,H′ − 1], we transpose and multiple it with the coefficients:

2d

ϕi θj × 2d

- 1

ϕi 2d

2

ϕi ··· 2dD′ϕi

- 2d

. (6) Eq. 6’s result is shaped 2×D′. We then apply the sine-cosine embedding on each unit of this matrix:

### 2d

··· 2dD′ =

1

2

### θj 2d

θj ··· 2dD′θj

1

2

### ϕi)]⊤ [sin(2d

### ϕi)]⊤ ··· [sin(2dD′ϕi),cos(2dD′ϕi)]⊤ [sin(2d

### [sin(2d

### ϕi),cos(2d

### ϕi),cos(2d

1

1

2

2

. (7)

### θj)]⊤ [sin(2d

θj)]⊤ ··· [sin(2dD′ θj),cos(2dD′ θj)]⊤

### θj),cos(2d

### θj),cos(2d

1

1

2

2

Eq. 7 has a shape of 2 × D′ × 2. Now, the flattened transformation of Eq. 7—with a dimension of 2 × D′ × 2 = D—is the unit (i,j) of the Spherical Embedding Esphere ∈ R(H

### ′×W′)×D.

As discussed in Sec. 1, all panoramas share the same 360◦×180◦ FoV, so the spherical embedding is fixed, reusable, and doesn’t need to be further refined. Thus, to inject spherical awareness, it’s only necessary to let image features Z “attend” to the embedding Esphere, but not vice versa. Accordingly, SphereViT replaces the usual self-attention (after addition: Z + Esphere) with cross-attention, where image features Z serve as queries and the spherical embeddings Esphere act as keys and values:

CrossAttn(Z,Esphere ) = SoftMax

ZWQ (Esphere WK)⊤ √Dk

(Esphere WV ), (8)

### ′×W′)×D.

k are learnable projection matrixs, and Z,Esphere ∈ R(H

where WQ,WK,WV ∈ RD×D

This cross-attention with spherical embedding Esphere allows the image features Z to “learn” the underlying spherical structures of the panoramas, producing distortion-aware representations and leading to clearly enhanced geometrical fidelity as demonstrated in Fig. 6 (a) and Tab. 3.

Training Losses. DA2’s SphereViT is trained end-to-end to estimate dense, scale-invariant distance Dˆ ∈ RH×W from a panoramic RGB input I ∈ RH×W×3. The supervision combines two terms: a distance loss Ldis that enforces globally accurate distance values, and a normal loss Lnor that promotes locally smooth, sharp geometrical surfaces, especially in regions where distance values are similar but surface normals vary significantly. Concretely, let Dˆ and D⋆ be the predicted and GT

- Table 1: Quantitative comparison. For a fair and comprehensive benchmark, we include both zeroshot / in-domain, panoramic / perspective approaches. The best and second best performances are highlighted (in zero-shot setting). In all settings (both zero-shot and in-domain), the best and second best performances are bolded and underlined. DA2 outperforms all other methods no matter in zero-shot or all settings, particularly showing large gains under the zero-shot setting. Median alignment (scale-invariant) is adopted by default. △: Affine-invariant alignment (scale and shiftinvariant), for prior relative depth estimators: DepthAnything v1v2 (Yang et al., 2024a;b), Lotus (He

- et al., 2024b), and PanDA (Cao et al., 2025). We also report PanDA’s results in median alignment for fairness. ⋆: Implemented by ourselves (code will be released). The unit is percentage (%).

Stanford2D3D Matterport3D PanoSUNCG Rank↓ Rank↓ AbsRel↓RMSE↓ δ1 ↑ δ2 ↑ AbsRel↓RMSE↓ δ1 ↑ δ2 ↑ AbsRel↓RMSE↓ δ1 ↑ δ2 ↑ Zero-shot All

Categories Method

OmniDepth 19.96 61.52 68.7788.91 29.01 76.43 68.3087.94 11.43 37.10 87.0593.65 – 26.33 FCRN 18.37 57.74 72.3092.07 24.09 67.04 77.0391.74 9.79 39.73 92.2396.59 – 24.00 BiFuse 12.09 41.42 86.6095.80 20.48 62.59 84.5293.19 5.92 25.96 95.9098.23 – 16.83

EGFormer 15.28 49.74 81.8593.38 14.73 60.25 81.5893.90 – – – – – 15.50 SliceNet 12.49 43.70 83.7794.14 17.64 61.33 87.1694.83 – – – – – 14.17 SphereDepth 11.58 45.12 86.6696.42 12.05 59.22 86.2095.19 – – – – – 12.42 BiFuse++ 11.17 37.20 87.8396.49 14.24 51.90 87.9095.17 5.24 24.77 96.3098.35 – 12.17

UniFuse 11.14 36.91 87.1196.64 10.63 49.41 88.9796.23 5.28 27.04 95.9198.25 – 11.25 HoHoNet 10.14 38.34 90.5496.93 14.88 51.38 87.8695.19 – – – – – 10.33 Elite360D 11.82 37.56 88.7296.84 11.15 48.75 88.1596.46 – – – – – 10.00

In-domain

PanoFormer 11.31 35.57 88.0896.23 9.04 44.70 88.1696.61 5.34 18.90 94.8798.83 – 9.50 HRDFuse 9.35 31.06 91.4097.98 9.67 44.33 91.6296.69 6.90 27.44 92.1597.42 – 9.50 SphereFusion 8.99 31.94 92.5797.55 11.45 48.85 87.0196.13 – – – – – 7.92 ACDNet 9.84 34.10 88.7297.04 10.10 46.29 90.0096.78 – – – – – 7.08 DepthAnywhere 11.80 35.10 91.0097.10 8.50 – 91.7097.60 – – – – – 5.33 OmniFusion 9.50 34.74 89.8897.69 9.00 42.61 91.8997.97 – – – – – 5.00 HUSH 7.82 33.32 93.8498.49 8.38 41.64 92.8796.98 – – – – – 3.67

Lotus-D⋆△ 45.88 48.86 37.6768.39 32.39 85.86 48.1578.23 37.96 77.02 46.0877.41 17.00 30.33 Lotus-G⋆△ 45.08 47.90 38.3869.18 31.82 84.51 49.1178.92 38.02 76.82 46.1677.51 16.17 29.50

DepthAnything⋆△ 37.21 43.41 47.0876.93 24.46 66.12 60.5488.32 24.58 52.22 64.8690.39 14.58 27.42 DepthAnythingv2⋆△ 36.79 43.39 47.6676.96 25.85 70.67 58.4286.19 23.90 50.74 66.8690.89 14.25 27.25 ZoeDepth⋆ 17.60 33.74 74.2692.86 18.43 53.46 72.1893.12 21.16 44.81 69.3494.45 11.75 22.58 360MonoDepth 16.50 28.23 74.5692.98 20.83 79.09 65.5888.95 11.43 28.29 90.7598.12 10.83 21.67

Zero-shot ((fusion)

VGGT⋆ 18.70 33.50 74.0883.90 10.78 38.80 88.7097.72 8.43 25.67 94.0498.19 8.42 15.08 Metric3D⋆ 12.93 20.80 84.7796.52 14.11 45.11 83.0996.59 11.42 26.95 90.4597.33 7.67 15.17 UniDepth⋆ 15.06 20.48 76.9990.34 11.12 36.20 88.6697.94 10.40 27.29 92.5998.00 7.50 13.92

MoGe 15.81 25.76 79.0283.32 10.04 35.91 90.8098.45 8.60 25.80 93.8598.31 6.33 12.08 UniDepthv2⋆ 13.08 20.46 82.1289.21 10.86 37.68 88.7697.86 9.74 25.94 93.0698.30 6.25 12.17 Metric3Dv2⋆ 11.59 21.78 86.0797.36 17.78 62.55 72.3593.22 7.30 24.54 94.2598.25 6.08 14.08

MoGev2 14.69 24.24 79.9884.39 10.34 36.91 89.4898.24 8.26 24.67 94.1598.52 5.58 11.25

PanDA 48.44 53.06 33.9251.33 37.10 101.5 42.5167.29 34.73 79.69 44.4971.45 17.50 30.83

DepthAnyCamera 15.26 22.80 75.4792.90 15.60 61.85 77.2795.62 12.78 27.88 89.6797.85 9.75 19.42 PanDA△ 16.48 23.64 73.2685.42 8.88 33.25 92.0998.26 6.71 21.85 95.4298.25 5.33 10.33 UniK3D 11.31 19.72 88.9495.33 9.66 32.66 93.0098.58 11.46 25.38 90.1898.02 4.58 8.75

Zero-shot (end2end)

DA2 (Ours) 7.23 14.00 95.4598.38 6.67 28.82 95.6198.60 5.96 19.07 96.1298.55 1.00 1.67

distances. Then the surface normals can be obtained with a distance-to-normal operator D2N, giving Nˆ = D2N(Dˆ) and N⋆ = D2N(D⋆) when GT normals are not directly available. Since we focuses on scale-invariant distance, Dˆ is median-aligned before loss computing: Dˆmed = Dˆ × Median(D

⋆)

Median(Dˆ) . While training the SphereViT, we minimize the per-pixel L1 difference for both Ldis and Lnor:

1 |Ω| p∈Ω

1 |Ω| p∈Ω

D ˆpmed − Dp⋆ , Lnor =

N ˆp − Np⋆ , (9)

Ldis =

where Ω is the set of valid pixels. For Lnor, we prefer the L1 norm over the commonly-used angular discrepancy 1−⟨Nˆp,Np⟩ as the latter may introduce gradient collapse and destabilize training. The total loss is a weighted sum: L = λdLdis + λnLnor, where λd and λn are scalar weights.

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETTINGS

Training Datasets. DA2 is trained using 7 high-quality datasets. 6 perspective: Hypersim (Roberts et al., 2021), Virtual-KITTI 2 (Cabon et al., 2020), MVS-Synth (Huang et al., 2018), Unreal-

- Table 2: Ablation study on training data scaling. The results show clear, steady performance gains as the size of training data grows. Pano indicates a perspective dataset converted into panoramic through

- our data curation engine. The average results across multiple datasets are reported (also in Tab. 3). Please see Supp’s Sec. B for more discussions about the date curation engine and the curated data.

S3D HPSPano VKPano MVSPano US4KPano 3DKBPano DRPano Data Size AbsRel↓ RMSE↓ δ1 ↑ δ2 ↑

✓ ✗ ✗ ✗ ✗ ✗ ✗ 63,097 8.07 25.13 92.91 97.29 ✓ ✓ ✗ ✗ ✗ ✗ ✗ 96,677 7.10 21.94 94.69 98.09 ✓ ✓ ✓ ✗ ✗ ✗ ✗ 136,326 6.84 21.50 95.09 98.25 ✓ ✓ ✓ ✓ ✗ ✗ ✗ 148,326 6.78 21.40 95.25 98.31 ✓ ✓ ✓ ✓ ✓ ✗ ✗ 164,726 6.76 21.42 95.35 98.31 ✓ ✓ ✓ ✓ ✓ ✓ ✗ 316,722 6.66 21.00 95.55 98.41

✓ ✓ ✓ ✓ ✓ ✓ ✓ 606,522 6.62 20.63 95.73 98.51

Stereo4K (Tosi et al., 2021), 3D-Ken-Burns (Niklaus et al., 2019), Dynamic Replica (Karaev et al., 2023), totaling 543,425 samples; 1 panoramic: Structured3D (Zheng et al., 2020) (63,097 samples).

Evaluation Datasets & Metrics. For a fair and reproducible comparison, DA2 is evaluated on three widely-used, well-recognized benchmarks in panoramic depth estimation: Stanford2D3D-S (Armeni et al., 2017) (all splits), Matterport3D (Chang et al., 2017) (test split), and PanoSUNCG (Wang et al., 2018) (test split), using 2 error metrics (AbsRel, RMSE), and 2 accuracy metrics (δ1, δ2). Please see the implementation details (Sec. E) and metric formulations (Sec. C) in our Supp.

4.2 QUANTITATIVE & QUALITATIVE COMPARISONS

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

DA2 (~0.1s) UniK3D (~0.1s) MoGev2 (~28s)

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

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

Figure 5: Qualitative comparisons. Compared with UniK3D and MoGev2, DA2 delivers more accurate geometric predictions and, as an end-to-end approach, achieves significantly higher inference efficiency than fusion-based methods.

Tab. 1 presents a comprehensive comparison of DA2 with previous SoTA approaches. Following (Wang et al., 2025c;d), we also include prior perspective methods for a more thorough comparison. As demonstrated in Tab. 1, DA2 consistently outperforms all other methods across vari-

- ous settings. Particularly in the zeroshot setting, DA2 shows significant gains over the second-best method by an average of 38% in AbsRel and 22% in RMSE, achieving a remarkable average δ1 of 95.73% and δ2 of 98.51%. Notably, even as a zero-shot model, DA2 surpasses earlier in-domain methods as well, further underscoring its superior zeroshot generalization ability.

In addition, for better access the DA2’s performance, we also conduct qualitative comparisons with UniK3D (Piccinelli et al., 2025a)—the strongest prior zero-shot, end-to-end method, and MoGev2 (Wang et al., 2025d)—the strongest prior zero-shot, fusion-based method, as highlighted in Fig. 5. Thanks to our data curation engine, DA2 is trained with about 21× more panoramic data than UniK3D, exhibiting clearly more accurate geometrical predictions. DA2 continuously yields better results over MoGev2, as its panoramic performance is restricted by the multi-view inconsistencies during fusion, e.g., irregular walls, fragmented buildings, etc. We also report the inference times: as an end-to-end method, DA2 achieves significantly higher efficiency than fusion-based approaches.

- 4.3 ABLATION STUDIES

Training Data. As reported in Tab. 2, DA2’s performance steadily improves as more perspective depth data converted into panoramic, thanks to our data curation engine. Fig. 2 further shows rapid gains once the curated perspective data is introduced, with performance gradually converging as the data scales. Even near convergence, further improvements are still anticipated with additional data.

- Table 3: Ablation studies on: 1) the panoramic out-painting in the data curation engine, 2) spherical embedding Esphere in the SphereViT, and 3) the auxiliary normal loss Lnor. The results below demonstrate that each design plays a vital role in achieving the final remarkable performance of DA2.

Pano. Out-painting Spherical Emb. Esphere Normal Loss Lnor Data Size AbsRel↓ RMSE↓ δ1 ↑ δ2 ↑

✗ ✓ ✓ 606,522 7.59 23.80 94.12 97.86 ✓ ✗ ✓ 606,522 6.84 20.87 95.26 98.43 ✓ ✓ ✗ 606,522 6.99 21.53 95.25 98.37

##### ✓ ✓ ✓ 606,522 6.62 20.63 95.73 98.51

[Figure 67]

[Figure 68]

[Figure 69]

|𝒘/ Spherical<br><br>Emb. 𝐸sphere|
|---|

|𝒘/ Normal Loss ℒnor|
|---|

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

| |
|---|

[Figure 74]

[Figure 75]

[Figure 76]

|𝒘/𝒐 Spherical Emb. 𝐸sphere|
|---|

|𝒘/𝒐 Normal Loss ℒnor|
|---|

Curved Curved

(a) (b)

Figure 6: Ablation studies of DA2. (a) Removing the spherical embedding Esphere causes curved, distorted geometry. (b) Omitting the normal loss Lnor yields rougher surfaces and more artifacts.

Panoramic Out-painting. It is a crucial step in the panoramic data curation engine, generating full RGB panoramas from P2E-projected perspective images (Fig. 3). Comparing Tab. 2’s 1st row with Tab. 3’s 1st row, DA2’s performance can be improved only modestly via scaling up the perspective w/o panoramic out-painting, yielding a 0.48 gain in AbsRel. In contrast, incorporating (w/) outpainting yields a much larger boost than “w/o out-painting” (∼3 times), with a 1.45 gain in AbsRel (Tab. 2’s 1st row vs. Tab. 3’s last row), clearly showing the importance of panoramic out-painting.

Spherical Embedding. We here ablate the impact of spherical embedding Esphere in the SphereViT. As shown in Tab. 3 (2nd vs. last row), including Esphere noticeably boosts DA2’s performance. Fig. 6 (a) further illustrates that incorporating the spherical embedding produces more accurate geometric understandings on panoramas, while its absence often leads to suboptimal performance (e.g., curved walls), highlighting its effectiveness in mitigating the spherical distortions.

Training Losses. We further ablate the auxiliary normal loss Lnor used for training the SphereViT. As shown in Tab. 3 (3rd vs. last row), adding Lnor boosts DA2’s performance clearly. Also, as highlighted in Fig. 6 (b), normal supervision yields flatter, smoother, and more coherent geometry, reducing the artifacts that typically appear in ambiguous regions (e.g., corners, edges, and the upper or lower poles), where distance values may be similar but surface normals differ substantially.

- 5 LIMITATION & CONCLUSION

[Figure 77]

[Figure 78]

[Figure 79]

Limitation. Despite the strong performance enabled by the large-scale training data thanks to our panoramic data curation engine and distortion-aware SphereViT, DA2 still faces several constraints. As the training resolution (1024×512) is lower than higher-definition formats such as 2K or 4K, and the curated perspective data provide only partially available GT depth in the spherical space, DA2 may occasionally miss fine details (Fig. 7 (a)) and produce visible seams along the panorama’s left–right boundaries (which should ideally be seamlessly aligned), as illustrated in Fig. 7 (b).

(a)

[Figure 80]

[Figure 81]

[Figure 82]

(b)

| |
|---|

| |
|---|

| |
|---|

Figure 7: DA2’s limitations. (a) The white lamp’s predicted distance is mistakenly aligned with the desk surface. (b) Visible seams appear along the predictions at lower left–right boundaries.

Conclusion. We introduce DA2, an end-to-end, zero-shot generalizable, panoramic distance (scaleinvariant) estimator that unites a panoramic data curation engine with the distortion-aware SphereViT. Trained on over 600K samples (∼543K curated from perspective and ∼63K native panoramas), DA2 delivers SoTA zero-shot performance, outperforming prior methods (both zero-shot and in-domain) by a clear margin while remaining efficient and fully end-to-end. This work shows that scaling up panoramic data and explicitly modeling the spherical geometry enables high-quality and robust 360◦×180◦ geometrical estimation, paving the way for high-fidelity 3D scene applications, e.g., immersive 3D scene creation, AR/VR, robotics simulation, physical simulation, etc.

- 6 ACKNOWLEDGMENT

We sincerely thank Dr. Hualie Jiang from CUHK(SZ) for providing access to Matterport3D’s test split, which is invaluable in enabling the comprehensive benchmark. We also thank Luigi Piccinelli from ETH Zurich and Ruicheng Wang from USTC and MSRA for their thoughtful discussions.

- 7 LLMS IN PAPER WRITING

LLMs (e.g., GPT-4, GPT-5) were employed solely to polish the grammar and sentence structures, for improving the readability, clarity, and fluency. They made no contribution to the original research content. All the scientific and technical content of this paper was written entirely by humans.

## SUPPLEMENTARY MATERIALS OF DA2: DEPTH ANYTHING IN ANY DIRECTION

A APPLICATIONS OF DA2

Leveraging its remarkable capability in zero-shot generalizable panoramic depth estimation, DA2 effectively enables a wide range of 3D reconstruction-related applications.

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

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

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

- Figure 8: Pano3R: Panoramic Multi-view Reconstruction. Given panoramic images of different rooms from a house / apartment, DA2 enables the reconstruction of a globally aligned 3D point cloud, ensuring the spatial coherence across multiple panoramic views of different rooms.

- A.1 PANO3R: PANORAMIC MULTI-VIEW RECONSTRUCTION

- A house / apartment typically consists of multiple distinct rooms, which may exhibit substantial geometric variations. Thanks to the strong zero-shot generalization and high geometric consistency in panoramic depth estimation, DA2 is able to reconstruct a holistic 3D point cloud representation of the indoor layout, leveraging multiple panoramic images captured from different rooms. As shown in Fig. 8, the rooms can be consistently aligned via simple translation, without requiring any scaling or rotation operations. This characteristic highlights the robustness and superior geometric consistency of DA2’s depth estimation, enabling seamless alignment of shared structures such as walls and doors, facilitating applications such as VR-based indoor apartment tours and layout visualization.

- A.2 LAYERED HOME RENOVATION

As illustrated in Fig. 9 (a), given indoor panoramas with three distinct complexity levels—“empty”, “simple”, and “full”—the multiple sets of 3D point clouds reconstructed from DA2’s panoramic distance maps exhibit high consistency. They can be seamlessly aligned with fine details. As demonstrated in the zoom-in regions of Fig. 9 (a), the fused point clouds are free of distortions: the text on the blackboard is sharp, and the wall boundaries are consistently aligned.

- A.3 ROBOTICS SIMULATION

Benefiting from DA2’s robust panoramic distance estimation, the reconstructed 3D point cloud can serve as a reliable 3D simulation environment for robot manipulation. As illustrated in Fig. 9 (b), it provides a practical 3D platform for simulating and demonstrating robotic tasks.

- (a)

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Empty Only Empty + Simple Empty + Simple + Full

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

- (b)

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

- Figure 9: More applications of DA2: (a) Layered Home Renovation. The three input panoramas correspond to different levels of foreground object complexity, denoted as “empty”, “simple”, and “full”. The zoom-in views show that the reconstructed 3D point clouds from these panoramas remain consistently aligned (primarily in backgrounds). (b) Robotics Simulation. The reconstructed 3D point cloud can serve as a practical 3D platform for evaluating robot manipulation performance.

- B PANORAMIC DATA CURATION ENGINE (MORE DETAILS)

As discussed in Sec. 4.1, 6 perspective datasets (Hypersim (Roberts et al., 2021), Virtual-KITTI 2 (Cabon et al., 2020), MVS-Synth Huang et al. (2018), UnrealStereo4K (Tosi et al., 2021), 3D-KenBurns (Niklaus et al., 2019), Dynamic Replica (Karaev et al., 2023)) are transformed into panoramic via the proposed panoramic data curation engine. The curated datasets are summarized in Tab. 4. As shown, the sampling probabilities are normalized across datasets primarily considering data size to ensure a balanced influence during DA2’s training process. Each dataset represents a domain, this balanced mixture ensures DA2’s performance will not be over influenced by a few strong datasets, achieving stable scaling behavior across datasets, and optimal cross-domain generalization. To this end, our data curation engine generates ∼543K high-quality panoramic image–depth pairs from perspective data, expanding the total dataset to ∼607K samples. This substantially enriches the quantity and diversity of panoramic data, constructs a solid data foundation for DA2, and in turn significantly enhances the zero-shot performance of DA2, as demonstrated in Fig. 2 and Tab. 2.

- Table 4: Perspective datasets processed by the panoramic data curation engine. For each dataset, the

vertical FoV (YFoV) is derived directly from the horizontal FoV (XFoV) as YFoV = XFoV × WH , where (W,H) denotes the input panorama’s width and height.

Category Dataset Name Abbreviation (Tab. 2) Data Size In-or-outdoor XFoV Sam. Probability

Perspective Hypersim HPS 39,649 In 60◦ 16.59% Perspective Virtual-KITTI 2 VK 33,580 Out 80◦ 14.05% Perspective MVS-Synth MVS 12,000 Out 80◦ 5.02% Perspective UnrealStereo4K US4K 16,400 Various 90◦ 6.86% Perspective 3D-Ken-Burns 3DKB 151,996 Various 60◦−90◦ 15.91% Perspective Dynamic Replica DR 289,800 In 85◦ 15.16%

Panoramic Structured3D S3D 63,097 In 360◦ 26.41%

- C EVALUATION METRICS

Concretely, given the predicted panoramic depth Dˆ and GT D⋆, median alignment is performed on the predicted distance Dˆ before computing the metrics:

Dˆmed = Dˆ ×

Median(D⋆) Median(Dˆ)

, (10)

following the evaluation protocols in prior works (Lee et al., 2025; Wang & Liu, 2024; Yun et al., 2023; Yan et al., 2025; 2022; Li et al., 2022; Shen et al., 2022; Zhuang et al., 2022; Wang et al., 2022; Sun et al., 2021; Pintore et al., 2021; Jiang et al., 2021; Wang et al., 2020; Rey et al., 2022; Piccinelli et al., 2025a; Cao et al., 2025), After that, the AbsRel and RMSE are given by:

AbsRel =

1 |Ω| p∈Ω

|Dˆpmed − Dp⋆| Dp⋆

, RMSE =

1 |Ω| p∈Ω

(Dˆpmed − Dp⋆)2, (11)

where Ω is the set of valid pixels. δ1 and δ2 denotes the proportion of pixels satisfying Max(Dp⋆/Dˆpmed,Dˆpmed/Dp⋆) < 1.25 and < 1.252 respectively.

- D DIFFERENCE AMONG: METRIC & SCALE-INVARIANT (BIASED) & AFFINE-INVARIANT (RELATIVE)

Metric and Scale-invariant Depth. In depth (or distance) estimation, metric depth Dmetric is the strictest setting, where the predicted values correspond to absolute physical distances and can be

directly used to reconstruct a “real-scale” point cloud. Scale-invariant (or biased) depth Dbiased is still strict but slightly more relaxed than metric: predictions include a global bias or shift, but not in the absolute global scale. Although the depths are not metric, the underlying 3D structure is preserved perfectly (Tab. 5), because the global bias or shift is preserved. During training & evaluation, for scale-invariant depth, median alignment (scale-invariant) is typically adopted to re-scale the underlying 3D structure to real-world size (please see Sec. C). For metric depth, no alignment should ideally be required, but median alignment is still commonly applied because absolute scales can be ambiguous (cameras with different focal lengths can capture visually similar pictures but with substantially different absolute depths) (Hu et al., 2024; Yin et al., 2023; Piccinelli et al., 2025a).

DA2 focuses on panoramic scale-invariant (or biased) distance estimation for two reasons: 1) like metric distance, scale-invariant distance also preserves the full underlying 3D geometry, and 2) DA2 targets on the strong zero-shot generalization across diverse domains, enforcing absolute scales would introduce significant optimization challenges, as indoor and outdoor scenes differ drastically in scale, making the additional cost outweigh the benefits.

Affine-invariant Depth. Affine-invariant (or relative) depth Drelative is the loosest definition, much more relaxed than either biased or metric depth, preserving only the “ordering” of depths (which

point is closer or farther). Since neither scale nor shift is preserved, affine-invariant depth Drelative cannot be used to reconstruct a reasonable 3D point cloud (Tab. 5), but it’s useful for tasks where only relative geometry matters. Affine-invariant alignment (scale and shift-invariant) is usually adopted during training & evaluation of affine-invariant depth estimators. Concretely, given the predicted Dˆrelative and GT depth D⋆, least squares fitting is performed:

### min

scale, shift

p∈Ω

scale × (Dˆrelative,p + shift) − Dp⋆ 22, (12)

where Ω is the set of valid pixels and the aligned predicted depth is: Dˆaff = scale×(Dˆrelative+shift). The summarized difference is listed in Tab. 5. Note that for the “Illustration with Dmetric” of scaleinvariant and affine-invariant depth, we only list the most widely adopted formats, passing over other scales for Dbiased and other specific transformations for Drelative like exp(·) and log(·).

- Table 5: Summarized difference on depth maps among metric, scale-invariant (biased), and affineinvariant (relative). Both metric and scale-invariant depth fully preserve the 3D geometry. Due to the absence of bias or shift, affine-invariant depth is unable to reconstruct an accurate 3D structure.

[Figure 138]

[Figure 139]

[Figure 140]

Depth Category Metric Depth Scale-invariant Depth Affine-invariant Depth Illustration with Dmetric Dmetric max(DmetricD

###### Dmetric−min(Dmetric) max(Dmetric)−min(Dmetric)

metric)

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

3D Point Cloud on:

[Figure 159]

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |

5m

1m (for instance)

5m 1m (for instance)

5m 1m (for instance)

1m

1m

1m

(for instance)

(for instance)

(for instance)

- E IMPLEMENTATION DETAILS

DA2 is implemented in PyTorch (Paszke et al., 2019). In SphereViT, the backbone of “ViT (DINO)” is initialized from DINOv2-ViT-L (Oquab et al., 2023) with 24 self-attention blocks, following (He et al., 2024b; Ke et al., 2024), to leverage the pre-trained visual priors. The “ViT w/ Esphere” is a lightweight ViT contains only 4 cross-attention blocks. Training the SphereViT takes ∼5,000 optimization iterations on 32 NVIDIA H20 GPUs, with a batch size of 768. The distributed training is implemented with Accelerate (Gugger et al., 2022). We set λdis = 1.0,λnor = 2.0 for balanced loss values. Panoramas and GT depth maps are fed to SphereViT at a resolution of 1024×512. Please see the sampling probabilities of different data sources in Tab. 4. In the panoramic data curation engine, the FLUX-I2P is fine-tuned on FLUX.1 [dev] (BFL, 2024), largely following Tencent (2025). The LoRA rank is set to 256 during the LoRA (Hu et al., 2022) fine-tuning. The positive prompt is: a clean, realistic, high-quality, high-resolution, panoramic image of a [*] scene, where [*] is either indoor or outdoor. The negative prompt is: messy, low-quality, blur, noise, low-resolution, abnormal. The ϕc,θc are randomly selected from ±30◦ and ±15◦, respectively. The panoramic out-painting of 543,425 perspective RGB images from various datasets is performed on 64 NVIDIA H20 GPUs and over nearly 9 days. The running time reported in Fig. 5 is tested on a NVIDIA H20 GPU at a resolution of 1024×512, excluding I/O operations.

- F PRIOR SOTA METHODS FOR COMPARISONS

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

5m 1m (for instance)

1m

(for instance)

In-domain Baselines. 17 previous in-domain, panoramic depth estimation approaches are selected for the quantitative comparison in Tab. 1: HUSH (Lee et al., 2025), DepthAnywhere (Wang & Liu, 2024), Elite360D (Ai & Wang, 2024), EGFormer (Yun et al., 2023), SphereFusion (Yan et al.,

- 2025), SphereDepth (Yan et al., 2022), OmniFusion (Li et al., 2022), HRDFuse (Ai et al., 2023), PanoFormer (Shen et al., 2022), ACDNet (Zhuang et al., 2022), BiFuse++ (Wang et al., 2022), HoHoNet (Sun et al., 2021), SliceNet (Pintore et al., 2021), UniFuse (Jiang et al., 2021), BiFuse (Wang et al., 2020), FCRN (Laina et al., 2016), and OmniDepth (Zioulis et al., 2018).

Zero-shot, fusion-based baselines. 13 zero-shot, fusion-based panoramic depth estimators are selected or implemented. 1 is originally panoramic: 360MonoDepth (Rey et al., 2022). The other 16 are prior SoTA perspective depth estimators: Metric3D & Metric3Dv2 (Yin et al., 2023; Hu

- et al., 2024), VGGT (Wang et al., 2025a), MoGe & MoGev2 (Wang et al., 2025c;d), UniDepth & UniDepthv2 (Piccinelli et al., 2024; 2025b), ZoeDepth (Bhat et al., 2023), DepthAnything & DepthAnythingv2 (Yang et al., 2024a;b), and Lotus-D & Lotus-G (He et al., 2024b). These methods are implemented for panoramic scenarios via multi-view splitting and fusion.

Zero-shot, end-to-end baselines. Prior zero-shot, end-to-end methods are rare, and their performance are limited by the scarcity of high-quality panoramic depth data. Only 3 methods are compared: UniK3D (Piccinelli et al., 2025a), PanDA (Cao et al., 2025), and DepthAnyCamera (Guo

- et al., 2025). As evident in Tab. 1, PanDA predicts affine-invariant (relative) depth, while other methods including DA2 predict at least the scale-invariant (biased) depth.

REFERENCES

Hao Ai and Lin Wang. Elite360d: Towards efficient 360 depth estimation via semantic-and distanceaware bi-projection fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9926–9935, 2024.

Hao Ai, Zidong Cao, Yan-Pei Cao, Ying Shan, and Lin Wang. Hrdfuse: Monocular 360deg depth estimation by collaboratively learning holistic-with-regional depth distributions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13273–13282, 2023.

Iro Armeni, Sasha Sax, Amir R Zamir, and Silvio Savarese. Joint 2d-3d-semantic data for indoor

scene understanding. arXiv preprint arXiv:1702.01105, 2017. BFL. Flux. https://github.com/black-forest-labs/flux, 2024. Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨uller. Zoedepth: Zeroshot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023.

Alexey Bochkovskiy, Ama¨el Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan Richter, and Vladlen Koltun. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second. In The Thirteenth International Conference on Learning Representations, 2025.

Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020.

Zidong Cao, Jinjing Zhu, Weiming Zhang, Hao Ai, Haotian Bai, Hengshuang Zhao, and Lin Wang. Panda: Towards panoramic depth anything with unlabeled panoramas and mobius spatial augmentation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 982–992, 2025.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niebner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. In 2017 International Conference on 3D Vision (3DV), pp. 667–676. IEEE Computer Society, 2017.

Timothy Chen, Miguel Ying Jie Then, Jing-Yuan Huang, Yang-Sheng Chen, Ping-Hsuan Han, and Yi-Ping Hung. spellorama: An immersive prototyping tool using generative panorama and voiceto-prompts. In ACM SIGGRAPH 2023 Posters, pp. 1–2, 2023.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, G Heigold, S Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.

Mengyang Feng, Jinlin Liu, Miaomiao Cui, and Xuansong Xie. Diffusion360: Seamless 360 degree

panoramic image generation based on diffusion models. arXiv preprint arXiv:2311.13141, 2023. Penglei Gao, Kai Yao, Tiandi Ye, Steven Wang, Yuan Yao, and Xiaofeng Wang. Opa-ma: Text

guided mamba for 360-degree image out-painting. arXiv preprint arXiv:2407.10923, 2024.

Songen Gu, Wei Yin, Bu Jin, Xiaoyang Guo, Junming Wang, Haodong Li, Qian Zhang, and Xiaoxiao Long. Dome: Taming diffusion model into high-fidelity controllable occupancy world model. arXiv preprint arXiv:2410.10429, 2024.

Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. Accelerate: Training and inference at scale made simple, efficient and adaptable. https://github.com/huggingface/accelerate, 2022.

Yuliang Guo, Sparsh Garg, S Mahdi H Miangoleh, Xinyu Huang, and Liu Ren. Depth any camera: Zero-shot metric depth estimation from any camera. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26996–27006, 2025.

Jing He, Haodong Li, Yongzhe Hu, Guibao Shen, Yingjie Cai, Weichao Qiu, and Ying-Cong Chen. Disenvisioner: Disentangled and enriched visual prompt for customized image generation. arXiv preprint arXiv:2410.02067, 2024a.

Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and Ying-Cong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024b.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multi-view stereopsis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

Hualie Jiang, Zhe Sheng, Siyu Zhu, Zilong Dong, and Rui Huang. Unifuse: Unidirectional fusion for 360 panorama depth estimation. IEEE Robotics and Automation Letters, 6(2):1519–1526, 2021.

Nikolai Kalischek, Michael Oechsle, Fabian Manhardt, Philipp Henzler, Konrad Schindler, and Federico Tombari. Cubediff: Repurposing diffusion-based image models for panorama generation,

2025. URL https://arxiv.org/abs/2501.17162. Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. CVPR, 2023.

Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9492– 9502, 2024.

Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Federico Tombari, and Nassir Navab. Deeper depth prediction with fully convolutional residual networks. In 2016 Fourth international conference on 3D vision (3DV), pp. 239–248. IEEE, 2016.

Jongsung Lee, Harin Park, Byeong-Uk Lee, and Kyungdon Joo. Hush: Holistic panoramic 3d scene understanding using spherical harmonics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 16599–16608, 2025.

Haodong Li, Hao Lu, and Ying-Cong Chen. Bi-tta: Bidirectional test-time adapter for remote physiological measurement. In European Conference on Computer Vision, pp. 356–374. Springer, 2024a.

Renjie Li, Panwang Pan, Bangbang Yang, Dejia Xu, Shijie Zhou, Xuanyang Zhang, Zeming Li, Achuta Kadambi, Zhangyang Wang, Zhengzhong Tu, et al. 4k4dgen: Panoramic 4d generation at 4k resolution. In The Thirteenth International Conference on Learning Representations, 2025.

Xiao-Lei Li, Haodong Li, Hao-Xiang Chen, Tai-Jiang Mu, and Shi-Min Hu. Discene: Object decoupling and interaction modeling for complex scene generation. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–12, 2024b.

Yuyan Li, Yuliang Guo, Zhixin Yan, Xinyu Huang, Ye Duan, and Liu Ren. Omnifusion: 360 monocular depth estimation via geometry-aware fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2801–2810, 2022.

Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6517–6526, 2024.

TaiMing Lu, Tianmin Shu, Alan Yuille, Daniel Khashabi, and Jieneng Chen. Genex: Generating an explorable world. In The Thirteenth International Conference on Learning Representations, 2025.

Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 3d ken burns effect from a single image. ACM Transactions on Graphics, 38(6):184:1–184:15, 2019.

Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems, 32, 2019.

William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.

Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10106–10116, 2024.

Luigi Piccinelli, Christos Sakaridis, Mattia Segu, Yung-Hsu Yang, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unik3d: Universal camera monocular 3d estimation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1028–1039, 2025a.

Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110, 2025b.

Giovanni Pintore, Marco Agus, Eva Almansa, Jens Schneider, and Enrico Gobbetti. Slicenet: deep dense depth estimation from a single indoor panorama using a slice-based representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11536– 11545, 2021.

Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020.

Manuel Rey, Mingze Yuan Area, and Christian Richardt. 360monodepth: High-resolution 360 monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, volume 3, 2022.

Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In International Conference on Computer Vision (ICCV) 2021, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Uzair Shah, Sara Jashari, Muhammad Tukur, Mowafa Househ, Jens Schneider, Giovanni Pintore, Enrico Gobbetti, and Marco Agus. Virtual staging of indoor panoramic images via multi-task learning and inverse rendering. IEEE Computer Graphics and Applications, 2025.

Zhijie Shen, Chunyu Lin, Kang Liao, Lang Nie, Zishuo Zheng, and Yao Zhao. Panoformer: panorama transformer for indoor 360◦ depth estimation. In European Conference on Computer Vision, pp. 195–211. Springer, 2022.

Skywork AI. Matrix-3d: Omnidirectional explorable 3d world generation. https:// matrix-3d.github.io/, 2025.

Cheng Sun, Min Sun, and Hwann-Tzong Chen. Hohonet: 360 indoor holistic understanding with latent horizontal features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2573–2582, 2021.

Surge AI. Surge ai. https://www.surgehq.ai/, 2020. Tencent. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from

words or pixels. https://3d.hunyuan.tencent.com/sceneTo3D, 2025. Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. Smd-nets: Stereo mixture density networks. In Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Fu-En Wang, Hou-Ning Hu, Hsien-Tzu Cheng, Juan-Ting Lin, Shang-Ta Yang, Meng-Li Shih, Hung-Kuo Chu, and Min Sun. Self-supervised learning of depth and camera motion from 360 videos. In Asian Conference on Computer Vision, pp. 53–68. Springer, 2018.

Fu-En Wang, Yu-Hsuan Yeh, Min Sun, Wei-Chen Chiu, and Yi-Hsuan Tsai. Bifuse: Monocular 360 depth estimation via bi-projection fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 462–471, 2020.

Fu-En Wang, Yu-Hsuan Yeh, Yi-Hsuan Tsai, Wei-Chen Chiu, and Min Sun. Bifuse++: Selfsupervised and efficient bi-projection fusion for 360 depth estimation. IEEE transactions on pattern analysis and machine intelligence, 45(5):5448–5460, 2022.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294–5306, 2025a.

Jiyuan Wang, Chunyu Lin, Cheng Guan, Lang Nie, Jing He, Haodong Li, Kang Liao, and Yao Zhao. Jasmine: Harnessing diffusion prior for self-supervised depth estimation. arXiv preprint arXiv:2503.15905, 2025b.

Ning-Hsu Albert Wang and Yu-Lun Liu. Depth anywhere: Enhancing 360 monocular depth estimation via perspective distillation and unlabeled data augmentation. Advances in Neural Information Processing Systems, 37:127739–127764, 2024.

Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5261–5271, 2025c.

Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546, 2025d.

Qingsong Yan, Qiang Wang, Kaiyong Zhao, Bo Li, Xiaoweo Chu, and Fei Deng. Spheredepth: Panorama depth estimation from spherical domain. In 2022 International Conference on 3D Vision (3DV), pp. 1–10. IEEE, 2022.

Qingsong Yan, Qiang Wang, Kaiyong Zhao, Jie Chen, Bo Li, Xiaowen Chu, and Fei Deng. Spherefusion: Efficient panorama depth estimation via gated fusion. In International Conference on 3D Vision 2025, 2025.

Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10371–10381, 2024a.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024b.

Shuai Yang, Jing Tan, Mengchen Zhang, Tong Wu, Gordon Wetzstein, Ziwei Liu, and Dahua Lin. Layerpano3d: Layered 3d panorama for hyper-immersive scene generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–10, 2025a.

Xin Yang, Jiantao Lin, Yingjie Xu, Haodong Li, and Yingcong Chen. Advancing high-fidelity 3d and texture generation with 2.5 d latents. arXiv preprint arXiv:2505.21050, 2025b.

Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9043–9053, 2023.

Ilwi Yun, Chanyong Shin, Hyunku Lee, Hyuk-Jae Lee, and Chae Eun Rhee. Egformer: Equirectangular geometry-biased transformer for 360 depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6101–6112, 2023.

Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3d: A large photo-realistic dataset for structured 3d modeling. In European Conference on Computer Vision, pp. 519–535. Springer, 2020.

Chuanqing Zhuang, Zhengda Lu, Yiqun Wang, Jun Xiao, and Ying Wang. Acdnet: Adaptively combined dilated convolution for monocular panorama depth estimation. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pp. 3653–3661, 2022.

Nikolaos Zioulis, Antonis Karakottas, Dimitrios Zarpalas, and Petros Daras. Omnidepth: Dense depth estimation for indoors spherical panoramas. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 448–465, 2018.

