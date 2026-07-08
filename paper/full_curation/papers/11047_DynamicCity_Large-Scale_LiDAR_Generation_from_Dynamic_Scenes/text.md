# arXiv:2410.18084v3[cs.CV]3Dec2025

[Figure 1]

## DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes

Hengwei Bian, Lingdong Kong , Haozhe Xie, Liang Pan , Yu Qiao, Ziwei Liu

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### WorldBench Team

[Figure 6]

[Figure 7]

Project Lead Corresponding Authors

Urban scene generation has been developing rapidly recently. However, existing methods primarily focus on generating static and single-frame scenes, overlooking the inherently dynamic nature of real-world driving environments. In this work, we introduce DynamicCity, a novel 4D occupancy generation framework capable of generating large-scale, high-quality dynamic 4D scenes with semantics. DynamicCity mainly consists of two key models. (1) A VAE model for learning HexPlane as the compact 4D representation. Instead of using naive averaging operations, DynamicCity employs a novel Projection Module to effectively compress 4D features into six 2D feature maps for HexPlane construction, which significantly enhances HexPlane fitting quality (up to 12.56 mIoU gain). Furthermore, we utilize an Expansion & Squeeze Strategy to reconstruct 3D feature volumes in parallel, which improves both network training efficiency and reconstruction accuracy than naively querying each 3D point (up to 7.05 mIoU gain, 2.06x training speedup, and 70.84% memory reduction). (2) A DiT-based diffusion model for HexPlane generation. To make HexPlane feasible for DiT generation, a Padded Rollout Operation is proposed to reorganize all six feature planes of the HexPlane as a squared 2D feature map. In particular, various conditions could be introduced in the diffusion or sampling process, supporting versatile 4D generation applications, such as trajectory- and command-driven generation, inpainting, and layout-conditioned generation. Extensive experiments on the CarlaSC and Waymo datasets demonstrate that DynamicCity significantly outperforms existing state-of-the-art 4D occupancy generation methods across multiple metrics. The code and models have been released to facilitate future research.

[Figure 8]

Project Page: https://dynamic-city.github.io GitHub Repo: https://github.com/3DTopia/DynamicCity

[Figure 9]

1 Introduction

Urban scene generation has garnered growing attention recently, which could benefit various related applications, such as robotics and autonomous driving. Compared to its 3D object generation counterpart, generating urban scenes remains an under-explored field, with many new research challenges such as the presence of numerous moving objects, large-scale scenes, and long temporal sequences [15]. For example, in autonomous driving scenarios, a scene typically comprises multiple objects from various categories, such as vehicles, pedestrians, and vegetation, captured over a long sequence (e.g., 200 frames) spanning a large area (e.g., 80×80×6.4 meters3). Although in its early stages, 4D occupancy generation holds great potential to enhance the understanding of the 3D world, with wide-reaching and profound implications.

Due to the complexity of occupancy data, many efficient learning frameworks have been introduced for large-scale 3D scene generation. X3 [30] utilizes a hierarchical voxel diffusion model to generate outdoor 3D scenes based on VDB data structure. PDD [18] introduces a pyramid discrete diffusion model to progressively generate high-quality 3D scenes. SemCity [17] resolves outdoor scene generation by leveraging a triplane

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

##### . . .

T=1 T=2 T=3 T=4 T=5 T=N

[Figure 17]

Command-Driven Scene Generation

Dynamic Object Generation

1 2

[Figure 18]

[Figure 19]

- 1 2

[Figure 20]

[Figure 21]

Forward Turn Left

[Figure 22]

3 4

[Figure 23]

[Figure 24]

[Figure 25]

3 4

Turn Left Turn Left

[Figure 26]

5 6

[Figure 27]

[Figure 28]

[Figure 29]

5 6

Turn Left Forward

Trajectory-Guided Generation Dynamic Scene Inpainting Layout-Conditioned Generation

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Before After

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

- Figure 1 Dynamic occupancy generation from DynamicCity. We introduce a new 4D generation model that generates diverse 4D scenes of large spatial scales (80 × 80 × 6.4 meter3) and long sequential modeling (up to 128 frames), enabling a diverse set of downstream applications. For more detailed examples, kindly refer to our Project Page.

diffusion model. Despite achieving impressive occupancy generation, they primarily focus on generating static and single-frame 3D occupancy, and hence fail to effectively capture the dynamic nature of outdoor environments. Recently, a few works [41, 49] have explored 4D scene generation. However, generating high-quality long-sequence 4D scenes is still a challenging and open problem [21, 23].

In this work, we propose a novel 4D occupancy generation framework, DynamicCity, enabling the generation of large-scale, high-quality dynamic occupancy scenes, which mainly consists of two stages:

- • A VAE network for learning compact 4D representations, i.e., HexPlanes [7, 11].
- • A HexPlane Generation model based on DiT [26].

VAE for 4D Occupancy. Given a set of 4D occupancy scenes, DynamicCity first encodes the scene as a 3D feature volume sequence with a 3D backbone. Afterward, we propose a novel Projection Module based on transformer operations to compress the feature volume sequence into six 2D feature maps. In particular, the proposed projection module significantly enhances HexPlane fitting performance, offering an improvement of up to 12.56% mIoU compared to conventional averaging operations. After constructing the HexPlane based on the projected six feature planes, we employ an Expansion & Squeeze Strategy (ESS) to decode the HexPlane into multiple 3D feature volumes in parallel. Compared to individually querying each point, ESS further improves HexPlane fitting quality (with up to 7.05% mIoU gain), significantly accelerates training speed (by up to 2.06x), and substantially reduces memory usage (by up to a relative 70.84% memory reduction).

DiT for HexPlane. Using the encoded HexPlane, we use a DiT-based framework for generating HexPlane, enabling 4D occupancy generation. Training a DiT with token sequences naively generated from HexPlane may not achieve optimal quality, as it could overlook spatial and temporal relationships among tokens. Therefore, we introduce the Padded Rollout Operation (PRO), which reorganizes the six feature planes into a square feature map, providing an efficient way to model both spatial and temporal relationships within the token

sequence. Leveraging the DiT framework, DynamicCity seamlessly incorporates various conditions to guide the 4D generation process, enabling a wide range of applications including hexplane-conditional generation, trajectory-guided generation, command-driven scene generation, layout-conditioned generation, and dynamic scene inpainting.

Our contributions can be summarized as follows:

- • We propose DynamicCity, a high-quality, large-scale 4D occupancy generation framework consisting of a tailored VAE for HexPlane fitting and a DiT-based network for HexPlane generation, which supports various downstream applications.
- • In the VAE architecture, DynamicCity employs a novel Projection Module to benefit in encoding 4D scenes into compact HexPlanes, significantly improving HexPlane fitting quality. Following, an Expansion & Squeeze Strategy is introduced to decode the HexPlanes for reconstruction, which improves both fitting efficiency and accuracy.
- • Building on fitted HexPlanes, we design a Padded Rollout Operation to reorganize HexPlane features into a masked 2D square feature map, enabling compatibility with DiT training.
- • Extensive experimental results demonstrate that DynamicCity achieves significantly better 4D reconstruction and generation performance than previous SoTA methods across all evaluation metrics, including generation quality, training speed, and memory usage.

- 2 Related Work
- 3D Object Generation has been a central focus in machine learning, with diffusion models playing a significant role in generating realistic 3D structures. Many techniques utilize 2D diffusion mechanisms to synthesize

- 3D outputs, covering tasks like text-to-3D object generation [20], image-to-3D transformations [43], and

- 3D editing [31]. Meanwhile, recent methods bypass the reliance on 2D intermediaries by generating 3D outputs directly in three-dimensional space, utilizing explicit [1], implicit [19], triplane [44], and latent representations [30]. Although these methods demonstrate impressive 3D object generation, they primarily focus on small-scale, isolated objects rather than large-scale, scene-level generation [13, 46]. This limitation underscores the need for methods capable of generating complete 3D scenes with complex spatial relationships.

Urban Scene Generation extends the scope to larger, more complex environments. Earlier works used VQVAE [50] and GAN-based models [5, 23] to generate LiDAR scenes. However, recent advancements have shifted towards diffusion models [14, 22, 24, 27, 47, 50], which better handle the complexities of expansive outdoor scenes. For example, [17] utilize voxel grids to represent large-scale scenes but often face challenges with empty spaces like skies and fields. While some recent works incorporate temporal dynamics to extend single-frame generation to sequences [41, 49], they often lack the ability to fully capture the dynamic nature of 4D environments. Thus, these methods typically remain limited to short temporal horizons or struggle with realistic dynamic object modeling, highlighting the gap in generating high-fidelity 4D scenes.

- 4D Generation represents a leap forward, aiming to capture the temporal evolution of scenes. Prior works often leverage video diffusion models [4, 35] to generate dynamic sequences [36], with some extending to multi-view [33, 45] and single-image settings [32] to enhance 3D consistency. In the context of video-conditional generation, approaches such as [16, 28, 29] incorporate image priors for guiding generation processes. While these methods capture certain dynamic aspects, they lack the ability to generate long-term, high-resolution

- 4D scenes with versatile applications. Our method, DynamicCity, fills this gap by introducing a novel 4D generation framework that efficiently captures large-scale dynamic environments, supports diverse generation tasks (e.g., trajectory-guided [2], command-driven generation), and offers substantial improvements in scene fidelity and temporal modeling.

- 3 Preliminaries HexPlane [7, 11] is an explicit and structured representation designed for efficient modeling of dynamic

- 3D scenes, leveraging feature planes to encode spacetime data. A dynamic 3D scene is represented as six

4D Scene 𝐐

###### HexPlane

[Figure 48]

[Figure 49]

| | |
|---|---|
| | |
| | |

[Figure 50]

c(p)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Space-TimeSpace

PE(p)

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

3D Backbone Projection Head

[Figure 63]

[Figure 64]

[Figure 65]

###### …

[Figure 66]

Hadamard Product

[Figure 67]

…

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Class Probabilities

[Figure 76]

(a) Learning HexPlane as an Efficient 4D Scene Representation

[Figure 77]

Diffusion

| |[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>…|
|---|---|
| | |
| | |
| |4D Scene 𝐐0|

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

[Figure 93]

| | |
|---|---|

…

…

…

Decoder Decoder

[Figure 94]

[Figure 95]

[Figure 96]

DiT … DiT

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

Conditions

Denoising

- 4D Scene 𝐐t

(b) HexPlane Diffusion with DiT for 4D Scene Generation

- Figure 2 Pipeline of dynamic scene generation. Our DynamicCity framework consists of two key procedures: (a) Encoding HexPlane with an VAE architecture (cf. Sec. 4.1), and (b) 4D Scene Generation with HexPlane DiT (cf. Sec. 4.2).

2D feature planes, each aligned with one of the major planes in the 4D spacetime grid. These planes are represented as H = [Pxy,Pxz,Pyz,Ptx,Pty,Ptz], comprising a Spatial TriPlane [8] with Pxy, Pxz, and Pyz, and a Spatial-Time TriPlane with Ptx, Pty, and Ptz. To query the HexPlane at a point p = (t,x,y,z), features are extracted from the corresponding coordinates on each of the six planes and fused into a comprehensive representation. This fused feature vector is then passed through a lightweight network to predict scene attributes for p.

DiffusionTransformers (DiT) [26] are diffusion-based generative models using transformers to gradually convert Gaussian noise into data samples through denoising steps. The forward diffusion adds Gaussian noise over time, with a noised sample at step t given by xt = √αtx0 + √1 − αtϵ, ϵ ∼ N(0,I), where αt controls the noise schedule. The reverse diffusion, using a neural network ϵθ, aims to denoise xt to recover x0, expressed as: xt−1 = √1α

√1 − αtϵθ(xt,t)). New samples are generated by repeating this reverse process.

(xt −

t

- 4 Our Approach

DynamicCity strives to generate dynamic 3D occupancy with semantic information, which mainly consists of a VAE for 4D occupancy encoding using HexPlane [7, 11] (Sec. 4.1), and a DiT for HexPlane generation (Sec. 4.2). Given a 4D scene, i.e., a dynamic 3D occupancy sequence Q ∈ RT×X×Y ×Z×C, where T, X, Y , Z, and C denote the sequence length, height, width, depth, and channel size, respectively, the VAE first aims to encode an efficient 4D representation, HexPlane H = [Pxy,Pxz,Pyz,Ptx,Pty,Ptz], which is then decoded for reconstructing 4D scenes with semantics.

After obtaining HexPlane embeddings, DynamicCity leverages a DiT-based framework for 4D occupancy generation. Diverse conditions could be introduced into the generation process, facilitating a range of downstream applications (Sec. 4.3). The overview of the proposed DynamicCity pipeline is illustrated in Fig. 2.

[Figure 107]

[Figure 108]

Encoding HexPlane

Space

|𝓧𝑥𝑦𝑧|
|---|

| | | | | | |𝒫𝑥𝑦|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 109]

𝒉𝒛

[Figure 110]

| | |
|---|---|
| | |
| | |

[Figure 111]

|𝓧𝑡𝑥𝑦𝑧|
|---|

XYZ

[Figure 112]

[Figure 113]

| | | | | | |𝒫𝑥𝒛|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

𝒉𝒚

[Figure 118]

𝒉𝒕

|[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>…|[Figure 122]<br><br>|
|---|---|
| | |
|4D Scene 𝐐| |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |𝒫𝑦𝑧|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

XYZ Feature Volume

𝒉𝒙

[Figure 123]

[Figure 124]

#### …

Space-Time

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

𝒇𝜽

###### T

[Figure 125]

[Figure 126]

[Figure 127]

𝒉𝒚𝒛 𝒉𝒙𝒛

𝒫𝑡𝑥

#### …

[Figure 128]

[Figure 129]

[Figure 130]

| | | | | | |𝒫𝑡𝑦|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 131]

[Figure 132]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |𝒫𝒕𝒛|
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

𝒉𝒙𝒚

Projection Module

HexPlane ℋ

Feature Volume Sequence

Reconstruction Loss

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

|𝓧𝑡𝑥𝑦𝑧′|
|---|

[Figure 147]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

[Figure 148]

Expansion

Hadamard

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

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Product

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

T

𝒈𝝓 T

#### …

#### …

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Decoding HexPlane

Reconstructed

Expand ൛𝒫𝑥𝑦, 𝒫𝑥𝑧, 𝒫𝑦𝑧, 𝒫𝑡𝑥, 𝒫𝑡𝑦, 𝒫𝑡𝑧ൟ𝑡𝑥𝑦𝑧′

XYZ

- 4D Scene 𝐐′

- Figure3 VAEforencodingdynamicscenes. We use HexPlane H as the 4D representation. fθ and gϕ are convolution-based networks with downsampling and upsampling operations, respectively. h(·) denotes the projection network based on transformer modules.

- 4.1 VAE for 4D Occupancy

Encoding HexPlane. As shown in Fig. 3, the VAE could encode 4D occupancy Q as a HexPlane H. It first utilizes a shared 3D convolutional feature extractor fθ(·) to extract and downsample features from each occupancy frame, resulting in a feature volume sequence Xtxyz ∈ RT×X×Y ×Z×C.

To encode and compress Xtxyz into compact 2D feature maps of H, we propose a novel Projection Module with multiple projection networks h(·).

1

k×Dk2×···×Dkn×Dr1×Dr2×···×Drm×C as a lower-dimensional feature output Xout ∈ RD

To project a high-dimensional feature input Xin ∈ RD

1

k×Dk2×···×Dkn×C, the projection network hS

(·) first reshapes Xin into a 3-dimensional feature XS′kSr ∈ RS

r

k×Sr×C by grouping the dimensions into the two new dimensions, i.e., Sk the dimension that will be kept, and Sr the dimension that will be reduced, where Sk = Dk1 × Dk2 × ··· × Dkn, and Sr = Dr1 × Dr2 × ··· × Drm. Afterward, hS

(·) utilizes a transformer-based operation to project the reshaped feature XS′kSr to XS′′k ∈ RS

r

k×C, which is then reshaped to the expected lower-dimensional feature output Xout. Formally, the projection network is formulated as:

1 k×Dk2×···×Dkn}×C

1 k×Dk2×···×Dkn}×{Dr1×Dr2×···×Drm}×C

X{D

(X{D

in ) , (1) where their feature dimensions are added as the upscript for Xin and Xout, respectively.

out = hS

r

To construct the spatial feature planes Pxy, Pxz, and Pyz, the Projection Module first generates the XYZ Feature Volume Xxyz = ht(Xtxyz). Rather than directly access the heavy feature volume sequence Xtxyz, hz(·), hy(·), and hx(·) are applied to Xxyz for reducing the spatial dimensions of Xxyz along the z-axis, y-axis, and x-axis, respectively. The temporal feature planes Ptx,Pty, and Ptz are directly obtained from Xtxyz by simultaneously removing two spatial dimensions with hzy(·),hxz(·), and hxy(·), respectively.

Consequently, we could construct the HexPlane H based on the encoded six feature planes, including Pxy,Pxz,Pyz,Ptx,Pty, and Ptz.

###### Space Space-Time

###### Image Conditions

###### Numeric Conditions

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

HexPlane Layout

Trajectory Command Timestamp

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

###### MLP

Patchify Patchify

Embedding MLP

(a) HexPlane

[Figure 197]

[Figure 198]

### +

|XY|TY|YZ|
|---|---|---|
|XZ|TZ| |
|TX| | |

+

###### Cross-Attention Conditioning

###### Conditioning

Multi-Head Cross-Attention

Multi-Head Self-Attention

InputTokens

Pointwise Feedforward

LayerNorm

LayerNorm

LayerNorm

Scale,Shift

Scale,Shift

Scale

Scale

+

+

+

(b) Padded Rollout

Figure 5 Condition Injection for DiT

Figure 4 Padded Rollout

Decoding HexPlane. Based on the HexPlane H = [Pxy,Pxz,Pyz,Ptx,Pty,Ptz], we employ an Expansion & Squeeze Strategy (ESS), which could efficiently recover the feature volume sequence by decoding the feature planes in parallel for 4D occupancy reconstruction.

ESS first duplicates and expands each feature plane P to match the shape of Xtxyz, resulting in the list of six feature volume sequences: {XtxyzPxy ,XP

txyz,XtxyzPyz ,XP

txyz,XtxyzPty ,XP

txyz} . Afterward, ESS squeezes the corresponding six expanded feature volumes with the Hadamard Product:

xz

tx

tz

txyz,XtxyzPty ,XP

txyz,XtxyzPyz ,XP

{XtxyzPxy ,XP

Xtxyz′ =

txyz} . (2)

tz

tx

xz

Hadamard

Subsequently, the convolutional network gϕ(·) is employed to upsample the volumes for generating dense semantic predictions Q′:

Q′ = gϕ(Concat(Xtxyz′ ,PE(Pos(Xtxyz′ )))) , (3) where Concat(·) and PE(·) denote the concatenation and sinusoidal positional encoding, respectively. Pos(·) returns the 4D position p of each voxel within the 4D feature volume Xtxyz′ . Optimization. The VAE is trained with a combined loss LVAE, including a cross-entropy loss, a Lovász-softmax loss [3], and a Kullback-Leibler (KL) divergence loss:

LVAE = LCE(Q,Q′) + αLLov(Q,Q′) + βLKL(H,N(0,I)) , (4)

where LCE is the cross-entropy loss between the input Q and prediction Q′, LLov is the Lovász-softmax loss, and LKL represents the KL divergence between the latent representation H and the prior Gaussian distribution N(0,I). Note that the KL divergence is computed for each feature plane of H individually, and the term LKL refers to the combined divergence over all six planes.

- 4.2 Diffusion Transformer for HexPlane

After training the VAE, 4D semantic scenes can be embedded as HexPlane H = [Pxy,Pxz,Pyz,Ptx,Pty,Ptz]. Building upon H, we aim to leverage a DiT [26] model Dτ to generate novel HexPlane, which could be further decoded as novel 4D scenes (see Fig. 2(b)).

However, training a DiT using token sequences naively generated from each feature plane of HexPlane could not guarantee high generation quality, mainly due to the absence of modeling spatial and temporal relations among the tokens.

PaddedRolloutOperation. Given that the feature planes of HexPlane may share spatial or temporal dimensions, we employ the Padded Rollout Operation (PRO) to systematically arrange all six planes into a unified square feature map, incorporating zero paddings in the uncovered corner areas. As shown in Fig. 4, the dimension of the 2D square feature map is (dX

), which minimizes the area for padding, where dX, dZ, and dT represent the downsampling rates along the X, Z, and T axes, respectively. Subsequently, we follow DiT to first “patchify” the constructed 2D feature map, converting it into a sequence of N = ((dX

+ dZ

+ dT

X

Z

T

)/p)2 tokens, where p is the patch size, chosen so each token holds information from one feature plane. Following patchification, we apply the frequency-based positional embeddings to all tokens similar to DiT. Note that tokens corresponding to padding areas are excluded from the diffusion process. Consequently, the proposed PRO offers an efficient method for modeling spatial and temporal relationships within the token sequence.

+ dZ

+ dT

X

Z

T

Conditional Generation. DiT enables conditional generation through the use of Classifier-Free Guidance (CFG) [12]. To incorporate conditions into the generation process, we designed two branches for condition insertion (see Fig. 5). For any condition c, we use the adaLN-Zero technique from DiT, generating scale and shift parameters from c and injecting them before and after the attention and feed-forward layers. To handle the complexity of image-based conditions, we add a cross-attention block to better integrate the image condition into the DiT block.

- 4.3 Downstream Applications

Beyond unconditional 4D scene generation, we explore novel applications of DynamicCity through conditional generation and HexPlane manipulation.

First, we showcase versatile uses of image conditions in the conditional generation pipeline: 1) HexPlane: By autoregressively generating the HexPlane, we extend scene duration beyond temporal constraints. 2) Layout: We control vehicle placement and dynamics in 4D scenes using conditions learned from bird’s-eye view sketches.

To manage ego vehicle motion, we introduce two numerical conditioning methods: 3) Command: Controls general ego vehicle motion via instructions. 4) Trajectory: Enables fine-grained control through specific trajectory inputs.

Inspired by SemCity [17], we also manipulate the HexPlane during sampling to: 5) Inpaint: Edit 4D scenes by masking HexPlane regions and guiding sampling with the masked areas. For more applications and implementation details, kindly refer to Sec. 7.5 in the Appendix.

###### 5 Experiments

- 5.1 Experimental Details

Datasets. We train the proposed model on the 1Occ3D-Waymo, 2Occ3D-nuScenes, and 3CarlaSC datasets. The former two from Occ3D [40] are derived from Waymo [37] and nuScenes [6], where LiDAR point clouds have been completed and voxelized to form occupancy data. Each occupancy scene has a resolution of 200 × 200 × 16, covering a region centered on the ego vehicle, extending 40 meters in all directions and 6.4 meters vertically. The CarlaSC dataset [42] is a synthetic occupancy dataset, with a scene resolution of 128 × 128 × 8, covering a region 25.6 meters around the ego vehicle, with a height of 3 meters.

Implementation Details. Our experiments are conducted using eight NVIDIA A100-80G GPUs. The global batch size used for training the VAE is 8, while the global batch size for training the DiT is 128. Our latent HexPlane H is compressed to half the size of the input Q in each dimension, with the latent channels C = 16. The weights for the Lovász-softmax and KL terms are set to 1 and 0.005, respectively. The learning rate for the VAE is 10−3, while the learning rate for the DiT is 10−4.

###### Occ3D-Waymo CarlaSC

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

T=1 T=1

T=1

T=1

T=1

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

T=8 T=8

T=8

T=8

T=8

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

T=16

T=16

T=16 T=16

T=16

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 6 Dynamic Scene Generation Results. We provide unconditional generation scenes from the 1st, 8th, and 16th frames on Occ3D-Waymo (Left) and CarlaSC (Right), respectively. Kindly refer to the Appendix for complete sequential scenes and longer temporal modeling examples.

- Table 1 Comparisons of 4D Scene Reconstruction. We report the mIoU scores of OccSora [41] and our DynamicCity framework on the CarlaSC, Occ3D-Waymo, and Occ3D-nuScenes datasets, respectively, under different resolutions and sequence lengths. Symbol † denotes score reported in OccSora [41]. Other scores are reproduced using the official code.

OccSora Ours

Dataset #Classes Resolution #Frames

[41] (DynamicCity)

10 128×128×8 4 41.01% 79.61% (+38.6%) 10 128×128×8 8 39.91% 76.18% (+36.3%) 10 128×128×8 16 33.40% 74.22% (+40.8%) 10 128×128×8 32 28.91% 59.31% (+30.4%)

CarlaSC [42]

Occ3D-Waymo [40]

9 200×200×16 16 36.38% 68.18% (+31.8%)

11 200×200×16 16 13.70% 56.93% (+43.2%) 11 200×200×16 32 13.51% 42.60% (+29.1%) 17 200×200×16 32 13.41% 40.79% (+27.3%) 17 200×200×16 32 27.40%† 40.79% (+13.4%)

Occ3D-nuScenes [40]

Evaluation Metrics. The mean intersection over union (mIoU) metric is used to evaluate the reconstruction results of VAE. For DiT, Inception Score, FID, KID, Precision, and Recall are calculated for evaluation. Specifically, we follow prior work [17, 41] by rendering 3D scenes into 2D images and utilizing conventional 2D evaluation pipelines for assessment. Additionally, we train the 3D Encoder to directly extract features from the 3D data and calculate the metrics. For more details, kindly refer to Sec. 7.2 in the Appendix.

- Table 2 Comparisons of 4D Scene Generation. We report the Inception Score (IS), Fréchet Inception Distance (FID), Kernel Inception Distance (KID), and the Precision (P) and Recall (R) rates of OccSora [41] and our DynamicCity framework on the CarlaSC and Occ3D-Waymo datasets, respectively, in both the 2D and 3D spaces.

Dataset Method #Frames

|Metric2D<br><br>|Metric3D|
|---|---|
|IS2D↑ FID2D↓ KID2D ↓ P2D↑ R2D↑|IS3D↑ FID3D↓ KID3D↓ P3D↑ R3D↑|

CarlaSC [42]

OccSora

16

1.030 28.55 0.008 0.224 0.010 2.257 1559 52.72 0.380 0.151

Ours 1.040 12.94 0.002 0.307 0.018 2.331 354.2 19.10 0.460 0.170 Occ3D-Waymo [40]

OccSora

16

1.005 42.53 0.049 0.654 0.004 3.129 3140 12.20 0.384 0.001 Ours 1.010 36.73 0.001 0.705 0.015 3.206 1806 77.71 0.494 0.026

- Table 3 Ablation Study on VAE Network Structures. We report mIoU, training time (seconds-per-iteration), and trainingtime memory consumption (VRAM) of different Encoder and Decoder configurations on CarlaSC and Occ3D-Waymo. Note that “ESS” denotes “Expansion & Squeeze”. The best and second-best values are in bold and underlined.

|CarlaSC<br><br>|Occ3D-Waymo|
|---|---|
|mIoU↑ Time (s)↓ VRAM (G)↓<br><br>|mIoU↑ Time (s)↓ VRAM (G)↓|

Encoder Decoder

Average Pooling Query 60.97% 0.236 12.46 49.37% 1.563 69.66 Average Pooling ESS 68.02% 0.143 4.27 55.72% 0.758 20.31

Projection Query 68.73% 0.292 13.59 61.93% 2.128 73.15 Projection ESS 74.22% 0.205 5.92 62.57% 1.316 25.92

- 5.2 4D Scene Reconstruction & Generation

Reconstruction. To evaluate the effectiveness of the proposed VAE in encoding the 4D occupancy sequence, we compare it with OccSora [41] using the CarlaSC, Occ3D-Waymo, and Occ3D-nuScenes datasets. As shown in Tab. 1, DynamicCity outperforms OccSora on these datasets, achieving mIoU improvements of 38.6%, 31.8%, and 43.2%, respectively, when the input number of frames is 16. These results highlight the superior performance of the proposed VAE.

Generation. To demonstrate the effectiveness of DynamicCity in 4D scene generation, we compare the generation results with OccSora [41] on the Occ3D-Waymo and CarlaSC datasets. As shown in Tab. 2, the proposed method outperforms OccSora in terms of perceptual metrics in both 2D and 3D spaces. These results show that our model excels in both generation quality and diversity. Fig. 6 and Fig. 16 show the 4D scene generation results, demonstrating that our model is capable of generating large dynamic scenes in both real-world and synthetic datasets. Our model not only exhibits the ability to generate moving scenes with static semantics shifting as a whole, but it is also capable of generating dynamic elements such as vehicles and pedestrians.

Applications. Fig. 7 presents the results of our downstream applications. In tasks that involve inserting conditions into the DiT, such as command-conditional generation, trajectory-conditional generation, and layout-conditional generation, our model demonstrates the ability to generate reasonable scenes and dynamic elements while following the prompt to a certain extent. Additionally, the inpainting method proves that our HexPlane has explicit spatial meaning, enabling direct modifications within the scene by editing the HexPlane during inference.

- 5.3 Ablation Study We conduct ablation studies to demonstrate the effectiveness of the components of DynamicCity.

VAE. The effectiveness of the VAE is driven by two key innovations: Projection Module and Expansion & Squeeze Strategy (ESS). As shown in Tab. 3, the proposed Projection Module substantially improves HexPlane fitting performance, delivering up to a 12.56% increase in mIoU compared to traditional averaging operations. Additionally, compared to querying each point individually, ESS enhances HexPlane fitting quality with up to

###### Command-Driven Scene Generation

Dynamic Object Inpainting

###### Turn Right

###### Static

###### Forward

###### Turn Left

###### Before After

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

T=1 T=1

T=1

T=1

T=1

T=1

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

T=8

T=8

T=8

T=8

T=8 T=8

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

T=16

T=16

T=16

T=16

T=16 T=16

Trajectory-Guided Generation

###### Layout-Conditioned Scene Generation

[Figure 232]

[Figure 233]

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

T=1

T=1 T=1

T=1 T=1

T=1

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

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

T=8

T=8

T=8 T=8

T=8 T=8

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

T=16

T=16

T=16 T=16

T=16 T=16

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 7 Dynamic Scene Generation Applications. We demonstrate the capability of our model on a diverse set of downstream tasks. We show the 1st, 8th, and 16th frames for simplicity. Kindly refer to the Appendix for complete sequential scenes and longer temporal modeling examples.

a 7.05% mIoU improvement, significantly boosts training speed by up to 2.06x, and reduces memory usage by a substantial 70.84%.

HexPlane Dimensions. The dimensions of HexPlane have a direct impact on both training efficiency and reconstruction quality. Table 4 provides a comparison of various downsample rates applied to the original HexPlane dimensions, which are 16 × 128 × 128 × 8 for CarlaSC and 16 × 200 × 200 × 16 for Occ3D-Waymo. As the downsampling rates increase, both the compression rate and training efficiency improve significantly, but the reconstruction quality, measured by mIoU, decreases. To achieve the optimal balance between training

- Table 4 Ablation Study on HexPlane Downsampling (D.S.) Rates. We report the compression ratios (C.R.), mIoU scores, training speed (seconds per iteration), and training-time memory consumption on CarlaSC and Occ3D-Waymo. The best and second-best values are in bold and underlined.

|D.S. Rates|CarlaSC<br><br>|Occ3D-Waymo|
|---|---|---|
|dT dX dY dZ|C.R.↑ mIoU↑ Time (s)↓ VRAM (G)↓<br><br>|C.R.↑ mIoU↑ Time (s)↓ VRAM (G)↓|

- 1 1 1 1 5.78% 84.67% 1.149 21.63 Out-of-Memory >80

- 1 2 2 1 17.96% 76.05% 0.289 8.49 38.42% 63.30% 1.852 32.82 2 2 2 2 23.14% 74.22% 0.205 5.92 48.25% 62.37% 0.935 24.9 2 4 4 2 71.86% 65.15% 0.199 4.00 153.69% 58.13% 0.877 22.30

- Table 5 Ablation Study on Organizing HexPlane as Image Tokens. We report the Inception Score (IS), Fréchet Inception Distance (FID), Kernel Inception Distance (KID), and the Precision (P) and Recall (R) rates on CarlaSC. The best values are highlighted in bold.

Method

|Metric2D<br><br>|Metric3D|
|---|---|
|IS2D↑ FID2D↓ KID2D ↓ P2D↑ R2D↑|IS3D↑ FID3D↓ KID3D↓ P3D↑ R3D↑|

Direct Unfold 2.496 205.0 0.248 0.000 0.000 2.269 9110 723.7 0.173 0.043 Vertical Concatenation 2.476 12.79 0.003 0.191 0.042 2.305 623.2 26.67 0.424 0.159 Padded Rollout 2.498 10.96 0.002 0.238 0.066 2.331 354.2 19.10 0.460 0.170

efficiency and reconstruction quality, we select a downsampling rate of dT = dX = dY = dZ = 2.

Padded Rollout Operation. We compare the Padded Rollout Operation with different strategies for obtaining image tokens: 1) Direct Unfold: directly unfolding the six planes into patches and concatenating them; 2) Vertical Concat: vertically concatenating the six planes without aligning dimensions during the rollout process.

- As shown in Tab. 5, Padded Rollout Operation (PRO) efficiently models spatial and temporal relationships in the token sequence, achieving optimal generation quality.

- 6 Conclusion

We present DynamicCity, a framework for high-quality 4D occupancy generation that captures the temporal dynamics of real-world environments. Our method introduces HexPlane, a compact 4D representation generated using a VAE with a Projection Module, alongside an Expansion & Squeeze Strategy to enhance reconstruction efficiency and accuracy. Additionally, our Masked Rollout Operation reorganizes HexPlane features for DiT-based diffusion, enabling versatile 4D scene generation. Extensive experiments demonstrate that DynamicCity surpasses state-of-the-art methods in both reconstruction and generation, offering significant improvements in quality, training speed, and memory efficiency. DynamicCity paves the way for future research in dynamic scene generation.

Appendix

In this appendix, we supplement the following materials to support the findings and conclusions drawn in the main body of this paper.

###### 7 Additional Implementation Details 13

- 7.1 Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 7.2 DiT Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 7.3 Model Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 7.4 Classifier-Free Guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 7.5 Downstream Applications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

###### 8 Additional Quantitative Results 18

- 8.1 Per-Class Generation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 8.2 Occupancy Forecasting Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 8.3 User Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 8.4 Model Stats . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 8.5 Comparisons with SemCity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

###### 9 Additional Qualitative Results 21

- 9.1 Unconditional Dynamic Scene Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 9.2 HexPlane-Guided Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 9.3 Layout-Guided Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 9.4 Command- & Trajectory-Guided Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 9.5 Dynamic Inpainting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- 9.6 Comparisons with OccSora . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- 9.7 Dynamic Outpainting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- 9.8 Single Frame Occupancy Conditional Generation . . . . . . . . . . . . . . . . . . . . . . . . . 30

###### 10 Potential Societal Impact & Limitations 31

- 10.1 Societal Impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- 10.2 Broader Impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- 10.3 Known Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

###### 11 Public Resources Used 32

- 11.1 Public Datasets Used . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- 11.2 Public Implementations Used . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

###### 7 Additional Implementation Details

In this section, we provide additional implementation details to assist in reproducing this work. Specifically, we elaborate on the details of the datasets, DiT evaluation metrics, the specifics of our generation models, and discussions on the downstream applications.

- 7.1 Datasets

Our experiments primarily utilize two datasets: Occ3D-Waymo [40] and CarlaSC [42]. Additionally, we also evaluate our VAE on Occ3D-nuScenes [40].

The Occ3D-Waymo dataset is derived from real-world Waymo Open Dataset [37] data, where occupancy sequences are obtained through multi-frame fusion and voxelization processes. Similarly, Occ3D-nuScenes is generated from the real-world nuScenes [6] dataset using the same fusion and voxelization operations. On the other hand, the CarlaSC dataset is generated from simulated scenes and sensor data, yielding occupancy sequences.

Using these different datasets demonstrates the effectiveness of our method on both real-world and synthetic data. To ensure consistency in the experimental setup, we select 11 commonly used semantic categories and map the original categories from both datasets to these 11 categories. The detailed semantic label mappings are provided in Tab. 6.

- • Occ3D-Waymo. This dataset contains 798 training scenes, with each scene lasting approximately 20 seconds and sampled at a frequency of 10 Hz. This dataset includes 15 semantic categories. We use volumes with a resolution of 200 × 200 × 16 from this dataset.
- • CarlaSC. This dataset contains 6 training scenes, each duplicated into Light, Medium, and Heavy based on traffic density. Each scene lasts approximately 180 seconds and is sampled at a frequency of 10 Hz. This dataset contains 22 semantic categories, and the scene resolution is 128 × 128 × 8.
- • Occ3D-nuScenes. This dataset contains 600 scenes, with each scene lasting approximately 20 seconds and sampled at a frequency of 2 Hz. Compared to Occ3D-Waymo and CarlaSC, Occ3D-nuScenes has fewer total frames and more variation between scenes. This dataset includes 17 semantic categories, with a resolution of 200 × 200 × 16.

- Table 6 Summary of Semantic Label Mappings. We unify the semantic classes between CarlaSC [42], Occ3D-Waymo [40], and Occ3D-nuScenes [40] datasets for semantic scene generation.

Class CarlaSC Occ3D-Waymo Occ3D-nuScenes

- ■ Building Building Building Manmade

- ■ Barrier Barrier, Wall, Guardrail - Barrier

- ■ Other Other, Sky, Bridge, Rail track, Static, Dynamic,

Water

General Object General Object

- ■ Pedestrian Pedestrian Pedestrian Pedestrian

- ■ Pole Pole, Traffic sign, Traffic

light

Sign, Traffic light, Pole, Construction Cone

Traffic cone

- ■ Road Road, Roadlines Road Drivable surface

- ■ Ground Ground, Terrain - Other flat, Terrain

- ■ Sidewalk Sidewalk Sidewalk Sidewalk

- ■ Vegetation Vegetation Vegetation, Tree trunk Vegetation

- ■ Vehicle Vehicle Vehicle Bus, Car, Construction

vehicle, Trailer, Truck

- ■ Bicycle - Bicyclist, Bicycle,

Bicycle, Motorcycle

Motorcycle

- 7.2 DiT Evaluation Metrics

Inception Score (IS). This metric evaluates the quality and diversity of generated samples using a pre-trained Inception model as follows:

IS = exp EQ∼p

[DKL(p(y|Q) ∥ p(y))] , (5)

g

where pg represents the distribution of generated samples. p(y|Q) is the conditional label distribution given by the Inception model for a generated sample Q. p(y) = p(y|Q)pg(Q)dQ is the marginal distribution over all generated samples. DKL(p(y|Q) ∥ p(y)) is the Kullback-Leibler divergence, defined as follows:

p(yi|Q) p(yi)

. (6)

DKL(p(y|Q) ∥ p(y)) =

p(yi|Q)log

i

Fréchet Inception Distance (FID). This metric measures the distance between the feature distributions of real and generated samples:

FID = ∥µr − µg∥2 + Tr Σr + Σg − 2(ΣrΣg)1/2 , (7)

where µr and Σr are the mean and covariance matrix of features from real samples. µg and Σg are the mean and covariance matrix of features from generated samples. Tr denotes the trace of a matrix.

Kernel Inception Distance (KID). This metric uses the squared Maximum Mean Discrepancy (MMD) with a polynomial kernel as follows:

KID = MMD2(ϕ(Qr),ϕ(Qg)) , (8)

where ϕ(Qr) and ϕ(Qg) represent the features of real and generated samples extracted from the Inception model.

MMD with a polynomial kernel k(x,y) = (x⊤y + c)d is calculated as follows:

1 m(m − 1) i̸=j

1 n(n − 1) i̸=j

2 mn i,j

MMD2(X,Y ) =

k(xi,yj) , (9)

k(yi,yj) −

k(xi,xj) +

where X = {Q1,...,Qm} and Y = {y1,...,yn} are sets of features from real and generated samples. Precision. This metric measures the fraction of generated samples that lie within the real data distribution as follows:

N

1 N

I (fg − µr)⊤Σ−r 1(fg − µr) ≤ χ2 , (10)

Precision =

i=1

where fg is a generated sample in the feature space. µr and Σr are the mean and covariance of the real data distribution. I(·) is the indicator function. χ2 is a threshold based on the chi-squared distribution.

Recall. This metric measures the fraction of real samples that lie within the generated data distribution as follows:

M

1 M

I (fr − µg)⊤Σ−g 1(fr − µg) ≤ χ2 , (11)

Recall =

j=1

where: fr is a real sample in the feature space. µg and Σg are the mean and covariance of the generated data distribution. I(·) is the indicator function. χ2 is a threshold based on the chi-squared distribution.

- 2D Evaluations. We render 3D scenes as 2D images for 2D evaluations. To ensure fair comparisons, we use the same semantic colormap and camera settings across all experiments. Fig. 8 shows an example of a rendered 2D semantic colormap. We use an InceptionV3 [38] model to compute the Inception Score (IS), Fréchet Inception Distance (FID), and Kernel Inception Distance (KID) scores, while Precision and Recall are computed using a VGG-16 [34] model. We train both 2D backbones using semantic colormap data.
- 3D Evaluations. For 3D data, we train a MinkowskiUNet [9] as an autoencoder. We adopt the latest implementation from SPVNAS [39], which supports optimized sparse convolution operations. The features were extracted by applying average pooling to the output of the final downsampling block.

[Figure 286]

- Figure 8 Example of 2D Evaluation Rendering.

- 7.3 Model Details

General Training Details. We implement both the VAE and DiT models using PyTorch [25]. We utilize PyTorch’s mixed precision and replace all attention mechanisms with FlashAttention [10] to accelerate training and reduce memory usage. AdamW is used as the optimizer for all models.

We train the VAE with a learning rate of 10−3, running for 20 epochs on Occ3D-Waymo and 100 epochs on CarlaSC. The DiT is trained with a learning rate of 10−4, and the EMA rate for DiT is set to 0.9999.

VAE. Our encoder projects the 4D input Q into a HexPlane, where each dimension is a compressed version of the original 4D input. First, a 3D CNN is applied to each frame for feature extraction and downsampling, with dimensionality reduction applied only to the spatial dimensions (X, Y , Z). Next, the Projection Module projects the 4D features into the HexPlane. Each small transformer within the Projection Module consists of two layers, and the attention mechanism has two heads. Each head has a dimensionality of 16, with a dropout rate of 0.1. Afterward, we further downsample the T dimension to half of its original size.

During decoding, we first use three small transpose CNNs to restore the T dimension, then use an ESS module to restore the 4D features. Finally, we apply a 3D CNN to recover the spatial dimensions and generate point-wise predictions.

Diffusion. We set the patch size p to 2 for our DiT models. The Waymo DiT model has a hidden size of 768, 18 DiT blocks, and 12 attention heads. The CarlaSC DiT model has a hidden size of 384, 16 DiT blocks, and

- 8 attention heads.

Discussion on VAE Structure Improvements. Some prior work utilizes sparse 3D structures to enhance the efficiency of their backbones. For example, XCube [30] employs a fully sparse 3D encoder, significantly improving model efficiency. Similarly, our VAE could potentially improve the 3D convolutional feature

extractor fθ(·) by adopting sparse convolution. However, using sparse convolution offers only limited efficiency gains, as convolution accounts for only a small portion of our VAE. Moreover, like XCube, we cannot apply sparse convolution in our decoder. In the future, we plan to explore more efficient operations to further optimize our 3D backbone.

- 7.4 Classifier-Free Guidance

Classifier-Free Guidance (CFG) [12] could improve the performance of conditional generative models without relying on an external classifier. Specifically, during training, the model simultaneously learns both conditional generation p(x|c) and unconditional generation p(x), and guidance during sampling is provided by the following equation:

xˆt = (1 + w) · xˆt(c) − w · xˆt(∅) , (12)

where xˆt(c) is the result conditioned on c, xˆt(∅) is the unconditioned result, and w is a weight parameter controlling the strength of the conditional guidance. By adjusting w, an appropriate balance between the accuracy and diversity of the generated scenes can be achieved.

- 7.5 Downstream Applications

This section provides a comprehensive explanation of five tasks to demonstrate the capability of our 4D scene generation model across various scenarios.

HexPlane. Since our model is based on Latent Diffusion Models, it is inherently constrained to generate results that match the latent space dimensions, limiting the temporal length of unconditionally generated sequences. We argue that a robust 4D generation model should not be restricted to producing only short sequences. Instead of increasing latent space size, we leverage CFG to generate sequences in an auto-regressive manner. By conditioning each new 4D sequence on the previous one, we sequentially extend the temporal dimension. This iterative process significantly extends sequence length, enabling long-term generation, and allows conditioning on any real-world 4D scene to predict the next sequence using the DiT model. Theoretically, our HexPlane conditional generation can model sequence of arbitrary length, but less stable generation may occur when generating very long sequences.

We condition our DiT by using the HexPlane from T frames earlier. For any condition HexPlane, we apply patch embedding and positional encoding operations to obtain condition tokens. These tokens, combined with other conditions, are fed into the adaLN-Zero and Cross-Attention branches to influence the main branch.

Layout. To control object placement in the scene, we train a model capable of generating vehicle dynamics based on a bird’s-eye view sketch. We apply semantic filtering to the bird’s-eye view of the input scene, marking regions with vehicles as 1 and regions without vehicles as 0. Pooling this binary image provides layout information as a T × H × W tensor from the bird’s-eye perspective. The layout is padded to match the size of the HexPlane, ensuring that the positional encoding of the bird’s-eye layout aligns with the XY plane. DiT learns the correspondence between the layout and vehicle semantics using the same conditional injection method applied to the HexPlane.

Command. While we have developed effective methods to control the HexPlane in both temporal and spatial dimensions, a critical aspect of 4D autonomous driving scenarios is the motion of the ego vehicle. To address this, we define four commands: STATIC, FORWARD, TURN LEFT, and TURN RIGHT, and annotate our training data by analyzing ego vehicle poses. During training, we follow the traditional DiT approach of injecting class labels, where the commands are embedded and fed into the model via adaLN-Zero.

Trajectory. For more fine-grained control of the ego vehicle’s motion, we extend the command-based conditioning into a trajectory condition branch. For any 4D scene, the XY coordinates of the trajectory traj ∈ RT×2 are passed through an MLP and injected into the adaLN-Zero branch.

Inpaint. We demonstrate that our model can handle versatile applications by training a conditional DiT for the previous tasks. Extending our exploration of downstream applications, and inspired by [17], we leverage the 2D structure of our latent space and the explicit modeling of each dimension to highlight our model’s ability to perform inpainting on 4D scenes. During DiT sampling, we define a 2D mask m ∈ RX×Y on the XY plane, which is extended across all dimensions to mask specific regions of the HexPlane.

- At each step of the diffusion process, we apply noise to the input Hin and update the HexPlane using the following formula:

Ht = m ⊙ Ht + (1 − m) ⊙ Htin , (13) where ⊙ denotes the element-wise product. This process inpaints the masked regions while preserving the unmasked areas of the scene, enabling partial scene modification, such as turning an empty street into one with heavy traffic.

Outpaint. Outpainting extends the spatial dimensions of a given occupancy sequence. We use the same procedure for outpainting as we do for inpainting. Specifically, we mask half of the scene, shift the latent representation, and apply the inpainting process. Consequently, we could obtain a larger scene with consistent dynamics.

Single frame occupancy. We apply the same procedure for single-frame occupancy conditional generation as for HexPlane conditional generation. Specifically, we preprocess the data, encode the first frame of each training sequence as a HexPlane, and fine-tune our HexPlane generation model for single-frame conditional generation.

###### 8 Additional Quantitative Results

In this section, we present additional quantitative results to demonstrate the effectiveness of our VAE in accurately reconstructing 4D scenes.

- 8.1 Per-Class Generation Results

We include the class-wise IoU scores of OccSora [41] and our proposed DynamicCity framework on CarlaSC [42]. As shown in Tab. 7, our results demonstrate higher IoU across all classes, indicating that our VAE reconstruction achieves minimal information loss. Additionally, our model does not exhibit significantly low IoU for any specific class, proving its ability to effectively handle class imbalance.

Table 7 Comparisons of Per-Class IoU Scores. We compared the performance of OccSora [41], and our DynamicCity framework on CarlaSC [42] across 10 semantic classes. The scene resolution is 128 × 128 × 8. The sequence lengths are 4, 8, 16, and 32, respectively.

Method mIoU

Building

Barrier

Other

Pedestrian

Pole

Road

Ground

Sidewalk

Vegetation

Vehicle

Resolution: 128 × 128 × 8 Sequence Length: 4 OccSora 41.009 38.861 10.616 6.637 19.191 21.825 93.910 61.357 86.671 15.685 55.340

Ours 79.604 76.364 31.354 68.898 93.436 87.962 98.617 87.014 95.129 68.700 88.569 Improv. 38.595 37.503 20.738 62.261 74.245 66.137 4.707 25.657 8.458 53.015 33.229 Resolution: 128 × 128 × 8 Sequence Length: 8 OccSora 39.910 33.001 3.260 5.659 19.224 19.357 93.038 57.335 85.551 30.899 51.776

Ours 76.181 70.874 50.025 52.433 87.958 85.866 97.513 83.074 93.944 58.626 81.498 Improv. 36.271 37.873 46.765 46.774 68.734 66.509 4.475 25.739 8.393 27.727 29.722 Resolution: 128 × 128 × 8 Sequence Length: 16 OccSora 33.404 19.264 2.205 3.454 11.781 9.165 92.054 50.077 82.594 18.078 45.363

Ours 74.223 66.852 51.901 49.844 79.410 82.369 96.937 84.484 94.082 58.217 78.134 Improv. 40.819 47.588 49.696 46.390 67.629 73.204 4.883 34.407 11.488 40.139 32.771 Resolution: 128 × 128 × 8 Sequence Length: 32 OccSora 28.911 16.565 1.413 0.944 6.200 4.150 91.466 43.399 78.614 11.007 35.353

Ours 59.308 52.036 25.521 29.382 56.811 57.876 94.792 78.390 89.955 46.080 62.234 Improv. 30.397 35.471 24.108 28.438 50.611 53.726 3.326 34.991 11.341 35.073 26.881

- 8.2 Occupancy Forecasting Results

We train our HexPlane conditional generation pipeline on Occ3D-nuScenes [40] as an occupancy forecasting model. We set T = 4 to ensure the model receives a HexPlane with a context length of 2 seconds, aligning with OccWorld [48], and generates the next 2 seconds for evaluation. As shown in Tab. 8, our model outperforms OccWorld on most metrics.

- 8.3 User Study

We conduct a user study comparing OccSora [41] with our proposed DynamicCity. The study includes 20 samples, with 10 from each method. Participants rate each sample on four metrics: 1) overall quality, 2) time consistency, 3) background quality, and 4) foreground quality. Ratings range from 1 to 5, with 5 being the highest. We collect results from 42 volunteers and get 840 valid scores in total, as shown in Tab. 9. Our method receives better user feedback across all metrics.

- Table 8 4D Occupancy Forecasting Performance. We compare the performance of OccWorld [48] and our proposed DynamicCity framework on Occ3D-nuScenes [40].

Method

mIoU IoU T = 0 T = 1 T = 2 T = 0 T = 1 T = 2

OccWorld-O 66.38 25.78 15.14 62.29 34.63 25.07 Ours 80.52 26.18 16.94 67.64 34.12 25.82

- Table 9 User Study Results. We conduct user study comparing OccSora [41] and DynamicCity. The rating is of scale 1-5, the higher the better.

Method Overall Quality Time Consistency Background Quality Foreground Quality

OccSora 2.21 2.05 2.17 2.11

Ours 4.03 4.02 3.95 4.04

- 8.4 Model Stats

We compare the training speed, inference speed, training VRAM, and inference VRAM of OccSora [41] and DynamicCity. The results are presented in Tab. 10, Tab. 11, Tab. 12, and Tab. 13. While some of our models may be slightly slower and consume more memory compared to OccSora, they achieve significantly better performance. We also compare the total model size of OccSora and our model in Tab. 14. Our model is significantly smaller than OccSora while achieving superior performance.

Table 10 VAE Model Statistics on CarlaSC Dataset [42]. We compare the training time, inference time, training VRAM, inference VRAM of OccSora [41] and our DynamicCity.

Method Training Time (s)↓ Inference Time (s)↓ Training VRAM (G)↓ Inference VRAM (G)↓

OccSora 0.36 0.21 4.86 3.25 Ours 0.21 0.41 5.92 1.43

- 8.5 Comparisons with SemCity

###### We compare the generation quality of SemCity [17] and our DynamicCity in Tab. 15. Despite using a more compact latent representation and generating dynamic scenes, our model outperforms SemCity on most metrics.

###### Table 11 VAE Model Statistics on Occ3D-Waymo Dataset [40]. We compare the training time, inference time, training VRAM, inference VRAM of OccSora [41] and our DynamicCity.

Method Training Time (s)↓ Inference Time (s)↓ Training VRAM (G)↓ Inference VRAM (G)↓

OccSora 0.63 0.21 10.05 3.93 Ours 0.94 0.54 24.90 4.62

###### Table 12 DiT Model Statistics on CarlaSC Dataset [42]. We compare the training time, inference time, training VRAM, inference VRAM of OccSora [41] and our DynamicCity.

Method Training Time (s)↓ Inference Time (s)↓ Training VRAM (G)↓ Inference VRAM (G)↓

OccSora 0.19 6.10 1.50 1.15 Ours 0.19 3.91 10.22 1.28

###### Table 13 DiT Model Statistics on Occ3D-Waymo Dataset [40]. We compare the training time, inference time, training VRAM, inference VRAM of OccSora [41] and our DynamicCity.

Method Training Time (s)↓ Inference Time (s)↓ Training VRAM (G)↓ Inference VRAM (G)↓

OccSora 0.35 6.09 15.16 1.15 Ours 0.45 4.41 22.33 1.29

###### Table 14 Model Size. We compare the total model size of OccSora [41] and our DynamicCity. Dataset Method Model Size (M)

OccSora 169.1

CarlaSC

- Ours 44.7

Occ3D-Waymo

OccSora 174.2

- Ours 45.6

###### Table 15 Comparisons of 2D and 3D Evaluation Metrics. We report the Inception Score (IS), Fréchet Inception Distance (FID), Kernel Inception Distance (KID), and the Precision (P) and Recall (R) rates for SemCity [17] and our method in both the 2D and 3D spaces.

|Metric2D|Metric3D|
|---|---|
|IS↑ FID↓ KID↓ P↑ R↑|IS↑ FID↓ KID↓ P↑ R↑|

Method

###### SemCity 1.039 35.40 0.010 0.213 0.058 2.288 1113 53.948 0.253 0.787 Ours 1.040 12.94 0.002 0.307 0.018 2.331 427.5 27.869 0.460 0.170

###### 9 Additional Qualitative Results

In this section, we provide additional qualitative results on the Occ3D-Waymo [40] and CarlaSC [42] datasets to demonstrate the effectiveness of our approach.

- 9.1 Unconditional Dynamic Scene Generation

First, we present full unconditional generation results in Fig. 9 and 10. These results demonstrate that our generated scenes are of high quality, realistic, and contain significant detail, capturing both the overall scene dynamics and the movement of objects within the scenes.

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

T=2 T=3 T=4

T=1

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

- T=8 T=7 T=6 T=5

- T=9 T=10 T=11 T=12

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

T=16 T=15 T=14 T=13

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 9 Unconditional Dynamic Scene Generation Results. We provide qualitative examples of a total of 16 consectutive frames generated by DynamicCity on Occ3D-Waymo [40]. Best viewed in colors and zoomed-in for additional details.

Waymo unconditional1 scene 18

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

T=1 T=2 T=3 T=4

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

- T=8 T=7 T=6 T=5

- T=9 T=10 T=11 T=12

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

T=16 T=15 T=14 T=13

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 10 Unconditional Dynamic Scene Generation Results. We provide qualitative examples of a total of 16 consectutive frames generated by DynamicCity on CarlaSC [42]. Best viewed in colors and zoomed-in for additional details.

- 9.2 HexPlane-Guided Generation

We show results for our HexPlane conditional generation in Fig. 11. Although the sequences are generated in groups of 16 due to the settings of our VAE, we successfully generate a long sequence by conditioning on the previous one. The result contains 64 frames, comprising four sequences, and depicts a T-intersection with many cars parked along the roadside. This result demonstrates strong temporal consistency across sequences, proving that our framework can effectively predict the next sequence based on the current one.

1 2 3 4 5 6 7 8

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

16 15 14 13 12 11 10 9

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

17 18 19 20 21 22 23 24

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

32 31 30 29 28 27 26 25

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

33 34 35 36 37 38 39 40

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

48 47 46 45 44 43 42 41

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

49 50 51 52 53 54 55 56

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

64

63 62 61 60 59 58 57

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

###### Figure 11 HexPlane-Guided Generation Results. We provide qualitative examples of a total of 64 consectutive frames generated by DynamicCity on Occ3D-Waymo [40]. Best viewed in colors and zoomed-in for additional details.

- 9.3 Layout-Guided Generation

The layout conditional generation result is presented in Fig. 12. First, we observe that the layout closely matches the semantic positions in the generated result. Additionally, as the layout changes, the positions of the vehicles in the scene also change accordingly, demonstrating that our model effectively captures the condition and influences both the overall scene layout and vehicle placement.

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

T=1 T=2 T=3

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

- T=6 T=5 T=4

- T=7

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

T=8 T=9

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

- T=12 T=11 T=10

- T=13 T=14 T=15

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 12 Layout-Guided Generation Results. We provide qualitative examples of a total of 16 consectutive frames generated by DynamicCity on Occ3D-Waymo [40]. Best viewed in colors and zoomed-in for additional details.

- 9.4 Command- & Trajectory-Guided Generation

We present command conditional generation in Fig. 13 and trajectory conditional generation in Fig. 14. These results show that when we input a command, such as "right turn," or a sequence of XY-plane coordinates, our model can effectively control the motion of the ego vehicle and the relative motion of the entire scene based on these movement trends.

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

Turn Right Turn Right Turn Right Turn Right

T=1 T=2 T=3 T=4

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

Turn Right Turn Right Turn Right Turn Right

- T=8 T=7 T=6 T=5

- T=9 T=10 T=11 T=12

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

Turn Right Turn Right Turn Right Turn Right

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

Turn Right Turn Right Turn Right Turn Right

T=16 T=15 T=14 T=13

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 13 Command-Guided Scene Generation Results. We provide qualitative examples of a total of 16 consectutive frames generated under the command RIGHT by DynamicCity on CarlaSC [42]. Best viewed in colors and zoomed-in for additional details.

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

T=1 T=2 T=3 T=4

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

- T=8 T=7 T=6 T=5

- T=9 T=10 T=11 T=12

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

T=16 T=15 T=14 T=13

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 14 Trajectory-Guided Scene Generation Results. We provide qualitative examples of a total of 16 consectutive frames generated by DynamicCity on CarlaSC [42]. Best viewed in colors and zoomed-in for additional details.

- 9.5 Dynamic Inpainting

We present the full inpainting results in Fig. 15. The results show that our model successfully regenerates the inpainted regions while ensuring that the areas outside the inpainted regions remain consistent with the original scene. Furthermore, the inpainted areas seamlessly blend into the original scene, exhibiting realistic placement and dynamics.

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

Before After Before After Before After

T=1 T=2 T=3

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

Before After Before After Before After

- T=6 T=5 T=4

- T=7 T=8 T=9

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

Before After Before After Before After

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

Before After Before After Before After

- T=12 T=11 T=10

- T=13 T=14 T=15

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

Before After Before After Before After

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 15 Dynamic Inpainting Results. We provide qualitative examples of a total of 16 consectutive frames generated by DynamicCity on CarlaSC [42]. Best viewed in colors and zoomed-in for additional details.

- 9.6 Comparisons with OccSora

We compare our qualitative results with OccSora [41] in Fig. 16, using a similar scene. It is evident that our result presents a realistic dynamic scene, with straight roads and complete objects and environments. In contrast, OccSora’s result displays unreasonable semantics, such as a pedestrian in the middle of the road, broken vehicles, and a lack of dynamic elements. This comparison highlights the effectiveness of our method.

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

OccSora Ours OccSora Ours OccSora Ours

T=1 T=2 T=3

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

OccSora Ours OccSora Ours OccSora Ours

- T=6 T=5 T=4

- T=7 T=8 T=9

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

OccSora Ours OccSora Ours OccSora Ours

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

OccSora Ours OccSora Ours OccSora Ours

- T=12 T=11 T=10

- T=13 T=14 T=15

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

OccSora Ours OccSora Ours OccSora Ours

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

- Figure 16 Comparisons of Dynamic Scene Generation. We provide qualitative examples of a total of 16 consectutive frames generated by OccSora [41] and our proposed DynamicCity framework on CarlaSC [42]. Best viewed in colors and zoomed-in for additional details.

- 9.7 Dynamic Outpainting

We present the full outpainting results in Fig. 17. The results demonstrate that our model can extend a scene into a larger dynamic scene.

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

T=1 T=2 T=3

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

- T=6 T=5 T=4

- T=7 T=8 T=9

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

- T=12 T=11 T=10

- T=13 T=14 T=15

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

- Figure 17 Dynamic Outpainting Results. We provide qualitative examples of a total of 16 consectutive frames generated by DynamicCity on CarlaSC [42]. Best viewed in colors and zoomed-in for additional details.

- 9.8 Single Frame Occupancy Conditional Generation

We present the results of generating frames based on a single-frame occupancy condition in Fig. 18. The results demonstrate good temporal consistency with the condition frame, highlighting our model’s ability to condition on easily accessible data.

Build Vehicle Ped Road Sidewalk Barrier Ground Veg Other Pole

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

Condition T=2 T=3 T=4

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

- T=8 T=7 T=6 T=5

- T=9 T=10 T=11 T=12

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

T=16 T=15 T=14 T=13

- Figure 18 Single Frame Occupancy Conditional Generation Results. We provide qualitative examples of a total of 16 consectutive frames generated by DynamicCity on CarlaSC [42]. Best viewed in colors and zoomed-in for additional details.

###### 10 Potential Societal Impact & Limitations

In this section, we elaborate on the potential positive and negative societal impact of this work, as well as the broader impact and some potential limitations.

- 10.1 Societal Impact

Our approach’s ability to generate high-quality 4D occupancy holds the potential to significantly impact various domains, particularly autonomous driving, robotics, urban planning, and smart city development. By creating realistic, large-scale, dynamic scenes, our model can aid in developing more robust and safe autonomous systems. These systems can be better trained and evaluated against diverse scenarios, including rare but critical edge cases like unexpected pedestrian movements or complex traffic patterns, which are difficult to capture in real-world datasets. This contribution can lead to safer autonomous vehicles, reducing traffic accidents, and improving traffic efficiency, ultimately benefiting society by enhancing transportation systems.

In addition to autonomous driving, DynamicCity can be valuable for developing virtual reality (VR) environments and augmented reality (AR) applications, enabling more realistic 3D simulations that could be used in various industries, including entertainment, training, and education. These advancements could help improve skill development in driving schools, emergency response training, and urban planning scenarios, fostering a safer and more informed society.

Despite these positive outcomes, the technology could be misused. The ability to generate realistic dynamic scenes might be exploited to create misleading or fake data, potentially undermining trust in autonomous systems or spreading misinformation about the capabilities of such technologies. However, we do not foresee any direct harmful impact from the intended use of this work, and ethical guidelines and responsible practices can mitigate potential risks.

- 10.2 Broader Impact

Our approach’s contribution to 4D scene generation stands to advance the fields of autonomous driving, robotics, and even urban planning. By providing a scalable solution for generating diverse and dynamic scenes, it enables researchers and engineers to develop more sophisticated models capable of handling real-world complexity. This has the potential to accelerate progress in autonomous systems, making them safer, more reliable, and adaptable to a wide range of environments. For example, researchers can use DynamicCity to generate synthetic training data, supplementing real-world data, which is often expensive and time-consuming to collect, especially in dynamic and high-risk scenarios.

The broader impact also extends to lowering entry barriers for smaller research institutions and startups that may not have access to vast amounts of real-world occupancy data. By offering a means to generate realistic and dynamic scenes, DynamicCity democratizes access to high-quality data for training and validating machine learning models, thereby fostering innovation across the autonomous driving and robotics communities.

However, it is crucial to emphasize that synthetic data should be used responsibly. As our model generates highly realistic scenes, there is a risk that reliance on synthetic data could lead to models that fail to generalize effectively in real-world settings, especially if the generated scenes do not capture the full diversity or rare conditions found in real environments. Hence, it is important to complement synthetic data with real-world data and ensure transparency when using synthetic data in model training and evaluation.

- 10.3 Known Limitations

Despite the strengths of DynamicCity, several limitations should be acknowledged. First, our model’s ability to generate extremely long sequences is still constrained by computational resources, leading to potential challenges in accurately modeling scenarios that span extensive periods. While we employ techniques to extend temporal modeling, there may be degradation in scene quality or consistency when attempting to generate sequences beyond a certain length, particularly in complex traffic scenarios.

Second, the generalization capability of DynamicCity depends on the diversity and representativeness of the training datasets. If the training data does not cover certain environmental conditions, object categories, or dynamic behaviors, the generated scenes might lack these aspects, resulting in incomplete or less realistic dynamic occupancy data. This could limit the model’s effectiveness in handling unseen or rare scenarios, which are critical for validating the robustness of autonomous systems.

Third, while our model demonstrates strong performance in generating dynamic scenes, it may face challenges in highly congested or intricate traffic environments, where multiple objects interact closely with rapid, unpredictable movements. In such cases, DynamicCity might struggle to capture the fine-grained details and interactions accurately, leading to less realistic scene generation.

Lastly, the reliance on pre-defined semantic categories means that any variations or new object types not included in the training set might be inadequately represented in the generated scenes. Addressing these limitations would require integrating more diverse training data, improving the model’s adaptability, and refining techniques for longer sequence generation.

- 11 Public Resources Used In this section, we acknowledge the public resources used, during the course of this work.

- 11.1 Public Datasets Used

- • nuScenes1 ........................................................................CC BY-NC-SA 4.0
- • nuScenes-devkit2 .................................................................Apache License 2.0
- • Waymo Open Dataset3 ......................................................Waymo Dataset License
- • CarlaSC4 .............................................................................. MIT License
- • Occ3D5 ................................................................................ MIT License

- 11.2 Public Implementations Used

- • SemCity6 ................................................................................. Unknown
- • OccSora7 ........................................................................ Apache License 2.0
- • MinkowskiEngine8 ..................................................................... MIT License
- • TorchSparse9 ...........................................................................MIT License
- • SPVNAS10 .............................................................................MIT License
- • spconv11 .........................................................................Apache License 2.0

- 1https://www.nuscenes.org/nuscenes
- 2https://github.com/nutonomy/nuscenes-devkit
- 3https://waymo.com/open
- 4https://umich-curly.github.io/CarlaSC.github.io.
- 5https://tsinghua-mars-lab.github.io/Occ3D.
- 6https://github.com/zoomin-lee/SemCity.
- 7https://github.com/wzzheng/OccSora.
- 8https://github.com/NVIDIA/MinkowskiEngine.
- 9https://github.com/mit-han-lab/torchsparse.
- 10https://github.com/mit-han-lab/spvnas.
- 11https://github.com/traveller59/spconv.

References

- [1] Antonio Alliegro, Yawar Siddiqui, Tatiana Tommasi, and Matthias Nießner. PolyDiff: Generating 3D polygonal meshes with diffusion models. arXiv preprint arXiv:2312.11417, 2023.
- [2] Sherwin Bahmani, Xian Liu, Yifan Wang, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, Andrea Tagliasacchi, and David B. Lindell. TC4D: Trajectory-conditioned text-to-4D generation. arXiv preprint arXiv:2403.17920, 2024.
- [3] Maxim Berman, Amal Rannen Triki, and Matthew B Blaschko. The lovász-softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4413–4421, 2018.
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023.
- [5] Lucas Caccia, Herke van Hoof, Aaron Courville, and Joelle Pineau. Deep generative modeling of LiDAR data. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5034–5040, 2019.
- [6] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11621–11631, 2020.
- [7] Ang Cao and Justin Johnson. HexPlane: A fast representation for dynamic scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130–141, 2023.
- [8] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3D generative adversarial networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16123–16133, 2022.
- [9] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4D spatio-temporal convnets: Minkowski convolutional neural networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3075–3084, 2019.
- [10] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with io-awareness. In Advances in Neural Information Processing Systems, volume 35, pages 16344–16359, 2022.
- [11] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-Planes: Explicit radiance fields in space, time, and appearance. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12479–12488, 2023.
- [12] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [13] Fangzhou Hong, Lingdong Kong, Hui Zhou, Xinge Zhu, Hongsheng Li, and Ziwei Liu. Unified 3D and 4D panoptic segmentation via dynamic shifting networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):3480–3495, 2024.
- [14] Qianjiang Hu, Zhimin Zhang, and Wei Hu. RangeLDM: Fast realistic LiDAR point cloud generation. In European Conference on Computer Vision, pages 115–135, 2024.
- [15] Siyuan Huang, Yichen Xie, Song-Chun Zhu, and Yixin Zhu. Spatio-temporal self-supervised representation learning for 3D point clouds. In IEEE/CVF International Conference on Computer Vision, pages 6535–6545, 2021.
- [16] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4D: Consistent 360° dynamic object generation from monocular video. arXiv preprint arXiv:2311.02848, 2023.
- [17] Jumin Lee, Sebin Lee, Changho Jo, Woobin Im, Juhyeong Seon, and Sung-Eui Yoon. SemCity: Semantic scene generation with triplane diffusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28337–28347, 2024.
- [18] Yuheng Liu, Xinke Li, Xueting Li, Lu Qi, Chongshou Li, and Ming-Hsuan Yang. Pyramid diffusion for fine 3D large scene generation. arXiv preprint arXiv:2311.12085, 2023.
- [19] Zhen Liu, Yao Feng, Michael J. Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu. MeshDiffusion: Score-based generative 3D mesh modeling. In International Conference on Learning Representations, 2023.

- [20] Zhiyuan Ma, Yuxiang Wei, Yabin Zhang, Xiangyu Zhu, Zhen Lei, and Lei Zhang. ScaleDreamer: Scalable text-to-3D synthesis with asynchronous score distillation. In European Conference on Computer Vision, pages 1–19, 2024.
- [21] Kazuto Nakashima and Ryo Kurazume. Learning to drop points for LiDAR scan synthesis. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 222–229, 2021.
- [22] Kazuto Nakashima and Ryo Kurazume. LiDAR data synthesis with denoising diffusion probabilistic models. In IEEE International Conference on Robotics and Automation, pages 14724–14731, 2024.
- [23] Kazuto Nakashima, Yumi Iwashita, and Ryo Kurazume. Generative range imaging for learning scene priors of 3D LiDAR data. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1256–1266, 2023.
- [24] Lucas Nunes, Rodrigo Marcuzzi, Benedikt Mersch, Jens Behley, and Cyrill Stachniss. Scaling diffusion models to real-world 3D LiDAR scene completion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14770–14780, 2024.
- [25] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32:8026–8037, 2019.
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [27] Haoxi Ran, Vitor Guizilini, and Yue Wang. Towards realistic scene generation with LiDAR diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14738–14748, 2024.
- [28] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. DreamGaussian4D: Generative 4D gaussian splatting. arXiv preprint arXiv:2312.17142, 2023.
- [29] Jiawei Ren, Kevin Xie, Ashkan Mirzaei, Hanxue Liang, Xiaohui Zeng, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, and Huan Ling. L4GM: Large 4d gaussian reconstruction model. arXiv preprint arXiv:2406.10324, 2024.
- [30] Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. XCube: Large-scale 3D generative modeling using sparse voxel hierarchies. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4209–4219, 2024.
- [31] Sara Rojas, Julien Philip, Kai Zhang, Sai Bi, Fujun Luan, Bernard Ghanem, and Kalyan Sunkavall. DATENeRF: Depth-aware text-based editing of nerfs. arXiv preprint arXiv:2404.04526, 2024.
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [33] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. MVDream: Multi-view diffusion for 3D generation. arXiv preprint arXiv:2308.16512, 2023.
- [34] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2015.
- [35] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-A-Video: Text-to-video generation without text-video data. In International Conference on Learning Representations, 2022.
- [36] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, and Yaniv Taigman. Text-to-4D dynamic scene generation. arXiv preprint arXiv:2301.11280, 2023.
- [37] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2446–2454, 2020.
- [38] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818–2826, 2015.

- [39] Haotian Tang, Zhijian Liu, Shengyu Zhao, Yujun Lin, Ji Lin, Hanrui Wang, and Song Han. Searching efficient 3D architectures with sparse point-voxel convolution. In European Conference on Computer Vision, pages 685–702, 2020.
- [40] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3D: A large-scale 3D occupancy prediction benchmark for autonomous driving. In Advances in Neural Information Processing Systems, volume 36, pages 64318–64330, 2023.
- [41] Lening Wang, Wenzhao Zheng, Yilong Ren, Han Jiang, Zhiyong Cui, Haiyang Yu, and Jiwen Lu. OccSora: 4D occupancy generation models as world simulators for autonomous driving. arXiv preprint arXiv:2405.20337, 2024.
- [42] Joey Wilson, Jingyu Song, Yuewei Fu, Arthur Zhang, Andrew Capodieci, Paramsothy Jayakumar, Kira Barton, and Maani Ghaffari. MotionSC: Data set and network for real-time semantic mapping in dynamic environments. IEEE Robotics and Automation Letters, 7(3):8439–8446, 2022.
- [43] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3D: High-quality and efficient 3D mesh generation from a single image. arXiv preprint arXiv:2405.20343, 2024.
- [44] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3D: Scalable image-to-3D generation via 3D latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024.
- [45] Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. CityDreamer4D: Compositional generative model of unbounded 4D cities. arXiv preprint arXiv:2501.08983, 2025.
- [46] Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. Generative gaussian splatting for unbounded 3D city generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [47] Yuwen Xiong, Wei-Chiu Ma, Jingkang Wang, and Raquel Urtasun. UltraLiDAR: Learning compact representations for LiDAR completion and generation. arXiv preprint arXiv:2311.01448, 2023.
- [48] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. OccWorld: Learning a 3D occupancy world model for autonomous driving. In European Conference on Computer Vision, pages 55–72, 2024.
- [49] Zehan Zheng, Fan Lu, Weiyi Xue, Guang Chen, and Changjun Jiang. LiDAR4D: Dynamic neural fields for novel space-time view LiDAR synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5145–5154, 2024.
- [50] Vlas Zyrianov, Xiyue Zhu, and Shenlong Wang. Learning to generate realistic LiDAR point clouds. In European Conference on Computer Vision, pages 17–35, 2022.

