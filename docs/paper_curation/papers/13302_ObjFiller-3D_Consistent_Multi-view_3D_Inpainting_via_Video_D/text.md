# arXiv:2508.18271v2[cs.CV]27May2026

## ObjFiller3D: Scaling 3D Object Inpainting to Dense Multi-View Consistency

HAITANG FENG∗, Nanjing University, China XINKAI CHEN∗, Great Bay University, China JIE LIU†, Nanjing University, China JIE TANG, Nanjing University, China GANGSHAN WU, Nanjing University, China BEIQI CHEN, Harbin Institute of Technology, China JIANHUANG LAI, Sun Yat-sen University, China GUANGCONG WANG†, Great Bay University, China

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>PSNR↑/LPIPS↓<br><br>PSNR↑/LPIPS↓<br><br>GT SD Image Cond NeRFiller Fillerbuster Ours<br><br>13.4 / 0.31 17.2 / 0.21 21.7 / 0.19 26.8 / 0.05<br>14.8 / 0.22 17.9 / 0.18 20.9 / 0.11 26.4 / 0.07<br>|
|---|

Fig. 1. ObjFiller3D reconstructs complete 3D objects from partial inputs. Regions requiring inpainting are marked in pink. Our method outperforms previous state-of-the-art methods across multiple benchmark datasets.

3D object inpainting is commonly achieved via multi-view 2D image completion, yet independently inpainted views often suffer from cross-view inconsistencies, leading to blurred textures, geometric discontinuities, and visual artifacts in the reconstructed 3D objects. To overcome these limitations, we propose ObjFiller-3D, a novel method designed for the completion and editing of high-quality and consistent 3D objects. Instead of relying on sparse-view editing or per-view 2D inpainting, our method jointly optimizes a sequence of densely sampled views along a 360◦ trajectory, enabling global coherence across viewpoints. We design a new framework with three complementary components: a Temporal-Driven Generative Encoder for modeling dense-view dependencies, a Semantic-Aware Completion Encoder for object-level inpainting, and a Cycle-Consistent 3D Encoder that enforces global coherence through a closed-loop formulation. Our framework also

∗Equal contribution. †Corresponding authors.

Authors’ Contact Information: Haitang Feng, Nanjing University, China; Xinkai Chen, Great Bay University, China; Jie Liu, Nanjing University, China; Jie Tang, Nanjing University, China; Gangshan Wu, Nanjing University, China; Beiqi Chen, Harbin Institute of Technology, China; Jianhuang Lai, Sun Yat-sen University, China; Guangcong Wang, Great Bay University, China.

supports reference-guided 3D inpainting, allowing fine-grained control over appearance. Extensive experiments on diverse datasets demonstrate that ObjFiller-3D significantly outperforms prior methods, achieving higher reconstruction fidelity (PSNR 26.6 vs. 15.9 of NeRFiller) and perceptual quality (LPIPS 0.19 vs. 0.25 of Instant3dit), while reducing reconstruction time from over 40 minutes to under 10 minutes. These results highlight the effectiveness and practical potential of our approach for real-world 3D editing applications.

1 Introduction

High-quality completion of incomplete 3D objects is essential for many graphics applications, including digital reconstruction, 3D asset creation, and game or film production. Environmental constraints during scanning often result in data loss, which poses significant challenges in recovering the original geometry of objects. Despite its importance, research on 3D object-centric completion remains relatively scarce, with only a few methods [Barda et al. 2025; Li et al. 2025; Weber et al. 2024, 2026] have focused on this

problem, and the task is fundamentally distinct from 3D generation and mainstream editing tasks.

Some existing methods offer a partial solution to 3D object inpainting. 3D generation methods [Gao et al. 2024; Hong et al. 2023; Li et al. 2023b, 2024; Liu et al. 2023; Long et al. 2024; Poole et al.

- 2022; Wang and Shi 2023; Wu et al. 2025b; Xiang et al. 2025; Xu et al. 2024] typically aim to produce high-quality 3D assets from noise, a single image, or a few reference views. These approaches generally assume that the input images depict complete views of the object and are not designed to handle partial or degraded observations. On the other hand, mainstream 3D editing [Chen and Wang 2024; Chen et al. 2024b,a; Haque et al. 2023; Liu et al. 2024; Mirzaei et al.
- 2023; Shi et al. 2025; Wang et al. 2024; Wu et al. 2025a] focuses on modifying global textures or performing style transfer, without accommodating significant geometric alterations, which makes it unsuitable for restoring missing object parts.

One straightforward approach is to first employ a 2D inpainting model [Lugmayr et al. 2022; Suvorov et al. 2022] to restore multiview images of an incomplete object. Score Distillation Sampling (SDS) [Poole et al. 2022] uses a pre-trained 2D diffusion model as a prior to guide 3D optimization. It renders images from a 3D representation under random viewpoints and backpropagates score-based gradients from the diffusion model to iteratively update the 3D parameters toward the given condition. In contrast, Iterative Dataset Update (IDU) [Haque et al. 2023] follows a data-centric paradigm that alternates between 2D image editing and 3D reconstruction. It iteratively refines input images using a generative model and reconstructs the 3D structure from the updated images, progressively improving alignment with the target. However, these methods lack explicit modeling of cross-view dependencies of the same object, which often results in inconsistencies across different viewpoints (see Fig. 1), which in turn introduces visual artifacts and degrades reconstruction accuracy. Moreover, they do not support high-quality localized 3D editing.

To learn cross-view-aware consistency, several sparse-view editing methods have been proposed. NeRFiller [Weber et al. 2024] extends IDU by introducing a 2×2 image grid that packs four views into a single image. It shows that jointly denoising these four views leads to more consistent multi-view inpainting compared to independent single-view processing, referred to as a grid prior. Building upon this idea, Instant3dit [Barda et al. 2025] further introduces a masked object dataset and trains a Stable Diffusion inpainting model [Lugmayr et al. 2022] to generate consistent 2×2 grid images. The resulting multi-view images are then fed into a Large Reconstruction Model (LRM) [Hong et al. 2023] to efficiently produce a full 3D reconstruction. Despite these efforts toward cross-view consistency, such approaches are inherently limited to enforcing consistency over only a small number of views (e.g., four), which restricts viewpoint coverage and leads to insufficient geometric and textural details. Consequently, they are mainly effective for objects with simple structures and relatively uniform textures, while struggling with more complex shapes and dense-view settings, where high-quality reconstruction remains challenging. Simply increasing the grid size leads to a trade-off between spatial coverage and resolution, resulting in degraded reconstruction quality.

To address the limitations of sparse-view 3D object editing, we propose to model 3D inpainting as a dense-view sequence problem and introduce a new framework that explicitly captures crossview dependencies. Unlike prior approaches that process views independently or rely on sparse observations, our method jointly optimizes a sequence of densely sampled views along a 360◦ trajectory, enabling globally consistent reconstruction. A key insight of our approach is that achieving high-quality 3D inpainting requires simultaneously modeling appearance generation, region completion, and object-level structural consistency. Motivated by video generation models [Jiang et al. 2025], we design a new 3D-objectcentric architecture, tri-encoder, composed of three complementary components: a Temporal-Driven Generative Encoder that models dependencies across densely sampled views, a Semantic-Aware Completion Encoder that infers plausible content for missing regions, and a Cycle-Consistent 3D Encoder that enforces global coherence through a closed-loop formulation. By duplicating the first view as the last, we explicitly impose cyclic constraints that enable longrange consistency across the entire view sequence. Furthermore, our framework naturally supports reference-guided 3D inpainting by introducing a reference view with an all-zero mask, which serves as an appearance anchor while preserving geometric consistency across views. Extensive experiments demonstrate that our method significantly outperforms prior approaches in both reconstruction quality and efficiency. In particular, ObjFiller-3D achieves more faithful and fine-grained 3D reconstructions, surpassing NeRFiller and Instant3dit with a PSNR of 26.6 (vs. 15.9) and an LPIPS of 0.19 (vs. 0.25), respectively.

Overall, our main contributions can be summarized as follows: (1) We introduce a cycle-consistent dense-view formulation for 3D object inpainting that enforces long-range coherence across 360◦ viewpoints, and instantiate it with a tri-encoder framework that jointly models dense-view generation, region completion, and global 3D consistency. (2) We develop a reference-guided inpainting mechanism that enables controllable appearance generation while preserving structural consistency. (3) We achieve state-of-the-art performance on multiple benchmarks, significantly improving both reconstruction fidelity and efficiency over prior methods.

- 2 Related Work
- 3D object inpainting is challenging due to the need for cross-view consistency under incomplete observations. Unlike 2D inpainting, inconsistencies across viewpoints can accumulate and degrade the reconstructed 3D results. Our problem is related to several research directions, including 3D object generation and 3D editing. We review these directions below and discuss their limitations for dense-view consistent 3D inpainting.

3D Object Generation. 3D object generation methods are typically categorized into 2D prior-based and feedforward approaches. The former utilizes image generative models (e.g., Stable Diffusion [Rombach et al. 2022], Imagen [Saharia et al. 2022]) trained on large-scale datasets like LAION-400M/5B [Schuhmann et al. 2022, 2021] to extend text or image inputs into 3D. DreamFusion [Poole et al. 2022] introduces Score Distillation Sampling (SDS) for this purpose, followed by improved but costly variants [Huang et al. 2023; Wang

[Figure 11]

N×

[Figure 12]

Temporal-Driven Generative Encoder

|[Figure 13]<br><br>[Figure 14]<br><br>VAE Decoder|
|---|

[Figure 15]

[Figure 16]

|[Figure 17]<br><br>[Figure 18]<br><br>View Enc.| |
|---|---|
| | |

360° Sampled Rendering

[Figure 19]

[Figure 20]

[Figure 21]

View Enc.

[Figure 22]

[Figure 23]

|Layer Norm|[Figure 24]<br><br>[Figure 25]<br><br>|
|---|---|
| | |

|Self-Attention|[Figure 26]<br><br>[Figure 27]<br><br>|
|---|---|
| | |

|[Figure 28]<br><br>Cross-Attention|[Figure 29]<br><br>|
|---|---|
| | |

|[Figure 30]<br><br>[Figure 31]<br><br>Layer Norm| |
|---|---|
| | |

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

VAE Decoder

Self-Attention

Cross-Attention

Layer Norm

Layer Norm

Layer Norm

[Figure 37]

[Figure 38]

[Figure 39]

FFN

[Figure 40]

Noisy Latent

Looped Seq.

Obj. Dense Views

Inp. Dense Views

|[Figure 41]<br><br>[Figure 42]<br><br>Text Enc.| |
|---|---|
| | |

[Figure 43]

[Figure 44]

[Figure 45]

Text Enc.

###### Text

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

< A statue … >

ℒ

#### ℒ

Semantic-Aware Completion Encoder

3D Object

[Figure 50]

[Figure 51]

|[Figure 52]<br><br>[Figure 53]<br><br>Layer Norm| |
|---|---|
| | |

|[Figure 54]<br><br>[Figure 55]<br><br>Self-Attention| |
|---|---|
| | |

|[Figure 56]<br><br>Cross-Attention|[Figure 57]<br><br>|
|---|---|
| | |

|[Figure 58]<br><br>[Figure 59]<br><br>Layer Norm| |
|---|---|
| | |

[Figure 60]

[Figure 61]

Self-Attention

Cross-Attention

[Figure 62]

Layer Norm

Layer Norm

Layer Norm

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Render Loss

FFN

[Figure 77]

[Figure 78]

Optional Ref.

3D Mask

3D Gaussians

Projected 2D Mask

[Figure 79]

[Figure 80]

[Figure 81]

|[Figure 82]<br><br>[Figure 83]<br><br>Decomposition|
|---|

|[Figure 84]<br><br>[Figure 85]<br><br>VAE Encoder| |
|---|---|
| | |
| | |
| | |

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Share Share Share

[Figure 91]

Decomposition

|Layer Norm|[Figure 92]<br><br>[Figure 93]<br><br>|
|---|---|
| | |

|[Figure 94]<br><br>[Figure 95]<br><br>Self-Attention| |
|---|---|
| | |

|Layer Norm|[Figure 96]<br><br>[Figure 97]<br><br>|
|---|---|
| | |

|[Figure 98]<br><br>Cross-Attention|[Figure 99]<br><br>|
|---|---|
| | |

|[Figure 100]<br><br>[Figure 101]<br><br>Layer Norm| |
|---|---|
| | |

[Figure 102]

[Figure 103]

Self-Attention

Cross-Attention

VAE Encoder

Layer Norm

Layer Norm

Layer Norm

[Figure 104]

[Figure 105]

Inactive Partition

𝑖−1

FFN

𝑐 = ෍

𝑐𝑖𝛼𝑖𝐺 𝑥𝑖 ෑ 𝑗=1

(1 − α𝑗G(x𝑗))

Loop

[Figure 106]

𝑖

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Inpainted 3D Object

Masked 3D Obj.

Cycle-Consistent3D Encoder

Reactive Partition

Fig. 2. Overview of ObjFiller-3D. A complete 3D object is rendered into a looped dense-view target sequence, while a 3D mask is applied to construct a masked 3D input. The masked dense views are decomposed into inactive and reactive partitions, encoded together with the text prompt and an optional reference image, and processed by the Temporal-Driven Generative Encoder, Semantic-Aware Completion Encoder, and Cycle-Consistent 3D Encoder. The predicted latent completions are decoded into globally consistent inpainted dense views and subsequently reconstructed as a complete 3D Gaussian object.

- et al. 2023]. Alternatively, methods such as MVDream [Shi et al. 2023b], ImageDream [Wang and Shi 2023], Zero123++ [Shi et al.

- 2023a], SyncDreamer [Liu et al. 2023], Wonder3D [Long et al. 2024], Cat3D [Gao et al. 2024], and MV-Adapter [Huang et al. 2024] directly generate multi-view consistent images using finetuned diffusion models, offering a more efficient path to 3D reconstruction. Feedforward generation is another key paradigm for 3D synthesis, often employing encoder-decoder architectures such as Latent Diffusion Models (LDMs) [Rombach et al. 2022] to extract image tokens and generate latent 3D representations. Models such as Large Reconstruction Model (LRM) [Hong et al. 2023], MeshLRM [Wei et al.
- 2024], Instant3D [Li et al. 2023b], and InstantMesh [Xu et al. 2024] use Transformers to map tokens into implicit triplane representations. CLAY [Zhang et al. 2024] and CraftsMan3D [Li et al. 2024] reconstruct meshes via neural signed distance or occupancy fields, while Trellis [Xiang et al. 2025] and Direct3D-S2 [Wu et al. 2025b] adopt voxel grids for more interpretable latent spaces. However, these methods focus on generation rather than completing partially observed objects, making them less suitable for 3D inpainting tasks.

3D Editing. Recent advances in 3D field editing mainly leverage NeRF [Mildenhall et al. 2021] and 3DGS [Kerbl et al. 2023] for superior realism and flexibility over traditional mesh or point clouds. Instruction-guided methods use pre-trained 2D diffusion models due to scarce 3D data, optimizing NeRF via Score Distillation Sampling (SDS) to achieve view-consistent, relightable 3D outputs. Iterative Dataset Update (IDU) approaches such as InstructNeRF2NeRF [Haque et al. 2023] train 3D scenes using edited images. Methods such as VcEdit [Wang et al. 2024], GaussianEditor [Chen

- et al. 2024a], and ProEdit [Chen and Wang 2024] extend 3DGS for efficient, controllable editing, with DGE [Chen et al. 2024b] improving multi-view coherence via epipolar attention. For 3D object removal, SPIn-NeRF [Mirzaei et al. 2023] combines depth and

- 2D inpainting but struggles with full 360◦ views. Recent Gaussian Splatting inpainting works such as InFusion [Liu et al. 2024], AuraFusion360 [Wu et al. 2025a], and IMFine [Shi et al. 2025] address 360◦ unbounded scene restoration. However, these methods focus on scene-level editing or object removal, and are not designed for consistent completion of partially observed 3D objects.

Among the most closely related methods to ours are NeRFiller [Weber et al. 2024], Instant3dit [Barda et al. 2025], and Fillerbuster [Weber et al. 2026]. Both NeRFiller and Instant3dit follow sparse-view editing paradigms, where consistency is enforced over only a limited number of input views, and NeRFiller further relies on a timeconsuming IDU process. These limitations inherently restrict viewpoint coverage and hinder efficient modeling, leading to incomplete geometry and degraded texture fidelity in the reconstructed 3D objects. Fillerbuster does not leverage strong video priors but instead trains a 1B DiT from scratch. It uses camera ray embeddings to provide geometric information for multi-view inputs and processes all views simultaneously, which leads to increasing cross-view inconsistencies as the number of views grows.

- 3 Method

We present ObjFiller-3D, a new framework for consistent and controllable 3D object inpainting. Our key idea is to reformulate 3D inpainting as a dense-view modeling problem, where a sequence of views sampled along a 360◦ trajectory is jointly optimized to enforce global coherence across viewpoints. Given a partially observed 3D object, we construct dense-view inputs with corresponding masks, text conditions, and an optional reference image. Our model predicts a set of globally consistent inpainted views, which are subsequently used to reconstruct the completed 3D object. Motivated by video generation models [Jiang et al. 2025], we introduce a novel 3D object-centric framework based on a tri-encoder architecture,

[Figure 115]

[Figure 116]

[Figure 117]

which explicitly captures cross-view dependencies while enabling semantic-aware completion and global structural consistency. In the following, we first introduce the dense-view formulation and conditional representation, and then present the design of the three encoders and the overall optimization process.

- 3.1 Preliminary and Problem Formulation

Preliminary. 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] uses 3D Gaussians as scene primitives and represents the scene with a large number of 3D Gaussians. A Gaussian ellipsoid centered at point 𝜇 ∈ R3 with covariance matrix Σ can be written as

𝐺(x) = 𝑒−12x𝑇 Σ−1x, and Σ = 𝑅𝑆𝑆𝑇𝑅𝑇, (1)

where x denotes the distance from a specific point in space to the mean 𝜇, and 𝑅 ∈ R3×3 and 𝑆 ∈ R3×3 represent the Gaussian’s rotation and scaling matrices, respectively. After formalizing the 3D Gaussians, volumetric rendering techniques [Kajiya and Von Herzen 1984] can be used to splat these Gaussians onto the 2D image plane. Specifically, the color 𝑐 for a pixel along a ray is given by

𝑐 = ∑︁

𝑖

𝑐𝑖𝛼𝑖𝐺(xi)

𝑖−1

𝑗=1

1 − 𝛼𝑗𝐺(xj) , (2)

where 𝛼 ∈ R and 𝑐 ∈ R3 represent the opacity and color of the

- 3D Gaussians, respectively. 3DGS is recognized for its efficiency in real-time radiance field rendering and has thus been integrated into our method.

Problem Formulation. We define the task of 3D object inpainting as follows: given an incomplete 3D object 𝑂, a 3D mask region 𝑀, a text description 𝑦 of the object, and an optional reference image 𝐼𝑟, our goal is to generate a plausible shape 𝑂′ within the masked region 𝑀, conditioned on 𝑦 and 𝐼𝑟, such that the completed object {𝑂 ∪𝑂′} forms a geometrically consistent and visually coherent 3D structure. This task is inherently challenging, as it requires jointly reasoning about geometry, appearance, and semantic consistency under partial observations. The generated content 𝑂′ must not only align with the existing structure of 𝑂, but also maintain consistency across viewpoints when rendered from different camera poses. In addition, the completion should respect both the semantic guidance provided by the text description and the appearance cues from the reference image, if available.

- 3.2 Dataset Preparation

Convexhull Surface Volume

Fig. 3. Three types of 3D masks.

3D Inpainting Results

Input Masked Images

[Figure 118]

…

[Figure 119]

[Figure 120]

SD Image Cond

[Figure 121]

…

[Figure 122]

NeRFiller

[Figure 123]

…

ObjFiller-3D

More Consistent!

[Figure 124]

[Figure 125]

…

Fig. 4. Visual comparison of different multi-view inpainting methods. Compared to baseline approaches, ObjFiller-3D achieves more consistent and coherent inpainting across views.

To exploit the inpainting prior of generative video diffusion mod-

els, we project 3D objects into 2D sequences. Specifically, object 𝑂 and mask 𝑀 are rendered into image frames 𝐹𝑠 = {𝑓𝑖}𝑛𝑖=1 and binary masks 𝑀𝑠 = {𝑚𝑖}𝑛𝑖=1 under predefined camera poses Π = {𝜋𝑖}𝑛𝑖=1:

⟨𝑓𝑖,𝑚𝑖⟩ = R(𝑂,𝑀,𝜋𝑖), 𝑖 ∈ {1, . . .,𝑛}, (3)

where R denotes the rendering operator. The resulting dataset is approximately 18 GB and will be publicly released.

We adopt the Instant3dit dataset [Barda et al. 2025], which contains ∼7k high-quality 3D objects from Objaverse [Deitke et al. 2023] and three mask types: convexhull, surface, and volume (Fig. 3). 1) Convexhull: the missing part 𝑂′ lies inside mask 𝑀. 2) Surface: 𝑀 covers a local surface region of object 𝑂. 3) Volume: 𝑀 tightly encloses the entire object 𝑂.

3.3 Dense-view Consistent 3D Object Inpainting

To overcome the limitations of sparse-view editing, where each view is processed largely independently and often leads to cross-view inconsistencies, we adopt a dense-view formulation to explicitly model global coherence across viewpoints. Our method jointly optimizes a sequence of densely sampled views arranged along a 360◦ trajectory, enabling consistent 3D inpainting with improved geometric and appearance fidelity. Let 𝐹𝑠 = {𝑓𝑖}𝑖𝐾=1 denote the rendered view sequence of an object, where each 𝑓𝑖 corresponds to a viewpoint sampled along a 360◦ trajectory and 𝐾 is the number of dense views. Let 𝑀𝑠 = {𝑚𝑖}𝑖𝐾=1 denote the corresponding binary masks

Since Instant3dit does not provide complete objects or 2D groundtruth images, we retrieve the original objects from Objaverse and render 16-view images at 512 × 512 resolution using Blender. The views are sampled with a fixed 20° elevation and uniformly distributed azimuths from 0 to 2𝜋. We further employ Cap3D [Luo et al. 2023] to generate captions using BLIP-2 [Li et al. 2023a], CLIP [Radford et al. 2021], and GPT4 [Achiam et al. 2023].

indicating regions to be inpainted, and let 𝑦 denote the text condition for semantic guidance. An optional reference image 𝐼𝑟 provides additional appearance cues. Each object is thus represented as a tuple ⟨𝐹𝑠,𝑀𝑠,𝑦,𝐼𝑟⟩.

To enforce cyclic consistency, we duplicate the first view and its mask as the (𝐾 + 1)-th element, forming a closed-loop sequence. Based on this dense-looped representation, our model consists of three core components: a Temporal-Driven Generative Encoder, a Semantic-Aware Completion Encoder, and a Cycle-Consistent 3D Encoder. The first branch captures long-range dependencies over the entire dense-view loop from noisy latent views, the second branch performs text- and mask-guided completion on the conditional latent sequence, and the third branch further refines the completion stream to enforce cross-view cycle consistency. Each of these components is implemented as a stack of multiple encoder layers. Together, they transform the input dense-view sequence into the inpainted sequence 𝐹𝑠′, which remains globally consistent across viewpoints and is subsequently used for 3DGS reconstruction.

Conditional Embedding. Inspired by the pretrained VACE[Jiang

- et al. 2025] input pipeline, we reuse three inherited front-end modulesinFig.2: aviewencoder,atextencoder, and a VAE encoder/decoder. The view encoder maps the looped dense-view target sequence rendered from the complete object into latent video tokens, which are

further perturbed with Gaussian noise to form the noisy latent 𝑥𝑡 used by the generative branch. The text encoder transforms the prompt 𝑦 into semantic tokens 𝑐𝑦, which are injected into the transformer blocks through cross-attention. The VAE encoder projects the masked conditional inputs, including the inactive partition, the reactive partition, the projected 2D masks, and the optional reference image, into a shared latent space, while the inherited VAE decoder finally maps the predicted latent sequence back to dense-view images. Specifically, during training, we first render the complete 3D object into a looped dense-view supervision sequence 𝐹𝑠obj = {𝑓𝑖obj}𝑖𝐾=+11, where the first view is duplicated as the (𝐾 + 1)-th frame to enforce cyclic consistency. We then apply a 3D mask to obtain the masked dense-view input 𝐹𝑠 = {𝑓𝑖}𝑖𝐾=+11 and masks 𝑀𝑠 = {𝑚𝑖}𝑖𝐾=+11. Each masked view is decomposed into inactive and reactive partitions:

𝑓𝑖inactive = 𝑓𝑖 ⊙ (1 −𝑚𝑖), 𝑓𝑖reactive = 𝑓𝑖 ⊙ 𝑚𝑖. (4) The VAE encoder maps them into latent representations:

𝑧𝑖 = Concat 𝜙(𝑓𝑖inactive),𝜙(𝑓𝑖reactive),𝑚˜𝑖 , (5)

where 𝜙(·) denotes the inherited VAE encoder and 𝑚˜𝑖 is the projected latent-aligned mask. If an optional reference image 𝐼𝑟 is provided, it is encoded as 𝑧𝑟 = 𝜙(𝐼𝑟) and prepended temporally:

𝑍 = Concat(𝑧𝑟,𝑧1,𝑧2, . . .,𝑧𝐾). (6)

After transformer-based dense-view completion, the final latent sequence𝑍final is decodedbytheinheritedVAEdecoder𝐹𝑠′ =𝜓(𝑍final), where 𝜓(·) denotes the VAE decoder and 𝐹𝑠′ is the predicted inpainted dense-view sequence. With conditional embedding, the model operates on dense-view latent representations through a unified encoder module composed of three branches.

The Temporal-Driven Generative Encoder is responsible for modeling the global dependency of the looped dense-view sequence. It

takes the noisy dense-view latent 𝑥𝑡 as input, following the inherited VACE generative path, and propagates information over the entire view loop rather than over isolated viewpoints. This design is particularly important for our setting, since dense-view object completion requires the model to preserve appearance and structure under large viewpoint changes, while simultaneously maintaining first-last view consistency in the closed loop. The output of this branch is a sequence of generative dense-view features that provides the global completion prior.

The Semantic-Aware Completion Encoder focuses on the maskedregion recovery conditioned on the observed object content. Its input is the conditional latent sequence 𝑍, which is constructed from the inactive partition, the reactive partition, the projected

- 2D masks, and the optional reference image. Unlike the generative branch that mainly captures global dense-view dynamics, this branch explicitly emphasizes which regions should be preserved and which regions should be regenerated. Through cross-attention to the text tokens, it further injects semantic guidance into the masked-region completion process, allowing the model to recover missing object parts in a way that is both structurally plausible and semantically aligned with the prompt. The output of this branch is a sequence of completion-oriented dense-view features.

The Cycle-Consistent 3D Encoder is designed to inject closed-loop multi-view consistency into the completion stream. Rather than predicting an independent dense-view sequence, it operates on the corresponding completion features and estimates cycle-consistency residuals that are fused back into the main completion branch. This enables the model to explicitly refine cross-view coherence in transformer space, especially under large pose variation and loop closure, where purely video-driven completion may still exhibit subtle view inconsistency. In practice, this encoder is realized by a factorized residual parameterization attached to the corresponding completion blocks, such that cycle-consistent 3D cues can be introduced without changing the inherited front-end encoders or the main backbone.

Together, these branches form the core transformer-based representation module of ObjFiller-3D, where global dense-view generation, semantics-aware completion, and cycle-consistent 3D refinement are jointly modeled before decoding the final inpainted views. As shown in Fig. 2, the above three encoders are instantiated on top of the pretrained VACE backbone and share the same DiT-style block topology. The residual features produced by the Cycle-Consistent

- 3D Encoder are fused into the completion stream, and the final latent sequence is decoded by the inherited VAE decoder to obtain

the inpainted dense-view sequence 𝐹𝑠′.

Two-stage Optimization. Our framework jointly addresses conditional dense-view completion and 3D novel view synthesis. We adopt a two-stage optimization scheme and train the model using a flow matching objective. Let𝑢𝜃 𝑡,𝑥 | 𝐹𝑠,𝑀𝑠,𝑦 denote the predicted velocity field, and let 𝑢∗(𝑡,𝑥) denote the target velocity induced by interpolating between the noise and data distributions. The loss is defined as:

2 2

, (7)

LFM(𝜃) = E𝑡∼U(0,1) E𝑥∼𝑝𝑡 𝑢𝜃 (𝑡,𝑥 | 𝐹𝑠,𝑀𝑠,𝑦) −𝑢∗(𝑡,𝑥)

where𝑝𝑡 denotes the marginal distribution at time𝑡. Minimizing this loss encourages a smooth and consistent transformation from noise

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

ConvexhullSurfaceVolume

…

[Figure 132]

[Figure 133]

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

…

| |
|---|

| |
|---|

| |
|---|

Reference image 𝐼r Repaired other views F’s

Fig. 5. Reference-based 3D inpainting. Given a reference image, the generated views are well aligned with the input.

to data, yielding coherent inpainted sequences. After inpainting, we obtain completed frames 𝐹𝑠′ and corresponding camera poses Π. These are used to reconstruct a 3D object represented by Gaussian primitives G:

Masked grid image Instant3dit Ours (1.3B) Ours (14B)

Fig. 6. Qualitative results compared with Instant3dit. We apply our method to three different mask types, listed from top to bottom as Convexhull, Surface, and Volume. ObjFiller-3D-14B (the rightmost column) exhibits the highest consistency across all mask types.

∑︁

Ginpainted = argmin

### L R(G,𝜋𝑖), 𝑓𝑖′ , (8)

G

𝜋𝑖 ∈Π,𝑓𝑖′∈𝐹𝑠′

where R denotes the differentiable renderer and L is the 3DGS reconstruction loss. Compared to NeRFiller, our method benefits from strong cross-view coherence established during inpainting, eliminating the need for iterative dataset update (IDU). This significantly reduces reconstruction time: our method completes the pipeline in under 10 minutes, while NeRFiller [Weber et al. 2024] requires over 40 minutes. Furthermore, our approach achieves substantially higher reconstruction quality, as shown in Figs. 1 and 4.

LRM [Xu et al. 2024]. If the views lack consistency, LRM will produce a distorted object; otherwise, the reconstruction will be coherent. To quantify this, we render the resulting object from the original viewpoints and compute the perceptual LPIPS score [Zhang et al. 2018] between the rendered images and the inpainted ones. A lower LPIPS score indicates better multi-view consistency. For comparison with NeRFiller and Fillerbuster, the final object reconstruction is a 3D reconstruction task. Therefore, we evaluate using three standard metrics: SSIM, PSNR, and LPIPS.

- 4 Experiments

Implementation Details. We train our models using 3,000 objects from the reprocessed dataset. In our implementation, We optimize only the factorized residual branch of the Cycle-Consistent 3D Encoder while keeping the inherited VACE backbone fixed. The Cycle-Consistent 3D Encoder is instantiated by attaching factorized residual parameters to the corresponding completion blocks. We set the factorization rank of the Cycle-Consistent 3D Encoder to 32, use a learning rate of 10−4, a batch size of 4, and train for 10 epochs. For the 14B backbone, training is conducted in half precision, consuming approximately 60 GB of VRAM, and takes around 3 days on a single NVIDIA A800 GPU. During inference, we use the UniPC sampler [Zhao et al. 2023] with 20 sampling steps, a CFG[Ho and Salimans 2022]guidance scale of 4, and a residual branch scale of 1.

4.1 Comparisons with State-of-the-Arts

Comparisons on Instant3dit Dataset. We consider four variants: the vanilla VACE-1.3B and VACE-14B backbones, and their counterparts equipped with the proposed Cycle-Consistent 3D Encoder. We evaluate all variants on 300 held-out objects. For Instant3dit, we stack masked multi-view images and masks from four orthogonal viewpoints (0◦, 90◦, 180◦, and 270◦) into 2 × 2 grids. These image grids, along with text prompts generated by Cap3D [Luo et al. 2023], serve as input to Instant3dit’s open-source 2D multi-view inpainting model, which has been trained on SDXL [Podell et al. 2023], exactly as expected by its design. For our method, we concatenate all dense views and projected masks into 17-frame looped sequence, and feed them into the inherited VACE pipeline together with text prompts. From the predicted inpainted dense-view sequence, we extract the same four orthogonal views and arrange them into 2 × 2 grids for evaluation. Since object reconstruction in later steps is uniformly performed by LRM for all methods, this comparison focuses on the quality, consistency, and semantic fidelity of the generated dense-view images.

Evaluation Metrics. For the comparison between our method and Instant3dit, we focus on three aspects. 1) Text-image similarity: we measure the semantic relevance between the generated 2 × 2 grid images and the given text prompt using the CLIP [Radford et al. 2021] similarity score. 2) Image quality and fidelity: this evaluates the quality of the generated images and the similarity between the inpainted images and the ground-truth (GT) images. Specifically, we inpaint 300 grid images from the evaluation dataset and compare them against GT grids using the FID score [Heusel et al. 2017]. 3) Multi-view consistency: to assess consistency across the four views in the inpainted grid image, we feed the four images directly into

Comparisons on NeRF Blender and NeRFiller Dataset. We choose ObjFiller-3D-14B, i.e., the 14B VACE backbone equipped with the proposed Cycle-Consistent 3D Encoder and followed by 3DGS reconstruction, as our main model for comparison with NeRFiller and

Table 1. Quantitative comparison on NeRF Blender and NeRFiller datasets. Cells are highlighted as follows: best , second best .

NeRF Blender [Mildenhall et al. 2021] NeRFiller [Weber et al. 2024]

Method

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

GT 35.65 0.97 0.03 35.22 0.98 0.02 SD Image Cond 14.15 0.76 0.28 19.89 0.89 0.12 NeRFiller (CVPR’24) [Weber et al. 2024] 15.89 0.82 0.23 28.85 0.95 0.04 Fillerbuster (3DV’26) [Weber et al. 2026] 20.43 0.85 0.21 32.82 0.97 0.06 ObjFiller-3D-1.3B (Ours) 21.38 0.88 0.11 30.75 0.96 0.06 ObjFiller-3D-14B (Ours) 26.62 0.93 0.07 33.68 0.97 0.03

Table 2. Quantitative comparison on Instant3dit dataset. Cells are highlighted as follows: best , second best .

Instant3dit [Barda et al. 2025]

Method

FID ↓ LPIPS ↓ Clip ↑

GT - 0.140 30.53 SD Image Cond 120.5 0.277 28.73 Instant3dit (CVPR’25) [Barda et al. 2025] 100.9 0.253 29.81 Fillerbuster (3DV’26) [Weber et al. 2026] 107.4 0.259 29.23 ObjFiller-3D-1.3B (Ours) 92.07 0.190 29.87 ObjFiller-3D-14B (Ours) 90.75 0.195 30.19

Fillerbuster. For a more comprehensive evaluation, we also include SD Image Cond, where each view is individually inpainted by Stable Diffusion and directly used for reconstruction. Experiments are conducted on the NeRF Blender synthetic dataset [Mildenhall et al. 2021], with a 256 × 256 occlusion at the image center (see Figure

- 4, top), and on the NeRFiller dataset [Weber et al. 2024], which contains scanned meshes with corresponding masks.

4.2 Further Analysis

The quantitative results are summarized in Table 1 and Table 2. We exclude Instant3dit from the NeRF Blender and NeRFiller benchmarks in Table 1 because Instant3dit is strictly limited to four-view inputs, making it inapplicable to scenarios with numerous sparse views. Similarly, NeRFiller is omitted from the Instant3dit dataset in Table 2 as it reduces to the SD Image Cond baseline under the fourview constraint. Overall, the 14B variant of ObjFiller-3D achieves the best performance across datasets. The qualitative results in Fig. 6 further verify that introducing the proposed Cycle-Consistent 3D Encoder consistently improves dense-view coherence and visual fidelity.

Number of Input Views. More incomplete views provide extra object information, but also impose greater constraints. As shown in Table 3, the performance of our inpainter improves with more inputs. Therefore, we argue that our method effectively handles the negative aspects introduced by additional views, while also benefiting from the positive information they provide.

Extend to 3D Scene. Leveraging our dense-view-based approach, we naturally extend it to 3D scene inpainting, which, like objectlevel inpainting, is fundamentally a mask-filling task. We evaluate our method on four diverse scenes, comparing it with NeRFiller and SPIn-NeRF [Mirzaei et al. 2023], as shown in Fig. 7. Unlike

- Table 3. Evaluation of our 3D inpainting method across different numbers of input views.

Input Views 80 100 120 140 PSNR ↑ 22.76 25.63 26.62 26.68 SSIM ↑ 0.89 0.92 0.93 0.93 LPIPS ↓ 0.11 0.08 0.07 0.06

- Table 4. Ablation study of Cycle-Consistent 3D Encoder (CCE) on 1.3B and 14B models. The inclusion of CCE significantly enhances reconstruction quality and semantic consistency.

Model Name Configuration FID ↓ LPIPS ↓ CLIP ↑ ObjFiller-3D-1.3B

w/o CCE 107.20 0.205 29.76 w/ CCE 92.07 0.190 29.87

w/o CCE 104.80 0.219 30.19 w/ CCE 90.75 0.195 30.19

ObjFiller-3D-14B

SPIn-NeRF, which assumes compact mask regions suited for object removal, our approach accommodates broader, unconstrained masks, enabling more general 3D inpainting and demonstrating greater applicability. In complex scenes, our method outperforms NeRFiller, as illustrated in the second row of Fig. 7.

Effectiveness of Cycle-Consistent 3D Encoder. A key component of our method is the proposed Cycle-Consistent 3D Encoder, which is trained under three types of multi-view-consistent masks. To assess their effectiveness, we conduct ablation studies focusing on this aspect. Fig. 8a shows the difference. The quantitative comparisons are provided in Table 4.

Applications. Since inpainting and editing are closely related tasks, our model can also be applied to object editing. Specifically, we can import our object into Blender and manually place a 3D geometric shape as the 3D mask in the desired location. Fig. 8b illustrates a replace example. Similarly, ObjFiller-3D supports add and remove.

Additional Results. We randomly select several outputs from our method and Instant3dit for qualitative comparison. Overall, our 14B model demonstrates superior consistency across multiple views. Figure 9 presents additional visualization results for comparative analysis with Instant3dit.

- 5 Conclusion

In this paper, we presented ObjFiller-3D, a novel framework for highquality and consistent 3D object inpainting. Unlike prior sparseview or per-view completion methods, our approach jointly models densely sampled views along a 360◦ trajectory, enabling coherent geometry and appearance across viewpoints. Our framework combines dense-view modeling, object-level completion, and global structural consistency within a unified architecture, while also supporting reference-guided 3D inpainting for controllable appearance editing. Extensive experiments showed that ObjFiller-3D consistently outperforms existing methods in reconstruction fidelity and perceptual quality. We believe our work provides a promising framework for scalable 3D content completion and editing.

Limitation. Our approach relies on foundation models for denseview generation, and is therefore inherently limited by their generative and consistency capabilities. We believe future advances in foundation models will further improve the quality and robustness of our framework.

References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report.

Amir Barda, Matheus Gadelha, Vladimir G Kim, Noam Aigerman, Amit H Bermano, and Thibault Groueix. 2025. Instant3dit: Multiview inpainting for fast editing of 3d objects. In Proceedings of the Computer Vision and Pattern Recognition Conference. 16273–16282.

Jun-Kun Chen and Yu-Xiong Wang. 2024. Proedit: Simple progression is all you need for high-quality 3d scene editing. Advances in Neural Information Processing Systems 37 (2024), 4934–4955.

Minghao Chen, Iro Laina, and Andrea Vedaldi. 2024b. Dge: Direct gaussian 3d editing by consistent multi-view editing. In European Conference on Computer Vision. Springer, 74–92.

Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. 2024a. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21476–21485.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 13142–13153.

Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo MartinBrualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. 2024. Cat3d: Create anything in 3d with multi-view diffusion models.

Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. 2023. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF international conference on computer vision. 19740–19750.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. Neural Information Processing Systems,Neural Information Processing Systems (Jan 2017).

Jonathan Ho and Tim Salimans. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022).

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).

Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, Zheng-Jun Zha, and Lei Zhang.

2023. Dreamtime: An improved optimization strategy for text-to-3d content creation. 14 pages.

Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. 2024. Mv-adapter: Multi-view consistent image generation made easy. Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. 2025. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598 (2025). James T Kajiya and Brian P Von Herzen. 1984. Ray tracing volume densities. ACM SIGGRAPH computer graphics 18, 3 (1984), 165–174. Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.

- 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
- 4 (2023), 139–1.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023a. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning. PMLR, 19730–19742.

Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. 2023b. Instant3d: Fast textto-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214 (2023).

Lin Li, Zehuan Huang, Haoran Feng, Gengxiong Zhuang, Rui Chen, Chunchao Guo, and Lu Sheng. 2025. VoxHammer: Training-Free Precise and Coherent 3D Editing in Native 3D Space. arXiv preprint arXiv:2508.19247 (2025).

Weiyu Li, Jiarui Liu, Hongyu Yan, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. 2024. Craftsman3d: High-fidelity mesh generation with 3d native generation and interactive geometry refiner.

Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2023. Syncdreamer: Generating multiview-consistent images from a single-view image.

Zhiheng Liu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jie Xiao, Kai Zhu, Nan Xue, Yu Liu, Yujun Shen, and Yang Cao. 2024. Infusion: Inpainting 3d gaussians via learning depth completion from diffusion prior.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2024. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 9970–9980.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. 2022. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11461–11471.

Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. 2023. Scalable 3d captioning with pretrained models. Advances in Neural Information Processing Systems 36 (2023), 75307–75337.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G Derpanis, Jonathan Kelly, Marcus A Brubaker, Igor Gilitschenski, and Alex Levinshtein. 2023. Spin-nerf: Multiview segmentation and perceptual inpainting with neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20669–20679.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent diffusion models for high-resolution image synthesis.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems 35 (2022), 36479–36494.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems 35 (2022), 25278–25294.

Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. 2021. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023a. Zero123++: a single image to consistent multi-view diffusion base model.

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2023b. Mvdream: Multi-view diffusion for 3d generation.

Zhihao Shi, Dong Huo, Yuhongze Zhou, Yan Min, Juwei Lu, and Xinxin Zuo. 2025. Imfine: 3d inpainting via geometry-guided multi-view refinement. In Proceedings of the Computer Vision and Pattern Recognition Conference. 26694–26703.

Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. 2022. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2149–2159.

Peng Wang and Yichun Shi. 2023. Imagedream: Image-prompt multi-view diffusion for 3d generation. Yuxuan Wang, Xuanyu Yi, Zike Wu, Na Zhao, Long Chen, and Hanwang Zhang.

2024. View-consistent 3d editing with gaussian splatting. In European conference on computer vision. Springer, 404–420.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems 36 (2023), 8406–8441.

Ethan Weber, Aleksander Holynski, Varun Jampani, Saurabh Saxena, Noah Snavely, Abhishek Kar, and Angjoo Kanazawa. 2024. Nerfiller: Completing scenes via generative 3d inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20731–20741.

Ethan Weber, Norman Müller, Yash Kant, Vasu Agrawal, Michael Zollhöfer, Angjoo Kanazawa, and Christian Richardt. 2026. Fillerbuster: Multi-View Scene Completion for Casual Captures.

Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. 2024. Meshlrm: Large reconstruction model for high-quality meshes.

Chung-Ho Wu, Yang-Jung Chen, Ying-Huan Chen, Jie-Ying Lee, Bo-Hsu Ke, ChunWei Tuan Mu, Yi-Chuan Huang, Chin-Yang Lin, Min-Hung Chen, Yen-Yu Lin, et al. 2025a. AuraFusion360: Augmented Unseen Region Alignment for Reference-based

360deg Unbounded Scene Inpainting. In Proceedings of the Computer Vision and Pattern Recognition Conference. 16366–16376.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, et al. 2025b. Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2025. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 21469–21480.

Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. 2024. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models.

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. 2024. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–20.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. doi:10.1109/cvpr.2018.00068

Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. 2023. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems 36 (2023), 49842–49869.

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

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

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

Masked scenes SPIn-NeRF NeRFiller Ours

Fig. 7. Qualitative results of scene inpainting. The leftmost column shows the scenes we need to process, where the areas to be repaired are masked in pink. We compare our method, ObjFiller-3D (the rightmost column), with other approaches.

[Figure 174]

[Figure 175]

w/o Cycle-Consistent 3D Encoder (14B)

w/o Cycle-Consistent 3D Encoder (1.3B)

[Figure 176]

Masked object

[Figure 177]

[Figure 178]

3D Mask

[Figure 179]

[Figure 180]

[Figure 181]

w/ Cycle-Consistent 3D

w/ Cycle-Consistent 3D

“Statue of Liberty lift

[Figure 182]

Encoder (14B)

Encoder (1.3B)

up a glass of beer”

(a) Cycle-Consistent 3D Encoder ablation.

(b) Object editing.

Fig. 8. Additional results and ablations of ObjFiller-3D.

[Figure 183]

Masked grid image Instant3dit Ours (1.3B) Ours (14B)

Fig. 9. Additional visualization results compared to Instant3dit. We randomly select several outputs from our method and Instant3dit for qualitative comparison. Overall, our 14B model demonstrates superior consistency across multiple views.

