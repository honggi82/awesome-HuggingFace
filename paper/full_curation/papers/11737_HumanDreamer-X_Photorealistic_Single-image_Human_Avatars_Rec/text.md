# arXiv:2504.03536v2[cs.CV]12Nov2025

## HumanDreamer-X: Photorealistic Single-image Human Avatars Reconstruction via Gaussian Restoration

Boyuan Wang1,2*, Runqi Ouyang1,2*, Xiaofeng Wang1,2*, Zheng Zhu1*†, Guosheng Zhao1,2 Chaojun Ni1,3, Xiaopei Zhang4, Guan Huang1, Yijie Ren2, Lihong Liu2, Xingang Wang2†

1GigaAI 2Institute of Automation, Chinese Academy of Sciences 3Peking University 4University of California Los Angeles

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

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

###### Reference Image Coarse Avatar Restored Avatar Animation

Figure 1. Illustration of HumanDreamer-X. The pipeline initiates with a single-image reconstruction to generate a coarse 3D avatar, providing essential geometric and appearance priority for the restoration process. This approach facilitates the attainment of a higherquality 3D avatar, suitable for subsequent animation tasks.

### Abstract

3D Gaussian Splatting serves as an explicit 3D representation to provide initial geometry and appearance priority. Building upon this foundation, HumanFixer is trained to restore 3DGS renderings, which guarantee photorealistic results. Furthermore, we delve into the inherent challenges associated with attention mechanisms in multi-view human generation, and propose an attention modulation strategy that effectively enhances geometric details identity consistency across multi-view. Experimental results demonstrate that our approach markedly improves generation and reconstruction PSNR quality metrics by 16.45% and 12.65%, respectively, achieving a PSNR of up to 25.62 dB, while also showing generalization capabilities on inthe-wild data and applicability to various human reconstruction backbone models.

Single-image human reconstruction is vital for digital human modeling applications but remains an extremely challenging task. Current approaches rely on generative models to synthesize multi-view images for subsequent 3D reconstruction and animation. However, directly generating multiple views from a single human image suffers from geometric inconsistencies, resulting in issues like fragmented or blurred limbs in the reconstructed models. To tackle these limitations, we introduce HumanDreamer-X, a novel framework that integrates multi-view human generation and reconstruction into a unified pipeline, which significantly enhances the geometric consistency and visual fidelity of the reconstructed 3D models. In this framework,

*These authors contributed equally to this work. †Corresponding authors. zhengzhu@ieee.org, xingang.wang@ia.ac.cn. ‡Project Page: https://humandreamer-x.github.io/

### 1. Introduction

Creating 3D human avatars is gaining increasing significance across various domains, including virtual reality, gaming, and film production. Among the numerous methods, creating from a single image represents a common, practical, and user-friendly approach. Nevertheless, constructing a versatile human avatar with diverse shapes, appearances, and clothing from just a single image remains a substantial challenge.

Traditional reconstruction methods based on mesh, Neural Radiance Fields (NeRF) [37] and 3D Gaussian Splatting (3DGS) [24] enable multi-view reconstruction but are incapable of achieving single-image reconstruction [18, 20, 23, 26, 27, 29, 55, 72].

Despite these advancements, single-image reconstruction remains particularly challenging. To address these limitations, current approaches for single-view human reconstruction often integrate techniques from image or video generation [12, 13, 28, 32, 35, 40, 42, 66, 71]. A common strategy, as seen in methods like PSHuman [28] and AniGS [42], involves first generating multi-view images using generative models [17, 49, 52, 65, 69] and then performing subsequent reconstruction with mesh-based or Gaussian-based techniques. However, this decoupled paradigm heavily depends on the geometric and appearance consistency of the generative model. Any inconsistency in these aspects can lead to severe artifacts in the reconstruction stage, such as fragmented or distorted limbs.

To alleviate the aforementioned issues, we introduce HumanDreamer-X, a novel framework that integrates multiview human generation and reconstruction into a unified pipeline. This integration facilitates mutual enhancement between the two processes, offering supplementary information on geometry and appearance. Consequently, this significantly improves the geometric consistency and visual fidelity of the reconstructed 3D models, effectively alleviating problems related to fragmented and blurred limbs in the generated models. As illustrated in Figure. 2, our framework first utilizes 3DGS to reconstruct the human body from a single image and then renders multi-view images. Leveraging the geometric representation capability of 3DGS, these rendered multi-view images provide strong geometric and appearance priority for the subsequent generative process. Building upon this, HumanFixer is introduced to refine the renderings, producing photorealistic images. Finally, these restored images are further utilized to guide 3DGS in reconstructing a high-quality human model. Compared to decoupled approaches, our unified framework effectively bridges the gap between reconstruction and generation, producing higher-quality avatars suitable for diverse downstream applications. Moreover, in the context of multi-view human generation, directly generating such videos often leads to blurriness because of temporal incon-

sistencies. To tackle this issue, we explore the inherent challenges associated with attention mechanisms within attention layers and propose a modulation strategy. This strategy effectively enhances geometric detail and identity consistency across multiple views, thus overcoming the limitations of traditional approaches.

Notably, our experiments demonstrate that our approach significantly improves the generation metrics by 16.45% in PSNR and the reconstruction metrics by 12.65% in PSNR. And also demonstrating its generalization capabilities on inthe-wild data and its adaptability to various human reconstruction backbone models.

The main contributions of this paper are summarized as follows:

- • We propose a novel framework, HumanDreamer-X, for generating animatable avatars from a single-view image by coupling 3D reconstruction with video restoration. This unified approach significantly enhances the quality and consistency of reconstructed avatars compared to existing decoupled methods.
- • We identify and address attention-related deficiencies in the generation of multi-view videos, which often lead to inconsistencies and blurriness. To mitigate these issues, we introduce an attention correction module that refines the temporal attention mechanism, thereby improving the quality of restored videos and ensuring better geometric and identity preservation.
- • Extensive experimental results demonstrate the superior performance of HumanDreamer-X across multiple datasets. Specifically, our method achieves higher fidelity in avatar reconstruction, showcasing its practical applicability and efficiency. It also demonstrates generalization capabilities on in-the-wild data and is applicable to various human reconstruction backbone models.

### 2. Related Work

#### 2.1. Single-image Human Recosntruction

With the advancement of 3D representation methods such as mesh [7, 21], NeRF [1–3, 37, 47], and 3DGS [24, 31, 58, 62], human reconstruction has seen significant improvements [9, 14, 16, 22, 38, 39, 41, 48, 68]. However, these methods typically require a large number of multi-view images or videos for reconstruction, which limits their applicability. In contrast, Single-image human reconstruction is a more flexible task, aiming to reconstruct a 3D human model from just one image [12, 13, 28, 35, 40, 42, 66, 71]. However, it is inherently an ill-posed problem, presenting considerable challenges.

Current single-image human reconstruction methods integrate techniques from generative approaches [4, 45, 59, 63, 70]. These methods first generate unseen viewpoints [13] or multi-view images [12, 28, 35, 40, 42], followed

by employing 3D representations to reconstruct the human. To ensure geometric consistency, methods such as SMPL [28, 42], masks [13], segmentation maps [12], and pose estimation [40] are utilized to drive the generation process. However, these generate-then-reconstruct approaches heavily rely on the geometric and appearance consistency of the generated images, and inconsistencies can lead to issues like fragmented limbs and blurriness in the reconstructed models. Recently, there have been attempts [71] to infer 3DGS parameters directly from a single image using feed-forward networks, but this requires extensive datasets for training. The approach most closely related to ours is SIFU [66], which refines the coarse texture of reconstructed models using generative models. However, SIFU uses a frozen image generation model for frame-by-frame refinement, making it difficult to maintain inter-frame consistency.

#### 2.2. Controllable Human Generation

With the rapid advancements in image and video generation [4–6, 33, 53, 57, 59, 63], diffusion-based human generation models [17, 19, 34, 36, 49, 50, 54, 60, 64, 65, 69] have gained increasing attention. Among these, Animate Anyone [17] stands out as a representative work, which constructs a 3D U-Net upon Stable Diffusion [45] for modeling temporal information, and incorporates skeleton information as driving signals for controlling motion generation.

To further enhance temporal consistency, MimicMotion [65] leverages pre-trained video generation model SVD [4] to improve the coherence of generated sequences. In contrast, UniAnimate [52] employs MAMBA-based [10] techniques for enhanced temporal modeling. Additionally, several approaches explore different motion control signals. For instance, DisCo [51] and MagicAnimate [56] utilize DensePose [11] as a representation of the human body [51, 56], while Champ [69] integrates multi-modal information including depth, normal, and semantic signals derived from the 3D parametric human model SMPL [30]. Despite their ability to generate photorealistic frames, these models often struggle to fully ensure geometric and appearance consistency across frames. This inconsistency poses significant challenges for subsequent reconstruction tasks.

### 3. Method 3.1. Preliminary

3D Gaussian Splatting. 3DGS represents a voxelbased rendering technique for scene representation, where the core concept involves modeling the scene using an optimized set of 3D Gaussian distributions G = {N(xi,Σi)}Ni=0. Each Gaussian distribution N is parameterized by its position xi ∈ R3, covariance matrix Σi (which defines the ellipsoidal shape), and radiance attributes such as color ci and opacity αi. Dur-

ing rendering, these Gaussian distributions are projected onto the image plane. The contribution of each pixel u is computed via radial basis functions defined as wi(u) = αi exp −12(u − ui)⊤Σ−i 1(u − ui) , where the weight wi decays with distance from the center of the Gaussian. This process is achieved through differentiable rasterization, which integrates depth sorting and weighted blending ( i wici/ i wi), thereby facilitating gradient-based optimization for high-quality scene reconstruction.

Video Diffusion Model. Video Diffusion Models (VDM) are generative frameworks that extend diffusion-based image synthesis to dynamic scenes by modeling sequential data in a latent space. The core principle involves a forward diffusion process that gradually corrupts video data x0 into noise via T steps: q(xt|xt−1) = N(xt;√1 − βtxt−1,βtI), where βt controls noise injection. VDM enhances temporal coherence by integrating spatiotemporal layers into a pretrained latent space, enabling high-fidelity video generation from images or text while leveraging efficient VAE-based compression. As a reverse denoising network it learns to approximate the posterior pθ(xt−1|xt), typically parameterized as xt−1 = µθ(xt,t) + σtϵ (ϵ ∼ N(0,I)), reconstructing coherent video frames through iterative refinement.

#### 3.2. Overall Framework of HumanDreamer-X

Traditional human reconstruction methodologies [18, 20, 23, 27, 29, 55] encounter substantial challenges due to their reliance on multi-view images. Recent advances [12, 13, 28, 32, 35, 40, 42, 66, 71] have endeavored to address this limitation by leveraging generative priors to compensate for the absence of invisible views. Nevertheless, the decoupling of generation from reconstruction has resulted in a deficiency of geometric consistency among the generated multi-view images. Our proposed framework, HumanDreamer-X, integrates reconstruction and generation into a unified pipeline. In this pipeline, the reconstruction step provides initial geometry and appearance priorities, and the generative model then refines the reconstructions by restoring coarse renderings. The overall framework is illustrated in Fig. 2.

Specifically, we first use the 3DGS model [18, 29] to reconstruct an avatar Ac from a single reference image IR:

Ac = 3DGS(IR,θSMPL), (1)

Following previous avatar reconstruction models, the initial point cloud for 3DGS is derived from the estimated Skinned Multi-Person Linear (SMPL) θSMPL. This ensures that even a single image can reconstruct coarse human geometry and appearance, providing a basic priority for subsequent refinement. Then, a multi-view video Vc of the avatar is rendered using the trained 3DGS avatar:

###### Vc = {Ac(di) | i = 1,2,...,n}, (2)

[Figure 25]

##### Reconstruction

##### Video Restoration

Attention Modulation

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Noise

|[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>… …<br><br>…<br><br>…<br><br>…| |
|---|---|
| | |

[Figure 47]

[Figure 48]

###### C

Enc

…

…

[Figure 49]

[Figure 50]

Render

[Figure 51]

[Figure 52]

[Figure 53]

🔥 HumanFixer

[Figure 54]

CLIP

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Dec

Coarse 3DGS Avatar

Rendered Video

[Figure 61]

Before Restoration

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Init

[Figure 66]

After Restoration

[Figure 67]

Reference Image

Coarse Stage

Refine Stage

Maintained 3DGS Restored Video

Update Refined 3DGS Avatar

Figure 2. Overall framework of the proposed HumanDreamer-X. The process begins by initializing a coarse 3DGS avatar using a reference image. A rendered video serves as a , providing geometric and appearance priors. Subsequently, HumanFixer performs video restoration, wherein an attention modulation is employed to enhance video consistency. Throughout this process, the restored video is used to continuously update the 3DGS model, ultimately resulting in a refined 3DGS avatar.

#### 3.3. Training and Inference of HumanFixer

where di represents a specific horizontal angle, and Ac(di) renders the avatar at that angle. Due to the limited availability of only one viewpoint, the resulting model Ac provides only basic geometric and appearance priority of the avatar, leaving issues such as blurriness and artifacts in unseen views unresolved. To address these issues, we introduce HumanFixer, a video generation model designed to restore details in the initial 3DGS renderings.

Reconstructed avatars derived directly from single-image inputs exhibit issues such as blurring at invisible views. To address these problems, we introduce HumanFixer for refining coarse avatars. This section will present the methodologies for the training and inference of HumanFixer.

Training. HumanFixer hinges on constructing a coarserefined pair dataset. In this section, we propose a concrete pipeline for dataset collection, as illustrated in Fig. 3. The steps are as follows: Initially, we utilize Blender to render a multi-view video Vgt from each 3D scan, serving as the ground truth video. Subsequently, we leverage the GS model to reconstruct human avatar from single image, and use Eq. 2 to render multi-view coarse images Vc, paired with their corresponding ground truth videos Vgt, form the repair dataset. This dataset is designed to refine avatar reconstructions by providing examples of both lowquality and high-quality renderings, allowing the HumanFixer model to learn how to enhance the coarse images into refined, detailed videos.

HumanFixer is built upon the pretrained video diffusion model [4], utilizing the coarse video Vc and the reference image IR as conditions to generate multi-view refined videos (see Sec. 3.3 for more details). The refined videos Vr capture the textures and geometry present in the reference image IR, making them suitable for modeling a refined human avatar Ar.

This approach leverages the geometry and appearance priority provided by 3DGS and exploits the temporal consistency inherent in video generation models, ensuring enhanced multi-view consistency in the repaired avatar. Additionally, to address blurriness in multi-view video generation, we investigate attention mechanisms and propose an attention modulation strategy (see Sec. 3.4 for more details). The refined video, enhanced through this strategy, is then utilized to optimize the human avatar.

The architecture of HumanFixer is illustrated in Fig. 2. It employs SVD [4] as the backbone and leverages the lowquality video to guide the generation process. Additionally, it integrates a CLIP [44] encoder to inject reference image information through cross-attention mechanisms.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

| | |
|---|---|
| | |

|[Figure 77]|
|---|

[Figure 78]

Supervise

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

🔥

HumanFixer

HumanFixer

[Figure 84]

[Figure 85]

GT Video

frame

frame

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

frame frame

[Figure 95]

w/o attention mask with attention mask

[Figure 97]

Figure 4. Attention weights visualization. The left and right sides show the head 0 attention weights at the temporal self-attention stage for training on cyclic videos without and with an attention modulation, respectively. Brighter colors indicate higher weights.

[Figure 98]

[Figure 99]

[Figure 101]

[Figure 103]

Maintained 3DGS

Coarse Rendered Video

Maintained 3DGS

frames increases. However, in multi-view videos, which encompass a full circle of views, the final frames have a strong relationship with the initial s. Given that our model is fine-tuned on SVD, which is pretrained on standard videos, the pretrained parameters align more closely with the assumption of strong correlations between adjacent frames.

Figure 3. The creation of the dataset for training HumanFixer. First, we use Blender to render scans and obtain the ground truth video. Next, we employ the frontal image and its corresponding SMPL prior to reconstruct a coarse 3DGS model, followed by rendering multi-view videos. This process yields paired video data for training.

|[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>…<br><br>…<br><br>…<br><br>…<br><br>…<br><br>frames.|
|---|

…

…

Attention Mask

For a multi-view video with a total of N views, we define a non-cyclic video as one where each frame corresponds sequentially to views 0,1,...,N − 1. Conversely, a cyclic video involves frames corresponding to views 0,1,...,N − 1,0, effectively looping back to the initial view.

During the training of HumanFixer, we first feed the coarse video Vc into a Variational Autoencoder (VAE) [25] encoder E to obtain the latent feature zc = E(Vc). Then, zc is used as a condition, providing both geometric and appearance priority. It is concatenated with zgt = E(Vgt), serving as the input to the model. To maintain identity consistency across multiple views, we utilize a reference image IR as a source of identity information. The face embedding of the reference image Mf ∈ Rn

Compared to non-cyclic videos, cyclic videos append the first frame to the end, making the training data more suitable for models pretrained on the assumption of strong correlations between adjacent frames. This should theoretically enhance overall consistency. However, during training, we observed that directly training on non-cyclic videos allows view 0 to develop stronger associations with views N − 1, N − 2, and N − 3. Nevertheless, a significant discontinuity was noted between view N − 1 and view 0, whereas the difference between view 1 and view 0 was relatively minor.

t×C is extracted using an existing facial embedding extraction model from IR. nt denotes number of tokens and C means dimension of crossattention. The model’s output is:

ϵtarget = hθ(zgtt ,zc,Mf), (3)

To address this issue, we further analyzed the temporal self-attention mechanism within the model. As illustrated in the left panel of Fig. 4, the attention weights for the starting and ending frames are significantly higher than those for intermediate frames. We hypothesize that this occurs because, in cyclic videos, the first and last frames are identical, leading to naturally higher attention weights compared to those calculated between different frames. This phenomenon suppresses information flow among intermediate frames.

Where hθ denotes the HumanFixer module with parameters θ, zgtt represents the latents of the ground truth video Vgt at the noising time step t, and zc signifies the latents of the coarse video Vc. As described in [4], we use a predicted target loss [4] for optimizing the HumanFixer model.

Inference. After training HumanFixer, the coarse video Vc to be refined and the reference image IR are used as inputs to obtain the refined video Vr.

Vr = EDM(hθ(zT,zc,Mf)), (4)

In summary, while cyclic videos aim to improve consistency by leveraging adjacency assumptions, they introduce challenges related to discontinuities and uneven attention weight distribution, necessitating further refinement of the temporal self-attention mechanism. To mitigate this issue, we apply an attention modulation, as shown in Fig. 2, which ensures that the first and last frames do not receive attention from the model, either from themselves or from each other.

where zT is the initial latent noise, and we use EDM scheduler [4] for denoising.

#### 3.4. Attention Mechanism Analysis

Generating multi-view videos differs from generating standard videos due to the distinct temporal relationships involved. In typical videos, adjacent frames are strongly correlated, with correlation decreasing as the distance between

Formally, let the attention mask M ∈ R(N+1)×(N+1) be

Table 1. Multi-view video generation comparison of other SOTA methods. Bold indicate the best result.

Method Testset PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ PSHuman [28]

21.998 0.826 0.1945 103.808 Champ [69] 20.228 0.889 0.2438 115.031 HumanFixer (Ours) 25.618 0.882 0.0687 87.149

CustomHumans

17.547 0.859 0.2701 129.629 HumanFixer (Ours) 23.741 0.889 0.0720 94.570

Champ [69]

THuman2.1

defined as:

M(i,j) = −∞, if (i,j) = (0,0),(0,N),(N,0),(N,N) 0, otherwise.

(5)

Then, this attention mask is added to every temporal selfattention module in the model:

Attn(Q,K,V,M) = softmax

QKT √dk

+ M V, (6)

where Q for query, K for key, V for value, dk the dimension for scaling down the dot product results, M denotes the attention mask on temporal self attention.

This formulation effectively suppresses attention weights for the first and last frames, preventing the model from being unduly influenced by these boundary frames during video generation. Fig. 4 illustrates that, after training on cyclic videos with the attention modulation, the attention weights shift from being predominantly focused between the 0th and Nth frames to resulting in a more evenly distributed attention pattern.

### 4. Experiments

This section outlines our experimental framework, including the datasets, implementation specifics, and evaluation criteria. We then provide both quantitative and qualitative results to highlight the outstanding performance of the proposed HumanDreamer-X.

#### 4.1. Experiment Setup

Dataset. Our experiments are conducted on a variety of datasets. To ensure that the human subject occupies a significant portion of the frame during training, we filter out instances where the arms are excessively spread, which would otherwise result in an overabundance of background content. The final training set comprises 388 scans from CustomHumans [15] and 1929 scans from THuman2.1 [61]. For testing, we use 39 samples from [15], 32 samples from [61]. All training data is standardized to a resolution of 960 × 640, with cyclic video sequences consisting of 19 frames (18 multi-view frames plus the first frame repeated).

To ensure the reconstruction of a complete avatar, it is stipulated that the input to HumanFixer must include facial

information as identity cues. For the THuman2.1 dataset [61], facial detection is performed using InsightFace [8], and the image with the largest facial area is designated as the reference image. For the CustomHumans dataset [15], the first frame is directly selected as the reference image.

Baselines. For multi-view video generation, we utilize PSHuman [28] and Champ [69] as baselines. For 3D avatar reconstruction, we compare our method with a variety of existing single-image human reconstruction approaches. These include mesh-based methods such as PIFU [46], PaMIR [67], SiTH [13], and PSHuman [28], as well as recent feed-forward 3D Gaussian Splatting methods like IDOL [71] and LHM [43]. In contrast to these end-to-end approaches, our framework adopts a two-stage generationthen-reconstruction pipeline. To validate the effectiveness of our framework across different generation strategies, we use either Champ [69] or HumanDreamer-X for the initial human prior generation stage, while keeping the subsequent 3D reconstruction stage fixed—implemented on two distinct 3DGS backbones: Animatable Gaussians [29] and GaussianAvatar [18].

#### 4.2. Main Results

Quantity comparison results on multi-view generation. We compare the multi-view generation performance of HumanFixer with several SOTA methods, and the results are presented in Tab. 1. The findings indicate that HumanFixer achieves superior video consistency. Fig. 5 visually illustrates the enhanced multi-view consistency produced by our method.

Quantity comparison results on 3D reconstruction. To evaluate the 3D reconstruction quality, we render the reconstructed models from 18 different views and calculate the PSNR, LPIPS, SSIM, and FID metrics. We compare our approach against various baselines to demonstrate the broad applicability of our proposed framework. As shown in Tab. 2, our framework supports the integration of more advanced 3DGS backbones for enhanced reconstruction performance, thereby achieving superior results. We also conducted a visualization comparison experiment, as shown in Fig. 6. Compared to other methods, ours delivers higher detail fidelity while mitigating issues like poor identity con-

[Figure 120]

[Figure 121]

Reference Champ PSHuman HumanFixer (Ours)

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

custom

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

custom

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

thuman2317

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

| |
|---|

character 38

Figure 5. Comparison of generation with SOTA methods. Note that PSHuman’s training dataset contains all of the CustomHumans. Best viewed with zoom-in.

character 34⼩丑

Table 2. 3D reconstruction comparison. * denotes the metric is from PSHuman[28].

考虑更换Thuman/character，因为Pshuman⽤Customhumans做训练集

Method Testset PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ PSHuman

⽣成效果⽐较

20.089 0.8439 0.1770 87.816 LHM 20.153 0.9122 0.1121 110.341

- IDOL 18.333 0.9209 0.0882 187.710 Champ with GaussianAvatar 19.673 0.8789 0.2643 164.554 HumanDreamer-X with GaussianAvatar 23.639 0.9100 0.2427 114.804 Champ with Animatable gaussians 16.853 0.9157 0.1251 122.752 HumanDreamer-X with Animatable gaussians 22.631 0.9458 0.0729 71.250 PIFU∗

THuman2.1

19.3957 0.8327 0.1001 PaMIR∗ 19.4130 0.8324 0.0988 SiTH∗ 18.458 0.8200 0.1004 PSHuman∗ 20.855 0.8636 0.0764 LHM 19.275 0.8913 0.1062 139.566

- IDOL 19.348 0.9321 0.0919 201.377 Champ with GaussianAvatar 18.264 0.8842 0.2639 129.413 HumanDreamer-X with GaussianAvatar 19.328 0.8945 0.2578 132.200 Champ with Animatable gaussians 18.908 0.9328 0.1278 176.836 HumanDreamer-X with Animatable gaussians 21.091 0.9403 0.0968 78.174

CustomHumans

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

IDOL

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

| |
|---|

PSHumanGaussians

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

withAnimatable

Gaussians

Champ

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

HumanDreamer-X

withAnimatable

LHM

Figure 6. Comparison of 3D reconstruction with SOTA methods. Best viewed with zoom-in.

cus619 cus642 thuman2328

重建⽐较

character41

###### Table 5. Ablation Study on Attention Modulation in the Reconstruction Stage using Animatable Gaussians [29].

sistency, missing hands, and limb sticking artifacts. Specifically, compared to the second-best method, PSHuman, our approach shows improvements in PSNR, SSIM, LPIPS and FID by 12.65%, 12.07%, 58.81% and 18.86% on CustomHumans.

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓

w/o attention mask (non-cyclic) 21.309 0.9398 0.0867 112.812 w/o attention mask (cyclic) 20.867 0.9351 0.0955 139.693 w attention mask (cyclic) 22.631 0.9458 0.0729 71.250

Table 3. Ablation Study on the Use of Coarse 3DGS.

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ w/o coarse 3DGS 16.824 0.621 0.3122 170.659 w/ coarse 3DGS 25.618 0.882 0.0687 87.149

Table 4. Ablation Study on Attention Modulation in the Generation Stage on the CustomHumans Subset.

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ w/o attention mask (non-cyclic) 25.514 0.885 0.0704 95.065 w/o attention mask (cyclic) 25.667 0.800 0.1453 106.705 w attention mask (cyclic) 25.618 0.882 0.0687 87.149

#### 4.3. Ablation Study

We conduct an ablation study to evaluate the importance of coarse 3DGS-based multi-view reconstruction as a conditioning signal during training. Specifically, we train the generation model without using the coarse 3DGS rendering as input condition. As shown in Tab. 3, the significant performance drop in all metrics, demonstrates the critical role of this geometric prior in guiding high-fidelity human video generation. These results validate that coarse 3DGS rendering provides effective structural supervision, significantly improving the quality and consistency of the generated multi-view videos.

Furthermore, we train HumanFixer on the CustomHumans dataset [15], comparing non-cyclic videos (18 frames) and cyclic videos (19 frames) with and without attention modulation, under both multi-view video generation (see Tab. 4) and 3D reconstruction settings (see Tab. 5). Experimental results show that while video quality degrades when transitioning from non-cyclic to cyclic inputs, the incorporation of attention modulation notably improves performance, achieving the best overall results. This confirms the effectiveness of our proposed attention mechanism in enhancing the fidelity and temporal coherence of generated avatar sequences.

### 5. Limitations and Conclusions

Limitations. Our method follows a two-stage pipeline: it first reconstructs a coarse human avatar, followed by restoration and final refinement. As described in the paper, this approach results in relatively high computational cost. Depending on the chosen 3DGS baseline, each full human avatar modeling process takes approximately 7 to 15 minutes, with the reconstruction stage alone accounting for about 90% of the total time. Employing faster reconstruction techniques could help alleviate this computational bottleneck.

Conclusions. Single-image human reconstruction is vital for digital human modeling but remains challenging due to geometric inconsistencies and limited visual fidelity. To address these issues, we propose HumanDreamer-X, a unified framework for multi-view human generation and reconstruction. It leverages 3DGS for robust initialization and introduces HumanFixer to refine geometry and appearance, along with an attention modulation strategy to enhance multi-view consistency. Experimental results show that our method significantly improves reconstruction quality, generalizes well to in-the-wild data, and is compatible with various backbone models.

### References

- [1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, 2021. 2
- [2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022.
- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. In ICCV, 2023. 2
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3, 4, 5

- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023.
- [6] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In CVPR, 2022. 3
- [7] Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, et al. Meshanything: Artist-created mesh generation with autoregressive transformers. arXiv preprint arXiv:2406.10163,

2024. 2

- [8] Jiankang Deng, Jia Guo, Xue Niannan, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019. 6
- [9] Zijian Dong, Xu Chen, Jinlong Yang, Michael J Black, Otmar Hilliges, and Andreas Geiger. Ag3d: Learning to generate 3d avatars from 2d image collections. In ICCV, 2023. 2
- [10] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 3
- [11] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In CVPR, 2018. 3
- [12] Xu He, Xiaoyu Li, Di Kang, Jiangnan Ye, Chaopeng Zhang, Liyang Chen, Xiangjun Gao, Han Zhang, Zhiyong Wu, and Haolin Zhuang. Magicman: Generative novel view synthesis of humans with 3d-aware diffusion and iterative refinement. arXiv preprint arXiv:2408.14211, 2024. 2, 3
- [13] I Ho, Jie Song, Otmar Hilliges, et al. Sith: Single-view textured human reconstruction with image-conditioned diffusion. In CVPR, 2024. 2, 3, 6
- [14] Fangzhou Hong, Zhaoxi Chen, Yushi Lan, Liang Pan, and Ziwei Liu. Eva3d: Compositional 3d human generation from 2d image collections. arXiv preprint arXiv:2210.04888,

2022. 2

- [15] Jie Song Hsuan-I Ho, Lixin Xue and Otmar Hilliges. Learning locally editable virtual humans. In CVPR, 2023. 6, 9
- [16] Hezhen Hu, Zhiwen Fan, Tianhao Wu, Yihan Xi, Seoyoung Lee, Georgios Pavlakos, Zhangyang Wang, et al. Expressive gaussian human avatars from monocular rgb video. NeurIPS,

2025. 2

- [17] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In CVPR, 2024. 2, 3
- [18] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In CVPR,

2024. 2, 3, 6

- [19] Li Hu, Guangyuan Wang, Zhen Shen, Xin Gao, Dechao Meng, Lian Zhuo, Peng Zhang, Bang Zhang, and Liefeng Bo. Animate anyone 2: High-fidelity character image animation with environment affordance. arXiv preprint arXiv:2502.06145, 2025. 3

- [20] Shoukang Hu, Tao Hu, and Ziwei Liu. Gauhuman: Articulated gaussian splatting from monocular human videos. In CVPR, 2024. 2, 3
- [21] Tao Hu, Liwei Wang, Xiaogang Xu, Shu Liu, and Jiaya Jia. Self-supervised 3d mesh reconstruction from single images. In CVPR, 2021. 2
- [22] Mustafa I¸sık, Martin R¨unz, Markos Georgopoulos, Taras Khakhulin, Jonathan Starck, Lourdes Agapito, and Matthias Nießner. Humanrf: High-fidelity neural radiance fields for humans in motion. ACM ToG, 2023. 2
- [23] Tianjian Jiang, Xu Chen, Jie Song, and Otmar Hilliges. Instantavatar: Learning avatars from monocular video in 60 seconds. In CVPR, 2023. 2, 3
- [24] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM ToG, 2023. 2
- [25] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. 5
- [26] Yong-Hoon Kwon, Ju Hong Yoon, and Min-Gyu Park. Text2avatar: Articulated 3d avatar creation with text instructions. IEEE Transactions on Multimedia, 27:3797–3806,

2025. 2

- [27] Jiahui Lei, Yufu Wang, Georgios Pavlakos, Lingjie Liu, and Kostas Daniilidis. Gart: Gaussian articulated template models. In CVPR, 2024. 2, 3
- [28] Peng Li, Wangguandong Zheng, Yuan Liu, Tao Yu, Yangguang Li, Xingqun Qi, Mengfei Li, Xiaowei Chi, Siyu Xia, Wei Xue, et al. Pshuman: Photorealistic single-view human reconstruction using cross-scale diffusion. arXiv preprint arXiv:2409.10141, 2024. 2, 3, 6, 7
- [29] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling. In CVPR, 2024. 2, 3, 6, 8
- [30] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers. 2023. 3
- [31] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In CVPR, 2024. 2
- [32] Yixing Lu, Junting Dong, Youngjoong Kwon, Qin Zhao, Bo Dai, and Fernando De la Torre. Gas: Generative avatar synthesis from a single image, 2025. 2, 3
- [33] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 3
- [34] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia, 2024. 3
- [35] Yifang Men, Biwen Lei, Yuan Yao, Miaomiao Cui, Zhouhui Lian, and Xuansong Xie. En3d: An enhanced generative model for sculpting 3d humans from 2d synthetic data. In CVPR, 2024. 2, 3

- [36] Yifang Men, Yuan Yao, Miaomiao Cui, and Liefeng Bo. Mimo: Controllable character video synthesis with spatial decomposed modeling. arXiv preprint arXiv:2409.16160,

2024. 3

- [37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021. 2
- [38] Arthur Moreau, Jifei Song, Helisa Dhamo, Richard Shaw, Yiren Zhou, and Eduardo P´erez-Pellitero. Human gaussian splatting: Real-time rendering of animatable avatars. In CVPR, 2024. 2
- [39] Panwang Pan, Zhuo Su, Chenguo Lin, Zhen Fan, Yongjie Zhang, Zeming Li, Tingting Shen, Yadong Mu, and Yebin Liu. Humansplat: Generalizable single-image human gaussian splatting with structure priors. NeurIPS, 2024. 2
- [40] Hao-Yang Peng, Jia-Peng Zhang, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu. Charactergen: Efficient 3d character generation from single images with multi-view pose canonicalization. ACM ToG, 2024. 2, 3
- [41] Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, and Siyu Tang. 3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting. In CVPR, 2024. 2
- [42] Lingteng Qiu, Shenhao Zhu, Qi Zuo, Xiaodong Gu, Yuan Dong, Junfei Zhang, Chao Xu, Zhe Li, Weihao Yuan, Liefeng Bo, et al. Anigs: Animatable gaussian avatar from a single image with inconsistent gaussian reconstruction. arXiv preprint arXiv:2412.02684, 2024. 2, 3
- [43] Lingteng Qiu, Xiaodong Gu, Peihao Li, Qi Zuo, Weichao Shen, Junfei Zhang, Kejie Qiu, Weihao Yuan, Guanying Chen, Zilong Dong, and Liefeng Bo. Lhm: Large animatable human reconstruction model from a single image in seconds.

2025. 6

- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 4
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3
- [46] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2304–2314, 2019. 6
- [47] Shuai Shen, Wanhua Li, Xiaoke Huang, Zheng Zhu, Jie Zhou, and Jiwen Lu. Sd-nerf: Towards lifelike talking head animation via spatially-adaptive dual-driven nerfs. IEEE Transactions on Multimedia, 26:3221–3234, 2024. 2
- [48] Qingping Sun, Yi Xiao, Jie Zhang, Shizhe Zhou, Chi-Sing Leung, and Xin Su. A local correspondence-aware hybrid cnn-gcn model for single-image human body reconstruction. IEEE Transactions on Multimedia, 25:4679–4690, 2023. 2
- [49] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. Animate-x: Universal character image ani-

- mation with enhanced motion representation. arXiv preprint arXiv:2410.10306, 2024. 2, 3
- [50] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In ECCV, pages 244–260. Springer, 2024. 3
- [51] Tan Wang, Linjie Li, Kevin Lin, Yuanhao Zhai, ChungChing Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Disco: Disentangled control for realistic human dance generation. arXiv preprint arXiv:2307.00040,

2023. 3

- [52] Xiang Wang, Shiwei Zhang, Changxin Gao, Jiayu Wang, Xiaoqiang Zhou, Yingya Zhang, Luxin Yan, and Nong Sang. Unianimate: Taming unified video diffusion models for consistent human image animation. arXiv preprint arXiv:2406.01188, 2024. 2, 3
- [53] Xiaofeng Wang, Zheng Zhu, Guan Huang, Boyuan Wang, Xinze Chen, and Jiwen Lu. Worlddreamer: Towards general world models for video generation via predicting masked tokens. arXiv preprint arXiv:2401.09985, 2024. 3
- [54] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694, 2024. 3
- [55] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In CVPR, 2022. 2, 3
- [56] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In CVPR, 2024. 3
- [57] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 3
- [58] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In CVPR,

2024. 2

- [59] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 3
- [60] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [61] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qionghai Dai, and Yebin Liu. Function4d: Real-time human volumetric capture from very sparse consumer rgbd sensors. In CVPR, 2021. 6
- [62] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In CVPR, 2024. 2
- [63] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 3

- [64] Shufang Zhang, Minxue Ni, Shuai Chen, Lei Wang, Wenxin Ding, and Yuhong Liu. A two-stage personalized virtual tryon framework with shape control and texture guidance. IEEE Transactions on Multimedia, 26:10225–10236, 2024. 3
- [65] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. arXiv preprint arXiv:2406.19680, 2024. 2, 3
- [66] Zechuan Zhang, Zongxin Yang, and Yi Yang. Sifu: Sideview conditioned implicit function for real-world usable clothed human reconstruction. In CVPR, 2024. 2, 3
- [67] Zerong Zheng, Tao Yu, Yebin Liu, and Qionghai Dai. Pamir: Parametric model-conditioned implicit representation for image-based human reconstruction. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–1, 2021. 6
- [68] Zerong Zheng, Xiaochen Zhao, Hongwen Zhang, Boning Liu, and Yebin Liu. Avatarrex: Real-time expressive fullbody avatars. ACM ToG, 2023. 2
- [69] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In ECCV, 2024. 2, 3, 6
- [70] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, Chi Zhang, et al. Is sora a world simulator? a comprehensive survey on general world models and beyond. arXiv preprint arXiv:2405.03520, 2024. 2
- [71] Yiyu Zhuang, Jiaxi Lv, Hao Wen, Qing Shuai, Ailing Zeng, Hao Zhu, Shifeng Chen, Yujiu Yang, Xun Cao, and Wei Liu. Idol: Instant photorealistic 3d human creation from a single image. arXiv preprint arXiv:2412.14963, 2024. 2, 3, 6
- [72] Xinxin Zuo, Sen Wang, Jiangbin Zheng, Weiwei Yu, Minglun Gong, Ruigang Yang, and Li Cheng. Sparsefusion: Dynamic human avatar modeling from sparse rgbd images. IEEE Transactions on Multimedia, 23:1617–1629, 2021. 2

