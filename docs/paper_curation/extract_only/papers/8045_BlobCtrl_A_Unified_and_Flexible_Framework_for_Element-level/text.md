## BlobCtrl: Taming Controllable Blob for Element-level Image Editing

YAOWEI LI, SECE, Peking University, China LINGEN LI, The Chinese University of Hong Kong, China ZHAOYANG ZHANG†, ARC Lab, Tencent, China XIAOYU LI∗, ARC Lab, Tencent, China GUANGZHI WANG, ARC Lab, Tencent, China HONGXIANG LI, The Hong Kong University of Science and Technology, China XIAODONG CUN, GVC Lab, Great Bay University, China YING SHAN, ARC Lab, Tencent, China YUEXIAN ZOU†, SECE, Peking University, China

# arXiv:2503.13434v2[cs.CV]1Oct2025

Element-level Editing

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

[Figure 25]

[Figure 26]

[Figure 27]

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

[Figure 42]

[Figure 43]

[Figure 44]

Addition Translation Translation Reduction Enlargement

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

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Removal Removal Replacement Replacement Combined

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

Source Image Replacement Removal Reduction Reduction + Translation

Multi-round Interactive Editing

Fig. 1. BlobCtrl enables comprehensive element-level editing, supporting diverse operations such as addition, translation, scaling, removal, replacement, and their arbitrary combinations (top). Via iterative refinement, BlobCtrl achieves precise, fine-grained control to realize the desired visual outcomes (bottom).

∗Project Lead. †Corresponding Author.

Kong University of Science and Technology, Hongkong, China, lihxxxxxx@gmail.com; Xiaodong Cun, GVC Lab, Great Bay University, China, vinthony@gmail.com; Ying Shan, ARC Lab, Tencent, China, yingsshan@tencent.com; Yuexian Zou, SECE, Peking University, China, zouyx@pku.edu.cn.

Authors’ Contact Information: Yaowei Li, liyaowei01@gmail.com, SECE, Peking University, China; Lingen Li, The Chinese University of Hong Kong, Hongkong, China, lgli@link.cuhk.edu.hk; Zhaoyang Zhang, ARC Lab, Tencent, China, zhaoyangzhang@ link.cuhk.edu.hk; Xiaoyu Li, ARC Lab, Tencent, China, xliea@connect.ust.hk; Guangzhi Wang, ARC Lab, Tencent, China, guangzhi.wang@u.nus.edu; Hongxiang Li, The Hong

for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed

As user expectations for image editing continue to rise, the demand for flexible, fine-grained manipulation of specific visual elements presents a challenge for current diffusion-based methods. In this work, we present BlobCtrl, a framework for element-level image editing based on a probabilistic blob-based representation. Treating blobs as visual primitives, BlobCtrl disentangles layout from appearance, affording fine-grained, controllable object-level elements manipulation. Our key contributions are twofold: 1) an in-context dual-branch diffusion model that separates foreground and background processing, incorporating blob representations to explicitly decouple layout and appearance; and 2) a self-supervised disentangle-thenreconstruct training paradigm with an identity-preserving loss function, along with tailored strategies to efficiently leverage blob-image pairs. To foster further research, we introduce BlobData for large-scale training, and BlobBench, a benchmark for systematic evaluation. Experimental results demonstrate that BlobCtrl achieves state-of-the-art performance in a variety of element-level editing tasks—such as object addition, removal, scaling, and replacement—while maintaining computational efficiency.

[Figure 101]

[Figure 102]

#### Geometry Statistics

𝒃

𝚺

𝒂

𝜽 𝒙, 𝒚

[Figure 103]

[Figure 104]

𝝁

[Figure 105]

Interconvertible

Fig. 2. Blob Formula. A blob can be represented in two equivalent forms: geometrically as an ellipse and statistically as a 2D Gaussian distribution. The two forms are exactly equivalent and interchangeable.

The essence of element-level visual representation lies in the flexible decoupling of layout and visual appearance. To this end, BlobCtrl employs blobs as visual primitives to make the layout and appearance of the edited elements controllable. Formally, a blob is a probabilistic two-dimensional Gaussian distribution [Carson et al. 1999], and geometrically, it appears as an ellipse [Nie et al. 2024]. While prior works [Epstein et al. 2022; Nie et al. 2024] use blobs to specify layouts for image synthesis from scratch, we further tame blobs to enable precise layout rearrangement and appearance replacement for fine-grained element-level editing, leveraging their 5-DoF (x, y, a, b, 𝜃) and opacity-aware operations to accurately control position, scale, and orientation.

CCS Concepts: • Computing methodologies → Computer vision. Additional Key Words and Phrases: Artificial Intelligence Generated Content, Computer Vision, Video Customization ACM Reference Format:

Yaowei Li, Lingen Li, Zhaoyang Zhang, Xiaoyu Li, Guangzhi Wang, Hongxiang Li, Xiaodong Cun, Ying Shan, and Yuexian Zou. 2025. BlobCtrl: Taming Controllable Blob for Element-level Image Editing. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 14 pages. https://doi.org/10. 1145/3757377.3763897

We propose an in-context dual-branch diffusion architecture that decouples foreground and background processing using a blobbased representation. To better utilize blob-image pairs and avoid artifacts commonly seen in methods trained on video data, we introduce a self-supervised disentangle-then-reconstruct training paradigm with a carefully designed identity-preserving optimization objective. Additionally, we introduce several tailored strategies: random data augmentation to prevent the model from falling into copypaste local optima, and random feature dropout to enable more flexible diffusion inference. These design choices make BlobCtrl an efficient, flexible solution for element-level image editing.

- 1 Introduction

Element-level image editing aims to achieve fine-grained refinement of the layout and appearance of visual elements in existing images. While recent generative models[Esser et al. 2024; Labs 2023; Ramesh

- et al. 2022; Sheynin et al. 2024; Shi et al. 2024; Yu et al. 2025] excel in high-quality image synthesis and editing, they often lack a straightforward approach for fine-grained control over individual visual elements. Conventional controllable generative approaches [Li et al. 2023; Wang et al. 2024; Ye et al. 2023; Zhang et al. 2023b] introduce spatial conditions (such as edge maps, bounding boxes) or identity conditions (like reference images or ID features) to generate new images from scratch. However, these methods cannot modify the layout and appearance of existing images, nor do they support interactive, multi-round, element-based editing operations such as visual element rearrangement.

To scale up our method and ensure comprehensive evaluation, we introduce a new training dataset, BlobData, and a benchmark, BlobBench. Extensive quantitative and qualitative results demonstrate BlobCtrl’s effectiveness in fine-grained element-level editing (addition, translation, scaling, removal, and replacement).

Recent methods [Alzayer et al. 2024; Li et al. 2025; Mao et al. 2025; Mu et al. 2025; Shi et al. 2023; Song et al. 2025; Zhang et al. 2023a] have explored fine-grained visual editing through optimization, segmentation, clustering, and drag-based approaches. However, these methods lack robust and flexible editing capabilities due to two main limitations: 1) undesirable changes in unedited regions during the editing process, and 2) reliance on video data for training, which leads to artifacts in edited content (e.g., failed inpainting of the original location when moving elements).

In a nutshell, our main contributions include:

- • We propose BlobCtrl, a novel approach that tames blobs as visual primitives to enable precise and flexible visual element editing, while effectively preserving their intrinsic characteristics.
- • We introduce aself-superviseddisentangle-then-reconstruct training paradigm with an identity-preserving loss function, along with tailored strategies to efficiently leverage blob-image pairs.
- • We introduce BlobData, a comprehensive dataset specifically curated for training blob-based editing, alongside BlobBench, a rigorous benchmark for assessing element-level editing capabilities.
- • Through extensive experimentation, we demonstrate that BlobCtrl achieves superior performance compared to existing methods in element-level editing tasks, while maintaining computational efficiency and practical applicability.

republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. SA Conference Papers ’25, Hong Kong, Hong Kong

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2137-3/2025/12 https://doi.org/10.1145/3757377.3763897

- 2 Related Works Image Editing. Prompt-based image editing methods [Brooks

- et al. 2023; Cao et al. 2023; Hertz et al. 2023; Huang et al. 2024; Li et al. 2024; Shi et al. 2024] primarily rely on text as editing instructions. Reference-based image editing methods [Gal et al. 2022; Kumari et al. 2023; Ruiz et al. 2023; Wang et al. 2024; Ye et al. 2023] focus on preserving the visual appearance of reference images in new scenarios. Most relevant to our work are spatial-based editing methods, which typically employ per-sample optimization algorithms [Yenphraphai et al. 2024; Zhang et al. 2023a], point-based drag methods [Lu et al. 2024; Mou et al. 2023, 2024; Shi et al. 2023; Shin et al. 2024], grounding-based approaches (such as bounding boxes) [Chen et al. 2023; Xiong et al. 2024], compositing-based algorithms [Alzayer et al. 2024], and VAE decoupling methods [Mu et al. 2025]. While these methods demonstrate capabilities in object manipulation and attribute manipulation, they often struggle to achieve effective element-level editing operations such as addition, translation, scaling, removal, and replacement within a unified framework.

Blob-basedControllableSynthesis. Earlyworkestablishedblobs as mid-level primitives for controllable synthesis, primarily in indoor scenes. BlobGAN [Epstein et al. 2022] first leveraged unsupervised learning to decompose scenes into blobs, enabling layoutlevel control; BlobGAN-3D [Wang et al. 2023] extended this paradigm to 3D, enabling control over camera and 3D object locations. Diffusion-based methods further leveraged blob parameters as conditioning signals for text-to-image generation in BlobGEN [Nie et al. 2024]. DiffUHaul [Avrahami et al. 2024] further employs a trainingfree procedure to adapt BlobGEN for object dragging in images. BlobGEN-3D [Liu et al. 2024] formalized blobs as a compositional, 3D-consistent representation to lift 2D scenes into 3D and support free-view synthesis, while BlobGEN-VID [Feng et al. 2025] used blobs as grounding cues for compositional text-to-video generation. In contrast, we target image editing rather than generation: we treat blobs as manipulable visual primitives that disentangle layout from appearance, enabling precise element-level operations on existing images with strong identity preservation.

- 3 Method

Sec. 3.1 introduces the blob-related formulations as foundational knowledge. Sec. 3.2 presents the architecture of our model, while Sec. 3.3 and 3.4 elaborate on the carefully designed training paradigm and strategies tailored for effective learning.

- 3.1 Blob-Based Element-level Representation Blob Formula. Fig. 2 illustrates a blob. Geometrically, a blob

can be modeled as an ellipse parameterized by 𝒆𝜏 = [𝐶𝑥,𝐶𝑦,𝑎,𝑏,𝜃], where (𝐶𝑥,𝐶𝑦) denote the center coordinates,𝑎 and𝑏 are the lengths of the semi-minor and semi-major axes, respectively, and 𝜃 ∈ [0,𝜋) is theorientation. Statistically,ablob is modeled as a two-dimensional

Gaussian distribution with mean 𝝁 = [𝜇𝑥, 𝜇𝑦] and covariance 𝚺 =

𝜎𝑥𝑥 𝜎𝑥𝑦 𝜎𝑥𝑦 𝜎𝑦𝑦

, where 𝜎𝑥𝑥 and 𝜎𝑦𝑦 are the variances along the 𝑥 and

𝑦 directions, and 𝜎𝑥𝑦 is the covariance indicating the correlation between 𝑥 and 𝑦.

Blob Opacity. Notably, representing the blob as a Gaussian enables the calculation of opacity across spatial dimensions [Epstein et al. 2022]. In particular, the squared Mahalanobis distance [Mahalanobis 1936] to the blob center is computed as:

𝑑𝑀(𝒙grid, 𝑸) = (𝒙grid − 𝝁)𝑇 𝚺−1(𝒙grid − 𝝁), (1)

where 𝒙grid ∈ 𝑊 𝑤 , 𝐻ℎ

denotes a two-dimensional coordinate map over the image grid, and 𝑸 = (𝝁, 𝚺) are the parameters of the blob’s bivariate Gaussian. The distance 𝑑𝑀 ∈ R𝐻×𝑊 is the corresponding distance map that quantifies how far each grid point is from the center 𝝁 while accounting for the shape encoded by 𝚺. Specifically, for each grid index (𝑤,ℎ), 𝑑𝑀 [𝑤,ℎ] =

𝑤=1..𝑊,ℎ=1..𝐻

𝒙grid[𝑤,ℎ] − 𝝁 𝑇 𝚺−1 𝒙grid[𝑤,ℎ] − 𝝁 . Then, the blob opacity is defined based on this distance:

𝑂(𝒙grid) = sigmoid(−𝑑𝑀), (2)

which maps the distance𝑑𝑀 to values in (0, 1). This yields a smooth, center-peaked opacity that gradually decays toward the edges.

Blob Composition and Splatting. Blob splatting [Epstein et al. 2022] projects 𝑖𝑡ℎ feature vectors 𝒇𝑖 ∈ R𝑑 into a 2D grid with composed blob opacities 𝑂𝑐𝑖 ∈ R𝐻×𝑊 , producing spatially-aware features 𝑭𝒊 ∈ R𝐻×𝑊 ×𝑑. With blobs ordered by depth, the composed opacity, modeling inter-blob occlusion, is

𝑚

𝑂𝑐𝑖 = 𝑂𝑖 ⊙

𝑗=𝑖+1

and per-blob splatting is

1 − 𝑂𝑗 , (3)

𝑭𝑖 = 𝑔splatting(𝒇𝑖,𝑂𝑐𝑖 ) = 𝑂𝑐𝑖 ⊗ 𝒇𝑖, (4)

where ⊙ denotes element-wise multiplication on maps and 1 ∈ R𝐻×𝑊 is the all-ones map. In Eq. (4), the map–vector product uses outer-product broadcasting, i.e., (𝑂𝑐𝑖 ⊗ 𝒇𝑖)[ℎ,𝑤, :] = 𝑂𝑐𝑖 [ℎ,𝑤] 𝒇𝑖.

3.2 In-Context Dual Branch Architecture

Overview. Our approach addresses element-level image editing by segmenting the target object as the foreground element and constructing a background through dual masking—removing both the original and target positions of the foreground element. We define foreground as countable "things" (e.g., birds, dogs) and background as uncountable "stuff" regions (e.g., sky, grass), assuming one foreground and one background element per image for simplicity. We design a dual-branch architecture that processes foreground and background separately, where composed opacities 𝑂𝑐0 (background) and 𝑂𝑐1 (foreground) encode their respective layouts. To enhance flexible control over foreground elements, we splat DINOv2 [Oquab et al. 2023] features with the foreground opacity 𝑂𝑐1, yielding spatially-aware foreground semantics map 𝑭1. The foreground branch extracts hierarchical features that are progressively fused into the background branch, enabling fine-grained controllable editing through blob representations. Henceforth, we use subscript 1 for foreground and 0 for background.

[Figure 106]

[Figure 107]

[Figure 108]

##### Dual-Branch Diffusion Model

###### Legend

###### Foreground Element

[Figure 109]

[Figure 110]

Output Layer

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

(Training Only) Random Augment & Encode

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Noisy Latent

[Figure 120]

[Figure 121]

[Figure 122]

Score Function

🔗

[Figure 123]

[Figure 124]

[Figure 125]

Random Blob

Self Attn

Conv

Simulating Blob Manipulation via

[Figure 126]

t

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Blob to Mask & Encode t Timestep Embedding A Cross Attention C Channel Concatenate

###### Splatting C

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Weighted Sum

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Text Encoder

[Figure 143]

[Figure 144]

Self + Cross Attn

[Figure 145]

🔗

Width Concatenate

[Figure 146]

A

[Figure 147]

[Figure 148]

Full Fine-tune LoRA Fine-tune Foreground Blob

Conv

[Figure 149]

[Figure 150]

[Figure 151]

Noisy Latent

[Figure 152]

t

[Figure 153]

🔗

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Score Function

M

C

[Figure 158]

[Figure 159]

[Figure 160]

Background Blob

Background Element

- Fig. 3. Overview of BlobCtrl. Our framework employs a dual-branch architecture: a foreground branch for identity encoding and a background branch for scene preservation and fusion. Inputs are concatenated in an in-context manner (Sec. 3.2), and the model is trained using the proposed strategy (Sec. 3.3).

Foreground Branch. The foreground branch extracts control-

The background branch uses a complete diffusion backbone with cross-attention layers. The input projection layer is modified to handle the dimensionally-changed input 𝑿0𝑡. We employ hierarchical feature fusion, progressively injecting foreground features at multiple resolution levels using zero-initialization [Zhang et al. 2023b] Z—initializing the linear layer weights between foreground and background fusion to zero. Feature fusion for the 𝑖-th block is formulated as:

lable features for injection into the background branch. Let cat0(·) denote channel-wise concatenation and cat𝑤(·) denote in-context concatenation along the width axis. We construct the inputs as

𝑪1 = cat0 𝒛1, 𝑂𝑐1, 𝑭1 ∈ R(𝑐+1+𝑑)×ℎ×𝑤, 𝑿1𝑡 = cat𝑤 𝑪1, cat0 𝒛𝑡1, 𝑂𝑐1, 𝑭1 ∈ R(𝑐+1+𝑑)×ℎ×2𝑤,

(5)

where 𝒛1 ∈ R𝑐×ℎ×𝑤 are foreground VAE latents, 𝑂𝑐1 ∈ R1×ℎ×𝑤 is the foreground composed opacity, 𝑭1 ∈ R𝑑×ℎ×𝑤 is the foreground semantic feature map, and 𝒛𝑡1 ∈ R𝑐×ℎ×𝑤 is the noisy foreground latent at timestep 𝑡.

𝝐𝜃𝑖,enhanced(𝑡,𝑿0𝑡,𝑿1𝑡) = 𝝐𝜃𝑖,bg(𝑡,𝑿0𝑡) + 𝜔 · Z(𝝐𝜃𝑖,fg(𝑡,𝑿1𝑡)), (8)

where 𝑿0𝑡 and 𝑿1𝑡 are the input conditions for the background and foreground branches, respectively, and 𝜔 is a hyperparameter con-

trolling the fusion strength. For clarity, we omit text-conditioning inputs in the formulation.

We use a modified pre-trained diffusion backbone without crossattention layers to process the foreground input. The input projection layer is modified to handle the dimensionally-changed input 𝑿1𝑡. This design leverages pre-trained weights for effective foreground feature processing while focusing solely on visual content.

3.3 Self-supervised Training Paradigm

Disentangle-then-Reconstruct. Obtaining element-level paired supervision for realistic edit operations is challenging and costly. Prior works [Alzayer et al. 2024; Chen et al. 2023] turn to video proxies, which introduce confounds that degrade performance. We therefore adopt a self-supervised Disentangle-then-Reconstruct paradigm: we treat each existing image as a post-edit result, disentangle the foreground element from the background, and construct dual masks that remove the element at both a hypothesized pre-edit source and the actual target. We then reconstruct by inpainting background at the source and synthesizing the foreground at the target to enforce scene harmony. Concretely, for each image we identify the foreground element’s blob in the image as the target (post-edit) state, and sample a synthetic pre-edit blob by randomly perturbing its parameters (center/scale/orientation), upon which we form dual masks for source and target, as illustrated in Fig. 3. We optimize our model using a noise-prediction objective during training:

The foreground branch extracts hierarchical features at multiple resolution levels through the diffusion backbone. For the 𝑖-th bottleneck block, the extracted features are:

𝝐𝜃𝑖,fg(𝑡,𝑿1𝑡) ∈ R𝑐𝑖×ℎ𝑖×𝑤𝑖, (6)

where𝑐𝑖,ℎ𝑖, and𝑤𝑖 are the channel, height, and width dimensions at the𝑖-th resolution level, respectively. These hierarchical features are progressively injected into the background branch for integration.

Background Branch. The background branch integrates foreground elements into the scene for controllable generation. Unlike the foreground branch which processes only the foreground region 𝒛𝑡1, the background branch operates on the entire image latent 𝒛𝑡 for proper scene integration. We concatenate the noisy latent 𝒛𝑡 with reference background conditions 𝑪0 via in-context format:

0,𝑿1𝑡,𝜖∼N(0,I) ∥𝜖 − 𝜖𝜃enhanced(𝑡,𝑿0𝑡,𝑿1𝑡)∥22 , (9)

𝑪0 = cat0 𝒛0, 𝑂𝑐0 ∈ R(𝑐+1)×ℎ×𝑤, 𝑿0𝑡 = cat𝑤 𝑪0, cat0 𝒛𝑡, 𝑂𝑐0 ∈ R(𝑐+1)×ℎ×2𝑤,

L = E𝑿𝑡

(7)

Here, 𝑿0𝑡 and 𝑿1𝑡 are the background and foreground in-context inputs constructed per Eq. (7) and Eq. (5). This loss drives the model

where 𝒛0 ∈ R𝑐×ℎ×𝑤 are background VAE latents and 𝑂𝑐0 ∈ R1×ℎ×𝑤 is the background composed opacity.

to synthesize the foreground at the target while inpainting the background at the source, ensuring scene harmony.

Identity Preservation Loss Function. We impose an identitypreservation loss on the foreground branch to disentangle responsibilities: the foreground branch preserves element-level identity, while the background branch focuses on scene harmonization. During training, the foreground head predicts the noise over masked regions; at inference, we disable this head. Concretely, given the

foreground head prediction 𝝐𝜃fg(𝑡,𝑿1𝑡), the loss is

2 2

1,𝜖∼N(0,I) 𝑀1 ⊙ 𝜖 − 𝝐𝜃fg(𝑡,𝑿1𝑡)

, (10)

Lid = E𝑿𝑡

where 𝑀1 ∈ {0, 1}𝐻×𝑊 is the binary foreground mask, and 𝑿1𝑡 is defined in Eq. (5). The overall training objective is

Ltotal = L + 𝜆id Lid, (11)

where 𝜆id controls the strength of identity preservation. We decay 𝜆id from 1.0 to 0.6 over training, which shifts emphasis toward scene harmonization in later stages while retaining identity consistency.

- 3.4 Tailored Training Strategies

Random Data Augmentation. To prevent naive copy–paste behavior, we apply extensive augmentations to foreground elements during training, including color jittering, scaling, rotation, random erasing, and perspective transforms. These augmentations (i) compel the model to place foregrounds harmoniously under diverse layouts and appearances, and (ii) strengthen inpainting robustness for incomplete elements. This fosters flexible, context-aware manipulation while maintaining coherence with the background.

Random Dropout. With probability 𝑝𝜔 we disable foreground– background fusion by setting 𝜔 ← 0 in Eq. (8); with probabilities 𝑝feat and 𝑝vae we set 𝑭1←0 and 𝒛1←0 in Eq. (5). At inference, these hyperparameters can be user-set (e.g., adjust 𝜔 to modulate identity preservation, toggle 𝑭1 or 𝒛1 to trade semantics vs. appearance).

4 Experiments

- 4.1 Datasets, Benchmark and Metrics

BlobData Curation. Building on the BrushData dataset with instance segmentation [Ju et al. 2024], we curate BlobData (1.86M samples) by filtering images and masks, annotating blob parameters, and generating captions. Specifically: (1) Retain images whose shorter side exceeds 480 pixels. (2) Keep masks with area ratios in [0.01, 0.9] of the image area and not touching image boundaries. (3) Fit ellipse parameters to each mask 1 and derive 2D Gaussian. (4) Discard samples with ill-conditioned covariance (below 1e-5). (5) Generate detailed captions using InternVL-2.5 [Chen et al. 2024].

BlobBench Curation. Existing benchmarks [Lin et al. 2014; Ruiz et al. 2023; Yang et al. 2023; Zhang et al. 2024] evaluate either grounding capability or identity preservation, but not both. They also do not cover the full spectrum of element-level manipulations (addition, translation, scaling, removal, and replacement). To bridge these gaps, we present BlobBench, a benchmark of 100 curated images evenly spanning the five operation types. Each image is annotated with ellipse parameters, a foreground mask, and expert-written detailed descriptions. BlobBench contains both real-world and AI-generated

1https://docs.opencv.org/4.x/de/d62/tutorial_bounding_rotated_ellipses.html

images across diverse scenarios (indoor and outdoor scenes, animals, landscapes), enabling fair and comprehensive evaluation.

Evaluation Metrics. For objective evaluation, we assess:

- • Identity Preservation. We employ CLIP-I [Radford et al. 2021] and DINO-I [Caron et al. 2021] scores to measure the appearance similarity between objects in generated and reference images by extracting and comparing object-level features. For the Removal task, we denote CLIP-I∗ and DINO-I∗ in the table, with smaller values indicating cleaner removal.
- • Grounding Accuracy. To assess layout control, we extract masks from generated images using SAM [Kirillov et al. 2023], fit ellipses to these masks, and measure the Mean Squared Error (MSE) against the ground-truth annotations to quantify spatial accuracy.
- • Generation Quality. We use standard image-quality metrics (FID [Heusel et al. 2017], PSNR [Wikipedia contributors 2024], SSIM [Wang et al. 2004], LPIPS [Zhang et al. 2018]) to assess image quality and harmonization.

For human evaluation, we conducted a user study in which 10 participants each assessed 20 result sets. For each metric (fidelity, layout accuracy, and visual harmony), participants selected the single best result among the candidates.

4.2 Implementation Details.

Training Details. BlobCtrl builds on Stable Diffusion v1.5 [Rombach et al. 2022]. All images and annotations are resized to 512×512 pixels. We initialize both foreground and background branches with pretrained UNet weights. The foreground branch is fully fine-tuned, and the background branch is fine-tuned using LoRA [Hu et al. 2021] (rank=64). We use the Adam optimizer [Kingma and Ba 2014] with a learning rate of 1e-5 and weight decay of 0.01. Training is conducted on our curated BlobData for 7 days using 24 NVIDIA V100 GPUs with a batch size of 192. To control the fidelity–diversity trade-off, we set dropout probabilities 𝑝𝜔, 𝑝feat, and 𝑝vae to 0.1. The identity-preservation loss weight 𝜆id is gradually decayed from 1.0 to 0.6 during training. We use a caption dropout of 0.1 to enable classifier-free guidance at inference.

Evaluation Details. We evaluate BlobCtrl on the BlobBench benchmark against six representative open-source baselines. For each editing type, the inputs consist of: (i) the image, including a foreground and the corresponding hole-filled background (for addition, we directly provide the clean background and foreground); (ii) blob parameters specifying the initial and target layouts; and (iii) the target foreground element for addition and replacement.

We categorize the baselines into two groups. General Methods include grounding-based approaches (GliGEN [Li et al. 2023], Anydoor [Chen et al. 2023]) and compositing-based methods (Magic Fixup [Alzayer et al. 2024]). Translation-only Methods consist of point-based dragging approaches for images (DiffEditor [Mou et al. 2024], InstantDrag [Shin et al. 2024]) and videos ( DragAnything [Wu et al. 2024]). These methods perform only positional edits, since only such edits can be easily specified using point annotations. Artifacts in InstantDrag and DragAnything prevent reliable object segmentation; as a result, standard object-level metrics (e.g., DINOI) are computed on the entire image instead of individual objects,

- Table 1. Comprehensive comparison of general-purpose methods. This table quantitatively compares our method against Anydoor [Chen et al. 2023], GliGEN [Li et al. 2023], and Magic Fixup [Alzayer et al. 2024] on multiple element-level manipulations. We evaluate identity preservation (CLIP-I/DINO-I), grounding accuracy (MSE), and removal completeness (CLIP-I∗/DINO-I∗). ↑ indicates higher is better, while ↓ indicates lower is better.

Method

Addition Translation Scaling Replacement Removal

CLIP-I↑ DINO-I↑ MSE↓ CLIP-I↑ DINO-I↑ MSE↓ CLIP-I↑ DINO-I↑ MSE↓ CLIP-I↑ DINO-I↑ MSE↓ CLIP-I∗ ↓ DINO-I∗ ↓ Anydoor 86.7 81.2 6.7 85.4 81.7 6.8 83.3 83.7 9.6 81.7 80.2 9.7 39.5 13.6

GliGEN 70.7 57.8 6.9 71.2 62.4 7.1 78.2 69.4 9.7 68.4 60.6 9.6 40.2 15.3 Magic Fixup 83.7 84.5 6.6 86.0 83.7 6.8 85.5 84.2 9.2 84.5 81.2 9.2 37.2 9.7

BlobCtrl (Ours) 88.3 86.9 6.4 88.9 87.8 6.3 86.5 89.1 8.9 86.2 86.0 9.0 35.3 8.6

- Table 2. Comparison with point-based dragging methods on the translation task [Mou et al. 2024; Shin et al. 2024; Wu et al. 2024]. N/A indicates object localization failed, making the metric incomputable.

Method CLIP-I↑ DINO-I↑ MSE↓ InstantDrag 80.6 77.7 N/A DragAnything 65.2 50.4 N/A DiffEditor 78.7 71.8 6.9 BlobCtrl (Ours) 88.9 87.8 6.3

- Table 3. Comparison of generation quality [Alzayer et al. 2024; Chen et al. 2023; Li et al. 2023; Mou et al. 2024; Shin et al. 2024; Wu et al. 2024].

Method Edit Type PSNR↑ SSIM↑ LPIPS↓ FID↓ Anydoor General 32.06 0.742 0.239 145.3 GliGEN General 27.92 0.241 0.696 307.8 Magic Fixup General 31.36 0.748 0.223 131.7 BlobCtrl (Ours) General 32.16 0.751 0.220 102.8 InstantDrag Translation Only 17.38 0.680 0.185 141.9 DragAnything Translation Only 10.75 0.282 0.611 245.7 DiffEditor Translation Only 24.22 0.956 0.100 122.8 BlobCtrl (Ours) Translation Only 29.48 0.975 0.081 74.6

while grounding accuracy, which cannot be meaningfully converted to an image-level metric, is omitted.

Baseline Details. For Anydoor, originally designed for maskguided foreground insertion with harmonization, we adopt a twopass strategy: (i) inpaint hole-filled backgrounds by feeding them as both foreground and background inputs to obtain a clean background, and (ii) use an operation-specific mask to insert the true foreground at the target location. For GliGEN, whose boundingbox-conditioned insertion cannot handle hole-filled backgrounds, we first recover a clean background via our removal operation, and then insert the foreground at the specified bounding box. For Magic Fixup, a compositing-based harmonization method, we apply rigid transformations to foreground elements according to the editing operation before harmonization. For point-based dragging methods, we use the blob centroids before and after editing as start and end positions to form the dragging points input.

- 4.3 Quantitative Evalution

Table 4. Human evaluation results [Alzayer et al. 2024; Chen et al. 2023; Li et al. 2023; Mou et al. 2024; Shin et al. 2024; Wu et al. 2024].

Method Edit Type Fidelity↑ Layout↑ Harmony↑

Anydoor General 7.5% 9.5% 8.0% GliGEN General 3.0% 4.0% 5.5% Magic Fixup General 10.0% 11.5% 8.0% BlobCtrl (Ours) General 79.5% 75% 78.5%

InstantDrag Translation Only 3.0% 2.0% 1.0% DragAnything Translation Only 1.5% 2.0% 1.5% DiffEditor Translation Only 11.0% 13.5% 17.5% BlobCtrl (Ours) Translation Only 84.5% 82.5% 80.0%

- • Identity Preservation: For general methods, BlobCtrl achieves substantially higher identity scores on tasks that require preserving elements (addition, translation, scaling, replacement), with average CLIP-I of 87.48 and DINO-I of 87.45, outperforming the previous best baseline, Magic Fixup (84.93 and 83.40). For removal tasks, it attains lower CLIP-I∗ and DINO-I∗ scores (avg. 21.95 vs. 23.45), indicating more complete elimination of target elements. In addition, for translation-only tasks, BlobCtrl consistently surpasses all drag-based methods.
- • Grounding Accuracy: BlobCtrl demonstrates superior spatial control, achieving a lower average layout MSE than the previous best method, Magic Fixup (7.65 vs. 7.95), corresponding to a 3.8% relative improvement. This highlights the effectiveness of our blob-based representation for precise element-level manipulation.
- • Generation Quality: BlobCtrl achieves state-of-the-art performance across standard image quality metrics. For general elementlevel editing, it attains PSNR 32.16, SSIM 0.751, LPIPS 0.220, and FID 102.8, outperforming all baselines and demonstrating superior global fidelity and realism. For translation-only tasks, our method achieves PSNR 29.48, SSIM 0.975, LPIPS 0.031, and FID 74.6, consistently surpassing all drag-based methods and highlighting its ability to maintain high-fidelity outputs.

We attribute these significant improvements to two key contributions: (1) a high-DoF blob-based representation, enabling precise control over element position, scale, and orientation; and (2) a selfsupervised disentangle-then-reconstruct framework, supported by a tailored dual-branch architecture and specialized training strategies, which effectively decouples identity from layout while ensuring robust and harmonious element-level editing.

Comparison to State-of-the-Art Methods. As shown in Tab. 1, Tab. 2 and Tab. 3, BlobCtrl demonstrates consistent and significant improvements over existing methods across all evaluation metrics:

Human Evaluation. The results of Tab. 4 demonstrate the consistent superiority of BlobCtrl across all assessment criteria. For general element-level editing, BlobCtrl achieves higher preference

(a) General Methods

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

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

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

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

MagicFixupBlobCtrl(Ours)AnydoorGliGENMove polar bear.

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Addition Translation Scaling Replacement Removal

(b) Translation-only Methods

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Input Image InstantDrag DragAnything DiffEditor BlobCtrl (Ours)

- Fig. 4. Element-level editing comparison across methods. (a) General Methods supporting diverse element-level operations; (b) Translation-only Methods limited to point-based object relocation. Please zoom in to view source images and manipulation instructions in detail.

rates than existing baselines, with Fidelity 79.5% (vs 10.0%), Layout 75.0% (vs 11.5%), and Harmony 78.5% (vs 8.0%). For translation-only tasks, BlobCtrl also outperforms all drag-based methods, achieving Fidelity 84.5%, Layout 82.5%, and Harmony 80.0%.

- 4.4 Qualitative Evaluation

- Fig. 4 shows qualitative comparisons between BlobCtrl and state-ofthe-art methods. Several consistent observations can be made:

- • General methods. GliGEN [Li et al. 2023] offers layout control but often breaks identity consistency. Anydoor [Chen et al. 2023] and Magic Fixup [Alzayer et al. 2024] produce plausible edits but with lower accuracy and visual coherence than ours.
- • Translation-only methods. InstantDrag [Shin et al. 2024] fails with large displacements, DragAnything [Wu et al. 2024] tends to misinterpret translation as camera motion, and DiffEditor [Mou et al. 2024] often compromises identity preservation.

In contrast, BlobCtrl consistently preserves identity, ensures precise layout control, and generalizes well across diverse scenarios while maintaining visual coherence.

[Figure 241]

[Figure 242]

[Figure 243]

TranslationScaling

AdditionalcomparisonswithTranslation-only methods are shown in Fig. 8, where our approach achieves the best results. Fig. 9 illustrates that by adjusting the hyperparameter 𝜔 (Eq. 8) and the input prompt, our method can switch between reference-followed edits and text-prompt-driven appearance edits. Figures 10 and 11 present additional element-level editing results under more complex settings, including diverse edit types (e.g., translate+rotate, replace+scale, translate+scale), challenging scenes (e.g., underwater, crowded scenes, occlusions, shadows, reflections), and varied styles (e.g., real-world, anime, LEGO). Our method produces consistently visually satisfactory results.

[Figure 244]

[Figure 245]

[Figure 246]

Input Image Bounding-box Blob

Fig. 7. Ablation on Blob Representation. Replacing blobs with bounding boxes reduces layout flexibility, while blobs better preserve shapes (e.g., reduced wings) and yield more plausible edits (hat relocation).

4.5 Ablation Studies

Ablation of Foreground–Background Fusion. Fig. 5 presents an ablation study on foreground–background fusion by varying key hyperparameters: fusion weight 𝜔 (Eq. (8)), fusion step ratio 𝑡𝜏 (fraction of diffusion steps with foreground–background fusion), and foreground inputs 𝒛1 and 𝑭1 (Eq. (5)). Results show that our method enables flexible control over the trade-off between semantic alignment and identity preservation, producing diverse outputs.

the foreground branch output corresponding to 𝜆id—an output not used during inference. This loss acts as a regularizer, encouraging the foreground branch to focus on foreground content and decoupling the functions of the foreground and background branches.

Ablation on Blob Representation. To evaluate their effectiveness, we replace blobs with bounding boxes (Fig. 7). While bounding boxes are the standard representation for objects, they only have 4-DoF (x, y, w, h), which limits their ability to represent complex shapes. In contrast, blobs have 5-DoF (x, y, a, b, 𝜃), allowing them to better capture irregular shapes and fine details. As a result, our method, which utilizes blobs, offers superior control over object deformation and produces more realistic outcomes (see top of Fig. 7).

[Figure 247]

[Figure 248]

[Figure 249]

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

𝒕𝝉 ∈ [𝟎,𝟎.𝟐] 𝒕𝝉 ∈ [𝟎,𝟎.𝟒] 𝒕𝝉 ∈ [𝟎.𝟎,𝟎.𝟔] 𝒕𝝉 ∈ [𝟎.𝟎,𝟎.𝟖] 𝒕𝝉 ∈ [𝟎.𝟎,𝟏.𝟎]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

In addition, our blobs are both geometrically and statistically well-defined, interpretable, and interchangeable—taking the form of ellipses geometrically and 2D Gaussian distributions statistically (see Sec. 1 and Sec. 2 of the supplementary materials). This welldefined representation enables smooth and coherent transitions when using blob opacity to represent layouts (Section 3.1), allowing more precise handling of object details, better preservation of shape, and more natural visual results (see bottom of Fig. 7.

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

###### 𝝎=0.2 𝝎=0.4 𝝎=0.6 𝝎=0.8 𝝎=1.0

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

𝒛𝟏 = 𝟎 𝒛𝟏 = 𝟎 𝑭𝟏 = 𝟎 𝑭𝟏 = 𝟎 𝒛𝟏 = 𝟎 , 𝑭𝟏 = 𝟎

5 Limitations and Conclusions

- Fig. 5. Foreground–Background Fusion Ablation. Effect of fusion step

ratio 𝑡𝜏, fusion weight 𝜔, and foreground inputs 𝒛1, 𝑭1 on identity preservation and semantic alignment, showing flexible control and diverse outputs.

Source Image w/o ℒ𝒊𝒅 w/. ID ℒ𝒊𝒅 Foreground Branch Output

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

ℒ =0.0399 ℒ =0.0235

- Fig. 6. Ablation of Identity Preservation Loss. Results of full-image denoising loss and foreground branch outputs.

We present BlobCtrl, a flexible framework for element-level editing based on a probabilistic blob representation. Blobs encode spatial information, enabling precise element-level manipulation. With a novel self-supervised dual-branch architecture and customized techniques, BlobCtrl achieves consistent edits, controllable flexibility, and state-of-the-art performance on BlobBench.

Despite its strong capabilities, BlobCtrl supports only iterative single-element operations within a single forward pass. Nevertheless, the blob-based representation naturally extends to depth-aware composition, suggesting promising directions for future work.

Acknowledgments

This work is supported by NSFC (No. 62176008), Tencent University Relations (Tencent AI Lab RBFR2024006) and Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology (Grant No. 2024B1212010006).

### AblationofIdentityPreservation Loss Function. Fig. 6presents

an ablation of the Identity Preservation Loss 𝜆id (Eq. 10): without it, the model converges slower (full-image denoising loss 0.0399 vs. 0.0235) and produces lower-quality outputs. We additionally decode

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Move white bird.Move metal knife.Move magazine.Move lantern.

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

Move top hat.

Input Image Editable Blob InstantDrag DragAnything DiffEditor BlobCtrl (Ours)

Fig. 8. Additional comparison with translation-only methods. InstantDrag [Shin et al. 2024] and DragAnything [Wu et al. 2024] fail, while DiffEditor [Mou et al. 2024] shows lower fidelity compared to our method.

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

Reduce wing.

Input Image Editable Blob Angel’s Wing Steel Wing Curly Wing

Source Prompt

Fig. 9. Results with different text prompts. The foreground branch is disabled (setting 𝜔 in Eq. 8 to 0), and different prompts guide image edting.

Complex Environment

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

CarttoonStyle

Complex Environment

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

Realistic Style

Complex Environment

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

Lego Style

Complex Environment

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

AI Style

Complex Environment

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Anime Style

[Figure 436]

[Figure 437]

Output Image Input Image Editable Blob Output Image

Input Image Editable Blob

Fig. 10. Editing results under complex settings. We perform diverse element-level edits—including combined operations such as translation+scale, translation+rotate, and replace+translation—across challenging scenarios (e.g., underwater, crowded scenes, occlusion) and various styles (AI, anime, real-world, LEGO). Our method produces visually plausible results.

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

Remove chess.

| |
|---|

| |
|---|

Reflection

Remove

Editable Blob Output Image

Editable Blob Output Image

Input Image

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

Remove

Shadow

| |
|---|

Editable Blob Output Image

Editable Blob Output Image

Fig. 11. Results of reflection and shadow removal. In this setting, shadows and reflections are treated as blob entities and iteratively removed.

References

Hadi Alzayer, Zhihao Xia, Xuaner Zhang, Eli Shechtman, Jia-Bin Huang, and Michael Gharbi. 2024. Magic Fixup: Streamlining Photo Editing by Watching Dynamic Videos. arXiv preprint arXiv:2403.13044 (2024).

Omri Avrahami, Rinon Gal, Gal Chechik, Ohad Fried, Dani Lischinski, Arash Vahdat, and Weili Nie. 2024. Diffuhaul: A training-free method for object dragging in images. In SIGGRAPH Asia 2024 Conference Papers. 1–12.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18392–18402.

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 22560–22570.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 2021. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 9650–9660.

Chad Carson, Megan Thomas, Serge Belongie, Joseph M Hellerstein, and Jitendra Malik. 1999. Blobworld: A system for region-based image indexing and retrieval. In Visual Information and Information Systems: Third International Conference, VISUAL’99 Amsterdam, The Netherlands, June 2–4, 1999 Proceedings 3. Springer, 509–517.

Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. 2023. AnyDoor: Zero-shot Object-level Image Customization. arXiv preprint (2023).

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. 2024. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 (2024).

Dave Epstein, Taesung Park, Richard Zhang, Eli Shechtman, and Alexei A Efros. 2022. Blobgan: Spatially disentangled scene representations. In European Conference on Computer Vision. Springer, 616–635.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning.

Weixi Feng, Chao Liu, Sifei Liu, William Yang Wang, Arash Vahdat, and Weili Nie. 2025. Blobgen-vid: Compositional text-to-video generation with blob video representations. In Proceedings of the Computer Vision and Pattern Recognition Conference. 12989–12998.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2023. Prompt-to-Prompt Image Editing with Cross-Attention Control. In ICLR.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in Neural Information Processing Systems (NIPS) 30 (2017).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. 2024. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8362–8371.

Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. 2024. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In European Conference on Computer Vision. Springer, 150–168.

Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu.

2023. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1931–1941.

Black Forest Labs. 2023. FLUX. https://github.com/black-forest-labs/flux. Yaowei Li, Yuxuan Bian, Xuan Ju, Zhaoyang Zhang, Ying Shan, Yuexian Zou, and

Qiang Xu. 2024. BrushEdit: All-In-One Image Inpainting and Editing. arXiv preprint arXiv:2412.10316 (2024).

Yaowei Li, Xiaoyu Li, Zhaoyang Zhang, Yuxuan Bian, Gan Liu, Xinyuan Li, Jiale Xu, Wenbo Hu, Yating Liu, Lingen Li, et al. 2025. IC-Custom: Diverse Image Customization via In-Context Learning. arXiv preprint arXiv:2507.01926 (2025).

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. 2023. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22511–22521.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer, 740–755.

Chao Liu, Weili Nie, Sifei Liu, Abhishek Badki, Hang Su, Morteza Mardani, Benjamin Eckart, and Arash Vahdat. 2024. Blobgen-3d: Compositional 3d-consistent freeview image generation with 3d blobs. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Jingyi Lu, Xinghui Li, and Kai Han. 2024. Regiondrag: Fast region-based image editing with diffusion models. In European Conference on Computer Vision. Springer, 231– 246.

PC Mahalanobis. 1936. On the generalized distance in Statistics. National Institute of Science of India.

Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. 2025. Ace++: Instruction-based image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487 (2025).

- Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. 2023. DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models. arXiv:2307.02421 [cs.CV]
- Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. 2024. Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8488–8497.

Jiteng Mu, Michaël Gharbi, Richard Zhang, Eli Shechtman, Nuno Vasconcelos, Xiaolong Wang, and Taesung Park. 2025. Editable image elements for controllable synthesis. In European Conference on Computer Vision. Springer, 39–56.

Weili Nie, Sifei Liu, Morteza Mardani, Chao Liu, Benjamin Eckart, and Arash Vahdat.

2024. Compositional Text-to-Image Generation with Dense Blob Representations. In Forty-first International Conference on Machine Learning.

Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2023. DINOv2: Learning Robust Visual Features without Supervision.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1, 2 (2022), 3.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In CVPR.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 22500–22510.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. 2024. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8871–8879.

Yichun Shi, Peng Wang, and Weilin Huang. 2024. SeedEdit: Align Image Re-Generation to Image Editing. arXiv preprint arXiv:2411.06686 (2024). Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai.

2023. DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing. arXiv preprint arXiv:2306.14435 (2023).

Joonghyuk Shin, Daehyeon Choi, and Jaesik Park. 2024. Instantdrag: Improving interactivity in drag-based image editing. In SIGGRAPH Asia 2024 Conference Papers. 1–10.

Wensong Song, Hong Jiang, Zongxing Yang, Ruijie Quan, and Yi Yang. 2025. Insert anything: Image insertion via in-context editing in dit. arXiv preprint arXiv:2504.15009

(2025).

Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga. 2023. Objectstitch: Object compositing with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18310–18319.

Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, He Zhang, Wei Xiong, and Daniel Aliaga. 2024. Imprint: Generative object compositing by learning identity-preserving representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8048–8058.

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. 2024. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024).

Qian Wang, Yiqun Wang, Michael Birsak, and Peter Wonka. 2023. Blobgan-3d: A spatially-disentangled 3d-aware generative model for indoor scenes. arXiv preprint arXiv:2303.14706 (2023).

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.

Wikipedia contributors. 2024. Peak signal-to-noise ratio — Wikipedia, The Free Encyclopedia. https://en.wikipedia.org/w/index.php?title=Peak_signal-to-noise_ratio& oldid=1210897995 [Online; accessed 4-March-2024].

Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. 2024. Draganything: Motion control for anything using entity representation. In European Conference on Computer Vision. Springer, 331–348.

Zhexiao Xiong, Wei Xiong, Jing Shi, He Zhang, Yizhi Song, and Nathan Jacobs.

2024. GroundingBooth: Grounding Text-to-Image Customization. arXiv preprint arXiv:2409.08520 (2024).

Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. 2023. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18381–18391.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).

Jiraphon Yenphraphai, Xichen Pan, Sainan Liu, Daniele Panozzo, and Saining Xie. 2024. Image sculpting: Precise object editing with 3d geometry control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4241–4251.

Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. 2025. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference. 26125–26135.

Hui Zhang, Dexiang Hong, Tingwei Gao, Yitong Wang, Jie Shao, Xinglong Wu, Zuxuan Wu, and Yu-Gang Jiang. 2024. CreatiLayout: Siamese Multimodal Diffusion Transformer for Creative Layout-to-Image Generation. arXiv preprint arXiv:2412.03859 (2024).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023b. Adding Conditional Control to Text-to-Image Diffusion Models.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 586–595.

Zhiyuan Zhang, Zhitong Huang, and Jing Liao. 2023a. Continuous layout editing of single images with diffusion models. In Computer Graphics Forum. Wiley Online Library, e14966.

A Gaussian to Ellipse Conversion

- A 2D Gaussian distribution is defined by its mean 𝝁 = (𝜇𝑥, 𝜇𝑦) and covariance matrix 𝚺:

𝚺 =

𝜎𝑥2 𝜌𝜎𝑥𝜎𝑦 𝜌𝜎𝑥𝜎𝑦 𝜎𝑦2

. (12) The level sets of this distribution are ellipses. For a confidence

level 𝛼, the corresponding confidence ellipse is given by:

(x − 𝝁)𝑇 𝚺−1(x − 𝝁) = 𝜒22(𝛼), (13)

where 𝜒22(𝛼) is the upper𝛼-quantile of the chi-square distribution with 2 degrees of freedom. The semi-major and semi-minor axes of the ellipse are proportional to the square root of the eigenvalues of

𝚺 multiplied by √︃𝜒22(𝛼), and the rotation angle is determined by the eigenvectors.

- B Ellipse to Gaussian Conversion

Conversely, given an ellipse with center (ℎ,𝑘), semi-major axis 𝑎, semi-minor axis 𝑏, and rotation angle 𝜃 (corresponding to a confidence level 𝛼), the Gaussian distribution can be reconstructed as:

𝝁 =

ℎ 𝑘

, 𝚺 =

1 𝜒22(𝛼)

R(𝜃)

𝑎2 0 0 𝑏2

R(𝜃)𝑇, (14) with the rotation matrix defined by

R(𝜃) =

cos𝜃 −sin𝜃 sin𝜃 cos𝜃

. (15)

This relationship allows a precise mapping between probabilistic blob representations and geometric ellipse controls, taking into account both the confidence level and the orientation of the ellipse.

- C Justification of Baseline Selection

In Sec. 4, we compare our approach against six representative baselines. Specifically, we include three methods capable of handling multiple types of element-level editing:

- (1) GliGEN [Li et al. 2023], a method that specifies layouts using bounding boxes.
- (2) Anydoor [Chen et al. 2023], a method that specifies layouts using segmentation.
- (3) Magic Fixup [Chen et al. 2023], a method based on compositing and harmonization.

as well as three methods restricted to translation-based editing:

- (1) DiffEditor [Mou et al. 2024], a point-based dragging method that designs different diffusion sampling algorithms for each type of edit.
- (2) InstantDrag [Shin et al. 2024], a point-based dragging method that predicts sparse optical flow from drags and uses it to guide the editing.
- (3) DragAnything[Wuetal.2024],apoint-baseddraggingmethod that represents objects using segmentation. This method was originally developed for motion-controllable video generation, and we use the final frame as the edited output.

We exclude several other methods for the following reasons:

- (1) DiffUHaul [Avrahami et al. 2024] has not been released.
- (2) Image Sculpting [Yenphraphai et al. 2024] relies on per-image optimization rather than generalizable editing.
- (3) DragonDiffusion [Mou et al. 2023] is an earlier version of DiffEditor [Mou et al. 2024].
- (4) RegionDrag [Lu et al. 2024], published earlier than both InstantDrag [Shin et al. 2024] and DiffEditor [Mou et al. 2024], is a point-based image dragging method similar to these two approaches.
- (5) ObjectStitch [Song et al. 2023], published earlier than Magic Fixup [Chen et al. 2023], and IMPRINT [Song et al. 2024], published around the same time as Magic Fixup, are both similar to Magic Fixup, being methods based on compositing and harmonization.
- (6) Image Sculpting [Yenphraphai et al. 2024] involves a complex process that requires manual adjustments for each sample during editing.

Taken together, the six selected baselines encompass a range of approaches, including point-based dragging, grounding-based methods, compositing techniques, and even a motion-controllable video generation model. This diverse set provides a comprehensive foundation for evaluating our method.

- D Limitations of Methods Without Multiple Task Support

Among the baselines introduced in the previous section, some methods do not support multiple types of generative editing tasks:

Additionally, several other methods have their own limitations:

- (1) Point-based dragging methods like DiffEditor [Mou et al. 2024], RegionDrag [Lu et al. 2024], InstantDrag [Shin et al. 2024], and DragonDiffusion [Mou et al. 2023] are constrained to translation-based editing due to their reliance on sparse point trajectories as input. While interpolation between start and end points is possible, these methods cannot handle more complex operations such as object addition, removal, scaling, or replacement. For example, scaling requires defining points in multiple directions, not just a single direction.
- (2) DiffUHaul [Avrahami et al. 2024] is a training-free approach that supports only translation-based operations and has not been publicly released.
- (3) Image Sculpting [Yenphraphai et al. 2024] is a 3D perceptionbased method designed for editing meshes in 3D space. While it supports various operations, the process is complex and requires per-sample reconstruction, along with specialized software like Blender for mesh editing.
- (4) ObjectStitch [Song et al. 2023] and IMPRINT [Song et al. 2024] focus on compositing and harmonizing foreground and background, but do not explicitly support translation, removal, or scaling operations.

These limitations highlight the advantages of our method, which supports a broader range of generative editing tasks, offering greater flexibility and control over the final output.

- E BlobBench and BlobData

BlobBench is a comprehensive benchmark consisting of 100 curated images, evenly distributed across various element-level operations,

[Figure 458]

- Fig. 12. Overview of the BlobBench.

[Figure 459]

- Fig. 13. The BlobData curation workflow.

including addition, translation, scaling, removal, and replacement. Each image is annotated with ellipse parameters, foreground masks, and textual descriptions, incorporating both real-world and AIgenerated images from diverse scenarios such as indoor/outdoor environments, animals, and landscapes (see Fig. 12).

In parallel, BlobData is a large-scale dataset comprising 1.86 million samples sourced from BrushData [Ju et al. 2024]. The curation process involves several key steps:

• Image Filtering. The source images are filtered to retain those with a minimum short side length of 480 pixels, valid instance segmentation masks, and masks with area ratios

SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

between 0.01 and 0.9 of the total image area. Masks touching image boundaries are excluded.

- • Parameter Extraction. Ellipse parameters are extracted using OpenCV’s ellipse fitting algorithm, followed by the derivation of corresponding 2D Gaussian distributions. Invalid samples with covariance values below 1e-5 are removed.
- • Annotation. Detailed textual descriptions for each image are generated using InternVL-2.5 [Chen et al. 2024], providing rich contextual information for each sample.

This curated dataset, combining detailed annotations and a diverseset of real-worldand synthetic images, serves as the foundation for diverse element-level operations.

