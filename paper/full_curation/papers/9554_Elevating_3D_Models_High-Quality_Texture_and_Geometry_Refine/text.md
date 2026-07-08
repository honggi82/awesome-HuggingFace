# arXiv:2507.11465v1[cs.GR]15Jul2025

## Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model

NURI RYU, POSTECH, South Korea JIYUN WON, POSTECH, South Korea JOOEUN SON, POSTECH, South Korea MINSU GONG, POSTECH, South Korea JOO-HAENG LEE, Pebblous, South Korea SUNGHYUN CHO, POSTECH, South Korea

[Figure 1]

[Figure 2]

[Figure 3]

(b) Ours (Elevate3D) (d) Ours (Elevate3D)

Fig. 1. 3D refinement examples from (a) a degraded real-world scan [Downs et al. 2022] and (c) a state-of-the-art image-to-3D generative model [Xiang et al. 2024]. Our method, Elevate3D, effectively refines both texture and geometry while preserving their alignment, as shown in (b) and (d). Inputs for the experiment: the GSO dataset [Downs et al. 2022], and ©MasaStojanovic/pixabay.

High-quality 3D assets are essential for various applications in computer graphics and 3D vision but remain scarce due to significant acquisition costs. To address this shortage, we introduce Elevate3D, a novel framework that transforms readily accessible low-quality 3D assets into higher quality. At the core of Elevate3D is HFS-SDEdit, a specialized texture enhancement method that significantly improves texture quality while preserving the appearance and geometry while fixing its degradations. Furthermore, Elevate3D operates in a view-by-view manner, alternating between texture and geometry refinement. Unlike previous methods that have largely overlooked geometry refinement, our framework leverages geometric cues from images refined with HFS-SDEdit by employing state-of-the-art monocular geometry predictors. This approach ensures detailed and accurate geometry that aligns seamlessly with the enhanced texture. Elevate3D outperforms recent competitors by achieving state-of-the-art quality in 3D model refinement, effectively addressing the scarcity of high-quality open-source 3D assets.

CCS Concepts: • Computing methodologies → Computer graphics. Additional Key Words and Phrases: 3D Asset Refinement, Diffusion models

Authors’ Contact Information: Nuri Ryu, POSTECH, South Korea, ryunuri@postech.ac. kr; Jiyun Won, POSTECH, South Korea, w1jyun@postech.ac.kr; Jooeun Son, POSTECH, South Korea, jeson@postech.ac.kr; Minsu Gong, POSTECH, South Korea, gongms@ postech.ac.kr; Joo-Haeng Lee, Pebblous, South Korea, joohaeng@pebblous.ai; Sunghyun Cho, POSTECH, South Korea, s.cho@postech.ac.kr.

This work is licensed under a Creative Commons Attribution 4.0 International License. SIGGRAPH Conference Papers ’25, Vancouver, BC, Canada

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1540-2/2025/08 https://doi.org/10.1145/3721238.3730701

#### ACM Reference Format:

NuriRyu,Jiyun Won, Jooeun Son, Minsu Gong, Joo-Haeng Lee, and Sunghyun Cho. 2025. Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers ’25), August 10–14, 2025, Vancouver, BC, Canada. ACM, New York, NY, USA, 20 pages. https://doi.org/10.1145/3721238.3730701

1 Introduction

High-quality 3D models are in unprecedented demand, serving as essential components for various applications and as training data in computer graphics and 3D vision [He et al. 2025; Shi et al. 2023b; Tewari et al. 2022]. Despite the growth in large-scale open-source 3D model datasets [Deitke et al. 2023, 2022; Wu et al. 2023], high-quality models remain scarce due to high acquisition costs. To address this, we tackle the problem of texture and geometry refinement of easily accessible low-quality models, bridging the gap between abundant low-quality data and the pressing need for high-quality models.

Constructing high-quality 3D models from low-quality counterparts, a process we refer to as 3D model refinement, is of great importance but has received relatively less attention. Traditional methods, such as mesh subdivision and denoising, primarily focus on refining geometry by suppressing noisy structures or smoothing surfaces using simple geometric priors. However, these methods fall short of producing high-quality geometric details as they rely solely on simple priors and do not address texture refinement.

Recently, several 3D model refinement methods have emerged, either as post-processing steps within 3D model generation pipelines

or as stand-alone methods for refining existing low-quality models [Lee et al. 2024; Sun et al. 2024; Tang et al. 2024b; Wu et al. 2024a; Yang et al. 2024c; Zhang et al. 2023a, 2024]. These approaches leverage generative priors, such as single-image and multi-view diffusion models [Ho et al. 2020; Sohl-Dickstein et al. 2015] and GANs [Goodfellow et al. 2014], to enhance both texture and geometry or texture alone. Specifically, they render multiple views of a low-quality model, enhance each view, and update the 3D model. One common strategy for view enhancement is to apply diffusionor GAN-based image enhancement models specifically trained to refine the appearance of 3D models [Wu et al. 2024a; Yang et al. 2024c; Zhang et al. 2024]. Another popular approach is to employ SDEdit [Meng et al. 2022], which offers high flexibility, enabling refinement of low-quality models with various degradations [Sun et al. 2024; Tang et al. 2024b; Yang et al. 2024c; Zhang et al. 2023a].

Despite the advances in 3D model refinement, limitations in both texture and geometry still remain. Most recent approaches refine each view independently, leading to cross-view inconsistencies and blurry textures. While multi-view diffusion models may improve consistency, they are limited in image resolution due to large memory requirements, restricting refinement quality [Long et al. 2023].

SDEdit-based approaches also suffer from a trade-off between fidelity to the input and the quality of refined 3D models. SDEdit generates structured noise by interpolating the input image with Gaussian noise, and applies the diffusion model’s denoising process [Ho et al. 2020; Sohl-Dickstein et al. 2015]. This aligns the image with the high-quality distribution learned by the diffusion model while preserving key features of the input. However, this process imposes a quality-fidelity trade-off based on the noise level. Stronger noise enhances conformance to the diffusion model’s distribution but harms fidelity. In contrast, lower noise maintains fidelity but offers only minor enhancements.

Lastly, previous approaches rely solely on image-based priors derived from large-scale image generative models. This exclusive reliance on photometric constraints leaves the geometry underconstrained; even if the refined models appear visually appealing from certain viewpoints, their underlying geometric structure may still be of poor quality [Barron et al. 2022; Yu et al. 2022; Zhang et al. 2020]. To address these issues, some approaches incorporate separate geometry and texture optimization stages [Sun et al. 2024; Tang et al. 2024b]. However, decoupling the two processes often results in misaligned texture and geometry [I Ho et al. 2024].

This paper proposes Elevate3D, a novel 3D model refinement approach that produces a high-quality 3D model with well-aligned texture and geometry. Elevate3D operates iteratively, refining textures and geometries view-by-view. For each view, it first enhances the texture and then leverages the refined texture to adjust the geometry, ensuring alignment. In subsequent views, the texture refinement is based on the visible parts of the updated geometry and the original geometry. This process ensures precise alignment between texture and geometry and allows the use of high-quality priors trained on large-scale image datasets for both texture and geometry, ultimately producing superior 3D models.

For texturerefinement,Elevate3DadoptsHigh-frequency-Swapping

SDEdit (HFS-SDEdit), a specialized method that resolves the fidelity–quality trade-off in SDEdit. Our key insight leverages the

coarse-to-finenatureofthediffusion model’s generative process [Rissanen et al. 2023], where low-frequency features are established first and strongly influence subsequent high-frequency detail generation. If these low-frequency features are constrained to match a lowquality input, the final image inevitably inherits unwanted artifacts. Instead, we let the diffusion model freely generate low-frequency features while applying constraints only high-frequency components to match the input image during the early denoising steps. In this way, the generative path is steered toward the high-quality image distribution while minimal high-frequency guidance preserves crucial edges and details—sufficient to maintain the input’s unique identity without embedding its artifacts.

After refining the texture at a viewpoint, we enhance the geometry for that viewpoint using the refined texture. We employ a state-of-the-art monocular normal predictor [Martin Garcia et al. 2025] to infer detailed surface normals aligned with the updated texture. The predicted normals may not perfectly match the initial 3D geometry. Thus, we employ a regularized normal integration scheme to estimate a geometry consistent with the initial 3D geometry from the predicted normals. We then stitch the estimated geometry with previously refined or unrefined regions, maintaining consistent geometry across viewpoints. This process produces a 3D representation faithful to the refined textures and supports direct texture projection onto the enhanced geometry without misalignment.

Comprehensive experiments demonstrate that Elevate3D successfully produces high-quality 3D models from low-quality ones, including those generated by previous 3D generation models and low-quality scanned models. To summarize, our main contributions are as follows:

- • We propose a novel 3D model refinement framework, Elevate3D, which alternates between texture and geometry refinement in a view-by-view fashion to produce a high-quality 3D model with well-aligned texture and geometry
- • We introduce HFS-SDEdit for texture refinement, leveraging high-frequency guidance to achieve high-quality and highfidelity enhancements while mitigating the limitations of previous SDEdit-based methods.
- • We demonstrate that our framework can achieve state-of-theart quality refinement of 3D models compared with recent competitors through comprehensive experiments.

- 2 Related Work
- 3D Model Generation. Recent advances in 3D model generation

leverage diffusion models due to their powerful image priors [Li et al. 2024]. Initially, the SDS loss was introduced to use pre-trained textto-image diffusion models for generating 3D models directly from text or image prompts [Melas-Kyriazi et al. 2023; Poole et al. 2023; Tang et al. 2023; Wang et al. 2023; Xu et al. 2023]. Although promising, these methods often produce textures with over-saturated colors and blurred details. Alongside these optimization approaches, early efforts also adapted 2D diffusion models to generate multi-view consistent novel views via inpainting for 3D reconstruction [Kant et al. 2023; Ryu et al. 2023]. These were succeeded by a line of works that fine-tune pre-trained diffusion models to generate multi-view

images for 3D reconstruction [Liu et al. 2024; Long et al. 2023; Shi

- et al. 2023a, 2024]. However, these models require additional components on top of the large-scale diffusion model [Rombach et al. 2022] to enforce multi-view consistency. This added complexity increases computational costs and memory requirements, limiting both the number of views that can be generated simultaneously and their resolution, consequently affecting the quality of the generated 3D models [Long et al. 2023]. In our work, we demonstrate that our refinement method can successfully produce high-quality 3D models from low-quality ones generated by previous approaches.

3D Model Refinement. Recently, 3D model refinement methods have primarily emerged as components of 3D model generation pipelines. To improve the low-quality outputs from a coarse 3D generation step, recent pipelines often introduce a relatively straightforward refinement stage. One widely adopted approach is to use SDEdit [Meng et al. 2022] with an image-based loss [Sun et al. 2024; Tang et al. 2024b; Yang et al. 2024c; Zhang et al. 2023a]. Specifically, a low-quality image is rendered from a coarse 3D model at an arbitrary viewpoint and then refined using SDEdit. The 3D model is updated using the refined images through an MSE loss. However, this approach has significant drawbacks. First, refined details at each view are not multi-view consistent; details introduced in one viewpoint are averaged out across others, resulting in blurry outcomes. Second, the image-based loss under-constrains the geometry, forcing such methods to keep the geometry fixed and focus solely on texture refinement. Another line of approaches [Wu

- et al. 2024a; Zhang et al. 2024] generates multi-view images and refines them using off-the-shelf super-resolution networks such as Real-ESRGAN [Wang et al. 2021]. The refined images are then used to synthesize 3D models. However, these methods perform superresolution on each image independently, resulting in blurriness due to multi-view inconsistency.

Standalone 3D refinement methods outside of 3D generation pipelinesencountersimilarchallenges. For instance, MagicBoost [Yang et al. 2024c] employs multi-view diffusion models combined with SDS optimization to enhance both geometry and texture. However, artifacts introduced during SDS optimization necessitate additional refinement using SDEdit, leading to the same multi-view consistency issues. DiSR-NeRF [Lee et al. 2024] pairs an SDS loss [Poole et al. 2023] with a diffusion-based 2D super-resolution model [Rombach et al. 2022]. Although this method iteratively refines a lowresolution NeRF by enhancing 2D images and synchronizing the 3D model, it prioritizes consistency in low-resolution features. Consequently, high-resolution details generated during refinement can still be averaged out during 3D synchronization.

In contrast, Elevate3D adopts a novel approach to refinement by preserving and distinguishing previously enhanced regions in each view. This ensures a multi-view consistent result across all viewpoints. Additionally, by alternating between texture and geometry refinement steps, Elevate3D allows the geometry to be refined according to the improved textures. This strategy addresses the limitations of earlier methods, which either average out high-frequency details or neglect geometry refinement altogether.

3D Model Texturing. Another relevant task related to ours is 3D model texturing [Chen et al. 2023; Richardson et al. 2023; Tang et al.

2024a; Youwang et al. 2024; Zeng et al. 2024]. Texturing approaches synthesize textures for 3D models without altering the geometry. Hence, these methods, like 3D model refinement approaches, face the limitation of texture-geometry misalignment. Even if an input 3D model lacks geometric details, these methods may still produce textures with rich details learned from generative priors, leading to inconsistencies between texture and geometry. This highlights the need for methods that jointly refine both texture and geometry. Our proposed Elevate3D addresses the shortcomings of one-sided methods by integrating both aspects within a unified framework.

3 HFS-SDEdit for Image Refinement

HFS-SDEdit, built on top of SDEdit [Meng et al. 2022], is a critical component of Elevate3D for high-quality, high-fidelity texture refinement. This section reviews SDEdit then introduces HFS-SDEdit.

SDEdit. SDEdit is a technique designed to guide the image synthesis process of diffusion models. Specifically, for a given reference image, such as a stroke image or a low-quality image, the goal of SDEdit is to generate a realistic image that aligns with the learned distribution of the diffusion model and adheres to the structure of the reference image. SDEdit leverages the observation that adding more noise to images in both the reference image domain (I𝑟) and the realistic image domain (I) gradually merges them, transforming them into the domain of Gaussian noise. Based on this, SDEdit proposes a simple, training-free strategy. Let 𝑧𝑟 ∈ I𝑟 represent a reference image in the latent space of a diffusion model. SDEdit then initalizes a noisy latent 𝑧𝑡𝑠 by adding noise to 𝑧𝑟 as:

𝑧𝑡𝑠 = 𝛼(𝑡𝑠)𝑧𝑟 + 𝛽(𝑡𝑠)𝜖, (1) where 𝑡𝑠 is a timestep in the denoising schedule of the diffusion model such that 𝑡𝑠 ∈ {𝑇,𝑇 − 1, · · · , 0}. The functions 𝛼(𝑡) ∈ [0, 1] and 𝛽(𝑡) ∈ [0, 1] control the noise level at timestep 𝑡. Starting from this initial noisy latent, SDEdit performs the iterative denoising process to produce a realistic image 𝑧0.

SDEdit offers several distinct benefits, including easy integration into diffusion-based image synthesis pipelines, training-free implementation, and computational efficiency. Thus, it has been widely adopted for various tasks, e.g., image editing [Wu and De la Torre

- 2023], video generation [Zhang et al. 2023b], 3D editing [Chen et al.
- 2024b]. However, it exhibits a fidelity-quality trade-off, which will be discussed in detail later.

HFS-SDEdit. To overcome the fidelity-quality trade-off of SDEdit, our proposed HFS-SDEdit replaces the high-frequency component of the latent representation with that of the reference image to ensure that the synthesized image 𝑧0 aligns with the structural details of the reference image 𝑧𝑟. Specifically, HFS-SDEdit starts with Eq. (1). Then, in the subsequent timesteps 𝑡, HFS-SDEdit replaces the highfrequency component of the latent 𝑧ˆ𝑡, the output of the standard denoising process, by evaluating:

𝑧˜𝑡 = 𝛼(𝑡)𝑧𝑟 + 𝛽(𝑡)𝜖, and (2) 𝑧𝑡′ = (𝛿 − 𝐺𝜎) ∗ 𝑧˜𝑡 + 𝐺𝜎 ∗ 𝑧ˆ𝑡 (3)

where 𝑧˜𝑡 is a noised reference image and 𝑧𝑡′ is a calibrated latent whose high-frequency component is replaced with that of 𝑧˜𝑡. 𝛿 is a Dirac delta function,𝐺𝜎 is a Guassian kernel with standard deviation

[Figure 4]

[Figure 5]

### Naïve SDEdit High Quality Low Quality

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

###### (a) Ours (b) SDEdit(High) (c) Input (d) SDEdit(Low)

𝜖𝜖 ~𝑁𝑁 0,1 𝜖𝜖

Ours

𝑧𝑧𝑡𝑡ℎ

𝑧𝑧𝑡𝑡𝑙𝑙 𝑧𝑧𝑟𝑟 Denoising Time Steps

Fig. 2. Overview of HFS-SDEdit. HFS-SDEdit addresses the quality-fidelity trade-off in SDEdit. By adding a substantial amount of noise 𝜖 to the low-quality reference image 𝑧𝑟 in (c) and initiating the denoising process from the noisy latent 𝑧𝑡h, SDEdit removes domain information, enabling the diffusion model to generate a high-quality image as depicted in (b). However, this approach compromises fidelity to the reference image. Conversely, adding a small amount of noise and starting the denoising process from the noisy latent 𝑧𝑡l preserves the low-quality domain information, resulting in only minor refinements, as seen in (d). In contrast, HFS-SDEdit incorporates high-frequency feature injection-based guidance, as detailed in Section 3, allowing for high-fidelity generation even when starting the denoising process from 𝑧𝑡h. This approach achieves both high quality and high fidelity in the refinement process. Input: ©Watts/flickr.

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

𝜎, and ∗ is the convolution operator. Once 𝑧𝑡′ is obtained, denoising is performed with 𝑧𝑡′ resulting in 𝑧ˆ𝑡−1. We perform high-frequency replacement until 𝑡 reaches a predefined timestep 𝑡stop to ensure that the resulting image does not reproduce all the details, such as the degraded details of the low-quality reference image.

Texture Refinement

[Figure 14]

[Figure 15]

| |
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

HFS-SDEdit is based on the intuition that the domain information of an image is encoded in the mid-frequency component of the latent representation. Specifically, we may decompose the latent representation into two frequency bands: low-, and high-frequency components. Similar to conventional images, the high-frequency component describes small-scale structures [Park et al. 2023]. On the other hand, the low-frequency component contains not only large-scale structures but also domain information.

Texture

Texture

[Figure 25]

[Figure 26]

Geometry Refinement

Geometry

Geometry

Coarse Geometry & Unrefined Texture

Refined Geometry & Texture

[Figure 27]

[Figure 28]

Fig. 3. Framework Overview. Given a low-quality 3D model, Elevate3D alternatingly refines texture and geometry. Input for the experiment: ©Momentmal/pixabay.

Fig. 2 illustrates the intuition behind HFS-SDEdit as well as the quality-fidelity trade-off of SDEdit with an image enhancement example. Adding more noise to the reference image 𝑧𝑟 gradually removes information from high-frequency to low-frequency components. Thus, to sufficiently merge the realistic image domain and the reference image domain, or equivalently to remove domain information, SDEdit requires the use of large noise to remove a sufficient amount of low-frequency components. However, such excessive noise also destroys small- and large-scale structures, causing highquality but low-fidelity results. Conversely, SDEdit with small noise removes only the high-frequency component, producing an image not only with different small-scale details but also within the same domain as the reference image, the low-quality image domain.

4 Elevate3D

Fig. 3 visualizes an overview of Elevate3D. Given a low-quality textured mesh, our approach progressively refines both the texture and geometry of the mesh by iterating through a predefined camera path V = {𝑣0, . . .,𝑣𝑘}. Specifically, at the 𝑖-th iteration corresponding to 𝑣𝑖, we denote the initial partially refined model as 𝑀𝑖, where 𝑀0 is initialized as the input low-quality textured mesh. Our method then refines the unrefined regions of the texture and geometry of 𝑀𝑖 that are visible at 𝑣𝑖 through texture refinement and geometry refinement stages. The texture refinement stage focuses on enhancing the unrefined regions while maintaining consistency with the already refined areas. Afterward, the geometry refinement stage extracts geometric cues from the newly refined texture and refines the mesh geometry accordingly. This ensures that the geometry accurately reflects the details present in the refined texture, maintaining texture-geometry consistency.

To improve both quality and fidelity, HFS-SDEdit starts with large noise and injects high-frequency components of the reference image into the synthesis process. This approach effectively removes the domain information of the reference image, resulting in a high-quality image. The injected high-frequency component not only constrains the high-frequency details of a synthesized image, but also guides the diffusion model to synthesize a low-frequency component that aligns with the injected details, achieving high-fidelity synthesis.

This view-by-view refinement strategy enables our method to utilize predefined image and geometry priors, which contribute to achieving high-quality texture and geometry. Additionally, by leveraging the refined texture and geometry when processing unrefined regions in subsequent viewpoints, our method ensures cross-view consistency. Also, the geometry refinement is performed based on

the refined texture, significantly enhancing texture-geometry consistency. In the following, we explain each stage in detail.

- 4.1 Texture Refinement with HFS-SDEdit The texture refinement stage improves the texture of the partially

refined 3D model 𝑀𝑖 for the current view 𝑣𝑖 by leveraging HFSSDEdit with a large-scale pretrained image diffusion model. We begin by rendering 𝑀𝑖 from 𝑣𝑖 to obtain the initial image 𝐼𝑖 (Fig. 4-a). This image contains regions refined at previous views {𝑣0, . . .,𝑣𝑖−1} and unrefined regions that still exhibit low-quality textures.

To isolate the unrefined regions in 𝐼𝑖, we identify pixels that are not visible in previous views. Specifically, we rasterize normal vectors for 𝑣𝑖 and compare them with the viewing directions of the previous views {𝑣0, . . .,𝑣𝑖−1} by computing cosine similarities. We detect pixels whose cosine similarity values exceed a threshold 𝜏 for all previous views and construct a binary mask 𝑚𝑖 (Fig. 4-b). We set 𝜏 = 0.5, equivalent to a 60◦ angle difference. This approach may include already-refined pixels in the mask, but it allows regions seen at oblique angles to be further refined.

Once 𝑚𝑖 is obtained, we refine the unrefined regions in 𝐼𝑖 using HFS-SDEdit. To refine only the unrefined regions specified by 𝑚𝑖, we introduce a slight modification to HFS-SDEdit. Specifically, at each timestep of the diffusion sampling process, we first compute a noised reference 𝑧˜𝑡 and a latent 𝑧𝑡′ using Eqs. (2) and (3). To ensure that the regions outside 𝑚𝑖 are not overwritten, we blend 𝑧𝑡′ with 𝑧˜𝑡 using 𝑚˜ , a downsampled version of 𝑚𝑖. This blending is done as:

zˆ𝑡 = m˜ ⊙ z𝑡′ + 1 − m˜ ⊙ z˜𝑡. (4) Finally, zˆ𝑡 goes through the denoising step. After iterative denoising with HFS-SDEdit, we obtain a refined image 𝐼𝑖′ with improved textures faithfully reflecting 𝐼𝑖 for the unrefined regions while preserving the textures from 𝐼𝑖 in the already-refined regions (Fig. 4-c).

- 4.2 Geometry Refinement with Refined Texture The geometry refinement step enhances the geometry of the partially refined 3D triangle mesh 𝑀𝑖 by utilizing the refined texture

image 𝐼𝑖′ from the previous texture refinement stage. This image not only exhibits improved textures but also provides valuable cues for

geometric details. These details can be extracted using monocular geometry estimation models [Ke et al. 2024; Martin Garcia et al.

- 2025; Yang et al. 2024b]. In Elevate3D, we start by inferring a normal map n𝑖 from the re-

finedimage𝐼𝑖′ usingastate-of-the-artnormalestimation model [Martin Garcia et al. 2025]. We then integrate the estimated normals n𝑖 to obtain a refined surface 𝑆𝑖 corresponding to the viewpoint 𝑣𝑖. This surface is consistent with the refined texture image 𝐼′. However, because n𝑖 is derived solely from 𝐼𝑖′ and not conditioned on the existing geometry of 𝑀𝑖, the refined geometry 𝑆𝑖 can significantly deviate from that of 𝑀𝑖. Consequently, directly stitching 𝑆𝑖 onto 𝑀𝑖 could introduce severe geometric distortion.

To address this, we introduce a regularized normal integration scheme to estimate 𝑆𝑖 while ensuring consistency with the existing geometry of𝑀𝑖. Our regularized normal integration scheme is implemented as follows. We assume an orthographic camera model and rasterize a depth map𝑑 of 𝑀𝑖 from viewpoint 𝑣𝑖. We define 𝑆𝑖 and n𝑖 as𝑆𝑖(𝑢,𝑣) = [𝑢,𝑣,𝑧(𝑢,𝑣)]⊤,andn𝑖(𝑢,𝑣) = [𝑛𝑥 (𝑢,𝑣),𝑛𝑦(𝑢,𝑣),𝑛𝑧(𝑢,𝑣)]⊤,

respectively, where 𝑢 and 𝑣 represent pixel coordinates. Our goal is to estimate the depth component 𝑧(𝑢,𝑣) such that the resulting surface 𝑆𝑖 satisfies the following two criteria: First, the normals of 𝑆𝑖 must be close to n𝑖, or equivalently, its tangent vectors 𝜕𝑆𝑖(𝑢,𝑣)/𝜕𝑢 and 𝜕𝑆𝑖(𝑢,𝑣)/𝜕𝑣 are orthogonal to n𝑖(𝑢,𝑣). Second, the estimated depth 𝑧(𝑢,𝑣) must be close to the depth 𝑑(𝑢,𝑣) rendered from 𝑀𝑖. Based on these two criteria, we define an energy functional:

𝐸(𝑧) = ∬ 𝜕𝑧 𝜕𝑢 +

2

2

𝑛𝑦(𝑢,𝑣) 𝑛𝑧(𝑢,𝑣)

𝑛𝑥 (𝑢,𝑣) 𝑛𝑧(𝑢,𝑣)

𝜕𝑧 𝜕𝑣 +

+

𝑑𝑢 𝑑𝑣

+ 𝜆 ∬ 𝑧(𝑢,𝑣) − 𝑑(𝑢,𝑣)

(5)

2

𝑑𝑢 𝑑𝑣,

where the first and second terms on the right-hand-side correspond to the first and second criteria, respectively. 𝜆 is a regularization parameter that balances the two terms. Minimizing 𝐸(𝑧) yields a refined depth map for 𝑆𝑖. In our implementation, we follow the normal integration method of Cao et al. [2022] to minimize 𝐸(𝑧).

Once we obtain the refined geometry 𝑆𝑖 represented by the refined depth map, we update the mesh 𝑀𝑖. Specifically, the refined geometry region 𝑆𝑖 is integrated with the unchanged parts of 𝑀𝑖 using Poisson surface reconstruction [Kazhdan et al. 2006]. This step generates the updated triangular mesh 𝑀˜𝑖. While Poisson reconstruction does not strictly preserve the original mesh topology, it rarely introduces geometric artifacts in our case. This robustness stems from our regularized integration in Eq. (5), which constrains the geometric update using the coarse input mesh 𝑀𝑖 via the depth map 𝑑 as a guide. This process ensures that each view’s improved geometry is seamlessly integrated without disrupting previously refined areas of the model. Furthermore, thanks to the geometry refinement process leveraging the refined texture, Elevate3D ensures proper texture-geometry alignment.

Fig. 5 illustrates this geometry refinement process. Given a partially refined mesh 𝑀𝑖 (Fig. 5-a), our geometry refinement stage estimates a refined geometry 𝑆𝑖, visualized via its depth map (Fig. 5b). This refined region is then merged with the other regions of 𝑀𝑖 using Poisson reconstruction (Fig. 5-c), resulting in the updated mesh 𝑀˜𝑖 (Fig. 5-d). We note that the seams in Fig. 5-b are the result of filtering of unreliable depth values around discontinuities, which we provide details in the supplementary material.

After the geometry refinement stage, we project the refined tex-

ture image 𝐼𝑖′ onto the updated mesh 𝑀˜𝑖 to obtain the textured mesh 𝑀𝑖+1 for the next view refinement iteration. For this, we employ projection mapping, a form of UV-free texture mapping, similar to recent mesh texturing methods [Chen et al. 2023; Richardson et al. 2023; Tang et al. 2024a].

5 Experiments 5.1 Implementation Details

In all experiments, we use FLUX1—an open-source large-scale textto-image diffusion model trained with the rectified flow-matching formulation [Albergo and Vanden-Eijnden 2022; Lipman et al. 2023; Liu et al. 2022]. We follow the default parameters and the denoising schedule used by FLUX. For all experiments, we use a total of 𝑇 = 30 denoising steps. We set the initial noise timestep 𝑡𝑠 to 29

1https://github.com/black-forest-labs/flux

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

(a) Initial image 𝐼𝐼𝑖𝑖 (c) Refined image 𝐼𝐼′𝑖𝑖

(b) Refinement Mask 𝑚𝑚𝑖𝑖

- Fig. 4. Texture Refinement. Given a partially-refined texture image 𝐼𝑖 in (a), the texture refinement stage detects a refinement mask 𝑚𝑖 in (b), and produces a refined image in (c) using HFS-SDEdit.

(a) Partially Refined Mesh 𝑀𝑀𝑖𝑖

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

(b) Refined Geometry 𝑆𝑆𝑖𝑖

(c) Region of 𝑀𝑀𝑖𝑖 Invisible from 𝑣𝑣𝑖𝑖

(d) Updated Mesh 𝑀𝑀𝑖𝑖

- Fig. 5. Geometry Refinement. Given a partially refined geometry 𝑀𝑖 in (a), we obtain a refined surface 𝑆𝑖 in (b), and stitch it with the other regions of 𝑀𝑖 shown in (c), resulting in the updated mesh 𝑀˜𝑖 in (d). Input for the experiment: the GSO dataset [Downs et al. 2022]

and the frequency swapping threshold 𝑡stop to 18. We employ a Gaussian low-pass filter with 𝜎 = 4. These parameters were selected by qualitatively comparing different combinations and choosing the setting that yielded the best qualitative results. A quantitative comparison for various combinations of 𝜎 and 𝑡stop is provided in the supplementary material. For the depth regularization, we set 𝜆 = 0.008. We use an orthographic camera, and for selecting the camera schedule, we adopt a strategy similar to Text2Tex [Chen

- et al. 2023]: starting with a pre-defined set of camera poses V, we use an automatic view selection scheme to choose the refinement view that covers the largest unrefined region. We leave the specific details in the supplementary.

5.2 Evaluation of Elevate3D

We evaluate the 3D model refinement quality of Elevate3D using a real-world scan dataset, GSO [Downs et al. 2022], where we formed a test set comprising 59 objects. To this end, we degrade the 3D models from the GSO dataset by reducing the number of faces to 20% and applying a Gaussian low-pass filter with 𝜎 = 8 to the textures. We then compare our method with recent 3D model refinement approaches: MagicBoost [Yang et al. 2024c], DiSR-NeRF [Lee et al. 2024], and DreamGaussian [Tang et al. 2024b]. DreamGaussian focuses solely on texture refinement, while the others refine both texture and geometry. Using each method, we refine the degraded

- 3D models and compare the quality of the refined models.

- Table 1. Quantitative Comparison on 3D Refinement. Elevate3D consistently achieves the best scores across various non-reference quality metrics, highlighting its high-quality 3D refinement. DreamGaussian [Tang et al. 2024b] refines only textures while others refine both texture and geometry. All refinement time were measured using an NVIDIA RTX A6000 GPU.

Model MUSIQ ↑ LIQE ↑ TOPIQ ↑ Q-Align ↑ Time ↓ DreamGaussian [2024b] 61.667 2.1185 0.4690 2.7416 1 min DiSR-NeRF [2024] 48.940 1.2869 0.3879 2.6794 6 hrs MagicBoost [2024c] 51.646 2.1085 0.3915 2.4992 20 min Elevate3D (Ours) 66.527 2.7744 0.5295 3.2151 25 min

- Table 2. Quantitative Comparison on 2D Image Refinement. The fullreference metrics are evaluated against the high-quality source images. Baseline (LQ) denotes the degraded version of the source images.

Full-Reference No-Reference PSNR ↑ SSIM ↑ LPIPS ↓ MUSIQ ↑ QAlign ↑ LIQE ↑ TOPIQ ↑

Model

Baseline (LQ) 20.701 0.521 0.662 21.918 2.018 1.303 0.159 SDEdit (strength = 0.4) 19.214 0.473 0.679 22.863 2.237 1.261 0.174 SDEdit (strength = 0.8) 15.255 0.379 0.746 29.190 2.860 1.321 0.214 NC-SDEdit 17.737 0.442 0.697 25.257 2.476 1.329 0.184 HFS-SDEdit (Ours) 15.588 0.391 0.598 39.519 3.337 2.105 0.283

Fig. 10 shows a qualitative comparison. As the figure shows, the results of previous approaches show blurry textures and less accurate geometries that are inconsistent with the textures. In contrast, our method successfully refines input low-quality 3D models, producing detailed textures and geometries, substantially surpassing previous methods. Our results also exhibit high texture-geometry consistency, thanks to our geometry refinement strategy that leverages refined textures. We also report a quantitative comparison of the quality of the refined 3D models in Table 1. For quantitative evaluation, we render the refined models and assess the quality of the rendered images using various image quality metrics: MUSIQ [Ke et al. 2021], LIQE [Zhang et al. 2023c], TOPIQ [Chen et al. 2024a], and Q-Align [Wu et al. 2024b]. As reported in the table, our method consistently outperforms other approaches across all quality metrics. Regarding computation times,the unoptimized prototype implementation of Elevate3D operates slower than DreamGaussian, which exclusively refines textures. However, it achieves similar efficiency to MagicBoost and is considerably faster than DiSR-NeRF.

Elevate3D can also be utilized to produce superior 3D models when combined with state-of-the-art (SoTA) image/text-to-3D synthesis methods. Fig. 11 illustrates the refinement of 3D models generated by TRELLIS [Xiang et al. 2024], a leading 3D model synthesis method. A common issue with 3D generation models like TRELLIS is that they often fail to produce high-quality results for inputs outside the domain of their 3D training datasets, which mostly consist of synthetic objects. Elevate3D effectively enhances these results, as shown in the figure, producing superior 3D models.

5.3 Evaluation of HFS-SDEdit

We analyze HFS-SDEdit in the image enhancement task using the validationset ofLSDIR [Liet al.2023], alarge-scaleimage restoration dataset. From LSDIR’s high-quality images, we create low-quality images by downsampling and upsampling them by a factor of 8.

For text prompts, we use ChatGPT Vision [OpenAI et al. 2024] to generate descriptions for the images.

Impact of Low-Frequency Component in Diffusion Sampling. We validate our intuition that the domain information is encoded in the low-frequency component in the latent representation. To this end, we conduct an experiment as follows. We prepare a low-quality reference image. Then, we sample three different images from the same pure Gaussian noise using different strategies. We sample the first sample following the conventional diffusion process. We sample the second sample similarly, but we replace the low-frequency component of its latent representation with that of the low-quality reference. To this end, we modify Eq. (3) as𝑧𝑡′ = 𝐺𝜎 ∗𝑧˜𝑟 +(𝛿−𝐺𝜎)∗𝑧𝑡. Finally, for the third image, we replace its high-frequency component with that of the low-quality reference. We swap the frequency components only for the first four denoising timesteps for both images when low-frequency features primarily emerge.

Fig. 6 compares the three images with their mean radially averaged power spectral density (RAPSD) graphs. As shown in the figure, when the low-frequency component is replaced with that of the low-quality reference, the diffusion model struggles to synthesize high-frequency details, indicating a shift in the generation path toward the low-quality domain. Conversely, when only highfrequency component is swapped, the model still produces detailed textures regardless of the reference image’s quality. This observation clearly indicates that the domain information is not in the high-frequency component but in the low-frequency component.

Image Refinement Comparisons. We validate the effectiveness of HFS-SDEdit by comparing it with SDEdit and NC-SDEdit [Yang et al. 2024a]. NC-SDEdit is a video enhancement approach that updates the low-frequency component of latents to match reference frames during the diffusion denoising process, enhancing fidelity. For comparison, we apply NC-SDEdit to single images. We evaluate the fidelity of the refinement results using full-reference metrics against the high-quality source images: PSNR, SSIM, and LPIPS [Zhang et al. 2018]. For quality assessment, we employ no-reference metrics: MUSIQ [Ke et al. 2021], LIQE [Zhang et al. 2023c], TOPIQ [Chen

- et al. 2024a], and Q-Align [Wu et al. 2024b]. Table 2 shows the quantitative comparison, where HFS-SDEdit

achieves the best performance in no-reference metrics, demonstrating its capability to produce high-quality outputs. It also obtains the best LPIPS score among its competitors, indicating that the resulting images preserve a high degree of perceptual similarity to the original images. However, because HFS-SDEdit uses a generative approach to refine the original input, it does not achieve the best PSNR and SSIM scores. This is common in generative-refinement methods, which prioritize plausible refinement over exact pixel-level fidelity [Blau and Michaeli 2018; Gu et al. 2020, 2022; Yu et al. 2024]. Despite this, Fig. 7 illustrates that HFS-SDEdit still produces images with convincing fidelity and high-frequency details.

When examining SDEdit across various strengths, we observe a fidelity-quality trade-off. Lower strength values yield better fullreference metrics but lower no-reference scores, indicating that fidelity is achieved at the expense of quality. Conversely, higher strength values improve perceived quality at the expense of fidelity.

[Figure 36]

[Figure 37]

(b) Low Quality Input (c) No Swap

[Figure 38]

[Figure 39]

[Figure 40]

(a) Mean RAPSD Plot (d) Low-Frequency Swap (e) High-Frequency Swap

- Fig. 6. Impact of Low-frequency Component in Diffusion Sampling. The mean radially averaged power spectral density (RAPSD) graphs of different example images shown in (b)-(e) are shown in (a). Image (b): ©patrick janicek/flickr.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

(a) Image Before Degradation (b) HFS-SDEdit (Ours) (c) NC-SDEdit

(d) Image After Degradation (e) SDEdit (Strength 0.4) (f) SDEdit (Strength 0.8)

- Fig. 7. Qualitative Comparison on 2D Image Refinement. The refinement results in (b), (c), (e), and (f) are obtained from the low-quality image in (d), which was degraded from the image in (a). Image (a): ©Mathias Appel/flickr.

Meanwhile, NC-SDEdit’s low-frequency retention condition inadvertently carries over low-quality domain information, leading to subpar results. Consequently, its outputs exhibit both lower fidelity and perceptual quality compared to those of HFS-SDEdit, reinforcing the advantage of our method’s high-frequency-based guidance.

5.4 Ablation Studies

Finally, we conduct ablation studies to justify our design choices. To demonstrate the necessity of texture and geometry refinement stages, we compare scenarios where only texture or geometry refinement is performed, as shown in Fig. 8. Without geometry refinement, the resulting 3D model retains the crude geometry of the input model (Fig. 8-a). Conversely, without texture refinement, the geometry refinement stage relies on the low-quality input texture, resulting in minimal geometry improvement (Fig. 8-b). Employing both texture and geometry refinement stages yields high-quality textures and geometry that are consistent with each other (Fig. 8-c).

Fig. 9 illustrates the effect of the regularized normal integration. As discussed in Section 4.2, normals predicted solely from a refined texture may be inconsistent with the input geometry (Fig. 9-a), thus directly using them for geometry refinement causes severe

- (a) Only Texture Refinement (b) Only Geometry Refinement (c) Full Refinement

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

- Fig. 8. Effect of Texture and Geometry Refinement Stages. Top: textured meshes, bottom: geometries. All results are rendered using flat shading. Input for the experiment: the GSO dataset [Downs et al. 2022]

(a) Input Geometry 𝑀𝑀𝑖𝑖

[Figure 53]

| |
|---|

(c) Refined w/ Regularizer

[Figure 54]

| |
|---|

(b) Refined w/o Regularizer

[Figure 55]

| |
|---|

- Fig. 9. Effect of Regularized Normal Integration. (a) Initial geometry.

- (b) Geometry refinement using normal integration w/o regularization. (c) Geometry refinement using normal integration w/ regularization (Ours). Input for the experiment: the GSO dataset [Downs et al. 2022]

distortions (Fig. 9-b). Our regularized normal integration effectively addresses this issue, resulting in high-quality geometry (Fig. 9-c).

- 6 Conclusion and Future Work

In this work, we proposed Elevate3D, a novel 3D model refinement framework that alternates between texture and geometry refinement in a view-by-view fashion to produce high-quality 3D models with well-aligned texture and geometry. We introduced HFS-SDEdit for texture refinement, leveraging high-frequency guidance to achieve high-fidelity enhancements while mitigating the limitations of previous SDEdit-based methods. Through comprehensive experiments, we demonstrated that our framework achieves state-of-the-art quality refinement of 3D models compared to recent competitors.

Limitations and Future Work. While our framework produces high-quality textured meshes, it shares a common limitation with similar diffusion-based texturing methods [Chen et al. 2023; Richardson et al. 2023; Tang et al. 2024a]: refinement time increases proportionally with the number of views that need to be generated by the diffusion model. Recent advances in increasing the efficiency of diffusion models [Kim et al. 2024; Sauer et al. 2024] offer potential reductions in our framework’s computational cost. Future work will explore integrating such models to optimize Elevate3D’s processing time while maintaining its high-quality output.

Acknowledgments

This work was supported by Pebblous Inc., and the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants (RS-2019-II91906, Artificial Intelligences Graduate School Program (POSTECH), RS-2024-00457882, AI Research Hub Project) funded by the Korea government (MSIT).

References

Michael S. Albergo and Eric Vanden-Eijnden. 2022. Building Normalizing Flows with Stochastic Interpolants. arXiv:2209.15571 [cs.LG]

Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. 2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 5470– 5479.

Colin Barré-Brisebois and Stephen Hill. 2012. Blending in Detail. https://blog.selfshadow.com/publications/blending-in-detail/.

Yochai Blau and Tomer Michaeli. 2018. The Perception-Distortion Tradeoff. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 6228–6237. https://doi.org/10.1109/cvpr.2018.00652

Xu Cao, Hiroaki Santo, Boxin Shi, Fumio Okura, and Yasuyuki Matsushita. 2022. Bilateral Normal Integration. In ECCV.

Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. 2024a. TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment. Trans. Img. Proc. 33 (March 2024), 2404–2418. https://doi.org/10.1109/TIP.2024.3378466

Dave Zhenyu Chen, Yawar Siddiqui, Hsin-Ying Lee, Sergey Tulyakov, and Matthias Nießner. 2023. Text2Tex: Text-driven Texture Synthesis via Diffusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 18558–18568.

Hansheng Chen, Ruoxi Shi, Yulin Liu, Bokui Shen, Jiayuan Gu, Gordon Wetzstein, Hao Su, and Leonidas Guibas. 2024b. Generic 3D Diffusion Adapter Using Controlled Multi-View Editing. arXiv:2403.12032 [cs.CV]

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. 2023. Objaverse-XL: A Universe of 10M+ 3D Objects. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track. https://openreview.net/forum?id=Sq3CLKJeiz Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2022. Objaverse: A Universe of Annotated 3D Objects. arXiv:2212.08051 [cs.CV]

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke. 2022. Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items. In 2022 International Conference on Robotics and Automation (ICRA) (Philadelphia, PA, USA). IEEE Press, 2553–2560. https://doi.org/10.1109/ICRA46639.2022.9811809

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. 2024. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. In Forty-first International Conference on Machine Learning. https://openreview.net/forum?id=FPnUhsQJ5B

Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. Advances in neural information processing systems 27 (2014).

Jinjin Gu, Haoming Cai, Haoyu Chen, Xiaoxing Ye, Jimmy Ren, and Chao Dong. 2020. PIPAL: a Large-Scale Image Quality Assessment Dataset for Perceptual Image Restoration. arXiv:2007.12142 [eess.IV] https://arxiv.org/abs/2007.12142

Jinjin Gu, Haoming Cai, Chao Dong, Jimmy S. Ren, and Radu Timofte. 2022. NTIRE 2022 Challenge on Perceptual Image Quality Assessment. arXiv:2206.11695 [cs.CV] https://arxiv.org/abs/2206.11695

Yong He, Hongshan Yu, Xiaoyan Liu, Zhengeng Yang, Wei Sun, Saeed Anwar, and Ajmal Mian. 2025. Deep learning based 3D segmentation in computer vision: A survey. Information Fusion 115 (2025), 102722. https://doi.org/10.1016/j.inffus.2024.102722

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models, Vol. 33. 6840–6851.

Hsuan I Ho, Jie Song, and Otmar Hilliges. 2024. SiTH: Single-view Textured Human Reconstruction with Image-Conditioned Diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 538–549.

Yash Kant, Aliaksandr Siarohin, Michael Vasilkovsky, Riza Alp Guler, Jian Ren, Sergey Tulyakov, and Igor Gilitschenski. 2023. iNVS: Repurposing Diffusion Inpainters for Novel View Synthesis. In SIGGRAPH Asia 2023 Conference Papers (Sydney, NSW, Australia) (SA ’23). Association for Computing Machinery, New York, NY, USA, Article 16, 12 pages. https://doi.org/10.1145/3610548.3618149

Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe. 2006. Poisson surface reconstruction. In Proceedings of the Fourth Eurographics Symposium on Geometry Processing (Cagliari, Sardinia, Italy) (SGP ’06). Eurographics Association, Goslar, DEU, 61–70.

Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. 2024. Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. 2021. MUSIQ: Multi-scale Image Quality Transformer. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). 5128–5137. https://doi.org/10.1109/ICCV48922.2021.00510

Geonung Kim, Beomsu Kim, Eunhyeok Park, and Sunghyun Cho. 2024. Diffusion Model Compression for Image-to-Image Translation. In Computer Vision – ACCV 2024: 17th Asian Conference on Computer Vision, Hanoi, Vietnam, December 8–12, 2024, Proceedings, Part V (Hanoi, Vietnam). Springer-Verlag, Berlin, Heidelberg, 148–166. https://doi.org/10.1007/978-981-96-0917-8_9

Jie Long Lee, Chen Li, and Gim Hee Lee. 2024. DiSR-NeRF: Diffusion-Guided ViewConsistent Super-Resolution NeRF. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 20561–20570.

Xiaoyu Li, Qi Zhang, Di Kang, Weihao Cheng, Yiming Gao, Jingbo Zhang, Zhihao Liang, Jing Liao, Yan-Pei Cao, and Ying Shan. 2024. Advances in 3D Generation: A Survey. arXiv:2401.17807 [cs.CV] https://arxiv.org/abs/2401.17807

Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, Rakesh Ranjan, Radu Timofte, and Luc Van Gool. 2023. LSDIR: A Large Scale Dataset for Image Restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. 1775–1787.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. 2023. Flow Matching for Generative Modeling. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=PqvMRDCJT9t

Xingchao Liu, Chengyue Gong, and Qiang Liu. 2022. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. arXiv:2209.03003 [cs.LG]

Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2024. SyncDreamer: Generating Multiview-consistent Images from a Single-view Image. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=MN3yH2ovHb

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2023. Wonder3D: Single Image to 3D using Cross-Domain Diffusion. arXiv preprint arXiv:2310.15008 (2023).

Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan de Geus, Alexander Hermans, and Bastian Leibe. 2025. Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).

Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. 2023. RealFusion: 360deg Reconstruction of Any Object From a Single Image. In CVPR. 8446–8455.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations. https: //openreview.net/forum?id=aBsCjcPu_tE

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li,

Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2024. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL]

Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, and Youngjung Uh. 2023. Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=VUlYp3jiEI

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2023. DreamFusion: Text-to-3D using 2D Diffusion. In ICLR.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision, Vol. 139. 8748–8763.

Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. 2023. TEXTure: Text-Guided Texturing of 3D Shapes. In ACM SIGGRAPH 2023 Conference Proceedings (, Los Angeles, CA, USA,) (SIGGRAPH ’23). Association for Computing Machinery, New York, NY, USA, Article 54, 11 pages. https://doi.org/10.1145/ 3588432.3591503

Severi Rissanen, Markus Heinonen, and Arno Solin. 2023. Generative Modelling with Inverse Heat Dissipation. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=4PJUBT9f2Ol

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-Resolution Image Synthesis With Latent Diffusion Models. In CVPR. 10684–10695.

Nuri Ryu, Minsu Gong, Geonung Kim, Joo-Haeng Lee, and Sunghyun Cho. 2023. 360° Reconstruction From a Single Image Using Space Carved Outpainting. In SIGGRAPH Asia 2023 Conference Papers (SA ’23). Association for Computing Machinery, New York, NY, USA, Article 75, 11 pages. https://doi.org/10.1145/3610548.3618240

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. 2024. Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation. arXiv:2403.12015 [cs.CV] https://arxiv.org/abs/2403.12015

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023a. Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model. arXiv:2310.15110 [cs.CV]

Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. 2024. MVDream: Multi-view Diffusion for 3D Generation. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=FUgrjq2pbB

Zifan Shi, Sida Peng, Yinghao Xu, Andreas Geiger, Yiyi Liao, and Yujun Shen. 2023b. Deep Generative Models on 3D Representations: A Survey. arXiv:2210.15663 [cs.CV] https://arxiv.org/abs/2210.15663

Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep Unsupervised Learning Using Nonequilibrium Thermodynamics. 2256–2265.

Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. 2024. DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior. In The Twelfth International Conference on Learning Representations. https: //openreview.net/forum?id=DDX1u29Gqr

Jiaxiang Tang, Ruijie Lu, Xiaokang Chen, Xiang Wen, Gang Zeng, and Ziwei Liu. 2024a. InTeX: Interactive Text-to-Texture Synthesis via Unified Depth-aware Inpainting. arXiv preprint arXiv:2403.11878 (2024).

Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2024b. DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation. In The Twelfth International Conference on Learning Representations. https://openreview. net/forum?id=UyNXMqnN3c

Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. 2023. Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior. arXiv:2303.14184 [cs.CV]

- A. Tewari, J. Thies, B. Mildenhall, P. Srinivasan, E. Tretschk, W. Yifan, C. Lassner, V. Sitzmann, R. Martin-Brualla, S. Lombardi, T. Simon, C. Theobalt, M. Nießner, J. T. Barron, G. Wetzstein, M. Zollhöfer, and V. Golyanik. 2022. Advances in Neural Rendering. Computer Graphics Forum 41, 2 (2022), 703–735. https://doi.org/10.1111/ cgf.14507 arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14507

Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. 2021. RealESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. arXiv:2107.10833 [eess.IV] https://arxiv.org/abs/2107.10833

Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13, 4 (2004), 600–612.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=ppJuFSOAnM

Chen Henry Wu and Fernando De la Torre. 2023. A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 7378–7387.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin. 2024b. Q-ALIGN: teaching LMMs for visual scoring via discrete text-defined levels. In Proceedings of the 41st International Conference on Machine Learning (Vienna, Austria) (ICML’24). JMLR.org, Article 2216, 15 pages. Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. 2024a. Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=UO7Mvch1Z5

Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Liang Pan Jiawei Ren, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, and Ziwei Liu. 2023. OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2024. Structured 3D Latents for Scalable and Versatile 3D Generation. arXiv preprint arXiv:2412.01506 (2024).

Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, and Zhangyang Wang. 2023. NeuralLift-360: Lifting an In-the-Wild 2D Photo to a 3D Object With 360deg Views. In CVPR. 4479–4489.

Fan Yang, Jianfeng Zhang, Yichun Shi, Bowen Chen, Chenxu Zhang, Huichao Zhang, Xiaofeng Yang, Jiashi Feng, and Guosheng Lin. 2024c. Magic-Boost: Boost 3D Generation with Mutli-View Conditioned Diffusion. arXiv:2404.06429 [cs.CV] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and

Hengshuang Zhao. 2024b. Depth Anything V2. arXiv:2406.09414 (2024).

Qinyu Yang, Haoxin Chen, Yong Zhang, Menghan Xia, Xiaodong Cun, Zhixun Su, and Ying Shan. 2024a. Noise Calibration: Plug-and-Play Content-Preserving Video Enhancement Using Pre-trained Video Diffusion Models. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XXXVI (Milan, Italy). Springer-Verlag, Berlin, Heidelberg, 307–326. https://doi.org/10.1007/978-3-031-72764-1_18

Kim Youwang, Tae-Hyun Oh, and Gerard Pons-Moll. 2024. Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 4347–4356.

Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. 2024. Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild. arXiv:2401.13627 [cs.CV]

Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger.

2022. MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction.

Xianfang Zeng, Xin Chen, Zhongqi Qi, Wen Liu, Zibo Zhao, Zhibin Wang, Bin Fu, Yong Liu, and Gang Yu. 2024. Paint3D: Paint Anything 3D with Lighting-Less Texture Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 4252–4262.

David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. 2023b. Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation. arXiv:2309.15818 [cs.CV] https: //arxiv.org/abs/2309.15818

Junwu Zhang, Zhenyu Tang, Yatian Pang, Xinhua Cheng, Peng Jin, Yida Wei, Munan Ning, and Li Yuan. 2023a. Repaint123: Fast and High-quality One Image to 3D

Generation with Progressive Controllable 2D Repainting. arXiv:2312.13271 [cs.CV]

Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. 2020. NeRF++: Analyzing and Improving Neural Radiance Fields. arXiv:2010.07492 [cs.CV] https://arxiv.org/ abs/2010.07492

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. 2024. CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets. ACM Trans. Graph. 43, 4, Article 120 (July 2024), 20 pages. https://doi.org/10.1145/3658146

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.

Weixia Zhang, Guangtao Zhai, Ying Wei, Xiaokang Yang, and Kede Ma. 2023c. Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 14071–14081.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

- (c) MagicBoost(a) Input (GSO)(b) DreamGaussian(d) DiSR-NeRF(e) Ours (Elevate3D)

- Fig. 10. Qualitative Comparison on 3D Refinement. We compare the 3D refinement results from a low-quality degraded input shown in (a). DreamGaussian [Tang et al. 2024b] refines only the texture, leaving the geometry degraded as seen in (b). MagicBoost lacks a fidelity constraint, resulting in large deviations from the input as seen in (c). DiSR-Nerf maintains high fidelity but struggles to generate high-frequency details as seen in (d). In contrast, our method effectively refines both texture and geometry while preserving input fidelity while producing high-quality, well-aligned textures and geometries with high quality as shown in (e). Inputs: the GSO dataset [Downs et al. 2022]

(a) Input Image (b) Generated Result (TRELLIS) (c) Refined Result (Elevate3D)

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

- Fig. 11. Qualitative Results on Refining TRELLIS Outputs. Due to the domain gap between synthetic training data and real-world images, TRELLIS often struggles to generate high-quality results from real-world inputs images such as in (a), as shown in (b). Therefore, we apply Elevate3D to refine TRELLIS’s outputs. As seen in (c), our method produces realistic textures and accurate geometry, resulting in high-quality refinements. Inputs: ©mec4411/pixabay, ©vimleshtailor/pixabay, ©maja7777/pixabay, ©jacksonmoccelin/pixabay

Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model • i

A Additional Technical Details

- A.1 Details on Geometry Refinement and Filtering

As described in Section 4.2 of the main paper, the geometry refinement stage aims to find a refined depth map 𝑧(𝑢,𝑣) for the surface patch 𝑆𝑖 by minimizing the following energy functional:

𝐸(𝑧) = ∬ 𝜕𝑧 𝜕𝑢 +

2

2

𝑛𝑦(𝑢,𝑣) 𝑛𝑧(𝑢𝑉𝑣)

𝑛𝑥 (𝑢,𝑣) 𝑛𝑧(𝑢,𝑣)

𝜕𝑧 𝜕𝑣 +

+

𝑑𝑢 𝑑𝑣

+ 𝜆 ∬ 𝑧(𝑢,𝑣) − 𝑑(𝑢,𝑣)

(6)

2

𝑑𝑢 𝑑𝑣.

Here, the first term enforces consistency with the estimated normal map n𝑖 = [𝑛𝑥,𝑛𝑦,𝑛𝑧]⊤ and the second term regularizes the solution towards the depth 𝑑(𝑢,𝑣) rendered from the existing mesh 𝑀𝑖. We adapt the normal integration method proposed by Cao et al. [2022] to perform this minimization. Their core contribution is a bilaterally weighted functional designed to handle potential depth discontinuities inherent in surfaces estimated from normal maps. Instead of assuming a globally smooth surface, their method operates under the semi-smooth surface assumption, allowing for one-sided discontinuities. They introduce bilateral weights 𝑤𝑢(𝑢,𝑣) and 𝑤𝑣(𝑢,𝑣) at each pixel (𝑢,𝑣) during optimization. These weights reflect the local surface continuity; values close to 0.5 indicate local smoothness in the respective direction (horizontal for 𝑤𝑢, vertical for 𝑤𝑣), while values approaching 0 or 1 suggest a likely discontinuity boundary.

Filtering of Unreliable Depth Estimate. The depth map 𝑧(𝑢,𝑣), obtained by minimizing Eq. (6), represents the refined geometry 𝑆𝑖. However, areas near depth discontinuities, indicated by 𝑤𝑢 or 𝑤𝑣 deviating from 0.5, can lead to unreliable depth estimates in 𝑧(𝑢,𝑣). Directly using these unreliable values during the Poisson surface reconstruction could introduce geometric artifacts when merging 𝑆𝑖 with the mesh 𝑀𝑖.

To mitigate this, we filter the depth map 𝑧(𝑢,𝑣) based on the continuity weights 𝑤𝑢 and 𝑤𝑣 computed during the optimization process. We identify pixels (𝑢,𝑣) where the estimated surface is potentially unreliable or discontinuous by checking if either weight significantly deviates from the ideal smooth value of 0.5. We mark a pixel as unreliable if:

𝑤𝑢(𝑢,𝑣) < 0.4 or 𝑤𝑢(𝑢,𝑣) > 0.6, or 𝑤𝑣(𝑢,𝑣) < 0.4 or 𝑤𝑣(𝑢,𝑣) > 0.6.

(7)

- A binary mask is then created, keeping only those pixels where

both 𝑤𝑢(𝑢,𝑣) and 𝑤𝑣(𝑢,𝑣) fall within the confidence interval [0.4, 0.6]. This mask highlights regions deemed to be reliably estimated and locally smooth. To further ensure robustness and remove potentially isolated unreliable pixels or thin artifacts near discontinuity boundaries, this binary mask is processed with morphological erosion using a 3 × 3 kernel. The final eroded mask defines the reliable region of the depth map 𝑧(𝑢,𝑣) that is subsequently used in the Poisson surface reconstruction step to update the mesh 𝑀𝑖, ensuring a cleaner and more robust integration of the refined geometry. The seams visible in Fig. 5-b of the main paper are a direct result of this filtering process of removing pixels around detected discontinuities.

- A.2 Projection Mapping Implementation Details As mentioned in Section 4.2 of the main paper, after geometry refinement yields the updated mesh 𝑀˜𝑖, we project the correspond-

ing refined texture image 𝐼𝑖′ along with other relevant textures onto 𝑀˜𝑖 to produce the textured mesh 𝑀𝑖+1. This projection mapping is implemented via a custom OpenGL fragment shader. This shader requires several pre-computed data structures passed as uniforms to operate. These include arrays storing all source texture images (Textures[]) corresponding to the camera path views V = {𝑣0, . . .,𝑣𝑘}, alongside a status array (IsRefined[]) indicating which textures have been refined by HFS-SDEdit. Additionally, parameters defining the projection for each view are needed: the projection direction (ProjDirections[]) and the orthographic view and projection matrices (ProjViewMats[], ProjProjMats[]), derived from the camera’s pose for that view. Finally, pre-rendered depth maps (DepthMaps[]) from each viewpoint are crucial for handling occlusions during the blending process. With these inputs prepared, the fragment shader executes the logic outlined in Algorithm 1 for each surface fragment.

The core of the blending logic within Algorithm 1 lies in calculating an appropriate weight (weight) for each texture’s potential contribution. This weighting is carefully designed to ensure highquality results:

- • View-dependent Alignment: The smoothstep(0.3, 1.0,

...) function applied to the cosine similarity between the surface normal and projection direction ensures that views nearly perpendicular to the surface contribute strongly, while contributions smoothly fall off to zero for views at grazing angles (beyond approximately 72.5◦), preventing artifacts from oblique projections.

- • Refinement Priority: By drastically reducing the weight of unrefined textures (×10−8) compared to refined ones (×1.0), the algorithm ensures that the high-quality details introduced by HFS-SDEdit are preferentially used in the final texture wherever a refined view provides relevant, visible information.
- • Occlusion and Transparency: Setting the weight to zero for occluded fragments (based on depth map comparison using the conceptual checkOcclusion function) or for fragments projecting onto transparent background regions of the source textures (via the conceptual sampleTexture function checking alpha) prevents projecting incorrect colors or background details onto the foreground mesh surface.

This combination of projection (represented conceptually by the project function), visibility checks, and carefully designed weighting allows the shader to synthesize a seamless, UV-free texture on the final mesh, integrating the best available information from multiple viewpoints.

- A.3 Further Technical Details

Texture Refinement. To refine the texture at a target view 𝑣𝑖, we adopt several techniques from the state-of-the-art mesh texturing method Paint3D [Zeng et al. 2024] in the texture refinement process with HFS-SDEdit. Specifically, we implement a multi-view depthaware texture sampling strategy by horizontally concatenating the

ALGORITHM 1: Projection Mapping Fragment Shader Logic Input: fragPosition, fragNormal // Fragment attributes Data: Textures[], DepthMaps[], NumTextures, IsRefined[] // Texture data Data: ProjDirections[], ProjViewMats[], ProjProjMats[] // Projection parameters Data: Epsilon // Depth tolerance Output: finalColor // Output fragment color

accumColor ← (0, 0, 0) totalWeight ← 0.0 normFragNormal ← normalize(fragNormal)

for 𝑗 ← 0 to NumTextures −1 do // Initialize weight for texture j weight ← 1.0 // Calculate view-dependent alignment weight alignment ← dot(normFragNormal, ProjDirections[j]) alignmentWeight ← smoothstep(0.3, 1.0, clamp(alignment, 0.0, 1.0)) weight ← weight × alignmentWeight

// Modulate weight based on refinement status if IsRefined[j] then

// Keep weight if refined weight ← weight

#### else

// Down-weight if unrefined weight ← weight × 1e-8

end // Project fragment and get texture coordinates fragTexUV ← project(fragPosition, ProjViewMats[j], ProjProjMats[j]) // Perform occlusion test using depth maps isOccluded ← checkOcclusion(fragPosition, fragTexUV, DepthMaps[j], ProjViewMats[j], ProjProjMats[j], Epsilon) if isOccluded then

weight ← 0.0

end // Sample texture and check transparency texColor ← sampleTexture(Textures[j], fragTexUV) if texColor.alpha < 1.0 then

// This case indicates backround in our implementation weight ← 0.0

end // Accumulate weighted color if weight > 0 then

accumColor ← accumColor + texColor.rgb × weight totalWeight ← totalWeight + weight

#### end

end if totalWeight > 1e-8 then

finalColor.rgb ← accumColor / totalWeight finalColor.alpha ← 1.0

#### else

// Default color (e.g., black) finalColor ← (0, 0, 0, 0)

end return finalColor

Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model • iii

renderings from the initial viewpoint 𝑣0 and the target viewpoint 𝑣𝑖, where the camera path is defined as V = {𝑣0, . . .,𝑣𝑘}. We render depth, RGB, and refinement mask images, resulting in corresponding depth, RGB, and mask grid images. In the mask grid, only the right half is used as the refinement mask. Subsequently, we perform multiview depth-aware texture refinement with HFS-SDEdit. For the initial viewpoint 𝑣0, we render the next planned view for refinement but utilize only the refined image corresponding to 𝑣0. For depth conditioning, we employ a variant of FLUX2.

Refinement View Selection. Inspired by the automatic camera selectionschemeintroducedinthe mesh texturing literature, Text2Tex [Chen

et al. 2023], we choose the camera path V = {𝑣0, . . .,𝑣𝑛, . . .,𝑣𝑘} as follows. We begin by defining a sparse camera schedule with polar angles [45◦, 90◦, 135◦] and corresponding azimuthal angles [0◦, 45◦, 180◦, 270◦]. Since an orthographic camera is used, these viewpoints cover most of the visible regions for refinement. However, we employ an automatic camera selection process to address areas that remain unrefined. We sample 100 views from the sphere. For each viewpoint 𝑣𝑛, we render the object to obtain a foreground mask 𝑚fg and the refinement mask 𝑚. We also compute a cosine similarity map𝑚cos to evaluate viewpoint quality. We then calculate the ratio 𝑟𝑛 as

𝑚𝑖 ⊙ 𝑚cos 𝑚fg

𝑟𝑛 =

for each viewpoint and select the one with the highest ratio for further refinement, ensuring we prioritize the most informative angle. Finally, when the ratio across the object falls below 0.02, we conclude that sufficient coverage has been achieved and terminate the refinement process.

Geometry Refinement Detail. Section 4.2 of the main paper de-

scribes inferring the normal map n𝑖 from the refined texture 𝐼𝑖′. In practice, our implementation incorporates an additional normal

blending step prior to normal integration to further enhance the robustness of the geometry refinement stage. Specifically, before minimizing the energy functional, we blend two normal maps: the map n𝑖, inferred from 𝐼𝑖′ that captures fine, texture-consistent details, and a base normal map representing the current geometry of the mesh 𝑀𝑖. This blending, performed using the UDN blending method [Barré-Brisebois and Hill. 2012], helps combine the details from the refined texture’s normal map with the underlying structure of the existing mesh geometry.

- B Additional Experiments

- B.1 The Effect of the Parameters in HFS-SDEdit As discussed in the main paper, HFS-SDEdit employs a Gaussian kernel 𝐺𝜎, and the high-frequency replacement is performed until it reaches the time step 𝑡stop. To analyze the effect of different parameter choices for 𝜎 and 𝑡stop in HFS-SDEdit, we conduct the same low-quality image refinement experiment presented in the main paper. For assessing refinement fidelity, we utilize full-reference metrics, including PSNR, SSIM [Wang et al. 2004], and LPIPS [Zhang et al. 2018]. To evaluate perceptual quality, we employ non-reference metrics such as MUSIQ [Ke et al. 2021], LIQE [Zhang et al. 2023c],

2https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev

- Table S1. Parameter Sweep of𝜎 in𝐺𝜎 and replacement threshold𝑡stop. Starting the denoising process from 𝑇 = 30, the swapping is performed until 𝑡stop. The table presents the refinement results according to various combinations of 𝜎 and 𝑡stop values. Full-reference metrics include PSNR, SSIM, and LPIPS, while non-reference metrics include MUSIQ, QAlign, LIQE, and TOPIQ.

𝜎 𝑡stop

Full-Reference Metrics Non-Reference Metrics

PSNR ↑ SSIM ↑ LPIPS ↓ MUSIQ ↑ QAlign ↑ LIQE ↑ TOPIQ ↑

2

22 12.8984 0.3144 0.6427 67.1750 4.5764 4.0468 0.5798 20 13.5825 0.3353 0.6215 58.0347 4.1461 3.2241 0.4679 18 14.2574 0.3562 0.6136 49.3993 3.6700 2.5849 0.3793 16 14.8523 0.3739 0.6084 41.9640 3.2286 2.0822 0.3057

4

22 14.1419 0.3514 0.6020 54.7414 4.1851 3.0206 0.4206 20 14.8375 0.3702 0.5980 45.7231 3.6959 2.4237 0.3354 18 15.5876 0.3906 0.5982 39.5193 3.3370 2.1052 0.2828 16 16.2478 0.4057 0.5969 34.8861 3.0465 1.8206 0.2390

16

22 16.2883 0.4014 0.6864 31.1365 3.0541 1.4288 0.2341 20 17.0381 0.4210 0.6840 28.2405 2.7874 1.3988 0.2074 18 17.7739 0.4390 0.6806 26.0664 2.6046 1.3661 0.1913 16 18.3762 0.4529 0.6752 25.1298 2.4729 1.3231 0.1845

- Table S2. Comparison with ProlificDreamer. Elevate3D consistently achieves the better scores across various quality metrics, highlighting its high-quality, visually appealing 3D refinement.

Method

Full-Reference No-Reference PSNR↑ SSIM↑ LPIPS↓ MUSIQ↑ LIQE↑ TOPIQ↑ Q-Align↑

LQ (Baseline) 33.202 0.966 0.057 61.241 2.069 0.457 2.690 Ours 26.163 0.941 0.070 66.527 2.774 0.529 3.215 ProlificDreamer 18.037 0.904 0.156 57.307 1.989 0.439 1.893

- Table S3. Quantitative Results on the Ablation. We compare with the cases where only texture or geometry refinement is performed.

Model

Full-Reference No-Reference PSNR↑ SSIM↑ LPIPS↓ MUSIQ↑ LIQE↑ TOPIQ↑ Q-Align↑ Normal FID↓

LQ (Baseline) 33.202 0.966 0.057 61.241 2.069 0.457 2.690 60.786 FULL 26.163 0.941 0.070 66.527 2.774 0.529 3.215 52.195 Only Geometry Refinement 28.365 0.954 0.068 61.086 2.032 0.465 2.599 48.043 Only Texture Refinement 26.393 0.939 0.067 68.717 3.188 0.572 3.454 55.465

- Table S4. Experimental Results with a Different Diffusion Backbone. The full-reference metrics are evaluated against the high-quality source images. LQ (Baseline) means the low-quality reference images degraded from the high-quality source images.

Full-Reference No-Reference PSNR↑ SSIM↑ LPIPS↓ MUSIQ↑ QAlign↑ LIQE↑ TOPIQ↑

Model

LQ (Baseline) 20.701 0.521 0.662 21.918 2.018 1.303 0.159 SDEdit (strength=0.4) 18.982 0.457 0.689 23.090 2.204 1.316 0.155 SDEdit (strength=0.8) 14.809 0.372 0.741 31.714 2.854 1.446 0.224 NC-SDEdit 17.344 0.424 0.717 24.730 2.347 1.274 0.168 Ours 15.853 0.376 0.601 41.754 3.415 1.876 0.295

TOPIQ [Chen et al. 2024a], and Q-Align [Wu et al. 2024b]. As shown in Table S1, for a fixed𝜎, increasing the number of replacement steps improves fidelity, leading to better full-reference metrics. However, this also increases the risk of preserving degraded details from the

- Table S5. Quantitative Analysis of Fig. 10 in the Main Paper. We compare the quantitative results of the objects shown in Fig. 10 in the Main Paper.

Method

Full-Reference No-Reference PSNR↑ SSIM↑ LPIPS↓ CLIP Sim↑ MUSIQ↑ LIQE↑ TOPIQ↑ Q-Align↑

LQ (Baseline) 32.086 0.940 0.096 88.640 61.203 2.022 0.429 2.577 Ours 25.342 0.902 0.098 90.577 67.674 2.756 0.540 3.277 DreamGaussian 31.559 0.942 0.086 88.387 61.669 2.083 0.457 2.704

- Table S6. Quantitative Comparison on 3D Refinement Using Fullreference metrics. DreamGaussian [Tang et al. 2024b] refines only textures while other methods refine both texture and geometry.

geometry and texture metrics, while geometry-only and textureonly refinements perform well in their respective domains but not both. This highlights our method’s holistic improvement across both geometry and texture aspects.

- B.4 Experimental Results with a Different Diffusion Backbone

To examine the robustness of HFS-SDEdit with respect to the diffusion backbone, we repeated our main experiment using a less powerful model, Stable Diffusion 3.5 medium [Esser et al. 2024], under identical experimental conditions. Results in Table S4 reflect similar trends to the main experiments, demonstrating improvements in non-reference metrics and LPIPS [Zhang et al. 2018] scores.

- B.5 Quantitative Analysis of Fig. 10 in the Main Paper Fig. 10 in the main paper qualitatively demonstrates substantial

visual enhancements achieved by our method over DreamGaussian [Tang et al. 2024b]. However, the corresponding quantitative analysis using non-reference metrics in Table S5 reveals less dramatic numerical differences. This highlights a common limitation where such metrics may not fully reflect perceived visual quality improvements. Despite this, the scores do confirm a relative improvement provided by our approach.

- B.6 Quantitative Comparison of 3D Refinement with Full Reference Metrics

We compare our method with recent 3D model refinement approachesinterms ofvariousfullreference metrics: PSNR, SSIM [Wang et al. 2004], LPIPS [Zhang et al. 2018], and CLIP similarity [Radford et al. 2021]. As summarized in Table S6, we see that when the initial coarse 3D input is already of moderate fidelity, these metrics yield similar scores across different refinement methods. Thus, these metrics may not adequately capture actual perceptual enhancements for the generative refinement task that involves generating details absent in the reference.

- B.7 Robustness to Normal Prediction Failures

Model PSNR↑ SSIM↑ LPIPS↓ CLIP↑ LQ (Baseline) 33.202 0.966 0.057 90.688 Ours 26.163 0.941 0.070 89.886 DreamGaussian [Tang et al. 2024b] 32.720 0.965 0.056 90.890 DiSR-NeRF [Lee et al. 2024] 30.294 0.949 0.088 88.696 MagicBoost [Yang et al. 2024c] 23.865 0.934 0.106 83.412 ProlificDreamer [Wang et al. 2023] 18.037 0.904 0.156 78.509

low-quality image, lowering the performance on non-reference metrics. Conversely, increasing 𝜎 allows a broader band of frequencies to be preserved for a fixed number of replacement steps. While this improves fidelity and enhances the full-reference metrics, it also risks incorporating low-quality domain information from the lowquality reference image, thereby negatively impacting non-reference metrics. The parameter combination of 𝜎 = 4 and 𝑡stop = 18, used in the main experiment, is observed to provide a balance, achieving both high-fidelity and high-quality refinement.

- B.2 Comparison with ProlificDreamer

We provideacomparisonbetweenour method and ProlificDreamer [Wang et al. 2023]. While ProlificDreamer was initially developed for textto-3D synthesis, it can also be applied to refine existing 3D models due to its VSD-loss-based framework. We use the geometry and texture refinement stages of ProlificDreamer for refinement. As shown in Table S2, our method significantly outperforms ProlificDreamer across all evaluated metrics, including MUSIQ [Ke et al. 2021], LIQE [Zhang et al. 2023c], TOPIQ [Chen et al. 2024a], and Q-Align [Wu et al. 2024b]. ProlificDreamer showed lower fidelity refinements due to the lack of explicit fidelity constraints. Quantitatively, ProlificDreamer exhibits substantially lower PSNR, SSIM [Wang et al. 2004], and higher LPIPS [Zhang et al. 2018] values. Qualitatively, we can see in Fig. S3 that direct VSD application often results in artifacts, such as multi-faced Janus effects, due to the absence of explicit fidelity constraints.

After the input image (Fig. S1-a) is refined with HFS-SDEdit (Fig. S1b), we extract extract geometric cues from the refined image using an off-the-shelf monocular normal estimator [Martin Garcia et al. 2025]. However, the normal estimator may occasionally produce failure cases. A common failure mode produces overly smooth or detailless predicted normal maps (Fig. S1-c). Directly Integrating such a compromised normal map produces flat, featureless surfaces that negate the purpose of refinement (Fig. S1-e). Our method, however, is robust to such failures due to the regularization term in our energy functional Eq. (6). This term explicitly encourages the refined surface 𝑆𝑖 to remain close to the depth map derived from the input coarse geometry 𝑀𝑖 (Fig. S1-d). This allows our method to produce meaningfully refined geometry (Fig. S1-e) without severe distortions, avoiding catastrophic failure during the refinement iterations.

B.3 Quantitative Ablation Study of Texture and Geometry Refinement

We performed a quantitative ablation study to evaluate contributions from geometry and texture refinements as shown in Table S3. For geometry evaluation, we computed the FID between high-quality normal maps and those from each refinement method. The full refinement achieves balanced and competitive results across both

Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model • v

[Figure 66]

[Figure 67]

[Figure 68]

- B.8 Additional 2D Refinement Results

In Fig. S2, we show more qualitative comparsion results comparing SDEdit [Meng et al. 2022], NC-SDEdit [Yang et al. 2024a], and HFSSDEdit. Our method shows a good balance between fidelity and quality.

- B.9 Additional 3D Refinement Results

In this section, we present additional qualitative examples. In Fig. S3, we show more qualitative comparison results for the refinement results of the degraded GSO [Downs et al. 2022] dataset. In Fig. S4, we further show more qualitative results for refining the 3D generation results of TRELLIS [Xiang et al. 2024]. We also provide an additional supplementary video. We strongly suggest the readers also see the video for a better understanding of our model’s output quality.

(a) Initial Image (b) Refined Image (c) Predicted Normal Map

[Figure 69]

[Figure 70]

[Figure 71]

(d) Coarse Geometry (e) Geometry Refined w/ Regularization

(e) Normal Integration w/o Regularization

- Fig. S1. Robustness to Normal Prediction Failure. Even when the normal predictor generates poor results, our regularized normal integration successfully preserves the coarse geometry structure, preventing severe distortion.

(d) NC-SDEdit (e) SDEdit (Strength = 0.4)

(f) SDEdit (Strength = 0.8)

(a) Before Degradation

(b) After Degradation

(c) HFS-SDEdit (Ours)

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

- Fig. S2. Additional Qualitative Comparison on 2D Image Refinement. The refinement results in (c), (d), (e), and (f) are obtained from the low-quality image in (b), which was degraded from the image in (a).

Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model • vii

(f) Ours (Elevate3D)

[Figure 102]

[Figure 103]

- Fig. S3. Additional Qualitative Comparison on 3D Refinement We show additional 3D refinement comparisons on the degraded GSO dataset. Among all methods, our model produces the highest quality textures with a well-aligned geometry.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

- (a) Input Image (b) Generated Result (TRELLIS) (c) Refined Result (Elevate3D)
- Fig. S4. Additional Qualitative Comparison on TRELLIS Outputs. We show additional examples of refining 3D generation results of TRELLIS where it generates the 3d in (b) taking real world input image in (a). As seen in (c), our method produces realistic textures and accurate geometry, resulting in high-quality refinements.

