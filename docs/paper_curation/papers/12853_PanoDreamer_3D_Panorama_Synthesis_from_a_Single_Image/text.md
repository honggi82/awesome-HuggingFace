# arXiv:2412.04827v3[cs.CV]21Jan2026

### PanoDreamer: Optimization-Based Single Image to 360 3D Scene With Diffusion

AVINASH PALIWAL, Texas A&M University, USA and Morphic Inc, USA XILONG ZHOU, Max Planck Institute for Informatics, Germany ANDRII TSAROV, Leia Inc., USA NIMA KALANTARI, Texas A&M University, USA

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Generation Trajectory

[Figure 5]

[Figure 6]

Input LucidDreamer WonderJourney Ours

Fig. 1. We introduce a novel method for 360° 3D scene synthesis from a single image. Our approach generates a panorama and its corresponding depth in a coherent manner, addressing limitations in existing state-of-the-art methods such as LucidDreamer [Chung et al. 2023] and WonderJourney [Yu et al. 2024b]. These methods sequentially add details by following a generation trajectory, often resulting in visible seams when looping back to the input image. In contrast, our approach ensures consistency throughout the entire 360° scene. The yellow bars show the regions corresponding to the input in each result.

In this paper, we present PanoDreamer, a novel method for producing a coherent 360° 3D scene from a single input image. Unlike existing methods that generate the scene sequentially, we frame the problem as single-image panorama and depth estimation. Once the coherent panoramic image and its corresponding depth are obtained, the scene can be reconstructed by inpainting the small occluded regions and projecting them into 3D space. Our key contribution is formulating single-image panorama and depth estimation as two optimization tasks and introducing alternating minimization strategies to effectively solve their objectives. We demonstrate that our approach outperforms existing techniques in single-image 360° 3D scene reconstruction in terms of consistency and overall quality1.

CCS Concepts: • Computing methodologies → Image-based rendering. Additional Key Words and Phrases: Single Image to 3D, 3D Scene Generation, Diffusion Models, Panorama Generation, Panorama Depth

###### ACM Reference Format:

Avinash Paliwal, Xilong Zhou, Andrii Tsarov, and Nima Kalantari. 2025. PanoDreamer: Optimization-Based Single Image to 360 3D Scene With Diffusion. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 18 pages. https://doi.org/10.1145/3757377.3763883

1people.engr.tamu.edu/nimak/Papers/PanoDreamer

Authors’ Contact Information: Avinash Paliwal, Texas A&M University, College Station, TX, USA and Morphic Inc, San Jose, CA, USA, avinashpaliwal@tamu.edu; Xilong Zhou, Max Planck Institute for Informatics, Saarbrücken, Germany, xzhou@mpi-inf.mpg.de; Andrii Tsarov, Leia Inc., Mountain View, CA, USA, andrii.tsarov@leiainc.com; Nima Kalantari, Texas A&M University, College Station, TX, USA, nimak@tamu.edu.

SA Conference Papers ’25, Hong Kong, Hong Kong © 2025 Copyright held by the owner/author(s). This is the author’s version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record was published in SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong, https://doi.org/10.1145/3757377.3763883.

1 Introduction

Generating immersive and realistic 3D scenes from a single input image has emerged as one of the important topics in computer vision/graphics, driven by its broad applications including virtual/augmented reality (VR/AR) and gaming. While early algorithms [Jampani et al. 2021; Kopf et al. 2020; Li and Kalantari 2020; Niklaus et al. 2019; Shih et al. 2020; Srinivasan et al. 2017; Tucker and Snavely 2020; Zhou et al. 2016] have achieved high-quality results, they are generally limited to synthesizing novel views with only minor deviation from the input camera position. Consequently, these techniques cannot reconstruct a full 360° scene, which is the primary goal of our work.

With the introduction of diffusion models, the more recent approaches have focused on utilizing these powerful models for 3D scene reconstruction. Specifically, several methods [Li et al. 2024; Zhou et al. 2024, 2025] propose various ways to generate 3D scenes frominputtextprompts.These methods first generate entire panorama from text prompt using pretrained text-to-panorama diffusion models (DMs) and then lift it to 3D. Unfortunately, these approaches are fully generative and do not have a mechanism for reconstructing a 3D scene which is also consistent with a single input image.

Several methods [Chung et al. 2023; Höllein et al. 2023; Ouyang et al. 2023; Shriram et al. 2024; Yu et al. 2024b; Zhang et al. 2024] specifically address the problem of 3D scene reconstruction from a single image. Starting from the input image, these methods typically project it into 3D space, render it from a novel view, and then inpaint the missing regions using a diffusion model. They repeat this process for a series of cameras along a specific path to reconstruct the complete 3D scene. However, a major limitation of these approaches is that, due to the progressive nature of the scene building, they often fail to synthesize coherent 360° scenes, i.e., the start and end of the 360° scenes are contextually different.

In this work, we propose a novel framework, coined PanoDreamer, for generating a coherent 360° 3D scene from a single input image. Departing from the existing methods, which generate the 3D scene one image at a time, we start by producing a coherent 360° panorama from the input image using standard pre-trained inpainting diffusion models. Inspired by MultiDiffusion [Bar-Tal et al. 2023], we formulate the problem as an optimization with two loss terms and propose an alternating minimization strategy to optimize the objective, resulting in a coherent and seamless panoramic image.

The next stage of our approach involves estimating the depth of the panoramic image to project pixels into 3D space and reconstruct the 3D scene. While powerful monocular depth estimation methods [Yang et al. 2024a] exist, these techniques are typically optimized for specific resolutions and struggle to handle large panoramic images effectively. To address this problem, we formulate panoramic depth reconstruction as an optimization task, aiming to simultaneously produce a coherent panoramic depth map and a parametric function that aligns the range of monocular depth to the target depth. We propose an alternating minimization approach to efficiently solve this objective, resulting in a coherent and seamless panoramic depth map.

Given the panoramic image and depth, we directly apply the approach of Shih et al. [2020] to construct a layered depth image (LDI) and inpaint the missing regions in each layer. Next, we build a 3D Gaussian splatting (3DGS) representation [Kerbl et al. 2023] by initializing a set of Gaussians through the projection of LDI pixels into 3D space. We then optimize the 3DGS representation to sharpen details and obtain the final scene. We demonstrate that PanoDreamer can reconstruct consistent 360° 3D scenes from single input images that outperform existing methods. In summary, our work makes the following contributions:

- • We propose a novel framework for synthesizing a coherent 3D panoramic scene from a single image.
- • We formulate the problem of single-image panorama generation using an inpainting diffusion model as an optimization task and solve it using an alternating minimization strategy.
- • We frame the task of obtaining panoramic depth from existing monocular depth estimation methods as an optimization problem and propose an alternating minimization method to solve it.

- 2 Related Work 2.1 Panorama Generation

Diffusion models (DMs) have shown promising results across various generative tasks. In particular, several approaches [Bar-Tal et al. 2023; Deng et al. 2023; Frolov et al. 2024; Lee et al. 2023; Li and Bansal

- 2023; Wang et al. 2024; Ye et al. 2024; Zhang et al. 2023] have proposed leveraging pretrained DMs to synthesize panoramic images. For example, DiffCollage [Zhang et al. 2023] reconstructs complex factor graphs and aggregates intermediate output from DMs defined by nodes to generate a panorama. PanoGen [Li and Bansal 2023] utilizes latent diffusion models combined with recursive outpainting to create indoor panoramic images. MultiDiffusion [Bar-Tal et al. 2023] frames the problem of panoramic image generation from

pretrained DMs as an optimization process to produce globally consistent images. SynDiffusion [Lee et al. 2023] builds on this idea and incorporates the LPIPS score [Zhang et al. 2018] between neighboring denoised images into the optimization process. StitchDiffusion [Wang et al. 2024] further proposes averaging the overlapping denoising predictions and fine-tuning a low-rank adaptation (LoRA) module [Hu et al. 2021]. To improve the efficiency of the generation process, SpotDiffusion [Frolov et al. 2024] shifts non-overlapping denoising windows over time to synthesize a coherent panorama efficiently. All of these methods generate panoramas from a text prompt and cannot incorporate an input image into the generation process.

In contrast to these approaches, PanoDiffusion [Wu et al. 2023b] is designed to generate panoramas from a masked input image. Similarly, MVDiffusion [Deng et al. 2023] can produce a panorama from a single image by stitching multiple images using pixel-wise correspondences and attention modules. However, both of these approaches require training and struggle to generalize to diverse scenarios.

- 2.2 View synthesis from a single input image

Numerous methods have been proposed to synthesize scenes from a single input image. One category of these methods [Jampani et al. 2021; Kopf et al. 2020; Niklaus et al. 2019; Pu et al. 2023; Shih et al. 2020] addresses this problem in a modular manner, decomposing the synthesis process into several independent components. For example, Shih et al. [2020] estimate a layered depth image (LDI) representation to reconstruct novel views. Niklaus et al.[2019] use the estimated depth to map the input image to a point cloud and train a network to fill in the missing areas.

The second group of methods [Li and Kalantari 2020; Srinivasan et al. 2017; Tucker and Snavely 2020; Zhou et al. 2016] synthesizes scenes from a single input image in an end-to-end manner. Among these approaches, Zhou et al.[2016] propose synthesizing scenes by first estimating optical flow and then warping the input image to novel views. Srinivasan et al.[2017] use two sequential convolutional neural networks to estimate disparity and refine the warped images. Several approaches propose synthesizing intermediate scene representations to achieve view synthesis. For example, SynSin [Wiles et al. 2020] estimates a point cloud of a scene, and several methods [Li and Kalantari 2020; Tucker and Snavely 2020] synthesize light fields using the estimated multi-plane image (MPI) representation. PixelNeRF [Yu et al. 2021] trains a NeRF prior and can synthesize NeRF from a single input image without performing test-time optimization. Additionally, several approaches [Bello and Kim 2024; Paliwal et al. 2023] focus on improving the viewdependent effects for single-view view synthesis. However, all of these methods are designed only for view synthesis within a narrow angle or restricted camera movement and cannot be generalized to the entire 360° scene.

- 2.3 3D Scene Generation

Reconstructing an entire 3D scene is a challenging problem, as it requires maintaining both content and depth consistency across a wide range of camera trajectories. Many approaches have been

proposed to achieve 3D scene generation, typically leveraging pretrained, powerful 2D diffusion priors, such as latent diffusion models (LDMs), to synthesize 3D scenes by optimizing different 3D representations, such as NeRF and 3DGS. These approaches can be categorized into two groups based on the input condition.

The first group of methods [Chung et al. 2023; Engstler et al.

- 2024; Liang et al. 2025; Ni et al. 2025; Ouyang et al. 2023; Shriram et al. 2024; Yu et al. 2024a,b; Zhang et al. 2024] generates 3D scenes from text or images in a progressive manner. Starting from a single image, either provided by the user or generated from a text prompt, these methods typically perform progressive inpainting, monocular depth estimation, and 3D optimization for novel views in the 3D scene. These approaches differ in their 3D representation, image inpainting, and depth refinement strategies. However, since the 3D scene is generated through progressive inpainting of single inputs, these methods struggle to preserve coherency, making it difficult to synthesize consistent 360° scenes.

The second group of methods [Li et al. 2024; Zhou et al. 2024, 2025] generates 360° 3D scenes in a two-step process. They synthesize coherent panoramas by leveraging pretrained text-to-panorama DMs, which are then lifted to 3D using different inpainting and depth estimation strategies. Although these approaches are capable of generating consistent 3D scenes from inputs, they are textconditioned only and do not have any mechanism to reconstruct a scene consistent with a single input image. In comparison, our method, PanoDreamer, not only generates coherent 3D scenes but also allows users to condition the generation on any single input image.

- 3 Preliminaries

In this section, we describe MultiDiffusion [Bar-Tal et al. 2023], an approach that leverages a pre-trained diffusion model, without any fine-tuning, to produce results in various image or condition spaces. For example, this technique can generate outputs at resolutions different from the base model’s native resolution (e.g., panoramas) or synthesize images using region-based text prompts. Here, we focus our discussion on the former example, as it is most relevant to our approach.

MultiDiffusion uses a pre-trained diffusion model, Φ, which operates on images of size 𝐻 × 𝑊 as the base model. Starting with an image 𝐼𝑇 initialized with Gaussian noise and conditioned on a text prompt 𝑝, the base model iteratively denoises 𝐼𝑇, producing a sequence of intermediate images 𝐼𝑇−1, · · · ,𝐼1 and ultimately generating a clean image 𝐼0 as follows:

𝐼𝑡−1 = Φ(𝐼𝑡|𝑝). (1)

The goal of MultiDiffusion is to leverage this base model to generate an image 𝐽0 at a larger resolution 𝐻′ ×𝑊 ′. The MultiDiffusion process begins with a noisy high-resolution image, 𝐽𝑇, and produces a clean image 𝐽0 through a sequence of gradually denoised images 𝐽𝑇−1, · · · , 𝐽0. Given the optimal high-resolution image at the current step, 𝐽𝑡∗, the key idea of MultiDiffusion is to ensure that the output of the base diffusion model Φ(𝐹𝑖(𝐽𝑡∗)|𝑝) is as close as possible to the high-resolution image at the next step 𝐹𝑖(𝐽𝑡−1), locally. Note that 𝐹𝑖 is an operator that maps the high-resolution image space to the base

model’s space (via cropping, in this case). Enforcing this similarity in the 𝐿2 sense, we arrive at the following objective:

##### ∑︁𝑛

𝑊𝑖 ⊙ 𝐹𝑖(𝐽) − Φ(𝐹𝑖(𝐽𝑡∗)|𝑝) 2 , (2)

𝐽𝑡∗−1 = argmin

𝐽

𝑖=1

where𝑊𝑖 is a weight map (𝑊𝑖 = 1 in this case), 𝑛 refers to the total number of crops, and ⊙ denotes the element-wise product.

Since this objective is quadratic, the solution can be easily obtained in closed form as follows:

∑︁𝑛

𝐹𝑖−1(𝑊𝑖)

⊙ 𝐹𝑖−1(Φ(𝐹𝑖(𝐽𝑡∗)|𝑝)), (3)

𝐽𝑡∗−1 =

𝑛 𝑗=1 𝐹𝑗−1(𝑊𝑗)

𝑖=1

where 𝐹𝑖−1 is the inverse of the cropping operator, which places the content into the appropriate location in the high-resolution

image. At a high level, this solution aggregates (adds) the outputs of the base diffusion model for overlapping crops and normalizes the resulting image by the total number of crops at each pixel.

Starting from the noisy high-resolution image 𝐽𝑇∗ = 𝐽𝑇, MultiDiffusion uses this process to obtain the optimal intermediate highresolution images 𝐽𝑡∗, resulting in the final clean high-resolution image 𝐽0∗.

4 Algorithm

Given a single input image𝐼, our goal is to reconstruct a coherent 360 scene using a 3D Gaussian representation [Kerbl et al. 2023]. Unlike existing methods that produce the 3D scenes through progressive projection and inpainting, we begin by generating a coherent 360° panorama from the input image (Sec. 4.1). We then estimate a coherent and consistent depth from the generated panorama (Sec. 4.2). Finally, we inpaint the occluded regions using layered depth image (LDI) inpainting and use the inpainted layers to reconstruct a 3DGS representation (Sec. 4.3).

4.1 Single-Image Panorama Generation

We begin by discussing our method for generating a larger image from a single input image, then explain the specific details for panorama generation in Sec. 4.1.1. The overview of our approach is provided in Fig. 2. Given an input image 𝐼 placed on a larger canvas 𝐿 of size 𝐻′ × 𝑊 ′, our goal is to fill in missing areas in 𝐿 using an inpainting diffusion model Φ, which operates on fixed lowerresolution images of size 𝐻 ×𝑊 . In addition to a text prompt 𝑝, this model takes a mask 𝑀 denoting the missing regions and a masked image (1 − 𝑀) ⊙ 𝑋 as inputs. It progressively denoises a Gaussian noise image 𝐼𝑇 to obtain a clean image 𝐼0 containing the hallucinated details, with each step following 𝐼𝑡−1 = Φ(𝐼𝑡|𝑝,𝑀, (1 − 𝑀) ⊙ 𝑋). A straightforward approach is to use this model to gradually outpaint the high-resolution image, starting from the regions covered by the input. However, this approach often results in noticeable contextual inconsistencies and seams, as shown in Fig. 3 (Progressive Inpainting).

Inspired by MultiDiffusion, we address this issue by formulating the problem as an optimization. MultiDiffusion can be adapted to this problem in a straightforward manner by replacing the base diffusion model with an inpainting model and reformulating the objective in Eq. 2 as follows:

S1 Iterate until convergence S2

[Figure 7]

[Figure 8]

Inpainting Mask

[Figure 9]

[Figure 10]

[Figure 11]

| |
|---|
| |

[Figure 12]

Random

[Figure 13]

Di usion Model

[Figure 14]

[Figure 15]

[Figure 16]

Stage 1 (S1)S2

Input image

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Random Crop

Di usion Model

[Figure 22]

[Figure 23]

Input condition Initialization

Denoised

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Fig. 2. We provide an overview of our proposed MultiConDiffusion process, which consists of two stages. In the first stage, we fix the input condition 𝐿 and apply the diffusion model to overlapping crops of the image at the current time step. The outputs are then aggregated to produce the image at the next time step. This process is repeated until the fully denoised image 𝐽0 is obtained. In the next stage, we replace the current input condition with 𝐽0. These two stages are repeated until convergence.

∑︁𝑛

∥𝑀𝑖 ⊙ 𝐹𝑖(𝐽𝑡−1) − Φ(𝐹𝑖(𝐽𝑡∗)|C𝑖) ∥,2 (4) where

L(𝐽𝑡−1|𝐽𝑡∗) =

𝑖=1

C𝑖 = {𝑝,𝑀𝑖,𝑀𝑖 ⊙ 𝐹𝑖(𝐿)}. (5)

Here, 𝑀𝑖 is a random inpainting mask for the 𝑖th crop, and 𝐿 is the high-resolution condition image. This objective ensures that the

output of the inpainting diffusion model, Φ(𝐹𝑖(𝐽𝑡∗)|C𝑖), is as close

- as possible to the corresponding crop of the high-resolution image
- at the next step 𝐹𝑖(𝐽𝑡−1) in the masked areas 𝑀𝑖. This objective can be minimized similarly to Eq. 3 as follows:

∑︁𝑛

𝐹𝑖−1(𝑀𝑖)

##### ⊙ 𝐹𝑖−1(Φ(𝐹𝑖(𝐽𝑡∗)|C𝑖)), (6)

𝐽𝑡∗−1 =

𝑛 𝑗=1 𝐹𝑗−1(𝑀𝑗)

𝑖=1

Based on this equation, it is easy to infer that the solution depends on the high-resolution input condition, 𝐿, of the MultiDiffusion process. As shown in Fig. 3, MultiDiffusion results vary drastically depending on how the input condition is set. In particular, both simple methods for obtaining the input condition, such as placing the input image on a black canvas or using progressive inpainting, produce inconsistent results.

To address this issue, we make a key observation that the ideal input condition is a coherent and consistent high-resolution image. However, obtaining such an image, 𝐽0, is the goal of our optimization and is not available beforehand. Therefore, we propose to incorporate this observation as an additional term in our objective as follows:

∑︁ 1

𝐽˜0 ··· 𝐽˜𝑇−1,𝐿˜ = argmin 𝐽0···𝐽𝑇 −1,𝐿

##### L(𝐽𝑡−1|𝐽𝑡∗) + ∥𝐿 − 𝐽0∥2 (7)

𝑡=𝑇

where the first term is the adapted MultiDiffusion objective for all time steps, while the second term forces the condition image 𝐿 to be close to the clean high-resolution image 𝐽0. Note that the output

[Figure 28]

Progressive Inpaitning

[Figure 29]

|[Figure 30]|
|---|

MultiDi usionOurs

|[Figure 31]|
|---|

[Figure 32]

MultiDi usion

[Figure 33]

Fig. 3. We compare the results of our MultiConDiffusion process against MultiDiffusion and progressive inpainting. The green bar shows the location of the input image. We show MultiDifussion results with two different input conditions (shown on the top left): black canvas with input image (second row), and progressive inpainting result (third row). Our method produces coherent results, while the alternative approaches produce images with seams and inconsistencies.

of this process, 𝐽𝑇−1, . . ., 𝐽0, depends on the condition image 𝐿. As such, 𝐽𝑡∗ is the optimal solution at time 𝑡 given the current condition image 𝐿, and it differs from the final optimal solution, 𝐽˜𝑡, which is obtained using the optimal condition image 𝐿˜. We call this equation the MultiConDiffusion objective, as the high-resolution diffusion process in our case is conditional.

Simultaneously solving for all the images in this objective is a difficult task. Therefore, we propose an alternating minimization strategy that solves for 𝐽𝑇−1, . . ., 𝐽0 and 𝐿 in the following two stages:

Stage 1: Here, we fix 𝐿 and minimize Eq. 7 by finding the optimal 𝐽𝑇−1, . . ., 𝐽0. Since 𝐽𝑇−1, . . ., 𝐽1 do not influence the second term (as different steps are assumed to be independent), we can use Eq. 6 to obtain their solution in closed form. On the other hand, since 𝐽0 appears in both terms, and both terms are quadratic with respect to it, the final solution is a weighted combination of the solution to the first term (Eq. 6) and the second term (𝐽0∗ = 𝐿). In practice, however, we found that plausible results can still be obtained even when the second term is ignored.

To summarize, as shown in Fig. 2, starting from 𝐽𝑇, we aggregate the output of the inpainting diffusion model over different overlapping crops to obtain the image at the next time step, resulting in a sequence of optimal 𝐽𝑇∗−1, . . ., 𝐽0∗ given the current fixed high-resolution input condition 𝐿.

Stage 2: During this stage, we fix 𝐽𝑇−1, . . ., 𝐽0 and find the optimal 𝐿 that minimizes Eq. 7. 𝐿 influences both the first term, as the diffusion model is conditioned on it (see Eq. 5), and the second

[Figure 34]

[Figure 35]

OursInputDAV2

[Figure 36]

[Figure 37]

[Figure 38]

- Fig. 4. We compare the result of our method, PanoDepthFusion, against applying Depth Anything V2 (DA V2) [Yang et al. 2024b] on the full image. The results obtained by DA V2 lacks details and is geometrically inconsistent. Our approach, on the other hand, produces highly detailed and consistent depth maps.

[Figure 39]

[Figure 40]

[Figure 41]

Patchwise Average (Initialization)

Iteration 1 Iteration 4

[Figure 42]

Input Initialization

[Figure 43]

[Figure 44]

Iteration 4

- Fig. 5. Averaging the patch depth estimates leads to banding artifacts since the depth maps are relative and not consistent. On the top right, we show that projecting the image into 3D using such a depth map results in clear

banding artifacts. Since we initialize𝐺𝜃𝑖 with the identity line, the patchwise average serves as our initial depth estimate during the optimization of Eq. 8.

We also show our results after one and four iterations of optimization. After only four iterations, the seams disappear. As seen on the bottom right, the banding artifacts also disappear from the projected image.

term. Obtaining the optimal solution to each term independently is straightforward. The optimal solution to the first term is 𝐿∗ = 𝐿, since if 𝐿 was used to produce the current 𝐽𝑇−1, . . ., 𝐽0, it is likely the best option for reproducing the same results. Moreover, since the second term is quadratic, the solution is simply 𝐿∗ = 𝐽0. Although obtaining the solution to each term is straightforward, computing the optimal solution considering both terms is difficult. However, assuming that 𝐿 and 𝐽0 are close to each other—i.e., MultiConDiffusion does not diverge significantly from the condition image in one pass—it is reasonable to assume that 𝐿∗ = 𝐽0 is close to the optimal solution for both terms.

We perform the optimization by first initializing 𝐽𝑇 with Gaussian noise and 𝐿 by placing the input image on a black canvas. We then alternate between stages 1 and 2 iteratively until convergence. At the end of this process, we can use either 𝐽˜0 or 𝐿˜ as the final result. Fig. 3 compares MultiConDiffusion with MultiDiffusion and progressive inpainting.

- 4.1.1 Panorama Generation Details. We slightly modify the MultiConDiffusion process to adapt it for generating panoramas from a single image. Our goal is to produce a cylindrical panorama, so in this case, MultiConDiffusion operates in the cylindrical domain,

and the sequence 𝐽𝑇, . . ., 𝐽0 is defined within this domain. Since the base inpainting diffusion model operates on perspective images, 𝐹𝑖 performs both cropping and projection from the cylindrical to

Step 1 - Fixed and aggregate

[Figure 45]

S1 Iterate until convergence S2

Step 2 Fixed

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Optimize

Depth Estimator

[Figure 52]

[Figure 53]

[Figure 54]

Optimize

[Figure 55]

Depth Estimator

[Figure 56]

[Figure 57]

[Figure 58]

Optimize

[Figure 59]

Depth Estimator

Panorama Depth

[Figure 60]

[Figure 61]

Fig. 6. We show the overview of PanoDepthFusion. We first apply an existing depth estimator to the overlapping patches of the input image to obtain a set of patch depth estimates. We then perform optimization in two stages. In the first stage, the depth patches are adjusted using a piecewise linear function 𝐺𝜃𝑖 , and the adjusted patches are then aggregated to obtain the panoramic depth. In the second stage, we optimize the parameters 𝜃𝑖 of the parametric functions to match the adjusted patch depth estimates with the corresponding regions in the panoramic depth. These two steps are repeated until convergence.

the perspective domain. Similarly, 𝐹𝑖−1 projects the pixels from the perspective image back to the cylindrical image, placing them in

the appropriate locations.

We experimented with bilinear interpolation during the projection; however, interpolation smoothed out the noise, which negatively affected the performance of the diffusion model. Therefore, we instead use nearest-neighbor interpolation for both 𝐹𝑖 and 𝐹𝑖−1. Additionally, we use an FOV of 45° for the perspective camera and carefully set the resolution of the cylindrical image to ensure a near one-to-one mapping between the pixels of the cylindrical and perspective images to preserve the noise pattern during projections.

This process allows us to produce a contextually coherent and seamless 360° cylindrical panorama, which we use to reconstruct the 3D scene. In our experiments, we apply 20 iterations of MultiConDiffusion (Stage 1 + Stage 2) to obtain the final cylindrical panorama.

4.2 Panorama Depth Estimation

Given the panoramic image 𝐽˜0, our goal is to estimate its depth 𝐷. In recent years, several powerful monocular depth estimation methods [Yang et al. 2024a,b] have been introduced. These approaches can estimate highly detailed relative depth but typically perform best at a specific image size. Beyond this optimal resolution, they often produce results that lack detail and geometric consistency. Consequently, applying these methods directly to panorama depth estimation leads to poor results, as shown in Fig. 4.

We address this problem by obtaining 𝐷 through a combination of estimated depth maps on patches using an existing technique, Ψ, i.e., Ψ(𝐹𝑖(𝐽˜0)). However, naïvely combining the patches (e.g., through averaging) leads to unsatisfactory results (see Fig. 5), as the patch depth estimates are relative and can be inconsistent. To overcome this, we pose the problem of obtaining panoramic depth from patch depth estimates as an optimization task. Our key insight is that the panoramic depth 𝐷 should be close to the estimated depth after it

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

Panorama Image

Project 3D Gaussians 3DGS Optimization

and Depth

Inpaint

Initialization

LDI

- Fig. 7. We present an overview of our inpainting and 3DGS optimization process. Given a cylindrical panorama and its corresponding depth, we first convert them to the LDI representation. We then inpaint both the image and depth layers. Note that while all images and depth maps are cylindrical, we show only a small crop for clarity. Next, we initialize the Gaussians by assigning a single Gaussian to each pixel and projecting them into 3D space. Finally, we perform 3DGS optimization to obtain the 3D representation.

- Table 1. Numerical comparison of MultiConDiffusion for single image wide-image generation against other relevant methods. CLIP-IQA+ and Q-Align measure the quality, A-CLIP and A-Align asses the aesthetic, and C-CLIP and C-Style evaluate the consistency of the results.

|Method|Q-IQA ↑ Q-Align ↑ A-CLIP ↑ A-Align ↑ C-CLIP ↑ C-Style ↓|
|---|---|
|Progressive L-MAGIC [Cai et al. 2024] MultiDiffusion [Bar-Tal et al. 2023] SyncDiffusion [Lee et al. 2023] MultiConDiffusion (ours)|0.520 4.164 5.779 3.314 0.862 0.019 0.550 4.331 5.865 3.426 0.842 0.069 0.523 4.390 5.953 3.516 0.869 0.030 0.535 4.290 5.893 3.429 0.864 0.016 0.530 4.481 5.992 3.696 0.881 0.011<br><br>|

has been globally aligned through a parametric function. This can be formally written as:

##### ∑︁𝑛

∥𝐹𝑖(𝐷) − 𝐺𝜃𝑖 (Ψ(𝐹𝑖(𝐽˜0)))∥2, (8)

𝐷∗,𝜽∗ = argmin

𝑖=1

𝐷,𝜽

where 𝐺𝜃𝑖 is the parametric function, and 𝜽 = {𝜃1, . . .,𝜃𝑛} represents the set of parameters for different patches. In our implementation, we use a piecewise linear function, where each parameter consists of a series of scale and shift values.

Solving for both 𝐷 and 𝜽 simultaneously is challenging. Therefore, we propose performing this optimization through alternating minimization, consisting of two stages (see Fig. 6). In the first stage, we fix 𝜽 and find the optimal 𝐷. Since the objective is quadratic, the solution can be obtained in closed form, similar to Eq. 3. The only difference is that Φ, the diffusion model, is replaced with Ψ, and 𝑊𝑖 = 1. In the second stage, we fix 𝐷 and find the optimal 𝜽. This is a least-squares regression problem, which we solve using standard packages.

Starting with all 𝐺𝜃𝑖 as the identity line (i.e., a linear function with a slope of 1), we alternate between the first and second stages iteratively until convergence (four iterations in our implementation). Once converged, we obtain a coherent and consistent panoramic depth, as shown in Figs. 4 and 5.

- 4.3 Inpainting and 3DGS Optimization

We now discuss our process for reconstructing the 3D scene using our generated panorama and the corresponding depth map (see the overview in Fig. 7). While the estimated depth can be used to project the cylindrical image into 3D space, when the scene is viewed from any position other than the panorama’s center of projection, occluded regions become visible. To address this, we utilize the layered depth image (LDI) inpainting approach by Shih et al. [2020], which performs effective depth-aware texture inpainting while also providing the corresponding depth. We use a four-layer LDI representation (foreground, background, and two intermediate layers) based on agglomerative clustering by disparity.

We then use these cylindrical layered images and depth maps to initialize a set of 3D Gaussians. Specifically, we assign a Gaussian to each pixel of the image at each layer and project them into 3D according to the corresponding depth. The color of each Gaussian is initialized based on the pixel color (without spherical harmonics); we initialize the rotation matrix with identity, assign the scale following Paliwal et al. [2025], and set the opacity to 0.5. During this process, we keep track of which Gaussians correspond to which layer, as this information is required for optimization.

Next, we perform 3DGS optimization for 1000 iterations. To do this, we set up 240 evenly rotated cameras from the center of projection and project the layered images and depth maps to these cameras. During the optimization, we randomly sample one of these cameras and optimize the Gaussians according to their corresponding layer. Additionally, we composite all the four layers and use the composited image as a reference to optimize all the Gaussians. Note that per layer and composite losses are optimized simultaneously. We use the original 3DGS reconstruction loss along with an 𝐿2 loss between the rendered and layered depth maps. In addition, to be able to produce consistent results from novel views, we use the depth-based novel view loss, proposed by Zhu et al. [2024]. Furthermore, pruning and densification are performed following 3DGS. Once the optimized 3DGS representation is obtained, we can synthesize novel views of the scene and produce coherent and seamless results.

5 Results

In this section, we evaluate both the generated texture and depth of our proposed algorithms. Specifically, we compare our approach against state-of-the-art wide-image generation and 3D scene generation methods, both visually and numerically. For evaluation, we compile a test set of 28 real and synthetic scenes sourced from LucidDreamer [Chung et al. 2023] and WonderJourney [Yu et al. 2024b]. For numerical evaluation, we employ several metrics to evaluate different aspects of the results: (1) Quality - we assess the quality of results using CLIP-IQA+ [Wang et al. 2023] and Q-Align [Wu et al. 2023a] scores. CLIP-IQA+ and Q-Align are built upon contrastive language-image pre-training (CLIP) [Radford et al. 2021] and large

[Figure 89]

Progressive

InpaitningOursL-MAGICOurs

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

MultiDi usionOursSyncDi usionOurs

[Figure 94]

[Figure 95]

[Figure 96]

- Fig. 8. We compare the wide-images generated by MultiConDiffusion with those from other methods. Other approaches often result in sharp discontinuities and contextual inconsistencies. For instance, in the top example, the MultiDiffusion result shows a mismatch between the generated sky and the input sky.

multi-modality models (LMMs) for image quality assessment, respectively. (2) Aesthetic - we use the CLIP aesthetic score (A-CLIP) and A-Align [Wang et al. 2023] to measure results aesthetic. (3) Consistency - we compute the similarity (C-CLIP) and style loss [Gatys et al. 2016] (C-Style) of the CLIP embeddings of random pairs of non-overlapping patches in the results.

Moreover, we numerically evaluate our depth estimation against other techniques, on 30 randomly selected scenes from the realworld Stanford2D3D dataset [Armeni et al. 2017]. For this evaluation, we use Absolute Relative Error (AbsRel), Root Mean Squared Error (RMSE), and three percentage metrics, i.e., the percentages of pixels where the ratio (𝛿) between the estimated and ground truth depth is smaller than 1.25, 1.252, and 1.253.

- 5.1 Wide-Image Reconstruction Comparisons

- Table 1 numerically compares MultiConDiffusion against vanilla progressive inpainting using our base inpainting diffusion model, L-MAGIC [Cai et al. 2024], SyncDiffusion [Shih et al. 2020], and MultiDiffusion [Bar-Tal et al. 2023]. Note that, since these images are not as wide as cylindrical panoramas, we perform the optimization for 15 iterations instead of 20. As seen, our method produces better results than all the other approaches across nearly all metrics. More importantly, the images generated by MultiConDiffusion show better consistency in terms of both style and content, demonstrating the effectiveness of our optimization strategy.

In Fig. 8, we show visual comparison against the other approaches. As seen, progressive inpainting generates results with noticeable seams and strong inconsistency. L-Magic which works based on

progressive inpainting, gradually changes the style of the image closer to the two sides. Similarly SyncDiffusion and MultiDiffusion produce results that are not consistent with the input images. Note that the walkway in center of Multidiffusion’s result does not align with the surrounding regions. In contrast, MultiConDiffusion can generate coherent and seamless wide-images that are significantly better than other approaches.

- 5.2 Panorama Depth Estimation

In Table 3, we numerically evaluate the performance of PanoDepthFusion with two baseline depth estimation methods: Depth Anything V2 [Yang et al. 2024b] (“DA V2 + ours”) and MoGe [Wang et al. 2025] (“MoGe + ours”), a recently introduced (concurrent) singleimage depth estimation method. As seen, our approach with MoGe produces better results than DA V2. We also report the metrics for MoGe with its own patch-wise depth estimation and blending approach (“MoGe”). Comparing “MoGe” with “MoGe + ours” demonstrates that PanoDepthFusion achieves better performance across most metrics. For our 3D scene reconstruction experiments, we adopt DA V2 as the base estimator, since it performs reliably across diverse settings (indoor, outdoor, synthetic) and integrates well with our pipeline. Nonetheless, our approach is agnostic to the choice of depth estimator and can be combined with MoGe or future methods, potentially yielding further improvements.

- 5.3 3D Scene Reconstruction Comparisons

We show numerical comparisons of our PanoDreamer against LucidDreamer [Chung et al. 2023] and WonderJourney [Yu et al. 2024b] in

- Table 2. Numerical comparisons of our approach against the state-of-the-art methods on novel view synthesis. The evaluation metrics are the same as the ones in Table 1.

Method Q-IQA ↑ Q-Align ↑ A-CLIP ↑ A-Align ↑ C-CLIP ↑ C-Style ↓ LucidDreamer [Chung et al. 2023] 0.495 2.911 5.253 2.705 0.848 0.058 WonderJourney [Yu et al. 2024b] 0.504 3.506 5.368 2.834 0.820 0.058 PanoDreamer (ours) 0.443 3.305 5.673 2.772 0.869 0.025

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

Input LucidDreamer WonderJourney Ours

Fig. 9. We compare renderings of PanoDreamer with LucidDreamer [Chung et al. 2023] and WonderJourney [Yu et al. 2024b]. For each methods, we render 3D scene from two novel views. LucidDreamer and WonderJourney produces results with seams and inconsistencies. In comparison, PanoDreamer is capable of generating coherent renderings from novel views. For more visual results and video comparison, please refer to our supplementary materials.

- Table 3. Comparison of PanoDepthFusion with Moge’s [Wang et al. 2025] patch-wise depth estimation + blending approach. Rows 2 − 3 correspond to PanoDepthFusion combined with different depth estimators.

comparing our results against LDI rendering. As seen in Fig. 10, LDI’s rendering contains artifacts around the depth layer discontinuities. These are significantly reduced with our 3DGS optimization partly due to our use of the depth-based novel view loss.

|Method AbsRel ↓ RMSE ↓|𝛿1 ↑ 𝛿2 ↑ 𝛿3 ↑<br><br>|
|---|---|
|MoGe 0.21 0.56 DA V2 + ours 0.31 0.91 MoGe + ours 0.19 0.67|61.66 90.63 98.32 47.05 77.39 92.17 66.47 92.66 98.71<br><br>|

6 Conclusion, Limitation, and Future Work

In conclusion, we have presented a novel method for generating 360° 3D scenes from a single input image. Our approach first generates a panoramic image along with its corresponding depth map. After inpainting occluded regions, these images are used to optimize a 3DGS representation from which novel views can be rendered. To create a coherent and globally consistent panorama, we frame the task as an optimization problem with two terms, solving it effectively through an alternating minimization strategy. Additionally, we pose the problem of estimating panorama depth using an existing monocular depth estimation method as an optimization and address it with alternating minimization. Extensive experiments show that our approach outperforms state-of-the-art methods in both panorama generation and reconstructed 3D scenes. We note that although our main focus has been 3D scene reconstruction, our proposed components, MultiConDiffusion and PanoDepthFusion, are general and have potential applications in related areas such as wide image generation and high-resolution depth prediction.

- Table 2. We use the official code released by the authors, and utilize the same training cameras as ours for a fair comparison. While our image quality and aesthetic scores are slightly worse than WonderJourney, our consistency scores are significantly better. This is because their novel view images are often reasonable when viewed one image at a time, however, different novel view images differ in style and thus are not consistent. Our approach, on the other hand, produces results that are consistent across all views.

We further show visual comparisons against the other methods in Fig. 9. LucidDreamer and WonderJourney produce results with seams and inconsistent style across the two views shown here. In contrast, PanoDreamer can synthesize consistent and seamless scene at both novel views. Please refer to our supplementary materials for video comparison and more visual results.

- 5.4 Novel View Synthesis Ablations

While our method demonstrates clear advantages over prior work, it also has some limitations. First, for our approach to generate appropriate panoramas, like all existing methods, the input image must have a mostly horizontal horizon. Additionally, our approach only reconstructs the front of objects, limiting our ability to capture the areas behind them. In the future, it would be interesting to combine our approach with existing projection-based approaches

We evaluate the effectiveness of different components in our 3D reconstruction pipeline on novel view synthesis task, both visually (Fig 10) and numerically (Table 4). While incorporating our coherent panorama improves LucidDreamer’s performance, their generated novel views still contain artifacts due to inconsistent depth estimation. Moreover, we show the impact of 3DGS optimization by

- Table 4. Numerical ablations to evaluate the effectiveness of different components of our novel view synthesis pipeline. The evaluation metrics are the same as the ones in Table 1.

Method Q-IQA ↑ Q-Align ↑ A-CLIP ↑ A-Align ↑ C-CLIP ↑ C-Style ↓ LucidDreamer [Chung et al. 2023] 0.495 2.911 5.253 2.705 0.848 0.058 LuciDreamer w/ our panorama 0.469 3.012 5.465 2.871 0.862 0.022 Ours + LDI 0.469 2.546 5.479 2.542 0.863 0.016 PanoDreamer (ours w/ 3DGS) 0.443 3.305 5.673 2.772 0.869 0.025

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Input LucidDreamer LucidDreamer with our panorama

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Ours without 3DGS Ours

- Fig. 10. We highlight the effect of various components of our novel view synthesis pipeline. Using our consistent panorama in progressive techniques like LucidDreamer [Chung et al. 2023] improves the texture quality around input image boundaries. However, the rendered texture still contains artifacts due to inconsistent scene depth estimation. Moreover, using the intermediate layered depth image (LDI) representation instead of 3DGS (Ours) leads to subpar rendered images with with artifacts around the depth layer discontinuities.

to address this limitation. Moreover, in some cases the generated textures can appear slightly blurry, particularly near the panorama edges. This results from our optimization process, which propagates content outward from the center to maintain contextual consistency, but occasionally at the cost of sharpness. Despite this, our results remain significantly better than those of existing methods. Lastly, while our approach already enables some level of scene exploration, a promising future direction is to extend this capability by using our 360° representation as a scaffold for state-of-the-art video generation models, enabling a more immersive scene exploration.

Acknowledgments

The project was funded by Leia Inc. (contract #415290). Portions of this research were conducted with the advanced computing resources provided by Texas A&M High Performance Research Computing.

References

Iro Armeni, Sasha Sax, Amir R Zamir, and Silvio Savarese. 2017. Joint 2d-3d-semantic data for indoor scene understanding. arXiv preprint arXiv:1702.01105 (2017). Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. 2023. Multidiffusion: Fusing diffusion paths for controlled image generation. (2023).

Juan Luis Gonzalez Bello and Munchurl Kim. 2024. Novel View Synthesis with ViewDependent Effects from a Single Image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10413–10423.

Zhipeng Cai, Matthias Mueller, Reiner Birkl, Diana Wofk, Shao-Yen Tseng, Junda Cheng, Gabriela Ben-Melech Stan, Vasudev Lai, and Michael Paulitsch. 2024. L-MAGIC: Language Model Assisted Generation of Images with Coherence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7049–7058.

Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee.

2023. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384 (2023).

Zijun Deng, Xiangteng He, Yuxin Peng, Xiongwei Zhu, and Lele Cheng. 2023. MVDiffusion: Motion-aware video diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia. 7255–7263.

Paul Engstler, Andrea Vedaldi, Iro Laina, and Christian Rupprecht. 2024. Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting. arXiv preprint arXiv:2404.19758 (2024).

Stanislav Frolov, Brian B Moser, and Andreas Dengel. 2024. SpotDiffusion: A Fast Approach For Seamless Panorama Generation Over Time. arXiv preprint arXiv:2407.15507 (2024).

Leon A Gatys, Alexander S Ecker, and Matthias Bethge. 2016. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2414–2423.

Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. 2023. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7909–7920.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

Varun Jampani, Huiwen Chang, Kyle Sargent, Abhishek Kar, Richard Tucker, Michael Krainin, Dominik Kaeser, William T Freeman, David Salesin, Brian Curless, et al. 2021. Slide: Single image 3d photography with soft layering and depth-aware inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12518–12527.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.

Johannes Kopf, Kevin Matzen, Suhib Alsisan, Ocean Quigley, Francis Ge, Yangming Chong, Josh Patterson, Jan-Michael Frahm, Shu Wu, Matthew Yu, et al. 2020. One shot 3d photography. ACM Transactions on Graphics (TOG) 39, 4 (2020), 76–1.

Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. 2023. Syncdiffusion: Coherent montage via synchronized joint diffusions. Advances in Neural Information Processing Systems 36 (2023), 50648–50660.

Jialu Li and Mohit Bansal. 2023. Panogen: Text-conditioned panoramic environment generation for vision-and-language navigation. Advances in Neural Information Processing Systems 36 (2023), 21878–21894.

Qinbo Li and Nima Khademi Kalantari. 2020. Synthesizing light field from a single image with variable MPI and two network fusion. ACM Trans. Graph. 39, 6 (2020), 229–1.

Wenrui Li, Yapeng Mi, Fucheng Cai, Zhe Yang, Wangmeng Zuo, Xingtao Wang, and Xiaopeng Fan. 2024. SceneDreamer360: Text-Driven 3D-Consistent Scene Generation with Panoramic Gaussian Splatting. arXiv preprint arXiv:2408.13711 (2024).

Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N. Plataniotis, Sergey Tulyakov, and Jian Ren. 2025. Wonderland: Navigating 3D Scenes from a Single Image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 798–810.

Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Weijie Wang, Haoyun Li, Guosheng Zhao, Jie Li, Wenkang Qin, Guan Huang, and Wenjun Mei. 2025. Wonderturbo: Generating interactive 3d world in 0.72 seconds. arXiv preprint arXiv:2504.02261 (2025).

Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 2019. 3d ken burns effect from a single image. ACM Transactions on Graphics (ToG) 38, 6 (2019), 1–15.

Hao Ouyang, Kathryn Heal, Stephen Lombardi, and Tiancheng Sun. 2023. Text2immersion: Generative immersive scene with 3d gaussians. arXiv preprint arXiv:2312.09242 (2023).

Avinash Paliwal, Brandon G Nguyen, Andrii Tsarov, and Nima Khademi Kalantari.

2023. ReShader: View-Dependent Highlights for Single Image View-Synthesis. ACM Transactions on Graphics (TOG) 42, 6 (2023), 1–9.

Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko, Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalantari. 2025. Coherentgs: Sparse novel view synthesis with coherent 3d gaussians. In European Conference on Computer Vision. Springer, 19–37.

Guo Pu, Peng-Shuai Wang, and Zhouhui Lian. 2023. Sinmpi: Novel view synthesis from a single image with expanded multiplane images. In SIGGRAPH Asia 2023 Conference Papers. 1–10.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

Meng-Li Shih, Shih-Yang Su, Johannes Kopf, and Jia-Bin Huang. 2020. 3d photography using context-aware layered depth inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8028–8038.

Jaidev Shriram, Alex Trevithick, Lingjie Liu, and Ravi Ramamoorthi. 2024. Realmdreamer: Text-driven 3d scene generation with inpainting and depth diffusion. arXiv preprint arXiv:2404.07199 (2024).

Pratul P Srinivasan, Tongzhou Wang, Ashwin Sreelal, Ravi Ramamoorthi, and Ren Ng. 2017. Learning to synthesize a 4D RGBD light field from a single image. In Proceedings of the IEEE International Conference on Computer Vision. 2243–2251. Richard Tucker and Noah Snavely. 2020. Single-view view synthesis with multiplane images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 551–560.

Hai Wang, Xiaoyu Xiang, Yuchen Fan, and Jing-Hao Xue. 2024. Customizing 360-degree panoramas through text-to-image diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 4933–4943.

Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. 2023. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 2555–2563.

Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. 2025. MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details. arXiv preprint arXiv:2507.02546 (2025).

Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. 2020. Synsin: End-to-end view synthesis from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 7467–7477.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. 2023a. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090 (2023).

Tianhao Wu, Chuanxia Zheng, and Tat-Jen Cham. 2023b. PanoDiffusion: 360-degree Panorama Outpainting via Diffusion. In The Twelfth International Conference on Learning Representations.

Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024a. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

10371–10381.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024b. Depth Anything V2. arXiv preprint arXiv:2406.09414

(2024).

Weicai Ye, Chenhao Ji, Zheng Chen, Junyao Gao, Xiaoshui Huang, Song-Hai Zhang, Wanli Ouyang, Tong He, Cairong Zhao, and Guofeng Zhang. 2024. DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion. arXiv preprint arXiv:2410.24203 (2024).

Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. 2021. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4578–4587.

Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu. 2024a. WonderWorld: Interactive 3D Scene Generation from a Single Image. arXiv preprint arXiv:2406.09394 (2024).

Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. 2024b. Wonderjourney: Going from anywhere to everywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6658–6667.

Jingbo Zhang, Xiaoyu Li, Ziyu Wan, Can Wang, and Jing Liao. 2024. Text2nerf: Textdriven 3d scene generation with neural radiance fields. IEEE Transactions on Visualization and Computer Graphics (2024).

Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, and Ming-Yu Liu. 2023. Diffcollage: Parallel generation of large content with diffusion models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 10188–10198.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Haiyang Zhou, Xinhua Cheng, Wangbo Yu, Yonghong Tian, and Li Yuan. 2024. Holodreamer: Holistic 3d panoramic world generation from text descriptions. arXiv preprint arXiv:2407.15187 (2024).

Shijie Zhou, Zhiwen Fan, Dejia Xu, Haoran Chang, Pradyumna Chari, Tejas Bharadwaj, Suya You, Zhangyang Wang, and Achuta Kadambi. 2025. Dreamscene360: Unconstrained text-to-3d scene generation with panoramic gaussian splatting. In European Conference on Computer Vision. Springer, 324–342.

Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A Efros. 2016. View synthesis by appearance flow. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14. Springer, 286–301.

Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. 2024. Fsgs: Real-time fewshot view synthesis using gaussian splatting. In European Conference on Computer Vision. Springer, 145–163.

#### PanoDreamer: Optimization-Based Single Image to 360 3D Scene With Diffusion Supplementary Material

In this supplementary material, we present additional supporting and result figures to further validate and illustrate our findings.

- 7 Context Propagation

First, Fig. S1 illustrates the denoised outputs obtained across different optimization iterations. As shown, with an increasing number of iterations, the central input context progressively propagates outward, ultimately converging to a consistent final result.

- 8 Diversity

As illustrated in Fig. S2, our approach successfully generates diverse, high-quality results for different random initializations.

- 9 Timing and Implementation Details

All experiments are conducted on an NVIDIA RTX A5000 GPU. Our method requires 100 minutes per scene on average, compared to 65 minutes for WonderJourney [Yu et al. 2024b], with panorama generation constituting the primary bottleneck (4 minutes 40 seconds per iteration; 93 minutes for 20 iterations). Computational efficiency was not the focus of this work as the method is built upon MultiDiffusion [Bar-Tal et al. 2023]. Since the introduction of MultiDiffusion, however, several approaches such as SpotDiffusion [Frolov et al. 2024] have achieved up to a 10x speedup, which could potentially be directly integrated into our framework. Further acceleration can be obtained by replacing the current Stable Diffusion model (50 sampling steps) with faster variants that require as few as 4–8 steps (e.g., few-steps distilled models). Moreover, because all panorama windows (crops) are independent, the method can be parallelized across multiple GPUs to substantially reduce runtime.

- 10 MVDiffusion

MVDiffusion [Deng et al. 2023] is a recent diffusion-based approach designed to generate panoramic images conditioned on input views. However, since their model is trained on indoor panoramic data, it tends to overfit, resulting in poor generalization and diminished performance on out-of-distribution scenes, as shown in Fig. S3.

- 11 Additional Results

We present additional qualitative comparisons between our method and recent state-of-the-art wide-image generation approaches [BarTal et al. 2023; Cai et al. 2024; Lee et al. 2023] in Fig.S4 and Fig.S5. As illustrated, MultiConDiffusion (Ours) consistently generates more coherent and seamless panoramas, significantly outperforming existing methods.

We also present further depth comparisons with the state-of-theart depth estimators, Depth-Anything V2 [Yang et al. 2024b] (DA V2) and MoGe [Wang et al. 2025] (patch-wise panorama depth estimator), in Fig. S6 and Fig. S7. As shown, our method generates depth maps with greater detail and improved consistency, particularly

around panorama boundaries (left corners). We highlight prominent artifacts in DA V2’s and MoGe’s results using white arrows. Similar to our approach, MoGe also computes patch-wise depth and blends them to obtain panoramic depth. However, it generates slightly blurry results and can contain scene scale inconsistency in some local patches.

[Figure 120]

## Iteration1Iteration2Iteration3Iteration15

[Figure 121]

[Figure 122]

[Figure 123]

Fig. S1. We show the results of MultiConDiffusion during different iterations of the optimization.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

###### Fig. S2. We show the results of our approach on the same input image across multiple runs. As shown, our approach produces diverse yet consistent results.

###### In training distribution

[Figure 128]

[Figure 129]

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

[Figure 140]

[Figure 141]

- Fig. S3. MVDiffusion [Deng et al. 2023] is a single image panorama generation approach. Since their model is trained on indoor panoramic data, it tends to overfit, resulting in poor generalization and diminished performance on out-of-distribution scenes.

[Figure 142]

[Figure 143]

Progressive Inpaitning

[Figure 144]

[Figure 145]

OursMultiDi usionOursSyncDi usionOursL-MAGICOurs

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

- Fig. S4. We compare the wide-images generated by MultiConDiffusion (Ours) with those from other methods. Other approaches often result in sharp discontinuities and contextual inconsistencies.

[Figure 158]

[Figure 159]

Progressive Inpaitning

[Figure 160]

[Figure 161]

OursMultiDi usionOursSyncDi usionOursL-MAGICOurs

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

- Fig. S5. We compare the wide-images generated by MultiConDiffusion (Ours) with those from other methods. Other approaches often result in sharp discontinuities and contextual inconsistencies.

[Figure 174]

[Figure 175]

###### DAV2DAV2DAV2MoGeMoGeInputMoGeDAV2+OursDAV2+OursInputInputDAV2+Ours

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

###### DAV2DAV2DAV2MoGeMoGeInputMoGeDAV2+OursDAV2+OursInputInputDAV2+Ours

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

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

