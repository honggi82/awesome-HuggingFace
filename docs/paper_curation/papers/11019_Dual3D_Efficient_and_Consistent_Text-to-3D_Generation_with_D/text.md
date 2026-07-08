## Dual3D: Efficient and Consistent Text-to-3D Generation with Dual-mode Multi-view Latent Diffusion

Xinyang Li1 Zhangyu Lai1 Linning Xu2 Jianfei Guo3 Liujuan Cao1 Shengchuan Zhang1 Bo Dai3 Rongrong Ji1

# arXiv:2405.09874v1[cs.CV]16May2024

### Abstract

We present Dual3D, a novel text-to-3D generation framework that generates high-quality 3D assets from texts in only 1 minute. The key component is a dual-mode multi-view latent diffusion model. Given the noisy multi-view latents, the 2D mode can efficiently denoise them with a single latent denoising network, while the 3D mode can generate a tri-plane neural surface for consistent rendering-based denoising. Most modules for both modes are tuned from a pre-trained text-to-image latent diffusion model to circumvent the expensive cost of training from scratch. To overcome the high rendering cost during inference, we propose the dual-mode toggling inference strategy to use only 1/10 denoising steps with 3D mode, successfully generating a 3D asset in just 10 seconds without sacrificing quality. The texture of the 3D asset can be further enhanced by our efficient texture refinement process in a short time. Extensive experiments demonstrate that our method delivers state-of-the-art performance while significantly reducing generation time. Our project page is available at https://dual3d.github.io.

### 1. Introduction

3D generation is a significant topic in the computer vision and graphics fields, which boasts wide-ranging applications across diverse industries, including gaming, robotics, and VR/AR. With the rapid development of the 2D diffusion models, DreamFusion (Poole et al., 2022) introduces Score Distillation Sampling (SDS) to use a pre-trained text-

1Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University 2The Chinese University of Hong Kong 3Shanghai Artificial Intelligence Laboratory. Work done during an internship of Xinyang Li with Shanghai Artificial Intelligence Laboratory. Correspondence to: Liujuan Cao <caoliujuan@xmu.edu.cn>.

conditioned 2D diffusion model (Saharia et al., 2022) for generating 3D assets from open-world texts. However, owing to the absence of 3D priors in 2D diffusion models, these methods frequently encounter low success rates and multi-faceted Janus problem (Poole et al., 2022). On a different trajectory, direct 3D diffusion models (Tang et al., 2023b) offer alternative text-to-3D approaches with the denoising of 3D representations, but they always struggle with incomplete geometry and blurry textures due to the quality disparity between images and 3D representations.

To solve the multi-faceted Janus problem and generate highquality assets with 3D-consistency, multi-view diffusion has garnered increasing interest since it can incorporate the rich knowledge of multi-view datasets. Representative methods, MVDream (Shi et al., 2023) and DMV3D (Xu et al., 2023), introduce multi-view supervision into 2D and renderingbased diffusion models, respectively. Specifically, MVDream fine-tunes a 2D latent diffusion model (LDM) (Rombach et al., 2022) into a multi-view latent diffusion model using multi-view image data, enabling efficient denoising of multi-view images. However, it still necessitates a timeconsuming per-asset SDS-based optimization process to generate a specific 3D asset. Conversely, DMV3D leverages multi-view diffusion in combination with a Large Reconstruction Model (LRM) (Hong et al., 2023), enabling to generate a clean 3D representation during denoising without additional per-asset optimization. Nevertheless, its denoising speed is inferior to MVDream as it necessitates full-resolution rendering at each denoising step. Moreover, DMV3D trains the entire LRM from scratch, leading to a substantial increase in training cost relative to MVDream.

Our goal is to develop a high-quality text-to-3D generation framework with fast generation speed and reasonable training cost. The cornerstone of our framework is a dual-mode multi-view latent diffusion model. Both modes are trained with shared modules and only multi-view images as supervision. During inference, we can toggle to 2D mode to reduce the inference time, or to 3D mode to obtain a noisy-free 3D neural surface for 3D-consistent multi-view rendering. Also, to avoid the high training cost, we leverage the unified formulation of 2D latent features and 3D tri-plane features to design a novel architecture and training process that al-

joinyly tuned from

Text Text

2D LDM

Z˜T 1 Z˜T 2 Z˜T 3 Z˜2 2D Z˜1 3D Z˜0

ZT

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Text

2D 2D

###### …

3D

Render

[Figure 1]

mutli-view noises

mutli-view latents

2D mode

E icient Texture Refinement

shared weights

[Figure 2]

Neural Surface

Tri-planes

| | |
|---|---|
|3D mode| |

Dual-mode Toggling Inference of Dual-mode Multi-view LDM

Refined Mesh

Mesh Extraction

10s 40s

- Figure 1: The Framework of Dual3D. Firstly, we fine-tune a pre-trained 2D LDM into a dual-mode multi-view LDM. Subsequently, we employ a dual-mode toggling inference strategy to choose different denoising modes during inference to balance the inference speed and 3D consistency. Finally, the mesh extracted from the neural surface is further optimized via our efficient texture refinement process, enhancing the photo-realism and details of the asset.

lows the dual-mode multi-view LDM to be tuned from a pre-trained 2D LDM. The insight of this architecture is to replace the single-view latent denoising with synchronized and interconnected denoising of the multi-view latents and tri-plane representations. We further discover that the texture of the 3D assets generated by only denoising exhibits a noticeable style difference from real-world textures, primarily due to the style bias in the synthesized multi-view datasets. To address this, we propose an efficient texture refinement process that quickly optimizes the texture map of the extracted mesh from the 3D neural surface. The overall framework of our method is shown in Figure 1 and the entire generation process of our framework requires only 50 seconds per asset on a single NVIDIA RTX 3090 GPU. The efficient and high-quality generation ability makes our framework well-suited for compositional generation, and all generated 3D assets can seamlessly integrate into traditional rendering engines, as shown in Figure 2.

### 2. Related Works

Text-to-3D Generation. DreamField (Jain et al., 2022) pioneers the open-world text-to-3D generation domain by integrating vision language model CLIP (Radford et al., 2021) with NeRF-based (Mildenhall et al., 2021) 3D rendering. DreamFusion (Poole et al., 2022) and SJC (Wang et al., 2023a) introduce 2D image diffusion models to optimize the 3D representation with SDS loss, improving the visual quality of text-to-3D generation. With advancements in 3D representation (Liu et al., 2020; Chen et al., 2022) and rendering techniques (Wang et al., 2021; Yariv et al., 2021; Tang et al., 2023a), there has been a growing focus on extending these techniques to the text-to-3D generation domain. Notably, recent works (Lin et al., 2023; Tang et al., 2023a; Liu et al., 2023c) have specifically targeted this area. Furthermore, some methods propose alternative score distillation losses (Wang et al., 2023b; Katzir et al., 2023; Zou et al., 2023; Bahmani et al., 2023; Wu et al., 2024) to

better leverage 2D diffusion models for stabilizing text-to3D generation. There are also methods (Shi et al., 2023; Liu et al., 2023c; Long et al., 2023; Liu et al., 2023b) that propose to introduce additional 3D priors to 2D diffusion models to improve the stability and 3D consistency of the generation. However, SDS-based methods often require a long optimization time for each asset, making it challenging to apply to large-scale generation.

On the other hand, some approaches accomplish this task by directly training 3D generative models. Early works (Chan

- et al., 2021; Schwarz et al., 2020; Gu et al., 2021; Or-El
- et al., 2022; Chan et al., 2022; Deng et al., 2022; Xiang et al., 2023) combine neural rendering and GANs (Goodfellow et al., 2020; Karras et al., 2018; 2019; 2020) techniques, yet their applicability is limited to specific categories. High-capacity diffusion model methods (Nichol

- et al., 2022; Jun & Nichol, 2023; Tang et al., 2023b; Shue
- et al., 2023; Gupta et al., 2023; Xu et al., 2023), nevertheless, either rely on 3D datasets or necessitate the reconstruction of multi-view datasets into 3D representations, resulting in high pre-processing cost. These methods often encounter geometric artifacts and unrealistic textures due to the inherent discrepancy between 3D datasets and real-world images.

Our framework, Dual3D, aims to generate high-quality and realistic 3D assets for category-agnostic texts while reducing the generation time to less than 1 minute.

### 3. Preliminary

Latent diffusion models (LDMs) (Rombach et al., 2022; Saharia et al., 2022) consist of two key components: an autoencoder (Kingma & Welling, 2014) and a latent denoising network. The autoencoder establishes a bi-directional mapping from the space of the original data to a low-resolution latent space:

###### z = E(x),x = D(z), (1)

“a wooden sliding door”

“a blue door”

“a pair of slippers”

“a women's fashion bag”

“a blue and green schoolbag”

“a vertical air conditioner”

“a floor-to-ceiling window with beautiful scenery”

“a chinese beautiful painting”

“a bookshelve and several books”

“a cowboy hat”

“a brown paper bag”

|[Figure 3]|[Figure 4]<br><br>|[Figure 5]|[Figure 6]|[Figure 7]<br><br>|[Figure 8]|
|---|---|---|---|---|---|
|[Figure 9]| |[Figure 10]|[Figure 11]| |[Figure 12]<br><br>|

“A teddy bear plush toy”

“a bag of crisps”

“a wooden sliding door”

“a WIFI router”

“a LCD TV”

“a wooden TV cabinet with drawers”

“a colorful lamp”

“a thermos cup”

“a wooden nightstant”

“a plush floor mat”

“a white cashmere pillow”

“a wooden window”

“a black and grey camera with a lens”

- Figure 2: Two compositional 3D scenes rendered by Blender, where all visible assets are generated by our method with only texts as inputs. The text prompts for some assets are indicated by arrows. Please refer to our project page for the tour videos.

where E and D are the encoder and decoder, respectively. The latent denoising network ϵ˜θ is designed to denoise noisy latent given a specific timestep t and condition y. Its training objective for ϵ-prediction is defined as:

L = Ex,ϵ∼N(0,1),t ∥ϵ − ϵ˜θ(zt,y,t)∥22 , (2)

where√1 − α¯thetϵ andnoisyα¯t islatenta monotonicallyis obtaineddecreasingby zt = √noiseα¯tEsched-(x) + ule. During inference, a random noise is sampled as zT ∼ N(0,1). By continuously denoising the random noise zT with condition y, we can derive a fully denoised latent z˜. Then, the denoised latent z˜ is fed into the latent decoder D to generate the high-resolution image x˜ = E(˜z).

Multi-view diffusion models aim to model the distribution of multi-view images X with 3D consistency, where each image is captured by a different camera within the same scene. Its objective for x0-prediction can be written as:

L = EX,ϵ∼N(0,1),t ∥X − X˜θ(Xt,c,y,t))∥22 , (3)

where c represents the camera parameters for the different views. Early works (Shi et al., 2023; Liu et al., 2023c) in this field are based on 2D LDMs. They fine-tune the 2D LDMs by integrating cross-view connections between the multiview images into the original single-view 2D LDMs, using multi-view data rendered from 3D datasets. These methods lack strict 3D consistency since there is no actual 3D representation during multi-view denoising. They also require the use of SDS-based optimization in conjunction with the fine-tuned multi-view LDMs to generate 3D assets. A more advanced approach, DMV3D (Xu et al., 2023), employs a 3D reconstruction model to generate noise-free 3D representations and predict multi-view images from noisy multi-view inputs by a 3D-consistent rendering process. This allows for 3D generation tasks to be achieved without any per-asset optimization during inference. However, the efficiency of the rendering-based multi-view diffusion models is significantly reduced due to the necessity of performing rigorous rendering on full-resolution images.

### 4. Method

In this section, we outline the algorithm for tuning the pretrained 2D LDM into the dual-mode multi-view LDM in Section 4.1. We then introduce the dual-mode toggling inference strategy in Section 4.2. Finally, we introduce the efficient texture refinement process in Section 4.3.

##### 4.1. Dual-mode Multi-view Latent Diffusion Model

The key insight of our approach is to utilize the strong prior of 2D LDM, (i.e., the compression and detail recovery abilities of the auto-encoder, and the generative ability of the latent denoising network) to jointly train a dual-mode multi-view LDM, where the 3D mode can directly generate a clean 3D representation from noisy multi-view latents. We take multi-view images X ∼ RN×3×H×W and feed them into the frozen image encoder E of the 2D LDM to obtain the latents Z ∼ RN×c×h×w. During training, we add noise to the multi-view latents Z to derive the noisy latents Zt = √α¯tZ + √1 − α¯tϵ, where ϵ is noise sampled from the Gaussian distribution N(0,1) and t is a random timestep. One of our inspirations is that one of the popular 3D representations, tri-plane (Chan et al., 2022), has a very similar formulation with 2D image features. As such, we treat the tri-plane as three special latents. Since we do not have the ground truth of the tri-plane latents, we initialize three learnable latents V ∼ R3×c×h×w to serve as the noisy latents of the tri-planes, as illustrated in Figure 3.

Denoising. Similar to the noisy multi-view latents Zt, the noisy tri-plane latents are also fed into the latent denoising network Zθ in parallel, note that we change the ϵ-prediction of the original 2D LDM into x0-prediction for convenience. The denoised tri-plane latents and multi-view latents can be obtained by Z˜2d,V˜ = Zθ({Zt,V},c,y,t). The camera condition c is injected into the network by concentrating the rays r = (o × d,d), parameterized by Plucker coordinates, into the noisy latents following LFN (Sitzmann et al., 2021). Here, o and d represent the origin and direction of the downsampled pixel rays aligned with the latent resolution, respec-

! # ❄!#

! ❄

###### Dual3D: Efficient and Consistent Text-to-3D Generation with Dual-mode Multi-view Latent Diffusion

c y t 2D mode 3D mode

Cameras

Text Time

Zt

###### Z˜2d Z˜3d

Input-view Cameras

|| |[Figure 13]<br><br>|
|---|---|
| | |
<br><br>Occupancy grids<br><br>Uniformly sampling SDF<br><br>0<br><br>depth Ray marching & upsampling<br><br>Xn<br><br>i=1<br><br>Ti↵ici<br><br>Volume Rendering|
|---|

| | |
|---|---|
| | |

c

2D Latent Encoder

L2d

inference

2D Latent Denoising Network

[Figure 14]

Render

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 15]

[Figure 16]

[Figure 17]

| | |
|---|---|
| | |

2D Latent Decoder

Tiny Transformer

Cross-view Self-a ention

#### L3d

training

c0 V˜0 D(V˜0)

[Figure 18]

Frozen Fine-tuned

[Figure 19]

[Figure 20]

[Figure 21]

Tri-planes

[Figure 22]

Learnable Tri-plane Latents

Neural New Surface

Novel-view Cameras

Rendered Images

V V˜

[Figure 23]

- Figure 3: The architecture of dual-mode multi-view LDM. The noisy multi-view latents and three learnable tri-plane latents

are fed into the 2D latent denoising network Zθ in parallel, where all self-attention blocks are replaced by cross-view self-attention blocks. A tiny transformer is used to enhance the connections between the multi-view features and the tri-plane features. The denoised tri-plane latents are decoded into higher resolution with the 2D latent decoder D and rendered to images with volume rendering of the tri-plane surface. Two main objectives, L2d and L3d, are used to optimize the model.

tively. To make connections between the tri-plane features and multi-view features, we follow MVDream (Shi et al., 2023) to simply replace self-attention blocks in the original

- 2D latent denoising network with cross-view self-attention blocks. A straightforward multi-view latent diffusion objective is used to supervise the 2D-mode denoised multi-view latents Z˜2d, which can be defined as:

L2d = EX,ϵ,c,y,t ∥Z − Z˜2d∥22 . (4)

We add a tiny transformer to further enhance the connections between the tri-plane latents and multi-view latents and get the final denoised tri-plane latents V˜′. The final denoised tri-plane latents V˜′ are then fed into the latent decoder D to get a high-resolution tri-planes D(V˜′), note that we reinitialize the last convolutional layer of the latent decoder D for a higher number of tri-plane channels.

Rendering. Instead of using NeRF-based rendering, we follow TextMesh (Tsalicoglou et al., 2023) to use NeuS (Wang et al., 2021) as our base rendering method. This allows for better geometric quality, and we propose some improvements for more efficient and accurate rendering. First, we uniformly sample a certain resolution of dense grids within the bounding box and obtain the SDF values through bi-linear sampling of the tri-planes D(V˜′) and a tiny 2layer MLP, following EG3D (Chan et al., 2022). Then, we determine whether there could be a surface within the grid based on the SDF values of the grid center, marking the positive grids as occupied. Next, for each ray, we obtain the initial sampled points within the occupancy grid via ray marching. These initial sampled points are refined through an upsampling strategy similar to NeuS, making the final sampled points close to the zero-set of the SDF. We also concatenate some uniformly sampled

points within the bounding box to explore unoccupied areas, the final color of the ray is calculated by volume rendering ni=1 Tiαici, where Ti = nj=1(1 − αj), n is the number of sampled points along the ray (sorted by depth), and αi and ci are the transparency and color of point pi, respectively. The transparency αi is calculated by αi = max(Φ

s(f(pi))−Φs(f(pi−1))

Φs(f(pi)) ,0), where f(·) is the SDF value of a specific point and Φs is the Sigmoid function with a learnable inverse standard deviation. During training, we use novel-view cameras c′ instead of input-view cameras c to supervise 3D-mode denoising in image space by:

L3d = EX′,X,ϵ∼N(0,1),t ℓ(X′,R(D(V˜′),c′)) , (5)

where R is the rendering process, D(V˜′) is the tri-planes, ℓ(·,·) is an image reconstruction loss penalizing the difference between images, and X′ is the ground truth of the novel-view images. We use a combination of MSE loss and LPIPS (Zhang et al., 2018) loss with equal weights for the reconstruction loss ℓ. During inference, the images are rendered with the input-view cameras c and encoded by the latent encoder E to obtain the 3D-mode denoised latents, represented as Z˜3d = E(R(D(V˜′),c)). To regularize the surface to be physically valid and reduce floating geometry, we follow StyleSDF (Or-El et al., 2022) to employ the eikonal loss Leik = (∥∇f(p)∥2 − 1)2 and the minimal surface loss Lsurf = exp(−64|f(p)|) as constraints on the normal vectors and SDF values of sampled points p.

Total loss. Our total loss is the weighted sum of the above losses, which is:

L = λ2dL2d + λ3dL3d + λeikLeik + λsurfLsurf, (6) where λ2d, λ3d, λeik, and λsurf are the weights of different

losses. We empirically set them to be 1, 1, 0.1, and 0.01, respectively, for all experiments.

##### 4.2. Dual-mode Toggling Inference

Our model is capable of performing both 2D-mode and 3Dmode denoising for multi-view latent diffusion. Since the inputs and outputs of the two modes are perfectly aligned, we can toggle between them during inference. While 3D-mode denoising ensures strict 3D consistency, it is significantly slower than 2D-mode denoising as it requires rendering full-resolution images, similar to DMV3D. However, using too few 3D-mode denoising steps can lead to 3D inconsistency in the multi-view latents, resulting in artifacts in the final 3D assets. We also find that the 3D-mode denoising is more difficult to deal with unseen texts due to the limited multi-view dataset, while the 2D-mode denoising can better handle unseen texts and concept combinations, as it is closer to the original 2D LDM. Therefore, we propose dual-mode toggling inference to balance the inference speed, generation quality, and 3D consistency. Specifically, we toggle between 2D-mode and 3D-mode denoising at a certain frequency throughout the entire inference process:

Z ˜3d, if (t − 1) mod m = 0 Z˜2d, otherwise

Z˜ =

(7)

where t is the current timestep and m ∈ N+ is the frequency to use 3D mode. The denoised multi-view latents Z˜ are then used to denoise Z˜t into less noisy latents Z˜t−1 using the x0prediction formulation of diffusion. This design also ensures that the final denoising step is 3D-mode. We experimentally find that only 1/10 of the denoising steps need to use 3D mode (i.e., m = 10 when using 100 steps with DDIM (Song et al., 2021)), which can essentially ensure 3D consistency and reduce the inference time to 10 seconds.

##### 4.3. Efficient Texture Refinement

The final 3D-mode denoising step of our method often yields

- 3D assets with good geometric shapes, but due to the limitations of the synthesized multi-view dataset, the textures are not always realistic. Therefore, we propose an efficient texture refinement process to further enhance the texture, while keeping the time cost reasonable. Specifically, we first extract the original neural surface into a mesh model, fix its geometry, and convert its texture into a learnable texture map. Then, we use differentiable mesh rendering (Laine et al., 2020) to render the mesh into an image I with a random viewpoint. The image is encoded into the latent space, perturbed with an annealing strength of noise at timestep t, and denoised using a multi-step denoising process F(·,y,t) with the original 2D latent diffusion model. Finally, We optimize the texture map by constructing a reconstruction loss between the rendered image and the refined image decoded

from the denoised latent, which is:

∥I − D(F(√α¯tE(I) + √1 − α¯tϵ),y,t)∥22. (8)

This process significantly enhances the texture quality in a short time, thanks to the good surface quality of the neural surface generated from denoising and the efficiency of differentiable mesh rendering. This process is inspired by the second stage optimization of DreamGaussian (Tang et al., 2023a). Still, our method is more robust and concise as the geometry and texture generated by our denoising stage provide better initialization, avoiding the complex color back-projection process in DreamGaussian.

### 5. Experiments

##### 5.1. Settings

Implementation details. We train our dual-mode multiview LDM using the Adam optimizer (Kingma & Ba, 2015) with a constant learning rate of 5e−5 and (β1,β2) = (0.9,0.95). The resolutions of the images and latents are 256 and 32, respectively. We use 4 input views following the practice of DMV3D (Xu et al., 2023). During training, we render 4 × 128 × 128 image patches for supervision to save GPU memory. The batch size is set to 128. Training takes about 4 days with 32 NVIDIA Tesla A100 GPUs for 100K iterations. We use Stable Diffusion v2.1 as our initial model. We use 1000 steps during training with a cosine schedule and reduce it to 100 steps with DDIM (Song et al., 2021) during inference. The classifier-free guidance scale is 7.5 for 2D-mode denoising. For rendering, we use 24 uniformly sampled and 24 upsampled points for each ray with the implementation of batch-wise neural surface rendering in StreetSurf (Guo et al., 2023). We directly use rendered multi-view images provided by Zero123 (Liu et al., 2023a) with Objaverse (Deitke et al., 2023) dataset to train our model. The text prompts are generated by Cap3D (Luo et al., 2023). Like MVDream, our model also supports regularization with 2D text-to-image datasets such as LAION (Schuhmann et al., 2022) to enhance generalization. For texture refinement, we use a learning rate of 1e−1 and a total of 100 iterations. The pre-defined timestep is annealing from 0.20T to 0.05T, where T is the total denoising steps.

Baselines. We compare our method with state-of-the-art category-agnostic text-to-3D generation approaches, including Point-E (Nichol et al., 2022), Shap-E (Jun & Nichol, 2023), DreamGaussian (Tang et al., 2023a), MVDream (Shi et al., 2023), and VolumeDiffusion (Tang et al., 2023b). Point-E and Shap-E are 3D diffusion models based on point clouds and implicit functions, respectively. DreamGaussian and MVDream are optimization-based methods that utilize the prior of 2D and multi-view LDMs, respectively. VolumeDiffusion is a recent work that employs a two-stage framework for text-to-3D generation, combining a 3D vol-

[Figure 24]

[Figure 25]

Dual3D: Efficient and Consistent Text-to-3D Generation with Dual-mode Multi-view Latent Diffusion

Table 1: Quantitative comparison.

inference-only methods

CLIP Similarity ↑

CLIP R-Precision ↑

Aesthetic Score ↑

Generation Time ↓

Method

Point-E 66.2 47.2 4.39 21s Shap-E 70.4 60.0 4.40 8s

- VolumeDiffusion-I 59.6 18.6 4.03 12s

- Ours-I 72.0 72.3 5.22 10s DreamGaussian 65.1 31.9 5.09 3m MVDream 69.8 56.7 5.27 45m

VolumeDiffusion-II 63.0 32.4 4.17 8m

- Ours-II 73.1 74.3 5.50 50s

optimization-based methods

ume denoising stage and an SDS-based refinement stage. For a fair comparison, all generated assets are converted into meshes and rendered with Blender for evaluation to avoid differences in quality caused by different rendering processes. We categorize the compared methods into inferenceonly and optimization-based according to whether there is a per-asset optimization process during generation. Therefore, we classify the denoising stage of VolumeDiffusion and our method as inference-only methods (i.e., VolumeDiffusion-I and Ours-I), and the refining stage as optimization-based methods (i.e., VolumeDiffusion-II and Ours-II), with independent evaluation. We would also report the results of DMV3D once its code is released.

Figure 4: User study

methods except for MVDream. After texture refinement, our method surpasses all baseline methods, demonstrating that our method can generate aesthetically pleasing 3D assets.

Generation time. Considering application deployment and large-scale generation, we also evaluate the generation time of different methods. For a fair comparison, we use a single NVIDIA RTX 3090 GPU to evaluate the generation time of different methods. The results are reported in Table 1. Point-E, Shap-E, VolumeDiffusion-I, and Ours-I are all inference-only methods, so the generation speed is fast. DreamGaussian and MVDream are SDS-based methods, so the generation speed is slow. Ours-II adopts an optimizationbased refinement stage, but due to better initialization and the efficiency of mesh rendering, its speed is far superior to other optimization-based methods.

##### 5.2. Quantitative results.

CLIP score. We compute the CLIP Similarity and RPrecision (Park et al., 2021) to compare the alignment between assets generated by different methods and the texts. For each method, we use 36 texts from Shap-E to generate 36 assets and render 24 fixed views for each asset. The CLIP Similarity is calculated by computing the average cosine distance between the CLIP (Radford et al., 2021) embeddings of the rendered view and the text. The R-Precision is calculated by computing the Top-1 Precision of the zero-shot classification of the 36 groups. The results are shown in Table 1. Our method demonstrates superior semantic alignment capability compared to baseline models using only the denoising stage. After texture refinement, our method achieves better performance on both metrics, demonstrating the texture-enhancing ability of the texture refinement.

User study. We also conduct a user study to compare the subjective quality of 3D assets generated by different methods. We collect 36 votes from 24 users in 2 tracks (a total of 1728 votes) and count the percentage of votes obtained by each method, and the results are shown in Figure 4. Different stages of our method all win the first place in the corresponding track, demonstrating that our method can generate 3D assets that align with user preferences.

##### 5.3. Qualitative results.

We also qualitatively compare our method with baseline methods as shown in Figure 5. The inference-only and optimization-based methods are listed on the left and right of the dotted line, respectively. For inference-only methods, Point-E, Shap-E, and VolumeDiffusion all produce discontinuous geometry, floating shapes, and poor texture. Thanks to the image-space supervision and the effective utilization of the 2D prior model, our method can generate a complete shape and good texture that aligns with the text using only the denoising stage. The good geometry and texture of the denoising stage also provide a better initialization for later refinement. MVDream, as the method closest to our method

Aesthetic score. We also evaluate the aesthetic scores of the 3D assets generated by different methods. We adopt the open-source LAION Aesthetics Predictor1, which trains a single linear layer based on CLIP embeddings to predict the aesthetic quality of images from 0 to 10. We use it to score the rendered images of the generated objects from the previous experiment and report the average. The results are also shown in Table 1. The results show that our method, using only the denoising stage, surpasses all other baseline

1https://github.com/LAION-AI/aesthetic-predictor

Point-E

Shap-E VolumeDiffusion-I Ours-I DreamGaussian MVDream VolumeDiffusion-II Ours-II

11 seconds 8 seconds 12 seconds 10 seconds 3 minutes 45 minutes 8 minutes 50 seconds

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

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

“a brown paper bag”

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

“a large metal bell on a red wooden stand”

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

“a stone water well with a wooden shed”

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

“a wooden chest with golden trim”

inference-only methods optimazation-based methods

Figure 5: Qualitative comparison.

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

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

“a house” “a hamburger”

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

“a traditional chinese chair”

“a SUV car” “a supercar” “a truck car” “a car made of sushi”

“a sofa chair” “an esports chair” “a bar chair”

Figure 6: Diverse and fine-grained generation results.

in quantitative evaluation, also produces objects with realistic textures, but there are some holes in the geometry. Although our method does not introduce explicit lighting or shadow during rendering, the strong prior of the 2D diffusion model helps to generate realistic lighting and shadow effects after texture refinement, making the generated 3D assets more photo-realistic. Also, we find that the assets generated by our model are more consistent with the given text, especially in materials and colors, which aligns with our leading performance in the CLIP Score.

##### 5.4. Diverse and Fine-grained Generation

In this experiment, we demonstrate some beneficial properties of our model. On one hand, our model can generate diverse 3D assets given the same text. On the other hand, our model can generalize to fine-grained abstract semantic changes in the text. We select some texts and denoise different latent noises with different random seeds as shown

- in Figure 6. We first select some general sentences and

generate four different objects. We find that our model can generate 3D assets that satisfy the given texts with varying contents, shapes, textures, and colors. We also select some base sentences and replace words in them. Our model can translate these fine-grained semantic modifications into changes in the 3D geometry and shape details of the assets. This experiment demonstrates that our model has a strong ability for diverse and fine-grained generation.

##### 5.5. Ablation Study

w/o network prior. In this ablation, we use a randomly initialized latent denoising network Zθ instead of resuming from the weights of the pre-trained 2D LDM. Other settings remain the same as in the original model. We present the quantitative metrics in Table 2 to compare our full model with other methods. We find that the model without prior experiences a significant drop in all metrics, especially for CLIP R-Precision. This demonstrates the effectiveness of training the dual-mode multi-view LDM with prior.

###### Table 2: Ablation study.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Generation Time ↓ Ours-I 72.0 72.3 5.22 10s w/o network prior 61.7 21.2 4.44 10s w/o tiny transformer 70.6 66.1 5.18 9s w/o dual-mode inference 66.0 44.6 4.81 1m30s

CLIP Similarity ↑

CLIP R-Precision ↑

Aesthetic Score ↑

Method

“a plush toy of a corgi nurse”

“a ghost eating a hamburger”

“a bicycle” “a pineapple”

Figure 8: Some failure cases.

100/100 20/100 10/100 5/100 1/100

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

##### 5.6. Limitations

Although our framework has a high success rate of generating in-distribution 3D assets and a certain generalization ability thanks to the prior of 2D LDM and the joint training of multi-view data and real-world 2D data, there are still some failure cases, as shown in Figure 8. On the one hand, since most multi-view data are single-object scenes, it is difficult for our model to handle innovative text prompts with complex concepts or multi-object combinations (e.g., the “eating” action between the ghost and the hamburger is misunderstood). This issue may be addressed by introducing more real-world multi-view data (Yu et al., 2023; Reizenstein et al., 2021; Ling et al., 2023) and parameter-efficient fine-tuning techniques (Hu et al., 2022; Liu et al., 2022; He

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

“a chair that looks like a tree”

“a fast car”

1min20s 20s 10s 7s 5s

Figure 7: Ablation study on dual-mode toggling inference.

w/o tiny transformer. In this ablation, we remove the tiny transformer after the latent denoising network. We find that this does not affect the overall convergence but has a significant adverse effect on the quality of the final 3D neural surface. Because the original 2D latent denoising network uses a convolutional UNet (Ronneberger et al., 2015), too little cross-view connection may make it difficult for the model to extract reasonable 3D information from the multi-view features. As shown in Table 2, the CLIP R-Precision of this ablated model has a drop. Also, since we use a dual-mode toggling inference, the overall generation time of this ablated model only has a slightly decreased. Based on the above complaints, we think that introducing additional tiny transformers is necessary for our framework.

- et al., 2022). On the other hand, although our texture refinement is efficient, using mesh rendering limits the further improvement of geometry, leading to the failure of generating very complex or thin shapes (e.g., The fine steel wire in the bicycle wheels and the small thorns on pineapple). This issue may be addressed by introducing more efficient 3D representations, such as 3D Gaussian Splatting (Kerbl
- et al., 2023), to further improve the rendering quality and efficiency. These failure cases prompt us to further improve our framework design in our future works.

### 6. Conclusion

w/o dual-mode inference. In this ablation, we first remove the dual-mode toggling inference and use 3D mode for all denoising steps. The metrics in Table 2 show that the model performance has a significant decrease. We then try different frequencies for 3D-mode denoising to observe the impact on efficiency and visual quality with a typical example shown

In this paper, we propose an efficient and consistent textto-3D generation framework capable of generating realistic 3D assets in one minute. We first introduce a dual-mode multi-view LDM that can be trained from a pre-trained 2D LDM, where the 3D mode can generate a clean neural surface from the noisy multi-view latents. The proposed dual-mode toggling inference strategy further allows the dual-mode multi-view LDM to significantly improve the inference speed while ensuring consistency and generation quality. The neural surface generated by the dual-mode multi-view LDM can be further extracted into a mesh and refined by the proposed efficient texture refinement to enhance the realism and details. We demonstrate the effectiveness of our method with extensive experiments and show the effect of each component. We believe our work makes essential contributions to the text-to-3D generation community, especially in discovering the potential of dual-mode multi-view diffusion for fast and high-quality 3D generation.

- in Figure 7. The percentage of 3D-mode denoising and the inference time are on the top and the bottom, respectively. We find that both too many and too few 3D-mode denoising steps lead to poor generation quality. Too many 3D-mode steps lead to semantic misalignment, which we suspect is because the 3D-mode denoising is more difficult to utilize the original 2D prior, resulting in a poor understanding of complex semantics. Too few 3D-mode steps lead to messy and floating geometry because the multi-view latents generated by 2D mode do not come from consistent 3D rendering. Hence, we choose 10/100 denoising steps to use 3D mode to basically ensure 3D consistency and generation quality with a reasonable time cost.

### Acknowledgements

This work was supported by National Science and Technology Major Project (No. 2022ZD0118202), the National Science Fund for Distinguished Young Scholars (No.62025603), the National Natural Science Foundation of China (No. U21B2037, No. U22B2051, No. 62176222, No. 62176223, No. 62176226, No. 62072386, No. 62072387, No. 62072389, No. 62002305 and No. 62272401), and the Natural Science Foundation of Fujian Province of China (No.2021J01002, No.2022J06001).

### References

Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Bahmani, S., Skorokhodov, I., Rong, V., Wetzstein, G., Guibas, L., Wonka, P., Tulyakov, S., Park, J. J., Tagliasacchi, A., and Lindell, D. B. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. arXiv preprint arXiv:2311.17984, 2023.

Chan, E. R., Monteiro, M., Kellnhofer, P., Wu, J., and Wetzstein, G. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5799–5809, 2021.

Chan, E. R., Lin, C. Z., Chan, M. A., Nagano, K., Pan, B., De Mello, S., Gallo, O., Guibas, L. J., Tremblay, J., Khamis, S., et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16123–16133, 2022.

Chen, A., Xu, Z., Geiger, A., Yu, J., and Su, H. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision, pp. 333–350. Springer, 2022.

Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi,

- A., and Farhadi, A. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13142– 13153, 2023.

Goyal, P., Doll´ar, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

Gu, J., Liu, L., Wang, P., and Theobalt, C. Stylenerf: A style-based 3d-aware generator for high-resolution image synthesis. arXiv preprint arXiv:2110.08985, 2021.

Guo, J., Deng, N., Li, X., Bai, Y., Shi, B., Wang, C., Ding, C., Wang, D., and Li, Y. Streetsurf: Extending multiview implicit surface reconstruction to street views. arXiv preprint arXiv:2306.04988, 2023.

Gupta, A., Xiong, W., Nie, Y., Jones, I., and O˘guz, B. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023.

He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., and Neubig, G. Towards a unified view of parameter-efficient transfer learning. In International Conference on Learning Representations, 2022. URL https://openreview.

net/forum?id=0RDcd5Axok. Hendrycks, D. and Gimpel, K. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., and Tan, H. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023.

Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=nZeVKeeFYf9.

Jain, A., Mildenhall, B., Barron, J. T., Abbeel, P., and Poole, B. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 867–876,

- 2022.

Jun, H. and Nichol, A. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463,

- 2023.

Deng, Y., Yang, J., Xiang, J., and Tong, X. Gram: Generative radiance manifolds for 3d-aware image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10673–10683, 2022.

Karras, T., Aila, T., Laine, S., and Lehtinen, J. Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations, 2018. URL https://openreview.

net/forum?id=Hk99zCeAb.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial networks. COMMUNICATIONS OF THE ACM, 63(11), 2020.

Karras, T., Laine, S., and Aila, T. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401–4410, 2019.

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., and Aila, T. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8110– 8119, 2020.

Karras, T., Aittala, M., Laine, S., H¨ark¨onen, E., Hellsten, J., Lehtinen, J., and Aila, T. Alias-free generative adversarial networks. In Proc. NeurIPS, 2021.

Katzir, O., Patashnik, O., Cohen-Or, D., and Lischinski, D. Noise-free score distillation. arXiv preprint arXiv:2310.17590, 2023.

Kerbl, B., Kopanas, G., Leimk¨uhler, T., and Drettakis, G. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023.

Kingma, D. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), San Diega, CA, USA, 2015.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, 2014.

Laine, S., Hellsten, J., Karras, T., Seol, Y., Lehtinen, J., and Aila, T. Modular primitives for high-performance differentiable rendering. ACM Transactions on Graphics, 39(6), 2020.

Lin, C.-H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.-Y., and Lin, T.-Y. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 300–309, 2023.

Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al. Dl3dv-10k: A largescale scene dataset for deep learning-based 3d vision. arXiv preprint arXiv:2312.16256, 2023.

Liu, L., Gu, J., Zaw Lin, K., Chua, T.-S., and Theobalt, C. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651–15663, 2020.

Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., and Vondrick, C. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9298–9309, 2023a.

Liu, X., Ji, K., Fu, Y., Tam, W., Du, Z., Yang, Z., and Tang, J. P-tuning: Prompt tuning can be comparable to finetuning across scales and tasks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 61–68, Dublin, Ireland, May 2022. URL https://aclanthology.

org/2022.acl-short.8.

- Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., and Wang, W. Syncdreamer: Generating multiviewconsistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023b.
- Liu, Z., Li, Y., Lin, Y., Yu, X., Peng, S., Cao, Y.-P., Qi, X., Huang, X., Liang, D., and Ouyang, W. Unidream: Unifying diffusion priors for relightable text-to-3d generation, 2023c.

Long, X., Guo, Y.-C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.-H., Habermann, M., Theobalt, C., et al. Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008, 2023.

Luo, T., Rockwell, C., Lee, H., and Johnson, J. Scalable 3d captioning with pretrained models. arXiv preprint arXiv:2306.07279, 2023.

Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

Nichol, A., Jun, H., Dhariwal, P., Mishkin, P., and Chen, M. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022.

Or-El, R., Luo, X., Shan, M., Shechtman, E., Park, J. J., and Kemelmacher-Shlizerman, I. Stylesdf: High-resolution 3d-consistent image and geometry generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13503–13513, 2022.

Park, D. H., Azadi, S., Liu, X., Darrell, T., and Rohrbach, A. Benchmark for compositional text-to-image synthesis. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Reizenstein, J., Shapovalov, R., Henzler, P., Sbordone, L., Labatut, P., and Novotny, D. Common objects in 3d:

Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10901– 10911, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234–241. Springer, 2015.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan,

- B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35: 36479–36494, 2022.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35: 25278–25294, 2022.

Schwarz, K., Liao, Y., Niemeyer, M., and Geiger, A. Graf: Generative radiance fields for 3d-aware image synthesis. Advances in Neural Information Processing Systems, 33: 20154–20166, 2020.

Shi, Y., Wang, P., Ye, J., Mai, L., Li, K., and Yang, X. Mvdream: Multi-view diffusion for 3d generation. arXiv:2308.16512, 2023.

Shue, J. R., Chan, E. R., Po, R., Ankner, Z., Wu, J., and Wetzstein, G. 3d neural field generation using triplane diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20875– 20886, 2023.

Sitzmann, V., Rezchikov, S., Freeman, B., Tenenbaum, J., and Durand, F. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34:19313– 19325, 2021.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021.

Tang, J., Ren, J., Zhou, H., Liu, Z., and Zeng, G. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023a.

Tang, Z., Gu, S., Wang, C., Zhang, T., Bao, J., Chen, D., and Guo, B. Volumediffusion: Flexible text-to-3d generation with efficient volumetric encoder, 2023b.

Tsalicoglou, C., Manhardt, F., Tonioni, A., Niemeyer, M., and Tombari, F. Textmesh: Generation of realistic 3d meshes from text prompts. arXiv preprint arXiv:2304.12439, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Wang, H., Du, X., Li, J., Yeh, R. A., and Shakhnarovich, G. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12619–12629, 2023a.

Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., and Wang, W. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In 35th Conference on Neural Information Processing Systems, pp. 27171–27183. Curran Assoicates, Inc., 2021.

Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu, J. Prolificdreamer: High-fidelity and diverse textto-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023b.

Wu, J., Gao, X., Liu, X., Shen, Z., Zhao, C., Feng, H., Liu, J., and Ding, E. Hd-fusion: Detailed text-to-3d generation leveraging multiple noise estimation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 3202–3211, 2024.

Wu, Y. and He, K. Group normalization. In Proceedings of the European conference on computer vision (ECCV), pp. 3–19, 2018.

Xiang, J., Yang, J., Deng, Y., and Tong, X. Gram-hd: 3d-consistent image generation at high resolution with generative radiance manifolds. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2195–2205, 2023.

Xu, Y., Tan, H., Luan, F., Bi, S., Wang, P., Li, J., Shi, Z., Sunkavalli, K., Wetzstein, G., Xu, Z., et al. Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. arXiv preprint arXiv:2311.09217, 2023.

Yariv, L., Gu, J., Kasten, Y., and Lipman, Y. Volume rendering of neural implicit surfaces. Advances in Neural Information Processing Systems, 34:4805–4815, 2021.

Yu, X., Xu, M., Zhang, Y., Liu, H., Ye, C., Wu, Y., Yan, Z., Zhu, C., Xiong, Z., Liang, T., et al. Mvimgnet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9150–9161, 2023.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.

Zou, Z.-X., Cheng, W., Cao, Y.-P., Huang, S.-S., Shan, Y., and Zhang, S.-H. Sparse3d: Distilling multiviewconsistent diffusion for object reconstruction from sparse views. arXiv preprint arXiv:2308.14078, 2023.

### A. More Implementation Details.

- A.1. Detailed Architecture of Dual-mode Multi-view LDM

Here, we provide a more detailed explanation of our modifications in the 2D LDM, as shown in Table 3. The latent encoder is completely frozen, so it does not require any modifications. For the latent denoising network, we additionally input the camera condition in the form of Plucker coordinates, therefore increasing the channel dimension by 6 in the input. The first dimension is made up of the number of views and three triplanes. We also increase the output of the latent denoising network and the input of the latent decoder by 508 dimensions to match the difference in the amount of information between the tri-plane latents and the image latents. The output of the latent decoder is increased from the original 3 dimensions to 64 dimensions, therefore the shape of the tri-planes is 3 × 64 × 256 × 256.

Table 3: Detailed Networks.

Module Architecture Input Output Latent Encoder CNN 4 × 3 × 256 × 256 4 × 4 × 32 × 32 Latent Denoising Network UNet (4 + 3) × (4 + 6) × 32 × 32 (4 + 3) × (4 + 508) × 32 × 32 Tiny Transformer Transformer (4 + 3) × (4 + 508) × (32 × 32) 3 × (4 + 508) × (32 × 32) Latent Decoder CNN 3 × (4 + 508) × 32 × 32 3 × 64 × 256 × 256

- A.2. Tiny Transformer

Our tiny transformer contains 16 self-attention layers (Vaswani et al., 2017). After each self-attention layer, there is a feed-forward MLP, which consists of two linear layers and a GeLU activation (Hendrycks & Gimpel, 2016). Layer Normalization (Ba et al., 2016) is used before each self-attention and feed-forward layer. We follow DIT (Peebles & Xie, 2023) to introduce zero-initialized layer scaling (Goyal et al., 2017) for each block to stabilize the training. For attention layer, the numbers of the channels and heads are set to 512 and 8, respectively. The parameter number is about 50M.

- A.3. Adding Normalization Layers into Latent Decoder

Our early experiment finds that although the loss of training the VAE decoder as the triplane upsampler is consistently decreased, it suddenly crashes during middle training. By checking the data distribution of each layer output, we find that the main reason is that the output values of the decoder layers are gradually increasing, which leads to the numerical explosion. We first try to add normalization layers (e.g., GroupNorm (Wu & He, 2018)) to each upsampling layer but find that it affects the original data distribution suddenly after initialization, which leads to slower convergence. Finally, we adopt the exponential moving average normalization layer proposed by StyleGAN3 (Karras et al., 2021) to each upsampling layer of the triplane decoder. Specifically, we record the moving average norm σ = E(x2) of each upsampling layer output x and divide x by √σ for stabilizing. σ is initialized to be 1, so it does not affect the distribution of each layer in the early stage, and it can be updated iteratively to stabilize the values during training.

### B. Evaluation Details.

##### B.1. CLIP Similarity and R-precision

We directly use the text-to-3D evaluation from Cap3D (Luo et al., 2023) to calculate CLIP Similarity and R-precision with ViT-B/32 model. For a fair comparison, we use 24 fixed views (i.e., azimuth ∈ {0◦,45◦,90◦,135◦,180◦,225◦,270◦,315◦} and elevation ∈ {−30◦,0◦,30◦}) to render 3D assets from different methods. The CLIP Similarity is calculated by E[cos(fx,fy) × 100 × 2.5], where fx and fy are the CLIP embeddings of the rendered image x and text y, respectively. For R-precision, we calculate the similarity of each rendered image and all 36 texts. Then, we report the proportion of all rendered images that the similarity with the ground truth text is the Top-1.

##### B.2. User study

The user study is conducted in the form of an anonymous questionnaire. We invite 24 users to participate, and each user must complete 36 votes for two tracks. We provide an example for both tracks, as shown in Figure 9. Each example requires users to choose the most satisfactory one from four different methods according to the text description.

[Figure 176]

[Figure 177]

Figure 9: Examples of user study.

### C. More Results.

##### C.1. Comparison with DMV3D

Our framework has the following advantages compared to DMV3D: 1. We have greatly reduced the required training cost and time. DMV3D requires 128 A100 cards to train for one week, while we only need 32 cards to train for 4 days, which is about 1/8 GPU days of DMV3D. 2. The proposed dual-mode toggling inference reduces the inference time of denoising. DMV3D takes approximately 30 seconds to 1 minute with an NVIDIA Tesla A100 GPU, while we only need 10 seconds with an NVIDIA RTX 3090 GPU for denoising. 3. Our generated mesh after texture refinement is more realistic, and our model has more potential for generalization of unseen texts due to the effective utilization of the 2D LDM. We supply some qualitative comparisons with DMV3D in Figure 10. Since the code of it has not been released, we directly use the examples on the project page. We find that the textures generated by DMV3D are more abundant while our model tends to generate more details (e.g., the rearview mirrors of “a rusty old car” and the tires of “a race car”).

DMV3D Ours-I Ours-II

30 ~ 60 seconds 10 seconds 50 seconds Tesla A100 RTX 3090 RTX 3090

[Figure 178]

[Figure 179]

[Figure 180]

“a rusty old car”

[Figure 181]

[Figure 182]

[Figure 183]

“a donut with pink icing”

[Figure 184]

[Figure 185]

[Figure 186]

“a race car”

Figure 10: Qualitative comparison with DMV3D.

##### C.2. Additional Qualitative Comparison with Baselines

We provide some additional qualitative comparisons with the baseline models in Figure 11. The text prompts are from Shap-E and VolumeDiffusion.

##### C.3. Additional Visualization Results

We also provide some additional visualization results of our framework in Figure 12. For more interactive results, please kindly refer to our project page.

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

DreamGaussianVolumeDiffusion-IIShap-EVolumeDiffusion-IMVDreamOurs-IOurs-IIPoint-E

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

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

“a blue and white vase with a lid”

“a wooden church with a cross on top”

“a chair that looks like a zebra”

“a penguin” “a tra ic cone”

Figure 11: More qualitative comparison.

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

“a brown wooden chair with metal legs”

“a large banyan tree with many extended vines”

“a wooden nightstand” “a basket of fruits”

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

“a bag of crisps” “a dog” “a o ice building” “a wooden cutout chair”

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

“a four-legged urban co ee shop leather chair”

“a small round wooden table”

“a castle made of stone”

“a white boot”

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

“a white and black police SUV car”

“a tall brown lighthouse with a light on top”

“a teddy bear plush toy”

“a co ee maker”

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

“a wooden TV cabinet with cabinets and drawers”

“a stone fountain with a spout”

“a wooden desk with a drawer”

“a women's fashion bag”

Figure 12: More results.

