## Streetscapes: Large-scale Consistent Street View Generation Using Autoregressive Video Diffusion

Boyang Deng

Stanford University USA bydeng@stanford.edu

Richard Tucker

Google Research USA richardt@google.com

Zhengqi Li

Google Research USA zhengqili@google.com

[Figure 1]

Leonidas Guibas

Noah Snavely*

Gordon Wetzstein*

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Stanford University Google Research USA guibas@cs.stanford.edu

Google Research USA snavely@google.com

Stanford University USA gordon.wetzstein@stanford.edu

# arXiv:2407.13759v2[cs.CV]25Jul2024

Map

View A View B View C View D

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

(b) Long-range Consistent Street View Generation

Streetscapes Generation

Height Map

[Figure 23]

“New York City, …” “Barcelona, …” “…, heavy snow” “…, at sunset”

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

(a) Input (c) Geographic Style Control (d) Weather and Time-of-Day Control

Figure 1: Generating Streetscapes. We design a system that generates Streetscapes, i.e., consistent street views along long-range paths through large-scale synthesised urban scenes. Given a) a street map and its corresponding height map, our system can generate: b) high-quality street views spanning a long camera path with layout controlled by the map, with the ability to generate c) different geographic styles as well as d) different weather conditions or times of day, both controlled by text prompts. We recommend the reader to visit our project page at boyangdeng.com/streetscapes to view our results as videos.

#### ABSTRACT

Compared to recent models for video generation or 3D view synthesis, our method can scale to much longer-range camera trajectories, spanning several city blocks, while maintaining visual quality and consistency. To achieve this goal, we build on recent work on video diffusion, used within an autoregressive framework that can easily scale to long sequences. In particular, we introduce a new temporal imputation method that prevents our autoregressive approach from drifting from the distribution of realistic city imagery. We train our Streetscapes system on a compelling source of data—posed imagery from Google Street View, along with contextual map data—which allows users to generate city views conditioned on any desired city layout, with controllable camera poses.

We present a method for generating Streetscapes—long sequences of views through an on-the-fly synthesized city-scale scene. Our generation is conditioned by language input (e.g., city name, weather), as well as an underlying map/layout hosting the desired trajectory.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0525-0/24/07...$15.00 https://doi.org/10.1145/3641519.3657513

* Equal Contributions

#### CCS CONCEPTS

• Computing methodologies → Computer vision; Neural networks.

#### KEYWORDS

Image Synthesis, Video Synthesis, Generative Models, Diffusion Models, Scene Generation, Neural Rendering.

ACM Reference Format:

Boyang Deng, Richard Tucker, Zhengqi Li, Leonidas Guibas, Noah Snavely*, and Gordon Wetzstein*. 2024. Streetscapes: Large-scale Consistent Street View Generation Using Autoregressive Video Diffusion. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers ’24 (SIGGRAPH Conference Papers ’24), July 27-August 1, 2024, Denver, CO, USA. ACM, New York, NY, USA, 16 pages. https://doi.org/10. 1145/3641519.3657513

#### 1 INTRODUCTION

Methods for generating visual contents like images, videos, and 3D models have captured the world’s imagination [Po et al. 2023]. While the results are astounding, these methods often struggle when asked to produce large-scale outputs. For instance, current text-to-video methods are limited to generating short videos of scenes with bounded extent. Text-to-3D methods can generate individual objects [Poole et al. 2023], but not whole cities. In this work, we focus on the domain of urban scenes, and generate longrange, 3D-consistent views along paths through synthesized city streets, an output we call Streetscapes.

There are several challenges associated with generating outputs at the scale of Streetscapes:

- • What input to use? Text is a popular form of conditioning. But at the scale of a city, text is a blunt tool that does not allow fine-grained control over the output. Instead, we condition our Streetscapes on an overhead scene layout, consisting of a street map and corresponding height map, inspired by recent layout-conditioned generative models [Bahmani et al. 2023]. The unique combination of our input data allows us to render semantic labels, a height map, and a depth map from any camera pose, transforming the map–space information to screen space. Our method also takes in an optional text input describing the desired style, like weather and time of day.
- • What output to produce? We want our outputs to be view consistent, as if they were rendered from a 3D model. But it is difficult to scale generative 3D methods to entire cities. Instead, we build on text-to-video models and produce a long sequence of images as output. However, current video generation methods cannot ensure long-range quality and 3D consistency. To address these issues, we ensure view consistency by conditioning each output image on the scene layout, projected to an image-space G-buffer [Deering et al. 1988], and scale to long-range sequences via a new autoregressive video synthesis technique that minimizes drift away from the space of natural street scenes, while ensuring frame to frame continuity.
- • What training data to use? Training our method requires large amounts of real street view sequences, along with corresponding scene layouts. Fortunately, such data has been

captured at scale by mapping services like Google Street View [Anguelov et al. 2010]. One of our key insights is to leverage such geographic datasets for generative tasks. However, there are also challenges associated with such data, like the fact that the scene layouts from mapping sites are relatively coarse-grained and not pixel-perfect when projected into ground views. We show how to leverage mapping corpora like Google Street View for training, while achieving robustness to coarse-grained, noisy data.

In short, our work presents a new method for synthesizing longrange, consistent Streetscapes by training on an overlooked source of data—large-scale collections of street view imagery—and conditioning on a new kind of input, namely scene layouts spanning multiple city blocks. To achieve high-quality Streetscapes over long scales, we propose (i) a layout-conditioned generation approach, (ii) a motion module that enables consistent two-frame generation, and (iii) an autoregressive temporal imputation technique that modifies the pre-trained two-frame motion module at inference time to enable consistent long-range video generation. With our results, we demonstrate that our system autoregressively generates Streetscapes covering long-range camera trajectories with consistently high quality. Our system also enables numerous creative scene generation applications, thanks to our flexible control over scene layout, camera poses, and scene conditions.

#### 2 RELATED WORK

The generation of walkthroughs in large environments is a classic problem in computer graphics that motivated a lot of the early work in efficient visibility culling [Cohen-Or et al. 2003; Teller and Séquin 1991]. That line of works, however, assumes that a model of the environment is given, while our key focus is in the synthesis of views along a path through an environment, where the environment itself is created on the fly using modern generative neural methods, such as diffusion. A comprehensive review of the theory and practice of diffusion models for visual computing, as applied to 3D and video generation, can be found in the recent survey paper by Po et al. [2023]. We briefly cover the most relevant works here.

3D Object Generation. Diffusion models have emerged as state-ofthe-art generative models for 2D images [Ho et al. 2020; Rombach et al. 2022; Song et al. 2020b]. These models have also been extended to the 3D domain. For example, some strategies [Gupta et al. 2023a; Jun and Nichol 2023; Nichol et al. 2022; Ntavelis et al. 2023; Shue et al. 2023] use direct 3D supervision. The quality and diversity of their results, however, is far from that achieved for 2D image generation. DreamFusion [Poole et al. 2023] and related works enable text-to-

###### 3D generation by combining a neural scene representation with the image priors of pre-trained 2D diffusion models [Chen et al. 2023a; Gupta et al. 2023a; Lin et al. 2023a; Shi et al. 2023; Wang et al. 2022]. These methods successfully generate 3D objects of moderately high quality. Image-conditioned 3D generation has also become a popular approach [Chan et al. 2023; Gu et al. 2023; Hong et al. 2023; Liu et al. 2023; Watson et al. 2022], with the latest methods in this category [Hong et al. 2023] resolving flicker and other viewinconsistent artifacts that plagued earlier approaches. Based on the idea of training 3D diffusion models directly on image data [Chen

et al. 2023b; Karnewar et al. 2023; Shen et al. 2023b], state-of-the-art results for 3D generation have recently been achieved by techniques that solve the 3D generation problem using 2D diffusion models with 3D-aware denoisers [Anciukevičius et al. 2023; Xu et al. 2024].

While different variants of these 3D diffusion approaches have become very successful, they are all limited to generating individual objects. This is in part because available training datasets, such as Objaverse [Deitke et al. 2023], only contain individual objects; score distillation sampling (SDS)–based methods, such as DreamFusion, suffer from mode collapse that also effectively limits these methods to generating individual objects.

3D Scene Generation. Large-scale and perpetual scene generation has been a topic of interest and much progress has been made [Chai et al. 2023; Hao et al. 2021; Li et al. 2022; Lin et al. 2023b; Liu et al. 2021; Yang et al. 2023b]. Among these, Infinicity [Lin et al. 2023b] is closest to our work because they also aim for large-scale street view generation. Yet, these approaches build on generative adversarial networks (GANs) that are challenging to train and that have shown limited quality and diversity in practice. Moreover, the outputs of these GANs are not easily editable, for example by text prompts.

Emerging diffusion models have been used for 3D scene generation of rooms [Höllein et al. 2023] and outdoor scenes [Cai et al. 2023; Fridman et al. 2023; Shen et al. 2023a]. These methods iteratively apply depth estimation, depth-based image warping, and diffusion-based inpainting to generate a scene. Several very recent methods extend this approach to 3D Gaussian–based representations [Chung et al. 2023; Ouyang et al. 2023]. Monocular depth estimation, however, is not always reliable, especially for street views, which makes it difficult to apply these methods to our setting. Moreover, a desired camera trajectory cannot easily be defined because the scene geometry is generated and not known a priori.

Another class of methods uses 3D bounding boxes to guide an object-centric diffusion model in generating a compositional scene [Po and Wetzstein 2023; Schult et al. 2023]. While successful, these methods are not directly applicable to consistent 3D generation at a city scale and, like other SDS-based approaches, suffer from mode collapse and limited quality. CC3D also generates compositional room-sized scenes, albeit conditioned on a 2D image of a room’s layout [Bahmani et al. 2023]. CC3D employs a GAN architecture and inherits its limitations, namely limited editability and diversity.

A set of recent, unpublished, and concurrent works discuss 3D street view generation [Gao et al. 2023; Swerdlow et al. 2023; Yang et al. 2023a]. These methods take a bird’s eye view—a layout map with each object in the scene—and ground truth 3D bounding boxes of all objects as input to generate street view images. Among these MagicDrive [Gao et al. 2023] achieves the highest-quality results and, similar to our approach, builds on a latent 2D diffusion model. Our work operates in a different setting: we do not have access to individual objects, such as cars, or their 3D bounding boxes. Our maps only contain the layout of streets and buildings, along with their height. Moreover, their approach operates in a frameby-frame manner and promotes consistency among frames using cross-frame attention; they discuss extensions to fine-tuning their model on a text-to-video model [Wu et al. 2023a]. In contrast, our autoregressive approach is not limited to the small numbers of frames that can be generated by existing text-to-video models.

Another very recent and concurrent approach to 3D scene generation is WonderJourney [Yu et al. 2023], where several keyframes are generated first and then interpolated using diffusion models.

Unlike these approaches, the Streetscape setting is unique in offering ground truth proxy geometry of the buildings via the given map and corresponding height information. Simple as it may seem, this small amount of extra information enables unprecedented levels of control over the generation process, including controllable camera trajectories, 3D multi-view consistency, and coarse grounding of the generated scene layout for consistent long-range generation. Interestingly, our unique setting allows us to formulate the 3D generation process in a way that is more akin to video generation than to 3D generation.

VideoGeneration. Videodiffusionmodels [Blattmannet al.2023a,b;

Guo et al. 2023b; Ho et al. 2022; Li et al. 2023; Singer et al. 2022; Villegas et al. 2022] have recently emerged as data-driven approaches to training foundation models that can generate short video clips of impressive quality and consistency. ControlNets can be used to condition the video generation process on depth maps or human poses [Guo et al. 2023a; Hu et al. 2023; Xu et al. 2023].

The primary limitation of most existing video generation models is being restricted to generating very short clips containing limited frames. Inspired by diffusion imputation techniques, we propose a new autoregressive mechanism that enables us to generate long sequences with high temporal and multi-view consistency.

#### 3 STREETSCAPES

To generate long-range, high-quality Streetscapes, we introduce a method that builds on video diffusion models, but adds two crucial ingredients: (1) a method for conditioning on coarse scene layout information, and (2) an autoregressive method for generating video frames that avoids drift—that is, that keeps the imagery on the manifold of natural images—even over long sequences. We train our system on a novel data source, namely street view imagery and corresponding map information (Sec. 3.4).

Our system first trains a diffusion model that jointly generates two frames by iteratively denoising two random noise images. This model also takes as input conditioning information rendered from the given layout for two camera views (Sec. 3.2). Our goal is to generate many consistent frames in our output, however, not just two. For this purpose, we modify the pre-trained two-frame generation model to allow it to operate in an autoregressive temporal imputation mode without the need for retraining the model. In this mode, the two random noise images used as input to the model are replaced by noised versions of the frame generated for the current camera view and of the current frame warped into the next camera view, respectively (Sec. 3.3). With the generated frames, we can also run an optional 3D reconstruction (Sec. B) to get a 3D scene model. We begin this section with a brief review of the basic concepts of diffusion models.

#### 3.1 Background on Diffusion Models

The goal of an image generation algorithm is to learn to sample images from an underlying data distribution 𝑝data(𝑥). Diffusion models achieve this goal by sequentially denoising samples of random noise 𝑥𝑇 ∼ N (0, I) into samples from the data distribution

𝑥0 ∼ 𝑝data(𝑥). In this context, the (forward) diffusion process, 𝑝 (𝑥𝑡 |𝑥0) = N 𝑥𝑡;√𝛼𝑡𝑥0, 1 − 𝛼𝑡I , (1)

defines the distribution of 𝑥𝑡 as a noised version of 𝑥0 at diffusion step 𝑡 = 0, . . .,𝑇; 𝛼𝑡 is a predefined noise scheduling term.

The above formulation is approximately inverted by the reverse diffusion process by iteratively computing

𝑥𝑡−1 = DDIM (𝑥𝑡,𝜖𝜃 (𝑥𝑡,𝑡) ,𝑡) , (2) where 𝜖 is a learned neural network with parameters 𝜃; 𝜖 predicts the noise we have added on 𝑥𝑡, which is used by a DDIM denoiser [Song et al. 2020a] to compute 𝑥𝑡−1.

#### 3.2 Layout-conditioned Scene Generation

Our objective is to create high-quality Streetscapes that adhere to a predefined scene layout and optional text prompts. Hence, we design a Streetscapes generation system that builds on robust, scalable, and controllable diffusion model architectures. Specifically, we begin by designing a two-view generation module that leverages a pretrained text-to-image diffusion model, incorporates a motion module inspired by AnimateDiff [Guo et al. 2023b] to enable twoframe generation, and injects control by integrating a G-buffer– conditioned ControlNet [Zhang et al. 2023].

Text-to-Image Foundation Model. We build our framework on top of a pretrained Latent Diffusion Model [Rombach et al. 2022]. Given a noise feature map 𝑧𝑇 along with a text prompt, this model generates a feature map 𝑧0 by iteratively denoising 𝑧𝑇 using Eq. 2. The generated feature map is then decoded into a high-quality image 𝑥0 using decoder D,𝑥0 = D(𝑧0). This decoder, along with a corresponding encoder E,𝑥 ≈ D (E(𝑥)), enables the diffusion model to operate in a low-resolution feature space, rather than in image space, for computational efficiency. Although not specialized for street-view generation, this foundation model is trained on generic text-to-image generation tasks. We do not further modify its parameters, and so we naturally inherit text control from this foundation model, even for scenarios that are scarce or absent in our data, e.g., streets full of snow.

Two-Frame Generation via Motion Module. A critical ingredient in our system is simultaneous generation of two consecutive frames in a Streetscape. We draw inspiration from recent advances in video diffusion models. In particular, we follow AnimateDiff [Guo et al. 2023b] to insert a motion module into our foundation model. This motion module enables information exchange across frames at each diffusion step𝑡 and effectively extends our noise prediction network

to 𝜖𝜃 𝑧𝑡(𝑖),𝑧𝑡(𝑖+1),𝑡 and jointly estimates 𝑧𝑡(𝑖−)1,𝑧𝑡(𝑖−+11) . Here, 𝑖 is the index of the first generated frame and𝑖+1 the following. Similar to the single-frame case of Eq. 2, this process starts from random

Gaussian noise for both 𝑧𝑇(𝑖) and 𝑧𝑇(𝑖+1). In the following, we refer to this standard video diffusion approach as parallel denoising,

because two frames are generated in parallel. The parameters of this motion module are trained with our custom dataset (see Sec. 3.4).

Note that we choose to output two frames as the minimal and most flexible design, yet one can easily generalize the formulation to more than two frames. That said, current video diffusion models are limited to generating a few tens of frames, so simply adding

G(i) G-buffers G(i+1) ControlNet

Generated

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Streetscapes Map

Motion Module

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Decoder

[Figure 40]

[Figure 41]

C(i+1) C(i)

𝒟

[Figure 42]

[Figure 43]

[Figure 44]

Semantics

Semantics

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Render

[Figure 52]

- (Eq. (1))

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

DDIM

- (Eq. (2))

[Figure 59]

[Figure 60]

Disparity

Disparity

Encoder

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

ℰ

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Add Noise

Height Map

Noisy Latents

Height

Height

Figure 2: Layout-conditioned Scene Generation. Using the input scene layout (overhead street map and height map), we render two geometry buffers (G-buffers), 𝐺(𝑖) and 𝐺(𝑖+1), that contain semantic labels encoded in an RGB image as well as image-space disparity maps and height maps, for camera poses 𝐶(𝑖) and 𝐶(𝑖+1). These G-buffers condition a motion-aware latent diffusion model that generates a pair of images. Orange and purple boxes illustrate spatial and temporal layers, respectively.

more than two frames to the parallel denoising process does not scale to the long Streetscape sequences we aim to generate.

Layout Conditioning and Camera Control. While the aforementioned motion module can successfully generate two coherent video frames, it does not provide us with direct control over the content. For this purpose, we additionally adopt a ControlNet-based conditioning mechanism [Zhang et al. 2023]. Specifically, our input street maps and corresponding height maps are rendered from the desired two camera poses 𝐶(𝑖) and 𝐶(𝑖+1) into G-buffers [Deering et al. 1988], 𝐺(𝑖) and 𝐺(𝑖+1). The G-buffers transform our conditioning information from map space into screen space. This conditioning mechanism adds direct control over the camera pose and scene layout and is closely related to depth-conditioned ControlNets used with concurrent video diffusion models [Hu et al. 2023; Wang et al. 2023; Xu et al. 2023]. We illustrate our system in Fig. 2. Note that our system doesn’t take camera pose parameters as input.

#### 3.3 Autoregressive Video Diffusion

The parallel denoising method discussed above can generate the first two frames of a Streetscape. For frames beyond those, we would like to ensure that they are consistent with all previously generated frames. G-buffer conditioning alone cannot guarantee consistency because this approach only controls the layout of the scenes, but not their appearance. Thus, stacking pairs of images generated with parallel denoising over time does not achieve the desired consistency, because consistency is only achieved within each pair. Alternatively, one can also append the generated frame to the ControlNet input, adding RGB control for appearance. However, we empirically find it suffering from severe quality drift.

Inspired by imputation techniques that were used in time-series analysis [Tashiro et al. 2021], image inpainting [Song et al. 2020b], monoculardepthestimation[Saxena et al.2023],andtext-conditioned generic video generation [Ho et al. 2022], we propose a temporal

C(1)

𝒩(0,1) 𝒩(0,1)

[Figure 71]

[Figure 72]

zT(i+1)

C(0)

C(2) C(3)

Trajectory G-buffers

[Figure 73]

[Figure 74]

[Figure 75]

Camera

[Figure 76]

. . .

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

zt(0) zt(1) zt(i) zt(i+1)

[Figure 96]

[Figure 97]

[Figure 98]

AddNoise,(Eq.(1))

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

DDIM DDIM

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

G(0) G(1)

G(i) G(i+1)

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

G(0) G(1) G(2) G(3)

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

[Figure 142]

zt(0)−1 zt(1)−1

zt(−1i+1)

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Temporal Imputation

Temporal Imputation

Parallel Denoising

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

z1(1) z0(1) z0(i)

z(0̂ i+1)

z0(i+1)

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

𝒟 𝒟

ℰ

###### ℰ

𝒟

Streetscapes

Generated

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

. . .

[Figure 179]

[Figure 180]

x(0) x(1) x(2) x(3)

x(0) x(1)

x(i)

x(i+1)

Warped

(a) Autoregressive Video Diffusion

(b) Parallel Denoising (c) Temporal Imputation

- Figure 3: Autoregressive Video Diffusion. The Streetscapes system generates a sequence of consistent frames along a desired camera trajectory. Consistency is achieved by generating the first 2 frames jointly using parallel denoising, then generating each subsequent frame via temporal imputation, guided by the previous frame in an autoregressive manner. Both procedures use the same model, but with different reverse diffusion formulations.

imputation method to enable consistent long-range autoregressive video generation for generating Streetscapes.

Temporal imputation is a surprisingly simple, yet effective idea. Importantly, it directly leverages the pre-trained two-frame model without the need to retrain or refine it. Given the last generated frame 𝑥(𝑖), we aim to generate the next frame 𝑥(𝑖+1) and continue the same procedure over and over again. This is possible using a twist on the reverse diffusion process in Eq. (2) with pretrained noise predictor 𝜖𝜃. At each reverse diffusion step, instead of reusing the estimation of 𝑥𝑡(𝑖) from the previous diffusion step 𝑡 + 1, we directly sample 𝑥𝑡(𝑖) from the known generated frame 𝑥(𝑖),

𝑧𝑡(𝑖) ∼ N 𝑧𝑡(𝑖);√𝛼𝑡E 𝑥(𝑖) , 1 − 𝛼𝑡I , (3)

Here, Eq. 3 essentially adds noise to the latent corresponding to the known frame 𝑥(𝑖). Because we apply Eq. (3) to all reverse diffusion

steps, the ultimate denoised𝑧0(𝑖) results in the same image 𝑥(𝑖) after decoding. Note that this is only applied to the first frame of the

two-frame generation. The second frame 𝑧𝑡(𝑖+1) still comes from the estimation of step 𝑡 + 1. In our autoregressive setting, this change

ensures that the first frame of the next pair of generated images is the same as the last frame of the previous pair. After generating the new pair, however, we discard the first frame of the generated pair and only append the second, 𝑥(𝑖+1), to our sequence.

One key insight of our temporal imputation method is that, through imputation, the autoregressive generation takes multiple noised copies of the condition frame 𝑥(𝑖) throughout the reverse diffusion process. This effectively makes the conditioning more robust to any generated slightly out-of-domain frames, because the uncanny details will be submerged by the noise we add.

While the two-frame diffusion model promotes consistency between pairs of generated frames in aforementioned imputation, we empirically observe that the consistency between 𝑥(𝑖) and 𝑥(𝑖+1) is not always satisfying. For better consistency, we devise two

additional techniques. First, inspired by image inpainting techniques [Lugmayr et al. 2022], we find that after denoising 𝑧𝑡(𝑖+1) to 𝑧𝑡(𝑖−+11), adding noise to 𝑧𝑡(𝑖−+11) and rerun the imputation can further homogenise 𝑧𝑡(𝑖−+11) with the noised condition 𝑧𝑡(𝑖). We only proceed to the next step after a few rounds of such back-and-forth. We refer to this as Resample. Second, we notice that the appearance of 𝑥(𝑖+1) is still influenced by the choice of the noisy latent 𝑧𝑇(𝑖+1). Hence, we propose to warp the known RGB image 𝑥(𝑖) from its camera pose 𝐶(𝑖) to the pose of the next view 𝐶(𝑖+1) using the disparity map available in G-buffer𝐺(𝑖). Even though the warped image has holes and other artifacts, its noised version provides an excellent choice for 𝑧𝑇(𝑖+1). We denote this technique as WarpInit. Together, they complete our temporal imputation method (see Fig. 3). To the best of our knowledge, our system is the first to adopt imputation techniques, specifically, temporal imputation, to autoregressive video generation, using our two-frame diffusion model.

#### 3.4 Streetscape Data

Because our Streetscapes task is unique, we also require a novel dataset that allows us to learn control over both scene layout and camera pose. A key insight behind our work is to identify geographic data captures from mapping sites like Google Street View [Anguelov et al. 2010] as a fitting source of training data for our generative task. Google Street View offers a large corpus of high-quality, diverse, and worldwide street view imagery, geolocated and situated in a geographic context. In our work, we show a proof-of-concept use of this data source by collecting 1.5M images covering 33 km2 from four major cities, Paris, London, Barcelona, and New York City, across four countries. These data also come with structural information like scene layout and image metadata including the camera pose and capture timestamp. This enables our system to associate layout and camera pose control signal with street view generation. On the other hand, along with its scale, coverage, and multi-modality, come data challenges; in particular, the aerial data is coarse and imperfectly aligned when projected to ground level views. Thus, our system must be sufficiently robust to these sources of noise. We show in Sec. 4 that our system generates controlled high-quality Streetscapes, a task otherwise impossible without this precious data.

#### 4 RESULTS AND APPLICATIONS

We validate our Streetscapes system on several generative tasks. We first describe experimental details in Sec. 4.1. In Sec. 4.2, we present results on a task we call long-range consistent street view generation. This task involves sythesizing all images from scratch, given a desired scene layout and camera trajectory. In Sec. 4.3 we evaluate on a second task, perpetual view generation [Liu et al. 2021], where unlike the first task, we start from a known, real input view, and autoregressively generate a long sequence of new views. In Sec. 4.4, we discuss applications of our Streetscapes system, including controlling an output Streetscape via text description and mix-and-match map and geographic styles, as well as an application of interpolating real street view images in Sec. C.

#### 4.1 Experimental Details

Model and Training. We base our system on a pretrained textto-image latent diffusion model [Rombach et al. 2022]. To enable two-frame generation, we insert motion modules as proposed in AnimateDiff [Guo et al. 2023b], but change their 16-frame formulation to a 2-frame one. We further add a ControlNet [Zhang et al. 2023] to enable layout and camera pose control. In contrast to the standard single-image ControlNet, we form a two-frame ControlNet by also cloning the motion modules. Hence, our ControlNet contains both spatial and temporal modules. We use the diffusion noise prediction objective from DDPM [Ho et al. 2020] to train our model with learning rate of 1𝑒−5 and batch size 256 for 1.5M iterations. Note we keep the base diffusion model U-Net intact in training, so as to preserve the text control from the base model. Our model is trained using image data from Google Street View.1 For two-frame training, we group pairs of 512 × 512 images captured within 80s and 5 meters.

Inference. We use both parallel sampling and temporal imputation in our generations, as defined in Sec. 3. For parallel sampling, we follow the deterministic sampling process in DDIM [Song et al.

- 2020a]. We use50denoise steps with the classifier-free guidance [Ho and Salimans 2021] scale set to 7.5. For the temporal imputation stage defined in Sec. 3.3, we use 10 Resample steps and WarpInit. Note for any inference in our experiments, we use maps from regions not seen during training.

#### 4.2 Long-range Consistent Street View Generation

We first evaluate our system on the task of generating long-range consistent street views of a large-scale urban scene, where we generate all views in the output from scratch. In this task, the model is given a street map, its corresponding height map, and a sequence of camera poses defining a camera trajectory, and generates consistent street views corresponding to the trajectory. We compare our method to InfiniCity [Lin et al. 2023b], a state-of-the-art consistent street view generation system. Unlike Streetscapes, InfiniCity requires aligned CAD models as training data, and it only works for cities where CAD models are readily available data (just London, in their case). For a fair comparison, we also only use maps from London when evaluating our system. We show example results in Fig. 4. We find that Streetscapes produced by our system, either the 2-frame version or the 4-frame version, are consistently more realistic compared to those of InfiniCity. Our results have higher-fidelity details such as windows, pavement, and vegetation. In addition, our results simulate richer lighting effects, including natural shadows (missing in InfiniCity’s results). To quantitatively evaluate the generated results, we use FID and KID metrics. These metrics measure the distance between distributions defined by the generated results and by ground truth imagery, and are commonly used to evaluate generative models. We summarize the results in Tab. 1. While our system generates Streetscapes autoregressively, we observe in Tab. 1 that our quality is consistently better than InfiniCity’s across all autoregressive steps by a fair margin. All InfiniCity results are obtained from the authors.

1Google Maps Street View images used with permission from Google.

- Table 1: Large-scale Consistent Street View Generation. We compare Streetscapes to InfiniCity on image quality metrics FID and KID. Because we generate Streetscapes autoregressively, we evaluate Streetscapes quality at different autoregressive steps. We find that our quality is significantly better than InfiniCity’s, across all steps.

Metric InfiniCity

Streetscapes @ Steps all 1–16 16–32 32–64

FID↓ 108.47 17.79 16.12 21.70 29.93 KID↓ 0.084 0.023 0.017 0.022 0.025

- Table 2: Perpetual Street View Generation. We compare our Streetscapes with results from state-of-the-art autoregressive

view generation methods ( best , 2nd best ). We compute a holistic image similarity measure (LPIPS) to assess consistency in the near-range (1–16 steps). For image quality, we measure the FID and KID at different autoregressive step ranges. These results indicate that Streetscapes have similar near-range quality and consistency to Zero123-A, while maintaining image quality over significantly longer ranges (32+ steps) than all alternative methods.

LPIPS ↓ FID↓/KID↓ @ Steps

Metric

1–16 1–16 16–32 32–64 InfNat0-mono 0.600 16.72/0.014 48.97/0.042 71.10/0.056 InfNat0-proxy 0.614 26.80/0.025 51.48/0.052 59.00/0.048 DiffDreamer 0.589 45.96/0.050 164.23/0.208 207.37/0.242 Zero123-A 0.497 7.09/0.005 34.64/0.035 64.77/0.062 Streetscapes 0.519 11.69/0.012 25.59/0.028 35.47/0.034

In addition to our comparison to InfiniCity, in Fig. 5, we also show two example Streetscapes from our system, simulating longrange, synthesized street cruises. Each scene spans a map of ∼ 1 km2. We generate 100 frames for each scene with flexible camera motion, to mimic “looking around” the synthesized city. We find that our system robustly generates high-quality street views across long camera trajectories with various camera motions.

#### 4.3 Perpetual Street View Generation

Core to our system is our novel autoregressive video diffusion method. To further validate this method, we also compare it with alternative autoregressive view generation methods. We choose the perpetual view generation [Liu et al. 2021] task, tailored to street scenes, as the benchmark. For this task, we take a real street view image as a starting view, and then generate new views autoregressively along a desired camera trajectory, with the goal of generating views that are consistent with the starting image. We additional provide our system with the corresponding scene layout for the starting view. We compare to several state-of-the-art methods:

• InfiniteNature-Zero (InfNat0) [Li et al. 2022] is a GAN-based method. It uses scene geometry estimated via monocular depth prediction to forward-warp views during generation.

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Streetscapes SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA

InfiniCity:

- Generated Street A

Streetscapes (Ours):

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Generated Street A (camera moving forwards)

- Generated Street B Generated Street B (camera moving backwards)

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

- Generated Street C Generated Street C (camera moving backwards)

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

- Figure 4: Long-range Consistent Street View Generation. We compare Streetscapes to a state-of-the-art street view generation approach, InfiniCity [Lin et al. 2023b], on the task of generating consistent views on long paths through large-scale street scenes. In this task, we let Streetscapes generate consistent views autoregressively following various camera tracks. We find that Streetscapes consistently generates higher-fidelity street views that are noticeably more realistic than InfiniCity’s results.

We test two variantsofInfNat0,one using the original monocular depth formulation (InfNat0-mono) and one using proxy geometry derived from the input height map (InfNat0-proxy).

- • DiffDreamer [Cai et al. 2023] is a diffusion-based alternative to InfNat0-proxy that replaces the GAN approach with a more powerful diffusion model.
- • Zero123[Liuetal.2023] is a diffusion-basedgenerativemethod that does not use explicit 3D geometry. It takes in an input view and a relative camera pose directly as condition embedding parameters. Note that the original Zero123 always conditions on the given first view because it focuses on object generation, which has a canonical, bounded camera space. We modify Zero123 to generate unbounded long-range street views autoregressively by conditioning on the last generated view. We refer this variant as Zero123-A.

We obtain results for InfNat0-mono and InfNat0-proxy using the code from its authors to train the model on our data. For diffusionbased approaches (DiffDreamer and Zero123-A), for fair comparison, we train these models using the same pretrained diffusion base models as on our data.

We show examples of generated views in Fig. 6. We find that at the first autoregressive step, most methods generate high-quality views. At step-20, the quality of DiffDreamer’s results start to decrease. At step-40, the realism of samples from InfNat0-mono and DiffDreamer have deteriorated, and the results from Zero123-A and InfNat0-proxy are also quite poor. In contrast, our Streetscapes maintain consistent, realistic quality across all steps.

For quantitative evaluation, we follow Infinite Nature [Liu et al.

- 2021] and evaluate with two kinds of metrics: First, we compute short-range accuracy for the first few images in the sequence, where we have known ground truth street view imagery, and where

the view synthesis problem has not yet become fully generative due to overlap with the seed view. For this metric, we compute LPIPS [Zhang et al. 2018] to measure the holistic image similarity over the first 16 views. We prefer the holistic measure LPIPS over local per-pixel errors because even for early views, the task still has generative aspects due to occlusions, super-resolution from zooming-in, and scene dynamics such as moving cars. Second, we evaluate long-range image quality for more distant views (where the task is almost entirely generative), we compute FID and KID between generated street views and real street views. We show quantitative comparisons in Tab. 2. From the comparison, we conclude that our generated Streetscapes have short-range accuracy comparable to a state-of-the-art method, Zero123-A. Meanwhile, unlike prior methods that suffer from serious degradation in longrange quality (e.g., after 32 steps), our method scales well such long sequences.

#### 4.4 Creative Applications of Streetscapes

Beyond standard view generation tasks, our Streetscapes system enables numerous creative applications. We showcase two example use cases of our system: text-controlled Streetscapes generation and geographic style mix-and-match.

Text Controlled Streetscapes. Thanks to the use of a pre-trained text-to-image model as the foundation of our system, we can control our Streetscapes generation using text prompt. While our dataset mainly contains street view images in the daytime without extreme weather conditions, due to the real-world street view capture constraints, our generation can hallucinate various conditions on the street, including both weather and time-of-day. For instance, we can ask for “Paris, Before Sunset” or “A Rainy Day in New York”. In

Fig. 7, we display a few example text-controlled Streetscapes with specific weather and time-of-day. We find that our system can effectively generate Streetscapes reflecting the given text description.

What Makes Paris Look Like New York? In a seminal prior work, researchers investigated “What Makes Paris Look Like Paris?” [Doersch et al. 2012]. Our Streetscapes system imagines what a street looks like from a map. Therefore, in this work, we make an alternative quest to our system — What makes Paris look like New York? Here, “Paris” denotes the scene layout — we take a street map of Paris blocks; “New York” indicates the geographic style of the generation — we hope our generated Streetscapes look distinctly like New York City. In Fig. 8, we show our generated Streetscapes for this task. We observe that our system can mix-and-match the layout and the geographic style and generate Streetscapes that demonstrate distinguishable signature details of the target location, e.g., New York style street sign post and Parisian Juliet balcony (or “balconet”).

#### 5 DISCUSSION AND CONCLUSIONS

Limitations. While our system generally generates high-fidelity, realistic Streetscapes, we observe occasional unnatural movements of transient objects in our Streetscapes, e.g., a car suddenly appearing or disappearing. We hypothesize the cause to be the limited capture frequency of street view data, and the pervasive presence of such objects in the data. Explicitly modeling transient objects and allowing for control of them would be a promising area of exploration. In addition, our results exhibit some short-range inconsistencies (e.g., slight color flicker from frame to frame), due to a balance between quality and consistency that is inherent to many diffusion problems.

Conclusion. We have designed a system called Streetscapes capable of generating highly realistic, consistent street views of a synthesised urban scene. The generated imagery can be controlled by scene layout, camera pose, and optionally, a “style” text input. This system is only possible due to our choice of the scale and high quality two-frame diffusion model, as well as identifying street view and map corpora as a unique data source. Furthermore, we overcome the limitation of video diffusion models to a finite number of frames via an autoregressive video diffusion method powered by our novel temporal imputation approach, enabling truly longrange view generation. Our work marks a forward step in visual generative models, enriching generative capabilities from objects to unbounded large-scale scenes.

#### ACKNOWLEDGMENTS

We thank Thomas Funkhouser, Kyle Genova, Andrew Liu, Lucy Chai, David Salesin, David Fleet, Jonathon T. Barron, Qianqian Wang, Shiry Ginosar, Luming Tang, Hansheng Chen, and Guandao Yang for their comments and for constructive discussions. We thank William T. Freeman and John Quintero for helping review our draft. We thank all anonymous reviewers for their helpful suggestions. G.W. was in part supported by Google, Samsung, and Stanford HAI. B.D. was supported by a Meta PhD Research Fellowship. The initial idea of this project was partly inspired by the Star Guitar video created by Michel Gondry and The Chemical Brothers.

#### REFERENCES

Titas Anciukevičius, Zexiang Xu, Matthew Fisher, Paul Henderson, Hakan Bilen, Niloy J Mitra, and Paul Guerrero. 2023. Renderdiffusion: Image diffusion for 3d reconstruction, inpainting and generation. In Proc. Conference on Computer Vision and Pattern Recognition (CVPR).

Dragomir Anguelov, Carole Dulong, Daniel Filip, Christian Frueh, Stéphane Lafon, Richard Lyon, Abhijit Ogale, Luc Vincent, and Josh Weaver. 2010. Google Street View: Capturing the World at Street Level. Computer 43 (2010). http://ieeexplore. ieee.org/xpls/abs_all.jsp?arnumber=5481932&tag=1

Sherwin Bahmani, Jeong Joon Park, Despoina Paschalidou, Xingguang Yan, Gordon Wetzstein, Leonidas Guibas, and Andrea Tagliasacchi. 2023. CC3D: Layoutconditioned generation of compositional 3d scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Universal Guidance for Diffusion Models. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=pzpWBbnwiJ

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. 2024. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945 (2024).

Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. 2023. Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields. ICCV (2023). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023a. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023b. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. (2024). https: //openai.com/research/video-generation-models-as-world-simulators

Shengqu Cai, Eric Ryan Chan, Songyou Peng, Mohamad Shahbazi, Anton Obukhov, Luc Van Gool, and Gordon Wetzstein. 2023. DiffDreamer: Towards Consistent Unsupervised Single-view Scene Extrapolation with Conditional Diffusion Models. In ICCV.

Lucy Chai, Richard Tucker, Zhengqi Li, Phillip Isola, and Noah Snavely. 2023. Persistent Nature: A Generative Model of Unbounded 3D Worlds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Eric R Chan, Koki Nagano, Matthew A Chan, Alexander W Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. 2023. Generative novel view synthesis with 3d-aware diffusion models. Proc. International Conference on Computer Vision (ICCV) (2023).

Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, and Hao Su. 2023b. Single-Stage Diffusion NeRF: A Unified Approach to 3D Generation and Reconstruction. arXiv preprint arXiv:2304.06714 (2023).

Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. 2023a. Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. 2023. LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes. arXiv preprint arXiv:2311.13384 (2023).

D. Cohen-Or, Y.L. Chrysanthou, C.T. Silva, and F. Durand. 2003. A survey of visibility for walkthrough applications. IEEE Transactions on Visualization and Computer Graphics 9, 3 (2003), 412–431. https://doi.org/10.1109/TVCG.2003.1207447

Michael Deering, Stephanie Winner, Bic Schediwy, Chris Duffy, and Neil Hunt. 1988. The triangle processor and normal vector shader: a VLSI system for high performance graphics. Acm siggraph computer graphics 22, 4 (1988), 21–30.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. 2023. Objaverse-XL: A Universe of 10M+ 3D Objects. arXiv:cs.CV/2307.05663

Carl Doersch, Saurabh Singh, Abhinav Gupta, Josef Sivic, and Alexei Efros. 2012. What makes paris look like paris? ACM Transactions on Graphics 31, 4 (2012). Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. 2023. SceneScape: TextDriven Consistent Scene Generation. In NeurIPS.

Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. 2023. Magicdrive: Street view generation with diverse 3d geometry control. arXiv preprint arXiv:2310.02601 (2023).

Ruiqi Gao*, Aleksander Holynski*, Philipp Henzler, Arthur Brussee, Ricardo MartinBrualla, Pratul P. Srinivasan, Jonathan T. Barron, and Ben Poole*. 2024. CAT3D: Create Anything in 3D with Multi-View Diffusion Models. arXiv (2024).

Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi. 2023. Nerfdiff: Single-image view synthesis with nerf-guided distillation from 3d-aware diffusion. In ICML.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2023a. SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models. arXiv preprint arXiv:2311.16933 (2023).

Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023b. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023).

Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas Oğuz. 2023a. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371

(2023).

Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. 2023b. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662 (2023).

Zekun Hao, Arun Mallya, Serge Belongie, and Ming-Yu Liu. 2021. Gancraft: Unsupervised 3d neural rendering of minecraft worlds. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 14072–14082.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS) 33 (2020), 6840–6851.

Jonathan Ho and Tim Salimans. 2021. Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. 2022. Video diffusion models. Advances in Neural Information Processing Systems 35 (2022), 8633–8646.

Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. 2023. Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 7909–7920.

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).

Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. 2023. Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation. arXiv preprint arXiv:2311.17117 (2023).

Heewoo Jun and Alex Nichol. 2023. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463 (2023).

Animesh Karnewar, Andrea Vedaldi, David Novotny, and Niloy J Mitra. 2023. Holodiffusion: Training a 3D diffusion model using 2D images. In Proc. Conference on Computer Vision and Pattern Recognition (CVPR).

Xuanyi Li, Daquan Zhou, Chenxu Zhang, Shaodong Wei, Qibin Hou, and Ming-Ming Cheng. 2024. Sora Generates Videos with Stunning Geometrical Consistency. arXiv preprint arXiv:2402.17403 (2024).

Zhengqi Li, Richard Tucker, Noah Snavely, and Aleksander Holynski. 2023. Generative image dynamics. arXiv preprint arXiv:2309.07906 (2023).

Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. 2022. InfiniteNatureZero: Learning Perpetual View Generation of Natural Scenes from Single Images. In ECCV.

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023a. Magic3D: HighResolution Text-to-3D Content Creation. In Proc. Conference on Computer Vision and Pattern Recognition (CVPR). 300–309.

Chieh Hubert Lin, Hsin-Ying Lee, Willi Menapace, Menglei Chai, Aliaksandr Siarohin, Ming-Hsuan Yang, and Sergey Tulyakov. 2023b. Infinicity: Infinite-scale city synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. 2021. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proc. International Conference on Computer Vision (ICCV). 14458–14467.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9298–9309.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. 2022. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11461–11471.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2021. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations. Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.

Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. 2022. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751 (2022).

Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, Chaoyang Wang, Luc Van Gool, and Sergey Tulyakov. 2023. AutoDecoding Latent 3D Diffusion Models. arXiv preprint arXiv:2307.05445 (2023).

Hao Ouyang, Tiancheng Sun, Stephen Lombardi, and Kathryn Heal. 2023. Text2Immersion: Generative Immersive Scene with 3D Gaussians. Arxiv (2023). Ryan Po and Gordon Wetzstein. 2023. Compositional 3D Scene Generation using Locally Conditioned Diffusion. ArXiv abs/2303.12218 (2023). https://api. semanticscholar.org/CorpusID:257663283

Ryan Po, Wang Yifan, Vladislav Golyanik, Kfir Aberman, Jonathan T Barron, Amit H Bermano, Eric Ryan Chan, Tali Dekel, Aleksander Holynski, Angjoo Kanazawa, et al. 2023. State of the art on diffusion models for visual computing. arXiv preprint arXiv:2310.07204 (2023).

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2023. DreamFusion: Text-to-3D using 2D Diffusion. In The Eleventh International Conference on Learning Representations.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, and David J Fleet. 2023. The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation. arXiv preprint arXiv:2306.01923 (2023).

Jonas Schult, Sam Tsai, Lukas Höllein, Bichen Wu, Jialiang Wang, Chih-Yao Ma, Kunpeng Li, Xiaofang Wang, Felix Wimbauer, Zijian He, Peizhao Zhang, Bastian Leibe, Peter Vajda, and Ji Hou. 2023. ControlRoom3D: Room Generation using Semantic Proxy Rooms. arXiv:2312.05208 (2023).

Bokui Shen, Xinchen Yan, Charles R Qi, Mahyar Najibi, Boyang Deng, Leonidas Guibas, Yin Zhou, and Dragomir Anguelov. 2023b. GINA-3D: Learning to Generate Implicit Neural Assets in the Wild. In Proc. Conference on Computer Vision and Pattern Recognition (CVPR). 4913–4926.

Liao Shen, Xingyi Li, Huiqiang Sun, Juewen Peng, Ke Xian, Zhiguo Cao, and Guosheng Lin. 2023a. Make-It-4D: Synthesizing a Consistent Long-Term Dynamic Scene Video from a Single Image. In Proceedings of the 31st ACM International Conference on Multimedia. 8167–8175.

Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. 2023. MVDream: Multi-view Diffusion for 3D Generation. arXiv:2308.16512 (2023).

J. Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 2023. 3D Neural Field Generation Using Triplane Diffusion. In Proc. Conference on Computer Vision and Pattern Recognition (CVPR).

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022).

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020a. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020).

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020b. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations. Alexander Swerdlow, Runsheng Xu, and Bolei Zhou. 2023. Street-View Image Genera-

tion from a Bird’s-Eye View Layout. arXiv preprint arXiv:2301.04634 (2023).

Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. 2021. Csdi: Conditional score-based diffusion models for probabilistic time series imputation. Advances in Neural Information Processing Systems 34 (2021), 24804–24816.

Seth J. Teller and Carlo H. Séquin. 1991. Visibility preprocessing for interactive walkthroughs. In Proceedings of the 18th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH ’91). Association for Computing Machinery, New York, NY, USA, 61–70. https://doi.org/10.1145/122718.122725

Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. 2022. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399 (2022).

Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, and Greg Shakhnarovich.

2022. Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation. arXiv preprint arXiv:2212.00774 (2022).

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2023. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. arXiv preprint arXiv:2312.03641 (2023).

Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. 2022. Novel View Synthesis with Diffusion Models. arXiv:cs.CV/2210.04628

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023a. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7623–7633.

Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P. Srinivasan, Dor Verbin, Jonathan T. Barron, Ben Poole, and Aleksander Holynski. 2023b. ReconFusion: 3D Reconstruction with Diffusion Priors. arXiv (2023).

Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, and Kai Zhang. 2024. DMV3D: Denoising Multi-View Diffusion using 3D Large Reconstruction Model. In Proc. International Conference on Learning Representations (ICLR).

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. 2023. MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model. arXiv preprint arXiv:2311.16498 (2023).

Kairui Yang, Enhui Ma, Jibin Peng, Qing Guo, Di Lin, and Kaicheng Yu. 2023a. BEVControl: Accurately Controlling Street-view Elements with Multi-perspective Consistency via BEV Sketch Layout. arXiv preprint arXiv:2308.01661 (2023).

Yuanbo Yang, Yifei Yang, Hanlei Guo, Rong Xiong, Yue Wang, and Yiyi Liao. 2023b. UrbanGIRAFFE: Representing Urban Scenes as Compositional Generative Neural Feature Fields. In ICCV.

Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. 2023. WonderJourney: Going from Anywhere to Everywhere. arXiv preprint arXiv:2312.03884 (2023).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Showing Frame: 1, 25, 80, 100

Showing Frames: 5, 23, 41, 94

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Map Scale: 297m x 297m

Map Scale: 297m x 297m

Camera Trajectory: 100 frames covering 176m

Camera Trajectory: 100 frames covering 160m

Text Prompt: “Paris Street”

Text Prompt: “London Street”

- Figure 5: Long-range Street Cruise. While alternative autoregressive generation methods suffer from severe quality degradation after 32 frames (see Fig. 6 and Tab. 2), our Streetscapes system can generate long street cruises without noticeable drift in quality for 100 frames along paths of over 170 meters spanning multiple city blocks. Note how our results are consistent with the specified scene layout and camera poses (illustrated on the maps). In addition, in contrast to prior methods that support either forward-only [Liu et al. 2021] or backward-only [Fridman et al. 2023] camera motion, our system allows for flexible camera control wherein the user can freely move and turn the camera.

InfNat0mono

InfNat0proxy

Zero123A

Street scapes (Ours)

Diff Dreamer

Init Step-1 Step-20 Step-40 Init Step-1 Step-20 Step-40

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

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

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

- Figure 6: Perpetual Street View Generation. We compare Streetscapes with state-of-the-art autoregressive view generation methods, including InfiniteNature-Zero [Li et al. 2022] (using either monocular depth, i.e. InfNat0-mono, or proxy depth, i.e. InfNat0-proxy), DiffDreamer [Cai et al. 2023], and an autoregressive variant of Zero123 [Liu et al. 2023], i.e. Zero123-A. In this task, we are given an initial input image and a camera track. Each method aims to generate a consistent street view following the camera track autoregressively. We pick generation step-1 to demonstrate degradation-free generation quality, as well as step-20 and step-40 for long-range generation quality. Note how the results of our Streetscapes method remain highly realistic, while those of other methods degrade significantly.

### + +

Weather Text Control:

G-buffer:

Time-of-day Text Control:

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

“…” + “rainy day”

“…” + “at sunrise”

Map Semantics

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

“…” + “at noon”

Disparity (Inverse Depth)

“…” + “cloudy day”

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

“…” + “snowy day” “…” + “in the evening”

Normalized Height

- Figure 7: Text-Controlled Streetscapes. We build our Streetscapes system on a pre-trained text-to-image model. This enables text-prompt based interaction with the generation process. We demonstrate a few examples of text-controlled Streetscapes in this figure. Given the same map and camera poses (and hence the same G-buffers), our system can condition on time of day and weather conditions described by text prompt. It generates high-quality street views in diverse conditions consistent with the text description.

[Figure 282]

[Figure 283]

Map Scale: 314m x 314m

Map From: Rue de Tocqueville Paris, France

Map Center: (48.88336656, 2.31259942)

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

Signatures of “NYC” streets:

Signatures of “Paris” streets:

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Text Prompt: “New York City Street”

Text Prompt: “Paris Street”

G-buffer/ Map Semantics

G-buffer/ Disparity (Inverse Depth)

G-buffer/ Normalized Height

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

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

- Figure 8: What Makes Paris Look Like New York? We demonstrate an example of mix-and-match scene layout with geographic styles using our Streetscapes system. Given a street map from Paris, we can generate Streetscapes that look like New York City, thanks to our disentanglement of layout and scene appearance, and to the text control inherited from text-to-image models. Following prior works [Doersch et al. 2012], we could analyze our generation to summarize the “signature” of a geographic style our system learns, i.e., “what makes Paris look like New York”? For instance, in the above example, our model learns the balustrade window and French balconies unique to Paris, and the very NYC-style road sign posts.

Streetscapes SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA

#### A EXTENSION TO 4-FRAME VIDEO DIFFUSION MODELS

Ground Image

In Sec. 3, we discuss the Streetscapes system based on a 2-frame video generation model. The same system design also applies to models that generate more frames at once. We validate this generalization by extending the video generation model to a 4-frame version. In a 4-frame model, the Parallel Denoising step in Fig. 3 generates four frames, and the Temporal Imputation step uses the last two generated frames as the conditioning to generate the next two frames. We find in our experiments that the 4-frame model improves the consistency, likely due to increased context size. On our project page, we show result videos from this 4-frame model.

- (a) Poor Boundary Alignment

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

G-buffer / Disparity Image / Disparity Overlay

- (b) Poor Scene Content Alignment

#### B 3D RECONSTRUCTION FROM STREETSCAPES

The aforementioned system generates Streetscapes that are reasonably multi-view consistent. Nevertheless, for applications such

Figure 9: Misalignment of Aerial Data and Ground Images. We show two representative examples of misalignment of aerial data (height maps) and the ground images in our dataset. By overlaying disparity maps rendered from the height maps atop the corresponding ground-level RGB views, we observe noticeable misalignment. Such misalignment includes the offsets near occlusion boundaries that may be attributable to systematic error due to the precision of aerial captures. We also see misalignment of scene content, due to the discrepancy between the the static height maps we use and the ever-changing street scenes characteristic of vibrant cities like NYC. These discrepancies and misalignments necessitate that our system be robust to data noise.

- as VR/AR that demands perfect 3D consistency, generating videos alone may not meet the criteria. Hence, we introduce an optional 3D reconstruction step to our Streetscapes system that reconstructs a Neural Radiance Field [Mildenhall et al. 2020] from generated Streetscapes. We base our reconstruction on Zip-NeRF [Barron et al. 2023]. Similar to prior works [Gao* et al. 2024; Wu et al. 2023b], we add an LPIPS [Zhang et al. 2018] loss to steer the focus of reconstruction to holistic rendering quality instead of pixel-level details that might not be perfectly consistent in Streetscapes. Additionally, we find that using Temporal Imputation with SDEdit [Meng et al. 2021] to resample each frame conditioning on neighbouring frames and picking the sample that is closest to the mode helps remove random transient objects and stabilizes the results. Please see 3D reconstruction results on our project page.

C INTERPOLATING SPARSE STREET VIEW IMAGES

In addition to generating non-existing realistic streets, Streetscapes can also assist the virtual exploration of real streets, thanks to the flexible conditioning of Temporal Imputation. In particular, we can interpolate real street view captures, which by themselves are too sparse to simulate smooth experiences of driving or walking along a street. Instead of conditioning on the first two frames in Temporal Imputation using a 4-frame model, we can condition on the first and last frame and use Temporal Imputation to generate the two frames in between. This effectively interpolates between the sparse image sequence, increasing the frame rate by about 3x. Applying such interpolation iteratively can further increase the frame rate. Alternatively, we can run the 3D reconstruction described in Sec. B to obtain a 3D model, from which we can re-render new paths

- at any camera speed. Please see our interpolation results on our project page.

map data we use can be coarse and noisy, due to aerial capture precision constraints. This results in misalignment between our G-buffers and the actual street view images. We show examples of such misalignment in Fig. 9. 2) The camera poses are not always pixel-accurate. To obtain camera poses, we use the latitude and longitude from the data source as coordinates on the ground plane, though we estimate the height above ground ourselves, using the approximate height of the capture vehicle as a proxy estimate. As a result of approximations in the calculated pose, our G-buffers deviate from perfectly aligned G-buffers. To mitigate the effects of this noise, we design our system to be more driven by images, via the choice of training two-frame motion modules directly from street view image pairs. Additionally, our choice of ControlNet for layout and camera control is robust to infrequent noise in the control signal in training, which helps ensure our generation quality after training even with the presence of imperfect G-buffers. In addition to these issues, some source image regions are blurred for privacy reasons. We use street view metadata to filter out these examples. For instance, we remove images with blurred areas over 0.04%.

#### D CHALLENGES WITH GEOGRAPHIC DATA

Some characteristics of the map and street view image data we train from lead to robustness challenges for our Streetscapes system due to data noise when used at scale without manual curation. We observe two main types of noise in our data: 1) The aerial height

#### E ABLATION STUDY ON AUTOREGRESSIVE VIDEO DIFFUSION

In Secs. 4.2 and 4.3 of the main paper, we compared our diffusion model–based Streetscapes generation system to GAN-based approaches such as InfiniCity and InfNat0, and showed that diffusion

Init Step-1 Step-20 Step-40 Step-1 Step-20 Step-40

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

###### RGB Control

Temporal Imputation (Ours)

- Figure 10: Temporal Imputation (Ours) vs RGB Control. We compare temporal imputation, our autoregressive video diffusion method, to an alternative method, RGB Control, that adds the condition image as extra RGB channels to the ControlNet [Zhang et al. 2023]. Both methods are using the same two-frame diffusion model. We observe that while RGB Control still suffers from serious quality degeneration, our temporal imputation method robustly generates high quality Streetscapes across a long range.

Table 3: Ablation on Autoregressive Video Diffusion ( best ,

2nd best ). We first validate our choices of using imputation against an alternative method where we add extra RGB channels to ControlNet by feeding the condition image to the first frame and a black image to the second frame (RGB Control). We find that RGB Control still demonstrates severe quality degeneration after only 16 steps. Moreover, we validate the impact of the resampling in temporal imputation (Ours vs Ours - Resample) and the initialisation from a noised image warped from the condition image to the target view (Ours vs Ours - WarpInit). We notice that both techniques help improve the consistency (lower LPIPS) of generation.

FID↓/KID↓ @ Steps 1–16 16–32 32–64

Metric LPIPS ↓

RGB Control 0.603 25.29/0.024 96.29/0.112 136.68/0.153

Ours - Resample 0.525 14.29/0.013 25.56/0.028 31.02/0.030 Ours - WarpInit 0.522 11.51/0.012 26.57/0.029 36.00/0.035 Ours 0.519 11.69/0.012 25.59/0.028 35.47/0.034

models lead to higher generation quality. In this section, we conduct ablation studies on the design choices of our proposed autoregressive video diffusion method. Specifically, we investigate:

(1) the choice of imputation-based autoregressive generation against an alternative autoregressive approach using our

system by adding an extra RGB control channel to the ControlNet, where for the condition frame it is the condition image and for the target frame it is all black (RGB Control).

- (2) the impact of resampling in temporal imputation (Ours vs Ours - Resample).
- (3) the impact of initialize each temporal imputation from a noised image warped from the condition image to the target view (Ours vs Ours - WarpInit).

We compare all variants on the perpetual street view generation task (Sec. 4.3) where we can compare both near-range accuracy and long-range image quality.

From the results in Tab. 3, we first find that the alternative RGB Control approach still suffers from severe quality degeneration. This again supports our key insight that using multiple noised copies of the condition image in imputation-based autoregressive generation is much more robust than simply conditioning on the image itself as input. In Fig. 10, we show the significant difference in image quality between RGB Control and temporal imputation after just 20 steps in autoregressive generation.

While imputation generally improves image quality over longrange autoregressive generation, as shown in Tab. 3 for all variants of temporal imputation, we aim to strike a balance between nearrange accuracy (i.e. alignment with the given view) and long-range quality. Inspired by prior works [Bansal et al. 2024; Lugmayr et al. 2022; Meng et al. 2021] that apply diffusion models to image processing problems, we adopt Resample and WarpInit to our imputationbased autoregressive generation pipeline. Resample is a common practice to homogenize diffusion generation when partial guidance

Streetscapes SIGGRAPH Conference Papers ’24, July 27-August 1, 2024, Denver, CO, USA

Sample A

Sample B Sample C Sample D

| |[Figure 345]|
|---|---|
| | |

[Figure 346]

[Figure 347]

[Figure 348]

Source View Ours

##### - Resample

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

Ours

##### - WarpInit

Reference Target View

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

Ours

- Figure 11: Impact of Resample and WarpInit. In this experiment, we validate the impact of Resample and WarpInit techniques (see Sec. 3.3) in temporal imputation. Given the source view on the left, we use different variants of temporal imputation, one without Resample (Ours - Resample), one without WarpInit (Ours - WarpInit), and the full temporal imputation, to autoregressively generate a target view, for a few samples using different random seeds. Compared to the reference real target view image, we find that our full temporal imputation produces most similar samples, indicating the best consistency in near-range view generation.

signals are injected [Bansal et al. 2024; Lugmayr et al. 2022]. It can be connected to SDE solvers for the stochastic sampling in the diffusion denoising process. WarpInit is also a principled guidance approach adopted from SDEdit [Meng et al. 2021] in image editing. We find in Tab. 3 that both Resample and WarpInit help increase the similarity of generated Streetscapes to the reference images. This is further backed by our qualitative results in Fig. 11, where we find that among a few random samples from each temporal imputation variants, our full method with both Resample and WarpInit generates the most similar images to the reference, indicating the best near-range accuracy.

#### F MORE TRAINING DETAILS

We base our system on a pretrained text-to-image latent diffusion model [Rombach et al. 2022]. To enable two-frame generation, we first insert motion modules as proposed in AnimateDiff [Guo et al. 2023b], but change their 16-frame formulation to a 2- or 4-frame one. We conduct a two-stage training procedure for our diffusion model. First, we train only the motion modules but keep parameters of the base diffusion model U-Net frozen. We also use per-location text in our training. For each image, we associate it with the text “[location] street” where “[location]” is the city name for the image, e.g., “London”. After training the motion modules, we train a ControlNet [Zhang et al. 2023] for layout and camera pose control. while training the ControlNet, we keep both the base diffusion model and

the base motion modules frozen. For both training stages, we use the standard diffusion noise prediction objective from DDPM [Ho et al. 2020]. We train the motion modules with a learning rate of 1𝑒−5 and batch size at 256 for 1M iterations. We train the ControlNet with a learning rate of 1𝑒−5 and batch size of 256 for 500k iterations.

#### G VIDEO DIFFUSION MODELS VS 3D REASONING

In our system, we choose to use Video Diffusion Models (VDMs) trained on street view image sequences to generate consistent Streetscapes, instead of implementing exact 3D rendering in our generated views. This design choice is driven by two observations. The first is the underlying data noise. As described in Sec. D, street view data is noisy, with issues including inconsistency across modalities (e.g. map-vs-image), inaccurate and coarse height maps, frequent transient objects, and inaccurate camera poses. This noise is a significant challenge to generative methods that rely on exact 3D reasoning, as shown by the results of Infinite-Nature and DiffDreamer in Table 2 of our main paper. Second, VDMs are becoming more and more consistent (as evidenced by methods like Sora [Brooks et al. 2024]), and seem like a promising route towards generating large-scale 3D scenes. Yet they are difficult to control and still can’t scale to full city-scale scenes. Hence, in our Streetscapes

system, we addresses challenges of controllability and scaling via G-buffer conditioning and autoregressive generation.

#### H RECENT PROGRESS IN VIDEO DIFFUSION MODELS

As we were developing our Streetscapes system using the best Video Diffusion Model design that was publicly available, namely AnimateDiff [Guo et al. 2023b], there are exciting concurrent progresses in video diffusion models. For example, WALT [Gupta et al. 2023b] uses a Diffusion Transformer architecture to enable longer and more stable video generation. [Bar-Tal et al. 2024] changes the latent diffusion to a pixel diffusion framework and uses multiple levels of spatiotemporal compression and decompression to improve the duration and quality of its video generations. Thanks to our system design, both methods are compatible with our Streetscapes system. Indeed, incorporating these models into Streetscapes would

be an interesting future direction. Even more impressive is the concurrently released Sora [Brooks et al. 2024], which already “generates videos with stunning geometrical consistency” [Li et al. 2024]. This provides further evidence that our design choice of building Streetscapes systems based on video diffusion models is promising.

#### I STREETSCAPES BEYOND STREET VIEW

In this work, we aim to design a globally applicable system. We identify Google Street View as our data source that can potentially provide global coverage. Therefore, we only conduct experiments on street view data. However, our system design, in part or whole, is applicable to other datasets such as indoor datasets with floorplans (leading to building walk-through generations). Also, our autoregressive video generation approach is applicable to any diffusion-based video generation model. It would be an interesting follow-up direction to study incorporating this technique to recent video diffusion models such as WALT [Gupta et al. 2023b] and Lumiere [Bar-Tal et al. 2024].

