## TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models

# arXiv:2502.06608v3[cs.CV]27Mar2025

Yangguang Li1 * Zi-Xin Zou1 * Zexiang Liu1 Dehu Wang1 Yuan Liang1 Zhipeng Yu1 Xingchao Liu3 Yuan-Chen Guo1 Ding Liang1 † Wanli Ouyang2,4 Yan-Pei Cao1

Project Homepage: TripoSG-Page

[Figure 1]

Figure 1. High-quality 3D shape samples from our largest TripoSG model. Covering various complex structures, diverse styles, imaginative designs, multi-object compositions, and richly detailed outputs, demonstrates its powerful generation capabilities.

### Abstract

Recent advancements in diffusion techniques have propelled image and video generation to unprecedented levels of quality, significantly accelerating the deployment and application of generative AI. However, 3D shape generation technology has

1VAST

- 2The Chinese University of Hong Kong
- 3The University of Texas at Austin 4Shanghai AI Laboratory ∗: Equal Contribution †: Corresponding Author

.

so far lagged behind, constrained by limitations in 3D data scale, complexity of 3D data processing, and insufficient exploration of advanced techniques in the 3D domain. Current approaches to 3D shape generation face substantial challenges in terms of output quality, generalization capability, and alignment with input conditions. We present TripoSG, a new streamlined shape diffusion paradigm capable of generating high-fidelity 3D meshes with precise correspondence to input images. Specifically, we propose: 1) A large-scale rectified flow transformer for 3D shape generation, achieving state-of-the-art fidelity through training on extensive, high-quality data. 2) A hybrid super-

vised training strategy combining SDF, normal, and eikonal losses for 3D VAE, achieving highquality 3D reconstruction performance. 3) A data processing pipeline to generate 2 million highquality 3D samples, highlighting the crucial rules for data quality and quantity in training 3D generative models. Through comprehensive experiments, we have validated the effectiveness of each component in our new framework. The seamless integration of these parts has enabled TripoSG to achieve state-of-the-art performance in 3D shape generation. The resulting 3D shapes exhibit enhanced detail due to high-resolution capabilities and demonstrate exceptional fidelity to input images. Moreover, TripoSG demonstrates improved versatility in generating 3D models from diverse image styles and contents, showcasing strong generalization capabilities. To foster progress and innovation in the field of 3D generation, we open source our model and code at: TripoSG.

Various Original Datasets Training Data

[Figure 2]

[Figure 3]

Data Processing

[Figure 4]

Watertight Direction Align Many Low-Quality Data Watertight Direction Align Filtered High-Quality Data

(i) Data-Building System

Output

[Figure 5]

[Figure 6]

Noised Latents

Latent Tokens

[Figure 7]

[Figure 8]

[Figure 9]

CLIP

MoE Residual Attention Block

DINO

Linear + LayerNorm

VAE Decoder

(ii) TripoSG

Input

Figure 2. The overview of our method consists of two main components: (i) Data-Building System and (ii) TripoSG Model. The databuilding system processes the 3D models from various datasets (e.g., Objaverse and ShapeNet) through a series of data processing steps to create the training data. Our TripoSG model is then trained on this curated dataset for high-fidelity shape generation from a single input image.

Key words: 3D Generation, Rectified Flow, Image-to-3D

### 1. Introduction

Recent advancements in large-scale visual datasets (Bain et al., 2021; Schuhmann et al., 2022; Wang et al., 2023c) have propelled remarkable progress in generative models. These models effectively compress high-dimensional visual data, such as images and videos, into latent spaces, enabling the generation of high-quality visual content conditioned on various input modalities. State-of-the-art generative AI models, including SD3 (Esser et al., 2024), FLUX (blackforestlabs, 2024), and Sora (Brooks et al., 2024), exemplify this capability, producing strikingly realistic images and videos from diverse conditional inputs. This breakthrough has revolutionized human visual creation, opening new avenues for artistic expression and content generation.

In the domain of 3D content creation, the pursuit of highquality, production-ready 3D generation remains a primary objective for researchers, artists, and designers. Substantial progress has been made in generating 3D models from single images, with approaches broadly categorized into two paradigms: large-scale reconstruction-based methods (Hong et al., 2023; Li et al., 2023; Tochilkin et al., 2024; Wang

- et al., 2023b; 2024; Wei et al., 2024; Xu et al., 2024; 2023; Zhang et al., 2024a; Zou et al., 2024b) and diffusion-based methods (Li et al., 2024c; Wu et al., 2024b; Zhang et al.,

- 2023; 2024b; Zhao et al., 2024). Large-scale reconstructionbased methods primarily utilize a network to regress the 3D model in a deterministic way. While effective, these approaches often struggle with inconsistencies in overlapping regions from multiple input views (which can be generated

from a single view by multi-view diffusion models (Liu et al., 2023d; Shi et al., 2023b; Wang & Shi, 2023b)) and exhibit artifacts in occluded areas. Conversely, diffusion-based methods train on 3D representations or latent representations compressed by Variational AutoEncoders (VAEs). As generative rather than regression methods, they circumvent some challenges inherent to reconstruction approaches. However, current methods predominantly rely on occupancy representations, often necessitating additional post-processing to mitigate aliasing artifacts and lacking fine-grained geometric details. Moreover, vanilla diffusion architectures and sampling strategies yield suboptimal 3D model quality, resulting in a significant alignment gap between generated models and input images. A common limitation across both approaches is their heavy reliance on the Objaverse dataset (Deitke et al., 2023). The necessity for rigorous data filtering often reduces the usable data samples by nearly half, presenting a substantial challenge in scaling data compared to the image and video domains.

Given these challenges in 3D generation and the substantial successes observed in image and video synthesis, we posit a critical question: What is the optimal paradigm for generating high-fidelity 3D models with precise alignment to input conditions?

In response to this inquiry, we present TripoSG, a highfidelity 3D generative model that leverages a rectified flow transformer trained on meticulously curated, largescale data. Our approach achieves unprecedented quality in image-to-3D generation, characterized by finer details and superior input condition alignment. Inspired by 3DShape2VecSet (Zhang et al., 2023), we train our genera-

tive model on latent representations efficiently compressed by a VAE model. We identify that the quality of the generated models is heavily dependent on the capacity of the latent space (i.e., the number of tokens) and the volume of high-quality 3D data available. Thus, it’s crucial to scale up the model with a more efficient neural architecture. To address this challenge, we propose several key advancements:

- 1. We pioneer the use of a rectified flow transformer architecture (Liu et al., 2023c) in 3D generation, drawing inspiration from recent successes in scaling up image/video models (e.g., SD3 (Esser et al., 2024), FLUX (blackforestlabs, 2024), and Meta Movie Gen (team at Meta, 2024)). This choice is motivated by its simplicity and superior performance in terms of training stability and convergence.
- 2. Our model builds upon the DiT (Peebles & Xie, 2023) framework, incorporating critical enhancements for improved scalability. These include skip-connections, RMSNorm (Zhang & Sennrich, 2019), and the injection of both global and local features. Furthermore, we adopt the DiT-MoE (Fei et al., 2024) approach, replacing standard Feed-Forward modules in each block with a mixture-of-experts (MoE) mechanism. This configuration retains one shared expert while selecting the top two experts for each operation.
- 3. We have successfully trained a 4 billion (4B) parameter 3D generative model at a latent resolution of 4096 tokens, leveraging this unprecedented scale to produce highly detailed and geometrically precise structures.

Recognizing the critical role of VAE reconstruction quality in latent diffusion or flow models (Rombach et al., 2022;

- Zhang et al., 2023), we introduce several improvements to our VAE training process, including 1) We employ SDF (Signed Distance Function) representation, which offers superior geometric expressiveness compared to occupancy representations. Occupancy grids often introduce quantization errors, resulting in aliasing artifacts or “staircasing” effects when representing continuous surfaces. 2) We implement geometry-aware supervision on SDF with surface normal guidance, significantly enhancing 3D model reconstruction. This approach enables sharper and more precise geometry without the quantization artifacts. These improvements can better connect the latent space and the 3D model space, benefiting the quality of 3D models generated by our flow transformers.

To address the scarcity of high-quality 3D data, we have developed a sophisticated data-building system. This system ensures the generation of standardized, high-quality 3D training data (Image-SDF pairs) from diverse sources through a rigorous process of 1) scoring, 2) filtering, 3) fixing and augmentation, and 4) field data production. More

importantly, we find that both data quality and quantity are crucial, demonstrating that improperly processed data can substantially impede the training process.

Building on our proposed solution, we have designed an optimized training configuration that incorporates the proposed improvements, achieving a new state-of-the-art (SOTA) performance in 3D generation with our largest TripoSG model. Fig. 2 provides an overview of our method, illustrating both the data-building system and the TripoSG model architecture. We have validated the effectiveness of each design component through a series of mini-setting experiments.

Our core contributions are as follows:

- • We introduce a large-scale rectified flow transformer for 3D shape generation, setting a new benchmark in fidelity by leveraging extensive, high-quality data.
- • We present a novel hybrid supervised training strategy that combines SDF, normal, and eikonal losses for 3D VAE, achieving state-of-the-art 3D reconstruction performance.
- • We demonstrate the crucial rule of data quality and quantity in training 3D generative models, introducing a robust data processing pipeline capable of producing 2M high-quality 3D data samples.

### 2. Related Work

#### 2.1. Lifting 2D Prior to 3D Modeling

The diffusion model (Ho et al., 2020) has demonstrated strong generative capabilities in image (Ramesh et al., 2021; Rombach et al., 2022) or video (Singer et al., 2023; Wu et al., 2023) generation. However, due to the limitations of high-quality 3D data, it has long been challenging to directly transfer the techniques from text and image generation to 3D tasks. DreamFusion (Poole et al., 2023) pioneered the use of image diffusion priors for 3D generation by proposing a Score Distillation Sampling method enabling iterative optimization of the 3D representation via differentiable volume rendering (Mildenhall et al., 2020). Subsequent work introduced numerous improvements in areas such as 3D represetation (Lin et al., 2023; Tang et al., 2024; Yi et al., 2024), sampling strategy (Liang et al., 2024; Wang et al.,

- 2023a;d; Zou et al., 2024a), incorporating additional geometric cues (Long et al., 2024; Tang et al., 2023), and multiview image generation consistency (Liu et al., 2024b; Shi et al., 2024; Wang & Shi, 2023a). Different from text-to-3D generation methods using off-the-shell text-to-image diffusion model, e.g., Stable Diffusion (Rombach et al., 2022), many works explore to train a viewpoint-aware image diffusion based on input images (Chan et al., 2023; Liu et al.,
- 2023b; Shi et al., 2023a). When image generation across

multiple views becomes more consistent or with normal or depth generation, the 3D model can be directly optimized by pixel-level loss instead of time-consuming distillation sampling, resulting in 3D generation in a few minutes (Li

- et al., 2024b; Long et al., 2024; Wu et al., 2024a).

#### 2.2. Large 3D Reconstruction Modeling

Unlike the previously introduced methods, which require a time-consuming optimization process lasting several minutes or even hours, various works propose to learn geometry with diverse representation types (e.g., point cloud (Fan et al., 2017; Wu et al., 2020), voxel (Girdhar et al., 2016; Wu et al., 2017), mesh (Wang et al., 2018; Worchel et al., 2022) or implicit field (Mescheder et al., 2019; Xu et al., 2019; Yu et al., 2021)) from input images in a deterministic process, with encoder-decoder network architecture. Recently, equipped with a ginormous collection of high-quality 3D models in Objaverse (-XL) (Deitke et al., 2023; 2024) as well as an advanced and scalable Transformer-based architecture (Vaswani et al., 2017), Large Reconstruction Model (LRM) (Hong et al., 2023) and its many subsequent variants (Li et al., 2023; Tochilkin et al., 2024; Wang et al.,

- 2023b; Wei et al., 2024; Xu et al., 2024; 2023; Zhang et al.,

- 2024a; Zou et al., 2024b) has greatly promoted the development of reconstruction-based methods. MeshFormer (Liu et al., 2024a) additionally leverages sparse UNet to downsample the voxel for Transformer layers, leading to impressive reconstruction quality. One-2-3-45 (Liu et al., 2023a) first proposes combining a 2D image diffusion model and a multiview reconstruction model, achieving generation capabilities while maintaining fast reconstruction speed. Bridged with the text-to-image and image-to-multi-view diffusion models, these multiview reconstruction methods can be easily extended to text-to-3D or image-to-3D generation tasks and achieve impressive results. However, the inconsistency between input images from different views can lead to a decline in reconstruction quality, and the unobserved regions may yield blurred results. Thus, these are only ‘reconstruction’ methods rather than ‘generation’ methods, fundamentally limiting the quality ceiling of such methods.

on the latent space (Cheng et al., 2023; Zhang et al., 2023). For a long time, these 3D diffusion methods have struggled to match the performance of the two major categories of approaches mentioned above due to the lack of large and high-quality 3D model datasets. These methods are mostly trained on simple 3D datasets (e.g., ShapeNet (Chang et al., 2015)), with limited generation ability and effectiveness, which hinders their practical application. Recently, some researchers have attempted to train a latent 3D diffusion model based on a large amount of high-quality 3D models (Hong et al., 2024; Lan et al., 2024; Li et al., 2024c; Wu et al., 2024b; Zhang et al., 2024b) and have demonstrated impressive 3D generation results. However, these methods still have limitations on high-fidelity generation with image alignment. In this paper, we adopt a 3D representation with better geometry expression ability and improve the diffusion model architecture and training strategy, achieving state-of-the-art performance on 3D shape generation.

### 3. TripoSG

This section outlines the specific framework of the TripoSG paradigm, which consists of three main parts: the flow-based generation architecture and sampling schedule ( Sec.3.1); the scaling-up strategy (Sec.3.2); the VAE architecture and supervision (Sec.3.3).

#### 3.1. Rectified Flow Transformer

Leveraging a meticulously designed VAE architecture and robust supervision information, TripoSG’s VAE, described in detail in Sec.3.3, following extensive training on largescale datasets, is capable of encoding arbitrary 3D shapes into multi-scale latent representations X = L × C, L ∈ {512,2048}, C = 64, as well as decoding them back into 3D meshes. Drawing inspiration from models such as LDM (Rombach et al., 2022) and 3DShape2VecSet (Zhang et al., 2023), we further train a rectified flow model on these latent representations, aiming to generate high-quality, semantically consistent 3D shapes under image-controlled conditions.

#### 2.3. 3D Diffusion Modeling

Training a 3D diffusion model for 3D generation is a natural idea that stems from advancements in the field of image (Ramesh et al., 2021; Rombach et al., 2022) and video (Singer et al., 2023; Wu et al., 2023) generation. Many previous works train a diffusion model based on various 3D representations, such as voxel (M¨uller et al., 2023), point cloud (Melas-Kyriazi et al., 2023; Zeng et al., 2022; Zhou et al., 2021), triplane (Shue et al., 2023), or Occupancy/SDF grid (Hui et al., 2022; Zheng et al., 2023). Some other works utilize a VAE to transfer the original representation to a compact latent space, and then train a diffusion model

3.1.1. IMAGE-TO-3D FLOW ARCHITECTURE

Our flow architecture is inspired by DiT (Peebles & Xie, 2023) and 3DShape2VecSet (Zhang et al., 2023), utilizing standard transformer blocks to construct the backbone. While this architecture has demonstrated success in classconditional tasks on ImageNet (Deng et al., 2009) and ShapeNet (Chang et al., 2015), we found that naively stacking multiple transformer blocks leads to suboptimal modeling capabilities due to insufficient information fusion between shallow and deep feature. Drawing inspiration from U-ViT (Bao et al., 2023) and the UNet structure in Stable Diffusion (Rombach et al., 2022), we follow Michelan-

| |Linear|
|---|---|
| | |

Sample

| | |
|---|---|
|Layer Norm| |

| | |
|---|---|
|MoE Residual Attention Block| |

Concatenate Weighting

| | |
|---|---|
| | |

MoE

| | |
|---|---|
|Layer Norm| |

| | |
|---|---|
|Layer Norm| |

| | |
|---|---|
|Linear| |

| | |
|---|---|
|Output Hidden| |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

…

|2-Multi-head Cross-Attention|
|---|

|1-Multi-head Cross-Attention|
|---|

|MoE Residual Attention Block|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
|Layer Norm| |

| | |
|---|---|
|Layer Norm| |

| | |
|---|---|
|Layer Norm| |

| | | |
|---|---|---|
|Expert-2| | |
| | | |

| | |
|---|---|
|Expert-1| |

| | |
|---|---|
|Linear| |

|Shared Expert|
|---|

ExpertN-1

…

Expert-N

| | |
|---|---|
| | |

|Multi-head Self-Attention|
|---|

###### MoE Residual Attention Block

| | |
|---|---|
|Layer Norm| |

###### Learned Weights

| | |
|---|---|
|MoE Residual Attention Block| |

|Router|
|---|

…

| | |
|---|---|
|Input Hidden| |

| | |
|---|---|
|MoE Residual Attention Block| |

|DINO|
|---|

|CLIP|
|---|

| | |
|---|---|
| | |

|CLIP|
|---|

|DINO|
|---|

DINO

CLIP

|Linear|
|---|

|Linear|
|---|

|Timesteps|
|---|

Image Input Noised Latents t

Image Input

- Figure 3. Left: the overall architecture of TripoSG. Middle: the detailed internal module of each block. Right: the detailed internal components of the MoE.

gelo (Zhao et al., 2024) by introducing long skip residual connections between blocks to capture comprehensive feature information, enhancing the network’s representational capacity.

encoded by the VAE, we project it to the hidden dimension

- W using an MLP, producing a L×W feature. Following the design of Michelangelo(Zhao et al., 2024) and CLAY(Zhang et al., 2024b), we concatenate the features of timestep t and latent X, yielding a (L + 1) × W feature, which is then fed into the flow backbone.

For image conditioning, Michelangelo(Zhao et al., 2024) implements conditioning by concatenating global features extracted by CLIP(Radford et al., 2021) with the input latent

- X. However, using global image features and concatenationbased injection results in a loss of fine control over the generated 3D shapes. In contrast, CLAY(Zhang et al., 2024b), which pioneered explored the large-scale 3D generation, replaces the concatenation method with a cross-attention mechanism for injecting image information. CLAY trains a 1.5 billion parameter text conditioning base model at high computational cost (256 A800 GPUs over 15 days), then freezes the base model and trains an additional 352 million parameters for image conditioning via cross-attention with reduced computational cost (over 8 hours). However, while this approach shortens the training time for image conditioning compared to text conditioning, it limits the ability to fully update model parameters based on image information, resulting in challenges in achieving fine-grained consistency between the generated 3D shapes and the image condition. Furthermore, captions generated from rendered images can introduce semantic gaps due to lighting, shadows, and textures, which deviate from the actual 3D geometry. Even when using tools like GPT-4V(OpenAI, 2023b) to generate captions for 3D models, accuracy issues persist. These

As shown on the left side of Fig.3, the backbone is divided into three parts: encoder blocks, a middle block, and decoder blocks, with an equal number N of blocks in both the encoder and decoder. Each encoder block is connected to its corresponding decoder block via skip connections. Specifically, the output of the i-th encoder block is skipconnected to the output of the (N − i)-th decoder block. Our flow backbone is composed of 2N + 1 transformer blocks with residual connections between them. In this setup, N is 10, the hidden dimension W is 2048, and each transformer block has 16 attention heads. The entire flow architecture comprises approximately 1.5 billion parameters. The following equation describes the flow architecture with skip-connections, where DB denotes the decoder block and EB denotes the encoder block.

Z(DBN−i) = DB(N−i) Z(DBN−i−1) + EB(i) Z(EBi−1) ,

(1)

i ∈ {0,1,...,N}.

Building on TripoSG’s backbone, we designed a method to inject both timestep and image conditioning, enabling controllable 3D generation. For timestep t we first encode it using the Timesteps layer from the diffusers library(von Platen et al., 2022), followed by an MLP layer that projects it to the hidden dimension W, obtaining a 1 × W feature. Similarly, for the input latent X, with dimensions L × C

added noise to the alignment between captions and shapes, slowing down training convergence and posing challenges to precision.

In contrast, our approach directly leverages CLIP-ViTL/14(Radford et al., 2021) to extract global image features Iglobal and DINOv2-Large(Oquab et al., 2023) to extract local image features Ilocal. In each flow block, both global and local features are injected simultaneously using separate cross-attention mechanisms. The outputs are then combined with the original input and passed to the next stage. This method allows the model to attend to both global and local image information in every block, enabling faster training convergence while maintaining strong detail consistency between the generated 3D model and the input image.

The process within each block of the flow architecture can be expressed by the following equation.

Z = Concat(X,t) (2) Z = Z + SelfAttn(Norm(Z)) (3)

Z = Z + CrossAttn(Norm(Z),Ilocal)+ (4) CrossAttn(Norm(Z),Iglobal) (5) Z = Z + FFN(Norm(Z)) (6)

- 3.1.2. RECTIFIED FLOW BASED GENERATION

We trained the 3D generation model using our designed flow architecture, exploring sampling strategies including DDPM, EDM, and Rectified Flow, and ultimately selected Rectified Flow for the final generative model.

DDPM leverages a Markov chain to establish a connection between Gaussian noise space and the data distribution, enabling high-quality data generation. Specifically, noise ϵ is progressively added to the data x0, transforming it into a standard Gaussian distribution. The data sample xt at any time step t can be expressed by the following equation:

##### xt = √α¯t x0 + √1 − α¯t ϵ (7)

Where α¯t = ts=1 αs, αt = 1 − βt and βt is the predefined noise scheduling parameter. From the perspective of

interpolation, DDPM models a relatively complex curved trajectory from x0 to xt.

EDM redesigns the noise schedule and sampling method, adopting a continuous-time framework to improve both the sampling speed and generation quality of DDPM. The data sample xt at any time step t is modeled using the original data x0 and noise ϵ as follows:

##### xt = x0 + σ(t)ϵ (8)

σ(t) is a continuous noise standard deviation function, allowing for more flexible noise scheduling strategies, such

ρ

as the power form σmax1/ρ − σmin1/ρ t + σmin1/ρ

, σmin, where and σmax are the minimum and maximum noise standard deviation, and ρ is the hyperparameter that controls the shape of the curve. EDM provides a more streamlined approach to modeling xt compared to DDPM. From an interpolation perspective, EDM also models a curved trajectory from x0 to xt.

Is there a simpler linear trajectory modeling process from x0 to xt? To explore this, we further investigated Rectified Flow, which learns a vector field to map the noise distribution to the data distribution. The data sample xt at any time step t is modeled using the original data x0 and noise ϵ as follows:

##### xt = tx0 + (1 − t)ϵ (9)

This represents a simpler linear trajectory, offering a more efficient and streamlined approach compared to DDPM (Eq.7) and EDM (Eq.8).

Rectified flow’s linear sampling simplifies network training, making it more efficient and stable, which we leverage to train our 3D flow model. Additionally, drawing inspiration from SD3 logit-normal sampling, we increase the sampling weight for intermediate steps, as predictions for t in the middle of the range (0,1) are more challenging during Rectified Flow training. The sampling weight is adjusted using the following equation, where m is the biasing location parameter and s is the distribution width parameter.

2

πln(t;m,s) = s√2πt1(1−t) exp −(log(t/(1−t))−m)

2s2 (10)

It is well-known that higher resolutions require more noise to sufficiently disrupt the signal. As resolution increases, the uncertainty in the noised latent at the same timestep decreases. Therefore, following SD3, we introduce ResolutionDependent Shifting of Timestep to adjust the timestep during both training and sampling. By remapping to a new timestep, we maintain the same level of uncertainty as with the original resolution. We define the resolution of the first stage of our progressive training as the base resolution, denoted as n, with its timestep represented as tn. The subsequent stage’s resolution is defined as the fine-tune resolution, denoted as m, with its timestep represented as tm. The relationship between tm and tn is expressed by the following equation.

- m

- n tn

tm =

1 + mn − 1 tn

(11)

Leveraging Rectified Flow with logit-normal sampling and resolution-dependent shifting of timestep, we train our 3D flow model.

|[Figure 10]|
|---|

Normal Texture

[Figure 11]

[Figure 12]

K,V

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Occupancy

| |
|---|

| |
|---|

Cross-Attention

Self-Attention

Self-Attention

DownSample

Q

| |
|---|

|[Figure 17]|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

| |
|---|

SDF

[Figure 22]

Surface Normal Guidance

Latent Tokens

Figure 5. Comparison between model reconstruction based on Occupancy (top) and SDF (bottom).

Surface Points with Normals

Randomly Sampled Points

Supervise

[Figure 23]

[Figure 24]

Cross-Attention

Self-Attention

Self-Attention

Marching Cubes

K,V

this approach, we opted to scale using a Mixture-of-Experts (MoE) architecture. This method not only boosts performance by increasing model parameters but also maintains nearly constant resource usage and inference latency due to the sparse activation of the network during inference.

Q

Query Points

- Figure 4. TripoSG’s transformer-based VAE architecture. The upper is the encoder and the lower is the decoder.

As shown on the right side of Fig.3, and following previous work(Fei et al., 2024; Riquelme et al., 2021), we extended the FFN (Feed-Forward Networks) within the transformer blocks using a Mixture-of-Experts (MoE) approach. Instead of a single FFN module in the dense network, N parallel FFN expert models are employed, controlled by a gating module to scale up the model parameters. Specifically, the latent X of length L are distributed token-wise to different FFN experts by the gating module based on top-K probabilities and then reconcatenated to restore the original length L. Inspired by DiT-MoE(Fei et al., 2024)’s approach of sharing certain experts to capture common knowledge and balancing expert loss to reduce redundancy, we apply weighted activation of the top-2 experts in each token, introduce a shared expert branch across all tokens, and use an auxiliary loss to balance expert routing. Unlike DiT-MoE, we used the base model’s FFN architecture (a two-layer MLP with one GELU activation) for constructing FFN experts. Instead of training the MoE model from scratch, we initialized it from the base model, where the weights of the multiple FFN experts in each block were inherited from the corresponding FFN weights in the base model. Additionally, due to the shallow layers focusing on general features and the deeper layers capturing more object-specific details(Zeiler & Fergus, 2014), we limited MoE application to the final six layers of the decoder, where deep feature modeling is critical. This targeted scaling of parameters was applied to the most crucial part of the flow architecture. Under the action of MoE, we modified formula 17 into the following new formula.

#### 3.2. Model and Resolution Scale-up Strategy.

Larger latent resolutions and more extensive models undoubtedly lead to performance improvements. To generate even better results, we aim to scale up both latent resolution and model size while minimizing training and inference costs. Specifically, we increased the latent resolution from 2048 to 4096, and scaled the model parameters from 1.5B to 4B using a Mixture-of-Experts (MoE).

Since the VAE training does not incorporate additional positional encoding in its input, and the varying number of query points used to learn the latent representations are downsampled from a fixed set of surface points, the VAE can generalize to resolutions beyond the training set. And higher number of query points (latent resolution) improves modeling capacity. This extrapolation ability eliminates the need for retraining the VAE, allowing us to directly encode and decode at a 4096 resolution using the VAE trained on {512,2048} resolutions. By leveraging this method to directly increase the latent resolution to 4096, we provide the flow model with finer geometric latent representations for training.

Additionally, to mitigate the risk of unstable training and potential loss divergence during mixed-precision training, (Dehghani et al., 2023) recommend normalizing Q and K before attention operations. Following this approach, during fine-tuning at higher resolutions in our flow architecture, we apply learnable RMSNorm(Zhang & Sennrich, 2019) to normalize Q and K within the transformer blocks.

Z = Norm(Z), (12) Z = Z + Concat FFN(i) (Z) ,i ∈ {1,2,...,N} (13)

Directly scaling a dense model is the most straightforward approach for increasing model parameters. While this enhances performance, it significantly increases the computational resource demands and inference latency. Rather than

In our MoE scaling up, we used 8 expert models, activating the top 2 FFN experts per MoE block while sharing one FFN.

Additionally, MoE expansion was applied to the final 6 layers of the decoder, increasing the overall model parameters from 1.5B to about 4B.

#### 3.3. 3D Variational Autoencoder (VAE)

- 3.3.1. 3D MODEL REPRESENTATION

Most existing 3D shape generation works (Li et al., 2024c; Zhang et al., 2024b) adopt occupancy field or semicontinuous occupancy (Wu et al., 2024b) as the neural implicit representation for 3D model. For each query position x ∈ R3, these methods utilize a neural network D to predict the occupancy value o from the latent features f, supervised by the ground-truth occupancy value oˆ with Binary Cross Entropy (BEC) loss:

o = D (x,f), (14) L = Ex∈R3 [BCE(o,oˆ)]. (15)

Learning geometry through occupancy representation as a classification task is easier to train and converge compared to the signed distance function (SDF) as a regression task. However, the occupancy representation has limited geometric representation capabilities compared to SDF, which provides more precise and detailed geometry encoding. Additionally, models reconstructed using occupancy representation often exhibit noticeable aliasing artifacts and typically require further post-processing (e.g., smooth filter or super-sampling) to address these issues. Without postprocessing, these aliasing artifacts sometimes also impact the subsequent texture generation. Fig.5 shows some examples of geometry reconstruction and texture generation results based on occupancy and SDF, respectively.

Given these considerations, we adopt neural SDF as our 3D model representation. This method, built upon a set of latent tokens, provides a stronger geometric detail than occupancy-based approaches (Li et al., 2024c; Wu et al., 2024b; Zhang et al., 2024b). Specifically, we predict the SDF value s of each query position as:

s = D (x,f). (16)

For efficiency, we employ the truncated signed distance function (TSDF) in our VAE model. In the following paragraph, we use s to represent TSDF for simplicity.

- 3.3.2. GEOMETRY LEARNING WITH SURFACE NORMAL GUIDANCE

More importantly, SDF representation theoretically ensures the effectiveness of supervision in the gradient domain of the neural implicit field. We think geometric details are relevant to the gradient domain of the neural implicit field, which represents the higher-order information compared to the value domain of the implicit field. Therefore, we apply

surface normal guidance during VAE training to capture finer-grained geometric details, providing a better latent space for model sampling. Specifically, in addition to the commonly used SDF loss, our approach also includes direct supervision of finer detailed geometry learning using the ground-truth surface normals and an additional eikonal regularization:

Lvae = Lsdf + λsnLsn + λeikLeik + λklLkl, (17)

Lsdf = |s − sˆ| + ∥s − sˆ∥22, (18) Lsn = 1 −

∇D (x,f) ∥∇D (x,f)∥

,nˆ , (19)

##### Leik = ∥∇D (x,f) − 1∥22, (20)

where nˆ is the ground-truth surface normal, ⟨·,·⟩ denotes the cosine similarity of two vectors, Leik is the eikonal regularization, and Lkl represents the KL-regularization in the latent space. Unlike SDF loss, which involves sampling points near the surface and randomly throughout space, the surface-normal loss is applied exclusively to surface points, making it a more efficient method for supervising fine-grained geometry learning.

3.3.3. NETWORK ARCHITECTURE Following the design of 3DShape2Vecset(Zhang et al.,

- 2023), we choose the latent vector set as our latent representation, which encodes a point cloud into latent space and subsequently decodes a geometry function (i.e., SDF) from it. To facilitate more efficient scaling up, we adopt a state-of-the-art transformer-based encoderdecoder architecture (Zhang et al., 2023; 2024b; Zhao et al.,
- 2024). Specifically, we choose the downsampled version in 3DShape2Vecset (Zhang et al., 2023) that subsamples M points X′ from the full set of surface points X, and directly utilizes the point cloud itself as initial latent queries instead of learnable embeddings. Then, the surface points information, encoded by concatenating positional embedding and surface normal, is integrated into latent queries via cross-attention, resulting in compact latent tokens Z rich in geometric information, as shown in the following:

Z0 = CrossAttn(PosEmb(X),PosEmb(X′)), (21) Z = Linear SelfAttn(i) (Z0) ,i ∈ {0,1,...,Lenc},

(22)

where CrossAttn denotes a cross-attention layer, SelfAttn(i) denotes the self-attention layers, and Linear is a linear layer.

After obtaining the latent representation, we can decode the

||Direction Model|
|---|
<br><br>Character Y Model<br><br>N<br><br>|ControlNet| |
|---|---|
| | |
<br><br>Y<br><br>Fixed 3D Data<br><br>|Blender Render| |
|---|---|
| | |
<br><br>N<br><br>Untextured Model|
|---|

||Mesh2SDF| |
|---|---|
| | |
<br><br>Watertight Model<br><br>|Sampler & PySDF| |
|---|---|
| | |
<br><br>3D Points & SDF|
|---|

||Blender Render| |
|---|---|
| | |
<br><br>Raw 3D Data<br><br>Multi-view Normal<br><br>|Score Model| |
|---|---|
| | |
<br><br>Scored 3D Data|
|---|

||Large Flat Base Filter| |
|---|---|
| | |
<br><br>|Multi-Object Filter| |
|---|---|
| | |
<br><br>Filtered 3D Data<br><br>|Bad Animation Filter| |
|---|---|
| | |
|
|---|

filtering, fixing and augmentation, and field data producing, respectively.

#### 4.1. I: Data Scoring

Each 3D model is scored, with only the high-score models advancing to the subsequent processing stages. Specifically, we randomly selected approximately 10K 3D models and used Blender to render four different views of normal maps for each model. These multi-view normal maps are then manually evaluated by 10 professional 3D modelers, assigning scores on a scale from 1 (lowest) to 5 (highest). Using this annotated data, we trained a linear regression-based scoring model concatenating their CLIP(Radford et al., 2021) and DINOv2(Oquab et al., 2023) features as input. This model was subsequently used to infer quality scores from the multi-view normal maps of all 3D models for filtering.

I II III IV

- Figure 6. Demonstration of the TripoSG data-building system. I: Data scoring procedure; II: Data filtering procedure; III: Data fixing and augmentation procedure. IV: Field data producing procedure. signed distance value for each query position x ∈ R3:

Z = SelfAttn(i) (Linear(Z)),i ∈ {0,1,...,Ldec},

(23) s = CrossAttn PosEmb(x), Z . (24)

#### 4.2. II: Data Filtering

Finally, the mesh of the 3D model can be extracted by applying Marching Cubes (Lorensen & Cline, 1987) at a given resolution.

After scoring, further filtering is applied to exclude models with large planar bases, rendering errors in animations, and those containing multiple objects. Specifically, models with large planar bases are filtered by determining if different surface patches can be classified as a single plane, based on features composed of their centroid positions, normal vectors, and the area of the resulting plane. Blender identifies animated models, sets them to the first frame, and filters out any models that still exhibit rendering errors after being set. And models containing multiple objects are filtered by evaluating the proportion of the largest connected component on the opaque mask, along with the magnitude of the solidity of both the largest connected component and the entire mask.

To implement progressive flow model training for faster convergence, we follow (Zhang et al., 2024b) to adopt a multi-resolution VAE with M ∈ {512,2048} tokens, where the VAE weights are shared across different resolutions. This training strategy, combined with the position-encodingfree feature of the VAE transformer, provides the VAE with strong extrapolation capabilities, allowing it to directly infer the 3D models with higher-resolution (e.g., 4096) tokens without requiring additional fine-tuning. Unlike previous works, which used only a few surface points (only 2048 or 8192 points)(Zhang et al., 2024b) as the VAE input, we opted to use a denser surface point for each 3D model. We think the purpose of the VAE is to capture as much geometric information of the 3D model as possible, rather than functioning as a sparse point cloud reconstruction task. The more input points provided, the more geometric information is encoded in the latent space, resulting in higher-quality geometry being decoded.

#### 4.3. III: Data Fixing and Augmentation

After data filtering, we perform the orientation fixing of character models to ensure they face forward. Specifically, we select 24 orientations around the x, y, and z axes, and for each, render images from six orthogonal views: front, back, left, right, top, and bottom. The DINOv2(Oquab et al., 2023) features from these six views are concatenated to train an orientation estimation model, which is then used to infer and fix the orientation of all character models. Additionally, for all untextured models, we render multi-view normal maps and use ControlNet++(Li et al., 2024a) to generate corresponding multi-view RGB data, which serve as conditional inputs during training.

### 4. Data-Building System.

TripoSG is trained on existing open-source datasets such as Objaverse (-XL)(Deitke et al., 2023; 2024) and ShapeNet(Chang et al., 2015), which contains approximately 10 million 3D data. Since most of these data are sourced from the Internet, their quality varies significantly, requiring extensive preprocessing to ensure suitability for training. To overcome these challenges, TripoSG developed a dedicated 3D data processing system that produces high-quality, large-scale datasets for model training. As illustrated in Fig.6, the system comprises four processing stages (Data Process I∼IV), responsible for data scoring,

#### 4.4. IV: Field Data Production

Although Objaverse (-XL) (Deitke et al., 2023; 2024) contains a large amount of data, most of the models are un-

suitable for direct training, even after processing steps such

- as scoring, filtering, and fixing. Since we adopt the neural implicit field as our 3D model representation, it’s necessary to convert the original non-watertight mesh to watertight ones for computing geometry supervision (e.g., occupancy or SDF). Rather than using common methods like TSDFfusion (Newcombe et al., 2011) or ManifoldPlus (Huang et al., 2018; 2020), we are inspired by (Wang et al., 2022; Zhang et al., 2024b) to construct a Unsigned Distance Function (UDF) field with a resolution of 5123 grid from the original non-watertight mesh, and then apply Marching Cubes (Lorensen & Cline, 1987) to extract the iso-surface

with a small threshold τ = 5123 . To remove interior structure for more efficient geometry learning, we follow (Zhang

et al., 2024b) by resetting the UDF value of the invisible grids to prevent the extraction of interior iso-surface before applying Marching Cubes. We then remove some small and invisible interior mesh components by calculating the area and the ambient occlusion ratio of each mesh component. Finally, we uniformly sample surface points along with their normals, and randomly sample points both within the volume and near the surface.

### 5. Experiments

#### 5.1. Implementation Details

The shape generation experiments are divided into two parts. In the TripoSG experiment, we progressively scaled both resolution and model size. First, we trained a 1.5B parameter model on a 2M dataset with a latent resolution of 512 tokens, using a learning rate of 1e-4 for 700k steps. Next, we switched to a latent resolution of 2048 tokens and continued training for an additional 300k steps with a learning rate of 5e-5. Finally, to scale up, we expanded the model parameters to 4B using MoE and increased the latent resolution to 4096 tokens. Training resumed on 1M high-quality dataset with a learning rate of 1e-5 for 100k steps. The batch size of the three processes is set to 16, 10, and 8 per GPU respectively. The entire training process took approximately

- 3 weeks across 160 A100 GPUs.

In the ablation experiments, we still use a small dataset (180K) filtered from Objaverse and a 975M parameter model for training. For the non-scaling ablation experiments, as shown in Tab.1, we trained the model with a latent resolution of 512 tokens, a learning rate of 1e-4, for about 300k steps, over approximately 3 days on 32 A100 GPUs. For the scaling-up ablation experiments, as shown in rows 2-4 of Tab.2, we progressively continued training from the previous experiment with latent resolutions of 2048 tokens, 4096 tokens, and 4096 tokens with the MoE model architecture, respectively, for an additional 100k steps. The learning rates were 5e-5, 1e-5, and 1e-5, respectively. These three scaling-up experiments took around 9 days in total on 32

A100 GPUs. For all ablation experiments, the batch size was set to 16 per-GPU.

It is also worth mentioning that during training, the image foreground is resized to a fixed ratio (90%) and rotated around the center within a range of [−10◦,10◦] with a probability of 0.2. This setting enables the model to generalize well to various input images. During inference, the image is first detected for the foreground and then resized to the same ratio as the training foreground to obtain the best generation effect.

Following (Zhao et al., 2024), our VAE model adopts a network architecture with an 8-layer encoder and a 16-layer decoder. We use a larger decoder to enhance the ability to decode geometry from the latent space, without increasing the inference cost of the VAE during the flow model training stage. The weights of surface normal loss λsn, eikonal regularization λeik, and KL-regularization λkl are set to 10, 0.1, and 0.001, respectively. For each training data item, our model takes 20,480 surface points as input and randomly samples 8,192 near-surface points, 8,192 volume points, and 8,192 on-surface points for supervision.

The VAE experiments are divided into two parts as well: TripoSG experiments and ablation experiments. In the TripoSG experiment, we train the VAE via SDF supervision with surface normal guidance and eikonal regularization using a learning rate of 5e-5 and a per-GPU batch size of 6 for 2.5M steps. The training process takes approximately 12 days on 32 A100 GPUs, then the VAE is used for the scale-up flow model training. For the ablation experiment, we evaluate the VAE reconstruction quality from different experiment settings on a small dataset (180K filtered data from Objaverse). We train the VAE with different settings using a learning rate of 1e-4 and a per-GPU batch size of 8 for 286K steps on 8 A100 GPUs.

5.2. Dataset, Metrics and Baselines 5.2.1. DATASET

We train our model on Objaverse (-XL) (Deitke et al., 2023; 2024), the largest publicly available 3D dataset, which contains over 10 million unique 3D objects from various sources. Since most of the data in Objaverse(-XL) cannot be directly used for our model’s training, we apply the preprocessing steps introduced in Sec.4 to prepare the training data, including scoring, filtering, orientation fixing, and training data reproducing. After preprocessing, we get 2 million high-quality 3D objects. We compute the ground-truth SDF from sampled points based on the reproduced 3D models. For single-image conditioned flow model training, we render from 8 random viewpoints in front of the 3D models with randomly sampled parameters from the camera’s focal length, elevation, and azimuth ranges after orientation fixing

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

InputMeshFormerMeshLRMCRMInstantMeshTripoSGTripoSRCraftsMan-1.5

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

- Figure 7. Comparison of 3D generation performance of TripoSG and other previous state-of-the-art methods under the same image input.

but before data reproducing. Specifically, the range of elevation is [−15◦,30◦]. The range of azimuth is [0◦,180◦]. The range of focal length is randomly selected from a discrete focal length list. The focal length list is [orthogonal, 50mm, 85mm, 135mm, 2 randomly selected from 35mm-65mm].

- 5.2.2. METRICS

For flow model generation quality, we use FID(Heusel et al., 2017) to evaluate our model performance. The original goal of FID is to evaluate the quality and photorealism of the generated shapes. We aim to introduce FID-related metrics to quantitatively assess the quality of the 3D models

generated by TripoSG. Typically, the input for generating 3D models is a 2D RGB image, while TripoSG primarily generates textureless 3D models, creating an evaluation gap between the two. To bridge this gap and enable evaluation on a consistent semantic level, we propose an improved evaluation process. Specifically, for 3D ground-truth models, we render paired RGB images Igt and normal maps Ngt under the same viewpoints. The RGB images Igt are input into TripoSG to generate 3D shapes, from which we render normal maps Ngen from the same viewpoints as the input images. We then compute the Normal-FID between the generated normal maps Ngen and the ground-truth normal maps Ngt to evaluate the overall performance of TripoSG.

[Figure 89]

- Figure 8. Radar chart of the score of different methods in 5 aspects, including 3D plausibility, text-asset alignment, geometry details, texture details, texture-geometry coherency.

In addition, the existing metrics, even the Normal-FID, are not flexible enough to adapt to various evaluation criteria and may not accurately reflect human preferences. Thus, we also incorporate a recently introduced evaluation metric GPTEval3D (Wu et al., 2024c) for further comparison.

We adopt some commonly used metrics to evaluate VAE reconstruction quality and flow model generation quality. For VAE reconstruction quality, we focus on the accuracy of the reconstruction mesh from input points (with corresponding normals). We use the Chamfer distance, F-score with 0.02 threshold, and the normal consistency as metrics.

#### 5.3. Quantitative and Qualitative Evaluation1

- 5.3.1. COMPARISON WITH DIFFERENT METHODS IN VISUALIZATION

As shown in Fig.7, we compare TripoSG with the most popular image-to-3D generation methods(Li et al., 2024c; Liu et al., 2024a; Tochilkin et al., 2024; Wang et al., 2024; Wei et al., 2024; Xu et al., 2024). It is worth noting that for Craftsman(Li et al., 2024c), we used the online demo of Craftsman-1.5 on Huggingface for inference, which is a more advanced version of the Craftsman. The first row in the figure shows the original input image, while rows 2-7 present a comparison between the generation 3D models of other methods and TripoSG. We compared their geometric quality by rendering normal maps. The results shown in the figure are all 3D normal maps rendered from the same viewpoint. Notably, we preprocessed the original image by removing the background and fed the processed images to different open-source models via Huggingface demos for online inference and generation. Unlike previous works that typically compare 3D generation results on simple, standard images, we conducted comparisons on complex and widely varying cases.

1The original images in Fig.1, Fig.7, Fig.10 and Fig.11, Fig.12 are sourced from various 3D generation platforms, benchmarks (such as Rodin, Meshy, 3D Arena), and our own collections.

Specifically, we evaluated the methods across five dimensions from left to right: (1) Semantic Consistency: TripoSG generates 3D models with better semantic consistency, as shown in the first and second cases, with greater detail and semantic alignment. (2) Detail: The third and fourth cases demonstrate TripoSG’s ability to capture finer details, such as clothing textures and accessories, providing richer visual fidelity. (3) Generalization: The fifth and sixth cases highlight TripoSG’s ability to generate high-quality 3D models from both comic-style and cartoon-style images, showcasing its strong generalization capability. (4) Spatial Structure Generation: The seventh and eighth cases show TripoSG excels at generating complex spatial structures, demonstrating superior spatial modeling capabilities. (5) Overall Performance: We compared TripoSG with the latest and most advanced open-source methods, including both reconstruction and generation approaches, and it is evident that TripoSG delivers significantly superior results, leaving a strong impression and outperforming previous approaches by a wide margin.

- 5.3.2. COMPARISON WITH DIFFERENT METHODS IN METRIC

Benefiting from the development of Large Multimodal Models (LMMs), we can easily leverage them to obtain evaluation results more aligned with human preferences. We adapt the evaluation script and test text prompt from (Wu et al., 2024c) and use the Claude3.5 instead of GPT-4Vision (OpenAI, 2023a) as the Large Multimodal Models (LMMs). We compare our results with various types of previous SOTA methods(Chen et al., 2023; Jun & Nichol, 2023; Li et al., 2023; Lin et al., 2023; Liu et al., 2024b; Long et al., 2024; Metzer et al., 2023; Nichol et al., 2022; Poole et al., 2023; Shi et al., 2024; Tang et al., 2024; Wang et al., 2023a;d). Specifically, we use an off-the-shelf text-to-image model Flux(blackforestlabs, 2024) to generate the input image for our method. Fig.8 demonstrates a Radar chart comparing the evaluation results across five aspects for different methods, as assessed by the LMM model. The result indicates that our TripoSG outperforms the other methods in all aspects.

- 5.3.3. SOTA PERFORMANCE OF TRIPOSG

Fig.1, Fig.11 and Fig.12 showcase various image-to-3D results generated by TripoSG. Notably, there are no duplicates among these cases, and the generated models have not undergone any post-processing (such as smoothing or removing floaters). Textured cases were produced through texture map generation, while non-textured cases were rendered from the original mesh. The process of texture generation is detailed in Sec.6. The first image of each case in Fig.11 and Fig.12 is the input image, and the following four images are multi-view results rendered from the generated 3D model.

Table 1. The ablation for flow model improvements. ‘Skip-C’ is the skip-connection operation and ‘Sample-S’ is the sample schedule.

|CONDITION|SKIP-C|SAMPLE-S|NORMAL-FID ↓|
|---|---|---|---|
|DINOV2<br><br>|✗|R-FLOW<br><br>|10.69|
|CLIP-DINOV2<br><br>|✗<br><br>|R-FLOW|10.61|
| |✓<br><br>|DDPM<br><br>|9.63|
| |✓|EDM|9.50|
| |✓<br><br>|R-FLOW|9.47|

From these results, it is evident that TripoSG delivers outstanding 3D model generation. Across the wide range of showcased cases—covering various complex structures, diverse styles, imaginative designs, multi-object compositions, thin surfaces, and richly detailed scenarios—TripoSG consistently produces impressive 3D models. Achieving this level of performance is challenging for existing methods. The strong generalization highlights the advantages of largescale datasets, while the rich detail and interpretive capability underscore the benefits of our high latent resolution and large model size, collectively reflecting TripoSG’s state-ofthe-art performance.

### 6. Texture Generation

Thanks to the finely detailed and high-quality 3D geometry generated by TripoSG, referring to Meta 3D TextureGen (Bensadoun et al., 2024), we can leverage the rendered normal maps as input conditions for existing mature multiview generation methods to produce consistent multi-view texture images. These multi-view texture images are then projected onto the geometric surface to obtain detailed texture maps. Fig.12 shows the 3D result with texture maps generated by TripoSG.

### 7. Ablation and Analysis

#### 7.1. Ablation for Flow Model

To validate the effectiveness of the proposed flow model improvements and scaling-up strategies, we performed specific ablation experiments and comparative analyses for each improvement. Using a further filtered 180K high-quality Objaverse dataset, we conducted ablation experiments following the training settings in 5.1, and evaluated the results using the Normal-FID metric introduced in 5.2.2.

For the evaluation of Normal-FID, we selected 1K data samples from the 180K dataset as a dedicated test set for 3D generation performance validation, with the remaining data samples used for training. For the test set, we rendered each 3D ground-truth model’s front view paired RGB and normal map using a 50mm camera focal length and a 10◦ elevation (the test set rendering settings are included within the training set settings). The RGB images are used to

Table 2. The ablation for flow model scaling up.

|DATASET|TOKEN NUMBER<br><br>|MOE|NORMAL-FID ↓|
|---|---|---|---|
|OBJAVERSE<br><br>|512|✗<br><br>|9.47|
| |2048<br><br>|✗|8.38|
| |4096|✗<br><br>|8.12|
| |4096|✓<br><br>|7.94|
|TRIPOSG|4096<br><br>|✓|3.36|

generate 3D shapes, and the normal maps are compared with the normal maps rendered from the generated 3D shapes to calculate the Normal-FID.

Our flow model ablation experiment consists of two parts: flow model improvement training and flow model scaling up. For the flow model improvement experiments, as shown in Tab.1, we used a 975M parameter model with a latent resolution of 512 tokens and trained for 300K steps for each experiment on the high-quality Objaverse dataset. We conducted comparative analyses on Condition, Skip − connection, and Sampling − schedule improvements. From the last three rows in Tab.1, we observe that RFlow sampling yields better generation results compared to EDM and DDPM. Combined with its training efficiency, RFlow demonstrates clear advantages in 3D generation tasks.

- Comparing rows 2 and 5 shows that the skip-connection operation significantly affects generation results, with the fusion of deep and shallow features improving flow modeling. Additionally, the comparison between the first two rows indicates that the CLIP condition also slightly improves generation results. From the overall quantitative results, the skip-connection operation has the most obvious effect among these ablations.

For the flow model scaling-up experiments, as shown in rows 2-4 of Tab.2, we used a 975M parameter model with CLIP-DINOv2 dual-conditioning, skip-connection operations, and rectified-flow sampling schedule. These models are trained for a total of 300K steps on the high-quality Objaverse data to conduct comparative analyses on latent resolution and MoE. The last row of Tab.2 represents our largest TripoSG model, encompassing the largest data, model size, resolution, and training cost. From the first three rows of Tab.2 we observe that as the latent resolution increases, the generated results consistently improve, with the most significant improvement occurring from 512 to 2048 tokens.

- Comparing rows 3 and 4 shows the gains from increasing model parameters through MoE. Comparing rows 4 and 5 demonstrates the performance improvement achieved by increasing high-quality data size. When combined with the results from row 1, we can see that the improvement from the increased high-quality data size surpasses that from higher resolution. Overall, the large-scale dataset, large model size, and high resolution contribute to significant performance improvements, allowing TripoSG to achieve remarkable 3D generation results.

Table 3. The ablation of different VAE, including 3D representation, training supervision and training dataset. ‘Repr’ refers to the type of 3D representation used, and ‘Lsn’ and ‘Leik’ refer to surface normal loss and eikonal regularization respectively. The ‘Dataset’ indicates whether a large (TripoSG) or a small dataset (Objaverse) is used.

|DATASET<br><br>|REPR. LSN Leik<br><br>|CHAMFER ↓ F-SCORE ↑ N.C. ↑|
|---|---|---|
|OBJAVERSE|OCC ✗ ✗<br><br>|4.59 0.999 0.952|
| |SDF ✗ ✗<br><br>|4.60 0.999 0.955|
| |SDF ✓ ✗|4.56 0.999 0.956|
| |SDF ✓ ✓<br><br>|4.57 0.999 0.957|
|TRIPOSG<br><br>|SDF ✓ ✓<br><br>|4.51 0.999 0.958|

Occupancy SDF SDF+Normal SDF+Normal+Eikonal

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

| |
|---|

- Figure 9. Qualitative comparison for the ablation of VAE with different types of 3D representation and training supervision.

#### 7.2. Ablation for VAE

To evaluate the effectiveness of neural SDF implicit representation with surface normal guidance, we experiment with different VAE model settings, including the formulation of neural implicit representation, training supervision, and training dataset. Tab.3 demonstrates the qualitative results of VAE reconstruction quality with different training settings. We can observe that the SDF representation, combined with surface normal guidance and eikonal regularization, improves the reconstruction quality and geometry details, achieving lower Chamfer distance and higher normal consistency compared to occupancy-based results. As the amount of training data increases (as demonstrated by TripoSG), the reconstruction quality of the VAE further improves. Fig.9 provides qualitative comparisons between them. Occupancy-based reconstruction results suffer from aliasing artifacts (highlighted by the blue box), thin structures, and floaters (highlighted by the red boxes). While SDF representation avoids aliasing artifacts, there remains a gap in achieving high-quality reconstruction, particularly for thin-shell structures where performance may worsen. Incorporating surface normal guidance can result in sharper reconstructions with finer details. However, over-emphasizing surface normal guidance during training introduces slight aliasing artifacts (as seen in the first row of Fig.9), which

Table 4. The ablation for data quality and quantity.

|DATASET|SIZE<br><br>|DATA-BUILDING SYSTEM<br><br>|NORMAL-FID ↓|
|---|---|---|---|
|OBJAVERSE|800K|✗<br><br>|11.61|
| |180K<br><br>|✓<br><br>|9.47|
|TRIPOSG|2M<br><br>|✓|5.81|

can be mitigated by introducing eikonal regularization.

#### 7.3. Ablation for Data-Building System

To demonstrate the effectiveness of the data-building system proposed by TripoSG, we implement ablation experiments on both data quality and quantity. Using the optimal R-Flow training settings (first row of Tab.1), we replaced the 180K Objaverse dataset produced by TripoSG with the original 800K Objaverse dataset, which had not undergone scoring, filtering, orientation fixing, untextured model processing, or internal processing of the converted watertight model. This experiment demonstrates the effect of data quality. Similarly, under the same R-Flow settings, we expanded the highquality dataset from 180K Objaverse to 2M TripoSG to evaluate the effect of data quantity.

As shown in the first two rows of Tab.4, although our databuilding system reduced the 800K Objaverse dataset to 180K, the improved data quality resulted in better generation results, demonstrating that, when training with in-thewild data, quality outweighs quantity. Furthermore, as seen in last two rows of Tab.4, increasing the high-quality dataset from 180K to 2M led to a significant boost in generation performance, showing that with high-quality data, scaling up data size is crucial for achieving better results. Additionally, the overall quantitative results in the Tab.4 show that the performance improvement gained from 2M high-quality data size is greater than that from improving data quality alone. Furthermore, after enhancing data quality, performance continues to improve with an increase in data size, without encountering a bottleneck at the current training scale.

#### 7.4. The Visualization for Flow Model Ablation

In addition to the quantitative results, we also performed a visualization analysis of the core experiments, as shown in Fig.10. The rows 1, 2, 4 correspond to the three experimental results in Tab.4, while the row 3 corresponds to the row 4 of results in Tab.2. The visualization reveals several insights consistent with the quantitative results: (1) Data quality is more important than the size of raw in-the-wild data (row 1 vs. row 2). (2) The improvement from increasing highquality data size is more obvious than the improvement from resolution (row 2 vs. row 3 vs. row 4). (3) Increasing the size of high-quality data (2M) provides a greater boost to performance than merely improving data quality. After improving data quality, performance continues to improve

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Input

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

TripoSGw/oDBS

512Token TripoSGw/DBS

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

2MData512Token TripoSGw/DBS

512Token TripoSGw/DBS

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

4096Token

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

- Figure 10. Visualization results of the flow model ablation experiments. ‘DBS’ is the abbreviation for Data-Building System. with increased data size without encountering bottlenecks

models as priors for 3D generation via the SDS solution, such as DreamFusion(Poole et al., 2022), and leveraging decoder-only transformer architectures to reconstruct 3D models from single or multiple views, such as LRM(Hong et al., 2023). However, due to the scarcity of large-scale datasets and limited experience in scaling up training for 3D generation tasks, the large-scale flow models, which have proven highly successful in 2D image and video generation, have not been widely applied to 3D generation. TripoSG has deeply explored the 3D flow route from the perspective of data and training, successfully achieving 3D generation with strong generalization, exceptional detail, and high fidelity. It has effectively replicated the success of image and video generation architectures in the field of 3D generation. Through TripoSG, 3D generation now aligns with image and video generation in terms of architecture and development stage, allowing the field of 3D generation to draw upon the wealth of architectures and training experience from 2D image and video generation.

- at the current training scale.

### 8. Conclusion and Discussion

#### 8.1. Conclusion

We present TripoSG, a new image-to-3D generation model via the rectified-flow-based transformer. To efficiently train the model for high-fidelity shape generation, we propose a data-building system to process data from original datasets. Compared to using all in-the-wild 3D models in the training dataset, filtered and fixed high-quality data can be properly reproduced into training data and effectively improve the model’s training performance. Additionally, we leverage the advance of SDF representation with surface normal guidance and eikonal regularization for finer geometry details and avoid aliasing artifacts. Furthermore, a rectified-flowbased transformer with MoE and a high-resolution strategy is introduced for the scale-up training. Experiments demonstrate that TripoSG can generate high-fidelity 3D shapes, leading to a new state-of-the-art performance.

Looking ahead, we can further scale up model parameters and training data, and employ more fine-grained conditional information injection methods to generate even more detailed 3D models. Additionally, based on the TripoSG foundation, we can also explore tasks such as 3D model super-resolution, scene generation, and stylization.

#### 8.2. Discussion

In recent years, 3D generation has followed a unique exploration route, with methods such as using text-to-image

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

###### Figure 11. A diverse array of texture-free 3D shapes generated by TripoSG.

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

###### Figure 12. A diverse array of textured 3D shapes generated by TripoSG.

### References

on Computer Vision and Pattern Recognition, pp. 13142– 13153, 2023.

Bain, M., Nagrani, A., Varol, G., and Zisserman, A. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1728–1738, 2021.

Deitke, M., Liu, R., Wallingford, M., Ngo, H., Michel, O., Kusupati, A., Fan, A., Laforte, C., Voleti, V., Gadre, S. Y., et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36, 2024.

Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., and Zhu, J. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22669–22679, 2023.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Bensadoun, R., Kleiman, Y., Azuri, I., Harosh, O., Vedaldi, A., Neverova, N., and Gafni, O. Meta 3d texturegen: fast and consistent texture generation for 3d objects. arXiv preprint arXiv:2407.02430, 2024.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

blackforestlabs. Flux. https://github.com/black-forestlabs/flux, 2024.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., and Ramesh, A. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Fan, H., Su, H., and Guibas, L. J. A point set generation network for 3d object reconstruction from a single image. In IEEE/CVF CVPR, pp. 2463–2471, 2017.

Fei, Z., Fan, M., Yu, C., Li, D., and Huang, J. Scaling diffusion transformers to 16 billion parameters. arXiv preprint arXiv:2407.11633, 2024.

Chan, E. R., Nagano, K., Chan, M. A., Bergman, A. W., Park, J. J., Levy, A., Aittala, M., Mello, S. D., Karras, T., and Wetzstein, G. Generative novel view synthesis with 3d-aware diffusion models. In IEEE/CVF ICCV, pp. 4194–4206, 2023.

Girdhar, R., Fouhey, D. F., Rodriguez, M., and Gupta, A. Learning a predictable and generative vector representation for objects. In Leibe, B., Matas, J., Sebe, N., and Welling, M. (eds.), ECCV, volume 9910, pp. 484–499, 2016.

Chang, A. X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Chen, R., Chen, Y., Jiao, N., and Jia, K. Fantasia3d: Disentangling geometry and appearance for high-quality textto-3d content creation. In IEEE/CVF ICCV, pp. 22189– 22199, 2023.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), NeurIPS, 2020.

Cheng, Y., Lee, H., Tulyakov, S., Schwing, A. G., and Gui, L. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In IEEE/CVF CVPR, pp. 4456–4465, 2023.

Hong, F., Tang, J., Cao, Z., Shi, M., Wu, T., Chen, Z., Wang, T., Pan, L., Lin, D., and Liu, Z. 3dtopia: Large text-to-3d generation model with hybrid diffusion priors. CoRR, abs/2403.02234, 2024.

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos, R., Alabdulmohsin, I., et al. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, pp. 7480–7512. PMLR, 2023.

Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., and Tan, H. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023.

Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., and Farhadi, A. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference

Huang, J., Su, H., and Guibas, L. J. Robust watertight manifold surface generation method for shapenet models. CoRR, abs/1802.01698, 2018.

Huang, J., Zhou, Y., and Guibas, L. J. Manifoldplus: A robust and scalable watertight manifold surface generation method for triangle soups. CoRR, abs/2005.11621, 2020.

Hui, K., Li, R., Hu, J., and Fu, C. Neural wavelet-domain diffusion for 3d shape generation. In Jung, S. K., Lee, J., and Bargteil, A. W. (eds.), SIGGRAPH Asia, pp. 24:1– 24:9. ACM, 2022.

Jun, H. and Nichol, A. Shap-e: Generating conditional 3d implicit functions. CoRR, abs/2305.02463, 2023.

Lan, Y., Hong, F., Yang, S., Zhou, S., Meng, X., Dai, B., Pan, X., and Loy, C. C. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation. In ECCV, 2024.

Li, J., Tan, H., Zhang, K., Xu, Z., Luan, F., Xu, Y., Hong, Y., Sunkavalli, K., Shakhnarovich, G., and Bi, S. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214, 2023.

Li, M., Yang, T., Kuang, H., Wu, J., Wang, Z., Xiao, X., and Chen, C. Controlnet++: Improving conditional controls with efficient consistency feedback. arXiv preprint arXiv:2404.07987, 2024a.

Li, P., Liu, Y., Long, X., Zhang, F., Lin, C., Li, M., Qi, X., Zhang, S., Luo, W., Tan, P., Wang, W., Liu, Q., and Guo, Y. Era3d: High-resolution multiview diffusion using efficient row-wise attention. CoRR, abs/2405.11616, 2024b.

Li, W., Liu, J., Chen, R., Liang, Y., Chen, X., Tan, P., and Long, X. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024c.

Liang, Y., Yang, X., Lin, J., Li, H., Xu, X., and Chen, Y. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In IEEE/CVF CVPR, pp. 6517–6526, 2024.

Lin, C., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M., and Lin, T. Magic3d: Highresolution text-to-3d content creation. In IEEE CVPR, pp. 300–309, 2023.

Liu, M., Xu, C., Jin, H., Chen, L., T., M. V., Xu, Z., and Su, H. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. In NeurIPS,

- 2023a.

Liu, M., Zeng, C., Wei, X., Shi, R., Chen, L., Xu, C., Zhang, M., Wang, Z., Zhang, X., Liu, I., Wu, H., and Su, H. Meshformer: High-quality mesh generation with 3d-guided reconstruction model. CoRR, abs/2408.10198,

- 2024a.

Liu, R., Wu, R., Hoorick, B. V., Tokmakov, P., Zakharov, S., and Vondrick, C. Zero-1-to-3: Zero-shot one image to 3d object. In IEEE/CVF ICCV, pp. 9264–9275, 2023b.

- Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023c.
- Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., and Wang, W. Syncdreamer: Generating multiview-consistent images from a single-view image. In ICLR, 2024b.
- Liu, Z., Li, Y., Lin, Y., Yu, X., Peng, S., Cao, Y.-P., Qi, X., Huang, X., Liang, D., and Ouyang, W. Unidream: Unifying diffusion priors for relightable text-to-3d generation. arXiv preprint arXiv:2312.08754, 2023d.

Long, X., Guo, Y.-C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.-H., Habermann, M., Theobalt, C., et al. Wonder3d: Single image to 3d using cross-domain diffusion. In IEEE/CVF CVPR, pp. 9970–9980, 2024.

Lorensen, W. E. and Cline, H. E. Marching cubes: A high resolution 3d surface construction algorithm. In Stone, M. C. (ed.), Proceedings of the SIGGRAPH, pp. 163–169, 1987.

Melas-Kyriazi, L., Rupprecht, C., and Vedaldi, A. Pc2: Projection-conditioned point cloud diffusion for singleimage 3d reconstruction. In IEEE/CVF CVPR, pp. 12923– 12932, 2023.

Mescheder, L. M., Oechsle, M., Niemeyer, M., Nowozin, S., and Geiger, A. Occupancy networks: Learning 3d reconstruction in function space. In IEEE/CVF CVPR, pp. 4460–4470, 2019.

Metzer, G., Richardson, E., Patashnik, O., Giryes, R., and Cohen-Or, D. Latent-nerf for shape-guided generation of 3d shapes and textures. In IEEE/CVF CVPR, pp. 12663– 12673, 2023.

Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. Nerf: Representing scenes as neural radiance fields for view synthesis. In Vedaldi, A., Bischof, H., Brox, T., and Frahm, J. (eds.), ECCV, volume 12346, pp. 405–421, 2020.

M¨uller, N., Siddiqui, Y., Porzi, L., Bul`o, S. R., Kontschieder, P., and Nießner, M. Diffrf: Rendering-guided 3d radiance field diffusion. In IEEE/CVF CVPR, pp. 4328–4338, 2023.

Newcombe, R. A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A. J., Kohli, P., Shotton, J., Hodges, S., and Fitzgibbon, A. W. Kinectfusion: Real-time dense surface mapping and tracking. In IEEE ISMAR, pp. 127– 136, 2011.

Nichol, A., Jun, H., Dhariwal, P., Mishkin, P., and Chen, M. Point-e: A system for generating 3d point clouds from complex prompts. CoRR, abs/2212.08751, 2022.

OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023a.

OpenAI. Gpt-4v(ision) system card. https: //cdn.openai.com/papers/GPTV_System_ Card.pdf, 2023b. Accessed: 2023-09-25.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., ElNouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-toimage generation. In ICML, volume 139, pp. 8821–8831, 2021.

Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., and Houlsby, N. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34: 8583–8595, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35: 25278–25294, 2022.

Shi, R., Chen, H., Zhang, Z., Liu, M., Xu, C., Wei, X., Chen, L., Zeng, C., and Su, H. Zero123++: a single image to consistent multi-view diffusion base model. CoRR, abs/2310.15110, 2023a.

Shi, Y., Wang, P., Ye, J., Long, M., Li, K., and Yang, X. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023b.

Shi, Y., Wang, P., Ye, J., Mai, L., Li, K., and Yang, X. Mvdream: Multi-view diffusion for 3d generation. In ICLR, 2024.

Shue, J. R., Chan, E. R., Po, R., Ankner, Z., Wu, J., and Wetzstein, G. 3d neural field generation using triplane diffusion. In IEEE/CVF CVPR, pp. 20875–20886, 2023.

Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., Parikh, D., Gupta, S., and Taigman, Y. Make-a-video: Text-to-video generation without text-video data. In ICLR, 2023.

Tang, J., Wang, T., Zhang, B., Zhang, T., Yi, R., Ma, L., and Chen, D. Make-it-3d: High-fidelity 3d creation from A single image with diffusion prior. In IEEE ICCV, pp. 22762–22772, 2023.

Tang, J., Ren, J., Zhou, H., Liu, Z., and Zeng, G. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In ICLR, 2024.

team at Meta, T. M. G. Movie gen: A cast of media foundation models. 2024.

Tochilkin, D., Pankratz, D., Liu, Z., Huang, Z., Letts, A., Li, Y., Liang, D., Laforte, C., Jampani, V., and Cao, Y.P. Triposr: Fast 3d object reconstruction from a single image. arXiv preprint arXiv:2403.02151, 2024.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In NeurIPS, pp. 5998–6008, 2017.

von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., Nair, D., Paul, S., Berman, W., Xu, Y., Liu, S., and Wolf, T. Diffusers: State-of-the-art diffusion models. https://github.

com/huggingface/diffusers, 2022.

Wang, H., Du, X., Li, J., Yeh, R. A., and Shakhnarovich, G. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In IEEE CVPR, pp. 12619– 12629, 2023a.

Wang, N., Zhang, Y., Li, Z., Fu, Y., Liu, W., and Jiang, Y. Pixel2mesh: Generating 3d mesh models from single RGB images. In Ferrari, V., Hebert, M., Sminchisescu, C., and Weiss, Y. (eds.), ECCV, volume 11215, pp. 55–71, 2018.

Wang, P. and Shi, Y. Imagedream: Image-prompt multiview diffusion for 3d generation. CoRR, abs/2312.02201, 2023a.

Wang, P. and Shi, Y. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023b.

Wang, P., Liu, Y., and Tong, X. Dual octree graph networks for learning adaptive volumetric shape representations. ACM Transactions on Graphics (TOG), 41(4): 103:1–103:15, 2022.

Wang, P., Tan, H., Bi, S., Xu, Y., Luan, F., Sunkavalli, K., Wang, W., Xu, Z., and Zhang, K. Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction. arXiv preprint arXiv:2311.12024, 2023b.

- Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., et al. Internvid: A largescale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023c.
- Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu, J. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS, 2023d.

Wang, Z., Wang, Y., Chen, Y., Xiang, C., Chen, S., Yu, D., Li, C., Su, H., and Zhu, J. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034, 2024.

Wei, X., Zhang, K., Bi, S., Tan, H., Luan, F., Deschaintre, V., Sunkavalli, K., Su, H., and Xu, Z. Meshlrm: Large reconstruction model for high-quality mesh. arXiv preprint arXiv:2404.12385, 2024.

Worchel, M., Diaz, R., Hu, W., Schreer, O., Feldmann, I., and Eisert, P. Multi-view mesh reconstruction with neural deferred shading. In IEEE/CVF CVPR, pp. 6177–6187, 2022.

Wu, J., Wang, Y., Xue, T., Sun, X., Freeman, B., and Tenenbaum, J. Marrnet: 3d shape reconstruction via 2.5d sketches. In NeurIPS, pp. 540–550, 2017.

- Wu, J. Z., Ge, Y., Wang, X., Lei, S. W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., and Shou, M. Z. Tune-a-video: One-shot tuning of image diffusion models for text-tovideo generation. In IEEE/CVF ICCV, pp. 7589–7599,

- 2023.

Wu, K., Liu, F., Cai, Z., Yan, R., Wang, H., Hu, Y., Duan, Y., and Ma, K. Unique3d: High-quality and efficient 3d mesh generation from a single image. CoRR, abs/2405.20343,

- 2024a.

- Wu, R., Zhuang, Y., Xu, K., Zhang, H., and Chen, B. PQNET: A generative part seq2seq network for 3d shapes. In IEEE/CVF CVPR, pp. 826–835, 2020.
- Wu, S., Lin, Y., Zhang, F., Zeng, Y., Xu, J., Torr, P., Cao, X., and Yao, Y. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024b.
- Wu, T., Yang, G., Li, Z., Zhang, K., Liu, Z., Guibas, L. J., Lin, D., and Wetzstein, G. Gpt-4v(ision) is a humanaligned evaluator for text-to-3d generation. In IEEE/CVF CVPR, pp. 22227–22238, 2024c.

Xu, J., Cheng, W., Gao, Y., Wang, X., Gao, S., and Shan, Y. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024.

Xu, Q., Wang, W., Ceylan, D., Mech, R., and Neumann, U. DISN: deep implicit surface network for high-quality single-view 3d reconstruction. In NeurIPS, pp. 490–500, 2019.

Xu, Y., Tan, H., Luan, F., Bi, S., Wang, P., Li, J., Shi, Z., Sunkavalli, K., Wetzstein, G., Xu, Z., et al. Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. arXiv preprint arXiv:2311.09217, 2023.

Yi, T., Fang, J., Wang, J., Wu, G., Xie, L., Zhang, X., Liu, W., Tian, Q., and Wang, X. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In IEEE CVPR, pp. 6796–6807, 2024.

Yu, A., Ye, V., Tancik, M., and Kanazawa, A. pixelnerf: Neural radiance fields from one or few images. In IEEE/CVF CVPR, pp. 4578–4587, 2021.

Zeiler, M. D. and Fergus, R. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13, pp. 818– 833. Springer, 2014.

Zeng, X., Vahdat, A., Williams, F., Gojcic, Z., Litany, O., Fidler, S., and Kreis, K. LION: latent point diffusion models for 3d shape generation. In NeurIPS, 2022.

Zhang, B. and Sennrich, R. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

Zhang, B., Tang, J., Niessner, M., and Wonka, P. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–16, 2023.

- Zhang, K., Bi, S., Tan, H., Xiangli, Y., Zhao, N., Sunkavalli, K., and Xu, Z. Gs-lrm: Large reconstruction model for 3d gaussian splatting. arXiv preprint arXiv:2404.19702, 2024a.
- Zhang, L., Wang, Z., Zhang, Q., Qiu, Q., Pang, A., Jiang, H., Yang, W., Xu, L., and Yu, J. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4): 1–20, 2024b.

Zhao, Z., Liu, W., Chen, X., Zeng, X., Wang, R., Cheng, P., Fu, B., Chen, T., Yu, G., and Gao, S. Michelangelo: Conditional 3d shape generation based on shape-imagetext aligned latent representation. Advances in Neural Information Processing Systems, 36, 2024.

Zheng, X., Pan, H., Wang, P., Tong, X., Liu, Y., and Shum, H. Locally attentional SDF diffusion for controllable 3d shape generation. ACM Trans. Graph., 42(4):91:1–91:13, 2023.

Zhou, L., Du, Y., and Wu, J. 3d shape generation and completion through point-voxel diffusion. In IEEE/CVF ICCV, pp. 5806–5815, 2021.

Zou, Z., Cheng, W., Cao, Y., Huang, S., Shan, Y., and Zhang, S. Sparse3d: Distilling multiview-consistent diffusion for object reconstruction from sparse views. In AAAI, pp. 7900–7908, 2024a.

Zou, Z.-X., Yu, Z., Guo, Y.-C., Li, Y., Liang, D., Cao, Y.P., and Zhang, S.-H. Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10324–10335, 2024b.

