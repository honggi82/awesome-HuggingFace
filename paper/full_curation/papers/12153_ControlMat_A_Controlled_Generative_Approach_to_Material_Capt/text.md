# arXiv:2309.01700v3[cs.CV]27Jul2024

## ControlMat: A Controlled Generative Approach to Material Capture

GIUSEPPE VECCHIO, Adobe Research, France ROSALIE MARTIN, Adobe Research, France ARTHUR ROULLIER, Adobe Research, France ADRIEN KAISER, Adobe Research, France ROMAIN ROUFFET, Adobe Research, France VALENTIN DESCHAINTRE, Adobe Research, UK TAMY BOUBEKEUR, Adobe Research, France

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

Render

ControlMat

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

Render

ControlMat

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

Render

ControlMat

[Figure 43]

[Figure 44]

[Figure 45]

Input Photo Rendering Visualization on Mesh

Estimated Maps

Fig. 1. We present ControlMat, a diffusion based material generation model conditioned on input photographs. Our approach enables high-resolution, tileable material generation and estimation from a single naturally or flash lit image, inferring both diffuse (Basecolor) and specular (Roughness, Metallic) properties as well as the material mesostructure (Height, Normal, Opacity).

Material reconstruction from a photograph is a key component of 3D content creation democratization. We propose to formulate this ill-posed problem as a controlled synthesis one, leveraging the recent progress in generative deep networks. We present ControlMat, a method which, given a single photograph with uncontrolled illumination as input, conditions a diffusion model to generate plausible, tileable, high-resolution physically-based digital materials. We carefully analyze the behavior of diffusion models for multichannel outputs, adapt the sampling process to fuse multi-scale information and introduce rolled diffusion to enable both tileability and patched diffusion for high-resolution outputs. Our generative approach further permits exploration of a variety of materials which could correspond to the input image, mitigating the unknown lighting conditions. We show that our approach outperforms recent inference and latent-space-optimization methods, and carefully validate our diffusion process design choices. Supplemental materials and additional details are available at: https://gvecchio.com/controlmat/.

CCS Concepts: • Computing methodologies → Appearance and texture representations.

Authors’ addresses: Giuseppe Vecchio, Adobe Research, France, gvecchio@adobe. com; Rosalie Martin, Adobe Research, France, rmartin@adobe.com; Arthur Roullier, Adobe Research, France, roullier@adobe.com; Adrien Kaiser, Adobe Research, France, akaiser@adobe.com; Romain Rouffet, Adobe Research, France, rouffet@adobe.com; Valentin Deschaintre, Adobe Research, UK, deschain@adobe.com; Tamy Boubekeur, Adobe Research, France, boubek@adobe.com.

Additional Key Words and Phrases: material appearance, capture, generative models

1 INTRODUCTION

Materials are at the core of computer graphics. Their creation, however, remains challenging with complex tools, requiring significant expertise. To facilitate this process, material acquisition has been a long-standing challenge, with rapid progress in recent years, leveraging massively machine learning for lightweight acquisition [Deschaintre et al. 2018; Guo et al. 2021; Vecchio et al. 2021]. However, many of these methods focused on the use of a single flash image, leading to typical highlight-related artifacts and limiting the range of acquisition [Deschaintre et al. 2020]. Another strategy has been to trade acquisition accuracy for result quality with environment lit images as input [Li et al. 2017; Martin et al. 2022]. We follow this strategy and propose to leverage the recent progress in diffusion models [Dhariwal and Nichol 2021; Ho et al. 2020; Rombach et al. 2022] to build an image-conditioned material generator. We design our method with two key properties of modern graphics pipelines in mind. First, we ensure tileability of the generated output, for both

unconditional and conditional generation. Second, we enable highresolution generation, allowing artists to directly use our results in high quality renderings.

Generative models have previously been used in the context of materials [Guo et al. 2020; Zhou et al. 2022], but relied on Generative Adversarial Networks (GANs) [Goodfellow et al. 2014] and optimization in their latent space, usually using multiple inputs for material acquisition. While GANs achieved impressive quality on structured domains such as human faces [Karras et al. 2020], domains with a wider variety of appearances, e.g., materials and textures, remain challenging to train for at high resolution and quality. This is due to a high memory consumption, but also to the inherent instability during the min-max training, leading to mode collapse. We choose to leverage diffusion models [Dhariwal and Nichol 2021; Ho et al. 2020; Rombach et al. 2022] which were recently proved to be stable, scalable generative models for different modalities (e.g. images [Ramesh et al. 2022; Saharia et al. 2022], music [Huang et al. 2023]). Further, as opposed to GANs which rely on latent space projection/optimization for inversion, Diffusion Models can be conditioned during the inference "denoising" process. Therefore, we adapt the architecture to material generation and evaluation and improve the sampling process to enable tileability and patch-based diffusion for high resolution.

Our backbone network, MatGen, is a latent diffusion model [Rombach et al. 2022] that we train to output 9 Spatially-Varying Bidirectional Reflectance Distribution Function (SVBRDF) channels (basecolor (3), normal (2), height (1), roughness (1), metalness (1), opacity (1)). Latent diffusion models generate results by "denoising" a latent space representation, which is then decoded by a jointly trained Variational Auto Encoder (VAE) [Kingma and Welling 2013]. We train this model for both unconditional and CLIP [Radford et al. 2021] conditioned generation, and train a separate ControlNet [Zhang et al. 2023] to condition the generation on material photograph. We train these models using the Substance 3D Materials database [Adobe 2022], generating for this purpose 10, 000, 000 renderings and corresponding ground truth materials. Once trained, if sampled naively, these models lead to non-tileable and limited resolution materials. We propose various modifications of the generation process – noise rolling, patched and multi-scale diffusion, patched decoding and border inpainting – to enable tileability and arbitrary resolution while preserving the generated materials quality.

We evaluate our method qualitatively and quantitatively against previous work [Martin et al. 2022; Vecchio et al. 2021; Zhou et al. 2022] and carefully analyze our diffusion process, demonstrating the benefit of each introduced components through ablation studies. In summary, we propose a method for material reproduction from a single photograph under uncontrolled lighting, through a conditional diffusion model adapted to the domain of material generation. This is enabled by the following contributions:

- • ControlMat, a single image SVBRDF estimator which guides a latent diffusion model (MatGen) by means of ControlNet.
- • Patchedandmulti-scalediffusion process for high-resolution, high-quality material generation.
- • Noise rolling diffusion and inpainting for the generation of tileable materials.

2 RELATED WORK

We discuss methods related to material capture and generation, as well as different generative models options. In particular, we focus on the recent progress relying on Neural Networks. For a thorough overview of pre-learning material modeling and acquisition methods, we recommend the survey by Guarnera et al. [2016].

2.1 Material capture & estimation

Material capture aims to recover a virtual representation of an existing material. In recent years, the field has focused on lightweight acquisition methods, leveraging neural network to build priors to resolve the inherent ambiguity when capturing a material from few photographs, unknown illumination, or both.

Flash-based. Single flash image material acquisition leveraging deep network has seen significant progress, leveraging the U-Net architecture [Ronneberger et al. 2015]. A first method was proposed by Deschaintre et al. [2018] and further improved to reduce the over-exposed flash artefacts using GANs [Vecchio et al. 2021; Zhou and Kalantari 2021], highlight aware training [Guo et al. 2021], meta learning [Fischer and Ritschel 2022; Zhou and Kalantari 2022], semiprocedural priors [Zhou et al. 2023b] or cross-scale attention [Guo et al. 2023]. Some methods focused on stationary material recovery, leveraging self similarity of the material, trading spatial resolution for angular resolution [Aittala et al. 2016, 2015; Henzler et al. 2021]. Other approaches for spatially varying materials proposed to combine multiple flash images with direct inference [Deschaintre et al.

- 2019] or using refinement optimisation [Gao et al. 2019] to improve quality, or even flash and non flash photographs [Deschaintre et al.
- 2020]. Closer to our approach are methods using generative models to build a latent space prior in which a latent vector can be optimized to reproduce an input picture [Guo et al. 2020]. Zhou et al. [2022] further modified the generative architecture to enforce tileability of outputs through circular padding and tensor rolling and allow for structurally conditioned generation with grayscale patterns. More recently, a self-supervised approach leveraging a GAN training proposed to train a material generator solely on flash images without material ground truth supervision [Zhou et al. 2023a].

Unknown illumination. While flash illumination provides important information about the acquired material specularity, it also leads to challenges: the over-exposure artefacts we mentioned earlier, limiting the acquisition scale [Deschaintre et al. 2020] and requirement to capture the material in a lightly controlled setup, limiting their use with existing online resources. An alternative approach is to estimate the properties of a material from unknown environment illumination as proposed by Li et al. [2017] and improved by Martin et al. [2022]. These methods however do not generate spatially-varying roughness properties, or approximate it from the mesostructure, and specialise on more diffuse materials (e.g "outdoor categories").

Inverse procedural Material. A recent line of work proposes to leverage procedural representation to represent materials from photographs. Hu et al. [2019; 2022a] and MATch [Shi et al. 2020] propose to use existing material graphs and to infer or optimise their parameters to match a given target image. The challenge of creating

[Figure 46]

|[Figure 47]|
|---|

Input photo + inpaint mask

Inference

ControlNet

LDM

[Figure 48]

[Figure 49]

Gaussian noise

| | |
|---|---|
| | |

[Figure 50]

[Figure 51]

UNet

Training

Q KV

Q KV

Q KV

Q KV

Noise

Noise

[Figure 52]

|rolling| | | |unrolling|
|---|---|---|---|---|
| | | | | |

[Figure 53]

Predicted

[Figure 54]

[Figure 55]

Patched material maps decoding

Text / image prompt

Ground truth material maps

"Alienground"rocky CLIP

MatGen

Fig. 2. Overview of ControlMat. During training, the PBR maps are compressed into the latent representation 𝑧 using the encoder E. Noise is then added to 𝑧 and the denoising is carried out by a U-Net model. The denoising process can be globally conditioned with the CLIP embedding of the prompt (text or image) and/or locally conditioned using the intermediate representation of a target photograph extracted by a ControlNet network. After 𝑛 denoising steps the new denoised latent vector 𝑧ˆ is projected back to pixel space using the decoder D. We enable high resolution diffusion through splitting the input image in N patches which are then diffused, decoded and reassembled through patched decoding.

procedural models has been tackled by selecting components for a generic graph architecture [Hu et al. 2022b] or through graph generation based on a combination of Transformer models [Guerrero et al. 2022; Hu et al. 2023]. While the procedural representation has interesting benefits, these methods target the recovery of the appearance style, rather than the exact material, as they rely on procedural functions. Further, many of these methods rely on test time access to a pre-existing large library of material graphs.

In this work, we explore the combination of generative models with unknown illumination, leveraging the recent progress in diffusion modeling [Dhariwal and Nichol 2021; Ho et al. 2020; Rombach et al. 2022] and their conditioning [Zhang et al. 2023]. Doing so, we propose an uncontrolled lighting photograph to material method with tileable, high-resolution results, reproducing the input photograph layout, and estimating specular properties.

2.2 Generative models

Image generation is an open challenge in computer vision due to the complexity of modeling high-dimensional data distributions. While Generative Adversarial Networks (GANs) [Goodfellow et al. 2014] have shown great results in generating high-quality images [Brock et al. 2018; Karras et al. 2017, 2020], their training process is plagued by unstable convergence and mode collapse behavior [Arjovsky et al. 2017; Gulrajani et al. 2017; Mescheder 2018; Metz et al. 2016]. Some recent approaches have tackled the problem by decoupling decoding and distribution learning into a two-stage approach [Dai and Wipf 2019; Esser et al. 2021; Rombach et al. 2020a,b]. In this case a Variational Autoencoder (VAE) [Kingma and Welling 2013; Rezende et al. 2014; Van Den Oord et al. 2017] is first trained to learn a data representation, then another network learns its distribution.

Recently, a new family of generative models, the Diffusion Models (DMs), have emerged as a promising alternative to GANs. These models allow to learn complex distributions and generate diverse, high quality images [Dhariwal and Nichol 2021; Ho et al. 2020; SohlDickstein et al. 2015], leveraging a more stable training process. However, these models are computationally expensive to evaluate and optimize, particularly for high-resolution images. To address this issue, Latent Diffusion Model [Rombach et al. 2022] propose to diffuse in a smaller latent space. This space is learned through pre-training an image compression model such as a Variational Autoencoder (VAE). By using the same diffusion model architecture, this shift to the latent space reduces computational requirements and improves inference times, without significantly compromising generation quality. Recently, different methods [Bar-Tal et al. 2023; Jiménez 2023] have been proposed to extend the generative capabilities of LDMs by enforcing consistency between multiple parallel diffusion processes. This enables a fine-grained control over the localization of content inside the generated image, while scaling to higher resolution, and non-square aspect ratio through latent vector patching and independent processing. Furthermore, diffusion in the latent space enabled novel conditioning mechanisms for the generative process. These conditioning mechanisms are, however, bound to the training of the diffusion model, requiring a new training for each type of conditioning. To address this limitation, ControlNet [Zhang et al. 2023] was proposed as a way to control pre-trained large diffusion models, through a relatively small, task-specific network.

We leverage this diffusion-specific progress and adapt the diffusion model and inference process to the material generation and acquisition domain, leveraging ControlNet for its conditioning capabilities. Recent concurrent work also explores the use of diffusion

models for material generation. Vecchio et al. [2024] introduces MatFuse, which focuses on multi-conditional generation and editing. Similarly, Yuan et al. [2024] proposes DiffMat, which applies the diffusion process to the latent vectors of a pretrained StyleGAN generator. MatFusion [Sartor and Peers 2023] uses a diffusion-based approach, applied directly in pixel space, for material capture. However, these approaches do not enable high-resolution nor tileable material estimation/generation.

### 3 OVERVIEW

ControlMat, summarized in Fig. 2, is a generative method based on a Latent Diffusion Model (LDM) [Rombach et al. 2022] and designed to produce high-quality Spatially Varying Bidirectional Reflectance Distribution Functions (SVBRDFs) material maps from an input image. The method’s generative backbone, MatGen, is made of a variational autoencoder (VAE) [Kingma and Welling 2013] network trained to represent SVBRDF maps in a compact latent space, and a diffusion model trained to sample from this latent space. This backbone can sample materials unconditionally or with global conditioning (text or image). To accurately guide sampling in following a spatial condition and achieve high-quality generation, we propose using a ControlNet [Zhang et al. 2023].

We modify the diffusion process to enable high-resolution generation and tileability. We achieve high resolution through patched diffusion, introducing the notion of noise rolling (and unrolling). At each diffusion step, the noise tensor is rolled by a random factor, preventing the presence of seams between patches and ensuring consistency across them. This process not only allows diffusion per patch, but also ensures tileability in the generated materials and preserves the possible tileability of the input photograph. When diffusing at high resolution, we note that low-frequency elements of the SVBRDF, in particular in the mesostructure, tend to disappear. We propose to solve this through a multi-scale diffusion, maintaining low-frequency signal for high-resolution generations by merging multiple scales of generated material maps through masked diffusion. We combine this with a patched decoding process in the VAE to enable the generation of arbitrary resolution material, which we demonstrate up to 4K. Finally, we enable tileability for non-tileable inputs photographs by inpainting the input photograph borders, and synthesizing them with our tileability constraint.

We discuss our material model in Section 4.1, our diffusion model in Section 4.2 and our different design choices for the diffusion process to enable conditioned (Section 4.3), tileable and high-resolution (Section 5) generation.

### 4 CONTROLLED GENERATIVE MODEL FOR MATERIALS 4.1 Material Representation

Our methodisdesignedtogenerate materials in the form of SVBRDFs represented as a collection of 2D texture maps. These maps represent a spatially varying Cook-Torrance micro-facet model [Cook and Torrance 1982; Karis 2013], using a GGX [Walter et al. 2007] distribution function, as well as the material mesostructure. In particular, we generate base color 𝑏, normal 𝑛, height ℎ, roughness 𝑟 and metalness 𝑚 properties. Our model also supports the generation of an opacity parameter map as exemplified in Figure 1. For visualization

space considerations, as this parameter is only relevant for a very small subset of materials, we omit it in the paper, but include this result map in the Supplemental Materials. The roughness is related to the width of the BRDF specular lobe, where a lower roughness represents a shinier material. Metalness defines which area of the material represents raw metal. We generate both normal and height properties separately, as artists typically include different signals in these maps [McDermott 2018]. We use a standard microfact BRDF model [Cook and Torrance 1982; Karis 2013] based on the GGX normal distribution function [Walter et al. 2007] and computing the diffuse and specular components of our renderings similarly to Deschaintre et al. [2018]. The exact model formulation is available in the Supplemental Materials.

4.2 Generative material model

MatGen adapts the original LDM architecture to output a set of SVBRDF maps instead of a single RGB image. This core generative model consists of two parts: a compression VAE [Kingma and Welling 2013] E, learning a compact latent representation of the material maps, and a diffusion [Rombach et al. 2022] U-Net [Ronneberger et al. 2015] model 𝜖𝜃, which learns the distribution of the latent VAE features (see Fig. 2).

The VAE compression model consists of an encoder E and a decoder D, trained to jointly encode and decode a set of 𝑁 maps 𝑀 = {M1, M2, . . ., M𝑁 } concatenated over the channels dimension. This compresses a tensor M ∈ R𝐻×𝑊 ×𝐶, where 𝐶 = 9 is the concatenation of the different material maps defined in Section 4.1, into a latent representation 𝑧 = E(M), where 𝑧 ∈ Rℎ×𝑤×𝑐, and 𝑐 is the dimensionality of the encoded maps. We set ℎ = 𝐻8 , 𝑤 = 𝑊8 as in the original LDM architecture [Rombach et al. 2022] and set 𝑐 = 14 which we empirically found to lead to the best compression/quality compromise.

Following Rombach et al. [2022], we train the VAE E using a combination of pixel-space 𝐿2 loss, a perceptual LPIPS loss [Zhang et al. 2021], and a patch-based adversarial objective [Dosovitskiy and Brox 2016; Esser et al. 2021; Isola et al. 2017] for each map separately. Furthermore, we follow the VAE latent space regularization and impose a Kullback–Leibler divergence penalty to encourage the latent space to follow a Normal distribution [Kingma and Welling 2013; Rezende et al. 2014].

We train our diffusion model 𝜖𝜃 to learn to sample the latent distribution of the VAE E. In particular, we train a diffusion model, using a time-conditional U-Net core, as proposed in [Rombach et al. 2022]. During training, noised latent vectors are generated, following the strategy defined in [Ho et al. 2020], through a deterministic forward diffusion process 𝑞 (𝑧𝑡 |𝑧𝑡−1), transforming them into an isotropic Gaussian distribution. The diffusion network 𝜖𝜃 is then trained to perform the backward diffusion process 𝑞 (𝑧𝑡−1|𝑧𝑡), effectively learning to "denoise" the latent vector and reconstruct its original content.

Once trained, our models allow sampling a normal distribution, "denoising" the samples into a valid latent space point, and decoding it using the VAE into high quality SVBRDF maps.

- 4.3 Controlled Synthesis

To control the generation process we just described, we use two different mechanisms: a) global conditioning for text or visual, highlevel prompts and b) spatial conditioning (e.g. a material photograph).

- 4.3.1 Global conditioning. Following the work by Rombach et al.

- [2022] we implement global conditioning through cross-attention [Vaswani et al. 2017] between each block of convolutions of the denoising U-Net and an embedding of the condition 𝑦, which is extracted by an encoder 𝜏𝜃, with the attention defined as:

Attention(𝑄,𝐾,𝑉) = softmax

𝑄𝐾𝑇 √

𝑑

𝑉, (1)

where𝑄 = 𝑊𝑄𝑖 ·𝜑𝑖(𝑧𝑡),𝐾 = 𝑊𝐾𝑖 ·𝜏𝜃 (𝑦),𝑉 = 𝑊𝑉𝑖 𝜏𝜃 (𝑦). Here,𝜑𝑖(𝑧𝑡) ∈ R𝑁×𝑑𝜖𝑖 is the flattened output of the previous convolution block of

𝜖𝜃 –the diffusion network–, and𝑊𝑄𝑖 ∈ R𝑑𝜖𝑖 ×𝑑,𝑊𝐾𝑖 ∈ R𝑑𝜏𝑖 ×𝑑,𝑊𝑉𝑖 ∈ R𝑑𝜏𝑖 ×𝑑, are learnable projection matrices. A visual representation of the attention layer is included in the Supplemental Materials. The training objective in the conditional setting becomes

𝐿𝐿𝐷𝑀 := EE(𝑀),𝑦,𝜖∽N(0,1),𝑡 ∥𝜖 − 𝜖𝜃 (𝑧𝑡,𝑡,𝜏(𝑦))∥22 (2)

We use a pre-trained CLIP [Radford et al. 2021] model as feature extractor 𝜏 to encode the global condition. MatGen is trained using the CLIP embeddings of the material renderings for global conditioning. At inference time, it is possible to condition MatGen via both text and image prompts, respectively encoded through the corresponding heads of CLIP. As material captions are not available in our dataset, we use a prior as proposed in Dall-e 2 [Aggarwal et al. 2023; Ramesh et al. 2022]. We employ the architecture proposed in the original paper and train the prior on an internal dataset of images for which we have titles and tags. This prior lets us transform text CLIP embeddings into image CLIP embeddings, which can be directly used to condition our model.

4.3.2 Spatial conditioning. We build ControlMat by adding spatial conditioning to the Latent Diffusion Model using a ControlNet [Zhang et al. 2023], leveraging its conditioning capabilities to tackle the tasks of (i) SVBRDF estimation from single input image, and (ii) inpainting for material editing and tileability.

Our ControlNet replicates the original design of Zhang et al.

- [2023]: It consists of a convolutional encoder, a U-Net encoder and middle block, and a skip-connected decoder. As our main model for material generation MatGen, we train our ControlNet to receive a 4-channel input consisting of an RGB image and a binary mask, concatenated along the channel dimension (see the center part of Fig. 2). This binary mask guides where the diffusion model has to generate new content, allowing for in-painting use cases. A small

encoder E𝑐, of four convolutional layers, is used to encode the condition to the same dimension as the noise processed by our LDM. In particular, given an image spatial condition 𝑐𝑖 ∈ R𝐻×𝑊 ×4, the encoder produces a set of feature maps 𝑐𝑓 = E𝑐(𝑐𝑖), with 𝑐𝑓 ∈ Rℎ×𝑤×𝑐.

The ControlNet is trained to guide the diffusion process of our main diffusion model, MatGen. During this training, MatGen is kept

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

(a) input (b) naïve approach (c) patches + overlap (d) noise rolling

Fig. 3. Patch diffusion comparison. Examples of height map results using different approaches for patched latent diffusion.

frozen and only the ControlNet is actually optimized. This allows for faster convergence and less computationally intensive training of the conditioning [Zhang et al. 2023]. Further, this makes our main diffusion model independent from the ControlNet, letting us choose during inference whether to use it unconditionally, with a global-condition or with a spatial condition, without retraining.

- 4.3.3 Material acquisition. For material acquisition, we input the material photograph both as a global condition as described in Section 4.3.1 and through the ControlNet [Zhang et al. 2023], as described in Section 4.3.2. The global conditioning provides guidance where there are regions of the material to inpaint –for which we do not want to follow the condition image perfectly. And ControlNet provides guidance to locally condition our generation model on the input photograph. This condition combination lets us sample likely material properties for a given photograph, compensating for the ill-posedness of material acquisition under unknown illumination. We compare the different conditioning in Figure 8 and show that our model recovers materials that better match the input photographs than previous work, while better separating albedo from mesostructures, light and shading in Figures 9 and 10.

5 LARGE SCALE, TILEABLE MATERIALS GENERATION

A simple combination of Latent Diffusion [Rombach et al. 2022] with ControlNet [Zhang et al. 2023] would be limited to non-tileable, lower-resolution results, due to memory constraints. This is particularly limiting in the context of material creation, where typical artists work at least at 4K resolution. To achieve high-resolution, we employ a patch–based diffusion and introduce a noise rolling technique (see Sec. 5.1) to (i) enforce consistency between patches and (ii) remove visible seams, without the additional cost of overlapping patches. This separation in different patches tends to create inconsistency in normal estimation and looses some of the low-frequency meso-structure content. To preserve this low-frequency content at high resolution materials and ensure normal consistency throughout the generation, we propose a multiscale diffusion method (Sec. 5.3). Finally, we introduce a simple, yet effective, patched decoding (see Sec. 5.4) achieving high-resolution generation with limited memory requirement.

- 5.1 Noise rolling

Diffusing noise by patches allows to reduce the memory consumption of diffusing large noise tensor, but also introduce artifacts in the output with visible seams, as shown in Fig. 3(b). This effect can be mitigated by the use of overlapping patches, similar to previous

- 5.2 Border inpainting

While our noise rolling ensures tileability in the generation process, if spatially conditioned on a non-tileable photograph, the condition prevails, preventing the output from being tileable. To enable tileability for any arbitrary input photograph, we adopt an inpaint-

ing technique, by masking the input border (with a border size of 161 of the image size) and letting the diffusion model regenerate it. This

is made possible by our use of a mask in the training of our ControlNet. During training, the input image to the ControlNet is masked with a random binary masks –between 0% and 40% of the image size is masked randomly at each step–, provided to the network

- as an additional input channel. At the same time, we encode the full unmasked image with CLIP, and provide it to the LDM through Cross-Attention. This allows the generation of the masked region through inpainting while using the global conditioning to generate a coherent material.

When masking the borders to make an input photograph tileable, the masking itself does not guarantee tileability on the estimated maps. It is the combination of the masking and the noise rolling (Sec. 5.1) which allows to inpaint and ensure that the generated region is tileable. As the inpainted region is generated using the global conditioning, it is inherently tileable thanks to the noise rolling approach, and since this inpainted region is on the border of the material, it enforces the material to be tileable. This approach is particularly efficient for stationary materials, but gives reasonable results on structured ones too. We provide results using this process in Figure 11 and in the Supplemental Materials.

5.3 Multiscale diffusion.

Estimating SVBRDFs is highly dependent on the network receptive field compared to the physical size of the captured area, and tends to produce flat geometry when evaluated at a higher resolution than trained for [Deschaintre et al. 2020; Martin et al. 2022]. To overcome this issue, we propose a multiscale diffusion approach. In particular, we leverage the information extracted at a lower resolution, to better estimate the low-frequency geometry when diffusing at higher resolutions, and ensure normal consistency throughout the patched process. This is akin to the cascaded diffusion models [Ho et al. 2022] but we use a single model where the starting noise comes from the output of the previous generation rather than one base model for generation and one for super-resolution as done with cascaded diffusion.

To achieve the proposed multiscale diffusion, we apply a hierarchical process where diffusion takes place at multiple scales, coarse and fine. More precisely, we proceed as follow:

- (1) we perform a low-resolution diffusion,
- (2) we upsample and renoise the low resolution output,
- (3) we perform a higher-resolution diffusion process, using the output of step (2) as initial noise.

Multiscale diffusion requires to perform a complete denoising

- at all the lower resolution. In particular a 4K generation requires estimations at a 512 × 512, 1024 × 1024 and 2048 × 2048 resolution. We found this to be the best approach to multiscale generation, preserving low frequencies while still generating high quality high frequencies and ensuring consistent normal orientations. A detailed

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Unrolled image

Rolled image

Rolling

- Fig. 4. Noise rolling. Visual representation of the noise rolling approach. The input is "rolled" over the x and y axes by a random translation, represented in the figure by replicating the image 2x2 and cropping the region contained in the blue square. Unrolling consists in doing the inverse process.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Input picture (non tileable)

Masked input Rolled input Rolled estimated

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

basecolor (tileable)

Estimated area

Inpainted area

- Fig. 5. Tileable estimation. Visual representation of the tileable estimation via border inpainting and noise rolling approach. We mask the input image border letting the diffusion model entirely regenerate it (blue area in the figure) while estimating the properties of the unmasked area (red area in the figure). In combination with rolling it ensures tileability while keeping the content of the image mostly unaltered.

material [Deschaintre et al. 2020] and diffusion [Bar-Tal et al. 2023] methods. However, this approach does not perfectly correct the problem with visible seams and low-frequency artifacts remaining (Fig. 3(c)). Further, overlapping patches requires to generate more patches, requiring extra computation.

To tackle this issue, inspired by the training procedure of tileable GANs [Zhou et al. 2022], we propose to take advantage of the iterative nature of the diffusion process at inference time. By “rolling” the noise tensor on itself by a random translation (and subsequently unrolling after each diffusion step), we remove seams stemming from diffusion (Fig. 5). This approach provides better consistency between patches (Fig. 3(d)), matching the statistics of the generated image at each diffusion step to match the learned distribution. As the learned distribution does not contain seams randomly placed in the images, our noise rolling approach naturally pushes the generation towards tileable images in an unconditional or globally-conditioned setting. We provide the pseudo-code for the noise rolling algorithm in Alg. 1 and evaluate the effect of the noise rolling for patched generation and materials tileability in Sec. 6.3. While circular padding [Zhou et al. 2022] is a viable solution for tileable generation of non-patched images, in our patched diffusion context it would lead to tileable individual patch, instead of enforcing tileability of the entire material.

ALGORITHM 1: Patched diffusion with noise rolling Data:𝑇 = 100, 𝑚𝑎𝑥_𝑟𝑜𝑙𝑙 ≥ 0, 𝑝 = 32 Result: 𝑧

- 1 𝑡 ← 0
- 2 𝑧 ← sample_noise()
- 3 while 𝑡 < 𝑇 do

- 4 𝑟𝑥 ← random(0, max_roll)
- 5 𝑟𝑦 ← random(0, max_roll)
- 6 𝑧 ← roll(𝑧, (𝑟𝑥,𝑟𝑦))
- 7 𝑧_𝑠ℎ𝑎𝑝𝑒 ← shape(𝑧)
- 8 𝑧 ← patch_noise(z, p)
- 9 𝑧 ← sample(𝑧,𝑡)
- 10 𝑧 ← unpatch_noise(z, z_shape, p)
- 11 𝑧 ← roll(𝑧, (−𝑟𝑥, −𝑟𝑦))
- 12 𝑡 ← 𝑡 + 1 end Procedure patch_noise(Noise tensor 𝑧, Patch size 𝑝)

- 1 𝑃 ← empty list
- 2 for 𝑖 = 0 to rows(𝑧) − 𝑝 by 𝑝 do

- 3 for 𝑗 = 0 to columns(𝑧) − 𝑝 by 𝑝 do

- 4 patch ← 𝑧[𝑖 : 𝑖 + 𝑝, 𝑗 : 𝑗 + 𝑝]
- 5 append(𝑃, patch) end

end

- 6 return 𝑃 Procedure unpatch_noise(Patched noise 𝑃, Original shape (𝐻,𝑊 ),

Patch size 𝑝)

- 1 𝑧 ← zero matrix of shape (𝐻,𝑊 )
- 2 𝑘 ← 0
- 3 for 𝑖 = 0 to 𝐻 − 𝑝 by 𝑝 do

- 4 for 𝑗 = 0 to𝑊 − 𝑝 by 𝑝 do

- 5 𝑧[𝑖 : 𝑖 + 𝑝, 𝑗 : 𝑗 + 𝑝] ← 𝑃[𝑘]
- 6 𝑘 ← 𝑘 + 1 end

end

- 7 return 𝑧

evaluation of the effect of our proposed multi-scale diffusion on the generation is presented in Sec. 6.4.2.

- 5.4 Patched decoding.

While our rolled diffusion process allows us to diffuse larger resolutions, decoding high-resolution maps with the VAE in a single step is a memory-intensive operation. We propose to use a patch decoding approach, which was also shown to be viable in community-driven projects like Automatic11111, decoding patches of the latent representation separately, thus reducing the maximum memory requirements. However, if applied naively, this approach produces visible seams and inconsistencies between decoded patches. To mitigate this issue we adopt a simple, yet effective solution: we first decode a low-resolution version of the material, decoded in a single pass, and match the mean of each high-resolution patch with the corresponding patch in the low-res version. We show a visual overview of the patch decoding in Fig. 6. This does not require any changes

1https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Downsampling

mean matching

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

| | |
|---|---|
| |Patching|

[Figure 97]

[Figure 98]

Fig. 6. Overview of our patched decoding. Decoding our latent vector 𝑧 per patch (reducing peak-memory usage) introduces seams between them. We propose to encourage similarity between patches by first decoding a low resolution material and applying a mean matching operation between the corresponding regions. Combined with an overlapping patches blending approach, this prevents the apparition of seams.

in the architecture, and significantly mitigates the inconsistency issues. However, as minor seams may remain, we decode overlapping patches and blend them using truncated Gaussian weights, maximally preserving the signal at the center of the patches and giving a 0 weight to the borders. Design choices for the patch decoding approach are ablated in Sec. 6.4.3 and are particularly important for the material domain, where material properties with discontinuities in lightness or contrast result in significant appearance change when rendered. In our experiments, we use a patch-size of 512.

6 IMPLEMENTATION & RESULTS 6.1 Dataset and metrics

6.1.1 Training dataset. We train our model using a synthetic dataset we generate. In particular, we follow the approach proposed by Deschaintre et al. [2018] and Martin et al. [2022]: starting from a collection of parametric material graphs in Substance 3D format [Adobe 2022], we generate a large amount of material variations. In particular, our dataset is composed of material maps (Basecolor, Normal, Height, Roughness and Metalness) and associated renderings under varying environment illuminations. To generate as much data as possible, we leverage the Adobe Substance 3D Assets library [Adobe 2022], consisting of 8615 material graphs. As opposed to Martin et al. [2022], we do not constrain our method to outdoor, mostly diffuse materials (e.g Ground, Stone, Plaster, Concrete), but use materials from all available categories 2, including those typically designed for indoor (e.g Ceramic, Marble, Fabrics, Leather, Metal, ...) use.

Similar to Martin et al. [2022], we generate large amounts of variations of the procedural materials by tweaking the parameters exposed by the artists during design. For each parameter we ensure we sample them in a range defined by the artists presets to obtain realistic material variations. Each material variation is represented by a set of textures/maps representing the SVBRDF, with a resolution of 2048x2048 pixels.

We follow previous work [Martin et al. 2022] and produce 14 renderings for each material variations, maintaining the 2048x2048

2See https://substance3d.adobe.com/assets/allassets?assetType=substanceMaterial for a complete list of categories.

pixels resolution, with ray-tracing and IBL lighting. We adapt the HDR environment map to the material category, using outdoor environment maps for material categories typically found outdoors and indoor environment maps for indoor categories.

The pairs of renderings and SVBRDF maps are then augmented togetherwith rotation,scalingand cropping to obtain512×512pixels training pairs. From 8, 615 material graphs, we generate ∼ 126, 000 material variations, ∼ 800, 000 materials crops, and ∼ 10, 000, 000 cropped pairs of renderings and materials.

We use different parts of the dataset for different components of our method. Our VAE is trained on the cropped materials (∼ 800𝑘 training points), the diffusion model is trained on the complete dataset (∼ 10𝑚 training pairs), and our ControlNet is trained on ∼ 25% of the complete dataset (∼ 2.5𝑚 training pairs).

- 6.1.2 Synthetic evaluation dataset. We design a synthetic dataset withoutrelyingonanySubstance3D library asset to ensure train/test dataset separation. Our collected dataset consists of 66 materials from three sources with permissive licenses: 27 materials from AmbientCG3; 26 from PolyHaven4; and 13 from Textures.com5.

For each material, we generate 4 renderings under different environment illuminations (not shared with the ones used for the training dataset generation).

- 6.1.3 Real photographs evaluation dataset. Our real photographs datasetwascapturedusingsmartphones, DSLR cameras, and sourced from the Web. This dataset offers a comprehensive representation of real-world image materials encountered in various lighting scenarios. Smartphone photos represent images captured by typical consumer-grade mobile devices, exhibiting medium resolution and noise levels commonly encountered in everyday photography. On the other hand, DSLR images represent high-quality photographs captured by professional-grade cameras, characterized by superior resolution, dynamic range, and color fidelity. Finally, web photos comprise images sourced from the Internet, which often exhibit low resolution, JPEG compression, and poor dynamic range. The different images and their origin are available in the Supplemental Material. 6.2 Technical details

- 6.2.1 VAE. We train our Kullback–Leibler-regularized VAE compression model [Kingma and Welling 2013] with stochastic gradient descent, using the Adam optimizer on 8×A100, 40 GB VRAM GPUS in parallel. We define the batch size to 9 per GPU, the learning rate is set to 4.5·10−6 and we train the model for 390k iterations, lasting 240 hours. Following the original work [Esser et al. 2021], we only enable the adversarial loss from iteration 50k onward.
- 6.2.2 Latent Diffusion model and ControlNet. We train successively our Latent Diffusion Model [Rombach et al. 2022] and ControlNet [Zhang et al. 2023] with stochastic gradient descent, using the Adam optimizer on 8×A100, 40 GB VRAM GPUS in parallel. We define the batch size to 44 per GPU to maximize memory usage. The learning rate is set to 2 · 10−7 and we train the LDM/ControlNet

- 3https://ambientcg.com/
- 4https://polyhaven.com/
- 5https://www.textures.com/

models for, respectively, 300k/350k iterations, lasting 180/240 hours. The U-Net employed in our experiments uses 320 base channels and the same channel multiplier as the original LDM implementation. The trainable copied weights of the ControlNet are initialized to be a copy of the trained LDM weights. During training of the ControlNet, only its parameters are optimized, while the LDM is kept frozen. Conditional generation is achieved by means of classifier free guidance training, with a condition dropout probability of 10%. Our implementation is based on the official ControlNet repository 6. At training time, ControlNet receives randomly downsampled and JPEG compressed rendering to further enhance robustness. This allows ControlMat to better reconstruct fine-grained details when provided with a low-resolution or compressed input image.

- 6.2.3 Inference. We evaluate execution speed and memory consumption on a A10G GPU with 24 GB of VRAM. As the complete ControlMat architecture (VAE + LDM + CLIP encoder + ControlNet) memory requirements and timings heavily depend on the target resolution, we provide values up to 4K results. Generation is performed by denoising a random noise latent for 50 steps, using the DDIM sampler [Song et al. 2020] with a fixed seed. Material estimation from an input image takes 3 seconds at 512 × 512, 18 seconds at 1024 × 1024 and 12GB of VRAM, 43 seconds at 2048 × 2048 and 18GB VRAM, and 350 seconds at 4096 × 4096 and 20GB of VRAM. We report timing and memory requirements for a maximum of 8 patches processed in parallel during the diffusion process, using the multiscale approach, which requires the diffusion steps at all the lower resolutions to be completed. It is possible to further reduce memory requirements in exchange for longer computation times by reducing the patches batch size.

6.3 Results and comparisons

For all results we show material maps and renderings, as well as a clay rendering, highlighting the material mesostructure. As inferred height maps are normalized between [0;1], we automatically approximate a displacement factor by matching the scale of normals computed from the inferred height to the scale of the normals directly inferred by the methods. We find this approach to be effective in matching the displacement with the predicted normals, achieving an average RMSE of 0.05 between the ground truth and derived normals on the test set samples.

- 6.3.1 Material generation. We evaluate our MatGen model and compare it to TileGen [Zhou et al. 2022], a GAN-based material generative model, and MatFuse [Vecchio et al. 2024], a recent diffusionbased material generation method, in Figure 7. In particular, we compare to the per-class models trained by TileGen on Stone, Metals, and Bricks. We generate results for both our method and MatFuse using global conditioning by embedding the class names as a condition. For this comparison, we used a version of MatFuse trained on the recently released MatSynth dataset [Vecchio and Deschaintre 2024] that was provided by the MatFuse authors. We emphasize that, in this experiment, the generation is only conditioned on the desired class, resulting in varied appearances. Despite not having been specifically trained per semantic class as TileGen,

6https://github.com/lllyasviel/ControlNet

Color Normal Height Rough Metal Render Clay

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

TileGen

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Metal MatFuseMatGenTileGen

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

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

[Figure 132]

[Figure 133]

Stone MatFuseMatGenTileGen

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

Bricks MatFuseMatGen

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Fig. 7. Comparison on global conditioning. We compare MatGen, our underlying generative model, to TileGen models trained on three categories (Metal, Stone, Bricks). We condition our model with the category name as a global condition. We can see that despite MatGen being trained on all categories in a single network, it can generate results with similar quality to TileGen instances specialized on these categories. We also include results of concurrent work MatFuse which performs well, but does not support local conditioning and is limited to non tileable, low resolution generation. Additional results can be found in Supplemental Materials.

Input Color Normal Height Rough Metal Render

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

“terracotta bricks”

Text

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

TextGlobalLocal

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

“rusted metal panel”

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

TextGlobalLocal

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

“light brown stone pavement”

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

GlobalLocal

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Fig. 8. Comparison between conditioning options. “Text” lines show results using only the text global conditioning, defining the overall appearance, without control over the details. “Global” lines show results using only the image global conditioning, providing a more complete appearance condition than text. However, the global conditioning approach doesn’t preserve the spatial arrangement and details. “Local” lines show results combining the global conditioning and the ControlNet [2023], providing both a global and local guidance, better reproducing the target appearance.

our model can generate high-quality materials with a large amount of details. As our dataset is larger than the one MatFuse was trained on, our results show better quality. This is further confirmed by a higher CLIP-score [Hessel et al. 2021] of 26.15 obtained by MatGen on a set of 80 samples, compared to 23.19 with MatFuse, proving a better match of the generated materials with the guiding text prompt. MatFuse generation is, additionally, limited to 256x256 nontileable results. In both cases, we can see in the Clay renderings that our approach leads to significantly more details in the generated materials.

We evaluate the difference between the different types of conditions discussed in Section 4.3.1: global condition (text (Text) or image (Global)) and the combination of global image condition and local condition using ControlNet (Local). As expected, the text conditioning generates materials matching the overall description, but lacks information for precise control. Due to the global condition input mechanism, the global image control generates materials which

overall match the desired appearance, but result in significant variation (e.g. different scale or the pavement which tiles are differently arranged). This is a very interesting property to explore possible material variations, which however does not fit the requirements for acquisition. In contrast, our local conditioning, used for material acquisition scenarios, combines global image conditioning with the ControlNet, as described in Section 4.3.2, generating materials that match the input spatial conditioning.

We include more generation results with a wider prompt variety in the Supplemental Material, as well as a comparison to a CLIP based Nearest-Neighbour search in the training database, validating the network generative capability.

6.3.2 Material estimation. Our method’s main application is material estimation from an input photograph. In this section we compare to recent state-of-the-art models, SurfaceNet [Vecchio et al. 2021] and MaterIA [Martin et al. 2022] on both synthetic and real materials. We retrain SurfaceNet on our dataset, while we use the available

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

GTMaterIAGTMaterIAGTMaterIASurfaceNetControlMatSurfaceNetControlMatSurfaceNetControlMat

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

[Figure 286]

[Figure 287]

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

- Fig. 9. Qualitative comparison on synthetic materials between SurfaceNet [Vecchio et al. 2021], MaterIA [Martin et al. 2022] and ControlMat (Ours). We show the Ground Truth and each method input, results parameter maps, Clay and 3 renderings under different lighting than the input image. We see that our approach suffers less from baked lighting and better estimates the mesostructure and roughness property. As MaterIA was not designed for conductors, it cannot recover well the first example’s appearance. As it does not estimate a metallic map, we replace it by a black image. We show the normalized height maps, and automatically adjust the displacement factor of renders to match the input.

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

MaterIAMaterIAMaterIAMaterIAControlMatSurfaceNetControlMatSurfaceNetControlMatSurfaceNetControlMatSurfaceNet

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

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

- Fig. 10. Qualitative comparison on real photographs between SurfaceNet [Vecchio et al. 2021], MaterIA [Martin et al. 2022] and ControlMat (Ours). We show each method’s input, results parameter maps, Clay and 3 renderings under different environment illumination (as we do not know the input illumination, we do not exactly reproduce it). We see that approach better separates mesostructure and reflectance, better removing the light from the input and reconstructing the surface geometry. As MaterIA doesn’t estimate a metallic map, we replace it by a black image. We show the normalized height maps, and automatically adjust the displacement factor of renders to match the input.

Input Base color Normal Height Roughness Metalness Render Render clay

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

2x

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

2x

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

2x

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

- Fig. 11. Examples of real materials captured with our method. We show here the diversity of materials our model can estimate. We demonstrate in the first three rows our acquisition results without border tiling. The next three rows demonstrate the tileability capacity of our border inpainting approach, with non tileable input photographs and tileable output materials. And the last three rows show results of our method when provided flash illuminated photographs, demonstrating our approach robustness, despite having not been explicitely train on this kind of illumination. While the tileable output materials do not pixel-match the input photograph at the borders, the general material appearance is preserved with the strong tileability benefit. We show the normalized height maps, and automatically adjust the displacement factor of our renders to match the input. We show additional results in Supplemental material.

MaterIA model in Substance Sampler which was trained on an "outdoor" dataset generated from the same procedural materials. This "outdoor" dataset contains the same amount of data as the "outdoor" part of our generated dataset.

We first compare to previous work and Ground-Truth on synthetic data in Figure 9, including the material maps, a "Clay" rendering to visualise the height and three renderings under different illuminations. In particular, we can see that both previous work tend to bake highlights in the diffuse albedo and artefacts in the height map. As MaterIA [2022] was trained for outdoor materials, it does not infer any metalness map nor handle well shiny materials. We also provide a quantitative comparison to these methods on our complete synthetic dataset (metrics are computed as the average of all materials in our test dataset) (Sec. 6.1.2) in Tables 1 and 2. We show in Table 1 RMSE comparisons for all parameters, and the error on re-renderings averaged for 4 different environment illumination, confirming that our model better reproduces the appearance of input images when relit, and better matches the ground truth material maps than previous work. The RMSE for the height is computed on the normalized map in the range [0, 1]. The narrow 95% confidence interval obtained, underlines the robustness of our method to different input lighting conditions and minimal variation in performance, confirming that ControlMat is able to generate highquality materials matching the target appearance with diverse input lighting conditions. Examples of estimation for the same material under different lighting conditions are included in the Supplemental Materials. We further evaluate the Albedo map and re-renderings with the SSIM [Wang et al. 2004] and LPIPS [Zhang et al. 2018] perceptual metrics as other parameter maps are not intepretable as natural images.

We also compare against previous work on real photographs in Figure 10. As we do not know the original lighting in the input picture, the renderings are relit with the same environment illumination than in Figure 9. Here again we show that our approach generates materials that better reflect the photographed material. Albedo and meso-structure are better disambiguated (1𝑠𝑡 row), light is less baked in the diffuse albedo (2𝑛𝑑 row), both high and low frequencies of the geometries are better recovered (3𝑟𝑑 & 4𝑡ℎ row) and overall albedo color better matches the input image (all rows).

Finally, we evaluate our methods on a variety of input photographs with different material types (stones, bricks, wood, metals, ...) under varying lighting conditions in Figure 11. We first show three results of acquisition without border inpainting, showing results that match better the input at the borders, but are not tileable. We then show three results with border inpainting for tileability, demonstrating appearance preservation while making the result tileable. The last three results demonstrate the robustness of our approach to different lighting, here with a typical flash-light illumination. Despite having not explicitly been trained on flash lighting, our model is capable of efficiently removing the light from the material maps and recovering plausible material properties.

We provide additional results and comparisons on both synthetic and real photographs in Supplemental Materials.

- Table 1. Quantitative results with MaterIA [Martin et al. 2022] and SurfaceNet [Vecchio et al. 2021]. We report here the RMSE↓ between predicted and ground-truth maps and renderings, except for Normal maps, showing the cosine error↓. As we average the RMSE over 4 different lighting, we also show the 95% confidence interval of the RMSE error across different lightings, showing that our approach not only performs better but is also more consistent.

Image Type SurfaceNet MaterIA ControlMat Renderings 0.114 ± 0.005 0.123 ± 0.006 0.097 ± 0.004 Base color 0.108 ± 0.008 0.103 ± 0.008 0.067 ± 0.005 Normal (Cos dist) 0.308 ± 0.026 0.318 ± 0.023 0.280 ± 0.022 Height 0.258 ± 0.014 0.251 ± 0.014 0.216 ± 0.012 Roughness 0.378 ± 0.014 0.368 ± 0.028 0.304 ± 0.013 Metallic 0.108 ± 0.028 x 0.076 ± 0.023

- Table 2. Quantitative results with MaterIA [Martin et al. 2022] and SurfaceNet [Vecchio et al. 2021]. We report here the perceptual metrics SSIM↑ and LPIPS↓ for images that can be intepreted as natural images: Renderings and base color.

Image Type Metric SurfaceNet MaterIA ControlMat Renderings

SSIM 0.677 0.719 0.729 LPIPS 0.239 0.211 0.184

SSIM 0.625 0.650 0.677 LPIPS 0.274 0.256 0.239

Base Color

Input Naïve Overlap Rolling

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

Fig. 12. Patch diffusion ablation results. We show the results of different approaches for patched ablation. We can see that naively concatenating separately diffused patches leads to significant seams in the recomposed image, here the height map. While merging overlapping patches reduces the seam problem, it doesn’t solve it and it requires to diffuse additional patches. Our noise rolling approch removes any visible seams while maintaining the same number of diffused patches as the Naïve approach.

6.4 Ablation study

We evaluate our different design choices by evaluating our model’s diffusion process elements –patched diffusion, multiscale diffusion, and patched decoding– against baseline solutions. We provide additional ablation results in the Supplemental Materials, their high resolutions make the difference between our elements and naive design even more apparent.

Input 1K Native 1K Multi-scale

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

- Fig. 13. Multiscale diffusion ablation results. Top row: without multiscale diffusion, our patched diffusion approach may sometime result in inconsistent normal orientation. Bottom row: When diffusing at higher resolution, geometries of the generated materials tend to appear flatter, losing details and flattening large elements. This is most visible when zooming on the normal maps shown here. Our Multi-scale approach preserves the normals orientation and the mesostructure, even when generating at higher resolution.

Input Naïve Overlap Mean match.

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

- Fig. 14. Patch decoding ablation results. Naively decoding patches and concatenating them leads to seams in final image. Generating overlapping patches and blending them with a gaussian kernel reduces the visibility of seams but can result in blurred images (see top example). Adding the mean matching helps remove the last visible seams and preserve the signal better.

6.4.1 Patched diffusion. We evaluate our method with and without Patched Diffusion in Figure 12 . The "naive" approach is simply the concatenation of the results of separate diffusion processes per patch. We can see that this naive approach results in largely varying patches, creating strong discontinuities where stitched. Using overlapping patches improves consistency and reduces the problem, but doesn’t completely solves it. Further, overlapping patches require the diffusion of a larger number of patches, leading to longer execution time. With our proposed noise rolling, we are able to generate consistent patches and ensure their tileability, preventing the apparition of seams when stitched while requiring the same number of patch diffusion as the naive approach.

- 6.4.2 Multiscale diffusion. During generation at higher resolution, if done naively, materials tend to lose details and appear flatter. We illustrate the effect of our proposed multi-scale diffusion in Figure 13. This approach ensures that both large elements (e.g. large stone in the first rows normal map) and high-frequency details are preserved. We can see that if generating results at a resolution of 1024 × 1024, the naive approach results in significantly flatter normals and loss of relief in the large stone, while our multi-scale approach preserves the original resolution details. The effects are particularly visible on high-resolution materials (2K+), we invite the reader to zoom in and see the full resolution results in Supplemental Material. This multiscale approach comes at the cost of additional (lower resolution) diffusion processes, requiring at most 40% more time for the total generation process (depending on the maximum resolution).
- 6.4.3 Patched decoding. We demonstrate here the effect of the VAE patch decoding step, allowing to reduce the peak memory consumption of our process. This lets us generate higher resolution materials, which we demonstrate up to 4K in Supplemental Material. We show a comparison of the results with and without patched decoding in Figure 14. We can see that once more, naively concatenating patches leads to visible seams in the results. Decoding overlapping patches and blending them with Gaussian weights significantly reduces the appearance of seams but does not solve it (bottom example) and loses some sharpness (top result). The addition of our mean matching from a lower resolution generation completely removes seams while preserving the original signal. The effect is particularly visible at high resolution as demonstrated in the Supplemental Materials. Further, the peak memory consumption of our method at 2K without patched decoding is 20GB while with it, it is 14 GB. At 4K we cannot generate results (out of memory) without the patched approach, while ours stays constant at 14GB (but requires decoding more patches, each of which is decoded in ∼150ms).

7 LIMITATIONS AND FUTURE WORK

Being generative, our ControlNet-conditioned diffusion model does not guarantee a pixel-perfect match with the input image, and some variations may arise such as slight color shifts or fine-scale texture alterations, as illustrated in Fig. 15 (top row). Such deviations are less likely to occur with neural image-translation methods [Martin et al. 2022; Vecchio et al. 2021]. Also, as diffusion models require multiple diffusion steps, the time required by our network to generate a material ranges from a few seconds to a few minutes, as reported in Sec. 6.2.3, which may hinder some application scenarios. However, recent progress has drastically reduced the number of required steps [Song et al. 2023], making consistency models a promising direction for further improving the performance of our approach. Our proposed inpainting to generate tileable materials from nontileable input works great for stochastic materials, but may break the structure a little in highly structured layouts (see Fig. 15, middle row). This is more likely to happen in challenging, or non-grid aligned structures, forcing the generation to forcefully fix misalignment, even by breaking the structure. In some cases, for shiny material where the input exhibits strong vignetting, our model mistakenly interprets the material as slightly metallic (see Fig. 15, bottom row.). This could be improved by randomly including strong vignetting

GT Render GT base color Estimation

[Figure 539]

[Figure 540]

[Figure 541]

Input Non tileable Tileable

[Figure 542]

[Figure 543]

[Figure 544]

Input Render

[Figure 545]

[Figure 546]

- Fig. 15. Limitations. (top row) We illustrate a small color shift happening in some results due to the generative nature of our architecture. In the middle row, we illustrate the clay renders of our approach’s results with and without tileability on strongly structured inputs. Our inpainting does not manage to enforce tileability while preserving the natural structure of challenging patterns. In the bottom row, we illustrate a result where our method erroneously attributed some degree of metalness to the leather properties, leading to a slightly metallic re rendered appearance. This is caused by the strong vignetting in the input.

in the training data augmentation. Finally, as shown in Fig. 11, our approach generates plausible results for flash-based acquisition, which is an interesting insight for future work targeting increased acquisition precision.

8 CONCLUSION

We proposed a generative diffusion model able to create tileable materials at high resolution. Our model can be conditioned by photographs, leveraging the recent ControlNet architecture and improving material evaluation from a single image. Further, our model can also be conditioned by an image or text prompt, for loose correspondence and exploration. To enable tileability and high resolution, we proposed an adapted inference diffusion process, with noise rolling, patched diffusion and decoding, multiscale diffusion, and border inpainting. We think that the insights gained in our material-specific diffusion process can extend beyond the material domain, e.g., for textures or 360° environment images.

9 ACKNOWLEDGMENTS

We thank Niloy Mitra for insightful suggestions to improve the exposition.

REFERENCES

Adobe. 2022. Substance Source. https://substance3d.adobe.com/assets/. Pranav Aggarwal, Hareesh Ravi, Naveen Marri, Sachin Kelkar, Fengbin Chen, Vinh Khuc, Midhun Harikumar, Ritiz Tambi, Sudharshan Reddy Kakumanu, Purvak Lapsiya, Alvin Ghouas, Sarah Saber, Malavika Ramprasad, Baldo Faieta, and Ajinkya Kale. 2023. Controlled and Conditional Text to Image Generation with Diffusion Prior. arXiv:2302.11710 [cs.CV]

Miika Aittala, Timo Aila, and Jaakko Lehtinen. 2016. Reflectance Modeling by Neural Texture Synthesis. ACM Trans. Graph. 35, 4, Article 65 (jul 2016), 13 pages. https: //doi.org/10.1145/2897824.2925917

Miika Aittala, Tim Weyrich, and Jaakko Lehtinen. 2015. Two-shot SVBRDF Capture for Stationary Materials. ACM Trans. Graph. 34, 4, Article 110 (July 2015), 13 pages. https://doi.org/10.1145/2766967

Martin Arjovsky, Soumith Chintala, and Léon Bottou. 2017. Wasserstein generative adversarial networks. In International conference on machine learning. PMLR, 214– 223.

Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. 2023. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation. arXiv preprint arXiv:2302.08113 2

(2023). Andrew Brock, Jeff Donahue, and Karen Simonyan. 2018. Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096 (2018). Robert L Cook and Kenneth E. Torrance. 1982. A reflectance model for computer graphics. ACM Transactions on Graphics (ToG) 1, 1 (1982), 7–24. Bin Dai and David Wipf. 2019. Diagnosing and enhancing VAE models. arXiv preprint arXiv:1903.05789 (2019).

Valentin Deschaintre, Miika Aittala, Frédo Durand, George Drettakis, and Adrien Bousseau. 2018. Single-Image SVBRDF Capture with a Rendering-Aware Deep Network. ACM Transactions on Graphics (SIGGRAPH Conference Proceedings) 37, 128 (aug 2018), 15. http://www-sop.inria.fr/reves/Basilic/2018/DADDB18

Valentin Deschaintre, Miika Aittala, Frédo Durand, George Drettakis, and Adrien Bousseau. 2019. Flexible SVBRDF Capture with a Multi-Image Deep Network. Computer Graphics Forum(Eurographics Symposium on Rendering Conference Proceedings) 38, 4 (jul 2019), 13. http://www-sop.inria.fr/reves/Basilic/2019/DADDB19

Valentin Deschaintre, George Drettakis, and Adrien Bousseau. 2020. Guided FineTuning for Large-Scale Material Transfer. Computer Graphics Forum (Proceedings of the Eurographics Symposium on Rendering) 39, 4 (2020). http://www-sop.inria.fr/ reves/Basilic/2020/DDB20

Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems 34 (2021), 8780–8794.

Alexey Dosovitskiy and Thomas Brox. 2016. Generating images with perceptual similarity metrics based on deep networks. Advances in neural information processing systems 29 (2016).

Patrick Esser, Robin Rombach, and Bjorn Ommer. 2021. Taming transformers for highresolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 12873–12883.

Michael Fischer and Tobias Ritschel. 2022. Metappearance: Meta-Learning for Visual Appearance Reproduction. ACM Trans Graph (Proc. SIGGRAPH Asia) 41, 4 (2022).

DUAN Gao, Xiao Li, Yue Dong, Pieter Peers, Kun Xu, and Xin Tong. 2019. Deep Inverse Rendering for High-Resolution SVBRDF Estimation from an Arbitrary Number of Images. ACM Trans. Graph. 38, 4, Article 134 (jul 2019), 15 pages. https://doi.org/10.1145/3306346.3323042

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (Eds.), Vol. 27. Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2014/file/ 5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf

D. Guarnera, G. C. Guarnera, A. Ghosh, C. Denk, and M. Glencross. 2016. BRDF Representation and Acquisition. In Proceedings of the 37th Annual Conference of the European Association for Computer Graphics: State of the Art Reports (Lisbon, Portugal) (EG ’16). Eurographics Association, Goslar, DEU, 625–650.

Paul Guerrero, Miloš Hašan, Kalyan Sunkavalli, Radomír Měch, Tamy Boubekeur, and Niloy J. Mitra. 2022. MatFormer: A Generative Model for Procedural Materials. ACM Trans. Graph. 41, 4, Article 46 (jul 2022), 12 pages. https://doi.org/10.1145/3528223. 3530173

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. 2017. Improved training of wasserstein gans. Advances in neural information processing systems 30 (2017).

Jie Guo, Shuichang Lai, Chengzhi Tao, Yuelong Cai, Lei Wang, Yanwen Guo, and LingQi Yan. 2021. Highlight-Aware Two-Stream Network for Single-Image SVBRDF

Acquisition. ACM Trans. Graph. 40, 4, Article 123 (jul 2021), 14 pages. https: //doi.org/10.1145/3450626.3459854

Jie Guo, Shuichang Lai, Qinghao Tu, Chengzhi Tao, Changqing Zou, and Yanwen Guo.

2023. Ultra-High Resolution SVBRDF Recovery from a Single Image. ACM Trans. Graph. (apr 2023). https://doi.org/10.1145/3593798 Just Accepted.

Yu Guo, Cameron Smith, Miloš Hašan, Kalyan Sunkavalli, and Shuang Zhao. 2020. MaterialGAN: Reflectance Capture Using a Generative SVBRDF Model. ACM Trans. Graph. 39, 6, Article 254 (nov 2020), 13 pages. https://doi.org/10.1145/3414685. 3417779

Philipp Henzler, Valentin Deschaintre, Niloy J Mitra, and Tobias Ritschel. 2021. Generative Modelling of BRDF Textures from Flash Images. ACM Trans Graph (Proc. SIGGRAPH Asia) 40, 6 (2021).

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 (2021).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33 (2020), 6840–6851.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. 2022. Cascaded diffusion models for high fidelity image generation. The Journal of Machine Learning Research 23, 1 (2022), 2249–2281.

Yiwei Hu, Julie Dorsey, and Holly Rushmeier. 2019. A Novel Framework for Inverse Procedural Texture Modeling. ACM Trans. Graph. 38, 6, Article 186 (Nov. 2019), 14 pages. https://doi.org/10.1145/3355089.3356516

Yiwei Hu, Paul Guerrero, Milos Hasan, Holly Rushmeier, and Valentin Deschaintre. 2022a. Node Graph Optimization Using Differentiable Proxies. In ACM SIGGRAPH 2022 Conference Proceedings (Vancouver, BC, Canada) (SIGGRAPH ’22). Association for Computing Machinery, New York, NY, USA, Article 5, 9 pages. https://doi.org/ 10.1145/3528233.3530733

Yiwei Hu, Paul Guerrero, Milos Hasan, Holly Rushmeier, and Valentin Deschaintre. 2023. Generating Procedural Materials from Text or Image Prompts. In ACM SIGGRAPH 2023 Conference Proceedings.

Yiwei Hu, Chengan He, Valentin Deschaintre, Julie Dorsey, and Holly Rushmeier. 2022b. An Inverse Procedural Modeling Pipeline for SVBRDF Maps. ACM Trans. Graph. 41, 2, Article 18 (jan 2022), 17 pages. https://doi.org/10.1145/3502431

Qingqing Huang, Daniel S. Park, Tao Wang, Timo I. Denk, Andy Ly, Nanxin Chen, Zhengdong Zhang, Zhishuai Zhang, Jiahui Yu, Christian Frank, Jesse Engel, Quoc V. Le, William Chan, Zhifeng Chen, and Wei Han. 2023. Noise2Music: Text-conditioned Music Generation with Diffusion Models. arXiv:2302.03917 [cs.SD]

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. 2017. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 1125–1134.

Álvaro Barbero Jiménez. 2023. Mixture of Diffusers for scene composition and high resolution image generation. arXiv preprint arXiv:2302.02412 (2023). Brian Karis. 2013. Real shading in unreal engine 4. Proc. Physically Based Shading Theory Practice 4, 3 (2013), 1.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. 2017. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196

(2017).

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. 2020. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8110–8119.

Diederik P Kingma and Max Welling. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013).

Xiao Li, Yue Dong, Pieter Peers, and Xin Tong. 2017. Modeling Surface Appearance from a Single Photograph Using Self-Augmented Convolutional Neural Networks. ACM Trans. Graph. 36, 4, Article 45 (jul 2017), 11 pages. https://doi.org/10.1145/ 3072959.3073641

Rosalie Martin, Arthur Roullier, Romain Rouffet, Adrien Kaiser, and Tamy Boubekeur. 2022. MaterIA: Single Image High-Resolution Material Capture in the Wild. Computer Graphics Forum 41, 2 (2022), 163–177. https://doi.org/10.1111/cgf.14466 arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14466

Wes McDermott. 2018. Maps common to both workflow. Allergorithmic, 75–79. https: //substance3d.adobe.com/tutorials/courses/the-pbr-guide-part-2 Lars Mescheder. 2018. On the convergence properties of gan training. arXiv preprint arXiv:1801.04406 1 (2018), 16. Luke Metz, Ben Poole, David Pfau, and Jascha Sohl-Dickstein. 2016. Unrolled generative adversarial networks. arXiv preprint arXiv:1611.02163 (2016).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1, 2 (2022), 3.

Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. 2014. Stochastic backpropagation and approximate inference in deep generative models. In International

conference on machine learning. PMLR, 1278–1286. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10684–10695.

Robin Rombach, Patrick Esser, and Björn Ommer. 2020a. Making sense of cnns: Interpreting deep representations and their invariances with inns. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16. Springer, 647–664.

Robin Rombach, Patrick Esser, and Bjorn Ommer. 2020b. Network-to-network translation with conditional invertible neural networks. Advances in Neural Information Processing Systems 33 (2020), 2784–2797.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 234–241.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. arXiv:2205.11487 [cs.CV]

Sam Sartor and Pieter Peers. 2023. Matfusion: a generative diffusion model for svbrdf capture. In SIGGRAPH Asia 2023 Conference Papers. 1–10.

Liang Shi, Beichen Li, Miloš Hašan, Kalyan Sunkavalli, Tamy Boubekeur, Radomir Mech, and Wojciech Matusik. 2020. MATch: Differentiable Material Graphs for Procedural Material Capture. ACM Trans. Graph. 39, 6, Article 196 (Dec. 2020), 15 pages.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning. PMLR, 2256–2265.

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020). Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency Models. arXiv:2303.01469 [cs.LG] Aaron Van Den Oord, Oriol Vinyals, et al. 2017. Neural discrete representation learning. Advances in neural information processing systems 30 (2017).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

Giuseppe Vecchio and Valentin Deschaintre. 2024. MatSynth: A Modern PBR Materials Dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 22109–22118.

Giuseppe Vecchio, Simone Palazzo, and Concetto Spampinato. 2021. SurfaceNet: Adversarial SVBRDF Estimation from a Single Image. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12840–12848.

Giuseppe Vecchio, Renato Sortino, Simone Palazzo, and Concetto Spampinato. 2024. MatFuse: Controllable Material Generation with Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 4429–4438.

Bruce Walter, Stephen R Marschner, Hongsong Li, and Kenneth E Torrance. 2007. Microfacet models for refraction through rough surfaces. In Proceedings of the 18th Eurographics conference on Rendering Techniques. 195–206.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.

Liang Yuan, Dingkun Yan, Suguru Saito, and Issei Fujishiro. 2024. DiffMat: Latent diffusion models for image-guided material generation. Visual Informatics (2024).

Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. 2021. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4791–4800.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Xilong Zhou, Milos Hasan, Valentin Deschaintre, Paul Guerrero, Yannick Hold-Geoffroy, Kalyan Sunkavalli, and Nima Khademi Kalantari. 2023a. PhotoMat: A Material Generator Learned from Single Flash Photos. In ACM SIGGRAPH 2023 Conference Proceedings (Los Angeles, CA, USA) (SIGGRAPH ’23). Association for Computing Machinery, New York, NY, USA.

Xilong Zhou, Milos Hasan, Valentin Deschaintre, Paul Guerrero, Kalyan Sunkavalli, and Nima Khademi Kalantari. 2022. TileGen: Tileable, Controllable Material Generation and Capture. In SIGGRAPH Asia 2022 Conference Papers (Daegu, Republic of Korea) (SA ’22). Association for Computing Machinery, New York, NY, USA, Article 34, 9 pages. https://doi.org/10.1145/3550469.3555403

Xilong Zhou, Miloš Hašan, Valentin Deschaintre, Paul Guerrero, Kalyan Sunkavalli, and Nima Khademi Kalantari. 2023b. A Semi-Procedural Convolutional Material

Prior. Computer Graphics Forum n/a, n/a (2023). https://doi.org/10.1111/cgf.14781 arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14781

Xilong Zhou and Nima Khademi Kalantari. 2021. Adversarial Single-Image SVBRDF Estimation with Hybrid Training. Computer Graphics Forum (2021).

Xilong Zhou and Nima Khademi Kalantari. 2022. Look-Ahead Training with Learned Reflectance Loss for Single-Image SVBRDF Estimation. ACM Transactions on Graphics 41, 6 (12 2022). https://doi.org/10.1145/3550454.3555495

