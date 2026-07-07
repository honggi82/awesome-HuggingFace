# arXiv:2412.08486v2[cs.CV]12Dec2024

## Learning Flow Fields in Attention for Controllable Person Image Generation

Zijian Zhou1,2,∗, Shikun Liu1, Xiao Han1, Haozhe Liu1, Kam Woh Ng1, Tian Xie1, Yuren Cong1, Hang Li1, Mengmeng Xu1, Juan-Manuel Pérez-Rúa1, Aditya Patel1, Tao Xiang1, Miaojing Shi3, Sen He1

1Meta AI, 2King’s College London, 3Tongji University

∗Work done at Meta

Controllable person image generation aims to generate a person image conditioned on reference images, allowing precise control over the person’s appearance or pose. However, prior methods often distort fine-grained textural details from the reference image, despite achieving high overall image quality. We attribute these distortions to inadequate attention to corresponding regions in the reference image. To address this, we thereby propose learning flow fields in attention (Leffa), which explicitly guides the target query to attend to the correct reference key in the attention layer during training. Specifically, it is realized via a regularization loss on top of the attention map within a diffusion-based baseline. Our extensive experiments show that Leffa achieves state-of-theart performance in controlling appearance (virtual try-on) and pose (pose transfer), significantly reducing fine-grained detail distortion while maintaining high image quality. Additionally, we show that our loss is model-agnostic and can be used to improve the performance of other diffusion models.

Date: December 13, 2024

[Figure 1]

Figure 1 We present Leffa, a unified framework for controllable person image generation that enables precise manipulation of both appearance (i.e., virtual try-on) and pose (i.e., pose transfer). Leffa’s generated images demonstrate high quality, with fine details preserved and minimal texture distortion. Please zoom in for better viewing.

- 1 Introduction

Controllable person image generation aims to generate an image of a person guided by reference images, allowing for control over attributes such as appearance and pose. Underpinning a wide variety of applications in virtual and augmented reality, gaming, and e-commerce, this task has made significant progress (Kim et al., 2024; Choi et al., 2024; Chong et al., 2024; Wan et al., 2024; Bhunia et al., 2023; Han et al., 2023; Lu et al., 2024; Pham et al., 2024) based on the advancements in diffusion-based image generation (Ho et al., 2020; Rombach et al., 2022).

The primary challenge in this task is to preserve the fine-grained details (e.g., textures, texts, logos, etc.) from the reference image while maintaining overall quality. Although current methods can generate high-quality images at first glance, they still suffer from the distortion of fine-grained textural details upon closer inspection (see Fig. 2b top, where the striped texture is wrong). Researchers have attempted to improve the preservation of fine-grained details by integrating inference strategies (e.g., disentangled guidance (Bhunia et al., 2023), multi-stage inference (Yang et al., 2024a), incorporating information from other modalities (Ning et al., 2024; Xu et al., 2024; Choi et al., 2024) (e.g., textual descriptions) and introducing auxiliary models for feature improvement (Kim et al., 2024; Xu et al., 2024; Choi et al., 2024; Wan et al., 2024; Sun et al., 2024; Lu et al., 2024; Pham et al., 2024) (e.g., using CLIP (Radford et al., 2021), DINOv2 (Oquab et al., 2024) features, and warping model, etc.). However, these methods typically increase model complexity and lack explicit supervision to establish visual consistencies between the target and reference images.

By visualizing the attention layer of various diffusion-based methods (Choi et al., 2024; Xu et al., 2024), we observe that in regions where details are distorted, the target query exhibits widelydistributed attention rather than accurately focusing on the corresponding reference key regions. This observation also aligns with findings from previous studies (Kim et al., 2024; Ren et al., 2022). For instance, in Fig. 2b, the incorrect generation of striped texture results from the target query failing to attend to the corresponding reference key regions. In contrast, when the attention map is manually corrected by swapping the highest-response values with those lying in the correct region, the model is able to generate significantly improved striped textures without any additional training (see Fig. 2c).

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

(a) (b) (c) (d)

Figure 2 Taking appearance control of a person image (virtual try-on) as an example: (a) input person and reference (garment) images; (b) generated image and attention map from a diffusion-based method (e.g., IDM-VTON (Choi et al., 2024)); (c) generated image after manually modifying the attention map in the diffusion-based method to focus on the correct regions; (d) generated image and attention map from Leffa. Our method generates high-quality images without detail distortion (see the colored striped texture).

Capitalizing on this observation, we proposea regularizationloss by learning flow fields in attention (Leffa) to alleviate the distortion of fine-grained details. Specifically, we transform the attention map between the target query and reference key into a flow field during training, which warps the reference image to more closely align with the target image. This process encourages the target query to accurately attend to the correct reference key.

To validate our method, we incorporate our Leffa loss into a diffusion-based baseline, observing a significant reduction in fine-grained detail distortion while maintaining high image quality (see Fig. 2d top), without additional inference cost. Additionally, visualization of our method’s attention map (see Fig. 2d bottom) confirms the model accurately focuses on the desired regions. We quantitatively and qualitatively evaluate our

method on virtual try-on (appearance control) with VITON-HD (Choi et al., 2021) and DressCode (Morelli et al., 2022); and pose transfer (pose control) with DeepFashion (Liu et al., 2016), both achieving state-of-theart performance and mitigating the distortion of fine-grained details across all datasets. Finally, we show that our Leffa loss is model-agnostic and generalizes effectively to other diffusion-based methods (Choi et al., 2024; Chong et al., 2024), demonstrating Leffa as a versatile and general solution for controllable person image generation.

The contributions of this work are as follows:

- • We propose Leffa, a regularization loss that explicitly guides the target queries in attention layers to focus on correct reference keys, mitigating detail distortion without additional parameters and inference cost.
- • We evaluate Leffa on both virtual try-on and pose transfer across various diffusion-based methods, achieving state-of-the-art quantitative performance and significantly reducing fine-grained detail distortion qualitatively.

### 2 Preliminary

- 2.1 Diffusion Model

Diffusion models (Ho et al., 2020; Rombach et al., 2022) have gained popularity in image generation models due to their stable training and exceptional generative capabilities. Among them, the Stable Diffusion (SD) series (e.g., SD1.5 (Rombach et al., 2022), SDXL (Podell et al., 2024)) is widely used. SD is a latent diffusion model (LDM), consisting of a variational autoencoder (VAE) (Kingma, 2013), a UNet (Ronneberger et al., 2015) ϵθ(·), and text encoders (Raffel et al., 2020; Radford et al., 2021). The VAE includes an encoder E(·) and a decoder D(·), which encodes input images from a pixel space into a latent space to reduce computational cost. During training, we first use the encoder E(·) of VAE to encode a given image into a latent feature z0. At a sampled diffusion time step t ∈ {1, · · · , T}, we apply a Gaussian noise ϵ ∼ N (0,1) to the latent feature z0 based on a noise scheduler, obtaining a noised latent feature zt. The UNet ϵθ(·) then denoises the noised latent feature zt under the control of the conditional feature c. Formally, the training loss can be formulated as:

Ldiffusion = Eϵ∼N(0,1),t[∥ϵ − ϵθ(zt,c, t)∥22], (1) where c can be either the language feature predicted by the text encoders or the visual feature derived from the reference image, which is the primary focus of our work.

- 2.2 Task Definition and Notations

Given a source person image Isrc ∈ RH×W×3 and a reference image Iref ∈ RH×W×3, controllable person image generation aims to generate a target person image Itgt ∈ RH×W×3, where H and W denote the height and width of the image. Specifically, in virtual try-on, Itgt represents the garment image Iref worn on the person in Isrc; while in pose transfer, Itgt represents the person in Iref adopting the pose from Isrc. During training, Isrc and Itgt are the same image. The data preprocessing for each task is defined as follows:

- • For virtual try-on, following Choi et al. (2024); Chong et al. (2024), we first extract a garment mask Im ∈ {0,1}H×W×1 from Isrc and construct a garment-agnostic image Iga ∈ RH×W×3 = Isrc ∗ (1 − Im). During training, we encode Isrc and Iga via the VAE encoder E(·) yields z0 and zga, and resize Im to latent space as zm. We add noise to z0 based on the sampled time step t which produces zt, and we channel concatenate it with zga and zm to construct zˆt.
- • For pose transfer, following Han et al. (2023), we first extract a DensePose image (Güler et al., 2018) Idp ∈ RH×W×3 from Isrc. During training, we encode Isrc with the VAE encoder E(·), obtaining z0, and resize the Idp to the latent resolution of zdp. We add noise to z0 at sampled time step t to produce zt, and we channel concatenate it with zdp to construct zˆt.

For the reference image Iref in both tasks, we encode it into a latent feature zref, and use that as a conditional feature within a diffusion loss defined in Eq. 1.

[Figure 6]

Figure3 An overview of our Leffa training pipeline for controllable person image generation. The left is our diffusion-based baseline; the right is our Leffa loss. Note that Isrc and Itgt are the same image during training.

### 3 Method

Our goal is to leverage diffusion-based methods to preserve fine-grained details while maintaining high overall image quality. To achieve this, we propose the Leffa loss that alleviates detail distortion through learning flow fields in attention. To validate the effectiveness of Leffa, we first construct a unified diffusion-based baseline suitable for both virtual try-on and pose transfer tasks in Sec. 3.1. We then introduce the formulation of Leffa loss in Sec. 3.2. Lastly, we explain its integration into the model training in Sec. 3.3.

- 3.1 Diffusion-based Baseline

We first design our diffusion-based baseline upon the SD (Rombach et al., 2022; Podell et al., 2024) to assess the effectiveness of our Leffa loss. Specifically, we modify SD in two steps as follows.

First, we duplicate the pre-trained SD UNet ϵθ(·) to create two separate UNets: i) a Generative UNet ϵθgen(·) for processing zˆt from the source image Isrc; and ii) a Reference UNet ϵθref(·) for processing zref from the reference image Iref. Since we only condition on images, we remove both the text encoders and the cross-attention layers used for text interaction in the Generative UNet ϵθgen(·) and Reference UNet ϵθref(·). Notably, both UNets are fully trainable, which we found to be effective in our early exploration.

Second, to condition the generation of the target person image Itgt on Isrc and Iref, we employ spatial concatenated self-attention. For l-th attention layer, we define the generative features fed into the Generative UNet ϵθgen(·)

- as Fgenl ∈ Rnl×dl, and the reference features in the Reference UNet ϵθref(·) as Frefl ∈ Rnl×dl, where nl = hl × wl with hl and wl as spatial dimensions, and dl as the feature dimension. We concatenate the Frefl with Fgenl along

the spatial dimension, yielding concatenated feature Fcatl ∈ R2nl×dl, and jointly feed it into the Generative UNet’s self-attention layer. We retain only the first half of the output corresponding to Fgenl for further processing. This process is repeated across all self-attention layers.

Using SD’s training method and the loss function in Eq. 1, our clean and simple diffusion-based baseline already achieves performance comparable to current state-of-the-arts (Kim et al., 2024; Choi et al., 2024; Chong et al., 2024; Wan et al., 2024; Bhunia et al., 2023; Han et al., 2023; Lu et al., 2024; Pham et al., 2024), while relying on no additional auxliary models (e.g., CLIP (Radford et al., 2021), DINOv2 (Oquab et al., 2024)) and/or complicated training techniques (e.g., DREAM (Zhou et al., 2024)). However, this diffusion-based baseline still suffers from detail distortion, which will be addressed by the Leffa loss, introduced in the next section.

- 3.2 Leffa: Learning Flow Fields in Attention

Our goal is to preserve fine-grained details without textural distortion. Detail distortion arises when the target query in the attention layer fails to attend to the corresponding region of the reference key correctly. To address this, we propose the Leffa loss, which explicitly guides the target query to spatially focus on the correct region of the reference key through supervision by learning flow fields in attention.

Specifically, in the lth attention layer of the Generative UNet ϵθgen(·), we compute the attention map Al by using Fgenl as the query Q and Frefl as the key K, formulated as the dot-product attention defined in Vaswani et al. (2017),

Al = softmax

QK⊤ √d

/τ , (2)

where τ is the temperature coefficient, and d is the token size of the query Q. We then construct the Leffa loss based on the attention map Al.

For each attention map Al, we aim to encourage all query tokens in Fgenl to attend to the correct regions on the reference key Frefl . Following Darcet et al. (2024), we also include additional learnable tokens to allow flexible attention on other regions. This design, rather than enforcing strict spatial attention alignment in all attention heads, we ensure that, on average, attention maps across different heads focus on the correct region. To achieve this, we average Al across the head dimension, resulting in Aˆl ∈ Rnl×nl.

Next, we convert Aˆl to a flow field representing the spatial correspondence between Isrc and Iref. To achieve this, we construct a normalized coordinate map Cl ∈ Rnl×2, where, reshaped as Rhl×wl×2, within the top-left coordinate (0,0) has the value of [−1, −1] and the bottom-right coordinate (h − 1, w − 1) has the value of [1,1]. We then multiply the attention map Aˆl with the coordinate map Cl to obtain the flow field Fl ∈ Rnl×2, formulated as

Fl = Aˆl · Cl, (3)

which indicates the coordinates of the region on the reference feature Frefl that each target query token attends to.

We then use this flow field Fl as a coordinate mapping and apply the grid sampling operation to warp the reference image Iref into the target image Itgt. Considering our method is based on the latent diffusion model, the resolution of the attention map is significantly lower than the original input image. To provide precise pixel-level training supervision, we upsample the flow field in lth attention layer Fl ∈ Rhl×wl×2 to the original image resolution using bilinear interpolation, resulting in the upsampled flow field Fupl ∈ RH×W×2. We then use this upsampled flow field Fupl to warp the reference image Iref, producing the warped image Iwarpl ∈ RH×W×3.

Finally, we compute the L2 loss between Iwarpl and the corresponding region of the target image Itgt, which defines our Leffa loss formulation:

Lleffa = ∑lL=1∥Itgt ∗ Im − Iwarpl ∗ Im∥22. (4)

This loss supervises the attention map without additional inputs or parameters, ensuring each target query token attends to the correct reference region and preserving fine-grained detail consistency in the generated images.

Note that, we apply Leffa loss only on selected L attention layers, explained next.

- 3.3 Model Training with Leffa Loss

In this section, we detail the application of Leffa loss to the diffusion-based baseline, guiding the target query to attend to the correct reference key and thus alleviating fine-grained detail distortion.

We follow Zhu et al. (2024) to adopt progressive training with Leffa loss, applied in the final finetuning phase to avoid early-stage performance degradation. The model is initially trained at a low resolution (e.g., 256, 512),

then at a higher resolution (e.g., 1024). In the final phase, we fine-tune at 1024 resolution using a combined loss:

Lfinetune = Ldiffusion + λleffaLleffa, (5)

where λleffa is the loss weight of Lleffa. For optimal model performance with Leffa loss, we also address the following considerations.

Attention Layer Selection. Given that current LDM-based methods perform feature interaction between the target and reference images in the latent space, the resulting attention maps tend to have relatively low resolution. For such low-resolution attention maps, accurately warping the reference image Iref is not feasible. Therefore, we set a resolution threshold θresolution = h/H (e.g., 1/32), representing the ratio of the attention map size to the original image size (see Fig. 5). Only attention layers with a resolution greater than θresolution participate in the computation of Lleffa.

Timestep Selection. When the time step t is large, substantial noise is added to the image, making accurate feature interactions between the target and reference difficult due to the excessive noise, which hinders attending to the right semantics of the corresponding regions. Therefore, Leffa loss is not suitable for large time steps (see Fig. 5). We thereby set a timestep threshold θtimestep (e.g., 500 when T = 1000), and during training, only time steps smaller than θtimestep are included in the calculation of Lleffa.

Temperature Selection. Different temperatures adjust the smoothness of the attention map produced by softmax. A larger temperature results in a smoother attention map, allowing the target query token to attend to a broader range of reference features Frefl , which helps the model more easily learn the correct reference regions and increases tolerance to errors (see Fig. 5). Therefore, when using Leffa loss, we apply a relatively large temperature τ (i.e., 2.0).

### 4 Experiments

- 4.1 Datasets, Metrics and Implementation Details

Datasets. To evaluate our method’s capability in appearance and pose control, we conduct experiments on the virtual try-on and pose transfer tasks, respectively. For virtual try-on, we use the VITON-HD (Choi et al., 2021) and DressCode (Morelli et al., 2022) datasets; for pose transfer, we use DeepFashion (Liu et al., 2016). Please refer to the supplementary material for more details.

Metrics. In virtual try-on, following previous methods (Zhu et al., 2023; Gou et al., 2023; Cui et al., 2023, 2024; Zeng et al., 2024; Ning et al., 2024; Kim et al., 2024; Xu et al., 2024; Choi et al., 2024; Yang et al., 2024a; Chong et al., 2024; Yang et al., 2024b), we evaluate the model under two settings: paired, where the garment in the person image same as the garment image; and unpaired, where the garments differ between the two images. For both settings, we use FID (Soloveitchik et al., 2021) and KID (Bińkowski et al., 2018) as evaluation metrics. Additionally in the paired setting, we include SSIM (Wang et al., 2004) and LPIPS (Zhang et al., 2018) to evaluate image-level metrics with ground-truth data. In pose transfer, we follow the previous methods (Bhunia et al., 2023; Han et al., 2023; Lu et al., 2024; Pham et al., 2024) to use FID (Soloveitchik et al., 2021), calculated between the generated test images and all training images; SSIM (Wang et al., 2004) and LPIPS (Zhang et al., 2018), computed between the generated images and all images in the test set, as metrics.

ImplementationDetails. In this paper, we use SD1.5 (Rombach et al., 2022) to build our diffusion-based baseline. We adopt the progressive training strategy to train the model. For virtual try-on, the model is first trained at 512 × 384 resolution for 10k steps with a batch size of 256, then at 1024 × 768 for 36k steps with a batch size of 64. Finally, we finetune the model for an additional 10k steps, incorporating Lleffa to enhance its capabilities. For pose transfer, we start at a resolution of 256 × 176 with a batch size of 256 for 80k steps, then increase to 512 × 352 for another 80k steps. Finally, we finetune for 24k steps with Lleffa. AdamW optimizer (Loshchilov, 2017) with a learning rate of 10−5 is applied to all training stages. The hyperparameters for Lleffa are set as follows: λleffa = 10−3, θresolution = 1/32, θtimestep = 500 and τ = 2.0.

[Figure 7]

- Figure 4 Qualitative visual results comparison with other methods. The input person image for the pose transfer is generated using our method in the virtual try-on. The visualization results demonstrate that our method not only generates high-quality images but also greatly reduces the distortion of fine-grained details. Please zoom in for better viewing.

- 4.2 Quantitative and Qualitative Results

Quantitative Comparison. We conduct experiments on both virtual try-on and pose transfer tasks. For virtual try-on, we conduct experiments on VITON-HD (Choi et al., 2021) and DressCode (Morelli et al., 2022), with results shown in Tabs. 1 and 2, respectively. In VITON-HD, our method significantly outperforms previous methods, achieving an FID reductions of -0.88/-0.5 in the paired/unpaired setting. In DressCode, our method also surpasses existing methods across all garment categories (upper body, lower body, and dresses), with an FID reductions of -1.93/-1.66 in the paired/unpaired setting. We also conduct evaluations for each garment category, please refer to the supplementary material. For pose transfer, we conduct experiments on DeepFashion (Liu et al., 2016) at different resolutions, significantly outperforming previous methods. As shown in Tab. 3, our method achieves -0.54 and -1.61 reductions in FID at 256×176 and 512×352, respectively. While our model shows a minor SSIM gap over CFLD (Lu et al., 2024), visual results and human study indicate substantial improvement over prior methods (see Fig. 7 and supplementary material).

Qualitative Comparison. Fig. 4 shows that our method significantly improves detail preservation compared to previous methods. For virtual try-on, in the first row, our method accurately preserves the texture details of the colored stripes and maintains the correct color order. In the second row, it generates very small text accurately, whereas other methods, despite achieving reasonable overall quality, exhibit varying levels of text distortion, failing to convey the correct meaning. In the third row, our method generates evenly spaced

paired unpaired

Method

FID ↓ KID ↓ SSIM ↑ LPIPS ↓ FID ↓ KID ↓

FS-VTON (He et al., 2022) 6.17 0.69 0.886 0.074 9.91 1.10 HR-VITON (Lee et al., 2022) 11.38 3.52 0.865 0.122 13.27 4.38 GP-VTON (Xie et al., 2023b) 6.03 0.60 0.885 0.080 9.37 0.79 LADI-VTON (Morelli et al., 2023) 6.60 1.09 0.866 0.094 9.35 1.66 DCI-VTON (Gou et al., 2023) 5.52 0.41 0.882 0.080 8.75 0.68 SD-VTON (Shim et al., 2024) 6.98 1.00 0.874 0.101 9.85 1.39 StableVITON (Kim et al., 2024) 8.23 0.49 0.888 0.073 - IDM-VTON (Choi et al., 2024) 5.76 0.73 0.850 0.063 9.84 1.12 OOTDiffusion (Xu et al., 2024) 9.30 4.09 0.819 0.088 12.41 4.68 CatVTON (Chong et al., 2024) 5.42 0.41 0.870 0.057 9.02 1.09

Leffa (Ours) 4.54 0.05 0.899 0.048 8.52 0.32

- Table 1 Quantitative results comparison with other methods on the VITON-HD dataset for virtual try-on. Note bold indicates the best result, and underline indicates the second-best result.

Method

paired unpaired

FID ↓ KID ↓ SSIM ↑ LPIPS ↓ FID ↓ KID ↓

GP-VTON (Xie et al., 2023b) 9.93 4.61 0.771 0.180 12.79 6.63 LADI-VTON (Morelli et al., 2023) 9.56 4.68 0.766 0.237 10.68 5.79 IDM-VTON (Choi et al., 2024) 6.82 2.92 0.879 0.056 9.55 4.32 OOTDiffusion (Xu et al., 2024) 4.61 0.96 0.885 0.053 12.57 6.63 CatVTON (Chong et al., 2024) 3.99 0.82 0.892 0.045 6.14 1.40

Leffa (Ours) 2.06 0.07 0.924 0.031 4.48 0.62

- Table 2 Quantitative results comparison with other methods on the DressCode dataset (all garment categories) for virtual try-on.

Resolution Method FID ↓ SSIM ↑ LPIPS ↓

256 × 176

Def-GAN (Siarohin et al., 2018) 18.46 0.679 0.233 PATN (Zhu et al., 2019) 20.75 0.671 0.256 ADGAN (Men et al., 2020) 14.46 0.672 0.228 GFLA (Ren et al., 2020) 10.57 0.707 0.223 PISE (Zhang et al., 2021) 13.61 0.663 0.206 DPTN (Zhang et al., 2022) 11.39 0.711 0.193 CASD (Zhou et al., 2022) 11.37 0.725 0.194 NTED (Ren et al., 2022) 8.68 0.718 0.175 PIDM (Bhunia et al., 2023) 6.37 0.731 0.168 PoCoLD (Han et al., 2023) 8.07 0.731 0.164 CFLD (Lu et al., 2024) 6.80 0.737 0.152 X-MDPT-L (Pham et al., 2024) 6.25 0.729 0.167 Leffa (Ours) 5.71 0.732 0.114

512 × 352

CocosNet-v2 (Zhou et al., 2021) 13.33 0.724 0.226 NTED (Ren et al., 2022) 7.78 0.738 0.198 PIDM (Bhunia et al., 2023) 5.84 0.742 0.177 PoCoLD (Han et al., 2023) 8.42 0.743 0.192 CFLD (Lu et al., 2024) 7.14 0.748 0.152 X-MDPT-L (Pham et al., 2024) 5.93 0.742 0.179

Leffa (Ours) 4.23 0.755 0.119

- Table 3 Quantitative results comparison with other methods on the DeepFashion dataset for pose transfer.

buttons in the correct quantity, and in the fifth row, it produces a more realistic, gauzy fabric effect. For pose transfer, in the first to fourth rows, our method better predicts a plausible side/back-view image based on frontal information. Other rows also show that our method maintains fine details more effectively when altering the pose. The last row demonstrates that our method can generate vivid full-body images from half-body images.

Paired Unpaired

Paired Unpaired

Paired Unpaired

Paired Unpaired

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

- 4
- 5
- 6
- 7
- 8
- 9
- 10

- 4
- 5
- 6
- 7
- 8
- 9
- 10

- 4
- 5
- 6
- 7
- 8
- 9
- 10

- 4
- 5
- 6
- 7
- 8
- 9
- 10

FID

FID

FID

FID

250 500 750 1000

0.1 0.5 1.0 2.0 5.0

1/8 1/16 1/32 1/64

0 10−5 10−4 10−3 10−2 10−1 1.0

temperature τ

θtimestep

λlef f a

θresolution

- Figure 5 Ablation study for (a) Leffa loss weight λleffa, (b) resolution threshold θresolution, (c) timestep threshold θtimestep, (d) temperature τ on VITON-HD dataset of virtual try-on.

Method Lleffa paired unpaired FID ↓ KID ↓ SSIM ↑ LPIPS ↓ FID ↓ KID ↓

✗ 5.76 0.73 0.850 0.063 9.84 1.12 ✓ 5.12 0.35 0.876 0.054 9.24 0.87

IDM-VTON

✗ 5.42 0.41 0.870 0.057 9.02 1.09 ✓ 4.86 0.28 0.886 0.056 8.77 0.58

CatVTON

✗ 5.31 0.30 0.885 0.058 9.38 0.91 ✓ 4.54 0.05 0.899 0.048 8.52 0.32

Ours

- Table 4 Applying our Leffa loss Lleffa consistently enhances performance across different diffusion-based methods.

- 4.3 Generalization to Other Diffusion Models

The proposed Leffa loss is model-agnostic and can be applied to different attention-based methods. We evaluate this by adding Leffa loss to two prior art diffusion-based methods (i.e., IDM-VTON (Choi et al., 2024) and CatVTON (Chong et al., 2024)) on the VITON-HD (Choi et al., 2021) dataset. Since both methods also use attention mechanism to interact with the reference image, our Leffa loss can be seamlessly integrated without introducing extra trainable parameters. As shown in Tab. 4, our method achieves FID reductions of -0.64/-0.6 for IDM-VTON and -0.56/-0.25 for CatVTON in paired/unpaired settings, confirming its generalization ability.

- 4.4 Ablation Study

To further validate the effectiveness of our method, we conduct ablation experiments on the VITON-HD dataset.

EffectofLleffa. We first ablate Leffa loss to evaluate the benefits of learning flow fields in attention. As shown in Tab. 4, all metrics degrade without Lleffa. Specifically, the FID worsens by +0.77/+0.86 in the paired/unpaired setting, clearly demonstrating the contribution of our Leffa loss to improving overall generation quality. We further present visualizations of attention maps using Leffa in Fig. 6, which clearly show that the attended regions (highlighted in red) accurately correspond to the desired reference regions.

We additionally conduct a list of ablations to search the optimal training hyperparameters for our Leffa loss. Effect of Different λleffa. In Fig. 5, we find that by increasing λleffa from 0.0 to 10−3 improves performance, but further increasing to a degradation in performance. This suggests that while the Leffa loss effectively guides attention, overly strong supervision can hinder generalization, ultimately degrading generation quality.

Effect of Resolution Threshold θresolution. In Fig. 5, we observe that the optimal performance appears when θresolution is within 1/32 and 1/16. Otherwise, attention maps are either too large (resulting in too few attention layers) or too small to focus on the correct regions.

Effect of Timestep threshold θtimestep. When the sampled timestep t is large, the target query would include too much noise, making it harder for the model to recognize the correct reference key regions. In Fig. 5, we

observe that the optimal performance is achieved at θresolution = 500. Increasing it to 750 or 1000 results in degraded performance even compared to one without using Lleffa.

Effect of Temperature τ. Adjusting the temperature τ controls the range of focus that the target query has on the reference key. As shown in Fig. 5, we test temperatures from 0.1 to 5.0 and find optimal performance at

[Figure 8]

- Figure6 Visualization of feature maps to assess the impact of our Leffa loss Lleffa. With our Leffa loss added, our method not only maintains overall generation quality but also more accurately preserves fine-grained details. Additionally, attention map visualizations indicate that, with our loss, the target query focuses more precisely on the correct reference region.

###### VITON-HD DressCode DeepFashion

|72%<br><br>5%<br><br>18%<br><br>5%<br><br>|
|---|

|86%<br><br>3% 6% 5%<br><br>|
|---|

|69%<br><br>10%<br><br>14%<br><br>7%<br><br>|
|---|

CatVTON IDM-VTON OOTDiffusion Leﬀa

CatVTON IDM-VTON OOTDiffusion Leﬀa

CFLD NTED PIDM Leﬀa

- Figure 7 Preference percentage via human studies. Our Leffa consistently outperforms previous methods across datasets.

τ = 2.0. At τ = 0.1, the attention map becomes overly sparse, making optimization difficult; while at τ = 5.0, the attention map is overly dense, making it difficult to focus on the correct regions and eventually resulting in a significant performance drop.

- 4.5 Human Study

We have observed that some evaluation metrics, such as FID, are highly sensitive to implementation details (Heusel et al., 2017; Kynkäänniemi et al., 2022; Peebles and Xie, 2023) and may not accurately reflect image generation quality. To provide a more comprehensive evaluation, we conduct a human study comparing our Leffa with other methods.

Specifically, VITON-HD (Choi et al., 2021) and DressCode (Morelli et al., 2022) are selected for virtual try-on , where we compare against OOTDiffusion (Xu et al., 2024), IDM-VTON (Choi et al., 2024), and CatVTON (Chong et al., 2024); while DeepFashion (Liu et al., 2016) is selected for pose transfer, comparing against PIDM (Bhunia et al., 2023), NTED (Ren et al., 2022), and CFLD (Lu et al., 2024). We invite 50 participants to evaluate 90 cases (30 randomly selected cases from each dataset), with each participant selecting the image with the best generation quality from each test set. As shown in Fig. 7, our method receives the highest number of preferred selections, significantly outperforming previous methods on both virtual try-on and pose transfer tasks.

- 5 Related Work

- 5.1 Controllable Person Image Generation

Appearance Control. Virtual try-on seeks to automate outfit changes in person images without distorting garment details, which is a longstanding challenge (Yang et al., 2020; Choi et al., 2021; Ge et al., 2021a,b; He

- et al., 2022). Recently, diffusion-based methods have emerged as the de facto solutions (Zhu et al., 2023; Kim et al., 2024; Xu et al., 2024; Wan et al., 2024). Some pioneering works (Morelli et al., 2023; Gou et al., 2023; Zeng et al., 2024) enhance fine-grained detail preservation by incorporating additional modules, such as a textual inversion module with CLIP (Radford et al., 2021) for enhanced conditions, a warping model to merge garments with garment-agnostic person images, and ControlNet (Zhang et al., 2023) with DINOv2 (Oquab et al., 2024) to extract garment details. Further improvements (Kim et al., 2024; Choi et al., 2024; Xu et al., 2024; Wan et al., 2024) leverage complex model structure designs, external models and textual information for enhanced performance. Recently, Chong et al. (2024) achieve detail preservation without extra models but requires complex training strategies (e.g., DREAM (Zhou et al., 2024)) to reduce the training instability.

Pose Control. Similarly, pose transfer presents a challenge in capturing the intricate structure of the pose transformation while preserving the fine-grained details of the textures (Ma et al., 2017; Siarohin et al., 2018; Li et al., 2019; Zhu et al., 2019; Ren et al., 2020; Men et al., 2020; Zhang et al., 2021; Albahar et al., 2021; Zhou et al., 2021; Sanyal et al., 2021; Zhang et al., 2022; Zhou et al., 2022; Ren et al., 2022). Bhunia et al. (2023) first leverage diffusion models with a texture diffusion module for quality enhancement. Han et al. (2023) further propose a pose-constrained attention module and replace skeletons with dense pose (Güler et al., 2018) for more accurate pose conditioning. Lu et al. (2024) introduce a perception-refined decoder to progressively generate person images from coarse to fine while employing hybrid-granularity attention to enhance detail. Recently, Pham et al. (2024) replace UNet (Ronneberger et al., 2015) with the masked diffusion transformer (Peebles and Xie, 2023; Gao et al., 2023), achieving efficient and high-quality generation.

Although the existing methods have achieved significant progress on both virtual try-on and pose transfer tasks, most of them rely on complex module designs and extra models for fine-grained detail preservation. In contrast, we introduce a method that ensures stable training without complex designs, reducing detail distortion without adding any inference cost or extra parameters.

- 5.2 Attention in Controllable Generation

The attention mechanism, a crucial module in generative models, has been extensively studied and shown to significantly enhance model generation quality when effectively guided (Liu et al., 2024; Xie et al., 2023a; Phung et al., 2024; Guo and Lin, 2024; Chefer et al., 2023; Cao et al., 2023; Xiao et al., 2024; Hertz et al., 2023). How to improve attention mechanism has also been studied in controllable person image generation. Some methods (Ren et al., 2020; Zhou et al., 2021; Ren et al., 2022; Tang et al., 2020) propose customized attention mechanisms. For instance, Ren et al. (2020) develop a differentiable global-local attention to reassemble inputs at the feature level, while Zhou et al. (2021) leverage iterative patch matching to enhance attention quality. Ren et al. (2022) apply neural texture extraction and distribution, using semantic filters to improve attention maps and image quality. On the other hand, several methods directly optimize attention maps via supervision.

- Han et al. (2023) propose a pose-constrained attention loss to model appearance-pose interactions. Kim et al.

(2024) introduce an attention total variation loss to enhance focus on garment regions, though it lacks explicit guidance to attend to correct regions. In contrast, we propose a regularization loss that guides attention maps to attend to correct regions without additional annotations, enhancing attention quality and reducing fine-grained detail distortion.

- 6 Conclusion

This paper introduces a regularization loss, learning flow fields in attention (Leffa), to enhance controllable person image generation. Our approach not only preserves high overall image quality but also mitigates fine-grained detail distortion. We validate the effectiveness and generalization ability of Leffa by integrating it with different diffusion-based methods, achieving significant qualitative and quantitative improvements in

both virtual try-on and pose transfer tasks. Future work will focus on developing a unified model that can simultaneously control appearance and pose.

- 7 Limitation

While Leffa significantly improves controllable person image generation in appearance and pose control, it has several limitations. First, it requires multi-stage training with the Leffa loss applied only in the final stage. In future work, we aim to design a single-stage model to simplify the training process. Second, appearance control relies on garment segmentation, which impacts performance when segmentation is inaccurate. We plan to develop a mask-free approach to ensure high quality generation and preserve fine-grained details without distortion. Third, our method struggles to preserve extremely fine-grained details, such as small text, due to the 8× resolution compression brought by the latent encoder in the latent diffusion model. It is worth noting that the issues mentioned above are not unique to our method but are also present in other methods.

- 8 Statement

All experiments, data collection, and processing activities were conducted at King’s College London. Meta was involved solely in an advisory role and NO experiments, data collection or processing activities were conducted on Meta infrastructure.

References

Badour Albahar, Jingwan Lu, Jimei Yang, Zhixin Shu, Eli Shechtman, and Jia-Bin Huang. Pose with style: Detail-preserving pose-guided image synthesis with conditional stylegan. ACM TOG, 2021.

Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Jorma Laaksonen, Mubarak Shah, and

Fahad Shahbaz Khan. Person image synthesis via denoising diffusion model. In CVPR, 2023. Mikołaj Bińkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. In ICLR, 2018. Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual

self-attention control for consistent image synthesis and editing. In ICCV, 2023. Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM TOG, 2023. Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In CVPR, 2021. Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In ECCV, 2024.

Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models. arXiv preprint arXiv:2407.15886, 2024.

Aiyu Cui, Sen He, Tao Xiang, and Antoine Toisoul. Learning garment densepose for robust warping in virtual try-on. arXiv preprint arXiv:2303.17688, 2023.

Aiyu Cui, Jay Mahajan, Viraj Shah, Preeti Gomathinayagam, and Svetlana Lazebnik. Street tryon: Learning in-the-wild virtual try-on from unpaired person images. In CVPRW, 2024.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In ICLR, 2024. Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image

synthesizer. In ICCV, 2023. Chongjian Ge, Yibing Song, Yuying Ge, Han Yang, Wei Liu, and Ping Luo. Disentangled cycle consistency for highlyrealistic virtual try-on. In CVPR, 2021a. Yuying Ge, Yibing Song, Ruimao Zhang, Chongjian Ge, Wei Liu, and Ping Luo. Parser-free virtual try-on via distilling appearance flows. In CVPR, 2021b.

Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In ACM MM, 2023.

Rıza Alp Güler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In CVPR, 2018.

Qin Guo and Tianwei Lin. Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation. In CVPR, 2024.

Xiao Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, and Tao Xiang. Controllable person image synthesis with pose-

constrained latent diffusion. In ICCV, 2023. Sen He, Yi-Zhe Song, and Tao Xiang. Style-based global appearance flow for virtual try-on. In CVPR, 2022. Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image

editing with cross-attention control. In ICLR, 2023. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two

time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence

with latent diffusion model for virtual try-on. In CVPR, 2024. Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. Tuomas Kynkäänniemi, Tero Karras, Miika Aittala, Timo Aila, and Jaakko Lehtinen. The role of imagenet classes in

fr\’echet inception distance. arXiv preprint arXiv:2203.06026, 2022.

Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In ECCV, 2022.

Yining Li, Chen Huang, and Chen Change Loy. Dense intrinsic appearance flow for human pose transfer. In CVPR, 2019. Haozhe Liu, Wentian Zhang, Jinheng Xie, Francesco Faccio, Mengmeng Xu, Tao Xiang, Mike Zheng Shou, Juan-Manuel

Perez-Rua, and Jürgen Schmidhuber. Faster diffusion via temporal attention decomposition. arXiv e-prints, pages arXiv–2404, 2024.

Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and

retrieval with rich annotations. In CVPR, 2016. I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. Yanzuo Lu, Manlin Zhang, Andy J Ma, Xiaohua Xie, and Jianhuang Lai. Coarse-to-fine latent diffusion for pose-guided

person image synthesis. In CVPR, 2024. Liqian Ma, Xu Jia, Qianru Sun, Bernt Schiele, Tinne Tuytelaars, and Luc Van Gool. Pose guided person image generation. NeurIPS, 2017. Yifang Men, Yiming Mao, Yuning Jiang, Wei-Ying Ma, and Zhouhui Lian. Controllable person image synthesis with attribute-decomposed gan. In CVPR, 2020. Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: Highresolution multi-category virtual try-on. In CVPR, 2022. Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In ACM MM, 2023. Shuliang Ning, Duomin Wang, Yipeng Qin, Zirong Jin, Baoyuan Wang, and Xiaoguang Han. Picture: Photorealistic virtual try-on from unconstrained designs. In CVPR, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. TMLR, 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. Trung X Pham, Kang Zhang, and Chang D Yoo. Cross-view masked diffusion transformers for person image synthesis.

In ICML, 2024.

Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. In CVPR, 2024.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICLR, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020.

Yurui Ren, Xiaoming Yu, Junming Chen, Thomas H Li, and Ge Li. Deep image spatial transformation for person image generation. In CVPR, 2020.

Yurui Ren, Xiaoqing Fan, Ge Li, Shan Liu, and Thomas H Li. Neural texture extraction and distribution for controllable person image synthesis. In CVPR, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015.

Soubhik Sanyal, Alex Vorobiov, Timo Bolkart, Matthew Loper, Betty Mohler, Larry S Davis, Javier Romero, and Michael J Black. Learning realistic human reposing using cyclic self-supervision with 3d shape, pose, and appearance consistency. In ICCV, 2021.

Sang-Heon Shim, Jiwoo Chung, and Jae-Pil Heo. Towards squeezing-averse virtual try-on via sequential deformation. In AAAI, 2024.

Aliaksandr Siarohin, Enver Sangineto, Stéphane Lathuiliere, and Nicu Sebe. Deformable gans for pose-based human image generation. In CVPR, 2018.

Michael Soloveitchik, Tzvi Diskin, Efrat Morin, and Ami Wiesel. Conditional frechet inception distance. arXiv preprint arXiv:2103.11521, 2021.

Ke Sun, Jian Cao, Qi Wang, Linrui Tian, Xindi Zhang, Lian Zhuo, Bang Zhang, Liefeng Bo, Wenbo Zhou, Weiming Zhang, et al. Outfitanyone: Ultra-high quality virtual try-on for any clothing and any person. arXiv preprint arXiv:2407.16224, 2024.

- Hao Tang, Song Bai, Li Zhang, Philip HS Torr, and Nicu Sebe. Xinggan for person image generation. In ECCV, 2020.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017.

Siqi Wan, Yehao Li, Jingwen Chen, Yingwei Pan, Ting Yao, Yang Cao, and Tao Mei. Improving virtual try-on with garment-focused diffusion models. In ECCV, 2024.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 2004.

Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. IJCV, 2024.

Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In ICCV, 2023a.

Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gp-vton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In CVPR, 2023b.

Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arXiv preprint arXiv:2403.01779, 2024.

Han Yang, Ruimao Zhang, Xiaobao Guo, Wei Liu, Wangmeng Zuo, and Ping Luo. Towards photo-realistic virtual try-on by adaptively generating-preserving image content. In CVPR, 2020.

Xu Yang, Changxing Ding, Zhibin Hong, Junhao Huang, Jin Tao, and Xiangmin Xu. Texture-preserving diffusion models for high-fidelity virtual try-on. In CVPR, 2024a.

Zhaotong Yang, Zicheng Jiang, Xinzhe Li, Huiyu Zhou, Junyu Dong, Huaidong Zhang, and Yong Du. D4-vton: Dynamic semantics disentangling for differential diffusion based virtual try-on. In ECCV, 2024b.

Jianhao Zeng, Dan Song, Weizhi Nie, Hongshuo Tian, Tongtong Wang, and An-An Liu. Cat-dm: Controllable accelerated virtual try-on with diffusion model. In CVPR, 2024.

Jinsong Zhang, Kun Li, Yu-Kun Lai, and Jingyu Yang. Pise: Person image synthesis and editing with decoupled gan. In CVPR, 2021.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.

Pengze Zhang, Lingxiao Yang, Jian-Huang Lai, and Xiaohua Xie. Exploring dual-task correlation for pose guided person image generation. In CVPR, 2022.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Jinxin Zhou, Tianyu Ding, Tianyi Chen, Jiachen Jiang, Ilya Zharkov, Zhihui Zhu, and Luming Liang. Dream: Diffusion rectification and estimation-adaptive models. In CVPR, 2024.

Xingran Zhou, Bo Zhang, Ting Zhang, Pan Zhang, Jianmin Bao, Dong Chen, Zhongfei Zhang, and Fang Wen. Cocosnet v2: Full-resolution correspondence learning for image translation. In CVPR, 2021.

Xinyue Zhou, Mingyu Yin, Xinyuan Chen, Li Sun, Changxin Gao, and Qingli Li. Cross attention based style distribution for controllable person image synthesis. In ECCV, 2022.

Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In CVPR, 2023.

Luyang Zhu, Yingwei Li, Nan Liu, Hao Peng, Dawei Yang, and Ira Kemelmacher-Shlizerman. M&m vto: Multi-garment virtual try-on and editing. In CVPR, 2024.

Zhen Zhu, Tengteng Huang, Baoguang Shi, Miao Yu, Bofei Wang, and Xiang Bai. Progressive pose attention transfer for person image generation. In CVPR, 2019.

## Appendix

In the supplementary material, we provide additional experimental details along with qualitative and quantitative results. Additionally, we discuss our diffusion-based baseline and Leffa loss, along with their limitations.

- A Experimental Details

A.1 Datasets

In this section, we provide a detailed introduction to the three datasets used in our study.

VITON-HD (Choi et al., 2021) dataset is the most commonly used dataset for virtual try-on task. The training set contains 11,647 person and garment image pairs, and the test set contains 2,032 pairs. All images are front-view, upper-body garments with a resolution of 1024 × 768.

DressCode (Morelli et al., 2022) dataset is composed of various types of garments, comprising 48,392 person and garment image pairs in the training set. This includes 13,563 upper-body, 7,151 lower-body, and 27,678 dress garment pairs. The test set contains 5,400 pairs, evenly distributed across 1,800 pairs each for upper-body, lower-body, and dress garments. All images have a resolution of 1024 × 768.

DeepFashion (Liu et al., 2016) dataset includes high resolution 52,712 person images in the fashion domain. Following Zhu et al. (2019), we split the dataset into training and test subsets with 101,966 and 8,570 pairs, respectively. Each pair includes the same person in the same garment but with different poses.

- B More Results for DressCode Dataset

To further validate our performance, we conduct separate evaluations on different garment categories for the DressCode dataset. In Tab. 5, results show that our method significantly outperforms previous methods across all garment categories. Specifically, for the upper body category, it achieves an FID reductions of -2.33 (paired) and -0.91 (unpaired); for the lower body category, -2.94 (paired) and -2.55 (unpaired); and for dresses, -1.25 (paired) and -1.71 (unpaired).

- C More Ablation Studies

Unless stated otherwise, the ablation studies are conducted on the VITON-HD dataset for the virtual try-on task, in alignment with the main paper.

Effect ofLleffa. To validate the effectiveness of Leffa loss, we conduct ablation studies on the DressCode dataset for virtual try-on and the DeepFashion dataset for pose transfer. i) As shown in Tab. 6, our Leffa reduces FID by -1.58 in the paired setting and -1.5 in the unpaired setting. ii) Tab. 7 further shows an FID reduction of -1.49 on pose transfer task. These results confirm that Leffa loss significantly improves controllable person image generation, enhancing both appearance and pose control.

Qualitative impact of our Leffa loss Lleffa. In Fig. 8, we visualize attention maps with varying λleffa to assess its impact on model training. The first row shows generated results with different λleffa values, with λleffa = 0 indicating no Lleffa. Rows 2 to 5 highlight reference key regions attended by the target query, marked by arrows of various colors. Without Lleffa, attention is dispersed. Increasing λleffa improves focus, guiding the query to correct regions. However, when λleffa > 10−3, the attention becomes overly narrow and less accurate, hindering image generation.

Effect of averaging attention map across multi-head. Inspired by Darcet et al. (2024), we average the attention maps across all heads before computing the Leffa loss. As shown in Tab. 8 (Leffa w/o average A), computing the loss for each head individually degrades performance, emphasizing the role of redundancy in attention maps for better generalization.

paired unpaired

Method

FID ↓ KID ↓ SSIM ↑ LPIPS ↓ FID ↓ KID ↓

upper body FS-VTON (He et al., 2022) 11.29 3.65 0.941 0.035 16.34 5.93

- HR-VITON (Lee et al., 2022) 15.36 5.27 0.916 0.071 16.82 5.70 GP-VTON (Xie et al., 2023b) 7.38 0.74 0.945 0.039 12.21 1.19 LADI-VTON (Morelli et al., 2023) 9.53 0.20 0.928 0.049 13.26 2.67 DCI-VTON (Gou et al., 2023) 7.47 1.07 0.942 0.041 11.64 0.86 StableVITON (Kim et al., 2024) 9.94 0.12 0.937 0.039 - OOTDiffusion (Xu et al., 2024) 11.03 0.29 - - - -

- Leffa (Ours) 5.05 0.02 0.949 0.021 10.73 0.77 lower body

FS-VTON (He et al., 2022) 11.65 3.82 0.934 0.053 22.43 9.81 HR-VITON (Lee et al., 2022) 11.41 3.20 0.937 0.045 16.39 4.31 GP-VTON (Xie et al., 2023b) 7.73 0.71 0.938 0.042 16.70 2.89

- LADI-VTON (Morelli et al., 2023) 8.52 1.04 0.922 0.051 14.80 3.13

- DCI-VTON (Gou et al., 2023) 7.97 0.96 0.939 0.045 15.45 1.60 Leffa (Ours) 4.79 0.05 0.941 0.024 12.25 1.66

dresses FS-VTON (He et al., 2022) 13.04 4.44 0.888 0.070 20.95 8.96 HR-VITON (Lee et al., 2022) 16.82 4.89 0.865 0.113 18.81 5.41 GP-VTON (Xie et al., 2023b) 7.44 0.32 0.881 0.073 12.64 1.83 LADI-VTON (Morelli et al., 2023) 9.07 1.12 0.868 0.089 13.40 2.50

- DCI-VTON (Gou et al., 2023) 8.48 1.08 0.887 0.070 12.35 1.36

- Leffa (Ours) 6.19 0.32 0.891 0.044 10.64 0.59

- Table 5 Quantitative results comparison with other methods on the DressCode dataset for virtual try-on. Our Leffa achieves state-of-the-art results across all categories of garment.

Method Lleffa paired unpaired FID ↓ KID ↓ SSIM ↑ LPIPS ↓ FID ↓ KID ↓

Ours

✗ 3.64 0.33 0.911 0.040 5.98 1.42 ✓ 2.06 0.07 0.924 0.031 4.48 0.62

- Table 6 Ablation study on DressCode dataset for virtual try-on. Our Leffa loss significantly improves model performance. Method Lleffa FID ↓ SSIM ↑ LPIPS ↓

Ours (512 × 352)

✗ 5.72 0.744 0.153 ✓ 4.23 0.755 0.119

- Table 7 Ablation study on the DeepFashion dataset for pose transfer. Our Leffa loss significantly improves model performance.

Method

paired unpaired

FID ↓ KID ↓ SSIM ↑ LPIPS ↓ FID ↓ KID ↓

Leffa (Ours) 4.54 0.05 0.899 0.048 8.52 0.32 Leffa w/o average A 6.02 0.74 0.863 0.072 9.78 0.98 Leffa w/o upsample F 4.94 0.32 0.888 0.064 9.33 0.78

- Table 8 Ablation study for the proposed Leffa loss.

Effect of upsampling flow fields. To evaluate the impact of upsampling the flow fields to image resolution, we experiment with retaining their latent resolution for the Leffa, requiring the ground truth image to be downsampled. This resizing reduces detail, hindering accurate supervision. As shown in Tab. 8 (Leffa w/o upsample F), not upsampling the flow fields results in a performance drop, further supporting our viewpoint.

paired unpaired

Method

FID ↓ KID ↓ SSIM ↑ LPIPS ↓ FID ↓ KID ↓

Our baseline 5.31 0.30 0.885 0.058 9.38 0.91 freeze Reference UNet 6.42 0.77 0.863 0.066 10.63 1.32 + CLIP visual feature 5.33 0.31 0.886 0.056 9.40 0.95 + CLIP textual feature 5.37 0.40 0.876 0.060 9.45 0.98

- Table 9 Ablation study for our diffusion-based baseline. Making both the generative and reference UNets trainable is key to performance improvement for our diffusion-based baseline.

- D Discussion

Why does our diffusion-based baseline perform comparably to state-of-the-art methods? Our baseline is similar to existing virtual try-on and pose transfer methods (Choi et al., 2024; Xu et al., 2024; Bhunia et al., 2023; Han

- et al., 2023), but we find that complex designs are unnecessary for strong performance. The key lies in making both UNets in the dual architecture fully trainable, as freezing one significantly degrades results. As shown in Tab. 9, freezing the reference UNet (as done in Choi et al. (2024)) leads to a significant performance drop (FID reduction of -1.11/-1.25 for the paired/unpaired settings). In contrast, adding CLIP visual and textual features (Choi et al., 2024; Xu et al., 2024) results in only a slight performance decline. This highlights that the key to improving the baseline lies in making both UNets trainable, while adding more complex designs (e.g., add CLIP, textual information) is unnecessary.

Why not add Lleffa from the first training stage? At the start of training, the attention maps are not well learned, and applying our Leffa loss too early forces the target query to prematurely attend to the reference key, hindering convergence rather than accelerating it. Instead, introducing our Leffa loss in a subsequent training stage significantly enhances performance, demonstrating its ability to correct inaccurate attention and guide the model toward more effective learning.

[Figure 9]

##### Figure 8 More visualizations of feature maps to assess the impact of Leffa loss. The third column is the optimal setting used in our paper.

