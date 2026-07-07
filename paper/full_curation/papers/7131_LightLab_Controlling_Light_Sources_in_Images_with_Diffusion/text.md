## LightLab: Controlling Light Sources in Images with Diffusion Models

NADAV MAGAR, Tel Aviv University and Google, Israel AMIR HERTZ, Google, United States of America ERIC TABELLION, Google, United States of America YAEL PRITCH, Google, Israel ALEX RAV-ACHA, Google, Israel ARIEL SHAMIR∗, Reichman University and Google, Israel YEDID HOSHEN∗, Hebrew University of Jerusalem and Google, Israel

# arXiv:2505.09608v1[cs.CV]14May2025

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

Fig. 1. Light editing results of LightLab. Our method enables explicit parametric control over light sources in an image, while producing physically plausible shadows and environmental effects (top). The method can manipulate the intensity of light sources, change their color, and adjust ambient illumination (middle). LightLab can be used for intricate sequential editing of lighting in images (bottom from left to right): starting with an input image, turning off even outside light coming from a window, turning off an inside light source, turning on another light source, and changing lights colors. Note how reflections and shadows are plausibly handled in all these cases (please zoom in for better view and examine the tabletop).

We present a simple, yet effective diffusion-based method for fine-grained, parametric control over light sources in an image. Existing relighting methods either rely on multiple input views to perform inverse rendering at inference time, or fail to provide explicit control over light changes. Our method fine-tunes a diffusion model on a small set of real raw photograph pairs, supplemented by synthetically rendered images at scale, to elicit its photorealistic prior for the relighting task. We leverage the linearity of light to synthesize image pairs depicting controlled light changes of either a

target light source or ambient illumination. Using this data and an appropriate fine-tuning scheme, we train a model for precise illumination changes with explicit control over light intensity and color. Lastly, we show how our method can achieve compelling light editing results, and outperforms existing methods based on user preference.

CCS Concepts: • Computing methodologies → Neural networks; Transfer learning; Computational photography; Rendering.

∗Equal advising.

This work is licensed under a Creative Commons Attribution 4.0 International License.

1 INTRODUCTION

A simple flick of a light switch can transform the appearance of an image. Turning on a light does not only increase an image’s brightness, but can also add depth, contrast, and shift a viewer’s

focus. Similarly, changing the temperature of a light source can set the mood of an image, warm light creates a comfortable and inviting atmosphere, whereas cool light induces a cold and pristine feeling. While turning a light on is easy at the time of capturing a photo, doing so after capture is far more challenging.

Traditionally, 3D graphics methods model scene geometry and intrinsics from multiple input captures, then simulate new lighting conditions with a physical illumination model. These methods enable precise and explicit control over light sources, however recovering a 3D model from a single image is an ill-posed problem, and often fails to produce appealing results. Recently, diffusion-based image editing methods have been applied to various relighting tasks, utilizing their strong statistical prior to bypass the need for an explicit physical model of the image. However, the stochastic nature of diffusion models, and their common reliance on textual conditioning, makes fine-grained parametric control of physical image properties challenging.

In this work, we present LightLab, a diffusion-based method for explicit, parametric control over light sources in an image. Specifically, we target two key properties of a light source: its intensity and color. To complement these abilities, we also enable control over the scene’s ambient illumination, and over the resulting tone mapping effects. Combined, these capabilities provide end users a set of rich light editing tools, and allow control over the look and feel of an image, by means of manipulating its illumination.

Editing the illumination in an image presents unique challenges. Firstly, creating physically plausible re-lit images requires an understanding of light transport. The distribution of light in a scene is highly dependent on its geometry, material properties, and light sourcecharacteristics,makingrelighting an extremely ill-conditioned problem. A second challenge arises in preserving intrinsic scene properties, such as geometry, diffuse and specular reflectance, and roughness, while changing illumination. These properties are intertwined in the image pixels, and disentangling them is a complex task, even when solved explicitly. Furthermore, editing illumination in Standard Dynamic Range (SDR), requires also disentangling changes to the physical scene from changes originating from the tone-mapping and imaging pipeline.

To overcome these challenges, we take a data-centric approach, using paired images representing the complex illumination changes induced by turning on a visible light source (Section 3.1). As acquiring a large scale dataset of paired examples depicting a diverse set of scenes and light configurations is infeasible, we propose a combination of three solutions. First, we capture a small set of photograph pairs, which depict a scene while a light source is switched on or off. Next, we use physically-based rendering of 3D scenes [Pharr et al. 2023] to create a large number of synthetic pairs, while also switching on and off several visible light sources and producing several ambient lighting configurations. To further increase the diversity of the data, we also procedurally place light sources at random locations in the scene. Finally, we use the linearity of light (Section 3.2) and a suitable tone-mapping strategy (Section 3.3) to synthesize image pairs for the full range of light source intensity.

Although simulated pairs cause a domain gap too large for training on their own, we show that combining a smaller set of real

photograph pairs mitigates the resulting domain drift. This combination mitigates the physical limitations of data capture, while allowing fine-grained control over light sources. Given such data, we fine-tune a conditional diffusion model (Section 3.4, Section 4.1) to learn to edit the color and intensity of the light sources.

We demonstrate our method on images that contain visible light sources, focusing on indoor settings, with additional results on outdoor images (Figure D.9), and out-of-domain examples (Figure 10). Further comparisons with other related works (Section 4.3) show that our method is the first to enable high-quality, precise control over visible local light sources, as confirmed by a user study.

We present several applications of our method (Section 5), including control over intensity and color of light sources in a photograph, applying a series of consecutive edits, changing illumination in nonphotorealistic images, and generating consistent lighting across multiple animation frames.

In summary, the main contributions of our paper are:

- • A method for effectively fine-tuning and conditioning a diffusion model for parametric control over light sources from a single image.
- • We show that even a small set of real, physically accurate training examples can complement large-scale synthetic rendering, where the former prevents domain drift while the latter improves physical plausibility.
- • A resulting high-quality image-based solution that enables users to make complex and consecutive illumination edits.

2 RELATED WORKS

Relighting with Generative Models. Generative image editing methods has been applied to various image relighting tasks. In portrait relighting, several works use images of a subject, captured by a light stage rig [Debevec et al. 2000], which are used to supervise a generative model [Anonymous 2024; Mei et al. 2023; Nestmeyer et al. 2020; Pandey et al. 2021; Ren et al. 2024]. Jin et al. [2024] enable general single image object relighting by fine-tuning a diffusion model conditioned on a normalized environment map, using synthetic relighting datasets from Deitke et al. [2023]. While effective for portrait relighting, light stage data is object-centric, and could not be easily used to control the illumination of arbitrary light sources in the wild. For outdoor scenes, several works assume a single dominant light source, such as the sun, to simulate cast shadows, which can be used to condition a generative model [Griffiths et al. 2022; Kocsis et al. 2024]. Compared to outdoor settings, indoor scenes often present complex multi-illumination conditions. Li et al. [2020] train an inverse rendering network that recovers a spatially-varying spherical Gaussian lighting from a single image. Bhattad et al. [2024]; Wang et al. [2022] find edit directions in StyleGAN’s latent space [Karras et al. 2020] that manipulate light. However, such methods do not enable direct control over specific light sources, and do not work on real images. Xing et al. [2024] train a ControlNet [Zhang et al. 2023] to condition a diffusion model with implicit latent intrinsics and illumination representations extracted using the method of [Zhang et al. 2024]. More closely related to our application, Choi et al. [2024] use a ControlNet to condition lighting changes on user scribbles. While their method enables fine-grained

[Figure 18]

|𝐢on|
|---|

|𝐢relit  , |
|---|

|relit  , |
|---|

|𝐢change| |
|---|---|
|[Figure 19]<br><br>[Figure 20]| |

𝐢off = 𝐢amb

[Figure 21]

|[Figure 22]<br><br>[Figure 23]|
|---|

[Figure 24]

|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]|[Figure 29]| |
|---|---|---|
|[Figure 30]| | |
| | | |

|[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]|[Figure 35]| |
|---|---|---|
| |[Figure 36]| |
| | | |

𝐢amb + 𝐢change

[Figure 37]

|𝐢relit  , |
|---|

|relit  , |
|---|

| |𝐢amb| |
|---|---|---|
|[Figure 38]<br><br>[Figure 39]| | |

𝐢change

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

- Fig. 2. Post processing pipeline. Top row. From a pair of real (raw) photograph pairs, we first isolate the target light change ichange. Bottom row. For synthetic data, we render each light component separately. After light disentanglement both domains undergo light arithmetic to create parameterized sequences of images irelit (𝛼,𝛾, ct), which are later tone mapped to SDR (either together or separately).

spatial control over illumination changes, it biases the model to make localized edits bounded by the user scribble and lacks control over light source intensity and color.

Light Control under Multi-illumination. Image based light editing under multi-illumination has been extensively studied in the context of flash-photography, with applications in white-balancing [Hui et al. 2016], automatic flash bounce capture [Murmann et al. 2016] and post-capture computational control of flash [Maralan et al. 2023]. Murmann et al. [2019] collect a multi-illumination dataset using a custom motorized capture device that fires flash at different directions. They use the linearity of light to synthesize mixed-illuminant data, and train a U-Net [Ronneberger et al. 2015] to change illuminant direction. Hui et al. [2017] use a flash/no-flash raw image pair to disentangle and manipulate scene illuminants based on their spectral differences. Notably, when their assumptions hold they can control the intensity and color of particular lights (the flash and joint ambient illumination). Our goal is to allow similar complex illumination edits, in a more general and accessible setting where only a single no-flash SDR image is given.

Aksoy et al. [2018] tackle this setting for the case of flash illumination. By curating a crowd-sourced dataset of raw flash/no-flash pairs, from which they decompose illuminants to ambient and flash pairs. Notably, they demonstrate that with ambient illumination variation augmentations, this dataset can be used to train a Pix2Pix model [Isola et al. 2017], to decompose ambient and flash illuminations from a single flash photograph. We also take a data-driven approach and train a diffusion model on a curated dataset of controlled illumination changes, synthesized from raw image pairs. Specifically, our setting is closer to that of [Haeberli 1992] in which an arbitrary visible light source is turned on/off.

3 METHOD

Our method uses paired images to implicitly model controlled light changes in image space, which are used to train a diffusion model. In Section 3.1, we describe our data collection procedure, which includes a small set of paired raw photographs, supplemented by a large set of synthetically rendered 3D indoor scenes. Sections 3.2 and 3.3 continue by explaining how we control the light source and ambient light parameters in post-processing. Finally, Section 3.4 explains how we fine-tune a pre-trained diffusion model for light source control using this mixture of collected datasets.

3.1 Datasets

Photography capture. We capture a set of 600 raw photograph pairs, using off-the-shelf mobile devices, a tripod, and triggering equipment. Each pair depicts the same scene, in which the only physical change is switching on a visible light source.

To ensure the captured images are well-exposed, we use each device’s default auto-exposure setting, and calibrate them using the raw image meta-data during post-capture (see supplementary Section B.1). This dataset provides intricate details of geometry, material appearance, and complex light phenomena that may not be found in synthetically rendered data (see below).

Following previous work [Aksoy et al. 2018; Haeberli 1992; Hui et al. 2017], we regard the "off image" as ambient illumination iamb := ioff and extract the illumination from the target light: ichange = ion − ioff. This difference may have negative values due to captured noise, errors in the post-capture calibration process, or slight differences in ambient illumination conditions between the two images. To avoid the resulting unintended dimming, we clip the difference to be non-negative: ichange = clip(ion − ioff, 0). We show statistics for this clip operation in Section B.1 in the supplementary.

F

e

r

a

e

- t
- u

i

r

u

- r

e

- s

o

Intensity

F

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Di usion

Noise

Input Image

- Fig. 3. Conditioning Signals. Spatial conditions (input image, target light mask and depth map) are embedded to the latent dimensions and concatenated to the input noise. Light intensity and color control are applied by scaling the intensity and color of the target light mask. Global controls (ambient light intensity and tone-mapping value) are projected to text embedding dimension and inserted through cross-attention.

As we show in Section 4.2, incorporating real data helps to disentangle the intended illumination changes from the style of synthetically rendered images, which do not include visual artifacts introduced by a real physical camera sensor such as lens distortion or chromatic aberrations, among others. In post-processing (Section 3.2), we inflate each real image pair by a factor of 60 to encompass a range of intensities and colors. The complete dataset after post-processing includes approximately 36K images.

Synthetic rendering. To complement the captured dataset, we use a larger set of procedurally generated synthetic images of 3D scenes, rendered using a physically-based renderer. We begin with 20 synthetic indoor scenes constructed by 3D artists using Blender [Community 2018], from which we procedurally render images. Our rendering pipeline randomly samples camera views around view target objects, and procedurally sets virtual light sources parameters (e.g. intensity, color temperature, area size, cone angle, etc.). We further expand this data by randomly sampling plausible light fixture locations, and adding them to the scenes. Furthermore, we apply different environment maps of varying strengths, and randomly configure background illumination. See the supplementary Section A.1 for further details.

We render synthetic images with each light component iamb and ichange separately, which are later combined in post-processing (Section 3.2). Images are created in linear RGB color space, using a Monte-Carlo path tracing renderer. Commonly, this may create pixels with unbounded outlier sampled values, when a path’ sampling probability is very low. To avoid subsequent noisy tone-mapping behavior, we apply a bound 𝐸𝑚𝑎𝑥, set to the top 5 × 10−4 percentile of pixel values, computed on a subset of 2000 random renders.

This dataset spans 16,000 combinations of target light source, camera view, and environmental lighting, which are then also inflated by a factor of 36 in post-processing totaling approximately 600K images. Despite the relatively low diversity in scene geometries and materials, in Section 4.2 we show that the synthetic data steers the model to create physically plausible, view-consistent lighting.

- 3.2 Image-based Relighting

Given the disentangled linear RGB images iamb and ichange, we generate a parametric sequence of images with varying target light source intensity and color and ambient illumination.

To change the target light color, we multiply it by a change coefficients c ∈ R3 in linear RGB space. To compute this coefficient, we first estimate the original light RGB co, and then multiply by the desired RGB coefficients ct to obtain c = ct ⊙co−1, where ⊙ denotes element-wise multiplication and co−1 element-wise inverse.

For a relative ambient illumination intensity 𝛼 ∈ [0, 1], a relative target light intensity 𝛾 ∈ [0, 1] and the target light RGB ct the relit image is computed by:

irelit 𝛼,𝛾, ct; iamb, ichange = 𝛼 iamb +𝛾 ichangec (1) See examples of post-processed samples in the supplementary.

- 3.3 Tone-Mapping Strategy

The relit image sequences (Equation 1) need to be tone-mapped to match the training distribution of the base diffusion model. One approach is to tone-map each relit image separately, ensuring it is well exposed. We find that this approach produces inconsistent

0.00 0.25 0.50 1.00

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Fig. 4. Tone mapping strategy. A sequence of images of increasing light intensity, tone mapped either separately or together. Top row. The images tone mapped separately, notice how the light source intensity appears constant when lit, while ambient light appears to be dimmed. Bottom row. Tone mapped together.

Input +0.25 +0.50 +1.00

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Input -0.50 -0.75 -1.00

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Fig. 5. Intensity control. Fine-grained control over a target light’s intensity using our method. Values represent the relative intensity change with respect to the source image.

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

Fig. 6. Color Control. We turn on the street lamp in the input image (top left) with different colors. Top row. artificial light blackbody temperatures. Bottom row arbitrary non-natural RGB colors.

image sequences, where the perceived changes in light intensity do not align with the physical change. For example, when the target light source dominates the dynamic range, it appears constant for increasing intensities, while ambient light is dimmed instead (Figure 4).

We address this problem by tone-mapping image sequences together using the same fixed exposures, based on [Hasinoff et al. 2016; Mertens et al. 2007]. Given an image pair in linear space: ioff, ion, we heuristically choose deciding intensities𝛾𝑑, 𝛼𝑑 for the target light source and ambient light, respectively. The synthetic exposures used in [Hasinoff et al. 2016] are computed with respect to the relit image irelit 𝛼𝑑,𝛾𝑑, ct; iamb, ichange and are applied across all different light source, and ambient light intensity combinations.

Although using fixed exposures produces intuitive sequences of varying light intensity, it can cause problems when solely used to train a relighting model. Firstly, at inference time we expect input images to be captured using auto-exposure, and that are tonemapped individually. Secondly, we want to allow the user to decide how the model’s output should be tone-mapped, providing control over the trade-off between a well-exposed output and intuitive light changes. As such, we tone-map our data using both strategies and expose the used strategy as an input condition to the diffusion model. We show the effect of this condition on the model’s output in Figure 10 (bottom row) and Figure D.11 in the Supplement.

- 3.4 Light Source Control with Diffusion Models

We fine-tune a pre-trained latent diffusion model [Ramesh et al. 2022; Rombach et al. 2022; Saharia et al. 2022] to enable parametric control over visible light source parameters, ambient light intensity, and tone mapping effects. We use different conditioning schemes for localized spatial signals and for global controls.

Spatial conditions. Spatial conditions include the input image, an automatically extracted depth map of the input image, and two spatial segmentation masks for the target light source intensity change and color respectively. To represent the target light source, we use a semantic segmentation mask acquired using [Ravi et al.

2024] conditioned on a user-given bounding box. The mask is used twice to condition the model on both the target light intensity change and on the target color. For the intensity condition, the mask is multiplied by the relative light intensity change scalar (corresponding to 𝛾 in Equation 1). For the target color condition, the mask is expanded to 3 channels and scaled to the requested target RGB color (corresponding to c in Equation 1). The input image is encoded using the model’s Variational Auto-Encoder (VAE), while other conditions are resized to match the spatial latent dimensions. Following this, a learned 1 × 1 convolution is used to match the number of channels of the spatial conditions tensor. The resulting tensor is then concatenated to the input noise [Nichol et al. 2021; Wang et al. 2023]. We provide additional details regarding the conditioning and sampling scheme of light change parameters during training in the supplementary Sections C.1 and C.3.

Global conditions. These include the ambient light change as a scalar value in the range [−1, 1], and the requested tone-mapping strategy as a binary scalar value, which affects the exposure of generated images during inference as described in Section 3.3. Global controls are encoded into Fourier features [Tancik et al. 2020] followed by a multi-layer perceptron (MLP) that projects them to the text embedding dimension. The projections are concatenated to the text embeddings and then inserted through the text-to-image cross-attention layers.

4 EXPERIMENTS 4.1 Implementation Details

Model and training. We fine-tune a text-to-image latent diffusion model with the same architecture layout and similar hidden dimensions as Stable Diffusion-XL [Podell et al. 2023], which was trained on a subset of Chen et al. [2022]. In Section 4.2 we train each model for 45,000 steps with a learning rate of 10−5, and a batch size of 128 at 1024 × 1024 resolution. Training took ∼12 hours on 64 v4 TPU’s. During training, we drop the depth and color conditions 10% of the time to allow unconditional inference. We use [Ravi et al. 2024] light source segmentation and [Yang et al. 2024]

Table 1. Effect of Training Domain. Ground truth similarity metrics for models trained on different weightings between domains. Binary. Image pairs that represent either completely turning on or off a light source.

PSNR↑ SSIM↑

Training Datasets

Binary Intensity Color Binary Intensity Color

23.2 28.6 24.2 0.818 0.879 0.874 w/o depth 23.2 28.5 24.1 0.816 0.876 0.871

Real + Synth. w/ depth

Real + Synth. (Binary) 22.7 - 23.8 0.818 - 0.864 Real Only 22.9 28.3 23.75 0.815 0.879 0.870 Synth. Only 20.71 27.38 22.33 0.7947 0.8730 0.8605

to create depth maps. See Section B.2 in the supplementary for the masking procedure. Running inference with 15 denoising steps took approximately 5 seconds on a single v4 TPU.

Evaluation datasets. For quantitative ablations and comparisons, we evaluate trained models on paired datasets, curated using the procedure described in Section 3.1. The real photograph dataset contains 200 photo pairs of different scenes and light sources, which is expanded by a factor of 60 during post-processing. The synthetic evaluation dataset includes renders from two held out scenes, containing unique light sources, objects, and materials. For qualitative evaluations, where a ground truth target is not needed, we evaluate the model on a collection of 100 images from [Bell et al. 2014]. For these images, we manually annotate the target light sources in each image and compute their respective segmentation masks and depths. Throughout the evaluation and to generate all results in the paper, the tone-mapping condition was set to "together", unless stated otherwise.

Evaluation metrics. We measure the model’s performance on paired images with two common metrics: Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) [Ren et al. 2024; Xing et al. 2024; Zeng et al. 2024b,a]. To supplement these measurements, we provide qualitative results in Figures 7, 8 and in the supplementary materials. Furthermore, we verify that these results align with user preference by conducting a user study to compare with other methods in Table 2. More information about our user study can be found in the supplementary materials.

- 4.2 The Effect of Different Domains

Empirically, we find that using a mixture of both a small set of real captures and synthetic 3D renders yields the best results. Nevertheless, understanding the effect of each domain dataset, and its corresponding augmentations, on the trained models performance provides insights for further research. In this section we isolate these effects we evaluate models fine-tuned from the same base model, with different data distributions.

Generalization across domains. We observe that the model trained only on synthetically rendered data does not generalize well to real images (Table 1 third row). We attribute this generalization error to differences in style, e.g. lack of complex and intricate geometries, fidelity of textures and materials, absence of camera artifacts such as glare, which are not present in the synthetic dataset.

Table 2. Comparison with other works. Left ground truth similarity in PSNR and SSIM on our paired evaluation set Section 4.1. Right user study preference rates for our method. Our method outperforms previous results both in physical plausibility and in user satisfaction.

#### Method PSNR SSIM Ours Win Rate

RGB ↔X 12.0 0.518 89.3 % IC-Light 12.2 0.507 83.0 % OmniGen 15.1 0.584 83.0 % ScribbleLight 13.8 0.418 84.8 %

Ours 23.2 0.818 -

Empirically, this domain gap causes a knowledge drift in the base diffusion model, as can be seen in Figure D.3 in the supplementary.

Using multiple domains. We train three models using the same procedure on three mixtures of the data domains: real captures only, synthetic renders only, and their weighted mixture. The results in Table 1 show that using a weighted mixture of data from both domains achieves the best results for all settings. Notably, we observe a small quantitative relative difference between the mixture dataset and real captures only, despite the significant difference in their size. For example, adding synthetic data results in only 2.2% averaged improvement in PSNR. This is likely due to perceivable local changes in illumination, such as small instance shadows, and specular reflections, are obscured by image-wide, low-frequency details in these metrics.

We corroborate this effect with qualitative comparisons in Figures 7, and Figure C.2 in the supplementary, which demonstrate how adding synthetic data encourages the model to produce intricate local shadows which are not present in the real-only model.

4.3 Comparisons

To the best of our knowledge, our method is the first to provide finegrained control over light sources in a real single image. Therefore, for a fair comparison, when comparing to other works, we evaluate only on the binary task. As baselines, we adapt four diffusion-based editing methods: OmniGen [Xiao et al. 2024], RGB ↔ X [Zeng et al. 2024a], ScribbleLight [Choi et al. 2024], and IC-Light [Anonymous 2024]. These methods use text prompts describing the light change and light source location in the input images, and various other scene intrinsics. The RGB ↔ X model is conditioned on an multiple pre-computed normals, albedo, roughness, and metallically maps of the input images. ScribbleLight receives the albedo and a mask layer that indicates where light should be turned on/off (as opposed to the light source mask for our method). Lastly, to use IC-Light to control light sources, we input the entire image as the foreground and provide our light source segmentation masks as the environment light source condition.

As can be seen from Table 2, our method outperforms previous methods by a significant margin. We also provide a qualitative comparison on evaluation images from [Bell et al. 2014] in 8. Notably, OmniGen fails to turn the target light sources on/off, and introduces local geometry changes. RGB ↔ X, ScribbleLight, and IC-Light can

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

LightLab: Controlling Light Sources in Images with Diffusion Models

#### Input (Off) Synthetic Only Captured Only Mixed Target (On)

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Fig. 7. Qualitative Comparison: The effect of adding synthetic data to the training dataset. Lime The specular reflection of the screen is simulated only in the mixed model. Red The mixed model generates a fine hard shadow for the cable which correlates with its shape and resemble ground truth.

Input (On/Off) OmniGen RGB ↔ X ScribbleLight IC-Light Ours

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

- Fig. 8. Qualitative Comparison. Comparison with other works on images from IIW dataset. Left. the input image where green / red contours specify which light source(s) should be turned on / off respectively. See Figures D.5, D.6 in the supplementary for more examples.

succeed in changing input lighting conditions, but usually results in additional unwanted illumination changes or color distortion. With respect to prior works, our method faithfully controls the target light source, and generates physically-plausible lighting.

- 4.4 Qualitative Results

We presentadditionalqualitativeresults, demonstrating our method’s ability to generate different illumination phenomena, and its representative failure cases.

Out-of-domain light sources. Figure 9 (b - red) shows an example where our model lights candles as light-tubes. This is likely due to bias in the light sources represented in the fine-tuning datasets. Interestingly, this bias is less prominent when turning off the light, as demonstrated in Figure 9 (d) where the model turns off the paper lantern. Furthermore, Figure 10 (second row) shows that the model can faithfully turn on a well-represented light source in outof-domain images (non-photorealistic cartoon image). Note how

the desk lamp is lit to match the image (zoom in on the slits at the top of the lamp shade).

Reflections. Figure 9 (b - green) shows an example where the model lights the mirror reflection of the chandelier, another example can be seen in Figure 1 (bottom row, second image from the left), where the model preserves the reflection of the ceiling lamp on the glass door when turning off ambient day light entering through it. Figures 9 (d - right box) and (f - green) show examples of removed / generated specular reflections from metallic materials, respectively.

Perspective failures. Our dataset contains mostly indoor images with bounded depth of field. This bias can create physically inaccurate results in cases such as forced perspective, as seen in Figure 5 (bottom row) where turning off the street light near the camera removes the red light projected on the distant pagoda.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Nadav Magar, Amir Hertz, Eric Tabellion, Yael Pritch, Alex Rav-Acha, Ariel Shamir, and Yedid Hoshen

#### Input Output Input Output Input Output

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

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

[Figure 127]

[Figure 128]

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

| |
|---|

| |
|---|

(a)

(b)

(c)

(d)

(e)

(f)

- Fig. 9. Qualitative Results. Select results on publicly sourced images, demonstrating successful lighting edits (lime boxes) and failures (red boxes) of our method. See the accompanying discussion in Section 4.4.

### 5 APPLICATIONS

7 CONCLUSION

We present several possible applications of our method in various settings. The primary one is the ability to control lights in a photograph post-capture. Examples of turning lights on or off can be seen in Figures 1, 8, 10. More accurate control of light intensity can be seen in Figures 5, control of color can be seen in Figures 6, and control of ambient light can be seen in Figure 10.

We presented LightLab- a diffusion-based method for fine-grained control of light sources in images. Using the linearity of light and synthetic 3D data, we create a high-quality data set of paired images, which implicitly model complex and controlled illumination changes.We showthatfine-tuningadiffusion model using physicallybased data, one can achieve physically-plausible control over light sources after capture. We believe that leveraging classical computational photography techniques and physics-based simulations to generate training data for generative models, is a promising direction for physically-based image editing.

In addition, we showcase how our method can be used for consistent light editing that combines light intensity, color and ambient changes. For example, notice how, in the tatami room image in Figure 1, modification of different light sources causes disentangled light changes over the input image. For more results, see our Supplemental Video and our interactive demo.

ACKNOWLEDGMENTS

In Figure 11 we demonstrate another possible application of using synthetic data, inserting virtual light sources - ones without any geometry into the scene. The images were rendered as in Section 3.1, without the light source geometry. This application alleviates the requirement of a visible light source in the image.

We thank Kiran Murthy for his valuable contribution and guidance in tone-mapping linear color images. We are also very grateful to Alberto García García, Erroll Wood, Jesús Pérez and Iker J. de los Mozos, for their help and contributions to the synthetic rendering effort. Finally, we also thank Dani Lischinski, Andrey Voynov, Bar Cavia, Tal Reiss, Daniel Winter, Yarden Frenkel, Asaf Shul, Matan Cohen, Julian Iseringhausen, Francois Bleibel, Chloe LeGendre, Dani Cohen-Or and Or Patashnik for discussion and their support that aided and improved our work.

The ability to turn on lights anywhere in a 2D image also opens up possibilities for animation. In figure 12 we present an example where we move the position of a turned-off lamp on a table, and in post-processing we turn it on. As can be seen, LightSwitch correctly interprets the geometry of the scene and creates plausible shadows and lighting throughout the sequence - allowing one to create a stop-motion type animation.

### 6 LIMITATIONS

While fine-tuning on a low-diversity dataset can generate pleasing relighting results in many settings, the resulting model suffers from bias in thedataset. Thisbiasis mostevident on the type of target light source as demonstrated in Section 4.4. We believe that combining our method with an unpaired fine-tuning method could mitigate this bias and is an interesting direction for future research.

Secondly, we use a simplistic data capture process, using consumer mobile devices and calibrating exposure post-capture. Although using this method allows us to easily collect the captured dataset, it does not enable precise relighting in physical units.

[Figure 129]

###### Fig. 10. Additional results.. Our method can control a target light intensity (third row), light color (second row) and a scene’s ambient light (fourth row) and tone-mapping effects (fifth row) across diverse scenes and image styles. The number above the results indicate the relative light intensity change used for the target light or ambient. See the complete inputs used to generate results in the supplementary.

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

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

###### Fig. 11. Virtual Point Lights. By using synthetic renderings of point lights without geometry we can insert an "invisible" light source into the scene. On the left of each sequence is the input image, red "X" marks the insertion point. Notice how the localization of the light source.

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

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

| |
|---|

| |
|---|

###### Fig. 12. Physically Plausible Lighting. Top row. The input sequence was created by capturing photographs of a turned-off lamp rotated around the polygon dog. Bottom rows. Inference results of our method, Orange a zoom-in on the dog. Note how self occlusions on the different faces, and dog’s shadow match the lamp’s position and angle.

REFERENCES

Yağız Aksoy, Changil Kim, Petr Kellnhofer, Sylvain Paris, Mohamed Elgharib, Marc Pollefeys, and Wojciech Matusik. 2018. A Dataset of Flash and Ambient Illumination Pairs from the Crowd. In Proc. ECCV.

Anonymous. 2024. Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing Consistent Light Transport. In Submitted to The Thirteenth International Conference on Learning Representations. https:// openreview.net/forum?id=u1cQYxRI1H under review.

Sean Bell, Kavita Bala, and Noah Snavely. 2014. Intrinsic Images in the Wild. ACM Trans. on Graphics (SIGGRAPH) 33, 4 (2014).

Anand Bhattad, James Soole, and D.A. Forsyth. 2024. StyLitGAN: Image-based Relighting via Latent Control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. 2022. PaLI: A Jointly-Scaled Multilingual Language-Image Model. arXiv:2209.06794 [cs.CV]

Jun Myeong Choi, Annie Wang, Pieter Peers, Anand Bhattad, and Roni Sengupta. 2024. ScribbleLight: Single Image Indoor Relighting with Scribbles. arXiv:2411.17696 [cs.CV] https://arxiv.org/abs/2411.17696

Blender Online Community. 2018. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam. http://www.blender. org

Paul Debevec, Tim Hawkins, Chris Tchou, Haarm-Pieter Duiker, Westley Sarokin, and Mark Sagar. 2000. Acquiring the reflectance field of a human face. In Proceedings of the 27th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH ’00). ACM Press/Addison-Wesley Publishing Co., USA, 145–156. https: //doi.org/10.1145/344779.344855

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. 2023. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems 36 (2023), 35799–35813.

David Griffiths, Tobias Ritschel, and Julien Philip. 2022. OutCast: Outdoor Single-image Relighting with Cast Shadows. arXiv:2204.09341 [cs.GR] https://arxiv.org/abs/2204. 09341

Paul Haeberli. 1992. Synthetic Lighting for Photography. https://www.graficaobscura. com/synth/index.html.

Samuel W. Hasinoff, Dillon Sharlet, Ryan Geiss, Andrew Adams, Jonathan T. Barron, Florian Kainz, Jiawen Chen, and Marc Levoy. 2016. Burst photography for high dynamic range and low-light imaging on mobile cameras. ACM Trans. Graph. 35, 6, Article 192 (Dec. 2016), 12 pages. https://doi.org/10.1145/2980179.2980254

Zhuo Hui, Aswin C. Sankaranarayanan, Kalyan Sunkavalli, and Sunil Hadap. 2016. White balance under mixed illumination using flash photography . In 2016 IEEE International Conference on Computational Photography (ICCP). IEEE Computer Society, Los Alamitos, CA, USA, 1–10. https://doi.org/10.1109/ICCPHOT.2016. 7492879

Zhuo Hui, Kalyan Sunkavalli, Sunil Hadap, and Aswin C. Sankaranarayanan.

2017. Illuminant Spectra-based Source Separation Using Flash Photography. arXiv:1704.05564 [cs.CV]

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. 2017. Image-to-Image Translation with Conditional Adversarial Networks. CVPR (2017).

Haian Jin, Yuan Li, Fujun Luan, Yuanbo Xiangli, Sai Bi, Kai Zhang, Zexiang Xu, Jin Sun, and Noah Snavely. 2024. Neural Gaffer: Relighting Any Object via Diffusion. In Advances in Neural Information Processing Systems.

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. 2020. Analyzing and Improving the Image Quality of StyleGAN. In Proc. CVPR.

Peter Kocsis, Julien Philip, Kalyan Sunkavalli, Matthias Nießner, and Yannick HoldGeoffroy. 2024. LightIt: Illumination Modeling and Control for Diffusion Models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 9359–9369. https://doi.org/10.1109/cvpr52733.2024.00894

Zhengqin Li, Mohammad Shafiei, Ravi Ramamoorthi, Kalyan Sunkavalli, and Manmohan Chandraker. 2020. Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and svbrdf from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2475–2484.

Sepideh Sarajian Maralan, Chris Careaga, and Yağız Aksoy. 2023. Computational Flash Photography through Intrinsics. Proc. CVPR.

Yiqun Mei, He Zhang, Xuaner Zhang, Jianming Zhang, Zhixin Shu, Yilin Wang, Zijun Wei, Shi Yan, HyunJoon Jung, and Vishal M. Patel. 2023. LightPainter: Interactive Portrait Relighting with Freehand Scribble. arXiv:2303.12950 [cs.CV] https://arxiv. org/abs/2303.12950

Tom Mertens, Jan Kautz, and Frank Van Reeth. 2007. Exposure Fusion. In 15th Pacific Conference on Computer Graphics and Applications (PG’07). 382–390. https://doi. org/10.1109/PG.2007.17

Lukas Murmann, Abe Davis, Jan Kautz, and Frédo Durand. 2016. Computational bounce flash for indoor portraits. ACM Transactions on Graphics (TOG) 35 (2016), 1 – 9. https://api.semanticscholar.org/CorpusID:6556192

Lukas Murmann, Michael Gharbi, Miika Aittala, and Fredo Durand. 2019. A Dataset of Multi-Illumination Images in the Wild. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE. https://doi.org/10.1109/iccv.2019.00418

Thomas Nestmeyer, Jean-Francois Lalonde, Iain Matthews, and Andreas Lehrmann. 2020. Learning Physics-Guided Face Relighting Under Directional Light. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5123–5132. https://doi.org/10.1109/cvpr42600.2020.00517

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. arXiv:2112.10741 [cs.CV]

Rohit Kumar Pandey, Sergio Orts Escolano, Chloe LeGendre, Christian Haene, Sofien Bouaziz, Christoph Rhemann, Paul Debevec, and Sean Fanello. 2021. Total Relighting: Learning to Relight Portraits for Background Replacement. In SIGGRAPH and TOG.

Matt Pharr, Wenzel Jakob, and Greg Humphreys. 2023. Physically based rendering: From theory to implementation. MIT Press.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv:2307.01952 [cs.CV] https: //arxiv.org/abs/2307.01952

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 (2022).

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2024. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714 [cs.CV] https://arxiv.org/abs/2408.00714 Mengwei Ren, Wei Xiong, Jae Shin Yoon, Zhixin Shu, Jianming Zhang, HyunJoon Jung, Guido Gerig, and He Zhang. 2024. Relightful Harmonization: Lighting-aware Portrait Background Replacement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6452–6462.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. Springer International Publishing, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. arXiv preprint arXiv:2205.11487 (2022).

Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. 2020. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. NeurIPS (2020).

Guangcong Wang, Yinuo Yang, Chen Change Loy, and Ziwei Liu. 2022. StyleLight: HDR Panorama Generation for Lighting Estimation and Editing. In European Conference on Computer Vision (ECCV).

Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J. Fleet, Radu Soricut, Jason Baldridge, Mohammad Norouzi, Peter Anderson, and William Chan. 2023. Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting. arXiv:2212.06909 [cs.CV] https://arxiv.org/abs/2212.06909

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. 2024. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340 (2024).

Xiaoyan Xing, Konrad Groh, Sezer Karagolu, Theo Gevers, and Anand Bhattad. 2024. LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting. arXiv preprint arXiv:2412.00177 (2024).

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024. Depth Anything V2. arXiv:2406.09414 [cs.CV] https: //arxiv.org/abs/2406.09414

Chong Zeng, Yue Dong, Pieter Peers, Youkang Kong, Hongzhi Wu, and Xin Tong. 2024b. DiLightNet: Fine-grained Lighting Control for Diffusion-based Image Generation. In ACM SIGGRAPH 2024 Conference Papers.

Zheng Zeng, Valentin Deschaintre, Iliyan Georgiev, Yannick Hold-Geoffroy, Yiwei Hu, Fujun Luan, Ling-Qi Yan, and Miloš Hašan. 2024a. RGBX: Image decomposition and synthesis using material- and lighting-aware diffusion models. In ACM SIGGRAPH 2024 Conference Papers (Denver, CO, USA) (SIGGRAPH ’24). Association for Computing Machinery, New York, NY, USA, Article 75, 11 pages. https://doi.org/10.1145/3641519.3657445

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding Conditional Control to Text-to-Image Diffusion Models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 3813–3824. https://doi.org/10.1109/iccv51070.2023. 00355

Xiao Zhang, William Gao, Seemandhar Jain, Michael Maire, David Forsyth, and Anand Bhattad. 2024. Latent Intrinsics Emerge from Training to Relight. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. https://openreview. net/forum?id=ltnDg0EzF9

A GENERATING DATA FOR ILLUMINATION CONTROL

A.1 Synthetic Rendering

To procedurally generate synthetic images as described in Section

- 3.1, we use a small set of indoor scenes from publicly available 3D asset stores. The set of scenes we use represent common indoor environments, e.g. living room, kitchen, bedroom, bar, restaurant, office, etc., complete with furniture and accessories. The scenes have physically plausible textured materials and can be rendered in Blender / Cycles, producing photorealistic results, as shown in Figure A.1.

We start by manually annotating the scenes, identifying specific scene objects that are plausible camera view targets, i.e. kitchen countertop, coffee table, bar counter, couch, side table, chair, etc. We also identify existing light fixture geometry (light bulbs’ glass, spot lights’ lens) and meshes onto which additional light fixture models can be randomly placed (floor, tables, shelves, furniture top). We also identify the floor geometry to validate the randomized camera placement described next.

To create randomized and plausible camera views with targeted content, we sample an object from the set of annotated view target objects and randomly place the camera view around that object given a range of valid camera distances, field of view angle, and polar and azimuth angles w.r.t. the center of the object and the up axis. The sampled candidate camera position is then tested to be above the floor geometry and additional visibility tests are performed to make sure the camera is not randomly placed outside of the scene.

Once a valid camera view is defined, the lighting is setup procedurally, creating virtual light sources corresponding to existing annotated light fixtures and to the randomly placed additional light fixtures. Light sources that are not seen directly by the camera are then discarded and we sample the remaining light sources’ parameters (intensity, color temperature, area size, spot angle, etc.) within pre-defined ranges.

Finally, we render the scene once per light source, with only the corresponding light source switched on and all other lights switched off. We also render the scene several more times, with all light sources switched off, each time using a different distant HDRI environment map. This produces several images of that same scene, lit with one light at a time and with a diverse set of outdoors lighting scenarios casting ambient lighting through windows and other openings.

The images are saved using a full float precision image format, in linear color space, without applying any tone-mapping, and passed

down to the image post-processing pipeline described in Section 3.2.

### B IMAGE BASED RELIGHTING

- B.1 Post-Capture Calibration

In this section we provide additional details on the processing of the raw photograph pairs. We demosaic each image pair with respect to the capturing camera color filter array. Our captured dataset was comprised of raw image pairs, captured without any specific photography expertise, using standard off-the-shelf mobile devices (Pixel 9 and Samsung S24). The images were captured using each device’s auto-exposure to ensure they are well-exposed and to match the distribution of source images captured in-the-wild.

To correctly perform manipulation on pairs of images captured on the same device but under different exposure setting, we calibrate them using the raw image meta-data during post-capture. We empirically find that linearly interpolating white balance gains according to the target light source intensity produces good results. For color correction, we opt to use the camera’s color correction matrix extracted from the raw image with the light on. Since each image in a pair is captured with different exposures, we calibrate it according to the product 𝑃 of the exposure value 𝐸, the applied analog and digital gain 𝐺, 𝐷 respectively. To un-normalize the relit images exposure, we linearly interpolated the two products elementwise, according to the chosen target light strength 𝛾 and ambient strength 𝛼:

𝑃relit = ((1 −𝛾) 𝛼𝐸off +𝛾𝐸on) ((1 −𝛾) 𝛼𝐺off +𝛾𝐺on)

((1 −𝛾) 𝛼𝐷off +𝛾𝐷on) (2) As explained in Section 3.2, under a theoretically perfect capture

process the target light residual ion − ioff is non-negative, corresponding with an addition of light to the scene. In Figure B.3 we

consider the negative component e = clip(ion − ioff, −∞, 0) as the error residual and the positive component d = clip(ion−ioff, 0, ∞) as a clean approximation. We measure both the relative error ∥𝑒∥2

∥𝑑∥2 , and the percentage of negative entries in ion − ioff across our captured dataset, and report the statistics in Figure B.3.

C LIGHT SOURCE CONTROL WITH DIFFUSION MODELS

- C.1 Parametric Conditioning

This section provides additional details on the conditioning scheme introduced in Section 3.4. We divide condition signals to spatial (localized) inputs and global conditions (defined to have spatially constant value).

Spatial Conditions. These include the input (source) image, a depth map, a segmentation mask of the target light source(s), relative light intensity change and the target light’s RGB color. The input image is passed through the model’s VAE, reducing it’s spatial dimensions to the diffusion models latent spatial dimensions 𝐻 × 𝑊 × 4. The scaled intensity change mask, the target color RGB mask and the depth map are resized to match the spatial dimensions of the latent input noise. Spatial conditions are stacked together,

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Fig. A.1. 3D scenes. Indoor scene examples used to generate our synthetic dataset.

overall forming a tensor with dimensions 𝐻 ×𝑊 × 9 merged via a learned 1 × 1 (zero-initialized) convolution layer, which reduces the number of channels to the base model’s 𝐻 ×𝑊 × 4. This tensor is concatenated to the input noise.

- C.2 Masking Procedure

To obtain semantic segmentation masks for the photography capture training set, the "ON" image from each pair was manually annotated with axis aligned bounding boxes which were used to condition the segmentation model [Ravi et al. 2024]. During inference the same procedure is applied, where a user provides a segmentation mask of the target light source (in the examples presented in the paper a bounding box condition was given to the segmentation model as for the training data). Note that the human annotation introduces variance to the distribution of segmentation masks on which the model is trained, which mirrors the variance expected by users at inference time. For the synthetically generated training and evaluation sets, a separate mask is rendered showing the geometry of each light fixture.

- C.3 Intensity Sampling

During training, for each sample we probabilistically choose a light component (light source or ambient lighting). Next, we sample two intensity values from the modified light component, and another intensity value for the unchanged component, all from the precomputed values in the dataset. Note that the model is trained in both directions, with intensity change values in the range [−1, 1], and

Table 3. Synthetic augmentations. ground truth similarity metrics for different synthetic data distributions, computed on the paired synthetic evaluation set.

PSNR↑ SSIM↑

Training Datasets

Binary Intensity Color Binary Intensity Color

Org. + Rand. 27.7 29.4 28.8 0.869 0.884 0.882 Org. + Rand. (reduced) 26.4 28.7 28.5 0.845 0.872 0.878 Org. 25.9 28.3 27.8 0.842 0.870 0.870 Rand. 25.8 28.1 26.4 0.845 0.867 0.866

term light intensity changes (of either the target light source or ambient light) within −1, 1 as "binary". As endpoint intensities - where the light is either "on" or "off" - are more common in-the-wild, we increase their weight during training by probabilistically setting either the source image or the target to an endpoint image. Importantly, using a relative intensity scale allows for iterative refinement in successive edits, and enables use without an additional radiance estimation stage at inference.

D EXPERIMENTS D.1 Random scene augmentations

We test whether extending the synthetic dataset by adding light sources at random plausible locations improves downstream performance, focusing on synthetic data. Note that this augmentation changes the scene light transport, as opposed to rendering the same

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

1.0

AmbientLightIntensity

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

0.5

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

0.14

##### 0.0 0.3 0.7 1.0 Light Source Intensity

Fig. B.2. Post-process examples. A parametric sequence of images generated from a single raw photograph pair in the post-processing procedure in Section

- 3.2

[Figure 175]

Fig. B.3. Calibration Errors. Error magnitudes in the target light residual ichange = ion − ioff for captured photograph pairs in our dataset.

scene from multiple views. We duplicate light sources that are already present in the original scenes in order to avoid increasing light source or material diversity for this experiment. Table 3 shows that while using only augmented scenes yields a lower ground truth similarity, incorporating both original and augmented scenes improves ground truth similarity across all settings. To test whether this improvement is a direct result of increasing the sample size, we match the size of the original and augmented datasets, using half of each. The results in Table 3 (second line from the top) show a noticeable improvement across tasks.

- D.2 Effect of Synthetic

This section provides additional qualitative examples demonstrating the effect of adding the synthetic data to the training data. Figure D.4 demonstrates domain leakage as indicated by synthetic-like lighting showing a lack of over-saturated halos around the light source, which are present in the model trained only on captured images (first two examples).

- D.3 Comparisons

In this section we provide additional results and details on the IIW dataset and on our paired evaluation dataset.

RGB ↔ XImplementation Details. The RGB-X model itself was used to compute all conditions from the input and ground truth images (RGB->X), we then fed back the input conditions (X>RGB) while replacing the irradiance condition with the ground truth irradiance

D.4 User Study

To verify that the results of our quantitative and qualitative comparisons, we also conducted a user study over the in-the-wild evaluation set. In each question, we randomly sampled one of the evaluation examples and one of the comparing methods. We presented to the user the input image, the target light source to turn either on or off, and the editing results both from our and the selected comparing method. Users were instructed to choose which edited image result depicts a better lighting change, with respect to the input image and other visible and non-visible light sources in it. Overall, we collected 3200 answers from 100 users using the Amazon Mechanical Turk service. The results are summarized in Table 2, where we report the percentage of judgments in our favor compared to each method. As can be seen, most participants favored our method by a large margin.

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Nadav Magar, Amir Hertz, Eric Tabellion, Yael Pritch, Alex Rav-Acha, Ariel Shamir, and Yedid Hoshen

#### Input (Off) Synthetic Only Captured Only Mixed Target (On)

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

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

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Fig. D.4. Qualitative Comparison: the effect of adding synthetic data to the training dataset. We compare the results of the models trained on a mixture of real and synthetic data, and only on real data. The condition image is displayed in column (d), and the ground truth target at column (a). Orange Note how the model trained on both modalities more accurately predicts the complex shadows of the chair frame. Lime The mixed model manages to recreate the "X" shadowing pattern, created by the supper-position of the two ceiling lamps. Third row This type of light fixture is well represented in our synthetic dataset, producing accurate lighting and shadows. The model trained with captured data only doesn’t predict as well the light fixture type and position. Blue and Red show two areas where shadows and specular reflections overlap.

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

###### Fig. D.5. Qualitative Comparisons. Additional comparisons of binary light switch on indoor images from IIW dataset. On the left, the input image where green / red contours specify which light source(s) should be turned on / off respectively. The results were computed with the "together" tone-mapping condition and without color conditioning.

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

###### Fig. D.6. Qualitative Comparisons. Additional comparisons of binary light switch on indoor images from IIW dataset. On the left, the input image where green / red contours specify which light source(s) should be turned on / off respectively. The results were computed with the "together" tone-mapping condition and without color conditioning.

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

- Fig. D.7. Results on Paired Captured Evaluation Dataset: A sample of inference results on the binary task, used in the quantitative ablation in Table 1 in the main paper. Light Intensity Mask. the target light intensity mask was scaled and translated from the range [-1,1] to the range [0,1] for display purposes. Gray pixels represent zeros, white represent 1 (turn light on) and black represent -1 (turn light off). The results were generated with the tone-map condition "together", with no ambient change (0 value) and without color conditions (0 values).

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

- Fig. D.8. Additional Results on Paired Captured Evaluation Dataset: A sample of inference results on the binary task, used in the quantitative ablation in Table 1 in the main paper. Light Intensity Mask. the target light intensity mask was scaled and translated from the range [-1,1] to the range [0,1] for display purposes. Gray pixels represent zeros, white represent 1 (turn light on) and black represent -1 (turn light off). The results were generated with the tone-map condition "together", with no ambient change (0 value) and without color conditions (0 values).

#### Input Result Input Result

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

- Fig. D.9. Results on Outdoor Scenes: Input images show the contour of the target light source mask. Contour color represents the color condition given to the model. green contours specify negative intensity changes (turning off a light). The input images were selected to represent of different environmental lighting conditions and different outdoor light sources. The results were generated with the tone-map condition "together", with no ambient change (0 value).

Input -0.34 -0.67 -1.00

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

Input +0.20 +0.40 +0.60

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

+0.80 +1.00 +1.20 +1.40

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

- Fig. D.10. Effect of Tone-Mapping Condition: The first row demonstrates gradual dimming of a colored neon light, which is not represented in the training dataset. Note how red light reflected from the adjacent brick-wall is being dimmed as-well. This figure also shows inaccuracies of the model, where red light on the lower section of the wall should not be removed. Bottom Rows. The bottom row shows that the model can extrapolate to intensity change values outside of the interval [-1,1]. The results were generated with the tone-map condition "together".

Input +0.30 +0.60 +1.00

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

Input -0.40 -0.60 -1.00

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

Input +0.30 +0.60 +1.00

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

Input -0.30 -0.60 -1.00

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

- Fig. D.11. Additional Ambient Light Intensity Results: We show ambient light edits in indoor and outdoor scenes. In all the above images the tone-mapping condition was set to "together". Fourth row. our method successfully disentangles daylight entering the room through windows from direct illumination emanating from the lamp. Second row. When turning ambient light completely off the sky becomes too dark and no night sky illuminated is generated, this is a limitation arising from not using image pairs where ambient illumination changes, and from indoor bias in the training datasets. The results were generated with the tone-mapping condition "together".

#### Input 0.33 0.67 1.00

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

- Fig. D.12. Effect of Tone-Mapping Condition on Sequences. This figure demonstrates the learned effect of the tone-mapping condition at inference time. It is complementary to Figure 4 in the main paper which shows the application of the two tone-mapping strategies on training data. Note that the learned tone-mapping control is not completely not disentangled, which could be seen by the perceptible variance in light intensity for the "separate" condition. First and third rows. the sequences tone-mapped separately. Second and fourth rows. the sequences tone-mapped together.

