## arXiv:2411.07126v1[cs.CV]11Nov2024

[Figure 1]

2024-11-12

### Edify Image: High-Quality Image Generation with Pixel Space Laplacian Diffusion Models

###### NVIDIA1

[Figure 2]

[Figure 3]

A photo of a couple doing pottery together in a well-lit room A chameleon showing colorful scales

- (a) Text-to-image generation

[Figure 4]

- (b) Finetuning (c) Additional control (d) Panorama

[Figure 5]

Figure 1: Edify Image can generate photorealistic high-resolution images from text prompts. Our models support a range of capabilities, including (a) Text-to-image generation, (b) Finetuning, (c) Generation with additional control, and (d) Panorama generation. For (b) and (c), an example of a finetuning image and the control input are provided in the bottom left corner, respectively. Best viewed with Acrobat Reader. Click the panorama image to play the video clip.

#### Abstract

We introduce Edify Image, a family of diffusion models capable of generating photorealistic image content with pixel-perfect accuracy. Edify Image utilizes cascaded pixel-space diffusion models trained using a novel Laplacian diffusion process, in which image signals at different frequency bands are attenuated at varying rates. Edify Image supports a wide range of applications, including text-to-image synthesis, 4𝐾 upsampling, ControlNets, 360∘ HDR panorama generation, and finetuning for image customization.

1A detailed list of contributors and acknowledgments can be found in App. A of this paper.

© 2024 NVIDIA. All rights reserved.

https://research.nvidia.com/labs/dir/edify-image

#### 1. Introduction

The field of text-to-image synthesis has witnessed remarkable progress in recent years, with stateof-the-art models (Baldridge et al., 2024; Betker et al., 2023; Esser et al., 2024; Podell et al., 2023) generating increasingly realistic and diverse images from natural language descriptions. These models typically leverage large-scale diffusion-based architectures trained on billions of image-text pairs. The ability to generate high-resolution, photorealistic images has far-reaching implications across domains such as content creation, gaming, synthetic data generation, and the design of digital avatars.

In this technical report, we present Edify Image, a family of pixel-space diffusion models capable of generating high-resolution images with exceptional controllability. We train our models in a cascaded fashion, where a base model generates low-resolution images, and subsequent models progressively upsample the images from the previous stage. Our models are trained using a novel multi-scale Laplacian diffusion process, in which image signals at different frequency bands are attenuated at varying rates. This enables our models to capture and refine details with precision across multiple scales, resulting in photorealistic, pixel-perfect generations.

Using the Laplacian Diffusion Model formulation, we train a suite of diffusion models capable of generating images from various input signals.

- • Text-to-image models. We train a two-stage cascaded text-to-image diffusion model that can generate 1𝐾 resolution images from input text. Our model handles long text prompts, generates images with varying aspect ratios, exhibits improved fairness and diversity when generating human subjects, and can support the use of camera controls such as pitch and depth of field.
- • 4𝐾 Upsampler. We train an upsampler model that takes a 1𝐾 resolution image as input and upsamples it to 4𝐾 resolution. The upsampler involves fine-tuning the 1𝐾 resolution generator on 4𝐾 images with an additional low-resolution input conditioning. Our model is capable of synthesizing high-frequency details while remaining faithful to the low-resolution input image.
- • ControlNets. We train ControlNets on the 256 resolution base models for various modalities, including depth, sketch, and inpainting mask. The 1𝐾 and 4𝐾 base models can be reused for upsampling. Our model can generate high-quality images while enabling flexible structural controls.

In addition, we also support the following two capabilities:

- • 360∘ HDR Panorama Generation. We design an algorithm for generating 4𝐾, 8𝐾 and 16𝐾 resolution HDR panorama from the input text. We utilize the base text-to-image models to perform sequential inpainting in which images from different perspectives are generated in an overlapping manner and stitched together with consistency.
- • Finetuning. We propose an algorithm for finetuning the base text-to-image models on a small subset of reference images. Our model is especially capable of generating various hyper-realistic humans with identities consistent with the reference set.

#### 2. Dimension-Varying EDM

Edify Image models are diffusion-based generators operating in the pixel space. Existing pixel-space generators employ a series of cascaded diffusion models in which subsequent stages upsample the low-resolution images produced in the previous stage, often leading to notorious artifact accumulation. To mitigate this issue, we introduce a new diffusion model that synthesizes large contexts in a single

diffusion process. The key innovation is the introduction of a multi-scale diffusion process, termed the Laplacian Diffusion Model. This model simulates a resolution-varying diffusion process in the time domain by simultaneously decaying different image frequency bands at different rates.

###### 2.1. Preliminary

- 2.1.1. Diffusion Model

Given an image data distribution 𝑝0(x0), where x0 ∈ 𝒳, a diffusion model derives a family of distributions 𝑝𝑡(x𝑡) by injecting i.i.d. Gaussian noise into data samples during the diffusion forward process, such that x𝑡 = x0 + 𝜎𝑡𝜖 with 𝜖 ∼ 𝒩(0,𝐼) and 𝜎𝑡 monotonically increasing with respect to time 𝑡 ∈ [0,𝑇]. To simulate the diffusion backward sampling process, which generates samples by iteratively removing noise starting from Gaussian noise, diffusion models obtain the score function ∇x𝑡

log 𝑝𝑡(x𝑡) (i.e., the gradient of log-probability) via a denoising score matching objective (Ho et al., 2020; Karras et al., 2022; Song et al., 2020; Vincent, 2011):

𝐿𝑡(𝜃) = Ex

0,x𝑡[‖𝐷𝜃(x𝑡,𝑡) − x0‖22], (1)

where 𝐷𝜃 : 𝒳 × [0,𝑇] → 𝒳 is a time-conditioned neural network that tries to denoise the noisy sample x𝑡. Assuming an infinite capacity of 𝐷𝜃, the predictions of the optimal model are related to the score function via Tweedie’s formula (Efron, 2011):

xˆ𝑡 := 𝐷𝜃(x𝑡,𝑡) = x𝑡 + 𝜎𝑡2∇x𝑡

log 𝑝𝑡(x𝑡), (2)

which represents the minimum mean squared error (MMSE) estimator of x0 given x𝑡 and 𝜎𝑡. We follow the precondition design for 𝐷𝜃(x𝑡,𝑡) and log normal distribution 𝜎 during training introduced in Karras

- et al. (2022).

- 2.1.2. Image Laplacian Decomposition The Image Laplacian Decomposition is a multi-scale representation technique that decomposes an image into a series of progressively lower-resolution images, capturing different frequency bands at each level. This hierarchical structure consists of a sequence of band-pass filtered images, where each level represents the difference between two successive versions of the original image. Specifically, we consider a simple image downsampling operation as a way to obtain the low-frequency component, where high-frequency details from the original image are effectively removed. We denote upsampling and downsampling operations as up(.) and down(.), respectively. We illustrate this decomposition in Fig. 2. Through this decomposition, for simplicity, we assume there are three resolution stages, i.e. x = x(1) + up(x(2)) + up(up(x(3))), where:

x(3) = down(down(x)), (3a) x(2) = down(x) − up(x(3)), (3b) x(1) = x − up(down(x)). (3c)

Note that even though we use a 𝑑 dimensional vector to present x(𝑖), their internal representation can be more compact. For example, we can use a downsampled 𝑑/16 dimensional vector to represent x(3) to tackle high-resolution image synthesis.

###### 2.2. Laplacian Diffusion Model We introduce our diffusion process, which is built upon image Laplacian decomposition using an intuitive approach. At its core, we explicitly control how image signals at different frequency bands are attenuated and synthesized at varying rates rather than entangling them together and allowing them

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

up up up

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

up up

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

attenuation factor

##### Image Pyramid

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

time

Forward Noising

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Backward Sampling

Noise Pyramid

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

up

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

up

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

- Figure 2: Laplacian diffusion for multi-resolution image generation. (Top) Image Laplacian Decomposition. Each image sample x can be decomposed into a set of components. The example shows three components, x = x(1) + up(x(2)) + up(up(x(3))). This decomposition is implemented using basic upsampling and downsampling operations, where each component corresponds to different frequency bands. The function 𝜇(x,𝑡) represents a weighted sum of these components across different frequency spaces. (Middle) Forward Noising Process. Components are attenuated at different rates, with higher frequencies attenuated more rapidly than lower ones. We use the decaying background color in the top part of the figure to illustrate the attenuation factors. As a result, the signal-to-noise ratio (SNR) diminishes faster in the high-frequency components, allowing them to be discarded without significant loss of information once their attenuation coefficients approach zero. (Bottom) Backward Sampling Process. Denoisers are trained at multiple stages to generate images at various resolutions. We decompose the noise into a noise Laplacian pyramid. The Laplacian Diffusion process synthesizes higher-resolution images by first upsampling a lower-resolution noisy sample and then denoising it, with random noise injected into the corresponding components during upsampling. When operating solely at the lowest resolution, the process reduces to standard EDM.

to be corrupted through an implicit approach. A rigorous treatment can be derived with stochastic differential equations. We start with a 3-stage image Laplacian decomposition in Eq. (3). The formulation can be extended to more stages easily.

###### 2.2.1. Forward Noising Process

We generalize the isotropic forward process utilized in standard diffusion models, where x𝑡 ∼ 𝒩(x0,𝜎𝑡𝐼), to a more flexible formulation: x𝑡 ∼ 𝒩(𝜇(x0,𝑡),𝜎𝑡𝐼). In this context, 𝜇 is defined as:

𝜇(x0,𝑡) =

∑︁3

𝛼𝑡(𝑖)x0(𝑖), (4)

𝑖=1

where the coefficients 𝛼𝑡(𝑖) are attenuation factors. We define attenuation factors to be monotonically non-increasing with respect to the diffusion time 𝑡. This forward process can be expressed as the summation of three diffusion models operating in different subspaces:

∑︁3

∑︁3

𝛼𝑡(𝑖)x(0𝑖) + 𝜎𝑡𝜖 =

𝛼𝑡(𝑖)x(0𝑖) + 𝜎𝑡𝜖(𝑖), (5)

x𝑡 =

𝑖=1

𝑖=1

where 𝜖(𝑖) can be obtained via the Laplacian decomposition as in Eq. (3). We also visualize this process at the bottom of Fig. 2. Most existing diffusion models choose 𝛼𝑡(𝑖) that are invariant to subspace, thereby entangling the three components at any given time 𝑡. Consequently, the denoising network is required to operate across all three subspaces to reconstruct the original signals for all diffusion processes.

In our study, we employ distinct rates for the set 𝛼𝑡(𝑖), such that the components in the high-frequency branch decay more rapidly than those in the lower-frequency branch, as illustrated in Fig. 2. We identify

two critical time points, 𝑡(1*) and 𝑡(2*), at which 𝛼𝑡(1) and 𝛼𝑡(2) respectively diminish to zero. Consequently, beyond these timestamps, a more compact, low-resolution representation suffices for the signal, as

the high-frequency components no longer contribute to x𝑡.

###### 2.2.2. Training

We utilize the same loss function, as defined in Eq. (1), to train the denoising network 𝐷𝜃(x𝑡,𝑡). However, the Laplacian forward process introduces greater flexibility in network design, allowing us to operate across different resolution ranges. Moreover, this approach greatly improves the training efficiency by separating the low-frequency and high-frequency components of the image, allowing the model to adapt more quickly. As illustrated in Fig. 2, we can train a large network for the whole time interval: [0,∞). Alternatively, we can employ a mixture of experts approach, where a low-resolution denoiser 𝐷𝜃(3) is trained on 𝒳(3) for the entire time range [0,∞), a mid-resolution denoiser 𝐷𝜃(2) is trained on 𝒳(2) ∪𝒳(3) for the interval [0,𝑡(2*)), and a high-resolution denoiser 𝐷𝜃(1) is trained on 𝒳 for the interval [0,𝑡(1*)).

###### 2.2.3. Backward Sampling Process Laplacian Diffusion Models offer a flexible approach to synthesizing samples at various resolutions, thanks to the Laplacian decomposition and the utilization of a mixture of denoiser experts trained across different denoising ranges. We illustrate the different sampling modes in Fig. 2.

- • To synthesize the lowest resolution images in 𝒳(3), the backward sampling process simplifies to that of regular diffusion models, as it involves only a single stage based on 𝐷𝜃(3).
- • For generating mid-resolution images, we can combine the outputs of the denoisers 𝐷𝜃(3) and

𝐷𝜃(2). Specifically, we perform backward sampling in 𝒳(3) up to 𝑡(2)*, then transition to using 𝐷𝜃(2) to complete the remaining sampling trajectory.

- • To synthesize the highest resolution images, we switch the sampling trajectory from 𝐷𝜃(2) at the

sampling timestamp 𝑡(1)*, and rely on 𝐷𝜃(1) to generate the remaining high-resolution details.

We include more discussions and derivations in App. B. We extend high-order sampling algorithms (Zhang and Chen, 2022; Zhang et al., 2023) from standard diffusion models to the Laplacian Diffusion Model following the similar spirit introduced by Zhang et al. (2022).

- 2.2.4. Switching Between Different Resolutions When synthesizing low-resolution images, we completely disregard the signals from the high-frequency band to reduce computational costs. This approach is justified by the fact that the signal-to-noise ratio is zero during the corresponding time interval. However, to synthesize high-resolution images, it

is necessary to switch the sampling trajectory by upsampling the noisy image x𝑡 and reintroducing the high-frequency noise components. We illustrate this concept using a low-resolution image (𝑟) and assume that we are at a noise level 𝜎 (under resolution 𝑟). Transitioning to a high-resolution (𝑅) image with a noise level 𝑅/𝑟 · 𝜎 involves two steps: first, upscale the low-resolution image to high resolution, and second, add the corresponding high-resolution Gaussian noise component, multiplied by (𝜎 · 𝑅/𝑟).

We justify the approach using a concrete example. Consider that a noisy state x𝑡 at resolution (𝑟) can be decomposed as:

###### x(𝑟) + 𝜎𝜖(𝑟), (6)

where 𝜖(𝑟) is the resolution-𝑟 standard Gaussian noise. Let us define 𝜖(𝑅) to be the standard Gaussian noise of resolution 𝑅, such that:

𝜖(𝑟) = down(𝜖(𝑅),𝑅/𝑟) · 𝑅/𝑟, (7)

where the coefficient is due to the averaging of Gaussian noise. Thus, we have that:

up(x(𝑟) + 𝜎𝜖(𝑟)) ⏟ ⏞

###### +𝜎𝑅/𝑟 · (𝜖𝑅 − up(down(𝜖(𝑅),𝑅/𝑟)) ⏟ ⏞

(8)

upscale

add noise

= up(x(𝑟)) + 𝜎𝑅/𝑟 · 𝜖𝑅 + 𝜎 · up(𝜖(𝑟) − down(𝜖(𝑅),𝑅/𝑟) · 𝑅/𝑟) (9) = up(x(𝑟)) + 𝜎𝑅/𝑟 · 𝜖𝑅, (10)

where the last equality is from Eq. (7). Here, we have translated the low-resolution Gaussian noise to high-resolution Gaussian noise.

#### 3. 1𝐾 Generation Using Two-Stage Laplacian Diffusion Models

To generate images of 1024 resolution, we train a two-stage cascaded pixel-space diffusion model where the first model generates an image of 256 resolution while the second model upscales the image to 1024 resolution. Our training pipeline is provided in Fig. 3. The 256 resolution model is trained on the full noise range [0,𝜎256𝑚𝑎𝑥], while the 1𝐾 model operates on a smaller noise range [0,𝜎1024𝑚𝑎𝑥] (𝜎1024𝑚𝑎𝑥 < 𝜎256𝑚𝑎𝑥). During inference, we first generate the 256 resolution sample by running the full sampling loop on the base model. Then, we apply forward diffusion on the generated sample with 𝜎 = 𝜎1024𝑚𝑎𝑥 and denoise the image using the upsampler. All our models are trained with the objective discussed in Section 2.

- 3.1. Model Architecture We use U-Net-based architecture for the base and upsampling models following Ho et al. (2020); Ramesh et al. (2022); Saharia et al. (2022). The U-Net model typically consists of a sequence of residual and attention blocks that progressively downsample (or upsample) feature maps with skip connections. For high-resolution synthesis, the spatial resolution of feature maps increases, which makes the

[Figure 50]

###### U-Net

[Figure 51]

InverseWavelet

ResidualBlock

ResidualBlock

ResidualBlock

ResidualBlock

ResidualBlock

ResidualBlock

Base model (256 res)

Upsampler (1024 res)

Upsample + forward noising

transform

transform

Wavelet

###### …

[Figure 52]

[Figure 53]

- Figure 3: Model architecture. As shown in the left panel, our diffusion models use a U-Net based architecture with a sequence of residual blocks with skip connections. We use wavelet and Inverse wavelet transform at the beginning and end of the network to bring down the spatial resolution of the images. In the right panel, we show how the 256 and 1𝐾-resolution models are combined in a 2-stage cascade to generate the 1024-resolution image.

computation of attention maps expensive. To address this issue, we propose to operate on the smaller spatial resolution by using invertible wavelet transforms at the beginning and the end of the network, inspired by Hoogeboom et al. (2023). This is illustrated in Fig. 3. We use 2-level Haar wavelets to downsample the images in the pixel space from resolution (3 × 𝐻 × 𝑊) to (48 × (𝐻/4) × (𝑊/4)). This reduces the number of spatial tokens in the attention layers by a factor of 16, dramatically improving the training efficiency. Our base model consists of 2.7𝐵 parameters, while the 1𝐾 upsampler consists of 1.6𝐵 parameters.

- 3.2. Conditioning Inputs Prior text-to-image generators Podell et al. (2023); Ramesh et al. (2022) mainly use text embeddings from CLIP Radford et al. (2021) and T5 Raffel et al. (2020) models as conditional inputs. To provide better controllability to our image generators, we use the following conditioning inputs.

- • T5 embeddings. We use text embeddings from the T5-XXL model. To enable support for long prompt generation, we use a sequence length of 512.
- • Camera embeddings. To provide better camera control while generating images, we additionally condition the synthesis using camera attributes. For each image, we obtain integer-valued pitch and depth of field annotations. These annotations are then passed through an embedding layer and used as a conditional signal during training.
- • Media type. Each image in the dataset is assigned a media type label such as ‘Photography’ and ‘Illustration’, which is then used as a conditional attribute during training.

All conditional embeddings are then concatenated along the sequence dimension and used in the cross-attention layer in the U-Nets. During training, we apply random embedding dropout to each of the conditional embeddings. This ensures that the model can generate using any combination of conditional signals. When all embeddings are dropped out, we obtain the unconditional score.

- 3.3. Data We train various Edify Images models for our AI foundry partners, who are responsible for sourcing the image dataset. To achieve better prompt alignment, having detailed and descriptive captions is critical, as shown in Betker et al. (2023). So, in addition to the ground truth captions, we use LLM based captioners to obtain long descriptive captions. During training, we randomly sample captions from ground truth and AI generations. This allows our model to generate images from both long and short text prompts.

[Figure 54]

[Figure 55]

A photo of a robot holding a brush and painting a picture. A group of friends sitting around a campfire.

[Figure 56]

[Figure 57]

A beautiful nature scene with snowy mountains, purple sky and bioluminescent blue icy lake

Large golden human brain sculpture on a marble pedestal in a modern museum

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

A photo of an old goldsmith making jewelry.

A dining table with lots of dishes and pastries.

A beautiful palace made of gold by a lake.

An astronaut meditating in a lush green forest.

###### Figure 4: Samples generated by our text-to-image model with 16:9, 1:1 and 9:16 aspect ratios.

[Figure 62]

[Figure 63]

A photo of a woman in a coffee shop reading a book. She is wearing glasses. Her hair is black in color. She is wearing a fancy hat and an orange shirt. There is a green coffee cup on a saucer placed next to the book. There are many plants in the background with string lights on the ceiling.

A happy couple is having a romantic dinner at a restaurant. In the table, there is a burger, fries, two glasses of red wine and a salad. The man is wearing a blue suit and the woman is wearing a green dress.

[Figure 64]

[Figure 65]

Watercolor painting of a nature scene. There are mountains in the background. There is a field with white and blue flowers in the foreground. There is a lake under the mountain where a boat is sailing. The sun is setting with orange sky. There are many hot air balloons floating in the sky.

A 4K dslr photo of a mouse kayaking in a stream of water set against the backdrop of a lush green forest. The mouse is wearing a Hawaiian shirt and a straw hat. There are several blocks of cheese stacked in the kayak.

- Figure 5: Long prompt generation. Edify Image can faithfully generate images from long descriptive prompts.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

- Figure 6: Human diversity. Our model is able to generate images with good gender and race diversity. The prompt used is "A studio portrait of a smart CEO".

Descending view Eye level view Ascending view

[Figure 74]

[Figure 75]

[Figure 76]

A photo of a beautiful ancient castle from medieval times

[Figure 77]

[Figure 78]

[Figure 79]

A photo of three women hanging out in a city

- Figure 7: Camera controls - Pitch.

- 3.4. Aspect Ratios We support five common aspect ratios in our image generator - 1 : 1, 4 : 3, 3 : 4, 16 : 9, and 9 : 16. Samples in the dataset are first grouped into one of these five bins according to the closest aspect ratio. During each training iteration, we randomly sample a batch of examples from a bin and train the diffusion model. We provide the aspect ratio information to the model using learnable spatial positional encodings. The positional encoding parameters are defined for the base 1 : 1 aspect ratio. For all other aspect ratios, we perform spatial interpolation to the required feature dimensions. We observed that the aspect ratios in our dataset had an imbalanced distribution. Despite this imbalance, the model was able to perform well across all aspect ratios.

Shallow depth of field Deep depth of field

[Figure 80]

[Figure 81]

A photo of a chef wearing a chef hat cooking in a kitchen. The background has kitchen supplies

[Figure 82]

[Figure 83]

A photo of a woman drinking coffee in a coffee shop

###### Figure 8: Camera controls - Depth of field.

- 3.5. Training We train both base and upsampler models for 2.7𝑀 iterations. The base model was trained with a global batch size of 4096, while the upsampler was trained with a batch size of 2048. We use AdamW optimizer with a constant learning rate and a warmup following Balaji et al. (2022). After 1.5M iterations, we use aesthetic weighted training, in which loss per sample is multiplied by a normalized aesthetic score computed using an aesthetic model.
- 3.6. Results Samples generated by our text-to-image model are shown in Fig. 4. Our model is able to generate highly detailed photorealistic images adhering to the input text prompt across a diverse set of categories nature, humans, animals, food, etc. We also show results across three aspect ratios - 16 : 9, 1 : 1, and

###### 9 : 16. The model can generate high-quality images in both aspect ratios despite having very few 9 : 16 images in the dataset. Our model can also generate images adhering to long and descriptive captions,

- as shown in Fig. 5.

- 3.6.1. Fairness and Diversity In Fig. 6, we show the results of our text-to-image model on some human generation prompts. We observe that our model can generate images with sufficient race and gender diversity.
- 3.6.2. Camera Control As discussed in Section 3.2, we condition our diffusion models on pitch and depth of field attributes during training. Fig. 7 shows the generations as we vary the pitch to ascending, eye level, and descending

view while using the same text prompt. We observe that the pitch of the resulting image changes as specified in the input. In Fig. 8, we vary the depth of field attribute to shallow and deep. The images with shallow depth of field have blurred backgrounds, while those with deep depth of field have all regions in focus.

#### 4. 4𝐾 Upsampling

Our Edify suite of image generation models also includes 4𝐾 upsampling, which helps users generate highly detailed images. While the 1𝐾 generator generates high-quality images with strong adherence to the input text prompts, the 4𝐾 upsampler adds additional fine-grained details to the 1𝐾 resolution image and outputs 4𝐾 resolution images.

- 4.1. Approach In theory, given our model formulation, it is very easy to train a 4𝐾 resolution diffusion model by simply adding a new resolution level while training. However, there is typically a large gap in the amount of data available at higher resolutions compared to lower-resolution data. This is indeed the case for our dataset as well, wherein the number of good-quality images with 4𝐾 or higher resolution is less than 1% of the data available to train the 4𝐾 model. Here, we refer to ‘good-quality’ images as those images that pass our criteria for containing high-frequency content and are aesthetically pleasing, which is required to train a good-quality upsampling model. To address this, we opt to utilize the existing 1𝐾 generator as the base for the upsampling model.

By scaling the noise levels appropriately, we can generate a high-quality 4𝐾 image directly from a pre-trained 1𝐾 generator. In the case of upsampling, similar to SDEdit, we can start with a low-resolution image, resize it to the desired resolution, add noise to it based on the forward diffusion process discussed in Sec. 2, and finally denoise it iteratively using our base 1𝐾 model to obtain the upsampled image. One issue with this approach, however, is that the model may change the content in the initial low-res image to a degree that may not be desirable to the user. To overcome this challenge, we design the upsampler as a ControlNet which conditions the base model on the clean low-resolution input image. Finally, we fine-tune the base model with the low-resolution ControlNet on a smaller number of 4𝐾 images available to us. This helps the model in two ways:

- • The pre-trained base model has not seen any high-frequency content which is crucial for generating 4𝐾 images. Fine-tuning on the 4𝐾 data enables the model to generate such details.
- • The clean low-resolution image conditioning allows the model to access the original content of the noisy input image and prevents it from deviating too much from the original.

Additionally, we utilize reconstruction guidance Ho et al. (2022) during sampling to further control the degree of change to the original low-resolution image.

- 4.2. Results Fig. 9 and Fig. 10 show how the upsampling model is able to add more details to the 1𝐾 resolution output from the base model. The difference is clearly observed by zooming into a region in the image showing the high-frequency content. Note that images in Fig. 9 and Fig. 10 are all upsampled to 4𝐾 using our upsampler.
- 5. Generation with Additional Control We add additional control to the Edify Image model by training ControlNet encoders following Zhang

- et al. (2023). Fig. 11 illustrates the architecture of the Edify Image model with ControlNet, which will be detailed in the following sections.

[Figure 84]

Edify Image: High-Quality Image Generation with Pixel Space Laplacian Diffusion Models

[Figure 85]

[Figure 86]

###### 1K Resolution 4K Resolution

- Figure 9: 4𝐾 Upsampling results. Full (top) and zoomed-in images (bottom) show the additional details.

[Figure 87]

Edify Image: High-Quality Image Generation with Pixel Space Laplacian Diffusion Models

[Figure 88]

[Figure 89]

###### 1K Resolution 4K Resolution

- Figure 10: 4𝐾 Upsampling results. Full (top) and zoomed-in images (bottom) show the additional details.

Hint Input Blocks

[Figure 90]

Pre-trained Base Model

[Figure 91]

# …

[Figure 92]

# …

+

Inpainting Task Indicator

[Figure 93]

[Figure 94]

# …

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

+ +

+

Replace Inpaint Outpaint …

Masked Images

Image Input Blocks

- Figure 11: Model architecture with additional control inputs. The base model is frozen when training the ControlNet encoders. The Image Input Blocks are initialized from the base model U-Net. The Hint Input Blocks are randomly initialized.

- 5.1. Approach After the base model is pre-trained, we freeze the model parameters and introduce an additional encoder whose parameters are partially initialized from the first half of the base model’s U-Net. As the control input, such as depth and sketch maps, may have different dimensions from images, we add several extra blocks, called Hint Input Blocks, to transform the control input into feature maps that will be added to the features from the noisy image input. Additionally, by scaling the control input feature maps (i.e., control weight), we can achieve the controllability of different strengths.

We view inpainting as another controlled image generation problem, similar to sketch and depthcontrolled generation, with the partial image and inpainting mask as the control input. We consider three sub-tasks for inpainting:

- • Replace: the unknown area in the image is an entire semantic area, which means the mask shape strictly follows the object shape. This is useful for replacing objects or backgrounds while we do not want to change the object shape.
- • Inpaint: the unknown area is not a semantic area and could partially cover both the background and foreground.
- • Outpaint: the unknown area is at the image boundary. This is usually called outpainting in the literature but can be viewed as a special case of inpainting.

The left side of Fig. 11 showcases the examples of the three sub-tasks. Only one shared inpainting model is trained for all sub-tasks. We use a one-hot vector to indicate different tasks, which is expanded to the image size and concatenated with the masked image and inpainting mask to serve as the control input.

- 5.2. Training We compute Canny edges, HED edges, and depth maps from input RGB images and use them to train the edge/depth2image models. For inpainting, we generate random masks or use object masks to train the inpainting model. Following Zhang et al. (2023), we only train the additional encoder and keep the base model frozen during training.

[Figure 100]

- Figure 12: Results with additional control inputs for inpainting, depth, and edge. For each input condition, we generate 3 variants using different text prompts.

- 5.3. Results Fig. 12 shows example results with various control inputs. The model can generate high-quality images while following the image structure indicated by the control input. Furthermore, we demonstrate

[Figure 101]

###### Figure 13: Results with different control weight values for depth-to-image and edge-to-image.

the effect of different control weights in Fig. 13, using edge and depth inputs as an example. The generated image can be aligned to the input more strictly with a higher control strength.

#### 6. 360∘ HDR Panorama Generation

Building on the foundation described in previous sections, we developed a high-dynamic range (HDR) 360-degree panorama generator. Given a text prompt and (optionally) a corresponding example image from a single viewpoint, the system generates omnidirectional equirectangular projection panoramas

- at 4𝐾, 8𝐾, or 16𝐾 resolution (Fig. 14). The generated outputs can provide content for 3D virtual reality headsets and backdrops for movies and games. Thanks to the high-dynamic range output, it can be used as image-based lighting (IBL).

- 6.1. Approach Unlike the case of images, which are cheap to obtain and available at scale on the Internet, gathering HDR panoramas is time-consuming. A single panorama requires capturing and combining multiple images across different directions and exposure levels. The amount of available HDR panorama data is orders of magnitude less than that used to train successful foundation image models. To address this data limitation, our algorithm relies on the base Edify Image model to provide a general text-to-image capability and assemble multiple generated images into the desired panorama. The limited panorama data are used only to fine-tune this process and for HDR estimation.

The algorithm adopts a sequential inpainting approach in which a number of conventional perspective images are synthesized with the foundation model and stitched together, with overlap from preceding images, to ensure continuity. During synthesis, each image is warped into equirectangular coordinates and projected into the coordinates of the neighboring image to provide the overlap region. The zenith (sky) and nadir (ground) images are inpainted with overlaps from all longitudinal images. The inpainting is trained as a controlnet, with an image containing the overlap area providing the control signal.

After we generate the panorama, we feed it to an LDR2HDR network to convert the low dynamic range

[Figure 102]

- (a) sunset at a lookout point in a gravel parking lot with blue sky and a few autumn maple trees and beautiful smokey mountains in the background, scenic nature, inspiring, landscape panoramic, mountains.

[Figure 103]

- (b) flat sand beach by a lake in the swiss alps mountains at noon with beautiful swiss alps mountains in the background, god rays, scenic nature, inspiring, landscape panoramic.

[Figure 104]

- (c) moss and grass plains in scottish highlands, scotland, remote, photography, wilderness, moody cloudy sky, rain, bluffs in background.

- Figure 14: Example 360 panorama generation results. The input prompts are described under each image. We also show zoomed-in crops at the right to better show the details in the images.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

(a) ev+0 (b) ev-2 (c) ev-5 (d) ev-9

- Figure 15: Crops of an HDR panorama at different exposure stops. This panorama image has 19 stops of dynamic range, showing the sun and bright clouds with values ‘‘above white’’.

(LDR) image to a high dynamic range (HDR) image. The LDR2HDR network is a multi-scale U-Net where we first generate a low-resolution HDR image and then concatenate it with the high-resolution LDR input to generate the high-resolution HDR output. To train the network, we convert the ground truth HDR dataset into LDR images and ask the network to reconstruct the original HDR input. For better training stability, we train the network to predict intensity values in logarithmic space.

- 6.2. Results Fig. 14 shows the results for our panorama synthesis at 16k resolution. Note that we are able to generate consistent panoramic scenes that properly follow the input prompt. We are also able to synthesize fine details for the trees, grass, etc, which are essential to make the results look realistic.

On the other hand, Fig. 15 demonstrates the result of our HDR generation from LDR input, which correctly assigns high-intensity values to bright objects like the sun and clouds. It also predicts a wide dynamic range (e.g., 19 stops) of intensities (crucial for IBL applications).

- 6.3. Limitations The panorama generator application shares assumptions and limitations of 360∘ panoramas in general; specifically, the panorama shows views in any direction from a single location. As a consequence, parallax is not possible, and translating the viewpoint results in visible distortions unless the translation is very small.

Another limitation is that while the generated panels are pairwise consistent, there is not necessarily any global consistency to the lighting. We hope to address this issue in future work. Note that this is not a crippling limitation in either backdrop or IBL applications: In the case where the panorama is visible as a backdrop, every individual view is plausible, and viewers may not notice the issue without additional study. In image-based lighting applications, when the sun is visible, it alone is usually responsible for most of the lighting effect due to its vastly greater intensity relative to reflected light.

- 7. Finetuning for Customization

We explore the Edify Image model’s capability to adapt to new personalization and stylization tasks. First, we describe our approach and then showcase several use cases, including single and multi-subject personalization, as well as single and multi-subject stylization. Finally, we demonstrate how the finetuned model can seamlessly integrate with various pre-trained frozen Edify ControlNets.

- 7.1. Approach Our finetuning approach does not modify the architecture of the Edify Image models, and we keep the text encoders frozen. We finetune only a subset of parameters in the cross-attention layers of the U-Nets, which accounts for just 3% of the total U-Net parameters. We finetune both the 256 and 1024-resolution U-Nets for 1500 steps.

- 7.2. Results We finetuned our models on four different datasets, each demonstrating the model’s ability to handle various customization tasks: single-subject personalization, multi-subject personalization, single-subject stylization, and multi-subject stylization. All the images in this section are upsampled by our 4𝐾 model in Sec. 4.

###### Learning a single human

We finetuned the Edify Image model using the finetuning data shown in Fig. 23. Fig. 16 demonstrates the model’s capability to generate images of the person at different ages and in various outfits, none of which were included in the training data.

Age Variation Scenario Variation

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

- Figure 16: The finetuned model is capable of generating realistic images of her at different ages and in a variety of scenarios that were not included in the finetuning dataset (Fig. 23).

###### Learning multiple humans

We also finetuned the model on a dataset (Fig. 24) that included multiple subjects. Interestingly, this dataset contains only three images with two individuals together, while the remaining 96% of the training images feature a single person. Despite the limited instances of multiple subjects, the finetuned model accurately generates images depicting both individuals engaging in various activities, as shown in Fig. 17. To distinguish between multiple subjects, distinct names were used for each individual in the training prompts.

[Figure 113]

[Figure 114]

[Figure 115]

###### Figure 17: The finetuned model can generate images featuring two individuals in various settings and correctly distinguish between them based on their names in the text prompts.

Learning a single style We finetuned the model on an icon dataset shown in Fig. 25. The results are presented in Fig. 18, demonstrating the model’s ability to produce clear and sharp line drawings.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

A rabbit. A woman is weightlifting. A man is playing soccer. A man is practicing Tai Chi.

###### Figure 18: The finetuned model successfully replicates the iconographic style with high fidelity.

Learning multiple styles

We also finetuned the model on the dataset in Fig. 26 to enable it to learn multiple styles. Different style names, such as "Epic" and "Line Art" were used in the training prompts to help the model distinguish among various styles. The results are shown in Fig. 19.

Learned styles from finetuning data Known styles in the base model

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Epic Line art Watercolor Comic sketch

###### Figure 19: The finetuned model is capable of learning multiple new styles while retaining knowledge of existing ones.

- 7.3. Compatibility with Edify ControlNet Our finetuning approach maintains the model architecture, allowing the finetuned model to be easily integrated with pre-trained, frozen ControlNet modules. In Fig. 20, we demonstrate that the finetuned U-Net can still effectively work with inpainting, sketch, and depth ControlNets, while faithfully preserving the learned subject in the controlled generation.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

- Figure 20: ControlNet compatibility. The finetuned U-Nets remain compatible with our inpainting, sketch, and depth ControlNets, which are not finetuned on the personalization datasets. The top row shows the control inputs, while the bottom row shows the generations.

- 7.4. Ablation Study We conducted an ablation study to investigate the impact of training data diversity on the generalization ability of the finetuned model. As shown in Fig. 21, we finetuned the model using two datasets: one containing the subject’s photos taken over six years, and another containing only recent photos. Although both datasets include images of the subject in her 20s, the first dataset makes it easier to generate the subject at younger or older ages (e.g., in her 30s or 40s) and yields more diverse outputs.
- 8. Related Work

###### Text-to-Image Generation

Diffusion models have emerged as the dominant approach for high-resolution image generation since the seminal work of Ho et al. (2020). For text-to-image synthesis, two main paradigms have gained popularity: pixel-space (Ramesh et al., 2022; Saharia et al., 2022) and latent-space diffusion (Baldridge et al., 2024; Betker et al., 2023; Podell et al., 2023; Rombach et al., 2022). Pixel-space models typically employ a cascaded architecture, where a base model generates low-resolution images, and subsequent models progressively upscale the generated images to higher resolutions. DALLE2 (Ramesh et al.,

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Training data includes photos taken over 6 years

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Training data includes only recent photos

age = 16 age = 26 age = 36 age = 46

FangyinWei, a professional {age}-year old woman sitting upright at a desk with one hand holding a pen in an office, smiling and engaged with her laptop, short, curly hair that frames her face, dressed in a white shirt and light brown vest.

- Figure 21: Effect of data diversity. Both training datasets lack explicit age labels in the captions and contain only images of the subject in her 20s. However, a more diverse training dataset facilitates generating the character across a broader range of ages.

2022) conditions the pixel-space diffusion on CLIP (Radford et al., 2021) text embeddings, while Imagen (Saharia et al., 2022) uses T5 (Raffel et al., 2020) embeddings. eDiff-I (Balaji et al., 2022) trains an ensemble of expert models, each specializing in a specific noise range, to enhance the generation quality. Latent-space models, on the other hand, employ an autoencoder to compress images into a low-dimensional latent representation, upon which a diffusion model is trained. Stable diffusion and Stable Diffusion XL (Podell et al., 2023) utilize U-Net based architectures for both autoencoders and diffusion models, with additional CLIP text conditioning. DALLE3 (Betker et al., 2023) trains the diffusion model using upsampled prompts from LLMs for generating images with long descriptive prompts. Stable Diffusion 3 (Esser et al., 2024) adopts a Diffusion Transformer (DiT) Peebles and Xie (2023) based architecture to scale the latent space models up to 8𝐵 parameters.

###### Generation with Additional Control

In addition to purely text-based image generation, many other methods aim at adding more control signals. They can be easily classified as training-free methods (Bansal et al., 2023; Chen et al., 2023; Meng et al., 2022; Xue et al., 2023), or methods that require further training to existing text-to-image models (Huang et al., 2023; Ju et al., 2023; Li et al., 2023; Mou et al., 2024; Qin et al., 2023; Zhang et al., 2023; Zhao et al., 2024). In general, training-based methods perform better than training-free methods. The most well-known is ControlNet (Zhang et al., 2023), which adds an identical encoder branch and exclusively trains that branch.

###### Panorama and HDR Synthesis

Panorama generation conditioned on text input has seen its emergence following the popularity of text-to-image synthesis using diffusion models. Most existing methods adopt a divide-and-conquer approach by first generating smaller image patches and then stitching them together (Chen et al., 2022; Li and Bansal, 2023; Tang et al., 2023; Wang et al., 2024; Zhang et al., 2024, 2023). For HDR synthesis from LDR images, recent trends have been using deep neural networks with the aid of either a perceptual loss (Liu et al., 2020; Santos et al., 2020) or a GAN loss (Wang et al., 2023, 2022).

###### Finetuning

A substantial amount of work has been devoted to personalization and stylization for text-to-image generative models. Training-based personalization approaches include Arar et al. (2024); Chen et al. (2023); Gal et al. (2022); Han et al. (2023); Hu et al. (2021); Kumari et al. (2023); Marjit et al. (2024); Ruiz et al. (2022); Shah et al. (2023); Tewel et al. (2023); Voynov et al. (2023); Xie et al. (2023); Yeh et al. (2023); Zhao et al. (2023). And training-based stylization methods are explored in Dong et al. (2022); Shah et al. (2023); Sinha et al. (2023); Sohn et al. (2023); Zhao et al. (2023). Our finetuning approach is more closely related to these training-based methods. Another branch of research focuses on fast finetuning or tuning-free techniques, which rely on pretrained image encoders or hypernetworks for personalization or stylization. The works in this area include, but are not limited to, Arar et al. (2023); Chen et al. (2024); Gal et al. (2023); Jia et al. (2023); Li et al. (2024); Ma et al. (2023); Rout et al. (2024); Ruiz et al. (2024); Shi

- et al. (2024); Valevski et al. (2023); Wang et al. (2024,); Wei et al. (2023); Xiao et al. (2023); Yuan et al.

(2023); Zeng et al. (2024).

#### 9. Conclusion

In this work, we presented Edify Image, a suite of image generation models producing high-fidelity images trained on a large dataset of image-text pairs. We proposed a novel multi-scale diffusion model, called Laplacian Diffusion Model, in which different image frequency bands are decayed at different rates in the diffusion process. Additionally, we explored several intriguing capabilities that can be adapted from our base model, including ControlNets, 4𝐾 upsampling, finetuning, and 360∘ panorama generation.

#### A. Contributors and Acknowledgements

- A.1. Core contributors Text-to-image model: Yogesh Balaji, Qinsheng Zhang, Jiaming Song, Ming-Yu Liu Super-resolution: Ting-Chun Wang, Siddharth Gururani, Seungjun Nah, Ming-Yu Liu ControlNets: Ting-Chun Wang, Yu Zeng, Grace Lam, Ming-Yu Liu 360∘ Panaroma Generation: Ting-Chun Wang, J. P. Lewis, Seungjun Nah, Ming-Yu Liu Finetuning: Jiaojiao Fan, Xiaohui Zeng, Yin Cui, Ming-Yu Liu Data Processing: Jacob Huffman, Yunhao Ge, Siddharth Gururani, Fitsum Reda, Seungjun Nah, Yin Cui, Arun Mallya, Ming-Yu Liu
- A.2. Contributors Yuval Atzmon, Maciej Bala, Tiffany Cai, Ronald Isaac, Pooya Jannaty, Tero Karras, Aaron Licata, Yen-Chen Lin, Qianli Ma, Ashlee Martino-Tarr, Doug Mendez, Chris Pruett, Fangyin Wei
- A.3. Acknowledgements We thank Timo Aila, Samuli Laine, Gal Chechik, Tsung-Yi Lin, Chen-Hsuan Lin and Zekun Hao for useful research discussions. We are grateful to Alessandro La Tona, Amol Fasale, Arslan Ali, Aryaman Gupta, Brett Hamilton, Devika Ghaisas, Gerardo Delgado Cabrera, Joel Pennington, Jason Paul, Jashojit Mukherjee, Jibin Varghese, Lyne Tchapmi, Mitesh Patel, Mohammad Harrim, Nathan Hayes-Roth, Raju Wagwani, Sydney Altobell, Thomas Volk and Vaibhav Ranglani for engineering and testing support. Special thanks to Andrea Gagliano, Bill Bon, Si Moran and Grant Farhall of Getty Images for providing useful feedback on our image generators, and to Dade Orgeron, Steve Chappell, Lucas Brown and Alex Ambroziak of Shutterstock for providing feedback on our panaroma generations. We also thank the NVIDIA Creative team and Peter Pang for providing stylization finetuning data. Finally, we would like to thank Amanda Moran, Sivakumar Arayandi Thottakara, John Dickinson, Herb Woodruff, Dane Aconfora, Yazdan Aghaghiri, Yugi Guvvala, David Page and Andrew Morse for the computing infrastructure support.

#### B. More Discussions on Laplacian Diffusion Models

- B.1. Diffusing Signals at Different Resolution In addition to our work, other concurrent works (Chen, 2023) also explore the diffusion effects at various resolutions and propose adjusting noise level sampling during training for different resolutions. We reiterate this from the perspective of signal-to-noise ratio and derive noise scaling factors in the diffusion process. First, let us consider the the average pooling (down) and nearest neighbor upsampling (up) operations. If we average pool the Gaussian noises, then each value in the downsampled tensor would have a lower variance. This is because in the case of 2 × 2 average pooling, assuming independent Gaussian random variables with 𝒩(0,1), we get

1 4𝒩(0,4) ∼

- 1

- 2𝒩(0,1). (11)

1 4

(𝒩(0,1) + 𝒩(0,1) + 𝒩(0,1) + 𝒩(0,1)) ∼

The reduced variance comes from summing independent Gaussian variables. From the definition of signal-to-noise ratio, we can see that signal-to-noise ratio will double when we downsample the noisy image. We illustrate the point in Fig. 22.

[Figure 138]

[Figure 139]

- Figure 22: (Top) Noise-free images x at different resolutions, 256,128,64,32. (Bottom) Noisy images x + 0.02𝜖 at different resolutions, 𝜖 has zero mean and identity matrix as covariance. From left to right in the bottom row, the images look less and less noisy, because the signal-to-noise ratio increases when we downsample the noisy images.

- B.2. Derivations Forward Diffusion Process For simplicity, we consider splitting the current sample x(0) into two subspaces, corresponding to the effective resolution of 𝑟 and 𝑅. As the signal to noise ratio is different for different resolutions, we use the variable 𝑡 to describe the ‘‘time’’ of the process. We use the notations 𝐿 and 𝐻 to represent low-frequency and high-frequency subspaces, respectively.

###### x(𝐿) = up(down(x(0),𝑅/𝑟)) (12) x(𝐻) = x − x(𝐿) (13)

The forward process also treat the two differently:

- • x(𝐿)(𝑡) = x(𝐿)(0) + 𝜎(𝑡)𝜖(𝐿), where 𝜖 ∼ 𝒩(0,𝐼).
- • x(𝑅)(𝑡) = 𝛼(𝑡)x(𝑟)(0) + 𝜎(𝑡)𝜖(𝐻).

Specifically, 𝛼(𝑡) is a function that vanishes after a certain time (denoted as 𝑡(𝐿)). We use EDM formulation to derive the drift and diffusion coefficients.

dx = 𝑓(𝑡)x + 𝑔(𝑡)d𝜔 (14)

leads to x(𝑡) = 𝑠(𝑡)x(0) + 𝑠(𝑡)2𝑢(𝑡)2𝜖, where:

𝑠(𝑡) = exp(︂∫︁ 𝑡

𝑓(𝜉)𝑑(𝜉))︂ and 𝑢(𝑡) =

0

√︃∫︁ 𝑡

𝑔(𝜉)2 𝑠(𝜉)2

d𝜉. (15)

0

For the 𝐿 subspace:

and for the 𝐻 subspace:

𝑠(𝑡) = 1, 𝑓(𝑡) = 0, 𝑢(𝑡) = 𝜎(𝑡),

𝑔(𝑡) = √︂d𝜎(𝑡)2 d𝑡

,

𝑠(𝑡) = 𝛼(𝑡), 𝑓(𝑡) =

dlog 𝛼(𝑡) d𝑡

,

𝜎(𝑡) 𝛼(𝑡)

𝑢(𝑡) =

,

𝑔(𝑡) = √︂d(𝜎(𝑡)2/𝛼(𝑡)2) d𝑡

𝛼(𝑡)

Backward Diffusion Process The backward diffusion process can be derived as:

𝑔(𝑡)2∇x log 𝑝𝑡(x)]︂d𝑡, (16)

dx = [︂𝑓(𝑡)x −

- 1

- 2

or equivalently as:

[︀

]︀

d𝑡, (17)

𝑠˙(𝑡)x/𝑠(𝑡) − 𝑠(𝑡)2𝑢˙(𝑡)𝑢(𝑡)∇x log 𝑝𝑡(x)

dx =

so we can derive the ODE as follows. For the 𝐿 subspace:

d𝑡 ∇x log 𝑝𝑡(x)]︂d𝑡, (18) and for the 𝐻 subspace:

dx = [︂−

d𝜎(𝑡)2

1 2

dx = [︂

𝜎(𝑡)(˙𝜎(𝑡)𝛼(𝑡) − 𝛼˙(𝑡)𝜎(𝑡)) 𝛼(𝑡) ∇x log 𝑝𝑡(x)]︂d𝑡. (19)

dlog 𝛼(𝑡) d𝑡

x −

- C. Finetuning Training Data The training images used for finetuning experiments in Sec. 7 are provided below.

[Figure 140]

###### Figure 23: Training images used for single-subject personalization.

[Figure 141]

###### Figure 24: Training images used for multi-subject personalization.

[Figure 142]

###### Figure 25: Training images used for single-subject stylization.

[Figure 143]

###### Figure 26: Training images used for multi-subject stylization.

#### References

- [1] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H. Bermano. Domain-agnostic tuning-encoder for fast personalization of text-to-image models. In SIGGRAPH Asia, 2023. 24
- [2] Moab Arar, Andrey Voynov, Amir Hertz, Omri Avrahami, Shlomi Fruchter, Yael Pritch, Daniel Cohen-Or, and Ariel Shamir. Palp: Prompt aligned personalization of text-to-image models. arXiv preprint arXiv:2401.06105, 2024. 24
- [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 11, 23
- [4] Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, Kelvin Chan, Yichang Chen, Sander Dieleman, Yuqing Du, Zach Eaton-Rosen, et al. Imagen 3. arXiv preprint arXiv:2408.07009, 2024. 2, 22
- [5] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In CVPR, 2023. 23
- [6] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2023. 2, 7, 22, 23
- [7] Hong Chen, Yipeng Zhang, Simin Wu, Xin Wang, Xuguang Duan, Yuwei Zhou, and Wenwu Zhu. Disenbooth: Identity-preserving disentangled tuning for subject-driven text-to-image generation. arXiv preprint arXiv:2305.03374, 2023. 24
- [8] Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. arXiv preprint arXiv:2304.03373, 2023. 23
- [9] Ting Chen. On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972, 2023. 25
- [10] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. In NeurIPS, 2024. 24
- [11] Zhaoxi Chen, Guangcong Wang, and Ziwei Liu. Text2light: Zero-shot text-driven hdr panorama generation. ACM Transactions on Graphics (TOG), 2022. 23
- [12] Ziyi Dong, Pengxu Wei, and Liang Lin. Dreamartist: Towards controllable one-shot text-to-image generation via positive-negative prompt-tuning. arXiv preprint arXiv:2211.11337, 2022. 24
- [13] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association,

2011. 3

- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2, 23
- [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 24

- [16] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG), 2023. 24
- [17] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In ICCV, 2023. 24
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS,

2020. 3, 6, 22

- [19] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022. 12
- [20] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. Simple diffusion: End-to-end diffusion for high resolution images. In ICML, 2023. 7
- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685,

2021. 24

- [22] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778,

2023. 23

- [23] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 24
- [24] Xuan Ju, Ailing Zeng, Chenchen Zhao, Jianan Wang, Lei Zhang, and Qiang Xu. Humansd: A native skeleton-guided diffusion model for human image generation. In ICCV, 2023. 23
- [25] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 3
- [26] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 24
- [27] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. In NeurIPS, 2024. 24
- [28] Jialu Li and Mohit Bansal. Panogen: Text-conditioned panoramic environment generation for vision-and-language navigation. In NeurIPS, 2023. 23
- [29] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023. 23
- [30] Yu-Lun Liu, Wei-Sheng Lai, Yu-Sheng Chen, Yi-Lung Kao, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Single-image hdr reconstruction by learning to reverse the camera pipeline. In CVPR, 2020. 23
- [31] Yiyang Ma, Huan Yang, Wenjing Wang, Jianlong Fu, and Jiaying Liu. Unified multi-modal latent diffusion for joint subject and text conditional image generation. arXiv preprint arXiv:2303.09319,

2023. 24

- [32] Shyam Marjit, Harshit Singh, Nityanand Mathur, Sayak Paul, Chia-Mu Yu, and Pin-Yu Chen. Diffusekrona: A parameter efficient fine-tuning method for personalized diffusion model. arXiv preprint arXiv:2402.17412, 2024. 24

- [33] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 23
- [34] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, 2024. 23
- [35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 23
- [36] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 7, 22, 23
- [37] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv preprint arXiv:2305.11147, 2023. 23
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 7
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 23
- [40] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 7, 23
- [41] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 6, 7, 22
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, 2022. 22
- [43] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024. 24
- [44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. arXiv preprint arXiv:2208.12242, 2022. 24
- [45] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In CVPR, 2024. 24
- [46] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 6, 22, 23
- [47] Marcel Santana Santos, Ren Tsang, and Nima Khademi Kalantari. Single image hdr reconstruction using a cnn with masked features and perceptual loss. ACM Transactions on Graphics (TOG), 2020. 23

- [48] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. arXiv preprint arXiv:2311.13600,

2023. 24

- [49] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In CVPR, 2024. 24
- [50] Animesh Sinha, Bo Sun, Anmol Kalia, Arantxa Casanova, Elliot Blanchard, David Yan, Winnie Zhang, Tony Nelli, Jiahui Chen, Hardik Shah, et al. Text-to-sticker: Style tailoring latent diffusion models for human expression. arXiv preprint arXiv:2311.10794, 2023. 24
- [51] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983, 2023. 24
- [52] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [53] Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. Mvdiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. In NeurIPS,

2023. 23

- [54] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In SIGGRAPH, 2023. 24
- [55] Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. Face0: Instantaneously conditioning a text-to-image model on a face. In SIGGRAPH Asia, 2023. 24
- [56] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 2011. 3
- [57] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-to-image generation. arXiv preprint arXiv:2303.09522, 2023. 24
- [58] Chao Wang, Ana Serrano, Xingang Pan, Bin Chen, Karol Myszkowski, Hans-Peter Seidel, Christian Theobalt, and Thomas Leimkühler. Glowgan: Unsupervised learning of hdr images from ldr images in the wild. In ICCV, 2023. 23
- [59] Guangcong Wang, Yinuo Yang, Chen Change Loy, and Ziwei Liu. Stylelight: Hdr panorama generation for lighting estimation and editing. In ECCV, 2022. 23
- [60] Hai Wang, Xiaoyu Xiang, Yuchen Fan, and Jing-Hao Xue. Customizing 360-degree panoramas through text-to-image diffusion models. In WACV, 2024. 23
- [61] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 24
- [62] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 24
- [63] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In ICCV, 2023. 24

- [64] Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 24
- [65] Enze Xie, Lewei Yao, Han Shi, Zhili Liu, Daquan Zhou, Zhaoqiang Liu, Jiawei Li, and Zhenguo Li. Difffit: Unlocking transferability of large diffusion models via simple parameter-efficient fine-tuning. In ICCV, 2023. 24
- [66] Han Xue, Zhiwu Huang, Qianru Sun, Li Song, and Wenjun Zhang. Freestyle layout-to-image synthesis. In CVPR, 2023. 23
- [67] Shih-Ying Yeh, Yu-Guan Hsieh, Zhidong Gao, Bernard BW Yang, Giyeong Oh, and Yanmin Gong. Navigating text-to-image customization: From lycoris fine-tuning to model evaluation. In ICLR,

2023. 24

- [68] Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. Inserting anybody in diffusion models via celeb basis. arXiv preprint arXiv:2306.00926, 2023. 24
- [69] Yu Zeng, Vishal M Patel, Haochen Wang, Xun Huang, Ting-Chun Wang, Ming-Yu Liu, and Yogesh Balaji. Jedi: Joint-image diffusion models for finetuning-free personalized text-to-image generation. In CVPR, 2024. 24
- [70] Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, and Jianfei Cai. Taming stable diffusion for text to 360 panorama image generation. In CVPR,

2024. 23

- [71] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 12, 15, 23
- [72] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022. 6
- [73] Qinsheng Zhang, Molei Tao, and Yongxin Chen. gddim: Generalized denoising diffusion implicit models. arXiv preprint arXiv:2206.05564, 2022. 6
- [74] Qinsheng Zhang, Jiaming Song, and Yongxin Chen. Improved order analysis and design of exponential integrator for diffusion models sampling. arXiv preprint arXiv:2308.02157, 2023. 6
- [75] Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, and Ming-Yu Liu. Diffcollage: Parallel generation of large content with diffusion models. In CVPR, 2023. 23
- [76] Brian Nlong Zhao, Yuhang Xiao, Jiashu Xu, Xinyang Jiang, Yifan Yang, Dongsheng Li, Laurent Itti, Vibhav Vineet, and Yunhao Ge. Dreamdistribution: Prompt distribution learning for text-to-image diffusion models. arXiv preprint arXiv:2312.14216, 2023. 24
- [77] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. In NeurIPS, 2024. 23

