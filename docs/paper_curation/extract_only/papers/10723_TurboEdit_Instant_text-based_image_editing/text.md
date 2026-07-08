# arXiv:2408.08332v1[cs.CV]14Aug2024

## TurboEdit: Instant text-based image editing

Zongze Wu Nicholas Kolkin Jonathan Brandt Richard Zhang Eli Shechtman

Adobe Research {zongzew, kolkin, jbrandt, rizhang, elishe}@adobe.com

Abstract. We address the challenges of precise image inversion and disentangled image editing in the context of few-step diffusion models. We introduce an encoder based iterative inversion technique. The inversion network is conditioned on the input image and the reconstructed image from the previous step, allowing for correction of the next reconstruction towards the input image. We demonstrate that disentangled controls can be easily achieved in the few-step diffusion model by conditioning on an (automatically generated) detailed text prompt. To manipulate the inverted image, we freeze the noise maps and modify one attribute in the text prompt (either manually or via instruction based editing driven by an LLM), resulting in the generation of a new image similar to the input image with only one attribute changed. It can further control the editing strength and accept instructive text prompt. Our approach facilitates realistic text-guided image edits in real-time, requiring only 8 number of functional evaluations (NFEs) in inversion (one-time cost) and 4 NFEs per edit. Our method is not only fast, but also significantly outperforms state-of-the-art multi-step diffusion editing techniques.

Keywords: Diffusion Models · Text-Guided Image Editing

### 1 Introduction

Large text-to-image diffusion models [23–25,27,28] have demonstrated remarkable ability to generate photorealistic and artistic images based on a text prompt, which allows people to visually express their ideas via natural language. Many methods try to repurpose the text-to-image diffusion for the real image editing tasks [2,4,9,11,20,22,33,35]. Specifically, given a real image, and a text prompt describing the target attribute, we want to disentangle alter the target attribute in the input image while keeping other attributes unchanged.

This task can be further decomposed into two subtasks, real image inversion and disentangled image editing. Real image inversion looks for a diffusion trajectory that can precisely reconstruct the input image, which relies on DDIM inversion [30], DDPM inversion [9,33], or their variants [2,11,20]. Disentangled image editing ensures only a single attribute change in image space, which relies on freezing attention maps [8,20,22], optimizing text embedding [34] or iterative classifier guidance [13]. Another line of works use the existing disentangled image editing methods to generate a large synthetic paired edit dataset, then train

Man→Fox Dog→Chihuahua Flower→Sunflower

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

+ Scarf + Leather Jacket Short Hair (0.5) Short Hair (1) Short Hair (1.5)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Replace his hair color to blue Change the kitchen to desert Make the image to child’s drawing

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- Fig. 1: We present a novel real-time text-based disentangled real image editing method built upon 4-step SDXL Turbo. Our method can handle both realistic and artistic images, supports manual or instruction-based prompt manipulation, and allow users to control the editing strength. We further show multi-attribute editing and continuous editing in Supplementary Fig 7.

a diffusion model to accept the input image and text instruction, and output the edited images in a supervised manner, therefore they do not need image inversion or disentangle editing tricks in inference [3,10,38,40].

While these methods show promising results, their speed is bottle-necked by iterative sampling of diffusion models, which often requires 50+ steps (> 5 seconds) to invert a real image and 30 − 50 steps (> 3 seconds) to generate a new edit. This dramatically slows down the iterative and trial-and-error nature of image editing.

Fortunately great strides have recently been made in the methods to distill diffusion models into more efficient variants which generate images in just 1-4 steps, taking less than half a second [16, 18, 29, 36]. Few-step diffusion models pose new challenges to real image editing, as methods developed for multi-step diffusion models cannot be effectively applied. The primary goal of our work is addressing these challenges and enabling high quality image editing using fast few-step diffusion models.

In particular, DDIM inversion [30], commonly used when inverting real images into the noise space of diffusion models, requires small step sizes and multiple steps (> 50). Using only 4 steps of DDIM inversion results in blurry reconstructed images and a significant loss of detailed structure. The inverted noise containing excessive information about the input image, which deviates significantly from Gaussian noise, the true distribution of the diffusion model’s noise

space. This distribution shift limits the ability for large structural changes and creates pronounced artifacts in Supplementary Fig 8.

Furthermore, while attention-based methods [8, 20, 22] are widely used in diffusion image editing, their direct application to few-step diffusion models poses challenges. These methods often involve freezing self-attention and crossattention maps to preserve structural similarity between the source and target images. Typically, attention maps are only frozen in the early generation steps to facilitate large structure changes and prevent artifacts. However, in the context of the one-shot diffusion model, attention control methods exert an overly restrictive influence on the generation process, leading to insufficient changes in the image space (horse to unicorn) or the introduction of artifacts (fox to dog). Even when applied solely during the initial generation steps of a four-step diffusion model, attention control can lead to either inadequate preservation of structure or the occurrence of artifacts, particularly in cases where the editing necessitates significant structural alterations in Supplementary Fig 9.

TurboEdit confronts both challenges, offering not only a fast feed-forward mechanism for accurately inverting real images, but also a simple and efficient mechanism to make disentangled edits. Our first contribution is an inversion network that predicts noise to reconstruct the input image, and is trained to iteratively correct the reconstruction image condition on the reconstruction in previous step. We demonstrate the effectiveness of our method on complex scenes with 2-3 objects. To the best of our knowledge, this is the first encoder based diffusion model inversion method.

Our second contribution is an analysis of an emergent property of the diffusion distillation process. Namely, we show that distillation leads to disentangled adherence to long detailed text prompts, allowing for precise attribute manipulation. By changing one attribute in the long detailed text prompt, only the corresponding attribute in the image space is altered, enabling users to easily edit the text prompt and obtain the desired disentangled edit image. This technique is very simple and requires no additional implementation effort, making it highly practical. We further show that the strength of editing could be controlled by a linear interpolation of the detailed text embedding. To allow users to input instructive text prompt, we utilize large language model to transform the descriptive source prompt and instructive prompt to descriptive target prompt, then input the target prompt to diffusion model, which enable instructive control in text-to-image diffusion model. Combined, these elements enable TurboEdit to make image edits that are high-quality, real-time, text-based, and disentangled.

Our method only requires 8 number of functional evaluations (NFEs) in inversion (one-time cost) and 4 NFEs per edit, compared to 50 NFEs in inversion and 30-50 NFEs per edit for methods based on multi-steps diffusion models. Despite being significantly faster (< 0.5 seconds instead of > 3 seconds per edit), our method shows better text image alignment and background preservation compared to methods based on multi-step diffusion models in both descriptive and instructive text prompt setting.

### 2 Related Works

#### 2.1 Text-To-Image diffusion models

Large text-to-image diffusion models [23–25,27,28] transform random Gaussian noise into natural images conditioned on text prompts through iterative denoising. While they produce high-quality images, they require 30-50 denoising steps and over 3 seconds per generation. Recently, distillation methods have been developed to speed up the process, enabling image generation in just 1-4 steps and less than 1 second [16,18,29,36]. These few-step distilled models offer the best balance between speed and quality, making them ideal for real-time image editing.

#### 2.2 Text-Based Image Editing

To edit an existing image using a text-to-image diffusion model, we first need to map a real image onto the reverse diffusion trajectory, ensuring the inverted images resemble the input image. Current methods often use DDIM inversion [30], requiring over 50 small diffusion steps. Null-Text inversion [20] improves DDIM’s reconstruction quality by optimizing the null text token at each timestep, but this process is time-consuming. DDPM inversion [9, 33] uses an over-complete noise representation for an inversion procedure to prevent error accumulation, resulting in perfect reconstruction for the input text prompt. LEdits++ [2] extends this idea with a higher-order solver and an automatic method to mask edits, preserving most of the image’s identity.

#### 2.3 Disentangled Editing with Generative Models

Attention-based methods [8,22] freeze self-attention and cross-attention maps to preserve structural similarity between the source and target images. Typically, attention maps are only frozen in the early generation steps to facilitate large structure changes and prevent artifacts. Similarly Cao et al. [4], propose a mechanism to directly share attention maps with the input image during generation, but this makes editing the texture of the foreground challenging and makes this method more appropriate to altering pose.

Other methods optimize text embedding tokens [20,34], or the model itself [12]; however the expensive optimization process per input image runs counter to our goal of real-time image editing. Another approach, pioneered by InstructPix2Pix, is to finetune a model on synthetic instruction-based image editing datasets [3]. While efficient, this method has poor disentanglement and requires expensive dataset preparation and training. In contrast our trained inversion network is easy to train, requires no training data beyond the original diffusion model, and offers more disentangled, identity preserving, edits.

### 3 Method

#### 3.1 Preliminary

The forward diffusion process gradually turns a clean image x0 into white Gaussian noise xT by adding Gaussian noise ϵt to it,

xt = √α¯tx0 + √1 − α¯tϵt (1) where α¯t controls noise schedule and ϵt is Gaussian noise. A network ϵˆθ is

trained to predict ϵt given xt, text prompt c and time step t with objective L(ˆϵθ) = Ex

0∼q;ϵt∼N(0,1)[∥ ϵˆθ(xt,c,t) − ϵt ∥2] (2)

We can easily rewrite the formulation from noise prediction to sample prediction

√1 − α¯tϵˆθ(xt,c,t) √α¯t

xt −

(3)

x0,t =

It usually takes 20-50 steps from a sampled Gaussian noise xT to a clean image x0. With the developed of distillation methods [16, 18, 29, 36], few-step diffusion models can obtains high quality images in 1-4 steps.

#### 3.2 Single Step Image Inversion

Current diffusion-based methods for real image editing have shown promising results in achieving high-quality disentangled edits [2–4,20,22,32]. However, these methods, which rely on multi-step diffusion models [24,27], are hindered by their computational demands, with each edit requiring at least 4-5 seconds, making

- them unsuitable for interactive applications. Moreover, these methods cannot be directly applied to few-step diffusion models due to fundamental differences in their design. For instance, many diffusion-based editing approaches rely on the DDIM inversion [30] or DDPM inversion [9] to project real images into diffusion noise space. However, DDIM inversion’s requirement for a small step size and a large number of inversion steps is inherently at odds with the design principles of few-step diffusion models. While DDPM inversion overfits to the input image and produce significant amount of artifact in the edited image. As illustrated in Supplementary Fig 8, both DDIM and DDPM inversion yield suboptimal editing results when applied to few-step inversion steps.

Several works [29,36] utilize adversarial loss to distill a multi-step diffusion model, and make the optimization target of few-shot diffusion models similar to GANs [7]. This inspire us to draw on ideas from the GAN inversion literature, where encoder based methods have been shown to be efficient and reliable [1,26, 31].

Consider a generator G (in our case SDXL-Turbo [29]) which accepts time step t, text prompt c, and noisy image xt = x0 +ϵt and outputs a reconstructed image x0,t. Given this, we predict the clean image x0,t from a noisy version as

x0,t = G(xt,c,t). We begin designing our inversion network using a single step approach where t = T. We train an inversion network Fsingle to predict xT, such that when inputting xT to G, x0,t will match x0. Leading to the loss function:

0∼q[∥ x0 − G(Fsingle(x0,c,T),c,T) ∥2] (4)

L(Fsingle) = Ex

The inversion network Fsingle is initialized from G (SDXL-Turbo), and the generator G is frozen during training. The information of input image x0 is stored in text prompt c (global information) and initial noise xT = Fsingle(x0,c,T) (spatial information). When we want to perform image editing, we use a new text prompt c′, and generate the edited image by

x′0,T = G(Fsingle(x0,c,T),c′,T), (5)

Despite its simplicity, the single-step encoder method is capable of impressive semantic edits while preserving background details, surpassing the performance of DDIM and DDPM inversion methods and emerging as the sole viable option for single-step inversion in Supplementary Fig 8. However, its results exhibit artifacts in regions such as hands and faces. The resulting images lack sharpness and contain salt-and-pepper noise, falling short of photorealistic quality. To combat this, we extend our method to multiple inversion steps.

#### 3.3 Multi-Step Image Inversion

In order to enhance the quality of image reconstruction, we leverage a multistep inversion approach which iteratively refines the reconstruction in each step, similar to the GAN inversion network proposed in ReStyle [1]. The inversion network F is designed to take the input image x0 along with the reconstruction from the previous step x0,t+1, and predict the injected noise ϵt for the current step. This injected noise ϵt, combined with the previous reconstruction x0,t+1, forms the new noisy image xˆt according to equation 1, which serves as input to G. Then we obtain a new reconstruction image x0,t according to equation 3. This yields the initial multi-step training objective:

0∼q[∥ x0 − G(xˆt,c,t) ∥2], xˆt = √α¯tx0,t+1 + √1 − α¯tF(x0,x0,t+1,c,t)

L(F) = Ex

(6)

It is worth emphasizing that generator G takes previous reconstruction x0,t+1 as input, therefore the loss function pushes F to output an ϵt which improve the previous reconstruction x0,t+1 relative to the input image x0. During training, we simulate the reconstruction from the previous step x0,t+1 using single step SDEdit [19]. Specifically, we add random Gaussian noise to input image to get xt+1, then input xt+1 to the generator to get x0,t+1. At maximum time step t = T, we use an all zeros matrix as x0,t+1.

[Figure 19]

##### TurboEdit: Instant text-based image editing 7

|[Figure 20]|
|---|

[Figure 21]

The image features a white plate filled with a delicious meal. The plate is topped with a variety of food items, including a piece of fish steak, asparagus, and tomatoes. The fish steak is placed towards the center of the plate, while the asparagus and tomatoes are scattered around it. The arrangement of the food items creates a visually appealing and appetizing presentation.

Captioning

[Figure 22]

| | | | | | | |
|---|---|---|---|---|---|---|
| |L|La|V|A| | |
| | | | | | | |

[Figure 23]

Edited caption

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

User-modified detailed caption 𝑐 → 𝑐′

Timestep 𝑡

Input image 𝑥

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

|[Figure 36]|
|---|

Reconstructions

Inversion

Input image

|Inv| |
|---|---|
| |𝑥|

|Inv| |
|---|---|
| |𝑥|

|Inv| |
|---|---|
| | |

|Inv| |
|---|---|
| | |

𝑐, 𝑡

𝑥 ,𝑐,𝑡 ,𝑐,𝑡 ,𝑐,𝑡 𝑥 ,𝑐,𝑡

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Noisetrajectory

𝐹

Singleinversioniteration

Our inversion network

Previous recon.

Predicted noise 𝜖

𝑥

𝑐′,𝑡 𝑐′,𝑡 𝑐′,𝑡 𝑐′,𝑡

𝑐, 𝑡

|Gen| |
|---|---|
| | |

|Gen| |
|---|---|
| | |

|Gen| |
|---|---|
| | |

|Gen| |
|---|---|
| | |

[Figure 43]

|[Figure 44]|
|---|

[Figure 45]

[Figure 46]

[Figure 47]

SDXL Turbo

Editing

Edits

[Figure 48]

Next recon. 𝑥

Final edited result

- Fig. 2: Given an input real image x0, we utilize the LLaVA to generate a detailed caption c. Users can modify c to create a new text prompt c′. The inversion process begins by feeding the x0, c, current time step t, and a previously reconstructed image x0,t+1 (initialized as a zero matrix) into the inversion network. This network then predicts the noise ϵt, which is subsequently input into a frozen SDXL-Turbo model to generate the new reconstruction image x0,t. Given the final inverted noise ϵt, along with c, we can use SDXL-Turbo to create an inversion trajectory and reconstruct x0,0, which is very similar to x0. Using the same noises ϵt and slightly different text prompt c′, starting from t = T to smaller t, the editing trajectory will be very similar to the inversion trajectory, and the generated image will closely resemble the input image, differing only in the specified attribute in c′.

Our analysis revealed that a naive implementation of this model results in predicted noise containing numerous high values (> 10) and excessive structural information from the input image, resulting in artifacts in the reconstruction image. Moreover, changing the text prompt had minimal effect on the output image. To address these issues, we employ the reparameterization trick [14] to constrain the injected noise to a distribution close to standard Gaussian. Instead of directly predicting the value of the injected noise, the inversion network outputs the mean and variance of each pixel, from which the injected noise is sampled. The KL loss required for this modification is:

0∼q[KL(F(x0,x0,t+1,c,t),N(0,1))] (7) This yields the final training objective:

LKL(F) = Ex

L(E) = LMSE(F) + λ ∗ LKL(F) (8)

Through experimentation, we determined that setting the λ = 10−6 achieved a favorable balance between reconstruction quality and editability. After training, we can perform iterative inversion as show in Fig 2, Algorithm 3.3. The

inversion process iterates from t = T to smaller t, with the intention of first encoding semantic information and then capturing finer details. The noise ϵt contain spatial information not explicitly encoded in c. Through experimentation, we determine that a four-step inversion is sufficient to faithfully reconstruct complex images and preserve facial identity in Figure 11. Given ϵt and new text prompt c′, we can generate a new image resembling the input image x0 while containing the target attribute in c′ as show in Algorithm 3.3. In summary, the inversion process takes 8 NFE (4 × 2) since each inversion step requires inference of both inversion network and generator. Once the image is inverted, all subsequent edit takes 4 additional NFE.

Algorithm 2 Iterative Editing

Algorithm 1 Iterative Inversion

Input target caption c′, {ϵT, ..., ϵ1} Optional Input mask m, {x0,T, ..., x0,1} x′0,T+1 = 0 for t=T to 1 do

Input real image x0, corresponding caption c x0,T+1 = 0 for t=T to 1 do

ϵt = E(t, c, x0, x0,t+1) xˆt = √α¯tx0,t+1

xˆt = √α¯tx0,t+1 + √1 − α¯tϵt x′0,t = G(t, c′, xˆt) if m is not None then

+√1 − α¯tF(t, c, x0, x0,t+1) x0,t = G(t, c, xˆt)

x′0,t = m ∗ x′0,t + (1 − m) ∗ x0,t Return x′0,1

Return {ϵT, ..., ϵ1}, {x0,T, ..., x0,1}

#### 3.4 Detailed Text Prompt Condition

Attention-based image editing methods [8,20,22] preserve the structural similarities between source and target images by freezing self-attention and crossattention maps. Although they work well in regular multi-step diffusion models, we show that it over constrains the structure of target images and trend to produce artifacts in both single-step or four-step diffusion model in Supplementary Fig 9.

To enable text guided image editing in few-step diffusion models, we propose an extremely simple method. Our intuition is that if the text prompt is highly detailed and encompasses semantic information across various aspects, modifying a single attribute in the text prompt will result in only a minor change in the text embedding. Consequently, the source and target sampling trajectories remain sufficiently close, resulting in generated images that are nearly identical except for the modified attribute in Fig 3. The same intuition applies to real image edit

- as we show in Supplementary Fig 10. Moreover, we can linear interpolate the detailed source and target text embeddings and generate a smooth interpolation in image space in Fig 1 and 4. Although it is hard for users to write a long text prompt, we can easily utilize ChatGPT [21] to expand a short text prompt (e.g., "please describe an image of a {short user-provided caption} in detail"), or use LLaVA [17] generate detailed captions of a given image.

A concurrent work also show image editing capability sololy based on text embedding [37] without freezeing attention map. We want to highlight the differ-

Black → Blond Trench → Down + Hat + Snow

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

ShortTextDetailedText

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

- Fig. 3: When presented with a concise source text prompt, minor edits in the text space can lead to substantial layout and structural changes in the image space. Conversely, making small text edits in a detailed text prompt tends to cause more disentangled changes in the image space. The results are from single step image generation with the same random seed. The captions and color-coded modification areas are provided below. Short Text: “a man wearing a trench coat (, a hat,) with black hair (in heavy snow).” Detailed Text: “a young man wearing a brown trench coat (and a hat,) and grey t-shirt with black hair, standing in front of subtropical flowers (in heavy snow). He is looking directly at the camera, giving a sense of focus and determination. The coat is open, revealing the man’s attire underneath. The overall scene is well-lit, with the man being the main subject of the image.”

ence between these two methods. To perform object replacement or style control, they substitute the keyword embeddings in the text embedding space, while we use long detailed text prompts and substitute the keyword directly in the text space. To control the editing strength, they rescale the weight of the descriptive word embedding or use singular value decomposition to discover editing direction in text embedding space, while we directly linearly interpolate the source and target text embedding.

#### 3.5 Local Mask

To facilitate localized edits, our method permits users to upload a binary mask indicating the region to be edited. We first Gaussian Blur the mask, then resize it to match the latent image size (64 × 64). Subsequently, we retain the edited image x′0,t at time step t only within the masked region, employing the inverted image x0,t for the remainder of the image, as outlined in Algorithm 3.3.

To provide an initialization of the masks, we propose utilizing a rough attention mask to denote the edited region. Drawing inspiration from the local blend mode in prompt2prompt [8], we automatically extract the attention mask

- at resolution 16×16 for words that only exist in either source or target prompt, sum it over channel dimension, divide it by its maximum value. This process yields a single-channel attention mask with values ranging from 0 to 1, where the edited region is characterized by high attention values and the unchanged region by low attention values. By default, we set the threshold at 0.6 and convert the continuous attention mask to a binary mask. Users can interactively adjust

Luxurious Sofa ←−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−→ Wood Chair

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

SUV ←−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−→ Bike

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

- Fig. 4: Given a detailed source text and corresponding target text, we can interpolating the text embeddings and generate a smooth interpolation in image space even for large structure change.

the threshold to control the size of the mask, as each edit (4 steps) requires less than 0.5 seconds. Although the attention mask is very rough, we show it can significantly improve the background and identity preservation in Supplementary Fig 12. In our figures we only use rough attention mask rather than accurate manual mask.

It is important to clarify that our approach utilizes attention masks solely to constrain the editing region, which is different from freezing attention maps for structural alignment in prompt2prompt [8]. Our method is orthogonal to attention freezing and can be combined with it. However, by default, we do not freeze attention maps at any time step, as doing so strongly constrains the object structure and tends to introduce artifacts in few-step diffusion models in Supplementary Figure 9.

#### 3.6 Instruction-based Editing

In many editing scenarios, users need to change multiple words from the source prompt to get the desired target prompt. For instance, when users want to change an image of a little dog to an image of a little cat, they need to change the word “dog” to “cat” and “puppy” to “kitten”, which can be cumbersome and unappealing.

Fortunately, instruction tuning and semantic editing in text space is well explored for large language models (LLM). We start with a base instruction like “please make the smallest change possible to the following sentence, but...” and

- then users only need to add task specific instruction like “change the dog to cat.” We concatenate the base instruction, user instruction, and source prompt, and input them to an LLM. The LLM figures out the best way to make the edit and generate the target prompt. In this way, the complex text edit is handled by the LLM, and users only need to input simple short instructions. To make this process more efficient and save memory, we reuse LLaVA as our LLM, but

any instruction-tuned LLM could be swapped in. LLaVA is built on top of a LLM Vicuna [5] and can still do text editing tasks even after being fine-tuned for vision and language tasks. Experiments show this simple method works well in Fig 1, 6 and Supplementary Table 3.

### 4 Experiments

#### 4.1 Training Details

In order to address computational and storage constraints, we select only 250k images larger than 512×512 pixels from an internal dataset, performing center cropping to obtain square images and resizing them to 512×512 pixels. To generate detailed captions, we input these images into the LLaVA model with the prompt “please describe the image as detailed as possible, including layout, objects, and color”. Subsequently, we precompute the image and text embeddings for the SDXL-Turbo model before training. The inversion network F is initialized from the SDXL-Turbo model, while the generator G (also SDXL-Turbo) remains fixed throughout training. Training is conducted over four different time steps (1000, 750, 500, 250), consistent with the approach employed in SDXLTurbo [29]. We utilize a learning rate of 10−5 and a batch size of 10, achieving model convergence within a day using eight A100 GPUs.

Method Background Preservation CLIP Similarity Efficiency PSNR LPIPS MSE SSIM Whole Edited Inverse Forward Steps ↑ 103 ↓ 104 ↓ 102 ↑ ↑ ↑ (s) ↓ (s) ↓ ↓

Null-Text Inv 27.03 60.67 35.86 84.11 24.75 21.86 56.98 3.66 50 Direct Inv 27.22 54.55 32.86 84.76 25.02 22.10 10.14 4.30 50 P2P-Zero 20.44 172.22 144.12 74.67 22.80 20.54 11.33 12.36 50 MasaCtrl 22.17 106.62 86.97 79.67 23.96 21.16 4.14 4.83 50 Inversion-Free 28.51 47.58 32.09 85.66 25.03 22.22 — 0.975 12 DDIM 18.59 177.96 184.69 66.86 23.62 21.20 0.344 0.341 4 Ours 29.52 44.74 26.08 91.59 25.05 22.34 0.668 0.508 4

- Table 1: Image editing comparison using descriptive text in PIE-Bench dataset. The efficiency is measured in a single H100 GPU. Our method achieves the best background preservation and clip similarity, while being significantly faster than other methods (except 4 steps DDIM).

#### 4.2 Quantitative Comparison

The PIE-Bench dataset [11] comprises 700 images, each associated with 10 distinct editing types. Each example includes a source prompt, target prompt, instruction prompt, and source image. In the descriptive setting, only the source and target prompts are used for text guidance, while in the instructive setting, only the instruction prompt is utilized.

However, the PIE-Bench dataset only provides short text prompts, whereas long, detailed text prompts are required to ensure disentangled edits and prevent

Original Ours InfEdit Pix2PixZero Ledits Ledits++

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

DogHusky+TopHatInFallWomanMan→→

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

- Fig. 5: We compare methods using descriptive text prompt as guidance. Despite requiring only four steps, our method outperforms multi-step methods, particularly in scenarios requiring significant structural changes for attributes such as adding a hat or transforming a man into a woman. In contrast, InfEdit and Pix2PixZero struggle with background and identity preservation. Similarly, Ledits and Ledits++ are unable

- to effectively handle large structural changes, as evidenced by their failure in adding a
- top hat or transforming a man into a woman.

artifacts. To ensure a fair comparison in the descriptive setting, we utilize short source and target prompts from the dataset and freeze the attention map [8] in the first sampling step. In the instructive setting, we employ LLaVA [17] to generate a long source caption and adhere to the short instruction from PIE-Bench to obtain a long target prompt as discuss in Section 3.6. Our results demonstrate that our method can better adhere to the text guidance and preserve the background compared to current state-of-the-art methods in both descriptive and instructive settings in Table 1 and Supplementary Table 3.

#### 4.3 Qualitative Comparison

Our method inherently supports various inversion steps. In the context of singlestep inversion, DDIM inversion [30] produces a significant number of artifacts, while DDPM inversion [9] generates images with the target attribute but fails to resemble the input image in Supplementary Fig 8. In contrast, our method successfully generates correct edits that closely resemble the input image, albeit with minor artifacts in the hand and face regions, as well as salt and pepper noise in the image. When considering four-step inversion, all methods exhibit superior

Original Ours MagicBrush HQEdit InstructPix2Pix

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Replace the cat with a dog

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Change the sweater to T-shirt

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Add snow to the scene

- Fig. 6: We evaluate methods utilizing instructive prompts as guidance. Although our method does not require any surprised training and only requires four sampling steps, it outperforms InstructPix2Pix and its variants, in terms of identity preservation (cat to dog) and text prompt alignment (sweater to T-shirt). It is worth to mention InstructPix2pIx and its variants require collecting a large scale surprised training set, computation intensive training, and multi-steps sampling.

results compared to their single-step counterparts. However, both DDIM inversion and DDPM inversion are prone to creating large artifacts when performing substantial structural changes (e.g., transforming a dog into a cat), whereas our method achieves photorealistic edits with significantly higher identity preservation in Supplementary Fig 8.

Furthermore, we conducted a comparative analysis between our four-step method and an image editing method based on a multi-step diffusion model using descriptive prompt as guidance in Fig 5. InfEdit [35] and Pix2PixZero [22] distorted the structure of objects such as houses, teddy bears, and guitars. Additionally, Ledits [32] and Ledits++ [2] struggled with large structural changes, such as adding a hat or transforming a man into a woman. In contrast, our method excels in performing realistic edits for both texture and structure changes while maintaining strong identity preservation. Compared to a concurrent work ReNoise [6] that also relies on SDXL-Turbo [29], we only need 8 NFE per inversion instead of 36 NFE, better preserve the face identity, and produce fewer artifacts in Supplementary Fig 14.

Similarly, we compare our method with other instruction based methods using instructive prompt as guidance in Fig 6. Although InstructPix2Pix [3] and its variant [10,38] need a large scale surprised training set, computation inten-

sive training, and multi-steps sampling, while our inversion network is trained unsupervised by reconstruction loss and only requires four sampling steps, our method outperform them in terms of identity preservation (cat to dog) and text prompt alignment (sweater to T-shirt).

- 4.4 Ablation Study We verify the necessity of each component in our framework through ablation study. First, we visualize the inversion results with varying numbers of inversion steps. Our findings indicate that multi-step inversion is essential for preserving facial identity and preventing blurring artifacts in Supplementary Fig 11. Subsequently, we calculate the reconstruction metric using 10k validation images, revealing a consistent improvement in reconstruction quality with an increasing number of inversion steps in Supplementary Table 2. Additionally, we demonstrate that a detailed text prompt condition is crucial for structure preservation and for preventing background artifacts in Supplementary Fig 10. Finally, we show local masking is important for preventing background structure change and identity shift in Supplementary Fig 12.
- 5 Limitations and Societal Impact

First, our method relies on LLaVA [17] to generate detailed captions. However, as we only perform a few-step inversion, the computationally intensive LLaVA model becomes a bottleneck. Hence, there is a need to explore alternative lightweight caption models to enable real-time image inversion. Secondly, while attention masks can effectively confine the editing region, they are often imprecise and may encompass nearby regions, which cannot be fully addressed by increasing the attention threshold. This imprecision can lead to slight identity shifts, particularly when the editing region is near a human face. We demonstrate that this issue can be mitigated by using a rough user-provided mask in Supplementary Fig 13. Lastly, our method is not able perform large pose change (turning a running man to a sitting man) in Supplementary Fig 15.

Our method, as a generative image editing tool, offers both creative opportunities and challenges. While it enables innovative image editing capability, it also raises concerns about the creation and dissemination of manipulated data, misinformation, and spam. One notable issue is the rise of deliberate image manipulation, commonly known as "deep fakes", which disproportionately affects women.

- 6 Conclusion

To the best of our knowledge, our method is the first work exploring image editing in the context of few-step diffusion models, and also the first to explore encoder based inversion in diffusion model. We demonstrate that disentangled controls can be easily achieved in the few-step diffusion model by conditioning on an (automatically generated) detailed text prompt. Our method enables users to make realistic text-guided image edits at interactive rates, running in milliseconds for both the inversion and editing processes.

### References

- 1. Alaluf, Y., Patashnik, O., Cohen-Or, D.: Restyle: A residual-based StyleGAN encoder via iterative refinement. In: Proc. ICCV (2021)
- 2. Brack, M., Friedrich, F., Kornmeier, K., Tsaban, L., Schramowski, P., Kersting, K., Passos, A.: Ledits++: Limitless image editing using text-to-image models. arXiv preprint arXiv:2311.16711 (2023)
- 3. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)
- 4. Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., Zheng, Y.: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465 (2023)
- 5. Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., Stoica, I., Xing, E.P.: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality (March 2023), https://lmsys.org/ blog/2023-03-30-vicuna/
- 6. Garibi, D., Patashnik, O., Voynov, A., Averbuch-Elor, H., Cohen-Or, D.: Renoise: Real image inversion through iterative noising. arXiv preprint arXiv:2403.14602

(2024)

- 7. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)
- 8. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022)
- 9. Huberman-Spiegelglas, I., Kulikov, V., Michaeli, T.: An edit friendly ddpm noise space: Inversion and manipulations. arXiv preprint arXiv:2304.06140 (2023)
- 10. Hui, M., Yang, S., Zhao, B., Shi, Y., Wang, H., Wang, P., Zhou, Y., Xie, C.: Hqedit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990 (2024)
- 11. Ju, X., Zeng, A., Bian, Y., Liu, S., Xu, Q.: Direct inversion: Boosting diffusionbased editing with 3 lines of code. arXiv preprint arXiv:2310.01506 (2023)
- 12. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6007–6017 (2023)
- 13. Kim, G., Kwon, T., Ye, J.C.: Diffusionclip: Text-guided diffusion models for robust image manipulation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2426–2435 (2022)
- 14. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)
- 15. Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. Communications of the ACM 60(6), 84–90 (2017)
- 16. Lin, S., Wang, A., Yang, X.: Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929 (2024)
- 17. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36 (2024)
- 18. Luo, S., Tan, Y., Huang, L., Li, J., Zhao, H.: Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378 (2023)

- 19. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021)
- 20. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6038– 6047 (2023)
- 21. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, 27730–27744 (2022)
- 22. Parmar, G., Kumar Singh, K., Zhang, R., Li, Y., Lu, J., Zhu, J.Y.: Zero-shot image-to-image translation. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–11 (2023)
- 23. Pernias, P., Rampas, D., Richter, M.L., Pal, C., Aubreville, M.: Würstchen: An efficient architecture for large-scale text-to-image diffusion models. In: The Twelfth International Conference on Learning Representations (2023)
- 24. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- 25. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022)
- 26. Richardson, E., Alaluf, Y., Patashnik, O., Nitzan, Y., Azar, Y., Shapiro, S., CohenOr, D.: Encoding in style: a stylegan encoder for image-to-image translation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2287–2296 (2021)
- 27. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 28. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022)
- 29. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042 (2023)
- 30. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)
- 31. Tov, O., Alaluf, Y., Nitzan, Y., Patashnik, O., Cohen-Or, D.: Designing an encoder for stylegan image manipulation. arXiv preprint arXiv:2102.02766 (2021)
- 32. Tsaban, L., Passos, A.: Ledits: Real image editing with ddpm inversion and semantic guidance. arXiv preprint arXiv:2307.00522 (2023)
- 33. Wu, C.H., De la Torre, F.: Unifying diffusion models’ latent space, with applications to cyclediffusion and guidance. arXiv preprint arXiv:2210.05559 (2022)
- 34. Wu, Q., Liu, Y., Zhao, H., Kale, A., Bui, T., Yu, T., Lin, Z., Zhang, Y., Chang, S.: Uncovering the disentanglement capability in text-to-image diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1900–1910 (2023)
- 35. Xu, S., Huang, Y., Pan, J., Ma, Z., Chai, J.: Inversion-free image editing with natural language. arXiv preprint arXiv:2312.04965 (2023)

- 36. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. arXiv preprint arXiv:2311.18828 (2023)
- 37. Yu, H., Luo, H., Wang, F., Zhao, F.: Uncovering the text embedding in text-toimage diffusion models. arXiv preprint arXiv:2404.01154 (2024)
- 38. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems 36 (2024)
- 39. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)
- 40. Zhang, S., Yang, X., Feng, Y., Qin, C., Chen, C.C., Yu, N., Chen, Z., Wang, H., Savarese, S., Ermon, S., et al.: Hive: Harnessing human feedback for instructional visual editing. arXiv preprint arXiv:2303.09618 (2023)

Shirt→Suit

Dog → Cat boy → Girl

Car → SUV

+Glasses

+Winter

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Straight Hair Black Hair Sweater → Suit + Sunglasses Image → Drawing

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

- Fig. 7: Our method can manipulate multiple attributes simultaneously (in the first row), or perform continuous editing (in the second row).

method steps L2↓10−3 PSNR ↑ LPIPS ↓ FID ↓ KID↓10−4 1 47.9 13.8 0.534 56.9 111.0

ddim 2 39.9 14.8 0.444 51.8 77.9 4 29.1 16.2 0.363 41.7 42.2 1 20.8 17.5 0.349 37.7 45.1

ours 2 8.90 21.7 0.220 24.9 10.5 4 5.87 23.7 0.163 18.7 4.33

- Table 2: Image reconstruction quality consistently improves with an increasing number of inversion steps. The evaluation metrics are computed over a validation set comprising 10,000 images. The LPIPS (Learned Perceptual Image Patch Similarity) loss is calculated using an AlexNet [15] backbone, following established practices in the field [39].

##### Ours DDIM DDPM Ours DDIM DDPM

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

DogCatYoungOld→→BuildingLakeShirtJacket→→

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

4 steps 1 step

- Fig. 8: Utilizing SDXL Turbo as the foundational model, we conduct a comparative analysis of DDIM inversion, DDPM inversion, and our method. While both DDIM and DDPM inversions completely fail in single step inversion, struggle to effect substantial structural changes (young2old) and tend to introduce pronounced artifacts (dog2cat) in 4 steps inversion. In contrast, our method performs disentangled edit even in a single step, and produce photo realistic edited images in 4 steps.

P2P Ours P2P Ours

HorseUnicorn→

[Figure 148]

[Figure 149]

[Figure 150]

FoxDog→

[Figure 151]

[Figure 152]

[Figure 153]

1 2 3 4 ours

CatTiger−>

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

- Fig. 9: Attention control methods [8, 20, 22] exert an overly restrictive influence on the generation process, leading to insufficient changes in the image space (horse to unicorn) or the introduction of artifacts (fox to dog) in single-step diffusion model. Even if we only apply attention control on the initial generation steps of a four-step diffusion model, it leads to either inadequate preservation of structure (1 and 2) or the occurrence of artifacts (3 and 4), particularly in cases where the editing necessitates significant structural alterations.

Original Short Text Detailed Text

[Figure 160]

[Figure 161]

[Figure 162]

+Sunglasses+GreyColor

[Figure 163]

[Figure 164]

[Figure 165]

- Fig. 10: We demonstrate a detailed text prompt is necessary for real image disentangled editing. If we use short text prompt for both inversion and editing, the edited images will have large structure change (in sunglasses example) or artifacts in background regions (in grey color example). In contrast, if we use long detailed text prompt

for inversion and editing, we can achieve disentangled edit without artifacts.

Woman short text: “a woman wearing sunglasses.” Woman detailed text: “The image features a woman wearing sunglasses with curly hair, wearing a brown sweater and smiling. She is posing for the camera, with her arm resting on her head. The woman is the main focus of the scene, and her smile is the central element of the image. The sweater she is wearing is a warm, earth-toned color, and her curly hair” Cat short text: “a grey color cat.” Cat detailed text: “The image features a grey color cat sitting in a woven basket, which is placed on a wooden table. The cat appears to be looking at the camera, possibly posing for a picture. The basket is filled with hay, providing a comfortable and cozy spot for the cat to rest. The overall scene is warm and inviting, showcasing the cat’s contentment”

Method

|Background Preservation CLIP Similarity| |
|---|---|
|PSNR ↑ LPIPS↓103 MSE↓104 SSIM↑102 20.58 171.51 305.47 74.82 12.47 308.77 793.64 57.43 27.36 64.21 121.54 84.02 32.63 33.88 17.95 93.57<br><br>|Whole ↑ Edited ↑ 23.43 21.58<br><br>22.46 21.24<br>23.24 20.58 23.78 20.97<br>|

InstructPix2Pix HQEdit MagicBrush Ours (instructive)

- Table 3: Image editing comparison using instructive text in PIE-Bench dataset. We only use instructive prompt in this comparison. Even though our model is not trained on the instructive setting, it outperforms the instructive training models by a large margins in background preservation, and achieves similar scores in CLIP similarity.

original 4 3 2 1

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

- Fig. 11: Given the input image, we visualize the inversion results with different inversion steps. While single-step inversion captures much of the semantic information from the input image, it tends to inadequately preserve identity and local detail, and produces noticeable artifacts in the background region. In contrast, multi-step inversion significantly enhances the quality of the reconstruction. Specifically, a four-step inversion approach can achieve near-perfect reconstruction of the input image.

Original Without Mask With Mask Attention Mask

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

ChickenSalmonShirtT-shirtShirtDownJacket→→→

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

- Fig. 12: In the absence of local masking, slight changes are observed in the background region after editing (indicating in red arrows). For instance, in the first example, the shape of sweet potatoes undergoes alteration, while in the second example, the shape of the background table change, and the identity of the face changes in the third example. Although attention mask is very rough, it significantly improves the background and identity preservation. It is worth to mention that manual mask will definitely help, since attention mask is too rough and sometime covers unedited region. For example, although we only modify the cloth region in the second example, the attention mask also covers the face region, which results in slightly identity changes. If users can provide a mask that only covers the cloth region, identity changes could be prevented.

Original With Attention Mask Attention Mask With Manual Mask Manual Mask

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

BlondBlackHorseZebra→→

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

- Fig. 13: While attention masks can generally identify the editing region, they often lack precision, leading to the inclusion of nearby regions. Consequently, nearby regions may undergo slight alterations post-editing, as indicated by the red arrows. Our approach enables users to upload a customized mask, which can be created manually or generated using an image segmentation model. We demonstrate that a manually created mask with a coarse outline can effectively minimize alterations in nearby pixels.

Original ReNoise Ours ReNoise Ours

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Young → Old Dog → Husky

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Shirt → Leather Jacket +Hat

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Black → White Kitten → Broccoli

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Monkey → Beaver Monkey → Teddy Bear

- Fig. 14: Compare to ReNoise method, our method generates realistic edited images without artifacts (dog → husky, young → old), maintains the face identity better (shirt → leather jacket), can perform large structure change (+hat). For the edits that ReNoise works well (we take the kitten and monkey images from ReNoise demo), our method generates comparable results.

Run → Sit + Blooming

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

- Fig. 15: Failure cases. Our method struggles with the difficult cases of large pose changes (e.g. run to sit, bud to blooming)

