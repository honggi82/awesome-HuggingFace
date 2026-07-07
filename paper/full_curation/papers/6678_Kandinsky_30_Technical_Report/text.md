# arXiv:2312.03511v3[cs.CV]28Jun2024

## KANDINSKY 3.0 TECHNICAL REPORT

Vladimir Arkhipkin1 Andrei Filatov1 Viacheslav Vasilev1 Anastasia Maltseva1 Said Azizov1

Igor Pavlov1 Julia Agafonova1 Andrey Kuznetsov1,2,∗ Denis Dimitrov1,2,∗

1Sber AI 2AIRI

### ABSTRACT

We present Kandinsky 3.0, a large-scale text-to-image generation model based on latent diffusion, continuing the series of text-to-image Kandinsky models and reflecting our progress to achieve higher quality and realism of image generation. In this report we describe the architecture of the model, the data collection procedure, the training technique, and the production system for user interaction. We focus on the key components that, as we have identified as a result of a large number of experiments, had the most significant impact on improving the quality of our model compared to the others. We also describe extensions and applications of our model, including super resolution, inpainting, image editing, image-to-video generation, and a distilled version of Kandinsky 3.0 – Kandinsky 3.1, which does inference in 4 steps of the reverse process and 20 times faster without visual quality decrease. By side-by-side human preferences comparison, Kandinsky becomes better in text understanding and works better on specific domains. The code is available at https://github.com/ai-forever/Kandinsky-3.

### 1 Introduction

Recently, the quality of text-to-image generation models increased significantly, which became possible thanks to the invention of Diffusion Probabilistic Models [1, 2]. To date, the zoo of text-to-image generation models is extremely rich [3, 4, 5, 6, 7, 8, 9, 10]. Some of these systems provide users with opportunities for almost real-time inference, a high level of photorealism, an understanding of fantastic ideas and concepts not found in the real world, and many user-friendly web platforms and interfaces for a generation. Despite this, the task of text-to-image generating continues to pose serious challenges to researchers. The growing number of practical applications in commerce and design leads to a new unprecedented level of realism and alignment with complex textual descriptions.

This paper presents Kandinsky 3.0, a new text-to-image generation model based on latent diffusion [5]. Earlier, we introduced other models of the Kandinsky family [11], the architecture of which is based on a two-stage pipeline using Diffusion Mapping between elements of latent vector spaces of images and text with subsequent decoding. In the Kandinsky 3.0 model, we focused on improving the text understanding, the image quality and simplifying the architecture by providing a single-stage pipeline in which generation takes place directly using text embeddings without any additional priors. The whole pipeline contains 11.9 billion parameters, almost three times more than the largest of the previous models of the Kandinsky family. Also, we integrated Kandinsky 3.0 into our user-friendly website interaction system. We made our model completely public to promote the development of new technologies and openness in the scientific community.

This technical report is arranged as follows:

- • Firstly, we describe the user interaction system (Section 2);
- • Secondly, we describe the key components of the Kandinsky 3.0 model architecture (Section 3), dataset usage strategy and training techniques (Section 4);

∗Corresponding authors: Andrey Kuznetsov <kuznetsov@airi.net>, Denis Dimitrov <dimitrov@airi.net>. ∗∗The family of models is named after Wassily Kandinsky, the great Russian artist and an art theorist, the father of abstract art.

- • We also report additional features such as distillation (aka Kandinsky 3.1, see Section 5.1), super resolution (aka Kandinsky SuperRes, Section 5.2) and prompt beautification (Section 5.3), and various applications: inpainting (Section 6.1), image editing (Section 6.2), image-to-video (Section 6.3) and text-to-video (Section 6.4);
- • Finally, we describe human evaluation methodology (Section 7) and report the extensive results of side-by-side comparisons based on human preferences (see in appendix C. Side-by-side human evaluation results).

### 2 Interaction System

- As in the previous work [11], we incorporated the Kandinsky 3.0 model in several user interaction systems with open free access. Here we will describe their functionality and capabilities. Fusionbrain.ai1 – this is a web-editor that has the following functionality for text-to-image generation2:

- • The system can accept text prompts in Russian, English and other languages. It is also allowed to use emoji in the text description. The maximum size of a text is 1000 characters;
- • In the “Negative prompt” field, the user can specify which information (e.g., colors) the model should not use for generation;
- • Maximum resolution is 1024 × 1024;
- • Choosing the sides ratio: 1 : 1, 16 : 9, 9 : 16, 2 : 3 or 3 : 2;
- • Choosing of generation style to accelerate inference: digital image, pixel art, cartoon, portrait photo, studio photo, cyberpunk, 3D render, classicism, anime, oil painting, pencil drawing, Khokhloma painting style, and styles of famous artists such as Aivazovsky, Kandinsky, Malevich, and Picasso;
- • Zoom in/out;
- • Using an eraser to highlight areas that can be filled both with and without a new text description (inpainting technique);
- • Using a sliding window to expand the boundaries of the generated image and further generation with new borders (outpainting approach);
- • We also implemented a content filter developed by us to process incorrect requests.

This website also supports image-to-video generation with the following characteristics:

- • Resolution: 640 × 640, 480 × 854 and 854 × 480;
- • The user can set up to 4 scenes by describing each scene using a text prompt. Each scene lasts 4 seconds, including the transition to the next;
- • For each scene, it is possible to choose the direction of camera movement: up, down, right, left, counterclockwise or clockwise, zoom-in, zoom-out, and different types of flights around the object;
- • The average generation time ranges from 1.5 minutes for one scene to 6 minutes for four scenes;
- • The generated video can be downloaded in mp4 format.

We also created a Telegram-bot3 in which text-to-image generation is available.

### 3 Kandinsky 3.0 Architecture

#### 3.1 Overall pipeline

Kandinsky 3.0 is a latent diffusion model, the whole pipeline of which includes a text encoder for processing a prompt from the user, a U-Net [12] for predicting noise during the denoising (reverse) process, and a decoder for image reconstruction from the generated latent (Fig. 1). The text encoder and image decoder were frozen during the U-Net training. The whole model contains 11.9 billion parameters. Table 1 shows the number of parameters for the components

1https://fusionbrain.ai/en/ 2A detailed description of the API can be found at https://fusionbrain.ai/docs/en/doc/api-dokumentaciya/. 3https://t.me/kandinsky21_bot

[Figure 1]

- Figure 1: Kandinsky 3.0 overall pipeline architecture. It consists of a text encoder, a latent conditioned diffusion model, and an image decoder.

of the Kandinsky 3.0 model, the Kandinsky 2.2 [11] and SDXL [9] models. Below, we take a look at each component of our new model.

- Table 1: Comparison of the number of parameters (in billions) of components for the Kandinsky 2.2, SDXL, and Kandinsky 3.0 models.

Kandinsky 2.2 [11] SDXL [9] Kandinsky 3.0

Model type Latent Diffusion Latent Diffusion Latent Diffusion Total parameters 4.6B 3.33B 11.9B Text encoder 0.62B 0.8B 8.6B Diffusion Mapping [11] 1.0B – – Denoising U-Net 1.2B 2.5B 3.0B Image decoder 0.08B 0.08B 0.27B

#### 3.2 U-Net architecture

This section, we describe the considerations and ideas that led us to create our denoising U-Net architecture. Based on the success of large transformer-based models in vision problems when learning on large amounts of data [13, 14, 15, 16], and the fact that convolutional architectures occupied the central place in diffusion models so far, we had to decide which types of layers would contain the main part of the parameters of the new model: transformer or convolutional. Our field of view mainly included classification model architectures that showed high quality on the ImageNet benchmark dataset [17]. We conducted about half a thousand experiments with various architectural combinations and noted the following two key points:

- • Increasing the network depth by increasing the number of layers while reducing the total number of parameters in practice gives better results in training. A similar idea of residual blocks with bottlenecks was previously exploited in the ResNet-50 [18] and BigGAN-deep architecture [19];
- • At the first network layers, in high resolution level, we decided to process the latents using only convolutional blocks, while more compressed latent representations were fed to the transformer layers too. This ensures the global interaction of image elements.

We also reviewed the MaxViT architecture [20], which is almost entirely based on transformer blocks adapted to work with images by reducing the quadratic complexity of self-attention. In the classification task, this architecture shows the best results in comparison with the models mentioned above. Despite this, during experiments, we found out that this architecture does not show good results in the generation task.

Having thus explored all of the above architectures, we settled on the ResNet-50 block as the main block for our denoising U-Net. Thus, the residual blocks of our architecture at the input and output contain convolutional layers with a 1 × 1 kernel, which correspondingly reduce and increase the number of channels. We also expanded it with one more convolutional layer with a 3 × 3 kernel, as in the BigGAN-deep residual block architecture. Using bottlenecks in residual blocks made it possible to double the number of convolutional layers, while maintaining approximately the same number of parameters as without bottlenecks. At the same time, the depth of our new architecture has increased by 1.5 times compared to previous versions of the Kandinsky 2.X model.

- At the higher levels of the upscale and downsample parts, we placed only our implementation of convolutional residual BigGAN-deep blocks. At the same time, at lower resolutions, the architecture includes self-attention and cross-attention

[Figure 2]

- Figure 2: Kandinsky 3.0 U-Net architecture. The architecture is based on modified BigGAN-deep blocks (left and right – downsample and upsample versions), which allows us to increase the depth of the architecture due to the presence of bottlenecks. The attention layers are arranged at levels with a lower resolution than the original image.

layers. The complete scheme of our U-Net architecture, residual BigGAN-deep blocks, and cross-attention blocks is shown in Fig. 2.

Our version of the BigGAN-deep residual blocks differs from the one proposed in [19] by the following components:

- • We use Group Normalization [21] instead of Batch Normalization [22];
- • We use SiLU [23] instead of ReLU [24];
- • As skip connections, we implement them in the standard BigGAN residual block. For example, in the upsample part of the U-Net, we do not drop channels but perform upsampling and apply a convolution with 1 × 1 kernel.

#### 3.3 Text encoder

For the text encoder, we use the 8.6B encoder of the Flan-UL2 20B model [25], which is based on the pre-trained UL2 20B [26]. In addition to pretraining on a large corpus of texts, Flan-UL2 was also trained using supervised fine-tuning on many language tasks using Flan Prompting [27]. Our experiments showed that such fine-tuning also significantly improves image generation.

#### 3.4 Sber-MoVQGAN

To achieve a high-quality image reconstruction in complex domains such as text and faces, we developed the SberMoVQGAN autoencoder, which showed good results in Kandinsky 2.2 [11].

The Sber-MoVQGAN architecture is based on the VQGAN [28] architecture with the addition of spatially conditional normalization from the MoVQ [29]. Spatial conditional normalization is implemented similarly to the Adaptive Instance Normalization (AdaIN) layers used in the StyleGAN [30] architecture and is calculated by the formula:

Fi−1 − µ(Fi−1) σ(Fi−1)

Fi = ϕγ(zq)

+ ϕβ(zq) (1)

where zq is the latent consisting of codebook codes, Fi−1 is the intermediate feature map, µ and σ are the functions for calculating the mean and standard deviation of the activation, ϕγ and ϕβ are the trainable affine transformations, which convert zq into the scaling and bias values. Other important features of our implementation include the addition of EMA (exponential moving average) weights and a modified loss function from ViT-VQGAN [31] during the training stage.

We trained three versions of Sber-MoVQGAN – 67M, 102M, and 270M. The 67M version is the same size as the standard VQGAN. The 102M model uses twice the number of residual blocks compared to the 67M, and the 270M model operates with twice the original number of channels. Kandinsky 3.0 uses the 270M model as the image decoder.

We trained Sber-MoVQGAN on the LAION HighRes dataset [32], obtaining the SOTA results in image reconstruction. The comparison of our autoencoder with competitors and Sber-VQGAN4 are presented in Table 2. We released the weights and code for these models under an open-source license 5.

- Table 2: Sber-MoVQGAN comparison with competitors on ImageNet dataset. Our 270B version outperform all other models in terms of all quality metrics.

Model Latent size Num Z Train steps FID ↓ SSIM ↑ PSNR ↑ L1 ↓ ViT-VQGAN [31] 32x32 8192 500,000 1.28 – – – RQ-VAE [33] 8x8x16 16384 10 epochs 1.83 – – – Mo-VQGAN [29] 16x16x4 1024 40 epochs 1.12 0.673 22.42 – VQ CompVis [34] 32x32 16384 971,043 1.34 0.650 23.85 0.0533 KL CompVis [34] 32x32 – 246,803 0.968 0.692 25.11 0.0474 Sber-VQGAN 32x32 8192 1 epoch 1.44 0.682 24.31 0.0503 Sber-MoVQGAN 67M 32x32 1024 5,000,000 1.34 0.704 25.68 0.0451 Sber-MoVQGAN 67M 32x32 16384 2,000,000 0.965 0.725 26.45 0.0415 Sber-MoVQGAN 102M 32x32 16384 2,360,000 0.776 0.737 26.89 0.0398 Sber-MoVQGAN 270M 32x32 16384 1,330,000 0.686 0.741 27.04 0.0393

### 4 Data and Training Strategy

Data. During the training procedure, we used a large dataset of text-image pairs collected online. The training dataset consists of popular open-source datasets and our internal dataset of approximately 150 million text-image pairs. To improve data quality, we pass the data through several filters: the aesthetics quality of the image, the watermarks detection, the CLIP similarity of the image with the text [15], and the detection of duplicates with perceptual hash.

We discovered that the collected data from Common Crawl [35] contains almost no images related to Russian culture. To fix this, we collected and labeled a dataset of 200 thousand text-image pairs from Soviet and Russian cartoons, famous people, and places. This dataset helped improve the model’s quality and text alignment when generating Russian-related images.

We also divided all the data into two categories. We used the first at the initial stages of low-resolution pretraining and the second for mixed and high-resolution fine-tuning at the last stage. The first category includes open large text-image datasets such as LAION-5B [36] and COYO-700M [37] and “dirty” data that we collected from the Internet. The second category contains the same datasets but with stricter filters, especially for the image aesthetics quality.

Training. We divided the training process into several stages to use more data and train the model to generate images in a wide range of resolutions:

- 1. 256 × 256 resolution: 1.1 billions of text-image pairs, batch size = 20, 600 thousand steps, 104 NVIDIA Tesla A100;
- 2. 384 × 384 resolutions: 768 millions of text-image pairs, batch size = 10, 500 thousand steps, 104 NVIDIA Tesla A100;
- 3. 512 × 512 resolutions: 450 millions of text-image pairs, batch size = 10, 400 thousand steps, 104 NVIDIA Tesla A100;
- 4. 768 × 768 resolutions: 224 millions of text-image pairs, batch size = 4, 250 thousand steps, 416 NVIDIA Tesla A100;
- 5. Mixed resolution: 7682 ≤ W × H ≤ 10242, 280 millions of text-image pairs, batch size = 1, 350 thousand steps, 416 NVIDIA Tesla A100.

- 4https://github.com/ai-forever/tuned-vq-gan
- 5https://github.com/ai-forever/MoVQGAN

### 5 Additional Features

#### 5.1 Kandinsky 3.1 (aka Kandinsky 3.0 Distilled)

[Figure 3]

- Figure 3: Discriminator architecture for Kandinsky 3.1. Gray blocks inherit the weight of Kandinsky 3.0 and remain frozen during training.

A serious problem of diffusion models is the generation speed. To obtain a single image, it is usually necessary to go through several dozen steps (for example, 50) in the reverse process, that is, to pass data through U-Net many times with a batch size of 2 (with conditioning and without it) for classifier free guidance. To solve this problem, we used the Adversarial Diffusion Distillation approach [38], but with a number of significant modifications:

- 1. If pretrained pixel models were used as a discriminator, it would be necessary to decode the generated image using MoVQ and throw gradients through it, which would lead to huge memory costs. These costs do not allow training the model in 1024 × 1024 resolution. Therefore, as a discriminator, we used the frozen downsample part of the U-Net from Kandinsky 3.0 with trainable heads after each layer of resolution reduction (Fig. 3);
- 2. We added cross-attention on text embeddings from FLAN-UL2 to the discriminator heads instead of adding text CLIP-embeddings. This improved the text alignment using a distilled model;
- 3. We used Wasserstein Loss [39]. Unlike Hinge Loss, it is unsaturated, which avoids the problem of zeroing gradients at the first stages of training, when the discriminator is stronger than the generator;
- 4. We removed the regularization in the Distillation Loss, since according to our experiments it did not affect the quality of the model;
- 5. We found that the generator quickly becomes more powerful than the discriminator, which leads to learning instability. To solve this problem, we have significantly increased the learning rate of the discriminator. For the discriminator, we set the learning rate is equal to 1e − 3, and for the generator 1e − 5. To prevent divergence, we also used gradient penalty, as in the [38].

We trained a distilled model on a dataset with 100 thousand of highly-aesthetic image-text pairs, which we manually selected from the pretraining dataset. As a result, we speed up the Kandinsky 3.0 by almost 20 times, making it possible to generate an image in only 4 passes through U-Net. The acceleration is also due to the fact that there is no need to use classifier free guidance in the distilled version. However, like in [38], for a serious acceleration, we had to sacrifice the quality of the text comprehension, which can be seen from the human evaluation results of side-by-side

comparison (Section Human evaluation results for distillation). Generation examples by Kandinsky 3.1 can be found in the sections Distillation and Comparison to prior works.

[Figure 4]

- 5.2 Kandinsky SuperRes

Based on Kandinsky 3.0, we developed the Kandinsky SuperRes model to generate high-resolution 4k images. Here we describe the modifications that we made.

- 1. For Kandinsky SuperRes model we used diffusion in pixel space instead of latent diffusion. This eliminated the loss of quality when encoding and decoding images using an autoencoder. In addition, our experiments have shown that model based on pixel diffusion converges faster and better in this task than with latent diffusion.
- 2. We implemented Efficient U-Net, similar to the one described in the Imagen [6]. Compared to U-Net from Kandinsky 3.0, Efficient U-Net consumes less memory and also has better convergence. Instead of 3 residual blocks at each downscaling, Efficient U-Net uses more low-resolution blocks and fewer high-resolution blocks. The order of convolution and downsampling/upsampling operations is changed relative to the original U-Net. In addition, we removed conditioning for text prompt, because it does not contribute to 4K high resolution generation. As a result, Efficient U-Net of Kandinsky SuperRes contains 413M parameters.
- 3. During training, Efficient U-Net predict x0 (i.e., the original image) instead of the noise level at a given time t, which avoided problems with changing the color of the generated SR image.

The training was carried out in 2 stages. First, the model learned on the LAION dataset [36] on 32 A100 for 1570 thousand steps with a batch size of 2 for a resolution of 256 to 1024. Then we trained the model on the aesthetic high-resolution sets, which we used to train Kandinsky 3.0 for 1500 thousand steps. At the second stage of training, we included JPEG compression similar to the scheme in the Real-ESRGAN [40].

Figure 4: Comparison of Kandinsky SuperRes, Stable Diffusion and Real-ESRGAN models at 1024 resolution. Better to zoom image.

The Kandinsky SuperRes model can work with images of various resolutions, but the main goal is 4K highresolution generation. Since the Kandinsky SuperRes model was already trained for resolutions from 256 to 1024, higher resolution training was not possible due to GPU Tesla A100 memory overflow. For this reason, we used the MultiDiffusion algorithm [41] to generate 4k images. More specifically, first we divide the image into overlapping patches, and then at each diffusion step we remove the noise and average the pixel/latent values of the overlapping areas. Thus, having gone through all the stages of diffusion, we obtain a seamless image of any resolution. Kandinsky SuperRes model works in inference in 5 steps using DPMSolverMultistepScheduler6. The inference time for image generation in 4K resolution takes 13 seconds, and in 1K it takes 0.5 seconds.

6https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver

[Figure 5]

#### Figure 5: Example of Kandinsky SuperRes generation in 4K resolution. Better to zoom image.

- Table 3: Comparison of Kandinsky SuperRes, Real-ESRGAN [40] and Stable Diffusion [5] models. Kandinsky SuperRes outperforms other models in terms of all quality metrics for all test datasets.

Datasets Model FID↓ PSNR↑ SSIM↑ L1↓ Real-ESRGAN 9.96 24.48 0.73 0.0428

Wikidata 5k Stable Diffusion 3.04 25.05 0.67 0.0435 Kandinsky SuperRes 0.89 28.52 0.81 0.0257 Real-ESRGAN 73.26 23.12 0.72 0.0610

RealSR(V3)[42] Stable Diffusion 47.79 24.85 0.67 0.0493 Kandinsky SuperRes 47.37 25.05 0.75 0.0462 Real-ESRGAN 115.94 22.88 0.62 0.0561

Set14 [43] Stable Diffusion 76.32 23.60 0.57 0.0520 Kandinsky SuperRes 61.00 25.70 0.70 0.0390

The table 3 shows a comparison of Kandinsky SuperRes with the Real-ESRGAN 7 and Stable Diffusion x4 Upscaler 8 [5] models in terms of FID, SSIM, PSNR and L1 metrics on the our own dataset Wikidata 5K and RealSR(V3) 9 [42] and Set14 10 [43] datasets. Our own dataset Wikidata 5K contains 5 thousand images collected from Wikipedia in 1K resolution. RealSR(V3) contains 100 test images in 1K and 2K resolutions. Set14 contains 14 low resolution images with JPEG artifacts. As a result, the Kandinsky SuperRes model shows the best results in terms of all quality metrics for all datasets. Figure 4 shows examples of generation of Kandinsky SuperRes, Stable Diffusion and Real-ESRGAN models at a resolution of 1024. Figure 5 shows examples of generation of Kandinsky SuperRes in 4K resolution. We released the weights and code for Kandinsky SuperRes under an open-source license 11.

#### 5.3 Prompt beautification

Many diffusion text-to-image models have some kind of inconvenience due to the fact that the visual quality and detail of the generation depends on the degree of detail of the text prompt. Sometimes, in practice, user has to use long, redundant prompts to generate desirable images. Normal user prompts are not such. To solve this problem, we have built a function into the generation pipeline to add details to the user’s prompt using LLM. An instruction is sent to the

- 7https://github.com/ai-forever/Real-ESRGAN/
- 8https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
- 9https://github.com/csjcai/RealSR/tree/master
- 10https://paperswithcode.com/dataset/set14
- 11https://github.com/ai-forever/KandiSuperRes/

input of the language model with a request to improve the prompt, and the model’s response is sent as the input into Kandinsky 3.0 model. As an LLM, we used Neural-Chat-7b-v3-1 [44] (based on Mistral 7B [45]) with the following system instruction:

### System:\nYou are a prompt engineer. Your mission is to expand prompts written by user. You should provide the best prompt for text to image generation in English. \n### User:\n{prompt}\n### Assistant:\n

Here {prompt} is the user’s text prompt. Examples of generation for the same prompt with and without beautification are presented in the section Prompt beautification. We conducted a side-by-side human evaluation comparison of the generation quality using the prompt beautification and without it. We conducted comparison for both Kandinsky 3.0 and Kandinsky 3.1 to assess how strongly the language model affects the generated images. Each generation was evaluated by visual quality and by the text-image alignment. The results of the comparison can be seen in the section Human evaluation results for prompt beautification. In general, human preferences are definitely more inclined towards generations with prompt beautification. The only exception is the correspondence of the generated image to the text for the Kandinsky 3.1 model. This is due to the fact that the distilled model generally understands the text worse.

### 6 Applications

#### 6.1 Inpainting and Outpainting

Implementation of the inpainting model is the same as GLIDE [46]: we initialize our model from base Kandinsky model weights. Then, we modify the input convolution layer of U-Net (Section 3.2) so that the input can additionally accept the image latent and mask. Thus, U-Net takes as many as 9 channels as input: 4 for the original latent, 4 for the image latent, and an additional channel for the mask. We zeroed the additional weights, so training starts with the base model.

For training, we generate random masks of the following forms: rectangular, circles, strokes, and arbitrary form. For every image sample, we use up to 3 masks, and for every image, we use unique masks. We use the same dataset as for the training base model with generated masks. We train our model using Lion [47] with lr=1e-5 and apply linear warmup for the first 10k steps. We train our model for 250 thousand steps. The inpainting results are in Fig. 6. The outpainting results can be found in the Outpainting appendix section. Additionally, we finetune our model using object detection datasets and LLaVA captions for 50k steps.

#### 6.2 Image editing

Kandinsky 3.0 provides image generation not only using a text prompt, but also using an image as a visual prompt. This allows one to edit an existing image, change its style and add new objects to it. To do this, we extended an IP Adapter-based approach [48]. To implement our own IP-Adapter based on our basic generation model, we used attention adapters. We used the ViT-L-14 as the CLIP model [15]. We get CLIP-embeddings of the size batch size × 768, which are then transformed by a linear layer into tensors of the size batch size × 4 × 4096. By adding a couple of new layers for key and value images in the cross-attention mechanism, we sum up the output of a text cross-attention with the output of cross-attention for images. We trained the IP-Adapter on the COYO 700m dataset [37] with a batch size 288 during 800 thousand iterations. As a result, Kandinsky 3.0 supports image variation, image-image mixing, and image-text mixing. For generation examples see section Image Editing with IP-Adapter.

In addition, we found that the IP Adapter-based approach does not preserve the shape of objects in the image, so we decided to train ControlNet [49] in addition to our generation model to consistently change the appearance of the image, preserving more information compared to the original one. We used HED detector [50] as a model to obtain the edges in the image fed to the ControlNet input. The training lasted 5 thousand iterations on the COYO 700m dataset [37] on 8 Tesla A 100 GPU with a batch size 512.

#### 6.3 Image-to-Video Generation

Image-to-video generation involves a series of iterative steps, encompassing four stages as illustrated in Fig. 7. Our animation pipeline is based on the Deforum technique [51]. It consists of a series of transformations applied to the scene:

- 1. Conversion of the image into a three-dimensional representation using a depth map;
- 2. Application of spatial transformations to the resulting scene to induce an animated effect;

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

- Figure 6: Examples of inpainting.. We use the following prompts for inpainting: “ship sailing in the water”, “rocket” and “robot sitting on a bench”, respectively. We generated inpainted images using FusionBrain.ai platform.

- 3. Projection of the 2.5D scene back onto a 2D image;
- 4. Elimination of transformation defects and updating of semantics through image-to-image conversion techniques.

The scene generation process involves defining depth values along the z-axis within the interval [(znear,zfar)] in the coordinate system. Depth estimation utilizes either MiDaS [52] or AdaBins [53]. The camera is characterized by the coordinates (x,y,z) in the three-dimensional space, and the direction of view, which is set by three angles (α,β,γ). Thus, we set the trajectory of the camera motion using the dependencies x = x(t), y = y(t), z = z(t), α = α(t), β = β(t), and γ = γ(t). The camera’s first-person motion trajectory includes perspective projection operations with the camera initially fixed at the origin and the scene at a distance of znear. Then, we apply transformations by rotating points around axes passing through the scene’s center and translating to this center. Acknowledging the limitations of a single-image-derived depth map, addressing distortions resulting from minor camera orientation deviations is crucial. Two essential considerations follow: adjusting scene position through infinitesimal transformations and employing the image-to-image approach after each transformation. The image-to-image technique facilitates the realization of seamless and semantically accurate transitions between frames, enhancing the aesthetic appeal of this approach. The examples of image-to-video generations are presented in the Image-to-Video appendix section.

#### 6.4 Text-to-Video

Based on the Kandinsky 3.0 model, we also created the text-to-video generation pipeline Kandinsky Video [54], consisting, in addition to a text encoder and an image decoder, of two models – for keyframes generation and interpolation between them. Both models use the pretrained weights of Kandinsky 3.0 as the backbone. We have

[Figure 15]

- Figure 7: Illustration of the Image-to-Video Generation process utilizing the image-to-image technique. The input image undergoes a right shift transformation. The resulting image enters the image-to-image process to eliminate transformation artifacts and update the semantic content in alignment with the provided prompt.

also implemented the publicly available text-to-video generation interface12 in the Fusionbrain.ai website, which we mentioned above (Section 2). Please refer to the main paper for additional details regarding the text-to-video model.

### 7 Human evaluation

Some examples of text-to-image generations can be found in the Text-to-Image appendix section. To compare Kandinsky 3.0 and Kandinsky 3.1 with other well-known models, we have compiled a balanced set of 2.1K prompts in 21 categories. Using this set, we first performed several side-by-side (SBS) comparisons between different versions of the Kandinsky 3.0 and then selected the best version. We conducted three rounds of SBS comparisons involving 28 people to do this. Next, we conducted side-by-side comparisons of Kandinsky 3.0 with the Kandinsky 2.2 [11], SDXL [9] and DALL-E 3 [10] models. Each study involved 12 people who voted an average of 30,000 times in each SBS. For this purpose, we developed chatbot that showed one of 2.1K pairs of images. Each person chose the best image according to two criteria:

- 1. Alignment between image content and prompt (text comprehension);
- 2. Visual quality of the image.

We compared the visual quality and text comprehension in total for all categories, and each category separately. The visual examples of Comparison to prior works and Human evaluation results for text-to-image generation are presented in the appendix.

### 8 Limitations

Even though the current system can generate high-quality realistic images and successfully cope with diverse and complex language concepts, there are ways for further improvement. Among them is the improvement of semantic coherence between the input text and the generated image due to more efficient use of the text encoder potential. Challenges still remain high-fidelity text generation and photorealistic faces, and physics-controlled scene (lighting, positioning, focus and exposure, etc.).

12https://fusionbrain.ai/en/t2v/

### 9 Border Impacts and Ethical Considerations

Generative models are an effective tool for creativity and media content creation. They are also of great importance for the development of artificial intelligence science. We made the code and the trained weights of the model available to promote openness in the scientific community and the development of technologies that improve people’s lives. We have provided free access to the user-friendly interface for everyone on the Internet.

At the same time, we know that generative models can be leveraged for blackmail, fraud, disinformation, creating fakes, inciting hatred and enmity between people, for unscrupulous political, financial, and other purposes. We warn against using our model in this way and strongly disagree with such malicious applications. We consider it necessary to note that the result of using the generations of our model for unfair purposes is entirely the user’s responsibility.

Despite this, we made many efforts to make sure that the generated images didn’t contain malicious, offensive, or insulting content. To this end, we cleaned the training dataset from samples marked as harmful/offensive/abusive and removed offensive textual prompts. While obvious queries, according to our tests, rarely generate abusive content, there is technically no guarantee that some carefully designed prompts may not yield undesirable content. Therefore, depending on the application, we recommend using additional classifiers to filter out unwanted content and use image/representation transformation methods adapted to a given application.

### 10 Conclusion

In this report we highlighted the most significant advantages of our new text-to-image generative model – Kandinsky 3.0. Improving the text encoder size and extending the main diffusion U-Net we achieved higher human evaluation scores in comparison with Kandinsky 2.2. It should be mentioned that both measured quality indicators – text understanding and visual quality improved. Comparing with SDXL we achieved significantly higher scores for both indicators.

We described the acceleration of our model using distillation (Kandinsky 3.1 model). This allowed us to increase the inference speed by 20 times and reduce the number of steps in the reverse diffusion process to 4 without decrease in quality. We also show that the Kandinsky 3.0 model can be successfully used in various applications – inpainting/outpainting, image editing, image-to-video, and text-to-video.

### References

- [1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France, 07–09 Jul 2015. PMLR.
- [2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [3] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 16784–16804. PMLR, 2022.
- [4] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [5] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [6] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
- [7] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [8] Midjourney. https://www.midjourney.com/.
- [9] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023.

- [10] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwa, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions, 2023.
- [11] Anton Razzhigaev, Arseniy Shakhmatov, Anastasia Maltseva, Vladimir Arkhipkin, Igor Pavlov, Ilya Ryabov, Angelina Kuts, Alexander Panchenko, Andrey Kuznetsov, and Denis Dimitrov. Kandinsky: an improved text-toimage synthesis with image prior and latent diffusion, 2023.
- [12] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.
- [14] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10012–10022, October 2021.
- [15] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763, 18–24 Jul 2021.
- [16] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8821–8831. PMLR, 18–24 Jul 2021.
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.
- [18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.
- [19] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis, 2019.
- [20] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer. ECCV, 2022.
- [21] Yuxin Wu and Kaiming He. Group normalization. arXiv:1803.08494, 2018.
- [22] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, ICML’15, page 448–456. JMLR.org, 2015.
- [23] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning, 2017.
- [24] Abien Fred Agarap. Deep learning using rectified linear units (relu), 2019.
- [25] Yi Tay. A new open source flan 20b with ul2. https://www.yitay.net/blog/flan-ul2-20b, 2023.
- [26] Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier García, Jason Wei, Xuezhi Wang, Hyung Won Chung, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Denny Zhou, Neil Houlsby, and Donald Metzler. Ul2: Unifying language learning paradigms. In International Conference on Learning Representations, 2022.
- [27] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022.
- [28] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis, 2021.
- [29] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for high-fidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022.

- [30] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. CoRR, abs/1812.04948, 2018.
- [31] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan, 2022.
- [32] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: an open large-scale dataset for training next generation image-text models. In NeurIPS, 2022.
- [33] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization, 2022.
- [34] Andreas Blattmann, Robin Rombach, Kaan Oktay, and Björn Ommer. Retrieval-augmented diffusion models, 2022.
- [35] Common crawl. https://commoncrawl.org/terms-of-use.
- [36] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022.
- [37] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.
- [38] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation, 2023.
- [39] Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 214–223. PMLR, 06–11 Aug 2017.
- [40] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In International Conference on Computer Vision Workshops (ICCVW), 2021.
- [41] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation, 2023.
- [42] Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image superresolution: A new benchmark and a new model. In Proceedings of the IEEE International Conference on Computer Vision, 2019.
- [43] Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed selfexemplars. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5197–5206, 2015.
- [44] Kaokao Lv, Wenxin Zhang, and Haihao Shen. Supervised fine-tuning and direct preference optimization on intel gaudi2. https://medium.com/intel-analytics-software/ the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3, 2023.
- [45] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023.
- [46] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [47] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, et al. Symbolic discovery of optimization algorithms. arXiv preprint arXiv:2302.06675, 2023.
- [48] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models, 2023.
- [49] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023.
- [50] Saining "Xie and Zhuowen" Tu. Holistically-nested edge detection. In Proceedings of IEEE International Conference on Computer Vision, 2015.

- [51] Deforum. https://deforum.art/.
- [52] René Ranftl, Jonas Gehrig, Martin Humenberger, and Vittorio Ferrari. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 5626–5635, 2019.
- [53] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. arXiv:2011.14141 [cs.CV], 2020.
- [54] Vladimir Arkhipkin, Zein Shaheen, Viacheslav Vasilev, Elizaveta Dakhova, Andrey Kuznetsov, and Denis Dimitrov. Fusionframes: Efficient architectural aspects for text-to-video generation pipeline, 2023.

### A. Acknowledgements

The authors would also like to extend their deepest gratitude to the following list of teams and persons, who made a significant contribution to Kandinsky 3.0 research and development:

- • Sber AI Research team: Nikolay Gerasimenko, Elizaveta Dakhova, Sofia Kirillova, Mikhail Shoytov, Zein Shaheen, Anastasia Yaschenko;
- • Anton Razzhigaev from FusionBrain research team at AIRI;
- • Konstantin Kulikov and his production team at Sber AI;
- • Sergey Markov and his research teams at SberDevices;
- • Polina Voloshina labelling team;
- • ABC Elementary labelling team;
- • TagMe labelling team;
- • Tatyana Nikulina, Angelina Kuts, Anton Bukashkin and prompt engineering team;
- • Arseniy Shakhmatov, Anastasia Lysenko, Sergey Nesteruk, Ilya Ryabov and Mikhail Martynov (ex-Sber AI).

Thanks to all of you for your valuable help, advice, and constructive criticism.

### B. Generation Examples Text-to-Image

[Figure 16]

#### Distillation

[Figure 17]

Figure 8: Prompt: Cute red lion cub in a bath with foam.

[Figure 18]

- Figure 9: Prompt: Close-up of a little boy in a tuxedo with a bouquet of white daisies, watercolor painting.

[Figure 19]

##### Figure 10: Prompt: Huge hot air balloon soars in the fabulous steppes and mountain ranges, freshness, high image quality.

[Figure 20]

##### Figure 11: Prompt: Cute little panda stands in a kimono, a very kind cute look, cute eyes, kindness.

#### Prompt beautification

[Figure 21]

##### Figure 12: Prompt: Close-up photo of a beautiful oriental woman, elegant hijab-adorned with hints of modern vintage style. Without/With LLM.

[Figure 22]

Figure 13: Prompt: A hut on chicken legs. Without/With LLM.

[Figure 23]

##### Figure 14: Prompt: Lego figure at the waterfall. Without/With LLM.

#### Outpainting

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Figure 15: Examples of outpainting.

#### Image Editing with IP-Adapter

[Figure 30]

- Figure 16

[Figure 31]

- Figure 17

[Figure 32]

- Figure 18

[Figure 33]

- Figure 19

#### Image-to-Video

[Figure 34]

Figure 20: Animations generated by Image-to-Video pipeline.

#### Comparison to prior works

[Figure 35]

Figure 21: Prompt: Tomatoes on a table, against the backdrop of nature, a still life painting depicted in a hyper realistic style.

[Figure 36]

- Figure 22: Prompt: Photo of Russian apartment buildings with many windows at night, lights on inside each window, the building is lighted from behind by warm yellow and white

lights, it’s a wide shot showing an entire multistorey block of flats, there are more houses in front of us, high resolution, hyper realistic.

[Figure 37]

##### Figure 23: Prompt: Funny cute wet kitten sitting in a basin with soap foam, soap bubbles around, photography.

[Figure 38]

##### Figure 24: Prompt: Maximally realistic portrait of a jolly old gray-haired Negro Rastaman with wrinkles around his eyes and a crooked nose in motley clothes.

### C. Side-by-side human evaluation results

#### Human evaluation results for text-to-image generation

[Figure 39]

- Figure 25: Overall results of side-by-side human comparison between Kandinsky 3.0 and Kandinsky 2.2.

[Figure 40]

- Figure 26: Results of side-by-side human comparison between Kandinsky 3.0 and Kandinsky 2.2 for text comprehension.

[Figure 41]

Figure 27: Results of side-by-side human comparison between Kandinsky 3.0 and Kandinsky 2.2 for visual quality.

[Figure 42]

Figure 28: Overall results of side-by-side human comparison between Kandinsky 3.0 and DALL-E 3.

[Figure 43]

###### Figure 29: Results of side-by-side human comparison between Kandinsky 3.0 and DALL-E 3 for text comprehension.

[Figure 44]

###### Figure 30: Results of side-by-side human comparison between Kandinsky 3.0 and DALL-E 3 for visual quality.

[Figure 45]

Figure 31: Overall results of side-by-side human comparison between Kandinsky 3.0 and SDXL.

[Figure 46]

Figure 32: Results of side-by-side human comparison between Kandinsky 3.0 and SDXL for text comprehension.

[Figure 47]

###### Figure 33: Results of side-by-side human comparison between Kandinsky 3.0 and SDXL for visual quality.

Human evaluation results for distillation

[Figure 48]

###### Figure 34: Overall results of side-by-side human comparison between Kandinsky 3.0 and Kandinsky 3.0 Distilled (Kandinsky 3.1).

[Figure 49]

###### Figure 35: Results of side-by-side human comparison between Kandinsky 3.0 and Kandinsky 3.0 Distilled (Kandinsky 3.1) for text comprehension.

[Figure 50]

###### Figure 36: Results of side-by-side human comparison between Kandinsky 3.0 and Kandinsky 3.0 Distilled (Kandinsky 3.1) for visual quality.

#### Human evaluation results for prompt beautification

[Figure 51]

- Figure 37: Overall results of side-by-side human comparison for Kandinsky 3.0 with and without prompt beautification.

[Figure 52]

- Figure 38: Overall results of side-by-side human comparison for Kandinsky 3.0 with and without prompt beautification for text comprehension.

[Figure 53]

###### Figure 39: Overall results of side-by-side human comparison for Kandinsky 3.0 with and without prompt beautification for visual quality.

[Figure 54]

###### Figure 40: Overall results of side-by-side human comparison for Kandinsky 3.0 Distilled (Kandinsky 3.1) with and without prompt beautification.

[Figure 55]

###### Figure 41: Overall results of side-by-side human comparison for Kandinsky 3.0 Distilled (Kandinsky 3.1) with and without prompt beautification for text comprehension.

[Figure 56]

###### Figure 42: Overall results of side-by-side human comparison for Kandinsky 3.0 Distilled (Kandinsky 3.1) with and without prompt beautification for visual quality.

