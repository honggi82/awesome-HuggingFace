# arXiv:2411.18552v1[cs.CV]27Nov2024

## FAM Diffusion: Frequency and Attention Modulation for High-Resolution Image Generation with Stable Diffusion

Haosen Yang1,2* Adrian Bulat1 Isma Hadji1 Hai X. Pham1 Xiatian Zhu2 Georgios Tzimiropoulos1,3 Brais Martinez1

1Samsung AI Center, Cambridge, UK 2University of Surrey, UK 3Queen Mary University, UK

### Abstract

Diffusion models are proficient at generating high-quality images. They are however effective only when operating at the resolution used during training. Inference at a scaled resolution leads to repetitive patterns and structural distortions. Retraining at higher resolutions quickly becomes prohibitive. Thus, methods enabling pre-existing diffusion models to operate at flexible test-time resolutions are highly desirable. Previous works suffer from frequent artifacts and often introduce large latency overheads. We propose two simple modules that combine to solve these issues. We introduce a Frequency Modulation (FM) module that leverages the Fourier domain to improve the global structure consistency, and an Attention Modulation (AM) module which improves the consistency of local texture patterns, a problem largely ignored in prior works. Our method, coined FAM diffusion, can seamlessly integrate into any latent diffusion model and requires no additional training. Extensive qualitative results highlight the effectiveness of our method in addressing structural and local artifacts, while quantitative results show state-of-the-art performance. Also, our method avoids redundant inference tricks for improved consistency such as patch-based or progressive generation, leading to negligible latency overheads.

### 1. Introduction

Diffusion models [22] demonstrate impressive generative power across a range of applications [6, 18, 20, 23, 29, 30, 33]. While powerful, one known shortcoming of diffusion models is their inability to seamlessly scale to higher resolutions beyond the one used during training. It is known that directly generating images at resolutions beyond the training resolution results in severe object repetition and unrealistic local patterns [1, 3, 7]. This is illustrated in Figure 1(a).

*This work was conducted while Haosen Yang was an intern at Samsung AI Center, Cambridge, UK.

While retraining diffusion models on higher-resolution images is a straightforward solution, the computational demands quickly become prohibitive. This restricts applications requiring flexible or high-resolution image generation, e.g. 4K. Therefore, adapting pre-trained diffusion models to generate high-resolution images without additional training is a topic of high interest that we tackle in this work.

Prior efforts addressing this important problem can be largely categorized into two tracks. The first set of approaches, e.g. [3, 15], propose mechanisms that improve the global structure consistency by steering the high-resolution generation using the image generated at native (i.e. training) resolution. However, the effectiveness of such mechanisms is mixed, with trailing issues like poor detail quality, inconsistent local textures, and even persisting pattern repetitions as shown in Figure 1(b). Furthermore, these works typically operate on a patch-based basis, generating one patch at a time. Concretely, this means that these methods resort to redundant and overlapping forward passes, leading to large latency overheads. The second group of approaches, e.g. [7, 11, 34], eschews patch-based generation in favor of a one-pass approach by directly altering the model architecture. This leads to faster generation, but unfortunately, it comes at the cost of image quality, as shown in Fig. 1 (c).

To address the aforementioned limitations, we propose a straightforward yet effective approach that takes the best of both worlds. Our method follows the single pass generation strategy for improved latency but, like patch-based approaches, leverages the native resolution generation to steer the high-resolution one. Specifically, our method starts by generating an image at native resolution conditioned on the input text prompt. We then resort to a test-time diffusedenoise strategy [3, 8, 27], where the high-resolution denoising stage is guided by the native resolution diffusion process. However, instead of blindly steering the high-res image toward the low-res one as done elsewhere [3, 15], we propose a Frequency Modulation (FM) module. In particular, we leverage the Fourier domain to selectively condition low-frequency components during the high-resolution

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

(a) Direct Inference (b) DemoFusion (c) HiDiffusion (d) Ours

Figure 1. Comparisons of 3× (3072 × 3072) image generation based on SDXL [19].

image generation stage, while providing full control over high-frequency components to the denoising process.

While the FM module resolves artifacts related to global consistency, artifacts related to inconsistent local texture might still be present, i.e. finer texture generated on semantically related parts of the image might be inconsistent. To tackle this second issue, largely ignored in the literature, we propose an Attention Modulation (AM) mechanism that leverages attention maps from the denoising process at native resolution to condition the attention maps of the denoising process at high resolution. Since attention maps at native resolution encode which regions of the image are semantically related, they regularize the high-res denoising towards consistent finer texture generation. Our method, coined Frequency and Attention Modulated diffusion (FAM diffusion), combines our FM and AM modules to yield superior quality results, see Fig. 1 (d).

Our method seamlessly integrates with any latent diffusion model without additional training or architectural changes. We empirically show that our method significantly enhances the quality and efficiency of high-resolution image generation, establishing a new state-of-the-art.

### 2. Related Work

Diffusion models have shown impressive performance in generating creative and accurate representations given text prompts [10, 22]. While early work [22] was limited to generating relatively low-resolution images (i.e. 256 × 256), follow-up work showed that their performance can scale to higher resolutions, e.g. 512 × 512 with SD1.5 [22] and 1024×1024 with SDXL [19]. However, a major shortcoming with all these models is that generation remains limited by the resolution used at training time. Naively targeting higher train-time resolutions quickly results in prohibitive training costs and computational requirements, and the limited availability of high-resolution training data also restricts the diversity of image generation. Thus, adapting pre-trained diffusion models to generate high-resolution im-

ages without retraining has emerged as a topic of interest.

Early works [1, 14] proposed using overlapping patches at native resolution and blending the outputs to produce an image without seams. However, this leads to frequent repetitions and inconsistent global image structure. Therefore, subsequent works introduced various mechanisms to encourage global structural consistency. For instance, DemoFusion [3] proposed a patch-based generation process with mechanisms such as skip residuals and progressive upsampling, while AccDiffusion [15] used localized prompting to guide high-resolution generation and improve consistency with images generated at native resolutions. However, these methods still suffer from issues like local repetitions, and inconsistent global coherence. They also have significant latency overheads due to the running cost of multiple backward passes. To mitigate the high latencies, other works aim to generate high-resolution images in a single pass by modifying the architecture of the UNet. For example, ScaleCrafter [7] employs dilated convolutions to adjust the receptive field of convolutions in the denoising UNet. HiDiffusion [34] introduces an alternative UNet that dynamically adjusts the feature map size during the denoising process. While these approaches achieve faster generation, they often result in image distortions.

More closely related to ours are methods that have approached structural consistency from a frequency domain perspective. FouriScale [12] splits the image in Fourier domain, then proceeds to incorporate a low-pass filtering operation and impose structural consistency with an image generated at natire resolution. However, this splitting operation results in unrealistic images. HiPrompt [16] decomposes images into spatial frequency components conditioned on local and global prompts, but it often relies on redundant operations that lead to high latencies. ResMaster [25] leverages low-frequency information from the latent representation of the native image to provide desirable global semantics during the denoising process. However, it ignores the noise distribution differences between the current high-resolution denoising step and the native image in

latent space. In addition, it still relies on patch-based denoising, making it inefficient. In contrast to these methods, we propose a one-pass method that does not alter the model architecture. Importantly, our method introduces a complementary novel attention modulation mechanism, which targets local structure consistency; an issue overlooked by all existing works.

### 3. Method

In this work, we leverage pretrained latent diffusion models (LDMs), which have been extensively trained on largescale high-quality data. Our goal is to generate images at higher resolutions than during training, without any additional finetuning or model modification. Sec. 3.1 briefly reviews the diffusion notation and the test-time diffusedenoise strategy. In Sec. 3.2 we present our Frequency Modulated (FM) denoising approach, which is designed to improve global consistency. Finally, we introduce our Attention Modulation (AM) mechanism, which is designed to improve the consistency of the local texture and highfrequency detail, in Sec. 3.3. We provide an overview of our method in Figure 2.

#### 3.1. Preliminaries

Latent Diffusion Models (LDM) [22]: We operate in the realm of LDMs, which first convert image x0 to a latent representation z0 using an encoder such that z0 = E(x0), z0 ∈ Rc×h×w. During training, a Markovian diffusion process progressively adds noise to the input latent z0 according to a predefined schedule βt,t ∈ [1,T] by sampling sequentially from:

q(zt|zt−1) := N(zt| 1 − βtzt−1,βtI) (1)

Conversely, a trainable denoising process progressively recovers the original latent z0 using a noise estimator Zθ = (µθ,Σθ) parametrized by θ by sampling from:

pθ(zt−1|zt) := N (zt−1|µθ(zt,t),Σθ(zt,t)) (2)

During inference, an image is generated by denoising from random noise, zT ∼ N(0,I) ∈ Rc×h×w, through sequential calls to Zθ. The quality of the generated image improves with the number of steps to finally yield the latent representation zn0 ∈ Rc×h×w, where we introduce the superscript n to indicates generation at native resolution h×w (i.e. same as training resolution).

Inference-time diffuse-denoise: Our goal is to use the pretrained parametric denoiser Zθ, without further finetuning, to generate zm0 ∈ Rc×sh×sw at a higher resolution m, m = sh × sw, where s is the target resolution scaling factor. The naive approach is to directly start from random noise at the target resolution, zmT ∼ N(0,I) ∈ Rc×sh×sw.

However, this has been repeatedly shown to lead to suboptimal results, with frequent artifacts and object duplication [3, 7, 34]. This is illustrated in Fig. 3a.

Instead, prior works proposed a test time diffuse-denoise process [3, 8, 27]. The idea is to start from the output of the denoising process at native resolution, zn0 rather than noise, which is then upsampled to the target resolution m to obtain z˜m0 = U(zn0,s), where U denotes an upsampling function. Next, T forward diffusion steps progressively add noise to the latents z˜mt=1...T. Finally, the backward process denoises from z˜mT to yield the final output zm0 . Note that we use z˜ and z to refer to the latents generated during diffusion and denoising respectively.

While a standard denoising process as in Eq. 2 could be used, it often leads to inconsistent global structures, as shown in Fig. 3b. Instead, the denoising process from Eq. 2 is now defined as:

pθ zmt−1|ft(z˜mt ,zmt ) (3)

where ft(.) is tasked with steering the denoising process and improving the consistency between the high-res and low-res images. Previous work [3, 15] define ft(.) as a simple weighted linear combination of z˜mt and zmt and coin the mechanism skip residual. We show in Fig. 3c that this yields to suboptimal results. In contrast, we propose a Frequency Modulated approach to defining ft(.).

#### 3.2. Frequency-Modulated Denoising

The conditioning of the denoising steps through the skip residual has been shown to improve consistency between low and high-resolution images. We however observe that it lacks control over the information transferred. More specifically, the goal of the test-time diffuse-denoise process is to take the upsampled low-resolution image and to produce an output that 1) preserves the global structure, and 2) improves the texture and high-frequency details. The skip residual mechanism however steers the output towards the input indiscriminately, which serves the first objective but can negatively impact the latter. It would be desirable to instead harness the global structure information from the diffused latents of the forward process, while allowing the denoising process to handle the generation of details. To this end, we appeal to the frequency domain, where global structure and finer details are captured by low- and highfrequency, respectively [17, 28, 31], and re-define the function ft(.), which controls information transfer from the forward diffusion into the denoising process, in accordance.

Let K(t) be a high-pass filter for timestep t, the function ft(.) in Eq. 3 is defined as follows:

ft(z˜mt ,zmt ) =IDFT2D(K(t) ⊙ DFT2D (zmt ) + (1 − K(t)) ⊙ DFT2D (z˜mt )),

(4)

[Figure 5]

[Figure 6]

(a) Overview (b) FM

FFT

[Figure 7]

[Figure 8]

...

[Figure 9]

IFFT

[Figure 10]

[Figure 11]

FFT

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

…

(c) AM

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

FM …

A M

- Figure 2. Overview of the FAM diffusion. (a) We first generate an image at native resolution, followed by a test-time diffuse-denoise process. We incorporate our Frequency Modulation module and Attention Modulation during high-res denoising to control global structure and fine local texture, respectively. (b) Details of the Frequency Modulation, where we use the Fourier domain to selectively condition low-frequency components during high-res denoising while leaving high-frequency components fully controllable. (c) Details of Attention Modulation, where attention maps from the native image denoising are used to correct the high-res denoising.

[Figure 22]

[Figure 23]

(a) DI (b) DI* (c) SR (d) FM (e) FM-AM

[Figure 24]

[Figure 25]

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

- Figure 3. Ablation on the components of FAM diffusion. Direct Inference (DI) at high resolution from noise, Direct Inference from low-res latent (DI*), Skip Residual (SR) from DemoFusion [3], Frequency Modulation (FM), Attention Modulation (AM).

where ⊙ denotes the Hadamard product. Essentially, the high-frequency coefficients of the denoised latent zmt are combined with the low-frequency coefficients of the diffused latent z˜mt , modulated by the filter K(t). Eq. 4 can be further reformulated in the time domain as below:

ft(z˜mt ,zmt ) = zmt + κ(t) ⊛ z ˜mt − zmt , (5) where κ(t) = IDFT2D (1 − K(t)) ∈ Rsh×sw is a convolutional kernel, and ⊛ denotes the circular convolution operator. Eq. 5 shows that the frequency modulation adds a lowfrequency update to the denoised latent zmt directed towards the diffused latent z˜mt , subsequently preserving the global

structural information from the upsampled latent. Furthermore, the circular convolution κ(t) in Eq. 5 can be interpreted as an additional (non-learnable) convolutional layer of the UNet, effectively providing it with a global receptive field and helping generate consistent structure without modifying the UNet architecture [11, 34] or using dilated sampling [3]. The result of our FM approach is shown in Fig. 3d. In comparison, the skip residual approach of DemoFusion, shown in Fig. 3c, produces inconsistencies like a missing left nostril and unnaturally small eyes.

#### 3.3. Attention Modulation

While the FM module successfully maintains global structure and solves the issue of object duplication as shown in Fig. 3d, we note that local structures can be inconsistently generated due to the discrepancy between trainingtime native resolution and the target inference-time high resolutions. For example, the top image in Fig. 3d shows a distorted mouth compared to the one at native resolution. Similarly, in the bottom example, fur texture is incorrectly generated on the shirt collar. That is, the high-frequency detail generated on the shirt collar is semantically related to one generated on the fox’s face and not to the other parts of the shirt. We hypothesize this stems from incorrect attention maps during the high-res denoising stage. This motivates us to propose our Attention Modulation (AM) approach. We take inspiration from attention swapping, a recent method to combine information from two diffusion processes in a more localized manner [4, 5, 13], and extend the idea to transfer local structural information from the denoising process at native resolution to the one at target resolution.

In particular, the attention of an input tensor z is computed by first projecting it linearly into a triplet of query, keys, and values, (Q,K,V ), respectively, and the selfattention is computed as:

Att(z) = softmax

Q · KT √

d

V = M · V (6)

where d indicates the feature dimensionality, and we refer to M as the attention matrix.

In our case, we modify the self-attention at specific layers of the UNet of the high-resolution denoising process to incorporate information from the attention maps of the native resolution as:

M¯ m = (λ · U(Mn,s) + (1 − λ) · Mm) (7)

where Mn and Mm are the attention matrices at native and target resolution respectively, λ is a hyperparameter, and U is an s-times upsampling function. The new attention matrix M¯ m is then used instead of Mm during the high-res denoising process in Eq. 6.

Applying our AM module at all layers of the UNet can lead to suboptimal performance due to over-regularization. We apply it instead only for layers in up-blocks of the UNet, as they are known to preserve layout information better [13]. Furthermore, we experimented with AM at various stages and found the highest benefit to be at up block 0. Results shown in Fig. 3e demonstrate the benefit of the proposed AM module, particularly regarding better preservation of local structures such as the mouth and shirt collar, highlighted in yellow boxes.

### 4. Experiment

#### 4.1. Experimental setup

To demonstrate the effectiveness of our approach, we pair it with a well-performing diffusion model like SDXL [19]. For completeness, we also pair our approach with the recent HiDiffusion [34], which specifically changes the attention mechanism of SDXL with windowed attention to improve the model latency. SDXL is trained at 1024×1024 resolution, which we refer to as 1×. We experiment with three unseen higher resolutions such that the model generates 2×2, 3 × 3, and 4 × 4 times more pixels than the training setup. In the supplementary, we also include results with various aspect ratios, e.g. 2 × 4, and also experiment with different variants of Stable Diffusion (SD); namely, SD 1.5 [22], SD 2.1 [22], which generate at 512×512 and 768×768 pixels respectively.

Evaluation set. Following previous work [3, 7, 11, 15] we evaluate performance on a subset of the Laion-5B dataset [24]. Given the number of compared methods and significant computational demands associated with the task, we randomly sample 10K images from Laion-5b which we use as our real images set, and we sample 1K captions, which we use as text prompts for the models.

Evaluation metrics. Following prior work, we evaluate the quality and diversity of the generated images using Frechet Inception Distance (FID) [9] and Kernel Inception Distance (KID) [2], computed between the generated and real images. Since FID requires resizing images to 299 × 299, which negatively impacts the assessment, it is typical to adopt their patch-level variants [3, 11, 15, 34]. Specifically, we extract 10 random crops from each image before calculating FID and KID, referring to these metrics as FIDc and KIDc. To further evaluate the semantic similarity between image features and text prompts, we report the CLIP score [21]. To measure the efficiency of each method, we compute latencies on a single A40 GPU.

#### 4.2. Main Results

We select Demofusion [3], AccDiffusion [15], FouriScale [11], and HiDiffusion [34] as representative methods of the current state-of-the-art among high-resolution generation methods. As shown in Table 1, FAM diffusion achieves the best overall performance on FIDc, KIDc, and CLIP Score in all cases. In the case of FID and KID, FAM diffusion provides substantial gains for larger scale factors, while producing similar results to DemoFusion on lower scale factors. However, these metrics heavily downsample high-resolution images before computing the metrics and thus do not capture finer details in the evaluation results. This is a widely-known issue for these metrics, as explained in Sec. 4.1. Finally, we note that

|Method<br><br>|Scaling Factor|FID↓ KID↓ FIDc ↓ KIDc ↓ CLIP ↑ Latency(mins)|
|---|---|---|
|DemoFusion [3] AccDiffusion [15] FouriScale* [12] HiDiffusion [34] HiDiffusion [34] + FAM diffusion SDXL [19] SDXL [19] + FAM diffusion|2 × 2<br><br>|63.24 0.0084 36.75 0.0096 32.0 2.5 59.42 0.0068 37.23 0.0105 31.69 2.6 78.54 0.0136 40.80 0.0130 29.8 2.3 78.02 0.0136 51.41 0.0139 30.5 0.6 69.61 0.0140 34.26 0.0084 32.32 0.8 59.47 0.0067 50.54 0.0136 30.6 0.8 58.91 0.0072 33.96 0.0080 32.35 1|
|DemoFusion [3] AccDiffusion [15] FouriScale* [12] HiDiffusion [34] HiDiffusion [34] + FAM diffusion SDXL [19] SDXL [19] + FAM diffusion<br><br>|3 × 3<br><br>|68.82 0.0159 40.24 0.0122 32.0 8.6 73.47 0.0210 43.64 0.014 31.50 10 73.57 0.0309 65.01 0.0357 28.54 6.2<br><br>112.51 0.0325 68.84 0.021 28.43 1.5 76.28 0.0007 36.70 0.010 32.26 1.8 78.41 0.0136 69.40 0.0210 28.44 2.2<br><br>69.25 0.0007 36.40 0.010 32.25 2.5<br><br><br>|
|DemoFusion [3] AccDiffusion [15] FouriScale* [12] HiDiffusion [34] HiDiffusion [34] + FAM diffusion SDXL [19] SDXL [19] + FAM diffusion<br><br>|4 × 4<br><br>|65.89 0.0087 48.44 0.0157 30.45 19.6 73.97 0.0090 54.80 0.0187 30.15 20.5<br><br>105.24 0.0342 70.45 0.0223 27.86 14.7 129.91 0.0483 156.98 0.0877 24.32 2.8<br><br>59.05 0.0074 44.65 0.0134 32.31 3.1 160.10 0.0602 74.37 0.0242 26.70 5.4<br><br>58.91 0.0073 43.65 0.0130 32.33 6.1|

Table 1. System-level comparisons with SDXL. * indicates inference with FreeU [26]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

(a) Low Resolution Att.(b) High Resolution Att. (c) Att. Modulation

- Figure 4. Visualization of Attention Maps in the UNet: (a) LowResolution Attention map, (b) High-Resolution Attention map, (c) Attention Map when using the AM module

our method adds only small latency overheads compared to direct inference on the target resolution, e.g. 0.2, 0.3, and 0.7 min at 2×, 3× and 4× scale factors respectively when combined with SDXL. In comparison, DemoFusion adds 14.2 sec latency vs SDXL direct inference at 4× scale factor. When compared to the frequency-based method FouriScale [11], FAM diffusion also shows notable improvements in both quality and latency. For instance, under 4K resolution image generation, it achieves 43.65

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

(a) Direct Upsampling (b) BSRGAN (c) Ours

Figure 5. Qualitative comparison between Direct Upsampling, BSRGAN, and our method. The patches shown were cropped from a 4096 × 4096 resolution image. Zoom in for best view.

vs. 70.45 on FIDc and 32.31 vs. 26.67 on CLIP score, while also being faster than FouriScale. Additionally, we observed that FAM diffusion can be seamlessly integrated

[Figure 55]

[Figure 56]

[Figure 57]

Prompt: A cute fluffy rabbit pilot walking on a military aircraft carrier, unreal engine render, 8k, cinematic

Prompt: A cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal

Prompt: Close-up portrait of an elderly man, deep wrinkles, expressive eyes

(a) Native Resolution Image

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

2048x2048 3072x3072 4096x4096

[Figure 64]

[Figure 65]

[Figure 66]

- (b) DemoFusion

[Figure 67]

[Figure 68]

[Figure 69]

2048x2048 3072x3072 4096x4096

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

- (c) FouriScale*

[Figure 76]

[Figure 77]

[Figure 78]

2048x2048 3072x3072 4096x4096

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

- (d) HiDiffusion

[Figure 85]

[Figure 86]

[Figure 87]

2048x2048 3072x3072 4096x4096

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

- (e) Our Method

- Figure 6. Qualitative comparison with other methods based on SDXL. Best viewed when zoomed in. * indicates inference with FreeU [26]

[Figure 94]

[Figure 95]

- (a) Constant LF (b) Time-aware LF
- Figure 7. Comparison between Constant LF and Time-aware LF.

into single-pass methods, such as HiDiffusion [34], to enhance performance while maintaining fast image generation, achieving an effective latency-quality trade-off. These results quantitatively validate the effectiveness of our method in improving the quality of image generation.

In Figure 6, we present a comparison between DemoFusion, FouriScale, HiDiffusion, and FAM diffusion. We selected three complex textual prompts to highlight the imagegeneration capabilities of the model. For FouriScale, we used the default setting with FreeU [26]. Firstly, as mentioned above, DemoFusion tends to generate repetitive content and artifacts with unreasonable local structures due to its patch-based generation approach (see for example the two small cat heads generated on the top-right image). FouriScale [11] and HiDiffusion [34] produce visually unappealing structures and extensive areas of irregular textures, which significantly degrade the overall visual quality. Additionally, we compare our method with the superresolution approach BSRGAN [32], as shown in Figure 5. We observe that FAM diffusion effectively introduces or modifies high-frequency details that were not present in the original image, while preserving structural information, leading to more appealing and detailed images.

To further illustrate the generality of our approach, in the supplementary material we provide results of our approach in combination with SD1.5 and SD2.1.

#### 4.3. Ablation Study

In this section, we conduct ablation studies and use SDXL with the 2 × 2 scale factor setting.

Effectiveness of the components in the FAM diffusion We study the effect of the two components of FAM diffusion, Frequency-Modulated Denoising (FM) and Attention Modulation (AM). The results shown in Figure 3 indicate the following: (1) both direct inference from random noise, and direct inference from the diffused latent at native resolution generate outputs with structural distortions and repeated patterns. (2) while the Skip Residuals of DemoFusion helps maintain the global structure of the image, it still

produces artifacts and poor local patterns. (3) Compared to Skip Residuals, FM reduces undesirable local patterns by leveraging the low-frequency information of the image at native resolution, which provides better structural guidance. (4) Attention Modulation resolves inconsistencies between local patterns and global structure by utilizing the attention map from the native resolution, offering strong guidance of the semantic relationships among latent tokens. Overall, FM and AM address structural distortions and local pattern inconsistencies in high-resolution images effectively, highlighting the meaningful contributions of FAM diffusion.

Effectiveness of the time-aware formulation on the FM module We show here the effect of the time-varying formulation of FM, as illustrated in Figure 7a. Specifically, the FM module incorporates low-frequency information from the corresponding diffused latent at each step t. Instead, we can avoid this time-varying nature and utilize the upsampled latent as a single static reference. However, this approach results in images that appear noticeably blurrier and lose finer details associated with high-frequency information, highlighting the importance of the dynamic nature of the FM module throughout the denoising process.

Analysis of Attention Modulation To better understand the principles underlying the AM module, we visualize in Figure 4 the self-attention maps of a tokens from the mouth region (marked with a star) as the query and all tokens as the key and value. The resulting attention map computed using the low-resolution latent primarily encodes coarse information of the semantic relations among parts of the image, but lacks fine-grained contextual information across the entire face. Instead, the attention maps at high resolution are more detailed, but fail to capture semantic relatedness, e.g. the mouth areas are not highlighted. After applying AM, the attention map effectively integrates local-global relationships with enhanced fine-grained detail. This analysis provides visual insights into how AM repairs inconsistencies in local patterns, contributing to more coherent global structures.

### 5. Conclusion

We introduced FAM diffusion, a training-free diffusion model for high-resolution image generation. To address issues of object repetition and structural distortion, we propose a Frequency Modulated strategy. By leveraging the Fourier domain, this method enhances guidance for highresolution generation while avoiding latency overheads associated with multi-patch approaches. Additionally, we propose an effective Attention Modulation mechanism to address inconsistent local texture patterns, a challenge largely overlooked in previous works. Extensive quantitative and qualitative evaluations highlight the effectiveness of our method. We further show that, contrary to previous works, our method incurs in marginal latency overheads.

### References

- [1] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. MultiDiffusion: fusing diffusion paths for controlled image generation. In International Conference on Machine Learning, 2023. 1, 2
- [2] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. International Conference on Learning Representations, 2018. 5
- [3] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. DemoFusion: Democratising highresolution image generation with no $$$. In IEEE Conference on Computer Vision and Pattern Recognition, 2024. 1, 2, 3, 4, 5, 6, 13
- [4] Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, and Xin Eric Wang. Photoswap: Personalized subject swapping in images. Neural Information Processing Systems, 2023. 5
- [5] Jing Gu, Nanxuan Zhao, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, Yilin Wang, and Xin Eric Wang. SwapAnything: Enabling arbitrary object swapping in personalized image editing. European Conference on Computer Vision, 2024. 5
- [6] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 1

- [7] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. In International Conference on Learning Representations, 2024. 1, 2, 3, 5
- [8] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 1, 3
- [9] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Neural Information Processing Systems, 2017. 5
- [10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Neural Information Processing Systems, 2020. 2
- [11] Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. FouriScale: A frequency perspective on training-free high-resolution image synthesis. In European Conference on Computer Vision,

2024. 1, 4, 5, 6, 8

- [12] Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. FouriScale: A frequency perspective on training-free high-resolution image synthesis. arXiv preprint arXiv:2403.12963, 2024. 2, 6, 13
- [13] Jaeseok Jeong, Junho Kim, Yunjey Choi, Gayoung Lee, and Youngjung Uh. Visual style prompting with swapping selfattention. arXiv preprint arXiv:2402.12974, 2024. 5

- [14] Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. SyncDiffusion: Coherent montage via synchronized joint diffusions. In Neural Information Processing Systems,

2023. 2

- [15] Zhihang Lin, Mingbao Lin, Zhao Meng, and Rongrong Ji. AccDiffusion: An accurate method for higher-resolution image generation. In European Conference on Computer Vision, 2024. 1, 2, 3, 5, 6, 13
- [16] Xinyu Liu, Yingqing He, Lanqing Guo, Xiang Li, Bu Jin, Peng Li, Yan Li, Chi-Min Chan, Qifeng Chen, Wei Xue, Wenhan Luo, Qifeng Liu, and Yike Guo. HiPrompt: Tuningfree higher-resolution generation with hierarchical MLLM prompts. arXiv preprint arXiv:2409.02919, 2024. 2
- [17] David Marr and Ellen Hildreth. Theory of edge detection. Proceedings of the Royal Society of London. Series B. Biological Sciences, 207(1167):187–217, 1980. 3
- [18] Mehdi Noroozi, Isma Hadji, Brais Martinez, Adrian Bulat, and Georgios Tzimiropoulos. You only need one step: Fast super-resolution with stable diffusion via scale distillation. European Conference on Computer Vision, 2024. 1
- [19] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024. 2, 5, 6, 13
- [20] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3D using 2D diffusion. arXiv preprint arXiv:2209.14988, 2022. 1
- [21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 5
- [22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition, 2022. 1, 2, 3, 5
- [23] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation. In IEEE Conference on Computer Vision and Pattern Recognition, 2023. 1
- [24] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In Neural Information Processing Systems - Datasets and Benchmarks Track, 2022. 5
- [25] Shuwei Shi, Wenbo Li, Yuechen Zhang, Jingwen He, Biao Gong, and Yinqiang Zheng. ResMaster: Mastering highresolution image generation via structural and fine-grained guidance. arXiv preprint arXiv:2406.16476, 2024. 2
- [26] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. FreeU: Free lunch in diffusion U-Net. In IEEE Conference

- on Computer Vision and Pattern Recognition, 2024. 6, 7, 8, 13, 14
- [27] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 1, 3
- [28] BA Wandell. Foundations of vision, 1995. 3
- [29] Wenqing Wang, Haosen Yang, Josef Kittler, and Xiatian Zhu. Single image, any face: Generalisable 3D face generation. arXiv preprint arXiv:2409.16990, 2024. 1
- [30] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In IEEE International Conference on Computer Vision, 2023. 1
- [31] Kai Xu, Minghai Qin, Fei Sun, Yuhao Wang, Yen-Kuang Chen, and Fengbo Ren. Learning in the frequency domain. In IEEE Conference on Computer Vision and Pattern Recognition, 2020. 3
- [32] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In IEEE International Conference on Computer Vision, 2021. 8
- [33] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In IEEE International Conference on Computer Vision, 2023. 1, 12, 17
- [34] Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Yuhao Chen, Yao Tang, and Jiajun Liang. HiDiffusion: Unlocking higherresolution creativity and efficiency in pretrained diffusion models. In European Conference on Computer Vision, 2024. 1, 2, 3, 4, 5, 6, 8, 13

### A. Appendix

To complement the main content of the paper, we provide here additional details about the method in Sec. B as well as additional quantitative and qualitative results in Sec C.

### B. Additional technical details

#### B.1. Frequency Modulation details

Time-varying high-pass filter definition. In our method, we rely on frequency domain and use a high pass filter to steer the denoising process as described in equation (4). In the following, we provide the formal definition of the timevarying high pass filter, K(t), that we used.

The high-pass filters K(t) have time-varying cut-off frequencies, defined as follows:

t T

(8)

ρ(t) =

τh(t) = h · c · (1 − ρ(t)) (9) τw(t) = w · c · (1 − ρ(t)) (10)

where τh(t) and τw(t) are the horizontal and vertical cutoff frequencies at timestep t, respectively. Subsequently, the mask K(t), which is applied on the shifted frequency spectrum centered on (xc,yc), is defined as

 

ρ(t), if |x − xc| < τ

w(t) 2

& |y − yc| < τ

h(t)

(11)

K(t) =

2 , 1, otherwise



The cut-off frequency grows as the denoising process progresses, while the scaling factor of the low-frequency coefficients decreases. Our frequency modulation is designed such that the guidance from the denoised latent z˜t becomes more significant as t → 0. In our experiments, we set c = 0.5.

Derivation of the Frequency Modulation in timedomain. In the main paper, we mention that our frequency modulation introduced in Eq. (4) can be reformulated in time domain as Eq. (5) and discuss the corresponding benefits. Here, we provide a formal derivation to support the equivalence between the two formulations. For ease of presentation, we omit the timestep t and resolution m notations from operands.

Let z ∈ Rh×w be the 2D latent, and Z = DFT2D (z) ∈ Ch×w be the Fourier transform of z. Written in matrix form,

###### Z = (WrzWc), (12)

where Wr ∈ Ch×h,Wc ∈ Cw×w are the row- and columnwise Fourier transform matrices, respectively. Let K ∈ Rh×w be the high-pass filter defined in the previous section,

[Figure 96]

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

(a) Swapping (b) Modulation

Figure 8. Comparison of Attention Swapping and Modulation

our proposed mixing operation in the frequency domain is formulated as below:

Zˆ = K ⊙ DFT2D(z) + (1 − K) ⊙ DFT2D(z˜) = K ⊙ (WrzWc) + (1 − K) ⊙ (Wrz˜Wc)

= WrzWc + (1 − K) ⊙ (Wr(z˜ − z)Wc)

The inverse DFT of Zˆ, which is the outcome of Eq. 4, is formulated as: zˆ = IDFT2D(Zˆ)

= Wr−1 (WrzWc + (1 − K) ⊙ (Wr(z˜ − z)Wc))Wc−1

= Wr−1WrzWcWc−1

+ Wr−1 ((1 − K) ⊙ (Wr(z˜ − z)Wc))Wc−1

= z + Wr−1(1 − K)Wc−1 ⊛ Wr−1Wr(z˜ − z)WcWc−1

= z + k ⊛ (z˜ − z),

resulting in Eq. 5 in the main paper, where k = Wr−1(1 − K)Wc−1 = IDFT2D(1 − K) is a convolutional kernel and ⊛ denotes a circular convolution operator.

#### B.2. Attention Modulation analysis

As mentioned in Sec. 3.3, we take inspiration from recent literature using attention swapping to control local texture. However, rather than swapping attention, we mix the two attention paths instead. In Figure 8 we compare attention swapping versus our proposed attention modulation. These results clearly show the benefit of including the attention

from the high resolution path rather than directly swapping with the low res pass to avoid loss of information from the high res denoising path. We empirically set λ used in Eq (6) to 0.7.

### C. Additional experimental results

#### C.1. FAM diffusion with different SD backbones

In Table 1 we show that our method outperforms several baselines when combined with SDXL. In addition to those main results, we further combine our FAM diffusion method with various SD backbones. The quantitative re-

- sults in Table 2 demonstrate that our approach can seamless combine with different variants of SD and provides similarly large improvements in quality and image-text alignment across all experimental settings.

- C.2. FAM diffusion with different aspect ratios

Thus far, we have used our method to generate highresolution images by equally upscaling both the height and width. Here, we study the effect of using Fam diffusion targeting different aspect ratios. In particular, starting from the SDXL model, we use our approach targeting higher resolutions with different aspect ratios. The quantitative results in Table 3 and qualitative results shown in Figures 9 through 11, clearly highlight the versatility of our method that can seamlessly adapt to various settings without compromising quality.

- C.3. FAM diffusion with different conditioning terms

Fam Diffusion enables seamless integration with various LDM-based applications, such as ControlNet [33]. As shown in Figure 12, Fam Diffusion combined with ControlNet [33] achieves controllable high-resolution generation, with examples showcasing the use of images and canny edges as conditions.

|Method<br><br>|Resolution Scale Factor<br><br>|FIDr ↓ KIDr ↓ FIDc ↓ KIDc ↓ CLIP Score ↑|
|---|---|---|
|SD 1.5 SD 1.5 + FAM diffusion<br><br>|2 × 2<br><br>|75.36 0.0122 43.99 0.0103 30.35 65.07 0.0087 34.06 0.0082 30.92 86.62 0.0163 53.67 0.0137 29.66 64.77 0.0084 38.18 0.0091 31.13 59.47 0.0067 50.54 0.0136 30.6 58.91 0.0072 33.96 0.0080 32.35|
|SD 2.1 SD 2.1 + FAM diffusion| | |
|SDXL SDXL+ FAM diffusion| | |
|SD 1.5 SD 1.5 + FAM diffusion<br><br>|3 × 3<br><br>|106.50 0.0251 48.92 0.0133 28.89 38.19 0.0011 43.99 0.0082 30.44 137.05 0.0384 63.91 0.01719 27.81<br><br>64.8 0.0089 40.49 0.0114 31.13 78.41 0.0136 69.40 0.0210 28.44 69.25 0.0007 36.40 0.0100 32.25|
|SD 2.1 SD 2.1 + FAM diffusion| | |
|SDXL SDXL + FAM diffusion| | |
|SD 1.5 SD 1.5 + FAM diffusion<br><br>|4 × 4<br><br>|150.84 0.0474 55.97 0.0155 27.40 67.77 0.0086 40.21 0.0012 30.36 177.06 0.0645 69.43 0.019 26.36 66.32 0.0085 41.37 0.0018 31.10 160.10 0.0602 74.37 0.0242 26.70 58.91 0.0073 43.65 0.0130 32.33|
|SD 2.1 SD 2.1+ FAM diffusion| | |
|SDXL SDXL + FAM diffusion| | |

Table 2. Comparison of vanilla Stable Diffusion and our FAM diffusion.

|Method<br><br>|Scaling Factor|FID↓ KID↓ FIDc ↓ KIDc ↓ CLIP ↑|
|---|---|---|
|DemoFusion [3] AccDiffusion [15] FouriScale* [12] HiDiffusion [34] SDXL [19] SDXL [19] + FAM diffusion|2 × 4<br><br>|81.69 0.0112 54.48 0.0165 29.3<br><br>70.42 0.0119 55.73 0.0205 29.0<br>71.86 0.0302 63.28 0.0322 25.8 118.56 0.038 65.46 0.021 26.3<br><br><br>80.62 0.0236 67.46 0.0302 25.5 63.48 0.0090 41.44 0.0115 30.6|

Table 3. System-level comparisons with SDXL. * indicates inference with FreeU [26]

[Figure 107]

[Figure 108]

nature in the reflection of a mirror which is located in the middle of the caos, realistic, well done, detailed, 8k

A micro-tiny clay pot full of dirt with a beautiful daisie planted in it, shining in the autumn sun on a road in an abandoned city, fiction, wallpaper, character, cg artwork, art, flash photography

(a) Native Resolution Image

[Figure 109]

[Figure 110]

[Figure 111]

2048x4096 4096x2048

[Figure 112]

[Figure 113]

[Figure 114]

- (b) DemoFusion

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

2048x4096 4096x2048

[Figure 119]

[Figure 120]

- (c) FouriScale*

Figure 9. Qualitative comparison with other methods based on SDXL. Best viewed when zoomed in. * indicates inference with FreeU [26]. (Continued in Fig. 10).

[Figure 121]

[Figure 122]

[Figure 123]

2048x4096 4096x2048

[Figure 124]

[Figure 125]

[Figure 126]

- (a) HiDiffusion

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

2048x4096 4096x2048

[Figure 131]

[Figure 132]

- (b) Our Method

Figure 10. Qualitative comparison with other methods based on SDXL (continued from Fig. 9). Best viewed when zoomed in.

[Figure 133]

[Figure 134]

2304x1664

[Figure 135]

[Figure 136]

1664x2304

[Figure 137]

[Figure 138]

(a) FouriScale*

[Figure 139]

[Figure 140]

2304x1664

[Figure 141]

[Figure 142]

1664x2304

[Figure 143]

[Figure 144]

(b) HiDiffusion

[Figure 145]

[Figure 146]

2304x1664

[Figure 147]

[Figure 148]

1664x2304

[Figure 149]

[Figure 150]

(c) Our Method

- Figure 11. Qualitative comparison with other methods based on SDXL with arbitrary resolutions. DemoFusion is unable to handle arbitrary resolutions, therefore not included. Best viewed when zoomed in.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Condition Prompt: A dog, 4k Prompt: A rabbit, 4k Prompt: A Fox, 4k

(a) Image to Image

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Condition Prompt:the grassCorgi is runing on Prompt:the waterCorgi is runing on Prompt:the snowCorgi is runing on

(b) Canny Edges to Image

- Figure 12. Results of FAM Diffusion combining with ControlNet [33]. All images are generated at 2× (2048 × 2048).Best viewed when zoomed in.

