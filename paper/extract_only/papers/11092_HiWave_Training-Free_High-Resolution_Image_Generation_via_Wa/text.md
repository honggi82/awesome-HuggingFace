arXiv:2506.20452v1[cs.CV]25Jun2025

HiWave: Training-Free High-Resolution Image Generation via Wavelet-Based Diffusion Sampling

TOBIAS VONTOBEL, ETH Zurich, Switzerland SEYEDMORTEZA SADAT, ETH Zurich, Switzerland FARNOOD SALEHI, Disney Research|Studios, Switzerland ROMANN WEBER, Disney Research|Studios, Switzerland

[Figure 1]

Fig. 1. We propose HiWave, a novel training-free approach for high-resolution image generation using pretrained diffusion models. While standard Stable Diffusion XL (SDXL) can produce globally coherent images, it lacks fine details when upscaled to 4096×4096 resolution (left column). Existing training-free methods (e.g., Pixelsmith [Tragakis et al. 2024]) enhance details in SDXL outputs but often introduce duplicated objects and visual artifacts (middle column). In contrast, HiWave leverages a patch-wise DDIM inversion strategy combined with a wavelet-based detail enhancer module to produce high-quality images with rich details and minimal duplication artifacts. The second and third rows show 10× and 5× magnified views of the red and green boxed regions, respectively.

Diffusion models have emerged as the leading approach for image synthesis, demonstrating exceptional photorealism and diversity. However, training diffusion models at high resolutions remains computationally prohibitive, and existing zero-shot generation techniques for synthesizing images beyond training resolutions often produce artifacts, including object duplication and spatial incoherence. In this paper, we introduce HiWave, a training-free, zero-shot approach that substantially enhances visual fidelity and structural coherence in ultra-high-resolution image synthesis using pretrained diffusion models. Our method employs a two-stage pipeline: generating a base image from the pretrained model followed by a patch-wise DDIM inversion step and a novel wavelet-based detail enhancer module. Specifically, we first utilize inversion methods to derive initial noise vectors that preserve

Authors’ Contact Information: Tobias Vontobel, , ETH Zurich, Zurich, Switzerland; Seyedmorteza Sadat, , ETH Zurich, Zurich, Switzerland; Farnood Salehi, , Disney Research|Studios, Zurich, Switzerland; Romann Weber, , Disney Research|Studios, Zurich, Switzerland.

global coherence from the base image. Subsequently, during sampling, our wavelet-domain detail enhancer retains low-frequency components from the base image to ensure structural consistency, while selectively guiding high-frequency components to enrich fine details and textures. Extensive evaluations using Stable Diffusion XL demonstrate that HiWave effectively mitigates common visual artifacts seen in prior methods, achieving superior perceptual quality. A user study confirmed HiWave’s performance, where it was preferred over the state-of-the-art alternative in more than 80% of comparisons, highlighting its effectiveness for high-quality, ultra-high-resolution image synthesis without requiring retraining or architectural modifications.

## CCS Concepts: • Computing methodologies → Computer vision.

Additional Key Words and Phrases: Image synthesis, diffusion models, highresolution generation, discrete wavelet transform, frequency-domain processing, training-free methods, patch-based diffusion

1 Introduction

Since the introduction of diffusion models, generative image synthesis has reached unprecedented levels of photorealism and creative control. Recent models such as Stable Diffusion [Esser et al. 2024; Podell et al. 2023a] can generate stunning images at resolutions up to 1024×1024 pixels. Despite these improvements in image quality, producing outputs beyond 1024×1024 remains technically challenging due to the substantial computational demands associated with training at higher resolutions.

Diffusion models typically rely on large-scale networks with high parameter counts to achieve optimal image quality and prompt alignment [Esser et al. 2024]. As input resolution increases, the computational cost of training these models becomes prohibitive—especially for attention-based architectures [Esser et al. 2024; Peebles and Xie 2023], where complexity scales quadratically with spatial dimensions. Moreover, most datasets used to train large-scale diffusion models lack high-resolution content beyond 1024×1024. As a result, current state-of-the-art models are typically limited to moderate resolutions, restricting their applicability in domains such as advertising and film production, where ultra high-resolution outputs (e.g., 4K) are required.

Motivated by these limitations, recent work has explored methods to extend pretrained diffusion models to higher resolutions—i.e., beyond their native training sizes. These approaches fall into two main categories: patch-based techniques that process image regions independently, such as Pixelsmith [Tragakis et al. 2024] and DemoFusion [Du et al. 2024], and direct inference methods that modify model architectures, such as HiDiffusion [Zhang et al. 2023] and FouriScale [Huang et al. 2024]. However, these methods face fundamental limitations. Patch-based approaches often produce duplicated objects (Figure 2a), while direct inference methods struggle to maintain global coherence at very high resolutions (e.g., beyond 2048×2048, see Figure 2b). This highlights the need for a training-free, highresolution generation approach that ensures both global coherence and robustness to duplication and artifacts.

In this paper, we propose HiWave, a novel pipeline for trainingfree high-resolution image generation that produces globally coherent outputs without object duplication. HiWave adopts a two-stage, patch-based approach that preserves the coherent structure of a base image generated by a pretrained model, while enhancing the fine details required for higher resolutions. In the first stage, a base image is generated at a standard resolution (e.g., 1024×1024) using a pretrained diffusion model. This image is then upscaled in the image domain to the target high resolution, though the upscaled result lacks fine-grained details. To enrich the image with high-frequency details, we introduce a novel sampling module based on patch-wise DDIM inversion. Specifically, each image patch is inverted using DDIM to recover the corresponding latent noise that would generate the given input. Sampling is then performed starting from this inverted noise. However, to prevent the model from simply reproducing the original image, we incorporate a detail enhancer into the sampling process. This component is based on discrete wavelet transform (DWT), and it preserves the low-frequency content of the base image to maintain global structure while guiding

[Figure 2]

[Figure 3]

[Figure 4]

(a) DemoFusion (b) HiDiffusion (c) HiWave (Ours)

Fig. 2. Qualitative comparison of high-resolution image generation methods at 4096×4096 resolution. DemoFusion, a patch-based method, exhibits object duplication artifacts (highlighted with red rectangles). HiDiffusion, a direct inference method, lacks structural coherence and fine details. In contrast, our method produces coherent generations with rich detail.

the high-frequency components to synthesize realistic additional details suitable for the target resolution.

Our approach enables standard diffusion models trained at a resolution of 1024×1024 to generate images at 4096×4096 resolutiona 16× increase in total pixel count. HiWave leverages the highfrequency priors inherently captured by pretrained diffusion models, as observed by Du et al. [2024], to generate coherent images with fine details. Compared to existing methods, HiWave excels at maintaining structural coherence while significantly reducing hallucinations and duplicated artifacts, addressing a key limitation in current zero-shot high-resolution generation pipelines. We validate the effectiveness of HiWave using Stable Diffusion XL [Podell et al. 2023b] and compare it against Pixelsmith [Tragakis et al. 2024], the current state-of-the-art in training-free high-resolution image generation. In a user study, HiWave was preferred over Pixelsmith in more than 80% of cases, highlighting its superior visual fidelity.

In summary, our main contributions are the following:

- • We introduce HiWave, a training-free, two-stage pipeline for high-resolution image synthesis that extends pretrained diffusion models to ultra high-resolutions, without requiring architectural modifications or additional training.
- • We propose a novel patch-wise DDIM inversion framework coupled with a wavelet-based detail enhancer, which preserves global structure from the base image while selectively enriching its high-frequency details.
- • We demonstrate that HiWave effectively mitigates common artifacts such as duplicated objects and structural inconsistencies that persist in prior methods.
- • We conduct extensive evaluations with Stable Diffusion XL and show that HiWave outperforms current state-of-the-art methods in qualitative comparisons and a preference study, with users favoring HiWave outputs in over 80% of cases.

2 Related Work

Diffusion models have become a dominant framework for image synthesis due to their strong generative capabilities [Denton et al. 2015; Ho et al. 2020; Song et al. 2020b]. They have rapidly surpassed previous generative modeling techniques in terms of fidelity and diversity [Dhariwal and Nichol 2021; Nichol and Dhariwal 2021],

achieving state-of-the-art performance across a wide range of applications, including unconditional image synthesis [Dhariwal and Nichol 2021; Karras et al. 2022], text-to-image generation [Balaji et al. 2022; Esser et al. 2024; Podell et al. 2023b; Ramesh et al. 2022; Rombach et al. 2022; Saharia et al. 2022b; Yu et al. 2022], video synthesis [Blattmann et al. 2023a,b; Gupta et al. 2023], image-to-image translation [Liu et al. 2023; Saharia et al. 2022a], motion synthesis [Tevet et al. 2023; Tseng et al. 2023], and audio synthesis [Chen et al. 2021; Huang et al. 2023; Kong et al. 2021].

Despite this progress, diffusion models often incur high computational costs and long training times [Chen et al. 2023], especially when handling high-resolution data. Latent diffusion models (LDMs) [Rombach et al. 2022] alleviate some of this burden by compressing inputs into a smaller latent space using a pretrained autoencoder. However, LDMs typically scale only up to moderate resolutions (e.g., 1024×1024). While some recent works have explored training LDMs at higher resolutions [Chen et al. 2024; Xie et al. 2024], they continue to face challenges related to prolonged training times and the limited availability of high-quality ultra-high-resolution datasets. These constraints have spurred growing interest in training-free methods that exploit pretrained models to generate images beyond their native resolutions.

One popular direction decomposes high-resolution generation into patches that are processed independently and later combined to form the full image. Examples include DemoFusion [Du et al. 2024], AccDiffusion [Lin et al. 2024], and Pixelsmith [Tragakis et al. 2024]. This patch-based strategy allows diffusion models to operate at their native resolution for each patch, thereby preserving the detail and expressiveness of the base model. However, these methods often fail to maintain global coherence at resolutions beyond 2048×2048, frequently producing boundary artifacts, duplicated content, and semantic inconsistencies across patches.

Another line of work modifies the architecture or inference process of pretrained diffusion models—without additional training—to enable single-pass generation of high-resolution images. Notable examples include FouriScale [Huang et al. 2024], MegaFusion [Wu et al. 2025], and HiDiffusion [Zhang et al. 2023]. These methods introduce attention scaling or downsampling blocks into the pretrained network to better align intermediate features with those seen at the model’s native resolution. While such methods avoid patch-based artifacts, their performance often degrades at ultrahigh resolutions (e.g., 4096×4096). As the gap between training and inference resolutions widens, these approaches struggle to preserve global coherence—leaving patch-based strategies as the current state-of-the-art for zero-shot high-resolution image generation.

In summary, current training-free approaches either fail to maintain global coherence compared to the base diffusion model or suffer from duplicated objects and artifacts at ultra-high resolutions (e.g., 4096×4096). HiWave proposes a novel patch-based strategy aimed at achieving coherent, high-quality image generation at such resolutions while avoiding the limitations of existing methods.

- 3 Background

Diffusion models Let 𝒙 ∼ 𝑝data(𝒙) represent a data sample, and let 𝑡 ∈ [0,𝑇] denote a continuous time variable. In the forward diffusion

process, noise is incrementally added as 𝒛𝑡 = 𝒙 + 𝜎(𝑡)𝝐, where 𝜎(𝑡) is a time-dependent noise scale. This noise schedule gradually corrupts the data, with 𝜎(0) = 0 (clean data) and 𝜎(𝑇) = 𝜎max (maximum corruption). As shown in Karras et al. [2022], this process corresponds to the following differential equation:

d𝒛 = − 𝜎(𝑡)𝜎(𝑡)∇𝒛𝑡 log𝑝𝑡 (𝒛𝑡)d𝑡, (1)

where 𝑝𝑡 (𝒛𝑡) denotes the distribution over noisy samples at time 𝑡, transitioning from 𝑝0 = 𝑝data to 𝑝𝑇 = N(0,𝜎max2 𝐼). Sampling from the data distribution involves solving this ODE in reverse (from 𝑡 = 𝑇 to 𝑡 = 0), provided the score function ∇𝒛𝑡 log𝑝𝑡 (𝒛𝑡) is known. Since this score function is intractable, it is approximated by a neural denoiser 𝐷𝜽 (𝒛𝑡,𝑡), trained to reconstruct the clean signal 𝒙 from its noisy observation 𝒛𝑡. For conditional generation, the denoiser is extended with a conditioning variable 𝒚—such as a label or text prompt—resulting in 𝐷𝜽 (𝒛𝑡,𝑡,𝒚).

Inversion methods Inversion techniques, such as DDIM inversion [Song et al. 2021] solve the forward-time version of the diffusion ODE in Equation 1 to map an observed image 𝒙 to its corresponding noise vector 𝒛𝑇 . By integrating Equation 1 from 𝑡 = 0 to 𝑡 = 𝑇, one obtains a deterministic mapping from the image back to the noise space (in the limit of small steps). This enables applications like image editing, as it provides a way to anchor sampling around a given input while retaining its global layout.

Classifier-free guidance Classifier-free guidance (CFG) [Ho and Salimans 2022] is an inference-time technique that improves generation quality by blending predictions from conditional and unconditional models. During sampling, CFG adjusts the denoiser output as:

𝐷ˆCFG(𝒛𝑡,𝑡,𝒚) = 𝐷𝜽 (𝒛𝑡,𝑡) + 𝑤(𝐷𝜽 (𝒛𝑡,𝑡,𝒚) − 𝐷𝜽 (𝒛𝑡,𝑡)), (2)

where 𝑤 is a guidance strength parameter (with 𝑤 = 1 representing unguided sampling). The unconditional model 𝐷𝜽 (𝒛𝑡,𝑡) is typically learned by randomly dropping conditioning inputs𝒚 during training. Alternatively, separate unconditional models can be used [Karras et al. 2023]. Analogous to the truncation trick in GANs [Brock et al. 2019], CFG improves visual fidelity, but may lead to oversaturation [Sadat et al. 2025] or reduced diversity [Sadat et al. 2024].

Discrete wavelet transform Discrete wavelet transforms (DWT) [Brewster 1993] are a fundamental tool in signal processing, commonly used to analyze spatial-frequency content in data. The transformation utilizes a pair of filters—a low-pass filter 𝐿 and a high-pass filter𝐻. For 2D signals, these are combined to form four distinct filter operations: 𝐿𝐿⊤, 𝐿𝐻⊤, 𝐻𝐿⊤, and 𝐻𝐻⊤. When applied to an image 𝒙, the 2D wavelet transform decomposes it into one low-frequency component 𝒙𝐿 and three high-frequency components 𝒙𝐻,𝒙𝑉,𝒙𝐷, capturing horizontal, vertical, and diagonal details, respectively. Each of these sub-bands has spatial dimensions of 𝐻/2 ×𝑊 /2 for an image of size 𝐻 ×𝑊 . Multiscale decomposition can be achieved by recursively applying the transform to the low-frequency component 𝒙𝐿. The transformation is fully invertible, enabling exact reconstruction of the original image 𝒙 from the set {𝒙𝐿,𝒙𝐻,𝒙𝑉,𝒙𝐷} using the inverse discrete wavelet transform (iDWT).

[Figure 5]

Fig. 3. Overview of HiWave, our training-free high-resolution image generation pipeline. We first generate a base image using a pretrained model through a standard sampling process that transforms random noise (𝑧𝑇 ) into a clean image (𝑧˜0) conditioned on a text prompt. This image is then upscaled in the image domain using Lanczos interpolation and encoded back into the latent space via the VAE encoder to enrich the base image with additional details. A patch-wise DDIM inversion process is then performed, mapping the upscaled image back to its corresponding noise representation. Finally, our DWT-based detail enhancement approach applies frequency-selective guidance during denoising, using wavelet decomposition to independently control low-frequency structure and guide high-frequency components for finer details. Skip residuals are also incorporated during the early sampling steps to further preserve the global coherence of the base image. This pipeline enables HiWave to generate high-quality, high-resolution images without duplications.

- 4 Method

We now describe the details of HiWave, our framework for highresolution image generation using pretrained diffusion models. An overview of the complete pipeline is shown in Figure 3, and we detail each component below. We define 𝐷𝑐(𝒛𝑡) 𝐷𝜽 (𝒛𝑡,𝑡,𝒚), 𝐷𝑢(𝒛𝑡) 𝐷𝜽 (𝒛𝑡,𝑡), and 𝐷ˆCFG(𝒛𝑡) 𝐷 ˆCFG(𝒛𝑡,𝑡,𝒚) to represent the conditional, unconditional, and CFG outputs, respectively.

- 4.1 Base image generation

We begin by generating a base image at the native resolution of the pretrained diffusion model, typically 1024×1024. This image is then upscaled in the image domain using Lanczos interpolation. Unlike most previous works that perform upscaling in the latent space [Du et al. 2024], we opt for image-domain upscaling to avoid artifacts commonly introduced by latent-space interpolation. These artifacts arise because standard VAEs used in diffusion pipelines are not equivariant to scaling operations, leading to inconsistencies when upscaling is applied in latent space [Kouzelis et al. 2025]. An example of such artifacts is illustrated in Figure 4. To avoid this, we first upscale the base image to the target resolution (e.g., 4096×4096) in the image domain. At this point, we obtain an image at the target resolution, albeit lacking fine-grained details. We then

[Figure 6]

[Figure 7]

(a) Upscaling in latent space (b) Upscaling in image space

Fig. 4. Comparison of upscaling in image space vs latent space. Interpolation performed directly in latent space introduces severe spatial artifacts, as standard VAEs are not equivariant to scaling operations. In contrast, interpolating in the image space after decoding preserves structural consistency and visual quality.

encode this upscaled image into the latent space via the VAE encoder and apply a patch-based sampling process (described next) to refine high-resolution details while preserving the global structure of the original image.

- 4.2 Patch-wise DDIM inversion

To preserve structural coherence during patch-wise generation, we initialize the diffusion process using DDIM inversion [Song et al. 2020a] instead of random noise. This inversion retrieves noise vectors for each image patch by integrating the diffusion ODE forward in time:

𝒛𝑡+1 ≈ 𝒛𝑡 − 𝜎(𝑡)𝜎(𝑡)∇𝒛𝑡 log𝑝𝑡 (𝒛𝑡)Δ𝑡, (3)

This deterministic initialization provides two key benefits: (1) Controlled noise for frequency decomposition: DDIM-inverted noise retains meaningful structural information and spatial layout from the original image, enabling our detail enhancement module to selectively refine high-frequency textures while preserving the low-frequency structural content. (2) Consistent patch initialization: Neighboring patches receive compatible noise vectors, which helps maintain continuity and avoid visible seams across patch boundaries. In contrast, initializing with random Gaussian noise loses the structural context of the base image and often introduces artifacts or structural inconsistencies. This controlled inversion lays the groundwork for our DWT-based detail enhancer in the next step, enabling detail enhancement while preserving global coherence.

- 4.3 Detail enhancement with DWT guidance

In the next step, we begin sampling from the inverted noise of each patch, with the goal of progressively adding details to the final image. A central challenge in patch-based high-resolution generation is maintaining a balance between global coherence and detailed texture synthesis. Existing methods often fall short in one of these areas, resulting in either artifact-prone high-frequency patterns or globally consistent images that lack detail.

To address this, we introduce a DWT-based detail enhancement module that leverages the complementary roles of the low- and high-frequency components of each latent. We argue that lowfrequency components typically capture structural coherence, while high-frequency components convey fine details and textures. This distinction is especially critical in patch-wise generation, where each patch must integrate seamlessly into the global image for overall coherence while still containing sufficient detail at high resolutions.

Since we used DDIM inversion to obtain the final noise, following the sampling process using only the conditional prediction 𝐷𝑐(𝒛𝑡) would reproduce the base image, i.e., globally coherent but lacking in fine detail. This motivates our decision to preserve the lowfrequency bands from 𝐷𝑐(𝒛𝑡), which retain much of the base image’s global layout due to the DDIM inversion. To enrich each patch with the required details for high-quality generation, we guide the highfrequency components adaptively using a modified CFG strategy.

Specifically, we apply the DWT to both the conditional and unconditional predictions to get

DWT(𝐷𝑐(𝒛𝑡)) = {𝐷𝑐𝐿(𝒛𝑡),𝐷𝑐𝐻 (𝒛𝑡),𝐷𝑐𝑉 (𝒛𝑡),𝐷𝑐𝐷(𝒛𝑡)}, (4) DWT(𝐷𝑢(𝒛𝑡)) = {𝐷𝑢𝐿(𝒛𝑡),𝐷𝑢𝐻 (𝒛𝑡),𝐷𝑢𝑉 (𝒛𝑡),𝐷𝑢𝐷(𝒛𝑡)}. (5)

We then construct the guided prediction in the frequency domain as follows:

𝐷˜CFG𝐿 (𝒛𝑡) = 𝐷𝑐𝐿(𝒛𝑡), (6) 𝐷˜CFG𝐻 (𝒛𝑡) = 𝐷𝑢𝐻 (𝒛𝑡) + 𝑤𝑑 (𝐷𝑐𝐻 (𝒛𝑡) − 𝐷𝑢𝐻 (𝒛𝑡)), (7) 𝐷˜𝑉CFG(𝒛𝑡) = 𝐷𝑢𝑉 (𝒛𝑡) + 𝑤𝑑 (𝐷𝑐𝑉 (𝒛𝑡) − 𝐷𝑢𝑉 (𝒛𝑡)), (8) 𝐷˜CFG𝐷 (𝒛𝑡) = 𝐷𝑢𝐷(𝒛𝑡) + 𝑤𝑑 (𝐷𝑐𝐷(𝒛𝑡) − 𝐷𝑢𝐷(𝒛𝑡)). (9)

Finally, we apply the inverse DWT to reconstruct the full guided signal:

𝐷˜CFG(𝒛𝑡) = iDWT({𝐷˜CFG𝐿 (𝒛𝑡),𝐷˜CFG𝐻 (𝒛𝑡),𝐷˜𝑉CFG(𝒛𝑡),𝐷˜CFG𝐷 (𝒛𝑡)}).

(10) This frequency-aware guidance strategy enables precise enhancement of details while preserving the global structure of the base image.

- 4.4 Skip residuals

To further preserve global structure during early denoising, we incorporate skip residuals by mixing the latents obtained from DDIM inversion with those from the sampling process for each patch. Let 𝒛𝑡 denote the current latent in the sampling process, 𝒛𝑡𝑠 the corresponding DDIM-inverted latent, 𝜏 a time step threshold, and 𝑐1 = ((1 + cos(𝑇𝑇−𝑡 𝜋))/2)𝛼 a cosine-decay weighting factor. The skip residual update is defined as

𝒛ˆ𝑡 =

𝑐1 × 𝒛𝑡 + (1 − 𝑐1) × 𝒛𝑡𝑠 𝑡 < 𝜏, 𝒛𝑡 𝑡 ≥ 𝜏.

(11)

Unlike prior work that applies skip residuals throughout all diffusion steps, we adopt a more conservative strategy: they are only used during the initial denoising phase. This allows the model to leverage the base image’s structure early on, and then progressively diverge to synthesize novel details guided by the DWT-enhanced predictions. In contrast, applying skip residuals at all time steps–as done in previous work–can suppress detail synthesis, while omitting them entirely causes duplication artifacts. Our DWT-based enhancer mitigates this trade-off by explicitly guiding different frequency bands, enabling the generation of rich textures while preserving global consistency.

- 4.5 Implementation Details

We use the sym4 wavelet for DWT due to its effective balance between spatial and frequency localization. The detail guidance strength is set to 𝑤𝑑 = 7.5, enhancing high-frequency features while preserving the low-frequency structure from 𝐷𝑐(𝒛𝑡).

Image generation is performed progressively—first at 1024×1024 (the model’s native resolution), then at 2048×2048, and finally at 4096×4096. While some prior works report increased duplication artifacts with iterative upscaling [Tragakis et al. 2024], we did not observe this in our experiments, likely due to the combination of patch-wise DDIM inversion and our DWT-based guidance.

Skip residuals are applied only up to time step 15 (out of 50) for 2048×2048 resolution and up to time step 30 for 4096×4096. This conservative strategy contrasts with methods that apply skip residuals across the entire diffusion process. By limiting their use to

early steps, we preserve the global structure initially while allowing the model to synthesize novel details in later time steps.

For patch processing, we use a 50% overlap between adjacent patches to ensure smooth transitions. To optimize memory usage, we employ a streaming approach in which patches are processed in batches rather than all at once, enabling 4096×4096 image generation on consumer GPUs with 24GB of VRAM.

- 5 Results

We now evaluate HiWave both qualitatively and quantitatively against state-of-the-art high-resolution image generation methods. Our evaluation focuses on three main aspects: (1) avoiding the duplication artifacts commonly seen in previous patch-based methods, (2) achieving global image coherence and high fidelity at 4096×4096 resolution, and (3) enhancing both quality and details over the original Stable Diffusion XL outputs.

- 5.1 Experimental setup

We select Pixelsmith [Tragakis et al. 2024] as the leading patch-based approach and HiDiffusion [Zhang et al. 2023] as the representative of direct inference methods. All methods were used to generate 4096×4096 resolution images from the same prompts and identical random seeds, using Stable Diffusion XL as the base model. To ensure a fair comparison, we employed the official codebases of all baselines. All experiments were conducted on a single RTX 4090 GPU with 24GB of VRAM.

To benchmark the methods, we used 1000 randomly sampled prompts from the LAION/LAION2B-en-aesthetic [Schuhmann et al.

- 2022] dataset, covering a diverse range of content including natural landscapes, human portraits, animals, architectural scenes, and close-up textures. This diversity allowed us to assess performance across a broad set of generation challenges.

For quantitative evaluation, prior work has shown that existing metrics are often unreliable at high resolutions, as they typically downscale images to lower resolutions (e.g., 224×224) before computing the score [Du et al. 2024; Tragakis et al. 2024; Zhang et al.

- 2023]. Consequently, we rely primarily on a human study, which provides the most reliable assessment of perceptual quality in this setting. Standard quantitative metrics are also reported in Appendix A for completeness.

- 5.2 Qualitative comparison with prior methods

Figure 7 presents a comprehensive visual comparison of our method against Pixelsmith and HiDiffusion across nine diverse test cases. Each row corresponds to a different subject, with the left columns displaying the full-resolution outputs and the right columns showing magnified regions (highlighted by green boxes) to facilitate detailed inspection of fine structures. HiDiffusion produces outputs that lack coherent structure and exhibit blurred textures across all nine examples. This demonstrates that, while direct inference methods avoid duplication and patch-based artifacts, they often struggle to produce globally coherent outputs with fine-grained detail at high resolutions. In contrast, Pixelsmith generates more detailed images but frequently suffers from object duplication. For instance, it produces duplicated humans in rows 1 through 4 and a phantom

figure in the grass background of row 5. By comparison, HiWave consistently generates high-quality images free from artifacts and duplication. This illustrates that HiWave effectively addresses a key limitation of prior approaches, enabling coherent and detailed high-resolution generation without duplication.

- 5.3 Detail enhancement over the base image

- Figure 5 illustrates how HiWave enhances fine details and can improve semantic plausibility in base images generated by the SDXL model at the original resolution. Our method retains the same overall composition but successfully extracts and amplifies fine details that were merely suggested in the original SDXL generations. In row 1, the intricate blue patterns on the porcelain bottle show substantially improved definition, with clear brushwork details and subtle variations in the blue pigment that are barely distinguishable in the original image. Row 2 illustrates how our method reveals individual yarn strands and stitch patterns of a knitted toy mouse with great clarity. In row 3, a child in the flower field shows how our method can handle natural scenes with finer hair detail, clothing texture, and surrounding flora detail without losing scene composition. These results demonstrate that HiWave effectively adds the fine-grained details necessary for high-quality high-resolution image generation.

5.4 Human evaluation study

To validate our qualitative observations, we conducted a comprehensive human preference study comparing HiWave against Pixelsmith. Using 32 image pairs generated with identical prompts and seeds at 4096×4096 resolution, we presented participants with randomized blind A/B tests. The 32 prompts were selected solely based on the SDXL base outputs–prior to running any methods–to avoid selection bias. The participants were asked to select their preferred image based on overall quality, coherence, and absence of artifacts.

- Figure 6 shows the preference scores for HiWave and Pixelsmith. Across 548 independent evaluations, HiWave was preferred in 81.2% of responses (445 out of 548), with seven test cases achieving 100% preference for our method. The full set of per-question preference percentages is provided in Figure 15 (Appendix). This strong preference aligns with our qualitative findings regarding artifact reduction and coherence preservation in Figure 7.

- 5.5 Ablation Study

Additional results and ablation studies analyzing HiWave’s performance and the contribution of its components are provided in the appendix. Figure 10 further confirms that HiWave improves both global structure and fine details compared to the base image generated with SDXL at 1024×1024. Figure 11 demonstrates that removing guidance from low-frequency components effectively eliminates duplication artifacts. Figure 12 highlights the importance of initializing sampling from DDIM-inverted noise to avoid geometric and color inconsistencies across patches, while the bottom-right subfigure in Figure 11 shows that this step alone is insufficient. Therefore, DWT-based frequency guidance remains essential for fully mitigating duplication and producing highly detailed, artifactfree images. Figure 13 confirms that multistep generation produces

[Figure 8]

- Fig. 5. Comparison between our HiWave method at 4096×4096 resolution (left) and the SDXL base model at 1024×1024 resolution (right). Each row displays a different image along with corresponding zoomed-in regions to highlight detail enhancement. Row 1 features a porcelain bottle with intricate blue patterns; Row 2 shows a knitted toy mouse with clearly visible yarn texture; and Row 3 depicts a child in a flower field, with enhanced hair and fabric details. The zoomed regions illustrate how HiWave preserves the overall composition generated by SDXL while significantly enhancing fine details that are only partially present in the original generations.

0 20 40 60 80 100

HiWave

Pixelsmith

81.2

18.8

Human preference (%)

- Fig. 6. User preference comparison between HiWave and Pixelsmith. HiWave was preferred by participants in over 80% of cases, validating its effectiveness in generating high-quality images at high resolutions.

without requiring architectural modifications or additional training. HiWave employs a two-stage, patch-based strategy that first generates a base image using a pretrained diffusion model and subsequently refines individual patches of the upscaled image to achieve higher resolutions. Specifically, HiWave leverages a patch-wise DDIM inversion approach to recover the latent noise from the base image and incorporates a novel wavelet-based detail enhancement module that selectively guides high-frequency components while preserving or enhancing the global structure via low-frequency components. With this frequency-aware guidance mechanism, HiWave overcomes common pitfalls of existing high-resolution methodsnamely, object duplication and structural incoherence—enabling models trained at 1024×1024 to generate coherent and detailed outputs at 4096×4096. Extensive experiments with Stable Diffusion XL demonstrated that HiWave not only produces visually compelling 4K images with fine details and strong global consistency, but also significantly outperforms prior zero-shot high-resolution approaches. A user study further supported these findings, with participants preferring HiWave’s results in more than 80% of cases. By enabling ultra-high-resolution synthesis without retraining, HiWave unlocks practical applications in domains where 4K (and beyond) outputs are essential. Future work could explore extensions to video, runtime optimizations, and the integration of more advanced guidance strategies within the detail enhancer module. Overall, HiWave represents a significant step toward democratizing high-fidelity, high-resolution generative modeling.

sharper details than single-step generation, without introducing duplication in our setup—unlike what has been reported in prior work [Tragakis et al. 2024]. Figure 14 illustrates that HiWave can also upscale natural (non-AI-generated) images in a zero-shot manner by leveraging the representations embedded in the SDXL noise space obtained via DDIM inversion. Figure 16 demonstrates HiWave’s ability to generate artifact-free images at an ultra-high resolution of 8192×8192. Lastly, additional comparisons in Figure 17 further shows that HiWave outperforms Pixelsmith by generating sharper, less blurry images while significantly reducing duplication artifacts.

- 6 Conclusion

In this work, we introduced HiWave, a novel zero-shot pipeline that enables pretrained diffusion models to generate ultra-highresolution images (e.g., 4096×4096) beyond their native resolution,

References

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. 2022. eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers. CoRR abs/2211.01324 (2022). doi:10.48550/arXiv.2211.01324 arXiv:2211.01324

Mikołaj Bińkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. 2018. Demystifying mmd gans. arXiv preprint arXiv:1801.01401 (2018).

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. 2023a. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. CoRR abs/2311.15127 (2023). doi:10.48550/ARXIV.2311. 15127 arXiv:2311.15127

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023b. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.

M. E. Brewster. 1993. An Introduction to Wavelets (Charles K. Chui). SIAM Rev. 35, 2

(1993), 312–313. doi:10.1137/1035061

Andrew Brock, Jeff Donahue, and Karen Simonyan. 2019. Large Scale GAN Training for High Fidelity Natural Image Synthesis. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net. https://openreview.net/forum?id=B1xsqj09Fm

Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. 2024. PIXART-𝛿: Fast and Controllable Image Generation with Latent Consistency Models. ArXiv abs/2401.05252 (2024). https://api.semanticscholar.org/ CorpusID:266902626

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. 2023. PixArt-𝛼: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426 (2023).

Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. 2021. WaveGrad: Estimating Gradients for Waveform Generation. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net. https://openreview.net/forum?id=NsMLjcFaO8O

Emily L Denton, Soumith Chintala, Rob Fergus, et al. 2015. Deep generative image models using a laplacian pyramid of adversarial networks. Advances in neural information processing systems 28 (2015).

Prafulla Dhariwal and Alexander Quinn Nichol. 2021. Diffusion Models Beat GANs on Image Synthesis. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (Eds.). 8780–8794. https://proceedings. neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. 2024. Demofusion: Democratising high-resolution image generation with no. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6159–6168.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning.

Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. 2023. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662 (2023).

Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. 2023. Scalecrafter: Tuning-free higher-resolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017). Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic

models. Advances in neural information processing systems 33 (2020), 6840–6851. Jonathan Ho and Tim Salimans. 2022. Classifier-Free Diffusion Guidance. CoRR abs/2207.12598 (2022). doi:10.48550/arXiv.2207.12598 arXiv:2207.12598

Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. 2024. Fouriscale: A frequency perspective on training-free highresolution image synthesis. In European Conference on Computer Vision. Springer, 196–212.

Qingqing Huang, Daniel S. Park, Tao Wang, Timo I. Denk, Andy Ly, Nanxin Chen, Zhengdong Zhang, Zhishuai Zhang, Jiahui Yu, Christian Havnø Frank, Jesse H. Engel, Quoc V. Le, William Chan, and Wei Han. 2023. Noise2Music: Text-conditioned Music Generation with Diffusion Models. CoRR abs/2302.03917 (2023). doi:10.48550/arXiv. 2302.03917 arXiv:2302.03917

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. 2022. Elucidating the Design Space of Diffusion-Based Generative Models. (2022). https://openreview.net/forum?

id=k7FuTOWMOc7

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. 2023. Analyzing and Improving the Training Dynamics of Diffusion Models. arXiv:2312.02696 [cs.CV]

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. 2021. DiffWave: A Versatile Diffusion Model for Audio Synthesis. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net. https://openreview.net/forum?id=a-xFK8Ymz5J

Theodoros Kouzelis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. 2025. EQ-VAE: Equivariance Regularized Latent Space for Improved Generative Image Modeling. arXiv preprint arXiv:2502.09509 (2025).

Zhihang Lin, Mingbao Lin, Meng Zhao, and Rongrong Ji. 2024. Accdiffusion: An accurate method for higher-resolution image generation. In European Conference on Computer Vision. Springer, 38–53.

Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, and Anima Anandkumar. 2023. I2SB: Image-to-Image Schrödinger Bridge. CoRR abs/2302.05872 (2023). doi:10.48550/arXiv.2302.05872 arXiv:2302.05872

Alexander Quinn Nichol and Prafulla Dhariwal. 2021. Improved Denoising Diffusion Probabilistic Models. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (Proceedings of Machine Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 8162–8171. http: //proceedings.mlr.press/v139/nichol21a.html

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023a. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023).

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023b. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. CoRR abs/2307.01952 (2023). doi:10. 48550/ARXIV.2307.01952 arXiv:2307.01952

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical Text-Conditional Image Generation with CLIP Latents. CoRR abs/2204.06125

(2022). doi:10.48550/arXiv.2204.06125 arXiv:2204.06125

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE, 10674–10685. doi:10.1109/CVPR52688.2022.01042

Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M. Weber. 2024. CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=zMoNrajk2X

Seyedmorteza Sadat, Otmar Hilliges, and Romann M. Weber. 2025. Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum? id=e2ONKX6qzJ

Chitwan Saharia, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim Salimans, David J. Fleet, and Mohammad Norouzi. 2022a. Palette: Image-to-Image Diffusion Models. In SIGGRAPH ’22: Special Interest Group on Computer Graphics and Interactive Techniques Conference, Vancouver, BC, Canada, August 7 - 11, 2022, Munkhtsetseg Nandigjav, Niloy J. Mitra, and Aaron Hertzmann (Eds.). ACM, 15:1– 15:10. doi:10.1145/3528233.3530757

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022b. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems 35 (2022), 36479– 36494.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. 2016. Improved techniques for training gans. Advances in neural information processing systems 29 (2016).

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. 2022. LAION-5B: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track. https: //openreview.net/forum?id=M3Y74vmsMcY

- Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020a. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020).
- Jiaming Song, Chenlin Meng, and Stefano Ermon. 2021. Denoising Diffusion Implicit Models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net. https://openreview.net/

forum?id=St1giarCHLP

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020b. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020).

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna.

2016. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2818–2826.

Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit Haim Bermano. 2023. Human Motion Diffusion Model. (2023). https://openreview.net/ pdf?id=SJ1kSyO2jwu

Athanasios Tragakis, Marco Aversa, Chaitanya Kaul, Roderick Murray-Smith, and Daniele Faccio. 2024. Is One GPU Enough? Pushing Image Generation at HigherResolutions with Foundation Models. arXiv preprint arXiv:2406.07251 (2024).

Jonathan Tseng, Rodrigo Castellon, and C. Karen Liu. 2023. EDGE: Editable Dance Generation From Music. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 448–458. doi:10.1109/CVPR52729.2023.00051

Haoning Wu, Shaocheng Shen, Qiang Hu, Xiaoyun Zhang, Ya Zhang, and Yanfeng Wang. 2025. Megafusion: Extend diffusion models towards higher-resolution image generation without further tuning. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 3944–3953.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341 (2023).

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. 2024. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629 (2024).

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. 2022. Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. Trans. Mach. Learn. Res. 2022 (2022). https://openreview.net/forum?id=AFDcYJKhND Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Zhenyuan Chen, Yao Tang, Yuhao Chen, Wengang Cao, and Jiajun Liang. 2023. Hidiffusion: Unlocking high-resolution creativity and efficiency in low-resolution trained diffusion models. CoRR (2023).

[Figure 9]

- Fig. 7. Qualitative comparison of high-resolution (4096×4096) image generation across three methods. HiDiffusion (left column) consistently struggles to produce realistic details and coherent structures, leading to blurry textures and distorted features. Pixelsmith (middle column) generally generates high-quality details but exhibits noticeable duplication artifacts—particularly in background elements and textures—as highlighted in the zoomed regions (green boxes). In contrast, HiWave (right column) maintains structural coherence and delivers sharp, artifact-free generations without duplications.

[Figure 10]

### Fig. 8. Examples of high-resolution (4096×4096) images generated by our method, illustrating a variety of subjects across diverse visual motifs.

A Quantitative evaluation

Table 1 presents a quantitative comparison using several standard metrics commonly used in evaluating generative image models. While these metrics provide useful insights into model performance at moderate resolutions, it is important to recognize their limitations when assessing high-resolution outputs (e.g., 2048×2048 and 4096×4096) produced by methods such as HiWave.

A key issue is that many of these metrics depend on neural network architectures designed for fixed, lower-resolution inputs (See below for details). As a result, evaluating our high-resolution images requires substantial downsampling, which may obscure improvements in fine-grained details, textures, and overall fidelity that high-resolution synthesis is intended to achieve.

Additionally, perceptual inconsistencies often go undetected by these scores. For instance, as shown in Figure 9, the version of the wedding photo generated without DWT guidance exhibits clear duplicationartifacts—hallucinatingasecond couple in the tree canopywhereas the DWT-guided version appears visually clean. However, metrics such as ImageReward (0.0443 vs 0.0168) fail to reflect this perceptual gap, further emphasizing the need for qualitative and human evaluation.

Despite these limitations, HiWave demonstrates comparable or superior performance relative to prior methods across both resolutions, underscoring its capacity to produce high-quality, detailed images.

We provide details of each metric used in Table 1 below.

FID (Fréchet Inception Distance) [Heusel et al. 2017] and KID (Kernel Inception Distance) [Bińkowski et al. 2018]: These widely used metrics compare feature representations extracted by the Inception V3 network [Szegedy et al. 2016]. Inception V3 requires input images to be resized to 299x299 pixels. Downsampling from 4096x4096 to 299x299 inevitably leads to a loss of high-frequency information.

IS (Inception Score) [Salimans et al. 2016]: Similarly, the Inception Score relies on the Inception V3 network and thus requires images to be resized to 299x299 pixels. While indicative of image quality at that scale, it does not assess details only visible at higher resolutions.

CLIP Score: This metric evaluates the semantic alignment between the generated image and the input text prompt using embeddings from the CLIP model [Radford et al. 2021]. Commonly used CLIP vision encoders, such as ViT-L/14 or ViT-B/32, process images at a resolution of 224x224 pixels. This significant downsampling primarily captures semantic content rather than high-resolution fidelity.

LPIPS (Learned Perceptual Image Patch Similarity) [Zhang et al. 2018]: LPIPS measures perceptual similarity using features from networks like AlexNet or VGG. While designed to correlate well with human perception, standard implementations often process images resized to resolutions like 256x256 pixels, limiting their ability to assess fine details present only at much higher resolutions.

HPS-v2 (Human Preference Score v2) [Wu et al. 2023]: While HPS v2 addresses limitations of earlier metrics by aligning better with human aesthetic judgment, it still relies on the CLIP ViT-H/14 architecture [Radford et al. 2021], which processes images at 224x224

[Figure 11]

Fig. 9. Failure example of current metrics in evaluating high-resolution images. The left image (HiWave w/o DWT) contains visible duplication artifacts, including a hallucinated couple in the trees and a disembodied hand in the grass. In contrast, the DWT-guided HiWave result (right) is clean and artifact-free. Despite this clear perceptual difference, common metrics faile to reflect the quality gap. ImageReward scores are 0.0443 vs 0.0168, respectively.

resolution. Therefore, evaluating our 4096x4096 outputs requires the same significant downsampling as the CLIP score, potentially losing some of the fine details our method aims to generate.

- B Runtime analysis

Table 2 presents the inference times for various high-resolution generation methods. Our method delivers high-quality results with a reasonable runtime compared to previous approaches. While direct inference methods are generally faster, they often sacrifice visual quality at high resolutions. Patch-based methods, though slower, offer superior fidelity and detail. HiWave embraces this trade-off, prioritizing image quality over speed—an approach validated by strong qualitative outcomes. Although our runtime exceeds that of Pixelsmith, this is due to our multistep upscaling strategy designed to enhance fidelity (as illustrated in Figure 13). In contrast, Pixelsmith’s one-shot pipeline prioritizes speed at the cost of quality. Considering these differences, our runtime remains comparable to other patch-based methods.

- C Improvements over the base image

In Figure 10, we compare the base generations of SDXL with the HiWave outputs across a variety of challenging content types. The examples include cropped regions representing 1/5 the width of the original images (SDXL at 1024×1024, HiWave at 4096×4096). HiWave consistently enhances facial definition (Rows 1–3) and produces sharper character renderings (Row 4), while faithfully preserving the global composition and visual style of the original outputs. These results demonstrate that HiWave effectively improves both the fine-grained details and structural coherence of the base images.

C.1 Analyzing the role of frequency-specific guidance

Figure 11 compares various configurations of our frequency guidance strategy. The top-left image shows the baseline SDXL output at 1024×1024 resolution, while the top-right image presents the result

- Table 1. Quantitative comparison of high-resolution image generation methods. Results are based on 1,000 random samples from the LAION/LAION2B-enaesthetic dataset. Although current metrics struggle to reliably evaluate high-resolution images, HiWave achieves comparable or superior performance to state-of-the-art methods in this domain.

Resolution Model FID↓ KID↓ IS↑ CLIP↑ LPIPS↓ HPS-v2 ↑

10242 SDXL 61.78 0.0020 18.6651 33.2183 0.7779 0.2639

20482

HiWave (Ours) 63.3460 0.0027 19.4784 33.2568 0.7784 0.2620 Pixelsmith 62.3054 0.002197 19.1222 33.1550 0.7808 0.2598 HiDiffusion 65.9073 0.002895 17.7217 31.9566 0.7834 0.2432

40962

HiWave (Ours) 64.7265 0.003241 18.7743 33.2729 0.7831 0.2585 Pixelsmith 62.5542 0.002373 19.4302 33.1504 0.8039 0.2598 HiDiffusion 93.4473 0.014934 14.6960 28.2296 0.7996 0.1821

- Table 2. Inference time comparison (in seconds) for various high-resolution image generation methods on an RTX 3090 GPU. HiWave achieves a runtime comparable to previous patch-based generation approaches. Runtimes for prior methods are reported from [Tragakis et al. 2024]. OOM indicates outof-memory on the RTX 3090.

- E One-Shot vs multistep generation

This experiment compares one-shot generation at 4096×4096 resolution with a progressive multistep upscaling strategy using HiWave. In the one-shot setting, the image is generated directly at the target resolution. In the multistep approach, generation begins at 1024×1024 and is successively upscaled to 2048×2048 and then to 4096×4096. As illustrated in Figure 13, the multistep strategy yields higher fidelity. For example, the raspberry exhibits a more realistic surface and texture, whereas the one-shot result lacks fine details. These findings highlight the importance of progressive upscaling for high-resolution synthesis, as each intermediate step allows HiWave to incrementally refine and enrich image details.

- F Upscaling real (non-generated) images

To evaluate the applicability of HiWave beyond synthetic images, we applied our method to real 1024×1024 photographs instead of generated images from SDXL. For these real photographs, we manually crafted descriptive prompts that matched the image content, as such prompt conditioning is necessary for HiWave to guide the high-resolution generation. Figure 14 compares the original photographs (left) with their corresponding high-resolution upscalings to 2048×2048 using HiWave. In the turtle example, HiWave convincingly enhances fine details such as the shell texture and flipper structure. The improvement is subtler for the market scene, possibly due to domain-specific differences in style or tone compared to the SDXL training distribution. In the final example, HiWave enhances structural elements like the hand and phone and sharpens some text regions, though it may also introduce minor variations in fine details—an area that could benefit from further refinement. Overall, these results demonstrate HiWave’s strong potential for real-image enhancement, even in challenging scenarios.

- G Detailed results of our user study

# Method 20482 40962

SDXL [Podell et al. 2023b] 71 515 ScaleCrafter [He et al. 2023] 80 1257 HiDiffusion [Zhang et al. 2023] 50 255 FouriScale [Huang et al. 2024] 162 OOM

Direct inference

DemoFusion [Du et al. 2024] 219 1632 AccDiffusion [Lin et al. 2024] 231 1710 Pixelsmith [Tragakis et al. 2024] 130 549 HiWave (Ours) 238 1557

Patch-based

of our full HiWave method using DWT-based guidance at 4096×4096. To isolate the effect of frequency-specific control, the bottom-left variant applies guidance only to low-frequency bands—essentially the inverse of HiWave—while the bottom-right replaces our DWTbased approach with standard CFG. Both the low-frequency-only and CFG-only variants exhibit noticeable duplication artifacts. In contrast, HiWave maintains the global composition of the original SDXL image while enhancing fine-grained details. These results suggest that maintaining low-frequency components while selectively guiding high frequencies is crucial for producing coherent and detailed generations.

D Importance of DDIM inversion

To illustrate the impact of DDIM inversion in HiWave, Figure 12 compares outputs with and without this component. The left image shows HiWave applied without DDIM inversion, while the right displays the result of the full HiWave method. Without DDIM inversion, patch-wise generation becomes uncoordinated and independent, resulting in visible seams, geometric inconsistencies, and color mismatches at patch boundaries. In contrast, the full HiWave method preserves spatial coherence across the entire image.

To evaluate perceptual quality, we conducted a blind A/B test comparing HiWave to Pixelsmith across 32 diverse image pairs. Participants were asked to indicate their preferred image in each pair. As shown in Figure 15, HiWave was preferred in all cases and received a substantially higher proportion of votes in the majority of comparisons. This strong preference shows the effectiveness of

[Figure 12]

Fig. 10. Comparison between SDXL base generations and HiWave generations across various challenging content types. The shown crops are 1/5 the length of the original images (1024×1024 for SDXL, and 4096×4096 for HiWave). Our approach enhances facial definition (Rows 1–3) and renders sharper characters (Row 4), while preserving the global composition and visual style.

our approach in producing visually compelling outputs and demonstrates consistent perceptual superiority over previous methods.

- H Scaling HiWave to 8K resolution

To examinethescalabilitylimitsofHiWave, we extended our method to generate images at 8192×8192 resolution—64 times the pixel count of the standard 1024×1024 baseline. As shown in Figure 16, HiWave successfully maintains structural coherence and fine-grained details

[Figure 13]

- Fig. 11. Comparison of different configurations for our DWT-based frequency guidance. Top-left: SDXL base image at 1024×1024. Top-right: full HiWave with DWT-based guidance at 4096×4096. Bottom-left: guidance applied only to low-frequency bands (inverse of our method). Bottom-right: standard CFG. Our DWT-based approach is crucial for reducing duplications and enhancing details.

[Figure 14]

- Fig. 12. Effect of omitting DDIM inversion in HiWave. Left: HiWave without DDIM inversion. Right: Full HiWave method. Without DDIM inversion, patch-wise generation becomes independent, resulting in visible seams, inconsistent geometry, and color mismatches across patches.

even at this extreme resolution. Its ability to preserve consistency and quality at such scale highlights the robustness of our approach, positioning it as a promising tool for applications demanding ultrahigh resolutions.

[Figure 15]

- Fig. 13. Comparison between single-step 4096×4096 generation (left) and progressive multistep upscaling (right). The multistep approach generates the image at 1024, then progressively upscales it to 2048 and 4096. This progressive generation process results in sharper details and enhanced overall quality.

- I More Qualitative comparisons with Pixelsmith

In this section, we present additional qualitative comparisons between HiWave and Pixelsmith on 4096×4096 resolution scenes to further evaluate the visual quality of our method. As shown in Figure 17, each row displays a different scene along with close-up crops for detailed inspection. HiWave consistently preserves global structural integrity while enhancing local textures and edge sharpness. Compared to Pixelsmith, our method more effectively reduces blurring, retains fine details, and avoids common artifacts such as duplications. These results further highlight HiWave’s advantage in generating high-fidelity images.

[Figure 16]

### Fig. 14. Upscaling natural 1024×1024 photographs to 2048×2048 resolution using HiWave. Each row shows one input image (left) and the corresponding HiWave result (right), along with zoomed-in crops from two marked regions (in red and green). The close-up views represent 8× magnifications. HiWave enhances fine-grained detail while preserving the semantics and appearance of the original photos.

Preference(%)

120

100

100

100

100

100

100

100

100

933.

917.

909.

905.

895.

895.

882.

875.

867.

857.

818.

813.

80

80

80

706.

688.

667.

667.

632.

583.

571.

562.

60

60

533.

467.

50

438.

429.

417.

368.

40

40

333.

333.

312.

294.

188.

182.

20

20

143.

20

133.

125.

118.

105.

105.

95.

91.

83.

67.

0

0

0

0

0

0

0

0

Q1Q2Q3Q4Q5Q6Q7Q8Q9Q10Q11Q12Q13Q14Q15Q16Q17Q18Q19Q20Q21Q22Q23Q24Q25Q26Q27Q28Q29Q30Q31Q32

Question

HiWave (Ours) Pixelsmith

### Fig. 15. Human preference (%) across 32 image pairs comparing HiWave and Pixelsmith. Blue bars indicate the proportion of votes favoring HiWave, while orange bars represent votes for Pixelsmith in the blind A/B test. HiWave significantly outperforms Pixelsmith in most cases, even achieving 100% preference in 7 instances.

[Figure 17]

### Fig. 16. 8K generation results using HiWave. The model continues to produce plausible images with fine details and coherent structure at this ultra-high resolution. This experiment demonstrates HiWave’s ability to scale effectively while preserving structural coherence.

[Figure 18]

### Fig. 17. Furthe comparison of HiWave with Pixelsmith across diverse 4096×4096 scenes. Each row presents a different scene, with corresponding crops for detailed inspection. HiWave consistently preserves global structure while enhancing local textures and edge sharpness. In several cases, HiWave reduces blurring, improves fine details, and avoids duplications seen in Pixelsmith outputs.

