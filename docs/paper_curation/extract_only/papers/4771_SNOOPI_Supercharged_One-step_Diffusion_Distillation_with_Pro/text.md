

[Figure 2]

## Supercharged One-step Text-to-Image Diffusion Models with Negative Prompts

Viet Nguyen1⋆, Anh Nguyen1⋆, Trung Dao1, Khoi Nguyen1, Cuong Pham1,2, Toan Tran1, Anh Tran1 1Qualcomm AI Research† 2Posts & Telecom. Inst. of Tech.

A bunny in high ornamented light armor, fluffy fur, foggy, wet, stormy atmosphere, 70mm cinematic, highly detailed

A polar bear wearing a glowing space suit, standing on an icy asteroid, distant planets in the background

A mystical deer with glowing antlers in a moonlit forest, fantasy art, detailed and enchanting

A portrait of a friendly monster

# arXiv:2412.02687v3[cs.CV]24Sep2025

— close-up, zoomed-in, facial focus

— white and blue background

— scary, aggressive, menacing, evil

— dim illumination, dark lighting

❌ ❌

❌

❌

Negative-Away Steer Attention

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

CFG One-step

NASA

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Multi-step

Improve Controllability Improve Quality

Figure 1. Negative prompting are a key technique for improving controllability and image quality in text-to-image diffusion models. However, classifier-free guidance (CFG) [19] only supports negative prompt integration in multi-step models due to its iterative nature. Our method, NASA, is the first to enable negative prompting in one-step models, allowing for the suppression of unwanted features and providing greater control over image attributes.

### Abstract

avoids the blending artifacts inherent in output-space guidance and achieves high efficiency, incurring only a minimal 1.89% increase in FLOPs compared to the computational doubling of CFG. Furthermore, NASA can be seamlessly integrated into existing timestep distillation frameworks, enhancing the student’s output quality. Experimental results demonstrate that NASA substantially improves controllability and output quality, achieving an HPSv2 score of 31.21, setting a new state-of-the-art benchmark for one-step diffusion models.

The escalating demand for real-time image synthesis has driven significant advancements in one-step diffusion models, which inherently offer expedited generation speeds compared to traditional multi-step methods. However, this enhanced efficiency is frequently accompanied by a compromise in the controllability of image attributes. While negative prompting, typically implemented via classifierfree guidance (CFG), has proven effective for fine-grained control in multi-step models, its application to one-step generators remains largely unaddressed. Due to the lack of iterative refinement, as in multi-step diffusion, directly applying CFG to one-step generation leads to blending artifacts and diminished output quality. To fill this gap, we introduce Negative-Away Steer Attention (NASA), an efficient method that integrates negative prompts into one-step diffusion models. NASA operates within the intermediate representation space by leveraging cross-attention mechanisms to suppress undesired visual attributes. This strategy

### 1. Introduction

Diffusion models have recently become popular due to their capacity to produce high-quality, diverse outputs in image [3, 7, 36, 37, 41], 3D [26, 38, 46], audio [15, 21] and video [5, 50] synthesis. Unlike other generative models, such as Generative Adversarial Networks (GANs) [16, 23], diffusion models gradually refine their output through a series of steps, allowing them to achieve diverse yet impressively detailed and realistic outputs. However, this iterative re-

⋆ Equal Contribution † Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.

finement is both time-consuming and computationally demanding, which limits their practicality in real-world applications. As a result, there is an increasing interest in techniques to speed up diffusion models while maintaining output quality. Recent distillation methods [12, 13, 34, 51, 52] compress the entire multi-step generation process into a single step, a breakthrough that enables real-time image synthesis. This speedup makes diffusion models practical for applications like content creation and interactive media, where their slow performance was previously a major drawback. Despite these impressive speed advancements in onestep generators, we have identified a significant limitation that affects all these models: the inability to support negative prompting [4, 48]. This feature, which allows users to explicitly exclude unwanted elements, is widely available in multi-step diffusion models but remains notably absent in their single-step alternatives.

The most widely adopted approach for negative prompting, classifier-free guidance (CFG) [19] works by generating two parallel noise predictions during each denoising step – one incorporating the positive prompt and another using the negative prompt. The system then subtracts the negative prediction from the positive one, effectively steering generation away from unwanted attributes. This process relies on the iterative nature of multi-step diffusion, where corrections can be applied and refined over dozens of steps.

One-step generators, however, collapse this entire process into a single forward pass, eliminating the iterative refinement cycles that make CFG possible. Without these multiple correction opportunities, there is no straightforward way to incorporate negative guidance into the generation pipeline. Consequently, users face an unappealing choice between the instant generation that might include unwanted elements or waiting longer for more precisely controlled results. This speed-versus-control dilemma represents the most significant practical barrier to widespread adoption of one-step generators in professional creative workflows, highlighting the urgent need for novel approaches that can bridge this gap.

To address this issue, we introduce a novel method called Negative-Away Steer Attention (NASA) used in inference, which incorporates negative prompt guidance directly into one-step diffusion models. NASA works by adjusting cross-attention layers, effectively reducing unwanted features in generated images. By steering attention within intermediate representation space, NASA offers image control capabilities that were previously achievable only in multi-step diffusion models.

With our newly developed capability, we have explored integrating negative prompting directly into one-step distillation frameworks, focusing on the SwiftBrush (SB) family of models [13, 34]. The SB approach is compelling for its unique, image-free distillation methodology. By match-

ing teacher model probability distributions and applying sophisticated training techniques, SBv2 can outperform its own teacher on standard image synthesis benchmarks. Despite these technical achievements, SB models still fall short in human preference evaluations. The core challenge lies in the distillation process: without real image supervision, the system lacks clear signals about which generation patterns should be avoided. This is precisely where negative prompting and our NASA framework come in.

Committed to maintaining SB’s valuable image-free distillation approach while incorporating guidance sampling to align with human preference, we provide the critical missing component, a way to steer the model away from undesirable outputs during training. NASA enables something previously impossible: guidance sampling within a one-step generator. Through this integration, our model achieves an HPSv2 [49] score of 31.21, establishing a new state-of-theart on human preference metrics for one-step models.

In summary, our contributions are twofold:

- • We introduce Negative-Away Steer Attention (NASA), the first method to integrate negative prompt guidance into one-step and few-step diffusion models in both inference and distillation settings.
- • By integrating NASA into the SwiftBrush (SB) distillation framework, we achieve a new state-of-the-art for onestep diffusion models on human preference metrics. This marks the first successful integration of negative prompt guidance directly into one-step model training.

### 2. Related Work

#### 2.1. Distribution Matching Distillation

Recent work [13, 34, 51, 52] has explored text-to-image diffusion distillation through distribution matching techniques. These methods directly align the output distributions of teacher and student models, particularly through Variational Score Distillation (VSD). These techniques can be broadly categorized into two approaches: image-dependent training and image-free training.

Image-dependent methods rely on real images for adversarial loss [42, 51] or synthetic ones generated by the teacher models for reconstruction loss [52]. Despite their effectiveness in preserving image quality, these approaches face significant practical limitations: they require substantial storage capacity, consume high memory during training, demand extensive computational resources, and risk mode collapse due to intensive image supervision.

In contrast, image-free approaches [13, 34] have demonstrated promising results by relying solely on prompt text for training, significantly reducing resource requirements while maintaining competitive performance. However, image-free techniques often encounter training instability, particularly within VSD-based frameworks. Existing solu-

tions to stabilize training [52] introduce significant memory and computational overhead, limiting their practical application. Our work addresses this gap by introducing a simple yet effective method that enhances training stability without additional computational burden, making image-free variational score distillation produce better models.

#### 2.2. Sampling Guidance with Negative Prompts

Classifier-free guidance (CFG) [19] and negative prompting [4, 48] are key techniques for content control during denoising. In multi-step diffusion models, these methods enhance desired image features and suppress undesired ones, improving visual fidelity and reducing artifacts [1, 14, 24]. Score-based approaches like NFSD [24] employ negative prompting to steer generated samples toward real-image distributions, while SDS-Bridge [33] uses negative prompts to model source distributions more accurately. Despite their success, negative prompting has yet to be integrated into one-step or few-step diffusion models. To bridge this gap, we introduce NASA, the first method to enable negative prompting in one-step and few-step diffusion models, enhancing control in faster generation settings.

### 3. Preliminaries

Diffusion Models [20, 40, 44] produce high-quality images by progressively denoising inputs. This technique involves a forward process that gradually adds noise to the data and a reverse process that reconstructs the original data by removing noise. While the original diffusion models act on the image space [20, 44], LDM [40] processes on the latent space produced by a pretrained VAE for more efficient computation. In particular, starting with a data point x0 sampled from an unknown distribution q(x0), the forward process gradually diffuses x0 into a standard Gaussian noise xT ∼ N(0,I) through T consecutive timesteps, where I is the identity matrix. At each time-step t, a noisier version of x0, denoted as xt, is drawn from qt(xt|x0) = N(αtx0,σt2I) using a standard Gaussian noise ϵ ∼ N(0,I):

xt = αtx0 + σtϵ, ∀t ∈ {0,...,T}, (1)

where {(αt,σt)}Tt=1 defines the noise schedule, with boundary conditions (α0,σ0) = (1,0) for the clean sample and (αT,σT) = (0,1) for pure noise.

Conversely, the reverse process aims to gradually reconstruct the original data by denoising the input over T steps, starting from an initial random Gaussian noise xT ∼ N(0,I). The model is trained by minimizing the difference between the noise estimated by the model, ϵϕ, parameterized by ϕ, and the actual noise in Eq. (1):

Et∼U(0,T),ϵ∼N(0,I)∥ϵϕ(xt,t) − ϵ∥22. (2)

min

ϕ

Text-to-Image Diffusion Models guide the sampling process using an additional text prompt y to produce outputs

that are both photorealistic and aligned with the provided textual descriptions. The training objective, slightly modified from the unconditional loss in Eq. (2), is defined as:

Et∼U(0,T),ϵ∼N(0,I),y∥ϵϕ(xt,t,y) − ϵ∥22. (3)

min

ϕ

Classifier-free Guidance (CFG) [19] is an inference technique designed to enhance the quality of generated images by blending the predictions from both a conditional and an unconditional model. At each sampling step, CFG adjusts the denoiser’s output using a control parameter κ > 1, allowing controlled guidance that aligns more closely with the desired conditions:

ˆϵϕ(xt,t,y,κ) = (1 − κ)ϵϕ(xt,t) + κϵϕ(xt,t,y). (4)

Negative Prompting [4, 48] provides enhanced control by suppressing unwanted features in the generated content. Instead of producing an unconditional output, the model generates an output conditioned on the negative prompt y−, as follows:

ˆϵϕ(xt,t,y,y−,κ) = (1−κ)ϵϕ(xt,t,y−)+κϵϕ(xt,t,y).

(5) Variational Score Distillation (VSD) is a powerful framework that utilizes pretrained diffusion-based text-to-image models to enhance text-based generation across both 3D [46] and 2D [13, 34, 51, 52] domains. The central aim of VSD is to align the renderings of a differentiable generator with the probability density of plausible images as guided by the 2D diffusion model. To accomplish this, VSD employs a two-teacher approach that uses a fixed pretrained diffusion model ϵψ and an adaptive LoRA-based teacher ϵπ. While training, the student model fθ produces 2D images xˆ0 = fθ(z,y) using an input noise z ∼ N(0,I) and a text prompt y. Noisy images xˆt = αtxˆ0 + σtϵ are then fed into both teacher models. The LoRA teacher model ϵπ aligns with the student distribution by minimizing a denoising L2 loss on single-step samples. This arrangement supports robust and adaptive guidance suitable for a variety of generative architectures. The gradient of the learning objective with respect to the student’s parameters θ is defined as:

∇θLVSD = Et,y,z,κ=C w(t)(ˆϵψ(xˆt,t,y,κ)

∂fθ(z,y) ∂θ

−ˆϵπ(xˆt,t,y,κ)

,

(6)

where C is a constant and w(t) is the time-dependent weight that adjusts the contribution of each timestep, aligning the student’s outputs with the diffusion model’s predictions. Through alternating updates of the student and LoRA teacher, VSD enables efficient, high-fidelity generation across both text-to-image and text-to-3D tasks.

step methods effectively support negative prompting. Our proposed method fills this gap by providing a simple, efficient, and training-free mechanism that integrates negative prompt into one-step diffusion models to (1) improve controllability and (2) enhance the quality of the output image.

###### NP Anime Photo CA Paintings Average

29.62 29.17 28.79 28.69 29.07 ✓ 31.66 29.17 30.58 30.38 30.42

- Table 1. The effect of negative prompts (NP) on the HPSv2 score of PixArt-α [9]. Using generic negative prompts to remove visual artifacts (e.g., “worst quality”, “blurry”, “bad anatomy”) leads to a significant improvement. Higher scores indicate better alignment with human preferences.

#### 4.2. Challenges

A widely used method for implementing negative prompting is the CFG mechanism. However, applying CFG directly to one-step diffusion models presents significant challenges. CFG was designed for multi-step models, where the output is iteratively refined, allowing for gradual adjustments using both negative and unconditional outputs. In contrast, one-step models generate images in a single pass without the iterative denoising process, making them inherently unsuitable for this method. Direct application of CFG to these models leads to undesirable image blending, resulting in unnatural artifacts, as shown in Fig. 2. This highlights a crucial limitation: the CFG mechanism, though effective in multi-step models, is not directly translatable to one-step models without significant modifications.

###### CFG NASA

A photo of livestock in a farm

###### ❌

— cow

[Figure 19]

[Figure 20]

[Figure 21]

- Figure 2. The inherent single-pass nature of one-step diffusion models renders them incompatible with direct CFG, resulting in artifact-laden outputs.

### 5. Proposed Method

### 4. Observations

Our proposed methodology is detailed in Sec. 5.1 for the core mechanism. We present two primary implementation strategies for NASA: a training-free approach (NASA–I) and a distillation-based approach (NASA–T), which are presented in Sec. 5.2 and Sec. 5.3, respectively.

#### 4.1. Motivation

Positive prompt is not enough. While vision generative models like diffusion models excel at incorporating positive attributes, they fundamentally struggle with understanding negation. Consider requesting “A portrait of a friendly monster, not scary” (Fig. 1), the model typically fails to exclude the specified negative attribute “scary”. Negative prompting addresses this limitation by creating a separate, dedicated input for exclusions – completely removing negation terms from the positive prompt and processing them independently. This clear separation allows the model to accurately distinguish what to avoid generating, providing significantly better control over the final output.

#### 5.1. NASA: Negative-Away Steer Attention

In contrast to CFG implementations that operate within the output space (synthesized image), as commonly practiced in multi-step diffusion models, we propose a novel approach that strategically relocates the guidance mechanism to the intermediate representation space of the diffusion model.

Specifically, cross-attention layers in these models capture semantic connections between image and text features [17], making them ideal for controlling the feature alignment. We hypothesize that cross-attention layer features are particularly well-suited for this approach. We selectively manipulate these cross-attention maps to attenuate attention responses associated with negative prompts relative to those elicited by positive prompts, thereby filtering out unwanted semantic attributes.

Community Need. The growing use of negative prompting in the generative AI community further highlights its importance. Platforms like Civit AI [10] and various open-source repositories [2, 11, 29] have shown that using both types of prompts together results in more refined, higher-quality model outputs that align more closely with user specifications. Users can clearly specify both desired and unwanted attributes in a prompt pair, which is particularly useful when working with models that generate complex or highly detailed content. Experimentally, Table 1 and Figure 1 confirm that using negative prompts effectively enhances image quality of the model.

This representation-space guidance strategy offers a crucial advantage: it enables the exclusion of undesirable elements prior to the final image synthesis stage. This removal mechanism significantly mitigates the risk of generating blending artifacts, which are common byproducts of image-space manipulation techniques.

Critical Gap. Despite the established need and widespread adoption of negative prompts in multi-step diffusion models and the rapidly growing prominence of one-step approaches, a significant limitation persists: no existing one-

Our representation-space guidance also extends to architectures like FLUX [25], which replace traditional crossattention with joint self-attention blocks. [47] show that

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Positive Text Features

Negative-Away Steer Attention

| | | | | | |
|---|---|---|---|---|---|

Fox in a suit

NAS Aegative-wayteerttention

Text Encoder

— Blurry, pixelated, low resolution

[Figure 22]

[Figure 23]

| | | | | | |
|---|---|---|---|---|---|

[Figure 24]

[Figure 25]

[Figure 26]

Negative Text Features

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Gaussian Noise

Output Image

One-step Diffusion

- Figure 3. Left: An overview of the Negative-Away Steer Attention (NASA) pipeline. Positive (green) and negative (red) prompts are fed into a text encoder to generate positive and negative text features. The NASA module then processes these features, which adjusts the one-step diffusion model to steer the output image away from the negative features, refining it based on the positive features. Right: The

details of the NASA module. It processes queries (Ql) in layer l, note we will omit the subscript l in subsequent notations to improve readability, with positive (V+, K+) and negative (V−, K−) key-value pairs to create positive (Z+) and negative (Z−) attention outputs. The final output (ZNASA) is calculated by subtracting the weighted negative features (Z−) from the positive features (Z+).

#### 5.2. NASA–I: Training-free Setting

FLUX.1-schnell SDXL-LCM SDXL-DMD2 4 steps 1 step 4 steps 1 step 4 steps 1 step

Method

Applicable to One-/Few-Step Models. To assess the effectiveness of NASA–I, we carry out experiments with pretrained one-step and few-step models, such as LCM [31], DMD2 [51] and FLUX.1-schnell [25]. We design a small set of positive and negative prompts and generate 100 images per model, both with and without negative prompts. We then calculate the percentage of images that successfully exclude the feature specified by the negative prompt. For example, given the positive prompt “A photo of a person” and the negative prompt “male”, we measure the percentage of generated images that do not depict a male person. The average results across all prompt sets are summarized in Tab. 2. Our findings show that while traditional CFG is ineffective for one-step and few-step models, NASA–I provides precise control over unwanted features in generated images. Additionally, this capability is also qualitatively demonstrated in Fig. 1, Fig. 4 and Fig. 5.

None 23% 44% 43% - 27% 25% CFG 30% 0% 14% - 25% 0% NASA–I 100% 99% 97% - 100% 100%

- Table 2. Comparison of success rate of unwanted feature removal in generated images. Note that SDXL-LCM [31] does not support one-step inference.

these blocks still embed distinct image-text cross-attention operations. This allows NASA to be seamlessly applied, proving its robustness across different backbones.

The overall pipeline of the NASA mechanism is illustrated in Fig. 3. Given the latent feature representation at a given layer l, denoted as Zl, and positive text features cp extracted from CLIP text encoder, we define the query, key, and value matrices as Ql = ZlWq,l, K+l = cpWk,l, and Vl+ = cpWv,l, where Wq,l, Wk,l, and Wv,l are projection matrices at current layer l. Analogously, for negative text features cn, we use the same projection matrices to define the corresponding key and value matrices for negative prompt conditioning: K−l = cnWk,l and Vl− = cnWv,l. The NASA mechanism is formulated as follows:

Efficiency Considerations. As described in Sec. 5.1, we reuse the projection matrices of the positive prompt for the negative prompt, ensuring that no additional model parameters are introduced and keeping the model size unchanged. Furthermore, applying NASA-I to all one-step and few-step models introduces only a minor computational overhead, increasing FLOPs by just 1.89 %. This represents a significant efficiency improvement over CFG, which doubles the computational cost due to the need for separate forward passes for positive and negative prompts. Consequently, NASA–I enables negative prompt guidance in one-step diffusion models without introducing substantial latency, making it a practical solution for real-time applications.

Ql(K+l )⊤

Z+l = Softmax

Vl+, (7)

√

d

Ql(K−l )⊤

Vl−, (8) ZNASAl = Z+l − α · Z−l , (9)

Z−l = Softmax

√

d

where α is the scale factor to control the degree of negative feature removal.

#### 5.3. NASA–T: Distillation Setting

Given their effectiveness in improving image quality during inference, their potential utility in model distillation is worth exploring. In this section, we introduce a method that integrates the negative prompt into one-step diffusion distillation. While our approach is applicable to various distillation methods, we select SwiftBrush (SB) [13, 34], an image-free method, as the baseline for improvement due to its efficiency. As described in Eq. (6), SB updates the student model using the gradient of the VSD loss. To incorporate the negative prompt y−, we modify the existing formulation by replacing ˆϵψ(xˆt,t,y,κ) (see Eq. (4)) with ˆϵψ(xˆt,t,y,y−,κ) (see Eq. (5)). In this way, the teacher model provides explicit guidance to the student model, helping it learn to steer away from undesirable features. Moreover, we incorporate the NASA mechanism into the student model, enabling it to process negative prompts during training. This allows the student model to better recognize and suppress unnatural artifacts, leading to improved learning and enhanced image quality. Note that with the LoRA teacher model ˆϵπ, we can choose whether to incorporate negative prompts. Empirically, we found that not using them yields better results. Detailed results and discussions are provided in Sec. 6.3.

### 6. Experiments 6.1. Experimental Setup

Metrics and Benchmark. In diffusion model evaluation, Fr´echet Inception Distance (FID) [18] on the zero-shot MSCOCO2014 [27] benchmark has been the standard metric. However, recent studies [8, 39, 51] suggest that FID in this benchmark does not always correlate with actual image quality and adopt human-focused alignment metrics like Human Preference Score v2 (HPSv2) [49] for benchmarking. Our experiments also follow this conclusion, so we choose HPSv2 as the primary metric for evaluating the models. Moreover, due to the absence of negative prompts in the MS-COCO2014 benchmark, we utilize the NegOpt dataset from [35], which contains pairs of positive and negative prompts. From this dataset, we randomly sample 30K prompt pairs to construct a subset for evaluation. Text alignment is then assessed using CLIP scores, where CLIP+ evaluates alignment with the positive prompt and CLIP− evaluates alignment with the negative prompt.

Training Datasets. Following SBv2 [13], we use total of 3.3M prompts from JourneyDB [45] and a subset of LAION [43]1. For negative prompts in the distillation setting (Sec. 5.3), we construct a dataset of 10K negative prompts derived from commonly used terms like “bad photo, duplicate, low resolution, low quality”.

1All datasets were downloaded and evaluated at MovianAI.

Implementation Details. We base our method on SBv2 [13] with our proposed modifications in Sec. 5.3. We utilize three backbones SDv1.5 [40], SDv2.1 [40] and PixArtα [9] as the frozen teachers and LoRA teachers are initialized with rank r = 64 and scaling γ = 128. The learning rates are set to 1e−6 for the student model and 1e−3 for the LoRA teacher, using the AdamW optimizer [30]. All our training is conducted on four NVIDIA A100 40GB GPUs with the total batch size of 64 for the SD models and 32 for PixArt-α. Moreover, [6] found that instead of using a fixed CFG scale κ during training, sampling κ from a uniform distribution within a given range improves training robustness. Based on this insight, we set the teacher models’ CFG scales as follows: κ ∼ U(0.5,4) for SDv1.5 and SDv2.1, and κ ∼ U(0.5,3) for PixArt-α. For NASA–T, in each iteration, we uniformly sample the scale α ∼ U(0,1) to help the student model adapt to different conditions and enhance generalization. For NASA–I, unless explicitly specified, we set α = 0.1 for SDv1.5-based models, α = 0.2 for SDv2.1-based models, and α = 0.5 for PixArt-α-based models. Further details can be found in the Appendix.

#### 6.2. Main Results

Quantitative Results. Table 3 presents a comparison between our approach and prior distilled text-to-image diffusion models. Using our reimplementation of SBv2 [13] as the baseline, NASA–T consistently enhances image quality across all tested backbones. Compared to other methods, our model achieves the highest HPSv2 scores on average, demonstrating its clear superiority in image generation quality. This suggests that integrating complex loss functions like DMD2 [51] may not always be necessary for achieving state-of-the-art results. Moreover, while applying CFG directly degrades one-step model outputs, NASA–I further enhances image quality across all models when using negative prompts like “worst quality, low quality, ugly, duplicate, out of frame, deformed, blurry, bad anatomy, watermark”. Notably, our method on the PixArt-α-based backbone achieves a record-breaking HPSv2 score of 31.21, surpassing all other one-step diffusion models baselines. Regarding the NegOpt dataset, with the support of NASA–I, all models demonstrate a reduction in the CLIP− score while maintaining the CLIP+ score. This indicates our method effectively eliminates unwanted features in generated outputs while preserving desired traits.

Qualitative Results. As discussed in Sec. 5.2, NASA–I is a versatile, training-free method for both one-step and fewstep diffusion models. Figure 4 and Figure 5 present qualitative results of integrating NASA–I into the SDXL-DMD2 [51] and FLUX.1-schnell [25] models, respectively. When using a conditional negative prompt, the generated images retain key semantic features while effectively suppressing unwanted attributes specified by the negative prompt.

Dataset NegOpt HPSv2 Method CLIP+ ↑ CLIP− ↓ Anime ↑ Photo ↑ CA ↑ Paintings ↑ Average ↑

PixArt-α-based backbone PixArt-α [9] [Teacher] 0.35 0.05 29.62 29.17 28.79 28.69 29.07 YOSO [32] 0.36 0.08 28.75 28.06 28.52 28.57 28.48 + NASA–I 0.36 0.06 28.74 28.05 28.56 28.60 28.49 (+0.01) DMD [52] 0.35 0.08 29.31 28.67 28.46 28.41 28.71

- + CFG = 1.5 0.34 0.09 30.02 27.07 28.36 28.07 28.38 (-0.33)

- + CFG = 2.5 0.31 0.13 26.74 23.86 25.13 24.66 25.10 (-3.61)

- + NASA–I 0.35 0.05 29.33 28.71 28.49 28.53 28.77 (+0.06)

- SBv2∗ 0.36 0.09 32.19 29.09 30.39 29.69 30.34

+ NASA–I 0.36 0.06 32.60 29.58 31.09 30.65 30.98 (+0.64) + NASA–T 0.35 0.08 32.33 29.26 30.75 30.10 30.61 (+0.27) + NASA–T + CFG = 1.5 0.34 0.10 29.47 26.50 28.22 27.68 27.97 (-2.37)

+ NASA–T + NASA–I 0.35 0.06 32.66 29.67 31.31 30.71 31.09 (+0.75) + NASA–T + NASA–I (α = 1) 0.35 0.05 32.65 29.65 31.45 31.06 31.21 (+0.87)

- Stable Diffusion 1.5-based backbone

SDv1.5 [40] [Teacher] 0.31 0.06 26.51 27.19 26.06 26.12 26.47 InstaFlow-0.9B [28] 0.33 0.11 26.10 26.62 25.92 25.95 26.15 + NASA–I 0.33 0.10 26.15 26.68 25.93 25.98 26.19 (+0.04) DMD2 [51] 0.31 0.10 26.39 27.00 25.80 25.82 26.25

- + NASA–I 0.31 0.09 26.39 27.02 25.80 25.83 26.26 (+0.01) SBv2∗ 0.32 0.09 27.18 27.58 26.69 26.62 27.02

- + NASA–I 0.32 0.08 27.19 27.59 26.71 26.63 27.03 (+0.01)

- + NASA–T 0.35 0.10 27.64 27.94 27.19 27.03 27.45 (+0.43)

- + NASA–T + NASA–I 0.35 0.09 27.65 27.97 27.18 27.04 27.46 (+0.44)

Stable Diffusion 2.1-based backbone SDv2.1 [40] [Teacher] 0.33 0.06 27.48 26.89 26.86 27.46 27.17 SBv2 [13] 0.34 0.10 27.25 27.62 26.86 26.77 27.13 + NASA–I 0.35 0.08 27.44 27.84 26.91 27.02 27.30 (+0.17) SBv2∗ 0.37 0.10 27.56 27.84 26.97 27.03 27.35

+ NASA–I 0.37 0.08 27.70 28.00 27.12 27.22 27.51 (+0.16) + NASA–T 0.36 0.09 28.00 28.65 27.44 27.26 27.84 (+0.49)

- + NASA–T + NASA–I 0.36 0.07 28.04 28.68 27.51 27.50 27.93 (+0.58)

- Table 3. Quantitative comparisons between our method and previous works. NASA–I refers to our training-free method introduced in Sec. 5.2, while NASA–T denotes our distillation-based approach described in Sec. 5.3. SBv2∗ refers to our reimplementation of SBv2 with a randomly sampled CFG scale κ.

Vincent Van Gogh

— starry night

[Figure 36]

One-step

NASA Few-step

A mystical deer with glowing antlers in a moonlit forest, fantasy art, detailed and enchanting

— dim illumination, dark lighting

[Figure 37]

0 0.2 0.4 0.6 0.8 1.0

promotional art of a very cute disney pixar pikachu round fluffy, very furry character, iconic film character

— simple background

[Figure 38]

0 0.2 0.4 0.6 0.8 1.0

NASA

A cute eldritch horror creature

— scary, terrifying, disturbing, nightmare, evil, demonic, horrific, ugly, repulsive, violent

[Figure 39]

0 0.2 0.4 0.6 0.8 1.0

- Figure 4. Effect of different scale values (0.0 to 1.0) in NASA-I with SDXL-DMD2 [51] for both one-step and few-step settings, illustrating the progressive influence on visual details and composition.

0 0.2 0.4 0.6 0.8 1.0

Goghwithout

Apetisrunningonagarden

atedprompt VincentVan

starrynight

❌

###### VincentVanGogh

❌

-starrynight

LLM-gener

—dog

A mystical deer with glowing antlers in a moonlit forest, fantasy art, detailed and enchanting

— dim illumination, dark lighting

[Figure 47]

[Figure 48]

0 0.2 0.4 0.6 0.8 1.0

Wideshot,apersonisjogging

Apetisrunningonagarden

A polar bear wearing a glowing space suit, standing on an icy asteroid, distant planets in the background

— white and blue background

❌—woman

❌—dog

[Figure 49]

[Figure 50]

[Figure 51]

0 0.2 0.4 0.6 0.8 1.0

A bunny in high ornamented light armor, fluffy fur, foggy, wet, stormy atmosphere, 70mm cinematic, highly detailed

— close-up, zoomed-in, facial focus

[Figure 52]

A mystical deer with glowing antlers in a moonlit forest, fantasy art, detailed and enchanting

— dim illumination, dark lighting

A high-tech pirate captain on the bridge of a sleek starship, holographic map glowing before her

A celestial whale swimming through a galaxy, stars and planets reflected in its skin

[Figure 53]

A majestic ice dragon with crystalline wings

❌ — fire, scales, warm colors

❌ — eye patch, hat, wooden leg

###### ❌

— ocean whale, simple whale

[Figure 54]

[Figure 55]

[Figure 56]

0 0.2 0.4 0.6 0.8 1.0

0 0.2 0.4 0.6 0.8 1.0

A polar bear wearing a glowing space suit, standing on an icy asteroid, distant planets in the background

Few-step

— white and blue background

[Figure 57]

NASA

Vincent Van Gogh

— starry night

0 0.2 0.4 0.6 0.8 1.0

[Figure 58]

A bunny in high ornamented light armor, fluffy fur, foggy, wet, stormy atmosphere, 70mm cinematic, highly detailed

Figure 5. Qualitative results of NASA-I in FLUX.1-schnell [25].

— close-up, zoomed-in, facial focus

[Figure 59]

[Figure 60]

[Figure 61]

Wideshot,apersonisjogging

Apetisrunningonagarden

VBench-Long benchmark

❌—woman

❌—dog

Method

Aesthetic Quality ↑

Imaging Quality ↑

0 0.2 0.4 0.6 0.8 1.0

0 0.2 0.4 0.6 0.8 1.0

[Figure 62]

[Figure 63]

promotional art of a very cute disney pixar pikachu round fluffy, very furry character, iconic film character

Few-step

— simple background

NASA

None 61.98 67.12 NASA–I 63.33 67.36

[Figure 64]

Vincent Van Gogh

— starry night

Figure 6. Qualitative results of NASA-I for text-to-video generation in CausVid [53].

Table 4. CausVid quality metrics.

[Figure 65]

#### 6.3. Ablation Studies

0 0.2 0.4 0.6 0.8 1.0

NASA ✓ ✓ Teacher ✓ ✓ ✓ ✓ LoRA Teacher ✓ ✓

0 0.2 0.4 0.6 0.8 1.0

Table 5 presents an ablation on integrating negative prompts during model distillation with the PixArt-α [9] backbone. The baseline is our reimplementation of SBv2 with a randomly sampled CFG scale κ. Simply applying negative prompts only in the teacher and/or the LoRA teacher models slightly degrades the HPSv2 score, as the student model is not explicitly trained on this negative conditioning. In contrast, incorporating NASA with negative prompts during training allows the student model to better capture and utilize this conditioning, leading to improved performance. Furthermore, we find that integrating negative prompts solely into the teacher model yields better results than applying them to both the teacher and LoRA teacher models.

promotional art of a very cute disney pixar pikachu round fluffy, very furry character, iconic film character

— simple background

[Figure 66]

HPSv2 Avg. ↑ 30.34 29.84 29.80 30.61 29.89

- Table 5. Ablation studies on the integration of negative prompts in model distillation using the PixArt-α backbone.

0 0.2 0.4 0.6 0.8 1.0

31.2

HPSv2average

31.0

30.8

Using our PixArt-α-based distilled model as the base, we varied the NASA scale α and evaluated the results with HPSv2 on the benchmark dataset. Figure 7 demonstrates that NASA–I generally enhances image synthesis, resulting in outputs that better align with human preferences. However, if the NASA scale α is set too high, the negative attributes may overly dominate the positive attributes, adversely affecting the final image quality.

30.6

0.00 0.25 0.50 0.75 1.00 1.25 1.50

(NASA scale)

- Figure 7. Ablation study on the NASA scale using our distilled model based on the PixArt-α backbone.

Application to Text-to-Video Models. To demonstrate the versatility of our approach, we extended NASA-I to a fewstep video diffusion model, CausVid [53]. As shown in Fig. 6, our method effectively suppresses unwanted visual attributes while preserving key semantic content and temporal consistency. Quantitative evaluation on the VBenchLong [22] benchmark further validates this. Table 4 shows that using NASA–I with common negative prompts like “low quality, bad anatomy, blurry, distortion, ugly, low resolution, unclear details” improves both the aesthetic and imaging quality scores of the generated videos. This highlights NASA–I’s potential as a general-purpose tool for enhancing guided generation across different modalities.

### 7. Discussion and Conclusion

This paper introduces NASA, the first method to integrate negative prompt guidance into one-step and few-step diffusion models for both inference and distillation. By leveraging cross-attention adjustments, NASA effectively suppresses unwanted features in generated images. Experimental results show that NASA enhances image quality and outperforms strong baselines, establishing a new stateof-the-art in one-step model distillation based on human preference metrics. While NASA’s effectiveness relies on selecting an appropriate scale for suppressing negative features, this parameter does not require extensive tuning.

### References

- [1] Mohammadreza Armandpour, Ali Sadeghian, Huangjie Zheng, Amir Sadeghian, and Mingyuan Zhou. Re-imagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond. ArXiv, abs/2304.04968, 2023. 3
- [2] AUTOMATIC1111. Automatic1111/stable-diffusion-webui,

2025. 4

- [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 1
- [4] Yuanhao Ban, Ruochen Wang, Tianyi Zhou, Minhao Cheng, Boqing Gong, and Cho-Jui Hsieh. Understanding the impact of negative prompts: When and how do they take effect? In european conference on computer vision, pages 190–206. Springer, 2024. 2, 3
- [5] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation. ArXiv, abs/2401.12945,

2024. 1

- [6] Clement Chadebec, Onur Tasar, Eyal Benaroche, and Benjamin Aubin. Flash diffusion: Accelerating any conditional diffusion model for few steps image generation. arXiv preprint arXiv:2406.02347, 2024. 6
- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James T. Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. ArXiv, abs/2310.00426, 2023. 1
- [8] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv: 2403.04692, 2024. 6
- [9] Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 4, 6, 7, 8
- [10] Civitai Platform. https://civitai.com. 4
- [11] comfyanonymous. comfyanonymous/comfyui, 2025. 4
- [12] Quan Dao, Hao Phung, Trung Tuan Dao, Dimitris N Metaxas, and Anh Tran. Self-corrected flow distillation for consistent one-step and few-step image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2654–2662, 2025. 2
- [13] Trung Dao, Thuan Hoang Nguyen, Thanh Le, Duc Vu, Khoi Nguyen, Cuong Pham, and Anh Tran. Swiftbrush v2: Make

- your one-step diffusion model better than its teacher. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. 2, 3, 6, 7, 11
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021. 3
- [15] Zach Evans, CJ Carr, Josiah Taylor, Scott H. Hawley, and Jordi Pons. Fast timing-conditioned latent audio diffusion. ArXiv, abs/2402.04825, 2024. 1
- [16] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 1
- [17] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control.(2022). In International Conference on Learning Representations, 2023. 4
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Neural Information Processing Systems, 2017. 6
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1, 2, 3
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems, 33, 2020. 3
- [21] Rongjie Huang, Jia-Bin Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiaoyue Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. ArXiv, abs/2301.12661, 2023. 1
- [22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 8
- [23] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1
- [24] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. Noise-free score distillation. arXiv preprint arXiv: 2310.17590, 2023. 3
- [25] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 4, 5, 6, 8
- [26] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023. 1
- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In

- European conference on computer vision, pages 740–755. Springer, 2014. 6
- [28] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation. arXiv preprint arXiv:2309.06380, 2023. 7
- [29] lllyasviel. lllyasviel/stable-diffusion-webui-forge, 2025. 4
- [30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6
- [31] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv: 2310.04378, 2023. 5
- [32] Yihong Luo, Xiaolong Chen, Xinghua Qu, and Jing Tang. You only sample once: Taming one-step text-to-image synthesis by self-cooperative diffusion gans. arXiv preprint arXiv: 2403.12931, 2024. 7
- [33] David McAllister, Songwei Ge, Jia-Bin Huang, David W. Jacobs, Alexei A. Efros, Aleksander Holynski, and Angjoo Kanazawa. Rethinking score distillation as a bridge between image distributions. ArXiv, abs/2406.09417, 2024. 3
- [34] Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3, 6
- [35] Michael Ogezi and Ning Shi. Optimizing negative prompts for enhanced aesthetics and fidelity in text-to-image generation. arXiv preprint arXiv:2403.07605, 2024. 6
- [36] Chau Pham, Quan Dao, Mahesh Bhosale, Yunjie Tian, Dimitris Metaxas, and David Doermann. AutoEdit: Automatic Hyperparameter Tuning for Image Editing. arXiv e-prints, art. arXiv:2509.15031, 2025. 1
- [37] Dustin Podell, Zion English, Kyle Lacey, A. Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. International Conference on Learning Representations, 2023. 1
- [38] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. ArXiv, abs/2209.14988, 2022. 1
- [39] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. arXiv preprint arXiv: 2404.13686, 2024. 6
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 3, 6, 7
- [41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022. 1

- [42] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023. 2
- [43] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 6
- [44] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 3
- [45] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Limin Wang, and Hongsheng Li. JourneyDB: A benchmark for generative image understanding. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track,

2023. 6

- [46] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024. 1, 3
- [47] Tianyi Wei, Dongdong Chen, Yifan Zhou, and Xingang Pan. Enhancing mmdit-based text-to-image models for similar subject generation. arXiv preprint arXiv:2411.18301, 2024. 4
- [48] Max Woolf. Stable diffusion 2.0 and the importance of negative prompts for good results. https://minimaxir. com/2022/11/stable-diffusion-negativeprompt/, 2023. 2, 3
- [49] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. ArXiv, abs/2306.09341, 2023. 2, 6
- [50] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023. 1
- [51] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024. 2, 3, 5, 6, 7
- [52] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fr´edo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024. 2, 3, 7
- [53] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. 2025. 8

## Supercharged One-step Text-to-Image Diffusion Models with Negative Prompts Supplementary Material

### 8. Implementation Details

NASA. We provide the PyTorch pseudo-code for the NASA algorithm, as outlined in Section 5.1 in the main paper. The implementation is straightforward and is provided in Algorithm 1.

NASA–T. Following SBv2 [13], we also use the clamped CLIP loss with a margin of τ = 0.37, starting with a weight of 0.1 and gradually reducing to zero. Table 6 provides the additional hyperparameters for training SBv2 with NASA–T.

Hyperparameter SDv1.5 SDv2.1 PixArt-α

Dataset JDB + LAION JDB + LAION JDB Batch size 64 64 32

Training iterations 60k 40k 50k

Mixed-Precision (BF16) Yes Yes Yes κ U(0.5, 4) U(0.5, 4) U(0.5, 3) α U(0, 1) U(0, 1) U(0, 1)

Clip weight 0.1 0.1 0.1

τ 0.37 0.37 0.37

lr of student 1e−6 1e−6 1e−6 lr of LoRA teacher 1e−3 1e−3 1e−3

LoRA rank r 64 64 64 LoRA scaling γ 128 128 128

Table 6. Hyperparameters used for training SBv2 [13] with NASA–T.

### 9. More Qualitative Results

- Figure 8 and Figure 9 illustrate the visual effect of different scale values α ranging from 0.0 to 1.0 in the NASA method when being applied on SDXL-DMD2. A higher scale value α results in more effective removal of the feature specified by the negative prompt.

##### Algorithm 1 NASA.

class NASA_AttnProcessor(nn.Module):

def __init__(self, nasa_scale=1.0): super().__init__() self.nasa_scale = nasa_scale

def __call__(

self, attn, z, emb, neg_emb, attn_mask, neg_attn_mask, temb, *args, **kwargs, ):

# Input preparation residual = z if attn.spatial_norm is not None:

z = attn.spatial_norm(z, temb)

input_ndim = z.ndim if input_ndim == 4:

bsz, channel, height, width = z.shape z = z.view(bsz, channel, height * width).transpose(1, 2)

bsz, sequence_length, _ = z.shape if emb is None else emb.shape _, neg_sequence_length, _ = z.shape if neg_emb is None else neg_emb.shape

if attn_mask is not None: attn_mask = attn.prepare_attention_mask(attn_mask, sequence_length, bsz) attn_mask = attn_mask.view(bsz, attn.heads, -1, attn_mask.shape[-1])

if neg_attn_mask is not None: neg_attn_mask = attn.prepare_attention_mask(neg_attn_mask, neg_sequence_length, bsz) neg_attn_mask = neg_attn_mask.view(bsz, attn.heads, -1, neg_attn_mask.shape[-1])

if attn.group_norm is not None:

z = attn.group_norm(z.transpose(1, 2)).transpose(1, 2)

query = attn.to_q(z) if emb is None:

emb = z else:

if attn.norm_cross: emb = attn.norm_emb(emb) neg_emb = attn.norm_emb(neg_emb)

# Compute cross-attention for positive embedding key = attn.to_k(emb) value = attn.to_v(emb) inner_dim = key.shape[-1] head_dim = inner_dim // attn.heads query = query.view(bsz, -1, attn.heads, head_dim).transpose(1, 2) key = key.view(bsz, -1, attn.heads, head_dim).transpose(1, 2) value = value.view(bsz, -1, attn.heads, head_dim).transpose(1, 2) if attn.norm_q is not None:

query = attn.norm_q(query) if attn.norm_k is not None:

key = attn.norm_k(key) z = F.scaled_dot_product_attention( query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False

) z = z.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim) z = z.to(query.dtype)

# Compute cross-attention for negative embedding

neg_key = attn.to_k(neg_emb) neg_value = attn.to_v(neg_emb) neg_key = neg_key.view(bsz, -1, attn.heads, head_dim).transpose(1, 2) neg_value = neg_value.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)

neg_z = F.scaled_dot_product_attention( query, neg_key, neg_value, attn_mask=neg_attn_mask, dropout_p=0.0, is_causal=False

) neg_z = neg_z.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim) neg_z = neg_z.to(query.dtype)

# Equation 10 in main paper

z_nasa = z - self.nasa_scale * neg_z

- z_nasa = attn.to_out[0](z_nasa)

- z_nasa = attn.to_out[1](z_nasa)

if input_ndim == 4:

z_nasa= z_nasa.transpose(-1, -2).reshape(bsz, channel, height, width) if attn.residual_connection:

z_nasa = z_nasa + residual z_nasa = z_nasa / attn.rescale_output_factor return z_nasa

A mystical deer with glowing antlers in a moonlit forest, fantasy art, detailed and enchanting

— dim illumination, dark lighting

[Figure 67]

Few-step

NASA

0 0.2 0.4 0.6 0.8 1.0

A polar bear wearing a glowing space suit, standing on an icy asteroid, distant planets in the background

— white and blue background

[Figure 68]

Vincent Van Gogh

— starry night

[Figure 69]

0 0.2 0.4 0.6 0.8 1.0

A bunny in high ornamented light armor, fluffy fur, foggy, wet, stormy atmosphere, 70mm cinematic, highly detailed

0 0.2 0.4 0.6 0.8 1.0

— close-up, zoomed-in, facial focus

[Figure 70]

promotional art of a very cute disney pixar pikachu round fluffy, very furry character, iconic film character

— simple background

[Figure 71]

0 0.2 0.4 0.6 0.8 1.0

Figure 8. Additional qualitative images of applying NASA–I to SDXL-DMD2 1-step.

0 0.2 0.4 0.6 0.8 1.0 A cute eldritch horror creature

— scary, terrifying, disturbing, nightmare, evil, demonic, horrific, ugly, repulsive, violent

[Figure 72]

0 0.2 0.4 0.6 0.8 1.0

A mystical deer with glowing antlers in a moonlit forest, fantasy art, detailed and enchanting

— dim illumination, dark lighting

[Figure 73]

0 0.2 0.4 0.6 0.8 1.0

A polar bear wearing a glowing space suit, standing on an icy asteroid, distant planets in the background

— white and blue background

[Figure 74]

0 0.2 0.4 0.6 0.8 1.0

A bunny in high ornamented light armor, fluffy fur, foggy, wet, stormy atmosphere, 70mm cinematic, highly detailed

— close-up, zoomed-in, facial focus

[Figure 75]

An underwater world teeming with bioluminescent sea creatures, including glowing jellyfish, neon fish, and a coral reef illuminated by their light, with a diver exploring the scene

— daylight, non-glowing creatures, above-water elements, plain fish

[Figure 76]

0 0.2 0.4 0.6 0.8 1.0

Few-step

NASA

0 0.2 0.4 0.6 0.8 1.0

A grand steampunk airship with steam-powered engines, soaring through a sky filled with fluffy clouds, setting sun

— modern aircraft, clear skies, nighttime, futuristic designs

[Figure 77]

Vincent Van Gogh

— starry night

[Figure 78]

0 0.2 0.4 0.6 0.8 1.0

A cute eldritch horror creature

0 0.2 0.4 0.6 0.8 1.0

— scary, terrifying, disturbing, nightmare, evil, demonic, horrific, ugly, repulsive, violent

[Figure 79]

promotional art of a very cute disney pixar pikachu round fluffy, very furry character, iconic film character

— simple background

[Figure 80]

0 0.2 0.4 0.6 0.8 1.0

Figure 9. Additional qualitative images of applying NASA–I to SDXL-DMD2 4-step.

0 0.2 0.4 0.6 0.8 1.0

