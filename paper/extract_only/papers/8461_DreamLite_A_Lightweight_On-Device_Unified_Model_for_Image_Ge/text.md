# arXiv:2603.28713v1[cs.CV]30Mar2026

[Figure 1]

## : A Lightweight On-Device Unified Model for Image Generation and Editing

[Figure 2]

### Kailai Feng, Yuxiang Wei†, Bo Chen, Yang Pan, Hu Ye, Songwei Liu, Chenqian Yan, Yuan Gao†

Intelligent Creation Lab, ByteDance †Corresponding Author

### Abstract

Diffusion models have made significant progress in both text-to-image (T2I) generation and textguided image editing. However, these models are typically built with billions of parameters, leading to high latency and increased deployment challenges. While on-device diffusion models improve efficiency, they largely focus on T2I generation and lack support for image editing. In this paper, we propose DreamLite, a compact unified on-device diffusion model (0.39B) that supports both T2I generation and text-guided image editing within a single network. DreamLite is built on a pruned mobile U-Net backbone and unifies conditioning through in-context spatial concatenation in the latent space. It concatenates images horizontally as input, using a (target | blank) configuration for generation tasks and (target | source) for editing tasks. To stabilize the training of this compact model, we introduce a task-progressive joint pretraining strategy that sequentially targets T2I, editing, and joint tasks. After high-quality SFT and reinforcement learning, DreamLite achieves GenEval (0.72) for image generation and ImgEdit (4.11) for image editing, outperforming existing on-device models and remaining competitive with several server-side models. By employing step distillation, we further reduce denoising processing to just 4 steps, enabling our DreamLite could generate or edit a 1024 × 1024 image in less than 1s on a Xiaomi 14 smartphone. To the best of our knowledge, DreamLite is the first unified on-device diffusion model that supports both image generation and image editing.

Date: March 30, 2026 Project Page: https://carlofkl.github.io/dreamlite/

### 1 Introduction

Recent advancements in large-scale diffusion models, such as FLUX series [23, 25], HunyuanImage 3.0 [7], Qwen Image [51] and Seedream series [12, 15, 44] have achieved remarkable progress in both text-to-image (T2I) generation and text-guided editing (I2I). Despite their superior semantic alignment and visual fidelity, these models typically rely on massive backbones with billions of parameters and iterative denoising processes. For instance, FLUX [25] scales its DiT backbone to 12B parameters, imposing prohibitive memory requirements and high inference latency that preclude efficient deployment on consumer-grade devices. To enhance efficiency, recent research has explored lightweight architectures such as SANA [55], DeepGen1.0 [50] and VIBE [1], which typically utilize backbones on the order of ∼2B parameters. However, achieving stable, real-time performance with these models on mobile hardware remains a significant challenge.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Vast mountain valley at sunrise, emerald forests and waterfalls cascading into mist, golden rays of morning light breaking through clouds, nature concept art ...

Change the season into autumn.

A risograph illustration of a cyborg woman, illustrated with ghosted vibrant offsets ... Blue and Yellow overlays to enhance the feeling of shifting perspectives

[Figure 7]

[Figure 8]

Transfer this robot-woman's into realistic style.

[Figure 9]

[Figure 10]

Cute 3D icon of a pastel mint green cactus with subtle smiling features, designed in clay texture. Clean composition with no branding. Replace the cactus with a sunflower.

Whimsical children’s storybook illustration ... a charming horse, pieced from antique newspaper clippings in a patchwork style ... a fluffy black cat with enormous yellow eyes ... a coral-hued bird, small and luminous ... moon hangs ...

[Figure 11]

Remove the black cat on the horse.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A loose, gestural oil painting of a candid moment inside a dimly lit modern apartment kitchen. A person cooks...with soft, moody lighting casting deep shadows across walls...

Turn this image into a pixel art style.

[Figure 16]

[Figure 17]

Ultra-realistic cinematic portrait of a woman on a rooftop at twilight, warm city lights behind her. 50 mm lens, shallow depth of field ... Let the womarn wear a red scarf.

[Figure 18]

[Figure 19]

A portrait of an hyper realistic and cute octopus sitting in a studio, He is also wearing a top hat. Make this octopus laugh.

[Figure 20]

[Figure 21]

Moroccan man, portrait, dynamic pose, detailed hair texture, ornate, sharp focus, in the style of National Geographic, photoportrait, Cinematic. Let the man be sad.

A close-up, high-detail portrait of a fluffy white cute cat against a solid black background. Let the cat wear a sun-glasses.

###### Figure 1 Generation (left) and Editing (right) examples of DreamLite.

To improve accessibility, several works [17, 18, 27, 62] focus on deploying compact diffusion models directly on mobile devices. For example, SnapFusion [27], Mobile Diffusion [62] and SnapGen [17] leverage lightweight U-Net backbones to achieve quality–efficiency trade-offs for on-device T2I generation. More recently, SnapGen++ [18] has explored the potential of Diffusion Transformer for efficient mobile generation. However, these approaches predominantly focus on T2I generation and lack support for image editing. In practice, creators demand a unified experience that seamlessly integrates “generate” and “edit” functionalities within a single application. Furthermore, deploying two separate models significantly increases system complexity and resource consumption, particularly in memory-constrained devices.

In this paper, we introduce DreamLite, a unified and compact diffusion model capable of performing both image generation and editing within a single network. Following SnapGen [17], we adopt a pruned UNet backbone and extend it for multi-task learning via an in-context conditioning mechanism. Specifically, we spatially concatenate the target and condition images (left-right) at the input level. For generation, the target is paired with a blank image; for editing, it is paired with the source image. To resolve task ambiguity, we prepend explicit task tokens (i.e., [Generate] or [Edit]) to the text prompts. This design enable effective task routing within a shared parameter space without introducing additional parameters or specialized branches.

Training the compact model with a unified scheme is challenging due to its limited capacity and the divergent optimization objectives of generation and editing tasks. To ensure stability, we propose a task-progressive joint pretraining scheme. Specifically, we introduce an intermediate editing pretraining stage between text-to-image pretraining and joint training. This stage aligns visual condition with the generative latent space prior to complex unified joint optimization, thereby facilitating stable training. Consequently, the pretraining of DreamLite is divided into three progressive stages: i.e., T2I Pretraining → Editing Training → Unified Joint Training. Following this, we further adopt a two-stage post-training strategy: supervised fine-tuning (SFT) on a curated high-quality dataset, followed by reinforcement learning (RL) for preference alignment. During RL training, we employ HPSv3 [36] as the reward model for generation and EditReward [53] for editing, optimizing the diffusion model via ReFL [56]. This post-training phase consistently enhances both perceptual quality and instruction following, enabling our compact model to outperform prior on-device diffusion baselines.

Overall, DreamLite achieves GenEval (0.72) and DPG (85.8) for text-to-image generation, along with ImgEdit (4.11) and GEdit (6.88) for text-guided image editing. It outperforms specialized lightweight baselines such as SnapGen, SANA-0.6B and VIBE while remaining competitive with larger unified models such as OmniGen2 and Bagel. These results validate the efficacy of our unified architecture. To minimize deployment overhead, we apply DMD2 [59] to compress the sampling process into 4 denoising steps. On a representative smartphone (i.e., Xiaomi 14), DreamLite completes a 1024 × 1024 generation or editing task in less than 1 second. To the best of our knowledge, DreamLite is the first unified on-device diffusion model to support both generation and editing within a single network for mobile deployment.

Our contributions are summarized as follows:

- • We propose, to the best of our knowledge, the first unified on-device model that supports both text-toimage generation and text-based image editing, eliminating the need to deploy two separate models.
- • We introduce an in-context conditioning mechanism for UNet to unify generation and editing, and propose a task-progressive joint pretraining scheme (i.e., T2I → Edit → Unified Joint Training) to stably train the model.
- • DreamLite achieves competitive performance on standard benchmarks and consistently outperforms prior mobile models. After deployment on Xiaomi 14, DreamLite could generate or edit a 1024 × 1024 image in less than 1s.

### 2 Related Work

#### 2.1 Unified Generative Models

Large-scale image models are increasingly built as unified generative systems that support both text-to-image generation and instruction-based editing through a single model. Recent model families such as FLUX 2 [24], HunyuanImage [7], Seedream 4.0 [44], Qwen-Image-2 [51], as well as commercial systems like Gemini-Image [38], GPT-Image [16], LongCat-Image [48] and DeepGen [50], all move toward “generate + edit” as first-class capabilities, typically by strengthening text–image alignment and instruction following with large backbones and curated post-training. A representative line is FLUX.2, which frames unified generation and editing via an in-context formulation. Compared to these large or cloud-oriented unified systems, our work targets a single compact diffusion model that supports both generation and instruction-based editing for on-device deployment under strict memory/latency constraints.

#### 2.2 Efficient Diffusion Models

A growing body of work improves diffusion efficiency by optimizing architectures, attention mechanisms, and training/inference recipes. PixArt-Σ [8] explores transformer efficiency for high-resolution generation with key-value compression to alleviate attention cost at large token counts. And SANA [55] proposes linear-attention diffusion transformers to reduce the quadratic complexity of self-attention at high resolutions while maintaining competitive generation performance. Beyond pure generation, EditMGT [9] and VIBE [1] presents a compact instruction-based editing pipeline that combines a lightweight vision-language model with an efficient diffusion backbone, demonstrating that strong editing can be achieved without relying on extremely large diffusion models. Our method is complementary to these efforts: rather than only accelerating T2I or only optimizing editing, we focus on an efficient unified interface that consolidates T2I and editing into a single compact model.

#### 2.3 On-Device Generative Models

To enable on-device deployment, prior works have explored quantization, pruning, and knowledge distillation to reduce model size and latency. Early on-device systems [27, 62] pruned and distilled U-Net architectures to generate 512-pixel images within seconds. SnapGen [17] demonstrated that a compact U-Net derived from SDXL can generate 1024 × 1024 images on mobile devices with a carefully engineered architecture and training recipe. More recently, SnapGen++ [18] explores efficient diffusion transformers tailored for mobile and edge deployment. Concurrent to our work, Mobile-O [45] attempts to unify both visual generation and understanding within a single compact framework. However, since Mobile-O relies on an understanding-centric paradigm to execute generation tasks, it struggles with fine-grained visual control and spatial consistency at in editing tasks. Consequently, its performance in complex image editing scenarios remains somewhat suboptimal. Despite this progress, most on-device works primarily emphasize T2I generation, while instruction-based editing often requires either separate models or additional editing-specific components. Our work targets this gap by providing a unified interface and training strategy so that a single compact model supports both “generate an image” and “edit my photo,” reducing deployment complexity and resource consumption on mobile devices.

#### 2.4 RLHF

Post-training alignment has emerged as a crucial stage for enhancing perceptual quality and instruction compliance, surpassing the limitations of noisy web-scale pre-training. A common practice involves utilizing learned reward models (e.g., ImageReward [56], HPSv2/v3 [36, 54], PickScore [22], and the editing-specific EditReward [53]) as optimization targets to improve aesthetics and prompt faithfulness while mitigating visual artifacts. On the optimization front, various methods adapt reinforcement learning (RL) to generative models. For instance, ReFL [56] performs reward-guided fine-tuning for diffusion models under constrained backprop settings to reduce training cost, and has inspired follow-up explorations of more efficient preference optimization. Parallel to RL-based approaches, DPO-like objectives (e.g., Diffusion-DPO [49], AlignProp [40]) have been developed to leverage pairwise preferences without explicit reward modeling. Recent flow-model

Pred Flow

Ref Img

Blank Img Noise

Blank Img

Noise

Qwen3-VL

U-Net

Pred. Flow

Cond. Latent

Noisy Latent

Cond. Latent

ℒ

[Figure 28]

[Figure 29]

[Figure 30]

[Edit]: A diptych with two sideby-side images of the same scene. Compared to the right side, the left one has + Let the woman wear a red scarf and sunglasses.

[Generate]: clean aesthetic portrait of a woman looking sideways, soft daylight, neutral tones, simple background, cinematic minimalism...

or

Gen Target Edit Target Edit Cond.

Blank Image

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Qwen3-VL

Tiny VAE

[Figure 35]

Predict Flow

Cond. Latent

Timestep

Text Embedding

Noise

In-context Width Concatenation

U-Net

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

or ℒfmw

ℒfm

Noisy Latent

Cond. Latent

or

Gen. Pair Edit Pair

Figure 2 Overview of DreamLite architecture. It consists of three primary modules: a UNet backbone, a Variational Autoencoder (VAE), and a text encoder. To support unified generation and editing, an in-context conditioning framework is adopted that unifies both tasks at the input level.

works also explore GRPO variants (e.g., Flow-GRPO [30], DanceGRPO [57]) for better stability and instruction following under preference feedback.

#### 2.5 Step Distillation

Few-step sampling is essential for interactive and on-device applications. A representative line is consistencybased distillation, including Latent Consistency Models (LCM) [33], which distill time-consistent behavior to enable generation in a small number of steps. Another influential family is distribution matching distillation. DMD [60] distills a multi-step teacher into a few-step student via distribution matching objectives, and DMD2 [59] further improves stability and sample quality under aggressive step reduction. In practice, these methods are often used to compress sampling to ≤ 4−8 steps with acceptable quality, and have been integrated into various backbones (including SD, SDXL and QwenImage, etc.). Adversarial-based distillation is also a prominent approach; for instance, SDXL-Turbo and SD-Turbo [42] utilize Adversarial Diffusion Distillation (ADD) to achieve high-fidelity, real-time synthesis in a single step. Recent efforts such as RG-LCD [26] and DI++ [34] incorporate reward model into the distillation process with an additional score model to maintain proximity to the original generator. To address this, LaSRO [21] optimizes arbitrary rewards via latent space exploration. Reward-Instruct [35] aligns generators with rewards without requiring training images. TAFS-GRPO [61] eliminates the need for differentiable reward functions by leveraging a policy gradient algorithm. In this work, we employ the DMD2 to compress our sampling process to 4 steps.

### 3 Method

This section presents the training pipeline and details of our DreamLite. We first describe the model architecture (Section 3.1), and then detail our training procedure, including task-progressive joint pretraining (Section 3.2), post training (Section 3.3), and few-step distillation (Section 3.4).

#### 3.1 Model Architecture

As illustrated in Fig. 2, the architecture of DreamLite consists of three primary modules: a UNet backbone, a Variational Autoencoder (VAE), and a text encoder. To support unified image generation and editing, we further introduce the in-context conditioning mechanism.

Variational Autoencoder. Following [23, 41], we adopt a latent diffusion framework. To enable efficient on-device deployment, we employ an extremely lightweight VAE (i.e., TinyVAE [37]), which contains only

3054

120

###### DreamLite (U-Net) Params (M) FLOPs (G)

(truncated)

2500

2567

100.0%

| |
|---|

ParamsRatiovs.StageA(%)

###### Params Ratio (%)

100

Params(M)/FLOPs(G)

2000

80

1500

60

1085

1042

1000

840

829

40

32.7% 32.3%

749 719

28.0%

657 654

25.5%

608

601 518 383

580

20.2%

500

14.9% 15.2%

390

20

0

0

(A) SDXL Baseline + Text Token Refiner

(B) Pruning (Blocks & Dims)

(C) Remove Self-Attn in High-Res Stages

(D) Expanded Separable Conv.

(E) Trim FFN Layers (ratio: 4 3)

(F) Multi-Query Attention (MQA)

(G) Stage Alignment (Down&Up: 3 2)

(H) QK-RMSNorm, Light Text Proj.

Figure 3 Architectural evolution of DreamLite. We calculate the Parameters (Params) and GFLOPs of the U-Net.

- 2.5M parameters for image tokenization. It maps image x as a 4-channel latent z with an 8 × 8 downsampling factor, facilitating efficient training and inference.

Compact UNet. Efficiency is the primary objective guiding the design of DreamLite. To this end, we build upon the mobile-efficient T2I architecture of SnapGen [17], a systematically compressed version of SDXL [39]. Specifically, we optimized the U-Net backbone by making it both shallower and thinner: the number of transformer blocks was reduced from [0, 2, 10] to [0, 2, 4], and the channel dimensions were shrunk from [320, 640, 1280] to [256, 512, 896] and a latent sample size is 128 × 128. We further enhanced efficiency through several key optimizations:

- • Remove self-attention layers at high-resolution stages to mitigate quadratic complexity;
- • Replace standard convolutions with expanded separable convolutions (i.e., depthwise convolution & pointwise convolution);
- • Set the hidden channel expansion ratio to 3 in the feed-forward network.
- • Adopt Multi-Query Attention (MQA) with a single KV head to reduce both computational overhead and memory footprint;
- • Stage alignment, add QK-RMSNorm and a light text projector.

As summarized in Fig. 3, this step-by-step optimization successfully compresses the 2.5B baseline into a highly efficient 389M parameter backbone, significantly reducing FLOPs while preserving generative performance. Additional architectural details can be found in the original SnapGen paper [17].

Text Encoder. For text conditioning, we utilize Qwen3-VL-2B [3] as our text encoder. We leverage its robust visual-language comprehension capabilities to accurately interpret complex user instructions and process multimodal inputs. This choice ensures precise semantic alignment between the input instructions and the generated content.

In-context Paradigm. Existing UNet-based image editing methods [4, 13, 20] typically follow the InstructPix2Pix paradigm [4], which concatenates the condition image with the noisy latent in the channel dimension and fine-tunes the model. However, this mechanism inevitably degrades the generative priors of the pretrained text-to-image (T2I) model and hinders the development of a unified architecture. To address these limitations, we propose extending the UNet with an in-context conditioning framework that unifies both image generation and editing tasks at the input level within a single compact network. As shown in Fig. 2, we construct a two-panel latent by concatenating the latent of the target image (ztgt) and the conditioning image zcond along the width (spatial) dimension:

zpair = Concat ztgt, zcond , (1) where the left panel corresponds to the target output and the right panel provides the visual condition. The concatenated latent zpair is fed directly into the U-Net. For text-to-image generation, we set the conditioning panel to a blank (all-black) image xblank (with latent zblank) to represent “no visual condition”. For image editing, we use the source image xsrc (latent zsrc) as the condition. This design allows the model to extend

from T2I to editing directly without introducing additional modules, making it highly suitable for a unified framework. To further reduce task ambiguity when training a single model for two behaviors, we prepend explicit task tokens to the text prompt: [Generate] for generation task and [Edit] for editing task. These tokens act as lightweight routing signals without requiring extra parameters or task-specific branches, thereby improving both generation quality and edit controllability under a unified framework. We also compare this in-context formulation with InstructPix2Pix; further details and motivation regarding the architectural design are provided in Section 4.

#### 3.2 Task-progressive Joint Pretraining

Training a compact model with a unified formulation is challenging due to its limited capacity and the divergent optimization objectives of generation and editing tasks. To ensure stable convergence, we propose

- a task-progressive joint pretraining scheme. Unlike standard approaches that transition directly from textto-image pretraining to joint training, we introduce an intermediate editing pretraining stage. This stage serves to align the visual conditioning representations with the generative latent space before the complex joint optimization, thereby mitigating task interference. Specifically, the pretraining of DreamLite is divided into three progressive stages: (i) T2I Pretraining, (ii) Editing Training, and (iii) Unified Joint Training.

##### 3.2.1 Text-to-Image Pretraining

We first train the DreamLite as a standard text-to-image diffusion model using the flow matching objective [29, 32]. During training, the noisy latent zt is constructed through linear interpolation between the initial Gaussian noise ϵ and the clean image latent z, i.e., zt = F(z,t) = t · z + (1 − t) · ϵ and t ∈ [0,1]. The model is then trained to predict the velocity of the vector field that defines the trajectory between the noise and data distributions, i.e., vt = z − ϵ. The training objective can be formulated as:

Lfm = Et,z,ϵ,y ∥vθ(zt,t,y) − (z − ϵ)∥2 , (2)

where vθ denotes the denoising UNet, θ represents the learnable parameters and y denotes the conditional embeddings. To improve convergence and stability, we adopt a progressive resolution curriculum following prior work [17]. The training proceeds sequentially from 256 × 256 to 512 × 512, and finally to 1024 × 1024 resolution, with a multi-scale training strategy applied at each stage. Furthermore, following Stable Diffusion

- 3 [11], we employ a logit-normal noise sampler to concentrate training on intermediate timesteps. We also utilize dynamic time shifting [23] to scale noise levels according to image resolution. Collectively, this stage establishes a strong generative prior for subsequent training stages.

##### 3.2.2 Edit Pretraining

Following T2I pretraining, we activate the in-context conditioning mechanism and continue training the model on paired text-guided image editing data. The primary objective of this stage is to align the newly introduced visual conditioning with the pre-trained generative latent space, enabling the model to generate images based on both condition images and editing instructions. We also employ a flow-matching loss during this phase. However, a major challenge in training for editing tasks is that the target edit regions in the target images are usually small. If a standard uniform loss is applied to the entire image, the gradient signal from these small edited regions can be overwhelmed by the dominant signal from the unchanged background, leading to unsatisfactory editing results. To mitigate this imbalance, following UniWorld-V1 [28], we apply a foreground-emphasis mask to re-weight the loss. Concretely, for localized editing tasks, we derive the edited region mask through a four-step pipeline:

- • Pixel-wise Differencing: we compute the absolute difference between source and target images, applying a tolerance threshold to identify candidate regions;
- • Dilation: a dilation operator is applied to reduce pixel-level noise;
- • Connected Component Filtering: we filter out small, isolated components to eliminate spurious artifacts;
- • Max-pooling Downsampling: max-pooling is utilized to remove internal noise within connected regions and we estimate the edited area size Aedit.

To prevent small edited regions from being overwhelmed by the static background, we assign higher loss weights to the masked pixels based on the area ratio x = Atotal/Aedit, where Atotal denotes the full image area. Subsequently, we use a logarithmic weighting function w(x) = log2(x) + 1 to balance training stability with sensitivity to minor edits. This mask strategy is applied exclusively to local editing; for global editing or style transfer, we maintain uniform weighting to preserve global distribution alignment.

Finally, we compute this weighting mask w to applied to the flow matching objective: Lfmw = Et,z,ϵ,y ∥w ⊙ (vθ(zt,t,y) − w ⊙ (z − ϵ))∥2 . (3)

Intuitively, the mask focuses the learning signal on regions that differ between source and target images, encouraging the model to make stronger, more faithful edits while preserving the unchanged background.

##### 3.2.3 Unified Pretraining

Subsequently, we perform unified joint training on a mixture of T2I and editing data. This stage is designed to consolidate the generative priors established in the T2I stage with the instruction-following capabilities acquired during Edit pretraining, ensuring the model converges to a single set of parameters that reliably supports both behaviors. As stated in Sec. 3.1, to mitigate task conflict in this unified setting, we prepend explicit task tokens to the text prompts: [Generate] to indicate the absence of a visual reference for generation tasks, and [Edit] for editing instructions. These tokens function as lightweight routing signals, guiding the model to switch behaviors dynamically based on the input context. Consequently, DreamLite achieves robust performance in both generation and editing under a strictly unified architecture, eliminating the need for additional parameters.

#### 3.3 Post Training

While our task-progressive joint pretraining establishes a strong foundation for both generation and editing, the model’s behavior remains unstable due to the high variance of the pretraining data. To improve stability and performance, we employ a two-stage post-training strategy: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).

##### 3.3.1 Supervised Fine-Tuning (SFT).

The primary goal of SFT is to refine the model’s behavior by exposing it to a curated distribution of highquality data. To achieve this, we construct a dataset comprising approximately 0.5M samples, selected for their high visual quality and caption diversity. By fine-tuning on these cleaner, denser signals, we effectively steer the model’s target distribution toward a manifold of higher realism and precise instruction following.

##### 3.3.2 Reinforcement Learning (RL).

To further align DreamLite with human preferences, we incorporate Reinforcement Learning from Human Feedback (RLHF). Specifically, we adopt the Reward Feedback Learning (ReFL) framework [56], which directly leverages scalar signals from a pre-trained reward model to guide the denoising trajectory via gradient backpropagation. During training, given a condition c and a timestep t, the model progressively denoises the latent to time t to predict the corresponding clean image xˆ. A scalar reward r(c,xˆ) is then computed using the reward model, and gradients are backpropagated through the denoising process. We adopt a ReLU-truncated reward formulation:

Lrl = −max(0,r(c,xˆ) − b), (4)

where b is a hyperparameter introduced to ensure training stability and mitigate reward hacking. In our experiments, we employ task-specific reward models to optimize distinct capabilities. For text-to-image generation, we utilize HPSv3 [36], a state-of-the-art preference model built upon the Qwen-VL backbone, with

- b = 11. For image editing, we employ EditReward [53], which is explicitly designed to evaluate adherence to editing instructions, and use b = 2.5. This post-training stage consistently enhances both perceptual quality and instruction following, enabling our compact model to outperform prior mobile diffusion baselines.

Category Source / Content Type Samples

- Part A: Generation Data (20.0M)

General Perception COYO, LAION, JoourneyDB, etc.. 14.8M Human & Portrait High-quality human, Portrait subsets 0.4M Graphic Design Text content, Canva subsets 0.4M Scene Text Dense/Large scene text 1.5M Artistic Styles Midjourney style prompts 0.6M Specialized Prompts Object relations, Text rendering 2.4M

- Part B: Editing Data (1.74M)

Understanding Feature Extraction & Parsing, Action 429k Local Edit Add, Remove, Color, Replace, etc.. 830k Global Edit Enhance, Lighting, Background 300k View Edit Camera View, Zoom In/Out, Outpainting 151k Style Edit Unreal ↔ Real 31k

Table 1 Detailed distribution of the 21.7M total samples.

#### 3.4 Step Distillation

While our post-training strategy ensures high-quality generation, it still requires tens of denoising steps to generate/edit an image, creating a significant bottleneck for real-time mobile applications. To bridge the gap between high-fidelity synthesis and low-latency deployment, we apply Distribution Matching Distillation (DMD) [59, 60] to reduce the sampling process to just 4 steps without compromising visual quality. DMD distills a multi-step diffusion teacher into a few-step generator G by minimizing the approximate KullbackLeibler (KL) divergence between the distribution of real images preal,t and the generator’s output distribution pfake,t. The gradient of DMD loss is computed as:

∇LDMD = Et (∇θKL(pfake,t∥preal,t))

= −Et sreal(F(Gθ(ϵ), t), t) − sfake(F(Gθ(ϵ), t), t)

dGθ(ϵ) dθ

dϵ ,

(5)

where ϵ ∼ N(0,I) is a random Gaussian noise, and θ denotes the generator parameters. F defines the forward diffusion process that transfers the generated sample Gθ(ϵ) the noise level corresponding to timestep t. sreal and sfake represent the scores, approximated using diffusion models vreal and vfake, respectively. To stabilize training, we also employ the GAN loss LGAN [59] to enhance the diversity and realism of the generated images for the distilled model. Our distilled model enables high-quality image generation and editing in only

- 4 sampling steps without the need for CFG, showcasing exceptional efficiency.

### 4 Experiments

#### 4.1 Implementation Details

DreamLite is implemented using the PyTorch. The training corpus consists of 20M text-to-image pairs and 1.7M image editing samples. During the unified joint training stage, we employ a sampling ratio of approximately 1:1 between T2I and editing data to balance performance. We utilize the AdamW optimizer with a staged learning rate schedule: 1 × 10−4 for initial T2I pre-training, 1 × 10−5 for editing training and 1 × 10−6 for unified joint training. The batch size is set to 576.

#### 4.2 Dataset Information

Text-to-image Generation Dataset. To establish a robust and diverse generative prior, our text-to-image (T2I) pre-training utilizes a curated corpus of approximately 20M samples. We categorize these data into five primary domains: General Perception, Human & Portrait, Graphic Design, Scene Text, and Artistic Styles. Specifically, we leverage large-scale open-source datasets (e.g., LAION [43], COYO [5], JourneyDB [47]) for general semantic alignment, while incorporating specialized subsets for high-quality human generation and

Model Param. Single Obj. Two Obj. Counting Colors Position Color Attr. Overall

FLUX.1-Dev [23] 12B 0.99 0.81 0.79 0.74 0.20 0.47 0.67 BAGEL [10] 7B 0.99 0.94 0.81 0.88 0.64 0.63 0.82 OmniGen2 [52] 4B 1.00 0.95 0.64 0.88 0.55 0.76 0.80 Z-Image [6] 6B 1.00 0.94 0.78 0.93 0.62 0.77 0.84 LongCat-Image [48] 6B 0.99 0.98 0.86 0.86 0.75 0.73 0.87 DeepGen1.0 [50] 2B 0.99 0.97 0.68 0.90 0.71 0.71 0.83

SANA-1.6B [55] 1.6B 0.99 0.80 0.62 0.89 0.27 0.43 0.67 Hunyuan-DiT [7] 1.5B 0.97 0.77 0.71 0.88 0.13 0.30 0.63 MEISSONIC [2] 1B 0.99 0.66 0.42 0.86 0.10 0.22 0.54

SnapGen [17] 0.38B - - - - - - 0.70 SnapGen++ [18] 0.4B - - - - - - 0.66 SANA-0.6B [55] 0.59B 0.98 0.78 0.59 0.86 0.22 0.40 0.64 Nitro-E-GRPO [46] 0.3B 1.00 0.88 0.63 0.86 0.32 0.48 0.70 Ours 0.39B 0.98 0.88 0.65 0.90 0.35 0.52 0.72

- Table 2 Evaluation on GenEval benchmark. All images are generated at 1024 × 1024 resolution. “Param.” denotes backbone parameters.

complex text rendering. For all samples, we prioritize high-resolution images and apply a stringent filtering pipeline to remove low-quality captions samples. The data composition is summarized in Table 1.

Image Editing Dataset. To empower DreamLite with comprehensive instruction-following capabilities, we curated an image editing corpus comprising approximately 1.7M samples. These samples are categorized into five primary groups:

- • Understanding Edit: A significant portion of the data is for editing-oriented understanding such as feature extraction and human action adjustments (e.g., age, expression, pose).
- • Local Edit: This includes fine-grained modifications such as object addition/removal and color/material changes, etc..
- • Global Edit: This category focuses on overall image enhancements, including lighting direction adjustments, background modifications, etc..
- • View Edit: It involves spatial and structural transformations, such as camera view adjustments, zooming (in/out) and object-level rotation or motion.
- • Style Edit: This one mainly focus on style transfer (unreal-to-real).

The detailed data distribution across these fine-grained categories is presented in Table 1.

#### 4.3 Quantitative Results

To evaluate the performance of DreamLite, we conduct extensive comparative experiments across multiple mainstream benchmarks, focusing on both Image Editing and Image Generation capabilities.

Performance on Image Generation. We first assess the generative quality using GenEval [14] and DPG [19] as the primary benchmarks with all images generated at 1024 × 1024 resolution for a fair comparison. The compared baselines are categorized into three groups:

- • Unified Models (e.g., BAGEL, OmniGen2), which typically feature backbone sizes exceeding 2B parameters;
- • Lightweight Generative Models (< 2B) (e.g., SANA-1.6B, Meissonic), which remain unsuitable for seamless on-device deployment;
- • On-device Generative Models (< 1B) (e.g., SnapGen series, SANA-0.6B), which are restricted to only generation tasks.

Model Param. Global Entity Attribute Relation Other Overall

FLUX.1-Dev [23] 12B 82.1 89.5 88.7 91.1 89.4 84.0 BAGEL [10] 7B 88.9 90.4 91.3 90.8 88.7 85.1 OmniGen2 [52] 4B 88.8 88.8 90.2 89.4 90.3 83.6 Z-Image [6] 6B 93.4 91.2 93.2 92.2 91.5 88.1 LongCat-Image [48] 6B 89.1 92.5 92.0 93.3 87.5 86.6 DeepGen1.0 [50] 2B 84.2 91.0 88.6 94.0 85.3 84.6

SANA-1.6B [55] 1.6B 86.0 91.5 88.9 91.9 90.7 84.8 Hunyuan-DiT [7] 1.5B 84.6 80.6 88.0 74.4 86.4 78.9 MEISSONIC [2] 1B 75.8 74.3 74.4 81.5 57.6 65.3

SnapGen [17] 0.38B - - - - - 81.1 SnapGen++ [18] 0.4B - - - - - 85.2 SANA-0.6B [55] 0.59B 83.0 89.5 89.3 90.1 90.2 83.6 Nitro-E-GRPO [46] 0.3B 82.7 87.6 86.9 92.1 74.9 81.1 Ours 0.39B 84.9 91.6 89.7 94.3 80.4 85.8

- Table 3 Evaluation on DPG benchmark. All images are generated at 1024 × 1024 resolution. “Param.” denotes backbone parameters.

Model Param. Add Remove Replace Adjust BG Style Extract Action Hybrid Overall

Kontext-Dev [25] 12B 4.04 2.92 4.28 3.89 4.00 4.43 2.71 4.52 3.08 3.76 BAGEL [10] 7B 3.81 3.16 3.85 3.59 3.39 4.51 1.58 4.25 2.67 3.42 OmniGen2 [52] 4B 3.57 3.20 3.74 3.06 3.57 4.81 1.77 4.68 2.52 3.44 Z-Image [6] 6B 4.40 4.13 4.57 4.14 4.14 4.85 4.30 4.50 3.63 4.30 LongCat-Image [48] 6B 4.66 4.68 4.75 4.64 4.35 4.81 3.90 4.83 3.80 4.49 DeepGen1.0 [50] 2B 4.52 4.43 4.42 4.09 4.38 4.48 1.75 4.57 3.61 4.03

VIBE [1] 1.6B 3.89 4.42 4.34 4.22 4.22 4.40 2.90 2.75 3.52 3.85 EditMGT [9] 0.96B 2.78 2.89 3.34 3.29 3.05 4.17 1.56 2.94 2.01 2.89

Ours 0.39B 4.59 4.27 4.32 4.46 4.28 4.42 2.70 4.73 3.24 4.11

Table 4 Evaluation on ImgEdit benchmark. “Param.” denotes backbone parameters.

Table 2 and Table 3 details the parameter counts, per-category scores and overall performance on GenEval and DPG benchmark. Note that for the SnapGen series, per-category scores are omitted as the models are not open-sourced. Nevertheless, DreamLite (0.39B) achieves competitive performance, rivaling models with nearly 10× more parameters.

Performance on Image Editing. We evaluate editing proficiency using the ImgEdit [58] and GEdit [31] benchmarks. For ImgEdit, we employ GPT-4o (2024-11-20) as the automated evaluation metric. For GEdit, we focus on English-based editing tasks and utilize Qwen2.5-VL [3] as the model-based judge. Similar to generation task, our method is compared against a broad spectrum of state-of-the-art (SOTA) baselines, categorized into:

- • Unified Models (e.g., FLUX, BAGEL, LongCat-Image);
- • Lightweight Editing Models (< 2B) (i.e., VIBE);
- • On-device Editing Models (< 1B) (i.e., EditMGT).

Crucially, to the best of our knowledge, DreamLite is among the first to successfully enable image editing tasks on-device with a model scale significantly under 0.5B parameters. Table 4 summarizes the model sizes, per-category scores and overall performance on the ImgEdit benchmark. DreamLite achieves SOTA among all lightweight models.

Model Param. QSC QPQ QO

Kontext-Dev [25] 12B 6.61 7.24 6.79 BAGEL [10] 7B 7.66 7.08 7.20 OmniGen2 [52] 4B 7.11 7.22 6.79 LongCat-Image [48] 6B 7.87 7.49 7.55 DeepGen1.0 [50] 2B 7.83 7.47 7.54

VIBE [1] 1.6B 7.61 7.34 7.28 EditMGT [9] 0.96B 6.77 6.74 6.33

Ours 0.39B 7.04 7.54 6.88

Table 5 Evaluation on GEdit-EN benchmark. “Param.” denotes backbone parameters. QSC, QPQ, and QO denote Semantic Consistency, Perceptual Quality and Overall score evaluated by Qwen-VL-2.5 respectively.

#### 4.4 Qualitative Results

To further validate the visual fidelity and instruction-following capability of our DreamLite, we present a qualitative comparison with both large unified models and specialized lightweight baselines.

Visual Analysis of Image Generation. As illustrated in Fig. 4, although DreamLite is significantly smaller than other large-scale models, it maintains high structural integrity and strong semantic alignment across diverse generation scenarios. For instance, in complex realistic prompts such as “A rain-soaked street at dawn...”, DreamLite accurately captures the atmospheric lighting and produces consistent structural details of the “bicycle”, avoiding the distortion often seen in smaller counterparts. Furthermore, in stylized and imaginative generation tasks, ranging from a “Cute isometric 3D icon of a pastel mint green cactus in clay texture” to a “Whimsical children’s storybook illustration” featuring multiple subjects (a horse and an oversized black cat), our model demonstrates a profound understanding of stylistic nuances and complex spatial compositions. Compared to other lightweight models like Nitro-E (0.3B) or Meissonic (1B), our results consistently exhibit superior scene composition, better adherence to stylistic constraints (e.g., clay animation-style or stylized digital blooms), and significantly fewer artifacts in high-resolution synthesis. While there might be a marginal gap in rendering extremely fine-grained textures in human portraits when compared to larger specialized models like SANA-1.6B, our model provides a more robust and coherent interpretation of environmental lighting, mood, and complex multi-object interactions (e.g., “A small French bulldog puppy sitting at a café table... looking at a croissant”).

Visual Analysis of Image Editing. The image editing comparisons in Fig. 5 further highlight the performance of DreamLite. For instance, in object addition and removal tasks (e.g., “Place a bicycle in the foreground of the flower field” or “Remove the white baby onesie from the image”), our model demonstrates a precise understanding of spatial relationships and occlusion comparable to large-scale models (e.g., OmniGen2 and BAGEL). Furthermore, when dealing with complex style and background transformations (e.g., “Transfer the image into a hand-sculpted claymation style” or “Change the background from a clear blue sky with bare branches to a sunset sky over a lush forest”), our approach successfully preserves the structural integrity of the main subjects while seamlessly applying the desired modifications. In addition, for fine-grained localized edits, such as texture alterations and multi-object manipulation (e.g., “Change the tortoise’s shell texture to a smooth surface” or “Remove the basket of fruit on the coffee table, and change the color of the left armchair cushion to dark green”), our model efficiently executes intricate instructions without unintended disruptions to the surrounding context, proving its high instruction-following capability despite having significantly fewer parameters (0.39B) compared to models like Kontext (12B) or LongCat (6B).

OmniGen2 (4B) LongCat (6B) DeepGen1.0 (2B) SANA-1.6B Meissonic (1B) Nitro-E (0.3B) Ours (0.39B)

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

A rain-soaked street at dawn, still glimmering from the night’s storm. A single bicycle leans against a lamppost, its shadow ...

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

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Cute isometric 3D icon of a pastel mint green cactus with subtle smiling features, designed in clay texture, soft lighting ...

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

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

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

An emotionally charged, evocative portrait of a lone figure standing beneath a vast, stormy sky, surrounded by swirling winds ...

[Figure 125]

[Figure 126]

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

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Whimsical children’s storybook illustration ... a charming horse, gallops playfully under a deep indigo sky...oversized black cat...

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

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

A small French bulldog puppy sitting at a café table wearing a cute scarf, looking at a croissant on a plate, latte art beside it...

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

[Figure 195]

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

A clay animation-style depiction of a detective lizard in a trench coat, navigating a neon-lit mushroom forest, full of solid color...

[Figure 209]

[Figure 210]

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

A friendly green alien with big eyes wearing a blue hoodie walking on an empty street at night.

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

Portrait of a young woman outside wearing fur coat and beads around her neck.

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

A highly stylized close up of a single chrysanthemum flower. The bloom appears digitally enhanced or rendered, with an...

Figure 4 Comparison of image generation. All input prompts are listed at the bottom of the images.

###### 13

6，8, 11, 18, 23, 24,27,30

Input Image OmniGen2 (4B) DeepGen(2B) EditMGT(0.96B) VIBE (1.6B) Ours (0.39B)

Kontext (12B) BAGEL (7B) LongCat (6B)

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

Place a bicycle in the foreground of the flower field, near the bottom left side, to add a sense of leisure and outdoor activity.

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Change the castle-like structure in the picture from a garden setting to a bustling cityscape with skyscrapers.

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Remove the yacht in the image.

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

Transfer the image into a hand-sculpted claymation style.

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

Change the tortoise's shell texture to a smooth surface.

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

Remove the basket of fruit on the coffee table, and change the color of the left armchair cushion to dark green.

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

Change the background from a clear blue sky with bare branches to a sunset sky over a lush forest.

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

Change the person's shirt color to blue.

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

Remove the white baby onesie from the image.

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

Remove the large white house in the center of the image.

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

Add a small sailboat floating near the end of the dock in the background, with its sails partially filled with wind.

###### Figure 5 Comparison of image editing. All source images and instructions are from ImgEdit Benchmark.

Exp. Mechanism Training Stage GenEval ↑ ImgEdit ↑ Text-to-image Pretraining 0.70 Condition Mechanism

Pix2Pix T2I → Edit 0.56 3.67 Pix2Pix T2I → Edit → Unified 0.61 3.65

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Input Image TPJ Pretrain TPJ + RL TPJ + RL + DMD

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

In-context T2I → T2I 0.65 In-context T2I → Edit 0.64 3.88 In-context T2I → Unified 0.65 3.14 In-context T2I → Edit → Unified 0.71 3.94

Training Recipe

Make the person raise her right arm.

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

Reinforcement Learning In-context TPJ Pretrain → RLHF 0.72 4.11 Step Distillation In-context TPJ Pretrain → RLHF → DMD 0.70 3.8

Make the person raise her right arm.

Table 6 Ablation study on GenEval and ImgEdit benchmarks. “TPJ” denotes “Task-progressive Joint”.

###### Ours w/o RL Ours Ours (4 step) Input Image Ours w/o RL Ours Ours (4 step)

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

A photo of a bench. Remove the camper van in the image and make the surrounding forest more seamless.

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

A young artist painting near a window, natural light on skin and fabric. Replace the ski lift chair in the image with a hot air balloon floating above the building.

Figure 6 Ablation study on generation (left) and editing (right) tasks. From left to right in each task: results without RL, our full model, and our model after 4-step distillation.

#### 4.5 Ablation Study

To dissect the contribution of our proposed training strategies, we conduct an extensive ablation study on DreamLite, evaluating generation and editing performance on GenEval and ImgEdit benchmarks. The quantitative results are summarized in Table 6 and the qualitative results are shown in Fig. 6.

Condition Mechanism. We first investigate the effectiveness of the In-context (IC) mechanism compared to the traditional Pix2Pix-style channel concatenation after Text-to-image pretraining. As shown in Table 6, when maintaining the same editing training stage (i.e., T2I → Edit, row 2&5), the IC mechanism yields a significant improvement from 3.67 to 3.88. Furthermore, compared to the results at unified training stage (i.e., T2I → Edit → Unified, row 3&7), the IC mechanism proves more suitable for joint training, particularly for the generation task (from 0.61 to 0.71). We attribute this to the fact that IC treats the generation as a special case of in-context referencing (i.e., the condition image is represented by a blank image). This formulation allows the model to leverage a unified spatial prior across disparate behaviors.

Training Recipe. The transition from specialized tasks to unified behavior often encounters a performance bottleneck. Our results indicate that direct joint training (i.e., T2I → Unified, row 6) sacrifices both generation and editing precision simultaneously (e.g., GenEval 0.65 and ImgEdit 3.14). By introducing Task-progressive Joint (TPJ) Pre-training (i.e., T2I → Edit → Unified, row 7), we observe that the model not only recovers but also surpasses the performance of separate tasks (GenEval 0.71 vs. 0.70 and ImgEdit 3.91 vs. 3.88). This demonstrates that TPJ allows the compact backbone to internalize the fundamental logic of IC referencing before tackling unified modeling. We also conduct in-context T2I training following standard T2I pre-training (row 4), which results in a slight decline in GenEval performance. We posit that since the blank image provides no information, the model fails to utilize it for contextual referencing, thereby slowing down convergence and compromising T2I performance.

Reinforcement Learning. As shown in Table 6, integrating RLHF into the TPJ pipeline yields the highest

overall performance, achieving 4.11 on ImgEdit and 0.72 on GenEval. Beyond these numerical gains, visual evidence in Fig. 6 demonstrates a substantial leap in image aesthetics and high-frequency details. Specifically, it markedly improves background realism in generation tasks and enhances human identity maintenance during editing.

Step Distillation. DMD significantly accelerates the sampling process, albeit with a slight performance penalty in complex semantic alignment tasks. Fig. 6 presents the visual results of DreamLite with 4-step inference after Distribution Matching Distillation. This performance-efficiency trade-off is expected, as the compression of the ODE trajectory inherently constrains the model’s capacity to navigate high-dimensional latent manifolds for intricate edits.

#### 4.6 On-device Deployment

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

Transfer into water-color style. Transfer into 8-bit pixel-art style

A dog is running on the rainbow. Let it wear a red scarf.

Change the background into forest.

Change the background into snow mountain.

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

Transfer into water-color style.

Change the background into desert. Change the background into forest. Let it wear a red scarf.

A tiger wearing a tuxedo. Change the suit into blue color.

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

Transfer into impasto oil-paint style. Change the background into forest. Let it wear a hat Let it wear sunglasses.

A cat holds a sign saying ‘hello’. Transfer into pencil drawing style.

Figure 7 DreamLite generation and editing examples on mobile device.

To evaluate the practical utility of DreamLite, we conduct comprehensive experiments on real mobile phones (i.e., Xiaomi 14 and vivo X100), equipped with the Snapdragon 8 Gen3 (Qualcomm NPU) and Dimensity 9300 (MTK APU). All measurements are performed using a quantized W8A8 (i.e., weights 8-bit, activations 8-bit) U-Net backbone at a 1024 × 1024 resolution with a 4-step sampling schedule.

Runtime Optimization. Given that the standard Qwen3-VL text encoder comprises approximately 2B parameters, direct on-device text encoding remains a primary latency bottleneck, We choose to pre-deploy common prompts (e.g., stylized or predefined editing tasks) as pre-computed embeddings for instantaneous interaction on mobile devices. For future iterations, we are developing a lightweight text encoder (<1B) to enable full-pipeline on-device flexibility without compromising inference speed.

Performance Analysis. As summarized in Table 7, the U-Net inference per step is merely 103.84ms on the Snapdragon 8 Gen3, leading to a total generation or editing time of approximately 0.42s (excluding VAE). Including VAE decoding (∼22ms) and system overhead, the end-to-end user experience remains near the 1s threshold. Fig. 7 shows our generation (left) and editing (right) results. The UNet is quantized as W8A8 with a 4-step inference.

Component Snapdragon 8 Gen3 Dimensity 9300 Model Size

VAE Encoder (fp16) 22.17 26.16 2.45 MB U-Net (w8a8, per step) 103.84 122.53 389.00 MB VAE Decoder (fp16) 22.17 26.16 2.45 MB

Total (4 steps, U-Net) 415.36 490.12 393.90 MB

Table 7 On-device inference latency (ms) for DreamLite components at 1024 × 1024 resolution. Measurements are conducted on W8A8 quantized models.

### 5 Limitations

Despite the efficiency and versatility of DreamLite, several limitations remain that define our future research directions.

Text Encoder Scale. Although our U-Net backbone is highly compact (0.39B), the current pipeline still relies on a standard 2B-parameter text encoder. During on-device deployment, this component introduces non-negligible latency. To achieve a fully optimized on-device pipeline, we are actively developing a lightweight text encoder (<1B) to ensure seamless end-to-end inference.

Reconstruction Fidelity and Specialized Tasks. While DreamLite performs competitively on major benchmarks, we observe relatively lower scores on GEdit and certain qualitative artifacts in challenging scenarios (i.e., text generation, text editing and identity preservation in portrait editing). We attribute these bottlenecks to our extremely compact VAE (1.2M), which may inevitably suffer from information loss or reconstruction blurriness when handling complex structural details. To mitigate this, we plan to train a slightly larger, high-fidelity VAE. Furthermore, we are going to implement specialized fine-tuning for text and facial generation or editing to enhance perceptual quality and instruction following.

Multi-modal Alignment. Future work will also explore a more advanced step distillation scheme combined with reward model during the post training stage to better align our compact model with human aesthetic preferences and complex spatial instructions.

### 6 Conclusion

We presented DreamLite, a compact unified on-device diffusion model that supports both text-to-image generation and text-guided image editing within a single network. By unifying conditioning through in-context spatial concatenation and explicit task tokens, DreamLite achieves seamless multi-tasking without parameter overhead. To stably train this small-capacity unified model, we introduced a task-progressive curriculum (T2I → Edit → Joint) followed by high-quality SFT and reinforcement learning. Finally, step distillation compresses sampling to 4 denoising steps for efficient mobile inference. Experiments show that DreamLite (GenEval 0.72, ImgEdit 4.11) significantly outperforms prior on-device baselines and reamins competitive with much larger server-side model. On representative smartphone (i.e., Xiaomi 14), DreamLite could generate or edit a 1024 × 1024 image in less than 1s.

### References

- [1] Grigorii Alekseenko, Aleksandr Gordeev, Irina Tolstykh, Bulat Suleimanov, Vladimir Dokholyan, Georgii Fedorov, Sergey Yakubson, Aleksandra Tsybina, Mikhail Chernyshov, and Maksim Kuprashevich. Vibe: Visual instruction based editor. arXiv preprint arXiv:2601.02242, 2026.
- [2] Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. In The Thirteenth International Conference on Learning Representations, 2024.
- [3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [5] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.
- [6] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.
- [7] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.
- [8] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024.
- [9] Wei Chow, Linfeng Li, Lingdong Kong, Zefeng Li, Qi Xu, Hang Song, Tian Ye, Xian Wang, Jinbin Bai, Shilin Xu, et al. Editmgt: Unleashing potentials of masked generative transformers in image editing. arXiv preprint arXiv:2512.11715, 2025.
- [10] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [12] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.
- [13] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Houqiang Li, Han Hu, et al. Instructdiffusion: A generalist modeling interface for vision tasks. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 12709–12720, 2024.
- [14] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [15] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.
- [16] Google. Gemini 2.5 flash. https://aistudio.google.com/models/gemini-2-5-flash-image, 2025.
- [17] Dongting Hu, Jierun Chen, Xijie Huang, Huseyin Coskun, Arpit Sahni, Aarush Gupta, Anujraaj Goyal, Dishani Lahiri, Rajesh Singh, Yerlan Idelbayev, et al. Snapgen: Taming high-resolution text-to-image models for mobile devices with efficient architectures and training. arXiv preprint arXiv:2412.09619, 2024.
- [18] Dongting Hu, Aarush Gupta, Magzhan Gabidolla, Arpit Sahni, Huseyin Coskun, Yanyu Li, Yerlan Idelbayev, Ahsan Mahmood, Aleksei Lebedev, Dishani Lahiri, et al. Snapgen++: Unleashing diffusion transformers for efficient high-fidelity image generation on edge devices. arXiv preprint arXiv:2601.08303, 2026.

- [19] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.
- [20] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8362–8371, 2024.
- [21] Zhiwei Jia, Yuesong Nan, Huixi Zhao, and Gengdai Liu. Reward fine-tuning two-step diffusion models via learning differentiable latent-space surrogate reward. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12912–12922, 2025.
- [22] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.
- [23] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [24] Black Forest Labs. Flux2. https://github.com/black-forest-labs/flux2, 2025.
- [25] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.
- [26] Jiachen Li, Weixi Feng, Wenhu Chen, and William Yang Wang. Reward guided latent consistency distillation. arXiv preprint arXiv:2403.11027, 2024.
- [27] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36:20662–20678, 2023.
- [28] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.
- [29] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [30] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.
- [31] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [32] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [33] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [34] Weijian Luo. Diff-instruct++: Training one-step text-to-image generator model to align with human preferences. arXiv preprint arXiv:2410.18881, 2024.
- [35] Yihong Luo, Tianyang Hu, Weijian Luo, Kenji Kawaguchi, and Jing Tang. Reward-instruct: A reward-centric approach to fast photo-realistic image generation. arXiv preprint arXiv:2503.13070, 2025.
- [36] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095, 2025.
- [37] Madebyollin. taesd. https://github.com/madebyollin/taesd, 2024.
- [38] OpenAI. Gpt-image. https://openai.com/index/gpt-4o-system-card/, 2025.

- [39] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [40] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. 2023.
- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [42] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer, 2024.
- [43] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.
- [44] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.
- [45] Abdelrahman Shaker, Ahmed Heakl, Jaseel Muhammad, Ritesh Thawkar, Omkar Thawakar, Senmao Li, Hisham Cholakkal, Ian Reid, Eric P Xing, Salman Khan, et al. Mobile-o: Unified multimodal understanding and generation on mobile device. arXiv preprint arXiv:2602.20161, 2026.
- [46] Tong Shen, Jingai Yu, Dong Zhou, Dong Li, and Emad Barsoum. E-mmdit: Revisiting multimodal diffusion transformer design for fast image synthesis under limited resources. arXiv preprint arXiv:2510.27135, 2025.
- [47] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in neural information processing systems, 36:49659–49678, 2023.
- [48] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, et al. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025.
- [49] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.
- [50] Dianyi Wang, Ruihang Li, Feng Han, Chaofan Ma, Wei Song, Siyuan Wang, Yibin Wang, Yi Xin, Hongjian Liu, Zhixiong Zhang, et al. Deepgen 1.0: A lightweight unified multimodal model for advancing image generation and editing. arXiv preprint arXiv:2602.12205, 2026.
- [51] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [52] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [53] Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, and Wenhu Chen. Editreward: A human-aligned reward model for instruction-guided image editing. arXiv preprint arXiv:2509.26346, 2025.
- [54] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [55] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.

- [56] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.
- [57] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.
- [58] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.
- [59] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.
- [60] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.
- [61] Zhixiong Yue, Zixuan Ni, Feiyang Ye, Jinshan Zhang, Sheng Shen, and Zhenpeng Mi. Know your step: Faster and better alignment for flow matching models via step-aware advantages. arXiv preprint arXiv:2602.01591, 2026.
- [62] Yang Zhao, Yanwu Xu, Zhisheng Xiao, Haolin Jia, and Tingbo Hou. Mobilediffusion: Instant text-to-image generation on mobile devices. In European Conference on Computer Vision, pages 225–242. Springer, 2024.

