## MoViE: Mobile Diffusion for Video Editing

Adil Karjauv* Noor Fathima* Ioannis Lelekas Fatih Porikli Amir Ghodrati Amirhossein Habibian Qualcomm AI Research†

12 FPS Editing on Mobile

[Figure 1]

[Figure 2]

# arXiv:2412.06578v1[cs.CV]9Dec2024

Input Video In Pixar 3D style

[Figure 3]

Figure 1. MoViE is a fast video editing model, capable of generating 12 frames per second on a mobile phone. It requires significantly fewer floating point operations (FLOPs) to edit a single video frame, making it computationally more efficient than competing methods.

### 1. Introduction

### Abstract

Diffusion-based generative models have shown impressive quality in image editing [2, 15, 42, 58]. Building on these advancements, zero-shot video-based models extend these capabilities to the temporal dimension for video editing [4, 11, 33, 45, 60, 61]. This is typically achieved by developing techniques to propagate information between frames ensuring that changes made in one frame are coherently reflected in subsequent frames. While these techniques achieve remarkable results in style transfer and attribute editing, they are computationally expensive, making them prohibitive to run on constrained edge devices, e.g. mobile phones.

Recent progress in diffusion-based video editing has shown remarkable potential for practical applications. However, these methods remain prohibitively expensive and challenging to deploy on mobile devices. In this study, we introduce a series of optimizations that render mobile video editing feasible. Building upon the existing image editing model, we first optimize its architecture and incorporate a lightweight autoencoder. Subsequently, we extend classifier-free guidance distillation to multiple modalities, resulting in a threefold on-device speedup. Finally, we reduce the number of sampling steps to one by introducing a novel adversarial distillation scheme which preserves the controllability of the editing process. Collectively, these optimizations enable video editing at 12 frames per second on mobile devices, while maintaining high quality. Our results are available at https:// qualcomm-ai-research.github.io/mobilevideo-editing/.

While significant efforts have been made to reduce the computational cost of image diffusion models for on-device generation [3, 8, 30, 68], there has been little effort towards developing video editing models on edge, mainly due to the aforementioned high computational and memory costs. Inversion-based models [11, 22, 45] that extract source video features using DDIM inversion incur high computational and memory costs during inference. On the other hand, methods that propagate features from one or more frames to the other, a process known as cross-frame attention [61], also remain expensive to run on-device, as

*Equal Contribution †Qualcomm AI Research is initiative of Qualcomm Technologies, Inc.

Snapdragon and Qualcomm branded products are products of Qualcomm Technologies, Inc. and/or its subsidiaries.

each frame must attend to multiple frames. For example, editing a 120-frame video at a resolution of 512 × 384 with Fairy [60], a fast video-to-video (V2V) model, takes 13.8 seconds on 8×A100 GPUs. This is even higher for TokenFlow [11], an inversion-based V2V model, which takes 744 seconds [60].

In this work, we introduce the first on-device video editing model capable of generating 12 frames per second on Xiaomi-14 Pro. Xiaomi-14 Pro uses a Snapdragon® 8 Gen. 3 Mobile Platform with a Hexagon

processor. To achieve this, we begin with a conventional video editing pipeline. Broadly speaking, diffusion editing pipelines [60, 61] first encode source frames into the latent space using a Variational AutoEncoder (VAE) [44]. A denoising UNet model then iteratively denoises latents to generate edited latents, which are then decoded by the VAE decoder to produce the edited frames. In addition, many diffusion models use the classifier-free guidance technique to balance quality and diversity. Specifically, editing models utilize two guidance scales — image and text — which can be adjusted to trade off how strongly the generated image correspond with the source image and the edit instructions, respectively.

TM

In this work, we identify the factors causing computational bottlenecks within an editing pipeline. First, we observed that the architectures of both the denoising UNet and the VAE encoder/decoder are major bottlenecks in the editing pipeline. Second, while classifier-free guidance technique enhances the model’s controllability, it increases the computational cost three times as we need 3 forward passes per diffusion step to obtain the output. Lastly, the intrinsic design of diffusion models necessitates iterative denoising to edit frames. Each of denoising steps involves complex computations, making the generation process expensive. To address the aforementioned challenges, we propose a series of contributions:

- • We introduce the first on-device video-to-video diffusion model, capable of editing a 120-frame video at a resolution of 512 × 384 on a mobile device in 9.6 seconds, achieving a generation rate of 12 fps.
- • We propose a method to jointly distill multiple guidance scales in the editing models, reducing the Number of Forward Evaluations (NFE) per diffusion step by 3 times.
- • We present an adjusted adversarial distillation recipe for reducing number of diffusion steps, ensuring the model’s controllability is maintained post-distillation while reducing NFE of denoiser by 10 times.

Additionally, unlike some methods [61] that can generate only a limited number of frames due to memory constraints, our model can operate on long videos. This is because our model generate video frames independent of the video length, and its complexity increases linearly with the number of frames.

### 2. Related work

Video generation. Recent methods can be categorised into two groups. The first group enhances image models with temporal components, i.e. 3D convolutions, spatiotemporal transformers [17, 19, 20, 54, 69], for modeling temporal dimension at the expense of the inflated computational footprint, along with the need for extensive training on large and diverse collections of data. In the effort to overcome these hurdles, zero-shot models [21], rely on training-free techniques for boosting temporal consistency [11, 25, 33, 45, 66]. Common methods are, utilization of structural conditioning from source video [5, 66], injection of diffusion inversion features into the generation process [11, 45] and employing cross-frame attention using a set of key frames [10, 45, 61, 66]. Our method aligns with this direction but is computationally much lighter, making it suitable for low-end edge devices.

Optimized diffusion. Optimizing diffusion mainly entails the reduction of required steps for traversing the probability path from pure noise to the denoised sample, as well as reducing the cost of each step. In line with the former are the utilization of higher order solvers [36, 37, 64], the straightening of underlying ODE trajectories or direct mapping to data via Rectified Flows [31, 34, 70] and consistency models [35, 56, 57], respectively and the employment of progressive-step distillation [30, 39, 49] or adversarial training [51, 52, 59, 67] for fewer or even single step diffusion. On the other hand, quantization [14, 41, 53] and pruning [8, 30] have been used, as well as extensive research conducted on simplifying the denoiser itself [9, 12, 26]. Our work incorporates both architectural optimization and a reduction in the number of steps. While extending [30] with multimodal distillation, we manage to address a limitation of [51, 67], by restoring control over text and image guidance, the level of adherance to text and image conditioning respectively.

On-device generation. The abundance of edge devices, along with reduced cost and more privacy-secure inference on edge comparing to cloud-based approaches, have led to recent works targeting mobile devices and their NPUs [3, 6, 8, 30, 68]. While there has been progress in the video domain with fast zero-shot video editing models [22, 60, 65], there have been no attempts to implement video editing models on device due to their high computational and memory requirements. Our method pushes the boundaries by unlocking on-device zero-shot video editing.

### 3. MoViE

Our goal is to develop an on-device video-editing model that optimizes trade-off between efficiency and quality. To achieve this, we follow four stages: (1) selecting a base model, (2) introducing a mobile-friendly denoiser called

Mobile-Pix2Pix, (3) reducing the cost of classifier-free guidance through multimodal guidance distillation, and (4) reducing number of steps using adversarial step distillation.

#### 3.1. Base model

Zero-shot video editing using diffusion models have achieved impressive results. The common approach among such models is to employ an off-the-shelf image editing model and replace self-attention layers with crossframe attention to ensure temporally consistent and coherent frame generation. In specific, each self-attention layer receives the latent from previous layer and linearly projects it into query, key and value Q,K,V to produce the output by Self-Attn(Q,K,V ) = Softmax(QKT/

√

d)V . Cross-frame attention extends the idea by concatenating keys and values from one or more other frames, commonly called as anchors, resulting in CrossFrame-Attn(Q,K′,V ′) = Softmax(Q[K1;,...;Kt]T/

√

d)[V1;...;Vt]. For example, Rerender-A-Video [63] uses Stable Diffusion [48] as image editing model and uses first and previous frames as anchors to attend to the current frame. Fairy [60] uses InstructPix2Pix [2] as an instruction-based image editing model and uniformly selects 3 frames with equal intervals as anchors. Following the same direction, we use InstructPix2Pix as the base image model. InstructPix2Pix processes a source image and a text instruction for the desired edit. The text is encoded using a text encoder (like CLIP [46]), and the source image is converted into latent representations via a Variational Auto-Encoder (VAE). A conditional U-Net model then refines these latents iteratively for a fixed number of sampling steps to produce the edited image, which is finally decoded back to pixel space using VAE decoder. Frames are generated in 10 diffusion steps, with the middle frame used as the sole anchor. In the subsequent sections, we detail our optimization strategies for the base model.

#### 3.2. Mobile-Pix2Pix

InstructPix2Pix pipeline is computationally intensive. For instance, denoising a single step of batch 1 with UNet requires approximately 600 GFLOPs. For 10 steps and three forward passes needed for classifier-free guidance, this totals to ∼ 18.05 TFLOPs. Additionally, encoding and decoding an image of resolution 480 × 480 using VAE costs 3.2 TFLOPs. We implement two changes to optimize InstructPix2Pix pipeline for video editing on mobile devices. First, similar to [30], we remove expensive self-attention and cross-attention layers at the highest resolutions, in both encoder and decoder part of the UNet. Removing them leads to 12% reduction in FLOPs.

Second, we reduce the latency induced by encoding and decoding latents using VAE by utilizing the Tiny Autoencoder for Stable Diffusion (TAESD) [1]. TAESD is a deterministic and lightweight autoencoder composed of a series of

Algorithm 1 Multimodal Guidance Distillation Require: Teacher ϵ, Student Mθ, dataset D, loss weighting

term λ, learning rate γ

- 1: while not converged do
- 2: cI,cT,x ∼ D ▷ Sample input image, prompt, edited image
- 3: t ∼ U[0,1000] ▷ Sample time
- 4: sI ∼ U[1,3] ▷ Sample image guidance scale
- 5: sT ∼ U[2,14] ▷ Sample text guidance scale
- 6: ϵn ∼ N(0,I) ▷ Sample noise
- 7: xt = αtx + σtϵn ▷ Add noise to data
- 8: Obtain ϵ˜from Eq. 1 ▷ CFG
- 9: L = λ∥Mθ(xt,cI,cT,sI,sT) − ϵ˜∥22 ▷ Loss
- 10: θ ← θ − γ∇θL ▷ Optimization
- 11: end while

residual blocks. Its encoder is a distilled version of the VAE encoder, whereas the decoder has been trained as a standalone GAN, optimized with adversarial and reconstruction losses. Moreover, its shared latent space with VAE enables hybrid autoencoding, with VAE decoding TAESD’s latents and vice versa. Incorporating TAESD in our pipeline leads to 92.6% FLOPs savings compared to original InstructPix2Pix VAE with a modest drop in quality (see Sec. 4.3).

#### 3.3. Multimodal Guidance Distillation

Classifier-free guidance (CFG) is a technique to improve the quality of text-to-image generative models. It is achieved by a linear combination of estimates from text-conditional and unconditional models, weighted by a so-called guidance scale. While this technique allows for trade-off between fidelity and diversity, it incurs computational costs due to two forward passes in each diffusion step, with and without text condition. To reduce this cost, [39] distill classifier-free guidance into a model that requires only one forward pass per diffusion step in single modality, i.e. text. However, several works have utilized multi-modal CFG for image-toimage editing [2], editing 3D scenes with text [13], novel view synthesis [32], and video generation [29]. In this work, we extend the concept of CFG distillation to multiple modalities, specifically text and image conditioning, to further reduce the cost of InstructPix2Pix pipeline.

InstructPix2Pix model accepts source image condition cI and text condition cT as input. During inference, the model performs three separate forward passes with different conditionings, that are subsequently combined to generate the final output [2]:

ϵ˜(xt,cI,cT) = ϵ(xt,∅,∅)

+ sI (ϵ(xt,cI,∅) − ϵ(xt,∅,∅))

+ sT (ϵ(xt,cI,cT) − ϵ(xt,cI,∅))

(1)

!!

!! !" "#

"# !"

prompt

prompt

[Figure 4]

[Figure 5]

[Figure 6]

%

"#$%

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

%

+

$&

[Figure 11]

#"

%

#!

Guidance Control

- Figure 2. Multimodal Guidance Distillation Overview: Standard classifier-free guidance inference pipeline (left) with two input conditionings (image and text) requires three inference runs per diffusion step. Our distilled pipeline (right) incorporates guidance scales sI and sT into UNet and only performs one inference run.

where ϵ(.) is a denoiser function and sI and sT are image and text guidance scales respectively. sI and sT are control signals that can be adjusted to trade off the edited image fidelity to the source image and the edit instruction, respectively. As shown in Fig. 2, we distill ϵ(.) in Eq. (1) to a single forward pass using the loss function:

I,cT∼pdata ∥Mθ(xt,cI,cT,sI,sT) − ϵ˜(xt,cI,cT)∥22 ,

Ec

where Mθ(.), a student model, receives guidance scales sI and sT along with the conditions. This is done first by projecting these scales to embeddings using sinusoidal timestep embeddings [18]. Each ResNet block is then modified to accept these embeddings similarly to timestep embedding, which are further processed through a linear layer and nonlinearity before being added to the latent representation. Note that both image and text guidance scales are distilled at once through our distillation procedure. For more details see Algorithm 1. As a result, the distilled model Mθ(.) allows for a single forward pass in each diffusion step without compromising the quality or control (See Fig. 5).

#### 3.4. Adversarial Step Distillation

Having reduced the computational footprint of each step, in this section, we seek to distill the multi-step model obtained in Sec. 3.3 into a single-step model to establish an efficient on-device editing pipeline. Improving sampling efficiency of diffusion models has been the focus of several works [30, 35, 49]. Recently, adversarial distillation diffusion models that require a single denoising step have shown impressive results for text-to-image (T2I) and imageto-image (I2I) editing applications [51, 52, 67]. However, the classifier-free guidance property is usually ignored during training [51, 67]. For example LADD [51] foregoes the flexibility of text and image guidance in favor of increased sampling speed. This oversight of guidance scales leads to models that lack control over edit strength during inference,

making them less practical for use. In this work, we propose an adversarial training algorithm, built upon LADD [51], that distills a multi-step teacher into a single-step student while preserving its controllability. Our adversarial distillation setup is outlined in Fig. 3. Student Aθ is initialized from pre-trained teacher Mθ of the previous stage. Discriminator Dϕ consists of a frozen feature extractor, initialized from the encoder arm of the teacher and a set of trainable spatial heads applied to each layer of feature extractor. Detailed architecture of spatial heads can be found in the Appendix B. During adversarial training, real edited sample x0 is obtained from the multi-step teacher. Student receives a noisy sample xt generated by applying forward diffusion process to x0 using student noise schedule. To retain controllability, the corresponding guidance scales sI,sT are also provided to the student. The student then generates the fake samples xˆ0 which is then evaluated by the discriminator. Similar to [51], we first add noise to x0 and xˆ0 before feeding them to the discriminator. For more details see Algorithm 2. Discriminator distinguishes between real and fake samples, providing feedback to the student model to improve its ability to generate high-quality, controlled edits.

Algorithm 2 Adversarial Distillation

Require: Teacher model M, Student model Aθ, Discriminator Dϕ, loss weighting term λ, learning rate γ and γd for A and D, timesteps Tg for A, preconditioning functions cin and c′in [23], R1 is gradient penalty [40]

- 1: while not converged do
- 2: cI,cT ∼ D, sI ∼ U[1,3], sT ∼ U[2,14]
- 3: C = [cI,cT,sI,sT]
- 4: ϵ ∼ N(0,I);x0 = M(ϵ,C) ▷ Real sample from teacher’s denoising process
- 5: t ∼ U[0,8],ϵ ∼ N(0,I)
- 6: xt = cin(t) ∗ (αtx0 + σtϵ) ▷ Add noise to real
- 7: xˆ0 = Aθ(xt,C,t) ▷ Fake (Student) sample
- 8: t′ ∼ U[0,1000],ϵ′ ∼ N(0,I)
- 9: xt′ = c′in ∗ (αt′x0 + σt′ϵ′) ▷ Add noise to real
- 10: xˆt′ = c′in ∗ (αt′xˆ0 + σt′ϵ′) ▷ Add noise to fake
- 11: L = Et′,x0[max(0,1 + Dϕ(xt′,C)) + λr1R1] + Et,t′,x0[max(0,1 − Dϕ(ˆxt′,C))]
- 12: ϕ ← ϕ − γd∇ϕL ▷ Update Disc.
- 13: Repeat Step-8 and 10
- 14: L = λmse ∥xˆ0 − x0∥22 + λgenEt,t′,x0[Dϕ(ˆx′t,C)]
- 15: θ ← θ − γ∇θL ▷ Update Student
- 16: end while

### 4. Experiments

#### 4.1. Experimental setup

Datasets and metrics. In our work, we utilize the InstructPix2Pix dataset [2] for finetuning and distillation. The

3%45 '()*#+,% -,#.%(0") →

| |[Figure 12]<br><br>× 𝑵<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>𝑥<br><br>𝑥<br><br>𝑅𝑒𝑎𝑙<br><br>𝐹𝑎𝑘𝑒<br><br>𝜀 +<br><br>+<br><br>𝜀<br><br>𝑆𝑡𝑢𝑑𝑒𝑛𝑡<br><br>𝑇𝑒𝑎𝑐ℎ𝑒𝑟<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>𝑥<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>𝐴<br><br>𝑀 𝑀 𝐷<br><br>𝑥|
|---|---|
| | |

𝐷𝑖𝑠𝑐𝑟𝑖𝑚𝑖𝑛𝑎𝑡𝑜𝑟

[Figure 23]

[Figure 24]

[Figure 25]

𝐶𝑜𝑛𝑑𝑖𝑡𝑖𝑜𝑛𝑖𝑛𝑔 (𝐶) 𝑐 𝑐

[Figure 26]

[Figure 27]

𝑠

prompt

[Figure 28]

𝑠

!"#$%'()*#+,%-,#.%(0)→!

- Figure 3. Adversarial Distillation: We distill a multi-step teacher into a single step student using adversarial losses. Unlike existing adversarial distillation approaches [51, 67] that forego guidance flexibility for faster sampling, we preserve guidance strength prop-

[Figure 29]

[Figure 30]

[Figure 31]

erty during adversarial training by providing the synthetic latent xt from teacher’s denoising process and conditioning the student on the corresponding guidance scales.

dataset has around 300k samples, each sample consisting of (source image, edit instruction, and edited image) triplet. For evaluation and ablations, we use the validation set consisting of around 5k samples. Following [2], we utilize CLIP-Image similarity and directional CLIP similarity to estimate fidelity of edited image to the input image and edit prompt respectively. Higher values for these metrics indicate superior performance. Following [7, 55], for comparison with the state-of-the-art, we employ the Text-Guided Video Editing (TGVE) benchmark [62]. The dataset comprises of 76 videos where each video includes 4 prompts on editing style, background, object, and multiple attributes. We measure PickScore [28] and CLIPFrame[16] to evaluate the quality and consistency of the edits respectively. To measure number of floating point operations (FLOPs), we use DeepSpeed [47, v0.15.2]. Latencies are measured on a single NVIDIA® A100 GPU and on a Xiaomi-14 Pro for GPU and Phone respectively. Xiaomi-14 Pro uses a Qualcomm Snapdragon® 8 Gen. 3 Mobile Platform with a Hexagon

- Figure 4. MoViE at text guidance [4.0, 8.0, 12.0] and image guidance [1.25, 1.75]. Our adversarial training maintains guidance scales, allowing us to control edit strength during inference. (Prompt: In Van Gogh Style)

- Figure 5. CLIP metrics for InstructPix2Pix, Mobile-Pix2pix, Multi-modal Guidance (MMG) Mobile-Pix2pix and MoViE. As shown in the graphs, proposed optimizations improve the efficiency greatly with minimum quality drop.

processor. Reported latencies correspond to a single-frame denoising.

TM

Implementation details. To obtain an efficient and mobile-friendly model of InstructPix2Pix, we make sequential architectural and inference optimizations as follows. Stage 1: We begin with the text-to-image Stable Diffusion (SD) UNet V-1.5 UNet. Inspired by [30], we remove the costly self-attention and cross-attention layers at the highest resolution and finetune the model. Stage 2: Similar to [2], we add image conditioning layers and transform the T2I model to an I2I editing model and train on InstructPix2Pix dataset for 20k iterations, resulting in MobilePix2Pix. Stage 3: We distill multimodal guidance scales as described in Algorithm 1. We use 5k iteration for training where the teacher is a model from the previous stage. Stage 4: For guidance-preserving adversarial distillation, we first convert the model obtained from the previous stage to v−prediction as we find that it helps to reduce noisy artifacts during adversarial training. We then distill a multi-step

teacher model into a single-step student using a weighted combination of MSE and GAN loss for 20k iterations. The detailed training steps are shown in Algorithm 2, and further details can be found in the Appendix.

#### 4.2. Results

MoViE. We compare the impact of different optimization stages on editing quality in terms of directional CLIP similarity and CLIP-image similarity. As shown in Fig. 5, our first optimization stage, Mobile-Pix2Pix, achieves comparable quality as the original model while being 24.4% cheaper in terms of FLOPS and 73% faster on-device (see Tab. 1). With multimodal guidance distillation, a single forward pass is now required, thus our pipeline becomes 66% cheaper computation-wise compared to previous stage, still at the cost of negligible quality drop. Finally, our adver-

###### Method Steps PickScore↑ CLIPFrame↑ TFLOPs (per frame) Latency (GPU) Latency (Phone)

Fairy 10 19.80 0.933 - - TokenFlow 50 20.49 0.940 109.35 2.45s Rerender-A-Video 20 19.58 0.909 107.52 2.13s ControlVideo 50 20.06 0.930 89.49 5.63s InsV2V 20 20.76 0.911 52.21 2.70s RAVE 50 20.35 0.932 83.09 4.31s EVE - 20.76 0.922 - - -

Base Model 10 20.34 0.943 21.31 1.37s 7s

+ Mobile-Pix2Pix 10 19.43 0.922 16.10 1.06s 1.9s + Multi-Guidance Dist. 10 19.60 0.919 5.50 0.82s 0.6s

+ Adversarial Distillation 1 19.40 0.913 0.76 0.11s 0.08s

Table 1. End-to-end FLOPs and latency of video editing models on 480 × 480 resolution on TGVE benchmark, normalized per frame. On-device latencies reported on 512×384 frames. PickScore and CLIPFrame for competing methods (except RAVE) are from InsV2V [7].

sarial distillation technique further improves UNet latency by a factor of 10 and end-to-end latency by a factor of 7.5, enabling on-device editing at 12 frames per second. In the last stage, we observe a slight decline in quantitative metrics; however, the substantial gains in efficiency justify the trade-off, as evidenced by Tab. 1 and the qualitative results in Fig. 8. Finally, in Fig. 4 we show that our adversarial distillation technique retains the flexible editing property allowing us to control the strength of edit at inference.

SOTA comparison. In Table 1 we compare several stateof-the-art methods with ours in terms of quality and computation cost. These methods include: Fairy [60], a anchorbased cross-frame attention method for fast and coherent video-to-video synthesis; TokenFlow [11], a video editing method that propagate inversion features across frames; Rerender-A-Video [63], a zero-shot text-guided model with temporal-aware patch matching; ControlVideo [66], a training-free controllable model with cross-frame interaction; InsV2V [7], a consistent video editing model, EVE [55], a frame editing method with video adapters, and RAVE [22] that uses a novel noise shuffling strategy for fast zero-shot video editing. We found these methods to be highly computationally intensive in terms of FLOPs, making them unsuitable for mobile video editing applications. In contrast, our series of optimizations demonstrates significant improvements in both TFLOPs and GPU latency per frame while maintaining good quality edits, as evidenced by PickScore [28] and CLIPFrame [16] metrics in the TGVE benchmark [62]. Finally, we profile the latency of our method on a Xiaomi 14 Pro with Snapdragon 8 Gen.

- 3 mobile platform. As shown in Table 1, our series of optimizations lead to the on-device denoising pipeline with the impressive frame rate of 12. Check Appendix A for the human evaluation study. Qualitative results. First, we present the qualitative results of MoViE on DAVIS [43] videos, as illustrated in Figure 6. Our method effectively performs both global edits and fine-grained attribute editing with high efficiency.

Type of Ablation Ablation detail CLIP-Image

m = 0, s = 1 0.759

Noise Distribution

- m = −1, s = 1 0.769
- m = −1, s = 2 0.765

No guidance conditioning 0.781 With guidance conditioning 0.786

Head Architecture

Table 2. Ablation study on discriminator design choices in adversarial training.

In Figure 8, we compare our model to the base model, demonstrating enhanced efficiency while maintaining editing quality. Finally, in Figure 9, we compare MoViE with several state-of-the-art methods on two DAVIS [43] videos featuring challenging editing prompts. In the first scenario, which involves changing the hair color of a woman, InsV2V [7] and TokenFlow [11] performed relatively well, whereas Rerender-A-Video [63] introduced numerous unnecessary changes. In the second scenario, where a fire effect was requested to be added in, most methods failed except for InsV2V [7]. For both prompts, our method produced good quality edits, indicating its competitiveness with more expensive video editing algorithms. Please refer to https://qualcomm-ai-research.github. io/mobile-video-editing/ for video results.

#### 4.3. Ablations

##### Adversarial Training.

To confirm the effectiveness of our design choices, we conduct an ablation study for both the discriminator and generator setup. We use the guidance-distilled MobilePix2Pix model from Stage-3 that has been finetuned for v − prediction. We evaluate the models for a single sampling step using the InstructPix2Pix validation dataset and report CLIP-Image similarity metric in Tab. 2 for guidance scale of 7.5 and image guidance scale of 1.2. For the discriminator noise (see Algorithm 2), we find similar observations as [51, 67], where the noise distribution affects the

[Figure 32]

[Figure 33]

Input Video Input Video

[Figure 34]

[Figure 35]

Make it desert

Turn the swan into flamingo

[Figure 36]

[Figure 37]

Input Video Input Video

[Figure 38]

[Figure 39]

Add grass Add snow

[Figure 40]

[Figure 41]

Input Video Input Video

[Figure 42]

[Figure 43]

Make him a zombie Turn him into the yeti

- Figure 6. Qualitative results of MoViE on DAVIS. Our method can handle complex global edits as well as perform more nuanced attribute editing while requiring very few computational resources. Please refer to the Appendix for video results.

- Figure 7. CLIP metrics for different autoencoder configurations. A substational FLOPs reduction can be achieved by incorporating TAESD, with minimal drop in editing performance.

signing more budget to the decoder, although more compute expensive, seems a better option for providing better and more detailed reconstruction. In this work, we use TAESD for both encoding and decoding as we find it a good balance between quality and efficiency.

### 5. Conclusion

In this work, we have explored several optimizations to accelerate diffusion-based video editing. Firstly, we introduced an architectural enhancement of denoising UNet and a lightweight autoencoder, resulting in our Mobile-Pix2Pix model. Secondly, we performed novel multimodal guidance distillation which consolidates classifier-free guidance inference with text and image as input conditionings into one forward pass per diffusion step, achieving a threefold increase in inference speed. Lastly, we optimized the number of diffusion steps to one through an adversarial distillation procedure, while maintaining the controllability of edits. These optimizations enable 12.5 frames per second video editing on-device, marking a significant milestone towards real-time text-guided video editing on mobile platforms.

edit quality. As shown, setting the mean and standard deviation of noise distribution to −1 and 1, respectively gives the best results. We also observe that using guidance conditioning on discriminator heads provides better prompt adherence and flexibility although the CLIP-Image scores do not appear to vary significantly.

Impact of TAESD. Incorporation of TAESD leads to a significant FLOPs reduction which can be seen in Tab. 1 and Fig. 7. More importantly this is accompanied by a moderate drop in relevant CLIP scores. In addition, the shared embedding space enables combinations of VAE’s and TAESD’s encoders - decoders, for incorporating partial gains to the diffusion pipeline. As shown in Fig. 7, as-

### References

- [1] Ollin Boer Bohan. Tiny autoencoder for stable diffusion.
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. In-

[Figure 44]

[Figure 45]

Input Video Input Video

[Figure 46]

[Figure 47]

BaseModelMoViE

In Monet Style Turn him into a wooden statue

[Figure 48]

[Figure 49]

- Figure 8. Qualitative comparison of our method to the base model. The efficiency is greatly improved whereas quality is not compromised both for style transfer and attribute edits. Please refer to the Appendix for video results.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

InputVideoInsV2VMoViE Rerender

[Figure 56]

[Figure 57]

aVideo TokenFlow

[Figure 58]

[Figure 59]

Make her hair blond Add fire

- Figure 9. Qualitative comparison of our method MoViE to other SOTA video editing algorithms. We evaluate on two challenging editing scenarios. MoViE produces good quality edits yet far outperforms other competing methods in terms of efficiency. Please refer to the Appendix for video results.

structpix2pix: Learning to follow image editing instructions. arXiv preprint arXiv:2211.09800, 2022.

- [3] Thibault Castells, Hyoung-Kyu Song, Tairen Piao, Shinkook Choi, Bo-Kyeong Kim, Hanyoung Yim, Changgwun Lee, Jae Gon Kim, and Tae-Ho Kim. Edgefusion: On-device text-to-image generation. arXiv preprint arXiv:2404.11925, 2024.
- [4] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23206–23217, 2023.
- [5] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023.
- [6] Yu-Hui Chen, Raman Sarokin, Juhyun Lee, Jiuqiang Tang, Chuo-Ling Chang, Andrei Kulik, and Matthias Grundmann. Speed is all you need: On-device acceleration of large diffusion models via gpu-aware optimizations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4651–4655, 2023.
- [7] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent videoto-video transfer using synthetic dataset, 2023.
- [8] Jiwoong Choi, Minkyu Kim, Daehyun Ahn, Taesu Kim, Yulhwa Kim, Dongwon Jo, Hyesung Jeon, Jae-Joon Kim, and Hyungjun Kim. Squeezing large-scale diffusion models for mobile. arXiv preprint arXiv:2307.01193, 2023.
- [9] Tim Dockhorn, Robin Rombach, Andreas Blatmann, and Yaoliang Yu. Distilling the knowledge in diffusion models. In CVPR Workshop Generative Modelsfor Computer Vision, volume 2, 2023.
- [10] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023.
- [11] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023.
- [12] Amirhossein Habibian, Amir Ghodrati, Noor Fathima, Guillaume Sautiere, Risheek Garrepalli, Fatih Porikli, and Jens Petersen. Clockwork diffusion: Efficient generation with model-step distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8352–8361, 2024.
- [13] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19740–19750, 2023.
- [14] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate post-training quantization for diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [15] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [16] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras,

- and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022.
- [17] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [19] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [20] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.
- [21] Kumara Kahatapitiya, Adil Karjauv, Davide Abati, Fatih Porikli, Yuki M Asano, and Amirhossein Habibian. Objectcentric diffusion for efficient video editing. In European Conference on Computer Vision, pages 91–108. Springer, 2025.
- [22] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M Rehg, and Pinar Yanardag. Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6507–6516, 2024.
- [23] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022.
- [24] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models, 2022.
- [25] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023.
- [26] Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. Bk-sdm: A lightweight, fast, and cheap version of stable diffusion. arXiv preprint arXiv:2305.15798, 2023.
- [27] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017.
- [28] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation, 2023.
- [29] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.
- [30] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snap-

- fusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36, 2024.
- [31] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport, 2022.
- [32] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023.
- [33] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8599–8608, 2024.
- [34] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [35] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.
- [36] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.
- [37] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
- [38] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference, 2023.
- [39] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023.
- [40] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge?, 2018.
- [41] Nilesh Prasad Pandey, Marios Fournarakis, Chirag Patel, and Markus Nagel. Softmax bias correction for quantized generative models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1453–1458, 2023.
- [42] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023.
- [43] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander SorkineHornung. A benchmark dataset and evaluation methodology for video object segmentation. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [44] Lucas Pinheiro Cinelli, Matheus Ara´ujo Marins, Eduardo Ant´unio Barros da Silva, and S´ergio Lima Netto. Variational autoencoder. In Variational Methods for Machine Learning with Applications to Deep Networks, pages 111–

149. Springer, 2021.

- [45] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei,

- Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15932–15942, 2023.
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [47] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020.
- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [49] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [50] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. CoRR, abs/2202.00512, 2022.
- [51] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024.
- [52] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer, 2025.
- [53] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1972–1981, 2023.
- [54] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.
- [55] Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. Video editing via factorized diffusion distillation, 2024.
- [56] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.
- [57] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.
- [58] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023.
- [59] Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans

- with diffusion. arXiv preprint arXiv:2206.02262, 2022.
- [60] Bichen Wu, Ching-Yao Chuang, Xiaoyan Wang, Yichen Jia, Kapil Krishnakumar, Tong Xiao, Feng Liang, Licheng Yu, and Peter Vajda. Fairy: Fast parallelized instruction-guided video-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8261–8270, 2024.
- [61] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.
- [62] Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, Rui He, Feng Hu, Junhua Hu, Hai Huang, Hanyu Zhu, Xu Cheng, Jie Tang, Mike Zheng Shou, Kurt Keutzer, and Forrest Iandola. Cvpr 2023 text guided video editing competition, 2023.
- [63] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023.
- [64] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022.
- [65] Youyuan Zhang, Xuan Ju, and James J Clark. Fastvideoedit: Leveraging consistency models for efficient text-to-video editing. arXiv preprint arXiv:2403.06269, 2024.
- [66] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023.
- [67] Zhixing Zhang, Yanyu Li, Yushu Wu, Yanwu Xu, Anil Kag, Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, Junli Cao, Dimitris Metaxas, et al. Sf-v: Single forward video generation model. arXiv preprint arXiv:2406.04324, 2024.
- [68] Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-image generation on mobile devices. arXiv preprint arXiv:2311.16567, 2023.
- [69] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.
- [70] Yuanzhi Zhu, Xingchao Liu, and Qiang Liu. Slimflow: Training smaller one-step diffusion models with rectified flow. In European Conference on Computer Vision, pages 342–359. Springer, 2025.

## MoViE: Mobile Diffusion for Video Editing Supplementary Material

[Figure 60]

MoViE Baseline Both Good

[Figure 61]

(a) vs. TokenFlow (b) vs. InsV2V (c) vs. Rerender-A-Video

- Figure 10. Human Evaluation results comparing MoViE to TokenFlow[11], InsV2V[7], and Rerender-A-Video[63].

### A. Additional results

#### A.1. Human Evaluation

We conducted an A/B comparison between MoViE and three state-of-the-art (SOTA) methods: TokenFlow [11], InsV2V [7], and Rerender-A-Video [63]. Participants were asked to compare the overall quality of the generated videos, choosing between: 1) video 1 is better, 2) video 2 is better, or 3) both are equal. Each participant compared 24 video pairs, with each pair evaluated by 32 participants. As shown in Fig. 10, participants generally preferred our method over the SOTA methods. Our model significantly outperformed Rerender-A-Video and was reasonably preferred over TokenFlow and InsV2V. However, due to the limited sample size, we cannot draw definitive conclusions.

Figure 11. Adversarial Head Architecture.

### B. Training Details

Spatial Head Architecture. As discussed in Section 3.4, discriminator Dϕ consists of a frozen feature extractor and trainable spatial heads applied to activations received at each layer of feature extractor. In Fig. 11 we show a schematic view of the spatial heads. Each spatial head receives as input the activation from layer i of feature extractor along with guidance scale (sembT ), image guidance scale (sembI ) and diffusion timestep (temb) embeddings. We condition the output of each spatial head on the prompt embedding cembT .

Training. Here we provide a detailed description of our training procedure for each stage in our pipeline. Throughout the finetuning stages we use InstructPix2Pix[2] dataset, batch size of 512 at 256 × 256 resolution and Adam Optimizer [27].

Stage-3. Our multimodal guidance distillation algorithm is specified in Algorithm 1 of the main manuscript. For

training, we use a learning rate of 10−5. We train on a single A100 GPU and training takes around 14 hours.

Stage-4. Image-editing UNet model is an ϵ − prediction model. However, we find that prior to adversarial training, finetuning the UNet model for v −prediction helped to reduce artifacts during training. We use a teacher-student setup and train the student to match the v − prediction from multimodal guidance distilled teacher obtained in Stage-3 (We obtain v − prediction from ϵ − prediction using the standard equation as in [50]). For training, we use a learning rate of 10−6 with 4k warm-up steps. Training takes about a day.

Our adversarial distillation pipeline is shown in Figure 3 and the corresponding training algorithm is specified in Algorithm 2 of the main manuscript. We train for 20k iterations and learning rate of 10−5. Training takes about 3 days on a single A100 GPU.

Here we add further details regarding our training pipeline. We provide details of each training component here.

• Teacher. Teacher model is initialized from the UNet checkpoint obtained in Stage-3 and remains frozen. We

evaluate the teacher model for 5 sampling steps to obtain the clean latent x0 using LCM scheduler [38]. We find that the teacher checkpoint generates good quality results for 5 sampling steps.

- • Student. Student model is initialized from the UNet checkpoint obtained after the v − prediction finetuning described above. Student is trained to denoise the latent

and generate a clean latent xˆ0 at each sampled diffusion timestep t. During training, we use EulerDiscrete [24] noise scheduler with 8 training timesteps. During evaluation, we evaluate the student for a single timestep using LCM scheduler.

- • Discriminator. Discriminator consists of Feature Extractor and spatial heads. We only train the spatial heads while the feature extractor remains frozen. Discriminator uses a separate EulerDiscrete [24] noise scheduler with 1000 training timesteps. During training, we find that fixing discriminator noise distribution to logit-normal as in [67] and using mean = −1,std = 1 gave us the best performance. We ablate this property in Section 4.3.

