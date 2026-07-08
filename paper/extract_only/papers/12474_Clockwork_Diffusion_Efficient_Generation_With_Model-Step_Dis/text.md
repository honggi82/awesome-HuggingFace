## Clockwork Diffusion: Efficient Generation With Model-Step Distillation

# arXiv:2312.08128v2[cs.CV]20Feb2024

Amirhossein Habibian* Amir Ghodrati* Noor Fathima* Guillaume Sautiere Risheek Garrepalli Fatih Porikli Jens Petersen Qualcomm AI Research†

{ahabibia, ghodrati, noor, gsautie, rgarrepa, fporikli, jpeterse}@qti.qualcomm.com

### Abstract

This work aims to improve the efficiency of text-to-image diffusion models. While diffusion models use computationally expensive UNet-based denoising operations in every generation step, we identify that not all operations are equally relevant for the final output quality. In particular, we observe that UNet layers operating on highres feature maps are relatively sensitive to small perturbations. In contrast, low-res feature maps influence the semantic layout of the final image and can often be perturbed with no noticeable change in the output. Based on this observation, we propose Clockwork Diffusion, a method that periodically reuses computation from preceding denoising steps to approximate low-res feature maps at one or more subsequent steps. For multiple baselines, and for both text-to-image generation and image editing, we demonstrate that Clockwork leads to comparable or improved perceptual scores with drastically reduced computational complexity. As an example, for Stable Diffusion v1.5 with 8 DPM++ steps we save 32% of FLOPs with negligible FID and CLIP change. We release code at https://github.com/QualcommAI-research/clockwork-diffusion

### 1. Introduction

Diffusion Probabilistic Models (DPM), or Diffusion Models for short, have become one of the most popular approaches for text-to-image generation [34, 36]. Compared to Generative Adversarial Networks (GANs), they allow for diverse synthesized outputs and high perceptual quality [5], while offering a relatively stable training paradigm [12] and high controllability. One of the main drawbacks of diffusion models is that they are comparatively slow, involving repeated operation of computationally expensive UNet models [35]. As a result, a

*Equal contribution †Qualcomm AI Research is an initiative of Qualcomm Technologies,

Inc

SD UNet

Efficient UNet

Distilled Efficient UNet

[Figure 1]

[Figure 2]

[Figure 3]

Baseline

454 ms 341 ms

330 ms 213 ms

240 ms 154 ms

24.9% 35.5% 35.8%

[Figure 4]

[Figure 5]

[Figure 6]

WithClockwork

Figure 1. Time savings with Clockwork, for different baselines. All pairs have roughly constant FID (computed on MS-COCO 2017 5K validation set), using 8 sampling steps (DPM++). Clockwork can be applied on top of standard models as well as heavily optimized ones. Timings computed on NVIDIA® RTX® 3080 at batch size 1 (for distilled model) or 2 (for classifier-free guidance). Prompt: “the bust of a man’s head is next to a vase of flowers”.

lot of current research focuses on improving their efficiency, mainly through two different mechanisms. First, some works seek to reduce the overall number of sampling steps, either by introducing more advanced samplers [26, 27, 43] or by performing so-called step distillation [29, 37]. Second, some works reduce the required computation per step e.g., through classifier-free guidance distillation [13,29], architecture search [21], or with model distillation [17].

Our work can be viewed as a combination of these two axes. We begin with the observation that lower-resolution representations within diffusion UNets (i.e. those further from input and output) are not only influencing the semantic layout more than smaller details [4,41,48], they are also more resilient to perturbations and thus more amenable to distillation into a smaller model. Hence, we propose to perform model distillation on the lower-resolution parts of the UNet by reusing their representations from previous sam-

pling steps. To achieve this we make several contributions: 1) By approximating internal UNet representations with those from previous sampling steps, we are effectively performing a combination of model- and step distillation, which we term model-step distillation. 2) We show how to design a lightweight adaptor architecture to maximize compute savings, and even show performance improvements by simply caching representations in some cases. 3) We show that it is crucial to alternate approximation steps with full UNet passes, which is why we call our method Clockwork Diffusion. 4) We propose a way to train our approach without access to an underlying image dataset, and in less than 24h on a single NVIDIA® Tesla® V100 GPU. We apply Clockwork to both text-to-image generation (MSCOCO [22]) and image editing (ImageNet-R-TI2I [48]), consistently demonstrating savings in FLOPs as well as latency on both GPU and edge device, while maintaining comparable FID and CLIP score. Clockwork is complementary to other optimizations like step and guidance distillation [29, 37] or efficient samplers: we show savings even on an optimized and DPM++ distilled Stable Diffusion model [27,34], as can be visualized in Fig. 1.

### 2. Related work

Faster solvers. Diffusion sampling is equivalent to integration of an ODE or SDE [46]. As a result, many works attempt to perform integration with as few steps as possible, often borrowing from existing literature on numerical integration. DDIM [44] introduced deterministic sampling, drastically improving over the original DDPM [12]. Subsequently, works have experimented with multistep [23], higher-order solvers [7, 15, 16], predictor-corrector methods [50, 51], or combinations thereof. DPM++ [26, 27] stands out as one of the fastest solvers, leveraging exponential integration, and we conduct most of our experiments with it. However, in our ablation studies in the AppendixTab. 4, we show that the benefit of Clockwork is largely independent of the choice of solver.

Step Distillation starts with a trained teacher model, and then trains a student to mirror the output of multiple teacher model steps [28, 37]. It has been extended to guided diffusion models [21,29], where Meng et al. [29] first distill unconditional and conditional model passes into one and then do step distillation following [37]. Berthelot et al. [1] introduce a multi-phase distillation technique similar to Salimans and Ho [37], but generalize the concept of distilling to a student model with fewer iterations beyond a factor of two. Other approaches do not distill students to take several steps simultaneously, but instead aim to distill straighter sampling trajectories, which then admit larger step sizes for integration [24,25,45]. In particular, InstaFlow [25] shows impressive results with single-step generation.

Our approach incorporates ideas from step distillation wherein internal UNet representations from previous steps are used to approximate the representations at the same level for the current step. At the same time, it is largely orthogonal and can be combined with the above. We demonstrate savings on an optimized Stable Diffusion model with step and guidance distillation.

Efficient Architectures. To reduce the architecture complexity of UNet, model or knowledge distillation techniques have been adopted either at output level or feature level [6, 17, 21]. Model pruning [3, 21] and model quantization [8, 30, 39] have also been explored to accelerate inference at lower precision while retaining quality. Another direction has been to optimize kernels for faster on-device inference [2], but such solutions are hardware dependent.

Our work can be considered as model distillation, as we replace parts of the UNet with more lightweight components. But unlike traditional model distillation, we only replace the full UNet for some steps in the trajectory. Additionally, we provide our lightweight adaptors outputs from previous steps, making it closer to step distillation.

### 3. Analysis of perturbation robustness

Our method design takes root in the observation that lowerresolution features in diffusion UNets are robust to perturbations, as measured by the change in the final output. This section provides a qualitative analysis of this behaviour.

During diffusion sampling, earlier steps contribute more to the semantic layout of the image, while later steps are more related to high-frequency details [4, 41]. Likewise, lowerres UNet representations contribute more to the semantic layout, while higher-res features and skip connections carry high-frequency content [41, 48]. This can be leveraged to perform image editing at a desired level of detail by performing DDIM inversion [46] and storing feature and attention maps to reuse during generation [48]. We extend this by finding that the lower-res representations, which contribute more to the semantic layout, are also more robust to perturbations. This makes them more amenable to distillation.

For our illustrative example, we choose random Gaussian noise to perturb feature maps. In particular, we mix a given representation with a random noise sample in a way that keeps activation statistics roughly constant. We assume a feature map to be normal f ∼ N(µf,σf2), and draw a random sample z ∼ N(0,σf2). We then update the feature map with:

f ← µf + √α · (f − µf) + √1 − α · z (1)

On average, this will leave the distribution unchanged. We set α = 0.3 to make the noise the dominant signal.

Perturb from step 0 Perturb from step 1 Perturb from step 2 Perturb from step 3 Perturb from step 4 Perturb from step 5 Perturb from step 10 Perturb from step 15

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Low-resfeatures

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Mid-resfeaturesHigh-resfeatures

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

|[Figure 30]<br><br>[Figure 31]|
|---|

- Figure 2. Perturbing Stable Diffusion v1.5 UNet representations (outputs of the three upsampling layers), starting from different sampling steps (20 DPM++ steps total, note the reference image as inset in lower-right). Perturbing low-resolution features after only a small number of steps has a comparatively small impact on the final output, whereas perturbation of higher-res features results in high-frequency artifacts. Prompt: ”image of an astronaut riding a horse on mars.”

In Fig. 2 we perform such perturbations on the outputs of the three upsampling layers of the Stable Diffusion v1.5 UNet [34]. Perturbation starts after a varying number of unperturbed steps and the final output is shown for each case. After only a small number of steps the lowest-resolution features can be perturbed without a noticeable change in the final output, whereas higher-res features are affected for longer along the trajectory. Moreover, early perturbations in lower-res layers mostly result in semantic changes, confirming findings from other works [4,41]. Implementation details and additional analyses for other layers are provided in Appendix C. Motivated by these findings, we propose to approximate lower-res UNet representations using more computationally lightweight functions, and in turn reuse information from previous sampling steps, effectively combining model and step distillation. However, we make another crucial and non-trivial contribution. Fig. 2 might suggest that one should approximate all representations after a certain sampling step. We instead find that it is beneficial to alternate approximation steps and full UNet passes to avoid accumulating errors. This makes our approach similar to others that run model parts with different temporal granularity [20,40], and we consequently name it Clockwork Diffusion.

### 4. Clockwork Diffusion

Diffusion sampling involves iteratively applying a learned denoising function ϵθ(·), or an equivalent reparametrization, to denoise a noisy sample xt into a less noisy sample xt−1 at each iteration t, starting from a sample from Gaussian noise at t = T towards a final generation at t = 0 [12,42].

As is illustrated in Fig. 3, the noise prediction function ϵ (we omit the parameters θ for clarity) is most commonly im-

plemented as a UNet, which can be decomposed into lowand high-resolution denoising functions ϵL and ϵH respectively. ϵH further consists of an input module ϵinH and an output module ϵoutH , where ϵinH receives the diffusion latent xt and ϵoutH predicts the next latent xt−1 (usually not directly, but by estimating its corresponding noise vector or denoised sample). The low-resolution path ϵL receives a lower-resolution internal representation rtin from ϵinH and predicts another internal representation rtout that is used by ϵoutH . We provide a detailed view of the architecture and how to separate it in the Appendix A. The basis of Clockwork Diffusion is the realization that the outputs of ϵL are relatively robust to perturbations — as demonstrated in Sec. 3 — and that it should be possible to approximate them with more computationally lightweight functions if we reuse information from previous sampling steps. The latter part differentiates it from regular model distillation [6, 17]. Overall, there are 4 key contributions that are necessary for optimal performance: a) joint model and step distillation, b) efficient adaptor design, c) Clockwork scheduling, and d) training with unrolled sampling trajectories. We describe each below.

#### 4.1. Model-step distillation

Model distillation is a well-established concept where a smaller student model is trained to replicate the output of a larger teacher model, operating on the same input. Step distillation is a common way to speed up sampling for diffusion models, where a student is trained to replace e.g. two teacher model passes. Here the input/output change, but the model architecture is usually kept the same. We propose to combine the two, replacing part of the diffusion UNet with a more lightweight adaptor, but in turn giving it access to outputs from previous sampling steps (as shown in Fig. 3).

High-res in 𝜖

𝑥 High-res in 𝑥 𝑥 𝜖

High-res out 𝜖

High-res out 𝜖

Low-res 𝜖 𝑟 𝑟

Low-res 𝜖 𝑟 𝑟

𝝐

Adaptor 𝜙

TRAINING VARIANTS

𝑡𝑒𝑥𝑡 𝑡

| | |
|---|---|
| | |
| | |

| | |
|---|---|
|Lin| |
| | |

- a) regular distillation forward noise uses images
- b) generation unroll doesnotuse images

[Figure 32]

𝑥 𝑥 𝑥

|ResBlock| |
|---|---|
| | |

| |ResBlock|
|---|---|
| | |

ResBlock

ResBlock

ConvT2d

Conv2d

⨁ ⨁

𝑡𝑒𝑥𝑡 𝑡

[Figure 33]

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥

//2 x2

- Figure 3. Schematic view of Clockwork. It can be thought of as a combination of model distillation and step distillation. We replace the lower-resolution parts of the UNet ϵ with a more lightweight adaptor, and at the same time give it access to features from the previous sampling step. Contrary to common step distillation, which constructs latents by forward noising images, we train with sampling trajectories unrolled from pure noise. Other modules are conditioned on text and time embeddings (omitted for readability). The gray panel illustrates the difference between regular distillation and our proposed training with unrolled trajectories.

We term this procedure model-step distillation. In its simplest form, an adaptor ϕθ is an identity mapping that naively copies a representation rout from step t + 1 to t. This works relatively well when the number of sampling steps is high, as for example in our image editing experiments in Sec. 5.3. For a more effective approximation in the low step regime, we rely on a parametric function ϕθ with additional inputs: rˆtout = ϕθ rtin,rtout+1,temb,textemb , which we describe as follows.

- 4.2. Efficient adaptor architecture

Time embedding temb is an additional input to the adaptor to make it conditional on the diffusion step t, instead of training separate adaptor models for each step. For this purpose we rely on the standard ResBlocks with time step embeddings, as in Rombach et al. [34].

Prompt embedding textemb is an additional input to the adaptor to make it conditional on the generation prompt. We rely on the pooled CLIP embedding [32] of the prompt, extracted using OpenCLIP’s ViT-g/14 [14], instead of the sequence to reduce the complexity.

The design of our adaptor is chosen to minimize heavy compute operations. It uses no attention, and is instead comprised of a strided convolutional layer resulting in two times spatial downsampling, followed by addition of a linear projection of the prompt embedding, two ResNet blocks with additive conditioning on t, and a final transposed convolution to go back to the original resolution. We further introduce a residual connection from input to output. The adaptor architecture is shown in Fig. 3, and we provide more details in Appendix A. We ablate several architecture choices in Sec. 5.4. The inputs to the adaptor are listed below.

#### 4.3. Clockwork scheduling

Instead of just replacing ϵL with an adaptor ϕθ entirely, we avoid accumulating errors during sampling by alternating lightweight adaptor steps with full UNet passes, which is the inspiration for our method’s name, following [20, 40]. Specifically, we switch between ϵL and ϕθ based on a predefined clock schedule C(t) ∈ {0,1} as follows:

ϵL rtin,temb,textemb , C(t) = 0 ϕθ rtin,rtout+1,temb,textemb , C(t) = 1

rˆtout =

Input representation rtin is the representation obtained from the high-res input module ϵinH at the current step, as shown in Fig. 3. It is concatenated with the next input.

where t and c are time step and prompt embeddings, respectively. C(t) can generally be an arbitrary schedule of switches between ϵL and ϕθ, but we find that interleaving them at a fixed rate offers a good tradeoff between performance and simplicity. Because we conduct our experiments mostly in the low-step regime with ≤ 8 steps, we simply alternate between adaptor and full UNet in consecutive steps (i.e. a clock of 2) unless otherwise specified. For sampling with more steps it is possible to use more consecutive adaptor passes, as we show in Appendix D.2 for the text-guided

Output representation rtout+1 is the equivalent representation from the previous sampling step that the adaptor tries to approximate for the current step. The high-res output module predicts the next diffusion latent from it. By conditioning on rtout+1, our approach depends on the sampler and step width (similar to step distillation).

image editing case. For the rest of the paper, we simply use the terminology a clock of N, which means every N steps, a full UNet pass will be evaluated, all other steps use the adaptor.

#### 4.4. Distillation with unrolled trajectories

We seek to train an adaptor that predicts an internal UNet representation, based on the same representation from the previous sampling step as well as further inputs. Formally, we minimize the following loss:

L = E

t

rtout − ϕθ rtin,rtout+1,temb,textemb 2 (2)

A common choice is to stochastically approximate the expectation over update steps, i.e. just sample t randomly at each training step. Most step distillation approaches [29,37] then construct xt from an image x0 via the diffusion forward process, and perform two UNet passes of a teacher model to obtain all components required for the loss. Instead of this, we start from a random noise sample and unroll a full sampling trajectory {xT,...,x0} with the teacher model, then use each step as a separate training signal for the adaptor. This is illustrated in Fig. 3. We construct a dataset of unrolled sampling trajectories for each epoch, which can be efficiently parallelized using larger batch sizes. We compare our unrolled training with the conventional approach in Sec. 5.4. Overall training can be done in less than a day on a single NVIDIA® Tesla® V100 GPU. As an added benefit, this training scheme does not require access to an image dataset and only relies on captions. We provide more details in Sec. 5 and include training pseudo-code in AppendixAlgorithm 1.

### 5. Experiments

We evaluate the effectiveness of Clockwork on two tasks: text-guided image generation in Sec. 5.2 and text-guided image editing in Sec. 5.3. Additionally, we provide several ablation experiments in Sec. 5.4.

#### 5.1. Experimental setup

Datasets and metrics We evaluate our text-guided image generation experiments by following common practices [21, 29, 34] on two public benchmarks: MS-COCO 2017 (5K captions), and MS-COCO 2014 [22] (30K captions) validation sets. We use each caption to generate an image and rely on the CLIP score from a OpenCLIP ViTg/14 model [14] to evaluate the alignment between captions and generated images. We also rely on Fr´echet Inception Distance (FID) [11] to estimate perceptual quality. For MS-COCO 2014, the images are resized to 256 × 256 before computing the FID as in Kim et al. [17]. We

evaluate our text-guided image editing experiments on the ImageNet-R-TI2I [48] dataset that includes various renderings of ImageNet-R [9] object classes. Following [48], we use 3 high-quality images from 10 different classes and 5 prompt templates to generate 150 image-text pairs for evaluation. In addition to the CLIP score, we measure the DINO self-similarity distance as introduced in Splice [47] to measure the structural similarity between the source and target images.

To measure the computational cost of the different methods, we report the time spent on latent generation, which we call latency for short, as it represents the majority of the total processing time. This measures the cost spent on UNet forward passes during the generation — and inversion in case of image editing — but ignores the fixed cost of text encoding and VAE decoding. Along with latencies we report the number of floating point operations (FLOPs). We measure latency using PyTorch’s benchmark utilities on a single NVIDIA® RTX® 3080 GPU, and use the DeepSpeed [33] library to estimate the FLOP count. Finally, to verify the efficiency of Clockwork on low-power devices, we measure its inference time on a Samsung Galaxy S23 device. It uses a Qualcomm “Snapdragon® 8 Gen. 2 Mobile Platform” with a Qualcomm® Hexagon

processor

TM

Diffusion models We evaluate the effectiveness of Clockwork on three latent diffusion models with varying computational costs: i) SD UNet, the standard UNet from Stable Diffusion v1.5 [34]. ii) Efficient UNet, which, inspired by Li et al. [21], removes the costly transformer blocks, including self-attention and cross-attention operations, from the highest resolution layer of SD UNet. iii) Distilled Efficient UNet, which further accelerates Efficient UNet by implementing progressive step distillation [37] and classifier-free guidance distillation [29]. Since there is no open source implementation [21,29,37] available, we rely on our replication as specified in the supplementary materials. In all experiments we use the DPM++ [27] multi-step scheduler due to its superiority in the low number of sampling steps regime, which is a key focus of our paper. An exception is the text-guided image editing experiment where we use the DDIM scheduler as in Plug-and-Play [48].

Implementation details We train Clockwork using a ResNet-based adaptor (as shown in Fig. 3) for a specific number of generation steps T and with a clock of 2, as described in Sec. 4.1, on 50K random captions from the LAION-5B dataset [38]. The training involves 120 epochs using the Adam optimizer [19] with a batch size of 16 and learning rate of 0.0001. Thanks to its parameter efficiency each training takes less than one day on a single NVIDIA® Tesla® V100 GPU.

SD UNet Efficient UNet Distilled Efficient UNet

40

Baseline

with Clockwork

FID[]

30

0.30

CLIP[]

0.28

0.26

4 6 8 10 TFLOPs

4 6 8 TFLOPs

2 3 4 TFLOPs

Figure 4. Clockwork improves text-to-image generation efficiency consistently over various diffusion models. Models are evaluated on 512 × 512 MS-COCO 2017-5K validation set.

#### 5.2. Text-guided image generation

We evaluate the effectiveness of Clockwork in accelerating text-guided image generation for three different diffusion models as specified in Sec. 5.1. For each model, we measure the generation quality and computational cost using 8, 6 and 4 steps with and without clockwork, as shown in Fig. 4. For the baselines (dashed lines) we also include a point with 3 sampling steps as a reference. Our results demonstrate that applying Clockwork for each model results in a high reduction in FLOPs with little changes in generation qualities (solid lines). For example, at 8 sampling steps, Clockwork reduces the FLOPs of the distilled Efficient UNet by 38% from 4.7 TFLOPS to 2.9 TFLOPS with only a minor degradation in CLIP (0.6%) and improvement in FID (5%). Fig. 5 shows generation examples for Stable Diffusion with and without Clockwork, while Fig. 1 shows an example for Efficient UNet and its distilled variant. See Appendix E for more examples.

Our improvement on the distilled Efficient UNet model demonstrates that Clockwork is complementary to other acceleration methods and adds savings on top of step distillation [37], classifier-free guidance distillation [29], efficient backbones [21] and efficient noise schedulers [27]. Moreover, Clockwork consistently improves the diffusion efficiency at very low sampling steps, which is the critical operating point for most time-constrained real-world applications, e.g. image generation on phones.

In Tab. 1 and Tab. 2 we compare Clockwork to state-ofthe-art methods for efficient diffusion on MS-COCO 2017 and 2014 respectively. The methods include classifier-free guidance distillation by Meng et al. [29], SnapFusion [21], model distillation from BK-SDM [17] and InstaFlow [25]. For BK-SDM [17] we use models available in the diffusers library [49] for all measurements. For Meng et al. [29], SnapFusion [21] and InstaFlow (1 step) [25] we report scores from the original papers and implement their architecture to measure latency and FLOPS. In terms of quan-

[Figure 34]

[Figure 35]

Figure 5. Text guided generations by SD UNet without (top) and with (bottom) Clockwork at 8 sampling steps (DPM++). Clockwork reduces FLOPs by 32% at a similar generation quality. Prompts given in Appendix E.

titative performance scores, Clockwork improves FID and slightly reduces CLIP on both datasets. Efficient UNet + Clockwork achieves the best FID out of all methods. InstaFlow has lowest FLOPs and latency as they specifically optimize the model for single-step generation, however, in terms of FID and CLIP, Clockwork is significantly better. Compared to SnapFusion, which is optimized and distilled from the same Stable Diffusion model, our Distilled Efficient UNet + Clockwork is significantly more compute efficient and faster.

#### 5.3. Text-guided image editing

We apply our method to a recent text-guided image-toimage (TI2I) translation method called Plug-and-Play (PnP) [48]. The method caches convolutional features and attention maps during source image inversion [46] at certain steps early in the trajectory. These are then injected during the generation using the target prompt at those same steps. This enables semantic meaning of the original image to be preserved, while the self-attention keys and queries allow preserving the guidance structure.

PnP, like many image editing works [10, 18, 31], requires DDIM inversion [46]. Inversion can quickly become the complexity bottleneck, as it is often run for many more steps than the generation. For instance, PnP uses 1000 inversion steps and 50 generation steps. We focus on evaluating PnP

###### Model FID [↓] CLIP [↑] TFLOPs Latency (GPU) Latency (Phone)

Meng et al. [29] 26.9 0.300 6.4 320 SnapFusion [21] 24.20 0.300 4.0 185 BK-SDM-Base [17] 29.26 0.291 8.4 348 BK-SDM-Small [17] 29.48 0.272 8.2 336 BK-SDM-Tiny [17] 31.48 0.268 7.8 313 InstaFlow (1 step) [25] 29.30 0.283 0.8 40 -

SD UNet 24.64 0.300 10.8 454 3968

+ Clockwork 24.11 0.295 7.3 (−32%) 341 (−25%) 3176 (−20%) Efficient UNet 24.22 0.302 9.5 330 1960

- + Clockwork 23.21 0.296 5.9 (−38%) 213 (−36%) 1196 (−39%)

Distilled Efficient UNet 25.75 0.297 4.7 240 980

- + Clockwork 24.45 0.295 2.9 (−38%) 154 (−36%) 598 (−39%)

Table 1. Text guided image generation results on 512 × 512 MSCOCO 2017-5K validation set. We compare to state-of-the-art efficient diffusion models, all at 8 sampling steps (DPM++) except when specified otherwise. Latency measured in ms.

and its Clockwork variants on the ImageNet-R-TI2I real dataset with SD UNet. Contrary to the rest of the paper, we use the DDIM sampler for these experiments to match PnP’s setup. To demonstrate the benefit of Clockwork in a training-free setting, we use an identity adaptor with a clock of 2 both in inversion and generation. We use the official open-source diffusers [49] implementation1 of PnP for these experiments, details in Appendix D.1.

In Fig. 6 we show qualitative examples of the same textimage pair with and without Clockwork for different DDIM inversion steps and generation fixed to 50 steps. For high numbers of inversion steps, Clockwork leads to little to no degradation in quality while consistently reducing latency by about 25%. At lower numbers of inversions steps, where less features can be extracted (and hence injected at generation), Clockwork outputs start diverging from the baseline’s, yet in semantically meaningful and perceptually pleasing ways.

On the right hand side of Fig. 6, we quantitatively show how, for various number of inversion steps, applying Clockwork enables saving computation cycles while improving text-image similarity and only slightly degrading structural distance. For PnP’s default setting of 1000 inversion steps and 50 generation steps (rightmost point on each curve) Clockwork allows saving 33% of the computational cycles while significantly improving CLIP score, and only slightly degrading DINO self-similarity.

#### 5.4. Ablation analysis

In this section we inspect different aspects of Clockwork. For all ablations, we follow the same training procedure explained in Sec. 5.1 and evaluate on the MS-COCO 2017 dataset, with a clock of 2 and Efficient Unet as backbone. Further ablations, e.g. results on different solvers, adaptor input variations are shown in Appendix B.

1https://github.com/MichalGeyer/pnp-diffusers

###### Model FID [↓] CLIP [↑] TFLOPs

SnapFusion [21] 14.00 0.300 4.0 BK-SDM-Base [17] 17.23 0.287 8.4 BK-SDM-Small [17] 17.72 0.268 8.2 BK-SDM-Tiny [17] 18.64 0.265 7.8 InstaFlow (1 step) [25] 20.00 - 0.8

SD UNet 12.77 0.296 10.8

+ Clockwork 12.27 0.291 7.3 (−32%) Efficient UNet 12.33 0.296 9.5

- + Clockwork 11.14 0.290 5.9 (−38%)

Distilled Efficient UNet 13.92 0.292 4.7

- + Clockwork 12.37 0.291 2.9 (−38%)

Table 2. Text guided image generation results on 256 × 256 MSCOCO 2014-30K validation set. We compare to state-of-the-art efficient diffusion models. Except for InstaFlow [25] all models are evaluated at 8 sampling steps using the DPM++ scheduler.

Adaptor Architecture. We study the effect of different parametric functions for the adaptor in terms of performance and complexity. As discussed in Sec. 4.1, ϕθ can be as simple as an identity function, where we directly reuse low-res features from the previous time step at the current step. As shown in Tab. 5, Identity function performs reasonably well, indicating high correlation in low-level features of the UNet across diffusion steps. In addition, we tried 1) a UNet-like convolutional architecture with two downsampling and upsampling modules, 2) a lighter variant of it with 3M parameters and less channels, 3) our proposed ResNet-like architecture (see Fig. 3). Details for all variants are given in Appendix A. From Tab. 5, all adaptors provide comparable performance, however, the ResNet-like adaptor obtains better quality-complexity trade-off.

Adaptor Clock. Instead of applying ϕθ in an alternating fashion (i.e. a clock of 2), in this ablation we study the effect of non-alternating arbitrary clock C(t). For an 8-step generation, we use 1) C(t) = 1 for t ∈ {5,6,7,8} and 2) C(t) = 1 for t ∈ {3,4,5,6}, C(t) = 0 otherwise. As shown in Tab. 5, both configurations underperform compared to the alternating clock, likely due to error propagation in approximation. It is worth noting that approximating earlier steps (config. 2) harms the generation significantly more than later steps (config. 1).

UNet cut-off. We ablate the splitting point where high-res and low-res representations are defined. In particular, we set the cut-off at the end of stage 1 or stage 2 of the UNet (after first and second downsampling layers, respectively). A detailed view of the architecture with splitting points can be found in the supplementary material. The lower the resolution in the UNet we set the cutoff to, the less compute we will save. As shown in Tab. 5, splitting at stage 2 is both more computationally expensive and worse in terms of FID. Therefore, we set the cut-off point at stage 1.

inv. steps 25

inv. steps 50

inv. steps 100

inv. steps 1000

ImageNet-R-TI2I real

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

0.280

0.278

CLIP[]

Baseline

0.276

0.274

0.272

0.270

14.4s 10.9s

17.0s 12.7s

22.0s 16.2s

113.4s (PnP's default)

-24.3% -25.3% -26.4% "a toy of a jeep" (ref)

0.050

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

WithClockwork

0.048

DINO[]

0.046

PnP Baseline

PnP with Clockwork

0.044

200 400 600 800 TFLOPs

Figure 6. Left: text-guided image editing qualitative results comparing the baseline Plug-and-Play to Clockwork with identity adaptor when using the reference image (bottom right) with the target prompt “an embroidery of a minivan”. Across configurations, applying Clockwork enables matching or outperforming the perceptual quality of the baseline Plug-and-Play while reducing latency by a significant margin. Right: Clockwork improves the efficiency of text-guided image translation on the ImageNet-R-TI2I real dataset. We evaluate both the baseline and its Clockwork variant at different number of DDIM inversion steps: 25, 50, 100, 500 and 1000. The number of DDIM generation steps is fixed to 50 throughout, except for 25 where we use the same number of generation steps as inversion steps.

Steps FID [↓] CLIP [↑] GFLOPs Efficient UNet 8 24.22 0.302 1187 Adaptor Architecture

Identity (0) 8 24.36 0.290 287 ResNet (14M) 8 23.21 0.296 301 UNet (152M) 8 23.18 0.296 324 UNet-light (3M) 8 23.87 0.294 289

###### Adaptor Clock

- Steps {2,4,6,8} 8 23.21 0.296 301 Steps {5,6,7,8} 8 28.07 0.286 301
- Steps {3,4,5,6} 8 33.10 0.271 301

###### UNet cut-off

- Stage 1 (res 32x32) 8 23.21 0.296 301
- Stage 2 (res 16x16) 8 24.49 0.296 734

Table 3. Ablations of Clockwork components. We use 512 × 512 MS-COCO 2017-5K, a clock of 2 and Efficient UNet as backbone. FLOPs are reported for 1 forward step of UNet with adaptor.

Training scheme and robustness. As outlined in Sec. 4.4, the adaptor ϕθ can be trained using 1) the regular distillation setup which employs forward noising of an image or 2) by unrolling complete sampling trajectories conditioned on a prompt. We compare the two at specific inference steps that use the same clock. Figure 7 shows that generation unroll performs on par with regular distillation at higher inference steps (6, 8, 16), but performs significantly better at 4 steps, which is the low compute regime that our work targets.

50

CLIP[]

FID[]

40

Generation Unroll

Regular Distillation

30

4 6 8 10 12 TFLOPs

0.300

0.275

0.250

0.225

4 6 8 10 12 TFLOPs

Figure 7. Training scheme ablation. We observe that our training with unrolled trajectories is generally on par with regular distillation, but performs significantly better in the low compute regime (4 steps). We use 512 × 512 MS-COCO 2017-5K, a clock of 2 and Efficient UNet as backbone.

### 6. Conclusion

We introduce a method for faster sampling with diffusion models, called Clockwork Diffusion. It combines model and step distillation, replacing lower-resolution UNet representations with more lightweight adaptors that reuse information from previous sampling steps. In this context, we show how to design an efficient adaptor architecture, and present a sampling scheme that alternates between approximated and full UNet passes. We also introduce a new training scheme that is more robust than regular step distillation at very small numbers of steps. It does not require access to an image dataset and training can be done in a day on a single GPU. We validate our method on text-to-image generation and text-conditioned image-to-image translation [48]. It can be applied on top of commonly used models like Stable Diffusion [34], as well as heavily optimized and dis-

tilled models, and shows consistent savings in FLOPs and runtime at comparable FID and CLIP score.

Limitations. Like in step distillation, when learned, Clockwork is trained for a fixed operating point and does not allow for drastic changes to scheduler or sampling steps at a later time. While we find that our unrolled trainings works better than regular distillation at low steps, we have not yet fully understood why that is the case. Finally, we have only demonstrated improvements on UNet-based diffusion models, and it is unclear how this translates to e.g. ViT-based implementations.

### References

- [1] David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbot, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. arXiv preprint arXiv:2303.04248, 2023. 2
- [2] Yu-Hui Chen, Raman Sarokin, Juhyun Lee, Jiuqiang Tang, Chuo-Ling Chang, Andrei Kulik, and Matthias Grundmann. Speed is all you need: On-device acceleration of large diffusion models via gpu-aware optimizations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4650–4654, 2023. 2
- [3] Jiwoong Choi, Minkyu Kim, Daehyun Ahn, Taesu Kim, Yulhwa Kim, Dongwon Jo, Hyesung Jeon, Jae-Joon Kim, and Hyungjun Kim. Squeezing large-scale diffusion models for mobile. arXiv preprint arXiv:2307.01193, 2023. 2
- [4] Kamil Deja, Anna Kuzina, Tomasz Trzci´nski, and Jakub M. Tomczak. On analyzing generative and denoising capabilities of diffusion-based deep generative models, 2022. 1, 2, 3
- [5] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis, 2021. 1
- [6] Tim Dockhorn, Robin Rombach, Andreas Blatmann, and Yaoliang Yu. Distilling the knowledge in diffusion models. In CVPR Workshop Generative Models for Computer Vision,

2023. 2, 3

- [7] Tim Dockhorn, Arash Vahdat, and Karsten Kreis. GENIE: Higher-Order Denoising Diffusion Solvers. In Advances in Neural Information Processing Systems, 2022. 2
- [8] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate post-training quantization for diffusion models. arXiv preprint arXiv:2305.10657,

2023. 2

- [9] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV,

2021. 5

- [10] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image

editing with cross attention control, 2022. 6

- [11] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems, 2017. 5
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 1, 2, 3
- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. 1
- [14] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 4, 5
- [15] Alexia Jolicoeur-Martineau, Ke Li, R´emi Pich´e-Taillefer, Tal Kachman, and Ioannis Mitliagkas. Gotta go fast when generating data with score-based models. arXiv preprint arXiv:2105.14080, 2021. 2
- [16] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022. 2
- [17] Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. On architectural compression of text-toimage diffusion models. arXiv preprint arXiv:2305.15798,

2023. 1, 2, 3, 5, 6, 7

- [18] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2022. 6
- [19] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. 5
- [20] Jan Koutn´ık, Klaus Greff, Faustino Gomez, and J¨urgen Schmidhuber. A Clockwork RNN, 2014. 3, 4
- [21] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. arXiv preprint arXiv:2306.00980, 2023. 1, 2, 5, 6, 7
- [22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 2, 5
- [23] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, Feb

2022. 2

- [24] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2
- [25] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. arXiv preprint arXiv:2309.06380, 2023. 2, 6, 7
- [26] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. NeurIPS,

2022. 1, 2

- [27] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 1, 2, 5, 6
- [28] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed,

2021. 2

- [29] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, 2023. 1, 2, 5, 6, 7
- [30] Nilesh Prasad Pandey, Marios Fournarakis, Chirag Patel, and Markus Nagel. Softmax bias correction for quantized generative models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1453–1458, 2023. 2
- [31] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Proceedings, SIGGRAPH ’23. ACM, July 2023. 6
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 4
- [33] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020. 5, 15
- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 2, 3, 4, 5, 8
- [35] O. Ronneberger, P.Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 1
- [36] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding, 2022. 1
- [37] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2021. 1, 2, 5, 6
- [38] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 2022. 5
- [39] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1972–1981, 2023. 2
- [40] Evan Shelhamer, Kate Rakelly, Judy Hoffman, and Trevor Darrell. Clockwork convnets for video semantic segmenta-

- tion. In Gang Hua and Herv´e J´egou, editors, ECCV Workshops, 2016. 3, 4
- [41] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. arXiv preprint arXiv:2309.11497, 2023. 1, 2, 3
- [42] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, 2015. 3
- [43] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2020. 1
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 2
- [45] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. 2
- [46] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, Nov 2020. 2, 6
- [47] Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Splicing vit features for semantic appearance transfer. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2022. 5
- [48] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, 2023. 1, 2, 5, 6, 8, 13, 17, 19, 20
- [49] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 6, 7
- [50] Qinsheng Zhang, Molei Tao, and Yongxin Chen. gddim: Generalized denoising diffusion implicit models, 2022. 2
- [51] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. arXiv preprint arXiv:2302.04867, 2023. 2

### A. Clockwork details

Steps FID CLIP TFLOPs

DPM++ 8 24.22 0.302 9.5 + Clockwork 8 23.21 0.296 5.9

UNet Architecture In Fig. 8 we show a detailed schematic of the SD UNet architecture. The parts in pink are replaced by our lightweight adaptor. We also show parameter counts and GMACs per block. In ablations we varied the level at which we introduce the adaptor, as shown in Table 3 of the main body. There we compare “Stage 1 (res 32x32)” (our default setup) and “Stage 2 (res 16x16)” (a variant where DOWN-1 and UP-2 remain in the model), finding better performance for the former. Interestingly, our sampling analysis suggested that introducing the adaptor at such a high resolution, replacing most parts of the UNet, should lead to poor performance. However, this is only true if we replace multiple consecutive steps (see adaptor clock ablations in Table 3 of the main body). By alternating adaptor and full UNet passes we recover much of the performance, and can replace more parts of the UNet than would otherwise be possible.

DPM 8 24.32 0.301 9.5 + Clockwork 8 23.24 0.296 5.9

PNDM 8 35.64 0.272 9.5 + Clockwork 8 33.15 0.280 5.9

DDIM 8 34.72 0.287 9.5 + Clockwork 8 38.38 0.280 5.9

Table 4. Clockwork works with different schedulers.

### B. Ablations

Scheduler. We evaluate Clockwork across multiple schedulers: DPM++, DPM, PNDM, and DDIM. With the exception of DDIM, Clockwork improves FID at negligible change to the CLIP score, while reducing FLOPs by 38%.

Adaptor inputs. We vary the inputs to the adaptor ϕθ. In the simplest version, we only input rtin and the time embedding. It leads to a poor FID and CLIP. Using only rtout+1 provides good performance, indicating the importance of using features from previous steps. Adding rtin helps for a better performance, showcasing the role of the early highres layers of the UNet. Finally, adding the pooled prompt embedding textemb doesn’t change FID and CLIP scores.

Adaptor Architecture In Fig. 9 we show a schematic of our UNet-like adaptor architecture, as discussed in ablations (Section 5.4 of the main paper). In addition to our ResNet-like architecture (Fig. 3 of the main paper) We tried

- 1) a UNet-like convolutional architecture with 640 channels in each block and 4 ResNet blocks in the middle level (N = 4), 2) a lighter variant of it with 96 channels and
- 2 ResNet blocks in the middle level. While all adaptors provide comparable performance, the ResNet-like adaptor obtains better quality-complexity trade-off.

Model Distillation In Tab. 5 In previous ablation, we used clock of 2. In this ablation, we explore the option to distill the low resolution layers of ϵ into the adaptor ϕθ, for all the steps. Here we train the adaptor in a typical model distillation setting - i.e., the adaptor ϕθ receives as input the downsampled features at current timesteps rtin along with the time and text embeddings temb and textemb. It learns to predict upsampled features at current timestep rtout. During inference, we use the adaptor during all sampling steps. Replacing the lower-resolution layers of ϵ with a lightweight adaptor results in worse performance. It is crucial that the adaptor be used with a clock schedule along with input from a previous upsampled latent.

Training We provide pseudocode for our unrolled training in Algorithm 1.

Algorithm 1 Adaptor training with unrolled trajectories Require: Teacher model ϵ Require: Adaptor ϕθ Require: Prompt set P Require: Clock schedule C(t)

for Ne epochs do PD ← RandomSubsetD(P) ▷ optional D ← GenerateTrajectories(PD,ϵ) for all Trajectory & prompt (T,text) ∈ D do

Timings for different GPU models In Tab. 6 we report latency of different UNet backbones on different GPU models.

for all (t,rtin,rtoutrtout+1) ∈ T do

if C(t) = 1 then rˆtout ← ϕθ rtin,rtout+1,temb,textemb L ← ∥rtout − rˆtout∥2 θ ← θ − γ∇L

### C. Additional perturbation analyses

In Section 3 of the main body, we introduced perturbation experiments to demonstrate how lower-resolution features in diffusion UNets are more robust to perturbations, and thus amenable to distillation with more lightweight components. As a quick reminder, we mix a given representation with a random noise sample by assuming that the feature

end if end for

end for end for

Figure8.DetailedviewoftheSDUNetarchitecture.Wereplacethepink/purplepartswithalightweightadaptor,theinputtowhichhasspatialresolution.Forthe×3232

Conv2d ResNet

18.81Mparams(2.19%)

66.0GMACs(19.48%%)

UP-3

Block Transformer

Block Transformer

Block Transformer

ResNet

ResNet

640x64x64

79.1GMACs(23.35%)

71.41Mparams(8.31%)

UP-2

Block TransformerUpSample

Block Transformer

Block Transformer

1280x32x32

ResNet

ResNet

ResNet

258.3Mparams(30.06%)

75.1GMACs(22.18%)

UP-1

Block TransformerUpSample

ablationsinthemainbodywealsotriedleavingDOWN-1andUP-2inthehigher-resolutionpath,onlyreplacingblocksbelow.

Block Transformer

Block Transformer

1280x16x16

ResNet

ResNet

ResNet

162.2Mparams(18.88%)

12.9GMACs(3.81%)

UP-0

Block UpSample

ResNet

ResNet

ResNet

Block

Block

1280x8x8

ResNet

Block

97.04Mparams(11.29%)

6.03GMACs(1.78%)

Block Transformer

MID

ResNet

1280x8x8

ResNet

ResNet

Block

Block

1280x8x8

62.28Mparams(7.25%)

3.78GMACs(1.12%)

DOWN-3

Block Transformer

Block Transformer

Sample

Down

140.0Mparams(16.29%)

31.5GMACs(9.29%)

DOWN-2

ResNet

ResNet

640x16x16

Block Transformer

Block Transformer

Sample

Down

36.82Mparams(4.28%)

31.3GMACs(9.24%)

DOWN-1

ResNet

ResNet

320x32x32

Block Transformer

Block Transformer

Sample

Down

10.52Mparams(1.22%)

859.52Mparams

32.9GMACs(9.72%)

338.61GMACs

DOWN-0

TOTAL

ResNet

ResNet

320x64x64

Conv2d

[Figure 44]

- Figure 9. Architecture of a variant of the adaptor: UNet and UNetlight. For UNet we set C = 640 and N = 4, while for UNet-light we set C = 96 and N = 2.

Steps FID [↓] CLIP [↑] GFLOPs Distilled Efficient UNet 8 25.75 0.297 150 Adaptor Input

rtIN + temb 8 40.73 0.262 150 rtOUT+1 + temb 8 24.76 0.295 150

+ rtIN 8 24.45 0.295 150 + textemb 8 24.45 0.295 150

###### Model Distillation

rtIN + temb + textemb 8 117.64 0.06 150

- Table 5. Ablation of adaptor inputs. We use the MSCOCO-2017 dataset, Distilled Efficient UNet as backbone and a clock of 2 (except for Model distillation where we use adaptor for all the steps)

. FLOPs are reported for 1 forward step of UNet with adaptor.

Latency [ms] RTX 3080 RTX 2080Ti V100 A100

SD v1.5 454 589 453 235 + Clockwork 341 440 360 183

(−24.9%) (−25.3%) (−20.5%) (−22.1%) Eff. UNet 330 427 312 176

+ Clockwork 213 268 212 118

(−35.5%) (−37.2%) (−32.1%) (−33.0%) Eff. UNet (distilled) 240 302 245 191

+ Clockwork 154 190 159 122 (−35.8%) (−37.1%) (−35.1%) (−36.1%)

- Table 6. Latency improvements [ms] using Clockwork on different GPU models. All measurements are averaged over 10 runs, using DPM++ with 8 steps and batch size 1 (distilled) or 2 (for classifierfree guidance).

map is normal f ∼ N(µf,σf2). We then draw a random sample z ∼ N(0,σf2) and update the feature map with:

f ← µf + √α · (f − µf) + √1 − α · z (3)

For the example in the main body we set α = 0.3, so that the signal is dominated by the noise. Interestingly, we can also fully replace feature maps with noise, i.e. use α = 0.0. The result is shown in Fig. 10. Changes are much stronger than before, but lower-resolution perturbations still result mostly in semantic changes. However, the output is of lower perceptual quality.

Inversion SD version 1.5 Sampler DDIM Inversion prompt “a <style> of an <instance>” Extract reverse False

Generation SD version 1.5 Sampler DDIM Guidance scale 15.0 (for both real and fake images) Negative prompt “ugly, blurry, black, low res, unrealistic”

τA 0.5 τf 0.8

Table 7. Plug-and-Play hyper-parameters in inversion and generation. τA and τf are expressed as fraction of the sampling trajectory. For instance, τf = 0.8 means that for the first 80% steps in the generation, convolutional features will be injected. If one uses 10 DDIM steps, this means that for the first 8 steps, convolutional features will be injected.

For the analysis in the main body, as well as Fig. 10, we perturb the output of the three upsampling layers in the SD UNet. We perform the same analysis for other layers in Fig. 11. Specifically, there we perturb the output of the bottleneck layer, the three downsampling layers, and the first convolutional layer of the network (which is also one of the skip connections). Qualitatively, findings remain the same, but perturbation of a given downsampling layer output leads to more semantic changes (as opposed to artifacts) compared to its upsampling counterpart.

Finally, we quantify the L2 distance to the unperturbed output as a function of the step where we start perturbation. Fig. 12 corresponds to the perturbations from the main body, while Fig. 13 shows the same but corresponds to the downsampling perturbations of Fig. 11. Confirming what we saw visually, perturbations to low-resolution layers result in smaller changes to the final output than the same perturbations to higher-resolution features.

### D. Text-Guided Image Editing

#### D.1. Implementation Details

We base our implementation of Plug-and-Play (PnP) [48] off of the official pnp-diffusers implementation. We summarize the different hyper-parameters used to generate the results for both the baseline and Clockworkvariant of PnP in Tab. 7. Additionally, while conceptually similar we outline a couple of important differences between what the original paper describes and what the code implements. Since we use this code to compute latency and FLOP, we will go over the differences and explain how both are computed. We refer to Fig. 14 for a visual reference of the implementation of the “pnp-diffusers”. For a better understanding, we encourage the reader to compare it to Fig. 2 from the PnP [48] paper.

Perturb from step 0 Perturb from step 1 Perturb from step 2 Perturb from step 3 Perturb from step 4 Perturb from step 5 Perturb from step 10 Perturb from step 15

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Low-resfeatures

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Mid-resfeaturesHigh-resfeatures

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

|[Figure 68]<br><br>[Figure 69]|
|---|

- Figure 10. Reproduction of Figure 2 from the main body, using α = 0.0 (where Figure 2 uses α = 0.3). This corresponds to full perturbation of the representation, i.e. the representation is completely replaced by noise in each step. Perturbation of low-resolution features still mostly results in semantic changes, whereas perturbation of higher-resolution features leads to artifacts.

[Figure 70]

Bottleneck

Perturb from step 0 Perturb from step 1 Perturb from step 2 Perturb from step 3 Perturb from step 4 Perturb from step 5 Perturb from step 10 Perturb from step 15

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Downsample2Downsample1Downsample0In-convolution

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

|[Figure 109]<br><br>[Figure 110]|
|---|

- Figure 11. Reproduction of Figure 2 from the main body, perturbation different layers. Figure 2 perturbs the outputs of the 3 upsampling layers in the SD UNet, here we perturb the outputs of the 3 downsampling layers as well as the bottleneck and the first input convolution. Qualitative findings remain the same.

100

L2Distancetounperturbedoutput

| |
|---|

75

| |
|---|

50

| |
|---|

25

Low-res features Mid-res features High-res features

0

0 2 4 6 8 10 12 14 16 18 20 Perturb from step ...

- Figure 12. L2 distance to the unperturbed output, when perturbing representations with noise (α = 0.7), starting after a given number of steps. This quantifies what is shown visually in Figure 2 in the main body. Lower-resolution representations are much more robust to perturbations, and converge to the unperturbed output faster.

0 2 4 6 8 10 12 14 16 18 20 Perturb from step ...

0

25

50

75

100

L2Distancetounperturbedoutput

| |
|---|

| |
|---|

Bottleneck

Downsample 2 Downsample 1 Downsample 0 In-convolution

- Figure 13. L2 distance to the unperturbed output, when perturbing representations with noise (α = 0.7), starting after a given number of steps. This quantifies what is shown visually in Figure 11. Lower-resolution representations are much more robust to perturbations, and converge to the unperturbed output faster.

When are features cached? The paper describes that the source image is first inverted, and only then features are cached during DDIM sampling. They are only cached at sampling step t falling within the injection schedule, which is defined by the two hyper parameters τf and τA which corresponds to the sampling steps until which feature and self-attention will be injected respectively. The code, instead of caching features during DDIM generation at time steps corresponding to injection schedule, caches during all DDIM inversion steps. This in theory could avoid running DDIM sampling using the source or no prompt. However as we will see in the next paragraph, since the features are not directly cached but the latents are, we end up spending the compute on DDIM sampling anyway.

What is cached? The paper describes the caching of spatial features from decoder layers ft4 along with their self-

attention Alt, where 4 and l indicate layer indices. The implementation trades off memory for compute by caching the

latent xt instead, and recomputes the activations on the fly by stacking the cached latent along the batch axis along with an empty prompt. The code does not optimize this operation and stacks such latent irrespective of whether it will be injected, which results in a batch size of 3 throughout the whole sampling trajectory: (1) unconditional cached latent forward (2) latent conditioned on target prompt and (3) latent conditioned on negative prompt. This has implications on the latency and complexity of the solution, and we reflected it on the FLOP count, we show the formula we used in Eq. (5).

Of note, this implementation which caches latents instead of features has a specific advantage for Clockwork, as it enables mismatching inversion and generation steps and clock. During inversion, when the latents are cached, it does not matter whether it is obtained using a full UNet pass or an adaptor pass. During generation, when the features should be injected, the cached latent is simply ran through the UNet to obtain the features on-the-fly. This is illustrated in Fig. 14 where features are injected at step t + 1 during the generation although the adaptor was used at the corresponding step during inversion.

How do we compute FLOP for PnP? To compute FLOP, we need to understand what data is passed through which network during inversion and generation. Summarizing previous paragraphs, we know that:

- • inversion is ran with a batch size of 1 with the source prompt only.
- • generation is ran with a batch size of 3. The first element in the batch corresponds to the cached latent and the empty prompt. The other two corresponds to the typical classifier-free guidance UNet calls using the target prompt and negative prompt.
- • both during inversion and generation, if the adaptor is used, only the higher-res part of the original UNet will be run, ϵH.

Let us denote N and C the number of steps and the clock, the indices I and G standing for inversion and generation respectively. We first count the number of full UNet pass in each, using integer division NIfull = NI div CI (we follow similar logic for NGfull. Additionally, we use FLOP estimate for a single forward pass with batch size of 1 in UNet, Fϵ = 677.8 GFLOPs, and UNet with identity adaptor, Fϵ

= 228.4 GFLOPs. The estimates are obtained using the DeepSpeed library [33]. Finally, we obtain the FLOP count F as follows:

H+ϕ = Fϵ

H

INVERSION source text 𝑡𝑒𝑥𝑡

source inverted latent 𝑥

“a toy of a jeep”

[Figure 111]

[Figure 112]

𝑥 𝑥 𝑥

[Figure 113]

[Figure 114]

𝜖 𝜙

source image 𝐼

cached latent injection

GENERATION

“”

[Figure 115]

empty text 𝑡𝑒𝑥𝑡

𝑥 𝑥

feature / self-attention injection

target image 𝐼

[Figure 116]

[Figure 117]

𝑥 𝑥 𝑥

“a cartoon of a jeep”

target text 𝑡𝑒𝑥𝑡

- Figure 14. Overview of the actual diffusers implementation of Plug-and-Play, which contrary to what the paper describes caches latent during inversion, not intermediate features during generation. The features to be injected are re-computed from the cached latents on-thefly during DDIM generation sampling. The red arrows indicate injection, the floppy disk icon indicate that only the latent gets cached / saved to disk. Inversion and generation are ran separately, all operations within each are ran in-memory.

turns out all layers which undergo injection are skipped if adaptor ϕ is ran instead of ϵL. Hence, when adaptor is ran during generation it means no features are being injected. As the number of inversion and generation steps decrease, the effect of skipping injection are more and more visible, in particular structural guidance degrades. One could look into caching and injecting adaptor features to avoid losing structural guidance. Note however that this would have no effect on complexity, and might only affect PnP + Clockworkperformance in terms of CLIP and DINO scores at lower number of steps. Since optimizing PnP’s performance at very low steps was not a focus of the paper, we did not pursue this thread of work.

FI = NIfull · Fϵ + (NI − NIfull) · Fϵ

(4) FG = 3 · NGfull · Fϵ + (NG − NGfull) · Fϵ

H

##### (5) F = FI + FG (6)

H

How do we compute latency for PnP? As described in Section 5, we only compute latency of the inversion and generation loops using PyTorch’s benchmark utilities. In particular, we exclude from latency computation any “fixed” cost like VAE decoding and text encoding. Additionally, similar to the FLOP computation, we did not perform optimization over the official PnP implementation, which leads to a batch size of 1 in the inversion loop, and a batch size of 3 in the generation loop.

Possible optimizations. The careful reader might understand that there are low hanging fruits in terms of both latency and FLOP optimizations for PnP. First, if memory would allow, one could cache the actual activations instead of the latent during inversion, which would allow not rerunning the latent through the UNet at generation time. Sec-

Interplay between injection and adaptor. The adaptor replaces the lower resolution part of the UNet ϵL. Based on where we split the UNet between low- and high-res, it

ond, it would be simple to modify the generation loop code not to stack the cached latent when t does not fall within the injection schedule. If implemented, a substantial amount of FLOP and latency could be saved on the generation, as the default PnP hyper parameters τf and τA lead to injection in only the first 80% of the sampling trajectory. Note however that both of these optimizations are orthogonal to Clockwork, and would benefit both the baseline and Clockworkimplementations of PnP, which is why we did not implement them.

#### D.2. Additional Quantitative Results

We provide additional quantitative results for PnP and its Clockworkvariants. In particular, we provide CLIP and DINO scores at different clocks and with a learned ResNet adaptor. In addition to the ImageNet-R-TI2I real dataset results, we report scores on ImageNet-R-TI2I fake [48]. In the Fig. 15, we can see how larger clock size of 4 enables bigger FLOP savings compared to 2, yet degrade performance at very low number of steps, where both CLIP and DINO scores underperform at 10 inversion and generation steps. It is interesting to see that the learned ResNet adaptor does not outperform nor match the baseline, which is line with our ablation study which shows that Clockworkworks best for all schedulers but DDIM at very low number of steps, see Tab. 4. We can see that results transfer well across datasets, where absolute numbers change when going from ImageNet-RTI2I real (top row) to fake (bottom row) but the relative difference between methods stay the same.

#### D.3. Additional Qualitative Results

We provide additional qualitative examples for PnP for ImageNet-R-TI2I real in Fig. 16 and Fig. 17. We show examples at 50 DDIM inversion and generation steps.

### E. Additional examples

We provide additional example generations in this section. Examples for SD UNet are given in Fig. 18, examples for Efficient UNet in Fig. 19, and those for the distilled Efficient UNet in Fig. 20. In each case the top panel shows the reference without Clockwork and the bottom panel shows generations with Clockwork. Fig. 18 includes the same examples already shown in the main body so that the layout is the same as for the other models for easier comparison. The prompts that were used for the generations are the following (left to right, top to bottom), all taken from the MSCOCO 2017 validation set:

- • “a large white bear standing near a rock.”
- • “a kitten laying over a keyboard on a laptop.”
- • “the vegetables are cooking in the skillet on the stove.”

- • “a bright kitchen with tulips on the table and plants by the window ”
- • “cars waiting at a red traffic light with a dome shaped building in the distance.”
- • “a big, open room with large windows and wooden floors.”
- • “a grey cat standing in a window with grey lining.”
- • “red clouds as sun sets over the ocean”
- • “a picnic table with pizza on two trays ”
- • “a couple of sandwich slices with lettuce sitting next to condiments.”
- • “a piece of pizza sits next to beer in a bottle and glass. ”
- • “the bust of a man’s head is next to a vase of flowers.”
- • “a view of a bathroom that needs to be fixed up.”
- • “a picture of some type of park with benches and no people around.”
- • “two containers containing quiche, a salad, apples and a banana on the side.”

Text-Image Similarity CLIP [ ]

Structure Distance DINO [ ]

0.050

0.28

0.045

ImageNet-R-TI2Ireal

0.27

0.040

0.26

0.035

0.25

0.030

PnP Baseline

PnP with Clockwork (identity clock of 2) PnP with Clockwork (identity clock of 4) PnP with Clockwork (learned clock of 2)

0.025

0.24

0.020

0.050

0.28

0.045

ImageNet-R-TI2Ifake

0.27

0.040

0.26

0.035

0.25

0.030

0.025

0.24

0.020

0 200 400 600 800 TFLOPs

0 200 400 600 800 TFLOPs

- Figure 15. Additional quantitative results on ImageNet-R-TI2I real (top) and fake (bottom) for varying number of DDIM inversion steps: [10, 20, 25, 50, 100, 200, 500, 1000]. We use 50 generation steps except for inversion steps below 50 where we use the same number for inversion and generation.

|a tattoo of a lion|
|---|

|a graffiti of a jeep|
|---|

|an embroidery of a bear|
|---|

|a sculpture of a castle|
|---|

|an image of a violin|
|---|

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

BaselineWithClockworkReference

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

|a sketch of a pizza|
|---|

|an embroidery of a bustard|
|---|

|a graphic of a eel|
|---|

|an origami of a tiger|
|---|

|a tattoo of a penguin|
|---|

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

BaselineWithClockwork

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

- Figure 16. Examples from ImageNet-R-TI2I real from Plug-and-Play [48] and its Clockworkvariant. We use 50 DDIM inversion and generation steps, and a clock of 2. Images synthesized with Clockworkare generated 34% faster than the baseline, while being perceptually close if at all distinguishable from baseline.

|a sketch of a jeep|
|---|

|a painting of a tiger|
|---|

|a painting of a penguin|
|---|

|an art of a castle|
|---|

|an image of a bobcat|
|---|

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

BaselineWithClockworkReference

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

|a photo of a cat|
|---|

|an embroidery of a hummingbird|
|---|

|a tattoo of a violin|
|---|

|an embroidery of a goldfish|
|---|

|an art of a baloon|
|---|

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

BaselineWithClockwork

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

- Figure 17. Examples from ImageNet-R-TI2I fake from Plug-and-Play [48] and its Clockworkvariant. We use 50 DDIM inversion and generation steps, and a clock of 2. Images synthesized with Clockworkare generated 34% faster than the baseline, while being perceptually close if at all distinguishable from baseline.

[Figure 178]

[Figure 179]

###### Figure 18. Additional example generations for SD UNet without (top) and with (bottom) Clockwork. We include the examples shown in the main body so that the layout of this figure matches that of Fig. 19 and Fig. 20.

[Figure 180]

[Figure 181]

###### Figure 19. Example generations for Efficient UNet without (top) and with (bottom) Clockwork.

[Figure 182]

[Figure 183]

###### Figure 20. Example generations for Distilled Efficient UNet without (top) and with (bottom) Clockwork.

