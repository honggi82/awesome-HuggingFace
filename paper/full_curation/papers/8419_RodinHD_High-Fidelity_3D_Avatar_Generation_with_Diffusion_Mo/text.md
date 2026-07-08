# arXiv:2407.06938v2[cs.CV]11Jul2024

## RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models

Bowen Zhang1∗ , Yiji Cheng2∗ , Chunyu Wang3† , Ting Zhang3 , Jiaolong Yang3 , Yansong Tang2 , Feng Zhao1 , Dong Chen3 , and Baining Guo3

1 University of Science and Technology of China

- 2 Tsinghua University
- 3 Microsoft Research Asia

[Figure 1]

[Figure 2]

3D avatars created from in-the-wild portrait images

[Figure 3]

[Figure 4]

“Blender Synthetic Avatar, brown hair, boy, green eyes, black and green jacket, stubble, leather"

Unconditional sampled avatar

Fig. 1: RodinHD generates detailed 3D avatars from single portrait images (dashed boxes) without compromising cross-view consistency (first row). It also supports textconditioned (second row left) or unconditional (second row right) generation.

Abstract. We present RodinHD, which can generate high-fidelity 3D avatars from a portrait image. Existing methods fail to capture intricate details such as hairstyles which we tackle in this paper. We first identify an overlooked problem of catastrophic forgetting that arises when fitting triplanes sequentially on many avatars, caused by the MLP decoder sharing scheme. To overcome this issue, we raise a novel data scheduling strategy and a weight consolidation regularization term, which improves the decoder’s capability of rendering sharper details. Additionally, we optimize the guiding effect of the portrait image by computing a finer-grained hierarchical representation that captures rich 2D texture cues, and injecting them to the 3D diffusion model at multiple layers via cross-attention. When trained on 46K avatars with a noise schedule optimized for triplanes, the resulting model can generate 3D avatars with notably better details than previous methods and can generalize to in-the-wild portrait input. See Fig. 1 for some examples. Project page: https://rodinhd.github.io/.

Keywords: 3D avatar generation · Diffusion · Catastrophic forgetting

⋆ Interns at Microsoft Research Asia. Equal contribution. †Corresponding author.

### 1 Introduction

High-fidelity 3D avatar generation has many applications in fields such as gaming and metaverse. Recent development of generative diffusion models [22, 24,27,52,55] and implicit neural radiance fields [44,71] has opened up new opportunities for automatic generation of 3D avatar [23,45,58,71] at scale. However, current methods struggle to generate fine details, which is a core challenge that has to be addressed. Otherwise, a “toy-like” avatar is less effective in delivering practical values.

The work of 3D generative diffusion models [3, 23, 31, 45, 50, 58, 71, 78, 79] usually follow a two-stage framework. First, they compute a proxy 3D representation of fixed length such as triplanes or volumes from the original unstructured meshes or point clouds so that they can be handled by diffusion models. A paired decoder is jointly learned to render 360◦ images from the representation. Different from the dominant NeRF fitting methods, the decoder here is shared among all avatars to decode novel generated triplanes. Then, they train diffusion models on the proxy representation to generate diverse avatars. However, they struggle to generate fine details such as sharp cloth textures and hair strands. To alleviate the problem, Rodin [71] applies a convolution refiner [72] to complement the missing details for each rendered image. Although the 2D refiner improves the visual quality in one view, it significantly compromises 3D consistency, which is not tolerable in many applications.

In this work, we introduce RodinHD, which aims to improving the fidelity of avatars without any refiners. We begin by empirically showing that fitting triplanes sequentially on a large number of avatars suffers from catastrophic forgetting, which can result in under-fitted decoders incapable of generating intricate details on novel triplanes. This occurs because the triplane of an avatar is typically trained for many iterations before switching to the next one, in order to reduce the data transfer costs between CPUs and GPUs. However, this process gradually causes the shared decoder to forget knowledge learned from previous avatars, leading to a lack of generalizability. Fig. 2 illustrates this issue with typical renderings of the resulting decoder. This problem has been largely overlooked in the literature, hindering the development of high-fidelity generation based on neural radiance fields.

To address this issue, we propose a novel data scheduling strategy called task replay, and a weight consolidation regularization term that effectively preserves the decoder’s capability of rendering sharp details. The idea of task-replay is to switch avatars more frequently so that each can be seen periodically for multiple times, preventing the decoder from over-fitting to a single avatar. The weight consolidation regularization term prevents the critical weights from deviating far from its consolidated values. As a result, knowledge learned from previous data can be retained during training. The method effectively alleviates the forgetting problem and improve the model’s capability to encode intricate details, paving the way for the subsequent generation step.

We train a cascaded diffusion model on the triplanes for conditional generation. It consists of a base model, which generates a low-resolution triplane

[Figure 5]

[Figure 6]

[Figure 7]

Identity4

Identity1

Identity9

PSNR=16.74

PSNR=8.30

PSNR=32.39

Fig. 2: Catastrophic forgetting. As training proceeds, decoder gradually forgets the knowledge learned on the previous avatars of 1&4 and is overly adapted to avatar 9.

conditioned on a portrait image, and an upsample model, which subsequently generates a high-resolution triplane. Differing from the previous work [71] which uses CLIP [54] to compute a global semantic token for the portrait image as conditions, we maximize its guiding effect by computing a finer-grained hierarchical representation to provide more detailed cues for the 3D diffusion model using a VAE-based image encoder. The multi-scale features are injected into different layers of U-Net via cross-attention, which significantly improves the coherence between the generated avatars and the portrait images. Fig. 3 shows the process. Besides, inspired by [9], we also optimize the noise schedule for the triplane considering its high redundancy in both spatial and channel dimensions.

We train the model on 46K digital avatars [73] of diverse identities, expressions, hairstyles, and clothing. We render high-quality images at the resolution of 1024 × 1024. The resulting model is capable of generating highly detailed avatars with clear clothing textures and hairstyles using a simple diffusion model without extra refinement models. While only validated on avatars, the proposed techniques are general and can be applied to other 3D generation tasks.

- 2 Related Work Early works in 3D generation have primarily focused on generating coarse
- 3D shapes, which are typically represented as meshes [38, 63], point clouds [1, 36], voxel grids [5,74], and implicit neural representations [51,60], using either GANs [19,32,80] or VAEs [34]. However, it remains unclear whether these methods can be effectively applied to generate complex 3D avatars with rich details.

3D-aware GANs [6, 7, 11, 14, 18, 49, 57, 66, 76, 77] are able to generate highresolution images with the aid of 2D upsampler and patch-based image discriminator. Nevertheless, they suffer from cross-view consistency [75]. Gram [76] performs upsampling on the surface manifold to promote multiview consistency and efficiency but it cannot handle large viewpoint changes and complex geometry. Moreover, they are prone to mode collapse due to training instabilities of GANs. Although score-distillation-based methods [10,39,53,62,64,65] are proposed to distill the 2D diffusion prior to a 3D representation with score function, they suffer from the Janus problem which prevents them from generating accurate geometry because the problem is under-determined.

Recently, diffusion models [16,26,61] have achieved notable success in textto-image synthesis, with a few on high-resolution image generation [17, 21, 43, 52,55,56,68]. Inspired by this, many recent works apply diffusion models for 3D generation [3,8,12,23,31,41,45,50,58,67,71,78]. They first compute a proxy 3D representation from the raw data, and then train a diffusion model on the proxy data with either text or image as conditions. However, most of them struggle to generate fine details such as cloth textures and hair strands, due to the lack of high-quality proxy data and a well-designed upsample network. Rodin [71] exploits a convolution refiner [72] to complement the missing details in each individually rendered image, but this compromises cross-view consistency.

Some recent work [17,21,43,52,55,56,68] have investigated high-resolution image generation, and obtained some interesting findings about optimal noise schedules, model architectures and capacity scaling principles. Despite this, we find the triplanes are essentially different from images, and directly transferring their findings barely works in our scenario. For instance, Stable Diffusion [55] uses a VAE to compress an image into a lower-resolution latent for diffusion. However, we observe that compressing triplanes by training a VAE will lose many high-frequency details in the renderings. Moreover, the considerations for designing the noise schedules are also different from those in the images.

### 3 Method

Our framework comprises two primary steps: fitting and modeling. In the first step, we fit a high-resolution triplane x0 ∈ R3×H

x×Wx×C for each avatar, and learn a decoder F, which is shared by all avatars, to render high-fidelity images from the triplanes. The parameters of Hx and Wx denote the height and width of triplanes, taking value on 512, and C is the number of channels, taking value on 32. In the second step, we train a 3D diffusion model G that can generate triplanes from random noises ϵ ∈ R3×H

x×Wx×C conditioned on a portrait image Ifront ∈ RH

I×WI×3. Fig. 3 shows the overall framework. During inference, we can render 360◦ high-resolution images ˆI ∈ RH

I×WI×3 from the following process:

G : (ϵ,Ifront) −−−−−−→Denosing x0, (1) F : x0 −−−−−−→Rendering ˆI . (2)

The primary challenge arises from the high-resolution nature of the triplane, presenting difficulties in both fitting and modeling. We will elaborate on several key design considerations in the following.

#### 3.1 Triplane Fitting

The increased resolution of the triplanes introduces a higher computational load, leading to potential bottlenecks in terms of both time and memory. Therefore we split this task into two stages to reduce computation cost. In the first

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Triplane Fitting Triplane Diffusion

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

VAE-based

[Figure 19]

[Figure 20]

Cross-Attention

Encoder

[Figure 21]

[Figure 22]

Consolidation

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Image Condition

Replay&

Rendered Image

[Figure 29]

[Figure 30]

[Figure 31]

Decoder

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Base Diffusion

Upsample Diffusion

[Figure 45]

[Figure 46]

Decoder

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Gaussian Noise

Triplane

##### Fig. 3: Overview of our method.

stage, we jointly train the MLP decoder and the triplanes on a smaller subset of avatars. In the second stage, we fix the decoder’s weights and fine-tune the triplane of each avatar independently, which can be performed in parallel. The first stage of getting an accurate yet generalizable MLP decoder is critical in order to generate details for all avatars including those generated ones.

The decoder F parameterized by ω learns identity-agnostic priors from a set of S avatars. Each avatar is represented as N multi-view renderings Ds = {(Is,n,cs,n)}Nn=1, where Is,n is the RGBA image and cs,n is the corresponding camera configuration. There are two critical issues overlooked in previous work [71]. 1) During training, batches typically comprise rays sampled from a single avatar due to limited GPU memory. Consequently, each avatar is treated as an independent task. To minimize data transfer between CPU and GPU, each avatar undergoes multiple training iterations until convergence before being replaced by the next one. As a result, the MLP may gradually forget the previously learned knowledge and become overly adapted to the current avatar. This is known as catastrophic forgetting in continual learning. Fig. 2 visualizes the phenomenon. 2) Additionally, the approach encounters training instabilities when switching between avatars due to the substantial gaps between them. This usually results in an under-fitted MLP that is incapable of decoding high-frequency details, even with the second triplane finetuning stage. Fig. 10 compares the differences. In the following, we formally describe the task-replay strategy and weight consolidation regularizer to address the above challenges.

Task replay. The core idea is to switch avatars more frequently, allowing each avatar to be seen periodically for multiple times and preventing the decoder from over-fitting to a single avatar. As a result, the decoder is exposed to triplanes of a variety of avatars, ensuring its generalization ability. To implement this strategy, each avatar is trained multiple times, with each time fitting for a shorter time without requiring convergence by tuning “outer_loop_iteration" and “inner_loop_iteration” in Algorithm 1, while the naive method trains each avatar only once, and fits it to convergence before switching to the next avatar. The task replay strategy proves effective in mitigating the risk of over-fitting to a single avatar, which may otherwise result in poor generalization performance.

Algorithm 1 The First Stage of Triplane Fitting

Require: dataset {Ds}Ss=1, triplane parameters {xs}Ss=1, shared MLP F parameters ω, IWC states {Ωs = 0, ωs,∗}Ss=1, learning rate α, β.

- 1: repeat
- 2: Sample from Ds and load triplane xs from disk
- 3: repeat
- 4: Sample rays r ∼ Ds
- 5: L(xs, ω; Ωs, ωs,∗) ← LDs + LIWC
- 6: xs ← xs − α∇xsL(xs, ω; Ωs, ωs,∗)
- 7: ω ← ω − β∇ωL(xs, ω; Ωs, ωs,∗)
- 8: until inner_loop_iteration
- 9: {Ωs, ωs,∗} ← (∇ωLDs)2, ω ▷ update state
- 10: until outer_loop_iteration

Weight consolidation. Learning the triplanes and the decoder also suffers from instabilities caused by the occasionally occurring large gradients when we switch between the avatars. To mitigate this issue, we introduce an Identity-aware Weight Consolidation (IWC) regularizer, a technique that stabilizes learning by consolidating knowledge and reducing drastic shifts in the learning landscape. Specifically, it uses the elastic weight consolidation regularizer [35] to prevent the most important weights of this avatar from deviating far from its consolidated values during training:

LFitting = LDs + DIWC

λ 2 i

Ωis ωi − ωis,∗ 2 ,

= LDs +

(3)

where LDs

is the conventional rendering loss between rendered images and ground-truth images, ωis,∗ is the ith MLP weight obtained when training the avatar in the previous epoch, and Ωis is the importance of ith weight, calculated by the ith diagonal element of the Fisher Information Matrix Ωs. Since the Fisher Information Matrix is equivalent to the second derivatives of the loss near a minimum and it can be computed from first-order derivatives, computing the IWC loss is efficient.

We find that our approach not only improves the details but also reduces the high-frequency components in the triplanes, making them easier to learn by diffusion models [37,43]. See Fig. 4 for comparison. We also use TV loss and L2 regularization on the triplanes to promote smoothness as in the prior work [71].

#### 3.2 Triplane Diffusion

We train a cascaded diffusion model to generate high-resolution triplanes. It consists of a base model, which generates a low-resolution triplane conditioned on a portrait image Ifront, and an upsample model, which subsequently generates a high-resolution triplane. During training, we first obtain the destructed triplane

Triplane w/ IWC - Image

Triplane w/ IWC - Triplane w/o IWC

0.0025

0.02

ProbabilityDifference

ProbabilityDifference

0.0020

0.00

0.0015

0.02

0.0010

0.0005

0.04

0.0000

0.06

0.0005

0.0010

0.08

0.0015

0 50 100 150 200 250 300

0 50 100 150 200 250 300

Amplitude

Amplitude

- Fig. 4: Frequency difference between two sources. Left: Triplanes have more high-frequency components than images. Right: Triplanes learned with our proposed IWC have fewer high-frequency components.

xt according to xt := αtx0 + σtϵ, where αt,σt define the noise schedule which determines the destruction strength of the signal, and t is a continuous number ranged from 0 to 1. We train the base and upsample diffusion models in two separate steps. For base diffusion training, we parameterize the diffusion model ϵˆθ to predict the noise [26] added to the low-resolution triplane xLRt :

0 ,ϵ ϵ ˆθ αtxLR0 + σtϵ,t,Ifront − ϵ 22 . (4) Despite the LLRsimple, we also use the variational lower bound to optimize the

LLRsimple = Et,xLR

negative log-likelihood of estimated distribution by incorporating LLRvlb following [48] for higher generation quality. Our upsample diffusion model is learned

to enhance the high-fidelity details of the low-resolution triplane, which is conditioned on both xLR0 and Ifront. We directly parameterize the model to predict the noiseless input xHR0 :

0 ,ϵ x ˆθ αtxHR0 + σtϵ,t,Ifront,xLR0 − xHR0 22 . (5)

LHRsimple = Et,xHR

To ensure the rendered images of our generated triplanes have compelling visual quality, we also adopt image-level supervision when training the upsample model inspired by previous works [71]. Specifically, we penalize the discrepancy between the rendered image patch ˆIpatch from predicted triplane xˆθ and the ground truth image patch Ipatch:

LHRImage = LPixel + LPerc

Ψl(ˆIpatch) − Ψl(Ipatch)

= Et,ˆI

(

patch

l

2 2

( ˆIpatch − Ipatch

+ Et,ˆI

),

patch

2 2

)

(6)

where Ψl denotes the multi-scale feature extracted using a pre-trained VGG [59]. Ensuring the diffusion model scales well with the increased resolution is crucial. As high-resolution triplanes involve a large number of parameters, processing

[Figure 51]

[Figure 52]

8 channels 32 channels

- Fig. 5: Rendered images from triplanes with 8 and 32 channels, respectively. The triplanes are destructed with the same noise level (logSNR(t) = 0.57). The 32-channel triplane has larger redundancy so it is less destructed.

such vast amounts of information efficiently can be a substantial hurdle. Next, we introduce improvements to our diffusion model. Multi-scale image feature conditioning. It is difficult to hallucinate detailed

- 3D avatars from scratch. So, we propose to supply ample details from the portrait image to alleviate the difficulty. Previous works such as [71] compute a global semantic token using the pre-trained CLIP image encoder [54], which results in a substantial loss of detailed information.

To fully utilize the information of the portrait, we compute a multi-scale feature representation using a pre-trained Variational Autoencoder [52]. Since the VAE is trained to accurately reconstruct the input images, the low-level visual details are well preserved in the latent features. Formally, we denote the VAE encoder as E, and the frontal portrait image Ifront ∈ RH×W×3. We compute the conditional signals for our diffusion model by:

{y1,y2,y3} = E(Ifront), (7)

where y1 ∈ RH2 ×W2 ×C,y2 ∈ RH4 ×W4 ×2C, and y3 ∈ RH8 ×W8 ×4C denote the multiscale spatial features, and C is the based channel dimension.

Since the 2D portrait is not aligned with the 3D triplane, directly harmonizing the conditional signals with diffusion U-Net features via concatenation [27,47] or addition [81] is problematic. Therefore, we elect to perform cross-attention [69]. In this way, the network is learned to automatically discover the spatial correspondence between triplanes and 2D images.

Formally, let xi be the feature maps of the i-th attention resolution in U-Net, we inject yi by

{yik}Kk=1 = PatchPartition(yi),

(8)

x′i = CrossAttn(xi;{yik}Kk=1),

where K and x′i are the number of patches and the output features, respectively. For the base model, the conditions are injected into both encoder and decoder layers that have the same resolutions. For the upsample model, the conditions are only injected into the middle latent features to reduce computation costs.

Base Diffusion

Default Linear Ours

10

0

−10

−20

0 0.2 0.4 0.6 0.8 1

Upsample Diffusion

Default Linear Ours

0

−10

−20

−30

0 0.2 0.4 0.6 0.8 1

- Fig. 6: Optimized noise schedule. LogSNR comparison between the default and our optimized noise schedules for the base (left) and upsample (right) networks.

Optimized noise schedule. In high-resolution image generation, some work [9, 20,28,40] find that using a stronger noise is critical to retain the learning difficulty for images with higher resolutions. We similarly study the optimal noise schedule for triplanes in 3D generation. Compared to images, triplanes have large spatial resolutions and channel numbers, and even long-range dependencies introduced by 3D correspondence. All these factors increase the redundancy in the representation. Fig. 5 shows an example. We fit two triplanes with the same spatial resolution but different channel dimensions (8 vs. 32), and add the same level of noises (SNR(t) = αt2/σt2) to them, respectively. However, we can see that the rendered images are destructed differently. The triplane with fewer channels suffers from more disruption compared with the one with larger channels.

Considering the larger redundancy in triplanes, we propose to apply a stronger noise to fully destruct triplanes to prevent the model from under-training. To be more specific, we utilize the adjusted cosine noise schedule [9,48] in our base diffusion and a much stronger sigmoid noise schedule [9,30] in the upsample diffusion stage. The adjusted noise schedules are shown in Fig. 6. They are much stronger than the default linear noise scheduling [26] designed for low-resolution images e.g. 32 × 32,64 × 64.

### 4 Experiments

#### 4.1 Dataset and Metrics

We conduct experiments on 46K avatars created from Blender [73]. For each avatar, we uniformly render 300 multi-view images at the resolution of 1024 × 1024. We evaluate our model’s conditional and unconditional generation capability, respectively. For image-conditioned generation, we compute numerical results using the common metrics of FID [25], LPIPS [82], and Structural Similarity Index Measure (SSIM) between 5K rendered multi-view images from generated avatars with ground-truth images. For unconditional generation, we report FID of 5K rendered images from randomly sampled avatars. We also measure cross-view consistency by fitting a NeuS model [70] from the multi-view renderings of the generated avatars.

[Figure 53]

Reference Rodin [71] Our RodinHD

Fig. 7: Avatars generated by different methods.

Table 1: Quantitative results of conditional avatar generation.

Models FID↓ LPIPS↓ SSIM↑

Rodin [71] 33.20 0.323 0.758 Ours 26.49 0.299 0.765

#### 4.2 Implementation Details

The triplane resolution is 512 × 512 and its channel number is 32. We randomly select 64 avatars for jointly training the triplanes and the NeRF decoder for two days on 8 Tesla V100 GPUs, with task replay and IWC regularization applied. Then, we fit the triplanes for all avatars independently, with each taking about 15 minutes. We adopt the U-Net architecture [16] as diffusion backbone. For the configuration of conditional feature injection, please refer to supplementary material. We utilize conditional augmentation [27, 71] when training the upsample diffusion to reduce the domain gap between training and inference. Both our base and upsample models are trained on 32 Tesla V100 (32G) GPUs, with batch sizes 96 and 32 respectively. We randomly drop the conditional features with 20% probability, which enables us to perform classifier-free guidance and unconditional generation.

#### 4.3 Main Results

Conditional generation. We compare our approach with Rodin [71]. We report the results of Rodin without 2D refinement for fair comparison. As shown in Fig. 7, our model captures the detailed appearance and vivid expression of the given portrait, benefiting from high-quality triplane fitting and strong guidance of the proposed multi-scale image feature conditioning. On the contrary, Rodin fails to synthesize details of avatars, e.g., hairstyles and closed eyes. Moreover, our model enables us to directly render compelling images in high resolution without any 2D refinement, demonstrating the strong capacity of the proposed

[Figure 54]

[Figure 55]

[Figure 56]

EG3D [6] Rodin [71] Our RodinHD

Fig. 8: Qualitative results of unconditional avatar generation.

##### Table 2: Quantitative results of unconditional avatar generation. The subscript ∗ indicates that 2D refinement is applied to the rendered images.

Pi-GAN GIRAFFE EG3D∗ Rodin∗ Rodin Ours FID ↓ 78.3 64.6 40.5 30.29 45.70 32.62

model. Furthermore, we achieve the best results across all metrics as shown in Tab. 1, demonstrating the superiority of our model.

Unconditional generation. We compare our method with the state-of-theart methods, including both 3D-aware GANs [6, 7, 49] and 3D-diffusion-based Rodin [71]. We present the results of Rodin before and after the 2D refiner (denoted as Rodin∗), respectively. Tab. 2 shows the results. Our approach achieves significantly better results than the methods except Rodin∗ which uses a 2D refiner. However, as we will discuss in the following, our method achieves notably better 3D consistency. We also provide visual comparison in Fig. 8. While prior arts tend to generate blur rendering, our model is able to provide complex and diverse avatars with rich details, e.g., hair and clothing.

3D consistency. We evaluate the 3D consistency of the generated results by visualizing the spatiotemporal textures following [76]. We include Rodin∗, ours and ground-truth triplane renderings for comparison in Fig. 9. As the camera moving smoothly, the texture of the fixed horizontal line is expected to be smooth and nature as GT triplane renderings. However, the spatiotemporal textures of Rodin∗ suffer from obvious flickering, which suggests inconsistency across views (see video in supplementary for more intuitive comparison). In contrast, our model produces smooth and natural results as GT, demonstrating the strong consistency of the proposed model. Moreover, the renderings of Rodin∗ fail to maintain the skin tone of the input while our model provides more faithful results. We also quantitatively measure the 3D consistency by training a multiview reconstruction method NeuS [70] on 300 rendering images of Rodin∗ and ours respectively. The results of our model achieve significantly better metrics

Table 3: 3D consistency measured by the fitting quality of NeuS [70].

Models PSNR↑ LPIPS↓ SSIM↑

Rodin∗ [71] 31.73 0.051 0.973 Ours 35.46 0.041 0.975

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

##### GT Rodin∗ [71] Our RodinHD

Fig. 9: Visual comparison of 3D consistency akin to the Epipolar Line Images [4]. We visualize the stacked texture of a fixed line as the camera smoothly rotates horizontally. Our model yields smooth and natural texture as GT, whereas a clear flickering pattern is shown in Rodin’s results, indicating the 3D inconsistency using 2D refinement.

thanks to the 3D consistency, whereas Rodin∗ obtains worse results since the 2D refiner breaks the 3D consistency.

#### 4.4 Ablation Study

We first study the factors that affect triplane fitting. As shown in Tab. 4(a), we present results for five baselines. Rodin (256) is the method proposed in [71]. In Rodin (512), we naively increase the triplane resolution to 512. We can see that directly increasing the triplane resolution provides very marginal gains due to the forgetting problem. With our proposed bunch of techniques, scaling the resolution can notably increase PSNR to 31.54.

Fig. 10 visualizes the differences of the baselines. Rodin (256) cannot obtain a sharp beard due to the low-resolution triplanes. Rodin (512) obtains sharp details but there are many noises caused by the under-fitted decoder. In contrast, our method with task replay and consolidation weight regularization is able to get clean and sharp details. Also, see the areas around the silver tie for differences.

We further evaluate the factors that affect the generation results of our diffusion model in Tab. 4(b). Starting from Rodin [71] which trains diffusion on 256 × 256 × 32 triplanes, we only scale the triplane to 512 × 512 × 32 with other factors unchanged. We can see that the results even become worse. We think this is because the original noises are not suitable for larger triplanes. Using a stronger noise schedule obtains more reasonable results. Replacing the original CLIP image encoder with the VAE encoder (single-scale) also improves

###### Table 4: Ablation study of the proposed components.

Training strategy PSNR↑ LPIPS↓ SSIM↑

Model configuration FID↓ LPIPS↓ SSIM↑

- A. Rodin (256) 30.31 0.131 0.862
- B. Rodin (512) 30.56 0.129 0.863
- C. + Task replay 30.93 0.106 0.890
- D. + Weight decay 31.00 0.094 0.909
- E. + Consolidation 31.45 0.086 0.911 (a)

- A. Rodin (256) 33.20 0.323 0.758
- B. + Naively scale to 512 35.84 0.324 0.740
- C. + Strong noise schedule 27.74 0.318 0.745
- D. + Single-scale cond. 27.13 0.301 0.763
- E. + Multi-scale cond. 26.49 0.299 0.765 (b)

|[Figure 69]<br><br>|
|---|

|[Figure 70]<br><br>|
|---|

|[Figure 71]<br><br>|
|---|

|[Figure 72]<br><br>|
|---|

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Tab. 4(a) A. Tab. 4(a) B. Tab. 4(a) C. Tab. 4(a) E. Fig. 10: Qualitative ablation for the representation fitting.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Reference Tab. 4(b) C. Tab. 4(b) D. Tab. 4(b) E.

Fig. 11: Qualitative ablation for 3D diffusion model. Multi-scale feature conditioning results in significantly faithful and detailed generation results.

the results. Our method with multi-scale features achieves the best results. We also provide visual comparison as shown in Fig. 11, the introduced multi-scale features of the input portrait provide significantly more detailed texture cues, enabling high-fidelity avatar creation.

#### 4.5 Applications

Avatar creation from in-the-wild portrait. Despite training on the synthetic dataset [73], our model is robust to create 3D avatars conditioned on single inthe-wild portraits. The results in Fig. 12 validate our method’s capability for generalization to real-world images, which outperforms Rodin [71] in retaining both identity and details.

Text-conditioned avatar creation. To enable high-quality avatar creation from text, we first convert the text prompt to reference portrait leveraging strong

- 2D diffusion network [55], thereafter generate a high-fidelity avatar with the

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

Reference Rodin Our RodinHD

Fig. 12: Avatars generated conditioned on single in-the-wild portraits.

[Figure 91]

[Figure 92]

“Blender Synthetic Avatar, blonde hair, boy,

“Blender Synthetic Avatar, girl, blue eyes, brown afro hair, curly hair, green sweater”

brown eyes, medium and facial hair, red and orange shirt, mustache”

- Fig. 13: RodinHD can create high-fidelity avatars based on the reference portraits (dashed boxes) generated by finetuned 2D text-to-image diffusion models.

reference portrait. To be more specific, we use 40 selected frontal images and corresponding text prompts in our synthetic dataset to perform LoRA-based [29] finetuning for Stable Diffusion. We employ the term “Blender Synthetic Avata” as our trigger words within prompts to generate frontally aligned image inputs, which is a widely adopted technique for steering the style of images produced by fine-tuned SD. We provide the samples of text-to-avatar creation in Fig. 13.

### 5 Conclusion

We present RodinHD for high-fidelity 3D avatar generation by improving both the data fitting and the diffusion modeling. To maintain the compelling visual details of avatars, we propose a task-relay strategy and identity-aware weight consolidation regularizer for high-quality and robust data fitting in largescale. Furthermore, to model the distribution of highly detailed avatars, we introduce a multi-scale visual feature conditioning mechanism in our cascaded diffusion model, which provides fine-grind guidance to diffusion generation. In addition, we study previous 2D optimized noise scheduling in both high-resolution and high-dimensional 3D diffusion training. Our optimized noise schedules effectively enhance the details of the generated avatars. Extensive experiments show the proposed framework can generate high-fidelity 3D avatars with rich details, which is also promising to apply our model to general 3D scene modeling.

### Acknowledgments

This work was supported in part by the Anhui Provincial Natural Science Foundation under Grant 2108085UD12. We acknowledge the support of GPU cluster built by MCC Lab of Information Science and Technology Institution, USTC.

### References

- 1. Achlioptas, P., Diamanti, O., Mitliagkas, I., Guibas, L.: Learning representations and generative models for 3d point clouds. In: International Conference on Machine Learning. pp. 40–49. PMLR (2018)
- 2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mipnerf 360: Unbounded anti-aliased neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5470– 5479 (2022)
- 3. Bautista, M.A., Guo, P., Abnar, S., Talbott, W., Toshev, A., Chen, Z., Dinh, L., Zhai, S., Goh, H., Ulbricht, D., et al.: Gaudi: A neural architect for immersive 3d scene generation. arXiv preprint arXiv:2207.13751 (2022)
- 4. Bolles, R.C., Baker, H.H., Marimont, D.H.: Epipolar-plane image analysis: An approach to determining structure from motion. International Journal of Computer Vision 1(1), 7–55 (1987)
- 5. Brock, A., Lim, T., Ritchie, J.M., Weston, N.: Generative and discriminative voxel modeling with convolutional neural networks. arXiv preprint arXiv:1608.04236

(2016)

- 6. Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., De Mello, S., Gallo, O., Guibas, L.J., Tremblay, J., Khamis, S., et al.: Efficient geometry-aware 3d generative adversarial networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16123–16133 (2022)
- 7. Chan, E.R., Monteiro, M., Kellnhofer, P., Wu, J., Wetzstein, G.: pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5799–5809 (2021)
- 8. Chen, H., Gu, J., Chen, A., Tian, W., Tu, Z., Liu, L., Su, H.: Single-stage diffusion nerf: A unified approach to 3d generation and reconstruction. arXiv preprint arXiv:2304.06714 (2023)
- 9. Chen, T.: On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972 (2023)
- 10. Chen, Y., Wang, T., Wu, T., Pan, X., Jia, K., Liu, Z.: Comboverse: Compositional 3d assets creation using spatially-aware diffusion guidance. arXiv preprint arXiv:2403.12409 (2024)
- 11. Cheng, Y., Yin, F., Huang, X., Yu, X., Liu, J., Feng, S., Yang, Y., Tang, Y.: Efficient text-guided 3d-aware portrait generation with score distillation sampling on distribution. arXiv preprint arXiv:2306.02083 (2023)
- 12. Dai, W., Chen, L.H., Wang, J., Liu, J., Dai, B., Tang, Y.: Motionlcm: Realtime controllable motion generation via latent consistency model. arXiv preprint arXiv:2404.19759 (2024)
- 13. Deng, J., Guo, J., Xue, N., Zafeiriou, S.: Arcface: Additive angular margin loss for deep face recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4690–4699 (2019)

- 14. Deng, Y., Yang, J., Xiang, J., Tong, X.: Gram: Generative radiance manifolds for 3d-aware image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10673–10683 (2022)
- 15. Deng, Y., Yang, J., Xu, S., Chen, D., Jia, Y., Tong, X.: Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 0–0 (2019)
- 16. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems 34, 8780–8794 (2021)
- 17. Ding, Z., Zhang, M., Wu, J., Tu, Z.: Patched denoising diffusion models for highresolution image synthesis. arXiv preprint arXiv:2308.01316 (2023)
- 18. Gao, J., Shen, T., Wang, Z., Chen, W., Yin, K., Li, D., Litany, O., Gojcic, Z., Fidler, S.: Get3d: A generative model of high quality 3d textured shapes learned from images. arXiv preprint arXiv:2209.11163 (2022)
- 19. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Communications of the ACM 63(11), 139–144 (2020)
- 20. Gu, J., Zhai, S., Zhang, Y., Bautista, M.A., Susskind, J.: f-dm: A multi-stage diffusion model via progressive signal transformation. arXiv preprint arXiv:2210.04955

(2022)

- 21. Gu, J., Zhai, S., Zhang, Y., Susskind, J., Jaitly, N.: Matryoshka diffusion models. arXiv preprint arXiv:2310.15111 (2023)
- 22. Gu, S., Chen, D., Bao, J., Wen, F., Zhang, B., Chen, D., Yuan, L., Guo, B.: Vector quantized diffusion model for text-to-image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10696– 10706 (2022)
- 23. Gupta, A., Xiong, W., Nie, Y., Jones, I., Oğuz, B.: 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371 (2023)
- 24. Hang, T., Gu, S., Li, C., Bao, J., Chen, D., Hu, H., Geng, X., Guo, B.: Efficient diffusion training via min-snr weighting strategy. arXiv preprint arXiv:2303.09556

(2023)

- 25. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems 30 (2017)
- 26. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33, 6840–6851 (2020)
- 27. Ho, J., Saharia, C., Chan, W., Fleet, D.J., Norouzi, M., Salimans, T.: Cascaded diffusion models for high fidelity image generation. The Journal of Machine Learning Research 23(1), 2249–2281 (2022)
- 28. Hoogeboom, E., Heek, J., Salimans, T.: simple diffusion: End-to-end diffusion for high resolution images. arXiv preprint arXiv:2301.11093 (2023)
- 29. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021)
- 30. Jabri, A., Fleet, D., Chen, T.: Scalable adaptive computation for iterative generation. arXiv preprint arXiv:2212.11972 (2022)
- 31. Jun, H., Nichol, A.: Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463 (2023)
- 32. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4401–4410 (2019)

- 33. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)
- 34. Kingma, D.P., Welling, M., et al.: An introduction to variational autoencoders. Foundations and Trends® in Machine Learning 12(4), 307–392 (2019)
- 35. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al.: Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences 114(13), 3521–3526 (2017)
- 36. Li, R., Li, X., Hui, K.H., Fu, C.W.: Sp-gan: Sphere-guided 3d shape generation and manipulation. ACM Transactions on Graphics (TOG) 40(4), 1–12 (2021)
- 37. Li, Z., Tucker, R., Snavely, N., Holynski, A.: Generative image dynamics. arXiv preprint arXiv:2309.07906 (2023)
- 38. Liao, Y., Schwarz, K., Mescheder, L., Geiger, A.: Towards unsupervised learning of generative models for 3d controllable image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5871– 5880 (2020)
- 39. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. arXiv preprint arXiv:2211.10440 (2022)
- 40. Lin, S., Liu, B., Li, J., Yang, X.: Common diffusion noise schedules and sample steps are flawed. arXiv preprint arXiv:2305.08891 (2023)
- 41. Liu, J., Dai, W., Wang, C., Cheng, Y., Tang, Y., Tong, X.: Plan, posture and go: Towards open-world text-to-motion generation. arXiv preprint arXiv:2312.14828

(2023)

- 42. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations, ICLR (2019)
- 43. Ma, H., Zhang, L., Zhu, X., Zhang, J., Feng, J.: Accelerating score-based generative models for high-resolution image synthesis. arXiv preprint arXiv:2206.04029 (2022)
- 44. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021)
- 45. Müller, N., Siddiqui, Y., Porzi, L., Bulo, S.R., Kontschieder, P., Nießner, M.: Diffrf: Rendering-guided 3d radiance field diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4328–4338 (2023)
- 46. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1– 15 (2022)
- 47. Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021)
- 48. Nichol, A.Q., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: International Conference on Machine Learning. pp. 8162–8171. PMLR (2021)
- 49. Niemeyer, M., Geiger, A.: Giraffe: Representing scenes as compositional generative neural feature fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11453–11464 (2021)
- 50. Ntavelis, E., Siarohin, A., Olszewski, K., Wang, C., Van Gool, L., Tulyakov, S.: Autodecoding latent 3d diffusion models. arXiv preprint arXiv:2307.05445 (2023)
- 51. Park, J.J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S.: Deepsdf: Learning continuous signed distance functions for shape representation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 165– 174 (2019)

- 52. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- 53. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022)
- 54. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763. PMLR (2021)
- 55. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)
- 56. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour, S.K.S., Ayan, B.K., Mahdavi, S.S., Lopes, R.G., et al.: Photorealistic textto-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487 (2022)
- 57. Schwarz, K., Liao, Y., Niemeyer, M., Geiger, A.: Graf: Generative radiance fields for 3d-aware image synthesis. Advances in Neural Information Processing Systems 33, 20154–20166 (2020)
- 58. Shue, J.R., Chan, E.R., Po, R., Ankner, Z., Wu, J., Wetzstein, G.: 3d neural field generation using triplane diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20875–20886 (2023)
- 59. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)
- 60. Sitzmann, V., Martel, J., Bergman, A., Lindell, D., Wetzstein, G.: Implicit neural representations with periodic activation functions. Advances in Neural Information Processing Systems 33, 7462–7473 (2020)
- 61. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020)
- 62. Sun, J., Zhang, B., Shao, R., Wang, L., Liu, W., Xie, Z., Liu, Y.: Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. arXiv preprint arXiv:2310.16818 (2023)
- 63. Szabó, A., Meishvili, G., Favaro, P.: Unsupervised generative 3d shape learning from natural images. arXiv preprint arXiv:1910.00287 (2019)
- 64. Tang, J., Wang, T., Zhang, B., Zhang, T., Yi, R., Ma, L., Chen, D.: Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184 (2023)
- 65. Tang, J., Zeng, Y., Fan, K., Wang, X., Dai, B., Chen, K., Ma, L.: Make-it-vivid: Dressing your animatable biped cartoon characters from text. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6243–6253 (2024)
- 66. Tang, J., Zhang, B., Yang, B., Zhang, T., Chen, D., Ma, L., Wen, F.: Explicitly controllable 3d-aware portrait generation. arXiv preprint arXiv:2209.05434 (2022)
- 67. Tang, Y., Liu, J., Liu, A., Yang, B., Dai, W., Rao, Y., Lu, J., Zhou, J., Li, X.: Flag3d: A 3d fitness activity dataset with language instruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22106–22117 (2023)
- 68. Teng, J., Zheng, W., Ding, M., Hong, W., Wangni, J., Yang, Z., Tang, J.: Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350 (2023)

- 69. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in Neural Information Processing Systems 30 (2017)
- 70. Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W.: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689 (2021)
- 71. Wang, T., Zhang, B., Zhang, T., Gu, S., Bao, J., Baltrusaitis, T., Shen, J., Chen, D., Wen, F., Chen, Q., et al.: Rodin: A generative model for sculpting 3d digital avatars using diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4563–4573 (2023)
- 72. Wang, X., Li, Y., Zhang, H., Shan, Y.: Towards real-world blind face restoration with generative facial prior. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9168–9178 (2021)
- 73. Wood, E., Baltrušaitis, T., Hewitt, C., Dziadzio, S., Cashman, T.J., Shotton, J.: Fake it till you make it: face analysis in the wild using synthetic data alone. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3681–3691 (2021)
- 74. Wu, J., Zhang, C., Xue, T., Freeman, B., Tenenbaum, J.: Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. Advances in Neural Information Processing Systems 29 (2016)
- 75. Xia, W., Xue, J.H.: A survey on deep generative 3d-aware image synthesis. ACM Computing Surveys 56(4), 1–34 (2023)
- 76. Xiang, J., Yang, J., Deng, Y., Tong, X.: Gram-hd: 3d-consistent image generation at high resolution with generative radiance manifolds. arXiv preprint arXiv:2206.07255 (2022)
- 77. Yin, F., Zhang, Y., Wang, X., Wang, T., Li, X., Gong, Y., Fan, Y., Cun, X., Shan, Y., Oztireli, C., et al.: 3d gan inversion with facial symmetry prior. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 342–351 (2023)
- 78. Zeng, X., Vahdat, A., Williams, F., Gojcic, Z., Litany, O., Fidler, S., Kreis, K.: Lion: Latent point diffusion models for 3d shape generation. arXiv preprint arXiv:2210.06978 (2022)
- 79. Zhang, B., Cheng, Y., Yang, J., Wang, C., Zhao, F., Tang, Y., Chen, D., Guo, B.: Gaussiancube: Structuring gaussian splatting using optimal transport for 3d generative modeling. arXiv preprint arXiv:2403.19655 (2024)
- 80. Zhang, B., Gu, S., Zhang, B., Bao, J., Chen, D., Wen, F., Wang, Y., Guo, B.: Styleswin: Transformer-based gan for high-resolution image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11304–11314 (2022)
- 81. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3836–3847 (2023)
- 82. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 586–595 (2018)

### A Additional Implementation Details

Triplane fitting. We split the triplane fitting into two stages to reduce computation costs. In the first stage, we jointly train the MLP decoder and the triplanes on a subset of 64 avatars. During each inner loop iteration, 8192 rays are randomly sampled for loss calculation. We optimize one avatar per GPU for each outer loop iteration due to large GPU memory consumption. The detailed hyper-parameters of the first stage are listed in Tab. 5, including ablation studies. In the second stage, we fix the decoder’s weights and fine-tune the triplanes of 46K avatars independently. The fitting iteration for each avatar is set to 25000. For rendering efficiency, an occupancy grid of 1283 resolution is maintained [46] to skip ray marching steps in empty space. Since we do not have the occupancy grid of the diffusion generated triplane, we update the occupancy grid 16 times from zero initialization before performing volumetric rendering.

Diffusion training. For triplane x = (xuv,xwu,xvw) of shape R3×H×W×C, we perform triplane roll-out x = hstack(yuv,ywu,yvw) ∈ RH×3W×C in order to employ the well-designed 2D UNet model in diffusion [16,48] following [71]. We also leverage 3D-aware convolution [71] for cross-plane feature communication. The portrait image is resized to 256 × 256 and the resulting multi-scale features have the resolution of 128 × 128, 64 × 64, and 32 × 32. The conditional features are injected to the base diffusion model at layers with resolutions of 32 × 32, 16×16, and 8×8, respectively, through cross attention. The upsample diffusion model only uses the conditional features of 128 × 128, which are injected to the middle latent features.

For our base diffusion model, we adopt the UNet model architecture from [16]. We train our base model using AdamW optimizer [42] with a learning rate 1e−5. To condition on the multi-scale image features of input portrait as illustrated in Sec. 3.2, we perform cross attention at resolutions (32,16,8). Our optimized noise schedule is based on the cosine schedule mentioned in [9], and we further adjust its hyper-parameters for 3D diffusion training. We provide the detailed configurations of the model and diffusion below.

For our upsample diffusion model, we also adopt the UNet model architecture from [16]. We train our upsample model using Adam optimizer [33] with a learning rate 1e − 5. We remove self-attention due to unaffordable computation cost at high resolutions, and perform cross attention at resolution 128 for conditioning on input portrait features. Our optimized noise schedule for upsample diffusion is based on the sigmoid noise schedule in [9], then we carefully adjust the hyper-parameters for 3D diffusion training. The detailed configurations of the model and diffusion are shown below.

# 128x128 Base diffusion UNet configuration = {

"channels": 192, "channel_mult": (1, 1, 2, 3, 4), "embed_dim": 768, "num_res_blocks": (3, 3, 3, 3, 3), "attn_resolutions": (32, 16, 8),

"ms_vae_feature_cross_attn_res": (32, 16, 8), "3D_aware_conv_res": (128), "dropout": 0, "feature_pooling_type": "attention", "use_scale_shift_norm": True

} Diffusion configuration = {

"Training steps": 1000, "Noise schedule": Cosine(start=0.2, end=1, tau=3), "Inference steps": 10, "Inference sampler": "DDPM"

} # 128x128 -> 512x512 Upsample diffusion UNet configuration = {

"channels": 128, "channel_mult": (1, 2, 4), "embed_dim": 512, "num_res_blocks": (2, 2, 6), "ms_vae_feature_cross_attn_res": (128), "3D_aware_conv_res": (512, 256, 128), "dropout": 0, "feature_pooling_type": "attention", "use_scale_shift_norm": False

} Diffusion configuration = {

"Training steps": 100, "Noise schedule": Sigmoid(start=0, end=3, tau=0.1), "Inference steps": 10, "Inference sampler": "DDPM"

}

### B Additional Analysis and Visualization

Choices of triplane resolution and channel. We argue that both triplane resolution and channel affect the preservation of high-frequency information in renderings. To validate this argument, we experiment with different triplanes from a resolution set of {128,256,512} and a channel set of {4,8,16,32} to fit 1024 × 1024 images of one subject and show the results in Tab. 6 and Fig. 14. Overall, the fitting quality increases with the triplane resolution and channel. High-resolution triplane can render high-frequency detail, and low-resolution triplane tends to produce blurring results. On the other hand, the triplane with more channels can keep high-fidelity appearance without introducing noisy pattern, but low-resolution triplane with more channels can not achieve better high-

Table 5: Hyper-parameters for the first stage of fitting, including ablation studies.

Rodin (512) + Task relay + Wight decay Ours

Inner loop iterations 15000 5000 5000 5000 Outer loop iterations per avatar 1 30 30 30 Loss Weight of TV regularization 1e-2 1e-2 1e-2 1e-2 Loss Weight of L2 regularization 1e-4 1e-4 1e-4 1e-4 Loss Weight of IWC regularization 0 0 0 0.1 Loss Weight of weight decay 0 0 1e-4 0 Triplane learning rate 2e-3 2e-3 2e-3 2e-3 Decoder learning rate 2e-4 2e-4 2e-4 2e-4 Ray batch size 8192 8192 8192 8192 Samples per ray 1024 1024 1024 1024

- Table 6: Comparison fitting quality (PSNR) of Triplane Resolution and Channel.

Ch.

4 8 16 32

Res.

128 30.15 30.71 31.21 31.59 256 30.24 31.01 31.44 31.67 512 30.38 31.31 31.60 31.71

frequency detail preservation than high-resolution triplane. We thus choose to utilize 512 × 512 × 32 triplanes in our experiments.

Visualization of intermediate results in the denoising process. During inference, our model starts from isotropic Gaussian noise and progressively reduces the noise to obtain the final high-quality triplanes. We visualize the renderings of generated results xt of intermediate timesteps t ∈ (0,1) in the denoising process to provide a comprehensive understanding of the triplane diffusion procedure. From Fig. 15, we observe that our model establishes the global structure of the avatar, and subsequently adds more detail, which is similar to [58,71].

### C Additional Comparison

More quantitative comparison with Rodin. We additionally compare our method with Rodin [71] on conditional avatar generation using more evaluation metrics. We evaluate the cosine similarity of identity embedding derived from ArcFace [13] between generated avatars and ground-truths (CSIM), as well as between paired renderings of generated avatars from different camera viewpoints (CSIM-CrossView). We also include Average Expression Distance (AED), Average Pose Distance (APD) and Average Shape Distance (ASD) between the

[Figure 93]

- Fig. 14: Comparison on Triplane Resolution and Channel. Zoom in for better visualization.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Fig. 15: Visualization of intermediate results in the denoising process.

reconstructed 3D faces [15] of generated avatars and ground-truth avatars. The results presented in Tab. 7 demonstrate that our model excels in preserving identity and accurately generating expression, pose, and geometry. While Rodin’s 2D refinement achieves lower FID scores, it struggles to maintain the identity and expression details of the conditioned portraits.

#### Comparison of 3D consistency with EG3D. We additionally compare our

- 3D consistency with SOTA 3D-aware GANs, EG3D. We evaluate the 3D consistency of unconditional generated results in Fig. 16, which is similar to Fig. 9 of main paper. Since EG3D also utilizes a 2D super-resolution module, the results in Fig. 16 yield obvious texture flickering, whereas our method leads to a natural and smooth texture pattern. We also provide numerical comparison

- in Tab. 8 by fitting a NeuS model from generated multi-views following Tab. 3 of main paper. Our generated results achieve significantly better metrics due to multi-view consistency.

##### Table 7: Additional comparison of conditional avatar generation. The subscript ∗ indicates that 2D refinement is applied to the rendered images.

Models FID↓ PSNR↑ CSIM↑ CSIM-CrossView↑ AED↓ APD%↓ ASD↓

Rodin 33.20 18.28 0.64 0.85 0.21 2.21 0.44 Rodin∗ 20.51 17.31 0.63 0.83 0.20 2.21 0.44 Ours 26.49 20.33 0.68 0.85 0.18 1.78 0.44

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

##### EG3D Our RodinHD

- Fig. 16: Visual comparison of 3D consistency akin to the Epipolar Line Images [4]. Our model yields smooth and natural texture, whereas EG3D produces obvious texture flickering, indicating the 3D inconsistency with 2D refinement.

### D Additional Results

Conditional avatar generation. We provide more renderings of generated avatars conditioned on the single portraits from our test set in Fig. 17. Our model is capable of creating high-fidelity avatars with compelling details and vivid expressions, demonstrating the strong capability of the proposed model.

Unconditional avatar generation. Fig. 18 show more unconditional avatars created by our model. Our model is able to produce diverse high-quality avatars with rich details, including complex clothing and hairstyles.

Avatar creation from in-the-wild portrait. In Fig. 19, we present additional generated avatars conditioned on real-world images. Our methodology demonstrates a higher fidelity in preserving the identity of the subjects when compared with [71]. Furthermore, our results exhibit a remarkable ability to retain intricate details such as hairstyle and clothing attributes.

Text-to-avatar creation. We provide more samples of high-quality text-toavatar creation in Fig. 20. We first convert the text prompt to reference portrait by our finetuned 2D text-to-image diffusion models, thereafter generate a highfidelity avatar conditioned on the reference portrait. It is worth noticing that the

- Table 8: Quantitative comparison of 3D consistency with EG3D.

PSNR↑ SSIM↑ LPIPS↓

EG3D 29.51 0.962 0.052 Ours 33.39 0.967 0.043

- Table 9: Average ranking of user study in conditional generation.

Models ID Similarity↓ 3D Consistency↓ Visual Fidelity↓

Rodin 2.54 2.12 2.61 Rodin∗ 2.06 2.77 1.26 Ours 1.39 1.11 2.13

trigger word we used “Blender Synthetic Avata” is not necessarily needed to be added in the prompts since we can omit it and perform cropping and alignment to the generated images, similar to how we handle realistic image inputs.

User study. We further conduct user study to measure the identity similarity (ID), 3D consistency and visual fidelity. We ask 15 subjects to rank different methods with 20 sets of comparisons in each study. The average ranking

- in Tab. 9 shows that our method earns user preferences the best in identity preservation and 3D consistency, only slightly worse than Rodin adding 2D refinement (Rodin∗) in fidelity. We think more follow-up research can be conducted to further improve the visual quality while ensuring the 3d consistency.

### E Responsible AI Considerations

Our model is trained on the synthetic dataset [73] of 3D digital avatars akin to those crafted by artists, as opposed to photo-realistic humans. This approach to training data selection alleviates privacy and copyright concerns associated with the use of real human face collections. Despite these precautions, it is important to acknowledge that 3D avatar created by our model from real-world images could potentially be exploited for the dissemination of disinformation, similar to other generative models. We must therefore emphasize the importance of responsible use of our technology. As a safeguard against misuse, we recommend the implementation of measures such as embedding visible tags or watermarks into the distributed renderings produced by our model.

### F Limitation

As illustrated in Fig. 21, our model still has some limitations. Floating points occasionally appear in the generated avatars as shown in Fig. 21 (a), which are typical NeRF artifacts [2]. Handling glasses remains challenging due to limited training data in Fig. 21 (b).

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

###### Fig. 17: Conditional generation samples by our model. Reference portraits are shown in dashed boxes.

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

###### Fig. 18: Unconditional generation samples by our model.

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

Reference Rodin Our RodinHD

- Fig. 19: Samples of generated avatars conditioned on single in-the-wild portraits. Compared with Rodin, our method preserves more details of identity and clothing.

[Figure 140]

“Blender Synthetic Avatar, brown hair, boy, brown eyes, scarf, beard, stubble”

[Figure 141]

“Blender Synthetic Avatar, brown hair, boy, brown eyes, beard, black sweater”

- Fig. 20: Samples of text-to-avatar creation using our model. The leftmost reference portraits are first created by finetuned 2D diffusion model given the text prompts. Then our 3D diffusion model creates 3D avatars conditioned on the generated portraits.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

(a) (b)

Fig. 21: Failure cases.

