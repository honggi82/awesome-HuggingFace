# arXiv:2406.04324v2[cs.CV]24Oct2024

## SF-V: Single Forward Video Generation Model

Zhixing Zhang1,2∗ Yanyu Li1 Yushu Wu1 Yanwu Xu1 Anil Kag1 Ivan Skorokhodov1 Willi Menapace1 Aliaksandr Siarohin1 Junli Cao1

Dimitris Metaxas2 Sergey Tulyakov1 Jian Ren1† 1Snap Inc. 2 Rutgers University Project Page: https://snap-research.github.io/SF-V

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Figure 1: Example generation results from our single-step image-to-video model. Our model can generate high-quality and motion consistent videos by only performing the sampling once during inference. Please refer to our webpage for whole video sequences.

### Abstract

Diffusion-based video generation models have demonstrated remarkable success in obtaining high-fidelity videos through the iterative denoising process. However, these models require multiple denoising steps during sampling, resulting in high computational costs. In this work, we propose a novel approach to obtain singlestep video generation models by leveraging adversarial training to fine-tune pretrained video diffusion models. We show that, through the adversarial training, the multi-steps video diffusion model, i.e., Stable Video Diffusion (SVD), can be trained to perform single forward pass to synthesize high-quality videos, capturing both temporal and spatial dependencies in the video data. Extensive experiments demonstrate that our method achieves competitive generation quality of synthesized videos with significantly reduced computational overhead for the denoising process (i.e., around 23× speedup compared with SVD and 6× speedup compared with existing works, with even better generation quality), paving the way for real-time video synthesis and editing.

∗Work done during an internship at Snap Inc. †Corresponding author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

### 1 Introduction

Video generation is experiencing unprecedented advancements by leveraging large-scale denoising diffusion probabilistic models [1, 2] to create photo-realistic frames with natural and consistent motion [3, 4], revolutionizing various fields, such as entertainment and digital content creation [5, 6].

Early efforts on image generation show that diffusion models have the significant capabilities when scaled-up to generate diverse and high-fidelity content [1, 2]. Additionally, these models benefit from a stable training and convergence process, demonstrating a considerable improvement over their predecessors, i.e., generative adversarial networks (GANs) [7]. Therefore, many studies on video generation are built upon the diffusion models. Some of them utilize the pre-trained image diffusion models for video synthesis through introducing temporal layers to generate high-quality video clips [8–11]. Inspired by this design paradigm, numerous video generation applications have emerged, such as animating a given image with optional motion priors [12–15], generating videos from natural language descriptions [16, 17, 5], and even synthesizing cinematic and minutes-long temporal-consistent videos [18, 4].

Despite the impressive generative performance, video diffusion models suffer from tremendous computational costs, hindering their widespread and efficient deployment. The iterative nature of the sampling process makes video diffusion models significantly slower than other generative models (e.g., GANs [19, 20]). For instance, in our benchmark, it only takes 0.3 seconds to perform a single denoising step using the UNet from the Stable Video Diffusion (SVD) [13] model to generate 14 frames on one NVIDIA A100 GPU, while consuming 10.79 seconds to run the UNet with the conventional 25-step sampling.

The significant overhead introduced by iterative sampling highlights the necessity to generate videos in fewer steps while maintaining the quality of multi-step sampling. Recent works [21–23] extend consistency training [24] to video diffusion models, offering two main benefits: reduced total runtime by performing fewer sampling steps and the preservation of the pre-trained ordinary differential equation (ODE) trajectory, allowing high-quality video generation with fewer sampling steps (e.g., 8 steps). Nevertheless, these approaches still struggle to achieve single-step high-quality video generation.

On the other hand, distilling image diffusion models into one step via adversarial training have shown promising progress [25–29]. However, scaling up such approaches for video diffusion model training to achieve single-step generation has not been well studied. In this work, we leverage adversarial training to obtain an image-to-vide o generation model that requires only single-step generation, with the contributions summarized as follows:

- • We build the framework to fine-tune the pre-trained state-of-the-art video diffusion model (i.e., SVD) to be able to generate videos in single forward pass, greatly reducing the runtime burden of video diffusion model. The training is conducted through adversarial training on the latent space.
- • To improve the generation quality (e.g., higher image quality and more consistent motion), we introduce the discriminator with spatial-temporal heads, preventing the generated videos from collapsing to the conditional image.
- • We are the first to achieve one-step generation for video diffusion models. Our one-step model demonstrates superiority in FVD [30] and visual quality. Specifically, for the denoising process, our model achieves around 23× speedup compared with SVD and 6× speedup compared with exiting works, with even better generation quality.

### 2 Related Work

Video Generation has been a long studied problem, aiming for high-quality image generation and consistent motion synthesis. Early efforts in this domain utilize adversarial training [31, 32]. Though extensively investigated, the trained models still suffer from low resolution, limited generated sequences, and inconsistent motion. Recent studies leverage denoising diffusion probabilistic models [1, 33, 34] to scale the video generators up to billions of model parameters, achieving high-fidelity generation sequences [35–39, 5, 4, 3, 18]. Nonetheless, the tremendous computation cost of video diffusion models hinders their wide deployment. It takes tens of seconds to generate a single video

batch even for high-tier server GPUs. Consequently, the reduction of denoising steps [21, 40, 22] is pivotal to efficient video generation, which linearly scales down the total runtime.

Step Distillation of Diffusion Models. Initially developed upon image diffusion models, progressive distillation [41, 42] aims to distill a less-step student mimicking the full-step counterpart. Specifically, at each step, the student learns to predict a teacher location in the ODE flow, resulting in fewer required denoising steps during inference time. Latent Consistency Models (LCM) [24, 43–48] instead proposes to refine the prediction objective into clean data, and achieves high-fidelity generation with fewer (2 ∼ 4) steps. Rectified flow [49, 50] progressively straights the ODE flow where each denoising step becomes a substitution of a long trajectory. UFOGen [25], ADD [27], and its latentspace successor LADD [28] further incorporate adversarial loss to distill teacher signal into the few-step student, enabling one-step generation with reasonable quality, and outperforming the teacher model with about 4 steps. DMD [26] proposes to combine a distribution matching objective and a regression loss to distill a one-step generator. The recent SDXL-Lightning [29] combines progressive distillation with adversarial loss to mitigate the blurry generation issue and ease the convergence of multi-step settings. In addition, SDXL-Lightning refines the design of the discriminator and proposes two adversarial loss objectives to balance sample quality and mode convergence.

When it comes to video models, VideoLCM [40] and AnimateLCM [21] adopt consistency distillation to enable 4-step generation with comparable quality to the full-step pre-trained video diffusion model. However, in the one-step setting, there are still considerable performance gaps observed for the visual quality. Animate-Diff Lightning [22] incorporates adversarial distillation to further reduce warps and blurs in the 1-2 step setting, despite that the model still underperforms full-step baselines.

### 3 Method

Our goal is to generate high-fidelity and temporally consistent videos in as few sampling steps as possible (i.e., 1 step). The adversarial objective has been proven effective in reducing the number of sampling steps required by diffusion models in image space [27, 28, 25, 51]. However, limited efforts have been conducted on scaling up the effective adversarial training to reduce the number of sampling steps for video diffusion models. In the following, we introduce the framework of latent adversarial training to obtain efficient video diffusion model by running sampling in single step. In this framework, we initialize the generator and part of the discriminator with the weights of a pre-trained video diffusion model. Moreover, we introduce a structure with separate spatial and temporal discriminator heads to enhance frame quality and motion consistency.

#### 3.1 Preliminaries of Stable Video Diffusion

Our method is built upon the Stable Video Diffusion (SVD) [13], which is an implementation of the EDM-framework [33] for conditional video generation, where the diffusion process is conducted in latent space. We choose the publicly released image-to-video generation pipeline of SVD due to its superior performance in generating high-quality and motion-consistent videos.

Training Diffusion Models with EDM. To facilitate the presentation, let pdata(x0) denote the data distribution and p(x;σ) represent the distribution obtained by adding σ2-variance Gaussian noise

to the data. For sufficiently large σmax, p(x;σmax) ≈ N(0,σmax2 ). Starting from high variance Gaussian noise xM ∼ N(0,σmax2 ), the diffusion models sequentially denoise towards σ0 = 0 through the numerical simulation of the Probability Flow ODE [52]. The denoiser, Dθ, attempts to predict the clean x0 and is trained via denoising score matching:

0∼pdata(x0),(σ,n)∼p(σ,n) λσ∥Dθ(x0 + n;σ) − x0∥22 , (1)

Ex

where p(σ,n) = p(σ)N(n;0,σ2), p(σ) is a distribution over noise levels σ, and λσ : R+ → R+ is a weighting function.

EDM [33] parameterizes the denoiser Dθ as:

##### Dθ(x;σ) = cskip(σ)x + cout(σ)Fθ(cin(σ)x;cnoise(σ)), (2)

where Fθ is the network to be trained. The preconditioning functions are set as cskip(σ) = (σ2+1)−1, cout(σ) = √σ−σ

2+1, cin(σ) = √σ1

2+1, and cnoise(σ) = 0.25log σ.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

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

[Figure 40]

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

- Figure 2: Training Pipeline. We initialize our generator and discriminator using the weights of a pre-trained image-to-video diffusion model. The discriminator utilizes the encoder part of the UNet as its backbone, which remains frozen during training. We add a spatial discriminator head and a temporal discriminator head after each downsampling block of the discriminator backbone and only update the parameters of these heads during training. Given a video latent x0, we first add noise σt through a forward diffusion process to obtain xt. The generator then predicts xˆ0 given xt. We calculate the reconstruction loss Lrecon between x0 and xˆ0. Additionally, we add noise level

σt′ to both x0 and xˆ0 to obtain real and fake samples, x′t and xˆ′t. The adversarial loss Ladv is then calculated using these real and fake sample pairs.

Stable Video Diffusion. The training of video model asks for a dataset of videos, each consisting of N frames with height H and width W. Given a video V0 = {Ii0}Ni=0, where Ii0 ∈ R3×H×W, SVD [13] maps each frame separately to latent space using a frame encoder, E. The encoded frames are represented as x0 = {E(Ii0)}Ni=0, resulting in x0 ∈ RN×4×H˜×W˜ . Here, x0 ∼ pdata(x0) is a sequence of N latent frames with 4 channels, height H˜, and width W˜ .

SVD inflates a text-to-image diffusion model to a text-to-video diffusion model [10]. The text conditioning is replaced with image conditioning to create an image-to-video diffusion model. Consequently, the parameterized denoiser Dθ in Eq. (2) is modified as follows:

Dθ(x;σ,c) = cskip(σ)x + cout(σ)Fθ(cin(σ)x;cnoise(σ),c), (3)

where c is the image condition I00, i.e., the first frame of the video. At sampling time, Dθ is leveraged to restore xt−1 from xt using the following relation [33]:

dt = (xt − Dθ(xt;σt,c))/σt; xt−1 = xt + (σt−1 − σt) · dt, (4) where σt is obtained with

t T − 1

σt = (σmin1/ρ +

(σmax1/ρ − σmin1/ρ))ρ, (5)

where T is the total number of denoising steps and ρ is a hyper-parameter controlling the emphasis level to low noise levels.

#### 3.2 Latent Adversarial Training for Video Diffusion Model

Design of Networks. Diffusion-GAN hybrid models are designed for training with large denoising step sizes [25, 27, 28, 51]. Our training procedure, illustrated in Fig. 2, involves two networks: a generator Gθ and a discriminator Dϕ. The generator is initialized from a pre-trained UNet diffusion model with weights θ (i.e., the UNet from SVD). The discriminator is partially initialized from a pretrained UNet diffusion model. Namely, the backbone of the discriminator shares the same architecture and weights as the pre-trained UNet encoder, and the weights of this backbone are kept frozen during training. Additionally, we augment the discriminator by adding a spatial discriminator head and

a temporal discriminator head after each backbone block. Therefore, in total, the discriminator comprises four spatial discriminator heads and four temporal discriminator heads. Only the parameters in these heads are trained during the discriminator training steps. The detailed architecture of these heads will be further discussed in Sec. 3.3.

Latent Adversarial Training. We use a pair of generated samples xˆ0 and real samples x0 to conduct the adversarial training. Specifically, during training, the generator Gθ produces generated samples xˆ0(xt;σt,c) from noisy data xt. The noisy data points are derived from a dataset of real latents x0 via a forward diffusion process xt = x0 + σtϵ. We sample σt uniformly from the set {σ1,··· ,σT

g−1}, obtained by setting T to Tg and t ∈ {1,2,··· ,Tg − 1} in Eq. (5). In practice, we set Tg = 4. The generated sample xˆ0 is given by:

##### xˆ0(xt;σt,c) = cskip(σt)xt + cout(σt)Gθ(cin(σt)xt;cnoise(σt),c). (6)

To train the discriminator, we forward the generated samples xˆ0 and real samples x0 into it, aiming to let the discriminator distinguish between them. However, for a more stabilized training, inspired by exiting works [28], we add noise to the samples before passing them to the discriminator, since the backbone of the discriminator is initialized from a pre-trained UNet with weights frozen during training. Namely, we sample σt′ from the set {σ1′ ,··· ,σT′

d−1}, obtained by setting T to Td and t ∈ {1,2,··· ,Td − 1} in Eq. (5), according to a discretized lognormal distribution defined as:

log(σt′−1 − Pmean) √2Pstd

log(σt′ − Pmean) √2Pstd − erf

p(σt′) ∝ erf

, (7)

where Pmean and Pstd control the noise level added to the samples before passing them to the discriminator. A visualization of how different Pmean and Pstd affect the probability of σ′ sampled is illustrated in Fig. 6. In practice, we set Td = 1,000. We diffuse the real and generated samples through the forward process to obtain xˆ′t = xˆ0 + σt′ϵ and x′t = x0 + σt′ϵ, respectively.

Following literature [27, 53, 54], we use the hinge loss [55] as the adversarial objective function for improved performance. The adversarial optimization for the generator LGadv(ˆx0,ϕ) is defined as:

##### LGadv = Eσ,σ′,x0[Dϕ (cin(σt′)ˆx′t)], (8)

Furthermore, we notice that a reconstruction objective, Lrecon, between x0 and xˆ0 can significantly improve the stability of the training process. We use Pseudo-Huber metric [56, 43] for reconstruction loss, as:

##### Lrecon(ˆx0,x0) = ∥xˆ0 − x0∥22 + c2 − c, (9)

where c > 0 is an adjustable constant. Thus, the overall objective for training the generator is as follows with λ balances two losses:

LG = LGadv + λLrecon(ˆx0,x0). (10) Other other hand, the discriminator is trained to minimize:

##### LDadv = Eσ′,x0[max(0,1+Dϕ (cin(σt′)x′t))+γR1]+Eσ,σ′,x0[max(0,1−Dϕ (cin(σt′)ˆx′t)))], (11)

where R1 denotes the R1 gradient penalty [57, 27]. Here, we omit other conditional input for Dϕ, such as cnoise(σ′) and image conditioning c, for simplicity.

Discussion. Our latent adversarial training framework is largely inspired by LADD [28]. Similar to LADD, we set Tg = 4 in practice and utilize a pre-trained diffusion model as part of the discriminator. However, our approach has several key differences compared with LADD [28]. First, we extend the image latent adversarial distillation framework to the video domain by incorporating spatial and temporal heads to achieve one-step generation for video diffusion models. The specifics of the spatial and temporal heads are discussed in Sec. 3.3. Second, based on the EDM-framework [33], we observe that sampling t′ using a discretized lognormal distribution provides more stable adversarial training compared to the logit-normal distribution used in LADD [28]. Finally, unlike LADD [28], we utilize real video data instead of synthetic data for training and incorporate a reconstruction objective (i.e., Eq. (9)) to ensure more stable training.

- Figure 3: Spatial & Temporal Discriminator Heads. Our discriminator heads take in intermediate features of the UNet encoder. Follow exiting arts [54, 53], we use image conditioning and frame index as the projected condition c. Left: For spatial discriminator heads, the input features are reshaped to merge the temporal axis and the batch axis, such that each frame is considered as an independent sample. Right: For temporal discriminator heads, we merge spatial dimensions to batch axis.

Table 1: Comparison Results. We compare our method against SVD [13], AnimateLCM [21], UFOGen [25], and LADD [28] using different numbers of sampling steps. AnimateLCM∗ indicates the usage of the officially provided 25-frame model, with only the first 14 frames considered for FVD calculation. † indicates our implementations. We also report the latency of the denoising process for each setting, measured on a single NVIDIA A100 GPU.

Name FVD↓ Steps Latency (s)

153.4 25 10.79 194.4 16 6.89 488.6 8 3.44 1687.0 4 1.72

SVD [13]

321.1 8 3.25 403.2 4 1.62 521.9 2 0.82

AnimateLCM∗ [21]

281.0 8 1.85 801.4 4 0.92 1158.4 2 0.46

AnimateLCM [21]

UFOGen† [25] 1917.2 1 0.30 LADD† [28] 1893.8 1 0.30

Ours 180.9 1 0.30

#### 3.3 Spatial Temporal Heads

To train the discriminator for better understanding of the spatial information and temporal correlation, we employ separate spatial and temporal discriminator heads for adversarial training [31, 32]. The backbone of the discriminator is the encoder from the pre-trained diffusion model (i.e., UNet), which consists of four spatial-temporal blocks sequentially [10]. The first three blocks downsample the spatial resolution by a factor of 2, and the last block maintains the spatial resolution. We extract the output features from each spatial-temporal block and utilize a spatial head and a temporal head to determine whether the sample is real or fake. The discriminator can be conditioned on additional information via projection [58] to enhance performance. In our setting, we use the image condition c and σ′ as the projected condition C.

Spatial Head. For an input feature of shape b×t×c×h×w, the spatial discriminator first reshapes it to (bt) × c × h × w. This way, each frame feature in a video is processed separately. The architecture for our proposed spatial head is illustrated in the left part of Fig. 3.

Temporal Head. Even though the features obtained from the discriminator backbone contain spatialtemporal information, we observe that using only spatial discriminator heads causes the generator to produce frames that are all identical to the image condition. To achieve better temporal performance (e.g., more vivid motion), we propose to add a temporal discriminator head parallel to the spatial discriminator head. The input features are reshaped to (bhw) × c × t instead. The architecture for our temporal head is illustrated in the right part of Fig. 3.

### 4 Experiment

Implementation Details. We apply Stable Video Diffusion [13] as the base model across our experiments. All the experiments are conducted on an internal video dataset with around one million videos. We fix the resolution of the training videos as 768 × 448 with the FPS as 7. The training is conducted for 50K iterations on 8 NVIDIA A100 GPUs, using the SM3 optimizer [59] with a learning rate of 1e − 5 for the generator (i.e., UNet) and 1e − 4 for the discriminator. We set the momentum and β for both optimizers as 0.5 and 0.999, respectively. The total batch size is set as 32 using a 4

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

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

- Figure 4: Video Generation on Single Conditioning Images from Various Domains. We employ our method on various images generated by SDXL [60] to synthesized videos. The videos contain 14-frame at a resolution of 1024 × 576 with 7 FPS. The results demonstrate that our model can generate high-quality motion-consistent videos of various objects across different domains. Please refer to our webpage for whole video sequences.

steps gradient accumulation. We set the EMA rate as 0.95. We set Pmean = −1,Pstd = −1, and λ = 0.1 if not otherwise noted. At inference time, we sample videos at resolution of 1024 × 576.

#### 4.1 Qualitative Visualization

To comprehensively evaluate the capabilities of our method, we use SDXL [60] (with refiner) to generate images of different scenes at the resolution of 1024 × 1024. We then perform center crop on the generated images to get resolution as 1024 × 576, which serves as the condition of our approach to synthesize videos of 14 frames at 7 FPS. As shown in Fig. 4, our method can successfully generate videos of high-quality frames and consistent object movements with only 1 step during inference.

#### 4.2 Comparisons Results

Quantitative Comparisons. We present a comprehensive evaluation of our method compared to the existing state-of-the-art approach, AnimateLCM [21], UFOGen [25], LADD [28], and SVD [13]. To

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

(25steps)

X0.64 SVD

X22.97 SVD

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

(16steps)

X1.00 SVD

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

(8steps)

X2.00 AnimateLCM

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

(4steps)

X7.49 LADD

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

(1step)

X22.97 Ours

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

X22.97 UFOGen

(1step)

X22.97 LADD

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

(1step)

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

(25steps)

X0.64 SVD

SVD

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

(16steps)

X1.00 SVD

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

(8steps)

X2.00 AnimateLCM

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

(4steps)

X7.49 Ours

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

(1step)

X22.97 UFOGen

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

(1step)

X22.97

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

(1step)

- Figure 5: Comparison between SVD [13], AnimateLCM [21], LADD [28], UFOGen [25], and Our Approach. We provide the synthesized videos (sampled frames) under various settings for different approaches. We use SVD to generate videos under 25, 16, and 8 sampling steps, AnimateLCM to synthesize videos under 4 sampling steps, LADD and UFOGen to generate videos under 1 sampling step. AnimateLCM, LADD and UFOGen generates blurry frames with few-steps and single-step sampling. Our approach can accelerate the sampling speed by 22.9× compared with SVD while maintaining similar frame quality and motion consistency.

conduct a fair comparison on the SVD model, we train the AnimateLCM, UFOGen, and LADD on SVD using our video dataset. We follow the released code and instructions provided by AnimateLCM authors. Additionally, we include the officially released AnimateLCM-xt1.1 [21] by evaluating the first 14 generated frames and denote the approach as AnimateLCM∗. We try our best to implement LADD [28] and UFOGen [25] and denote respectively as LADD†, and UFOGen†. Note that simply re-using the discriminator from LADD [28] and UFOGen [25] leads to out-of-memory issue, since the computation in the video model is much larger than the image model. Here we replace the discriminator from LADD [28] and UFOGen [25] with the one proposed in our work.

We follow exiting works [61] by using Fréchet Video Distance (FVD) [30] as the comparison metric. Specifically, we use the first frame from the UCF-101 dataset [62] as the conditioning input and generate 14-frame videos at a resolution of 1024 × 576 at 7 FPS for all methods. The generation results are then resized back to 320 × 240 for FVD calculation. Our method is compared against SVD [13] and AnimateLCM [21], each using a different number of sampling steps. Furthermore, to better demonstrate the effectiveness of our method, we measure the generation latency of each method, which is calculated on running the diffusion model (i.e., UNet). Note that only for SVD [13], classifier-free guidance [63] is used, leading to higher computational cost.

As shown in Tab. 1, our method achieves comparable results to the base model using 16 discrete sampling steps, resulting in approximately a 23× speedup. Our method also outperforms the 8steps sampling results for AnimateLCM and AnimateLCM∗, indicating a speedup of more than 6×. For single-step evaluation, our method performs much better than existing step-distillation methods [25, 28] built upon image-based-diffusion models.

Qualitative Comparisons. We further provide qualitative comparisons across different approaches by using publicly available web images. Fig. 5 presents generation results from SVD [13] with 25, 16, and 8 sampling steps, AnimateLCM [21] with 4 sampling steps, UFOGen [25], LADD [28], and our method with 1 sampling step. As can be seen, our method achieves results comparable to the sampling results of SVD using 16 or 25 denoising steps. We notice significant artifacts for videos synthesized by SVD when using 8 denoising steps. Compared to AnimateLCM [21],UFOGen [25], and LADD [28], our method produces frames of higher quality and better temporal consistency, with fewer or same denoising steps, demonstrating the effectiveness of our proposed approach.

#### 4.3 Ablation Analysis

Effect of Discriminator Heads. We explore the effect of our proposed spatial and temporal heads by measuring the FVD on the UCF-101 dataset. We conduct latent adversarial training with three different discriminator settings to analyze the impact of our spatial and temporal discriminators. As shown in Tab. 2, training with only spatial heads (denoted as SP) or only temporal heads (denoted as TE) results in significantly worse performance than using all of them (denoted as SP+TE).

Nevertheless, since our discriminator backbone shares the same architecture as the spatial-temporal generator, the receptive field of each pixel on the feature maps provided by the backbone can cover a region both spatially and temporally. Additionally, we embed the frame index as an additional projected condition. Consequently, even when using only spatial heads or only temporal heads, the generated videos still exhibit reasonable frame quality and temporal coherence.

Effect of Noise Distribution for Discriminator. As shown in Fig. 6, following Eq. (5), Pmean and Pstd control the distribution of σt′, which is the noise level added to x0 or xˆ0 before passing to the discriminator as real and fake samples, respectively. We explore the effect of different noise distributions on model performance by calculating FVD on the UCF-101 dataset.

When the sampled σt′ is concentrated on small values, e.g., Pmean = −2 and Pstd = −1 in our case, we notice that the discriminator can quickly learn to distinguish real samples from fake ones. This leads to a significant drop in performance, as shown in Tab. 3 and Fig. 7.

On the other hand, when the noise level becomes too high, e.g., Pmean = 1 and Pstd = 1, the discriminator input, which is cin(σt′)ˆx′t = xˆ0+σ

′

√ tϵ

, results in small adversarial gradients for the generator. This causes increased artifacts in the generated videos, as shown in Fig. 7 and Tab. 3.

σt′2+1

Table 2: Analysis of discriminator. We measure FVD for models with different discriminator configurations. “SP” indicates that spatial heads and “TE” indicates temporal heads.

Table 3: FVD vs. σ′ distributions.

Pmean Pstd FVD

PDF

−2.0 −1.0 3370.4 −1.0 −1.0 180.9

- 0.0 1.0 416.7
- 1.0 1.0 632.9

SP+TE SP TE FVD 180.9 514.7 539.2

0.010

|Pmean = −2.0,Pstd = −1.0 Pmean = −1.0,Pstd = −1.0<br><br>Pmean = 0.0,Pstd = 1.<br><br>Pmean = 1.0,Pstd = 1.<br>|
|---|

0 0

0.008

0.006

0.004

0.002

0.000

10−2 10−1 10σ0 101 102

Figure 6: PDF of σ′.

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

𝑃 = −2,𝑃 = −1 𝑃 = −1,𝑃 = −1 𝑃 = 0,𝑃 = 1 𝑃 = 1,𝑃 = 1

- Figure 7: Analysis of σ′ Distributions. We investigate the impact of changing the distribution of σ′

by adjusting Pmean and Pstd. The results are shown with the same image conditioning. The first row and the second row display the first and last frames generated, respectively.

5 Discussion and Conclusion

In this work, we leverage adversarial training to reduce the denoising steps of the video diffusion model and thus improve its generation speed. We further enhance the discriminator by introducing spatial-temporal heads, resulting in better video quality and motion diversity. We are the first to achieve 1-step generation for video diffusion models while preserving comparable visual quality and FVD scores, democratizing efficient video generation to a broader audience by delivering more than 20× speedup for the denosing process.

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

SVD

(25steps) Ours

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

(1step)

- Figure 8: Limitations. We show that, for some conditional images, our model tends to generate a few unsatisfactory frames when complex motion might be required (Second Row). Similar artifacts can also be observed in frames generated from SVD by sampling at 25-steps (First Row).

Limitations. We observe that when the given conditioning image indicates complex motion, e.g.running, our model tends to generate unsatisfactory results, e.g.blurry frames, as shown in Fig. 8. Such artifacts are introduced by the original SVD model, as can be observed in Fig. 8. We believe a better text-to-video model can solve such issue.

This work successfully achieves single sampling step for video diffusion models. However, under such setting, the temporal VAE decoder and the encoder for image conditioning take a considerable portion of the overall runtime. We leave the acceleration of these models as future work.

### References

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2
- [2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [3] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662, 2023. 2
- [4] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Joshua V. Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. 2
- [5] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7038–7048, 2024. 2
- [6] Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris Metaxas, and Licheng Yu. Avid: Any-length video inpainting with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7162–7172, 2024. 2
- [7] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2
- [8] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2
- [9] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.
- [10] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563– 22575, 2023. 4, 6
- [11] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 2
- [12] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. arXiv preprint arXiv:2311.16498, 2023. 2
- [13] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3, 4, 6, 7, 8, 9
- [14] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.
- [15] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Animateanything: Fine-grained open domain image animation with motion guidance, 2023. 2

- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [17] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2
- [18] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024. 2
- [19] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3626–3636, 2022. 2
- [20] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 5907–5915, 2017. 2
- [21] Fu-Yun Wang, Zhaoyang Huang, Xiaoyu Shi, Weikang Bian, Guanglu Song, Yu Liu, and Hongsheng Li. Animatelcm: Accelerating the animation of personalized diffusion models and adapters with decoupled consistency learning. arXiv preprint arXiv:2402.00769, 2024. 2, 3, 6, 7, 8, 9
- [22] Shanchuan Lin and Xiao Yang. Animatediff-lightning: Cross-model diffusion distillation. arXiv preprint arXiv:2403.12706, 2024. 3
- [23] Xiang Wang, Shiwei Zhang, Han Zhang, Yu Liu, Yingya Zhang, Changxin Gao, and Nong Sang. Videolcm: Video latent consistency model. arXiv preprint arXiv:2312.09109, 2023. 2
- [24] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 32211–32252. PMLR, 2023. 2, 3
- [25] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-toimage generation via diffusion gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8196–8206, 2024. 2, 3, 4, 6, 7, 8, 9
- [26] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024. 3
- [27] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023. 3, 4, 5
- [28] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024. 3, 4, 5, 6, 7, 8, 9
- [29] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 2, 3
- [30] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new metric for video generation. In Deep Generative Models for Highly Structured Data, ICLR 2019 Workshop, New Orleans, Louisiana, United States, May 6, 2019. OpenReview.net, 2019. 2, 9
- [31] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1526–1535, 2018. 2, 6
- [32] Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N. Metaxas, and Sergey Tulyakov. A good image generator is what you need for high-resolution video synthesis. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 2, 6
- [33] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022. 2, 3, 4, 5

- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2
- [35] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023. 2
- [36] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.
- [37] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, JiaBin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22930–22941, 2023.
- [38] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2022.
- [39] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [40] Xiang Wang, Shiwei Zhang, Han Zhang, Yu Liu, Yingya Zhang, Changxin Gao, and Nong Sang. Videolcm: Video latent consistency model. arXiv preprint arXiv:2312.09109, 2023. 3
- [41] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. 3
- [42] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36, 2024. 3
- [43] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023. 3, 5
- [44] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [45] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.
- [46] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. arXiv preprint arXiv:2404.13686, 2024.
- [47] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023.
- [48] Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao, and Tat-Jen Cham. Trajectory consistency distillation. arXiv preprint arXiv:2402.19159, 2024. 3
- [49] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 3
- [50] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for highquality diffusion-based text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023. 3

- [51] Jonas Kohler, Albert Pumarola, Edgar Schönfeld, Artsiom Sanakoyeu, Roshan Sumbaly, Peter Vajda, and Ali Thabet. Imagine flash: Accelerating emu diffusion models with backward distillation. arXiv preprint arXiv:2405.05224, 2024. 3, 4
- [52] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 3
- [53] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In International conference on machine learning, pages 30105–30118. PMLR, 2023. 5, 6
- [54] Axel Sauer, Kashyap Chitta, Jens Müller, and Andreas Geiger. Projected gans converge faster. Advances in Neural Information Processing Systems, 34:17480–17492, 2021. 5, 6
- [55] Jae Hyun Lim and Jong Chul Ye. Geometric gan. arXiv preprint arXiv:1705.02894, 2017. 5
- [56] Pierre Charbonnier, Laure Blanc-Féraud, Gilles Aubert, and Michel Barlaud. Deterministic edge-preserving regularization in computed imaging. IEEE Transactions on image processing, 6(2):298–311, 1997. 5
- [57] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In International conference on machine learning, pages 3481–3490. PMLR, 2018. 5
- [58] Takeru Miyato and Masanori Koyama. cgans with projection discriminator. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. 6
- [59] Rohan Anil, Vineet Gupta, Tomer Koren, and Yoram Singer. Memory efficient adaptive optimization. Advances in Neural Information Processing Systems, 32, 2019. 6
- [60] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 7
- [61] Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, Jun-Yan Zhu, and Jia-Bin Huang. On the content bias in fréchet video distance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 9
- [62] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 9
- [63] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 9

