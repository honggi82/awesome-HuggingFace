## MicroDreamer: Efficient 3D Generation in ∼20 Seconds by Score-based Iterative Reconstruction

Luxi Chen∗, Zhengyi Wang∗, Zihan Zhou, Tingting Gao, Hang Su, Member, IEEE, Jun Zhu, Fellow, IEEE, Chongxuan Li† Member, IEEE

### arXiv:2404.19525v3[cs.CV]18Oct2024

Abstract—Optimization-based approaches, such as score distillation sampling (SDS), show promise in zero-shot 3D generation but suffer from low efficiency, primarily due to the high number of function evaluations (NFEs) required for each sample and the limitation of optimization confined to latent space. This paper introduces score-based iterative reconstruction (SIR), an efficient and general algorithm mimicking a differentiable 3D reconstruction process to reduce the NFEs and enable optimization in pixel space. Given a single set of images sampled from a multi-view score-based diffusion model, SIR repeatedly optimizes 3D parameters, unlike the single-step optimization in SDS. With other improvements in training, we present an efficient approach called MicroDreamer that generally applies to various 3D representations and 3D generation tasks. In particular, MicroDreamer is 5-20 times faster than SDS in generating neural radiance field while retaining a comparable performance and takes about 20 seconds to create meshes from 3D Gaussian splatting on a single A100 GPU, halving the time of the fastest optimization-based baseline DreamGaussian with significantly superior performance compared to the measurement standard deviation. Our code is available at https://github.com/ML-GSAI/MicroDreamer.

Index Terms—3D Generation, Diffusion Model, Multi-view Diffusion, Score Distillation Sampling

I. INTRODUCTION

# R

ECENTLY, optimization-based approaches [1]–[19] particularly score distillation sampling (SDS) [1], [2] have

emerged as promising avenues for 3D generation based on text-to-image diffusion models [20]–[25]. These approaches are appealing due to their minimal, or even zero reliance on 3D data, in contrast to the data-intensive requirements of feedforward approaches [26]–[53]. This advantage is particularly significant given that 3D data are costly and scarce. Despite their promising capabilities, optimization-based approaches suffer from low efficiency due to the extensive number of

This work was supported by Beijing Nova Program (No. 20230484416); NSF of China (No. 62076145); Beijing Natural Science Foundation (No. L247030); the Kuaishou Research Fund. The work was partially done at the Engineering Research Center of Next-Generation Intelligent Search and Recommendation, Ministry of Education.

∗ Equal contribution.

Luxi Chen, Zihan Zhou, and Chongxuan Li are with the Gaoling School of AI, Renmin University of China, and Beijing Key Laboratory of Big Data Management and Analysis Methods, Beijing 100872, China. E-mail: clx1489@ruc.edu.cn; zhouzihan2@ruc.edu.cn; chongxuanli@ruc.edu.cn. †Corresponding author: Chongxuan Li.

Zhengyi Wang, Hang Su, and Jun Zhu are with Dept. of Comp. Sci. & Tech., BNRist Center, Tsinghua-Bosch Joint ML Center, Tsinghua University, Beijing 100084, China. E-mail: wang-zy21@mails.tsinghua.edu.cn; suhangss@tsinghua.edu.cn; dcszj@tsinghua.edu.cn

Tingting Gao is with Kuaishou Technology, Beijing, China. E-mail: lisize@kuaishou.com.

[Figure 1]

Fig. 1: MicroDreamer surpasses the fastest optimizationbased baseline DreamGaussian [8] in terms of both efficiency and sample quality. The optimization-based methods are highlighted in red. See Tab. III for a comprehensive comparison with more baselines.

function evaluations (NFEs), i.e. forward passes of the diffusion model, required for each 3D object generation. Moreover, when adopting the latent diffusion model (LDM) [23] framework, most of these approaches can only compute loss in latent space rather than pixel space, which requires backpropagation through an encoder [54], [55] and further lowers the efficiency. Even if the loss function can be written in a data-predicting form [4], this type of loss is challenging to optimize effectively (see evidence in Fig. 11). The fastest approach DreamGaussian [8] still requires about 40 seconds to generate a 3D object even employing 3D Gaussian splatting [56].

In comparison, the multi-step reconstruction process of 3D representations that enable differentiable rendering, such as neural radiance field (NeRF) [57], [58] and 3D Gaussian splatting (3DGS) [56], produces 3D contents extremely fast because they do not involve large generative neural networks. However, such approaches rely on true 3D data, i.e. abundant real multi-view images, making them unfeasible for text-to-3D and image-to-3D generation tasks. There exist works [59]–

[Figure 2]

- Fig. 2: MicroDreamer can generate a high-quality mesh (as illustrated above) in about 20 seconds on a single A100, built on a multi-view diffusion model without additional 3D data. See supplementary materials for 3D visualization.

[66] for 3D generation attempt to generate multi-views first to reconstruct 3D object directly. Still, such methods may require the diffusion model to simultaneously generate multiview images with their corresponding 3D priors [60], [63], and they require longer than one minute to reconstruct a 3D object. (see Sec. II for a review and Sec. VI for comparison).

To speed up the generation process, this paper presents an efficient and general 3D generation algorithm termed scorebased iterative reconstruction (SIR), leveraging reconstruction to reduce NFEs and enable optimization in pixel space. Like SDS, SIR iteratively updates 3D parameters, leveraging a multi-view diffusion model without additional 3D data or 3D prior. However, in each iteration, SIR distinguishes itself by repeatedly optimizing 3D parameters given a set of images produced by the diffusion model, mimicking the efficient 3D reconstruction process to reduce the total NFEs (see Fig. 7). To obtain 3D-consistent and high-quality images as the ground truth for better reconstruction in each iteration, we carefully design a hybrid forward process and a sampling process to refine the images rendered from the 3D object optimized in the latest iteration. Besides, even mapped back to pixel space through the decoder in LDM, the refined images are still of high quality for reconstruction, enabling optimization in pixel space to speed up further (see Fig. 10).

We provide a general and compatible configuration for SIR, and the comprehensive system is named MicroDreamer, highlighting its efficiency for 3D generation. As detailed in Sec. V, we introduce an initialization strategy for 3D objects, an annealed time schedule for diffusion, additional losses on

reference images for image-to-3D, and a refinement procedure for deriving high-quality meshes from 3DGS.

Comprehensive studies demonstrate the generality and efficiency of our proposed method. In particular, SIR and MicroDreamer broadly apply to NeRF and 3DGS and both text-to-3D and image-to-3D tasks, as detailed in Sec. VI. Employing three widely adopted multi-view diffusion models [67]–[69], we systematically compare SIR and SDS for NeRF generation. Retaining a comparable performance, SIR can accelerate the generation process by 5 to 20 times. Besides, MicroDreamer can efficiently produce 3DGS and further refine it into high-quality meshes, delivering consistent 3D meshes in about 20 seconds on a single A100 GPU – about twice as fast as the most competitive optimization-based alternatives, DreamGaussian [8], with significantly (compared to the measurement standard deviation) superior performance. Remarkably, MicroDreamer is on par with speed compared to representative feed-forward methods [70] trained on extensive 3D data, with a very competitive CLIP similarity [71].

II. RELATED WORK

Optimization-based 3D generation. Built upon text-to-image diffusion models, optimization-based approaches [1]–[19] usually generate 3D objects without additional 3D data. Among them, the most relevant line of work [1], [2], [4], [7], [9], [17]–[19] proposes various distillation algorithms. Besides SDS [1], ProlificDreamer [7] proposes variational score distillation (VSD) to produce high-fidelity 3D objects via variational inference. LucidDreamer [9] incorporates DDIM

###### Reconstruction

[Figure 3]

SIR (Ours)

[Figure 4]

[Figure 5]

[Figure 6]

Abundant

Iterative optimization

[Figure 7]

3D views

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Refine

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Reconstruction

Multi-view

[Figure 21]

diffusion

[Figure 22]

SDS

Iterative optimization

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Reconstruction

Diffusion

[Figure 30]

[Figure 31]

[Figure 32]

Lowing NFEs from reconstruction

- Fig. 3: Overview of SIR. SIR is an optimization-based 3D generation method that marries the strengths of reconstruction and iterative optimization. SIR reutilizes the samples from diffusion multiple times through reconstruction, reducing the total NFEs, enabling optimization in pixel space, and improving efficiency.

inversion to strengthen the forward process. ReconFusion [15] utilizes diffusion priors to generate novel views for single-step optimization in sparse view reconstruction tasks, aiming to enhance the quality of scene reconstruction. Though promising, many of these methods suffer from inefficiency problems due to large NFEs and optimization in latent space. Distinct from existing algorithms, SIR mimics 3D reconstruction processes to optimize the 3D parameters multiple times given a set of images produced by diffusion. Our experiments demonstrate that SIR is 5-20 times faster than SDS [1] on NeRF and twice as fast as the most competitive baseline DreamGaussian [8] with better generation quality on meshes.

Feed-forward 3D generation. Contrary to optimizationbased methods, feed-forward methods [26]–[42], [44]–[53] use large-scale 3D datasets [72], [73] training to achieve a process that can directly generate 3D objects. Early methods are characterized by their speed but often produce lowerquality 3D structures with simple textures. Several recent methods [32], [36], [37], [48], [53], [70] have exhibited the feasibility of training a Transformer utilizing more data to achieve reliable 3D content from a single image. Generating 3D content from a single view can be quite challenging. Therefore, several follow-up works [38]–[40], [42], [43], [45], [49], [51] have leveraged multi-view diffusion models to generate multiple views as input and then trained a feedforward model to produce 3D content from these views. Remarkably, MicroDreamer is on par with speed compared to such methods trained on extensive 3D data, with a very competitive 3D quality measured by CLIP similarity [71].

Multi-view prediction based 3D generation. There is also a line of work [59]–[66] dedicated to enhancing the output of multi-view diffusion models by training on 3D datasets to reconstruct 3D objects using a single reconstruction process with

no or few iterations. Some approaches like Wonder3D [60] typically rely on 3D prior knowledge and are limited to specific 3D representations with a long reconstruction time. Another aspect of works like IM-3D [64] and V3D [65] involves fine-tuning the video diffusion model using 3D data and employing the generated 3D-aware multi-view images as ground truth for reconstruction. The efficiency of these methods is limited by the sampling efficiency of the video diffusion model and the long reconstruction time. Consequently, the total time required to generate a 3D object using these methods typically exceeds one minute. In contrast, thanks to the carefully designed iterative process in SIR, MicroDreamer is more efficient and applies to various 3D representations (see Tab. III for comparison).

III. BACKGROUND

We present background on 3D representations, diffusion models, multi-view diffusion models, and current optimization-based 3D generation methods sequentially.

A. 3D representation

Neural radiance fields [57], [58] (NeRF) and 3D Gaussian splatting [56] (3DGS) have emerged as popular 3D representations. NeRF employs an MLP to predict the colors and density of the input space coordinates. 3DGS consists of multiple 3D Gaussians parameterized by the colors, centers, scales, and rotation quaternions. We denote the corresponding tunable parameters in both representations as θ. Given camera poses c, both approaches define a differentiable rendering process, denoted by g(θ,c). They are proven efficient and effective in 3D reconstruction [56], [57] and generation [1], [8].

- B. Diffusion models

- A diffusion model [20]–[22] consists of a forward process

and a sampling process. The forward process gradually adds Gaussian noise to an input image from time 0 to T. For any t ∈ (0,T), the noise-adding process can be written as follows:

xt = αtx0 + σtϵ := Noise-adding(x0,0 → t), ϵ ∈ N(0,I), (1)

where the coefficients αt and σt form the noise schedule of the diffusion model. A noise prediction network ϵϕ(xt,t) with parameters ϕ is trained to predict the noise in the input xt with the corresponding noise level, i.e. time t. Plugging in the noise prediction network into Eq. (1), we can solve x0 from a noisy image xt of time t by a single-step prediction as follows:

xˆt0 =

1 αt

xt −

σt αt

ϵϕ(xt,t), (2) which is an efficient approximation of the original input.

Instead of using Eq. (2) directly, the sampling process of diffusion gradually denoises through the noise prediction network and generates images from pure Gaussian noise. Among existing samplers [74]–[79], the denoising diffusion implicit models (DDIM) [80] facilitate a sequence of samplers with random noise control η. When η = 0, it solves the equivalent probability flow ordinary differential equation (ODE) [22] of the diffusion model and enjoys a fast sampling process with a small number of function evaluations (NFEs)1. In this setting, the one-step sampling of DDIM is given by:

xt−1 =

αt−1 αt

xt + σt−1 −

αt−1 αt

σt ϵϕ(xt,t) := Sampler(xt,t → t − 1). (3)

We refer the readers to the original paper [80] for the formula of η > 0. For η = 1, it represents a standard SDE sampler [21], with a higher tolerance for mismatches in the distributions of latent variables [81], [82].

Besides, when η = 0 there is an inverse process (called DDIM inversion) that maps the distribution of images to the same distributions of noisy images as in Eq. (1) but it can maintain the unique feature of an input image and reconstruct it accurately through the corresponding DDIM sampler in Eq. (3) (see Fig. 5a). The (one-step) DDIM inversion [80] is formulated as follows:

xt+1 =

αt+1 αt

xt + σt+1 −

αt+1 αt

σt ϵϕ(xt,t) := Inversion(xt,t → t + 1). (4)

- C. Multi-view diffusion models

After being trained on a modest amount of 3D data, diffusion models can generate 3D-consistent multi-view, known as multi-view diffusion models. Among them, MVDream [67] takes text inputs and outputs multi-view images consistent in 3D. In contrast, Zero-1-to-3 [6] and ImageDream [69] focus on the Image-to-3D task, taking an additional reference image as

1Throughout the paper, we refer to the number of forward passes through ϵϕ as NFEs.

input. These models output new viewpoint images consistent with the reference image. This paper directly utilizes these pre-trained multi-view diffusion models without any further fine-tuning. Notably, such multi-view diffusion models cannot provide sufficient consistent multi-view images for 3D reconstruction directly (see Fig. 12a). Given that the multi-view diffusion models in [59]–[64] often incorporate additional 3D priors and do not align with our setting in Sec. V, we choose not to employ them in this paper.

D. Optimization-based algorithms for 3D generation

Built upon (multi-view) diffusion models, optimizationbased methods (see Sec. II for a review) aim to generate 3D content in a zero-shot manner. Among them, score distillation sampling (SDS) [1], [2] is the most representative and popular approach. Formally, denoting the rendered images as x = g(θ,c), SDS repeats adding noise to x according to Eq. (1) and updating the 3D parameters θ by

∇θJSDS(x = g(θ);ϕ) :=Et w(t)(ϵϕ(αtx + σtϵ,t) − ϵ)

∂x ∂θ

, (5)

where w(t) is a fixed weighting function, we omit the dependency on the prompt y and camera c for simplicity. Notably, by reparameterization [4] according to Eq. (2), SDS has an equivalent data-prediction form:

∇θJSDS(x = g(θ);ϕ) = Et

w(t)αt σt

∂x ∂θ

(x − xˆt0)

. (6)

IV. SCORE-BASED ITERATIVE RECONSTRUCTION

We first analyze factors contributing to the efficiency bottleneck of existing work in Sec. IV-A, motivating score-based iterative reconstruction (SIR), an efficient and versatile algorithm combining the advantages of differentiable 3D reconstruction and optimization methods in Sec. IV-B. We introduce a carefully designed diffusion-based process to produce refined multi-view images as ground truth for better reconstruction in Sec. IV-C. Besides, we demonstrate how SIR facilitates the optimization of 3D content directly within pixel space in Sec. IV-D.

A. Efficiency bottleneck of existing work

As a motivation of SIR, we analyze the efficiency bottleneck of existing optimization-based methods [1], [2], [4], [7], [9], [17], [18]. We take the widely adopted SDS [1] as a representative example and our analysis also applies to other algorithms.

On the one hand, according to Eq. (5), SDS iteratively optimizes the 3D parameters based on a 2D diffusion. It necessitates high NFEs because it requires a forward pass of the 2D diffusion in each update of the 3D parameters. On the other hand, when employed within the latent diffusion model (LDM) [23] upon a variational auto-encoder (VAE) [55], [83], SDS computes the loss in latent space. This process requires backpropagation through the VAE encoder, further lowering generation efficiency. These arguments are validated by the empirical results in Fig. 4 quantitatively.

[Figure 33]

2.9 %

[Figure 34]

11.3 %

18.3 %

Time bottleneck

67.0 %

- Fig. 4: Time proportion in SDS optimization. We record the time proportions of all components in SDS on 3DGS. The bottleneck lies in the large NFEs and updating in latent space.

As for the latent space optimization, we emphasize that although SDS has a corresponding data-prediction form in Eq. (6), it is nontrivial to map the predicted x(0t) back to pixel space through the VAE decoder for loss calculation. This is because the single-step prediction in Eq. (2) yields poor samples in the pixel space, leading to suboptimal 3D results (See experiments in Tab. 11 of Sec. VI).

- B. Mimicking 3D reconstruction to reduce the NFEs

Motivated by the analysis of SDS, we propose SIR to reduce NFEs and enable optimization in pixel space to enhance efficiency. Naturally, reutilizing the outcomes of diffusion for successive updates of the 3D object—mimicking the process of differentiable 3D reconstruction could substantially decrease the overall NFEs required and enable optimization in pixel space. However, 3D reconstruction typically requires a sufficient number of consistently aligned multi-view images, which cannot be directly obtained from the current multi-view diffusion models [6], [67]–[69] (see Sec. II for a review and discussion). Therefore, similar to existing optimization-based methods, we introduce a reconstruction-based algorithm with multiple iterations of optimizing dubbed score-based iterative reconstruction (SIR), detailed as follows.

Formally, SIR consists of K reconstruction iterations. In the k-th iteration, where k = 0,1,...,K −1, given initial 3D parameters θ0(k), we randomly select several camera poses c(k) (detailed in Sec. V) and employ rendering function g(·,·) to obtain multi-view images of the current 3D object as follows:

##### x(k) = g(θ0(k),c(k)). (7)

For simplicity, let x(k) be a vector by flattening all images and concatenating them together, and so do any subsequent set of multi-view images.

As detailed in Sec. IV-C, x(k) is then refined via a pretrained multi-view diffusion model [6], [67]–[69]. The outcome, denoted as xˆ(k), serves as the ground truth for reconstruction in the current iteration. In particular, starting from θ0(k), keeping the camera poses c(k) unchanged, we optimize

Input Noisy Refined

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|Sampling<br><br>|
|---|---|
| | |

|[Figure 38]<br><br>|
|---|

Forward

|[Figure 39]|
|---|

DDIM inversion, Eq. (4), ~20s

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|Sampling<br><br>|
|---|---|
| | |

|[Figure 43]<br><br>|
|---|

Forward

|[Figure 44]|
|---|

Artifacts

Noise-adding, Eq. (1), ~14s

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|Sampling<br><br>|
|---|---|
| | |

|[Figure 48]<br><br>|
|---|

Forward

|[Figure 49]|
|---|

Hybrid process, Eq. (9), ~18s

Fig. 5: The hybrid forward process is more efficient than DDIM inversion and generates better samples than noiseadding. We present the final results and sampling time on 20 iterations of SIR for three forward processes. Notably, the Noise-adding process may generate artifacts that contain unexpected elements compared to the input.

the 3D parameters w.r.t. the following reconstruction loss for I steps:

JSIR(θ;c(k),xˆ(k)) = ∥g(θ,c(k)) − xˆ(k)∥, (8) where ∥·∥ can be any proper norm operator in principle while we choose ℓ1 norm in our experiments. We use the final 3D parameters of the k-th iteration as the initialization of the next one, i.e. θ0(k+1) = θI(k) and outputs θ0(K) finally. See Sec. V for the initialization of the 0-th iteration, i.e. θ0(0).

Here we assume the rendered images and those sampled by the diffusion model have the same dimensional for simplicity. In Sec. IV-D we will further discuss how to deal with the latent diffusion model (LDM) [23]. Before that, we discuss how to obtain 3D-consistent and high quality xˆ(k) from x(k) for reconstruction in Sec. IV-C.

C. Refining multi-view images for reconstruction by diffusion

We treat x(k) as noise-free (i.e., at time 0 of the diffusion process) but low-quality images, refined by a forward process followed by a sampling process.

The diffusion model has two theoretically equivalent forward processes [22]: the noise-adding process modeled by stochastic differential equations (SDEs) in Eq. (1) and the

Algorithm 1 Score-based iterative reconstruction (SIR)

- 1: Input: The number of iterations K, an initial 3D object

θ0(0), the number of reconstruction steps I and a set of camera poses {c(k)}Kk=0−1.

- 2: Output: A final 3D content θ0(K).
- 3: for k from 0 to K − 1 do
- 4: Render N images x(k) = g(θ0(k),c(k))
- 5: Obtain noisy x˜(k) from forward process for x(k)
- 6: Obtain refined xˆ(k) from sampling process for x˜(k)
- 7: for i from 0 to I − 1 do
- 8: Compute loss L = ∥g(θ,c(k)) − xˆ(k)∥
- 9: Compute the gradient ∇L, update θi(k) to θi(+1k)
- 10: end for
- 11: end for

inversion process based on the probability flow ordinary differential equations (ODEs) in Eq. (4). In comparison, the noiseadding process is more efficient without function evaluation, but sampling after the noise-adding process may produce unexpected artifacts (see Fig. 5). The inversion process can better preserve the 3D consistency and overall information of the current 3D object, but it necessitates more NFEs with low efficiency. We carefully design a hybrid forward process that initially adds noise and then performs DDIM inversion. Compared to common noise-adding and inversion processes, the hybrid forward process achieves a better balance in quality and efficiency, as Fig. 5 shows.

Specifically, the hybrid forward process adds noise to time

- t1(k) first and then performs DDIM inversion [80] to time t(2k), where t(1k) ∈ (0,T) and t(2k) ∈ [t(1k),T) are hyperparameters2.

In contrast to random sampled t(2k) in existing algorithms including SDS [1], we adopt a linearly decreased schedule for

- t2(k) as k progresses, detailed in Sec. V. Formally, the process is defined as:

##### x˜(k) = Inversion(Noise-adding(x(k),0 → t(1k)),t(1k) → t(2k)),

(9)

where Noise-adding(·) is defined in Eq. (1) and Inversion(·) is defined in Eq. (4). The ablation in Fig. 12d shows the effectiveness of our hybrid forward process.

Note that generally x(k) does not follow the model distribution defined by the diffusion and the resulting x˜(k) does not strictly adhere to the marginal distribution at the corresponding time. Nevertheless, we still use existing sampling algorithms to obtain refined images xˆ(k) from x˜(k) as follows:

##### xˆ(k) = Sampler(˜x(k),t(2k) → 0). (10)

Although other advanced sampling methods [74]–[79] exist, we choose the popular DDIM in Eq. (3) and tune its noise hyperparameter η. The optimal value of η depends on the base model and we search for the best one in {0,0.5,1} as detailed in Tab. I.

The whole process of SIR is presented in Algorithm 1. Compared with existing methods [1], [7], [9], [15] dedicated

2Note that if t(1k) = t(2k), the process is purely adding noise.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

- View 1
- View 2

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

- View 1
- View 2

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Initial Iter=0 Iter=10 Iter=20 Output

Fig. 6: Visualization of the optimization process in SIR. The visual quality of the 3D samples increases along with the iterations.

to enhancing 3D generation quality, SIR benefits from reconstruction and lowers the NFEs in total, consequently improving generation efficiency. (see Fig. 7 for the comparison to SDS).

We visualize the optimization process of SIR in Fig. 6. With the increase in the number of iterations, the visual quality of the 3D content improves, validating the effectiveness of our iterative reconstruction.

D. Enabling optimization in pixel space

Within the widely adopted framework of LDM [23], the rendered images are mapped through an encoder E [54], [55] to a latent space, where the loss function is calculated. Consequently, the gradients must be propagated back through the encoder, further reducing the efficiency of 3D object generation. In such a case, the SIR loss in latent space is given by:

##### JSIR-latent(θ;c(k),xˆ(k)) = ∥E(g(θ,c(k))) − xˆ(k)∥. (11)

An alternative approach maps the diffusion output back to pixel space via the corresponding decoder D, allowing for a similar loss function to be defined directly in pixel space. This method enables direct updates to 3D parameters without passing gradients through the encoder, thus enhancing efficiency.

As discussed in Sec. IV-A, previous optimization-based methods like SDS fail to be effective when applied in pixel space. In contrast, SIR achieves higher quality generation results through a carefully designed refinement process, thereby enabling optimization in pixel space. The SIR loss in pixel space is formalized as:

##### JSIR-pixel(θ;c(k),xˆ(k)) = ∥g(θ,c(k)) − D(ˆx(k))∥, (12)

TABLE I: Key hyperparameters of MicroDreamer on three base diffusion models. All models are employed to generate NeRF and the last two are employed to generate 3DGS and mesh. By default, the hyperparameters are shared across the two 3D representations. Otherwise, those for 3DGS are shown in brackets. T is the end time of diffusion.

Model Select MVDream [67] Stable Zero123 [68] ImageDream [69] Diffusion CFG [85] 7.5 3.0 3.0 (2.0) Forward process hybrid hybrid hybrid Time schedule of t2 0.8T → 0.5T

0.8T → 0.6T (0.8T → 0.4T) Time schedule of t1 t1 = t22/T t1 = 0.6t2

0.8T → 0.2T (0.9T → 0.2T)

t1 = t22/T (t1 = 0.6t2) Sampling process DDIM, η = 0.0 DDIM, η = 0.5 DDIM, η = 1.0 Discretization steps 50 20 10

3D training Resolution 64 → 128

64 → 128 (256) Background learned by NN always white always white # camera views 4 4 (6) 4 # initialized steps 50 15 50 # iterations K 50 30 (20 or 30) 30 # reconstruction steps I 15 15 15 Loss type ℓ1 ℓ1 ℓ1 Ref. color loss - 0.1 (0.3) 0 Ref. opacity loss - 0.001 (0.01) 0

64 → 128 → 196 (256)

which is about 2-3 times faster than Eq. (11) for 3D parameter optimization, as the analysis results shown in Fig. 10. It is set as the default loss throughout the paper.

V. MICRODREAMER

We provide a general configuration in the 3D training and diffusion for the SIR algorithm. The comprehensive system is called MicroDreamer to highlight its efficiency.

3D initialization and camera views. We utilize a direct reconstruction approach to initialize the 3D content. Specifically, we optimize the loss function in Eq. (8) by several steps (see Tab. I for detailed values) to update the 3D parameters, where xˆ represents the images sampled from random noise via the pre-trained multi-view diffusion models. The camera views are uniformly sampled following the corresponding baselines [8], [84] except the azimuth angles of different views in the same iteration are evenly distributed. In this setting, we are unable to directly utilize the diffusion model in [59], [60], [62], [63] due to their fixed camera views conditioned generation.

Annealed time schedule. We utilize an annealed time sched-

ule {t(2k)}Kk=0−1 for the end of the forward process. Intuitively, as the quality of the 3D assets improves, the input x in Eq. (7)

becomes more proximate to the model distribution, thus requiring fewer steps to refine. This differs from SDS, which samples uniformly random t2. Our preliminary experiments in Fig. 12c suggest that a linearly annealed schedule is sufficient. The endpoints of the schedule depend on the diffusion model, detailed in Tab. I.

Reference image loss for image-to-3D. A reference image is available in image-to-3D, which is regarded as the ground truth front view of the 3D object on Stable Zero123 [68] following DreamGaussian [8]. In this way, we add the reference loss in the same form as the reconstruction loss with Eq. (8) in each training iteration. The weight set for the reference loss can be seen in Tab. I.

TABLE II: Codebases and checkpoints. We provide URLs for the open-source assets we used in this paper.

Model URL Codebases

[84] https://github.com/threestudio-project/threestudio [8] https://github.com/dreamgaussian/dreamgaussian

###### Checkpoints

[87] https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k

- [67] https://github.com/bytedance/MVDream-threestudio
- [68] https://huggingface.co/stabilityai/stable-zero123
- [69] https://github.com/bytedance/ImageDream

Baselines [30] https://github.com/openai/point-e [29] https://github.com/openai/shap-e [39] https://github.com/One-2-3-45/One-2-3-45 [37] https://github.com/VAST-AI-Research/TriplaneGaussian [60] https://github.com/xxlong0/Wonder3D [43] https://github.com/3DTopia/LGM

- [70] https://github.com/3DTopia/OpenLRM

3DGS settings and mesh refinement. For simplicity, our approach on 3DGS largely adheres to the settings in DreamGaussian [8] unless specified. We incorporate a densify and prune procedure at every 100 updates during the initial 300 updates. At the end of optimization, we do a last prune and remove potential white Gaussians with scales larger than 0.01. We follow the mesh extraction method in LGM [43] and employ a threshold value of 5 in the marching cube algorithm [86]. We utilize SIR to optimize the exported mesh texture for one iteration with 30 steps, and we use a noiseadding process and DDIM with η = 0 for simplicity.

VI. EXPERIMENTS

We present the experimental details and results on NeRF, 3DGS, and ablation sequentially.

- A. Experimental details

We present key hyperparameters of MicroDreamer in Tab. I. For implementing SIR and SDS on NeRF, we choose the popularly used framework threestudio [84]. For the implementation of SIR on 3DGS, we follow the framework from DreamGaussian [8]. Most hyperparameters are not sensitive to the results, and we follow the previous framework [8], [84] for these settings. For important hyperparameters, including the number of iterations, camera views, and the time schedule, see Sec. VI-D for ablation. See Tab. II for details about the URL of the codebases and checkpoints we used in this paper.

For a fair comparison, we utilize the widely used CLIP similarity [71] for quantitative comparison following DreamGaussian [8]. We consider 8 views at 0 elevation and evenly distributed in azimuth angles starting from 0. For all methods, the corresponding images are rendered from NeRF in Sec. VI-B and mesh in Sec. VI-C. We report the average CLIP similarities between these images and the reference image or text.

- B. Results on NeRF

We apply SIR algorithm on NeRF [57], [58], [84] leveraging three multi-view diffusion models from MVDream [67], Stable

[Figure 241]

[Figure 242]

[Figure 243]

(a) Comparion on MVDream [67] (b) Comparion on Stable Zero123 [68] (c) Comparion on ImageDream [69]

- Fig. 7: Comparison of SIR and SDS on NeRF. We plot the curve of the CLIP similarity in the generation process and final NFEs on different models. SIR lowers the NFEs and achieves a 5-20 times acceleration to achieve a competitive quality.

SDS SIR (Ours)

“a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes”

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

“an astronaut riding a horse”

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

“A bald eagle carved out of wood”

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

Input MVDream Based, ~104s

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

SDS SIR (Ours)

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

Input Stable Zero123 Based, ~37s

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

SDS SIR (Ours)

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

Input ImageDream Based, ~32s

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

- Fig. 8: Qualitative comparison on NeRF. SIR can generate NeRF of higher visual quality than SDS in a short time.

DreamGaussian ~40 seconds 500 NFEs

MicroDreamer-20 iters (Ours) ~21 seconds ~350 NFEs

Input

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

Fig. 9: Qualitative comparisons on mesh from 3DGS. MicroDreamer produces superior meshes, characterized by enhanced texture and reduced geometric artifacts, with greater efficiency than DreamGaussian.

cess 5-20 times while holding a competitive CLIP similarity, as shown in Fig. 7.

To provide a comprehensive comparison, we qualitatively analyze the generation results of SDS and SIR under identical time conditions, as illustrated in Fig. 8. Notably, at the moment of convergence in SIR, we find that SDS has not achieved satisfactory results, such as producing ambiguous outcomes. These observations support the quantitative results, demonstrating that SIR is general and more efficient.

C. Results on 3D Gaussian splatting

Qualitative comparisons. In Fig. 9, we present the generated 3D meshes comparing the fastest optimization-based baseline, DreamGaussian [8], with our method. MicroDreamer reduces DreamGaussian’s generation time by half while exhibiting superior performance, including improved texture and enhanced geometric structure.

Zero123 [68], and ImageDream [69]. For each model, we selected 6 input prompts from a widely used codebase [84] for testing, calculated NFEs, and recorded the average CLIP similarity during the generation process. Compared with SDS, SIR lowers the total NFEs and accelerates the generation pro-

Quantitative comparisons. We compare MicroDreamer with eight competitive baselines including Point-E [30], Shap-

TABLE III: Quantitative comparisons. MicroDreamer significantly outperforms the strong optimization-based baseline DreamGaussian in quality and efficiency and remains competitive with feed-forward methods. All results are averaged over three runs.

Method CLIP similarity ↑ Generation time ↓ Point-E [30] 0.566 ± 0.0011 ∼ 24s Shap-E [29] 0.626 ± 0.0030 ∼ 5s One-2-3-45 [39] 0.617 ± 0.0025 ∼ 42s Wonder3D [60] 0.696 ± 0.0017 ∼ 170s TriplaneGaussian [37] 0.691 ± 0.0016 ∼ 7s LGM [43] 0.700 ± 0.0017 ∼ 3s Open-LRM-mix-base-1.1 [70] 0.704 ± 0.0000 ∼ 23s

DreamGaussian [8] 0.692 ± 0.0015 ∼ 30 + 10s DreamGaussian-300 iter [8] 0.641 ± 0.0032 ∼ 18 + 10s MicroDreamer-20 iter (Ours) 0.697 ± 0.0009 ∼ 18 + 3s MicroDreamer-30 iter (Ours) 0.711 ± 0.0007 ∼ 26 + 3s

[Figure 419]

[Figure 420]

(a) On Stable Zero123 [68]. (b) On ImageDream [69].

#### Fig. 10: Optimization in pixel space accelerates generation.

E [29], One-2-3-45 [39], TriplaneGaussian3 [37], Wonder3D [60], LGM [43], Open-LRM [70] and DreamGaussian [8]. We record the generation time for each model on a single NVIDIA A100 (80GB) GPU and compute the average CLIP similarity for mesh on a test dataset consisting of 87 images collected from previous works [8], [39], [59], as shown in Tab. III. MicroDreamer generates significantly better 3D content than DreamGaussian, as indicated by the standard deviation across multiple runs, and has higher efficiency.

In addition, MicroDreamer is on par with speed compared to feed-forward methods [70] trained on a substantial amount of 3D data and has a very competitive CLIP similarity. In conclusion, all those results suggest that SIR is a promising approach for efficient 3D generation.

|[Figure 421]|[Figure 422]|[Figure 423]|
|---|---|---|

Input Output Input Output Input Output

(a) Results of single-step prediction using Eq. (2) for SDS.

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

Input 6 views from generated 3D Gaussian splatting

(b) Results of applying SDS in pixel space.

- Fig. 11: Failure of applying SDS in pixel space. SDS optimizing in pixel space generates poor results.

[Figure 470]

(a) Iterative reconstruction. (b) Camera views.

[Figure 471]

[Figure 472]

(c) Time schedule. (d) Forward process.

[Figure 473]

- Fig. 12: Detailed analyses. (a) Iterative reconstruction is necessary. (b) More camera views benefit. (c) A linear schedule is sufficient. (d) A hybrid forward process can be effective.

- D. Analysis and ablation study

Analysis on optimization space. We implement SIR-latent on NeRF and compare its time consumption with SIR-pixel, as illustrated in Fig. 12a-b. We can see that SIR-pixel achieves 2-3 times more acceleration than SIR-latent, verifying the benefits of SIR in enabling optimization in pixel space. Contrary to SIR, we argue that SDS optimizing in pixel space is not effective even if it has the data predicting form by reparameterization as presented in Eq. (6). If we attempt to employ SDS for optimization in pixel space, i.e. optimizing

3As TriplaneGaussian has no official mesh export code, we apply the mesh exported code from LGM [43] for it.

3D objects with 2D images like those generated by Eq. (2) in Fig. 11a, the resulting 3D content would be of subpar quality, as the results shown in Fig. 11b.

Ablation on iterative reconstruction. Fig. 12a shows the necessity of iterative reconstruction. The case of K = 0 outputs our 3D initialization, which is reconstructed by a single process from images generated by random noise. When K > 30, the generation quality doesn’t improve significantly. For efficiency, we set K = 20 or 30.

Ablation on number of camera poses. SIR necessitates more than 1 camera pose for reconstruction, as shown in Fig. 12b, and the performance increases with more camera poses. We

[Figure 474]

Fig. 13: Limitation. Visualization of some less satisfactory cases generated by MicroDreamer.

choose 6 on 3DGS with Stable Zero123 [68] model to balance efficiency and quality. See Tab. I for values in other settings.

Ablation on time schedule. We compare our linear schedule, the random schedule employed in SDS [1], and a square schedule with t(2k) = (Kk )2 · (0.9T − 0.2T) + 0.2T. As shown in Fig. 12c, the schedules perform similarly. For simplicity and efficiency, we use the linear schedule by default.

Ablation on forward process. Fig. 12d compares three different forward processes on 3DGS with the Stable Zero123 [68] model. The hybrid strategy performs better than noise-adding. As discussed in Sec. IV-C, we adopt the hybrid process rather than inversion for efficiency.

- E. Limitation

We show some less successful cases generated by our method in Fig. 13. MicroDreamer faces challenges in generating complex geometric structures such as central hollows and may produce meshes with poor back surface textures. These problems may be mitigated as the quality of the generated multi-view images improves.

VII. CONCLUSION

We introduce SIR, an efficient and general algorithm combining the strengths of 3D reconstruction and iterative optimization to reduce total NFEs and enable optimization in pixel space in optimization-based 3D generation. SIR achieves a

- 5 to 20 times speed increase in NeRF generation compared to SDS. Remarkably, MicroDreamer generates high-quality meshes from 3DGS in about 20 seconds, outpacing the fastest optimization-based baseline DreamGaussian in quality and efficiency, and matching the speed of some feed-forward approaches with a competitive generation quality.

There is potential for further improving MicroDreamer’s efficiency via employing consistency models [88], [89] or alternative sampling models that require fewer steps [90], [91]. Additionally, the fidelity and 3D consistency of the objects produced by MicroDreamer are directly limited by the quality of the outputs from multi-view diffusion. Nevertheless, we believe SIR is promising and may inspire future work as the multi-view diffusion evolves.

Furthermore, MicroDreamer can provide artists with the convenience of creating 3D assets. However, as a generative approach, our method may also be used for fabricating data and news. Moreover, the 3D content generated from input images may infringe on the privacy or copyright of others. While automatic detection might mitigate these issues, they warrant careful consideration.

REFERENCES

- [1] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Textto-3d using 2d diffusion,” arXiv preprint arXiv:2209.14988, 2022.
- [2] H. Wang, X. Du, J. Li, R. A. Yeh, and G. Shakhnarovich, “Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation,” in CVPR, 2023, pp. 12619–12629.
- [3] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, “Magic3d: High-resolution text-to-3d content creation,” arXiv preprint arXiv:2211.10440, 2022.
- [4] J. Zhu and P. Zhuang, “Hifa: High-fidelity text-to-3d with advanced diffusion guidance,” arXiv preprint arXiv:2305.18766, 2023.
- [5] R. Chen, Y. Chen, N. Jiao, and K. Jia, “Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation,” arXiv preprint arXiv:2303.13873, 2023.
- [6] R. Liu, R. Wu, B. V. Hoorick, P. Tokmakov, S. Zakharov, and C. Vondrick, “Zero-1-to-3: Zero-shot one image to 3d object,” 2023.
- [7] Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu, “Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [8] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, “Dreamgaussian: Generative gaussian splatting for efficient 3d content creation,” arXiv preprint arXiv:2309.16653, 2023.
- [9] Y. Liang, X. Yang, J. Lin, H. Li, X. Xu, and Y. Chen, “Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching,” arXiv preprint arXiv:2311.11284, 2023.
- [10] J. Sun, B. Zhang, R. Shao, L. Wang, W. Liu, Z. Xie, and Y. Liu, “Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior,” arXiv preprint arXiv:2310.16818, 2023.
- [11] G. Metzer, E. Richardson, O. Patashnik, R. Giryes, and D. Cohen-Or, “Latent-nerf for shape-guided generation of 3d shapes and textures,” arXiv preprint arXiv:2211.07600, 2022.
- [12] T. Yi, J. Fang, G. Wu, L. Xie, X. Zhang, W. Liu, Q. Tian, and X. Wang, “Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors,” arXiv preprint arXiv:2310.08529, 2023.
- [13] F. Liu, D. Wu, Y. Wei, Y. Rao, and Y. Duan, “Sherpa3d: Boosting high-fidelity text-to-3d generation via coarse 3d prior,” arXiv preprint arXiv:2312.06655, 2023.
- [14] L. Qiu, G. Chen, X. Gu, Q. Zuo, M. Xu, Y. Wu, W. Yuan, Z. Dong, L. Bo, and X. Han, “Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d,” arXiv preprint arXiv:2311.16918, 2023.
- [15] R. Wu, B. Mildenhall, P. Henzler, K. Park, R. Gao, D. Watson, P. P. Srinivasan, D. Verbin, J. T. Barron, B. Poole et al., “Reconfusion: 3d reconstruction with diffusion priors,” arXiv preprint arXiv:2312.02981, 2023.
- [16] B. Tang, J. Wang, Z. Wu, and L. Zhang, “Stable score distillation for high-quality 3d generation,” arXiv preprint arXiv:2312.09305, 2023.
- [17] X. Yu, Y.-C. Guo, Y. Li, D. Liang, S.-H. Zhang, and X. Qi, “Text-to3d with classifier score distillation,” arXiv preprint arXiv:2310.19415, 2023.
- [18] O. Katzir, O. Patashnik, D. Cohen-Or, and D. Lischinski, “Noise-free score distillation,” arXiv preprint arXiv:2310.17590, 2023.
- [19] X. Yang, Y. Chen, C. Chen, C. Zhang, Y. Xu, X. Yang, F. Liu, and G. Lin, “Learn to optimize denoising scores for 3d generation: A unified and improved diffusion prior on nerf and 3d gaussian splatting,” arXiv e-prints, pp. arXiv–2312, 2023.
- [20] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International Conference on Machine Learning. PMLR, 2015, pp. 2256–2265.
- [21] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems, vol. 33, pp. 6840– 6851, 2020.
- [22] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” 2021.
- [23] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10684–10695.
- [24] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, B. K. Ayan, S. S. Mahdavi, R. G. Lopes et al., “Photorealistic text-to-image diffusion models with deep language understanding,” arXiv preprint arXiv:2205.11487, 2022.

- [25] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, 2022.
- [26] Z. Cao, F. Hong, T. Wu, L. Pan, and Z. Liu, “Large-vocabulary 3d diffusion model with transformer,” arXiv preprint arXiv:2309.07920, 2023.
- [27] H. Chen, J. Gu, A. Chen, W. Tian, Z. Tu, L. Liu, and H. Su, “Single-stage diffusion nerf: A unified approach to 3d generation and reconstruction,” arXiv preprint arXiv:2304.06714, 2023.
- [28] Z. Chen, F. Hong, H. Mei, G. Wang, L. Yang, and Z. Liu, “Primdiffusion: Volumetric primitives diffusion for 3d human generation,” arXiv preprint arXiv:2312.04559, 2023.
- [29] H. Jun and A. Nichol, “Shap-e: Generating conditional 3d implicit functions,” arXiv preprint arXiv:2305.02463, 2023.
- [30] A. Nichol, H. Jun, P. Dhariwal, P. Mishkin, and M. Chen, “Point-e: A system for generating 3d point clouds from complex prompts,” arXiv preprint arXiv:2212.08751, 2022.
- [31] Z. Liu, Y. Feng, M. J. Black, D. Nowrouzezahrai, L. Paull, and W. Liu, “Meshdiffusion: Score-based generative 3d mesh modeling,” arXiv preprint arXiv:2303.08133, 2023.
- [32] Y. Hong, K. Zhang, J. Gu, S. Bi, Y. Zhou, D. Liu, F. Liu, K. Sunkavalli, T. Bui, and H. Tan, “Lrm: Large reconstruction model for single image to 3d,” arXiv preprint arXiv:2311.04400, 2023.
- [33] N. M¨uller, Y. Siddiqui, L. Porzi, S. R. Bulo, P. Kontschieder, and M. Nießner, “Diffrf: Rendering-guided 3d radiance field diffusion,” in CVPR, 2023, pp. 4328–4338.
- [34] T. Wang, B. Zhang, T. Zhang, S. Gu, J. Bao, T. Baltrusaitis, J. Shen, D. Chen, F. Wen, Q. Chen et al., “Rodin: A generative model for sculpting 3d digital avatars using diffusion,” in CVPR, 2023, pp. 4563– 4573.
- [35] L. Yariv, O. Puny, N. Neverova, O. Gafni, and Y. Lipman, “Mosaic-sdf for 3d generative models,” arXiv preprint arXiv:2312.09222, 2023.
- [36] Z. Zhao, W. Liu, X. Chen, X. Zeng, R. Wang, P. Cheng, B. Fu, T. Chen, G. Yu, and S. Gao, “Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation,” arXiv preprint arXiv:2306.17115, 2023.
- [37] Z.-X. Zou, Z. Yu, Y.-C. Guo, Y. Li, D. Liang, Y.-P. Cao, and S.-H. Zhang, “Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers,” arXiv preprint arXiv:2312.09147, 2023.
- [38] J. Li, H. Tan, K. Zhang, Z. Xu, F. Luan, Y. Xu, Y. Hong, K. Sunkavalli, G. Shakhnarovich, and S. Bi, “Instant3d: Fast text-to-3d with sparseview generation and large reconstruction model,” arXiv preprint arXiv:2311.06214, 2023.
- [39] M. Liu, C. Xu, H. Jin, L. Chen, Z. Xu, H. Su et al., “One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization,” arXiv preprint arXiv:2306.16928, 2023.
- [40] M. Liu, R. Shi, L. Chen, Z. Zhang, C. Xu, X. Wei, H. Chen, C. Zeng, J. Gu, and H. Su, “One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion,” arXiv preprint arXiv:2311.07885, 2023.
- [41] P. Wang, H. Tan, S. Bi, Y. Xu, F. Luan, K. Sunkavalli, W. Wang, Z. Xu, and K. Zhang, “Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction,” arXiv preprint arXiv:2311.12024, 2023.
- [42] Z. Wang, Y. Wang, Y. Chen, C. Xiang, S. Chen, D. Yu, C. Li, H. Su, and J. Zhu, “Crm: Single image to 3d textured mesh with convolutional reconstruction model,” arXiv preprint arXiv:2403.05034, 2024.
- [43] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu, “Lgm: Large multi-view gaussian model for high-resolution 3d content creation,” arXiv preprint arXiv:2402.05054, 2024.
- [44] F. Hong, J. Tang, Z. Cao, M. Shi, T. Wu, Z. Chen, T. Wang, L. Pan, D. Lin, and Z. Liu, “3dtopia: Large text-to-3d generation model with hybrid diffusion priors,” arXiv preprint arXiv:2403.02234, 2024.
- [45] J. Xu, W. Cheng, Y. Gao, X. Wang, S. Gao, and Y. Shan, “Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models,” arXiv preprint arXiv:2404.07191, 2024.
- [46] Y. Wang, W. Lira, W. Wang, A. Mahdavi-Amiri, and H. Zhang, “Slice3d: Multi-slice, occlusion-revealing, single view 3d reconstruction,” arXiv preprint arXiv:2312.02221, 2023.
- [47] Y. Lan, F. Hong, S. Yang, S. Zhou, X. Meng, B. Dai, X. Pan, and C. C. Loy, “Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation,” arXiv preprint arXiv:2403.12019, 2024.
- [48] D. Tochilkin, D. Pankratz, Z. Liu, Z. Huang, , A. Letts, Y. Li, D. Liang, C. Laforte, V. Jampani, and Y.-P. Cao, “Triposr: Fast 3d object reconstruction from a single image,” arXiv preprint arXiv:2403.02151, 2024.

- [49] Y. Xu, Z. Shi, W. Yifan, H. Chen, C. Yang, S. Peng, Y. Shen, and G. Wetzstein, “Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation,” arXiv preprint arXiv:2403.14621, 2024.
- [50] X. Wei, K. Zhang, S. Bi, H. Tan, F. Luan, V. Deschaintre, K. Sunkavalli, H. Su, and Z. Xu, “Meshlrm: Large reconstruction model for highquality mesh,” arXiv preprint arXiv:2404.12385, 2024.
- [51] K. Wu, F. Liu, Z. Cai, R. Yan, H. Wang, Y. Hu, Y. Duan, and K. Ma, “Unique3d: High-quality and efficient 3d mesh generation from a single image,” arXiv preprint arXiv:2405.20343, 2024.
- [52] K. Zhang, S. Bi, H. Tan, Y. Xiangli, N. Zhao, K. Sunkavalli, and Z. Xu, “Gs-lrm: Large reconstruction model for 3d gaussian splatting,” arXiv preprint arXiv:2404.19702, 2024.
- [53] M. Boss, Z. Huang, A. Vasishta, and V. Jampani, “Sf3d: Stable fast 3d mesh reconstruction with uv-unwrapping and illumination disentanglement,” arXiv preprint arXiv:2408.00653, 2024.
- [54] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” stat, vol. 1050, p. 1, 2014.
- [55] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017.
- [56] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, 2023.
- [57] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.
- [58] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics primitives with a multiresolution hash encoding,” ACM Transactions on Graphics (ToG), vol. 41, no. 4, pp. 1–15, 2022.
- [59] Y. Liu, C. Lin, Z. Zeng, X. Long, L. Liu, T. Komura, and W. Wang, “Syncdreamer: Generating multiview-consistent images from a singleview image,” arXiv preprint arXiv:2309.03453, 2023.
- [60] X. Long, Y.-C. Guo, C. Lin, Y. Liu, Z. Dou, L. Liu, Y. Ma, S.-H. Zhang, M. Habermann, C. Theobalt et al., “Wonder3d: Single image to 3d using cross-domain diffusion,” arXiv preprint arXiv:2310.15008, 2023.
- [61] R. Shi, H. Chen, Z. Zhang, M. Liu, C. Xu, X. Wei, L. Chen, C. Zeng, and H. Su, “Zero123++: a single image to consistent multi-view diffusion base model,” 2023.
- [62] Y. Lu, J. Zhang, S. Li, T. Fang, D. McKinnon, Y. Tsin, L. Quan, X. Cao, and Y. Yao, “Direct2. 5: Diverse text-to-3d generation via multi-view 2.5 d diffusion,” arXiv preprint arXiv:2311.15980, 2023.
- [63] P. Li, Y. Liu, X. Long, F. Zhang, C. Lin, M. Li, X. Qi, S. Zhang, W. Luo, P. Tan et al., “Era3d: High-resolution multiview diffusion using efficient row-wise attention,” arXiv preprint arXiv:2405.11616, 2024.
- [64] L. Melas-Kyriazi, I. Laina, C. Rupprecht, N. Neverova, A. Vedaldi, O. Gafni, and F. Kokkinos, “Im-3d: Iterative multiview diffusion and reconstruction for high-quality 3d generation,” arXiv preprint arXiv:2402.08682, 2024.
- [65] Z. Chen, Y. Wang, F. Wang, Z. Wang, and H. Liu, “V3d: Video diffusion models are effective 3d generators,” arXiv preprint arXiv:2403.06738, 2024.
- [66] V. Voleti, C.-H. Yao, M. Boss, A. Letts, D. Pankratz, D. Tochilkin, C. Laforte, R. Rombach, and V. Jampani, “Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion,” arXiv preprint arXiv:2403.12008, 2024.
- [67] Y. Shi, P. Wang, J. Ye, M. Long, K. Li, and X. Yang, “Mvdream: Multiview diffusion for 3d generation,” arXiv preprint arXiv:2308.16512, 2023.
- [68] stability.ai, “Stable zero123,” 2023, https://stability.ai/news/ stable-zero123-3d-generation.
- [69] P. Wang and Y. Shi, “Imagedream: Image-prompt multi-view diffusion for 3d generation,” arXiv preprint arXiv:2312.02201, 2023.
- [70] Z. He and T. Wang, “Openlrm: Open-source large reconstruction models,” https://github.com/3DTopia/OpenLRM, 2023.
- [71] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” 2021.
- [72] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, “Objaverse: A universe of annotated 3d objects,” in CVPR, 2023, pp. 13142–13153.
- [73] M. Deitke, R. Liu, M. Wallingford, H. Ngo, O. Michel, A. Kusupati, A. Fan, C. Laforte, V. Voleti, S. Y. Gadre et al., “Objaverse-xl: A universe of 10m+ 3d objects,” arXiv preprint arXiv:2307.05663, 2023.
- [74] L. Liu, Y. Ren, Z. Lin, and Z. Zhao, “Pseudo numerical methods for diffusion models on manifolds,” arXiv preprint arXiv:2202.09778, 2022.

- [75] F. Bao, C. Li, J. Zhu, and B. Zhang, “Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models,” in International Conference on Learning Representations, 2022.
- [76] F. Bao, C. Li, J. Sun, J. Zhu, and B. Zhang, “Estimating the optimal covariance with imperfect mean in diffusion probabilistic models,” arXiv preprint arXiv:2206.07309, 2022.
- [77] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, “Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps,” in Advances in Neural Information Processing Systems, 2022.
- [78] ——, “Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models,” arXiv preprint arXiv:2211.01095, 2022.
- [79] W. Zhao, L. Bai, Y. Rao, J. Zhou, and J. Lu, “Unipc: A unified predictor-corrector framework for fast sampling of diffusion models,” arXiv preprint arXiv:2302.04867, 2023.
- [80] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.
- [81] C. Lu, K. Zheng, F. Bao, J. Chen, C. Li, and J. Zhu, “Maximum likelihood training for score-based diffusion odes by high order denoising score matching,” in International Conference on Machine Learning. PMLR, 2022, pp. 14429–14460.
- [82] S. Nie, H. A. Guo, C. Lu, Y. Zhou, C. Zheng, and C. Li, “The blessing of randomness: Sde beats ode in general diffusion-based image editing,” arXiv preprint arXiv:2311.01410, 2023.
- [83] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.
- [84] Y.-C. Guo, Y.-T. Liu, C. Wang, Z.-X. Zou, G. Luo, C.-H. Chen, Y.-P. Cao, and S.-H. Zhang, “threestudio: A unified framework for 3d content generation,” https://github.com/threestudio-project/threestudio, 2023.
- [85] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.
- [86] W. E. Lorensen and H. E. Cline, “Marching cubes: A high resolution 3d surface construction algorithm,” in Seminal graphics: pioneering efforts that shaped the field, 1998, pp. 347–353.
- [87] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman, G. Ilharco, C. Gordon, C. Schuhmann, L. Schmidt, and J. Jitsev, “Reproducible scaling laws for contrastive language-image learning,” arXiv preprint arXiv:2212.07143, 2022.
- [88] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in International Conference on Machine Learning. PMLR, 2023, pp. 32211–32252.
- [89] S. Luo, Y. Tan, L. Huang, J. Li, and H. Zhao, “Latent consistency models: Synthesizing high-resolution images with few-step inference,”

- arXiv preprint arXiv:2310.04378, 2023.

[90] T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, and T. Park, “One-step diffusion with distribution matching distillation,”

- arXiv preprint arXiv:2311.18828, 2023.

- [91] A. Sauer, D. Lorenz, A. Blattmann, and R. Rombach, “Adversarial diffusion distillation,” arXiv preprint arXiv:2311.17042, 2023.

[Figure 475]

Luxi Chen Luxi Chen received a BS degree from the Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China. He is pursuing a PhD degree in the Gaoling School of Artificial Intelligence, Renmin University of China. His research interests include deep generative models and 3D content generation.

[Figure 476]

Zihan Zhou Zihan Zhou received his BS degree from the School of Computer Science and Technology, Xidian University, Shaanxi, China. He is currently pursuing an MS degree in the Gaoling School of Artificial Intelligence, at Renmin University of China. His research interests include 3D mesh generation and deep generative models.

[Figure 477]

Tingting Gao Tingting Gao is the head of the Visual Understanding and Application Center at Kuaishou. Her research interests encompass computer vision, multimodality, and the industrial application of large models. She has previously worked as a Senior Algorithm Engineer at Baidu, where she accumulated a wealth of experience in the fields of search and recommendation

Hang Su Hang Su (Member, IEEE) is an associate professor with the Department of Computer Science and Technology, at Tsinghua University. His research interests lie in adversarial machine learning and robust computer vision, based on which he has published more than 50 papers including CVPR, ECCV, IEEE Transactions on Medical Imaging, etc. He has served as area chair in NeurIPS and the workshop co-chair in AAAI22. He received the “Young Investigator Award” from MICCAI2012, the “Best Paper Award” in AVSS2012, and the “Platinum Best

[Figure 478]

Paper Award” in ICME2018.

Jun Zhu Jun Zhu (Fellow, IEEE) received the BS and PhD degrees from the Department of Computer Science and Technology, Tsinghua University, where he is currently a Bosch AI professor. He was a postdoctoral fellow and adjunct faculty with the Machine Learning Department, at Carnegie Mellon University. His research interest is primarily in developing machine learning methods to understand scientific and engineering data arising from various fields. He regularly serves as senior area chairs and area chairs at prestigious conferences, including

[Figure 479]

ICML, NeurIPS, ICLR, IJCAI, and AAAI. He was selected as “AI’s 10 to Watch” by IEEE Intelligent Systems. He is a fellow of IEEE, a fellow of AAAI, and an associate editor-in-chief of IEEE Transactions on Pattern Analysis and Machine Intelligence.

[Figure 480]

Zhengyi Wang Zhengyi Wang received his BS degree from the Department of Computer Science and Technology, Tsinghua University, Beijing, China. He is currently working toward a PhD degree in the Department of Computer Science and Technology, at Tsinghua University, Beijing, China. His research interests include the theory and application of generative models.

[Figure 481]

Chongxuan Li Chongxuan Li (Member, IEEE) is an associate professor at Renmin University of China, Beijing, China. He obtained both his Bachelor’s and Ph.D. degrees from Tsinghua University. His research interests include machine learning and deep generative models. His works were recognized as the Outstanding Paper Award at ICLR 2022. Moreover, he served as an associate editor for IEEE Transactions on Pattern Analysis and Machine Intelligence and area chair for NeurIPS, ICLR, and ACM MM.

