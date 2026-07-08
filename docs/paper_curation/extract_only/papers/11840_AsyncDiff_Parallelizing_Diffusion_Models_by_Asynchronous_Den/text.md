# arXiv:2406.06911v3[cs.CV]26Sep2024

## AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising

##### Zigeng Chen, Xinyin Ma, Gongfan Fang, Zhenxiong Tan, Xinchao Wang∗

National University of Singapore zigeng99@u.nus.edu, xinchao@nus.edu.sg

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### 2.8x️

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

###### Speedup

Figure 1: We introduce a new distributed acceleration paradigm that attains a 2.8x speed-up on Stable Diffusion XL while maintaining pixel-level consistency, using four NVIDIA A5000 GPUs.

### Abstract

Diffusion models have garnered significant interest from the community for their great generative ability across various applications. However, their typical multistep sequential-denoising nature gives rise to high cumulative latency, thereby precluding the possibilities of parallel computation. To address this, we introduce AsyncDiff, a universal and plug-and-play acceleration scheme that enables model parallelism across multiple devices. Our approach divides the cumbersome noise prediction model into multiple components, assigning each to a different device. To break the dependency chain between these components, it transforms the conventional sequential denoising into an asynchronous process by exploiting the high similarity between hidden states in consecutive diffusion steps. Consequently, each component is facilitated to compute in parallel on separate devices. The proposed strategy significantly reduces inference latency while minimally impacting the generative quality. Specifically, for the Stable Diffusion v2.1, AsyncDiff achieves a 2.7x speedup with negligible degradation and a 4.0x speedup with only a slight reduction of 0.38 in CLIP Score, on four NVIDIA A5000 GPUs. Our experiments also demonstrate AsyncDiff can be readily applied to video diffusion models with encouraging performances. Code is available at https://github.com/czg1225/AsyncDiff

### 1 Introduction

Diffusion models [10] stand out in generative modeling and have significantly advanced various fields including text-to-image [38, 36, 40, 41, 64, 70] and text-to-video generation [56, 6, 54, 17, 2], image

∗Correspoding Author

Preprint. Under review.

Traditional: Denosing Model running in Sequential

[Figure 11]

[Figure 12]

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

2 3 4

1

GPU 0

Ours: Denoising Model running in Parallel

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

1 2 3 4

GPU 0 GPU 1 GPU 2 GPU 3

Communication within device Communication across devices

- Figure 2: By preparing each component’s input beforehand, we enable parallel computation of the denoising model, which substantially reduces latency while minimally affecting quality.

translation [44, 51, 19], audio generation[18, 11, 39], low-level vision tasks [42, 53, 35, 22, 5, 62], image editing [15, 58, 46, 69], and 3D model generation [37, 14, 32], among others. However, their widespread application is hindered by the high latency inherent in their multi-step sequential denoising process. This issue becomes more pronounced as the complexity and size of the models increase to enhance generative quality.

In response to these challenges, significant research efforts are directed toward enhancing the efficiency of diffusion models. Notably, training-free acceleration methods have garnered increasing popularity due to their low cost and convenience. Numerous studies [30, 55, 68, 59, 48, 21, 29] improve inference speed by skipping redundant calculations in the denoising process. As computational resources grow rapidly, distributing computations across multiple devices has become a more promising approach. Recent advances [47, 20] demonstrate that using distributed computing to parallelize inference effectively increases the acceleration ratio for diffusion models while maintaining acceptable generative quality. Though these methods succeed in parallelizing the diffusion models, they require iterative refining [47] or displaced patch parallelism [20], resulting in a larger number of model evaluations or low GPU utilization correspondingly.

Thus, we wish to propose a new parallel paradigm for diffusion, akin to the model parallelism in distributed computing [12, 33, 24, 13, 34, 57], which divides the denoising model into several components to be distributed on different GPUs. The primary challenge lies in the inherent sequential denoising process of diffusion models. Each step in this process depends on the completion of its predecessor, forming a dependency chain that impedes parallelization and significantly increases inference latency. Our approach seeks to disrupt this chain, allowing for the parallel execution of the denoising model while closely approximating the results of the sequential process.

In this paper, we introduce AsyncDiff, a universal, distributed acceleration paradigm that innovatively explores model parallelism in diffusion models. As shown in Fig 2, our method sequentially partitions the heavyweight denoising model ϵθ into multiple components {ϵnθ}Nn=1 based on computational load, assigning each to a separate device. Our core idea lies in decoupling the dependencies between these cascaded components by leveraging the high similarity in hidden states across consecutive diffusion steps. After the initial warm-up steps, each component takes the output from the previous component’s prior step as the approximation of its original input. This transforms the traditional sequential denoising into an asynchronous process, allowing components to predict noise for different time steps in parallel. Additionally, we incorporate stride denoising to skip redundant calculations and reduce the frequency of communication between devices, further enhancing efficiency.

Through extensive testing across multiple base models, our method effectively distributes the computational burden across various devices, substantially boosting inference speed while maintaining quality. Specifically, with the text-to-image model Stable Diffusion v2.1 [38], our method achieves a 1.8x speedup with only a marginal 0.01 drop in CLIP Score [8], and a 4.0x speedup with a slight 0.38 reduction in CLIP Score on two and four NVIDIA A5000 GPUs, respectively. For video diffusion models, AnimateDiff [6] and Stable Video Diffusion [2], our approach significantly reduces latency by tens of seconds, effectively preserving video quality.

In summary, we present a novel distributed acceleration method for diffusion models that significantly reduces inference latency with minimal impact on generation quality. This is achieved by replacing the sequential denoising process with an asynchronous process, allowing each component of the denoising model to run independently across different devices. Extensive experiments on both image and video diffusion models strongly demonstrate the effectiveness and versatility of our method.

### 2 Related Works

Diffusion Models. Diffusion models have attracted significant attention due to their powerful generative capabilities across various tasks. Sohl-Dickstein et al. [49] first proposed diffusion probabilistic models. Ho et al. [10] with the introduction of Denoising Diffusion Probabilistic Models (DDPM), enhancing training efficiency and generation quality. Rombach et al. [38] advanced these models by incorporating latent spaces, enabling high-resolution image generation. Despite these advancements, the high latency of the iterative denoising process remains a limitation.

Inference Acceleration. Training-based acceleration methods focus on reducing sampling steps [43, 63, 28, 45, 61] or optimizing model architectures [23, 71, 4, 65, 60]. However, these methods incur high training costs and complexity. Training-free methods are gaining popularity due to their ease of use. Some approaches develop fast solvers for SDE or ODE to improve sampling efficiency [27, 1, 26, 66, 72]. Other works [30, 55, 68, 59, 48, 21, 29] observed special characteristics of diffusion models and skipped the redundant computation within the denoising process.

Parallelism. The parallelism strategy presents a promising yet underexplored approach to accelerating diffusion models. ParaDiGMS [47] implements Picard iterations for parallel sampling, yet its practical speed-up ratio is modest, and it struggles to maintain consistency with original outputs. Faster Diffusion [21] introduces encoder propagation but significantly compromises quality, and its parallelization remains theoretical. Distrifusion [20] adopts patch parallelism, dividing highresolution images into sub-patches to facilitate parallel inference on each patch by reusing stale activation maps from each layer. However, this approach lacks flexibility across different data types or tasks, often encountering low resource utilization. Furthermore, its reliance on reusing per-layer activation maps greatly increases GPU memory demands thus introducing additional challenges for realistic applications. In contrast, our method uniquely implements model parallelism through asynchronous denoising, achieving substantial acceleration while maintaining a stable resource usage ratio and minimal impact on quality.

### 3 Methods

##### 3.1 Preliminary

Diffusion models [10] are a dominant class of generative models that transform Gaussian noise into complex data distributions via a Markov process. The forward process is defined by:

q(xt|xt−1) = N(xt; 1 − βtxt−1,βtI), (1)

where {βt} progressively increases noise until the data becomes indistinguishable from noise. The reverse process, essential for data reconstruction, involves iterative denoising:

pθ(xt−1|xt) = N(xt−1;µθ(xt,t),σt2I), (2)

where µθ(xt,t) is the predicted mean and σt2 is the variance. For DDIMs [50], the reverse update is deterministic:

1 − αt αt−1

αt−1 αt

ϵθ(xt,t), (3)

xt + 1 − αt−1 1 −

xt−1 =

where αt is the cumulative product of (1 − βt). These processes are computationally intensive, influencing the quality of generated samples and necessitating efficient inference methods for practical applications.

##### 3.2 Asynchronous Diffusion Model

Traditional diffusion models employ a sequential and synchronous denoising process. At each time step t, the noise-prediction model ϵθ estimates the noise ϵt based on the noisy image xt and the time

Warm-up Parallel Parallel Parallel Parallel

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

###### GPU 3 (  4)

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

 −4  −5

 −3

 −1

 −2

T

T-2

T-4

T-1

T-3

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

GPU 2 (  3)

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

 −1

T

T-2

T-4

T-1

T-3

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

GPU 1 (  2)

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

 −1

T

T-2

T-4

###### T-1

T-3

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

GPU 0 (  1)

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

 −1

T

T-2

T-4

T-1

T-3

Communication within device Communication across devices T: Time embedding

- Figure 3: Overview of the asynchronous denoising process. The denoising model ϵθ is divided

into four components {ϵnθ}4n=1 for clarity. Following the warm-up stage, each component’s input is prepared in advance, breaking the dependency chain and facilitating parallel processing.

embedding t. The image for the next step, xt−1, is then generated using a sampler function S(xt,ϵt,t). This process is iterative, where the generation of ϵt at each step is dependent on the completion of the previous denoising step, making the process slow, particularly when ϵθ is computationally intensive.

To address the limitations of high latency in diffusion models, leveraging multiple GPUs for distributed inference is a promising solution. Existing studies primarily focus on patch parallelism [20], where the input image is divided into patches, each processed on a different GPU. While this strategy efficiently distributes computational loads, it still retains the bottleneck of sequential denoising, as each patch must undergo the complete denoising process iteratively. In contrast, our asynchronous diffusion model innovatively introduces a model parallelism strategy. By approximating the sequential denoising as an asynchronous process, this approach enables parallel inference of the noise prediction model, effectively reducing latency and breaking the constraints of sequential execution.

Asynchronous Denoising. Figure 3 illustrates our approach to the asynchronous denoising. For a denoising process consisting of T steps, the initial w steps are designated as a warm-up phase, where w is significantly smaller than T. During this phase, the denoising model ϵθ operates using standard sequential inference. After warm-up steps, rather than splitting the input image, we partition the denoising model ϵθ into N sequential components, expressed as ϵθ = {ϵ1θ,ϵ2θ,...,ϵNθ }. Each component is divided to handle a comparable computational load and assigned to a distinct device. This equitable division aims to equalize the time cost of each component to approximately l(ϵθ)/N, thus minimizing the overall maximum latency. In this setup, original noise prediction for xt can be represented as a cascading operation through these sub-models, defined mathematically as:

ϵt = ϵθ(xt,t) = ϵNθ (ϵNθ −1(...ϵ2θ(ϵ1θ(xt,t),t)...,t),t). (4) Although each device can independently compute its assigned component, the dependency chain persists because the input for each component ϵθ,n is derived from the output from its preceding component ϵθ,n−1. Therefore, despite the distribution of model components across multiple devices, full parallelization is constrained by these sequential dependencies.

Our principal innovation is to break the dependency between cascaded components by utilizing hidden features from previous steps. Observations indicate that the hidden states of each block in the denoising model always exhibit substantial similarity across adjacent time steps. Leveraging this, each component at time step t can take the output from the preceding component at time step t − 1 as the approximation of its original input. Specifically, the n-th component ϵnθ(,t) receives the output of

###### GPU 3(  3)

###### GPU 0(  1) GPU 1(  2) GPU 2(  3)

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

###### Parallel

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

#### t-1 t-1

t t-1

[Figure 223]

[Figure 224]

[Figure 225]

 −1  −2

Communication across devices

Communication within device

t: Time embedding

- Figure 4: Illustration of stride denoising. The model ϵθ is divided into three components {ϵnθ}3n=1,

with a stride S of 2 for clarity. Components ϵ1θ and ϵ2θ are skipped at time step t. A single parallel batch results in the completion of denoising for two steps, producing xt−1 and xt−2.

ϵnθ−1(·,t − 1). This alteration allows the noise prediction for xt to be represented as follows: ϵt = ϵNθ (ϵNθ −1(...ϵ2θ(ϵ1θ(xt+N−1,t + N − 1),t + N − 2)...,t + 1),t). (5)

In this new framework, noise prediction ϵt is derived from components executed across N previous time steps. This transforms the denoising process from sequential to asynchronous, as the prediction

of noise ϵt already begins before denoising at step t + 1 is completed. At each time step, the N components are running as parts of the noise prediction model for the next N steps. Specifically, the n-th component ϵnθ, computed in parallel at time t, contributes to the noise prediction for the future time step t − N + n. Figure 3 depicts this asynchronous process using a U-net model with N set to 4. The strong resemblance of hidden states between consecutive diffusion steps enables the asynchronous process to closely mimic the denoising results of the original sequential process.

Model Parallelism. By transitioning to an asynchronous denoising strategy, the dependencies among components within the same time step are eliminated. This adjustment allows each component’s input for time step t to be prepared in advance, enabling the N split components to be processed concurrently across multiple devices. Once computed, the outputs from each component must be stored and then broadcasted to other devices to facilitate parallel processing for subsequent time steps. In contrast, in the traditional sequential denoising process, the time cost for each step accumulates as follows:

Cseq(t) = C(ϵ1θ) + C(ϵ2θ) + ... + C(ϵNθ ). (6) By adopting asynchronous denoising to enable parallel computation of each component, the cost for each time step is now given by:

Casy(t) = max(C(ϵ1θ),C(ϵ2θ),...,C(ϵNθ )) + C(comm.), (7)

where max() represents taking the maximum value, and C(comm.) indicates the communication cost across multiple GPUs. As the model components are equally divided by computational load, their time costs are similar, allowing us to approximate the overall cost of each time step as:

Cseq(t) N

+ C(comm.). (8)

Casy(t) ≈

Since the communication overhead C(comm.) is generally much lower than the model’s execution time, it leads to significant overall cost reductions. Moreover, increasing N further reduces time costs but complicates the accurate approximation of the original denoising process.

Stride Denoising. While asynchronous denoising reduces latency by parallelizing the denoising model, it completes only one denoising step at a time. To enhance efficiency, we introduce stride denoising, which completes multiple denoising steps simultaneously through a single parallel computation. The diagram is illustrated in Figure 4, where we set the stride to 2 for clarity. Unlike the continuous broadcasting of hidden states at each time step, stride denoising broadcasts them every two steps. As depicted, at time step t, we conduct denoising alone, and at time step t−1, we compute and broadcast the hidden states for the next parallel computation round. Consequently, the hidden states from time step t are not required, allowing us to skip the calculations for ϵ1θ and ϵ2θ at this step. In this stride, only ϵ3θ(·,t), ϵ1θ(·,t − 1), ϵ2θ(·,t − 1), and ϵ3θ(·,t − 1) need computing, all receiving the

(a) Qualitative Results on SDXL with different configurations

Original 2OursDevices1.7x(N=2SpeedupS=1) 3OursDevices2.4x(N=3SpeedupS=1) 4OursDevices2.7x(N=4SpeedupS=1) 3OursDevices2.8x(N=2SpeedupS=2)

###### Ours 3.8x Speedup 4 Devices (N=3 S=2)

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

[Figure 237]

(b) Qualitative Results on SDXL with different warm-up steps (N=3 S=2)

Original 4 DevicesOurs 2.5xWarm-up=11Speedup 4 DevicesOurs 2.8xWarm-up=9Speedup 4 DevicesOurs 3.0xWarm-up=7Speedup 4 DevicesOurs 3.4xWarm-up=5Speedup 4 DevicesOurs 3.8xWarm-up=3Speedup

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Ours 2.7x Speedup

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

- Figure 5: Qualitative Results. (a) Our method significantly accelerates the denoising process with minimal impact on generative quality. (b) Increasing warm-up steps achieves pixel-level consistency with the original output while maintaining a high speed-up ratio.

previously broadcast hidden states, enabling their parallel processing. Both ϵ3θ(·,t) and ϵ3θ(·,t − 1) share the same feature from ϵ2θ(·,t + 1), so the stride should be kept small to maintain quality. Stride denoising effectively reduces both computational load and communication demands by decreasing the parallel computing rounds needed to complete the process. Compared to the significant improvements it brings in efficiency, the quality sacrifice is minimal and can be entirely compensated for by slightly increasing the warm-up steps. We also illustrate the full schematic of it in Appendix Figure 7.

Multi-Device Communication. Parallel inference of the model necessitates efficient communication between devices, as each component ϵnθ must access the cached hidden state from the preceding component ϵnθ−1, which resides on a different device. Post each parallel computation batch, each device stores the current hidden state needed for the next parallel batch. These states, encompassing all component outputs, are then broadcast to all participating devices before the next parallel computation batch. Although each component ϵnθ primarily uses the cached output of ϵnθ−1 for its input, it may require residual features [7] from other components. Therefore, it’s crucial to broadcast the stored states from every component across all devices before each round of parallel computation.

### 4 Experiments

##### 4.1 Implementation Details

Base models. We validated the broad applicability of AsyncDiff through extensive testing on several diffusion models. For text-to-image tasks, we experimented with three versions of Stable Diffusion: SD 1.5, SD 2.1 [38], and Stable Diffusion XL (SDXL) [36]. Additionally, we explored the effectiveness of AsyncDiff on video diffusion models using Stable Video Diffusion (SVD) [2] and AnimateDiff [6]. All models were evaluated using 50 DDIM steps. We facilitated communication across multiple GPUs using the broadcast operation from torch.distributed, powered by the NVIDIA Collective Communication Library (NCCL) backend.

- Table 1: Quantitative evaluations of AsyncDiff on three text-to-image diffusion models, showcasing various configurations. ’N’ indicates the number of components into which the model is divided, and ’S’ represents the denoising stride. MACs quantifies the computational load per device for generating a single image throughout the denoising process.

Base Model Configuration Devices MACs↓ latency↓ Speed up↑ CLIP Score↑ FID↓ LPIPS↓

SD 2.1 (Text-to-Image)

Original Model 1 76T 5.51s 1.0x 31.60 27.89 –

- + Ours (N=2 S=1) 2 38T 3.03s 1.8x 31.59 27.79 0.2121

- + Ours (N=3 S=1) 3 25T 2.41s 2.3x 31.56 28.00 0.2755

- + Ours (N=4 S=1) 4 19T 2.10s 2.6x 31.40 28.28 0.3132

- + Ours (N=2 S=2) 3 19T 1.82s 3.0x 31.43 28.55 0.3458

- + Ours (N=3 S=2) 4 13T 1.35s 4.0x 31.22 29.41 0.3778

SD 1.5 (Text-to-Image)

Original Model 1 34T 2.70s 1.0x 30.63 29.96 –

- + Ours (N=2 S=1) 2 17T 1.52s 1.8x 30.62 29.94 0.1988

- + Ours (N=3 S=1) 3 11T 1.23s 2.2x 30.58 29.87 0.2645

- + Ours (N=4 S=1) 4 9T 1.01 2.6x 30.52 30.10 0.3073

- + Ours (N=2 S=2) 3 9T 0.94s 2.9x 30.46 30.98 0.3232

- + Ours (N=3 S=2) 4 6T 0.72s 3.7x 30.17 30.89 0.3811

SDXL (Text-to-Image)

Original Model 1 299T 13.81s 1.0x 32.33 27.43 –

- + Ours (N=2 S=1) 2 150T 8.00s 1.7x 32.21 27.79 0.2509

- + Ours (N=3 S=1) 3 100T 5.84s 2.4x 32.05 28.03 0.2940

- + Ours (N=4 S=1) 4 75T 5.12s 2.7x 31.90 29.12 0.3157

- + Ours (N=2 S=2) 3 75T 4.91s 2.8x 31.70 28.99 0.3209

- + Ours (N=3 S=2) 4 49T 3.65s 3.8x 31.40 30.27 0.3556

- Table 2: Quantitative evaluations of the effect of increasing warm-up steps. More warm-up steps can achieve pixel-level consistency with the original output while slightly reducing processing speed.

SD 2.1 SD 1.5 SDXL Speedup↑ CLIP↑ LPIPS↓ Speedup↑ CLIP↑ LPIPS↓ Speedup↑ CLIP↑ LPIPS↓

Configuration

Original Model 1.0x 31.60 – 1.0x 30.63 – 1.0x 32.33 – Warm-up = 3 3.5x 31.26 0.3289 3.3x 30.16 0.3676 3.8x 31.40 0.3556 Warm-up = 5 3.1x 31.27 0.2769 3.0x 30.14 0.3304 3.4x 31.60 0.2993 Warm-up = 7 2.9x 31.32 0.2309 2.7x 30.10 0.2839 3.0x 31.77 0.2521 Warm-up = 9 2.7x 31.40 0.1940 2.5x 30.17 0.2354 2.8x 31.92 0.2095 Warm-up = 11 2.4x 31.45 0.1628 2.4x 30.22 0.1927 2.5x 32.01 0.1740

Dataset and Evaluation Metrics. We assess the zero-shot generation capability using the MS-COCO 2017 [25] validation set, which comprises 5,000 images and captions. For image generation, quality is measured by the CLIP Score (on ViT-g/14) [8] and Fréchet Inception Distance (FID) [9], with LPIPS [67] used to check consistency with original outputs. In video generation, quality is evaluated by averaging the CLIP Score across all frames of a video. We also report MACs per device and latency to gauge efficiency comprehensively. All latency measurements were conducted on NVIDIA A5000 GPUs equipped with NVLINK Bridge.

##### 4.2 Experimental Results on Image Diffusion Models

Improvements on Base Models. Table 1 displays our acceleration outcomes for three fundamental image diffusion models under various configurations. In this context, ’N’ represents the number of segments into which the denoising model is divided, and ’S’ denotes the stride of denoising for each parallel computation batch. Our approach, AsyncDiff, not only significantly accelerates processing but also minimally impacts generative quality. The speedup ratio is almost proportional to the number of devices used, demonstrating efficient resource utilization. Visualization results in Figure 5 (a) illustrate the high generative quality achieved even with substantially reduced latency. Although achieving pixel-level consistency with the original output is challenging at high acceleration ratios, the generated image still effectively conveys the semantic information in the prompt, which is crucial for generative results.

Pixel-level Consistency by Warm-up. In Table 2, we explore the balance between pixel-level consistency and processing speed by adjusting the warm-up steps in the diffusion models. As the initial steps of these models play a crucial role in reconstructing the global structure based on text prompts [68], a modest increase in warm-up steps can significantly enhance consistency with the

- Table 3: Quantitative comparison with other parallel acceleration methods. To ensure a fair comparison with Distrifusion, we increased the warm-up steps in our method to match the speedup ratio of Distrifusion, allowing us to fairly compare generation quality and resource costs.

Method Speed up↑ Devices MACs↓ Memory↓ CLIP Score↑ FID↓ LPIPS↓ Original Model 1.0x 1 76T 5240MB 31.60 27.87 – Faster Diffusion 1.6x 1 57T 9692MB 30.84 29.95 0.3477

- Distrifusion 1.6x 2 38T 6538MB 31.59 27.89 0.0178

- Ours (N=2 S=1) 1.6x 2 44T 5450MB 31.59 27.79 0.0944

Distrifusion 2.3x 4 19T 7086MB 31.43 27.97 0.2710

- Ours (N=2 S=2) 2.3x 3 20T 5516MB 31.49 27.71 0.2117 Distrifusion 2.7x 8 10T 7280MB 31.31 28.12 0.2934

- Ours (N=3 S=2) 2.7x 4 14T 5580MB 31.40 28.03 0.1940

Oiginal 1 Device

Distrifusion 2.3x Speedup 4 Devices

Distrifusion 1.6x Speedup 2 Devices

Ours 2.3x Speedup 3 Devices

Ours 1.6x Speedup 2 Devices

Ours 2.7x Speedup 4 Devices

Distrifusion 2.7x Speedup 8 Devices

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

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

- Figure 6: Qualitative Comparison with Distrifusion on SD2.1. At the same acceleration ratio, AsyncDiff outperforms in generating higher quality and more consistent images with the original.

original images. Figure 5(b) illustrates this trend with qualitative comparisons of generative results on SDXL using gradually increasing warm-up steps. Increasing the warm-up steps to 9 achieves visual indistinguishability from the original output while maintaining an impressive 2.8x acceleration ratio.

Comparison with Acceleration Baselines. We evaluated our AsyncDiff method on SD 2.1 against two other parallel acceleration methods: Faster Diffusion [21] and Distrifusion [20]. Faster Diffusion employs encoder propagation but compromises significantly on generative quality. As its parallelism maintains theoretical and lacks a multi-device implementation, we cannot measure its realistic latency with more than one GPU. Its ideal speed-up on 2 devices is about 1.9x. Distrifusion, on the other hand, uses patch parallelism for distributed acceleration but faces potential issues with low resource utilization and high GPU memory demands.

According to Table 3, our method achieves the same operational speed using only 4 GPUs and 3 GPUs as Distrifusion does with 8 GPUs and 4 GPUs, respectively. Additionally, our method requires almost the same amount of memory as the original setup, whereas Distrifusion significantly increases memory requirements, posing extra challenges for practical applications. In terms of generative quality, AsyncDiff and Distrifusion both mirror the original diffusion model’s performance at a 1.6x acceleration ratio. However, at higher speedup ratios of 2.3x and 2.7x, our method demonstrates significantly superior generative quality. Qualitative comparisons in Fig 6 further show that AsyncDiff maintains better pixel-level consistency with the original input compared to Distrifusion.

##### 4.3 Experimental Results on Video Diffusion Models

As presented in Table 4, we conducted experiments with different configurations on two video diffusion models: SVD [2] (25 frames), and AnimentDiff [6] (16 frames), to demonstrate the efficacy

- Table 4: Quantitative evaluations of AsyncDiff on text-to-video and image-to-video diffusion models. We present the results with various configurations.

Base Model Configuration Devices MACs↓ latency↓ Speed up↑ CLIP Score↑

AnimateDiff (Text-to-Video)

Original Model 1 786T 43.5s 1.0x 30.65

- + Ours (N=2 S=1) 2 393T 24.5s 1.8x 30.65

- + Ours (N=3 S=1) 3 262T 19.1s 2.3x 30.54

- + Ours (N=2 S=2) 3 197T 14.2s 3.0x 30.32

- + Ours (N=3 S=2) 4 131T 11.5s 3.8x 30.20

SVD (Image-to-Video)

Original Model 1 3221T 184s 1.0x 26.88

- + Ours (N=2 S=1) 2 1611T 101s 1.8x 26.66

- + Ours (N=3 S=1) 3 1074T 80s 2.3x 26.56

- + Ours (N=4 S=1) 4 805T 68s 2.7x 26.19

- Table 5: Effect of stride denoising on SD 2.1. Stride denoising significantly lowers overall latency and the communication cost while only slightly compromising the generative quality

Communication

Configuration MACs↓ Latency↓ Speedup↑

CLIP Score↑ Nums↓ Latency↓

- AsyncDiff (3 devices) w/o stride denoising 25T 2.41s 2.3x Faster 49 times 0.23s(9.5%) 31.56

- AsyncDiff (3 devices) w/ stride denoising 19T 1.82s 3.0x Faster 25 times 0.12s(6.6%) 31.43

- AsyncDiff (4 devices) w/o stride denoising 19T 2.10s 2.6x Faster 49 times 0.40s(19.0%) 31.40

- AsyncDiff (4 devices) w/ stride denoising 13T 1.35s 4.0x Faster 25 times 0.10s(7.4%) 31.22

of our method. Video generation, often constrained by exceptionally high latency and substantial computation load, greatly benefits from our approach. For a 50-step video diffusion model, AsyncDiff significantly reduces latency—by tens or even hundreds of seconds—while preserving the quality of generated content. Qualitative results shown in the Appendix. D further corroborate the effectiveness of our method. AsyncDiff achieves an impressive acceleration ratio of over three times while still producing videos that closely match the prompt descriptions, ensuring the rationality of actions and details. These findings highlight the substantial potential of AsyncDiff in accelerating the inference process of video diffusion models.

##### 4.4 Effect of Stride Denoising

We introduce stride denoising to further enhance the efficiency of the asynchronous denoising process. Stride denoising completes multiple steps simultaneously through a single parallel computation, reducing the number of parallel rounds and communication frequency across devices. For a diffusion process with T steps and warm-up step W, the number of broadcasts decreases from T − W to (T − W)//2 with a stride of 2. This strategy also reduces the computational load on each device by skipping unnecessary calculations. Table 5 shows the effects of stride denoising in our parallel framework with 3 and 4 devices. Stride denoising significantly lowers overall latency and the proportion of communication time, especially as the number of devices used increases. While stride denoising slightly impacts generation quality, this effect is minimal and can be mitigated by a modest increase in warm-up steps, preserving efficiency and maintaining quality.

### 5 Conclusion

In this paper, we propose a new parallel paradigm, AsyncDiff, to accelerate diffusion models by leveraging model parallelism across multiple devices. We split the denoising model into several components, each assigned to a different device. We transform the conventional sequential denoising into an asynchronous process by exploiting the high similarity of hidden states between consecutive time steps, enabling each component to compute in parallel. Our method has been comprehensively validated on three image diffusion models (SD 2.1, SD 1.5, SDXL) and two video diffusion models (SVD, AnimateDiff). Extensive experiments demonstrate that our approach significantly accelerates inference with only a marginal impact on generative quality. This work investigates the practical application of model parallelism in diffusion models, establishing a new baseline for future research in distributed diffusion models.

### References

- [1] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. arXiv preprint arXiv:2201.06503, 2022.
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [3] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5):2567–2581, 2020.
- [4] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models. Advances in neural information processing systems, 36, 2024.
- [5] Lanqing Guo, Chong Wang, Wenhan Yang, Siyu Huang, Yufei Wang, Hanspeter Pfister, and Bihan Wen. Shadowdiffusion: When degradation prior meets diffusion model for shadow removal. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14049–14058, 2023.
- [6] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- [8] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.
- [9] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [11] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. In International Conference on Machine Learning, pages 13916–13932. PMLR, 2023.
- [12] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism. Advances in neural information processing systems, 32, 2019.
- [13] Zhihao Jia, Matei Zaharia, and Alex Aiken. Beyond data and model parallelism for deep neural networks. Proceedings of Machine Learning and Systems, 1:1–13, 2019.
- [14] Animesh Karnewar, Andrea Vedaldi, David Novotny, and Niloy J Mitra. Holodiffusion: Training a 3d diffusion model using 2d images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18423–18433, 2023.
- [15] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.
- [16] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021.
- [17] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023.
- [18] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020.

- [19] Bo Li, Kaitao Xue, Bin Liu, and Yu-Kun Lai. Bbdm: Image-to-image translation with brownian bridge diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, pages 1952–1961, 2023.
- [20] Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Ming-Yu Liu, Kai Li, and Song Han. Distrifusion: Distributed parallel inference for high-resolution diffusion models. arXiv preprint arXiv:2402.19481, 2024.
- [21] Senmao Li, Taihang Hu, Fahad Shahbaz Khan, Linxuan Li, Shiqi Yang, Yaxing Wang, Ming-Ming Cheng, and Jian Yang. Faster diffusion: Rethinking the role of unet encoder in diffusion models. arXiv preprint arXiv:2312.09608, 2023.
- [22] Xin Li, Yulin Ren, Xin Jin, Cuiling Lan, Xingrui Wang, Wenjun Zeng, Xinchao Wang, and Zhibo Chen. Diffusion models for image restoration and enhancement–a comprehensive survey. arXiv preprint arXiv:2308.09388, 2023.
- [23] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36, 2024.
- [24] Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E Gonzalez, et al. {AlpaServe}: Statistical multiplexing with model parallelism for deep learning serving. In 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23), pages 663–679, 2023.
- [25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.
- [26] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. arXiv preprint arXiv:2202.09778, 2022.
- [27] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022.
- [28] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [29] Zhaoyang Lyu, Xudong Xu, Ceyuan Yang, Dahua Lin, and Bo Dai. Accelerating diffusion models via early stop of the diffusion process. arXiv preprint arXiv:2205.12524, 2022.
- [30] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. arXiv preprint arXiv:2312.00858, 2023.
- [31] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.
- [32] Norman Müller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter Kontschieder, and Matthias Nießner. Diffrf: Rendering-guided 3d radiance field diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4328–4338, 2023.
- [33] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R Devanur, Gregory R Ganger, Phillip B Gibbons, and Matei Zaharia. Pipedream: generalized pipeline parallelism for dnn training. In Proceedings of the 27th ACM symposium on operating systems principles, pages 1–15, 2019.
- [34] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–15, 2021.
- [35] Ozan Özdenizci and Robert Legenstein. Restoring vision in adverse weather conditions with patch-based denoising diffusion models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
- [36] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [37] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [39] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10219–10228, 2023.
- [40] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.
- [41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [42] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence, 45(4):4713–4726, 2022.
- [43] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [44] Hiroshi Sasaki, Chris G Willcocks, and Toby P Breckon. Unit-ddpm: Unpaired image translation with denoising diffusion probabilistic models. arXiv preprint arXiv:2104.05358, 2021.
- [45] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.
- [46] Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv preprint arXiv:2306.14435, 2023.
- [47] Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, and Nima Anari. Parallel sampling of diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [48] Junhyuk So, Jungwon Lee, and Eunhyeok Park. Frdiff: Feature reuse for exquisite zero-shot acceleration of diffusion models. arXiv preprint arXiv:2312.03517, 2023.
- [49] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [50] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [51] Xuan Su, Jiaming Song, Chenlin Meng, and Stefano Ermon. Dual diffusion implicit bridges for image-toimage translation. arXiv preprint arXiv:2203.08382, 2022.
- [52] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 2555–2563, 2023.
- [53] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. arXiv preprint arXiv:2305.07015, 2023.
- [54] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
- [55] Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al. Cache me if you can: Accelerating diffusion models through block caching. arXiv preprint arXiv:2312.03209, 2023.

- [56] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.
- [57] Yuanzhong Xu, HyoukJoong Lee, Dehao Chen, Blake Hechtman, Yanping Huang, Rahul Joshi, Maxim Krikun, Dmitry Lepikhin, Andy Ly, Marcello Maggioni, et al. Gspmd: general and scalable parallelization for ml computation graphs. arXiv preprint arXiv:2105.04663, 2021.
- [58] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391, 2023.
- [59] Xingyi Yang and Xinchao Wang. Hash3d: Training-free acceleration for 3d generation. arXiv preprint arXiv:2404.06091, 2024.
- [60] Xingyi Yang, Daquan Zhou, Jiashi Feng, and Xinchao Wang. Diffusion probabilistic model made slim. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22552– 22562, 2023.
- [61] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. arXiv preprint arXiv:2311.18828, 2023.
- [62] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. arXiv preprint arXiv:2401.13627, 2024.
- [63] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image super-resolution by residual shifting. Advances in Neural Information Processing Systems, 36, 2024.
- [64] Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, and In So Kweon. Text-to-image diffusion model in generative ai: A survey. arXiv preprint arXiv:2303.07909, 2023.
- [65] Dingkun Zhang, Sijia Li, Chen Chen, Qingsong Xie, and Haonan Lu. Laptop-diff: Layer pruning and normalized distillation for compressing diffusion models. arXiv preprint arXiv:2404.11098, 2024.
- [66] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022.
- [67] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [68] Wentian Zhang, Haozhe Liu, Jinheng Xie, Francesco Faccio, Mike Zheng Shou, and Jürgen Schmidhuber. Cross-attention makes inference cumbersome in text-to-image diffusion models. arXiv preprint arXiv:2404.02747, 2024.
- [69] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single image editing with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6027–6037, 2023.
- [70] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and KwanYee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [71] Yang Zhao, Yanwu Xu, Zhisheng Xiao, and Tingbo Hou. Mobilediffusion: Subsecond text-to-image generation on mobile devices. arXiv preprint arXiv:2311.16567, 2023.
- [72] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36, 2024.

❈ In this document, we provide supplementary materials that extend beyond the scope of the main manuscript, constrained by space limitations.

###### Warm-up Parallel Parallel Parallel Parallel

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

GPU 3 (  3)

[Figure 291]

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

-  −8

[Figure 312]

-  −9

-  −6

[Figure 313]

-  −7

 −1

-  −2

[Figure 314]

-  −3

-  −4

[Figure 315]

-  −5

- T-7
- T-8

T

- T-5
- T-6

- T-1
- T-2

- T-3
- T-4

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

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

GPU 2 (  3)

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

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

 −1

T-8

T

T-6

T-2

T-4

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

GPU 1 (  2)

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

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

 −1

T-8

T

T-6

###### T-2

T-4

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

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

GPU 0 (  1)

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

 −1

T

Communication within device Communication across devices T: Time embedding

- Figure 7: Schematic of the asynchronous diffusion model with stride denoising. The model ϵθ is

divided into three components {ϵnθ}3n=1, with a stride S of 2 for clarity. A single parallel batch results in the completion of denoising for two steps

### A More Implementation Details.

Model Segmentation. In our method, we partition the cumbersome denoising model into multiple components, each assigned to a different device. After successfully parallelizing the computation of each component, the time cost for each time step now corresponds to the maximum latency among these components. To optimize parallel processing efficiency, we partition the model into segments that each carry a roughly equal computational load. This arrangement allows all modules to finish their computations nearly simultaneously, making full use of available computational resources. The segmentation strategy is sequential except for SDXL [36]. For the denoising U-net within the SDXL module, we group its first and last blocks into a single segment and apply sequential splitting to the remaining blocks. This is because SDXL has specific needs for high-frequency details, and res connections typically contain abundant high-frequency information.

Time Shifting. We introduce a technique called time shifting. Following the warm-up steps, the time embedding for each step is shifted back by one step. For instance, in a 50-step asynchronous denoising process with a warm-up of 2 steps, the original sequence of time embeddings is {50,49,48,47,...,3,2,1}. With time shifting, this sequence is adjusted to {50,49,49,48,...,3,2}. In certain extreme cases, asynchronous denoising might leave residual noise in the output. Time shifting addresses this by adjusting the time embeddings backward, enhancing the denoising effect. It’s important to note that time shifting is not a standard component of our method but is employed optionally. The quantitative results presented in this paper are achieved without the use of time shifting.

Stride Denoising. To further enhance efficiency, we introduce stride denoising, which completes multiple denoising steps simultaneously through a single parallel computation. Figure 7 illustrates the full schematic of applying stride denoising to AsyncDiff. In this depiction, the denoising model ϵθ is divided into three components ϵnθ3n=1, and for clarity, the stride S is set to 2. Unlike the continuous broadcasting of hidden states at each time step, stride denoising broadcasts them every two steps. As depicted, at time step {T − 1,T − 3,T − 5,T − 7}, we conduct denoising alone, and at time step {T − 2,T − 4,T − 6,T − 8}, we compute and broadcast the hidden states for the next parallel computation round. Consequently, the hidden states from time step {T − 1,T − 3,T − 5,T − 7}

are not required, allowing us to skip the calculations for ϵ1θ and ϵ2θ at these steps. Stride denoising effectively reduces both computational load and communication demands by decreasing the parallel

computing rounds needed to complete the process. Compared to the significant improvements it brings in efficiency, the quality sacrifice is minimal and can be entirely compensated for by slightly increasing the warm-up steps.

### B More Analysis.

Time cost. In Table 6, we present the time costs associated with model running and inter-device communication when using AsyncDiff on SD 2.1. Generally, communication expenses constitute only a minor fraction of the total time cost, demonstrating that AsyncDiff is an effective distributed acceleration technique suitable for practical application. It is important to note that as the number of devices increases, the time needed for data broadcasting between devices also rises, thereby increasing the proportion of communication costs. However, employing stride denoising can substantially reduce these costs by decreasing the number of parallel rounds needed to complete the denoising process.

Table 6: Time cost comparisons on SD 2.1. ’Ratio’ in this table represents the proportion of communication cost to overall latency. All measurements were conducted on NVIDIA A5000 GPUs equipped with NVLINK Bridge

Time Cost Overall Running Comm. Ratio

Config

- N=2 S=1 3.03s 2.90s 0.13s 4.30%

- N=3 S=1 2.41s 2.18s 0.23s 9.54%

- N=4 S=1 2.10s 1.80s 0.30s 14.29%

- N=2 S=2 1.82s 1.70s 0.12s 6.59%

- N=3 S=2 1.35s 1.25s 0.10s 7.40%

Speedup Ratio. We also evaluate the acceleration ratio on SD 2.1 with varying numbers of denoising steps. As indicated in Table 7, AsyncDiff significantly enhances processing speed, even with a denoising procedure consisting of only 25 steps. When the number of steps extends to 100, our approach achieves a speedup of up to 4.3x, surpassing the ratio of devices employed.

Table 7: Acceleration ratio on SD 2.1 under different num of denoising steps

Speedup↑ 25steps 50steps 100steps

Config

Origin 1.0x (2.89s) 1.0x (5.51s) 1.0x (10.96s)

- N=2 S=1 1.7x (1.70s) 1.8x (3.03s) 1.8x (6.04s)

- N=3 S=1 2.1x (1.35s) 2.3x (2.41s) 2.3x (4.71s)

- N=4 S=1 2.4x (1.21s) 2.6x (2.10s) 2.7x (4.01s)

- N=2 S=2 2.7x (1.05s) 3.0x (1.82s) 3.2x (3.39s)

- N=3 S=2 3.4x (0.86s) 4.0x (1.35s) 4.3x (2.52s)

### C More Quantitative Results.

To thoroughly assess the quality of images produced following acceleration, we provide quantitative analyses on three base models (SD 2.1 [38], SD 1.5 [38], SDXL [36]) using four additional metrics: the full reference metric, DISTS [3], and no-reference metrics including MUSIQ [16], CLIP-IQA [52], and NIQE [31]. The experimental results in Table 8 demonstrate that our method significantly reduces inference latency while maintaining a high level of quality in diffusion model-generated images. On SD 1.5, our approach not only accelerates the inference process but also brings the image quality closer to the natural distribution.

Table 8: Quantitative evaluations of AsyncDiff on three text-to-image diffusion models using more metrics including DISTS [3], MUSIQ [16], CLIP-IQA [52], and NIQE [31].

Base Model Configuration Devices DISTS↓ MUSIQ↑ CLIP-IQA↑ NIQE↓

Original Model 1 – 69.95 0.6653 3.9675

- + Ours (N=2 S=1) 2 0.1041 69.55 0.6539 3.8850

- + Ours (N=3 S=1) 3 0.1280 69.04 0.6441 3.9438

- + Ours (N=4 S=1) 4 0.1419 68.58 0.6365 3.9724

SD 2.1

- + Ours (N=2 S=2) 3 0.1556 68.03 0.6158 3.5761

- + Ours (N=3 S=2) 4 0.1689 67.13 0.5986 3.6761

Original Model 1 – 71.98 0.6534 3.5517

- + Ours (N=2 S=1) 2 0.1169 72.21 0.6569 3.7448

- + Ours (N=3 S=1) 3 0.1434 71.73 0.6481 3.8023

- + Ours (N=4 S=1) 4 0.1599 71.51 0.6442 3.8620

SD 1.5

- + Ours (N=2 S=2) 3 0.1668 71.14 0.6323 3.9613

- + Ours (N=3 S=2) 4 0.1905 69.42 0.6070 4.1047

Original Model 1 – 71.58 0.6633 4.0743

- + Ours (N=2 S=1) 2 0.1038 70.56 0.6498 4.1139

- + Ours (N=3 S=1) 3 0.1211 69.88 0.6389 4.1585

- + Ours (N=4 S=1) 4 0.1391 67.70 0.6056 4.0927

SDXL

- + Ours (N=2 S=2) 3 0.1329 69.56 0.6222 4.1685

- + Ours (N=3 S=2) 4 0.1527 68.16 0.5955 4.2745

### D More Qualitative Results.

Qualitative Results on Image Diffusion Models. As depicted in Figure 8, we present further qualitative results for SD 2.1 and SDXL under various configurations. The speedup achieved is nearly proportional to the number of devices utilized, indicating efficient resource usage by our method. Moreover, the images generated by our approach closely match the text descriptions and are of high quality.

Qualitative Results on Video Diffusion Models. We present qualitative evaluations of AsyncDiff applied to the video diffusion models. Figures 9, 10, and 11 illustrate the generated results using our method on the text-to-video model AnimateDiff [6]. Figure 12 displays results from applying our method to the image-to-video model SVD [2]. For a 50-step video diffusion model, AsyncDiff markedly decreases latency—saving tens or even hundreds of seconds—while maintaining the integrity and quality of the generated videos.

### E Limitations

As a distributed acceleration framework, AsyncDiff necessitates frequent communication between devices throughout the denoising process. Consequently, if the devices lack the capability to communicate effectively or have subpar communication infrastructure, our method may not perform optimally. Additionally, AsyncDiff operates as a plug-and-play acceleration solution that depends on pre-trained diffusion models. Therefore, if the baseline quality of the original diffusion models is unsatisfactory, achieving high-quality results with our method could be challenging.

### F Societal impacts

In this paper, we introduce a universal distributed acceleration approach for diffusion models. This method substantially speeds up the inference phase of diverse diffusion models by fully leveraging computational resources. It holds significant potential for practical applications, particularly in computationally intensive generation tasks like video and speech generation.

###### (a) More Qualitative Results on SD 2.1 with different configurations

- Original 2OursDevices1.7x(N=2SpeedupS=1) 3OursDevices2.4x(N=3SpeedupS=1) 4OursDevices2.7x(N=4SpeedupS=1) 3OursDevices2.8x(N=2SpeedupS=2) 4 Devices (N=3 S=2)

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

(b) More Qualitative Results on SDXL with different configurations

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

Ours 4.0x Speedup

- Original 2OursDevices1.8x(N=2SpeedupS=1) 3OursDevices2.3x(N=3SpeedupS=1) 4OursDevices2.6x(N=4SpeedupS=1) 3OursDevices3.0x(N=2SpeedupS=2) 4 Devices (N=3 S=2)

Ours 3.8x Speedup

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

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

- Figure 8: Qualitative results on SD 2.1 and SDXL with different configurations.Our method maintains excellent generation quality even when achieving speedups of up to four times.

Prompt: Brilliant fireworks on the town, Van Gogh style, digital artwork, illustrative, painterly, matte painting, highly detailed, cinematic

Original 43.5s

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

Ours 23.5s (2 devices)

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

Ours 11.5s (4 devices)

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

- Figure 9: Qualitative results on AnimateDiff (1)

Prompt: panda playing a guitar, on a boat, in the blue ocean, high quality

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

Original 43.5s

Ours 23.5s (2 devices)

Ours 11.5s (4 devices)

- Figure 10: Qualitative results on AnimateDiff (2)

Ours 2.7x Speedup

Prompt: comic book style, Batman is walking, colored, dynamic background, full body view, clean sharp focus

Original 43.5s

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

Ours 23.5s (2 devices)

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

Ours 11.5s (4 devices)

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

Figure 11: Qualitative results on AnimateDiff (3)

Original 184s Ous 101s 2 Devices

Ous 80s 3 Devices

Ous 64s 4 Devices

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

Figure 12: Qualitative results on Stable Video Diffusion

