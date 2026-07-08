## Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling

Euisoo Jung Byunghyun Kim Hyunjin Kim Seonghye Cho Jae-Gil Lee* School of Computing, KAIST

{jyssys, rooknpown, hjkim1228, orangingq, jaegil}@kaist.ac.kr

Speed-up Image Quality

[Figure 1]

[Figure 2]

Naïve diffusion

Model Generality

[Figure 3]

Ours

# arXiv:2602.21760v1[cs.CV]25Feb2026

Communication Efficiency

High-

Ours AsyncDiff DistriFusion

2.3x Speed-Up at 2 GPUs

resolution

Time

0

7.12 16.49

Figure 1. Summary of the proposed hybrid data-pipeline parallelism. Our method consistently outperforms prior distributed approaches across five key aspects: Speed-up, Image Quality, Generality, High-resolution Synthesis, and Communication Cost, demonstrating robust and balanced acceleration-quality trade-offs.

##### 1. Introduction

##### Abstract

Diffusion models have achieved remarkable progress in high-fidelity image, video, and audio generation, yet inference remains computationally expensive. Nevertheless, current diffusion acceleration methods based on distributed parallelism suffer from noticeable generation artifacts and fail to achieve substantial acceleration proportional to the number of GPUs. Therefore, we propose a hybrid parallelism framework that combines a novel data parallel strategy, condition-based partitioning, with an optimal pipeline scheduling method, adaptive parallelism switching, to reduce generation latency and achieve high generation quality in conditional diffusion models. The key ideas are to (i) leverage the conditional and unconditional denoising paths

Diffusion models have emerged as a powerful family of generative models because of their superior sample quality and broad applicability. However, the inherently iterative nature of diffusion processes, which consists of many denoising steps, leads to significant inference latency and computational bottlenecks. As model sizes continue to scale, these inefficiencies become increasingly limiting, making diffusion inference acceleration a pressing research challenge. Existing approaches have focused mainly on reducing the number of sampling steps[9, 19– 21, 26, 27, 34, 36, 37], designing optimal architectures[11, 13, 14, 35, 38, 39], or leveraging mathematical approximations[1, 17, 18, 22, 28, 40, 40]. Yet, these methods often require additional training or fail to deliver strong acceleration in practice, exhibiting a clear trade-off between generation quality and speed.

- as a new data-partitioning perspective and (ii) adaptively enable optimal pipeline parallelism according to the denoising discrepancy between these two paths. Our framework achieves 2.31× and 2.07× latency reductions on SDXL and SD3, respectively, using two NVIDIA RTX 3090 GPUs, while preserving image quality. This result confirms the generality of our approach across U-Net-based diffusion models and DiT-based flow-matching architectures. Our approach also outperforms existing methods in acceleration under high-resolution synthesis settings. Code is available
- at https://github.com/kaist-dmlab/Hybridiff.

Distributed parallelism across multiple GPUs offers a promising alternative. Using modern parallel computing resources, one can achieve substantial throughput improvements in diffusion inference without additional training. This direction is especially appealing given the success of distributed strategies in natural language processing, where large-scale language models have already benefited from extensive parallelism research[24, 29]. As in other domains, distributed parallelism for generative model inference can be broadly classified into data parallelism

∗ indicates corresponding author.

a. Diffusion Inference Data Parallel Processing b. Diffusion Inference Pipeline Parallel Processing

𝐱 ,  & 𝐱 , ,𝑐

[Figure 4]

𝐱 & 𝐱 ,𝑐

|𝐱 & 𝐱 ,𝑐<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]| |[Figure 8]|
|---|---|---|
|[Figure 9]| | |
| | | |

Multi-GPU

| |
|---|

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

𝐱

𝐱 & 𝐱 ,𝑐

[Figure 15]

[Figure 16]

𝐱 & 𝑥 ,𝑐 𝐱

[Figure 17]

[Figure 18]

⋯

𝐱 ,  & 𝐱 , ,𝑐

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Artifact in connection

- Speed-Up: 1.2x Quality ↓

Speed-Up: 1.3x Quality -

|[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]| |
|---|---|

⋯

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

Multi-GPU

[Figure 43]

[Figure 44]

|[Figure 45]|
|---|

[Figure 46]

[Figure 47]

[Figure 48]

c. Ours (Condition-Based Data & Pipeline Parallel Processing)

𝐱 & 𝐱 ,𝑐

𝐱

𝐱

- Speed-Up: 2.3x Quality ↑

Parallelism Step

|: Communication<br><br>: Pre-trained Diffusion Model<br><br>: Conditional Communication : Unconditional Communication<br><br>𝐱 : Latent Noise at time step t<br><br>: All-gather Operation<br><br>𝑐 : Conditional Information|
|---|

(Adaptive Parallelism Switching)

[Figure 49]

Multi-GPU

|[Figure 50]<br><br>[Figure 51]|
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

|[Figure 58]|
|---|

[Figure 59]

⋯

[Figure 60]

𝐱 ,𝑐

|[Figure 61]<br><br>[Figure 62]|
|---|

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Parallelism Step

- Figure 2. Comparison of parallel strategies for diffusion inference. (a) Patch-based data parallel frameworks suffer from bottlenecks caused by all-gather operations and artifacts at patch boundaries, leading to limited acceleration and quality degradation. (b) Pipeline parallel frameworks incur excessive asynchronous communication overhead and accumulate estimate errors. (c) Our hybrid parallelism, which incorporates condition-based data parallelism, adaptively combines both paradigms to achieve high fidelity and fast generation.

and pipeline parallelism[2, 12]. Both approaches enhance throughput by distributing either the input data or the model itself across multiple GPUs.

when using two GPUs, data and model parallelism achieved 1.2× and 1.3×speed-up, respectively, whereas our hybrid approach remarkably achieved a 2.3×speed-up under the same configuration, as shown in Figures 1 and 2.

Representative existing studies include DistriFusion[12] for data parallelism and AsyncDiff[2] for pipeline parallelism. In DistriFusion (Figure 2a), an input image is divided into N disjoint patches, and these patches are processed in parallel across N GPUs, where each device independently handles one patch. In AsyncDiff (Figure 2b), the entire model is divided into N sequential components, where each component is assigned to a GPU, and the output from the i-th GPU is asynchronously fed as the input to the (i + 1)-th GPU; thus, AsyncDiff enables pipelined execution across devices.

To achieve hybrid parallelism, one could combine the aforementioned representative methods. Specifically, an image is divided into disjoint patches, and each patch is fed into a corresponding model component (not necessarily the first one). As a result, each GPU trains a 1/N portion of the model using a 1/N portion of an input image. This hybrid approach can potentially achieve beyond-linear scaling; however, it may degrade generation quality for two main reasons. First, since each GPU processes only a portion of the image, artifacts are likely to appear particularly along patch boundaries. Second, this issue is exacerbated by asynchronous communication between model components; that is, errors introduced by asynchronous rather than sequential denoising can worsen the artifacts.

In theory, each form of parallelism can improve throughput linearly with respect to the number of GPUs, up to an ideal N×speed-up, but in practice, the gains are often sublinear due to communication overhead and synchronization costs. In this paper, we propose a hybrid strategy that combines data and model parallelism to further increase the throughput of generative model inference, achieving beyond-linear scaling relative to the number of GPUs, while maintaining generation quality. That is, if there are two GPUs, we aim to obtain more than a twofold speed-up without noticeable degradation in output fidelity. In practice,

In this paper, we aim to propose and further optimize the hybrid parallelism for diffusion inference from two complementary perspectives: (1) from the data parallelism perspective, transitioning from patch-based partitioning to condition-based partitioning; and (2) from the model parallelism perspective, advancing from static parallelism switching to adaptive parallelism switching.

- (1) Condition-Based Partitioning. The main limitation of patch-based partitioning is that each patch represents only a local subregion of an image, often leading to boundary artifacts and degraded visual coherence. To address this limitation, we leverage the classifier-free guidance(CFG)[7], a technique widely adopted in diffusion models, where the model simultaneously predicts conditional (prompted) and unconditional (unprompted) noise estimates. This dualpath prediction naturally provides a meaningful criterion for data partitioning: as shown in Figure 2c, the conditioned

(xt,c) and unconditioned (xt) inputs form two distinct dataparallel paths. Importantly, unlike patch-based partitioning, each image partition covers the entire image, thereby preserving global consistency. Consequently, condition-based partitioning yields improved visual coherence and reduced communication overhead during feature aggregation.

- (2) Adaptive Parallelism Switching. Because we revise the data partitioning strategy, the pipeline parallelism must also be adapted to align with it. In the early denoising steps, the conditional and unconditional noise estimates differ substantially due to the presence or absence of the condition. Consequently, asynchronous denoising at this stage can lead to divergence between the two paths. To mitigate this issue, we defer the onset of parallel execution until the noise estimates of the two paths become sufficiently similar, beyond the conventional warm-up phase used in prior works (e.g., [2]). Similarly, toward the final denoising steps, the noise estimates from the two paths begin to diverge again; at this point, parallel execution is terminated. The specific switching points between serial and parallel execution are determined automatically based on a novel metric, called the denoising discrepancy, which quantifies the difference between the two noise estimates. This adaptive switching mechanism effectively improves generation quality by reducing error propagation, while only marginally shortening the duration of parallel processing.

This novel framework demonstrates consistent acceleration not only on conventional denoising diffusion models but also on recent state-of-the-art generative frameworks such as flow matching[16]. As long as the model follows a sequential denoising process that allows quantifying the relative influence between conditional and unconditional branches, our framework remains robust and effective. Furthermore, due to the nature of pipeline parallelism, it is not restricted to specific architectures such as U-Net or DiT, showing strong generality across diverse networks.

As summarized in Figure 1, our proposed hybrid parallelism achieves superior performance across the five key aspects. In fact, compared to single-GPU inference, our method achieves a 2.3×speed-up with two GPUs (i.e., > 2), while preserving generation fidelity. See Appendix A and Section 5 for details of Figure 1. Finally, the key contributions are summarized as follows.

- • Hybrid Parallelism Framework for Diffusion Inference. We introduce a novel diffusion inference parallelism framework that integrates condition-based partitioning and adaptive parallelism switching into a unified hybrid parallelism design.
- • Novel Condition-Based Partitioning. At the data parallelism level, we exploit the intrinsic mechanism of diffusion by decoupling conditional and unconditional branches and performing multi-GPU denoising.
- • Adaptive Parallelism Switching. To align pipeline parallelism with the behavior of conditional guidance, our method adaptively switches to hybrid parallelism framework during inference. Switching points are automatically determined based on the denoising discrepancy between conditional and unconditional estimates, ensuring generation efficiency throughout the denoising process.
- • Robustness across Models and Architectures. Our framework consistently demonstrates strong acceleration and generation quality across various architectures (e.g., U-Net, DiT) and recent state-of-the-art generative frameworks, such as flow matching, even under high-resolution synthesis settings.

##### 2. Related Work

Single-GPU Diffusion Acceleration. Research on the acceleration of single-device diffusion inference can be classified into three categories. The first group focuses on reducing the number of sampling steps required for high-quality generation[9, 19–21, 26, 27, 30, 34, 36, 37, 40]. These approaches enable fast sampling by either reformulating the reverse process as an ordinary differential equation(ODE), distilling multi-step models into fewer steps, or directly predicting the reverse process in latent space. The second group targets model architecture optimization, aiming to reduce computational cost through network compression and efficient design[11, 13, 14, 35, 38, 39]. The third group leverages mathematical and algorithmic strategies, either exploiting the mathematical structure of diffusion processes or reusing intermediate computations to further accelerate inference[1, 17, 18, 22, 28, 33, 40]. While these methods reduce single-device inference time, they are inherently limited by the computational capacity of individual GPUs.

Multi-GPU Diffusion Acceleration. Recent studies have explored various distributed parallelism strategies to accelerate diffusion inference using multiple GPUs[2, 4, 5, 12, 32]. DistriFusion[12] introduces a data-parallel approach that divides the input image into independent patches, performing denoising in parallel across GPUs. This work has established a foundational paradigm for parallel diffusion inference. Building on this parallelization idea, AsyncDiff,[2] introduces model parallelism by dividing the U-Net into layer-wise segments and employing a stride-

###### Condition-Based Partitioning Condition-Based Partitioning + Adaptive Parallelism Switching Condition-Based Partitioning

[Figure 69]

[Figure 70]

[Figure 71]

###### (2) Parallelism Stage (𝜏 ,𝜏 ) (3) Fully-Connecting Stage [𝜏 ,0]

###### (1) Warm-Up Stage [𝑇,𝜏 ]

Unconditional

Unconditional

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

|[Figure 90]| |
|---|---|
| | |

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

|[Figure 98]|
|---|

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

𝑥

𝑥

𝑥

|[Figure 111]|
|---|

[Figure 112]

⋯

###### ⋯

⋯

Conditional

Conditional

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

𝑥

𝑥

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

|[Figure 130]| |
|---|---|
| | |

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

|[Figure 138]|
|---|

|[Figure 139]|
|---|

[Figure 140]

|[Figure 141]|
|---|

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

𝑥 ,𝑐

𝑥 ,𝑐

𝑥 ,𝑐 𝑥 ,𝑐

𝑥 ,𝑐

|Ordinal Communication Conditional Communication Unconditional Communication 𝑐 Conditional Information|
|---|

- Figure 3. Overview of the proposed diffusion inference hybrid parallel framework. Our method adaptively switches parallelism modes at τ1 and τ2, optimizing the trade-off between computational efficiency and consistency of conditional guidance, and demonstrates superior inference acceleration performance while preserving high generation quality.

based scheduling strategy to balance parallel execution, achieving a notable reduction in latency.

noise ϵθ(xt,t,c) and its unconditional variant ϵθ(xt,t,∅). At inference, the samples follow

Subsequently, PipeFusion[5] and XDiT[4] combine patch-level parallelism with transformer-oriented parallelism through ring attention. While additional adaptations such as CFG-based data parallelism have been introduced, these methods remain limited to inter-image processing and lack deeper architectural integration. Moreover, transformer-specific schemes such as ring attention exhibit limited scalability and inconsistent performance when applied to general diffusion architectures. More recently, ParaStep[32] proposes a reuse-then-predict mechanism that leverages the similarity of noise predictions between adjacent denoising steps. By reusing the noise from previous steps before re-prediction, ParaStep enables inter-step parallelization and significantly reduces communication overhead. However, because early and late diffusion steps exhibit larger discrepancies between adjacent noise states, the reuse mechanism can accumulate errors, leading to potential degradation in image quality or restricted speedup.

ϵcfg = ϵθ(xt,c,t) + w ϵθ(xt,c,t) − ϵθ(xt,t) ,

where w > 0 is the guidance scale. The adjusted reverse mean becomes

1 √αt

βt √1 − α¯t

xt −

µcfg(xt,t,y) =

ϵcfg .

Flow Matching. Given a target distribution q(x) and base distribution p0(x), flow matching defines an ordinary differential equation,

dx(t) dt

= v(x(t),t),

where the vector field vθ is learned by minimizing

xt − x0 t

2

LFM = Et,x

0∼q vθ xt,t −

,

with xt = x0 + te for e ∼ N(0,I). Sampling proceeds by integrating x˙ = vθ(x,t) from t = 1 to t = 0.

##### 3. Preliminaries

##### 4. Method

Denoising Diffusion Model. Let q(x0) denote the data distribution and define a forward noising process by

###### 4.1. Overview

Figure 3 illustrates the overall process of our proposed hybrid parallelism framework. The input isotropic noise latent xT is fed simultaneously into two denoising branches: the unconditional path fθ(xt,t) and the conditional path fθ(xt,c,t) guided by a textual prompt c. where fθ denotes the denoising diffusion network parameterized by θ (e.g. U-Net, DiT). To exploit both global consistency and conditional fidelity, our framework incorporates two complementary dimensions of parallelism, condition-based partitioning, and adaptive parallelism switching.

q(xt | xt−1) = N xt; 1 − βt xt−1,βtI ,

for t = 1,...,T, with variance schedule {βt}. The model learns a parameterized reverse denoising process,

pθ(xt−1 | xt) = N xt−1;µθ(xt,t),Σθ(t) , by optimizing the variational lower bound,

T

LVLB = Eq

DKL q(xt−1 | xt,x0)∥pθ(xt−1 | xt) .

Formally, given the denoising model fθ, the diffusion inference across N devices can be expressed as

t=1

Classifier-Free Guidance (CFG). For conditional generation with a condition c, the model is trained to predict the

x(t−n)1 = fθ(n)(x(tn),c(b

n),t), n ∈ {1,...,N}, bn ∈ {cond,uncond},

conditional and unconditional branches at each timestep t, (where ϵc = ϵθ(xt,c,t) and ϵu = ϵθ(xt,t)), and is formulated as

|Warm-Up Stage|Parallelism Stage<br><br>|Fully-Connecting Stage|
|---|---|---|
| | | |

rel−MAEϵ ,ϵ

Ex,ϵ[∥ϵθ(xt,c,t) − ϵθ(xt,t)∥1] Ex,ϵ[∥ϵθ(xt,t)∥1]

. (1)

rel-MAEt(ϵc,ϵu) =

Here, ϵθ(xt,c,t) and ϵθ(xt,t) denote the noise components predicted from the conditional and unconditional denoisers, respectively. A larger value indicates a stronger discrepancy between the two branches, reflecting a higher conditional influence on the denoising trajectory at that timestep.

T 0

#### 𝜏 𝜏

Time step

- Figure 4. Illustration of the rel-MAEt(ϵc, ϵu) curve. The rel-MAEt(ϵc, ϵu) value is relatively large before τ1 and after τ2, while it converges near zero between them, indicating stable alignment between conditional and unconditional branches during the parallelism phase.

According to the trend of denoising discrepancy shown in Figure 4, which exhibits a U-shaped curve over the entire denoising process, we divide the process into three stages: the Warm-Up Stage [T, τ1], the Parallelism Stage (τ1, τ2), and the Fully Connecting Stage [τ2, 0]. The two parameters, τ1 and τ2, define the boundaries between these stages and are automatically determined during the middle of the denoising process. The details of how they are determined are provided in the next section. Intuitively, τ1 marks the point where the denoising discrepancy ceases to decrease rapidly, while τ2 indicates the point where it begins to increase.

where each θ(n) corresponds to the subset of model parameters assigned to the n-th device in the pipeline, reflecting adaptive parallelism switching across different network stages. Meanwhile, bn ∈ {cond,uncond} indicates whether the device n handles the conditional or unconditional branch in condition-based partitioning. Accordingly, each device processes either a conditional input with c or an unconditional input without c. This formulation jointly represents both condition-based partitioning and adaptive parallelism switching within a unified diffusion framework.

By measuring denoising discrepancy across 5,000 prompts from the MS-COCO 2014 validation set[15], we observed that the variation of the error between the conditional and unconditional branches exhibits a clear U-shaped trend, as further demonstrated in Appendix B.

To further enhance performance, the denoising process is divided into three stages according to the temporal dynamics of conditional influence: (1) Warm-Up Stage, where only ordinal communication occurs between conditional and unconditional branches; (2) Parallelism Stage, where both branches are executed in parallel with conditional exchange; and (3) Fully-Connecting Stage, which merges the two branches for the final refinement. The rationale for this three-phase division and the quantitative criteria for determining the boundary points τ1 and τ2 are discussed in Section 4.2 and Section 4.3, respectively.

We now describe each denoising stage in detail.

- (1) Warm-Up Stage [T, τ1]. This stage captures the global outline of the generated image. The conditional branch establishes the overall composition from the text prompt, while the unconditional branch stabilizes the coarse structural forms. Since both branches have distinct influences, the denoising discrepancy remains low. Therefore, each branch is processed independently using condition-based partitioning, without adaptive parallelism switching.
- (2) Parallelism Stage (τ1, τ2). At this phase, the model refines local details within the preformed outline. The conditional and unconditional branches begin to converge, and the denoising discrepancy remains small and stable. To take advantage of this convergence, adaptive parallelism switching is activated, enabling a more powerful acceleration of the denoising process.
- (3) Fully-Connecting Stage [τ2, 0]. In the final phase, fine-grained conditional cues dominate generation. The framework reverts to condition-based partitioning, integrating conditional guidance to reconstruct the final image x0.

###### 4.2. Hybrid Parallel Inference Framework

- Figure 4 illustrates the relative-Mean Absolute Error of the

predicted noise (relative-MAE; rel-MAEt(ϵc,ϵu)) across the three stages of the proposed hybrid parallelism in the denoising diffusion model. To determine when the conditional and unconditional branches should interact or remain independent, we first quantify their denoising discrepancy during the denoising process.

Since conditional and unconditional denoisers contribute differently to generation, with one emphasizing semantic alignment to the text condition and the other stabilizing global structure, it is essential to measure how their noises ϵc and ϵu diverge over time. This discrepancy serves as a key indicator for determining the switching points between serial and parallel execution within our hybrid framework.

A similar three stages structure has also been observed in previous studies on diffusion conditional guidance[10], which further supports the validity of our framework. Building upon this, through this three stages hybrid parallelism framework, our method achieves efficient distributed denoising while preserving generation quality.

The denoising discrepancy, namely rel-MAEt(ϵc,ϵu), quantifies the difference in noise prediction ϵt between the

The denoising discrepancy, rel-MAEt(ϵc,ϵu) can be extended to flow matching models by replacing ϵθ with the predicted velocity vθ. In this case, rel-MAEt(vc,vu) maintains the same role in quantifying the conditionalunconditional discrepancy over the velocity field.

###### 4.3. Adaptive Switching via Denoising Discrepancy

The core of the proposed hybrid parallelism is to dynamically determine the timesteps τ1 and τ2 during the realtime denoising process, sequentially switching between the Warm-Up, Parallelism, and Fully-Connecting modes, while constructing a scheduling method based on the previously defined denoising discrepancy.

- (1) Determining τ1. For each timestep t, we compute denoising discrepancy and calculate the average slope of the most recent L steps by

Gt =

Mt − Mt−L L

. (2)

We then select τ1′ = min{t|0 ≤ Gt < gslope} and constrain it by safety-cap τcap. As shown in Appendix B, τcap is defined as the global minimum point of the denoising discrepancy curve, and it serves as an upper bound for τ1 during automatic selection. The introduction of τcap ensures stability by covering cases where τ1 is assigned too late or remains undefined due to outlier behaviors, thus maintaining generation quality while maximizing acceleration.

Consequently, τ1 is given by τ1 = min(τ1′,τcap), which marks the end of the warm-up stage where conditional influence stabilizes.

- (2) Determining τ2. During the parallelism phase, ϵc and ϵu converge to an identical value, making the denoising discrepancy measurement no longer meaningful. Therefore, τ2 is empirically fixed to a certain number of steps k after τ1,

τ2 = τ1 + k, k ∈ N, 1 ≤ k < T − τ1. (3) A larger k extends the parallelism phase, resulting in faster inference but lower generation quality, while a smaller k improves fidelity at the cost of latency. A detailed analysis of quality and speed trade-offs with respect to k is presented in Section 5.4, where we empirically verify the optimal balance across various k. We also provide the algorithm of Section 4.3 in Appendix C describes the overall process.

###### 4.4. Theoretical Analysis of Adaptive Switching

Analysis of Denoising Discrepancy by Score Decomposition. The denoising discrepancy can be theoretically interpreted as a ratio between the conditional information strength and the unconditional data prior. From the scoredecomposition perspective[8, 31], can be approximated as

t log p(c|xt)∥1

c−ϵu∥1 ∥ϵu∥1 ≈ ∥∇x

rel-MAEt(ϵc,ϵu) = ∥ϵ

∥su(xt,t)∥1 . (4) ∇xt

log p(c|xt) represents the conditional information strength and su(xt,t) denotes the unconditional score of

the data distribution. Consequently, denoising discrepancy measures the relative magnitude between conditional and unconditional components.

In the score formulation of Eq. (4), the unconditional score su(xt,t) = ∇xt

log p(xt) captures the intrinsic structure of the data distribution, while the conditional gradient ∇xt

log p(c|xt) encodes the semantic influence of the conditioning signal c. Their relative magnitudes evolve naturally along the diffusion process:

- • Warm-Up Stage: When xt is close to pure noise, su(xt,t) carries little structural information, whereas

∇xt

log p(c|xt) dominates by guiding the global semantic layout from the prompt, leading to a large denoising discrepancy.

- • Parallelism Stage: As denoising progresses, su(xt,t) reconstructs meaningful local structures and becomes

comparable in magnitude to ∇xt

log p(c|xt). This balance satisfies ∥su(xt,t)∥ ≈ ∥∇xt

log p(c|xt)∥, yielding

d dtrel-MAEt(ϵc,ϵu) ≈ 0 and motivates the activation of the parallel inference phase.

- • Fully-Connecting Stage: At high SNR, most patterns

have been recovered by su(xt,t), while ∇xt

log p(c|xt) contributes to fine-grained alignment and texture refinement, causing a mild increase in denoising discrepancy.

This interpretation provides an intuitive explanation of how the relative magnitudes of the conditional and unconditional scores evolve across timesteps, theoretically supporting the three stages proposed (Warm-Up → Parallelism → Fully Connecting). Detailed derivations of Eq. (4) and the robustness analysis of τ1 under stochastic denoising noise are shown in Appendix D and Appendix E, respectively.

###### 4.5. Extensibility to Many GPU Configurations

While the hybrid parallelism framework is optimized for two GPUs, it also scales well to larger even-numbered configurations. We present two extension strategies.

- (1) Batch-Level Extension. In this approach, the model generates N2 samples across N GPUs, where each pair of GPUs produces one image. This structure linearly increases acceleration with the number of GPUs while maintaining near-identical generation quality. However, it is most effective when a large number of samples are generated.

- (2) Layer-Wise Pipeline Extension. This method extends the adaptive parallelism switching mechanism by dividing the optimal pipeline interval into N layer-wise segments, thereby enabling finer-grained parallel execution across multiple devices. Unlike the batch-level scheme, it can be applied to single-sample generation, though it may incur slightly reduced acceleration efficiency and minor quality degradation due to finer partitioning.

The structures and details of both strategies are provided in Appendix F. Supporting a degree of parallelism greater than two for a single image is deferred to future work.

FID ↓ LPIPS ↓ PSNR ↑

Base Model Devices Methods Latency (s) ↓ Speed-Up ↑ Comm. (GB) ↓

w/ G.T. w/ Orig. w/ G.T. w/ Orig. w/ G.T. w/ Orig.

- 1 Original Model 16.49 - - 23.977 - 0.797 - 9.618 -

- 2

DistriFusion[12] 13.53 1.22× 0.525 24.164 4.864 0.7978 0.146 9.597 24.634 AsyncDiff[2] (stride=1) 12.54 1.31× 9.830 23.941 4.103 0.797 0.108 9.586 26.387 Ours (k=5) 7.12 2.31× 0.516 23.831 4.100 0.796 0.107 9.665 26.640

Stable Diffusion XL

- 1 Original Model 19.36 - - 33.433 - 0.810 - 8.086 -

- 2

AsyncDiff[2] (stride=1) 9.82 1.97× 1.290 33.379 2.032 0.813 0.052 8.155 27.812 xDiT-Ring[4] 14.31 1.35× 121.646 33.356 1.909 0.809 0.047 8.085 27.857 Parastep[32] 9.98 1.94× 0.032 33.340 3.350 0.810 0.112 8.091 22.917 Ours (k=5) 9.33 2.07× 0.189 33.322 1.878 0.780 0.046 8.229 27.875

Stable Diffusion 3

Table 1. Quantitative comparison of parallelism methods on the Stable Diffusion XL and Stable Diffusion 3 models. We compare our method with existing distributed inference techniques under 1- and 2-GPU. We report both the baseline latency and the corresponding acceleration ratio (Speed-Up), Communication efficiency (Comm.), and quantitative metrics assessing generation fidelity. Here, w/ G.T. denotes comparison with the ground-truth image, and w/ Orig. indicates comparison with the original (single-GPU) model output.

##### 5. Experiments

###### 5.1. Experimental Setup

Models. We evaluate our proposed hybrid parallelism framework on two representative diffusion backbones: Stable Diffusion XL(SDXL)[23] and Stable Diffusion 3.0(SD3), a DiT-based flow matching model[3]. SDXL represents U-Net–based latent diffusion models[25], while SD3 reflects the transformer-based paradigm, demonstrating the generality of our approach.

Datasets. All experiments are conducted on the MSCOCO Captions 2014 benchmark[15], using 5,000 validation prompts for text-to-image generation. Generated images are compared against both the ground-truth samples and the single-GPU original model outputs.

Metrics. We evaluate inference efficiency and generative quality. Latency and speed-up ratio measure acceleration. For quality, we report FID (Fr´echet Inception Distance)[6], LPIPS (Learned Perceptual Image Patch Similarity)[41], and PSNR (Peak Signal-to-Noise Ratio). Lower FID/LPIPS and higher PSNR indicate better generation quality.

For implementation details, please refer to Appendix G.

###### 5.2. Main Results

Quantitative Results. Table 1 reports a quantitative comparison across SDXL and SD3 pre-train diffusion models. On SDXL, our method achieves a 2.31× acceleration over the single-GPU baseline while slightly improving image fidelity. Compared to prior distributed inference methods such as DistriFusion,[12] and AsyncDiff,[2], our proposed method attains the best speed–quality trade-off with minimal communication overhead. Notably, our communication cost is reduced by 19.6× compared to AsyncDiff, due to adaptive parallelism switching that dynamically determines optimal parallel intervals to minimize communication cost.

FID ↓ (w/ Orig.) Original Model 16.49 - -

Methods Latency (s) ↓ Speed-Up ↑

Full Condition-Based Partitioning 9.24 1.78× 3.623 Ours (Hybrid Parallelism) 7.12 2.31× 4.100

Table 2. Ablation on hybrid parallel components. All experiments are conducted on the SDXL model at 1024×1024 resolution, comparing the single-GPU baseline, full condition-based partitioning, and our hybrid parallelism framework.

For SD3, a DiT-based flow-matching model, our approach not only surpasses earlier distributed frameworks such as DistriFusion and AsyncDiff, but also consistently outperforms more recent baselines, xDiT-Ring and Parastep. It achieves a 2.07× speed-up with negligible communication cost while maintaining comparable or superior generation quality. These results emphasize our method’s strong generality across both U-Net and DiT architectures, achieving generation efficiency.

Qualitative Results. Figure 5 presents qualitative comparisons among distributed inference methods. While DistriFusion and AsyncDiff exhibit boundary artifacts or spatial inconsistency, our method preserves global coherence and fine-grained details similar to the original model. These results confirm that the proposed hybrid parallelism framework maintains high visual fidelity while achieving substantial acceleration. Further results are shown in Appendix I.

###### 5.3. Ablation Study

Ablation on Hybrid Parallel Components. Table 2 analyzes the contribution of each hybrid parallel component. We compare three settings: (1) the original singleGPU model, (2) full condition-based partitioning applied to all denoising steps, and (3) our proposed hybrid parallelism combining both condition-based partitioning and adaptive parallelism switching. Condition-based partition-

###### Original DistriFusion AsyncDiff Ours

Speed-Up: 1.23x FID: 4.86 Speed-Up: 1.32x FID: 4.13 Speed-Up: 2.32x FID: 4.10

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Prompt: “A white tray with a vase holding flowers over a cat.”

- Figure 5. Qualitative results of the main experiments. We compare 1024×1024 image generations from the SDXL model. Our method achieves the best acceleration and FID performance, while producing visuals most similar to the original.

| | | | | | |
|---|---|---|---|---|---|
| |Naïve diffu|AsyncDiff<br><br>sion|Ours|(k = 5)| |
| |Distri|Fusion|O|urs (k = 10)| |
| | | | |Ours (k =|20)|
| | | | |Ours|(k = 30)|
| | | | | | |

Speed-Up ↑

4.0

6.0

8.0

10.0

FID ↓

1.0x 2.0x 3.0x

0.0

1.5x 2.5x

- Figure 6. Visualization of speed–quality trade-off across different parallelism intervals k. Smaller k values preserve higher fidelity, whereas larger k achieve greater acceleration. Our method consistently dominates prior works across the trade-off frontier. All experiments were conducted on 2 GPUs.

10

1.25x 1.35x

1.54x

Latency(s)

1.80x

2.36x

2.72x

- 5

2.5

7.5

20

10

5

15

40

20

10

30

1024 x 1024 2048 x 2048 2560 x 2560

| |
|---|

Original

| |
|---|

DistriFusion

| |
|---|

AsyncDiff

| |
|---|

Ours (k=5)

8.12

3.44

4.50

2.99

14.2

13.2

11.6

26.5

25.1

21.5

1.31x 1.38x

1.62x

17.8 34.7

Figure 7. Comparison of high-resolutions tasks. We compare different parallel inference methods on the SDXL model using NVIDIA H200 GPUs across 1024×1024, 2048×2048, and 2560×2560 high-resolutions.

and fidelity. Quantitative results are summarized in Appendix H, and qualitative comparisons across different k values are provided in Appendix J.

High-Resolution Generation. As shown in Figure 7, our method consistently achieves superior acceleration over existing distributed inference frameworks across increasing resolutions. On the SDXL model using NVIDIA H200 GPUs, our hybrid parallelism attains up to 2.72× speed-up at 1024×1024, 1.54× speed-up at 2048×2048, and 1.62× speed-up at 2560×2560, demonstrating strong scalability for high-resolution image generation.

- 6. Conclusion

ing achieves a 1.78× speed-up while maintaining image quality, whereas our hybrid parallelism further improves efficiency to 2.31× with comparable quality. This demonstrates that the addition of the pipeline component maximizes generation acceleration while minimizing quality degradation. Consequently, the proposed framework effectively integrates the advantages of condition-based partitioning and adaptive parallelism switching.

In this paper, we introduced a hybrid parallelism framework for diffusion inference that integrates condition-based partitioning with adaptive parallelism switching. Guided by the denoising discrepancy criterion, the method adaptively switches between parallelism modes to minimize redundant communication. It achieves 2.31× and 2.07× latency reductions on SDXL and SD3, respectively, while preserving fidelity. We also generalize across U-Net and DiT architectures, providing a unified parallelism paradigm for scalable multi-GPU diffusion inference.

###### 5.4. Sensitivity Analysis

Impact of Different k Values. As shown in Figure 6, the parallelism interval k clearly reveals a speed–quality tradeoff: smaller k values preserve higher fidelity, while larger k values yield faster generation. An appropriate balance is observed at k=5, achieving both strong quality and acceleration. Moreover, the interval k can be flexibly chosen by practitioners to adjust the trade-off between efficiency

##### References

- [1] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analyticdpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In Proceedings of the International Conference on Learning Representations (ICLR),

2022. 1, 3

- [2] Zigeng Chen, Xinyin Ma, Gongfan Fang, Zhenxiong Tan, and Xinchao Wang. Asyncdiff: Parallelizing diffusion models by asynchronous denoising. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 37:95170–95197, 2024. 2, 3, 7
- [3] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the International Conference on Machine Learning (ICML), 2024. 7
- [4] Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, and Jiannan Wang. xdit: an inference engine for diffusion transformers (dits) with massive parallelism. arXiv preprint arXiv:2411.01738, 2024. 3, 4, 7
- [5] Jiarui Fang, Jinzhe Pan, Jiannan Wang, Aoyu Li, and Xibo Sun. Pipefusion: Patch-level pipeline parallelism for diffusion transformers inference. arXiv preprint arXiv:2405.14430, 2024. 3, 4
- [6] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 30, 2017. 7
- [7] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshop on Deep Generative Models and Downstream Applications, 2021. 3, 2
- [8] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 35:26565–26577, 2022. 6, 2
- [9] Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models, 2021. 1, 3
- [10] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 37:122458– 122483, 2024. 5
- [11] Muyang Li, Ji Lin, Chenlin Meng, Stefano Ermon, Song Han, and Jun-Yan Zhu. Efficient spatially sparse inference for conditional gans and diffusion models. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 35:28858–28873, 2022. 1, 3
- [12] Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Kai Li, and Song Han. Distrifusion: Distributed parallel inference for high-resolution

- diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7183–7193, 2024. 2, 3, 7
- [13] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 17535–17545, 2023. 1, 3
- [14] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 36:20662– 20678, 2023. 1, 3
- [15] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Proceedings of the European Conference on Computer Vision (ECCV), pages 740–755. Springer, 2014. 5, 7, 1
- [16] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. 3
- [17] Haozhe Liu, Wentian Zhang, Jinheng Xie, Francesco Faccio, Mengmeng Xu, Tao Xiang, Mike Zheng Shou, JuanManuel Perez-Rua, and J¨urgen Schmidhuber. Faster diffusion via temporal attention decomposition. arXiv preprint arXiv:2404.02747, 2024. 1, 3
- [18] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In Proceedings of the International Conference on Learning Representations (ICLR), 2022. 1, 3
- [19] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 35:5775–5787, 2022. 1, 3
- [20] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, pages 1–22, 2025.
- [21] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 1, 3
- [22] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15762–15772, 2024. 1, 3
- [23] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In Proceedings of the International Conference on Learning Representations (ICLR), 2024. 7
- [24] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.

- In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pages 3505–3506, 2020. 1
- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 7
- [26] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. Proceedings of the International Conference on Learning Representations (ICLR),

2022. 1, 3

- [27] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 87–103. Springer, 2024. 1, 3
- [28] Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, and Nima Anari. Parallel sampling of diffusion models. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 36:4263–4276, 2023. 1, 3
- [29] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatronlm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019. 1
- [30] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proceedings of the International Conference on Learning Representations (ICLR),

2021. 3, 4

- [31] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. Proceedings of the International Conference on Learning Representations (ICLR), 2021. 6, 2
- [32] Kunyun Wang, Bohan Li, Kai Yu, Minyi Guo, and Jieru Zhao. Communication-efficient diffusion denoising parallelization via reuse-then-predict mechanism. arXiv preprint arXiv:2505.14741, 2025. 3, 4, 7
- [33] Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al. Cache me if you can: Accelerating diffusion models through block caching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6211–6220, 2024. 3
- [34] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. Proceedings of the International Conference on Learning Representations (ICLR), 2022. 1, 3
- [35] Xingyi Yang, Daquan Zhou, Jiashi Feng, and Xinchao Wang. Diffusion probabilistic model made slim. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22552–22562, 2023. 1, 3
- [36] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6613–6623,

2024. 1, 3

- [37] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image superresolution by residual shifting. Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 36:13294–13307, 2023. 1, 3
- [38] Dingkun Zhang, Sijia Li, Chen Chen, Qingsong Xie, and Haonan Lu. Laptop-diff: Layer pruning and normalized distillation for compressing diffusion models. arXiv preprint arXiv:2404.11098, 2024. 1, 3
- [39] Jiale Zhang, Yulun Zhang, Jinjin Gu, Jiahua Dong, Linghe Kong, and Xiaokang Yang. Xformer: Hybrid x-shaped transformer for image denoising. Proceedings of the International Conference on Learning Representations (ICLR), 2024. 1, 3
- [40] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. 1, 3
- [41] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595, 2018. 7

## Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling

### Supplementary Material

[Figure 158]

rel−MAE (𝜖 ,𝜖 )value

Image Quality ↑

Model General. ↑

High-res Synth. ↑

Comm.

MAE mean ±2σ

Methods Speed-Up ↑

Efficiency ↑ Distrifusion 2.5 3.5 2.5 3.3 5.0

AsyncDiff 3.0 4.5 5.0 3.5 1.0 Ours 4.7 4.5 5.0 4.4 5.0

- Table 3. Quantitative metrics comparison across five evaluation aspects. Scores are normalized to a 5-point scale. Higher values (↑) indicate better performance.

Timestep 𝑡

Figure 8. Empirical visualization of denoising discrepancy curve.

##### A. Evaluation of Hybrid Parallelism

##### B. Empirical Visualization of Denoising Discrepancy

Evaluation Protocol. All scores are computed based on a 5-point scale unified min–max scaling scheme, where the normalized values are re-centered around an average score of 3. Specifically, each metric is assessed as follows:

Figure 8 illustrates the average denoising discrepancy (rel-MAEt(ϵc,ϵu)) value measured during the denoising process based on 5,000 prompts from the MS-COCO 2014[15] validation set. The shaded region represents the ±2σ range, and the denoising model used is Stable Diffusion XL. The red dot denotes τcap = argmin t

- • Speed-Up. We measure the relative acceleration ratio with respect to the SDXL baseline latency in Table 1. The measured latencies are 13.53secs for DistriFusion, 12.54secs for AsyncDiff, and 7.12secs for our method.
- • Image Quality. We evaluate image quality using FID scores reported in Table 1 from the main results of SDXL. The reported FID values are 4.864 for DistriFusion, 4.103 for AsyncDiff, and 4.100 for our method.
- • Model Generality. We assign scores based on architecture compatibility. Each model receives 2.5 points for supporting U-Net and an additional 2.5 points for DiT support, resulting in scores of 2.5 for DistriFusion, 5 for AsyncDiff, and 5 for Ours.
- • High-resolution Synthesis. The score reflects both highresolution generation capability and inference latency. According to the results in Section 5.4 High-Resolution Generation, all three methods successfully generate three target resolutions. The corresponding average latencies are 14.73secs for DistriFusion, 14.27secs for AsyncDiff, and 11.99secs for Ours.
- • Communication Efficiency. We evaluate the communication efficiency based on the measured inter-GPU data transfer communication volume in the SDXL multi-GPU setting reported in Table 1 from the main results. The measured communication volumes are 0.525 GB for DistriFusion, 9.830 GB for AsyncDiff, and 0.516 GB for our method.

rel-MAEt(ϵc,ϵu), which is employed as a safetycap in the main method.

##### C. Adaptive Parallelism Switching Algorithm

Algorithm 1 Adaptive Parallelism Switching

via Denoising Discrepancy

Require: latent noise xt, prompt c, steps T, window L,

slope threshold g, safety-cap τcap, interval k

- 1: τ1,τ2 ← ∅
- 2: for t = T,T −1,...,1 do
- 3: ϵc,ϵu ← ϵθ(xt,c,t), ϵθ(xt,t)
- 4: Mt ←

Ex,ϵ∥ϵc − ϵu∥1 Ex,ϵ∥ϵu∥1

▷ rel-MAEt(ϵc,ϵu)

- 5: Gt = M

t−Mt−L L

- 6: if τ1 = ∅ and t > L and 0 ≤ Gt < g then
- 7: τ1 ← min(t, τcap); τ2 ← τ1 + k
- 8: Denoise:
- 9: if t ≥ τ1 then
- 10: WARM-UP
- 11: else if t > τ2 then
- 12: PARALLELISM
- 13: else
- 14: FULLY-CONNECTING
- 15: end if
- 16: xt−1 ← STEP DENOISE(xt,ϵc,ϵu,t)
- 17: end for
- 18: return x0, (τ1,τ2)

##### D. Derivation of Score-Based Interpretation of Denoising Discrepancy

The denoising discrepancy(rel-MAEt(ϵc,ϵu)) criterion in Eq. (4) can be theoretically derived from the score decomposition of diffusion models. Following the ϵparameterization of score-based generative modeling [8, 31], the preconditioned score can be expressed as

ϵθ(xt,t) σt

sθ(xt,t) ≈ −

, (5)

where σt denotes the noise standard deviation at timestep t. According to Bayes’ rule, the conditional score function can be decomposed as

sc(xt,t) = su(xt,t) + ∇xt

log p(c|xt), (6)

where su(xt,t) is the unconditional data score, and ∇xt

log p(c|xt) denotes the conditional information flow [7]. Substituting Eq. ((5)) into Eq. ((6)) yields

ϵc(xt,t) − ϵu(xt,t) ∝ σt ∇xt

log p(c|xt), (7)

which implies that the difference between conditional and unconditional denoiser outputs corresponds to the conditional gradient scaled by σt. Therefore, the rel-MAE at each timestep t can be approximated as

log p(c|xt)∥1 ∥su(xt,t)∥1

rel-MAEt = ∥ϵc − ϵu∥1 ∥ϵu∥1

∥∇xt

. (8)

≈

This formulation reveals that rel-MAEt(ϵc,ϵu) quantifies the relative magnitude between the conditional information and the unconditional data prior—forming the theoretical basis for the main method equation (Eq. (4)).

##### E. Robustness of Determine τ1 under Stochastic Denoising Noise

Diffusion inference is a stochastic denoising process; predicted noises ϵθ(xt) are subject to random sampling. Consequently, the observed {Mt} fluctuates slightly, and Gt≈0 may appear prematurely. To ensure robust detection, we define a finite-difference slope by

Mt − Mt−L L

, (9)

Gt =

which smooths out stochastic perturbations across L timesteps. The stability of Gt can be theoretically justified by Hoeffding’s inequality:

2Lδ2 (b − a)2

. (10)

Pr(|Gt − E[Gt]| > δ) ≤ 2exp −

Here, L denotes the window length used to compute the moving-average slope, δ represents the allowable deviation

from the expected slope E[Gt], and a,b correspond to the minimum and maximum possible range of the observed rel-MAEt(ϵc,ϵu) values, typically normalized within [0,1].

As L increases, the variance of the estimated slope decreases, and the probability of false detection decreases exponentially. showing that larger L exponentially reduces false-alarm probability.

Empirically, L and gslope, which are also established in our experiments, lie within a stable regime due to strong autocorrelation of rel-MAEt(ϵc,ϵu) sequences. Thus, τ1 can be reliably detected as the earliest timestep satisfying 0≤Gt<gslope and t≤τcap.

##### F. Extensibility to Many GPU Configurations Structures

Figure 9 presents two extensibility structures that scale the proposed hybrid parallelism framework from the baseline 2GPUs setup to many GPU configurations.

The first structure, shown in Figure 9a, demonstrates the batch-level extension under an N GPUs configuration. In this scheme, each pair of GPUs collaboratively denoises a single sample while following the three stages hybrid parallelism framework. As a result, the system can generate N/2 samples concurrently with N GPUs, enabling near-linear throughput scaling when multiple samples are produced.

The second structure, shown in Figure 9b, demonstrates the layer-wise pipeline extension on a 4GPUs configuration. Here, the denoising network is partitioned into multiple layer-wise segments distributed across devices, allowing the hybrid parallelism strategy to be applied to singlesample generation. While this configuration may exhibit slightly reduced acceleration efficiency and minor quality degradation compared to the batch-level extension, it provides a fine-grained pipeline scheduling mechanism. Importantly, the same structural principles naturally generalize beyond the 4GPUs example to arbitrary N GPUs configurations, demonstrating the flexibility and scalability of the proposed framework.

###### (2) Parallelism Stage (𝜏 ,𝜏 ) (3) Fully-Connecting Stage [𝜏 ,0]

###### (1) Warm-Up Stage [𝑇,𝜏 ]

Unconditional

Unconditional

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

|GPU 1|
|---|

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

|[Figure 177]| |
|---|---|
| | |

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

|[Figure 185]|
|---|

###### Sample 1

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

𝑥

𝑥

𝑥

|[Figure 198]|
|---|

[Figure 199]

⋯

###### ⋯

⋯

Conditional

Conditional

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

𝑥

𝑥

|GPU 2|
|---|

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

|[Figure 217]| |
|---|---|
| | |

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

|[Figure 225]|
|---|

|[Figure 226]|
|---|

|[Figure 227]|
|---|

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

[Figure 238]

[Figure 239]

[Figure 240]

𝑥 ,𝑐

𝑥 ,𝑐

𝑥 ,𝑐 𝑥 ,𝑐

𝑥 ,𝑐

###### (2) Parallelism Stage (𝜏 ,𝜏 ) (3) Fully-Connecting Stage [𝜏 ,0]

###### (1) Warm-Up Stage [𝑇,𝜏 ]

Unconditional

Unconditional

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

|GPU 3|
|---|

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

|[Figure 259]| |
|---|---|
| | |

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

|[Figure 267]|
|---|

###### Sample 2

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

𝑥

𝑥

𝑥

|[Figure 280]|
|---|

[Figure 281]

⋯

⋯

###### ⋯

Conditional

Conditional

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

[Figure 293]

𝑥

𝑥

|GPU 4|
|---|

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

|[Figure 299]| |
|---|---|
| | |

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

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

𝑥 ,𝑐

𝑥 ,𝑐

𝑥 ,𝑐 𝑥 ,𝑐

𝑥 ,𝑐

⋮

###### (2) Parallelism Stage (𝜏 ,𝜏 ) (3) Fully-Connecting Stage [𝜏 ,0]

###### (1) Warm-Up Stage [𝑇,𝜏 ]

Unconditional

Unconditional

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

###### Sample 𝐍𝟐

|GPU N-1|
|---|

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

|[Figure 341]| |
|---|---|
| | |

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

|[Figure 349]|
|---|

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

𝑥

𝑥

𝑥

|[Figure 362]|
|---|

[Figure 363]

⋯

###### ⋯

⋯

Conditional

Conditional

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

𝑥

𝑥

|GPU N|
|---|

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

|[Figure 382]| |
|---|---|
| | |

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

|[Figure 389]|
|---|

|[Figure 390]|
|---|

|[Figure 391]|
|---|

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

𝑥 ,𝑐

𝑥 ,𝑐

𝑥 ,𝑐

𝑥 ,𝑐 𝑥 ,𝑐

|Ordinal Communication Conditional Communication Unconditional Communication 𝑐 Conditional Information|
|---|

(a) Batch-level extension under N GPUs configuration.

|(3) Fully-Connecting Stage [𝜏 ,0]<br><br>𝑥 ,𝑐<br><br>𝑥<br><br>[Figure 405]<br><br>[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]<br><br>[Figure 410]<br><br>[Figure 411]<br><br>[Figure 412]<br><br>[Figure 413]<br><br>[Figure 414]<br><br>[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]<br><br>[Figure 418]<br><br>[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]<br><br>[Figure 422]<br><br>[Figure 423]<br><br>[Figure 424]<br><br>[Figure 425]<br><br>[Figure 426]<br><br>[Figure 427]<br><br>[Figure 428]<br><br>𝑥<br><br>|[Figure 429]|
|---|
<br><br>|[Figure 430]| |
|---|---|
| | |
<br><br>Conditional<br><br>Unconditional<br><br>|[Figure 431]| |
|---|---|
| | |
<br><br>⋯|
|---|

###### (2) Parallelism Stage (𝜏 ,𝜏 )

###### (1) Warm-Up Stage [𝑇,𝜏 ]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Unconditional

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

|[Figure 449]|
|---|

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

𝑥

𝑥

[Figure 458]

⋯

###### ⋯

Conditional

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

𝑥

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

|[Figure 471]|
|---|

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

𝑥 ,𝑐

𝑥 ,𝑐

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

|[Figure 485]|
|---|

|[Figure 486]|
|---|

𝑥 ,𝑐 𝑥 ,𝑐

|Ordinal Communication Conditional Communication Unconditional Communication 𝑐 Conditional Information|
|---|

(b) Layer-wise pipeline extension on a 4GPUs configuration.

- Figure 9. Extensibility to many GPU configurations structures. This figure illustrates two strategies for scaling the proposed hybrid parallelism framework to larger GPU configurations. These structures demonstrate how the proposed framework naturally generalizes from the 2GPUs setting to both batch-level and layer-wise many GPU configurations.

FID ↓ (w/ Orig.)

Parallelism Interval k Latency (s) ↓ Speed-Up ↑

k=5 7.12 2.31× 4.100 k=10 6.89 2.39× 5.942 k=20 6.44 2.56× 7.966 k=30 5.94 2.78× 9.191

- Table 4. Effect of speed-quality trade-off across different parallelism intervals k. All experiments are conducted on the SDXL model at 1024×1024 resolution with various parallelism intervals.

##### G. Implementation Details

All experiments adopt the DDIM scheduler[30] with T = 50 timesteps and generate images at a resolution of 1024 × 1024. Experiments are performed on NVIDIA GeForce 3090 GPUs (24GB each), connected via PCIe Gen3. The adaptive switching parameters are set as follows: for SDXL, we use L = 12, gslope = 0.4 × 10−3, k = 5, and τcap = 15; for SD3, we set L = 15, gslope = 0.1 × 10−3, k = 5, and τcap = 40.

##### H. Quantitative Results on the Parallelism Interval k

Table 4 summarizes the numerical values corresponding to the speed–quality trade-off illustrated in Figure 6. As described in the Section 5.4, smaller parallelism interval k preserve higher fidelity, whereas larger k values yield powerful acceleration. The table provides concrete measurements that reflect this trade-off, confirming the same trend observed in the pareto frontier visualization.

##### I. Additional Qualitative Results

###### Original DistriFusion AsyncDiff Ours

Speed-Up: 1.23x FID: 4.82 Speed-Up: 1.32x FID: 4.21 Speed-Up: 2.32x FID: 4.14

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

Prompt: “Two cats sitting on top of a pair of shoes.”

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

Prompt: “A white tray with a vase holding flowers over a cat.”

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

Prompt: “A bedroom with a bed next to a closet.”

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

Prompt: “A silver fork and some sliced carrots and fish.”

- Figure 10. Additional qualitative results of the main experiments. We compare 1024×1024 image generations from the SDXL model. Our method achieves the best acceleration and FID performance, while producing visuals most similar to the original.

##### J. Qualitative Comparion Results via Different k

𝑘 = 5 𝑘 = 10 𝑘 = 20 𝑘 =30

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

Prompt: “Astronaut in a jungle, cold color palette, muted colors, detailed, 8k”

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

Prompt: “A photo of running horse.”

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

Prompt: “A kid wearing headphones and using a laptop.”

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

Prompt: “A young girl is holding a small cat.”

- Figure 11. Additional qualitative comparisons across different k values. We compare 1024×1024 image generations from the SDXL model across various parallelism intervals. Smaller k values preserve higher visual fidelity, whereas larger k gradually reduce local detail due to the extended parallelism window. Although the overall appearance remains similar, fine-grained conditional attributes become subtly blurred as k increases.

