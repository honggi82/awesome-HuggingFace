## Dual-Expert Consistency Model for Efficient and High-Quality Video Generation

Zhengyao Lv2,3∗ Chenyang Si1‡∗ Tianlin Pan1,4 Zhaoxi Chen5 Kwan-Yee K. Wong2 Yu Qiao3 Ziwei Liu5†

# arXiv:2506.03123v2[cs.CV]6Aug2025

1Nanjing University 2The University of Hong Kong 3Shanghai Artificial Intelligence Laboratory 4University of Chinese Academy of Sciences 5S-Lab, Nanyang Technological University

cszy98@gmail.com chenyang.si@nju.edu.cn pantianlin23@mails.ucas.ac.cn zhaoxi001@ntu.edu.sg kykwong@cs.hku.hk yu.qiao@siat.ac.cn ziwei.liu@ntu.edu.sg

[Figure 1]

HunyuanDCM-4PCM-4LCM-4

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Figure 1. Comparison of visual results between our DCM (4 steps), the original HunyuanVideo, and other competing methods (left). Comparison of latency and VBench score across different methods (right). Latency is measured on two A100 GPUs under the video synthesis configuration of 129 frames at 1280 × 720 resolution.

### Abstract

ing semantic layout and motion, while a detail expert specializes in fine detail refinement. Furthermore, we introduce Temporal Coherence Loss to improve motion consistency for the semantic expert and apply GAN and Feature Matching Loss to enhance the synthesis quality of the detail expert. Our approach achieves state-of-the-art visual quality with significantly reduced sampling steps, demonstrating the effectiveness of expert specialization in video diffusion model distillation. Our code and models are available at https://github.com/Vchitect/DCM.

Diffusion Models have achieved remarkable results in video synthesis but require iterative denoising steps, leading to substantial computational overhead. Consistency Models have made significant progress in accelerating diffusion models. However, directly applying them to video diffusion models often results in severe degradation of temporal consistency and appearance details. In this paper, by analyzing the training dynamics of Consistency Models, we identify a key conflicting learning dynamics during the distillation process: there is a significant discrepancy in the optimization gradients and loss contributions across different timesteps. This discrepancy prevents the distilled student model from achieving an optimal state, leading to compromised temporal consistency and degraded appearance details. To address this issue, we propose a parameter-efficient Dual-Expert Consistency Model (DCM), where a semantic expert focuses on learn-

### 1. Introduction

Diffusion Models [8] have achieved remarkable progress in image and video synthesis [45, 48, 63]. However, they require multiple iterations to model the probability flow Ordinary Differential Equation (ODE) [52] and rely on increasingly large denoising networks, resulting in substantial computational overhead that limits their practicality in real-world applications.

To mitigate this constraint, Consistency Distillation [54]

*Equal Contribution. ‡ Project Leader †Corresponding Author.

has emerged as an efficient knowledge distillation framework to reduce the sampling timesteps. It leverages a pretrained diffusion model as the teacher and trains a student model to directly map any point along the ODE trajectory to the same solution, thereby ensuring the selfconsistency property. Despite enabling few-step sampling, it often struggles with visual quality, particularly in challenging video synthesis, leading to distorted layouts, unnatural motion, and degraded details.

To ground this inherent issue, we analyzed the sampling dynamics of video diffusion models, as shown in Fig. 2 (a). Our key observations are that the differences between adjacent steps are substantial in the early stages of sampling, whereas the changes become more gradual in the later stages. This discrepancy arises because the early steps primarily focus on synthesizing semantic layout and motion, while the later steps emphasize refining fine details. These findings suggest that the student model may learn different patterns and exhibit distinct learning dynamics when trained on high-noise and low-noise samples. We visualized the magnitude and gradient of the consistency loss during the distillation process and observed significant differences between high and low noise levels, as shown in Fig. 2 (b). This variation indicates that jointly distilling a single student model to capture both semantic layout and fine-detail synthesis may introduce optimization interference, potentially leading to suboptimal results.

To validate this assumption, we trained two expert denoisers. We first divide the ODE trajectory of the pretrained model into two phases: the semantic synthesis phase and the detail refinement phase. We then train two distinct student expert denoisers, each responsible for fitting one of these sub-trajectories. During inference, we dynamically select the corresponding expert denoiser based on the noise level of samples to predict the next position in the ODE trajectory. The results demonstrate that the combination of the two student expert denoisers achieves better performance, thereby confirming the validity of our hypothesis.

However, this straightforward baseline involves training two student models which is not efficient enough. To further enhance parameter efficiency, we analyze the parameter differences between the two expert denoisers and identify that the primary differences lie in 1) embedding layers where the input parameters include timesteps, and 2) the linear layers within the attention layers. Based on this insight, we propose a parameter-efficient Dual-Expert Consistency Model (DCM). Specifically, we first train a semantic expert denoiser on the semantic synthesis trajectory. We then freeze this expert and introduce a new set of timestepdependent layers, incorporating a LoRA [11] into the linear layers of the attention blocks. Subsequently, we fine-tune these newly added layers on the detail refinement trajectory. In this manner, we decouple the optimization of the two ex-

pert denoisers with minimal additional parameters and computational cost, achieving visual results comparable to those obtained with two separate experts.

Given the differing training dynamics of the semantic and detail expert denoisers, we introduce distinct optimization objectives beyond the original consistency loss. To enforce temporal coherence in the semantic expert denoiser, we introduce a Temporal Coherence Loss, which guides it to capture motion variations across frames. To enhance the fine-grained content synthesized by the detail expert denoiser, we introduce a generative adversarial (GAN) [4] loss and incorporate a Feature Matching loss. Specifically, we alternately optimize the student model and the discriminator in the feature space, encouraging the generator to synthesize visual content that aligns with the output distribution of the teacher model. The Feature Matching term enhances supervision over intermediate features, thereby stabilizing the GAN training.

Our proposed DCM accelerates sampling while preserving both semantic and detail quality, as shown in Fig. 1. In summary, our contributions are as follows:

- • We analyze the training dynamics of Consistency Models and identify a key conflict in the distillation process: discrepancies in loss contributions and optimization gradients across noise levels hinder optimal learning, leading to suboptimal visual quality.
- • We propose a parameter-efficient Dual-Expert Consistency Model that decouples the expert denoisers distillation, mitigating the conflict and improving visual quality with minimal parameter and computational cost.
- • To enhance visual quality, we introduce Temporal Coherence Loss for the semantic expert and GAN loss with Feature Matching term for the details expert, improving both temporal consistency and detail quality.

### 2. Related Work

#### 2.1. Diffusion Models For Video Synthesis

Video Diffusion Models have witnessed rapid advancements with diffusion models [2, 8, 9], driving a wide range of visual content generation applications[12, 13, 70–72]. Building on the Diffusion Transformer (DiT) [45] pretraining, a notable breakthrough is the development of highfidelity video diffusion models [10, 20–22, 42, 44, 55, 63]. However, scaling these models for long videos incurs significant training and inference costs. LTX-Video [7] designed Video-VAE that achieves a high compression ratio for efficient self-attention. Pyramid flow [15] introduced a unified pyramidal flow matching algorithm for efficient video generative modeling. The acceleration method based on sparse attention [68, 69] and feature cache [41, 73] has also improved inference efficiency. Moreover, while efficient fast diffusion samplers [17, 28, 32, 33, 52] reduce inference steps, further reduction often severely degrades per-

t=T t = 0

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Frame0Frame15Frame25

(a) Visualization of the video synthesis sampling process

(b) The trend of training loss and gradient norm

- Figure 2. Visualization of the video synthesis process and the trend of loss variation. (a) In the early stages of sampling, the results change significantly and rapidly, whereas in the later stages, the changes become gradual and smooth. (b) During distillation, the loss and gradient norm of the student model exhibit significant differences between samples with high and low noise levels.

by score distillation [46]. Notably, DMD [65] aligns the one-step generator with the distribution of teacher model by minimizing an approximate KL divergence, whose gradient is the difference between the target and synthetic distribution score functions. Recently works [19, 25, 47] have also tried to integrate the advantages of trajectory-preserving and distribution-matching methods. Hyper-SD [47] introduced the trajectory-segmented consistency distillation and used DMD [65] for one-step generation enhancement.

formance. Diffusion distillation offers a promising way to reduce sampling steps while maintaining visual quality.

- 2.2. Diffusion Model Distillation Diffusion distillation [37] aims to distill knowledge from pre-trained diffusion models to student models, reducing inference cost. Prior works can be generally classified into two categories based on their distillation mechanisms. Trajectory-preserving distillation methods exploit the fact that diffusion models learn an Ordinary Differential Equation (ODE) trajectory and aim to predict the exact teacher output in fewer steps. Among the earliest studies on diffusion distillation, Luhman et al. [34] and DSNO [74] proposed training the student model using noise-image pairs precomputed by the teacher model with an ODE solver. Progressive distillation [43, 49] reduces the final number of sampling steps by iteratively applying the distillation process to halve the number of sampling steps of previous model. Instaflow [29, 30] progressively learns straighter flows, enabling accurate one-step predictions over larger distances. Consistency models [27, 31, 35, 36, 53, 54, 75], BOOT [5] and TRACT [1] learn to map samples along the ODE trajectory to another point to achieve selfconsistency. The consistency trajectory model [18] was designed to mitigate discretization inaccuracies and accumulated estimation errors in the multistep consistency model sampling. PCMs [57] phase the ODE trajectory into several sub-trajectories and only enforce the self-consistency property on each sub-trajectory, thus alleviating the limitations of CMs. TCM [23] generalizes consistency training to the truncated time range to prevents the truncated-time training from collapsing to a trivial solution. Trajectory-preserving distillation enables stable optimization but can degrade visual quality, leading to blurriness or distortions when sampling with fewer steps. Distribution-matching distillation methods bypass the ODE trajectory and aim to train the student model to generate samples whose distribution aligns with that of the teacher diffusion model. Some methods reduce the distribution gap between the student and teacher models through adversarial training [3, 16, 26, 40, 50, 51, 61, 62]. Other methods [38, 39, 64–66, 76] achieve diffusion distillation

Previous works have primarily focused on distilling image synthesis diffusion models, with some efforts [24, 58, 67] extending to the distillation of small-scale video synthesis models [6, 59]. However, these methods are limited to synthesizing low-resolution and short-sequence videos. Seaweed-APT [26] proposed adversarial posttuning against real data following diffusion pre-training for one-step high-resolution 2-second duration video generation. A recent work [66] extended DMD [65] for video synthesis in four sampling steps. Due to the inherent complexity of video synthesis and the increasing model scale, research on diffusion distillation for video synthesis remains limited, and its performance is yet to be fully explored.

### 3. Methodology

#### 3.1. Preliminary

Diffusion Model is a generative framework with a forward and reverse process.

In the forward process, noise is progressively added to clean data x0 ∼ pdata(x0), degrading the signal:

q(xt|x0) = N(xt;√αtx0,√1 − αtI), (1) where {αt}Tt=1 controls the noise schedule.

The reverse process, typically parameterized by a UNet or transformer ϵθ, is trained to predict the noise:

LDM = Ex,ϵ∼N(0,1),t ||ϵ − ϵθ(xt,t)||22 . (2)

During inference, a clean sample x0 can be recovered through iterative denoising:

p(xt−1|xt) = N(xt−1;µθ(xt,t),Σθ(xt,t)), (3)

|78<br>79<br>80<br>81<br>82<br>83<br>84<br><br><br>VBench for Variants<br><br>VCM SemE+VCM VCM+DetE SemE+DetE<br><br>|
|---|

###### Vanilla Consistency Model

Semantic Expert Denoiser

Details Expert Denoiser

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

Semantic & Details Learning Semantic Learning Details Learning Mixed-noise Level Data

High-noise Level Data Low-noise Level Data

(a) Variants trained on data with different noise levels (b) VBench metric for different variants

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

|[Figure 46]<br><br>[Figure 47]|
|---|

|[Figure 48]<br><br>[Figure 49]|
|---|

|[Figure 50]<br><br>[Figure 51]|
|---|

|[Figure 52]<br><br>[Figure 53]|
|---|

|VCM|VCM + DetE|SemE + VCM|SemE + DetE|
|---|---|---|---|

(c) Example visual results of different variants

- Figure 3. Comparison of the visual quality of denoiser variants trained at different noise level samples. By optimizing two expert denoisers to decouple the distillation process into semantic learning and detail learning, and combining them during inference, we achieve the best quantitative and qualitative visual results.

where µθ and Σθ are learned parameters.

establish the semantic layout and motion of the video, while in the later stages, the model gradually refines the details with smooth adjustments. These findings imply that the student model may learn different patterns with distinct training dynamics when distilled with high-noise and low-noise samples. By visualizing the trends of consistency loss and gradient norm during distillation, we find significant differences in the student model when distilled with high- and low-noise samples, as shown in Fig. 2 (b). This suggests that jointly optimizing a student model for both semantic and fine-grained details synthesis may introduce inefficient optimization, constraining its fitting capacity and leading to suboptimal performance.

Consistency Distillation utilizes a pre-trained model ϵθ as the teacher FT to distill its knowledge into a student model FS initialized with ϵθ, allowing for faster sampling with fewer steps [54]. Specifically, consistency distillation trains the student model FS to directly map any point xt

on the solution trajectory of the ODE solver Φ to its endpoint xt

n

. The learning objective can be formulated as:

end

LCD = Ex,t

###### ,tn,c),tend)− Φ(xˆt

n||Φ(xt

###### ,FS(xt

n

n

(4)

###### ,FS−(xˆt

,tn−1,c),tend)||22.

n−1

n−1

Here, FS− is the exponential moving average (EMA) of FS and xˆt

is the next point on the ODE solution trajectory computed by the teacher model FT:

To validate our hypothesis, we conducted an experiment on the HunyuanVideo [20] text-to-video diffusion model. Specifically, we divided the ODE solution trajectory (xN,xN−1,...,x1,x0) of the pre-trained model into two sub-trajectories, using tκ as the boundary (we set N = 50 and κ = 37 by default). The first part ({xt

n−1

###### ,tn,c),tn−1). (5)

###### xˆt

###### = Φ(xt

###### ,FT(xt

n−1

n

n

Consistency distillation has garnered widespread attention and research [35, 47, 57] due to its ease of stable training.

i}Ni=κ) primarily focuses on synthesizing the semantic layout and motion, while the second part ({xt

- 3.2. Suboptimal Solution in Consistency Distillation Although consistency distillation has demonstrated promising results in class-conditioned image synthesis and text-toimage model distillation, it falls short in more challenging video synthesis diffusion models, where issues arise such as distorted layouts, unnatural motion, and degraded details.

j}κj=0) emphasizes semantic refinement and high-quality detail generation. As shown in Fig 3 (a), we optimized two distinct student models, semantic expert denoiser FSemE and details expert denoiser FDetE, to fit each sub-trajectory:

Due to the limited capacity of the student model, consistency distillation struggles to address these issues simultaneously. By tracking the video synthesis process, we find that in the early stages of sampling, the sampling results vary significantly and rapidly, whereas in the later stages, the transitions become more gradual and smooth, as shown in Fig. 2 (a). In the early stages of sampling, rapid changes

LSemE = Ex,t

m∈[tκ,tN]||Φ(xt

###### ,tm,c),tκ)− Φ(xˆt

###### ,FSemE(xt

m

m

###### ,FSemE− (xˆt

,tm−1,c),tκ)||22,

m−1

m−1

(6) LDetE = Ex,t

n∈[t0,tκ]||Φ(xt

###### ,tn,c),t0)− Φ(xˆt

###### ,FDetE(xt

n

n

(7)

###### ,FDetE− (xˆt

,tn−1,c),t0)||22.

n−1

n−1

Expert denoiser FSemE is optimized to synthesize coherent

|Trainable<br><br>[Figure 54]|Frozen<br><br>[Figure 55]|New context_embedder<br><br>|New time_text embedder<br><br>|New final layer<br><br>|LoRA<br><br>[Figure 56]<br><br>|
|---|---|---|---|---|---|

[Figure 57]

##### Initialization DetE

SemE

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

|𝑥ො𝑡𝜅|
|---|

|𝑥ො𝑡0|
|---|

[Figure 74]

[Figure 75]

|𝑥𝑡𝑛| |
|---|---|
| | |

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

𝑥t𝑚

[Figure 83]

|𝐿𝐶𝐷 𝐿𝐹𝑀<br><br>𝐿𝐺𝐴𝑁|
|---|

[Figure 84]

[Figure 85]

|𝐿𝐶𝐷 𝐿𝑇𝐶|
|---|

Teacher SemE Teacher

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

DetE

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

|x෤𝑡𝜅|
|---|

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

|x෤𝑡0|
|---|

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

|𝒙𝒕𝑵|
|---|

|𝒙𝒕𝟎|
|---|

𝒙𝒕𝜿

Stage I: Semantic Learning Stage II: Detail Learning

- Figure 4. The training process of DCM consists of two stages. In the semantic learning stage, we train SemE on high-noise samples with consistency loss and temporal coherence loss as the learning objectives. In the detail learning stage, we initialize DetE with the weights of SemE and introduce a set of time-dependent layers and LoRA. DetE is then trained on low-noise samples, where only the newly added layers and LoRA are updated. The learning objectives in this stage include consistency loss, GAN loss, and Feature Matching loss.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

attention layers

[Figure 141]

[Figure 142]

final layer

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

time_text_embedder

x_embedder

context_embedder

[Figure 149]

[Figure 150]

𝑡 𝑥𝑡 𝑡𝑒𝑥𝑡

|0<br><br>2<br><br>4<br><br>6<br><br>8<br><br>10<br><br>12<br><br>14<br><br>16<br><br>18<br><br>20<br><br>The weight difference distribution between expert denoisers<br><br>time_text_embedder x_embedder context_embedder<br><br>final_layer attention layers other layers<br><br>|
|---|

- Figure 5. Weight difference distribution between expert denoisers. We employ the normalized L1 distance to quantify the difference between the weights.

in the embedding layers Ψ where the input parameters include timesteps and the linear layers within the attention layers Λ, as illustrated in Fig. 5.

Based on the above observations, we propose the parameter-efficient Dual-Expert distillation strategy, as illustrated in Fig. 4. Specifically, our training scheme is divided into two stages. 1) Initialize the semantic expert denoiser FSemE with the teacher model FT and optimize all its parameters on the sub-trajectory {xt

i}Ni=κ. 2) Use the optimized FSemE as the initialization of FDetE and freeze it. Then add a new set of timestep-dependent embedding layers Ψ and LoRA [11] Λ† of attention blocks. Optimize the newly added parameters (Ψ and Λ†) on its sub-trajectory {xt

semantic layouts and motion, while expert denoiser FDetE learns to generate high-quality details. During inference, we dynamically switch the expert denoiser based on the sampling stage. To assess the impact of each expert denoiser and the effectiveness of decoupled training, we evaluate four variants: a) VCM: The vanilla consistency model is used throughout the sampling process. b) SemE + VCM: SemE is applied in the first sub-trajectory, transitioning to VCM in the second. c) VCM + DetE: VCM is applied initially, followed by DetE. d) SemE + DetE: SemE and DetE are integrated for the sampling process. According to the Fig. 3 (b) and (c), SemE and DetE have respectively learned to model semantics and details, each outperforming VCM in their respective aspects. This validates our hypothesis, demonstrating that decoupled optimization is superior to jointly training a single model for both tasks.

j}κj=0. In this way, we significantly reduce the number of parameters required for decoupling the optimization process of semantic modeling and detail learning, with minimal computational cost, while maintaining the visual quality of the synthesized videos.

#### 3.4. Expert-specific Optimization Objective

In addition to the consistency objectives mentioned in Eq. 6 and Eq. 7, we also designed expert-specific optimization objectives for the semantic expert denoiser FSemE and details expert denoiser FDetE, as shown in Fig. 4.

Temporal Coherence Loss To enhance the temporal coherence in the video synthesized by the semantic expert denoiser FSemE, we introduce the Temporal Coherence Loss LTC, which emphasizes and guides the FSemE to focus on the variations and motion at corresponding positions among different frames:

- 3.3. Parameter-efficient Dual-Expert Distillation While training two expert denoisers improves video quality, it significantly increases model parameters and GPU memory consumption during inference. Through the analysis of parameter similarity between the two expert denoisers, we found that the primary differences in model parameters lie

,tm,c),tκ), xˆt

###### xt

= Φ(xt

,FSemE(xt

κ

m

m

,FSemE− (xˆt

(8)

= Φ(xˆt

,tm−1,c),tκ), LTC = ||(xt

m−1

m−1

κ

0:L−l) − (xˆt

l:L − xˆt

l:L − xt

0:L−l)||22.

κ

κ

κ

κ

Here xt

l:L represents the video latents from the l-th to the L-th channel along the temporal axis. This temporal coherence loss encourages the semantic expert denoiser FSemE to preserve consistent motion and spatial relationships between frames, ensuring more fluid videos synthesis.

κ

Generative Adversarial Loss The effectiveness of the generative adversarial (GAN) [4] loss in high-quality detail synthesis has been validated in many distribution-matching distillation methods. We introduce the GAN loss into the training of the details expert denoiser and incorporate a Feature Matching loss to stabilize the training. We first obtain xt

with the details expert denoiser FDetE, teacher model FT and ODE solver Φ:

and xˆt

0

0

###### xt

,tn,c),t0), xˆt

= Φ(xt

,FDetE(xt

0

n

n

(9)

,FDetE− (xˆt

= Φ(xˆt

,tn−1,c),t0),

n−1

n−1

0

Then we perform the forward process and apply noise to them to obtain fake sample xfake and real sample xreal with Eq. 1. We use a frozen teacher model as the feature extraction backbone Ω, extracting intermediate features with a fixed stride for calculating the GAN loss and Feature Matching loss LFM. During training, we iteratively update the parameters of FDetE and the discriminator head fD:

n ∥Ω(xfake) − Ω(xreal)∥22 , LG = Ex,t

LFM = Ex,t

[1 − fD(Ω(xfake))] + LFM, LD = Ex,t

n

[fD(Ω(xfake))] + Ex,t

[1 − fD(Ω(xreal))].

n

n

(10) The integration of the GAN loss in combination with Feature Matching loss provides a robust framework for training the details expert denoiser FDetE, stabilizing its learning process and improving the quality of detail synthesis.

### 4. Experiments

- 4.1. Experimental Setup Backbones and Baselines We utilize HunyuanVideo [20] and CogVideoX [63] as the base models for distillation. The HunyuanVideo has 13 billion parameters, and CogVideoX has 2 billion parameters. Since most prior distillation methods for diffusion models have not been applied to video synthesis, we follow the official implementations of LCM [35] and PCM [57] to implement these two methods on the selected base models as baselines for comparison. Implementation Details For HunyuanVideo, we selected trajectories with 50 Euler steps and used the default sampling parameters from diffusers. The distillation was conducted on 129-frame video sequences at a resolution of 1280 × 720 with a batch size of 6. For the semantic expert denoiser, we performed 1000 iterations of distillation with a learning rate of 1e−6, while for the details expert denoiser, we trained for 1000 iterations with a learning rate of 5e−6. For CogVideoX-2B, we selected trajectories with 50 DDIM

steps. The distillation was conducted on 29-frame video sequences at a resolution of 720 × 480 with a batch size of 4. In the first-stage fine-tuning, we distilled for approximately 1000 steps, while in the second-stage fine-tuning, we distilled for around 500 steps, both with a learning rate of 1e − 6. All experiments were conducted on 24 NVIDIA A100 80GB GPUs.

Evaluation Metrics For video quality evaluation, we use VBench [14] as our assessment metric. VBench is a comprehensive benchmark suite for video generative models, designed to align closely with human perception and offer valuable insights from multiple perspectives. Additionally, we conducted a user study to help evaluate the visual quality of the generated videos.

#### 4.2. Main Results

Quantitative Comparison Table 1 presents the quantitative comparison of our method with LCM and PCM on HunyuanVideo and CogVideoX. We generate videos using the prompts provided by VBench to evaluate their performance in terms of semantic alignment and visual quality. It can be observed that on HunyuanVideo, our method achieves a VBench score comparable to the baseline with 4step sampling, significantly outperforming LCM and PCM. In terms of efficiency, our method incurs a nearly identical latency cost per inference step compared to LCM and PCM.

Table 1. Comparison of efficiency and visual quality of different methods. The latency of HunyuanVideo was measured on two A100 GPUs, and that of CogVideoX on a single A100 GPU.

|Method<br><br>|Step|Lat.(Sec.)<br><br>|VBench|
|---|---|---|---|
| | | |Total Quality Semantic|
|Hunyuan LCM PCM Ours<br><br>|50 4 4 4|1504.5 120.68 120.89 121.52<br><br>|83.87 85.00 79.34 80.33 80.83 78.32 80.93 81.94 76.90 83.83 85.12 78.67<br><br>|
|LCM PCM Ours<br><br>|8 8 8|242.80 242.96 244.72<br><br>|81.49 82.35 78.03 81.63 82.78 77.00 83.86 85.00 79.32|

|CogVideoX LCM PCM Ours|50 4 4 4<br><br>|76.50 3.22 3.23 3.31<br><br>|80.59 81.93 75.23<br><br>78.88 80.07 74.12<br>79.09 80.33 74.14 79.99 81.35 74.56<br>|
|---|---|---|---|
|LCM PCM Ours<br><br>|8 8 8|6.42 6.42 6.58<br><br>|79.34 80.64 74.21<br><br>79.70 80.98 74.60<br>80.26 81.57 75.03<br>|

Qualitative Comparison Fig. 6 presents a comparison of the videos generated by our method and those produced by the original model, LCM and PCM. The results demonstrate that our method maintains high semantic and detail quality in synthesized videos while reducing the number of inference steps. Additional qualitative results are provided in the supplementary material for further reference.

[Figure 151]

| |
|---|

Prompt:Apanda, wearing achef's hatand ared apron, stands in a cozy kitchen filled with wooden cabinets and colorful utensils. The panda chops vegetables on a wooden cutting board, its furry paws. Next, it stirs a bubbling pot on the stove. The kitchen is warmly lit, with pots and pans hanging from a rack above. The panda then tastes the soup with a wooden spoon.

Hunyuan

steps50

| |
|---|

| |
|---|

[Figure 152]

[Figure 153]

steps8

steps4

LCM

LCM

[Figure 154]

[Figure 155]

| |
|---|

| |
|---|

steps4

steps8

PCM

PCM

| |
|---|

| |
|---|

[Figure 156]

[Figure 157]

steps4

steps8

Ours

Ours

[Figure 158]

CogVideoX

Prompt:A golden retriever with a shiny coat stands by a serene,crystalclear streamin a lush forest, itstongue lapping up the refreshing water. The sunlight filters through the dense canopy, casting dappled light on the dog's fur. As the dog drinks, droplets of water glisten on its whiskers, and its tail wags contentedly.

steps50

[Figure 159]

[Figure 160]

steps8

steps4

LCM

LCM

[Figure 161]

[Figure 162]

steps4

steps8

PCM

PCM

[Figure 163]

[Figure 164]

steps8

steps4

Ours

Ours

Figure 6. Visual quality comparison of different methods. Differences are highlighted in boxes.

User Study To further evaluate the effectiveness of our method, we conduct a human evaluation to assess the perceived visual quality of the generated videos. Specifically, we randomly select 30 videos for each model. During the evaluation, each rater is presented with a text prompt along with two videos generated by different distillation methods, displayed in a randomized order to eliminate bias. Following the protocol of human preference evaluation in HunyuanVideo [20], the professional raters are asked to choose the video they perceive to have superior text alignment, motion quality, and visual quality. Each sample is evaluated by fifty independent raters, and the aggregated voting results are summarized in Table 2. As one can see, compared to other distillation methods, the raters significantly prefer the videos generated by our method.

HunyuanVideo, as shown in Table 3. All experiments were conducted on 29-frame videos with a resolution of 1280 × 720. Inference is performed with 4 sampling timesteps.

Table 3. Impact of different components of our method.

| |Variants<br><br>OD PE TC GF<br><br>|VBench<br><br>Total Quality Semantic|
|---|---|---|
|(1)<br>(2)<br><br>(3)<br><br>(4)<br><br>(5)<br><br>(6)<br><br><br>|✓<br><br>✓ ✓<br><br>✓ ✓ ✓<br><br>✓ ✓ ✓ ✓ ✓ ✓ ✓|80.30 80.74 78.36 83.08 84.20 78.59 83.03 84.16 78.53 83.42 84.63 78.63 83.71 84.99 78.59 83.80 85.10 78.62<br><br>|

Effect of Optimization Decoupling (OD) Through Experiments (1) and (2), decoupling the optimization of semantic and detail modeling significantly improves the semantic and quality scores. As shown in Fig. 7, the optimized decoupled model synthesizes videos with better semantic and detail quality, where the motion of characters and facial details appear more natural.

Table 2. User preference study. The numbers represent the percentage of raters who favor the videos synthesized by our method.

|Method comparison<br><br>|HunyuanVideo CogVideoX|
|---|---|
|Ours vs. LCM|82.67% 75.33%<br><br>|
|Ours vs. PCM|77.33% 72.67%|

Effect of Parameter-Efficient dual-expert distillation (PE) Through Experiments (2) and (3), we observe that compared to simply decoupling the optimization into two separate model training processes, the parameter-efficient

#### 4.3. Ablation Study

To thoroughly evaluate both the effectiveness of our method, we conduct extensive ablation studies based on

|80<br>81<br>82<br>83<br>84<br><br><br>VBench for Different κ<br><br>κ=28 κ=35 κ=37 κ=39 κ=46<br><br>|
|---|

[Figure 165]

Frame 5 Frame 15 Frame 25

[Figure 166]

Originalw/oOD withOD

Figure 10. Impact of different κ.

ModelTwo

based on the inference process. Fig. 10 (left) illustrates the L1 distance between adjacent time-step sampling results in HunyuanVideo during the sampling process. It can be observed that from approximately step 37 onward, the L1 Distance decreases to a very small value. We interpret this as the point where the semantic content and layout are established, and the remaining steps focus on synthesizing highfrequency details. Therefore, we set κ = 37 as the default value. To evaluate the impact of different κ values, we experimented with κ = 28,35,37,39,46. As shown in Fig. 10 (right), the results indicate that as κ deviates from the transition point between semantic synthesis and detail synthesis, the video quality gradually deteriorates. It further validates the effectiveness of our optimization decoupling strategy.

ModelPE

withOD

Figure 7. Impact of optimization decoupling and parameterefficient distillation.

[Figure 167]

losslossw/oTCTCw/

### 5. Conclusion and Discussion

Figure 8. Impact of temporal coherence loss.

In this paper, we identify a key optimization conflict in consistency distillation for video synthesis: there exists a significant discrepancy in the optimization gradients and loss contributions across different timesteps. Distilling the entire ODE trajectory into a single student model fails to balance these aspects, leading to degraded motion consistency and coarse synthesis quality. To address this issue, we propose a parameter-efficient Dual-Expert distillation framework that decouples semantic learning from fine-detail refinement. Additionally, we introduce a Temporal Coherence loss to enhance motion consistency for the semantic expert and apply GAN and Feature Matching loss to improve synthesis quality for the detail expert. Our method significantly reduces sampling steps while achieving stateof-the-art visual quality, demonstrating the effectiveness of expert specialization in video diffusion model distillation.

[Figure 168]

Original w/o GAN & FM +FM

[Figure 169]

[Figure 170]

[Figure 171]

+GAN +GAN+FM Local Zoom-in

Figure 9. Impact of the GAN loss and Feature Matching term.

Dual-Expert distillation significantly reduces both the parameters and memory requirements, with minimal computational overhead, while preserving visual quality. The last two rows of Fig. 7 also demonstrate that our parameterefficient Dual-Expert method does not result in a significant degradation in visual quality.

Limitation Although our method achieves favorable results with 4 steps inference, it still struggles to produce satisfactory outcomes with fewer steps (e.g., 2) due to limited training data and iterations. We will further explore highquality synthesis with fewer steps in future work.

Effect of Temporal Coherence Loss (TC) By comparing Experiments (3) and (4), or (5) and (6), we observe that the introduction of Temporal Coherence loss improves the quality scores of the synthesized videos. As shown in Fig. 8, the introduction of the TC loss enables more natural motion in the video and enhances its consistency.

### 6. Acknowledgement

Effect of GAN and Feature Matching Loss (GF) By comparing Experiments (3) and (5), or (4) and (6), we observe that the introduction of GAN Loss improves the quality scores of the synthesized videos. As shown in Fig. 9, the introduction of the GAN loss and Feature Matching term enhances the realism of the details in the synthesized video. Selection of κ In this paper, we determine the value of κ

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP202210012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

### References

- [1] David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbott, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. arXiv preprint arXiv:2303.04248, 2023. 3
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2
- [3] Dar-Yen Chen, Hmrishav Bandyopadhyay, Kai Zou, and Yi-Zhe Song. Nitrofusion: High-fidelity single-step diffusion through dynamic adversarial training. arXiv preprint arXiv:2412.02030, 2024. 3
- [4] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2, 6
- [5] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference {\&} Generative Modeling, 2023. 3
- [6] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [7] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024. 2
- [8] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2
- [9] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2
- [10] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2
- [11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 5
- [12] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 2
- [13] Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui Li, Wangmeng Zuo, and Rynson WH Lau. Dreamphysics:

- Learning physics-based 3d dynamics with video diffusion priors. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3733–3741, 2025. 2
- [14] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 6
- [15] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 2

- [16] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling diffusion models into conditional gans. In European Conference on Computer Vision, pages 428–447. Springer, 2024. 3
- [17] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. 35:26565–26577, 2022. 2
- [18] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023. 3
- [19] Jonas Kohler, Albert Pumarola, Edgar Sch¨onfeld, Artsiom Sanakoyeu, Roshan Sumbaly, Peter Vajda, and Ali Thabet. Imagine flash: Accelerating emu diffusion models with backward distillation. arXiv preprint arXiv:2405.05224,

2024. 3

- [20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 4, 6, 7, 1
- [21] Kuaishou. Kling, 2024.
- [22] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. 2
- [23] Sangyun Lee, Yilun Xu, Tomas Geffner, Giulia Fanti, Karsten Kreis, Arash Vahdat, and Weili Nie. Truncated consistency models. arXiv preprint arXiv:2410.14895, 2024. 3
- [24] Shanchuan Lin and Xiao Yang. Animatediff-lightning: Cross-model diffusion distillation. arXiv preprint arXiv:2403.12706, 2024. 3
- [25] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 3
- [26] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316,

2025. 3

- [27] Hongjian Liu, Qingsong Xie, Zhijie Deng, Chen Chen, Shixiang Tang, Fueyang Fu, Zheng-jun Zha, and Haonan Lu. Scott: Accelerating diffusion models with stochastic consistency distillation. arXiv preprint arXiv:2403.01505, 2024. 3

- [28] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. arXiv preprint arXiv:2202.09778, 2022. 2
- [29] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 3
- [30] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusionbased text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023. 3
- [31] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 3
- [32] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787,

2022. 2

- [33] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 2
- [34] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021. 3
- [35] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 3, 4, 6
- [36] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023. 3
- [37] Weijian Luo. A comprehensive survey on knowledge distillation of diffusion models. arXiv preprint arXiv:2304.04262,

2023. 3

- [38] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36:76525–76546, 2023. 3
- [39] Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. arXiv preprint arXiv:2410.16794, 2024. 3
- [40] Yihong Luo, Xiaolong Chen, Xinghua Qu, Tianyang Hu, and Jing Tang. You only sample once: Taming one-step text-toimage synthesis by self-cooperative diffusion gans. arXiv preprint arXiv:2403.12931, 2024. 3
- [41] Zhengyao Lv, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K Wong. Fastercache: Training-free video diffusion model acceleration with high quality. arXiv preprint arXiv:2410.19355, 2024. 2
- [42] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2

- [43] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 3
- [44] OpenAI. Sora, 2024. 2
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 1, 2

- [46] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 3
- [47] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. arXiv preprint arXiv:2404.13686, 2024. 3, 4
- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1
- [49] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 3
- [50] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 3
- [51] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2024. 3

- [52] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1, 2
- [53] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023. 3
- [54] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023. 1, 3, 4
- [55] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 2
- [56] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1
- [57] Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency model. arXiv preprint arXiv:2405.18407, 2024. 3, 4, 6
- [58] Fu-Yun Wang, Zhaoyang Huang, Weikang Bian, Xiaoyu Shi, Keqiang Sun, Guanglu Song, Yu Liu, and Hongsheng Li.

- Animatelcm: Computation-efficient personalized style video generation without personalized video data. In SIGGRAPH Asia 2024 Technical Communications, pages 1–5. 2024. 3
- [59] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 3
- [60] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025. 1
- [61] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. arXiv preprint arXiv:2112.07804, 2021. 3
- [62] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8196–8206, 2024. 3
- [63] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 2, 6
- [64] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024. 3
- [65] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024. 3
- [66] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast causal video generators. arXiv preprint arXiv:2412.07772, 2024. 3
- [67] Yuanhao Zhai, Kevin Lin, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Chung-Ching Lin, David Doermann, Junsong Yuan, and Lijuan Wang. Motion consistency model: Accelerating video diffusion with disentangled motion-appearance distillation. arXiv preprint arXiv:2406.06890, 2024. 3
- [68] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025. 2
- [69] Peiyuan Zhang, Haofeng Huang, Yongqi Chen, Will Lin, Zhengzhong Liu, Ion Stoica, Eric P Xing, and Hao Zhang. Faster video diffusion with trainable sparse attention. arXiv e-prints, pages arXiv–2505, 2025. 2
- [70] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 2
- [71] Yabo Zhang, Yuxiang Wei, Xianhui Lin, Zheng Hui, Peiran Ren, Xuansong Xie, and Wangmeng Zuo. Videoelevator: Elevating video generation quality with versatile text-to-image

- diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10266–10274, 2025.
- [72] Yabo Zhang, Xinpeng Zhou, Yihan Zeng, Hang Xu, Hui Li, and Wangmeng Zuo. Framepainter: Endowing interactive image editing with video diffusion priors. arXiv preprint arXiv:2501.08225, 2025. 2
- [73] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024. 2
- [74] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, and Anima Anandkumar. Fast sampling of diffusion models via operator learning. In International conference on machine learning, pages 42390–42402. PMLR,

2023. 3

- [75] Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao, and Tat-Jen Cham. Trajectory consistency distillation. arXiv preprint arXiv:2402.19159, 2024. 3
- [76] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024. 3

## Dual-Expert Consistency Model for Efficient and High-Quality Video Generation Supplementary Material

### 7. Further implementation details

Stage division and expert switching. During inference, we empirically observe that evenly dividing the total steps between the two experts produces favorable results. With 8 or 4 total steps, we assign 4 or 2 steps to each expert, respectively. These steps are uniformly sampled within each sub-trajectory.

### 8. Additional Results

#### 8.1. Compatibility with other acceleration techniques

DCM accelerates generation via sampling step reduction and is compatible with other methods like low precision computation and sparse modeling. For example, integrating SVG [60] (which leverages the sparsity of 3D full attention), yields an additional 1.33× speedup on top of DCMHunyuan while maintaining high fidelity (VBench 83.79%).

#### 8.2. Generality of DCM

DCM addresses discrepancies in loss and gradient contributions across noise levels—a problem inherent to consistency distillation itself, not from any specific model architecture. Beyond HunyuanVideo [20] and CogVideoX [63], we further apply it to the recent WAN2.1-T2V [56]. DCM significantly accelerates inference while preserving comparable visual quality, as evidenced by VBench scores (baseline: 83.2%, DCM: 82.9%).

#### 8.3. Visualization of the sampling process

To further verify the effectiveness of our method in semantic and detail synthesis, we visualize the results of each sampling step in a 4-step sampling process on HunyuanVideo. As shown in Fig. 11 and Fig. 12, our method achieves better performance in both semantic layout and fine details compared to competing methods.

#### 8.4. More visual comparison results

The additional visual comparison results for HunyuanVideo are presented in Fig. 13, Fig. 14 and Fig. 15. More visual results of CogVideoX are shown in Fig. 16. The results indicate that our method maintains reliable fidelity across diverse models, styles, and content in video synthesis while also achieving acceleration.

Semantic synthesis Detail synthesis

|Step 1<br><br>|Step 2|Step 3<br><br>|Step 4|
|---|---|---|---|

[Figure 172]

Frame5Frame15Frame25Frame15Frame25Frame5Frame15Frame25Frame5

| |
|---|

| |
|---|

LCM 4steps

[Figure 173]

| |
|---|

| |
|---|

PCM 4steps

[Figure 174]

| |
|---|

| |
|---|

Ours 4steps

Semantic synthesis Detail synthesis

|Step 1|Step 2<br><br>|Step 3|Step 4<br><br>|
|---|---|---|---|

[Figure 175]

Frame5Frame15Frame25Frame5Frame15Frame25Frame5Frame15Frame25

| |
|---|

LCM 4steps

| |
|---|

[Figure 176]

| |
|---|

| |
|---|

| |
|---|

PCM 4steps

[Figure 177]

| |
|---|

| |
|---|

Ours 4steps

[Figure 178]

Prompt:A fluffy,orange tabby catwith striking green eyes delicately laps water from acrystal-clear bowlplaced on a sunlit windowsill. The sunlight filters through the window, casting a warm glow on the cat's fur and creating a serene, peaceful atmosphere. The cat's whiskers twitch slightly as it drinks, and its ears perk up at the faint sounds of birds chirping outside.

Hunyuan

steps50

[Figure 179]

[Figure 180]

stepsstepsstepsstepsstepsstepsstepsstepssteps888888888

steps4

LCM

[Figure 181]

[Figure 182]

steps4

PCM

[Figure 183]

[Figure 184]

steps4

Ours

[Figure 185]

Prompt:A joyfulgolden retrieverwith a shiny coat sprints across a sunlit meadow, ears flapping and tongue lolling. The scene shifts to a close-up of the dog's face, eyes sparkling with excitement and mouth open. Its fur catching the sunlight. Finally, thedog runs towards the camera, tail wagging furiously, with a backdrop of vibrant wildflowers and a clear blue sky.

Hunyuan

steps50

[Figure 186]

[Figure 187]

steps4

LCM

[Figure 188]

[Figure 189]

steps4

PCM

[Figure 190]

[Figure 191]

steps4

Ours

[Figure 192]

Prompt: In a picturesque snowy landscape, a vibrant red frisbee soars through the crisp winter air, contrasting against the pristine white snow. Nearby, a sleek, modern skis, adorned with bold blue and white patterns, stand upright in the snow. The scene transitions to a close-up of the frisbee spinning gracefully, capturing the intricate details of its design.

Hunyuan

steps50

[Figure 193]

[Figure 194]

steps4

LCM

[Figure 195]

[Figure 196]

steps4

PCM

[Figure 197]

[Figure 198]

steps4

Ours

[Figure 199]

Prompt:In the neon-lit streets of Cyberpunk Beijing,a colossal robot towers over the cityscape, its sleek metallic frame adorned with glowing blue and red lights. The robot's design is a fusion of futuristic technology and ancient Chinese motifs. Holographic advertisements flicker around it, casting a kaleidoscope of colors on its polished surface.

Hunyuan

steps50 LCM

[Figure 200]

[Figure 201]

stepsstepsstepsstepsstepsstepsstepsstepssteps888888888

steps4 PCM

[Figure 202]

[Figure 203]

steps4 Ours

[Figure 204]

[Figure 205]

steps4 Hunyuan

[Figure 206]

Prompt:A graceful individual, dressed in a flowing shirt and black leggings, stands in a serene, sunlit room with wooden floors and large windows. She begin tobend slowly, her movements fluid and controlled. The sunlight filters through the windows, casting a warm glow on their form. The room's minimalist decor, with a fewpotted plants and a yoga mat

steps50 LCM

[Figure 207]

[Figure 208]

steps4 PCM

[Figure 209]

[Figure 210]

steps4 Ours

[Figure 211]

[Figure 212]

steps4 Hunyuan

[Figure 213]

Prompt:Ayoung woman with long, dark hair, wearing acozy gray sweater and jeans, stands in a laundry room. She sorts clothes into piles, the sunlight streaming through a nearby window casting a warm glow. As the machine starts, sheleans against the counter, sipping a cup of tea, her expression relaxed and content.

steps50 LCM

[Figure 214]

[Figure 215]

steps4 PCM

[Figure 216]

[Figure 217]

steps4 Ours

[Figure 218]

[Figure 219]

steps4

[Figure 220]

Hunyuan

steps50

Prompt:A vibrantcarousel spins under a twilight sky, its golden lights twinkling like stars. Painted horses with flowing manes and ornate saddles rise and fall gracefully, each one uniquely adorned with intricate details.

[Figure 221]

[Figure 222]

stepsstepsstepsstepsstepssteps888888stepsstepssteps888

steps4

LCM

[Figure 223]

[Figure 224]

steps4

PCM

[Figure 225]

[Figure 226]

steps4

Ours

[Figure 227]

Prompt:A well-dressed individual stands in front of a mirror, wearing a crisp white dress shirt and a sleek black suit jacket. The scene begins with a closeup of their handsskillfully looping a deep navy blue silk tie around their

Hunyuan

steps50

collar. The background is softly blurred, focusing attention on the tie and the person's meticulous technique.

[Figure 228]

[Figure 229]

steps4

LCM

[Figure 230]

[Figure 231]

steps4

PCM

[Figure 232]

[Figure 233]

steps4

Ours

[Figure 234]

Hunyuan

steps50

Prompt:A sleek,modern train glides over a towering steel bridge, its polished exterior reflecting the golden hues of the setting sun. As the train moves, its rhythmic clatter harmonizes with the distant calls of birds and the gentle rustling of leaves.

[Figure 235]

[Figure 236]

steps4

LCM

[Figure 237]

[Figure 238]

steps4

PCM

[Figure 239]

[Figure 240]

steps4

Ours

[Figure 241]

Prompt: In a futuristic setting, Iron Man, clad in his iconic red and gold armor, stands on a neon-lit stage, gripping a sleek, high-tech electronic guitar. As he strums the guitar, sparks fly, and holographic musical notes float around him, creating a mesmerizing visual symphony. His helmet's eyes glow intensely, syncing with the rhythm of the electrifying music.

CogVideoX

steps50

[Figure 242]

[Figure 243]

steps4

stepsstepsstepsstepsstepsstepsstepsstepssteps888888888

LCM

[Figure 244]

[Figure 245]

steps4

PCM

[Figure 246]

[Figure 247]

steps4

Ours

[Figure 248]

Prompt: A vibrant soccer ball, with its classic black and white hexagonal pattern, rests on a lush, green field under a clear blue sky. The camera zooms in to reveal the intricate stitching and slight scuffs from previous games, highlighting its well-loved nature. As the ball is gently nudged, it rolls smoothly across the grass, capturing the sunlight that glints off its surface.

CogVideoX

steps50

[Figure 249]

[Figure 250]

steps4

LCM

[Figure 251]

[Figure 252]

steps4

PCM

[Figure 253]

[Figure 254]

steps4

Ours

[Figure 255]

CogVideoX

Prompt:A vibrantblue jayperches gracefully on a slender branch, its feathers shimmering in the soft morning light. It flutters its wings briefly, showcasing the intricate patterns of blue, white, and black on its plumage. The background reveals a lush canopy of green leaves, with rays of sunlight filtering through, creating a dappled effect on the forest floor.

steps50

[Figure 256]

[Figure 257]

steps4

LCM

[Figure 258]

[Figure 259]

steps4

PCM

[Figure 260]

[Figure 261]

steps4

Ours

Figure 16. Visual quality comparison of different methods.

