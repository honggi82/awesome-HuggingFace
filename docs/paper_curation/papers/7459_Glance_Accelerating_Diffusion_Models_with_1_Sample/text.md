# arXiv:2512.02899v2[cs.CV]11Dec2025

## Glance: Accelerating Diffusion Models with 1Sample

[Figure 1]

Zhuobai Dong1∗, Rui Zhao2∗, Songjie Wu3, Junchao Yi4, Linjie Li5, Zhengyuan Yang5, Lijuan Wang5, Alex Jinpeng Wang3

1WHU 2NUS 3CSU 4UESTC 5Microsoft https://zhuobaidong.github.io/Glance/

#### Abstract

Diffusion models have achieved remarkable success in image generation, yet their deployment remains constrained by the heavy computational cost and the need for numerous inference steps. Previous efforts on fewer-step distillation attempt to skip redundant steps by training compact student models, yet they often suffer from heavy retraining costs and degraded generalization. In this work, we take a different perspective: we accelerate smartly, not evenly, applying smaller speedups to early semantic stages and larger ones to later redundant phases. We instantiate this phase-aware strategy with two experts that specialize in slow and fast denoising phases. Surprisingly, instead of investing massive effort in retraining student models, we find that simply equipping the base model with lightweight LoRA adapters achieves both efficient acceleration and strong generalization. We refer to these two adapters as Slow-LoRA and FastLoRA. Through extensive experiments, our method achieves up to 5× acceleration over the base model while maintaining comparable visual quality across diverse benchmarks. Remarkably, the LoRA experts are trained with only 1 samples on a single V100 within one hour, yet the resulting models generalize strongly on unseen prompts.

#### 1. Introduction

Diffusion and flow matching models [1, 20, 32, 52, 53] have shown strong capabilities in generating highfidelity images, marking a significant advancement in the field of generative modeling. Despite their impressive performance, a notable challenge is the high inference cost due to its iterative denoising nature. To address this issue, various methods are proposed to accelerate the sampling process of diffusion models, including improving the efficiency of samplers [25, 33, 35, 36] and employing model distillation techniques [14, 15, 26, 34, 49, 55] to reduce the number of inference steps. Recent advancements in trajectory

1* Equal contribution.

[Figure 2]

DMD2

𝜋-Flow

DataUsage(images)𝑥10

TimeStep Master

Qwen-Image-

Lightning

Qwen-Image-Distill-LoRA

Glance (ours)

- 0

- 3

2

- 4

1

8

7

6

- 5

10 100 1000

Training Time (A100 GPU Hours)

Figure 1. Comparison of data usage and training time. Glance achieves comparable generation quality with only 1 training samples and within 1 GPU-hour, demonstrating extreme data and compute efficiency. Note that the x-axis is in logarithmic scale, and values equal to zero are therefore not representable.

distillation methods and distribution matching techniques [50, 63, 65, 67], often enhanced by adversarial learning at scale, have shown considerable promise in generating high-fidelity images in extremely low steps such as one to four steps.

Despite significant advancements in timestepdistilled diffusion models, it remains unclear how to effectively fine-tune or customize such distilled models. Naively tuning the distilled model with diffusion loss will make the generation results blurry. An alternative approach is to fine-tune or customize the original diffusion model, and then repeat the diffusion distillation process to create a distilled model variant. However, the large computation cost of diffusion distillation, when compared with the customization training used for distillation (cf., 3840 A100 GPU hours for SDXL-DMD2 [63] and 3072 A100 GPU hours for Qwen-Image-Lightning [39]), often makes such distilled model tuning approach less feasible.

In this work, we take a different perspective by revisiting the denoising dynamics of diffusion models. We observe that the generation trajectory consists of

[Figure 3]

[Figure 4]

[Figure 5]

………………………………………….…………………………………………. 50 Steps

Base Generation Model

(a) Base model: Accurate but slow

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Distillation (Heavy Retraining)

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

….….

Base Generation Model Student Model

8 Steps

Large Set of Samples

(b) Distillation: Faster but costly retraining

[Figure 20]

[Figure 21]

Slow LoRA

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

….….

(Light Training)

8 Steps

Fast

One Sample

Glance LoRA Base Model

Base Generation Model

(c) Glance (Plug-and-speed-up): lightweight, cost-efficient, and generalizable

- Figure 2. Comparison of distill and accelerate strategies. Prior distillation pipelines rely on large training sets and costly retraining. Glance requires only one training sample to obtain Slow-LoRA and Fast-LoRA, providing plug-and-play acceleration of the base generation model.

two qualitatively distinct phases: an early semantic phase that determines global structure, and a late redundant phase that primarily refines texture. Uniform acceleration treats all steps equally, yet semantic steps are far more sensitive to perturbation than redundant ones. This motivates a phase-aware acceleration strategy that applies small speedups to semantic steps and large speedups to redundant steps.

To realize this idea, we introduce Glance, implemented as a pair of lightweight LoRA adapters that attach to the pretrained diffusion model. SlowLoRA stabilizes early semantic formation, while FastLoRA accelerates late-stage refinement. Crucially, our method does not require training a new student network and the base model remains unchanged. Both LoRA experts are trained using only one sample on a single V100 GPU within one hour (Fig. 1).

To demonstrate its scalability, we distill FLUX.112B [27] and Qwen-Image-20B [57] text-to-image models into 8- and 10-step students, respectively. Extensive experiments across six text-to-image benchmarks show that Glance exhibits performance curves that closely track those of the base models, indicating strong consistency under accelerated inference. On OneIG-Bench, HPSv2, and GenEval, the performance of Glance reaches 92.60%, 99.67%, and 96.71% of the teacher models, respectively. We further conduct dense ablation studies on slow–fast phase decomposition, timestep allocation, and training data scaling, all of which consistently validate the effectiveness and robustness of our design.

We summarize the contributions as follows:

- • We employ a phase-aware acceleration scheme that treats semantic and redundant steps differently for a more natural and stable speedup.
- • We introduce two lightweight adapters that plug into the base model, enabling effective acceleration with

only one sample and one hour of training.

• Glance achieves a 5× speed-up over FLUX.1 and Qwen-Image while retaining teacher-level performance across six benchmarks.

#### 2. Related Work

Diffusion Models. Recently, diffusion models (DMs) [20, 52, 54] have become the leading paradigm for visual generation, achieving state-of-the-art performance across a wide range of conditioning modalities, including images [48], depth, edges, poses [40, 66], and text [10, 12, 21, 43, 45, 48]. Advances in large-scale systems such as PixArt [6–8], SD3 [13], Qwen-Image [57], and FLUX [27] further push generation quality, controllability, and multilingual rendering capabilities. Despite these rapid developments, achieving high-fidelity synthesis still requires many denoising steps, resulting in substantial inference cost and limiting their use in real-time or resource-constrained applications.

Diffusion Distillation. Early work [37] directly regresses the teacher’s ODE integral in a single step, but the ℓ2-based x0 regression often produces overly smooth and blurry results. Progressive distillation methods [14, 34, 49] refine this paradigm via multistage training that enlarges step size and lowers NFE by merging teacher steps. While effective, these approaches suffer from error accumulation and substantial computational overhead. Consistency distillation [15, 26, 55] replaces x0 regression with velocity-based objectives to enforce trajectory consistency, improving fidelity but requiring costly Jacobian–vector products (JVPs) or inaccurate finite-difference approximations. Distribution matching methods [50, 63, 65, 67]instead adopt score-based or adversarial objec-

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

|𝑡 = 0|
|---|

|𝑡 = 𝑇|
|---|

𝑡 = 0 Timesteps of Diffusion 𝑡 = 𝑇

||[Figure 31]<br><br>𝚯|
|---|
<br><br>𝑨𝟐 𝑩𝟐<br><br>[Figure 32]<br><br>[Figure 33]<br><br>|
|---|

||[Figure 34]<br><br>𝚯|
|---|
<br><br>𝑨𝟏 𝑩𝟏<br><br>[Figure 35]<br><br>[Figure 36]<br><br>|
|---|

Baseline

[Figure 37]

Slow

###### Glance

Fast

Slow-LoRA Fast-LoRA

- Figure 3. Visualization of Slow-Fast paradigm. In the slow stage, we sample one timestep every two steps from the first 20 timesteps (i.e., 5 samples in total). In the fast stage, an additional 5 timesteps are uniformly sampled from the remaining 40 steps. During inference, the slow-stage timesteps are executed prior to the fast-stage ones.

tives to align the student’s output distribution with the teacher’s, achieving high perceptual quality yet prone to mode collapse and instability due to auxiliary discriminators. Recent concurrent studies [5, 39, 56] have distilled Qwen-Image and FLUX into compact 4-NFE or 8-NFE versions, but these still incur high distillation costs from trajectory-level supervision and extensive teacher sampling. In contrast, our approach achieves comparable or superior generation quality with minimal training cost, avoiding recursive distillation and auxiliary network overhead while maintaining stable optimization.

Efficient Tuning of Diffusion Models. LoRA [22] has been widely adopted for efficient adaptation of diffusion models via low-rank parameter updates [18, 24, 28, 38, 41, 46, 60–62, 66]. Several works integrate LoRA into distillation or controllable generation: DMD [64] enables faster inference through LoRA-based distillation; ControlNeXt [42] and TCLoRA [9] provide adaptive, time- or conditiondependent LoRA control; Timestep Master [68] assigns LoRA experts to different noise levels for better representation. Recent models such as QwenImage-Lightning [39] and Qwen-Image-Distill-LoRA [56] embed LoRA into distilled backbones, producing compact low-step (4–8 NFE) models. However, their generation quality remains limited. We address this by allocating slow and fast LoRA adapters according to denoising phases, yielding substantial improvements.

#### 3. Method

In this section, we introduce Glance, a phaseaware acceleration framework that improves both efficiency and adaptability of diffusion models through slow–fast paradigm. We first revisit the diffusion model and flow-matching formulation as preliminaries, then describe our phase-aware LoRA experts and their learning objectives.

##### 3.1. Preliminary

Diffusion and Flow Matching. Diffusion models [20] learn data distributions by gradually transforming noise into data through a parameterized de-

noising process. The flow matching formulation [12, 34] interprets diffusion as learning a continuous velocity field that transports a sample from Gaussian noise x1 ∼ N(0,I) to clean data x0. At timestep t ∈ [0,1], the intermediate state is defined as xt = tx0 + (1 − t)x1, and the model predicts the transport velocity vθ(xt,t,h) conditioned on guidance h (e.g., text embedding). The objective is a mean-squared error between the predicted and target velocities:

0,x1,t,h ∥vθ(xt,t,h) − vt ∥22 ,

LFM = Ex

where vt is the groundtruth velocity. To achieve superior performance, the diffusion model is often designed with a large number of network parameters that are pre-trained on large-scale web data. Apparently, it is computationally expensive to distill such a big model for step reduction.

Low-Rank Adaptation To alleviate the above difficulty, LoRA [22] has been recently applied for rapid distillation diffusion models on target data [3, 68]. Specifically, LoRA introduces low-rank decomposition of an extra matrix, Θ′ = Θ + BA, where Θ ∈ Rd×k denotes the frozen pretrained parameters, and the low-rank matrices A ∈ Rr×k and B ∈ Rd×r (with r ≪ d,k) constitute the learnable LoRA parameters.

##### 3.2.Phase-awareLoRAExpertsforPhase-wise Denoising

To accelerate the denoising process of pretrained diffusion models while maintaining generative quality, we retain the pretrained parameters Θ and introduce a compact yet effective augmentation: a set of phasespecific LoRA adapters. Each adapter specializes in a specific stage of the denoising trajectory, enabling the model to adapt dynamically to varying noise levels and semantic complexities during inference.

Beyond uniform timestep partitioning. Prior works such as Timestep Master [68] have demonstrated the potential of using multiple LoRA adapters trained over different timestep intervals. However, Uniform partitioning assumes equal contribution from all timesteps, which contradicts the intrinsic nonuniformity of diffusion dynamics. Empirical analyses

###### Table 1. Quantitative comparisons on COCO-10k dataset and HPSv2 prompt set.

COCO-10k prompts HPSv2 prompts

Model Distill method NFE

###### Data align. Prompt align. Pref. align. Prompt align. Pref. align. FID↓ pFID↓ CLIP↑ VQA↑ HPSv2.1↑ CLIP↑ VQA↑ HPSv2.1↑

FLUX.1 dev - 50 27.8 34.9 0.268 0.900 0.309 0.284 0.805 0.314 FLUX Turbo GAN 8 26.7 32.0 0.267 0.900 0.308 0.286 0.814 0.313 Hyper-FLUX CD+Re 8 29.8 33.3 0.268 0.894 0.309 0.285 0.807 0.315 π-Flow (FLUX) π-ID 8 29.0 35.4 0.268 0.901 0.311 0.285 0.810 0.316 Glance (FLUX) Slow-Fast 8 34.2 40.1 0.259 0.879 0.298 0.276 0.895 0.297 Glance (FLUX) Slow-Fast 10 30.4 37.5 0.265 0.891 0.303 0.282 0.799 0.303 Qwen-Image - 50×2 34.1 45.6 0.282 0.936 0.312 0.302 0.872 0.309 Qwen-Image - 10×2 39.1 52.1 0.265 0.918 0.287 0.288 0.835 0.305 Qwen-Image - 8×2 44.2 56.3 0.263 0.873 0.269 0.287 0.761 0.292 Qwen-Image-Lightning VSD 4 37.5 51.6 0.280 0.935 0.322 0.299 0.867 0.328 π-Flow (Qwen) π-ID 4 36.0 46.1 0.281 0.934 0.314 0.300 0.860 0.310 Glance (Qwen) Slow-Fast 8×2 38.9 52.4 0.273 0.926 0.289 0.287 0.842 0.302 Glance (Qwen) Slow-Fast 10×2 37.8 50.3 0.280 0.932 0.313 0.297 0.849 0.308

###### Table 2. Quantitative comparisons on OneIG-Bench. * denotes unavailable results.

Model Distill Method Training Cost NFE Alignment↑ Text↑ Diversity↑ Style↑ Reasoning↑ Data GPU hour

FLUX.1 dev - - - 50 0.790 0.556 0.238 0.370 0.257 FLUX Turbo GAN 1M * 8 0.791 0.334 0.234 0.370 0.239 Hyper-FLUX CD+Re 1.1M 800 8 0.790 0.530 0.198 0.369 0.254 π-Flow (FLUX) π-ID 2.3M 83 8 0.792 0.517 0.234 0.369 0.256 Glance (FLUX) Slow-Fast 1 0.6 8 0.774 0.284 0.196 0.353 0.208 Glance (FLUX) Slow-Fast 1 0.8 10 0.788 0.328 0.204 0.358 0.231 Qwen-Image - - - 50×2 0.880 0.888 0.194 0.427 0.306 Qwen-Image - - - 10×2 0.802 0.693 0.156 0.410 0.290 Qwen-Image - - - 8×2 0.752 0.611 0.148 0.411 0.276 Qwen-Image-Lightning VSD 0.4M 3072 4 0.885 0.923 0.116 0.417 0.311 π-Flow (Qwen) π-ID 2.3M 83 4 0.875 0.892 0.180 0.434 0.298 LoRA (Qwen) Uniform steps 1 0.8 10×2 0.621 0.332 0.097 0.298 0.193 Glance (Qwen) Slow-Fast 1 0.6 8×2 0.863 0.692 0.162 0.414 0.286 Glance (Qwen) Slow-Fast 1 0.8 10×2 0.868 0.734 0.160 0.421 0.303

and prior studies [2] reveal that different timesteps exhibit markedly different levels of semantic importance: in the early, high-noise regime, the model primarily reconstructs coarse global structures and high-level semantics (low-frequency information); in contrast, the later, low-noise regime refines textures and details (high-frequency information).

Phase-aware partitioning via SNR. To better align expert specialization with the intrinsic dynamics of the diffusion process, we introduce a phase-aware partitioning strategy guided by the signal-to-noise ratio (SNR). Unlike timestep indices, the SNR provides a physically meaningful measure of the relative dominance between signal and noise, and it decreases monotonically as denoising progresses. At the beginning of the process (t large, high-noise phase), the latent representation is dominated by noise with a low SNR, making coarse structural recovery the primary

objective. In contrast, as t decreases and SNR rises, the model transitions into a low-noise regime focused on texture refinement.

Based on this observation, we define a transition boundary ts corresponding to an SNR threshold (e.g., half of the initial SNR value). Two phase-specific experts are then employed: a slow expert specialized for the high-noise phase (t ≥ ts) that focuses on coarse semantic reconstruction, and a fast expert for the lownoise phase (t < ts) that enhances fine-grained details. This SNR-guided partition allows each expert to operate in the regime where it is most effective, forming a semantically meaningful decomposition of the denoising process.

Surprising effectiveness of extremely small training sets. To evaluate whether phase-wise LoRAs can recover accelerated inference, we initially conducted an overfitting-style experiment using only 10

training samples. Unexpectedly, the model rapidly learned a faithful approximation of the accelerated sampling trajectory. Even more remarkably, reducing the dataset to a single training sample still produced a stable acceleration behavior.

We attribute this data efficiency to the nature of flow matching. By directly predicting the target velocity field along the diffusion trajectory, the training objective bypasses redundant score-matching steps. Consequently, essential structural knowledge for fast inference can be distilled from only a few examples.

Necessity of carefully designed timestep skipping. Despite this promising data efficiency, subsequent ablation studies reveal that timestep skipping is far from arbitrary. Although few-step students can imitate the teacher behavior in aggregate, not all timesteps contribute equally to the reconstruction dynamics; naive skipping strategies can severely degrade performance.

To this end, we conducted a comprehensive investigation of different specialization schemes. We first explored assigning multiple timesteps to the slow stage LoRA adapters while keeping a single adapter for the fast stage, and vice versa. We also tested a degenerate configuration where a single LoRA was trained across the entire trajectory. However, these variants either lacked the expressiveness to capture high-noise complexity or failed to exploit temporal locality in the low-noise refinement phase.

Our experiments ultimately show that separating the trajectory into a dedicated slow region and a dedicated fast region yields the most robust specialization. This design preserves sufficient capacity for modeling the challenging high-noise dynamics while enabling lightweight refinement in later steps, achieving a compact yet effective acceleration mechanism.

Flow-matching supervision. Each phase-specific LoRA expert is trained under a flow-matching supervision scheme that aligns its predicted denoising direction with the underlying data flow. Given the noisy latent xt obtained during the diffusion process, the model predicts a velocity field vˆΘ,B

i,Ai(xt,t,c), which is supervised against the ground-truth flow vector vt⋆. The training objective is defined as a weighted mean-squared error:

i,Ai(xt,t,c) − vt⋆∥22 ,

LFM(t;i) = Ex,c,ε w(t)∥vˆΘ,B

where w(t) denotes an optional timestep-dependent weighting function. By restricting the training samples of each expert to its assigned denoising phase, the model effectively learns to specialize on distinct noise levels. The resulting mixture of phase-aware LoRA experts collectively improves both the denoising speed and generative quality, forming the foundation of our proposed slowfast paradigm.

|Qwen-Image FLUX.1-dev Glance (Qwen) Glance (FLUX)|
|---|

|0.54<br><br>0.44<br><br>0.59<br><br>0.50 0.309<br><br>0.314 0.308<br><br>0.91<br><br>0.66<br><br>0.88<br><br>88.3<br><br>83.8<br><br>34.1<br><br>27.8<br><br>37.8<br><br>94.3<br><br>61.2<br><br>79.6<br><br>0.38<br><br>0.303<br><br>30.4<br><br>48.2<br><br>85.2<br><br>81.1|
|---|

OneIG-Bench HPSv2 GenEval DPG COCO-10k LongText-Bench

Figure 4. Comparison between Image Generation Benchmarks. Glance further shows performance trajectories that closely follow those of the corresponding base models.

#### 4. Experiments

This section presents a comprehensive evaluation of Glance on the text-to-image generation task. We first report quantitative results compared with competitive baselines, followed by detailed ablation analyses. We then discuss the generalization behavior of the model and its sensitivity to data scale.

##### 4.1. Experimental Setup

Dstillation Setup. We distill two large-scale textto-image generators, FLUX.1-12B [27] and QwenImage-20B [57], into compact Slow-Fast students. During distillation, the base parameters inherited from the teacher are kept frozen, while only the LoRA adapters are optimized. Following Qwen-ImageDistill-LoRA [56], we extend the adapter placement beyond the standard attention projections. Specifically, LoRA modules are injected not only into the query, key, value, and output projections, but also into auxiliary projection layers and modality-specific MLPs across both visual and textual branches. This broader integration allows the student to more effectively capture cross-modal dependencies and retain generation fidelity despite its compact capacity.

Evaluation protocol. We conduct a comprehensive evaluation on 10242 high-resolution image generation from three distinct prompt sets: (a) 10K captions from the COCO 2014 validation set [30], (b) 3200 prompts from the HPSv2 benchmark [59], (c) 1120 prompts from OneIG-Bench [4], (d) 553 prompts from the GenEval benchmark [17], (e) 1065 prompts from the DPG-Bench [23], and (f) 160 prompts from the LongText-Bench [16]. For the COCO and HPSv2 sets, we report common metrics including FID [19], patch FID (pFID) [29], CLIP similarity [44], VQAScore [31], and HPSv2.1 [58]. On COCO prompts, FIDs are computed against real images, reflecting data alignment. On HPSv2, CLIP and VQAScore measure prompt alignment, while HPSv2 captures human preference alignment. For OneIG-Bench, GenEval, DPGBench, and LongText-Bench, we adopt their official evaluation protocols and report results based on their

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Qwen-Image

𝟓𝟎 × 𝟐 NFE

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Qwen-Image 𝟏𝟎 × 𝟐 NFE

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Qwen-Image

𝟖 × 𝟐 NFE

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Qwen-Image -Lightning

4 NFE

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

𝝅-Flow 4 NFE

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Glance 𝟖 × 𝟐 NFE

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Glance 𝟏𝟎 × 𝟐 NFE

- Figure 5. Visual comparison of different Slow–Fast configurations. All images are generated from the same initial noise using the 50-step base model, our 8/10-step students, and other few-step models. Slow-Fast preserves semantic fidelity under strong acceleration, while additional steps progressively enhance fine details. respective benchmark metrics.

Overall Performance. As illustrated in Figure 4, Glance exhibits performance curves that closely track those of the base models (Qwen-Image and Flux) across all benchmarks, indicating strong consistency under the accelerated setting. The detailed quantitative summaries in Table 1 and Table 2 confirm this trend: Glance achieves nearly the same generation quality as the 50-step base models while running 5× faster. Despite the aggressive acceleration, it maintains competitive FID, CLIP, and HPSv2 scores across all benchmarks, showing no clear degradation in visual fidelity or prompt alignment. Moreover, even when trained with only 1 sample within less than one GPU hour, Glance delivers results comparable to other few-step distillation approaches that require large-scale datasets and heavy computation. These results underscore the strong efficiency–performance balance achieved by our approach, validating that the proposed Slow-Fast LoRA framework can preserve high generation quality under minimal supervi-

##### 4.2. Main Results

We compare Glance against other few-step student models distilled from the same teacher. For FLUX, we compare against: 8-NFE Hyper-FLUX [47], trained with consistency distillation (CD) and reward models (Re); 8-NFE FLUX Turbo [51], based on GAN-like adversarial distillation; π-Flow (FLUX) [5], trained with policy-based imitation distillation (π-ID). For Qwen-Image, we compare with the 4-NFE QwenImage Lighting based on variational score distillation (VSD) and π-Flow (Qwen). To further evaluate under extremely low-data conditions, we additionally implement a LoRA-based uniform timestep distillation approach that uniformly distill time steps using only 1 sample. The results are shown in Table 1 and 2. In Appendix A, we also explore the Qwen-Image-Edit model [57] in the image-editing domain.

sion and limited computational resources.

Visual Fidelity. To further examine the trade-off between speed and quality, Figure 5 presents qualitative comparisons among our Glance models (8- and 10step), the base teacher models (8-, 10-, and 50-step), and other 4-step distillation models such as QwenImage-Lightning and π-Flow, all under identical noise initialization. Even at only eight steps, Glance maintains the teacher’s global semantics and color composition with minimal loss of fidelity. Increasing the step count gradually restores fine textures and small structures, indicating that phase-aware LoRA adaptation preserves the denoising trajectory of teacher despite extreme acceleration. These observations align with our quantitative findings: the 5× faster model achieves nearly the same quality as the teacher while requiring only a fraction of the computation.

##### 4.3. Ablation Study

We conduct comprehensive ablation studies to analyze the key factors that contribute to the effectiveness of the proposed Slow–Fast design. Unless otherwise stated, all experiments are evaluated on OneIG-Bench using Qwen-image as the teacher model.

The importance of Slow-Fast Design. To verify the effectiveness of the proposed phase-aware design, we divide the diffusion process into two distinct denoising stages and systematically vary the LoRA assignment strategy. Specifically, we experiment with five configurations to assess the role of Slow-LoRA and Fast-LoRA under different timestep allocations. We experiment with five configurations: (1) phase-aware Slow3–Fast5 setup (ours), (2) Slow3 + Base5, (3) Base3 + Fast5, (4) single LoRA at identical timesteps, and (5) single LoRA uniformly sampled across eight timesteps.

As summarized in Table 3, our asymmetric Slow–Fast configuration achieves the highest performance across all metrics, demonstrating its superior balance between quality and efficiency. This confirms that aligning LoRA updates with the semanticto-refinement progression of denoising leads to more effective knowledge transfer. In this process, the model performs slow adaptation during early stages and fast refinement during later stages, achieving better specialization than uniform or single-expert alternatives. Among all variants, the Single (uniform) setup performs the worst, confirming the necessity of phase-wise specialization. Notably, we observe that the early-stage Slow-LoRA contributes more significantly to final image quality, underscoring the importance of coarse-to-fine adaptation in guiding generation.

If more samples help? To investigate the effect of data composition on LoRA adaptation, we first

Table 3. Slowfast stage ablation study.

Model Alignment↑ Text↑ Diversity↑ Style↑ Reasoning↑

Slow3–Fast5 0.849 0.614 0.152 0.396 0.284 Slow3 + Base5 0.805 0.567 0.123 0.368 0.255 Base3 + Fast5 0.747 0.521 0.125 0.372 0.243 Single (identical) 0.702 0.453 0.110 0.342 0.218 Single (uniform) 0.621 0.332 0.097 0.298 0.193

randomly select 1 text–image pairs from the QwenImage-Self-Generated-Dataset [11] dataset as the minimal training set. We then scale up the dataset while keeping the total number of training epochs fixed. As shown in Table 4, increasing the number of training samples from 1 to 10 and further to 100 does not lead to notable performance gains. Most metrics remain nearly unchanged, while the style score even slightly declines, suggesting that simply enlarging the dataset without enhancing its diversity or phase alignment may weaken stylistic consistency. These results indicate that for phase-aware LoRA adaptation, data quality and phase alignment are more crucial than scale, and that even a single well-chosen sample can achieve effective adaptation.

Table 4. Training data ablation study.

Model Alignment↑ Text↑ Diversity↑ Style↑ Reasoning↑

1 sample 0.868 0.734 0.160 0.421 0.303 10 samples 0.874 0.758 0.163 0.414 0.296 100 samples 0.876 0.753 0.165 0.418 0.306

Timestep Ablation. In this experiment, we study the influence of the number of timesteps equipped with LoRA adapters while keeping the data selection and scale fixed at 1 text–image pairs. We progressively increase the number of timesteps on which LoRA modules are attached, thereby examining how the temporal coverage of adaptation affects generation quality. As shown in Table 5, with more LoRAequipped timesteps, the overall performance steadily improves, particularly showing a clear gain in the text metric. This trend indicates that broader temporal adaptation enables the model to capture more phase-specific denoising dynamics, leading to better overall reconstruction quality.

Table 5. Timestep ablation study.

Model Alignment↑ Text↑ Diversity↑ Style↑ Reasoning↑

Slow3 + Fast5 0.863 0.692 0.162 0.414 0.286 Slow5 + Fast5 0.868 0.734 0.160 0.421 0.303 Slow5 + Fast10 0.874 0.813 0.175 0.422 0.305

##### 4.4. Discussion

Generalization Ability of Different Single Training Samples. We conduct an ablation study using four distinct single-image settings to examine the generalization behavior of our framework under extreme data scarcity. Specifically, we first select in-distribution samples from [11], where the samples are generated from Qwen-image base model. We select three sample with representing diverse semantic and structural characteristics: (1) Fox: an anthropomorphic fox with vibrant red fur and white facial features, (2) Valley

Valley Landscape Bookstore Real-World Gaussian Noise

Fox

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Training Sample

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

Inference Sample

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

- Figure 6. Qualitative results from the one-sample training setting. Even trained on a single image, the model generalizes well to unseen prompts, producing coherent and detailed results across diverse scenes.

landscape: a unique natural scene featuring a winding spiral valley, and (3) Bookstore: a text-rich storefront window densely filled with books and signage. Additionally, an out-of-distribution (OOD) real-world image, depicting a bustling city crosswalk filled with pedestrians moving in different directions on a clear day, is included to assess robustness beyond the training domain. For completeness, we also experiment with an extreme setting where the model is trained purely on Gaussian noise, serving as a control to isolate the effect of meaningful visual content. All models are trained on different samples using the same configuration to assess their generalization ability.

Table 6. 1 data sample study.

Model Alignment↑ Text↑ Diversity↑ Style↑ Reasoning↑

Fox 0.868 0.734 0.160 0.421 0.303 Valley Landscape 0.842 0.712 0.146 0.409 0.299 Bookstore (text-rich) 0.797 0.751 0.131 0.373 0.267 Real-World (OOD) 0.857 0.728 0.153 0.420 0.298

From Table 6 and Fig. 6 we observe: i. While the three single-sample settings exhibit noticeable stylistic differences, their quantitative performance remains relatively close. ii. The Bookstore (text-rich) model exhibits the weakest generalization, while the foxtrained model demonstrates the strongest. The foxbased LoRA yields consistent improvements across all evaluation metrics, and Figure 6 further illustrates that it produces images with better convergence and richer fine-grained details. However, the gap between them is modest. iii. Surprisingly, the Real-world image (OOD) model also achieves competitive results, with scores on most metrics only slightly lower than those of the Fox-based LoRA, suggesting that out-ofdomain samples can still provide meaningful transferable cues for denoising adaptation. iv. When trained solely on Gaussian noise, the model fails to produce any meaningful images, indicating that effective de-

[Figure 112]

|[Figure 113]|
|---|

Qwen-Image 𝟓𝟎 × 𝟐 NFE

| |
|---|

[Figure 114]

|[Figure 115]|
|---|

Glance

𝟏𝟎 × 𝟐 NFE

| |
|---|

Figure 7. Text-render failure cases. Glance struggles on extremely small text, producing blurred or distorted characters.

noising requires exposure to data that resembles natural image distributions.

Overall, while different samples lead to consistent performance trends, the quantitative differences are minor. This shows that Glance is not particularly sensitive to which single image is used for training, and can still learn transferable denoising behaviors from extremely limited but coherent supervision.

Failure Analysis. Although Glance achieves performance curves that closely track those of the base models across all evaluated benchmarks, we identify a consistent weakness in text rendering quality. As reported in Table 2 and Fig. 4, Glance shows clear deficits in both text rendering, falling behind Qwen-Image and FLUX by 0.154 and 0.228 on the Text-Render metric, and by 14.7 and 13.0 points on LongText-Bench, respectively. To better understand this discrepancy, we conducted a detailed inspection of the generated samples. As illustrated in Fig. 7, failure cases predominantly occur in images containing extremely dense or very small text, where the model struggles to preserve sharp character boundaries, often producing blurred strokes or local artifacts. In contrast, when the text is shorter and occupies a larger spatial extent in the image, Glance is able to reproduce it faithfully.

These suggest that high-frequency textual details are harder for the student to capture than general visual content, Such fine-grained details require precise spatial alignment and are harder to capture during distillation, especially under few-step constraints.

#### 5. Conclusion

We present Glance, a lightweight distillation framework that accelerates diffusion inference through a phase-aware Slow–Fast design. The well studied LoRA adapters distinct denoising phases to efficiently capture both global semantics and local refinements. Glance enables high-quality generation with only eight steps, achieving an 5× speed-up over the base model. Despite being trained with as few as one image and a few GPU hours, Glance maintains comparable visual fidelity and exhibits strong generalization to unseen prompts. These results highlight that data- and

compute-efficient distillation can retain the expressive capacity of large diffusion models without sacrificing quality. We believe Glance serve as a strong candidate for accelerating large-scale diffusion models, particularly in data-scarce applications.

#### References

- [1] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022. 1
- [2] Sotiris Anagnostidis, Gregor Bachmann, Yeongmin Kim, Jonas Kohler, Markos Georgopoulos, Artsiom Sanakoyeu, Yuming Du, Albert Pumarola, Ali Thabet, and Edgar Sch¨onfeld. Flexidit: Your diffusion transformer can easily generate high-quality samples with less compute. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28316– 28326, 2025. 4
- [3] Clement Chadebec, Onur Tasar, Eyal Benaroche, and Benjamin Aubin. Flash diffusion: Accelerating any conditional diffusion model for few steps image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 15686–15695, 2025. 3
- [4] Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arXiv:2506.07977, 2025. 5
- [5] Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein, and Sai Bi. pi-flow: Policy-based few-step generation via imitation distillation. arXiv preprint arXiv:2510.14974, 2025. 3, 6
- [6] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-tostrong training of diffusion transformer for 4k text-toimage generation. arXiv preprint arXiv:2403.04692,

2024. 2

- [7] Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixartδ: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252, 2024.
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In International Conference on Learning Representations, 2024. 2
- [9] Minkyoung Cho, Ruben Ohana, Christian Jacobsen, Adityan Jothi, Min-Hung Chen, Z Morley Mao, and Ethem Can. Tc-lora: Temporally modulated conditional lora for adaptive diffusion control. arXiv preprint arXiv:2510.09561, 2025. 3
- [10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780– 8794, 2021. 2
- [11] DiffSynth-Studio. Qwen-image-self-generateddataset. https://www.modelscope.cn/ datasets / DiffSynth - Studio / Qwen -

- Image - Self - Generated - Dataset, 2025. Accessed: 2025-09-24. 7
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206,

2024. 2, 3

- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024. 2
- [14] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024. 1, 2
- [15] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 1, 2
- [16] Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, Linus, Di Wang, and Jie Jiang. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again, 2025. 5
- [17] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152,

2023. 5

- [18] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 3
- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 5
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2, 3
- [21] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 3
- [23] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 5
- [24] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image gen-

- eration. Advances in Neural Information Processing Systems, 36:78723–78747, 2023. 3
- [25] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. NeurIPS, 35:26565–26577, 2022. 1
- [26] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023. 1, 2
- [27] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 5
- [28] Han Lin, Jaemin Cho, Abhay Zala, and Mohit Bansal. Ctrl-adapter: An efficient and versatile framework for adapting diverse controls to any diffusion model. arXiv preprint arXiv:2404.09967, 2024. 3
- [29] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation, 2024. 5
- [30] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 5
- [31] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In ECCV, 2024. 5
- [32] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 1
- [33] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. arXiv preprint arXiv:2202.09778, 2022. 1
- [34] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 1, 2, 3
- [35] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022. 1
- [36] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 1
- [37] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021. 2
- [38] Mengyao Lyu, Yuhong Yang, Haiwen Hong, Hui Chen, Xuan Jin, Yuan He, Hui Xue, Jungong Han, and Guiguang Ding. One-dimensional adapter to rule them all: Concepts diffusion models and erasing applications. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7559–7568, 2024. 3
- [39] ModelTC. Qwen-Image-Lightning: Speed up qwen-image model with distillation. https :

/ / github . com / ModelTC / Qwen - Image Lightning, 2025. Accessed: 2025-11-07. 1, 3

- [40] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial In-

- telligence, pages 4296–4304, 2024. 2

[41] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial In-

- telligence, pages 4296–4304, 2024. 3

- [42] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, Ming-Chang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024. 3
- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2023. 2
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 5
- [45] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 2
- [46] Lingmin Ran, Xiaodong Cun, Jia-Wei Liu, Rui Zhao, Song Zijie, Xintao Wang, Jussi Keppo, and Mike Zheng Shou. X-adapter: Adding universal compatibility of plugins for upgraded diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8775– 8784, 2024. 3
- [47] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, XING WANG, and Xuefeng Xiao. Hyper-SD: Trajectory segmented consistency model for efficient image synthesis. In NeurIPS, 2024. 6
- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [49] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 1, 2
- [50] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–

103. Springer, 2024. 1, 2

- [51] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In ECCV, pages 87–103, 2024. 6
- [52] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised

- learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015. 1, 2
- [53] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32,

2019. 1

- [54] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020. 2
- [55] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023. 1, 2
- [56] ModelScope Team. Diffsynth-studio: Enjoy the magic of diffusion models! https://github. com/modelscope/DiffSynth-Studio, 2025. GitHub repository, accessed: 2025-09-13. 3, 5
- [57] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 2, 5, 6
- [58] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis,

2023. 5

- [59] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023. 5
- [60] Enze Xie, Lewei Yao, Han Shi, Zhili Liu, Daquan Zhou, Zhaoqiang Liu, Jiawei Li, and Zhenguo Li. Difffit: Unlocking transferability of large diffusion models via simple parameter-efficient fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4230–4239, 2023. 3
- [61] Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, and Yu-Gang Jiang. Simda: Simple diffusion adapter for efficient video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7827–7839, 2024.
- [62] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023. 3
- [63] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024. 1, 2
- [64] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution

- matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024. 3
- [65] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 1, 2
- [66] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836– 3847, 2023. 2, 3
- [67] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024. 1, 2
- [68] Shaobin Zhuang, Yiwei Guo, Yanbo Ding, Kunchang Li, Xinyuan Chen, Yaohui Wang, Fangyikang Wang, Ying Zhang, Chen Li, and Yali Wang. Timestep master: Asymmetrical mixture of timestep lora experts for versatile and efficient diffusion models in vision. arXiv preprint arXiv:2503.07416, 2025. 3

## Glance: Accelerating Diffusion Models with 1Sample

[Figure 116]

### Supplementary Material

#### Contents

- A. Qwen-Image-Edit task 12

- A.1. Training Setup . . . . . . . . . . . . . 12
- A.2. Results and Analysis . . . . . . . . . . 12

- B. Remote Sensing Domain Speedup 12

- B.1. Zero-Shot Behavior . . . . . . . . . . 12
- B.2. One-Shot Slow-Fast LoRA Adaptation 13

- C. More Implementation details 13

- C.1. Glance (Qwen-Image) . . . . . . . . . 13
- C.2. Glance (FLUX) . . . . . . . . . . . . . 13

- D. More Qualitative Results 14

- D.1. Glance (Qwen-Image) . . . . . . . . . 14
- D.2. Glance (FLUX) . . . . . . . . . . . . . 14

- E. Future Work 14

#### A. Qwen-Image-Edit task

Beyond the text-to-image setting, a natural question is whether our phase-aware distillation strategy can transfer to other generative tasks. Motivated by this curiosity, we further evaluate our approach on imageediting using the Qwen-Image-Edit model.

##### A.1. Training Setup

Our training configuration closely follows the setup used for Qwen-Image experiments. We train a new pair of Slow-LoRA and Fast-LoRA adapters on the base Qwen-Image-Edit model, while adjusting the learning rate to 2e-4 to account for the different task dynamics. Following the one-shot paradigm, we train the LoRA experts on a single image, shown in Fig. 8, with the editing instruction “Put a hat on the woman’s head.”

##### A.2. Results and Analysis

Despite being trained on only one image, our method exhibits surprisingly strong generalization on the Qwen-Image-Edit task. Across a broad set of test samples, the edited images consistently follow the prompt: the model accurately places a hat on the target person while leaving the rest of the scene intact. This ability to preserve non-edited regions demonstrates that the Slow- and Fast-LoRA experts successfully adapt the model’s denoising phases without degrading spatial consistency.

Interestingly, the model does not simply copy the hat from the training image. Instead, it generates varied, context-appropriate hats for different individuals—evidence that the distilled LoRA experts capture

[Figure 117]

[Figure 118]

Training Sample

Put a hat on the fox's head

[Figure 119]

[Figure 120]

Put a hat on the girl's head

[Figure 121]

[Figure 122]

Put a hat on the girl's head

Inference Sample

[Figure 123]

[Figure 124]

Put a hat on

the boy's head

[Figure 125]

[Figure 126]

Put glasses on the girl

Figure 8. Training and inference examples for the one-shot Qwen-Image-Edit adaptation.

high-level editing semantics rather than memorizing pixel-level details.

To further test the flexibility of our approach, we modify the instruction to “Put glasses on the girl.” Even with this completely new editing concept, the model performs accurately and robustly, inserting realistic glasses while maintaining the appearance and identity of the original subject. The strong performance under unseen editing instructions highlights the task-agnostic nature of our distillation strategy.

Overall, these findings suggest that the proposed phase-aware LoRA acceleration is not limited to textto-image synthesis; it naturally extends to other generative domains with minimal adaptation. In future work, we plan to explore broader applications, including controllable generation, inpainting, and video editing, to further uncover the potential of this lightweight and generalizable distillation framework.

#### B. Remote Sensing Domain Speedup

##### B.1. Zero-Shot Behavior

One-shot acceleration in Glance produced surprisingly strong results on common natural images, which motivated us to explore domains where data availabil-

ity is inherently limited. Remote sensing imagery is a representative low-resource domain: its acquisition is expensive, often restricted by geographic or institutional constraints, and large-scale datasets are difficult to obtain. This unique setting raises a natural question—can phase-aware acceleration with Glance generalize to such domains without requiring extensive retraining?

To investigate this, we directly applied a Glance model trained only on random natural images to remote sensing image generation. As shown in Fig. 9, the model produces visually plausible outputs and maintains high-fidelity structures. However, the generated perspectives resemble natural images rather than true overhead remote-sensing viewpoints. This observation indicates that while Glance generalizes appearance, zero-shot adaptation is insufficient to inject domain-specific geometric priors.

##### B.2. One-Shot Slow-Fast LoRA Adaptation

To address this, we performed a targeted adaptation by training the Slow-LoRA and Fast-LoRA experts using only a single remote sensing sample ((Fig. 9). After this minimal fine-tuning, the adapted Glance model exhibits a remarkable change in behavior. When tested on unseen prompts, it consistently produces images with proper aerial viewpoints and structural layouts characteristic of remote sensing photography. This demonstrates that the model successfully captures domain-specific priors from just one example, while maintaining the acceleration benefits provided by the phase-aware LoRA design. The strong generalization achieved with minimal supervision highlights the effectiveness of Slow–Fast LoRA in injecting domain-specific knowledge without sacrificing speed or requiring large-scale datasets.

For reference, the single training example used for adaptation is shown in Fig. 9. Its detailed description is as follows: A high-resolution overhead satellite image of a large urban roundabout intersected by an elevated roadway. A wide overpass runs diagonally across the scene from bottom-left to top-right, casting a long shadow onto the circular traffic island below. Beneath the overpass lies a landscaped roundabout with grass, shrubs, and groups of small trees arranged in patches. At the center of the roundabout is a paved triangular plaza with a statue or vertical monument casting a distinct shadow. Multiple cars of various colors—blue, white, black, and red—drive along the circular lanes around the roundabout, following curved road markings. Surrounding the intersection are urban buildings, parking lots, and paved sidewalks. The lighting suggests clear weather and midday sun, creating sharp shadows from vehicles, trees, and the elevated bridge.

Training

Inference

Sample

Sample

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

Qwen-Image

[Figure 138]

[Figure 139]

[Figure 140]

Ground Truth

Figure 9. Qualitative results from the one-sample training setting. After observing only a single remote sensing example, Glance adapts effectively and begins generating images that exhibit correct aerial viewpoints and characteristics consistent with real remote sensing imagery.

#### C. More Implementation details

##### C.1. Glance (Qwen-Image)

For Qwen-Image, in Slow-LoRA and Fast-LoRA, we apply LoRA to a broad set of multi-modal projection and modulation layers to ensure effective lowrank adaptation. Specifically, LoRA adapters are injected into the to q, to k, to v, add q proj, add k proj, add v proj, and to out.0 modules of the MM-DiT blocks. In addition, LoRA is also placed on the multimodal MLP and modulation pathways, including to add out, img mlp.net.2, img mod.1, txt mlp.net.2, and txt mod.1.

We set the LoRA rank and scaling parameter to r = 32 and α = 128, respectively, and follow the Gaussian initialization strategy for all LoRA weight matrices.

During training, we use a learning rate of 3×10−4 with a constant schedule. The optimizer is AdamW with β1 = 0.9, β2 = 0.999, weight decay of 10−2, and ϵ = 10−8. Training is performed with a global batch size of 1, mixed-precision (bf16), and gradient clipping of 1.0. We train for a total of 60 steps, which corresponds to effectively training the single data sample for 60 epochs, and enable 8-bit Adam and quantized weight loading to reduce memory footprint. Both image and text embeddings are precomputed to accelerate training.

##### C.2. Glance (FLUX)

For FLUX, in Slow-LoRA and Fast-LoRA, we employ LoRA on the to q, to k, to v and to out.0 modules of the MM-DiT.

We set the LoRA rank and scaling parameter to r = 16 and α = 64, respectively, and follow the Gaussian initialization strategy for all LoRA weight matrices.

During training, we use a learning rate of 5×10−4 with a constant schedule. The optimizer is AdamW with β1 = 0.9, β2 = 0.999, weight decay of 10−2, and ϵ = 10−8. Training is performed with a global batch size of 1, mixed-precision (bf16), and gradient clipping of 1.0. We train for a total of 60 steps, which corresponds to effectively training the single data sample for 60 epochs, and enable 8-bit Adam and quantized weight loading to reduce memory footprint.

#### D. More Qualitative Results

- D.1. Glance (Qwen-Image)

We show additional uncurated results of Glance (Qwen) in Fig. 10 and 11.

- D.2. Glance (FLUX)

We show additional uncurated results of Glance (FLUX) in Fig. 12, Fig. 13 and 14.

#### E. Future Work

Although our phase-aware acceleration framework already achieves strong performance with extremely lightweight adaptation, several promising directions remain open for exploration.

Dynamic Expert Switching Beyond Hard SNR Thresholds. In the current design, Glance employs a hard switch between Slow-LoRA and Fast-LoRA based solely on the SNR-based phase boundary. While effective, this strategy does not account for prompt-dependent difficulty. A more adaptive alternative is to dynamically adjust the switching point according to the complexity of the generation task. For challenging prompts that involve intricate structures or fine-grained semantics, the model could remain longer in the slow-denoising phase to preserve fidelity. Conversely, for simpler prompts, the model could transition earlier into the fast phase to maximize speedup. Learning such prompt-aware switching policies represents an exciting opportunity for further reducing inference cost while maintaining high visual quality.

Toward Zero-Shot Diffusion Distillation. Our method demonstrates that phase-aware LoRA with only a single training sample is already sufficient to capture strong domain priors and generalize effectively. A natural next step is to explore whether Glance can be extended to a complete zero-shot distillation setting. This would involve leveraging intrinsic diffusion priors, self-consistency constraints, or synthetic trajectories generated by the model itself, enabling fully data-free adaptation. Achieving robust zero-shot distillation would further push the boundary

of efficient diffusion acceleration and enable deployment in domains where even a single reference example is unavailable.

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

Figure 10. An uncurated random batch from the OneIG-Bench prompt set.

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

Figure 11. An uncurated random batch from the HPSv2 prompt set.

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

FLUX.1 dev 𝟓𝟎 NFE

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

FLUX.1 dev 𝟏𝟎 NFE

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

FLUX.1 dev

𝟖 NFE

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

FLUX Turbo

𝟖 NFE

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Hyper-FLUX 𝟖 NFE

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Glance 𝟖 NFE

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Glance 𝟏𝟎 NFE

Figure 12. Visual comparison of different Slow–Fast configurations. All images are generated from the same initial noise using the 50-step base model, our 8/10-step students, and other few-step models. Slow-Fast preserves semantic fidelity under strong acceleration, while additional steps progressively enhance fine details.

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

[Figure 293]

[Figure 294]

Figure 13. An uncurated random batch from the OneIG-Bench prompt set.

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

[Figure 328]

[Figure 329]

Figure 14. An uncurated random batch from the DPG prompt set.

