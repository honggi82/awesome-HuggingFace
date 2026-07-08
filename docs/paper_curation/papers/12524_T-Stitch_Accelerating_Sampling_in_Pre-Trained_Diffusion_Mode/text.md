## T-Stitch: Accelerating Sampling in Pre-Trained Diffusion Models with Trajectory Stitching

# arXiv:2402.14167v1[cs.CV]21Feb2024

Zizheng Pan1* Bohan Zhuang1 De-An Huang2 Weili Nie2 Zhiding Yu2 Chaowei Xiao23 Jianfei Cai1 Anima Anandkumar4

### Abstract

Sampling from diffusion probabilistic models (DPMs) is often expensive for high-quality image generation and typically requires many steps with a large model. In this paper, we introduce sampling Trajectory Stitching (T-Stitch), a simple yet efficient technique to improve the sampling efficiency with little or no generation degradation. Instead of solely using a large DPM for the entire sampling trajectory, T-Stitch first leverages a smaller DPM in the initial steps as a cheap drop-in replacement of the larger DPM and switches to the larger DPM at a later stage. Our key insight is that different diffusion models learn similar encodings under the same training data distribution and smaller models are capable of generating good global structures in the early steps. Extensive experiments demonstrate that T-Stitch is training-free, generally applicable for different architectures, and complements most existing fast sampling techniques with flexible speed and quality trade-offs. On DiT-XL, for example, 40% of the early timesteps can be safely replaced with a 10x faster DiT-S without performance drop on class-conditional ImageNet generation. We further show that our method can also be used as a drop-in technique to not only accelerate the popular pretrained stable diffusion (SD) models but also improve the prompt alignment of stylized SD models from the public model zoo. Code is released at https://github.com/NVlabs/T-Stitch.

### 1. Introduction

Diffusion probabilistic models (DPMs) (Ho et al., 2020) have demonstrated remarkable success in generating highquality data among various real-world applications, such

∗Work done during an internship at NVIDIA. 1Monash University 2NVIDIA 3University of Wisconsin, Madison 4Caltech. Correspondence to: Bohan Zhuang <bohan.zhuang@gmail.com>.

Preprint.

as text-to-image generation (Rombach et al., 2022), audio synthesis (Kong et al., 2021) and 3D generation (Poole et al., 2023), etc. Achieving high generation quality, however, is expensive due to the need to sample from a large DPM, typically involving hundreds of denoising steps, each of which requires a high computational cost. For example, even with a high-performance RTX 3090, generating 8 images with DiT-XL (Peebles & Xie, 2022) takes 16.5 seconds with 100 denoising steps, which is ∼ 10× slower than its smaller counterpart DiT-S (1.7s) with a lower generation quality.

Recent works tackle the inference efficiency issue by speeding up the sampling of DPMs in two ways: (1) reducing the computational costs per step or (2) reducing the number of sampling steps. The former approach can be done by model compression through quantization (Li et al., 2023b) and pruning (Fang et al., 2023), or by redesigning lightweight model architectures (Yang et al., 2023; Lee et al., 2023). The second approach reduces the number of steps either by distilling multiple denoising steps into fewer ones (Salimans & Ho, 2022; Song et al., 2023; Zheng et al., 2023; Luo et al., 2023; Sauer et al., 2023) or by improving the differential equation solver (Song et al., 2021a; Lu et al., 2022; Zheng et al., 2023). While both directions can improve the efficiency of large DPMs, they assume that the computational cost of each denoising step remains the same, and a single model is used throughout the process. However, we observe that different steps in the denoising process exhibit quite distinct characteristics, and using the same model throughout is a suboptimal strategy for efficiency.

Our Approach. In this work, we propose Trajectory Stitching (T-Stitch), a simple yet effective strategy to improve DPMs’ efficiency that complements existing efficient sampling methods by dynamically allocating computation to different denoising steps. Our core idea is to apply DPMs of different sizes at different denoising steps instead of using the same model at all steps, as in previous works. We show that by first applying a smaller DPM in the early denoising steps followed by switching to a larger DPM in the later denoising steps, we can reduce the overall computational costs without sacrificing the generation quality. Figure 1 shows an example of our approach using two DiT models

40

| | | | | | | | | |30.|11<br><br>33.|46|
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | |18.|04<br><br>25.|44| | |
|9.2|0 9.1|7 8.9|9 9.0|3 8.9|5 10.|06<br><br>12.|46| | | | |
| | | | | | | | | | | | |

30

FID

20

10

0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

#DiT-S Steps / Total Steps

[Figure 1]

16.5s 15.3s 13.9s 12.5s 11.0s 9.4s 7.9s 6.4s 4.8s 3.3s 1.7s

- Figure 1. Top: FID comparison on class-conditional ImageNet when progressively stitching more DiT-S steps at the beginning and fewer DiT-XL steps in the end, based on DDIM 100 timesteps and a classifier-free guidance scale of 1.5. FID is calculated by sampling 5000 images. Bottom: One example of stitching more DiT-S steps to achieve faster sampling, where the time cost is measured by generating 8 images on one RTX 3090 in seconds (s).

Ghibli Diffusion 40% T-Stitch InkPunk Diffusion 40% T-Stitch

A ghibli style princess with golden hair in New York City A photo of a white cat on a tropical beach, nvinkpunk style

Small SD Small SD

3.1s 2.6s 1.9s 3.1s 2.6s 1.9s

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

- Figure 2. By directly adopting a small SD in the model zoo, T-Stitch naturally interpolates the speed, style, and image contents with a large styled SD, which also potentially improves the prompt alignment, e.g., “New York City” and “tropical beach” in the above examples.

(DiT-S and DiT-XL), where DiT-S is computationally much cheaper than DiT-XL. With the increase in the percentage of steps from DiT-S instead of DiT-XL in our T-stitch, we can keep increasing the inference speed. In our experiments, we find that there is no degradation of the generation quality (in FID), even when the first 40% of steps are using DiT-S, leading to around 1.5× lossless speedup.

Our method is based on two key insights: (1) Recent work suggests a common latent space across different DPMs trained on the same data distribution (Song et al., 2021b; Roeder et al., 2021). Thus, different DPMs tend to share similar sampling trajectories, which makes it possible to stitch across different model sizes and even architectures. (2) From the frequency perspective, the denoising process focuses on generating low-frequency components at the early steps while the later steps target the high-frequency signals (Yang et al., 2023). Although the small models are not as effective for high-frequency details, they can still generate a good global structure at the beginning.

With comprehensive experiments, we demonstrate that TStitch substantially speeds up large DPMs without much loss of generation quality. This observation is consistent across a spectrum of architectures and diffusion model samplers. This also implies that T-Stitch can be directly applied

to widely used large DPMs without any re-training (e.g., Stable Diffusion (SD) (Rombach et al., 2022)). Figure 2 shows the results of speeding up stylized Stable Diffusion with a relatively smaller pretrained SD model (Kim et al., 2023). Surprisingly, we find that T-Stitch not only improves speed but also improves prompt alignment for stylized models. This is possibly because the fine-tuning process of stylized models (e.g., ghibli, inkpunk) degrades their prompt alignment. T-Stitch improves both efficiency and generation quality here by combining small SD models to complement the prompt alignment for large SD models specialized in stylizing the image.

Note that T-Stitch is complementary to existing fast sampling approaches. The part of the trajectory that is taken by the large DPM can still be sped up by reducing the number of steps taken by it, or by reducing its computational cost with compression techniques. In addition, while T-Stitch can already effectively improve the quality-efficiency tradeoffs without any overhead of re-training, we show that the generation quality of T-Stitch can be further improved when we fine-tune the stitched DPMs given a trajectory schedule (Section A.12). By fine-tuning the large DPM only on the timesteps that it is applied, the large DPM can better specialize in providing high-frequency details and further improve generation quality. Furthermore, we show that the

training-free Pareto frontier generated by T-Stitch improves quality-efficiency trade-offs to training-based methods designed for interpolating between neural network models via model stitching (Pan et al., 2023a;b). Note that T-Stitch is not limited to only two model sizes, and is also applicable to different DPM architectures.

We summarize our main contributions as follows:

- • We propose T-Stitch, a simple yet highly effective approach for improving the inference speed of DPMs, by applying a small DPM at early denoising steps while a large DPM at later steps. Without retraining, we achieve better speed and quality trade-offs than individual large DPMs and even non-trivial lossless speedups.
- • We conduct extensive experiments to demonstrate that our method is generally applicable to different model architectures and samplers, and is complementary to existing fast sampling techniques.
- • Notably, without any re-training overhead, T-Stitch not only accelerates Stable Diffusion models that are widely used in practical applications but also improves the prompt alignment of stylized SD models for textto-image generation.

### 2. Related Works

Efficient diffusion models. Despite the success, DPMs suffer from the slow sampling speed (Rombach et al., 2022; Ho et al., 2020) due to hundreds of timesteps and the large denoiser (e.g., U-Net). To expedite the sampling process, some efforts have been made by directly utilizing network compression techniques to diffusion models, such as pruning (Fang et al., 2023) and quantization (Shang et al., 2023; Li et al., 2023b). On the other hand, many works seek for reducing sampling steps, which can be achieved by distillation (Salimans & Ho, 2022; Zheng et al., 2023; Song et al., 2023; Luo et al., 2023; Sauer et al., 2023), implicit sam-

- pler (Song et al., 2021a), and improved differential equation (DE) solvers (Lu et al., 2022; Song et al., 2021b; JolicoeurMartineau et al., 2021; Liu et al., 2022). Another line of work also considers accelerating sampling by parallel sampling. For example, (Zheng et al., 2023) proposed to utilize operator learning to simultaneously predict all steps. (Shih et al., 2023) proposed ParaDiGMS to compute the drift at multiple timesteps in parallel. As a complementary technique to the above methods, our proposed trajectory stitching accelerates large DPM sampling by leveraging pretrained small DPMs at early denoising steps, while leaving sufficient space for large DPMs at later steps.

Multiple experts in diffusion models. Previous observations have revealed that the synthesis behavior in DPMs can change at different timesteps (Balaji et al., 2022; Yang et al.,

2023), which has inspired some works to propose an ensemble of experts at different timesteps for better performance. For example, (Balaji et al., 2022) trained an ensemble of expert denoisers at different denoising intervals. However, allocating multiple large denoisers linearly increases the model parameters and does not reduce the computational cost. (Yang et al., 2023) proposed a lite latent diffusion model (i.e., LDM) which incorporates a gating mechanism for the wavelet transform in the denoiser to control the frequency dynamics at different steps, which can be regarded as an ensemble of frequency experts. Following the same spirit, (Lee et al., 2023) allocated different small denoisers at different denoising intervals to specialize on their respective frequency ranges. Nevertheless, most existing works adopt the same-sized model over all timesteps, which barely consider the speed and quality trade-offs between different-sized models. In contrast, we explore a flexible trade-off between small and large DPMs and reveal that the early denoising steps can be sufficiently handled by a much efficient small DPM.

Stitchable neural networks. Stitchable neural networks (SN-Net) (Pan et al., 2023a) is motivated by the idea of model stitching (Lenc & Vedaldi, 2015; Bansal et al., 2021; Csisz´arik et al., 2021; Yang et al., 2022), where the pretrained models of different scales within a pretrained model family can be splitted and stitched together with simple stitching layers (i.e., 1 × 1 convs) without a significant performance drop. Based on the insight, SN-Net inserts a few stitching layers among models of different sizes and applies joint training to obtain numerous networks (i.e., stitches) with different speed-performance trade-offs. The following work of SN-Netv2 (Pan et al., 2023b) enlarges its space and demonstrates its effectiveness on downstream dense prediction tasks. In this work, we compare our technique with SN-Netv2 to show the advantage of trajectory stitching over model stitching in terms of the speed and quality trade-offs in DPMs. Our T-Stitch is a better, simpler and more general solution.

### 3. Method

#### 3.1. Preliminary

Diffusion models. We consider the class of score-based diffusion models in a continuous time (Song et al., 2021b) and following the presentation from (Karras et al., 2022). Let pdata(x0) denote the data distribution and σ(t): [0,1] → R+ is a user-specified noise level schedule, where t ∈ {0,...,T} and σ(t−1) < σ(t). Let p(x;σ) denote the distribution of noised samples by injecting σ2-variance Gaussian noise. Starting with a high-variance Gaussian noise xT, diffusion models gradually denoise xT into less noisy samples {xT−1,xT−2,...,x0}, where xt ∼ p(xt;σ(t)). Furthermore, this iterative process can be done by solving the

probability flow ordinary differential equation (ODE) if knowing the score ∇x log pt(x), namely the gradient of the log probability density with respect to data,

dx = −σˆ(t)σ(t)∇x log p(x;σ(t))dt, (1)

where σˆ(t) denote the time derivative of σ(t). Essentially, diffusion models aim to learn a model for the score function, which can be reparameterized as

∇x log pt(x) ≈ (Dθ(x;σ) − x)/σ2, (2)

where Dθ(x;σ) is the learnable denoiser. Given a noisy data point x0 + n and a conditioning signal c, where n ∼ N 0,σ2I , the denoiser aim to predict the clean data x0. In practice, the mode is trained by minimizing the loss of denoising score matching,

0,c)∼pdata,(σ,n)∼p(σ,n) λσ∥Dθ(x0 + n;σ,c) − x0∥22 , (3)

E(x

where λσ : R+ → R+ is a weighting function (Ho et al.,

- 2020), p(σ,n) = p(σ)N n;0,σ2 , and p(σ) is a distribution over noise levels σ.

This work focuses on the denoisers D in diffusion models. In common practice, they are typically large parameterized neural networks with different architectures that consume high FLOPs at each timestep. In the following, we use “denoiser” or “model” interchangeably to refer to this network. We begin with the pretrained DiT model family to explore the advantage of trajectory stitching on efficiency gain. Then we show our method is a general technique for other architectures, such as U-Net (Rombach et al., 2022) and U-ViT (Bao et al., 2023).

Classifier-free guidance. Unlike classifier-based denoisers (Dhariwal & Nichol, 2021) that require an additional network to provide conditioning guidance, classifier-free guidance (Ho & Salimans, 2022) is a technique that jointly trains a conditional model and an unconditional model in one network by replacing the conditioning signal with a null embedding. During sample generation, it adopts a guidance scale s ≥ 0 to guide the sample to be more aligned with the conditioning signal by jointly considering the predictions from both conditional and unconditional models,

Ds(x;σ,c) = (1 + s)D(x;σ,c) − sD(x;σ). (4)

Recent works have demonstrated that classifier-free guidance provides a clear improvement in generation quality. In this work, we consider the diffusion models that are trained with classifier-free guidance due to their popularity.

- 3.2. Trajectory Stitching

Why can different pretrained DPMs be directly stitched along the sampling trajectory? First of all, DPMs from the

same model family usually takes the latent noise inputs and outputs of the same shape, (e.g., 4×32×32 in DiTs). There is no dimension mismatch when applying different DPMs at different denoising steps. More importantly, as pointed out in (Song et al., 2021b), different DPMs that are trained on the same dataset often learn similar latent embeddings. We observe that this is especially true for the latent noises at early denoising sampling steps, as shown in Figure 3, where the cosine similarities between the output latent noises from different DiT models reach almost 100% at early steps. This motivates us to propose Trajectory Stitching (T-Stitch), a novel step-level stitching strategy that leverages a pretrained small model at the beginning to accelerate the sampling speed of large diffusion models.

Principle of model selection. Figure 4 shows the framework of our proposed T-Stitch for different speed-quality tradeoffs. In principle, the fast speed or worst generation quality we can achieve is roughly bounded by the smallest model in the trajectory, whereas the slowest speed or best generation quality is determined by the largest denoiser. Thus, given a large diffusion model that we want to speed up, we select a small model that is 1) clearly faster, 2) sufficiently optimized, and 3) trained on the same dataset as the large model or at least they have learned similar data distributions (e.g., pretrained or finetuned stable diffusion models).

Pairwise model allocation. By default, T-Stitch adopts a pairwise denoisers in the sampling trajectory as it performs very well in practice. Specifically, we first define a denoising interval as a range of sampling steps in the trajectory, and the fraction of it over the total number of steps T is denoted as r, where r ∈ [0,1]. Next, we treat the model allocation as a compute budget allocation problem. From Figure 3, we observe that the latent similarity between different scaled denoisers keeps decreasing when T flows to 0. To this end, our allocation strategy adopts a small denoiser as a cheap replacement at the initial intervals then applies the large denoiser at the later intervals. In particular, suppose we have a small denoiser D1 and a large denoiser D2. Then we let D1 take the first ⌊r1T⌉ steps and D2 takes the last ⌊r2T⌉ steps, where ⌊·⌉ denotes a rounding operation and r2 = 1 − r1. By increasing r1, we naturally interpolate the compute budget between the small and large denoiser and thus obtain flexible quality and efficiency trade-offs. For example, in Figure 1, the configuration r1 = 0.5 uniquely defines a trade-off where it achieves 10.06 FID and 1.76× speedup.

More denoisers for more trade-offs. Note that T-Stitch is not limited to the pairwise setting. In fact, we can adopt more denoisers in the sampling trajectory to obtain more speed and quality trade-offs and a better Pareto frontier. For example, by using a medium sized denoiser in the interme-

DiT-XL/S

DiT-XL/B

DiT-B/S

1.00

1.00

1.00

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

CosineSimilarity

CosineSimilarity

CosineSimilarity

0.80

0.80

0.80

0.60

0.60

0.60

0.40

0.40

0.40

100 75 50 25 0

100 75 50 25 0

100 75 50 25 0

Denoising steps

Denoising steps

Denoising steps

- Figure 3. Similarity comparison of latent embeddings at different denoising steps between different DiT models. Results are averaged over 32 images.

Large DPM

t = T t = 0

[Figure 8]

!

Small DPM

t = T t = 0

[Figure 9]

!

10% 20% 30% … 80%

- Figure 4. Trajectory Stitching (T-Stitch): Based on pretrained small and large DPMs, we can leverage the more efficient small DPM with different percentages at the early denoising sampling steps to achieve different speed-quality trade-offs.

diate interval, we can change the fractions of each denoiser to obtain more configurations. In practice, given a compute budget such as time cost, we can efficiently find a few configurations that satisfy this constraint via a pre-computed lookup table, as discussed in Section A.1.

Remark. Compared to existing multi-experts DPMs, TStitch directly applies models of different sizes in a pretrained model family. Thus, given a compute budget, we consider how to allocate different resources across different steps while benefiting from training-free. Furthermore, speculative decoding (Leviathan et al., 2023) shares a similar motivation with us, i.e., leveraging a small model to speed up large language model sampling. However, this technique is specifically designed for autoregressive models, whereas it is not straightforward to apply the same sampling strategy to diffusion models. On the other hand, our method utilizes the DPM’s property and achieves effective speedup.

### 4. Experiments

In this section, we first show the effectiveness of T-Stitch based on DiT (Peebles & Xie, 2022) as it provides a convenient model family. Then we extend into U-Net and Stable Diffusion models. Last, we ablate our technique with different sampling steps, and samplers to demonstrate that T-Stitch is generally applicable in many scenarios.

- 4.1. DiT Experiments Implementation details. Following DiT, we conduct the class-conditional ImageNet experiments based on pretrained DiT-S/B/XL under 256×256 images and patch size of 2. A detailed comparison of the pretrained models is shown in Table 3. As T-Stitch is training-free, for two-model setting, we directly allocate the models into the sampling trajectory

under our allocation strategy described in Section 3.2. For three-model setting, we enumerate all possible configuration sets by increasing the fraction by 0.1 per model one at a time, which eventually gives rise to 66 configurations that include pairwise combinations of DiT-S/XL, DiT-S/B, DiT-S/XL, and three model combinations DiT-S/B/XL. By default, we adopt a classifier-free guidance scale of 1.5 as it achieves the best FID for DiT-XL, which is also the target model in our setting.

Evaluation metrics. We adopt Fr´echet Inception Distance (FID) (Heusel et al., 2017) as our default metric to measure the overall sample quality as it captures both diversity and fidelity (lower values indicate better results). Additionally, we report the Inception Score as it remains a solid performance measure on ImageNet, where the backbone Inception network (Szegedy et al., 2016) is pretrained. We use the reference batch from ADM (Dhariwal & Nichol, 2021) and sample 5,000 images to compute FID. In the supplementary material, we show that sampling more images (e.g., 50K) does not affect our observation. By default, the time cost is measured by generating 8 images on a single RTX 3090 in seconds.

Results. Based on the pretrained model families, we first apply T-Stitch with any two-model combinations, including DiT-XL/S, DiT-XL/B, and DiT-B/S. For each setting, we begin the sampling steps with a relatively smaller model and then let the larger model deal with the last timesteps. In Figure 5, we report the FID comparisons on different combinations. In general, we observe that using a smaller model at the early 40-50% steps brings a minor performance drop for all combinations. Besides, the best/worst performance is roughly bounded by the smallest and largest models in the

DiT-XL/S

DiT-XL/B

DiT-B/S

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 9
- 10
- 11
- 12

30

30

25

FID

FID

FID

20

20

15

10

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

Fraction of Smaller Net

Fraction of Smaller Net

Fraction of Smaller Net

- Figure 5. T-Stitch of two model combinations: DiT-XL/S, DiT-XL/B and DiT-B/S. We adopt DDIM 100 timesteps with a classifier-free guidance scale of 1.5.

|DiT-S| | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | |DiT-XL|
| | | | |

5 10 15

Time Cost (s)

10

15

20

25

30

FID

5 10 15

Time Cost (s)

50

100

150

200

250

InceptionScore

DiT-S

DiT-XL

- Figure 6. T-Stitch based on three models: DiT-S, DiT-B and DiTXL. We adopt DDIM 100 timesteps with a classifier-free guidance scale of 1.5. We highlight the Pareto frontier in lines.

At the same time, we observe an approximately linear decrease in time cost when progressively using more LDM-S steps in the trajectory. Overall, the U-Net experiment indicates that our method is applicable to different denoiser architectures. We provide the generated image examples in Section A.16 and also show that T-Stitch can be applied with even different model families in Section A.10.

#### 4.3. Text-to-Image Stable Diffusion

Benefiting from the public model zoo on Diffusers (von Platen et al., 2022), we can directly adopt a small SD model to accelerate the sampling speed of any large pretrained or styled SD models without any training. In this section, we show how to apply T-Stitch to accelerate existing SD models in the model zoo. Previous research from (Kim et al., 2023) has produced multiple SD models with different sizes by pruning the original SD v1.4 and then applying knowledge distillation. We then directly adopt the smallest model BK-SDM Tiny for our stable diffusion experiments. By default, we use a guidance scale of 7.5 under 50 steps using PNDM (Liu et al., 2022) sampler.

pretrained model family.

Furthermore, we show that T-Stitch can adopt a mediumsized model at the intermediate denoising intervals to achieve more speed and quality trade-offs. For example, built upon the three different-sized DiT models: DiT-S, DiTB, DiT-XL, we start with DiT-S at the beginning then use DiT-B at the intermediate denoising intervals, and finally adopt DiT-XL to draw fine local details. Figure 6 indicates that the three-model combinations effectively obtain a smooth Pareto Frontier for both FID and Inception Score. In particular, at the time cost of ∼10s, we achieve 1.7× speedups with comparable FID (9.21 vs. 9.19) and Inception Score (243.82 vs. 245.73). We show the effect of using different classifier-free guidance scales in Section A.4.

Results. In Table 2, we report the results by applying TStitch to the original SD v1.4. In addition to the FID and Inception Score, we also report the CLIP score for measuring the alignment of the image with the text prompt. Overall, we found the early 30% steps can be taken by BK-SDM Tiny without a significant performance drop in Inception Score and CLIP Scores while achieving even better FID. We believe a better and faster small model in future works can help to achieve better quality and efficiency trade-offs. Furthermore, we demonstrate that T-Stitch is compatible with other large SD models. For example, as shown in Figure 7, under the original SD v1.4, we achieve a promising speedup while obtaining comparable image quality. Moreover, with other stylized SD models such as Inkpunk style1, we observe a natural style interpolation between the two models. More importantly, by adopting a small fraction of steps from a general small SD, we found it helps the image to be more aligned with the prompt, such as the “park” in InkPunk Diffusion. In this case, we assume finetuning in these stylized SD may unexpectedly hurt prompt alignment,

#### 4.2. U-Net Experiments

In this section, we show T-Stitch is complementary to the architectural choices of denoisers. We experiment with prevalent U-Net as it is widely adopted in many diffusion models. We adopt the class-conditional ImageNet implementation from the latent diffusion model (LDM) (Rombach et al.,

- 2022). Based on their official implementation, we simply scale down the network channel width from 256 to 64 and the context dimension from 512 to 256. This modification produces a 15.8× smaller LDM-S. A detailed comparison between the two pretrained models is shown in Table 4.

Results. We report the results on T-Stitch with U-Net in Table 1. In general, under DDIM and 100 timesteps, we found the first ∼50% steps can be taken by an efficient LDM-S with comparable or even better FID and Inception Scores.

1https://huggingface.co/Envvi/Inkpunk-Diffusion

0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

[Figure 10]

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

SD-1.4

Prompt: “a vase with different flowers”

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

InkPunk

Prompt: “a squirrel in the park, nvinkpunk style”

- Figure 7. Based on a general pretrained small SD model, T-Stitch simultaneously accelerates a large general SD and complements the prompt alignment with image content when stitching other finetuned/stylized large SD models, i.e., “park” in InkPunk Diffusion. Better viewed when zoomed in digitally.

- Table 1. T-Stitch with LDM (Rombach et al., 2022) and LDM-S on class-conditional ImageNet. All evaluations are based on DDIM and 100 timesteps. We adopt a classifier-free guidance scale of 3.0. The time cost is measured by generating 8 images on one RTX 3090.

|Fraction of LDM-S<br><br>|0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%|
|---|---|
|FID Inception Score Time Cost (s)<br><br>|20.11 19.54 18.74 18.64 18.60 19.33 21.81 26.03 30.41 35.24 40.92 199.24 201.87 202.81 204.01 193.62 175.62 140.69 110.81 90.24 70.91 54.41 7.1 6.7 6.2 5.8 5.3 4.9 4.5 4.1 3.6 3.1 2.9|

- Table 2. T-Stitch with BK-SDM Tiny (Kim et al., 2023) and SD v1.4. We report FID, Inception Score (IS) and CLIP score (Hessel et al.,

- 2021) on MS-COCO 256×256 benchmark. The time cost is measured by generating one image on one RTX 3090.

|Fraction of BK-SDM Tiny<br><br>|0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%|
|---|---|
|FID Inception Score CLIP Score Time Cost (s)|13.07 12.59 12.29 12.54 13.65 14.98 15.69 16.57 16.92 16.80 17.15 36.72 36.12 34.66 33.32 32.48 31.72 31.48 30.83 30.53 30.48 30.00 0.2957 0.2957 0.2938 0.2910 0.2860 0.2805 0.2770 0.2718 0.2692 0.2682 0.2653 3.1 3.0 2.9 2.8 2.6 2.5 2.4 2.3 2.1 2.0 1.9<br><br>|

Effect of T-Stitch with different samplers. Here we show T-Stitch is also compatible with advanced samplers (Lu et al., 2022) for achieving better generation quality in fewer timesteps. To this end, we experiment with prevalent samplers to demonstrate the effectiveness of T-Stitch with these orthogonal techniques: DDPM (Ho et al., 2020), DDIM (Song et al., 2021a) and DPM-Solver++ (Lu et al., 2022). In Figure 8, we use the DiT-S to progressively replace the early steps of DiT-XL under different samplers and steps. In general, we observe a consistent efficiency gain at the initial sampling stage, which strongly justifies that our method is a complementary solution with existing samplers for accelerating DPM sampling.

while adopting the knowledge from a general pretrained SD can complement the early global structure generation. Overall, this strongly supports another practical usage of T-Stitch: Using a small general expert at the beginning for fast sketching and better prompt alignment, while letting any stylized SD at the later steps for patiently illustrating details. Furthermore, we show that T-Stitch is compatible with ControlNet, SDXL, LCM in Section A.11.

#### 4.4. Ablation Study

Effect of T-Stitch with different steps. To explore the efficiency gain on different numbers of sampling steps, we conduct experiments based on DDIM and DiT-S/XL. As shown in Figure 9, T-Stitch achieves consistent efficiency gain when using different number of steps and diffusion model samplers. In particular, we found the 40% early steps can be safely taken by DiT-S without a significant performance drop. It indicates that small DPMs can sufficiently handle the early denoising steps where they mainly generate the low-frequency components. Thus, we can leave the highfrequency generation of fine local details to a more capable DiT-XL. This is further evidenced by the generation exam-

T-Stitch vs. model stitching. Previous works (Pan et al., 2023a;b) such as SN-Net have demonstrated the power of model stitching for obtaining numerous architectures that with different complexity and performance trade-offs. Thus, by adopting one of these architectures as the denoiser for sampling, SN-Net naturally achieves flexible quality and efficiency trade-offs. To show the advantage of T-Stitch in the Pareto frontier, we conduct experiments to compare with the framework of model stitching proposed in SN-Netv2 (Pan et al., 2023b). We provide implementation details in Section A.8. In Figure 10, we compare T-Stitch with model stitching based on DDIM sampler and 100 steps. Overall, while both methods can obtain flexible speed and quality trade-offs, T-Stitch achieves clearly better advantage over

- ples in Figure 17, where we provide the sampled images at all fractions of DiT-S steps across different total number of steps. Overall, we demonstrate that T-Stitch is not competing but complementing other fast diffusion approaches that focus on reducing sampling steps.

10 Steps

20 Steps

50 Steps

100

60

40

DDPM DDIM DPM Solver++

DDPM DDIM DPM Solver++

DDPM DDIM DPM Solver++

80

30

40

FID

FID

FID

60

20

40

20

10

20

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

Fraction of DiT-S

Fraction of DiT-S

Fraction of DiT-S

Figure 8. Effect of T-Stitch with different samplers, under guidance scale of 1.5.

T = 10 T = 20 T = 50 T = 100 T = 250

T = 10 T = 20 T = 50 T = 100 T = 250

50

40

TimeCost(s)

40

30

FID

30

20

20

10

10

0

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

Fraction of DiT-S

Fraction of DiT-S

250

M-Stitch T-Stitch

M-Stitch T-Stitch

60

InceptionScore

200

150

FID

40

100

20

50

5 10 15

5 10 15

Time Cost (s)

Time Cost (s)

Figure 9. Left: We compare FID between different numbers of steps. Right: We visualize the time cost of generating 8 images under different number of steps, based on DDIM and a classifierguidance scale of 1.5. “T” denotes the number of sampling steps.

Figure 10. T-Stitch vs. model stitching (M-Stitch) (Pan et al., 2023b) based on DiTs and DDIM 100 steps, with a classifierfree guidance scale of 1.5.

model stitching across different metrics.

Compared to training-based acceleration methods. The widely adopted training-based methods for accelerating DPM sampling mainly include lightweight model design (Zhao et al., 2023; Lee et al., 2023), model compression (Kim et al., 2023), and steps distillation (Salimans & Ho, 2022; Song et al., 2023; Luo et al., 2023). Compared to them, T-Stitch is a training-free and complementary acceleration technique since it is agnostic to individual model optimization. In practice, T-Stitch achieves wide compatibility with different denoiser architectures (DiT and U-Net, Section 4.1 and Section 4.2), and any already pruned (Section A.7) or step-distilled models (Section A.18).

Compared to other training-free acceleration methods. Recent works (Li et al., 2023a; Ma et al., 2023; Wimbauer et al., 2023) proposed to cache the intermediate feature maps in U-Net during sampling for speedup. T-Stitch is also complementary to these cache-based methods since the individual model can still be accelerated with caching, as shown in Section A.19. In addition, T-Stitch can also enjoy the benefit from model quantization (Shang et al., 2023; Li et al., 2023b), VAE decoder acceleration (Kodaira et al.,

- 2023) and token merging (Bolya et al., 2023) (Section A.20) since they are orthogonal approaches.

### 5. Conclusion

We have proposed Trajectory Stitching, an effective and general approach to accelerate existing pretrained large diffusion model sampling by directly leveraging pretrained

smaller counterparts at the initial denoising process, which achieves better speed and quality trade-offs than using an individual large DPM. Comprehensive experiments have demonstrated that T-Stitch achieved consistent efficiency gain across different model architectures, samplers, as well as various stable diffusion models. Besides, our work has revealed the power of small DPMs at the early denoising process. Future work may consider disentangling the sampling trajectory by redesigning or training experts of different sizes at different denoising intervals. For example, designing a better, faster small DPM at the beginning to draw global structures, then specifically optimizing the large DPM at the later stages to refine image details. Besides, more guidelines for the optimal trade-off and more in-depth analysis of the prompt alignment for stylized SDs can be helpful, which we leave for future work.

Limitations. T-Stitch requires a smaller model that has been trained on the same data distribution as the large model. Thus, a sufficiently optimized small model is required. Besides, adopting an additional small model for denoising sampling will slightly increase memory usage (Section A.14). Lastly, since T-Stitch provides a free lunch from a small model for sampling acceleration, the speedup gain is bounded by the efficiency of the small model. In practice, we suggest using T-Stitch when a small model is available and much faster than the large model. As DPMs are scaling up in recent studies (Razzhigaev et al., 2023; Podell et al., 2023), we hope our research will inspire more explorations and adoptions in effectively utilizing efficient small models to complement those large models.

### Societal Impact

Our approach is built upon pretrained models from the public model zoo, thus it avoids training cost while speeding up diffusion model sampling for image generation, contributing to lowering carbon emissions during deployment. However, it is important to acknowledge that the generated images are determined by user prompts and the chosen diffusion models. Therefore, our work does not address potential privacy concerns or misuse of generative models, as these fall outside our current scope.

### References

Balaji, Y., Nah, S., Huang, X., Vahdat, A., Song, J., Kreis, K., Aittala, M., Aila, T., Laine, S., Catanzaro, B., Karras, T., and Liu, M. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. CoRR, abs/2211.01324, 2022.

Bansal, Y., Nakkiran, P., and Barak, B. Revisiting model stitching to compare neural representations. In NeurIPS, pp. 225–236, 2021.

Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., and Zhu, J. All are worth words: A vit backbone for diffusion models. In CVPR, 2023.

Bolya, D., Fu, C., Dai, X., Zhang, P., Feichtenhofer, C., and Hoffman, J. Token merging: Your vit but faster. In ICLR, 2023.

Csisz´arik, A., Kor¨osi-Szab´o, P., Matszangosz, A.´ K., Papp, G., and Varga, D. Similarity and matching of neural network representations. In NeurIPS, pp. 5656–5668, 2021.

Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. NeurIPS, 34:8780–8794, 2021.

Fang, G., Ma, X., and Wang, X. Structural pruning for diffusion models. arXiv preprint arXiv:2305.10924, 2023.

Hessel, J., Holtzman, A., Forbes, M., Bras, R. L., and Choi, Y. Clipscore: A reference-free evaluation metric for image captioning. In EMNLP, pp. 7514–7528, 2021.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, pp. 6626–6637, 2017.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. CoRR, abs/2207.12598, 2022.

Jolicoeur-Martineau, A., Li, K., Pich´e-Taillefer, R., Kachman, T., and Mitliagkas, I. Gotta go fast when generating data with score-based models. CoRR, abs/2105.14080, 2021.

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022.

Kim, B.-K., Song, H.-K., Castells, T., and Choi, S. Bk-sdm: Architecturally compressed stable diffusion for efficient text-to-image generation. ICML Workshop on Efficient Systems for Foundation Models (ES-FoMo), 2023.

Kodaira, A., Xu, C., Hazama, T., Yoshimoto, T., Ohno, K., Mitsuhori, S., Sugano, S., Cho, H., Liu, Z., and Keutzer, K. Streamdiffusion: A pipeline-level solution for realtime interactive generation. arXiv, 2023.

Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B. Diffwave: A versatile diffusion model for audio synthesis. In ICLR. OpenReview.net, 2021.

Lee, Y., Kim, J., Go, H., Jeong, M., Oh, S., and Choi, S. Multi-architecture multi-expert diffusion models. CoRR, abs/2306.04990, 2023.

Lenc, K. and Vedaldi, A. Understanding image representations by measuring their equivariance and equivalence. In CVPR, pp. 991–999, 2015.

Leviathan, Y., Kalman, M., and Matias, Y. Fast inference from transformers via speculative decoding. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), ICML, volume 202, pp. 19274–19286, 2023.

Li, S., Hu, T., Khan, F. S., Li, L., Yang, S., Wang, Y., Cheng, M., and Yang, J. Faster diffusion: Rethinking the role of unet encoder in diffusion models. arXiv, 2023a.

Li, X., Lian, L., Liu, Y., Yang, H., Dong, Z., Kang, D., Zhang, S., and Keutzer, K. Q-diffusion: Quantizing diffusion models. ICCV, 2023b.

Liu, L., Ren, Y., Lin, Z., and Zhao, Z. Pseudo numerical methods for diffusion models on manifolds. In ICLR, 2022.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpmsolver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022.

Luo, S., Tan, Y., Huang, L., Li, J., and Zhao, H. Latent consistency models: Synthesizing high-resolution images with few-step inference, 2023.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020.

Ma, X., Fang, G., and Wang, X. Deepcache: Accelerating diffusion models for free. arXiv, 2023.

Pan, Z., Cai, J., and Zhuang, B. Stitchable neural networks. In CVPR, 2023a.

Pan, Z., Liu, J., He, H., Cai, J., and Zhuang, B. Stitched vits are flexible vision backbones. arXiv, 2023b.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. CoRR, abs/2212.09748, 2022.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. SDXL: improving latent diffusion models for high-resolution image synthesis. CoRR, 2023.

Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR. OpenReview.net, 2023.

Razzhigaev, A., Shakhmatov, A., Maltseva, A., Arkhipkin, V., Pavlov, I., Ryabov, I., Kuts, A., Panchenko, A., Kuznetsov, A., and Dimitrov, D. Kandinsky: An improved text-to-image synthesis with image prior and latent diffusion. In EMNLP Demos, pp. 286–295, 2023.

Roeder, G., Metz, L., and Kingma, D. On linear identifiability of learned representations. In ICML, volume 139, pp. 9030–9039, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, pp. 10684–10695, 2022.

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. In ICLR. OpenReview.net, 2022.

Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. Adversarial diffusion distillation. arXiv, 2023.

Segmind. Segmind Stable Diffusion Model (SSD1B). https://huggingface.co/segmind/ SSD-1B, 2023.

Shang, Y., Yuan, Z., Xie, B., Wu, B., and Yan, Y. Posttraining quantization on diffusion models. CVPR, Jun 2023.

Shih, A., Belkhale, S., Ermon, S., Sadigh, D., and Anari, N. Parallel sampling of diffusion models. CoRR, abs/2305.16317, 2023.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In ICLR. OpenReview.net, 2021a.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. ICLR, 2021b.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. In ICML, volume 202, 2023.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the inception architecture for computer vision. In CVPR, pp. 2818–2826. IEEE Computer Society, 2016.

von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., and Wolf, T. Diffusers: State-of-the-art diffusion models. https://github. com/huggingface/diffusers, 2022.

Wimbauer, F., Wu, B., Sch¨onfeld, E., Dai, X., Hou, J., He, Z., Sanakoyeu, A., Zhang, P., Tsai, S. S., Kohler, J., Rupprecht, C., Cremers, D., Vajda, P., and Wang, J. Cache me if you can: Accelerating diffusion models through block caching. arXiv, 2023.

Yang, X., Zhou, D., Liu, S., Ye, J., and Wang, X. Deep model reassembly. NeurIPS, 2022.

Yang, X., Zhou, D., Feng, J., and Wang, X. Diffusion probabilistic model made slim. In CVPR, pp. 22552–

22562. IEEE, 2023.

Zhang, L., Rao, A., and Agrawala, M. Adding conditional control to text-to-image diffusion models. In ICCV, pp. 3836–3847, 2023.

Zhao, Y., Xu, Y., Xiao, Z., and Hou, T. Mobilediffusion: Subsecond text-to-image generation on mobile devices. arXiv, 2023.

Zheng, H., Nie, W., Vahdat, A., Azizzadenesheli, K., and Anandkumar, A. Fast sampling of diffusion models via operator learning. In ICML, pp. 42390–42402. PMLR, 2023.

### A. Appendix

We organize our supplementary material as follows.

- • In Section A.1, we provide guidelines for practical deployment of T-Stitch.
- • In Section A.2, we provide frequency analysis during denoising sapmling process based on DiTs.
- • In Section A.3, we report the details of our adopted pretrained DiTs and U-Nets.
- • In Section A.4, we show the effect of using different classifier-free guidance scales based on DiTs and T-Stitch.
- • In Section A.5, we compare FID evaluation under T-Stitch with 5,000 images and 50,000 images.
- • In Section A.6, we compare T-Stitch with directly reducing sampling steps.
- • In Section A.7, we show T-Stitch is compatible with pruned and knowledge distilled models.
- • In Section A.8, we describe the implementation details of model stitching baseline under SN-Netv2 (Pan et al., 2023b).
- • In Section A.9, we show image examples when using T-Stitch with different sampling steps based on DiTs.
- • In Section A.10, we demonstrate that T-Stitch is applicable to different pretrained model families, e.g., stitching DiT with U-ViT (Bao et al., 2023).
- • In Section A.11, we show more image examples in stable diffusion experiments, including the original SDv1.4, stylized SDs, SDXL, ControlNet.
- • In Section A.12, we report our finetune experiments by further finetuning the large DiTs at their allocated steps.
- • In Section A.13, we compare our default stitching strategy with more baselines.
- • In Section A.14, we report the additional memory and storage overhead of T-Stitch.
- • In Section A.15, we report the precision and recall metrics on class conditional ImageNet-256 based on DiTs.
- • In Section A.16, we show image examples of T-Stitch with DiTs and U-Nets.
- • In Section A.17, we evaluate FID under T-Stitch by using pretrained DiT-S at different training iterations.
- • In Section A.18, we demonstrate that T-Stitch can still obtain a smooth speed and quality trade-off under 2-4 steps with LCM (Luo et al., 2023).
- • In Section A.19, we show T-Stitch is also complementary to cache-based methods such as DeepCache (Ma et al., 2023) to achieve further speedup.
- • In Section A.20, we evaluate T-Stitch and show image examples by applying ToMe (Bolya et al., 2023) simultaneously.

- Table 3. Performance comparison of pretrained DiT model family on class-conditional ImageNet. FLOPs are measured by a single forward process given a latent noise in the shape of 4 × 32 × 32.

| |Parameters (M) FLOPs (G) Train Iters Time Cost (s) FID|
|---|---|
|DiT-S DiT-B DiT-XL|33.0 5.5 5000K 1.6 33.46 130.5 21.8 1600K 4.0 12.30<br><br>675.1 114.5 - 16.5 9.20|

t = 999 t = 888 t = 777 t = 666 t = 555 t = 444 t = 333 t = 222 t = 111 t = 0

5.0

4.5

Logamplitude

4.0

3.5

3.0

2.5

0.0 0.2 0.4 0.6 0.8 1.0

Frequency

- Figure 11. Frequency analysis in denoising process of DiT-XL, based on DDIM 10 steps and guidance scale of 4.0. We visualize the log amplitudes of Fourier-transformed latent noises at each step. Results are averaged over 32 images.

Table 4. Performance comparison of LDM and LDM-S on class-conditional ImageNet.

|Model|Param (M) Train Iter Time Cost (s) FID<br><br>|
|---|---|
|LDM-S LDM<br><br>|25 400K 2.9 40.92 394 200K 7.1 20.11|

#### A.1. Practical Deployment of T-Stitch

In this section, we provide guidelines for the practical deployment of T-Stitch by formulating our model allocation strategy into a compute budget allocation problem.

Given a set of denoisers {D1,D2,...,DK} and their corresponding computational costs {C1,C2,...,CK} for sampling in a T-steps trajectory, where Ck−1 < Ck, we aim to find an optimal configuration set {r1,r2,...,rK} that allocates models into corresponding denoising intervals to maximize the generation quality, which can be formulated as

max

r1,r2,...,rK

subject to

M (F(D1,r1) ◦ F(D2,r2)··· ◦ F(DK,rK)) (5)

K

K

rk = 1, (6)

rkCk ≤ CR,

k=1

k=1

where F(Dk,rk) refers to the denoising process by applying denoiser Dk at the k-th interval indicated by rk, ◦ denotes to a composition, M represents a metric function for evaluating generation performance, and CR is the compute budget constraint. Since {C1,C2,...,CK} is known, we can efficiently enumerate all possible fraction combinations and obtain a lookup table, where each fraction configuration set corresponds to a compute budget (i.e., time cost). In practice, we can sample a few configuration sets from this table that satisfy a budget and then apply to generation tasks.

#### A.2. Frequency Analysis in Denoising Process

We provide evidence that the denoising process focuses on low frequencies at the initial stage and high frequencies in the later steps. Based on DiT-XL, we visualize the log amplitudes of Fourier-transformed latent noises at each sampling step. As shown in Figure 11, the low-frequency amplitudes increase rapidly at the early timesteps (i.e., from 999 to 555), indicating that low frequencies are intensively generated. At the later steps, especially for t = 111 and t = 0, we observe the log amplitude of high frequencies increases significantly, which implies that the later steps focus on detail refinement.

#### A.3. Pretrained DiTs and U-Nets

In Table 3 and Table 4, we provide detailed comparisons of the pretrained DiT model family, as well as our reproduced small version of U-Net. Overall, as mentioned earlier in Section 3, we make sure the models at each model family have a clear gap in model size between each other such that we can achieve a clear speedup.

- s = 1.5

- s = 2.0

- s = 3.0

400

30

InceptionScore

25

300

FID

20

200

- s = 1.5

- s = 2.0

- s = 3.0

15

100

10

5 10 15

5 10 15

Time Cost (s)

Time Cost (s)

- Figure 12. Trajectory stitching based on three models: DiT-S, DiT-B, and DiT-XL. We adopt DDIM 100 timesteps with a classifier-free guidance scale of 1.5, 2.0 and 3.0.

0.0 0.2 0.4 0.6 0.8 1.0

Fraction of DiT-S

10

15

20

25

30

FID-5K

0.0 0.2 0.4 0.6 0.8 1.0

Fraction of DiT-S

5

10

15

20

FID-50K

- Figure 13. Trajectory stitching based on three models: DiT-S, DiT-B, and DiT-XL. We adopt DDPM 250 timesteps with a classifier-free guidance scale of 1.5.

#### A.4. Effect of Different Classifier-free Guidance on Three-model T-Stitch

In Figure 12, we provide the results by applying T-Sittch with DiTs using different guidance scales under three-model settings. In general, T-Stitch performs consistently with different guidance scales, where it interpolates a smooth Pareto frontier between the DiT-S and DiT-XL. As common practice in DPMs adopt different guidance scales to control image generation, this significantly underscores the broad applicability of T-Stitch.

#### A.5. FID-50K vs. FID-5K

For efficiency concerns, we report FID based on 5,000 images by default. Based on DiT, we apply T-Stitch with DDPM 250 steps with a guidance scale of 1.5 and sample 50,000 images for evaluating FID. As shown in Figure 13, the observation between FID-50K and FID-5K are similar, which indicates that sampling more images like 50,000 does not affect the effectiveness.

Pretrained

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Finetuned

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

- Figure 14. Image quality comparison by stitching pretrained and finetuned DiT-B and DiT-XL at the later steps, based on T-Stitch schedule of DiT-S/B/XL of 50% : 30% : 20%.

2.5 5.0 7.5 10.0 12.5 15.0 17.5

Time Cost (s)

10

15

20

25

30

FID

s = 1.5

T-Stitch

Reduce Steps

2.5 5.0 7.5 10.0 12.5 15.0 17.5

Time Cost (s)

10.0

12.5

15.0

17.5

20.0

22.5

FID

s = 2.0

T-Stitch

Reduce Steps

- Figure 15. Based on DDIM, we report the FID and speedup comparisons on DiT-XL by using T-Stitch and directly reducing the sampling step from 100 to 10. “s” denotes the classifier-free guidance scale. Trajectory stitching adopts the three-model combination (DiT-S/B/XL) under 100 steps.

#### A.6. Compared to Directly Reducing Sampling Steps

Reducing sampling steps has been a common practice for obtaining different speed and quality trade-offs during deployment. Although we have demonstrated that T-Stitch can achieve consistent efficiency gain under different sampling steps, we show in Figure 15 that compared to directly reducing the number of sampling steps, the trade-offs from T-Stitch are very competitive, especially for the 50-100 steps region where the FIDs under T-Stitch are even better. Thus, T-Stitch is able to serve as a complementary or an alternative method for practical DPM sampling speed acceleration.

#### A.7. Compared to Model Compression

In practice, T-Stitch is orthogonal to individual model optimization/compression. For example, with a BK-SDM Tiny and SDv1.4, we can still apply compression into SDv1.4 in order to reduce the computational cost at the later steps from the large SD. In Figure 16, we show that by adopting a compressed SD v1.4, i.e., BK-SDM Small, we can further reduce the time cost with a trade-off for image quality.

- 0.98s 0.96s 0.95s 0.94s 0.93s 0.93s

0.85s

- 1.54s 1.46s 1.33s 1.26s 1.22s 1.14s

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

0% 10% 20% 30% 40% 50%

###### with BK-SDM Small at later steps

BK-SDM Tiny

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

a backpack in the snow

0% 10% 20% 30% 40% 50%

- Figure 16. Comparison of T-Stitch by adopting SDv1.4 and its compressed version (i.e., BK-SDM Small) at the later steps.

Table 5. T-Stitch with DiT-S and U-ViT H, under DPM-Solver++, 50 steps, guidance scale of 1.5.

|Fraction of DiT-S<br><br>|0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%|
|---|---|
|FID Time Cost (s)|15.04 13.68 12.44 12.76 14.19 17.6 29.4 53.75 74.14 84.33 121.95<br><br>15.90 13.21 11.91 10.61 9.42 7.92 6.57 5.23 3.84 2.50 1.40|

- A.8. Implementation Details of Model Stitching Baseline

We adopt a LoRA rank of 64 when stitching DiT-S/XL, which leads to 134 stitching configurations. The stitched model is finetuned on 8 A100 GPUs for 1,700K training iterations. We pre-extract the ImageNet features with a Stable Diffusion AutoEncoder (Rombach et al., 2022) and do not apply any data augmentation. Following the baseline DiT, we adopt the AdamW optimizer with a constant learning rate of 1 × 10−4. The total batch size is set as 256. All other hyperparameters adopt the default setting as DiT.

- A.9. Image Examples under the Different Number of Sampling Steps

Figure 17 shows image examples generated using different numbers of sampling steps under T-Stitch and DiT-S/XL. As the figure shows, adopting a small model at the early 40% steps has a negligible effect on the final generated images. When progressively increasing the fraction of DiT-S, there is a visible trade-off between speed and quality, with the final image becoming more similar to those generated from DiT-S.

- A.10. T-Stitch with Different Pretrained Model families

As different pretrained models trained on the same dataset to learn similar encodings, T-Stitch is able to directly integrate different pretrained model families. For example, based on U-ViT H (Bao et al., 2023), we apply DiT-S at the early sampling steps just as we have done for DiTs and U-Nets. Remarkably, as shown in Table 5, it performs very well, which demonstrates the advantage of T-Stitch as it can be applied for more different models in the public model zoo.

- A.11. More Examples in Stable Diffusion

We show more examples by applying T-Stitch to SD v1.4, InkPunk Diffusion and Ghibli Diffusion with a small SD model, BK-SDM Tiny (Kim et al., 2023). For all examples, we adopt the default scheduler and hyperparameters of StableDiffusionPipeline in Diffusers: PNDM scheduler, 50 steps, guidance scale 7.5. In Figure 25, we observe that adopting a small SD in the sampling trajectory of SD v1.4 achieves minor effect on image quality at the small fractions and obtain

0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

[Figure 55]

10 Steps

[Figure 56]

20 Steps

[Figure 57]

50 Steps

[Figure 58]

100 Steps

[Figure 59]

250 Steps

- Figure 17. Based on DDIM and a classifier-free guidance scale of 1.5, we stitch the trajectories from DiT-S and DiT-XL and progressively increase the fraction (%) of DiT-S timesteps at the beginning.

flexible trade-offs in speed and quality by using different fractions.

Stylized SDs. For stylized SDs, such as InkPunk-Diffusion and Ghibli-Diffusion2, we show in Figures 26 and 27 that T-Stitch helps to complement the prompt alignment by effectively utilizing the knowledge of the pretrained small SD. Benefiting from the interpolation on speeds, styles and image contents, T-Stitch naturally increases the diversity of the generated images given a prompt by using different fractions of small SD.

Generality of T-Stitch . In Figure 28, we show T-Stitch performs favorably with more complex prompts. Besides, by adopting a smaller and distilled SSD-1B, we can easily accelerate SDXL while being compatible with complex prompts and ControlNet (Zhang et al., 2023) for practical art generation, as shown in Figure 29 and Figure 30. Furthermore, we demonstrate that T-Stitch is robust in practical usage. As shown in Figure 31, 8 consecutive runs can generate stable images with great quality.

#### A.12. Finetuning on Specific Trajectory Schedule

When progressively using a small model in the trajectory, we observe a non-negligible performance drop. However, we show that we can simply finetune the model at the allocated denoising intervals to improve the generation quality. For example, based on DDIM and 100 steps, allocating DiT-S at the early 50%, DiT-B at the subsequent 30%, and DiTXL at the last 20% obtains an FID of 16.49. In this experiment, we separately finetune DiT-B and DiT-XL at their allocated denoising intervals, with additional 250K iterations on ImageNet-1K under the default hyperparameters in DiT (Peebles & Xie, 2022). In Table 6, we observe a clear improvement over FID, Precision and Recall by finetuning at stitched interval. This strategy also achieves better performance than finetuning for all timesteps. Furthermore, we provide a comparison of the generated images in Figure 14, where we observe that finetuning clearly improves local details.

Table 6. Performance comparison of stitching pretrained and finetuned DiTs at the later steps. We set the denoising interval of DiT-S/B/XL with 50% : 30% : 20%

| |FID Inception Score|
|---|---|
|Pretrained Finetuned at all timesteps Finetuned at stitched interval<br><br>|16.49 123.11 16.04 125.81 13.35 155.35|

2https://huggingface.co/nitrosocke/Ghibli-Diffusion

- Table 7. Compared to other trajectory stitching baselines based on DiT-S/XL, DDIM 100 steps and guidance scale of 1.5. FID is calculated by 5K images. Memory and time cost are measured by a batch size of 8 on one RTX 3090.

|Method|FID Inception Score Time Cost|
|---|---|
|Interleave Decreasing Prob Large to Small Small to Large (Ours)<br><br>|19.02 120.04 10.1 12.94 163.45 9.8 27.61 72.60 10.0 10.06 200.81 9.9|

- Table 8. Local storage and memory cost comparison between DiT-S, DiT-XL and T-Stitch. Memory and time cost are measured by generating 8 images in parallel on one RTX 3090.

|Name<br><br>|Parameter (M) Local Storage (MB) Memory Cost (MB) Time Cost (s)|
|---|---|
|DiT-S DiT-XL T-Stitch (50%)|33 263 3088 1.7 675 2576 3166 16.5 708 (×1.04) 2839 (×1.10) 3296 (×1.04) 9.4 (×1.76)<br><br>|

#### A.13. Compared with More Stitching Baselines

By default, we design T-Stitch to start from a small DPM and then switch into a large DPM for the last denoising sampling steps. To show the effectiveness of this design, we compare our method with several baselines in Table 7 based on DiT-S and DiT-XL, including

- • Interleaving. During denoising sampling, we interleave the small and large model along the trajectory. Eventually, DiT-S takes 50% steps and DiT-XL takes another 50% steps.
- • Decreasing Prob. Linearly decreasing the probability of using DiT-S from 1 to 0 during the denoising sampling steps.
- • Large to Small. Adopting the large model at the early 50% steps and the small model at the last 50% steps.
- • Small to Large (our default design). The default strategy of T-Stitch by adopting DiT-S at the early 50% steps and using DiT-XL at the last 50% steps.

As shown in Table 7, in general, our default design achieves the best FID and Inception Score with similar sampling speed, which strongly demonstrate its effectiveness.

#### A.14. Additional Memory and Storage Overhead of T-Stitch

Intuitively, T-Stitch adopts a small DPM which can introduce additional memory and storage overhead. However, in practice, the large DPM is still the main bottleneck of memory and storage consumption. In this case, the additional overhead from small DPM is considerably minor. For example, as shown in Table 8, compared to DiT-XL, T-Stitch by adopting 50% steps of DiT-S only introduces additional 5% parameters, 4% GPU memory cost, 10% local storage cost, while significantly accelerating DiT-XL sampling speed by 1.76×.

#### A.15. Precision and Recall Measurement of T-Stitch

Following common practice (Dhariwal & Nichol, 2021), we adopt Precision to measure fidelity and Recall to measure diversity or distribution coverage. In Table 9, we show that T-Stitch introduces a minor effect on Precision and Recall at the early 40-50% steps, while at the later steps we observe clear trade-offs, which is consistent with FID evaluations.

#### A.16. Image Examples of T-Stitch on DiTs and U-Nets

In Figures 23 and 24, we provide image examples that generated by applying T-Stitch with DiT-S/XL, LDM-S/LDM, respectively. Overall, we observe that adopting a small DPM at the beginning still produces meaningful and high-quality images, while at the later steps it achieves flexible speed and quality trade-offs. Note that different from DiTs that learn a null

Table 9. Precision and Recall evaluation based on DiT-S/XL, with DDIM 100 steps and guidance scale of 1.5.

|Fraction of DiT-S<br><br>|0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%|
|---|---|
|FID Precision Recall|9.20 9.17 8.99 9.03 8.95 10.06 12.46 18.04 25.44 30.11 33.46<br><br>0.81 0.81 0.81 0.81 0.80 0.76 0.72 0.67 0.62 0.59 0.58 0.74 0.74 0.74 0.74 0.75 0.75 0.74 0.73 0.69 0.65 0.63<br><br>|

class embedding during classifier-free guidance, LDM inherently omits this embedding in their official implementation 3. During sampling, LDM and LDM-S have different unconditional signals, which eventually results in various image contents under different fractions.

#### A.17. Effect of DiT-S under Different Training Iterations

In our experiments, we adopt a DiT-S that trained with 5,000K iterations as it can be sufficiently optimized. In Figure 18, we indicate that even under a short training schedule of 400K iterations, adopting DiT-S at the initial stages of the sampling trajectory also has a minor effect on the overall FID. The main difference is at the later part of the sampling trajectory. Therefore, it implies the early denoising sampling steps can be easier to learn and be handled by a compute-efficient small model.

50

40

FID

30

20

10

400K 1000K 2000K 3000K 4000K 5000K

0.0 0.2 0.4 0.6 0.8 1.0

Fraction of DiT-S

- Figure 18. Effect of different pretrained DiT-S in T-Stitch for accelerating DiT-XL, based on DDPM, 250 steps and guidance scale of 1.5. For example, “400K” indicates the pretrained weights of DiT-S at 400K iterations.

#### A.18. Compatibility with LCM

T-Stitch can further speed up an already accelerated DPM via established training-based methods, such as step distillations (Luo et al., 2023; Song et al., 2023). For example, as shown in Figure 32 and Figure 33, given a distilled SDXL from LCM (Luo et al., 2023), T-Stitch can achieve further speedup under 2 to 4 steps with high image quality by adopting a relatively smaller SD. In Table 10, Table 11, we report comprehensive FID, inception score and CLIP score evaluations by stitching LCM distilled SDXL and SSD-1B, where we show that T-Stitch smoothly interpolates the quality between SDXL and SSD-1B. Finally, we assume a better and faster small model in T-Stitch will help to obtain more gains in future works.

#### A.19. Compatibility with DeepCache

In this section, we demonstrate that recent cache-based methods (Ma et al., 2023; Wimbauer et al., 2023) such as DeepCache (Ma et al., 2023) can be effectively combined with T-Stitch to obtain more benefit. Essentially, as T-Stitch directly drops off the pretrained SDs, we can adopt DeepCache to simultaneously accelerate both small and large diffusion models

3https://github.com/CompVis/latent-diffusion

- Table 10. T-Stitch based on LCM (Luo et al., 2023) distilled models: LCM-SDXL and LCM-SSD-1B, under 2 sampling steps.

|Faction of SSD-1B<br><br>|0 0.5 1|
|---|---|
|FID IS CLIP Score Time Cost (ms)<br><br>|21.98 23.96 24.36 28.48 27.60 28.22 0.2929 0.2895 0.2844 768 685 634|

- Table 11. T-Stitch based on LCM (Luo et al., 2023) distilled models: LCM-SDXL and LCM-SSD-1B, under 4 sampling steps. Time cost is measured by generating one image on RTX 3090 in seconds.

|Faction of SSD-1B|0% 25% 50% 75% 100%<br><br>|
|---|---|
|FID IS CLIP Score Time Cost (ms)<br><br>|17.05 18.32 21.28 23.67 25.50 35.35 34.46 31.83 30.47 29.31 0.3062 0.3059 0.2984 0.2912 0.2897 1,029 968 921 870 823|

during sampling to achieve further speedup. The image quality and speed benchmarking as shown in Figure 21 have demonstrated that T-Stitch works very well along with DeepCache, while potentially further improving the prompt alignment for stylized SDs. We also comprehensively evaluate the FID, Inception score, CLIP score and time cost in Figure 19, where we observe combining T-Stitch with DeepCache brings improvement over all metrics. Note that under DeepCache, BK-SDM Tiny is 1.5× faster than SDv1.4, thus the speedup gain from T-Stitch is slightly smaller than applying T-Stitch only where the BK-SDM Tiny is 1.7× faster than SDv1.4. In addition, we observe DeepCache cannot work well with step-distilled models and ControlNet, while T-Stitch is generally applicable to many scenarios, as shown in Section A.11.

#### A.20. Compatibility with Token Merging

Our technique also complements Token Merging (Bolya et al., 2023). For example, during the denoising sampling, we can still apply ToMe into both small and large U-Nets. In practice, it brings additional gain in both sampling speed and CLIP score, and slightly improves Inception score, as shown in Figure 20. We also provide image examples in Figure 22.

3.0

0.30

T-Stitch T-Stitch + DeepCache

T-Stitch T-Stitch + DeepCache

T-Stitch T-Stitch + DeepCache

2.5

36

InceptionScore

16

TimeCost(s)

0.29

CLIPScore

2.0

34

FID

1.5

0.28

14

1.0

32

T-Stitch T-Stitch + DeepCache

0.27

0.5

12

30

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Fraction of Small Model

Fraction of Small Model

Fraction of Small Model

Fraction of Small Model

###### Figure 19. Effect of combining T-Stitch and DeepCache (Ma et al., 2023). We report FID, Inception Score and CLIP score (Hessel et al.,

2021) on MS-COCO 256×256 benchmark under 50 steps. The time cost is measured by generating one image on one RTX 3090. We adopt BK-SDM Tiny and SDv1.4 as the small and large model, respectively. For DeepCache, we adopt an uniform cache interval of 3.

2.75

- 13
- 14
- 15
- 16
- 17

T-Stitch T-Stitch + ToMe

T-Stitch T-Stitch + ToMe

T-Stitch T-Stitch + ToMe

T-Stitch T-Stitch + ToMe

36

2.50

InceptionScore

0.29

TimeCost(s)

CLIPScore

2.25

34

FID

0.28

2.00

32

1.75

0.27

1.50

30

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Fraction of Small Model

Fraction of Small Model

Fraction of Small Model

Fraction of Small Model

###### Figure 20. Effect of combining T-Stitch and ToMe (Bolya et al., 2023). We report FID, Inception Score and CLIP score (Hessel et al.,

- 2021) on MS-COCO 256×256 benchmark under 50 steps. The time cost is measured by generating one image on one RTX 3090. We adopt BK-SDM Tiny and SDv1.4 as the small and large model, respectively. For ToMe, we adopt a token merging ratio of 0.5.

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

SDv1.4 w/ DeepCache w/ DeepCache + T-Stitch

10%, 1101ms 20%, 1086ms 30%, 1041ms 40%, 979ms 50%, 975ms

10% 20%

InkPunk SD w/ DeepCache w/ DeepCache + T-Stitch

2440ms 1215ms

A blue cake topped with a beach scene.

a squirrel in the park, nvinkpunk style

30% 40% 50%

10% 20%

Ghibli SD w/ DeepCache w/ DeepCache + T-Stitch

A sky filled with vibrant hot air balloons, ghibli style

30% 40% 50%

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Figure 21. Image examples of combining T-Stitch with DeepCache (Ma et al., 2023). We adopt BK-SDM Tiny as the small model in T-Stitch and report the percentage on the top of images. All images are generated by the default settings in diffusers (von Platen et al.,

- 2022): 50 steps with a guidance scale of 7.5.

2.440s 2.272s

10%, 2.228s 20%, 2.138s 30%, 2.052s 40%, 1.954s 50%, 1.858s

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

SDv1.4 w/ ToMe w/ ToMe + T-Stitch

three horses in a snowy field with trees in the background

10% 20%

30% 40% 50%

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

InkPunk SD w/ ToMe w/ ToMe + T-Stitch

a man is riding a red motorcycle and some buildings, nvinkpunk.

10% 20%

30% 40% 50%

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

###### Ghibli SD w/ ToMe w/ ToMe + T-Stitch

a ghibli style village, cows eating by a river, sunset

- Figure 22. Image examples of combining T-Stitch and ToMe (Bolya et al., 2023). We adopt BK-SDM Tiny as the small model in T-Stitch and report the percentage on the top of images. All images are generated by the default settings in diffusers (von Platen et al., 2022): 50 steps with a guidance scale of 7.5. We adopt a token merging ratio of 0.5 in ToMe.

crate

0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

[Figure 102]

coral_fungus

[Figure 103]

restaurant

[Figure 104]

pillow

admiral

[Figure 105]

[Figure 106]

dowitcher

[Figure 107]

gyromitra

[Figure 108]

spotted salamander

[Figure 109]

convertible

[Figure 110]

- Figure 23. Image examples of T-Stitch on DiT-S and DiT-XL. We adopt DDIM and 100 steps, with a guidance scale of 4.0. From left to right, we gradually increase the fraction of LDM-S steps at the beginning, then let the original LDM to process later denoising steps.

promontory

0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

[Figure 111]

macaw

[Figure 112]

aircraft_carrier

[Figure 113]

[Figure 114]

German_shepherd

beacon

[Figure 115]

[Figure 116]

sewing_machine

hen

[Figure 117]

[Figure 118]

English_springer

bison

[Figure 119]

- Figure 24. Image examples of T-Stitch on U-Net-based LDM and LDM-S. We adopt DDIM and 100 steps, with a guidance scale of 3.0. From left to right, we gradually increase the fraction of LDM-S steps at the beginning, then let the original LDM to process later denoising steps.

[Figure 120]

SD-v1.4

[Figure 121]

SD-v1.4

[Figure 122]

SD-v1.4

10% 20% 30% 40% 50%

|[Figure 123]|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]|
|---|---|---|---|---|

60% 70% 80% 90% 100%

|[Figure 128]|[Figure 129]|[Figure 130]|[Figure 131]|[Figure 132]|
|---|---|---|---|---|

a bowl that has vegetables inside of it

10% 20% 30% 40% 50%

|[Figure 133]|[Figure 134]|[Figure 135]|[Figure 136]|[Figure 137]|
|---|---|---|---|---|

60% 70% 80% 90% 100%

|[Figure 138]|[Figure 139]|[Figure 140]|[Figure 141]|[Figure 142]|
|---|---|---|---|---|

a brown and white cat laying on a bed

10% 20% 30% 40% 50%

|[Figure 143]|[Figure 144]|[Figure 145]|[Figure 146]|[Figure 147]|
|---|---|---|---|---|

60% 70% 80% 90% 100%

|[Figure 148]|[Figure 149]|[Figure 150]|[Figure 151]|[Figure 152]|
|---|---|---|---|---|

A very large tower has a clock on it

- Figure 25. T-Stitch based on Stable Diffusion v1.4 and BK-SDM Tiny. We annotate the faction of BK-SDM on top of images.

[Figure 153]

Inkpunk SD

[Figure 154]

Inkpunk SD

[Figure 155]

Inkpunk SD

10% 20% 30% 40% 50%

|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|[Figure 160]|
|---|---|---|---|---|

60% 70% 80% 90% 100%

|[Figure 161]|[Figure 162]|[Figure 163]|[Figure 164]|[Figure 165]|
|---|---|---|---|---|

A polar bear on mars, nvinkpunk style

10% 20% 30% 40% 50%

|[Figure 166]|[Figure 167]|[Figure 168]|[Figure 169]|[Figure 170]|
|---|---|---|---|---|

60% 70% 80% 90% 100%

|[Figure 171]|[Figure 172]|[Figure 173]|[Figure 174]|[Figure 175]|
|---|---|---|---|---|

an lion wearing a suit in a meeting room, nvinkpunk style

10% 20% 30% 40% 50%

|[Figure 176]|[Figure 177]|[Figure 178]|[Figure 179]|[Figure 180]|
|---|---|---|---|---|

60% 70% 80% 90% 100%

|[Figure 181]|[Figure 182]|[Figure 183]|[Figure 184]|[Figure 185]|
|---|---|---|---|---|

A train pulling into a station on a cloudy day, nvinkpunk style

- Figure 26. T-Stitch based on Inkpunk-Diffusion SD an BK-SDM Tiny. We annotate the faction of BK-SDM on top of images.

[Figure 186]

Ghibli SD

[Figure 187]

Ghibli SD

[Figure 188]

Ghibli SD

10% 20% 30% 40% 50%

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

60% 70% 80% 90% 100%

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

a ghibli style princess with golden hair in New York City

10% 20% 30% 40% 50%

|[Figure 199]|[Figure 200]|[Figure 201]|[Figure 202]|[Figure 203]|
|---|---|---|---|---|

60% 70% 80% 90% 100%

|[Figure 204]|[Figure 205]|[Figure 206]|[Figure 207]|[Figure 208]|
|---|---|---|---|---|

ghibli style beautiful Caribbean beach tropical (sunset)

10% 20% 30% 40% 50%

|[Figure 209]|[Figure 210]|[Figure 211]|[Figure 212]|[Figure 213]|
|---|---|---|---|---|

60% 70% 80% 90% 100%

|[Figure 214]|[Figure 215]|[Figure 216]|[Figure 217]|[Figure 218]|
|---|---|---|---|---|

ghibli style ice field white mountains ((northern lights)) starry sky low horizon

- Figure 27. T-Stitch based on Ghibli-Diffusion SD and BK-SDM Tiny. We annotate the faction of BK-SDM on top of images.

|[Figure 219]|
|---|

SD-v1.4

[Figure 220]

SD-v1.4

[Figure 221]

SD-v1.4

10% 20% 30% 40% 50%

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

60% 70% 80% 90% 100%

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

A clock and several vases sit on a table in front of a gold framed mirror

10% 20% 30% 40% 50%

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

60% 70% 80% 90% 100%

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

Macro photography of dewdrops on a spiderweb, with morning sunlight creating rainbows.

10% 20% 30% 40% 50%

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

60% 70% 80% 90% 100%

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Aerial photography of a winding river through autumn forests, with vibrant red and orange foliage.

- Figure 28. T-Stitch with more complex prompts based on Stable Diffusion v1.4 and BK-SDM Tiny. We annotate the faction of BK-SDM on top of images.

SDXL, 13.6s 20%, 12.04s 40%, 11.05s 60%, 10.6s 80%, 9.7s 100%, 8.8s

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

###### Prompt:

highly detailed albert einstein at an epic laboratory oﬃce, shelves with detailed items in background, ((long shot)), highly detailed realistic painting by grandmaster, unreal engine, octane render, 4k, trending on artstation

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

Prompt:

concept art of dragon ﬂying over town, clouds. digital artwork, illustrative, painterly, matte painting, highly detailed, cinematic composition

Negative Prompt:

photo, photorealistic, realism, ugly

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Prompt:

anime artwork an empty classroom. anime style, key visual, vibrant, studio anime, highly detailed

Negative Prompt:

photo, deformed, black and white, realism, disﬁgured, low contrast

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Prompt:

claymation style captain jack sparrow on tropical island. sculpture, clay art, centered composition, play-doh

Negative Prompt:

sloppy, messy, grainy, highly detailed, ultra textured, photo, mutated

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Prompt: Negative Prompt:

16-bit pixel art, a cozy cafe side view, a beautiful day sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic

- Figure 29. T-Stitch with more complex prompts based on SDXL (Podell et al., 2023) and SSD-1B (Segmind, 2023). We annotate the faction of SSD-1B on top of images. Time cost is measured by generating one image on RTX 3090.

|[Figure 282]|
|---|

[Figure 283]

Image

Canny edges

SDXL, 19.7s 20%, 18.4s 40%, 17.5s 60%, 16.6s 80%, 15.6s 100%, 14.8s

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

Prompt: highly detailed cinematic aerial view of a futuristic research complex in Antarctica, 4K Negative Prompt: blur, low quality, bad quality, sketches

[Figure 290]

[Figure 291]

Image Depth

SDXL, 19.4s 20%, 18.4s 40%, 17.3s 60%, 16.4s 80%, 15.5s 100%, 14.6s

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Prompt: a beautiful bosai tree, masterpiece, 4K Negative Prompt: blur, low quality, bad quality, sketches

|Image<br><br>[Figure 298]|
|---|

[Figure 299]

Image Pose

SDXL, 19.4s 20%, 18.4s 40%, 17.5s 60%, 16.5s 80%, 15.6s 100%, 14.7s

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Prompt: Ironman dancing in a futuristic city, high quality Negative Prompt: low quality, bad quality

- Figure 30. T-Stitch with SDXL-based ControlNet. We annotate the faction of SSD-1B on top of images. Time cost is measured by generating one image on one RTX 3090.

8 runs

0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

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

Cluttered house in the woods, anime, oil painting, high resolution, cottagecore, ghibli inspired, 4k.

- Figure 31. Based on Stable Diffusion v1.4 and BK-SDM Tiny, we generate images by different fractions of BK-SDM for 8 consecutive runs (a for-loop) on one GPU. T-Stitch demonstrates stable performance for robust image generation. Best viewed in digital version and zoom in.

LCM-SDXL, 768ms 50%, 685ms 100%, 634ms

[Figure 394]

[Figure 395]

[Figure 396]

Prompt: Astronaut in a jungle, cold color palette, muted colors, detailed, 8k

LCM-SDXL 50% 100%

[Figure 397]

[Figure 398]

[Figure 399]

Prompt: A picture of a cute Welsh Corgi in a bucket

- Figure 32. T-Stitch based on distilled models: LCM-SDXL (Luo et al., 2023) and LCM-SSD-1B (Luo et al., 2023), under 2 sampling steps. We annotate the faction of LCM-SSD-1B on top of images. Time cost is measured by generating one image on RTX 3090 in milliseconds.

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

Prompt: Self-portrait oil painting, a beautiful cyborg with golden hair, 8k

Prompt: a close-up picture of an old man standing in the rain

LCM-SDXL, 1029ms 25%, 968ms 50%, 921ms 75%, 870ms 100%, 823ms

LCM-SDXL 25% 50% 75% 100%

- Figure 33. T-Stitch based on distilled models: LCM-SDXL (Luo et al., 2023) and LCM-SSD-1B (Luo et al., 2023), under 4 sampling steps. We annotate the faction of LCM-SSD-1B on top of images. Time cost is measured by generating one image on RTX 3090 in milliseconds.

