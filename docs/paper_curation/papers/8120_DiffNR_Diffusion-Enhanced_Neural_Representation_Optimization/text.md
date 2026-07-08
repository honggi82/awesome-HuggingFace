# arXiv:2604.21518v1[eess.IV]23Apr2026

## DiffNR: Diffusion-Enhanced Neural Representation Optimization for Sparse-View 3D Tomographic Reconstruction

### Shiyan Su1*, Ruyi Zha2*, Danli Shi3, Hongdong Li2, Xuelian Cheng1†

1Monash University 2The Australian National University 3Hong Kong Polytechnic University Xuelian.Cheng@monash.edu

##### Abstract

Neural representations (NRs), such as neural fields and 3D Gaussians, effectively model volumetric data in computed tomography (CT) but suffer from severe artifacts under sparseview settings. To address this, we propose DiffNR, a novel framework that enhances NR optimization with diffusion priors. At its core is SliceFixer, a single-step diffusion model designed to correct artifacts in degraded slices. We integrate specialized conditioning layers into the network and develop tailored data curation strategies to support model finetuning. During reconstruction, SliceFixer periodically generates pseudo-reference volumes, providing auxiliary 3D perceptual supervision to fix underconstrained regions. Compared to prior methods that embed CT solvers into time-consuming iterative denoising, our repair-and-augment strategy avoids frequent diffusion model queries, leading to better runtime performance. Extensive experiments show that DiffNR improves PSNR by 3.99 dB on average, generalizes well across domains, and maintains efficient optimization.

### Introduction

X-ray computed tomography (CT) is an essential imaging technique for noninvasive inspection of internal structures. A CT scanner captures multi-view projections that record the X-ray attenuation through the material. Given these projections, 3D tomographic reconstruction aims to recover a radiodensity volume. Conventional CT systems acquire hundreds of projections to produce a clean volume, but this results in substantial radiation exposure to subjects. Sparseview CT (SVCT) reconstruction, which aims to maintain high-quality recovery with only a few dozen projections, thus becomes a crucial direction for safer imaging.

Recent years have seen rapid progress in learning-based SVCT. While feedforward approaches exist (Jin et al. 2017; Lin et al. 2024), optimization frameworks are generally preferred to enforce consistency between predicted volumes and measured projections. They can be broadly categorized

*These authors contributed equally. Project page at https:// ooonesevennn.github.io/DiffNR/

†Corresponding author: Xuelian Cheng Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

into neural representation (NR) and neural prior (NP) approaches. NR methods model the volume as learnable neural fields (Zha, Zhang, and Li 2022) or 3D Gaussians (Zha et al. 2024), and optimize them in a self-supervised manner. They outperform traditional algorithms but yield artifacts in underconstrained regions. In contrast, NP methods pretrain networks to learn data-driven priors and then align network outputs with measurements using optimization solvers. Recent state-of-the-art NP approaches adopts unconditional 2D diffusion models (Ho, Jain, and Abbeel 2020) as network backbone, and embed local solvers into iterative denoising steps. While adequately steering unconditional generation towards the true data manifold, they suffer from inter-slice jitters, hallucinations, and long processing time.

In this work, we aim to marry neural representations with diffusion models. Unlike prior methods that embed local solvers into unconditional denoising processes, we adopt a fundamentally different strategy: enhancing a global NR with conditioned diffusion models. This design offers clear advantages: (1) learning a unified 3D representation promotes volumetric consistency, and (2) we can finetune powerful 2D foundation models instead of training one from scratch. Nevertheless, this integration is non-trivial, with key challenges in developing an NR-aware diffusion model and efficiently incorporating it into NR optimization.

To tackle these challenges, we propose DiffNR, Diffusion enhanced Neural Representation, for sparse-view 3D CT reconstruction. At its core is SliceFixer, a diffusion model specifically adapted to correct artifacts in NR-reconstructed slices. Leveraging 2D foundation models and recent advances in inference acceleration, we finetune a single-step diffusion model (Sauer et al. 2024) on a curated dataset of clean and corrupted slice pairs under varying sparsity levels. To improve structural awareness, we incorporate biplanar X-ray projections as additional conditioning inputs. During the reconstruction phase, SliceFixer periodically generates pseudo-reference volumes, which guide NR optimization in underconstrained regions. We adopt a perceptual SSIM-based regularization instead of voxel-wise losses to mitigate hallucinations and promote structural integrity. This repair-and-augment strategy reduces the need for frequent diffusion model queries, thus ensuring computational effi-

ProjectionI

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

 

  ω

[Figure 7]

- x
- y
- z

Query

[Figure 8]

| | |
|---|---|
|SliceF|ixer|
| | |

Neural Fields

Corrupted Slice

[Figure 9]

r I(r)

[Figure 10]

sn sf

[Figure 11]

[Figure 12]

Source

[Figure 13]

[Figure 14]

[Figure 15]

Update

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Volume ω(v)

3D Gaussians

NAF +DiffNR(+2.19 dB) R2-Gaussian +DiffNR(+5.79 dB)

Refined Slice

###### (a) 3D CT Geometry

(b) Method Overview (c) Qualitative Results (36-view)

- Figure 1: We propose DiffNR for sparse-view 3D CT reconstruction. (a) Geometry of a cone-beam CT scanner. (b) Method overview. (c) Comparison between the baseline methods (Zha, Zhang, and Li 2022; Zha et al. 2024) and our proposed DiffNR.

ciency. We evaluate DiffNR across in-distribution and outof-distribution datasets. Extensive experiments show that it improves NR reconstruction quality by 3.99 dB, generalizes well across domains, and maintains reasonable runtime.

3D CT, but they struggle in sparse-view settings. NP methods combine optimization solvers (traditional or NR-based) with pretrained networks. Some methods use deterministic networks (Kamilov et al. 2023; Tian et al. 2025; Vo et al. 2024) as regularizer, and the state of the art plugs traditional local solvers into unconditional diffusion models (Chung et al. 2023; Chung, Lee, and Ye 2023). Within this paradigm, there are some early diffusion-NR hybrids (Du et al. 2024; Chu et al. 2025) which adapt NR as local solvers. Compared with prior methods, our DiffNR takes a new direction by enhancing a global NR with conditional diffusion models.

We summarize our contributions as follows. (1) We propose DiffNR, a novel framework that combines neural representation with diffusion priors, fundamentally different from prior CT methods. (2) We design an effective pipeline to adapt diffusion models for artifact correction and efficiently integrate them into NR optimization, which may also inspire other inverse problems. (3) Experiments demonstrate that DiffNR outperforms existing methods in accuracy, generalization, and efficiency, highlighting its practical values.

Diffusion-Enhanced Neural Representation Enhancing NR with diffusion priors has proven to be effective in RGB view synthesis. Some works use diffusion models as scorers that must be queried at each optimization step (Gu et al. 2023; Warburg et al. 2023; Zhou and Tulsiani 2023), which significantly compromises efficiency. Other approaches finetune diffusion models to repair corrupted images rendered from NR and augment training views with these pseudoobservations (Liu, Zhou, and Huang 2024; Liu et al. 2024). This strategy avoids frequent diffusion queries, thereby reducing computational overhead. Notably, Difix3D+ (Wu et al. 2025) further improves efficiency by employing singlestep diffusion models (Sauer et al. 2024). Our method follows the repair-and-augment strategy but introduces key innovations designated for CT: (1) we correct artifacts on reconstructed slices rather than on rendered projections, and (2) we augment pseudo-volumes for direct 3D supervision instead of relying on intermediate image losses.

### Related Work

Computed Tomography CT is widely used in daily applications such as medical diagnosis and security screening. Conventional fan-beam CT reconstructs a 3D volume slice by slice from 1D projection arrays. More recently, conebeam CT has become popular as it swiftly captures 2D projection images, creating demand for direct volumetric reconstruction. Traditional algorithms fall into direct and iterative methods. Direct approaches (Feldkamp, Davis, and Kress 1984) instantly compute analytical results but produce severe artifacts. Iterative methods (Andersen and Kak 1984; Sidky and Pan 2008) formulate reconstruction as an optimization problem and solve it using numerical solvers. They reduce artifacts but oversmooth fine details.

Learning-Based Tomographic Reconstruction Similar to traditional algorithms, learning-based CT reconstruction can be performed directly or iteratively. Many works use feedforward networks to predict results from projections (Lin et al. 2024; Zhang et al. 2025) or low-quality reconstructions (Jin et al. 2017; Ma et al. 2023). Such a direct regression, however, lacks physical constraints. Consequently, more attention has shifted to optimization frameworks, broadly grouped into neural representation (NR) and neural prior (NP) approaches. NR methods, inspired by advances in RGB view synthesis such as NeRF (Mildenhall et al. 2020) and 3D Gaussian splatting (3DGS) (Kerbl et al.

### Background

X-ray Imaging This work adopts cone-beam geometry as a typical example of 3D CT, and the proposed method can be readily adapted to other geometries such as parallel-beam. As shown in Figure 1(a), an X-ray with initial intensity I0 travels along the trajectory r(s) = o + sd ∈ R3 where s ∈ [sn,sf], passes through a density field σ(v) : R3 → R where v is any spatial location, and eventually reaches the detector plane. According to the Beer-Lambert law (Kak and Slaney 2001), the corresponding raw pixel value is given by I′(r) = I0 exp(− s sf

σ(r(s))ds). In practice, raw data are transformed into logarithmic space for computational convenience, yielding the processed pixel value:

- 2023), optimize a learnable field via differentiable rendering. There are NeRF (Zha, Zhang, and Li 2022; Cai et al.
- 2024) and 3DGS (Zha et al. 2024; Li et al. 2025) variants for

n

Zero Conv

[Figure 20]

[Figure 21]

 

  ω

[Figure 22]

[Figure 23]

- x
- y
- z

Query

Neural Fields

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

VAE Encoder U-Net VAE Decoder

Corrupted Slice Refined Slice

[Figure 28]

[Figure 29]

Cross Attn

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Remove artifacts for this [Organ] CT slice.

Trainable Frozen

3D Gaussians

Neural Representations

Biplanar Projections Image Encoder Text Encoder Text Prompt

- Figure 2: SliceFixer Architecture. It takes as input a CT slice queried from NRs, along with biplanar projections and a text prompt as conditions. It outputs a refined slice without artifacts. The model is built on SD-Turbo (Sauer et al. 2024), a singlestep diffusion backbone. Trainable LoRA layers and zero convolutions are injected to adapt the model for our purpose.

I(r) = log I0 − log I′(r) = s sf

σ(r(s))ds. Unless otherwise stated, we use the logarithmic projections as inputs. The goal of tomographic reconstruction is to recover the underlying density field σ(v), typically output as a discrete voxel grid V ∈ RX×Y ×Z, from multi-angle projections {Ii}Ni=1. Note that real-world projections contain noise due to physical effects and hardware imperfections.

data, t∼pt,ϵ∼N(0,1) ∥ϵ − ϵθ(xt;c,t)∥22 , where c denotes optional conditioning information, such as text or images. Recent advances (Sauer et al. 2024) accelerate diffusion inference by distilling the multi-step denoising process into a single-step generation.

ing objective: Ex∼p

n

### Proposed Method

Neural Representations NR methods trains a 3D model via differentiable rendering. There are two primary types of NRs: neural fields and 3D Gaussians. Neural fields, as exemplified by NAF (Zha, Zhang, and Li 2022), represent the density field with a multilayer perceptron (MLP) f, which can be queried at any location v to produce the corresponding density σf(v). The rendering function is a discrete BeerLambert law: If(r) = Pi=1 σf(r(si)) · (r(si+1) − r(si)) where P is the number of sampled points along each ray.

Given N projection images {Ii}Ni=1 acquired at uniform angular intervals around an object, our goal is to reconstruct

its volumetric density field σ(v), with emphasis on underconstrained regions that are prone to artifacts. To tackle this, we introduce DiffNR, a neural representation optimization framework with diffusion-based augmentation. This section is organized as follows. We begin by introducing SliceFixer, a single-step diffusion model that repairs degraded CT slices. Next, we detail the data curation strategies for model finetuning. Finally, we illustrate how to efficiently integrate SliceFixer into the optimization pipeline.

R2-Gaussian (Zha et al. 2024) is a recent 3DGS-based approach, offering faster reconstruction than neural field methods. It represents the density field as a mixture of 3D Gaus-

#### SliceFixer: Diffusion Model for Slice Repairing

sians: σg(v) = Mi=1 Gi3(v), where M is the number of kernels. Each Gaussian Gi3 has learnable parameters: base density ρi, center pi ∈ R3, and covariance Σi ∈ R3×3. Its form is given by: Gi3(v) = ρi exp(−12(v − pi)⊤Σ−i 1(v − pi)). To render a projection image, each 3D Gaussian is splatted onto the image plane as a 2D Gaussian Gi2(u), where u ∈ R2. The final projection is then computed by summing all 2D Gaussians: Ig(u) = Mi=1 Gi2(u). We use NAF and R2-Gaussian as two NR backbones.

Previous NR methods (Wu et al. 2025) repair artifacts at the projection level and incorporate intermediate image losses to optimize 3D models. While effective for surface-based RGB reconstruction, this strategy is suboptimal for volumetric reconstruction, where errors in penetrable X-ray projections accumulate. To address this, we propose SliceFixer, a diffusion model that predicts a refined slice Sˆ ∈ RX

′×Y ′ from its counterpart S˜ queried from NRs. We build SliceFixer upon SD-Turbo (Sauer et al. 2024), a single-step diffusion model that has demonstrated strong performance in imageto-image translation tasks (Parmar et al. 2024) and providing good inference efficiency. Following Chung et al. (2023), we use axial (z-direction) slices in practice, though the approach can be extended to arbitrary slicing directions. Architecture is shown in Figure 2. A VAE encodes corrupted slices into latents, and a U-Net predicts the target latents conditioned on the encoded inputs, conditions, and denoising timestep. The refined slice is then reconstructed using the VAE decoder.

Diffusion Models Diffusion models (Ho, Jain, and Abbeel 2020; Song et al. 2020) learn to approximate the data distribution pdata through iterative denoising. During training, a noisy version of a data sample x ∼ pdata is generated as xt = √α¯t x + √1 − α¯tϵ, where ϵ ∼ N(0,1) is standard Gaussian noise, and α¯t controls noise level. The discrete diffusion timestep t is sampled from a uniform distribution pt ∼ U(0,tmax). The denoising network θ predicts the added noise ϵθ and is optimized with the score match-

Algorithm 1: Diffusion-Enhanced NR Optimization Input: Sparse-view projections {Ii}Ni=1, scanner calibration parameters {Ki}Ni=1, neural fields f or 3D Gaussians g Output: Density volume V

Neural Representations

Image Losses (Stage 1+2)

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Compute

 

  ω

[Figure 38]

[Figure 39]

- x
- y
- z

[Figure 40]

Low-Level Prior (Stage 1+2)

[Figure 41]

Update

Diffusion Prior (Stage 2)

Neural Fields 3D Gaussians

- 1: for j = 1 to J do
- 2: Render projection ˜Ii with geometry parameters Ki
- 3: Compute L1 and SSIM losses between ˜Ii and Ii
- 4: Query volume V˜ tv and compute total variation (TV)
- 5: if j mod ℓ = 0 then
- 6: Query volume V˜ ℓ
- 7: for each axial slice S˜ in V˜ ℓ do
- 8: Upsample S˜ to match SliceFixer input size
- 9: Generate repaired slice Sˆ with SliceFixer
- 10: Downsample Sˆ back to queried size
- 11: end for
- 12: Stack repaired slices into a volume Vˆ ℓ
- 13: end if
- 14: if Vˆ ℓ exists and j mod τ = 0 then
- 15: Query V˜ and compute its 3D SSIM loss with Vˆ ℓ
- 16: end if
- 17: Update f or g based on all losses
- 18: end for
- 19: Query final volume V from trained f or g

Image Losses Low-Level Prior

[Figure 42]

[Figure 43]

[Figure 44]

L1,SSIM

Render

Query

TV

Rendered Proj. Real Proj.

Small Vol.

Diffusion Prior

[Figure 45]

[Figure 46]

[Figure 47]

SliceFixer

Query

Query

SSIM

Queried Vol. Reference Vol. Every ℓ Iters

Figure 3: DiffNR Pipeline. During the training, we train neural representations using image losses and low-level regularization. In Stage 2, we generate a pseudo-reference volume with SliceFixer every ℓ iterations, and then apply SSIM regularization on queried and reference volumes.

#### Data Curation

Conditioning We aim to teach SliceFixer to remove artifacts while preserving anatomical structures in CT slices. To this end, our model is conditioned jointly on a text prompt ct and two orthogonal X-ray projections (Ia,Ib). The text prompt provides high-level semantic guidance, whereas the biplanar X-ray projections contains global structural cues. We employ the pretrained RAD-DINO (P´erez-Garc´ıa et al.

Training SliceFixer requires a large-scale dataset of paired slices, where one slice contains artifacts typically introduced during NR optimization and the other serves as the clean ground truth. However, no existing dataset satisfies these requirements. To address this, we leverage public 3D CT volumes to synthesize projection data and train a diverse set of neural representations. We explore various strategies to expand the training set and improve data diversity.

- 2025) tailored for radiographs to encode image features. These image features are subsequently aggregated with text embedding via a cross-attention layer to form the conditioning input c = Embed(Ia,Ib,ct) for the diffusion model.

View Distribution We use the tomography toolbox (Biguri et al. 2016) to synthesize K dense projections for each real CT volume over a full 360◦ angular range. To simulate sparse-view scenarios, we randomly sample subsets of these projections to train NR models. We explore both uniformly and non-uniformly distributed view configurations. This variation introduces diverse artifact patterns in the reconstructed volumes, thereby enhancing the model’s robustness to varying sparse-view conditions.

Finetuning We finetune a pretrained 2D foundation model SD-Turbo (Sauer et al. 2024) to leverage its rich visual priors. Following Pix2pix-Turbo (Parmar et al. 2024), we inject LoRA adapters (Hu et al. 2022) into the VAE and U-Net modules and incorporate skip connections between the encoder and decoder via zero-convolution layers (Zhang, Rao, and Agrawala 2023). Other parameters are kept frozen.

Losses We integrates several standard diffusion losses, including L2 loss, LPIPS loss (Zhang et al. 2018), CLIP alignment loss (Radford et al. 2021), and an adversarial loss implemented with a CLIP-based discriminator for the target domain (Parmar et al. 2024). Additionally, we introduce a structural similarity (SSIM) (Wang et al. 2004) loss that captures perceptual quality. Our final objective is defined as:

Model Underfitting We intentionally underfit the NR optimization by limiting training to a reduced number of iterations (e.g., 25–50% of the standard training steps). These underfitted reconstructions exhibit more pronounced artifacts due to incomplete convergence, thereby enriching the training set with challenging examples.

Mixed Neural Representation We mix reconstruction results from both neural fields and 3D Gaussians in a 1:1 ratio to encourage the diffusion model to learn generalized priors, rather than overfitting to specific patterns.

Ltotal = LL2 + LLPIPS + λCLIPLCLIP

+ λGANLGAN + λSSIMLSSIM.

ToothFairy (Cipriano et al. 2022) LUNA16 (Setio et al. 2017)

Methods

TIME 36-view 24-view 12-view 36-view 24-view 12-view

PSNR / SSIM PSNR / SSIM PSNR / SSIM PSNR / SSIM PSNR / SSIM PSNR / SSIM

Traditional Methods SART 27.41 / 0.581 27.13 / 0.596 25.66 / 0.604 22.34 / 0.438 21.77 / 0.437 19.96 / 0.412 1m25s ASD-POCS 29.65 / 0.775 28.34 / 0.765 25.91 / 0.721 23.93 / 0.661 22.63 / 0.616 20.04 / 0.512 48s

###### Diffusion-Based Iterative Methods

DiffusionMBIR 33.29 / 0.856 30.54 / 0.818 26.28 / 0.733 29.35 / 0.781 27.15 / 0.735 23.01 / 0.581 11h15m DDS 32.56 / 0.817 31.13 / 0.788 28.66 / 0.767 26.21 / 0.554 25.21 / 0.512 23.29 / 0.486 16m17s

###### Neural Representation Methods

SAX-NeRF 28.48 / 0.835 27.91 / 0.832 26.11 / 0.812 23.72 / 0.704 23.20 / 0.690 21.50 / 0.639 4h9m NAF 28.62 / 0.833 28.20 / 0.833 26.22 / 0.812 23.85 / 0.712 23.18 / 0.692 21.37 / 0.618 7m15s +DiffNR (Ours) 31.27 / 0.951 30.79 / 0.946 28.10 / 0.906 26.27 / 0.867 25.15 / 0.839 22.98 / 0.765 8m41s R2-Gaussian 28.56 / 0.695 26.36 / 0.634 22.63 / 0.537 24.11 / 0.577 22.06 / 0.497 18.32 / 0.364 5m52s +DiffNR (Ours) 33.52 / 0.900 32.92 / 0.895 29.71 / 0.852 28.82 / 0.822 27.43 / 0.793 24.37 / 0.712 11min35s

- Table 1: Quantitative results on ToothFairy and LUNA16 datasets. The best values are in bold, second-best are underlined.

#### DiffNR: Diffusion-Enhanced Neural Representation Optimization

While SliceFixer effectively suppresses artifacts, it may introduce hallucinated details, which is highly undesirable in medical diagnostics. Moreover, this 2D model fails to maintain volumetric consistency, resulting in noticeable interslice jitters. To address these issues, instead of treating SliceFixer as a post-processing module, we integrate it into the NR optimization process. DiffNR pipeline is illustrated in Figure 3, and the algorithm is shown in Algorithm 1.

Enhanced Volumes as Augmented Supervision We begin by optimizing a NR using standard image losses (L1 and SSIM) and low-level 3D regularization (total variation (Rudin, Osher, and Fatemi 1992)) to capture global structures. Every ℓ iterations, we query a volume V˜ ℓ from the current model. We then upsample its slices using bilinear interpolation, apply SliceFixer for artifact correction, and downsample the results to the original resolution, producing a pseudo-reference volume Vˆ ℓ. We show in ablation that this up-downsampling strategy improves reconstruction quality. For the remaining training steps, we augment with an additional 3D supervision between the queried volume V˜ and this reference volume Vˆ ℓ every τ steps. This repairand-augment strategy reduces the frequency of SliceFixer queries, thus preserving the overall optimization efficiency.

Perceptual Loss for Structural Integrity SliceFixer may introduce hallucinated details not perfectly aligned with measured projections. Consequently, directly minimizing voxel-wise L1 loss, as commonly adopted in image supervision, can lead to suboptimal performance. To address this, we adopt a perceptual loss based on 3D SSIM, computed as the average of 2D SSIM scores across axial, sagittal, and coronal planes. This promotes structural coherence and smoothness in underconstrained regions, rather than overfitting to fine-grained, potentially hallucinated details. We use a loss weight λdiff to balance the contribution of 3D SSIM.

### Experiments

#### Experimental Setup

Datasets We use two datasets: ToothFairy (Cipriano et al. 2022) and LUNA16 (Setio et al. 2017). ToothFairy consists of 443 dental scans, split into 393/25/25 for training/validation/testing, respectively. LUNA16 includes 888 chest scans, divided into 838/25/25. We train a separate SliceFixer on each dataset and apply the corresponding model for test-case reconstruction. We follow Lin et al. (2024); Zha, Zhang, and Li (2022) to preprocess raw CT volumes to a resolution of 2563 and X-ray projections to 2562. Sparse-view reconstruction is defined as using fewer than a hundred views, and we evaluate the challenging 36-, 24-, and 12-view settings.

Implementation Details SliceFixer is finetuned from SDTurbo on 5122 images, which are upsampled from 2562 slices. We integrate LoRA layers with ranks of 8 for the UNet and 4 for the VAE, and train the model with a learning rate of 1e−5 for 40k steps on ToothFairy and 70k steps on LUNA16, using a batch size of 4. Loss weights are set to λCLIP = 4, λGAN = 0.4, and λSSIM = 0.5. Finetuning is performed on 4 H100 GPUs. DiffNR is implemented in PyTorch and optimized using the Adam optimizer (Kingma 2014). We use NAF and R2-Gaussian as backbones, training them for 11k and 13.5k epochs, respectively, while keeping other hyperparameters unchanged. We empirically set ℓ = 10k, and use τ = 20 for NAF and τ = 10 for R2Gaussian. Pseudo-reference volumes have a resolution of 2563. All test-case reconstructions are performed on an RTX 3090 GPU. The code and model will be publicly available.

Compared Methods and Evaluation We compare with widely-used optimization-based methods, including (1) traditional iterative methods SART (Andersen and Kak 1984) and ASD-POCS (Sidky and Pan 2008), (2) self-supervised NR methods SAX-NeRF (Cai et al. 2024), NAF (Zha, Zhang, and Li 2022), and R2-Gaussian (Zha et al. 2024), and (3) diffusion-based iterative methods: DDS (Chung, Lee, and Ye 2023) and DiffusionMBIR (Chung et al. 2023). We

- 21.05 / 0.437 21.74 / 0.589 26.87 / 0.722 25.18 / 0.503 23.37 / 0.688 23.48 / 0.698 25.92 / 0.845 21.55 / 0.494 27.71 / 0.798 24-view
- 22.42 / 0.519 23.51 / 0.679 29.30 / 0.800 26.64 / 0.584 24.87 / 0.755 25.00 / 0.770 27.65 / 0.911 24.67 / 0.645 29.68 / 0.872 36-view

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

[Figure 65]

[Figure 66]

[Figure 67]

28.56 / 0.711 28.47 / 0.762 30.36 / 0.792 31.34 / 0.768 30.38 / 0.913 30.36 / 0.902 31.96 / 0.916 26.92 / 0.677 34.19 / 0.908 12-view

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

SART ASD-POCS DiffusionMBIR DDS SAX-NeRF NAF +DiffNR (Ours) R2-Gaussian +DiffNR (Ours) Ground Truth

Figure 4: Qualitative results of reconstructed volumes on two datasets, shown from different slicing directions and sparsity levels. We annotate PSNR/SSIM on the top-left of each image. DiffNR recovers finer details and effectively suppresses artifacts.

OOD Dataset (Zha et al. 2024) Methods 36-view 24-view 12-view

PSNR / SSIM PSNR / SSIM PSNR / SSIM

SART 30.50 / 0.740 29.43 / 0.721 27.53 / 0.695 ASD-POCS 32.28 / 0.852 30.16 / 0.811 27.36 / 0.750

DiffusionMBIR 33.26 / 0.839 30.97 / 0.796 26.82 / 0.668 DDS 29.45 / 0.638 26.97 / 0.536 25.17 / 0.520

R2-Gaussian 35.64 / 0.904 33.46 / 0.868 29.71 / 0.792 +DiffNR (Ours) 35.99 / 0.918 34.15 / 0.896 31.04 / 0.848

- Table 2: Quantitative results on the OOD dataset. The best values are in bold, second-best are underlined

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

25.89 / 0.631 28.58 / 0.800 31.33 / 0.882 12-view

Mount

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

23.37 / 0.749 26.84 / 0.895 27.15 / 0.927 24-view

Broccoli

DiffusionMBIR R2-Gaussian +DiffNR (Ours) Ground Truth

Figure 5: Qualitative results on OOD dataset.

quantitatively evaluate all methods using standard metrics PSNR and SSIM.

#### Results

In-Distribution Performance Table 1 presents quantitative results on ToothFairy and LUNA16. Traditional methods and self-supervised NR approaches produce significant artifacts. While diffusion-based methods achieve higher scores, they come at the cost of hallucinated details and significant computation time. Previous SOTA DiffusionMBIR takes 11 hours to process a single case. In contrast, our DiffNR consistently enhances NR baselines, yielding an average improvement of +2.19 dB in PSNR for NAF and +5.79 dB for R2-Gaussian. Although DiffNR introduces additional optimization time, it remains substantially faster than prior diffusion-based methods. Qualitative comparisons are provided in Figure 4, where DiffNR recovers fine structures and

substantially reduces artifacts present in NR baselines.

Out-of-Distribution Performance To evaluate generalization capability, we use SliceFixer pretrained on ToothFairy and apply R2-Gaussian+DiffNR to dataset from Zha et al. (2024), which includes 18 diverse cases spanning human organs, biological specimens, and artificial objects. Notably, this dataset contains real-world captured projections. Quantitative and qualitative results are shown in Table 2 and Figure 5, respectively. DiffNR outperforms other methods by suppressing hallucinations and artifacts, which shows that SliceFixer learns generalizable artifact patterns.

Downstream Application We further validate our method on downstream medical tasks such as segmentation. Specifically, we use the LungMask toolkit (Hofmanninger et al. 2020) to perform left/right lung segmentation on the reconstructed volumes. We use Dice (Dice 1945) and average sur-

36-view 24-view 12-view Dice↑/ASD↓ Dice↑/ASD↓ Dice↑/ASD↓

Methods

SART 81.89 / 11.73 74.12 / 16.63 56.92 / 26.62 ASD-POCS 76.47 / 15.71 70.06 / 19.64 57.98 / 22.27

DiffusionMBIR 90.33 / 6.13 86.96 / 6.97 77.75 / 11.96 DDS 80.03 / 16.66 75.98 / 16.60 68.23 / 19.20

R2-Gaussian 90.41 / 5.19 84.32 / 8.39 59.73 / 25.11

###### +DiffNR (Ours) 93.74 / 3.85 90.71 / 5.60 84.93 / 9.59

- Table 3: Quantitative results for lung segmentation of reconstructed results on LUNA16 dataset.

ID Res. SD-Turbo Pretrain Lssim Bip. Proj. PSNR SSIM

- (1) 256 ✓ 27.65 0.789

- (2) 512 ✓ 27.91 0.807

- (3) 512 ✓ ✓ 28.21 0.814

- (4) 512 ✓ ✓ ✓ 28.82 0.822

- Table 4: Ablation study of SliceFixer. We finetune different models and evaluate DiffNR under LUNA16 36-view cases.

face distance (ASD) metrics to evaluate performance. As shown in Table 3 and Figure 6(a), the segmentation masks generated from Gaussian-based DiffNR are more consistent with those obtained from the ground-truth volumes, demonstrating the practical utility of our method.

#### Ablation Study

SliceFixer Design We validate design choices of SliceFixer in Table 4 and Figure 6(b). We find that finetuning SliceFixer on 5122 images and applying up-downsampling to queried slices leads to better reconstruction quality compared to using the original 2562 resolution. Additionally, incorporating an SSIM loss into finetuning resulting in a 0.3 dB gain in PSNR. Finally, adding biplanar projections as additional conditioning inputs provides rich structural cues and further boosts finetuning performance by 0.6 dB in PSNR.

DiffNR Design We use R2-Gaussian as backbone to validate components of DiffNR as shown in Table 5. Augmenting NRs with novel-view images, commonly used in RGB surface reconstruction (Wu et al. 2025), is ineffective in volume reconstruction. This is because errors in penetrable projections can accumulate to the target volume across views. Instead, we choose to augment slice supervision, which proves to be more stable and effective. Moreover, applying SliceFixer as a standalone post-processing step leads to slice jitter and hallucinations (Figure 6(c)), highlighting the necessity of integrating it into the optimization pipeline. Lastly, we find that using voxel-wise L1 loss results in a performance drop, as the pseudo-reference volumes may contain details inconsistent with measured projections. A 3D perceptual loss is thus preferred. Overall, integrating our proposed components leads to the best performance.

Parameter Analysis We perform parameter analysis for Gaussian-based DiffNR to investigate the impact of 3D

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

GT (1) 256 Res. (2) 512 Res.

GT DiffusionMBIR

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

R2-Gaussian +DiffNR

Input (4) Bip. Proj. (3) SSIM

(a) Lung Segmentation (b) SliceFixer Ablation

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

GT SliceFixer DiffNR GT SliceFixer DiffNR

(c) SliceFixer vs. DiffNR

Figure 6: Qualitative results of downstream tasks and ablation study. (a) Lung segmentation with the left lung in blue and the right lung in red. (b) Visualization of different design choices for SliceFixer. (c) Comparison of standalone SliceFixer post-processing and our proposed DiffNR.

Methods PSNR SSIM R2-Gaussian 24.11 0.577 + Difix3D+ (augment projection) 23.23 0.579 + SliceFixer (post-processing) 26.70 0.776 + SliceFixer (L1) 26.42 0.678 + SliceFixer (Lssim) (Ours) 28.82 0.822

- Table 5: Ablation study of DiffNR design on LUNA16 dataset under 36-view setting.

λdiff 0.3 0.5 0.7 1.0 1.5 PSNR 28.65 28.82 28.79 28.72 28.63 τ 5 10 15 20 30

PSNR 28.76 28.82 28.67 28.43 27.87 TIME 27m35s 12m56s 10m02s 8m32s 7m26s

- Table 6: Ablation study of DiffNR hyperparameters on LUNA16 dataset (36-view) with our choices in bold.

SSIM loss weight λdiff and 3D supervision frequency τ. As shown in Table 6, λdiff = 0.5 achieves the best performance by balancing the guidance from 3D supervision and avoiding overfitting to projections or degradation from diffusion hallucination. For the supervision interval, τ = 10 yields optimal results. More frequent supervision (e.g., τ = 5) may lead to over-reliance on the 3D loss and increased computational cost, whereas sparse supervision (e.g., τ = 20) weakens structural regularization and degrades performance.

### Conclusion

We present DiffNR, a novel optimization framework for sparse-view 3D tomographic reconstruction. At its core is SliceFixer, a single-step diffusion model finetuned on curated datasets to correct artifacts in reconstructed CT slices. During reconstruction, the pretrained SliceFixer gen-

erates pseudo-reference volumes that provide augmented perceptual regularization. Such a repair-and-augment strategy avoids frequent diffusion model queries, therefore improving reconstruction quality without sacrificing efficiency. Experimental results demonstrate that DiffNR outperforms prior methods in reconstruction quality, generalization capability, and optimization efficiency, highlighting its practical potential. Further, this novel integration of diffusion models with neural representation optimization opens a promising direction for addressing broader classes of inverse problems.

### Acknowledgments

This research is supported in part by the Jiangsu Department of Technology Natural Science Fund (Grants No: BK20250441), the Center of Excellence for Antimicrobial Therapeutics Discovery and Innovation (CEATDI, Grants No: 8002003), and the ARC Discovery Grant (Grant ID: DP220100800) of the Australia Research Council.

### References

Andersen, A. H.; and Kak, A. C. 1984. Simultaneous algebraic reconstruction technique (SART): a superior implementation of the ART algorithm. Ultrasonic imaging, 6(1): 81–94.

Biguri, A.; Dosanjh, M.; Hancock, S.; and Soleimani, M. 2016. TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction. Biomedical Physics & Engineering Express, 2(5): 055010.

Cai, Y.; Wang, J.; Yuille, A.; Zhou, Z.; and Wang, A. 2024. Structure-aware sparse-view x-ray 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11174–11183.

Chu, J.; Du, C.; Lin, X.; Zhang, X.; Wang, L.; Zhang, Y.; and Wei, H. 2025. Highly accelerated MRI via implicit neural representation guided posterior sampling of diffusion models. Medical Image Analysis, 100: 103398.

Chung, H.; Lee, S.; and Ye, J. C. 2023. Decomposed diffusion sampler for accelerating large-scale inverse problems. arXiv preprint arXiv:2303.05754.

Chung, H.; Ryu, D.; McCann, M. T.; Klasky, M. L.; and Ye,

- J. C. 2023. Solving 3d inverse problems using pre-trained 2d diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22542– 22551.

Cipriano, M.; Allegretti, S.; Bolelli, F.; Di Bartolomeo, M.; Pollastri, F.; Pellacani, A.; Minafra, P.; Anesi, A.; and Grana, C. 2022. Deep segmentation of the mandibular canal: a new 3D annotated dataset of CBCT volumes. IEEE Access, 10: 11500–11510.

Dice, L. R. 1945. Measures of the amount of ecologic association between species. Ecology, 26(3): 297–302.

Du, C.; Lin, X.; Wu, Q.; Tian, X.; Su, Y.; Luo, Z.; Zheng, R.; Chen, Y.; Wei, H.; Zhou, S. K.; et al. 2024. DPER: Diffusion prior driven neural representation for limited angle and sparse view CT reconstruction. arXiv preprint arXiv:2404.17890.

Feldkamp, L. A.; Davis, L. C.; and Kress, J. W. 1984. Practical cone-beam algorithm. Josa a, 1(6): 612–619.

Gu, J.; Trevithick, A.; Lin, K.-E.; Susskind, J. M.; Theobalt, C.; Liu, L.; and Ramamoorthi, R. 2023. Nerfdiff: Singleimage view synthesis with nerf-guided distillation from 3daware diffusion. In International Conference on Machine Learning, 11808–11826. PMLR.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33: 6840–6851.

Hofmanninger, J.; Prayer, F.; Pan, J.; R¨ohrich, S.; Prosch, H.; and Langs, G. 2020. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. European radiology experimental, 4(1): 50.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W.; et al. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2): 3.

Jin, K. H.; McCann, M. T.; Froustey, E.; and Unser, M. 2017. Deep convolutional neural network for inverse problems in imaging. IEEE transactions on image processing, 26(9): 4509–4522.

Kak, A. C.; and Slaney, M. 2001. Principles of computerized tomographic imaging. SIAM.

Kamilov, U. S.; Bouman, C. A.; Buzzard, G. T.; and Wohlberg, B. 2023. Plug-and-play methods for integrating physical and learned models in computational imaging: Theory, algorithms, and applications. IEEE Signal Processing Magazine, 40(1): 85–97.

Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G. 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139–1.

Kingma, D. P. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Li, Y.; Fu, X.; Li, H.; Zhao, S.; Jin, R.; and Zhou, S. K. 2025. 3DGR-CT: Sparse-view CT reconstruction with a 3D Gaussian representation. Medical Image Analysis, 103585.

Lin, Y.; Yang, J.; Wang, H.; Ding, X.; Zhao, W.; and Li, X. 2024. Cˆ 2rv: Cross-regional and cross-view learning for sparse-view cbct reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11205–11214.

Liu, X.; Chen, J.; Kao, S.-H.; Tai, Y.-W.; and Tang, C.K. 2024. Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-Observations for High-Quality Sparse-View Reconstruction. In European Conference on Computer Vision, 337–355. Springer.

Liu, X.; Zhou, C.; and Huang, S. 2024. 3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with viewconsistent 2d diffusion priors. Advances in Neural Information Processing Systems, 37: 133305–133327.

Ma, C.; Li, Z.; Zhang, J.; Zhang, Y.; and Shan, H. 2023. FreeSeed: Frequency-band-aware and self-guided network for sparse-view CT reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 250–259. Springer.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.

Parmar, G.; Park, T.; Narasimhan, S.; and Zhu, J.-Y.

- 2024. One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036.

P´erez-Garc´ıa, F.; Sharma, H.; Bond-Taylor, S.; Bouzid, K.; Salvatelli, V.; Ilse, M.; Bannur, S.; Castro, D. C.; Schwaighofer, A.; Lungren, M. P.; Wetscherek, M. T.; Codella, N.; Hyland, S. L.; Alvarez-Valle, J.; and Oktay, O.

- 2025. Exploring scalable medical image encoders beyond text supervision. Nature Machine Intelligence.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR.

Rudin, L. I.; Osher, S.; and Fatemi, E. 1992. Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4): 259–268.

Sauer, A.; Lorenz, D.; Blattmann, A.; and Rombach, R. 2024. Adversarial diffusion distillation. In European Conference on Computer Vision, 87–103. Springer.

Setio, A. A. A.; Traverso, A.; De Bel, T.; Berens, M. S.; Van Den Bogaard, C.; Cerello, P.; Chen, H.; Dou, Q.; Fantacci, M. E.; Geurts, B.; et al. 2017. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge. Medical image analysis, 42: 1–13.

Sidky, E. Y.; and Pan, X. 2008. Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization. Physics in Medicine & Biology, 53(17): 4777.

Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2020. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.

Tian, X.; Chen, L.; Wu, Q.; Du, C.; Shi, J.; Wei, H.; and Zhang, Y. 2025. Unsupervised Self-Prior Embedding Neural Representation for Iterative Sparse-View CT Reconstruction. Proceedings of the AAAI Conference on Artificial Intelligence, 39(7): 7383–7391.

Vo, R.; Escoda, J.; Vienne, C.; and Decenci`ere, E.´ 2024. Neural Field Regularization by Denoising for 3D SparseView X-Ray Computed Tomography. In 2024 International Conference on 3D Vision (3DV), 1166–1176. IEEE.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.

Warburg, F.; Weber, E.; Tancik, M.; Holynski, A.; and Kanazawa, A. 2023. Nerfbusters: Removing ghostly artifacts from casually captured nerfs. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 18120–18130.

Wu, J. Z.; Zhang, Y.; Turki, H.; Ren, X.; Gao, J.; Shou, M. Z.; Fidler, S.; Gojcic, Z.; and Ling, H. 2025. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 26024–26035.

Zha, R.; Lin, T. J.; Cai, Y.; Cao, J.; Zhang, Y.; and Li, H. 2024. R2-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction. In Advances in Neural Information Processing Systems (NeurIPS).

Zha, R.; Zhang, Y.; and Li, H. 2022. NAF: neural attenuation fields for sparse-view CBCT reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 442–452. Springer.

Zhang, G.; Zha, R.; He, H.; Liang, Y.; Yuille, A.; Li, H.; and Cai, Y. 2025. X-lrm: X-ray large reconstruction model for extremely sparse-view computed tomography recovery in one second. arXiv preprint arXiv:2503.06382.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3836–3847.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586–595.

Zhou, Z.; and Tulsiani, S. 2023. Sparsefusion: Distilling view-conditioned diffusion for 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12588–12597.

