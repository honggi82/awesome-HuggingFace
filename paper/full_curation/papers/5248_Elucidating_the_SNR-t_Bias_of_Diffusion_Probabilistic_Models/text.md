## Elucidating the SNR-t Bias of Diffusion Probabilistic Models

Meng Yu1,2*, Lei Sun2†, Jianhao Zeng2, Xiangxiang Chu2, Kun Zhan1‡

1Lanzhou University, 2AMAP Alibaba Group

# arXiv:2604.16044v1[cs.CV]17Apr2026

### Abstract

Diffusion Probabilistic Models have demonstrated remarkable performance across a wide range of generative tasks. However, we have observed that these models often suffer from a Signal-to-Noise Ratio-timestep (SNR-t) bias. This bias refers to the misalignment between the SNR of the denoising sample and its corresponding timestep during the inference phase. Specifically, during training, the SNR of a sample is strictly coupled with its timestep. However, this correspondence is disrupted during inference, leading to error accumulation and impairing the generation quality. We provide comprehensive empirical evidence and theoretical analysis to substantiate this phenomenon and propose a simple yet effective differential correction method to mitigate the SNR-t bias. Recognizing that diffusion models typically reconstruct low-frequency components before focusing on high-frequency details during the reverse denoising process, we decompose samples into various frequency components and apply differential correction to each component individually. Extensive experiments show that our approach significantly improves the generation quality of various diffusion models (IDDPM, ADM, DDIM, A-DPM, EA-DPM, EDM, PFGM++, and FLUX) on datasets of various resolutions with negligible computational overhead. The code is at https://github.com/AMAP-ML/DCW.

### 1. Introduction

Due to their outstanding performance, Diffusion Probabilistic Models (DPMs) [17, 48, 51] have achieved remarkable success in various generative tasks, including image [11, 45], audio [6, 21], and video [4, 19, 68] generation. DPMs typically consist of two processes. In the forward process, a data sample is progressively perturbed by Gaussian noise until it becomes the standard Gaussian noise. In the reverse process, DPMs iteratively denoise from the standard Gaussian noise to generate the clean data sample. Despite their significant success, we identify that DPMs suffer severely

*Work done during the internship at AMAP Alibaba Group. †Project leader. ‡Corresponding author. Email: kzhan@lzu.edu.cn

from a Signal-to-Noise Ratio–timestep (SNR-t) bias.

The SNR-t bias refers to the misalignment between the SNR of predicted samples and their assigned timesteps during inference. Specifically, during training, the neural network is conditioned on both the perturbed sample and the corresponding timestep, establishing a deterministic correspondence between the SNR of the sample and the timestep. However, during inference, due to cumulative errors arising from both the model’s predictions [20] and the numerical solvers [31, 51], the denoising trajectory inevitably deviates from the ideal path, causing a misalignment between the SNR of the predicted sample and its designated timestep, as shown in Fig. 1a. Unlike previously studied exposure bias [38], which focuses on inter-sample discrepancies, the SNR-t bias emphasizes the misalignment between the predicted sample and its corresponding timestep. We argue that the SNR-t bias is a more fundamental bias that can induce exposure bias and is prevalent in current DPMs.

We provide a comprehensive experimental analysis and theoretical justification for SNR-t bias. Our experiments reveal two key findings: (1) the network demonstrates significantly inaccurate predictions when processing samples with mismatched SNR and timesteps. Specifically, as illustrated in Fig. 1b, samples with lower SNR tend to make the network produce larger noise predictions, while those with higher SNR yield smaller noise predictions. (2) Reverse denoising samples often exhibit lower SNR compared to their corresponding forward samples at the same timestep, as shown in Fig. 1c. These findings lead to a notable conclusion: the SNR-t bias severely degrades the model performance and often manifests as lower SNR for the corresponding timestep during the denoising process. To investigate the underlying mechanisms, we analyze the reverse process of DPMs and provide a theoretical proof of this bias, thereby offering a robust theoretical justification for our findings.

To mitigate the SNR-t bias, a natural solution is to align the distribution of reverse samples, which tends to have lower SNR, with the corresponding distribution of forward samples. Given the complexity of existing DPM frameworks, training or fine-tuning approaches would incur significant costs. Instead, we propose a dynamic differential correction method in the wavelet domain, which leverages the model’s

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|| θ(x10, 10)||2 || θ(x16, 16)||2 || θ(x10, 10)||2 || θ(x16, 16)||2

| |
|---|

| |
|---|

| |
|---|

| |
|---|

||||(x,s) θt2

||·||(,t) θ2

| |
|---|

s = 10 s = 16

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

Train Inference

t

t

(a) Schematic of SNR-t bias in DPMs.

(b) SNR-t bias causes inaccurate predictions.

(c) The reverse process exhibits lower SNR.

- Figure 1. (a) During training, the SNR of perturbed sample xt is strictly tied to timestep t. However, during inference, due to network prediction errors and discretization errors in numerical solvers, the SNR of predicted sample xˆt no longer matches the preset timestep t. (b) shows the network output ||ϵθ(xt, s)||2 when a trained network ϵθ(·, s) with fixed timestep s receives samples xt with mismatched SNR (samples are generated via forward process using Eq. 2 with different t). (c) shows the network output ||ϵθ(·, t)||2 using forward samples and reverse predicted samples, respectively. ||ϵθ(xˆt, t)||2 is always larger than ||ϵθ(xt, t)||2, which indicates that predicted samples exhibit lower SNR compared to forward samples at the same timestep. See the experiment details of (b) and (c) in Sec. 4.

inherent capabilities to alleviate the bias without additional training. Specifically, at each denoising step, we obtain the reconstruction sample, which directly predicts the clean sample from the current predicted sample. By analytically modeling the prediction distribution and the reconstruction distribution, we find that their difference signal contains gradient information that can guide the biased predicted sample toward the ideal perturbed sample. We incorporate this differential signal into each denoising step to ensure the predicted distribution aligns more closely with the perturbed distribution, thereby effectively mitigating the bias.

Additionally, to improve the correction effect, we introduce the method into the wavelet domain, allowing it to correct different frequency components of samples separately. This approach leverages the unique denoising characteristics [42, 61] of DPMs, which initially emphasize the reconstruction of low-frequency contours during the reverse process before focusing on restoring high-frequency details. Meanwhile, we assign dynamic weight coefficients to the correction operations for different components. By applying targeted corrections for varying frequency components at different stages of the denoising process, we achieve significant improvements in corrections and overall performance. Notably, our method can further enhance the performance of improved models [38, 39, 64] for exposure bias, which highlights the significance and superiority of our proposed problem and method. In summary, our contributions are:

- • We identify the SNR-t bias in DPMs and provide comprehensive experimental analysis and theoretical proof.
- • We propose a dynamic differential correction method in wavelet domain to effectively alleviate the SNR-t bias.
- • Our method is training-free and plug-and-play, effectively improving the generation quality of various DPMs. It can also be extended to other bias-correction models with significant gains and negligible computation.

### 2. Related Work

This section first reviews the development of DPMs, followed by some recent works on bias analysis in DPMs.

The foundational theory of DPMs is introduced by DPM [48], with major advances brought by DDPM [17]. ADM [11] employs classifier guidance to make DPMs outperform GANs [13], while EDM [18] systematically explores the training and inference design space to further boost generation quality and efficiency of DPMs. Notably, ODE-based DPMs [12, 31, 67, 69], knowledge distillation-based DPMs [28, 32, 34, 47], and consistency models [24, 30, 50, 52] are widely studied. Meanwhile, DPMs have advanced downstream tasks like text-to-image models [3, 7, 23, 45], image editing [10, 33, 40], and superresolution generation [15, 25, 46]. Furthermore, USP [9], SY-TDM [35], FE2E [56], S2-Guidance [5], and ADECOT [43] improve DPMs from different perspectives.

Research on exposure bias is closely related to our work. Exposure bias in DPMs refers to the sample mismatch between training and sampling. ADM-IP [38] re-perturbs training data to imitate the discrepancies in inference, exposing the model to possible prediction errors. MDSS [44] interprets exposure bias as deviations between predicted samples and network outputs and adopts a multi-step denoising schedule to reduce it. EP-DDPM [27] derives an upper bound on accumulated errors and incorporates it as a retraining regularizer to lessen the bias. While these models require retraining, TS-DPM [26] and ADM-ES [39] offer training-free, plugand-play alternatives. In addition, MCDO [60], DPM-AT [66], DPM-AE [57], BMGDM [63], and DPM-FR [64] also analyze and mitigate this bias from different perspectives.

Exposure bias acts across samples, whereas SNR-t bias arises between samples and timesteps.

### 3. Background

In this section, we review the preliminaries of DPMs.

DPMs generally comprise a forward process and a reverse process, with both formulated as Markov chains. Given a target data distribution q(x0) and a variance schedule βt, the forward process is defined as

T

q(xt|xt−1), (1)

q(x1:T|x0) =

t=1

where q(xt|xt−1) = N(xt;√1 − βtxt−1,βtI). Utilizing the attributes of the Gaussian distribution, the perturbed sample xt is directly expressed in a closed form as the conditional distribution q(xt|x0):

xt = √α¯tx0 + √1 − α¯tϵt, (2)

where αt = 1−βt, α¯t = ti=1 αi, and ϵt ∼ N(0,I). Then, by applying Bayes’ theorem, the corresponding posterior

distribution can be expressed as:

q(xt−1|xt,x0) = N(µ˜t(xt,x0),β˜tI), (3) where

√α¯t−1βt 1 − α¯t

√αt(1 − α¯t−1) 1 − α¯t

1 − α¯t−1 1 − α¯t

xt, β˜t =

µ˜t =

x0+

βt.

A neural network pθ(xt−1|xt) = N(xt−1;µθ(xt,t),σtI) is employed to approximate q(xt−1|xt,x0), which aims to minimize DKL(q(xt−1|xt,x0)∥pθ(xt−1|xt)). Through reparameterization, we are able to obtain:

√α¯t−1βt 1 − α¯t

√αt(1 − α¯t−1) 1 − α¯t

x0θ(xt,t) +

µθ(xt,t) =

##### xt

1 − αt √1 − α¯t

1 √αt

xt −

=

ϵθ(xt,t) ,

(4)

where x0θ(xt,t) represents the reconstruction of x0 given xt, and ϵθ(·) denotes the noise prediction network. Specifically, the relationship between the two is:

√1 − α¯tϵθ(xt,t) √α¯t

xt −

x0θ(xt,t) =

. (5)

Finally, we obtain the concise training objective:

0,ϵt∼N(0,I) ∥ϵθ(xt,t) − ϵt∥22 . (6)

Lsimple = Et,x

Once the noise prediction network is trained to convergence, we can start from a standard Gaussian noise, perform stepby-step iterative denoising via pθ(xt−1|xt), and ultimately generate the clean data sample.

### 4. SNR-t Bias

In this section, we present the specific definition of SNR-t bias and elaborate on two key findings.

The DPM takes the perturbed sample xt and the timestep t as input during training, as shown in Fig. 1a, and the SNR of xt is directly determined by the timestep t:

SNR(t) = α¯t/(1 − α¯t). (7)

Due to the forced binding between the SNR of samples and timesteps during the training phase, the network ϵθ(·,t) is proficient in accurately predicting samples with a corresponding SNR(t). But what happens if the network ϵθ(·,s) receives a sample xt with a mismatched SNR(t)?

To validate this, we design and conduct an experiment to assess the network predictions using samples with the mismatched SNR. Specifically, we select the ADM [11] model as our baseline model and utilize 2,000 samples from the CIFAR-10 [22] dataset. We first fix the timestep as s for the network ϵθ(·,s) and then generate a series of forward perturbed samples {x0,x1,··· ,xt,··· ,xT} using Eq. 2. These perturbed samples are subsequently fed into the network ϵθ(·,s), after which we compute their mean squared norm and present the results in Fig. 1b.

Key Finding 1. The network produces significantly inaccurate predictions when processing samples with mismatched SNR and timesteps. As illustrated in Fig. 1b, this bias exhibits a specific pattern: for the fixed timestep s, when handling the input sample xt with a lower SNR, where t > s, the network tends to overestimate the predicted output. In contrast, when dealing with the sample xt with a higher SNR, the predicted output is typically underestimated. In summary, samples with lower SNR lead the network to produce larger noise predictions, while those with higher SNR result in smaller noise predictions.

With the Key Finding 1 highlighting the significant performance degradation caused by SNR-t bias in DPMs, a natural subsequent question arises: how does SNR-t bias exactly manifest during the actual denoising process?

The inference process in DPMs can be understood as a numerical solution to a Stochastic Differential Equation (SDE) or an Ordinary Differential Equation (ODE), which inevitably introduces discretization errors during numerical computations. Additionally, the neural network within DPMs is subject to inherent prediction errors. Consequently, these two types of errors can cause the reverse denoising trajectory to deviate from the ideal path, resulting in a mismatch between the actual SNR of the reverse predicted samples xˆt and the designated timestep t. Thus, the actual reverse denoising process can be expressed as:

1 √αt

xˆt−1 =

1 − αt √1 − α¯t

ϵθ(xˆt,t) + σtz. (8)

x ˆt −

To further investigate the manifestations of SNR-t bias, we adopt the same experimental setup as in Fig. 1b and conduct the following comparative experiment. (1) We generate perturbed samples {x1,x2,...,xT} via Eq. 2, and feed xt and timestep t into the network to obtain ϵθ(xt,t). (2) Then, we initialize 2,000 samples of standard Gaussian noise and perform iterative denoising via Eq. 8 to obtain samples {xˆ1,xˆ2,...,xˆT} and corresponding network outputs ϵθ(xˆt,t). (3) Finally, we compute and plot the expectation of ℓ2 norms ||ϵθ(xt,t)||22 and ||ϵθ(xˆt,t)||22, as shown in Fig. 1c. Particularly, similar experiments were also conducted in ADM-ES [39], and we provide the evidence of the differences, together with more robust analyses in Appendix A. Building on this, we derive the second key finding:

Key Finding 2. Reverse denoising samples often exhibit lower SNR compared to their corresponding forward samples at the same timestep. Fig. 1c shows that for any timestep t, the mean ℓ2 norm of reverse predictions ϵθ(xˆt,t) consistently exceeds that of forward predictions ϵθ(xt,t). The Key Finding 1 shows that the network tends to produce an overestimated output when processing samples with lower SNR. Therefore, we have reason to conclude that the denoising sample xˆt generally maintains a lower SNR than the forward perturbed sample xt at the same timestep, leading to overestimated predictions at each denoising step.

### 5. Method

In this section, we first analytically model the reverse process of DPMs and derive the analytical form of the SNR-t bias, providing a comprehensive theoretical basis for this bias. Then, based on the theoretical analysis, we propose a simple yet effective differential correction method to mitigate the SNR-t bias, thereby improving the generation quality of DPMs. Finally, by incorporating the denoising laws of DPMs, we introduce differential correction into the wavelet domain and design a specialized weighting strategy to further enhance the correction effect.

#### 5.1. Theoretical Proof

For the theoretical analysis of bias in DPMs, prior works have proposed two distinct assumptions. ADM-ES [39] and TS-DPM [26] propose the following formulation:

###### x0θ(xt,t) = x0 + ϕtϵt, (9)

where ϵt ∼ N(0,I), with ϕt a scalar coefficient. LADPM [65] and DPM-FR [64] propose another formulation:

x0θ(xt,t) = γtx0 + ϕtϵt, with γt also a scalar coefficient. Unfortunately, these prior assumptions are overly strong and

lack sufficient theoretical grounding and empirical validation. Furthermore, there is a clear discrepancy in the coefficient of x0 between the two hypotheses. To address this issue,

we conduct extensive theoretical and experimental analyses in this work, and ultimately decide to adopt the second hypothesis for our subsequent analysis.

Assumption 5.1. During both the forward and reverse processes, the reconstruction sample x0θ(xt,t) can be expressed in terms of the original data x0 as follows:

###### x0θ(xˆt,t) = γtx0 + ϕtϵt, (10)

where 0 < γt ⩽ 1, ϕt < M, and M denotes a uniform upper bound constant across all timesteps.

Sketch of Proof. x0θ(xt,t) is the reconstruction output for predicting x0 given xt, which is also known as the posterior mean E[x0|xt] [54]. Based on Tweedie’s formula [54] and the L2-norm loss function [18], DPMs tend to predict the mean value of the target data. Thus, x0θ can be viewed as the mean prediction x¯0 of x0. By the variance identity E[∥x0∥2] = ∥x¯0∥2 + Var(∥x0∥) and the non-negativity of variance, we get

∥x¯0∥2 ⩽ E[∥x0∥2]. Since the expectation of a constant is itself, we can obtain:

E[||x0θ||2] ⩽ E[∥x0∥2]. (11)

The assumption x0θ = x0 + ϕtϵt implies that E[||x0θ||2] = E[||x0||2] + ϕ2t. Obviously, this conflicts with Eq. 11. Thus, a more accurate formulation is given in Eq. 10, where γt < 1 denotes energy and information loss during the reconstruction of x0. In particular, ASBGM [36] also provides indirect evidence for this view. Furthermore, more theoretical and experimental evidence is provided in Appendix B.

Based on Assumption 5.1, we can derive the analytical form of the SNR for xˆt in the reverse process:

Theorem 5.1. For a specific timestep t in the reverse denoising process of DPMs, the SNR of the biased denoising sample xˆt is given by:

√α¯tβt+1 1 − α¯t+1

SNR(t) = γˆt2α¯t/ 1 − α¯t + (

ϕt+1)2 , (12)

where 0 < γˆt ⩽ 1 and ϕt+1 is derived from the reconstruction model x0θ(xˆt+1,t + 1) in Eq. 10.

Sketch of Proof. For the sake of brevity of the formula, we present the denoising process from xˆt to xˆt−1. By substituting the reconstruction model in Eq. 5 into the inverse denoising Eq. 8, we can obtain:

√αt−1βt 1 − αt

√αt(1 − αt−1) 1 − αt

xt+ β˜tϵ1, (13)

x0θ(xt,t)+

xˆt−1 =

SNR of xˆt in the inverse process is always lower than that of xt at the same timestep t in the forward process. Thus, we can infer that if we move the predicted sample toward the perturbed sample, the SNR-t bias can be alleviated. Interestingly, this gradient information pushing xˆt toward xt is implicitly contained in each step of the denoising process. Now, we focus on the differential signal between the predicted sample xˆt−1 and the reconstructed sample x0θ(xˆt,t) in Eq. 14. Based on Eq. 15 for xˆt−1 and Eq. 10 for x0θ(xˆt,t), the differential signal is expressed as:

Type SNR xt Forward α¯t/(1 − α¯t) xˆt Reverse γˆt2α¯t/ 1 − α¯t + (

√α¯tβt+1

1−α¯t+1 ϕt+1)2

Table 1. The actual SNR of xt and xˆt.

DPM

γt γˆt−1

xˆt−1 − x0θ(xˆt,t) = γˆt−1(xt−1 −

x0) + ηtϵt (16)

DWT

DWT DCW iDWT

[Figure 1]

[Figure 2]

where ηt = ϕ2t + ψt2−1. Obviously, the differential signal based on Eq. 16 contains directional information pointing to xt−1. Inspired by various directional information guidance strategies [55, 65], we integrate this gradient information into each step of denoising to guide the predicted samples xˆt−1 to move toward the ideal perturbed samples xt−1:

iDWT

Pixel Space Wave Domain

- Figure 2. The overall framework of Differential Correction in Wavelet domain (DCW). At each denoising step, DPMs always generate the reconstructed sample x0θ for predicting x0 based on xt. After each denoising is completed, DCW maps x0θ and xt−1

xˆt−1 = xˆt−1 + λt(xˆt−1 − x0θ(xˆt,t)), (17)

where λt is a scalar guidance factor that adjusts the magnitude of the effect of the differential signal. More specifically, the difference guidance shifts the predicted sample toward the noisy direction targeting xt−1. When the parameter is properly selected, it will improve the accuracy of the predicted sample to mitigate the SNR-t bias.

to the wavelet domain via DWT to obtain xfθ and xft−1, where f ∈ {ll, lh, hl, hh}. Then, DCW corrects the different frequency

components of xt−1 using Eq. 18. Finally, DCW maps the corrected xft−1 back to the pixel space via iDWT.

- where ϵ1 ∼ N(0,I). Substituting Eqs. 10 and 2 into Eq. 13 yields the analytical form of xˆt−1:

xˆt−1 = γˆt−1√α¯t−1x0+ 1 − α¯t−1 +

√α¯t−1βt 1 − α¯t

ϕt

2

ϵ2,

(14)

- where ϵ2 ∼ N(0,I). By substituting timestep t + 1 into Eq. 14, we can calculate the actual SNR of the predicted

sample xˆt, thereby completing the proof of Theorem 5.1. With the aid of the forward noising Eq. 2, a more concise expression form is obtained:

xˆt−1 = γˆt−1xt−1 + ψt−1ϵ3. (15)

- where ϵ3 ∼ N(0,I), with more details in Appendix C. Tab. 1 and Eq. 15 clearly show that the actual SNR of

We emphasize that correcting xˆt−1 is more advantageous than correcting xˆt, as it not only enhances the quality of generation more effectively but also incurs less computational overhead. Specifically, Eq. 13 shows that the denoising re-

sult xˆt−1 of the current step t is jointly influenced by xˆt and x0θ(xˆt,t) (or ϵθ(xˆt,t)). Meanwhile, the acquisition of x0θ(xˆt,t) indicates the network has completed prediction. Thus, without increasing Neural Function Evaluations (NFE), Eq. 17 can correct xˆt−1 and has no effect on the network output. Additionally, correcting the denoising result xˆt−1 will bring gains to both the predicted sample and the network output in the next denoising process.

#### 5.3. Differential Correction in Wavelet Domain

In this subsection, we introduce the Differential Correction method into the Wavelet domain (DCW), as shown in Fig. 2, which stems from two key motivations: (1) During inference, DPMs first focus on reconstructing the low-frequency contours of images and then concentrate on the high-frequency details [61]. Thus, our method should align with this important characteristic of DPMs; (2) The direction indicated by the differential signal based on Eq. 16 is disturbed by Gaussian noise ηtϵt, thus performing correction in the timefrequency domain helps reduce noise interference.

the predicted samples xˆt in the reverse process is always lower than that of the perturbed sample xt in the forward process, thus there is always a SNR-t bias where the SNR of predicted samples does not match the timestep t during the inference phase, which provides solid theoretical evidence for the experimental conclusions in Sec. 4.

#### 5.2. Differential Correction in Pixel Space

In Sec. 4 and Sec. 5.1, we clarify the SNR-t bias of DPMs and its specific manifestations from both empirical and theoretical perspectives. Meanwhile, we find that the actual

Specifically, during the denoising process, DCW employs Discrete Wavelet Transform (DWT) [14] to decompose xˆt

and x0θ(xˆt,t) into four frequency subbands. For a given image sample x in the pixel space, after DWT is applied to

x, the following are obtained: xll, xlh, xhl, and xhh, where the size of all four subbands is RH/2×W/2. xll represents the low-frequency subband, which characterizes the lowfrequency information of the image, such as the shape of a human face or a house. xlh,xhl, and xhh correspond to the high-frequency subbands in different directions, which characterize the high-frequency information of the image, such as the wrinkles of an elderly person or the veins of leaves. Subsequently, we separately perform differential correction on each type of frequency subband:

xˆft−1 = xˆft−1 + λft (xˆft−1 − xfθ(xˆt,t)), (18)

where f ∈ {ll,lh,hl,hh}, λft is an adjustment coefficient related to both timesteps and frequency components. Then, we utilize the inverse discrete wavelet transform (iDWT) [14] to map the samples back to the pixel space, thereby forming a complete DCW operation:

x˜t−1 = iDWT(xˆft−1|f ∈ {ll,lh,hl,hh}) (19)

Next, we discuss the adjustment strategy for λft . For the low-frequency component, we propose a time-dependent weighting strategy that follows a decaying schedule as the denoising advances. Conversely, a decreasing strategy is adopted for the high-frequency components. Specifically, in early denoising steps, we assign a relatively large coefficient to the low-frequency correction term to prioritize the generation of low-frequency components, which also effectively mitigates the interference of high-frequency noise errors during the initial denoising phase. In the later denoising stages, we assign a larger coefficient to the high-frequency correction to focus on the restoration of high-frequency details, which helps suppress the over-expression of low-frequency components towards the end of the process.

Notably, the reverse process variance σt in DPMs serves as a robust indicator of the denoising progress and has been widely adopted for dynamic modulation in various sampling techniques [11, 53, 64]. Consequently, we leverage this reverse variance to implement our dynamic correction. The low-frequency component coefficient is formulated as:

λlt = λl · σt, (20)

where λl denotes a scalar coefficient. Similarly, the highfrequency component coefficient is defined as:

λht = (1 − λh)σt, (21)

where λh also denotes a scalar coefficient. Furthermore, inspired by SG-Minority [53], more weight design strategies are provided in Appendix D.

### 6. Experiments

In this section, we conduct extensive experiments on numerous datasets and DPMs to show the effectiveness, generality, superiority, and robustness of our method.

We evaluate it on multiple representative DPM frameworks and samplers, including IDDPM [37], ADM [11], DDIM [49], A-DPM [2], EA-DPM [1], EDM [18], DiT [41], PFGM++ [59], FLUX [3], and Qwen-Image [58]. Then, we choose DPM-AE [57] (ICLR 2025) and DPM-AT [66] (ICLR 2025) as comparative models. Furthermore, we also integrate our method into the open-source bias-corrected models ADM-IP [38] (ICML 2023), ADM-ES [39] (ICLR 2024), and DPM-FR [64] (ACM MM 2025) to further demonstrate the superiority of our approach. Meanwhile, experiments are conducted across datasets of varying resolutions, including CIFAR-10 [22], CelebA 64×64 [29], ImageNet 128×128 [8], and LSUN Bedroom 256×256 [62].

Overall, we categorize our evaluations into two main types: stochastic generation [17] and deterministic generation [49]. To comprehensively assess generation quality, we employ standard metrics including Fr´echet Inception Distance (FID) [16] and Recall [16], where FID serves as the primary metric. All quantitative results are computed over 50K generated samples with the full training set as the reference distribution. For qualitative evaluation, we visualize text-to-image results to intuitively demonstrate the effectiveness of our method.

#### 6.1. Results on Classic Diffusion Models

To verify the effectiveness and generality of the proposed method, we select several classic diffusion models, namely IDDPM, ADM, and ADM-IP. Additionally, we choose datasets with different resolutions, including CIFAR-10 [22] 32 × 32, CelebA 64 × 64 [29], ImageNet 128 × 128 [8], and LSUN Bedroom 256 × 256 [62]. Meanwhile, we select FID and Recall as evaluation metrics to assess fidelity and diversity, and use 20 and 50 as sampling steps.

Tab. 2 clearly shows that our method comprehensively improves the generation quality of the baseline models across all models and datasets. For example, on the CIFAR-10 dataset, DCW helps IDDPM reduce the FID score by 42.6% and 25% in the 20-step and 50-step tasks, respectively.

For a fair comparison with recent methods on exposure bias, we follow previous works and use the same baselines, namely DDIM [49] sampler applied to A-DPM and ADM. Tab. 3 clearly shows our method consistently outperforms DPM-AE [57] (ICLR 2025) and DPM-AT [66] (ICLR 2025) across all generation results, further validating its superiority.

#### 6.2. Results on Bias-Corrected Diffusion Models

To verify the generality and advancement of our method, we select several improved models for exposure bias as comparative models and integrate DCW into them, namely ADM-

###### T = 20 T = 50

Model Dataset FID↓ Rec↑ FID↓ Rec↑ IDDPM CIFAR-10 32 13.19 0.50 5.55 0.56 +Ours CIFAR-10 32 7.57 0.56 4.16 0.58 ADM-IP CelebA 64 11.95 0.42 4.52 0.55 +Ours CelebA 64 10.41 0.47 4.34 0.57 ADM ImageNet 128 12.28 0.52 5.18 0.58 +Ours ImageNet 128 10.34 0.54 4.52 0.58 IDDPM LSUN 256 18.69 0.27 8.42 0.41 +Ours LSUN 256 11.03 0.36 5.24 0.45

Table 2. FID and Recall (Rec) on datasets with different resolutions.

DDIM ADM Model 10 20 50 10 20 50 Base 14.40 6.87 4.15 22.62 10.52 4.55 Base-AE 13.98 6.76 4.10 - - Base-AT - - - 15.88 6.60 3.34 Base+Ours 9.36 4.64 3.33 13.01 5.59 2.95

Table 3. FID ↓ on CIFAR-10 using ADM and DDIM.

ES [39] and DPM-FR [64]. Notably, DPM-FR is the SOTA model for exposure bias. To be consistent with them, we divide the generation task into two categories: stochastic sampling and deterministic sampling. In stochastic sampling, we select A-DPM [2] and NPR-DM in EA-DPM [1] as the baseline models. In deterministic sampling, we use EDM [18] and PFGM++ [59] as baseline models and measure the sampling cost by Neural Function Evaluations (NFE) [55].

- Tab. 4 shows that in stochastic sampling, DCW com-

prehensively improves the generation quality of baseline models. For different models, noise scheduling strategies, and time-step settings, DCW consistently achieves a significant reduction in the FID scores. For the corrected models, even though they have already achieved extremely low FID scores, DCW can still further reduce the FID results, which demonstrates the advancement of our method.

- Tab. 5 shows that in deterministic sampling, DCW can

not only improve the generation quality of baseline models but also further reduce the FID of corrected models. For EDM, DCW reduces the FID by 47.1%, 47.4%, and 36.4% in the 13, 21, and 35 NFE generation tasks, respectively. Although ADM-ES and ADM-FR have already improved generation performance by alleviating exposure bias, DCW can still further improve the corrected models. For EDM-ES the reductions of FID under the three NFE tasks are 7.0%, 5.3%, and 3.5%, respectively. For PFGM-FR, the corresponding reductions are 6.6%, 5.7%, and 2.0%, respectively.

CIFAR-10 (LS) CIFAR-10 (CS) Model 10 25 50 10 25 50

A-DPM 34.26 11.60 7.25 22.94 8.50 5.50 +Ours 17.56 8.81 5.38 12.44 5.99 4.06

A-DPM-FR 12.38 6.63 4.52 11.61 4.40 3.62 +Ours 10.91 6.03 4.44 9.80 4.33 3.56

NPR-DM 32.35 10.55 6.18 19.94 7.99 5.31 +Ours 16.60 8.64 5.40 11.44 6.38 4.80

NPR-DM-FR 10.86 5.76 4.19 10.18 4.07 3.44 +Ours 9.81 5.30 4.11 8.46 3.96 3.33

Table 4. FID ↓ on CIFAR-10 using A-DPM and EA-DPM.

EDM PFGM++ Model 13 21 35 13 21 35 Base 10.66 5.91 3.74 12.92 6.53 3.88

- +Ours 5.67 3.37 2.41 6.98 3.83 2.64 Base-ES 6.59 3.74 2.59 8.79 4.54 2.91

- +Ours 6.13 3.57 2.50 8.00 4.41 2.84

Base-FR 4.68 2.84 2.13 6.62 3.67 2.53 +Ours 4.57 2.79 2.12 6.18 3.46 2.48

Table 5. FID ↓ on CIFAR-10 using different fast samplers.

Meanwhile, we also provide the DiT [41] experiments on the ImageNet 256 × 256 dataset in Appendix E.

#### 6.3. Qualitative Comparison

To intuitively demonstrate the impact of DCW on the generation quality, we set the same random seed and sampling steps for the baseline models and improved models during inference, ensuring that they follow similar denoising trajectories. Specifically, we adopt FLUX [3] as the baseline model and use 10 sampling steps. As shown in Fig. 3, images generated by FLUX suffer from distortion issues such as over-smoothing and overexposure. In contrast, DCW significantly mitigates these problems, substantially enhancing the aesthetic quality and visual appeal of the generated images. More qualitative results are provided in Appendix F, including the qualitative experiments on Qwen-Image [58].

#### 6.4. Ablation Study

In this subsection, we conduct detailed ablation experiments to examine the role of each component in DCW. We primarily use CIFAR-10 as the test dataset.

Effect of the Wavelet Domain. First, we investigate the impact of each component in DCW on generation quality via four comparative variants. Differential correction applied

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- Figure 3. Qualitative comparison between FLUX (first row) and FLUX-DCW (second row) using 10 steps.

Model Type 10 25 50 A-DPM Baseline 22.94 8.50 5.50 A-DPM-DC Pixel Space 15.71 6.38 4.31 A-DPM-DH High Frequency 16.72 6.05 4.06 A-DPM-DL Low Frequency 13.21 7.00 5.10 A-DPM-DCW High & Low 12.46 5.99 4.06

Table 6. Ablation study (FID ↓) of different frequency components.

solely in the pixel space is denoted as “DC”. Then, we denote differential correction applied only to high frequency or low frequency wavelet components as “DH” and “DL”, respectively. Finally, our complete framework involves applying differential correction to both the high frequency and low frequency components, denoted as “DCW”. Tab. 6 clearly shows that the differential correction method is effective in both the pixel and wavelet space, resulting in noticeable improvements in the generation quality. Furthermore, the simultaneous integration of differential correction into both high-frequency and low-frequency components enhances performance even further, underscoring the necessity and advantages of applying the method within the wavelet domain.

Sensitivity of Hyperparameter λf. Next, we examine the sensitivity of DCW to hyperparameters. DCW is robust to variations in hyperparameters: for both low-frequency and high-frequency adjustment factors, the intensity of differential correction gradually increases with the growth of the adjustment parameters, and the FID of the final generated results exhibits a trend of first decreasing and then increasing, as shown in Fig. 4. Therefore, we can quickly determine the optimal values of the hyperparameters through a simple two-stage search method, as presented in Appendix G.

Impact of DCW on computational overhead. Finally, we evaluate the impact of DCW on computational overhead. Without loss of generality, we fix the random seed,

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

λl

λh

(a) Search experiments of λl .

(b) Search experiments of λh .

Figure 4. Hyperparameter search experiments on CIFAR-10 (CS) using A-DPM and EA-DPM with T = 25.

Model Dataset Time DCW Time Overhead

ADM-IP CelebA 64 4.25 4.27 0.47% ADM ImageNet 128 12.59 12.60 0.08% IDDPM LSUN 256 15.57 15.61 0.26%

Table 7. Batch generation time on a single NVIDIA A6000 GPU.

the number of timesteps, and batch size, then conduct extensive experiments on datasets of varying resolutions: CelebA 64×64, ImageNet 128×128, and LSUN Bedroom 256×256. To address statistical bias, each experiment is repeated 100 times, and the average runtime is reported. Tab. 7 demonstrates that the computational cost incurred by DCW for DPMs is negligible, introducing virtually no generation latency. Specifically, DCW adds an additional time overhead of approximately 0.47%, 0.08%, and 0.26% for the three generation tasks, which is clearly minimal. These experimental results regarding time overhead further reinforce the practicality and superiority of DCW.

### 7. Conclusion

In conclusion, we find that DPMs often suffer from a signalto-noise ratio–timestep (SNR-t) bias. This bias refers to the mismatch between the SNR of a denoising sample and its associated timestep during inference. During training, the SNR of a sample is a deterministic function of its timestep, but this coupling is broken at inference due to accumulated prediction and discretization errors, which leads to error accumulation and degraded generation quality. We provide empirical evidence and theoretical analysis for this phenomenon and propose a simple differential correction method to mitigate the SNR-t bias. Since diffusion models tend to reconstruct low-frequency components before refining high-frequency details in the reverse process, we decompose samples into multiple frequency components and apply differential correction to each component separately. Extensive experiments show that our approach improves the generation quality of various diffusion models on datasets with different resolutions, while incurring negligible computational overhead.

### References

- [1] Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, and Bo Zhang. Estimating the optimal covariance with imperfect mean in diffusion probabilistic models. In ICML, 2022. 6, 7
- [2] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. AnalyticDPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In ICLR, 2022. 6, 7
- [3] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 6, 7
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 1
- [5] Chubin Chen, Jiashu Zhu, Xiaokun Feng, Nisha Huang, Chen Zhu, Meiqi Wu, Fangyuan Mao, Jiahong Wu, Xiangxiang Chu, and Xiu Li. Stochastic self-guidance for training-free enhancement of diffusion models. In ICLR, 2026. 2
- [6] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In ICLR, 2021. 1
- [7] Ruidong Chen, Yancheng Bai, Xuanpu Zhang, Jianhao Zeng, Lanjun Wang, Dan Song, Lei Sun, Xiangxiang Chu, and Anan Liu. Layer-wise instance binding for regional and occlusion control in text-to-image diffusion transformers. arXiv preprint arXiv:2603.05769, 2026. 2
- [8] Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of ImageNet as an alternative to the CIFAR datasets. arXiv preprint arXiv:1707.08819, 2017. 6
- [9] Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding. In ICCV, 2025. 2
- [10] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. DiffEdit: Diffusion-based semantic image editing with mask guidance. In ICLR, 2023. 2
- [11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In NeurIPS, 2021. 1, 2, 3, 6
- [12] Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Genie: Higher-order denoising diffusion solvers. In NeurIPS, 2022. 2
- [13] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS,

2014. 2

- [14] Amara Graps. An introduction to wavelets. IEEE Computational Science and Engineering, 1995. 5, 6
- [15] Haodong He, Xin Zhan, Yancheng Bai, Rui Lan, Lei Sun, and Xiangxiang Chu. Texts-diff: Texts-aware diffusion model for real-world text image super-resolution. arXiv preprint arXiv:2601.17340, 2026. 2
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 6, 5
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 1, 2, 6

- [18] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 2, 4, 6, 7
- [19] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In ICCV, 2023. 1
- [20] Dongjun Kim, Yeongmin Kim, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon. Refining generative process with discriminator guidance in score-based diffusion models. In ICML,

2023. 1

- [21] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. DiffWave: A versatile diffusion model for audio synthesis. In ICLR, 2021. 1
- [22] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 3, 6
- [23] Rui Lan, Yancheng Bai, Xu Duan, Mingxing Li, Dongyang Jin, Ryan Xu, Lei Sun, and Xiangxiang Chu. Flux-text: A simple and advanced diffusion transformer baseline for scene text editing. arXiv preprint arXiv:2505.03329, 2025. 2
- [24] Jiachen Lei, Keli Liu, Julius Berner, Y HoiM, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. There is no VAE: Endto-end pixel-space generative modeling via self-supervised pre-training. In ICLR, 2026. 2
- [25] Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. Srdiff: Single image super-resolution with diffusion probabilistic models. Neurocomputing, 2022. 2
- [26] Mingxiao Li, Tingyu Qu, Ruicong Yao, Wei Sun, and MarieFrancine Moens. Alleviating exposure bias in diffusion models through sampling with shifted time steps. In ICLR, 2024. 2, 4, 3
- [27] Yangming Li and Mihaela van der Schaar. On error propagation of diffusion models. In ICLR, 2024. 2
- [28] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In ICLR, 2023. 2
- [29] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In ICCV, 2015. 6
- [30] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. In ICLR, 2025. 2
- [31] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS,

2022. 1, 2

- [32] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021. 2
- [33] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2
- [34] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, 2023. 2

- [35] Yingmao Miao, Zhanpeng Huang, Rui Han, Zibin Wang, Chenhao Lin, and Chao Shen. Shining yourself: High-fidelity ornaments virtual try-on with diffusion model. In CVPR,

2025. 2

- [36] Amitoj Singh Miglani, Shweta Singh, and Vidit Aggarwal. Analysing the spectral biases in generative models. In The Fourth Blogpost Track at ICLR 2025, 2025. 4
- [37] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICLR, 2021. 6
- [38] Mang Ning, Enver Sangineto, Angelo Porrello, Simone Calderara, and Rita Cucchiara. Input perturbation reduces exposure bias in diffusion models. In ICML, 2023. 1, 2, 6
- [39] Mang Ning, Mingxiao Li, Jianlin Su, Albert Ali Salah, and Itir Onal Ertugrul. Elucidating the exposure bias in diffusion models. In ICLR, 2024. 2, 4, 6, 7, 3, 5
- [40] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH, 2023. 2
- [41] William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, 2023. 6, 7, 5
- [42] Yurui Qian, Qi Cai, Yingwei Pan, Yehao Li, Ting Yao, Qibin Sun, and Tao Mei. Boosting diffusion models with moving average sampling in frequency domain. In CVPR, 2024. 2
- [43] Xiangyan Qu, Zhenlong Yuan, Jing Tang, Rui Chen, Datao Tang, Meng Yu, Lei Sun, Yancheng Bai, Xiangxiang Chu, Gaopeng Gou, et al. From scale to speed: Adaptive test-time scaling for image editing. arXiv preprint arXiv:2603.00141,

2026. 2

- [44] Zhiyao Ren, Yibing Zhan, Liang Ding, Gaoang Wang, Chaoyue Wang, Zhongyi Fan, and Dacheng Tao. Multi-step denoising scheduled sampling: Towards alleviating exposure bias for diffusion models. In AAAI, 2024. 2
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 2
- [46] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image superresolution via iterative refinement. TPAMI, 2022. 2
- [47] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 2
- [48] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 1, 2
- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 6
- [50] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In ICLR, 2024. 2
- [51] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 1
- [52] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 2
- [53] Soobin Um and Jong Chul Ye. Self-guided generation of minority samples using diffusion models. In ECCV, 2024. 6
- [54] Soobin Um, Suhyeon Lee, and Jong Chul Ye. Don’t play favorites: Minority guidance for diffusion models. In ICLR,

2024. 4, 3

- [55] Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. In NeurIPS, 2021. 5, 7
- [56] JiYuan Wang, Chunyu Lin, Lei Sun, Rongying Liu, Lang Nie, Mingxing Li, Kang Liao, Xiangxiang Chu, and Yao Zhao. From editor to dense geometry estimator. arXiv preprint arXiv:2509.04338, 2025. 2
- [57] Zekun Wang, Mingyang Yi, Shuchen Xue, Zhenguo Li, Ming Liu, Bing Qin, and Zhi-Ming Ma. Improved diffusion-based generative model with better adversarial robustness. In ICLR,

2025. 2, 6

- [58] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 6, 7
- [59] Yilun Xu, Ziming Liu, Yonglong Tian, Shangyuan Tong, Max Tegmark, and Tommi Jaakkola. PFGM++: Unlocking the potential of physics-inspired generative models. In ICML,

2023. 6, 7

- [60] Yuzhe YAO, Jun Chen, Zeyi Huang, Haonan Lin, Mengmeng Wang, Guang Dai, and Jingdong Wang. Manifold constraint reduces exposure bias in accelerated diffusion sampling. In ICLR, 2025. 2
- [61] Mingyang Yi, Aoxue Li, Yi Xin, and Zhenguo Li. Towards understanding the working mechanism of text-to-image diffusion model. In NeurIPS, 2024. 2, 5
- [62] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015. 6
- [63] Meng Yu and Kun Zhan. Bias mitigation in graph diffusion models. In ICLR, 2025. 2
- [64] Meng Yu and Kun Zhan. Frequency regulation for exposure bias mitigation in diffusion models. In ACM MM, 2025. 2, 4, 6, 7, 3
- [65] Guoqiang Zhang, Kenta Niwa, and W Bastiaan Kleijn. Lookahead diffusion probabilistic models for refining mean estimation. In CVPR, 2023. 4, 5, 3
- [66] Junyu Zhang, Daochang Liu, Eunbyung Park, Shichao Zhang, and Chang Xu. Anti-exposure bias in diffusion models. In ICLR, 2025. 2, 6
- [67] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. In NeurIPS, 2024. 2
- [68] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 1
- [69] Zhenyu Zhou, Defang Chen, Can Wang, and Chun Chen. Fast ODE-based sampling for diffusion models in around 5 steps. In CVPR, 2024. 2

## Elucidating the SNR-t Bias of Diffusion Probabilistic Models Supplementary Material

### A. Difference from Prior Works

In this section, we outline the differences between the second experiment (Fig. 1c) in Sec 4 of this paper and prior work [38, 39]. We emphasize that ADM-ES [39] only provides a phenomenological conclusion and does not delve into the underlying causes of the phenomenon. In contrast, the SNRt bias discovered in this paper, along with the sliding window experiments on neural networks based on Fig. 1b, provide in-depth explanations and evidence for this phenomenon. Additionally, this section offers more robust experimental analyses for the phenomenon.

- (1) The SNR-t bias is the underlying cause of exposure

bias proposed by ADM-IP [38] and ADM-ES [39]. ADMIP and ADM-ES define the exposure bias as an intuitively inter-sample bias between the perturbed sample xt and the predicted sample xˆt. Meanwhile, ADM-ES also claims that exposure bias leads to the accumulation of errors, yet it fails to provide fundamental evidence for such error accumulation. In contrast, we explicitly demonstrate when the SNR of the input sample mismatches the timestep, the network’s predictive output exhibits significant errors, as shown in the Key Finding 1 (Fig. 1b). Furthermore, since the SNR of reverse-process samples is consistently lower than the ideal level, as shown in the Key Finding 2 (Fig. 1c), the network’s predictions during the reverse process are invariably erroneous, specifically manifesting as overestimated outputs. In summary, the SNR-t bias stems primarily from the forced coupling of sample SNR and timestep during training.

- (2) Unlike ADM-ES, this paper focuses on drawing

deeper conclusions and uncovering the underlying patterns. Specifically, Figure 2 in ADM-ES concludes that the L2norm of ϵθ(xˆt,t) in the reverse process is always larger than that of ϵθ(xt,t) in the forward process. However, ADM-ES does not explore the deep-seated reasons for this overestimation phenomenon. In this paper, we derive Finding 1 through the sliding window experiments in Sec. 4: for the fixed timestep s, when handling the sample xt with a lower SNR, where t > s, the network tends to overestimate the predicted output. Conversely, when dealing with the sample xt with a higher SNR, the predicted output is typically underestimated. Therefore, combining the findings of ADM-ES and Finding 1 of this paper, we arrive at Finding 2: Reverse denoising samples often exhibit lower SNR compared to their corresponding forward samples at the same timestep.

- (3) Unlike exposure bias, an inter-sample bias, the SNR-t

bias is a more specific SNR-timestep bias. Meanwhile, our method based on the SNR-t bias can be naturally integrated into state-of-the-art models for correcting exposure bias,

such as ADM-IP, ADM-ES, and DPM-FR, further improving the generation quality of these correction models as shown in Sec. 6.2. Additionally, our method can significantly enhance the generation quality in the latest text-to-image models, as shown in Appendix E. Thus, these experiments further illustrate the differences between SNR-t bias and exposure bias, as well as the necessity of researching SNR-t bias.

Furthermore, we also provide more robust experimental evidence for Fig. 1c to eliminate interference caused by random seeds and sampling batch sizes. Specifically, we fix the sampling batch size at 2000 and then select different random number seeds (16, 42, and 99) to obtain distinct sampling trajectories, as illustrated in Figs. 5a, 5b, and 5c, respectively. Subsequently, we fix the random number seed and vary the sampling batch sizes (10, 100, and 1000), as shown in Figs. 5d, 5e, and 5f, respectively. Fig. 5 clearly demonstrates that regardless of the random number seed and sampling batch size, the network output of the reverse process is consistently larger than that of the forward process, which provides more robust evidence for our analysis.

### B. Theoretical evidence of Assumption 5.1

Assumption 5.1. During both the forward and reverse processes, the reconstruction sample x0θ(xt,t) can be expressed in terms of the original data x0 as follows:

###### x0θ(xˆt,t) = γtx0 + ϕtϵt, (22)

where 0 < γt ⩽ 1, ϕt < M, and M denotes a uniform upper bound constant across all timesteps.

Specifically, we emphasize that both the forward reconstructed sample x0θ(xt,t) and the reverse reconstructed sample x0θ(xˆt,t) adhere to the form specified in Eq. 22.

In this section, we present the detailed proof of Assumption 5.1. As stated in the main text, previous work proposed two distinct linear assumptions but lacked supporting evidence. However, we provide both experimental evidence and theoretical proofs to support our findings. Under Gaussian perturbation qσ(y|x), the Tweedie’s formula is

E[x|y] = y + σ2∇ylogqσ(y), (23)

where qσ(y) := q(y|x)q(x)dx. Now, by substituting the forward perturbation distribution q(xt|x0) of DPMs into Eq. 23, we can obtain:

logq(xt) √α¯t

xt + (1 − α¯t)∇xt

. (24)

E[x0|xt] =

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

||·||(,t) θ2

||·||(,t) θ2

||·||(,t) θ2

t

t

t

(a) Seed=16, Batch Size=2000

(b) Seed=42, Batch Size=2000

(c) Seed=99, Batch Size=2000

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

||·||(,t) θ2

||·||(,t) θ2

||·||(,t) θ2

t

t

t

(d) Seed=42, Batch Size=10

(e) Seed=42, Batch Size=100

(f) Seed=99, Batch Size=1000

- Figure 5. Robust experimental results for Fig. 1c with varied random number seeds and sampling batch sizes. These figures show the network output ||ϵθ(·, t)||2 using forward samples xt via Eq. 2 and reverse predicted samples xˆt via Eq. 8, respectively. ||ϵθ(xˆt, t)||2 is always larger than ||ϵθ(xt, t)||2 in every figure.

Based on the relationship between the score and the noise sθ(xt,t) = −ϵ

θ(xt,t)

√1−α¯t , we further derive:

√1 − α¯tϵθ(xt,t) √α¯t

xt −

= x0θ(xt,t), (25)

E[x0|xt] =

which clearly demonstrates that the reconstructed sample x0θ(xt,t) is essentially the posterior mean based on the Tweedie formula. Furthermore, the score network trained with the L2 norm-MSE loss function always have a theoretical analytical solution [54], which is also the posterior mean:

log q(xt | x0)]. (26)

sθ(xt,t) = Eq(x

0|xt) [∇xt

Based on the equivalence between the score and noise, the optimal solution for noise prediction is also the same posterior mean. Therefore, based on the mean tendency of denoising operations and network predictions, we can regard x0θ(xt,t) as the mean estimate x¯0 of x0.

The variance formula is expressed as:

E[∥x0∥2] = ∥x¯0∥2 + Var(∥x0∥). (27) Based on the non-negativity of the variance, we obtain:

###### ∥x¯0∥2 ⩽ E[∥x0∥2].

We substitute x0θ(xt,t) for x¯0, then given that the expectation of a constant is the constant itself, we can take the

expectation of both sides of the above equation to obtain:

###### E[∥x0θ(xt,t)∥2] ⩽ E[∥x0∥2]. (28)

Eq. 28 clearly demonstrates that the L2 norm of reconstructed samples is always smaller than that of real samples, which indicates that the reconstruction operation is always accompanied by information loss.

However, previous work [26, 39] argues that reconstructed samples should be modeled as:

###### x0θ(xt,t) = x0 + ϕtϵt, (29)

which is clearly inconsistent with Eq. 28. Thus, We use the form in Eq. 22, consistent with the assumption of LADPM [65] and DPM-FR [64].

In addition, we also provide experimental evidence for the above proof. Following the experimental setup described in Sec. 4, we perform the following operations sequentially: (1) We generate perturbed samples {x1,x2,...,xT} via Eq. 2, and feed xt and timestep t into the network to obtain ϵθ(xt,t) to compute x0θ(xt,t) via Eq. 5. (2) Then, we initialize 2,000 standard Gaussian noise and iteratively denoise operation via Eq. 8 to obtain samples {xˆ1,xˆ2,...,xˆT} and corresponding network outputs ϵθ(xˆt,t) to compute x0θ(xˆt,t) via Eq. 5. (3) Finally, we compute and plot the expectation of ||x0θ(xt,t)||22, ||x0θ(xˆt,t)||22, and ||x0||22.

Fig. 6 clearly demonstrates that DPMs fail to fully reconstruct real data x0, both in the forward and reverse processes.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

2

t

- Figure 6. The expectation of ||x0θ(xt, t)||22, ||x0θ(xˆt, t)||22, together with the ground-truth norm of x0.

This further indicates that the reconstruction operation incurs information loss. Notably, similar experiments are also reported in DPM-FR [64]. However, it focuses on the differences between the forward and reverse processes, whereas our work places greater emphasis on whether DPMs can fully reconstruct real data. Additionally, we argue that conducting experiments in the data space is more persuasive. This experiment further demonstrates that x0θ(xt,t) and x0θ(xˆt,t) adhere to the form specified in Eq. 22.

### C. Proofs of Theorem 5.1 and Eq. 15

In this section, we present the detailed proofs of Theorem 5.1 and Eq. 15. Our derivation process is mainly based on DPMFR [64]. However, we provide a more rigorous derivation process, particularly for γt and γˆt. Specifically, we focus on SNR, the core theme of this work.

Theorem 5.1. For a specific timestep t in the reverse denoising process of DPMs, the SNR of the biased denoising sample xˆt is given by:

√α¯tβt+1 1 − α¯t+1

SNR(t) = γˆt2α¯t/ 1 − α¯t + (

ϕt+1)2 , (30)

where 0 < γˆt ⩽ 1 and ϕt+1 is derived from the reconstruction model x0θ(xˆt+1,t + 1) in Eq. 10.

Firstly, we emphasize that all subsequent noise terms ϵ follow the standard Gaussian distribution. We rewrite the fundamental formula of DPMs and the forward noising process is expressed as:

xt = √α¯tx0 + √1 − α¯tϵ0. (31)

We assume the current predicted sample is ideal. Thus, the reverse denoising process is expressed as:

1 √αt

xˆt−1 =

1 − αt √1 − α¯t

ϵθ(xt,t) + σtϵ1. (32)

xt −

Then, substituting Eq. 25 into Eq. 32, we can obtain an equivalent form of the reverse denoising process:

√α¯t−1βt 1 − α¯t

√αt(1 − α¯t−1) 1 − α¯t

xt+ β˜tϵ1,

x0θ(xt,t)+

xˆt−1 =

(33) By substituting Eqs. 31 and 22 into Eq. 33 to replace x0θ(xt,t) and xt, we can obtain:

√α¯t−1βt 1 − α¯t

xˆt−1 =

(γtx0 + ϕtϵt)+ √αt(1 − α¯t−1) 1 − α¯t

(√α¯tx0 + √1 − α¯tϵ0) + β˜tϵ1

√αt(1 − α¯t−1)√α¯t 1 − α¯t

√α¯t−1βtγt 1 − α¯t

x0 + β˜tϵ1

=

+

√αt(1 − α¯t−1)√1 − α¯t 1 − α¯t

√α¯t−1βtϕt 1 − α¯t

ϵt +

##### ϵ0

+

(34) For Eq. 34, we first focus on the coefficient of x0:

√αt(1 − α¯t−1)√α¯t 1 − α¯t

√α¯t−1βtγt 1 − α¯t

+

(35)

√α¯t−1 (1 − αt)γt + αt(1 − α¯t−1) 1 − α¯t

=

.

Given that γt ⩽ 1, we use the scaling method to amplify it to 1, yielding the following inequality:

√α¯t−1 (1 − αt)γt + αt(1 − α¯t−1)

1 − α¯t ⩽

√α¯t−1 (1 − αt) + αt(1 − α¯t−1) 1 − α¯t

(36)

= √α¯t−1

Given that 1 − αt > 0,γt ⩽ 1, We may rigorously define a novel coefficient γˆt−1 ⩽ 1 for xˆt−1 where

√α¯t−1 (1 − αt)γt + αt(1 − α¯t−1) 1 − α¯t

γˆt−1√α¯t−1 =

.

(37)

For the standard Gaussian noise component in Eq. 34, based on the properties of the Gaussian distribution, we define a new coefficient ψˆt−1 such that:

√α¯t−1βt 1 − α¯t

√αt(1 − α¯t−1) 1 − α¯t

ψˆt−1 = (

1 − α¯t−1)2 + β˜t

ϕt)2 + (

√α¯t−1βt 1 − α¯t

αt(1 − α¯t−1)2 1 − α¯t

(1 − α¯t−1)(1 − αt) 1 − α¯t

ϕt)2 +

+

= (

√α¯t−1βt 1 − α¯t

αt(1 − α¯t−1)2 + (1 − α¯t−1)(1 − αt) 1 − α¯t

ϕt)2 +

= (

√α¯t−1βt 1 − α¯t

ϕt)2 + 1 − α¯t−1.

= (

(38)

Based on Eqs. 37 and (38), we can obtain

√α¯t−1βt 1 − α¯t

xˆt−1 = γˆt−1√α¯t−1x0 + 1 − α¯t−1 + (

ϕt)2ϵt−1

(39) Ultimately, based on Eq. 39, we obtain the SNR of xt−1 as:

√α¯t−1βt 1 − α¯t

SNR(t − 1) = γˆt2−1α¯t−1/ 1 − α¯t−1 + (

ϕt)2

(40) By replacing the timestep in Eq. 40, we ultimately obtain the actual SNR of xˆt to complet the proof.

To obtain a more concise and intuitive form, we use the piecing-together method to derive:

√α¯t−1βt 1 − α¯t

ψˆt2−1 = (

ϕt)2 + (1 − γˆt2−1)(1 − α¯t−1)

+ γˆt2−1(1 − α¯t−1)

In conclusion, we have obtained the biased mean and variance of the reverse process:

xˆt−1 = γˆt−1√α¯t−1x0 + +ˆγt−1 (1 − α¯t−1)ϵˆ3

√α¯t−1βt 1 − α¯t

(41)

ϕt)2 + (1 − γˆt2−1)(1 − α¯t−1)ϵ˜3

+ (

= γˆt−1xt + ψt−1ϵ3,

√α¯t−1βt

where ψt−1 = (

1−α¯t ϕt)2 + (1 − γˆt2−1)(1 − α¯t−1). Thus, we have completed the proof of Eq. 15. Finally, we emphasize again that γt is the coefficient of the reconstruction sample x0θ(xt,t) in Eq. 22, and γˆt−1 is the coefficient of the predicted sample xˆt−1 in Eqs. 39 and 41.

### D. Weight Strategy Design

The denoising process of DPM inherently follows a coarseto-fine paradigm: the early stages primarily generate lowfrequency global structures, while the later stages progressively recover high-frequency details. To this end, our proposed differential correction method is designed to align with this intrinsic property, prioritizing low-frequency correction in the initial phases and shifting focus to high-frequency correction in the later stages.

Based on the above reasoning, we assign larger correction coefficients to low-frequency components in the early stage of denoising and higher weighting coefficients to highfrequency components in the later stage of denoising. On this basis, we propose three weighting scheduling strategies.

Firstly, considering that the variance σt in the reverse process of DPM can dynamically characterize the denoising progress, we adopt the weighting forms shown in Eqs. 20 and 21 in the main text. Second, we design a piecewise weighting strategy. For the timestep t (0 ⩽ t < T) and

Table 8. FID and Recall (Rec) on DiT.

###### T = 20 T = 50

Model Dataset FID↓ Rec↑ FID↓ Rec↑ DiT ImageNet 256 12.83 0.54 3.78 0.58 DiT-ES ImageNet 256 10.00 - 3.30 -

##### DiT+Ours ImageNet 256 7.99 0.51 3.09 0.56

threshold ts, based on empirical experience, we classify t > ts as the early stage of denoising and t ⩽ ts as the later stage of denoising. Accordingly, the piecewise weight for low-frequency components can be defined as:

wtl = wl · I{t ⩾ ts}, (42)

where I(·) denotes the indicator function. In a similar vein, the piecewise weight for high-frequency components is naturally defined as:

wth = wh · I{t < ts}. (43)

Furthermore, to simplify the implementation, we also design a constant weighting strategy, where the weights remain unchanged throughout the entire denoising process.

In particular, we emphasize that all three aforementioned weighting strategies are effective after extensive experimental evaluations, as shown in Sec.6. Specifically, the variance-based scheduling strategy and the piecewise weighting strategy achieve superior generation quality, which further demonstrates the necessity of aligning the weight design with the denoising dynamics of DPMs.

### E. Additional Results

Given the extensive influence of transformer-based diffusion models, we select DiT [41] as the baseline model, ADMES [39] as the comparative model. Subsequently, we adopt Fr´echet Inception Distance (FID) [16] and Recall [16] as evaluation metrics, and select ImageNet 256 × 256 as the test dataset for our experiments.

Tab. 8 clearly demonstrates that our method achieves a comprehensive reduction in the FID scores of DiT and outperforms the comparative models significantly. In the subsequent appendix, we also provide the evaluation results of two text-to-image models, which are also based on the DiT architecture.

### F. Qualitative Comparison

To show the improvement effect of DCW on the generation quality of DPMs, we select two state-of-the-art text-toimage models, namely Qwen-Image, which demonstrates strong instruction following and text rendering ability, and

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

- Figure 7. Qualitative comparison between Qwen-Image (first row) and Qwen-Image-DCW (second row) using 10 steps, where the prompt is “A woman is walking on the beach by the sea”.

FLUX, which is known for its high visual fidelity, to conduct extensive experiments. Given that our study focuses on the SNR-t bias, we conduct tests with a small number of steps to amplify the sampling errors of the baseline models as much as possible, thereby verifying how effectively DCW corrects such bias. As shown in Figs. 7, 8, 9, 10, 11, 12, 13, 14, and 15, our method can significantly enhance the aesthetic quality across different models and time steps.

Specifically, as shown in Figs. 7 and 10, our method consistently improves the visual quality of the generated images under a small number of sampling steps. Compared with the original models, our method produces results with more coherent scene structure, better semantic fidelity, and clearer details. It also alleviates common artifacts caused by sampling bias, leading to images that are more natural and visually appealing. These results demonstrate that DCW is effective across different baseline models and can reliably enhance generation quality in low-step sampling settings. Moreover, the improvements are consistently observed across diverse scenes and content types, further highlighting the robustness and generality of our method.

### G. Parameter sensitivity

To demonstrate the insensitivity of DCW to hyperparameters λl and λh, we first apply DCW to A-DPM to obtain the

Table 9. The search process of λl and λh on CIFAR-10 (CS) using A-DPM-DCW with 25 sampling steps.

Value 0.02 0.03 0.04 0.05 0.06 0.07 0.08 FID 7.64 7.37 7.24 7.18 7.19 7.35 7.66

optimal parameter λl on CIFAR-10 (CS). Then, based on the optimal parameter λl, we apply DCW to obtain the optimal parameter λh. Fig. 4 clearly shows that DCW can achieve performance gains over a wide range of λl and λh, indicating the insensitivity of DCW to hyperparameters.

Benefiting from the strong robustness of the proposed method to hyperparameter perturbations, the parameter search process is fast via the two-stage search. Firstly, a coarse search with a step size of 0.01 was performed. After identifying a turning point in the FID curve around 0.05, we conducted a fine-grained search with a step size of 0.001 and quickly determined the optimal value to be 0.052, as shown in Tab 9. Then, after fixing the optimal λ∗l at 0.052, quickly derive the optimal parameter λ∗h = 0.010 using the same method. In summary, the above experimental process further demonstrates the robustness and practicality of our method with respect to hyperparameters.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

###### Figure 8. Qualitative comparison between Qwen-Image (first row) and Qwen-Image-DCW (second row) using 10 steps, where the prompt is “There is a house and a path on a snowy mountain”.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

###### Figure 9. Qualitative comparison between Qwen-Image (first row) and Qwen-Image-DCW (second row) using 10 steps, where the prompt is “A balloon gently climbs into a serene blue sky”.

[Figure 30]

[Figure 31]

[Figure 32]

###### Figure 10. Qualitative comparison between FLUX (first row) and FLUX-DCW (second row) using 10 steps, where the prompt is “There is a house and a path on a snowy mountain”.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

###### Figure 11. Qualitative comparison between FLUX (first row) and FLUX-DCW (second row) using 10 steps, where the prompt is “A woman is walking on the beach by the sea”.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

###### Figure 12. Qualitative comparison between FLUX (first row) and FLUX-DCW (second row) using 10 steps, where the prompt is “A balloon gently climbs into a serene blue sky”.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

###### Figure 13. Qualitative comparison between Qwen-Image (first row) and Qwen-Image-DCW (second row) using 20 steps, where the prompt is “A woman is walking on the beach by the sea”.

[Figure 54]

[Figure 55]

[Figure 56]

###### Figure 14. Qualitative comparison between FLUX (first row) and FLUX-DCW (second row) using 20 steps, where the prompt is “There is a house and a path on a snowy mountain”.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

###### Figure 15. Qualitative comparison between FLUX (first row) and FLUX-DCW (second row) using 20 steps, where the prompt is “A balloon gently climbs into a serene blue sky”.

