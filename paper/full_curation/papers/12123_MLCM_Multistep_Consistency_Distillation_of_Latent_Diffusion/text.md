# arXiv:2406.05768v6[cs.CV]7Nov2024

## TLCM: TRAINING-EFFICIENT LATENT CONSISTENCY MODEL FOR IMAGE GENERATION WITH 2-8 STEPS

### Qingsong Xie1†, Zhenyi Liao2, Zhijie Deng2†, Chen Chen1 & Haonan Lu1 1AI Center, Guangdong OPPO Mobile Telecommunications Corp., Ltd 2Qing Yuan Research Institute, SEIEE, Shanghai Jiao Tong University Project: https://github.com/OPPO-Mente-Lab/TLCM

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Figure 1: 1024 × 1024 samples from TLCM, distilled from SDXL-base-1.0 (Podell et al., 2024) based on LoRA (Hu et al., 2022). From top to bottom, 2, 3, 4 and 8 sampling steps are adopted, respectively. Apart from satisfactory visual quality, TLCM can also yield improved metrics compared to strong baselines.

ABSTRACT

Distilling latent diffusion models (LDMs) into ones that are fast to sample from is attracting growing research interest. However, the majority of existing methods face two critical challenges: (i) They hinge on long training using a huge volume

† Corresponding authors.

of real data. (ii) They routinely lead to quality degradation for generation, especially in text-image alignment. This paper proposes a novel Training-efficient Latent Consistency Model (TLCM) to overcome these challenges. Our method first accelerates LDMs via data-free multistep latent consistency distillation (MLCD), and then data-free latent consistency distillation is proposed to efficiently guarantee the inter-segment consistency in MLCD. Furthermore, we introduce bags of techniques, e.g., distribution matching, adversarial learning, and preference learning, to enhance TLCM’s performance at few-step inference without any real data. TLCM demonstrates a high level of flexibility by enabling adjustment of sampling steps within the range of 2 to 8 while still producing competitive outputs compared to full-step approaches. Notably, TLCM enjoys the data-free merit by employing synthetic data from the teacher for distillation. With just 70 training hours on an A100 GPU, a 3-step TLCM distilled from SDXL achieves an impressive CLIP Score of 33.68 and an Aesthetic Score of 5.97 on the MSCOCO-2017 5K benchmark, surpassing various accelerated models and even outperforming the teacher model in human preference metrics. We also demonstrate the versatility of TLCMs in applications including image style transfer, controllable generation, and Chinese-to-image generation.

- 1 INTRODUCTION

Diffusion models (DMs) have made great advancements in the field of generative modeling, becoming the go-to approach for image, video, and audio generation (Ho et al., 2020; Kong et al., 2021; Saharia et al., 2022). Latent diffusion models (LDMs) further enhance DMs by operating in the latent image space, pushing the limit of high-resolution image and video synthesis (Ma et al., 2024; Peebles & Xie, 2023; Podell et al., 2024; Rombach et al., 2022). Despite the high-quality and realistic samples, LDMs suffer from frustratingly slow inference–generating a single sample requires tens to hundreds of evaluations of the model, giving rise to a high cost and bad user experience.

There is growing interest in distilling large-scale LDMs into more efficient ones. Concretely, progressive distillation (Lin et al., 2024; Meng et al., 2023; Salimans & Ho, 2023) reduces the sampling steps by half in each turn but finally hinges on a set of models for various sampling steps. InstaFlow (Liu et al., 2023), UFO-Gen (Xu et al., 2024b), DMD (Yin et al., 2024b), and ADD (Sauer et al.,

- 2023) target one-step generation, yet losing or weakening the ability to benefit from more (e.g., > 4) sampling steps. Latent consistency models (LCMs) (Luo et al., 2023) apply consistency distillation (Song et al., 2023) on LDMs’ reverse-time ordinary differential equation (ODE) trajectories to conjoin one- and multi-step generation, but the image quality degrades substantially, especially in 2-4 steps. HyperSD (Ren et al., 2024) applies consistency trajectory distillation (Kim et al., 2023) in segments of the ODE trajectory, yet suffers from a substantial performance drop in text-image alignment. Besides, all these methods rely on a huge volume of high-quality data and long training time, hindering their applicability to downstream scenarios with rare computing and data.

Before presenting our proposal, we argue that one-step generation may not always be the optimal choice in practical applications. Empirically, sampling with 2-8 steps introduces less than 2 times additional computational time compared to one step for SDXL (Podell et al., 2024) but can notably enhance the upper limit of sampling quality. Moreover, some practical applications typically have a low tolerance for quality degradation and hence can accept a moderate number of sampling steps. Thereby, this paper aims to develop a unified model with 2 to 8 sampling steps while achieving quality comparable to full-step models. We propose Training-efficient Latent Consistency Models (TLCMs) to achieve this at the expense of minimal computation and training data. Technically, we introduce data-free multistep latent consistency distillation (MLCD) to reduce the sampling steps. After MLCD, we further employ data-free latent consistency distillation (LCD) for global consistency. To enhance LCD, we enforce the consistency of TLCM at sparse predefined timesteps instead of the entire timestep range, which reduces LCD’s learning difficulty and accelerates convergence. A multistep solver is further explored to unleash the potential of the teacher in LCD. Besides, we train a latent LPIPS model to constrain the perceptual consistency of the distilled model in latent space. To optimize TLCM’s performance at few-step inference, we explore preference learning, distribution matching, and adversarial learning techniques for regularization in a data-free manner.

We have performed comprehensive empirical studies to evaluate TLCMs. We first assess the image quality on the MSCOCO-2017 5K benchmark. We observe the TLCM distilled from SDXL (Podell et al., 2024) gets an Aesthetic Score (AS) (Schuhmann) of 5.97, and a CLIP Score (CS) (Hessel et al., 2021) of 33.68 with only 3 steps, substantially surpassing 4-step LCM, 8-step SDXL-Lightning (Lin et al., 2024), and 8-step HyperSD, comparable to 25-step DDIM. Additionally, TLCM is obtained by only 70 A100 training hours without any real data, significantly reducing training costs. We also demonstrate the versatility of TLCMs in applications including image stylization, controllable generation, and Chinese-to-image generation.

We summarize our contributions as follows:

- • We propose TLCMs to accelerate LDMs to generate high-quality outputs within 2−8 steps, at the expense of minimal training compute and data.
- • We establish a data-free multistep latent consistency distillation and improved latent consistency distillation pipeline for fast LDM acceleration. Besides, bags of data-free techniques are incorporated to boost rare-step quality.
- • TLCM achieves a state-of-the-art CS (33.68) and AS (5.97) in 3 steps, surpassing competing baselines, such as 4-step LCM, 8-step SDXL-Lightning, and 8-step HyperSD.
- • TLCMs’ versatility extends to scenarios such as image stylization, controllable generation, and Chinese-to-image generation, paving the path for extensive practical applications.

- 2 RELATED WORKS

Diffusion models. (DMs) (Ho et al., 2020; Sohl-Dickstein et al., 2015; Song & Ermon, 2019; 2020; Song et al., 2021b) progressively add Gaussian noise to perturb the data, then are trained to denoise noise-corrupted data. In the inference stage, DMs sample from a Gaussian distribution and perform sequential denoising steps to reconstruct the data. As a type of generative model, they have demonstrated impressive capabilities in generating realistic and high-quality outputs in text-to-image generation (Podell et al., 2024; Rombach et al., 2022; Saharia et al., 2022), video generation (Peebles & Xie, 2023). To enhance the condition awareness in conditional DMs, the classifier-free guidance (CFG) (Ho & Salimans, 2021) technique is proposed to trade off diversity and fidelity.

Diffusion acceleration. The primary challenges that hinder the practical adoption of DMs is the slow inference involving tens to hundreds of evaluations of the model.

Early works like Progressive Distillation (PD) (Salimans & Ho, 2023) and Classifier-aware Distillation (CAD) (Meng et al., 2023) explore the approaches of progressive knowledge distillation to compress sampling steps but lead to blurry samples within four sampling steps. Consistency models (CMs) (Song et al., 2023), consistency trajectory models (CTMs) (Kim et al., 2023) and Diff-Instruct (Luo et al., 2024) distill a pre-trained DM into a single-step generator, but they do not verify the effectiveness on large-scale text-to-image generation.

Recently, the distillation of large-scale text-to-image DMs has gained significant attention. LCM (Luo et al., 2023) extends CM to text-to-image generation with few-step inference but synthesizes blurry images in four steps. InstaFlow (Liu et al., 2023), UFOGen (Xu et al., 2024b), BOOT (Gu et al., 2023), SwiftBrush (Nguyen & Tran, 2024), DMD (Yin et al., 2024a), and Diffusion2GAN (Kang et al., 2024) propose one-step sampling for text-to-image generation but are unable to extend their sampler to multiple steps for better image quality.

More recently, SDXL-Turbo (Sauer et al., 2023), SDXL-Lighting (Lin et al., 2024), and HyperSD (Ren et al., 2024) are proposed to further enhance the image quality with a few-step sampling but their training depends on huge high-quality text-image pairs and expensive online training. Our method not only enables the generation of high-quality samples using a 2-8 steps sampler but also enhances model performance with more inference cost. Furthermore, our training strategy is resource-efficient and does not require any images.

Human preference for text-to-image model. ImageReward (IR) (Xu et al., 2024a) and Aesthetic Score (Schuhmann) are proposed to evaluate the human preference for the text-to-image model. Multi-dimensional Preference Score (MPS) (Zhang et al., 2024) improves metrics by learning di-

verse preferences. To optimize TLCM towards human preference, we incorporate effective reward learning into TLCM to directly guide model tuning.

- 3 PRELIMINARY

- 3.1 DIFFUSION MODELS

Diffusion models (DMs) (Ho et al., 2020; Sohl-Dickstein et al., 2015; Song et al., 2021b) are specified by a predefined forward process that progressively distorts the clean data x0 into a pure Gaussian noise with a Gaussian transition kernel. It is shown that such a process can be described by the following stochastic differential equation (SDE) (Karras et al., 2022; Song et al., 2021b):

dxt = f(x,t)xtdt + g(t)dwt, (1)

where t ∈ [0,T], wt is the standard Brownian motion, and f(x,t) and g(t) are the drift and diffusion coefficients respectively. Let pt(xt) denote the corresponding marginal distribution of xt.

It has been proven that this forward SDE possesses the identical marginal distribution as the following probability flow (PF) ordinary differential equation (ODE) (Song et al., 2021b):

dxt = f(x,t)xt −

- 1

- 2

g2(t)∇xt log pt(xt) dt. (2)

As long as we can learn a neural model ϵθ(xt,t) for estimating the ground-truth score ∇xt log pt(xt), we can then draw samples that roughly follow the same distribution as the clean data by solving the diffusion ODE. In practice, the learning of ϵθ(xt,t) usually boils down to score matching (Song et al., 2021b).

The ODE formulation is appreciated due to its potential for accelerating sampling and has sparked a range of works on specialized solvers for diffusion ODE (Lu et al., 2022a;b; Song et al., 2021a). Let Ψ denote an ODE solver, e.g., the deterministic diffusion implicit model (DDIM) solver (Song et al., 2021a). The sampling iterates by

xt

n−1

= Ψ(ϵθ(xt

n

,tn),tn,tn−1), (3) where {tn}Nn=0 denotes a set of pre-defined discretization timesteps and tN = T,t0 = 0.

- 3.2 CONSISTENCY MODELS

Consistency model (CM) (Song & Dhariwal, 2024; Song et al., 2023) aims at generating images from Gaussian noise within one sampling step. Its core idea is to learn a model fθ(xt,t) that directly maps any point xt on the trajectory of the diffusion ODE to its endpoint. To achieve this, CMs first parameterizes fθ(xt,t) as:

#### fθ(xt,t) = cskip(t)xt + cout(t)Fθ(xt,t), (4)

where cskip(t),cout(t) is pre-defined to guarantee the boundary condition fθ(x0,0) = x0, and Fθ(xt,t) is the neural network (NN) to train.

via consistency distillation (CD) by minimizing (Song et al., 2023):

CM can be learned from a pre-trained DM ϵθ

0

,tn) , (5) where tm ∼ U[0,T], xt

#### LCD = d fθ(xt

#### ,tm),fθ−(xt

m

n

,tm),tm,tn), d(.,.) is some distance function, and θ− is the exponential moving average (EMA) of θ. Typically, xt

#### ), tn ∼ U[0,tm), xt

m ∼ pt

#### (xt

#### = Ψ(ϵθ

#### (xt

m

m

n

0

m

is obtained by a single-step solver (SS) Ψ.

n

Multistep consistency models (MCMs) (Heek et al., 2024) generalize CMs by splitting the entire range [0,T] into multiple segments and performing consistency distillation individually within each segment. Formally, MCMs first define a set of milestones {tsstep}Ms=0 (M denotes the number of segments), and minimize the following multistep consistency distillation (MCD) loss:

#### ,tm),tm,tsstep),DDIM(xt

,tn),tn,tsstep) , (6)

#### LMCD = d DDIM(xt

#### ,fθ(xt

#### ,fθ−(xt

m

m

n

n

where s is uniformly sampled from {0,...,M}, tm ∼ U[tsstep,tsstep+1], tn = tm − 1, and DDIM(xt

at timestep tm to timestep tsstep based on the estimated clean image fθ(xt

,tm),tm,tsstep) means one-step DDIM transformation from state xt

#### ,fθ(xt

m

m

m

,tm) (Song et al., 2021a).

m

ℒ𝑚𝑝𝑠

𝑔(𝑧𝑡𝑚,𝑐,𝑡𝑚,𝑡𝑠𝑡𝑒𝑝𝑠 )

𝑠𝑟𝑒𝑎𝑙

Teacher

[Figure 17]

[Figure 18]

TLCM

MPS

Gaussian noise

𝑧Ƹ𝑡′

ℒ𝑚𝑙𝑐𝑑

FD

[Figure 19]

𝑡𝑚 𝑡𝑠𝑡𝑒𝑝𝑠 -

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

TLCM

Fake model

[Figure 24]

𝑡𝑛 𝑡𝑠𝑡𝑒𝑝𝑠

ℒ𝑖𝑙𝑐𝑑2

𝑔(𝑧𝑡𝑛,𝑐,𝑡𝑛,𝑡𝑠𝑡𝑒𝑝𝑠 ) 𝑓(𝑧Ƹ𝑡𝑚, 𝑐,𝑡𝑚)

G𝑟𝑎𝑑𝑖𝑒𝑛𝑡

ℒ𝑑𝑖𝑓𝑓

[Figure 25]

𝑠𝑓𝑎𝑘𝑒

[Figure 26]

[Figure 27]

𝑡𝑛 Discriminator ℒ𝑔𝑎𝑛

𝑡𝑚

SS

MDS

𝑓(𝑧Ƹ𝑡𝑛,𝑐,𝑡𝑛)

𝑧𝑡𝑚 𝑧𝑡𝑛

Teacher Teacher

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

TLCM MS solver

FD

TLCM

Trainable model

Propagate

Stop gradient

Frozen

Gaussian 𝑧Ƹ0 𝑧Ƹ𝑡𝑚 𝑧Ƹ𝑡𝑛

Teacher

gradient

model

noise

- Figure 2: The overview for training TLCM. Data-free multistep latent consistency distillation is first used to accelerate LDM, obtaining initial TLCM (left part of the overview). Then, improved data-free latent consistency distillation is proposed to enforce the global consistency of TLCM. MPS optimization, DM, and adversarial learning are exploited to promote TLCM’s performance in a datafree manner (right part of the overview). Note that we omit the Latent LPIPS model for brevity.

- 4 METHODOLOGY

Our target is to accelerate LDM into a few-step model, with performance competitive to the fullstep teacher. The learning procedure should be executed with cheap cost in a data-free manner. In this section, we propose a novel and unified Training-efficient Latent Consistency Model (TLCM) with 2-8 step inference. We begin by introducing data-free multistep latent consistency distillation. Subsequently, we discuss data-free latent consistency distillation to enforce the global consistency of TLCM. Lastly, we explore various techniques to promote TLCM’s performance in a data-free manner. The overview of our training pipeline is presented in Figure 2.

- 4.1 DATA-FREE MULTISTEP LATENT CONSISTENCY DISTILLATION

We consider distilling representative pre-trained LDMs, e.g., SDXL (Podell et al., 2024). Previous LCM (Luo et al., 2023) has distilled SDXL into a few-step model, but it results in a big performance drop since it is hard to learn the mapping between an arbitrary state of the entire ODE trajectory to the endpoint. Drawing inspiration from MCM, we split the entire range [0, T] into M segments, and then only enforce the consistency at each separate segment. To speed up convergence, we change the skipping step (skip) =1 in MCM into 20. The EMA module is removed to save memory consumption. Let zt denote the states at timestep t in the latent space. We abuse ϵθ

(zt,c,t) and fθ(zt,c,t) to denote the pre-trained LDM and target model respectively, where c refers to the generation condition. We formulate the multistep latent consistency distillation (MLCD) loss as:

0

#### ,tn,tsstep,c)∥22, (7) where gθ(zt

#### ,tm,tsstep,c) − nograd(gθ(zt

#### Lmlcd = ∥gθ(zt

m

n

,c,tm),tm,tsstep represents initial TLCM. Given that CFG (Ho & Salimans, 2021) is critical for high-quality text-to-image generation, we integrate it to MLCD by:

#### ,tm,tsstep,c) = DDIM zt

#### ,fθ(zt

m

m

m

#### ,c,w,tm),tm,tn), (8) where ϵˆθ

#### zt

#### = Ψ(ˆϵθ

#### (zt

n

0

m

(zt,∅,t)) with w as the guidance scale.

#### (zt,∅,t) + w(ϵθ

#### (zt,c,t) − ϵθ

#### (zt,c,w,t) := ϵθ

0

0

0

0

However, this training procedure depends on huge high-quality data, which limits its applicability in scenarios where such data is inaccessible. To deal with this problem, we propose to draw samples from the teacher model as training data. Specifically, instead of obtaining zt

via adding noise to z0 as in MCM and HyperSD, we initialize zT as pure Gaussian noise ϵ and perform denoising with off-the-shelf ODE solvers based on the teacher model ϵθ

m

. Intuitively, with this strategy, we leverage and distill only the denoising ODE trajectory of the teacher without concerning

to obtain zt

0

m

can be acquired from ϵ by a single denoising step, but we empirically observe that this naive strategy is unable to accelerate LDM with desirable performance, due to poor quality of zt

the forward one. The latent state zt

m

contains less noise for smaller tm. Therefore, we design a multistep denoising strategy (MDS) to predict zt

. Theoretically, zt

m

m

, which executes more sampling iterations for smaller tm to get cleaner zt

m

. At this stage, the DDIM solver is used to estimate the ODE trajectory and generate samples from pure Gaussian noise. We present the details in Algorithm 1 in Appendix A.1.

m

- 4.2 IMPROVED DATA-FREE LATENT CONSISTENCY DISTILLATION

After a round of distillation on M segments, TLCM can naturally produce high-quality samples through M-step sampling. However, it is empirically observed that the performance decreases when using fewer steps, which is probably because of the larger discretization error caused by long sampling step sizes. To alleviate this, we advocate explicitly teaching TLCM to capture the mapping between the states that cross segments. Upon this goal, we propose data-free latent consistency distillation to promote the model to be consistent across the predefined timesteps.

We do not compile TLCM to keep consistency across the entire timestep range [0, T] since it is hard to learn the mapping that transforms any point along the trajectory into real data. Instead, we improve raw LCD by only keeping consistency at the predefined M timesteps, which makes LCD much easier to learn the mapping. Naturally, the skipping step skip is changed to T/M. The big skip offers an additional advantage that further accelerates model convergence. Benefiting from the pre-trained TLCM, we can fast yield clean data zˆ0 via a few-step (q-step) sampler, such as 4 steps, eliminating the requirement of real data. The latent state zˆt

m

is obtained by adding noise to zˆ0 in the forward diffusion process, where tm ∈ {tsstep}Ms=1. We formulate this procedure as

zˆt

m

= FD(TLCM(ϵ,T,c),tm), ϵ ∈ N(0,I), (9)

where FD and TLCM denote forward diffusion and multistep iterations by TLCM. Then, an ODE solver is used to estimate latent state zˆt

n

from zˆt

m

. Raw LCD adopts a one-step solver to predict zˆt

n

. We argue that it restricts the capability of the teacher due to discretization error, especially for big skip. As a result, we explore a multistep solver (MS) to unleash the potential of the teacher. Concretely, the time interval T/M is uniformly divided into p parts, and then p-step DDIM with CFG is used to calculate zˆt

n

. The improved data-free LCD loss in stage 2 is: Lilcd2 = ∥ fθ(ˆzt

m

,c,tm)) − nograd fθ(ˆzt

n

,c,tn)) ∥22. (10)

We present the details in Algorithm 2 in appendix A.1. Surprisingly, our improved data-free LCD only costs 2K-iteration training to achieve convergence.

- 4.3 INCORPORATING BAG OF TECHNIQUES INTO TLCM IN DATA-FREE MANNER

Latent LPIPS. Typical LCD directly adopts mean square error loss (Lmse) to enforce consistency in the latent space, but it can not capture perceptual features. LPIPS (Zhang et al., 2018) can extract the features matching human perceptual responses. Meanwhile, it has been widely used as an effective regression loss across many image translation tasks. Thereby, we aim to integrate LPIPS into our distillation pipeline to enhance TLCM’s performance. However, LPIPS is built in the pixel space, and hence we have to reconstruct latent codes to pixel space to use LPIPS. To reduce training time, we train a latent LPIPS (L-LPIPS) model, which computes perceptual features in latent space. The latent LPIPS model adopts the VGG network by changing the input to 4 channels and removing the

- 3 max-pooling layers, as the latent space in LDM is already 8× downsampled. The model is trained from scratch on BAPPS dataset (Zhang et al., 2018). Based on L-LPIPS, the outputs of the model

gθ and fθ are first fed into the L-LPIPS model, whose outputs are used to calculate consistency loss via Equation (7) or Equation (10).

MPS optimization. Since TLCMs transform the points on the trajectory to clean samples xˆ0, we can naturally directly maximize the feedback of the scorer on the sample s(ˆx0,c). Considering that multi-dimensional preference score (Zhang et al., 2024) can measure diverse human preferences, we leverage it to improve TLCM towards human preference. Formally, we optimize the following MPS loss (Lmps):

#### Lmps = max(s0 − s(ˆx0,cpos),0) + max(s(ˆx0,cneg),0), (11)

where cpos represents the text condition corresponding to the images while cpos denotes the irrelevant texts. Lmps maximizes s(ˆx0,cpos) with margin s0 and simultaneously minimizes s(ˆx0,cneg) with margin 0. The gradients are directly back-propagated from the scorer to model parameters θ for optimization. We do not use ImageReward or AS to optimize TLCM, because we find IR tends to cause overexposure and AS results in oversaturation for generated images.

Distribution matching. Distribution matching (Yin et al., 2024a) is proposed to transform LDM into a one-step model. We effectively integrate it into our distillation method to enhance the performance of TLCM. To remove the need of real data, we exploit Equation 9 to get noisy latent zˆt. Data-free DM loss in Ldfdm is applied to optimize TLCM at sparse-step inference as

#### [sreal(FD(fθ(ˆzt,t,c),t′)) − sfake(FD(fθ(ˆzt,t,c),t′))∇θfθ(ϵ)], (12)

Ldfdm = −Et,ϵ,zˆ

t

where sreal and sfake denote the pre-trained score model and fake score model, both initialized by SDXL. The model sfake is finetuned on synthetic data zˆ0 through noise prediction loss Ldiff in DM (Yin et al., 2024a).

Adversarial learning. For high-resolution text-to-image generation, considering the high data dimensionality and complex data distribution, simply using MSE loss fails to capture data discrepancy precisely, thus providing imperfect consistency constraints. We propose to use GAN loss to enforce the distribution consistency. Unlike previous methods needing real data to execute adversarial learning, we exploit Equation 9 to obtain zˆt. The student model fθ denoises zˆt by one step, obtaining z0. Through discriminator D, the GAN loss Lgan is formulated as

#### Lgan = log(D(FD(ˆz0,t′)) − log(D(FD( z0,t′))). (13)

- 5 EXPERIMENTS 5.1 IMPLEMENTATION DETAILS

We use the prompts from LAION-Aesthetics- 6+ subset of LAION-5B (Schuhmann et al., 2022) to train our model. We train the model with 12000 iterations for data-free MLCD and 2000 iterations for data-free LCD. After LCD, MPS optimization runs 500 iterations with a batch size of 8. Then, DM and adversarial learning are used to improve TLCM with 1000 iterations with a batch size of

- 4. The whole procedure uses AdamW optimizer and 4 A100. Only MLCD adopts a learning rate of 1e-4 and the other stages use a learning rate of 1e-5. The discriminator adopts a learning rate of 1e-4 and AdamW optimizer. The initial segment number M is 8 and s0 for MPS is 16. We set

the guidance scale w in CFG as 8.0, the denoising steps p = 3 for the teacher to compute zˆt

n

, and q = 4 for TLCM to compute zˆ0. As for model configuration, we use SDXL (Podell et al., 2024) as teacher to estimate trajectory while student model fθ is also initialized by SDXL. The discriminator is also initialized by SDXL. We train a unified Lora instead of UNet in all the distillation stages for convenient transfer to downstream applications.

- 5.2 MAIN RESULTS

We quantitatively compare our method with both the DDIM (Song et al., 2021a) baseline and acceleration approaches including LCM (Luo et al., 2023), SDXL-Turbo (Sauer et al., 2023), SDXLLightning (Lin et al., 2024), HyperSD (Ren et al., 2024), CS (Hessel et al., 2021) with ViTg/14 backbone, AS (Schuhmann), IR (Xu et al., 2024a), and Fr´echet Inception Distance (FID) are exploited as objective metrics. The evaluation is performed on MSCOCO-2017 5K validation dataset (Lin et al., 2014). All methods perform zero-shot validation except for HyperSD since it utilizes the MSCOCO-2017 dataset for training. Only SDXL-Turbo produces 512-pixel images while the others generate 1024-pixel images. We only report FID for reference and do not analyze it since FID on COCO is not reliable for evaluating text-to-image models (Sauer et al., 2023; Ren et al.,

- 2024).

The metrics of various methods are listed in Table 1. We use “-” to represent a metric when it is missing in the corresponding paper. We can observe that our TLCM only costs 70 A 100 training hours, even without any data. Compared to other methods, TLCM significantly reduces training resources, which is very valuable for most laboratories and scenarios when real data are inaccessible. our 3-step TLCM presents superior CS, AS, and IR than 4-8 step’s LCM (Luo et al., 2023),

- Table 1: Zero-shot performance comparison on MSCOCO-2017 5K validation datasets with the state-of-the-art methods. All models adopt SDXL architecture. Time: inference time (second) on A100. TH: Training hours using A100. TI: Training images.

Method Step Time FID CS AS IR TH TI DDIM (Song et al., 2021a) 25 3.29 23.29 33.97 5.87 0.82 0 0 LCM (Luo et al., 2023) 4 0.71 27.09 32.53 5.85 0.51 - SDXL-Turbo (Sauer et al., 2023) 4 0.38 28.52 33.35 5.64 0.83 - SDXL-Turbo (Sauer et al., 2023) 8 0.61 29.64 32.81 5.78 0.82 - SDXL-Lightning (Lin et al., 2024) 4 0.71 27.90 32.90 5.63 0.72 - >12M SDXL-Lightning (Lin et al., 2024) 8 0.99 27.04 32.74 5.95 0.71 - >12M HyperSD (Ren et al., 2024) 4 0.71 34.45 32.64 5.52 1.15 600 >12M HyperSD (Ren et al., 2024) 8 0.99 35.94 32.41 5.83 1.14 600 >12M

- TLCM 2 0.58 27.50 33.18 5.90 0.97 70 0
- TLCM 3 0.65 29.12 33.68 5.97 1.00 70 0
- TLCM 4 0.71 30.33 33.52 6.06 1.01 70 0
- TLCM 5 0.78 30.90 33.69 6.04 1.01 70 0
- TLCM 6 0.85 30.98 33.71 6.07 1.01 70 0 TLCM 8 0.99 32.40 33.53 6.08 1.02 70 0

SDXL-Lightning (Lin et al., 2024). These results indicate our TLCM’s synthetic images are much better aligned with texts and human preference than LCM, SDXL-Lightning. Excitingly, our 3-step TLCM outperforms the 25-step teacher in terms of AS and IR, and achieves comparable CS value, demonstrating that TLCM almost reserves all the information in the teacher and even introduces new human preference knowledge via the proposed distillation method. Our 3-step TLCM shows much higher CS than the 4-8 step HyperSD, indicating HyperSD loses much information in the distillation procedure because it fails to sufficiently ensure consistency constraint. We notice IR value of HyperSD is higher than our TLCM. This is because IR model has been used to optimize HyperSD. Moreover, we can see the performance of SDXL-Turbo drop with respect to CS and IR when increasing sampling steps. This is because it is designed for specific steps. Instead, our TLCM can improve at least one metric with additional steps. This is valuable since image quality is the primary consideration when affordable computation resource is determined in real applications.

We present the visual comparisons in Figure 3. Under the same conditions, we observe that the images generated by TLCM have better image quality and maintain higher semantic consistency on more challenging prompts, which also leads to greater human preference.

5.3 ABLATION STUDY

To analyze the key components of our method, we make a thorough ablation study to verify the effectiveness of the proposed TLCM. Table 2 depicts the performance of TLCM’s variants.

Data-free multistep latent consistency distillation. As shown in Table 2, only using Llcd−s which computes zt

by single step for LCD achieves CS score of 31.61, AS of 5.89, indicating our datafree method is able to accelerate LDM with good quality. Changing Llcd−s to single-step denoising MLCD Lmlcd−s, all metrics are improved. This result verifies that MLCD has a stronger capability to accelerate LDM than LCD. This is because it is hard for data-free LCD to enforce consistency across the entire timestep range while data-free MLCD alleviates this by performing LCD within predefined multiple segments.

m

Denoising strategy. We can observe from Table 2 that Lmlcd−m substantially enhances the performance of Lmlcd−s, verifying that the proposed multistep denoising strategy is critical to perform data-free MLCD. The probable reason is our multistep MDS yields better initial latent codes, whereas the latent codes have better quality with smaller timesteps.

TLCM (Ours) Step=4

DDIM Step=25

LCM Step=4

SDXL-Turbo Step=4

SDXL-Lightning Step=4

HyperSD Step=4

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

a high-resolution image or illustration of a diverse group of people facing me.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

A bat landing on a baseball bat.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

A giraffe with an owl on its head.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

A baseball player standing on the field in uniform.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

A black and white cat sitting on top of a chair.

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

A boy is eating donut holes while sitting at a dinner table.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

A boy on his phone outside near a red chair.

- Figure 3: Visual comparison between our TLCM and the state-of-the-art methods. Zoom in for more details.

- Table 2: Ablation study of TLCM with respect to latent LPIPS, data-free LCD with single denoising step (Llcd−s), data-free MLCD with single denoising iteration (Lmlcd−s), data-free MLCD with MDS (Lmlcd−m), data-free LCD in stage 2 (Llcd2), improved data-free LCD in stage 2 (Lilcd2), data-free DM (Ldfdm), multi-dimensional human preference (Lmhp), adversarial (Lgan). All the models adopt a 4-step sampler and SDXL backbone.

L-LPIPS Llcd−s Lmlcd−s Lmlcd−m Llcd2 Lilcd2 Lmhp Ldfdm Lgan CS FID AS IR

✓ 31.61 32.90 5.89 0.41 ✓ 31.76 27.01 5.98 0.58

- ✓ ✓ 31.99 27.61 5.92 0.61
- ✓ ✓ 32.31 30.99 6.01 0.69

- ✓ ✓ ✓ 32.74 32.05 6.00 0.72
- ✓ ✓ ✓ 33.06 25.44 5.96 0.77

✓ ✓ ✓ ✓ 33.16 28.40 6.01 0.90 ✓ ✓ ✓ ✓ ✓ 33.32 30.58 6.03 0.97 ✓ ✓ ✓ ✓ ✓ ✓ 33.52 30.33 6.06 1.01

- Table 3: Performance comparison of the teacher’s sampling steps for data-free LCD in stage 2.

Step CS FID AS IR Step CS FID AS IR 1 32.78 26.19 5.95 0.66 2 32.97 25.73 5.95 0.71 3 33.06 25.44 5.96 0.77 4 33.10 25.18 5.97 0.78

Latent LPIPS. As outlined in Table 2, Lmlcd−s using L-LPIPS introduces gains on all metrics. This result denotes it is more powerful to enforce consistency in latent LPIPS space than raw latent space as latent LPIPS can make perceptual consistency.

Data-free latent consistency distillation in stage 2. In 2, Llcd2 represents using multistep solver in LCD to enforce consistency across the entire timestep range. We can see that Llcd2 significantly improves CS values of TLCM trained in stage 1. This is because Llcd2 achieves inter-segment consistency of TLCM. The performance is further enhanced by substituting Llcd2 with Lilcd2. The reason lies in that it is easier to make consistency along the sparse predefined timesteps than the entire timestep range.

MHP optimization. Table 2 shows that adding Lmhp to the losses in line 7 introduces gains in terms of CS and IR. This result indicates that our MHP optimization method is capable of improving the text-image alignment and human preference of TLCM.

Data-free DM. We can see in Table 2 using our data-free DM loss Ldfdm leads to the performance improvements on all metrics. This result demonstrates that our DM in a data-free way is compatible with the proposed distillation method, boosting TLCM’s performance.

Discriminator. We also observe in Table 2 that discriminator loss Lgan improves CS, AS, and IR since the discriminator facilitates consistency in probability distribution space, which is critical for the low-step regime.

Teacher’s inference steps of data-free latent consistency distillation in stage 2. In Table 3, we study the effect concerning the teacher’s sampling steps of data-free LCD in stage 2. The results show as the sampling step increases from 1 to 4, the performance is consistently improved. Therefore, it is crucial to perform multi-step denoising to estimate zˆt

. The reason is that multi-step solvers are capable of reducing discretization error for big skipping step.

n

- 6 CONCLUSION

In this paper, we propose Training-efficient Latent Consistency Model (TLCM), a novel approach for accelerating text-to-image latent diffusion models using only 70 A100 hours, without requiring any text-image paired data. TLCM can generate high-quality, delightful images with only 2-8 sampling

steps and achieve better image quality than baseline methods while being compatible with image style transfer, controllable generation, and Chinese-to-image generation.

REFERENCES

Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling, 2023.

Jonathan Heek, Emiel Hoogeboom, and Tim Salimans. Multistep consistency models. arXiv preprint arXiv:2403.06807, 2024.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: a reference-free evaluation metric for image captioning. In EMNLP, 2021.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Chongxuan, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling diffusion models into conditional gans. arXiv preprint arXiv:2405.05967, 2024.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. Advances in Neural Information Processing Systems, 35:26565–26577,

- 2022.

Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023.

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2021.

Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.

Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022a.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022b.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

Jian Ma, Chen Chen, Qingsong Xie, and Haonan Lu. Pea-diffusion: Parameter-efficient adapter with knowledge distillation in non-english text-to-image generation. arXiv preprint arXiv:2311.17086, 2023.

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14297–14306, 2023.

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 4296– 4304, 2024.

Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7807–7816, 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024.

Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. arXiv preprint arXiv:2404.13686, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2023.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.

Christoph Schuhmann. Clip+mlp aesthetic score predictor. https://github.com/ christophschuhmann/improved-aesthetic-predictor. Accessed: 2024-05-20.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a.

Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In The Twelfth International Conference on Learning Representations, 2024.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. Advances in neural information processing systems, 33:12438–12448, 2020.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pp. 32211–32252. PMLR, 2023.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024a.

Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8196–8206, 2024b.

Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024a.

Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024b.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.

Sixian Zhang, Bohan Wang, Junqiang Wu, Yan Li, Tingting Gao, Di Zhang, and Zhongyuan Wang. Learning multi-dimensional human preference for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8018–8027, 2024.

A APPENDIX

- A.1 ALGORITHMS

- Algorithm 1: Data-free multistep latent consistency distillation

Input: Gaussian noise ϵ, timestep tm, segment index s, teacher model ϵθ

0

, student model gθ, text condition c, segment number M

Initialize zT with ϵ, calculate denoising steps L = M − s, time interval △T = (T − tm)/L for i in {0,1,··· ,L − 1} do

Calculate t = T − i ∗ △T, tm′ = t − △T Calculate zt

m′

= Ψ(ˆϵθ

0

(zt,c,w,t),t,tm′)

end Calculate zt

n

using Equation (8) Perform MLCD using Equation (7)

- Algorithm 2: Data-free latent consistency distillation in stage 2

Input: Gaussian noise ϵ, timestep tm, teacher model ϵθ

0

, student model fθ, text condition c, segment number M, denoising step of teacher p, denoising step q of student, diffusion coefficient sequence α1:T, timestep milestones {tsstep}Ms=0

Initialize zˆT with ϵ and timestep t with T for i in {0,1,··· ,q − 1} do

Calculate zˆ0 =

zˆt −

√1 − αtfθ(ˆzt,t,c) √αt

Calculate t = T − T/q × (i + 1), Calculate zˆt = √αtzˆ0 + √1 − αtϵ

end Randomly sample tm from {tsstep}Ms=1, detach zˆ0 and calculate zˆt

m

by forward diffusion zˆt

m

= √αt

m

zˆ0 + √1 − αt

m

ϵ

for i in {0,1,··· ,p − 1} do

Calculate t1 = tm − (T/M)/p × i and t2 = tm − (T/M)/p × (i + 1) Calculate zˆt

2

using Equation (8) based on current state zˆt

1

end Perform LCD using Equation (10)

- A.2 APPLICATION

- A.2.1 ACCELERATION OF IMAGE STYLE TRANSFER

Our TLCM LoRA is compatible with the pipeline of image style transfer (Mou et al., 2024). We present some examples in Figure 4 with only 2-step sampling.

- A.2.2 ACCELERATION OF CONTROLLABLE GENERATION

Our TLCM LoRA is compatible with Controlnet, enabling accelerated controllable generation. We utilize canny and depth ControlNet based on SDXL-base, together with TLCM LoRA in Figure 5. The results are sampled in 2 steps. We observe our model achieves superior image quality and demonstrates compatibility with other models, e.g. ControlNet, while also providing enhanced acceleration capabilities.

- A.2.3 ACCELERATION OF CHINESE-TO-IMAGE GENERATION

Our TLCM can accelerate the generation speed of the Chinese-to-image diffusion model (Ma et al.,

- 2023). We present some examples in Figure 6.

Van Gogh’s paintings

Ink and wash style

Pixar dreamworks

Source Japanese comics

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

- Figure 4: TLCM with image style transfer. The styles are presented at the top, and we apply image style transfer on the source image with our TLCM. Two-step sampling can produce highly stylized images with excellent results.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Source Canny edge A dog in the winter

A black dog in the autumn

A beautiful dog in the garden

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Source Depth map A black bird in the forest

A brown bird under the stars

A gray bird in the room

- Figure 5: TLCM with ControlNet. Our TLCM can be incorporated into ControlNet pipeline and produce satisfactory results with 2 steps sampling.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

落日，户外场景 咖啡店 Sunset, outdoor setting, café

海盗船被困在 宇宙漩涡星云中 A pirate ship trapped in a cosmic vortex nebula

宫崎骏风格，夏天池 塘，森林 Miyazaki-esque, summer, pond,forest

未来世界,虚幻引擎 Future world, Unreal Engine

薰衣草花海 Lavender field

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

江南水乡 小桥流水

孤舟蓑笠翁 独钓寒江雪 A lone boat, the fisherman in straw hat and raincoat fishing alone in the cold river under the snow

红烧狮子头 Red-braised Lion's Head

墙角数枝梅

北京，胡同，秋天 Beijing, hutong, autumn

凌寒独自开

Jiangnan water town

A few plum blossoms in the corner of the wall Blooming alone defying the cold

Small bridges and flowing streams

- Figure 6: TLCM for Chinese-to-image generation. With 3 steps sampling, our TLCM model can produce images that align with Chinese semantic meaning. The first line presents images in general Chinese contexts, while the second line showcases images in specific Chinese cultural settings.

