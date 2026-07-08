## Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation

Hongzhou Zhu*12 Min Zhao*12 Guande He3 Hang Su1 Chongxuan Li456 Jun Zhu12

# arXiv:2602.02214v5[cs.CV]1Jun2026

##### Abstract

To achieve real-time interactive video generation, current methods distill pretrained bidirectional video diffusion models into few-step autoregressive (AR) models, facing an architectural gap when full attention is replaced by causal attention. However, existing approaches do not bridge this gap theoretically. They initialize the AR student via ODE distillation, which requires frame-level injectivity, where each noisy frame must map to a unique clean frame under the PF-ODE of an AR teacher. Distilling an AR student from a bidirectional teacher violates this condition, preventing recovery of the teacher’s flow map and instead inducing a conditional-expectation solution, which degrades performance. To address this issue, we propose Causal Forcing, which uses an autoregressive teacher for ODE initialization to bridge the architectural gap, and then applies the same DMD procedure as in Self Forcing. Empirical results show that our method outperforms all baselines across all metrics, surpassing the SOTA Self Forcing by 19.3% in Dynamic Degree, 8.7% in VisionReward, and 16.7% in Instruction Following. Project page: https://thuml.github.io/CausalForcing.github.io/; the code: https://github.com/thu-ml/Causal-Forcing.

*Equal contribution 1Dept. of Comp. Sci. & Tech., BNRist Center, THU-Bosch ML Center, Tsinghua University. 2ShengShu. 3The University of Texas at Austin. 4Gaoling School of Artificial Intelligence Renmin University of China Beijing, China. 5Beijing Key Laboratory of Research on Large Models and Intelligent Governance. 6Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE.. Correspondence to: Jun Zhu <dcszj@tsinghua.edu.cn>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

##### 1. Introduction

Recent years have witnessed rapid progress in autoregressive (AR) video diffusion models (Jin et al., 2024; Teng et al., 2025; Chen et al., 2025; Wu et al., 2025). By adopting a frame-level autoregressive formulation with diffusion within each frame, AR video diffusion enables a wide range of real-time and interactive applications, including world modeling (Mao et al., 2025; Sun et al., 2025a; Hong et al., 2025), game simulation (Ball et al., 2025; Tang et al., 2025), embodied intelligence (Feng et al., 2025), and interactive content creation (Shin et al., 2025; Huang et al., 2025b; Ki et al., 2026; Xiao et al., 2025). Despite their promise, the computational burden of multi-step diffusion sampling severely limits their real-time capabilities.

To alleviate this latency bottleneck, recent works (Huang et al., 2025a; Yin et al., 2025) distill a powerful pretrained bidirectional video diffusion model into a few-step autoregressive student model. This is typically achieved via a twostage pipeline: an initial ODE distillation to initialize the AR student, followed by DMD (Yin et al., 2024) to further boost performance. However, compared to standard stepdistillation, such AR distillation faces a more fundamental challenge beyond the shared sampling-step gap, namely, the architectural gap. This gap arises from converting a bidirectional model, which has access to future frames, into a causal architecture that conditions solely on past context. Empirically, we find that even when distilled from the same bidirectional teacher, SOTA AR distillation methods (Huang et al., 2025a) still lag significantly behind standard DMD, which distills a bidirectional student (see Fig. 1).

In this paper, we show that the performance degradation stems from the failure of existing methods to properly address the architectural gap theoretically (see Fig. 3 and Sec. 3.2 ). Through a controlled experiment, we first show that this gap cannot be resolved by the DMD stage and should instead be addressed during the preceding ODE initialization. Crucially, a key requirement for ODE distillation is injectivity (Liu et al., 2022). In standard ODE distillation that distills a bidirectional teacher into a bidirectional student, injectivity naturally holds at the video level. In contrast, for an AR student, injectivity must hold at the frame level: each noisy frame must map to a unique clean

Standard DMD

[Figure 1]

[Figure 2]

[Figure 3]

…

Self Forcing

[Figure 4]

[Figure 5]

[Figure 6]

…

[Figure 7]

70

60

50

[Figure 8]

40

| |Standard DMD<br><br>Self Forcing|
|---|---|
| | |
| | |

Instruction Following

Dynamic Degree

- Figure 1. Limitations of existing methods. While distilling from the same bidirectional base model, SOTA autoregressive diffusion distillation methods like Self-Forcing still lag significantly behind standard DMD, which distills a bidirectional student.

frame under the PF-ODE of the AR teacher. We refer to this requirement as frame-level injectivity. However, existing methods (Huang et al., 2025a; Yin et al., 2025) distill an AR student directly from a bidirectional teacher, allowing the same noisy frame to correspond to multiple different clean frames. This violation of frame-level injectivity results in blurred and inconsistent video generation.

Building on the above analysis, we propose Causal Forcing, which bridges the architectural gap by performing ODE distillation initialization with an AR teacher (see Sec. 3.3). We first train an AR diffusion model using teacher forcing, where we show that diffusion forcing is inferior to teacher forcing for AR diffusion training both theoretically and empirically. With this AR diffusion model as the teacher, we then perform causal ODE distillation by sampling its PFODE trajectories and training the AR student accordingly. Crucially, since the teacher is autoregressive rather than bidirectional, its PF-ODE naturally satisfies frame-level injectivity, enabling the student to accurately learn the flow map. Finally, we apply the same DMD procedure as in Self Forcing to obtain a few-step AR student with high quality, enabling high-quality efficient real-time video generation.

To validate our approach, we conduct comprehensive evaluations against various baseline models (Wan et al., 2025; HaCohen et al., 2024; Deng et al., 2024; Jin et al., 2024; Chen et al., 2025; Teng et al., 2025; Yin et al., 2025; Huang et al., 2025a). Experiments show that our method consistently outperforms all baselines across all metrics, with significant gains in dynamic degree, visual quality, and instructionfollowing capability. Remarkably, under the same training budget as existing distilled autoregressive video models, it surpasses the SOTA Self-Forcing (Huang et al., 2025a) baseline by 19.3% in Dynamic Degree (Huang et al., 2024), 8.7% in VisionReward (Xu et al., 2024), and 16.7% in Instruction Following, while maintaining the same inference latency, demonstrating the effectiveness of our method.

##### 2. Background

###### 2.1. Diffusion Models

Diffusion models (Ho et al., 2020; Song et al., 2020) gradually perturb data x0 ∼ pdata(x0) through a forward diffusion process to learn its distribution. This process follows a transitional kernel qt|0(xt|x0) given by xt = αtx0 + σtϵ, ϵ ∼ N(0,I), t ∈ [0,T], where αt,σt are the predefined noise schedule. To match the data distribution, the model can be trained under a variety of parameterizations (Ho et al., 2020; Kingma & Gao, 2023; Salimans & Ho, 2022). A typical parameterization is flow matching (Lipman et al., 2022), which uses the velocity prediction. The model vθ is trained to minimize the weighted mean square error Ex

0,ϵ,t[w(t)||vθ(xt,t) − vt||2]. Under a typical noise schedule (Liu et al., 2022) that αt = 1 − t,σt = t and T = 1, vt is defined by vt := dx

dt = ϵ − x0. At this point, sampling can be done by solving the probability flow ordinary differential equation (PF-ODE) (Song et al., 2020)

t

dxt = vθ(xt,t)dt, xT ∼ N(0,I), t : T → 0. (1)

###### 2.2. Autoregressive Video Diffusion Models

Despite their success in video generation (Yang et al., 2024; Kong et al., 2024), full-sequence diffusion models generate all frames in a single shot, preventing user interaction. Autoregressive (AR) video diffusion models instead generate frames sequentially, aiming to model the distribution of N-frame videos via an autoregressive factorization1 pθ(x1:0 N) = Ni=1 pθ(xi0 | x<i0 ), where each conditional distribution pθ(xi0 | x<i0 ) is modeled by standard diffusion. This mechanism enables users to steer subsequent frames based on the generated content, thereby enabling interactivity, exemplified by Google’s Genie 3 (Ball et al., 2025).

To achieve this, two typical training strategies can be adopted, namely teacher forcing (TF) (Jin et al., 2024) and

1In practice, generation is typically performed in chunks rather than frame-by-frame. We omit this detail here for simplicity.

Standard DMD Only architectural gap

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

75

35

[Figure 13]

[Figure 14]

-5

Standard DMD

| |Only architectural|
|---|---|
| | |
| | |

gap

[Figure 15]

[Figure 16]

Dynamic Degree

Instruction Following

- Figure 2. DMD fails to bridge the architectural gap. Initializing the autoregressive student with standard DMD removes the samplingstep gap and isolates the architectural gap, yet still underperforms standard DMD. This indicates that the architectural gap cannot be resolved by the DMD stage and should instead be addressed during the preceding ODE initialization.

diffusion forcing (DF) (Chen et al., 2024; Song et al., 2025). TF aims to learn pdata(xi0 | x0<i) conditioning on a clean prefix of past frames x0<i. To enable this training paradigm in practice, a commonly used strategy (Teng et al., 2025) is concatenating the clean video with its noisy counterpart and applying a causal attention mask, so that xit can attend to x0<i. In contrast, DF targets the noisy-conditioned distribution pDF(xi0 | xt<i), with noise added independently to each frame via qt|0(xt<i | x0<i). Rather than feeding a clean prefix as in TF, DF lets xit directly attend to the noisy prefix xt<i. See more related work in Appendix A.

###### 2.3. Consistency Distillation and ODE Distillation

To enable real-time generation, multi-step diffusion models are typically distilled into few-step models. A typical approach is Consistency Distillation (CD) (Song et al., 2023; Song & Dhariwal, 2023). This is achieved by learning a flow map Gθ : (xt,t)  → x0 that maps xt to the clean endpoint x0 of the teacher diffusion model’s PF-ODE. Under the boundary condition Gθ(x,0) ≡ x, the model Gθ can be trained by minimizing Ex

0,ϵ,t[w(t)d(Gθ(xt,t),Gθ−(xˆt−∆t,t − ∆t))], where xt

is obtained via the forward diffusion process, xˆt−∆t is obtained by solving one ODE step from xt using the teacher diffusion model, θ− is a running average of θ with stopgradient, and d(·,·) is the distance under a chosen norm. Recent works (Lu & Song, 2024; Geng et al., 2025; Zheng et al., 2025) have further improved the method.

Recent works on real-time interactive video generation (Yin et al., 2025; Huang et al., 2025a) adopt a simplified variant of CD, which trains the student Gθ with direct regression: θ∗ = minθ Et,x

[||Gθ(xt,t) − x0||2], where xt and x0 lie on the same PF-ODE trajectory of the teacher model. We refer to this method as ODE distillation in the sequel.

t

###### 2.4. Score Distillation

Score distillation (Wang et al., 2023; Luo et al., 2023b) distills a multi-step diffusion model into a few-step student model by matching the student’s generative distribution pθ(x˜) to that of the data. A commonly used instantiation is Distribution Matching Distillation (DMD) (Yin et al., 2024),

which minimizes the KL divergence between the student and data distributions by descending along its gradient

∇θEt[DKL(pθ,t||pdata,t)]

∂x˜ ∂θ

= −Ex˜,t,x˜

[(sreal(x˜t,t) − sfake(x˜t,t))

t

], (2)

where x˜ ∼ pθ(x˜) is generated by the student, and x˜t ∼ qt|0(x˜t|x˜) is a noised version of x˜ that induces the distribution pθ,t(x˜t). Herein, a frozen diffusion model sreal is used to predict the score of x˜t under the noisy data distribution pdata,t(x˜t), while an online-trainable diffusion model sfake predicts the score under pθ,t(x˜t).

##### 3. Method

###### 3.1. Limitations of Existing Methods

As described in Sec. 2.2, real-time interactive video generation requires a few-step autoregressive generator. The most widely used strategy, exemplified by CausVid (Yin et al., 2025) and Self Forcing (Huang et al., 2025a), adopts asymmetric distillation: given a pretrained bidirectional video diffusion model, one distills a few-step autoregressive student generator. Compared to standard step-distillation, beyond the shared sampling-step gap, i.e., reducing multistep sampling to few-step sampling, a more fundamental challenge lies in the architectural gap: converting a bidirectional model with full attention (Peebles & Xie, 2023) into a causal attention architecture that conditions solely on past context, with no access to future frames.

Although the current state-of-the-art (SOTA) in autoregressive video diffusion distillation, Self Forcing (Huang et al., 2025a), achieves strong performance, it still falls short of standard DMD, which distills a few-step bidirectional student from a bidirectional video diffusion model. As shown in Fig. 1, Self Forcing is substantially worse than standard DMD (Yin et al., 2024) in terms of vision quality, dynamic degree, and instruction following. This gap suggests that existing autoregressive diffusion distillation pipelines remain suboptimal, motivating a further investigation of the underlying causes and more effective strategies.

Video-level injectivity Frame-level non-injectivity (Self Forcing)

Frame-level injectivity (Ours)

[Figure 17]

[Figure 18]

[Figure 19]

=

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

… … …

(a) Bidirectional teacher, bidirectional student (b) AR teacher, AR student (c) Bidirectional teacher, AR student

###### Figure 3. Necessary principle for ODE initialization and why Self Forcing is flawed. ODE distillation requires injective paired data.

- (a) Standard ODE distillation, which distills a bidirectional teacher to a bidirectional student, satisfies this requirement at the video level.
- (b) For an AR student, injectivity must hold at the frame level: each noisy frame maps to a unique clean frame via the PF-ODE of the AR teacher. (c) In contrast, Self Forcing distills an AR student from a bidirectional teacher, where the same noisy frame corresponds to multiple distinct clean frames, violating frame-level injectivity and results in blurred videos after ODE distillation. See Sec. 3.2 for details.

###### 3.2. Analysis: Suboptimality of Existing Methods

In this section, we analyze the reason for the performance degradation observed in existing methods and identify the key principle required to address it.

Recap of the Self Forcing Pipeline. The current SOTA Self Forcing (Huang et al., 2025a) adopts a two-stage distillation strategy. Given a bidirectional diffusion model, it first applies an ODE distillation to bridge the architectural gap and enable few-step sampling. Specifically, the bidirectional model samples along its PF-ODE trajectory, and the target autoregressive model Gθ learns to map noisy intermediates to the clean video. The training objective is

θ∗ = min

t , i[||Gθ(xit,x<it ,t) − xi0||2], (3)

Et,x1:N

θ

where i ∼ U(1,N), (x1:t N,x1:0 N) lie on the same PF-ODE trajectory of the bidirectional teacher model, and t ∈ S denotes a predefined set of timesteps used for sampling. Building on this ODE distillation stage, it then further applies asymmetric DMD, using the resulting model to initialize the autoregressive student and the bidirectional base model

- as the teacher. In what follows, we examine how each stage contributes to closing the architectural gap and whether it is theoretically well aligned with this objective.

DMD stage in Self Forcing does not address the architectural gap. We first examine whether the DMD stage can bridge the architectural gap. We initialize the autoregressive student with a few-step bidirectional model distilled via standard DMD, which removes the sampling-step gap while retaining only the architectural gap. As shown in

- Fig. 2, despite eliminating the sampling-step gap, performance remains significantly worse than the standard DMD. This indicates that a large architectural gap at initialization

cannot be resolved by the subsequent DMD stage. Consequently, in the two-stage design of Self Forcing, it is the ODE distillation stage that is expected to bridge the architectural gap. We next analyze its theoretical soundness.

Frame-level injectivity as a necessary principle for ODE initialization. We begin by identifying the necessary condition that ODE distillation must satisfy. For regressive MSE-loss-based ODE distillation to be well-defined, the paired data must be injective (Liu et al., 2022), meaning that at any timestep, each noisy sample corresponds to a unique clean sample in the sample space. In the setting where a bidirectional student is distilled from a bidirectional teacher, this injectivity naturally holds by nature at the video level due to the injectivity of diffusion PF-ODE. Formally, for any noisy video x1:t N, there exists a unique clean video x1:0 N in the sample space, such that x1:0 N = ϕ

(x1:t N,t), where ϕ

Bi

: (x1:t N,t)  → x1:0 N denotes the PF-ODE flow map of the bidirectional model (see Fig. 3a), and is exactly what the student model learns to fit. However, in autoregressive video models, frames are generated sequentially. This shifts the injectivity requirement from the entire video x1:0 N to each individual frame xi0, i ∈ {1,...,N}. Specifically, for any noisy frame xit, there must exist a unique corresponding clean frame xi0 in the sample space such that xi0 = ϕ

Bi

: (xit,t)  → x0 denotes PFODE flow map of the autoregressive diffusion model that the student model learns (see Fig. 3b). We formalize this requirement below:

(xit,t), where ϕ

AR

AR

Definition 3.1 (Frame-level injectivity). For the mapping ϕ

: (xit,t)  → xi0, frame-level injectivity holds if ∀t ∈ (0,1], for any two noisy videos {xjt}Nj=1,{ytj}Nj=1:

AR

∀i ∈ [N],xit = yti ⇒ ϕAR(xit,t) = ϕAR(yti,t), (4)

i.e., ϕ

(xit,t) is a well-defined function that maps xit to the i-th clean frame (we omit the conditional history frames for brevity).

AR

If the condition in Eq. (4) is violated, the regressive student trained via Eq. (3) cannot recover the teacher’s flow map, but instead collapses to the conditional expectation (Bishop & Nasrabadi, 2006):

G∗θ(xit,x<it ,t) = E[x0|xit,x<it ,t]. (5)

Intuitively, learning a conditional mean averages over frames, which manifests as blurred visual results.

Current ODE initialization in Self Forcing violates frame-level injectivity. Current ODE distillation in Self Forcing employs a bidirectional teacher to distill an autoregressive student. We show that this design violates the frame-level injectivity condition in Eq. (4), rendering the distillation fundamentally flawed.

As discussed above, the PF-ODE trajectory induced by a bidirectional model is injective only at the video level, but not at the frame level. We theoretically demonstrate that this leads to a non-negligible probability that the same noisy frame xit corresponds to multiple distinct clean frames (see Fig. 3c and Lemma 3.2). Formally, for a fixed timestep t, there exist xit = yti yet xi0 ̸= y0i in the paired data sample space, and such collisions occur on a set of non-zero measure. Consequently, the frame-level injectivity condition in Eq. (4) is violated with non-negligible probability. This shows that Self Forcing’s ODE distillation, which trains an autoregressive student from a bidirectional teacher, is theoretically misaligned. We formalize this issue in the following lemma:

Lemma 3.2 (Frame-level non-injectivity of PF-ODE, informal). Let x1:t N satisfy the PF-ODE in Eq. (1) of a bidirectional diffusion model. Denote xit as its i-th frame, and let xothert := x[tN]\{i} denote the remaining frames. Define the flow map ϕ

(x1:t N,t)i is not a.e. constant with respect to xothert , then

: (x1:t N,t)  → x1:0 N. If ϕ

Bi

Bi

∀t ∈ (0,1], ∀x1:t N ∈ Rd, ∃yt1:N ∈ Rd,such that

yti = xit, andϕBi(x1:t N,t)i ̸= ϕBi(yt1:N,t)i. (6) Moreover, P Var ϕ

(x1:t N,t)i | xit,t > 0 > 0. For a rigorous formalization and proof, see Appendix B.1. This implies that ϕ

Bi

(·,·)i is not a well-defined function. A key intuition for this issue is that a bidirectional diffusion model denoises the i-th frame using all frames. Thus, even with xit fixed, different x>it can yield different xi0. In Self Forcing, the autoregressive student is supervised without x>it , causing information loss and thus violating Eq. (4).

Bi

Similar to the issue identified in Eq. (5), this frame-level non-injectivity prevents the autoregressive student model from recovering the teacher’s true flow map:

Proposition 3.3 (Distribution mismatch in current Self Forcing ODE distillation, proof in Appendix B.1). Using the notation of Lemma 3.2, consider training a causal frame-wise model Gθ : (xit,t)  → xi0 with the MSE regression target θ∗ = minθ Ex1:N

t ,t Gθ(xit,t) − xi0 2 (we omit the con-

ditional x<it for brevity), where (x1:t N,x1:0 N) is paired data from PF-ODE of a bidirectional diffusion model. Then the optimal solution does not follow the data distribution, i.e.,

G∗θ(xit,t) = E[xi0 | xit,t] ≁ pdata(xi0). (7)

As shown in Fig. 3c, this leads to blurry videos and is markedly inferior to standard ODE distillation with a bidirectional student in Fig. 3a.

[Figure 29]

[Figure 30]

DF …

[Figure 31]

[Figure 32]

…

TF

Figure 4. TF vs. DF in AR diffusion training. Contrary to common belief, DF leads to video collapse due to the traininginference gap, whereas TF produces higher visual quality.

###### 3.3. Causal Forcing

Building on the above analysis, bridging the architectural gap requires ODE distillation to satisfy the frame-level injectivity condition in Eq. (4), which in turn requires an autoregressive diffusion model as the teacher. We therefore propose Causal Forcing, a three-stage method that sequentially consists of teacher forcing autoregressive diffusion training, causal ODE distillation, and asymmetric DMD.

Stage 1: autoregressive diffusion training. We begin by revisiting two standard training paradigms for AR diffusion models, namely teacher forcing (TF) and diffusion forcing (DF) (see Sec. 2.2), to obtain the AR diffusion model that serves as the teacher for subsequent ODE distillation. Somewhat surprisingly, and contrary to common belief, we find that teacher forcing is more suitable than diffusion forcing for training AR diffusion models, both theoretically and empirically. Specifically, when training the i-th frame xit, diffusion forcing is conditioned on heavily noised preceding frames x<it , whereas inference is conditioned on

clean preceding frames x<i0 . This discrepancy introduces a substantial training–inference distribution mismatch. We formalize this issue in the following proposition.

Proposition 3.4 (Distribution mismatch in autoregressive diffusion forcing). Under the notation of Sec. 2.2 and regularity conditions in Appendix B.2:

data(x<i0 ) DKL pDF(xi0 | y)||pdata(xi0 | y) > 0.

Ey∼p

That is, the model trained with autoregressive diffusion forcing does not follow the data distribution conditioned on the causal prefix y. See Appendix B.2 for the proof.

In contrast, teacher forcing conditions on clean preceding frames x<i0 during training, thereby aligning the training objective with the inference process and eliminating this gap. The empirical comparisons in Fig. 4 further corroborate this theoretical analysis. For further discussion of diffusion forcing and recent alternatives (Wu et al., 2025; Po et al., 2025; Guo et al., 2025), see Appendix C.1. Accordingly, we adopt an autoregressive diffusion model trained via teacher forcing. Notably, this multi-step AR diffusion model already provides a substantially improved initialization for asymmetric DMD, but still exhibits abrupt artifacts. Please refer to Appendix C.2 for both qualitative and quantitative results.

- Stage 2: causal ODE distillation. With the above AR diffusion model as the teacher, we next perform causal ODE distillation. Firstly, we need to sample and store the PF-

ODE trajectory {xit}t∈S∪{0} from the AR diffusion teacher at the required timesteps S. To achieve this, we sample the

clean frames x<igt from the real dataset and use the teacher to generate the current frame with ODE solver conditioned on these frames as history, starting from Gaussian noise xiT ∼ N(0,I). Then, the student Gθ is trained to regress the clean target xi0 from the intermediate noisy state xit, conditioned on the same history clean frames x<igt :

θ∗ = min

θ

Ex<i

gt , t∈S,i,xit[||Gθ(xit,x<igt ,t) − xi0||2], (8)

Notably, since the teacher here is autoregressive rather than bidirectional, its PF-ODE ensures the frame-level injectivity condition in Eq. (4) by nature. Therefore, our method avoids the collapse identified in Proposition 3.3 and enables the student to learn the flow map accurately. As shown in Fig. 3b, sampling such an AR teacher for ODE distillation indeed yields strong performance.

- Stage 3: asymmetric DMD. Building upon the above causal ODE initialization for the AR student, we perform asymmetric DMD following Self Forcing. Exactly as in Self Forcing’s DMD, this process lets the student model condition on its own generated prefix and perform self-rollout,

thereby reducing the training–inference gap. As shown in Fig. 5, DMD with our causal ODE initialization yields a final model that substantially outperforms Self Forcing, indicating that the architectural gap is effectively resolved.

###### 3.4. Extension to Causal Consistency Models

Beyond the score-distillation paradigm mentioned above, since ODE distillation can be viewed as a simplified form of consistency distillation (CD), our perspective also naturally extends to CD. In this section, we present the first causal CD framework and further show that it outperforms the asymmetric CD that uses a bidirectional model as a teacher.

Specifically, we use the aforementioned native autoregressive diffusion model as the teacher, and train the causal consistency model Gθ via teacher forcing as follows:

θ∗ = min

θ

gt,ϵ,t,i[w(t)d(Gθ(xit,x<igt ,t),

Ex

Gθ−(xˆit−∆t,x<igt ,t − ∆t))], (9)

where xˆit−∆t is obtained by solving ODE from xit using the autoregressive teacher model conditioned on clean prefix

x<igt , and other notations follow Sec. 2.3. Consistent with Sec. 3.3, this approach leverages the frame-level injectivity of the native AR teacher, enabling the student to learn the correct flow map. In contrast, asymmetric CD teacher violates this injectivity as stated in Lemma 3.2, leading to collapse. As shown in Fig. 10 in Appendix D, our causal CD substantially outperforms asymmetric CD.

Such causal CD may also serve as a substitute for causal ODE distillation, providing a strong initialization for the DMD stage. We leave this to future work.

[Figure 33]

[Figure 34]

SF …

[Figure 35]

[Figure 36]

…

###### Ours

Figure 5. Performance comparison between Self Forcing (SF) and ours. DMD with Self Forcing’s ODE initialization shows weaker dynamics and artifacts, whereas with causal ODE initialization, it achieves stronger dynamics with higher visual fidelity.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Wan2.1

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

CausVid

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Self Forcing

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

###### Ours

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Wan2.1

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

CausVid

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Self Forcing

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

###### Ours

Figure 6. Qualitative comparisons with existing methods. Our method achieves substantially higher dynamics and better visual quality than existing distilled autoregressive video models (Causvid and Self Forcing), while matching or even surpassing bidirectional diffusion models (Wan2.1). More video demos and all the prompts used in this paper are provided in the supplementary materials.

##### 4. Experiments

- et al., 2023a) setting. See more details in Appendix D.

Evaluation. We adopt VBench (Huang et al., 2024) as our primary evaluation benchmark following Self Forcing. For overall visual assessment, we employ VisionReward (Xu

- et al., 2024), which correlates well with human judgments, and we additionally report VisionReward’s Instruction Following sub-score to measure instruction adherence. Notably, VisionReward scores can be negative, but higher values are always better. Since many VBench prompts involve minimal motion, we further curate a 100-prompt set with rich motion and complex actions, provided in the supplementary materials. We evaluate VisionReward, Instruction Following, and Dynamic Degree on this 100-prompt set. For readability, all these metrics are scaled by 100. Additionally, we conduct a user study with 10 participants on 10 prompts, where users rank the overall video quality across all methods. Finally, to assess real-time capability, we report throughput and latency on a single H100 GPU by FPS and seconds, respectively. See more details in Appendix D.

- 4.1. Setup Implementation details. Following Self Forcing (Huang

- et al., 2025a), we adopt Wan2.1-T2V-1.3B (Wan et al., 2025) as our base model to fine-tune from, which generates 81frame videos at a resolution of 832 × 480. We first train an autoregressive diffusion model with teacher forcing for 2K steps on a 3K dataset DBi synthesized by the base bidirectional model. When constructing DBi, we also store noisy intermediates for the baseline ODE distillation for ablation. We then use the autoregressive diffusion model

as the teacher to sample 3K causal ODE-trajectories DCausal and perform causal ODE distillation for 1K steps. The resulting model initializes asymmetric DMD, trained on VidProM (Wang & Yang, 2024) for 750 steps until convergence under the same protocol as Self Forcing. Following Self Forcing, we implement all methods in a chunk-wise manner, where each chunk contains 3 latent frames. For causal CD in the following ablation section, we adopt the LCM (Luo

- Table 1. Quantitative comparisons with existing methods. Our method consistently outperforms all baselines across all metrics. Dynamic., Vision., Instruct., and Rating denote Dynamic Degree, VisionReward, Instruction Following, and user rating, respectively.

Model Throughput ↑ Latency ↓ Total↑ Quality↑ Semantic↑ Dynamic.↑ Vision.↑ Instruct.↑ Rating↓ Bidirectional Video Diffusion Models

LTX-1.9B (HaCohen et al., 2024) 8.98 13.5 79.83 81.88 71.62 46 – 6.218 – 38 6.40 Wan2.1-1.3B (Wan et al., 2025) 0.78 103 83.37 84.30 79.65 61 5.275 42 2.29

Autoregressive Video Diffusion Models

NOVA (Deng et al., 2024) 0.88 4.1 80.31 80.66 78.92 46 – 7.381 – 16 8.41 Pyramid Flow (Jin et al., 2024) 6.70 2.5 80.75 83.41 70.11 16 4.055 – 2 6.11 SkyReels-V2-1.3B (Chen et al., 2025) 0.49 112 81.97 83.96 74.01 37 3.584 32 6.57 MAGI-1-4.5B (Teng et al., 2025) 0.19 282 78.88 81.67 67.72 42 0.773 8 6.44

Distilled Autoregressive Video Models

CausVid (Yin et al., 2025) 17.0 0.69 81.33 83.98 70.72 62 5.741 12 4.27 Self Forcing (Huang et al., 2025a) 17.0 0.69 83.74 84.48 80.77 57 5.820 48 2.87 Causal Forcing (Ours) 17.0 0.69 84.04 84.59 81.84 68 6.326 56 1.64

###### 4.2. Results

Performance comparison with existing models. We compare our model against baselines of comparable scale, including bidirectional video diffusion models Wan2.11.3B (Wan et al., 2025), LTX-1.9B (HaCohen et al., 2024), autoregressive video diffusion models NOVA (Deng et al.,

- 2024), Pyramid Flow (Jin et al., 2024), SkyReels-V21.3B (Chen et al., 2025), MAGI-1-4.5B (Teng et al., 2025), and distilled autoregressive video models Causvid (Yin et al.,
- 2025) and Self Forcing (Huang et al., 2025a).

As shown in Tab. 1, our method consistently outperforms all baselines across all metrics, achieving the best dynamic degree, visual quality, and instruction following ability. Compared to bidirectional diffusion models with a similar parameter scale, our method matches the performance of the SOTA Wan2.1 and even surpasses it, while delivering a 2079% higher throughput and substantially faster inference. In comparison with existing autoregressive diffusion models, our method improves over the best baseline by 47.8% in dynamic degree, 56.0% in VisionReward, and 75.0% in instruction following. Compared to distilled autoregressive video models, we maintain the same exceptionally high throughput and outperform the current SOTA Self Forcing by 19.3% in dynamic degree, 8.7% in VisionReward, and 16.7% in instruction following. Qualitative results in Fig. 6 align with the quantitative findings, showing that our method significantly surpasses the SOTA distilled autoregressive models. Notably, the baseline distilled autoregressive models both perform at least 3K steps of ODE initialization before DMD, the same as our method. Thus, our method uses exactly the same training budget, yet delivers substantial improvements.

Ablation studies. We compare different strategies under autoregressive diffusion training, score distillation, and consistency distillation (CD). For autoregressive diffusion training, Self Forcing’s ODE initialization, and CD, we use

the DBi dataset; for our causal ODE initialization, we use DCausal. Since both datasets are internal synthetic data using the same prompts, we ensure that the data quality is identical, thus guaranteeing the fairness of the comparison. We report all results under the chunk-wise setting. In addition, for ODE initialization and DMD, we also report results under the frame-wise setting. See more details in Appendix D.

Tab. 2 shows that during autoregressive diffusion training, teacher forcing outperforms diffusion forcing across all metrics, with VisionReward improving by 111.2%, consistent with Fig. 4. Diffusion forcing attains a higher dynamic degree, but it largely stems from the collapse that pathologically inflates the motion metric. For ODE initialized DMD, our causal ODE initialization substantially outperforms Self Forcing’s ODE initialization. Under the chunk-wise setting, DMD with our causal ODE initialization improves VisionReward by 90.0%, dynamic degree by 183.3%, and instruction following by 47.4%. This improvement is even more pronounced under the frame-wise setting, with a 3100% improvement in dynamic degree and a 218.0% increase in VisionReward. This is consistent with the qualitative results in Fig. 5, demonstrating that causal ODE distillation provides the correct initialization for DMD. We also compare our causal CD with the asymmetric CD, where causal CD improves VisionReward by 9.781 and instruction following by 60. Qualitative visualizations are provided in Appendix D Fig. 10. Notably, our current CD is still a rudimentary instantiation that directly adopts vanilla LCM (Luo et al., 2023a), and therefore underperforms score distillation. Nevertheless, our formulation paves the way for future work (Lu & Song, 2024; Zheng et al., 2025).

##### 5. Further Discussion

In this section, we further discuss the extended applications of Causal Forcing and its distinctions from prior work.

- Table 2. Ablation study. Tot., Qua., Sem., Dy., Vis., and Inst. denote Total, Quality, Semantic VBench score, Dynamic Degree, VisionReward, and Instruction Following, respectively.

Method Tot.↑ Qua.↑Sem.↑Dy.↑ Vis.↑ Inst.↑ Autoregressive Diffusion Training

Diffusion Forcing 81.76 82.52 78.71 60 1.583 30 Teacher Forcing 82.12 82.73 79.67 50 3.343 32

Score Distillation (Chunk-wise)

Self Forcing’s ODE + DMD82.00 82.18 81.29 24 3.330 38 Causal ODE + DMD 84.04 84.59 81.84 68 6.326 56

Score Distillation (Frame-wise)

Self Forcing’s ODE + DMD81.83 82.66 78.50 2 1.951 – 4 Causal ODE + DMD 83.75 84.35 81.37 64 6.204 42

Consistency Distillation Asymmetric CD 79.07 79.99 75.37 59 – 7.983 – 42 Causal CD 81.48 82.13 78.88 51 1.798 18

Long video generation. A common view is that AR modeling naturally fits long video generation due to its streaming capability. However, this is non-trivial: like Self Forcing (Huang et al., 2025a), Causal Forcing is trained on 5s videos with at most 5s attention, so directly extrapolating it longer causes a training-inference gap, and leads to degradation. This issue can be addressed by long-video adaptation methods orthogonal to Causal Forcing, including trainingbased LongLive (Yang et al., 2025a) and Rolling Forcing (Liu et al., 2025), and training-free Infinity-RoPE (Yesiltepe et al., 2026) and Deep Forcing (Yi et al., 2025).

Distinction from other AR distillation paradigms. Beyond DMD-based AR distillation methods such as CausVid and Self Forcing, another line of work adopts GAN-based AR distillation (Goodfellow et al., 2014), e.g., APT2 (Lin

- et al., 2026). Unlike CausVid and Self Forcing, APT2 uses teacher-forcing CD for initialization, whose theoretical objective is aligned with the causal ODE initialization in Causal Forcing. Here we emphasize the fundamental distinctions and independent contributions of Causal Forcing.

First, Causal Forcing is the first work to theoretically show that forward-KL-based distillation, including ODE/CD, should use an AR teacher to train an AR student, whereas APT2 provides no theoretical analysis. Second, Causal Forcing significantly differs from APT2 in both algorithm and architecture: it follows the asymmetric DMD paradigm of CausVid and Self Forcing, using a bidirectional diffusion model as the DMD teacher, and supports temporal KV cache inference. In contrast, APT2 is GAN-based, has no bidirectional-teacher formulation, and uses no temporal KV cache, instead concatenating noise with conditions. Finally, built upon these distinct post-training algorithms and model architectures, Causal Forcing open-sources the first few-step AR model initialized by causal ODE. Since APT2 is not open-source, we do not compare with it in this paper.

##### 6. Conclusion

In this paper, we identify the limitations of existing methods for autoregressive video diffusion distillation and clarify that bridging the architectural gap is essential. Focusing on the ODE initialization, we show that a fundamental requirement is frame-level injectivity that existing methods violate. Building on this theoretical analysis, we propose Causal Forcing: we first train an autoregressive diffusion model via teacher forcing, and then use it as the teacher for ODE distillation to initialize the subsequent DMD stage. Experiments show that our method consistently outperforms all baselines across all metrics, demonstrating its effectiveness.

##### Acknowledgements

This work was supported by NSFC Projects (Nos. 62595773, 62522609, 62550004, 92470118, 92370124, U25B6003, 62350080), the Beijing Natural Science Foundation (No. L247030); partially supported by the Shandong Provincial Natural Science Foundation (No. ZR2022ZD01) and Beijing Natural Science Foundation L247011; supported by Tsinghua Institute for Guo Qiang, and the High Performance Computing Center, Tsinghua University. J. Zhu was also supported by the XPlorer Prize.

##### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

##### References

Ball, P. J., Bauer, J., Belletti, F., Brownfield, B., Ephrat, A., Fruchter, S., Gupta, A., Holsheimer, K., Holynski, A., Hron, J., Kaplanis, C., Limont, M., McGill, M., Oliveira, Y., Parker-Holder, J., Perbet, F., Scully, G., Shar, J., Spencer, S., Tov, O., Villegas, R., Wang, E., Yung, J., Baetu, C., Berbel, J., Bridson, D., Bruce, J., Buttimore, G., Chakera, S., Chandra, B., Collins, P., Cullum, A., Damoc, B., Dasagi, V., Gazeau, M., Gbadamosi, C., Han, W., Hirst, E., Kachra, A., Kerley, L., Kjems, K., Knoepfel, E., Koriakin, V., Lo, J., Lu, C., Mehring, Z., Moufarek, A., Nandwani, H., Oliveira, V., Pardo, F., Park, J., Pierson, A., Poole, B., Ran, H., Salimans, T., Sanchez, M., Saprykin, I., Shen, A., Sidhwani, S., Smith, D., Stanton, J., Tomlinson, H., Vijaykumar, D., Wang, L., Wingfield, P., Wong, N., Xu, K., Yew, C., Young, N., Zubov, V., Eck, D., Erhan, D., Kavukcuoglu, K., Hassabis, D., Gharamani, Z., Hadsell, R., van den Oord, A., Mosseri, I., Bolton, A., Singh, S., and Rockt¨aschel, T. Genie 3: A new frontier for world models. 2025.

Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., and Zhu, J. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22669–22679, 2023.

Bao, F., Xiang, C., Yue, G., He, G., Zhu, H., Zheng, K., Zhao, M., Liu, S., Wang, Y., and Zhu, J. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233,

- 2024.

Bishop, C. M. and Nasrabadi, N. M. Pattern recognition and machine learning, volume 4. Springer, 2006.

Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023a.

Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S. W., Fidler, S., and Kreis, K. Align your latents: Highresolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22563–22575, 2023b.

Chen, B., Mart´ı Mons´o, D., Du, Y., Simchowitz, M., Tedrake, R., and Sitzmann, V. Diffusion forcing: Nexttoken prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081– 24125, 2024.

- Chen, G., Lin, D., Yang, J., Lin, C., Zhu, J., Fan, M., Zhang, H., Chen, S., Chen, Z., Ma, C., et al. Skyreelsv2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.
- Chen, H., Xia, M., He, Y., Zhang, Y., Cun, X., Yang, S., Xing, J., Liu, Y., Chen, Q., Wang, X., et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.

Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai,

- A., Ban, Y., and Hsieh, C.-J. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.

Deng, H., Pan, T., Diao, H., Luo, Z., Cui, Y., Lu, H., Shan, S., Qi, Y., and Wang, X. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.

Evans, L. Measure theory and fine properties of functions. Routledge, 2018.

Feng, Y., Xiang, C., Mao, X., Tan, H., Zhang, Z., Huang, S., Zheng, K., Liu, H., Su, H., and Zhu, J. Vidarc: Embodied video diffusion model for closed-loop control. arXiv preprint arXiv:2512.17661, 2025.

Geng, Z., Pokle, A., Luo, W., Lin, J., and Kolter, J. Z. Consistency models made easy. arXiv preprint arXiv:2406.14548, 2024.

Geng, Z., Deng, M., Bai, X., Kolter, J. Z., and He, K. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Guo, Y., Yang, C., He, H., Zhao, Y., Wei, M., Yang, Z.,

- Huang, W., and Lin, D. End-to-end training for autoregressive video diffusion via self-resampling. arXiv preprint arXiv:2512.15702, 2025.

Gupta, A., Yu, L., Sohn, K., Gu, X., Hahn, M., Li, F.F., Essa, I., Jiang, L., and Lezama, J. Photorealistic video generation with diffusion models. In European Conference on Computer Vision, pp. 393–411. Springer, 2024.

HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

He, Y., Yang, T., Zhang, Y., Shan, Y., and Chen, Q. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221, 2022.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J., et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

Hong, W., Ding, M., Zheng, W., Liu, X., and Tang, J. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

Hong, Y., Mei, Y., Ge, C., Xu, Y., Zhou, Y., Bi, S., HoldGeoffroy, Y., Roberts, M., Fisher, M., Shechtman, E., et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.

- Huang, X., Li, Z., He, G., Zhou, M., and Shechtman, E. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025a.

- Huang, Y., Guo, H., Wu, F., Zhang, S., Huang, S., Gan, Q., Liu, L., Zhao, S., Chen, E., Liu, J., et al. Live avatar: Streaming real-time audio-driven avatar generation with infinite length. arXiv preprint arXiv:2512.04677, 2025b.
- Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., and Lin, Z. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.

Kallenberg, O. Foundations of modern probability. Springer, 1997.

Ki, T., Jang, S., Jo, J., Yoon, J., and Hwang, S. J. Avatar forcing: Real-time interactive head avatar generation for natural conversation. arXiv preprint arXiv:2601.00664, 2026.

Kingma, D. and Gao, R. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36:65484– 65516, 2023.

Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Schindler, G., Hornung, R., Birodkar, V., Yan, J., Chiu, M.-C., et al. Videopoet: A large language model for zeroshot video generation. arXiv preprint arXiv:2312.14125, 2023.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Li, X., Li, M., Cai, T., Xi, H., Yang, S., Lin, Y., Zhang, L., Yang, S., Hu, J., Peng, K., et al. Radial attention: o(nlog n) sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025.

Lin, B., Ge, Y., Cheng, X., Li, Z., Zhu, B., Wang, S., He, X., Ye, Y., Yuan, S., Chen, L., et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.

Lin, S., Yang, C., He, H., Jiang, J., Ren, Y., Xia, X., Zhao, Y., Xiao, X., and Jiang, L. Autoregressive adversarial post-training for real-time interactive video generation. Advances in Neural Information Processing Systems, 38: 41061–41086, 2026.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, K., Hu, W., Xu, J., Shan, Y., and Lu, S. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Lu, C. and Song, Y. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.

Luo, S., Tan, Y., Huang, L., Li, J., and Zhao, H. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023a.

Luo, W., Hu, T., Zhang, S., Sun, J., Li, Z., and Zhang, Z. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36:76525– 76546, 2023b.

Ma, G., Huang, H., Yan, K., Chen, L., Duan, N., Yin, S., Wan, C., Ming, R., Song, X., Chen, X., et al. Stepvideo-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.

Mao, X., Li, Z., Li, C., Xu, X., Ying, K., He, T., Pang, J., Qiao, Y., and Zhang, K. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Po, R., Chan, E. R., Chen, C., and Wetzstein, G. Bagger: Backwards aggregation for mitigating drift in autoregressive video diffusion models. arXiv preprint arXiv:2512.12080, 2025.

Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.-Y., Chuang, C.-Y., et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Shin, J., Li, Z., Zhang, R., Zhu, J.-Y., Park, J., Shechtman, E., and Huang, X. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.

Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al. Make-avideo: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

Song, K., Chen, B., Simchowitz, M., Du, Y., Tedrake, R., and Sitzmann, V. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025.

Song, Y. and Dhariwal, P. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. 2023.

Sun, W., Zhang, H., Wang, H., Wu, J., Wang, Z., Wang, Z., Wang, Y., Zhang, J., Wang, T., and Guo, C. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025a.

Sun, Z., Peng, Z., Ma, Y., Chen, Y., Zhou, Z., Zhou, Z., Zhang, G., Zhang, Y., Zhou, Y., Lu, Q., et al. Streamavatar: Streaming diffusion models for real-time interactive human avatars. arXiv preprint arXiv:2512.22065, 2025b.

Tang, J., Liu, J., Li, J., Wu, L., Yang, H., Zhao, P., Gong, S., Yuan, X., Shao, S., and Lu, Q. Hunyuan-gamecraft2: Instruction-following interactive game world model. arXiv preprint arXiv:2511.23429, 2025.

Teng, H., Jia, H., Sun, L., Li, L., Li, M., Tang, M., Han, S., Zhang, T., Zhang, W., Luo, W., et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, W. and Yang, Y. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, 37: 65618–65642, 2024.

Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu, J. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36:8406–8441, 2023.

Weissenborn, D., T¨ackstr¨om, O., and Uszkoreit, J. Scaling autoregressive video models. arXiv preprint arXiv:1906.02634, 2019.

Wu, C., Huang, L., Zhang, Q., Li, B., Ji, L., Yang, F., Sapiro, G., and Duan, N. Godiva: Generating opendomain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.

Wu, C., Liang, J., Ji, L., Yang, F., Fang, Y., Jiang, D., and Duan, N. N¨uwa: Visual synthesis pre-training for neural visual world creation. In European conference on computer vision, pp. 720–736. Springer, 2022.

Wu, X., Zhang, G., Xu, Z., Zhou, Y., Lu, Q., and He, X. Pack and force your memory: Long-form and consistent video generation. arXiv preprint arXiv:2510.01784, 2025.

Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Xiao, S., Zhang, X., Meng, D., Wang, Q., Zhang, P., and Zhang, B. Knot forcing: Taming autoregressive video diffusion models for real-time infinite interactive portrait animation. arXiv preprint arXiv:2512.21734, 2025.

Xing, J., Xia, M., Zhang, Y., Chen, H., Yu, W., Liu, H., Liu, G., Wang, X., Shan, Y., and Wong, T.-T. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pp. 399–417. Springer, 2024.

Xu, J., Huang, Y., Cheng, J., Yang, Y., Xu, J., Wang, Y., Duan, W., Yang, S., Jin, Q., Li, S., et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

Yan, W., Zhang, Y., Abbeel, P., and Srinivas, A. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.

Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al. Longlive: Realtime interactive long video generation. arXiv preprint arXiv:2509.22622, 2025a.

Yang, Y., Huang, H., Peng, X., Hu, X., Luo, D., Zhang, J., Wang, C., and Wu, Y. Towards one-step causal video generation via adversarial self-distillation. arXiv preprint arXiv:2511.01419, 2025b.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Yesiltepe, H., Meral, T., Akan, A. K., Oktay, K., and Yanardag, P. Infinity-rope: Action-controllable infinite video generation emerges from autoregressive self-rollout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 40256–40265, 2026.

Yi, J., Jang, W., Cho, P. H., Nam, J., Yoon, H., and Kim, S. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6613–6623, 2024.

Yin, T., Zhang, Q., Zhang, R., Freeman, W. T., Durand, F., Shechtman, E., and Huang, X. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22963–22974, 2025.

Zhao, M., Bao, F., Li, C., and Zhu, J. Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. Advances in Neural Information Processing Systems, 35:3609–3623, 2022.

Zhao, M., Zhu, H., Xiang, C., Zheng, K., Li, C., and Zhu, J. Identifying and solving conditional image leakage in image-to-video diffusion model. Advances in Neural Information Processing Systems, 37:30300–30326, 2024.

Zhao, M., He, G., Chen, Y., Zhu, H., Li, C., and Zhu, J. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025a.

Zhao, M., Wang, R., Bao, F., Li, C., and Zhu, J. Controlvideo: conditional control for one-shot text-driven video editing and beyond. Science China Information Sciences, 68(3):132107, 2025b.

Zhao, M., Yan, B., Yang, X., Zhu, H., Zhang, J., Liu, S., Li, C., and Zhu, J. Ultraimage: Rethinking resolution extrapolation in image diffusion transformers. arXiv preprint arXiv:2512.04504, 2025c.

Zhao, M., Zhu, H., Wang, Y., Yan, B., Zhang, J., He, G., Yang, L., Li, C., and Zhu, J. Ultravico: Breaking extrapolation limits in video diffusion transformers. arXiv preprint arXiv:2511.20123, 2025d.

Zheng, K., Wang, Y., Ma, Q., Chen, H., Zhang, J., Balaji, Y., Chen, J., Liu, M.-Y., Zhu, J., and Zhang, Q. Large scale diffusion distillation via score-regularized continuoustime consistency. arXiv preprint arXiv:2510.08431, 2025.

Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., and You, Y. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

##### A. Extended Related Work

Video Generative Models. Building on the tremendous success of diffusion models, many works have applied them to video generation (He et al., 2022; Ho et al., 2022; Singer et al., 2022; Blattmann et al., 2023a;b; Chen et al., 2023; Gupta et al., 2024; Zhao et al., 2024; Xing et al., 2024; Zhao et al., 2025b; 2022). With diffusion transformers (DiTs) demonstrating strong scalability (Bao et al., 2023; Peebles & Xie, 2023), many works have introduced DiT-based large-scale video models (Lin et al., 2024; Zheng et al., 2024; Polyak et al., 2024; Yang et al., 2024; HaCohen et al., 2024; Kong et al., 2024; Ma et al., 2025; Li et al., 2025), such as CogVideoX (Yang et al., 2024), Vidu (Bao et al., 2024) and Wan2.1 (Wan

- et al., 2025). Apart from the full-sequence diffusion models, some works adopt autoregressive next-token prediction to enable video generation (Wu et al., 2021; Hong et al., 2022; Wu et al., 2022; Weissenborn et al., 2019; Yan et al., 2021; Zhao et al., 2025c;a), such as NOVA (Deng et al., 2024) and VideoPoet (Kondratyuk et al., 2023). Video generation based on full-sequence diffusion models currently achieves better overall quality than autoregressive next-token prediction. However, full-sequence diffusion models must generate all frames in one shot, which incurs substantial latency and prevents displaying frames to users as they are produced, hindering interactivity and real-time use. In contrast, autoregressive models can generate videos in a streaming manner, enabling user interaction.

Autoregressive Diffusion Models for Interactive Video Generation. To combine the high quality of diffusion models with the interactivity of autoregressive models, recent works have proposed autoregressive diffusion video models (Jin et al., 2024; Deng et al., 2024; Teng et al., 2025; Chen et al., 2025). These models adopt a frame-wise autoregressive formulation while using diffusion within each frame, e.g., Pyramid Flow (Jin et al., 2024), MAGI-1 (Teng et al., 2025), and SkyReels-v2 (Chen et al., 2025). Such autoregressive diffusion models can display each frame to the user as soon as it is generated, and can adjust the conditioning for subsequent frames based on user feedback, enabling interactive generation. Nevertheless, interactivity typically requires real-time performance, meaning the generation speed should be comparable to the video playback rate. However, diffusion models rely on multi-step sampling and are therefore too slow to meet this requirement. To address this, recent works such as ASD (Yang et al., 2025b), CausVid (Yin et al., 2025) and Self Forcing (Huang et al., 2025a) introduce distillation strategies to obtain few-step generation models.

Such real-time, interactive video generation models are highly promising and have broad applications across many domains. One prominent application is video world modeling. HY-WorldPlay (Sun et al., 2025a), RELIC (Hong et al., 2025), Hunyuan-GameCraft-2 (Tang et al., 2025), and Yume-1.5 (Mao et al., 2025) train real-time interactive video models for realistic world simulation, allowing users to freely explore and take actions in the simulated environment. This interactive world-modeling paradigm further enables embodied intelligence, such as closed-loop control in Vidarc (Feng et al., 2025).

Another major application lies in entertainment and media, supporting interactive content generation (Sun et al., 2025b; Ki

- et al., 2026), including Knot Forcing (Xiao et al., 2025), Live avatar (Huang et al., 2025b), and Motionstream (Shin et al.,

2025). Beyond interactivity, these autoregressive diffusion models have also been shown to excel at long-video generation,

- as demonstrated by Rolling Forcing (Liu et al., 2025), LongLive (Yang et al., 2025a), Self-Forcing++ (Cui et al., 2025), and Deep Forcing (Yi et al., 2025).

##### B. Proofs of Propositions

We assume that all expectations appearing below are well-defined and finite, and that all probability density functions involved are integrable, which are mild regularity conditions standard in diffusion modeling (Song et al., 2020).

###### B.1. The Flaw of Self Forcing’s ODE Distillation

In this section, we first present a formal mathematical statement of Lemma 3.2 and provide its proof. Building on this result, we then prove Proposition 3.3.

Lemma B.1 (Chunk-wise non-injectivity of PF-ODE). Let xt ∈ Rd satisfy the PF-ODE dxt = vθ(xt,t)dt with the unique solution. Define the flow map ϕ : Rd × (0,1] → Rd by ϕ(xt,t) = x0 ∼ pdata(x0). Partition coordinates into xut := (x(tm),...,x(tn)) and xzt := x[td]\{m,...,n}, with k := n − m + 1 < d. If ϕ(xt,t)u is not a.e. constant in xzt for each fixed (xut ,t), then

###### ∀t ∈ (0,1], ∀xt ∈ Rd, ∃yt ∈ Rd,such thatytu = xut , andϕ(yt,t)u ̸= ϕ(xt,t)u. (10)

Moreover, if xT ∼ N(0,I) and pt(xt) > 0for Lebesgue-a.e. xt, then

P(Var(ϕ(xt,t)u | xut ,t) > 0) > 0. (11)

Proof. Fix t ∈ (0,1]. Write any x ∈ Rd as x = (xu,xz) with xu ∈ Rk and xz ∈ Rd−k. Fix an arbitrary xt ∈ Rd and denote u1 := xut and z1 := xzt. Define the measurable map

1,t : Rd−k → Rk, fu

1,t(z) := ϕ((u1,z),t)u. (12) By assumption, for this fixed (u1,t), the function z  → fu

fu

1,t(z) is not a.e. constant (with respect to Lebesgue measure on Rd−k). We claim that then for every z1 ∈ Rd−k there exists some z2 ∈ Rd−k such that fu

1,t(z1). Indeed, if for some z1 one had fu

1,t(z2) ̸= fu

1,t would be everywhere constant, contradicting the assumption. Choose such a z2 for the above z1, and set yt := (u1,z2). Then ytu = xut while

1,t(z1) for all z, then fu

1,t(z) = fu

1,t(z1) = ϕ(xt,t)u, (13) which proves the expression (10).

ϕ(yt,t)u = fu

1,t(z2) ̸= fu

Combining the above result with the fact that xT ∼ N(0,I) and xt admits a nondegenerate probability density, standard measure-theoretic arguments (Kallenberg, 1997; Evans, 2018) imply the following: for the above z1,z2, in a neighborhood of z2 there exist uncountably many zk, each of which maps to a distinct ϕ(xt,t)u, just as z2 does. Equivalently, P(Var(ϕ(xt,t)u | xut ,t) > 0) > 0.

| |
|---|

We next prove Proposition 3.3. First, we formalize this in the following statement.

Proposition B.2 (Distribution mismatch in chunk-wise regression). Using the notation of Lemma B.1, and for each fixed (xut ,t), ϕ(xt,t)u is not a.e. constant in xzt. Consider training a chunk-level model Gθ : Rk × (0,1] → Rk with the regression target

t,t ∥Gθ(xut ,t) − xu0∥2 , (14)

θ∗ = min

Ex

θ

where x0 = ϕ(xt,t). Then the optimal solution is the conditional mean, which does not follow the chunk-wise data distribution, i.e.,

G∗θ(xut ,t) = E[xu0 | xut ,t] ≁ pdata(xu0). (15)

Since DiT-based bidirectional diffusion models contain attention modules that are not constant as evidenced by attention-map experiments in prior works (Xi et al., 2025; Zhao et al., 2025d), the condition in Lemma B.1 holds: for each fixed (xut ,t), ϕ(xt,t)u is not a.e. constant in xzt. We then prove Proposition B.2 using the expression (11).

Proof. Let t ∼ q(t) be the time sampling used in training and let x0 = ϕ(xt,t). Denote

U := xut ∈ Rk, Y := xu0 ∈ Rk, Y := E[Y | U,t]. (16) By the standard squared-loss regression result (Bishop & Nasrabadi, 2006), the minimizer satisfies

G∗θ(U,t) = Y = E[xu0 | xut ,t]. (17)

It remains to show Y ≁ Y (hence Y ≁ pdata(xu0)). Using the L2-orthogonal projection identity (Bishop & Nasrabadi, 2006),

E∥Y ∥2 = E∥ Y ∥2 + E∥Y − Y ∥2 = E∥ Y ∥2 + E[Var(Y | U,t)], (18)

where Var(Y | U,t) := E[∥Y − E[Y | U,t]∥2 | U,t]. By Lemma B.1 (expression 11), P(Var(Y | U,t) > 0) > 0, hence E[Var(Y | U,t)] > 0 and therefore

E∥Y ∥2 > E∥ Y ∥2. (19)

If Y and Y have the same distribution, then E∥ Y ∥2 = E∥Y ∥2, a contradiction. Thus Y ≁ Y , i.e.,

G∗θ(xut ,t) = E[xu0 | xut ,t] ≁ pdata(xu0). (20)

- B.2. Distribution Mismatch in Autoregressive Diffusion Forcing In this section, we first present the basic regularity assumptions and then prove Proposition 3.4.

Let Y := x<i0 , X := xi0, and let Z := x<it be obtained by independently noising each frame of Y via the forward kernel qt|0(Z | Y ) at some fixed t > 0. We have the following mild assumptions.

- • (A1) The model yields the optimal conditional distribution under the diffusion training objective, denoted as pDF(X | Z) =: pDF(xi0 | x<it ) = pdata(x | z). This is a trivial assumption widely used (Song et al., 2020).
- • (A2) X is not independent of Y under pdata(X,Y ), i.e., pdata(X | Y = y) is not pdata(Y )-a.e. constant. The intuitive understanding of this assumption is that different frames within the same video are not independent, but are closely related.
- • (A3) The posterior kernel pdata(Y | Z = y) is positive and non-degenerate on the support of pdata(Y ). In particular, for y ∈ supp(pdata(Y )), the density of pdata(Y | Z = y) is positive.

Proof. By (A1), querying the DF-trained model at a clean prefix value y induces

pDF(x | y) = pDF(x | Z = y) = pdata(x | Z = y). (21) Therefore, it suffices to prove

EY DKL pdata(X | Z = Y )||pdata(X | Y ) > 0. (22)

We prove by contradiction. Assume for contradiction that the left-hand side of expression (22) equals 0. This implies

pdata(X | Z = y) = pdata(X | Y = y) for pdata(Y )-a.e. y. (23) Fix any measurable set A in the sample space of X and define

fA(y) := Pdata(X ∈ A | Y = y). (24) By Eq. (23), for pdata(Y )-a.e. y,

Pdata(X ∈ A | Z = y) = fA(y). (25)

On the other hand, since Z is generated from Y via independent noising, we have the Markov chain X → Y → Z under pdata. Thus, by the tower property (Kallenberg, 1997),

###### Pdata(X ∈ A | Z = y) = Edata[Pdata(X ∈ A | Y )|Z = y] (26)

= Edata[fA(Y )|Z = y]. (27) Combining Eq. (25) and Eq. (27) yields

###### fA(y) = Edata[fA(Y )|Z = y] for pdata(Y )-a.e. y. (28)

By the regularity conditions (A3), the conditional expectation operator Tf(y) := E[f(Y ) | Z = y] admits only pdata(Y )-a.e. constant bounded fixed points. Applying this fact to Eq. (28) implies that fA is pdata(Y )-a.e. constant for every measurable A. Hence pdata(X | Y = y) is pdata(Y )-a.e. constant, i.e., X ⊥⊥ Y, which contradicts (A2). Therefore, the contradiction assumption is false, and the expression (22) holds. Consequently,

data(Y ) DKL pDF(X | y)||pdata(X | y) > 0. (29)

Ey∼p

Table 3. Quantitative comparison of Autoregressive diffusion training strategies. The recent works perform on par with teacher forcing in our setting.

Method Dynamic Degree↑ VisionReward↑ Instruction Following↑

Teacher Forcing 50 3.343 32 PFVG (Wu et al., 2025) 61 2.857 32 BAgger (Po et al., 2025) 53 2.715 28 Resampling Forcing (Guo et al., 2025) 51 3.336 22

##### C. More Discussion of Our Method

###### C.1. Further Remarks on Autoregressive Diffusion Training Strategies

In this section, we first provide further remarks on diffusion forcing, and then report results for other training strategies, including PFVG (Wu et al., 2025), BAgger (Po et al., 2025), and Resampling Forcing (Guo et al., 2025).

As stated in Proposition 3.4, applying diffusion forcing to autoregressive diffusion training is suboptimal. However, this does not render diffusion forcing useless. Specifically, diffusion forcing was originally introduced to train a bidirectional diffusion model for video continuation to enable long-video generation (Song et al., 2025). In this setting, continuation

- at inference time concatenates a clean prefix (the tail frames of the given video) with noise, which matches the training setup and thus avoids a train–inference mismatch. Therefore, this bidirectional diffusion forcing regime is not covered by our suboptimality claim. Moreover, diffusion forcing allows different frames to have different noise levels. Even in autoregressive diffusion training, this is not an issue, since each frame is actually trained independently. The only practice we refute is conditioning on a noisy prefix for autoregressive diffusion training, as proposed by CausVid (Yin et al., 2025) and recent works (e.g., LiveAvatar (Huang et al., 2025b)).

Apart from diffusion forcing and teacher forcing, we also experiment with several recent alternatives, including PFVG (Wu et al., 2025), BAgger (Po et al., 2025), and Resampling Forcing (Guo et al., 2025). However, as shown in Tab. 3, these methods provide no significant improvement over teacher forcing. Notably, most of them are primarily designed for long-video training and generation, so the limited gains in our 5s setting are understandable. We leave a deeper investigation of these strategies for future work.

###### C.2. Multi-Step Autoregressive Diffusion as Initialization for Asymmetric DMD

In this section, we investigate directly using a teacher forcing-trained multi-step autoregressive diffusion model to initialize asymmetric DMD. We find that, compared to Self Forcing’s ODE distillation initialization, multi-step autoregressive diffusion initialization yields substantial improvements in both dynamics and visual quality, as illustrated in Fig. 8 (middle vs. left).

Despite these improvements, the multi-step autoregressive diffusion model only narrows the bidirectional-to-causal architectural gap under multi-step sampling (e.g., 50 steps) and does not fully resolve it in the few-step regime. Specifically, under few-step sampling, autoregressive generation induces an additional mismatch in the conditional context: the i-th frame conditions on preceding frames 0 ∼ i − 1, whose quality degrades at low step counts, whereas training conditions on a high-quality ground-truth prefix. As a result, this degraded conditioning accumulates throughout autoregressive generation, causing error propagation across chunks. As illustrated in Fig. 7 (top), before the DMD stage, we evaluate the autoregressive diffusion model on 4-step generation only; it exhibits abrupt transitions between chunks, indicating that a substantial architectural gap remains in the few-step setting.

This analysis suggests that it’s necessary to convert the multi-step autoregressive diffusion model to a few-step model before using it to initialize DMD. As illustrated in Fig. 7 (bottom), the causal ODE-distilled model exhibits stronger temporal consistency under few-step sampling, making it a more suitable DMD initialization. Consistently, Fig. 8 (middle vs. right) further shows that replacing multi-step autoregressive diffusion initialization with causal ODE initialization yields clear gains in the subsequent DMD stage. This is also supported by the quantitative results in Tab. 4, where causal ODE attains markedly better VisionReward and Instruction Following scores.

Table 4. Quantitative comparison of DMD with different initializations. DMD with Self Forcing’s ODE initialization shows weak dynamics and low visual quality. Initializing with a Teacher Forcing-trained autoregressive diffusion model yields a large improvement, while causal ODE initialization achieves the best overall quality and VisionReward score.

Method Total↑ Quality↑ Semantic↑ Dynamic Degree↑ VisionReward↑ Instruction Following↑

Self Forcing’s ODE + DMD 82.00 82.18 81.29 24 3.330 38 Autoregressive diffusion + DMD 84.02 84.72 81.23 66 5.863 48 Causal ODE + DMD 84.04 84.59 81.84 68 6.326 56

Type Generated video with 4 steps

[Figure 85]

[Figure 86]

[Figure 87]

#### …

Autoregressive diffusion

[Figure 88]

[Figure 89]

[Figure 90]

#### …

Causal ODE distillation

- Figure 7. Performance comparison with 4-step generation before the DMD stage. Without having reached the DMD stage yet, we directly compare the 4-step generation of the autoregressive diffusion model with the 4-step generation of the causal ODE-distilled model. Autoregressive diffusion exhibits inter-frame abrupt changes, indicating suboptimal causality under 4 steps, whereas the causal ODE–distilled model remains more stable.

Asymmetric DMD with Self Forcing’s ODE initialization

Asymmetric DMD with autoregressive diffusion initialization

Asymmetric DMD with causal ODE initialization

… … …

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

- Figure 8. Performance comparison of DMD with different initialization. DMD with Self Forcing’s ODE initialization shows weak dynamics and abrupt artifacts. Initializing with TF-trained autoregressive diffusion brings a large improvement but still exhibits abrupt changes (e.g., two red flowers turning into one), whereas causal ODE initialization yields the highest quality and the most stable results.

Self Forcing’s ODE distillation, bidirectional initial model

Causal ODE distillation, causal initial model

Causal ODE distillation, bidirectional initial model

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

###### … … …

- Figure 9. Student initialization is not the bottleneck of ODE distillation. With causal ODE distillation, the student with a bidirectional initial model achieves similar performance to that with a causal initial model, both better than Self Forcing’s ODE distillation.

###### C.3. Causal ODE Distillation from Bidirectional Initial Model

Recall from Sec. 3.2 that we claim that we should adopt causal ODE distillation rather than Self Forcing’s ODE distillation. For causal ODE distillation, the paired data (xit,xi0) are generated by an autoregressive diffusion model, and the student is initialized from this autoregressive diffusion model as well. For Self Forcing’s ODE distillation, the paired data are generated by a bidirectional diffusion model, and the student is initialized from this bidirectional diffusion model as well. Though our main argument attributes the performance gap of these two methods to how the paired data are constructed, yet these two methods also differ in their initialization, raising a natural question: is the observed difference in Fig. 3 mainly driven by the paired data construction, or by the initialization difference?

To answer this question, we use paired data synthesized by the autoregressive diffusion model, i.e., DCausal in Sec. 4, while initializing the student from the bidirectional diffusion model. As shown in Fig. 9, the resulting quality is comparable to initializing from the autoregressive diffusion model, still much better than that of the Self Forcing’s ODE distillation. This indicates that the performance gap in ODE distillation is not primarily due to student initialization, but rather to the distillation setup: the teacher should be an autoregressive diffusion model rather than a bidirectional one.

- D. More Implementation Details In this section, we provide more details of Sec. 4.

Training details of our method. We first construct a dataset DBi consisting of about 3K samples generated by Wan (bidirectional) (Wan et al., 2025) with the VidProM (Wang & Yang, 2024) prompts and train a teacher forcing autoregressive diffusion model for 2K steps. Next, we sample ODE trajectories from the autoregressive diffusion model to construct a causal ODE dataset DCausal with 3K samples. Notably, since the causal ODE distillation conducts teacher forcing conditioned on the ground-truth clean data, we save the corresponding relationship between each data point in DCausal and DBi. Throughout training, including the teacher forcing autoregressive diffusion model and both ODE-initialization variants, we use either DBi or DCausal, both internally synthesized by the model, and using about the same prompts, thus guaranteeing that there is no gap in terms of data quality. This ensures the fairness of the comparison in the ablation study. Then, we perform causal ODE distillation on DCausal for 1K steps via teacher forcing, conditioned on the corresponding clean data in DBi. In this stage, the ODE student is initialized from the autoregressive diffusion teacher. Finally, we use this model as initialization for the standard asymmetric DMD, where sreal is Wan2.1-14B, and sfake is Wan2.1-1.3B, strictly following the setting of Self Forcing (Huang et al., 2025a) to guarantee the fair comparison. For both the chunk-wise and frame-wise settings, we adopt the same overall formulation and pipeline.

Throughout all training stages, we use a batch size of 64 and the Adam optimizer with a learning rate of 2 × 10−6, β1 = 0, and β2 = 0.999, while keeping all other settings identical to Self Forcing. During inference, we use 4-step sampling with timesteps shared by causal ODE initialization and asymmetric DMD, i.e., 1, 0.9375, 0.8333, and 0.625.

For the extended causal consistency distillation, we adopt the LCM (Luo et al., 2023a) scheme with 48 discretized timesteps, using the UniPC ODE solver with an EMA rate of 0.99. We train the model for 3K steps on DBi, and inference also uses 4 steps, with the same timesteps as DMD. Notably, discrete-time consistency distillation requires the boundary condition Gθ(xi,x<igt ,0) ≡ xi, which is typically implemented by introducing a wraped network Fθ to parameterize Gθ:

Gθ(xi,x<igt ,t) = cskip(t)xi + cout(t)Fθ(xi,x<igt ,t), (30) where cskip(0) = 1, cout(0) = 0. However, since we use flow matching, i.e., a v-prediction parameterization for the diffusion

Type Generated video

[Figure 103]

[Figure 104]

[Figure 105]

### …

Asymmetric CD

[Figure 106]

[Figure 107]

[Figure 108]

#### CausalCD …

- Figure 10. Comparison between asymmetric CD and causal CD. Asymmetric CD appears highly blurry and exhibits abrupt artifacts, whereas causal CD results remain much better quality and more stable.

model vθ, an x0-prediction form for Gθ already satisfies the required boundary conditions, without any additional design: Gθ(xi,x<igt ,t) = xi − tvθ(xi,x<igt ,t). (31)

This simplified design may not be optimal and leaves substantial design space for further exploration (Geng et al., 2024; Lu & Song, 2024; Zheng et al., 2025), which we leave to future work. Fig. 10 visualizes that causal CD outperforms asymmetric CD, where asymmetric CD appears highly blurry and exhibits abrupt artifacts, whereas our results achieve better quality and are more stable. This agrees with Tab. 2 and highlights the necessity of a native causal teacher for autoregressive CD.

Evaluation details. In this section, we focus on the setups of Dynamic Degree, VisionReward, and Instruction Following. We use 100 prompts with rich action sequences and dynamics, provided in the supplementary material.

For Dynamic Degree, we use the official VBench evaluation code in the custom-input mode. Note that the Dynamic Degree reported in our table is evaluated on the 100-prompt motion set; however, when computing VBench’s Total, Quality, and Semantic scores, the Dynamic Degree term is still evaluated on the standard VBench official prompts rather than inherited from our custom set. In addition, we use VisionReward to evaluate overall visual quality. Each sub-score can be positive or negative and lies in [−1,1], where −1 indicates the worst quality and 1 the best. The final VisionReward score is computed as a weighted sum using the official weights. We additionally use VisionReward’s prompt-alignment sub-score to evaluate instruction following, by querying VisionReward with the official prompt: Does the video meet some of the requirements stated in the text ”[[prompt]]”?

Performance comparison details. All baselines use the same spatial resolution as Self Forcing. Throughput and latency on the H100 GPU for the baselines are taken directly from the Self Forcing paper.

Ablation details. For autoregressive diffusion training, both teacher forcing and diffusion forcing are trained for 3K steps. For Self Forcing’s ODE initialization, we start from the bidirectional diffusion model and train for 3K steps. For causal ODE initialization, we first train a teacher forcing autoregressive diffusion model for 2K steps, and then run an additional 1K-step causal ODE distillation initialized from it. This aligns the training overall computation (both 3K in total) of the two ODE-initialization variants and ensures a fair comparison. Both CD variants are trained for 3K steps as well, each distilled using the teacher forcing autoregressive diffusion model as the teacher, ensuring a fair within-CD comparison. All other settings follow the main experiments.

