# arXiv:2605.15141v3[cs.CV]1Jun2026

[Figure 1]

## Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation

Min Zhao1,2 ∗, Hongzhou Zhu1,2 ∗, Kaiwen Zheng1, Zihan Zhou3, Bokai Yan3, Xinyuan Li1, Xiao Yang1, Chongxuan Li3, Jun Zhu1,2 † 1Tsinghua University 2ShengShu 3Renmin University of China

∗Equal contribution. †Correspondence to: Jun Zhu.

{gracezhao1997, suinibian74}@gmail.com; dcszj@tsinghua.edu.cn

Abstract Real-time interactive video generation requires low-latency, streaming, and controllable rollout. Existing autoregressive (AR) diffusion distillation methods have achieved strong results in the chunk-wise 4-step regime by distilling bidirectional base models into few-step AR students, but they remain limited by coarse response granularity and non-negligible sampling latency. In this paper, we study a more aggressive setting: frame-wise autoregression with only 1–2 sampling steps. In this regime, we identify the initialization of a few-step AR student as the key bottleneck: existing strategies are either target-misaligned, incapable of few-step generation, or too costly to scale. We propose Causal Forcing++, a principled and scalable pipeline that uses causal consistency distillation (causal CD) for few-step AR initialization. The core idea is that causal CD learns the same AR-conditional ﬂow map as causal ODE distillation, but obtains supervision from a single online teacher ODE step between adjacent timesteps, avoiding the need to precompute and store full PF-ODE trajectories. This makes the initialization both more efﬁcient and easier to optimize. The resulting pipeline, Causal Forcing++, surpasses the SOTA 4-step chunk-wise Causal Forcing under the frame-wise 2-step setting by 0.1 in VBench Total, 0.3 in VBench Quality, and 0.335 in VisionReward, while reducing ﬁrst-frame latency by 50% and Stage 2 training cost by ∼4×. We further extend the pipeline to action-conditioned world model generation in the spirit of Genie3. Project Page: https://github.com/thu-ml/Causal-Forcing & https://github.com/shengshu-ai/minWM.

### 1. Introduction

Video generation models are rapidly evolving from passive content generators into interactive world models [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], where low latency, streaming rollout, and user-controllable interaction are essential. Autoregressive (AR) diffusion models [15, 16, 17] are a natural ﬁt for this goal, as they perform causal rollout across frames or chunks while retaining diffusion-based generation within each segment. Recent AR diffusion distillation methods [8, 10, 18, 19, 20] have achieved promising results by distilling bidirectional video diffusion models, such as Wan [3] and Hunyuan [4], into few-step AR students. However, these methods typically rely on chunk-wise autoregression with 4-step sampling, which still falls short of real-time interaction due to coarse response granularity and non-negligible sampling latency. We therefore push AR diffusion distillation to a more aggressive and largely underexplored regime: frame-wise autoregression with only 1–2 sampling steps.

We identify the initialization of a few-step AR student before asymmetric DMD as the key bottleneck in this aggressive regime, where existing strategies fail in complementary ways. ODE initialization with a bidirectional teacher, as used in CausVid [18] and Self Forcing [19], is architecturally misaligned with causal rollout: the teacher trajectory depends on future frames that are unavailable to an AR student, thereby providing an incorrect regression target. Directly using a multi-step AR diffusion model for initialization,

as in LiveAvatar [10] and WorldPlay [8], avoids this mismatch but lacks few-step generation capability; under frame-wise 1–2 step generation, its per-frame approximation error is severely ampliﬁed during selfrollout. Causal ODE initialization, as in Causal Forcing [20], corrects the learning target by distilling from an AR teacher, but requires generating full multi-step PF-ODE trajectories for every training sample, making it costly to scale. Therefore, a satisfactory initialization for this regime must be simultaneously AR, few-step, and scalable.

To this end, we introduce Causal Forcing++, a principled and scalable pipeline that uses causal consistency distillation (causal CD) for few-step AR student initialization. Our key observation is that causal ODE distillation and causal CD aim to learn the same object: the AR-conditional ﬂow map (or namely the consistency function [21, 22, 23, 24]) of the teacher. They differ, however, in how the supervision is obtained. Causal ODE distillation requires the AR teacher to generate an entire multi-step PF-ODE trajectory for each training sample, which must be precomputed and stored ofﬂine. In contrast, causal CD obtains supervision from a single online teacher ODE step between adjacent timesteps on real videos. Therefore, causal CD serves as a principled substitute for causal ODE initialization, while avoiding the expensive trajectory-generation bottleneck. Beyond efﬁciency, this local supervision also improves quality: adjacent-timestep consistency yields a smaller per-step optimization gap than causal ODE distillation, which regresses noisy intermediate states directly to clean endpoints [25]. As a result, causal CD is easier to optimize and empirically produces a stronger few-step AR student.

Experiments on Wan2.1-1.3B [3] validate the effectiveness of Causal Forcing++ in this aggressive lowlatency regime. Under frame-wise 2-step generation, Causal Forcing++ achieves the best overall performance among existing AR diffusion distillation methods, improving VBench Total, VBench Quality, and VisionReward over prior methods while reducing ﬁrst-frame latency by 50%. Ablation studies further show that causal CD consistently matches or outperforms causal ODE initialization across 1-step, 2-step, and 4step settings, while reducing the Stage 2 cost by about 4× and requiring no auxiliary trajectory storage. We also examine causal score-distillation initialization and ﬁnd that, although it can produce sharper early frames, its mode-seeking behavior makes it more sensitive to accumulated history errors during AR rollout, leading to stronger exposure bias. Finally, we demonstrate that Causal Forcing++ naturally extends to action-conditioned world model generation by distilling a camera-pose-conditioned generator into an interactive AR world models.

### 2. Background

#### 2.1. Autoregressive Diffusion for Interactive Video Generation

Generative modeling through diffusion. Score-based diffusion models [26, 27] typically perturb the data x0 ∼ pdata(x0) through the diffusion process xt = α(t)x0 + σ(t)ϵ, ϵ ∼ N(0,I),t ∈ [0,1], where α(·),σ(·) are functions of the timestep t, namely the noise schedules. Trained in this process, the model learns scorerelated information about pdata(x0). A recent trend is ﬂow matching [28, 29], which sets α(t) = 1 − t and σ(t) = t, and parameterizes the model using velocity prediction, denoted by vθ. The model is trained by minimizing Ex

dt = ϵ − x0. By using the optimal model vθ to solve the probability ﬂow ordinary differential equation (PF-ODE) [27] dxt = vθ(xt,t)dt, t : 1 → 0, one can generate samples x0 ∼ pdata(x0). Owing to their remarkable expressive power, diffusion models have achieved widespread success in image and video generation [2, 30, 31, 32, 33, 34, 35].

0,ϵ,t[||vθ(xt,t) − vt||2], where vt := dx

t

Autoregressive diffusion for video generation. Despite the remarkable success of diffusion models in video generation [1, 2, 3, 4, 5, 6, 7, 36], they are typically bidirectional, generating the entire video content in a single pass. This leads to high latency, namely a long delay from the start of generation to the completion of the ﬁrst frame, and also makes them non-interactive, since the user must specify the full conditioning signals in advance. By contrast, autoregressive (AR) generation [33, 35, 37, 38, 39, 40, 41, 42, 43], operates on much smaller generation units, thereby offering lower latency. It also enables interactive generation, as users can provide feedback based on the content already generated and adjust subsequent conditioning signals accordingly. To combine the high generation quality of diffusion models with the low latency and

fixing the theoretical flaw by frame-level injectivity

theoretically equivalent, but more efficient and effective

[Figure 2]

[Figure 3]

Self Forcing

Causal Forcing

###### Causal Forcing ++

Bidirectional base model

Bidirectional base model

Bidirectional base model

###### bidirectional teacher

AR diffusion

AR diffusion

teacher

teacher

teacher

AR teacher

ODE initialization with bidirectional teacher

AR teacher

Causal CD

ODE initialization

student init.

student init.

student init.

DMD with self rollout

DMD with self rollout

DMD with self rollout

flawed and costly

correct, efficient and scalable

correct but costly

- (a)
- (b)

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

83.74

1500

###### SF CF CF++

5000

0.6

11600

84.04

1900

0.6

84.14

4

0.27 2

2900 0 ~100% cost reduction

Initialization time Cost ↓ Extra data storage cost ↓ Latency ↓ Vbench total score↑

- Figure 1: Overall framework of our Causal Forcing++ and the comparison with existing methods. (a) Causal Forcing (CF) ﬁxes Self Forcing (SF)’s frame-level injectivity ﬂaw but remains costly; our Causal Forcing++ (CF++) is theoretically sound, efﬁcient, and scalable. (b) Our CF++ reduces training cost by 4×, requires no extra data curation, and achieves 50% lower latency and higher VBench scores than SF and CF.

interactivity of AR models, a recent trend is AR diffusion [15, 16, 17], which performs autoregression across frames (or chunks) and diffusion within each frame (or chunk). These models typically adopt a causal attention mask to be trained with teacher forcing or its variants [16, 44, 45, 46, 47], and perform self-rollout inference with a KV cache.

#### 2.2. Autoregressive Diffusion Distillation

Although AR diffusion enables interactivity, its multi-step generation process still hinders real-time generation. This has motivated the development of AR diffusion distillation [8, 48, 49, 50, 51, 52, 53, 54, 55].

CausVid. CausVid [18], one of the the earliest representatives of the current AR diffusion distillation paradigm, adopts a two-stage framework: (1) ODE initialization, which samples PF-ODE trajectories from a bidirectional diffusion model and trains an AR student via regression; (2) asymmetric DMD [56, 57, 58], which keeps the teacher and critic bidirectional while using the ODE-initialized AR model as the student, trained under diffusion forcing on real data. The rationale is that bidirectional diffusion models currently achieve stronger performance than AR diffusion models, so a stronger bidirectional teacher is expected to transfer a better generative distribution to the AR student.

Self Forcing. Self Forcing [19] improves upon CausVid by correcting the asymmetric DMD stage. Speciﬁcally, it observes that CausVid trains the student under diffusion forcing on real data, so each generated frame is conditioned on ground-truth context rather than on self-generated preﬁxes. Consequently, the resulting frames do not form a valid generated video when concatenated, leading to a substantial gap between training and inference-time self-rollout. To resolve this issue, Self Forcing replaces diffusion forcing generation in the DMD stage with student self-rollout, thereby aligning training with inference and substantially improving performance.

Causal Forcing. Causal Forcing [20] corrects the ODE initialization stage used in CausVid and Self Forcing, while retaining the asymmetric DMD design of Self Forcing. It points out that, unlike DMD which merely pursues distribution matching, ODE distillation is intended to match the generation trajectories. Consequently, an AR student is theoretically incapable of ﬁtting the ODE trajectories induced by a bidirectional teacher, and thus cannot properly bridge the architectural gap. Motivated by this observation, Causal Forcing ﬁrst ﬁne-tunes a bidirectional diffusion model into an AR diffusion model, and then uses the resulting AR teacher to generate ODE trajectories for initializing the AR student. The initialized student is subsequently optimized with asymmetric DMD, while the teacher and critic remain bidirectional.

In summary, the training pipeline consists of the following three stages, whose terminology we adopt below: (1) Stage 1: multi-step AR diffusion training via teacher forcing; (2) Stage 2: causal ODE initialization with the AR teacher; and (3) Stage 3: asymmetric DMD with student self-rollout.

### 3. Method

The AR diffusion distillation pipelines reviewed in Sec. 2.2 have achieved strong results in chunk-wise (typically 3 latent frames) 4-step AR generation, but two questions remain open. First, none of them has been validated in more aggressive low-latency regimes—in particular, frame-wise AR generation with as few as 1–2 sampling steps—which would better realize the low-latency promise of real-time interactive generation. Second, the existing few-step initialization strategy, ODE distillation, requires the teacher to generate full PF-ODE trajectories for every training datum; this is structurally expensive and makes systematic exploration of harder regimes costly. In this section we address both. We ﬁrst show that no existing initialization strategy is satisfactory in our target regime (Sec. 3.1); we then propose causal consistency distillation as a principled and scalable substitute for causal ODE distillation (Sec. 3.2); ﬁnally, we extend our method to action-conditioned world model generation (Sec. 3.3). The overview of our method and its relation to previous works are shown in Fig. 1.

#### 3.1. The Necessity of Few-Step AR Student Initialization

Since asymmetric DMD is highly sensitive to its few-step student initialization [20], we begin by examining whether existing initialization strategies sufﬁce in our target regime. In recent works, three options have been proposed: (i) distilling a few-step AR student from a bidirectional teacher via ODE distillation, as in CausVid [18] and Self Forcing [19]; (ii) skipping few-step distillation entirely and using the multi-step AR diffusion model directly, as in LiveAvatar [10] and WorldPlay [8]; and (iii) distilling a few-step AR student from an AR teacher via ODE distillation, as in Causal Forcing [20]. We compare these three candidates as the autoregressive unit shrinks from chunk-wise to frame-wise and the sampling step count drops from 4 to 1. As shown in Fig. 2, we ﬁnd that all three fall short for complementary reasons: one is architecturally misaligned, one is too weak, and one is too costly to scale. This motivates an initialization that is simultaneously AR, few-step, and scalable, as we now establish.

ODE initialization with a bidirectional teacher is misaligned. The ﬁrst candidate uses ODE distillation with a bidirectional teacher, which is the initialization mechanism behind CausVid [18] and Self Forcing [19]. As Causal Forcing [20] shows, this choice violates the frame-level injectivity required by an AR studentthe same noisy frame can correspond to multiple clean frames under different future contexts—so ODE distillation no longer recovers the AR ﬂow map but instead collapses toward the conditional expectation E[xi0 | xit,x<it ,t], producing a blurred and poorly aligned initialization. This mismatch becomes more damaging in our frame-wise low-step settings, where the student must repeatedly roll out from its own imperfect history. Quantitatively, asymmetric DMD with Self-Forcing-style initialization already collapses in the chunkwise 4-step setting, becomes even worse in the frame-wise 4-step setting, and eventually catastrophically breaks down in the frame-wise 1-step setting, as illustrated in Fig. 2(column 1). ODE initialization with a bidirectional teacher is therefore fundamentally ﬂawed and not a viable foundation for low-latency AR diffusion distillation.

DMD after Self Forcing ODE initialization DMD after multi-step AR diffusion initialization

DMD after Casual Forcing ODE initialization

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

chunk-wise 4-step

chunk size ↓

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

frame-wise 4-step

step ↓

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

frame-wise 1-step

fix the theoretical flaw and improve quality, but requires costly data curation

theoretical flaw leads to poor quality lack of few-step capability leads to poor quality

- Figure 2: Performance of existing initialization methods after DMD. Self Forcing ODE initialization and multistep AR diffusion initialization leads to poor quality and degrade as the setting becomes more aggressive, while Causal Forcing’s causal ODE initialization performs well but is costly and therefore difﬁcult to scale.

Multi-step AR diffusion initialization degrades sharply in aggressive settings. Using the multi-step AR diffusion model directly as the student initialization yields consistently weaker results than explicit few-step distillation, and this gap widens as we move toward lower-latency generation. As shown in Fig. 2(column 2 vs. column 3), the gap is moderate under chunk-wise generation, larger under frame-wise 4-step, and largest under frame-wise 1-step, where the model nearly collapses. The reason is structural: reducing the chunk size increases the number of AR calls needed to generate a video, while reducing the sampling step count increases the approximation error within each call. These two effects compound during self-rollout, amplifying exposure bias. In this sense, asymmetric DMD acts as a reﬁner rather than a complete trainer: if the initialization lacks few-step capability, DMD inherits an optimization burden it cannot reliably absorb.

Casual ODE initialization with an AR teacher works, but is difﬁcult to scale. Causal Forcing [20] addresses the architectural mismatch of Self-Forcing-style ODE initialization by replacing the bidirectional teacher with an AR teacher and performing causal ODE distillation between AR models. This produces a well-aligned few-step initialization and leads to strong chunk-wise 4-step results after asymmetric DMD, as illustrated in Fig. 2(column 3). Its limitation is not correctness but scalability: each training datum requires the AR teacher to generate a full multi-step PF-ODE trajectory (e.g., 48 steps per sample); these paired trajectories must be stored ofﬂine; and they must be regenerated whenever the teacher, data distribution, or chunk-size conﬁguration changes. Worse, the cost grows with task difﬁculty: harder regimes typically demand more training data and longer trajectories, compounding the bottleneck. At our 80K-video scale, this data curation along with the training costs roughly 11,600 A800-GPU hours and 1,900GiB of additional storage, as quantiﬁed in Tab. 2. Thus, causal ODE initialization works in principle but imposes a structural scaling bottleneck that limits its practical reach.

Taken together, the three existing options leave us with no satisfactory initialization for asymmetric DMD in aggressive low-latency regimes: one is architecturally misaligned, one lacks few-step capability, and one is too costly to scale. We therefore need an initialization that is simultaneously AR, few-step, and scalable.

#### 3.2. Causal Forcing++: Causal CD as a Principled and Scalable Substitute for Causal ODE Initialization

Sec. 3.1 established that the initialization of asymmetric DMD should be simultaneously AR, few-step, and scalable, and that causal ODE distillation [20] satisﬁes only the ﬁrst two. Our key observation is that causal ODE distillation and causal consistency distillation (CD) shares the same learning target: the ﬂow map (or the consistency function) of the AR teacher. Building on this equivalence, we propose causal CD as the few-

step AR student initialization. We show that this substitution is principled—sharing the learning target of causal ODE distillation—and brings two practical advantages: it eliminates the ofﬂine-trajectory bottleneck and yields a stronger initialization via a smaller per-step optimization gap. We refer to the resulting pipeline as Causal Forcing++.

Causal ODE distillation shares the same target as causal consistency distillation. Causal ODE distillation [20] collects intermediate states xit and corresponding clean outputs xi0 along the PF-ODE trajectory of the AR diffusion teacher, and trains the student by MSE regression with teacher forcing:

[

]

. (1) The minimizer of this objective is the AR-conditional ﬂow map of the teacher,

θ∗ = arg min

Gθ(xit,x<igt ,t) − xi0 2

Ex<i

gt , t, i, xit

θ

###### fϕ : (xit, x<igt , t)  → xi0, (2)

which maps xit at any time t to the PF-ODE sample xi0 of the teacher diffusion model ϕ at t = 0. Recognizing fϕ as the AR-conditional analog of the consistency function [21, 22, 23, 24, 59], we lift the standard CD objective to the AR setting via teacher forcing:

gt,ϵ, t, i[w(t)d

)], (3)

(

θ∗ = arg min

Gθ(xit,x<igt ,t), Gθ−(xˆit−∆t,x<igt ,t − ∆t)

Ex

θ

where xit is obtained from the ground-truth xigt via the forward diffusion process, xˆit−∆t is obtained by a single ODE step from xit using the AR teacher conditioned on the ground-truth preﬁx x<igt , θ− is the EMA of θ with stop-gradient, w(·) is a timestep-dependent weight, and d(·,·) is a distance under a pre-deﬁned norm. Under the ﬂow-matching parameterization of Sec. 2.1,

###### Gθ(xit,x<igt ,t) = xit − tvθ(xit,x<igt ,t), (4)

where vθ is the neural network. Following standard CD analysis [21], the error between the optimal model Gθ∗ and the target ﬂow map fϕ (namely the consistency function) is bounded by the numerical error of the ODE solver and is therefore negligible:

###### sup||fϕ(xit,x<igt ,t) − Gθ∗(xit,x<igt ,t)||2 = O((∆t)p), (5)

where ∆t denotes the maximum difference between adjacent timesteps, and the ODE solver is p-th order accurate.

In other words, causal ODE distillation and causal CD share the same target; they differ only in how that target is approached—ODE distillation regresses to it via large jumps from t to 0 on pre-generated trajectories, while CD enforces it locally between adjacent timesteps on real data. This equivalence motivates causal CD as a principled substitute for causal ODE distillation.

Causal CD is more efﬁcient and yields a stronger initialization. Beyond principled equivalence, Causal CD delivers two practical advantages over causal ODE distillation. (1) Efﬁciency. Causal ODE distillation requires the teacher to pre-generate a full multi-step PF-ODE trajectory for every training datum (e.g., 48 steps), store the paired trajectories ofﬂine, and regenerate them whenever the teacher or data distribution changes. Causal CD requires only one teacher ODE step per training iteration, performed online on groundtruth videos. At our 80K-video scale (Fig. 3(b)), this reduces the data curation and training cost from ∼11,600 to ∼2,900 A800-GPU hours (∼4× speedup) and the auxiliary storage from ∼1,900GiB to zero. (2) Quality. Causal ODE distillation regresses xit to xi0 across the entire t → 0 interval, asking the student to bridge a large gap in one step. Causal CD instead pairs xit with xˆit−∆t between adjacent timesteps, reducing the per-step gap to ∆t and yielding an easier optimization target—consistent with prior observations in bidirectional distillation [25]. Empirically (Fig. 3(a)), causal CD matches or surpasses causal ODE distillation on VBench [60] both at the end of Stage 2 and after the asymmetric DMD stage. These advantages are structural properties of the local-pairing scheme and apply uniformly across chunk-wise and frame-wise conﬁgurations.

Before asymmetric DMD After asymmetric DMD

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Causal ODE

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Causal CD

82

83.5

| |Causal CD<br><br>Causal ODE|
|---|---|
| | |

| |Causal CD Init.<br><br>Causal ODE Init.|
|---|---|
| | |

80

83.25

78

83

VBench VBench

12500

| |Causal CD Causal ODE|
|---|---|
| | |
| | |
| | |

10000

7500

5000

2500

Time cost (GPU · hours)

2000

| |Causal CD<br><br>Causal ODE|
|---|---|
| | |
| | |
| |[Figure 34]|

1500

1000

500

0

Extra data storage (GiB)

(a) Performance comparison. Causal CD achieves higher VBench scores than causal ODE both when directly evaluated at Stage 2 and when used as the initialization for Stage 3, while also producing videos with better visual quality. Init. denotes initialization.

(b) Training-efﬁciency. Causal CD dramatically reduces training time and requires no extra storage.

- Figure 3: Performance and efﬁciency comparison between causal CD and causal ODE. Causal CD outperforms causal ODE, while subsequently improving training efﬁciency in time and storage.

Causal Forcing++. Putting these pieces together, our pipeline inherits Stage 1 (teacher forcing AR diffusion training) and Stage 3 (asymmetric DMD with self-rollout) from Causal Forcing [20], and replaces its Stage 2 with the causal CD objective of Eq. (3). We refer to this pipeline as Causal Forcing++. In summary, Causal Forcing++ improves over prior AR diffusion distillation along two complementary axes. Relative to Self Forcing [19], whose ODE initialization from a bidirectional teacher violates frame-level injectivity and degenerates to a conditional-expectation target, Causal Forcing++ is both correct—targeting the correct AR ﬂow map—and more efﬁcient, since causal CD avoids ofﬂine trajectory generation entirely. Relative to Causal Forcing [20], whose causal ODE initialization recovers the correct target but is bottlenecked by ofﬂine trajectory generation, Causal Forcing++ is both more efﬁcient and higher-quality, owing to the smaller per-step optimization gap of local consistency pairing.

#### 3.3. Application to Action-Conditioned World Models

We further showcase the applicability of our pipeline to action-conditioned world model generation. Following the Genie3-style [9] camera-pose conditioning paradigm, we instantiate a chunk-wise 4-step generator based on Wan2.1-1.3B, using camera poses as the action signal. The pipeline involves three stages: (i) constructing a camera-pose-annotated training dataset with WorldPlay [8]; (ii) ﬁnetuning Wan2.1-1.3B into a bidirectional camera-pose-conditioned diffusion model by injecting pose information via PRoPE [61]; and (iii) distilling this bidirectional model with Causal Forcing++ into an interactive action-conditioned world model. Qualitative results are shown in Fig. 4, and full demos are provided on the project page. Further reducing the action-conditioned variant to the frame-wise 2-step setting for fully real-time interaction is left for future work.

#### 3.4. Further Discussion

##### 3.4.1. Whether Causal Score Distillation Yields Strong Initialization

Since causal consistency distillation has been shown to provide a strong few-step initialization above, a natural question is whether its counterpart in step distillation, namely score distillation [56, 57, 58], can also serve this role. Given that score distillation often yields higher quality than consistency distillation in bidirectional distillation [24, 58, 62], this appears promising. In this section, we examine whether causal

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

(a) Action: moving forward continuously.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

(b) Action: move forward ﬁrst, then tilt the camera downward.

- Figure 4: Application of Causal Forcing++ to the action-conditioned world model. Causal Forcing++ enables efﬁcient, high-quality distillation from a bidirectional model to a low-latency AR model, thereby enabling interactive generation toward a Genie3-style world model.

score distillation actually works.

Causal DMD with teacher forcing. We begin by adapting score distillation into a teacher forcing causal form. Speciﬁcally, we take DMD [56, 57, 58, 62] as a representative example and examine its causal variant. DMD optimizes the student generator through the gradient of the KL divergence between the noised student and data distributions. The student ﬁrst produces x˜ ∼ pθ(x˜), which is then perturbed to x˜t through the forward diffusion process, inducing pθ,t(x˜t). A frozen diffusion model sreal estimates the data score, while an online-trained diffusion model sfake estimates the student score. Their difference provides the distributionmatching direction, which is backpropagated to update the generator as follows:

###### ∇θEt[DKL(pθ,t(x˜t)||pdata,t(x˜t))] = −Ex˜,t,x˜

[(sreal(x˜t,t) − sfake(x˜t,t))

t

∂x˜ ∂θ

]. (6)

When distilling an AR diffusion model, the setting becomes slightly different: the teacher models sreal, sfake and the student Gθ all model conditional distributions autoregressively. Therefore, the above objective must be trained under teacher forcing, as follows:

∂x˜i ∂θ

]. (7)

∇θEt[DKL(pθ,t(x˜it|x<igt )||pdata,t(x˜it|x<igt ))] = −Ex<i

gt ,x˜i,t,x˜it[(sreal(x˜it,x<igt ,t) − sfake(x˜it,x<igt ,t))

With this training scheme, we can directly distill an AR diffusion model via DMD into an AR few-step generator. Note that all models involved in causal DMD, including the teachers and the student, are autoregressive. It is therefore a purely causal distillation procedure, fundamentally different from the subsequent asymmetric DMD, where the teacher is bidirectional while the student is autoregressive. In summary, these two DMD stages serve distinct roles: causal DMD provides a few-step AR student initialization for the subsequent asymmetric DMD.

Severe exposure bias undermines causal DMD. Although bidirectional DMD usually achieves better quality than bidirectional CD [24, 58, 62], we observe the opposite trend in the AR setting: causal DMD yields lower overall quality than causal CD, as shown by the Stage 2 VBench comparison in Fig. 5(a). More interestingly, this degradation is not uniform across frames. In the Stage 2 visualizations of Fig. 5(a), the ﬁrst few frames generated by causal DMD even appear sharper and better than those from causal CD. However, as autoregressive generation proceeds, the later frames from causal DMD rapidly drift, accompanied by severe

Before asymmetric DMD After asymmetric DMD

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

###### Causal CD

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Causal DMD

- 78
- 79
- 80
- 81
- 82

- 82

- 82.5
- 83

- 83.5

| |Causal CD<br><br>Causal DMD|
|---|---|
| | |
| | |
| | |

| |Causal CD Init.<br><br>Causal DMD Init.|
|---|---|
| | |
| | |

VBench VBench

(a) Performance comparison. Causal DMD yields lower VBench scores than causal CD both when directly evaluated after Stage 2 and when used as the initialization for Stage

- 3. Although causal DMD produces sharper early frames, its quality rapidly drifts during autoregressive rollout, leading to severe exposure bias. Init. denotes initialization.

[Figure 51]

Good quality Poor quality

Causal CD (mode-covering)

shift

[Figure 52]

Causal DMD (mode-seeking)

shift

(b) Under history shift, causal CD preserves good-quality mass; causal DMD shifts most mass into poor-quality regions.

- Figure 5: Performance comparison between causal CD and causal DMD and the intuitive explanation. Causal DMD achieves better early-frame quality than causal CD, but suffers from severe exposure bias in later frames, since mode-seeking DMD is more sensitive to accumulated errors.

camera shifts, and eventually degrade to an unacceptable level. In contrast, although causal CD also suffers from gradual quality decay during autoregressive rollout, its degradation is far less severe. The Stage 3 results in Fig. 5(a) further show that using causal DMD as the initialization for the subsequent asymmetric DMD performs worse than using causal CD initialization. In summary, causal DMD suffers from substantially stronger exposure bias and is therefore unsuitable as a few-step initialization.

We further analyze why causal DMD can outperform CD in the ﬁrst few frames but then rapidly suffers from ampliﬁed exposure bias, with an intuitive explanation illustrated in Fig. 5(b). As noted in prior work [24], DMD optimizes a reverse KL objective, whereas CD optimizes a forward KL objective. Compared with the mode-covering behavior induced by forward KL, the mode-seeking behavior of reverse KL tends to concentrate probability mass around high-density modes, leading to a sharper and less diverse distribution. This contrast is illustrated by the broader causal-CD distributions and the sharper causal-DMD distributions in Fig. 5(b). Early in autoregressive rollout, this concentration reduces the probability assigned to low-quality regions and thus improves generation quality, explaining the strong quality of the ﬁrst few frames. However, such a concentrated distribution is also more sensitive to accumulated errors in autoregressive conditional generation.

Speciﬁcally, the arrows in Fig. 5 denote the shift of the conditional distribution caused by history drift, i.e., the deviation of the self-generated preﬁx from the ground-truth history. Under such drift, the conditional distributions induced by both mode-covering causal CD and mode-seeking causal DMD shift toward the poorquality region. For mode-seeking DMD, since the probability mass is highly concentrated, once the shifted mode moves into the poor-quality region, most generated samples tend to follow this erroneous mode, as shown by the shifted red DMD peak in Fig. 5(b). In contrast, mode-covering CD maintains a more dispersed distribution; even after the shift, a substantial portion of its probability mass can remain in the good-quality region, as shown by the broader shifted CD distribution in Fig. 5(b). Therefore, intuitively, mode-seeking DMD is more sensitive to accumulated history errors, causing exposure bias to rapidly amplify in later frames, whereas mode-covering CD suffers from exposure bias with much lower sensitivity.

##### 3.4.2. Distinction from Other AR Distillation Paradigms

APT2 [63] represents another line of work that is substantially different from the classical asymmetric DMDbased AR diffusion distillation paradigm. When initializing its GAN [64] generator, APT2 adopts teacherforcing CD, which is highly related to our causal CD initialization.

However, the two works signiﬁcantly differ in both motivation and technical design. First, Causal Forcing++ is motivated by reducing the cost of causal ODE initialization in Causal Forcing while preserving theoretical equivalence, whereas Causal Forcing is the ﬁrst work to theoretically show why such causal initialization is needed. In contrast, APT2 adopts teacher-forcing CD heuristically, without a corresponding theoretical motivation. Second, Causal Forcing++ remains within the asymmetric DMD framework of CausVid and Self Forcing: it uses a bidirectional diffusion model as the DMD teacher and supports temporal KV-cache inference. APT2, by contrast, is built as a GAN-based method, has no bidirectional-teacher formulation, does not use temporal KV cache, and instead concatenates noise with conditions. Built upon these distinct training algorithms and model architectures, this work open-sources the ﬁrst asymmetric-DMD-driven few-step AR model initialized by causal CD. Since APT2 is not open-source, we do not compare with it below.

### 4. Experiments

#### 4.1. Setup

Implementation details. Following Causal Forcing [20], we adopt Wan2.1-1.3B [3] as the base model from which the AR few-step model is derived. The model generates videos at a resolution of 480 × 832 with 81 frames, using frame-wise autoregressive generation. We follow the three-stage distillation pipeline of Causal Forcing, namely: Stage 1, teacher forcing AR diffusion training; Stage 2, few-step initialization; and Stage 3, asymmetric DMD with self rollout. For Stage 2 causal CD, we use the square norm, adopt 48 discretized timesteps, and employ the Euler solver. For the number of sampling steps, in Stage 3 we train 4-step (t = 1,0.9375,0.8333,0.625), 2-step (t = 1,0.8333), and 1-step settings. For settings with fewer than

- 4 steps, we follow ASD [50] and keep the generation of the ﬁrst latent frame at 4 steps, while using 2 steps or 1 step, respectively, for the subsequent 20 latent frames. Note that this trick is used only in Stage 3 and is not involved in the ﬁrst two stages. All teacher and student models used throughout the pipeline are based on Wan2.1-1.3B, except for the real score model in the Stage 3 DMD, which is based on Wan2.1-14B. For Stage 1,2, we utilize an 80K dataset including videos sampled from OpenVid [65]; for Stage 3, we use the VidProM [66] dataset. The numbers of training steps for the three stages are 20K, 5K, and 1K, respectively, and the batch size is 64. All other hyperparameters are kept the same as in Causal Forcing.

Evaluation. Following Causal Forcing, we adopt two benchmarks: VBench [60] and VisionReward [67]. For VBench, we report the overall metrics and additionally evaluate dynamic degree separately using the 100 prompts from Causal Forcing. For VisionReward, we also use the 100 prompts from Causal Forcing and report both the overall score and instruction-following performance. For readability, all metrics are multiplied by 100. All other evaluation details follow Causal Forcing. In addition, we report ﬁrst-frame latency and throughput. These efﬁciency metrics are measured on the single A800 GPU without the VAE-related time cost, rather than on H100 as in the Self Forcing and Causal Forcing papers.

#### 4.2. Results

Performance comparison with existing methods. We compare Causal Forcing++ with prior AR diffusion distillation methods, including CausVid [18], Self Forcing [19], and Causal Forcing [20], which are all chunkwise models with 4-step sampling.

As illustrated in Tab. 1, our frame-wise Causal Forcing++ (2-step generation) achieves the best VBench total score. Its Total score (84.14) and Quality score (84.89) surpasses all previous methods, while its

*Because we adopt the ASD trick [50], the ﬁrst-frame latency for 1-step, 2-step, and 4-step generation is identical, and is determined by the 4-step generation of the ﬁrst frame.

- Table 1: Comparisons with existing methods. Our method reduces latency by half and improves throughput, while achieving generation quality comparable to or even better than previous SOTA methods. Dynamic., Vision. and Instruct. denote Dynamic Degree, VisionReward and Instruction Following, respectively. Throughput and latency are measured in FPS and seconds on an A800.*

Model Throughput ↑ Latency ↓ Total↑ Quality↑ Semantic↑ Dynamic.↑ Vision.↑ Instruct.↑

CausVid [18] 10.4 0.60 81.33 83.98 70.72 62 5.741 12 Self Forcing [19] 10.4 0.60 83.74 84.48 80.77 57 5.820 48 Causal Forcing [20] 10.4 0.60 84.04 84.59 81.84 68 6.326 56

- Causal Forcing++ (1-step) 20.7 0.27 83.35 84.50 78.75 66 5.412 38
- Causal Forcing++ (2-step) 14.1 0.27 84.14 84.89 81.13 64 6.661 51 Causal Forcing++ (4-step) 8.69 0.27 84.10 84.94 80.75 71 6.798 47

Semantic score remains comparable to Causal Forcing. Notably, while maintaining overall generation quality comparable to, and in some aspects better than, prior state-of-the-art (SOTA) methods, our 2-step frame-wise Causal Forcing++ improves throughput by about 1.4Œ and reduces latency by 50%. Furthermore, Causal Forcing++ under 4-step generation attains even higher Quality and VisionReward scores, and achieves a Dynamic Degree beyond all previous SOTA methods, while still maintaining 50% lower ﬁrst-frame latency.

These quantitative results are consistent with the visualization in Fig. 6. CausVid and Self Forcing both exhibit weaker dynamics; moreover, CausVid shows over-saturation artifacts, while Self Forcing suffers from object inconsistency, as illustrated by the spurious horizontal structure growing from the building next to the waterfall in the right-hand demo. In contrast, under fewer-step frame-wise generation, Causal Forcing++ attains quality and dynamics comparable to those of Causal Forcing, and even shows advantages in aesthetic aspects such as color vividness and brightness.

Ablation studies. In this section, we provide a detailed method ablation. We experiment under the framewise 1, 2, and 4-step settings, respectively, and evaluate Self Forcing [19] ODE initialization, multi-step AR diffusion initialization, causal ODE initialization, causal DMD initialization, and our causal CD initialization. Note that the difference in the number of generation steps only applies to Stage 3, while Stage 2 remains identical across settings, with special note that the causal ODE is trained under the 4-step setting. In addition to the quality metrics in Sec. 4.1, we also list here the training time (measured in A800 GPU· hours, rounded to the nearest hundred) for Stage 2 (with ODE including the 80K data curation time) and the extra storage for the ODE paired data (measured in GiB). The quantitative results and visual comparisons are illustrated in Tab. 2 and Fig. 7, respectively, from which we draw the following conclusions.

Self Forcing ODE initialization performs consistently poorly in the frame-wise setting. Speciﬁcally, under the 2-step and 4-step settings, Self Forcing ODE initialization underperforms all other methods across all metrics. Its VBench Total score remains below 80, and its Dynamic Degree reaches at most 2. Beyond its poor quality, the ODE stage also costs 5,000 A800 GPU· hours and requires 1,500 GB of storage.

A strong few-step initialization is indispensable, especially in aggressive low-step settings. Using the multi-step AR diffusion model directly as initialization yields consistently weak performance. This issue is most severe in the 1-step setting, where the model nearly collapses in dynamics and instruction following, achieving only 0 Dynamic Degree, 1.101 VisionReward, and even a -14 Instruction Following score. Although the performance improves as the number of steps increases, it still remains clearly inferior even under 4-step generation, where its VisionReward is only 5.166, still far below all explicit few-step initialization methods. This shows that AR diffusion initialization is insufﬁcient, and that explicit few-step adaptation before asymmetric DMD is crucial, corroborating Sec. 3.1.

CausVid

Self Forcing

Causal Forcing

Causal Forcing ++

- （1-step）

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

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Causal Forcing ++

- （2-step）

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

Causal Forcing ++ （4-step）

- Figure 6: Performance comparison. Our Causal Forcing++ achieves quality and dynamics comparable to or even surpassing Causal Forcing, while outperforming CausVid and Self Forcing.

Causal CD is a stronger and more efﬁcient substitute for causal ODE. Causal CD matches or surpasses causal ODE across all step settings while dramatically reducing Stage 2 cost. In 1-step generation, it improves Total, Quality, and Dynamic Degree over causal ODE, with comparable VisionReward and Instruction Following. In the 2-step setting, it achieves the best overall performance, with the highest Total, Quality, and VisionReward scores, while maintaining comparable Semantic score. In the 4-step setting, causal CD again gives the strongest overall results. Meanwhile, causal ODE requires around 11,600 A800 GPU· hours and

- 1,900GiB extra storage for ODE-paired data curation and training, whereas causal CD requires only around
- 2,900 A800 GPU· hours and no extra storage. Thus, causal CD reduces Stage 2 time cost by roughly 4×, completely removes storage overhead, and serves as a better substitute for causal ODE initialization, consistent with Sec. 3.2.

Causal DMD is not an ideal replacement for few-step initialization. Although causal DMD improves substantially over AR diffusion initialization, it is weaker than causal CD and is usually also worse than causal ODE. For example, its VisionReward remains about 0.5 lower than that of causal CD initialization across all settings. This suggests that the causal DMD initialization is suboptimal, which is we discuss the underlying reason in the analysis below.

These ablation results are highly consistent with Sec. 3, and provide strong evidence for the validity and superiority of replacing causal ODE with causal CD in our Causal Forcing++.

### 5. Conclusion

In this paper, we identify the training inefﬁciency of existing AR diffusion distillation methods and propose Causal Forcing++, which replaces causal ODE distillation with causal consistency distillation. Our method substantially improves training efﬁciency and, for the ﬁrst time, achieves performance comparable to or better than prior SOTA methods under the frame-wise 2-step setting, reducing latency by 50% . Experiments

- Table 2: Ablation study. Causal CD is comparable to or even outperforms causal ODE, while dramatically reducing both time and storage cost. In contrast, Self Forcing ODE initialization, multi-step and causal DMD initialization perform worse.

Model Total↑ Quality↑ Semantic↑ Dynamic.↑ Vision.↑ Instruct.↑ Time (Stage 2)↓ Extra Storage ↓

- 1-Step Asymmetric DMD

- Self Forcing initialization 78.87 79.85 74.95 0 1.992 -12 5000 1500 AR diffusion initialization 80.54 80.97 78.84 0 1.101 -14 - 0 Causal ODE initialization 83.06 83.88 79.77 46 5.464 40 11600 1900

- Causal DMD initialization 82.34 83.50 77.71 62 4.868 20 2900 0

- Causal CD initialization 83.35 84.50 78.75 66 5.412 38 2900 0

2-Step Asymmetric DMD Self Forcing initialization 79.44 80.43 75.47 0 2.826 -14 5000 1500

- AR diffusion initialization 82.43 83.04 80.00 8 3.645 16 - 0

- Causal ODE initialization 83.77 84.42 81.19 57 6.224 46 11600 1900

Causal DMD initialization 83.73 84.56 80.39 69 6.108 38 2900 0 Causal CD initialization 84.14 84.89 81.13 64 6.661 51 2900 0

4-Step Asymmetric DMD Self Forcing initialization 79.82 80.58 76.76 2 3.675 -2 5000 1500 AR diffusion initialization 83.41 84.42 79.38 62 5.166 30 - 0

- Causal ODE initialization 83.78 84.90 79.28 75 6.435 42 11600 1900 Causal DMD initialization 83.49 84.84 78.11 67 6.309 36 2900 0 Causal CD initialization 84.10 84.94 80.75 71 6.798 47 2900 0

show that our method achieves strong performance, demonstrating its effectiveness.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Multi Step Init.

Causal ODE Init.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Causal CD Init.

Causal DMD Init.

- (a) Final results of 1-step asymmetric DMD under different initializations. AR diffusion initialization results in severe blur and almost no motion, causal ODE suffers from scene collapse, and causal DMD blurs the mouses legs into a single indistinguishable mass. In contrast, causal CD initialization preserves both strong dynamics and high visual quality. Init. denotes initialization.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Multi Step Init.

Causal ODE Init.

Causal CD Init.

Causal DMD Init.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

- (b) Final results of 2-step asymmetric DMD under different initializations. Multi-step init. blurs the bouquet, causal ODE init. turns the persons head black, and causal DMD init. introduces a bright spot in the ﬁrst frame and a boat in later frames. In contrast, causal CD init. maintains stable quality. Init. denotes initialization.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Multi Step Init.

Causal ODE Init.

Causal CD Init.

Causal DMD Init.

[Figure 122]

[Figure 123]

[Figure 124]

- (c) Final results of 4-step asymmetric DMD under different initializations. Both multi-step and causal DMD initializations produce poor visual quality. Causal ODE exhibits antler separation artifacts, whereas causal CD initialization yields high-quality results. Init. denotes initialization.

- Figure 7: Visual comparisons after asymmetric DMD with different initializations. Causal CD achieves results comparable to or even better than, causal ODE, whereas multi-step and causal DMD initializations perform worse.

### References

- [1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.
- [2] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.
- [3] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [4] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [5] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [6] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [7] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efﬁcient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [8] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.
- [9] Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownﬁeld, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Oliveira, Jack Parker-Holder, Frank Perbet, Guy Scully, Jeremy Shar, Stephen Spencer, Omer Tov, Ruben Villegas, Emma Wang, Jessica Yung, Cip Baetu, Jordi Berbel, David Bridson, Jake Bruce, Gavin Buttimore, Sarah Chakera, Bilva Chandra, Paul Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin, Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson, Dimple Vijaykumar, Luyu Wang, Piers Wingﬁeld, Nat Wong, Keyang Xu, Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, Aäron van den Oord, Inbar Mosseri, Adrian Bolton, Satinder Singh, and Tim Rocktäschel. Genie 3: A new frontier for world models. 2025.
- [10] Yubo Huang, Hailong Guo, Fangtai Wu, Shifeng Zhang, Shijie Huang, Qijun Gan, Lin Liu, Sirui Zhao, Enhong Chen, Jiaming Liu, et al. Live avatar: Streaming real-time audio-driven avatar generation with inﬁnite length. arXiv preprint arXiv:2512.04677, 2025.
- [11] Taekyung Ki, Sangwon Jang, Jaehyeong Jo, Jaehong Yoon, and Sung Ju Hwang. Avatar forcing: Real-time interactive head avatar generation for natural conversation. arXiv preprint arXiv:2601.00664, 2026.
- [12] Zhiyao Sun, Ziqiao Peng, Yifeng Ma, Yi Chen, Zhengguang Zhou, Zixiang Zhou, Guozhen Zhang, Youliang Zhang, Yuan Zhou, Qinglin Lu, et al. Streamavatar: Streaming diffusion models for real-time interactive human avatars. arXiv preprint arXiv:2512.22065, 2025.
- [13] Yao Feng, Chendong Xiang, Xinyi Mao, Hengkai Tan, Zuyue Zhang, Shuhe Huang, Kaiwen Zheng, Haitian Liu, Hang Su, and Jun Zhu. Vidarc: Embodied video diffusion model for closed-loop control. arXiv preprint arXiv:2512.17661, 2025.
- [14] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, and Joel Jang. World action models are zero-shot policies, 2026.

- [15] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal ﬂow matching for efﬁcient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.
- [16] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [17] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Inﬁnite-length ﬁlm generative model. arXiv preprint arXiv:2504.13074, 2025.
- [18] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025.
- [19] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [20] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.
- [21] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023.
- [22] Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency model. arXiv preprint arXiv:2405.18407, 2024.
- [23] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.
- [24] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025.
- [25] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaﬂow: One step is enough for high-quality diffusionbased text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023.
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [27] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [28] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectiﬁed ﬂow. arXiv preprint arXiv:2209.03003, 2022.
- [29] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [30] Min Zhao, Fan Bao, Chongxuan Li, and Jun Zhu. Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. Advances in Neural Information Processing Systems, 35:3609–3623, 2022.
- [31] Min Zhao, Hongzhou Zhu, Chendong Xiang, Kaiwen Zheng, Chongxuan Li, and Jun Zhu. Identifying and solving conditional image leakage in image-to-video diffusion model. Advances in Neural Information Processing Systems, 37:30300–30326, 2024.
- [32] Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: conditional control for one-shot text-driven video editing and beyond. Science China Information Sciences, 68(3):132107, 2025.
- [33] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riﬂex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.

- [34] Min Zhao, Hongzhou Zhu, Yingze Wang, Bokai Yan, Jintao Zhang, Guande He, Ling Yang, Chongxuan Li, and Jun Zhu. Ultravico: Breaking extrapolation limits in video diffusion transformers. arXiv preprint arXiv:2511.20123, 2025.
- [35] Min Zhao, Bokai Yan, Xue Yang, Hongzhou Zhu, Jintao Zhang, Shilong Liu, Chongxuan Li, and Jun Zhu. Ultraimage: Rethinking resolution extrapolation in image diffusion transformers. arXiv preprint arXiv:2512.04504, 2025.
- [36] Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: o(n log n) sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025.
- [37] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.
- [38] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-tovideo generation via transformers. arXiv preprint arXiv:2205.15868, 2022.
- [39] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. Nüwa: Visual synthesis pre-training for neural visual world creation. In European conference on computer vision, pages 720–736. Springer, 2022.
- [40] Dirk Weissenborn, Oscar Täckström, and Jakob Uszkoreit. Scaling autoregressive video models. arXiv preprint arXiv:1906.02634, 2019.
- [41] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.
- [42] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.
- [43] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.
- [44] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [45] Xiaofei Wu, Guozhen Zhang, Zhiyong Xu, Yuan Zhou, Qinglin Lu, and Xuming He. Pack and force your memory: Long-form and consistent video generation. arXiv preprint arXiv:2510.01784, 2025.
- [46] Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, and Dahua Lin. End-toend training for autoregressive video diffusion via self-resampling. arXiv preprint arXiv:2512.15702, 2025.
- [47] Ryan Po, Eric Ryan Chan, Changan Chen, and Gordon Wetzstein. Bagger: Backwards aggregation for mitigating drift in autoregressive video diffusion models. arXiv preprint arXiv:2512.12080, 2025.
- [48] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025.
- [49] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316, 2025.
- [50] Yongqi Yang, Huayang Huang, Xu Peng, Xiaobin Hu, Donghao Luo, Jiangning Zhang, Chengjie Wang, and Yu Wu. Towards one-step causal video generation via adversarial self-distillation. arXiv preprint arXiv:2511.01419, 2025.
- [51] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efﬁcient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025.

- [52] Zehan Wang, Tengfei Wang, Haiyu Zhang, Xuhui Zuo, Junta Wu, Haoyuan Wang, Wenqiang Sun, Zhenwei Wang, Chenjie Cao, Hengshuang Zhao, et al. Worldcompass: Reinforcement learning for long-horizon world models. arXiv preprint arXiv:2602.09022, 2026.
- [53] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.
- [54] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.
- [55] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.
- [56] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Proliﬁcdreamer: Highﬁdelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36:8406–8441, 2023.
- [57] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36:76525–76546, 2023.
- [58] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.
- [59] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.
- [60] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [61] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, pages 15984–16009. Curran Associates, Inc., 2025.
- [62] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.
- [63] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. Advances in Neural Information Processing Systems, 38:41061–41086, 2026.
- [64] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.
- [65] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.
- [66] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, 37:65618–65642, 2024.
- [67] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

