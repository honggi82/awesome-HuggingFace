## LIVE: Long-horizon Interactive Video World Modeling

Junchao Huang1,2,3 Ziyang Ye1 Xinting Hu4 Tianyu He3,† Guiyu Zhang1 Shaoshuai Shi5 Jiang Bian3 Li Jiang1,2,‡

1The Chinese University of Hong Kong, Shenzhen 2Shenzhen Loop Area Institute 3Microsoft Research 4The University of Hong Kong 5Voyager Research, Didi Chuxing Project Page: https://junchao-cs.github.io/LIVE-demo/

# arXiv:2602.03747v1[cs.CV]3Feb2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Baseline

###### FID↓

|TF DF GF<br><br>DFoT Ours|
|---|

90

…

…

55

###### ComparisonMoreScenarios

45

1#!

256!"

[Figure 9]

35

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Ours

25

15

5

32 64 128 200

Length of Rollouts

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

Re10K

[Figure 36]

1#!

256!"

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Minecraft UE Engine

1#! 256!" 256!"

1#!

[Figure 51]

[Figure 52]

Figure 1. LIVE achieves bounded error accumulation for stable long-horizon video world modeling. Top: Qualitative comparison with baselines and FID curves showing LIVE maintains stable quality while other methods degrade as rollout length increases. Bottom: Applications in real-world (RealEstate10K) and gaming environments (Minecraft, UE Engine).

#### Abstract

ror accumulation via a novel cycle-consistency objective, thereby eliminating the need for teacher-based distillation. Specifically, LIVE first performs a forward rollout from ground-truth frames and then applies a reverse generation process to reconstruct the initial state. The diffusion loss is subsequently computed on the reconstructed terminal state, providing an explicit constraint on long-horizon error propagation. Moreover, we provide an unified view that encompasses different approaches and introduce progressive training curriculum to stabilize training. Experiments demonstrate that LIVE achieves state-of-the-art performance on long-horizon benchmarks, generating stable, high-quality videos far beyond training rollout lengths.

Autoregressive video world models predict future visual observations conditioned on actions. While effective over short horizons, these models often struggle with longhorizon generation, as small prediction errors accumulate over time. Prior methods alleviate this by introducing pre-trained teacher models and sequence-level distribution matching, which incur additional computational cost and fail to prevent error propagation beyond the training horizon. In this work, we propose LIVE, a Long-horizon Interactive Video world modEl that enforces bounded er-

†Team lead. ‡Corresponding authors.

!89:9

- J$)F/'
- K$F0L

&'' ()*%"

,-"./ ,)/0"10

>?@2 B3@@3CD

K".ML"$ !)--)F0

<*%0$. Matching Loss

! "# :399(43#);#44

EF-0* − %0"H <"/)*%*/I

2345647

!"#$%&

!)--)F0

!"#"$%" ⇌

- P$.O"%
- Q)$ -)%%

…

…

&'' ()*%" …

…

…

… &'' ()*%" …

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

!;

()*%" .0 #.$N*/I 0*O"%0"H%

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

:399(43#) ;#44

[Figure 64]

:399(43#) ;#44

+ ,#

$ %# & '()*

Teacher DiT

'"#()* +"(&,

&''

!89:9

()*%" !"#$%&

###### DiT … …

[Figure 65]

,-"./ ,)/0"10

7?R3STSRU

+-./,-" 0#11#(&

Student DiT

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

DiT

>?@2 B3@@3CD

:399(43#) ;#44

! "#

###### <*%0$. Matching Loss

[Figure 71]

0#11#(&

2345647

!"#"$%" ⇌

GT1

GT2 GT3 GT4 GT5

[Figure 72]

|… &''<br><br>[Figure 73]<br><br>[Figure 74]<br><br>DiT<br><br>[Figure 75]|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- 8".$-4
- 9#" 1#44

… &'' ()*%" …

()*%" …

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

…

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

…

…

2#34- .& 5."63)7 &3$-4&-%4

!;

DiT

[Figure 90]

:399(43#) ;#44

:399(43#) ;#44

+ ,#

$ %# & '()*

|Figure 2. inference<br><br>with<br><br>[Figure 91]|
|---|

|paradigms. Teacher noise but fails LIVE|
|---|

Fig Comparison of autoregressive training

para er Forcing (TF) uses ground truth context during training, causing train-

- inf mismatch. Diffusion Forcing (DF) injects n s to model real rollout errors. Self-Forcing (SF) employs sequence-level distillation unbounded error accumulation. Our performs forward rollout then reverse recovery with frame-level diffusion loss, bounding errors through the cycle-consistency objective.

Prompt: GT1

Rollout 2 Rollout 3 Rollout 4 Rollout 5

GT1

GT2 GT3 GT4 GT5

sue by injecting stochastic noise into the conditioning context during training, thereby exposing the model to imperfect inputs. While this provides some degree of robustness for short sequences, the approach remains ineffective for long-horizon generation, as there remains a substantial distributional gap between noised ground-truth data and genuine model rollouts with accumulated errors.

|[Figure 92]<br><br>[Figure 93]|
|---|

|[Figure 94]|
|---|

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

Prompt: GT1

Rollout 2 Rollout 3 Rollout 4 Rollout 5

[Figure 106]

[Figure 107]

[Figure 108]

|[Figure 109]<br><br>[Figure 110]|
|---|

|[Figure 111]|
|---|

To further mitigate the train-inference gap, Self-Forcing (SF) [28] has proposed training on rollouts generated by the model itself and distilling knowledge from a pre-trained teacher via holistic sequence-level distribution matching. While effective, this paradigm suffers from several limitations. First, the reliance on pre-trained, interaction-capable teacher models incurs substantial computational overhead, particularly in domain-specific settings. Second, knowledge distillation inherently constrains the student model by the teacher’s capacity and can induce mode-seeking behavior that degrades output diversity. Third, SF applies distribution matching at the sequence level without explicitly bounding error accumulation, limiting its ability to control long-horizon error propagation. As a result, the model is only exposed to errors within a fixed training rollout length, and inference beyond this horizon leads to unseen error patterns and potential catastrophic collapse.

- Figure 3. Rollout from GT produces semantically diverse content, making direct supervision infeasible. LIVE addresses this by requiring the model to generate back toward the original GT, enabling valid supervision through the cycle-consistency objective.

#### 1. Introduction

Video world models aim to learn action-conditioned future video predictions for interactive agents, based on past observations and control inputs such as camera poses and keyboard commands. Different from bidirectional video diffusion models that generate the entire video frames at once [15, 33, 37, 48], effective world models require finegrained interactivity and real-time inference. To achieve this goal, approaches such as Teacher Forcing (TF) [14, 26, 29] and Diffusion Forcing (DF) [7] have introduced causal attention mechanisms into video diffusion models, enabling autoregressive video generation with real-time interactivity.

To address these limitations, we propose LIVE, a Long-horizon Interactive Video world modEl that enforces bounded error accumulation via a novel cycle-consistency objective, thereby eliminating the reliance on teacher-based distillation. Instead of matching full sequence distributions [28], LIVE performs a forward rollout from groundtruth frames followed by a reverse generation process to reconstruct the initial state, on which the diffusion loss is computed. This formulation explicitly enforces cycle consistency: training the model to map its own imperfect rollouts back to the ground-truth manifold. Crucially, unlike sequence-level objectives that permit unbounded drift,

Despite its empirical success [8, 51], autoregressive video world modeling is fundamentally limited by the temporal accumulation of generation errors. This issue arises from exposure bias, where the model is trained on groundtruth frames but must condition on its own predictions at inference time, leading to compounding distributional shift over long horizons. DF [7, 39] attempts to mitigate this is-

the proposed design maintains distributional alignment between generated rollouts and supervision targets. By train-

- ing with fixed-length windows while explicitly modeling error accumulation, LIVE learns to operate within a controlled error bound, enabling stable generalization to longhorizon generation at inference time.

In addition, we present a unified view that encompasses TF, DF, and the proposed LIVE. Under this unified view, TF and DF emerge as special cases of LIVE by adjusting the proportion of ground-truth conditioning. Motivated by this observation, we introduce a progressive training curriculum that explicitly controls error tolerance by parameterizing the ratio of ground-truth frames to model-generated rollouts within each training window. This curriculum facilitates stable optimization while preserving high-quality generation through end-to-end diffusion training. In summary, our contributions are threefold:

- • We propose LIVE, a long-horizon interactive video world model that enforces bounded error accumulation via a cycle-consistency objective, eliminating the need for teacher-based distillation.
- • We present a unified view of TF, DF, and LIVE, and derive a progressive training curriculum that controls error tolerance by adjusting the ratio of ground-truth to rollout frames, enabling stable end-to-end diffusion training.
- • We demonstrate state-of-the-art performance on longhorizon interactive video benchmarks, with robust generalization to sequences far beyond the training horizon.

#### 2. Related Work

Video Diffusion Models. Early video diffusion methods extended image diffusion into temporal domains using UNet-based architectures [2–4, 18, 25]. The introduction of Diffusion Transformers (DiT) [20, 35] enabled better modeling of global spatiotemporal dependencies, leading to large-scale models like Sora [33], Seaweed [38], HunyuanVideo [31], and Wan [42]. These bidirectional approaches [1, 5] employ full-sequence attention where all frames interact simultaneously, achieving impressive temporal consistency and motion quality. However, they are constrained to fixed-length generation and lack frame-level interaction control, while facing computational complexity that scales quadratically with sequence length, making them unsuitable for real-time interactive world modeling.

Autoregressive Video Generation. Autoregressive methods synthesize videos sequentially by conditioning on preceding context [21, 24, 32, 40, 46]. Approaches include discrete token-based autoregression [17, 30, 44] and diffusionbased frameworks [8, 16]. This paradigm naturally supports interactive world modeling where environments are simulated step-by-step [6, 13, 23, 34, 41, 52]. Several works adopt causal attention with sliding windows for real-time

generation [9, 11, 27], while facing error accumulation challenges during long-horizon inference.

Mitigating Exposure Bias. Teacher Forcing [14, 29, 51] conditions on ground truth during training but causes exposure bias at inference when models encounter their own imperfect rollouts. Diffusion Forcing [7, 8, 16, 39, 49] injects noise into ground truth context during training to approximate rollout distributions, yet noised ground truth still fundamentally differs from actual rollouts. Self-Forcing [28] and its extensions [10, 47] align the model’s rollout distribution with that of a pre-trained bidirectional teacher during training, which slows down error accumulation but still suffers from degradation beyond training rollout lengths. Moreover, the reliance on pre-trained teachers complicates extension to interactive video generation models. Concurrent work [36] constructs corrective trajectories from the model’s rollouts to teach it to recover from its mistakes. Another approach [19] simulates rollouts via resampling ground truth, yet this still differs from genuine rollouts.

#### 3. Preliminaries

##### 3.1. Interactive Video World Modeling

We consider video world modeling as learning the conditional distribution p(x1:T|c1:T), where x1:T = (x1,...,xT) denotes a sequence of T video frames and c1:T = (c1,...,cT) represents conditioning information for each frame (e.g., camera poses, actions).

Video diffusion models learn to denoise Gaussian noise through an iterative process, where a forward diffusion process gradually adds noise to the data:

,βiI), (1) and a reverse denoising process learns to predict the noise:

; 1 − βixt

i|xt

) = N(xt

q(xt

i−1

i−1

i

ϵθ(xkt ,t,ck) ≈ ϵ, (2)

where t ∈ [t1,...,tN] denotes the diffusion timestep for frame k and ϵ ∼ N(0,I) is the Gaussian noise.

In video world modeling, the model generates frames sequentially conditioned on previous frames:

p(x1:T|c1:T) =

T

p(xk|x<k,c≤k), (3)

k=1

where each frame xk is generated conditioned on the context window x<k = (x1,...,xk−1) and corresponding conditions c≤k = (c1,...,ck). For interactive world models requiring real-time inference, we employ a sliding window approach where only the most recent K frames are used as context:

p(xk|xk−K:k−1,ck−K:k). (4)

9

!"!%% !"!$$ !!"& !!#' !!"! !!"" !!## !!#(

|<br><br><br><br>;&|
|---|

|<br><br>;&|
|---|

|;!|
|---|

|;<"|
|---|

|;<#|
|---|

|;#|
|---|

|;"|
|---|

|;!|
|---|

!"!%%

867567

!"!$$

!!"& !!#'

[Figure 121]

[Figure 122]

Diffusion Transformer

Diffusion Transformer

### ❄ !

:

!!"! !!"" !!##

Casual DiT

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

|[Figure 129]<br><br>;&|
|---|

|;!|
|---|

|[Figure 130]<br><br>;<$##|
|---|

|[Figure 131]<br><br>;<$"$|
|---|

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

34567

=>?@A ;$&%/$&

;$!!/$"

=>?@A

!!#(

[Figure 137]

[Figure 138]

9

|[Figure 139]<br><br>[Figure 140]<br><br>;&|
|---|

|[Figure 141]<br><br>;&|
|---|

|;!|
|---|

|;<"|
|---|

|;<#|
|---|

|;#|
|---|

|;"|
|---|

|;!|
|---|

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

!"!%% !"!$$ !!"& !!#' !!"! !!"" !!## !!#(

867567

!"!%%

!"!$$

!!"& !!#'

+noise

⇌

Diffusion Transformer Diffusion Transformer

[Figure 146]

[Figure 147]

[Figure 148]

### ❄ !

[Figure 149]

:

;$!" ;$!!

[Figure 150]

;$&& ;$&%

[Figure 151]

[Figure 152]

!!"! !!"" !!##

[Figure 153]

Casual DiT

|[Figure 154]<br><br>;&|
|---|

|;!|
|---|

noise noise

[Figure 155]

|[Figure 156]<br><br>;<$##|
|---|

|[Figure 157]<br><br>;<$"$|
|---|

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

34567

!!#(

(a) Forward Rollout

(b) Cycle-consistency Training

- Figure 4. LIVE training pipeline. Forward rollout (Left, frozen): Given p prompt frames xi, the model generates the remaining T − p frames x˜j via causal attention. Cycle-consistency objective (Right, trainable): The rollout is reversed and used as context to recover the original prompt frames via frame-level diffusion loss, employing reverse attention (right mask, shown for p = 2).

#### 4. Method

##### 3.2. Training Paradigms for AR Generation

Existing autoregressive video diffusion models typically employ one of three training strategies (Figure 2 (a)-(c)):

We introduce LIVE, a framework that enforces bounded error accumulation via a cycle-consistency constraint. Specifically, LIVE performs a forward rollout from ground-truth (GT) frames followed by a reverse generation process to reconstruct the initial state, on which the diffusion loss is computed. This formulation explicitly enforces cycle consistency by training the model to map its own imperfect rollouts back to the GT manifold.

Teacher Forcing (TF). During training, the model predicts noise conditioned on ground truth frames within a sliding window:

K

,x<k,ti,c≤k) 2 . (5)

ϵk − ϵθ(xkt

###### LTF = E

i

x1:K, ϵ,ti

k=1

##### 4.1. Bounded Error Accumulation

ϵk is the noised frame k at timestep ti, with ti independently sampled for each frame from the noise schedule [t1,...,tN], and K denotes the context window length. This creates a train-inference discrepancy: at inference, the model must condition on its own imperfect rollouts rather than ground truth.

where xkt

xk+σt

= αt

Consider an autoregressive video diffusion model that generates frames sequentially: p(x1:T|c1:T) =

i

i

i

T k=1 pθ(xk|x<k,c≤k), where x<k = (x1,...,xk−1)

denotes the context frames and c1:T represents conditioning information (e.g., camera poses, actions).

Problem Setup. During autoregressive generation with rollouts, we observe a general tendency toward quality degradation in expectation:

Diffusion Forcing (DF). To bridge this gap, DF injects noise into the conditioning context during training:

K

E[D(xk,x˜k)] ≲ E[D(xk+1,x˜k+1)], ∀k ∈ [1,T − 1],

,xˆ<k,ti,c≤k) 2 , (6)

ϵk − ϵθ(xkt

###### LDF = E

(8) where D(xk,x˜k) measures perceptual quality (e.g., FVD, FID). While this error accumulation pattern is empirically well-established, directly supervising rollouts to reduce D(xk,x˜k) faces a fundamental obstacle: rollouts naturally produce semantically diverse content that diverges from GT trajectories (Figure 3). Since x˜k and xk represent different but equally valid future states, computing diffusion loss between them is infeasible. This limitation hinders extending SF to efficient parallel diffusion supervision and increases its dependence on pretrained teacher models.

i

x1:K, ϵ,ti

k=1

ϵj represents noisy context frame j with independently sampled timestep ti, and ϵj ∼ N(0,I). However, the distribution of noised ground truth differs from genuine model rollouts with accumulated errors.

where xˆj = αt

xj + σt

i

i

Self-Forcing (SF). SF addresses this through knowledge distillation, where a student model learns from its own rollouts under the supervision of a teacher:

LSF = DKL pteacher(˜x1:T)∥pstudent(˜x1:T) . (7)

Cycle-consistency Objective. To address the above challenges, LIVE introduces a cycle-consistency objective that enables valid frame-level supervision without requiring distributional alignment between rollouts and GT. The key insight is: instead of supervising rollouts x˜p+1:T (where p de-

However, sequence-level distribution matching fails to constrain error accumulation within bounded ranges, leading to quality degradation that prevents generalization beyond training rollout lengths.

Algorithm 1 LIVE Training Pipeline Require: Window length T, minimum pmin

- 1: for each epoch do
- 2: Pre-training: p ← T
- 3: Post-training: decrease p gradually
- 4: for each batch (x1:T,c1:T) ∈ D do
- 5: x˜p+1:T ∼ pθ(xp+1:T|x1:p,c1:T) {Eq. 9}
- 6: x˜p+1:T,rev ← (˜xT,...,x˜p+1) {Eq. 10}
- 7: crev ← (cT,...,c1) {Eq. 10}
- 8: Inject noise: x˜k,ϵ ← αtx˜k + σtηk {Eq. 11}
- 9: Compute LLIVE {Eq. 13}
- 10: θ ← θ − α∇θLLIVE
- 11: end for
- 12: end for

notes the number of prompt frames used to initiate the rollout) directly against GT, we require them to be recoverable - the model must be able to reverse-generate the original GT prompt frames from the rollouts by reversing camera poses/actions. This creates a valid training signal while accommodating distributional diversity. For a video sequence x1:T:

- Step 1 (Forward Rollout): Given a training window of T frames with known camera/action conditions, we use the first p frames as prompt frames and generate the remaining T − p frames with gradients disabled. Unlike inference which requires frame-by-frame interaction, during training we can efficiently generate all T −p frames simultaneously (initialized from pure noise) since we have access to all future camera/action conditions. This uses the same causal attention mask as inference (Figure 4), ensuring traininginference consistency while dramatically improving efficiency:

x˜p+1:T ∼ pθ(xp+1:T|x1:p,c1:T). (9)

- Step 2 (Reverse Generation): Reverse the rollout temporally and reverse camera/action conditions, then attempt to recover the original p prompt frames. Since forward rollouts satisfy D(˜xk,xk) ≤ D(˜xk+1,xk+1) (quality degrades monotonically), after reversal the context quality improves monotonically. Without intervention, the model could recover x1 by attending primarily to the highest-quality context frame x˜2,rev, trivially satisfying recoverability without constraining forward errors. To prevent this shortcut, we first reverse the rollout:

x˜p+1:T,rev ← (˜xT,...,x˜p+1), crev ← (cT,...,c1), (10)

then inject random noise per frame. For each k ∈ [p+1,T], sample t ∼ U([t1,...,tN]) and ηk ∼ N(0,I), then:

x˜k,ϵ ← αtx˜k + σtηk, (11) and finally recover the original prompts:

xˆ1:p ∼ pθ(x1:p| x˜p+1:T,rev, ϵ, crev). (12)

Step 3 (Frame-Level Supervision): To enable efficient parallel training like TF/DF, we extend the p-frame supervision to the full window length T by repeating the p GT frames and applying different noise timesteps to each position. This allows computing noise prediction loss on all T frames in parallel:

###### LLIVE = E

x1:T ∼pdata t∼U([t1,...,tN]) ϵk∼N(0,I)

1 T

where each ϵkθ is predicted as:

T

k=1

ϵk − ϵkθ 2 , (13)

###### ϵkθ = ϵθ(xgtt (k),x˜<k,rev,ϵ,t,c≤k), (14)

with t independently sampled for each frame k from the noise schedule, xgt(k) denoting the GT frame (repeated from the p prompts), and x˜<k,rev,ϵ representing the reversed rollout context with injected noise. Specifically, x˜<k,rev,ϵ consists of the subset of frames from x˜p+1:T,rev,ϵ in Eq. 12 preceding position k.

Implicit Error Bounding. The cycle-consistency objective creates an implicit incentive to bound forward distortion through the recovery objective. Define Dctx = D(xk,x˜k) as the distortion between rollout and GT at frame k, and Drec = p1 pk=1 D(xk,xˆk) as the average recovery distortion over the p prompt frames. The training objective minimizes Drec, which encourages:

- (1) Maintaining forward distortion D(xk,x˜k) within bounded range to enable recovery from rollout context;
- (2) Optimizing ϵθ to recover GT frames from imperfect rollout context, directly reducing Drec via gradient descent.

Consequently, gradient optimization learns to maintain D(xk,x˜k) within a bounded range, preventing the monotonic degradation that plagues AR diffusion inference.

##### 4.2. Progressive Training Curriculum

Unified Training Objective. LIVE unifies existing training paradigms by controlling the GT ratio p ∈ [1,T], encompassing: (1) Teacher Forcing (p = T, perfect GT context x<k); (2) Diffusion Forcing (p = T, noisy GT context xˆ<k = x<k+ϵ); (3) LIVE (p < T, imperfect rollout context x˜<k with accumulated errors). By controlling p, our framework supports both pre-training and post-training: during pre-training, the model uses p = T since it has not yet learned to generate rollouts; during post-training, as shown in Figure 6, we progressively decrease p to adapt the model to increasing error levels.

Error Tolerance. When p is large, the context consists primarily of GT frames with small distortions, making recovery relatively easy; as p decreases, more rollout frames enter the context, accumulating larger errors and making recovery increasingly difficult. Through gradually exposing the

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

PSNR↑ FID

↓ PSNR↑ FID↓

Validation Steps (0~200 frames)

Validation Steps (0~128 frames)

- Figure 5. Post-training performance from a converged DF checkpoint. Continued DF training stagnates with oscillating metrics, while LIVE achieves substantial improvements that amplify at longer horizons. LIVE converges to comparable FID across 128-frame and 200frame generation, demonstrating uniform quality regardless of rollout length.

9

!!%% !!$$ !!"& !!#' !!"! !!"" !!## !!#(

:

!!%! !!$" !!"#

!!$$

!!%%

!!"& !!#'

!!#(

9

!"!%% !"!$$ !!"& !!#' !!"! !!"" !!## !!#(

9

!"!%% !"!$$ !"!"& !!#' !!#! !!#" !!## !!#(

:

!!"! !!"" !!##

!"!$$

!"!%%

!!"& !!#'

!!#(

:

!!#! !!#" !!##

!"!$$

!"!%%

!"!"& !!#'

!!#(

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

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

! = # ! = $ ! = %

FGH64I JG67K

LHMMH67

NGOPQR SHG MHRR

THURQ O7 VOGWU4X 7UPQR7Q5R

[Figure 220]

[Figure 221]

- Figure 6. Progressive training curriculum by increasing rollout ratio. From left to right, as p decreases, more generated frames enter the context, increasing the model’s error tolerance while maintaining recoverability through the cycle-consistency objective.

domly select one video per scene (12 videos total) as the test set. (3) Minecraft: We train on the WorldMem [45] dataset and collect 300 video-action pairs from MineDojo [12] for evaluation, testing long-horizon generation in interactive environments. We compare LIVE against multiple baselines including CameraCtrl [22], DFoT [39], GF (Geometry Forcing) [43], and NFD-TF/DF (Teacher Forcing/Diffusion Forcing) [9], assessing generation quality using PSNR, SSIM, LPIPS, and FID metrics. We focus our comparison on methods without interactive teacher model distillation. Training large-scale bidirectional teacher models [28] remains important future work beyond our current computational budget.

model to harder recovery tasks, this strengthens its ability to recover from imperfect contexts (Error Tolerance). This enhanced capability, in turn, produces better rollouts with reduced errors, enabling robust long-horizon generation.

#### 5. Experiments

Implementation Details. All experiments are conducted on a cluster of 32 NVIDIA H100 GPUs with a batch size of 64. Our model architecture follows the NFD [9] 774M configuration. For RealEstate10K, we train from scratch following DFoT [39] settings at 256 × 256 resolution with a frame skip of 2. For UE Engine Videos datasets [50], we initialize from RealEstate10K pre-trained weights and apply the same frame skip of 2. Our model use a fixed context window of 32 frames during both training and evaluation. Additional details are provided in Appendix 7.1.

Datasets and Baselines. We evaluate on three diverse benchmarks: (1) RealEstate10K: A large-scale dataset of real estate videos featuring diverse camera motions. We report results on the complete test set. (2) UE Engine Videos: Following Context-as-Memory [50], we use their dataset containing 100 videos of 7,601 frames across 12 scenes with camera pose annotations, collected from realistic game engine environments (UE engine). We ran-

##### 5.1. Main Results

Error Accumulation Analysis. Figure 1 demonstrates LIVE’s core advantage. Training models on RealEstate10K with TF (Teacher Forcing), DF (Diffusion Forcing), DFoT [39], GF (Geometry Forcing) [43], and LIVE (TF, DF, and LIVE use the same model architecture [9]), we evaluate FID at 32, 64, 128, and 200 frames. LIVE maintains stable FID around 10 across all lengths, while all baselines degrade dramatically beyond 64 frames, validating that our cycle-consistency objective successfully bounds error accumulation.

Figure 5 shows post-training from a converged DF checkpoint. Continued DF training stagnates with oscillating metrics, while LIVE achieves substantial gains that amplify at longer sequences. Critically, LIVE converges to comparable FID for both 128-frame and 200-frame generation, maintaining uniform quality across rollout horizons.

Quantitative Results. Tables 1 and 2 show that LIVE achieves substantial improvements over all baselines across three benchmarks, with particularly large gains at longer rollout lengths. Our method shares identical architecture and inference procedures with NFD [9], isolating training strategy as the sole differentiator. While DF improves upon TF by injecting noise during training, it remains insufficient for long-horizon generation since noised ground truth fails

- Table 1. RealEstate10K full test set results across different rollout lengths. LIVE achieves state-of-the-art performance, with particularly large gains at longer sequences demonstrating superior long-horizon generation capability.

Method

Realtime

0∼64 frames 0∼128 frames 0∼200 frames ≥256 frames PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ CameraCtrl [22] × 14.09 0.3829 0.4366 11.69 0.5224 0.3651 10.25 0.6115 0.3181 9.48 0.6585 0.2886

DFoT [39] × 15.65 0.3053 0.4989 12.55 0.4601 0.3936 10.86 0.5613 0.3287 10.02 0.6128 0.2921 GF [43] × 16.37 0.2450 0.5567 12.69 0.4190 0.4534 10.59 0.5400 0.3969 9.91 0.5936 0.3796

NFD-TF [9] ✓ 16.87 0.2571 0.5503 13.59 0.4302 0.4448 11.63 0.5526 0.3724 10.58 0.6222 0.3281 NFD-DF [9] ✓ 16.59 0.2558 0.5723 13.82 0.3922 0.5015 12.21 0.4956 0.4598 11.51 0.5506 0.4397

LIVE ✓ 18.11 0.2215 0.5810 15.91 0.3298 0.5096 14.57 0.4163 0.4630 13.89 0.4682 0.4400

- Table 2. Results on interactive game environments. LIVE achieves consistent improvements over baselines on both realistic game engine videos (UE Engine) and interactive gameplay (Minecraft), demonstrating strong performance for interactive world modeling.

Method

0∼64 frames 0∼128 frames 0∼256 frames ≥400 frames PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ UE Engine (Realistic Game Engine)

NFD-TF [9] 17.16 0.3387 0.4953 14.95 0.4597 0.4245 12.97 0.5702 0.3625 11.80 0.6318 0.3286 NFD-DF [9] 17.15 0.3357 0.5062 14.71 0.4586 0.4441 12.27 0.5799 0.3956 11.02 0.6456 0.3760

LIVE 17.83 0.3145 0.5204 15.85 0.4210 0.4600 14.04 0.5214 0.4085 12.96 0.5794 0.3834

Method

0∼32 frames 0∼64 frames 0∼128 frames 0∼200 frames PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ Minecraft (Interactive Gameplay)

NFD-TF [9] 16.09 0.3474 0.6224 14.62 0.4067 0.5930 13.06 0.4781 0.5560 12.10 0.5255 0.5311 NFD-DF [9] 17.39 0.2888 0.6401 15.54 0.3586 0.6036 13.52 0.4469 0.5594 12.34 0.5091 0.5332

LIVE 17.87 0.2698 0.6558 16.31 0.3271 0.6291 14.90 0.3877 0.6037 14.02 0.4299 0.5885

- Table 3. Ablation studies on RealEstate10K test set evaluating the impact of key components in LIVE.

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

TFDFGTOurs

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

0∼64 frames 0∼200 frames PSNR ↑ LPIPS ↓ SSIM ↑ PSNR ↑ LPIPS ↓ SSIM ↑ Effect of Cycle-consistency Objective

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

Variant

w/o Cycle 13.99 0.4041 0.4597 11.18 0.6024 0.3564 Effect of Context Noise Strategy

No Noise 17.76 0.2310 0.5752 13.83 0.4573 0.4487 Fixed Noise 17.48 0.2392 0.5551 14.09 0.4444 0.4508 Effect of Progressive Training Curriculum Fixed p = 1 16.78 0.2747 0.5265 13.58 0.4800 0.4279

1!" 256"#

[Figure 242]

Figure 7. Qualitative comparison on UE Engine dataset. We compare models with identical architecture trained using Teacher Forcing (TF), Diffusion Forcing (DF), and LIVE.

###### LIVE 18.11 0.2215 0.5810 14.57 0.4163 0.4630

to match the distribution of genuine rollouts with accumulated errors. LIVE addresses this limitation by training directly on imperfect rollouts with the cycle-consistency objective, achieving bounded error accumulation through endto-end diffusion optimization.

Qualitative Results. Figures 7 and 8 present qualitative comparisons on UE Engine and RealEstate10K datasets.

On UE Engine, we compare models with identical architecture trained using TF, DF, and LIVE, demonstrating LIVE’s superior generation quality. On RealEstate10K, our method maintains consistent visual quality over extended rollouts across both indoor and outdoor scenes, while competing methods exhibit noticeable degradation. Full videos and additional examples are provided in Appendix 7.2.

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

DFoTGFDFTFOurs

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

1!" 256"# 1!" 256"#

[Figure 293]

[Figure 294]

Figure 8. Qualitative comparison on RealEstate10K dataset. We showcase indoor and outdoor scenes comparing various methods. LIVE demonstrates stable visual quality during rollouts. Full videos and additional examples are provided in Appendix 7.2.

##### 5.2. Ablation Studies

ting p = 1 throughout post-training underperforms our progressive curriculum that gradually decreases p from T to pmin. Abruptly exposing the model to the maximum rollout length creates an overly difficult task before sufficient error tolerance develops. Progressive rollout extension allows gradual capability building, starting from easy recovery with mostly GT context, then progressively increasing the rollout proportion to expose harder error patterns. This enhanced recovery capability produces better rollouts, ultimately enabling the model’s error tolerance to converge smoothly toward its recovery capacity.

We conduct comprehensive ablation studies on the RealEstate10K test set to validate the key design choices in LIVE. Results are summarized in Table 3.

Effect of Cycle-consistency Objective. Removing the reverse generation step leads to substantial performance degradation. This validates our core hypothesis illustrated in Figure 3: direct supervision on forward rollouts is infeasible due to semantic divergence between rollouts and ground truth. The cycle-consistency objective addresses this by requiring the model to generate back toward the original GT, creating a valid training signal that accommodates distributional diversity while constraining error accumulation within recoverable limits.

#### 6. Conclusion

In this work, we introduce LIVE, a long-horizon interactive video world model that addresses the fundamental challenge of error accumulation in autoregressive generation. By enforcing a cycle-consistency objective through diffusion loss, LIVE explicitly bounds long-horizon error propagation without relying on teacher-based distillation. We further present a unified perspective that connects TF, DF, and LIVE, and derived a progressive training curriculum that stabilizes optimization while preserving generation quality. Extensive experiments demonstrate that LIVE achieves strong performance and robust generalization on long-horizon interactive video world modeling benchmarks, significantly extending the effective rollout horizon beyond the training window. In future work, we will further scale up LIVE on large-scale and diverse datasets.

Effect of Context Noise Strategy. We compare three noise injection strategies for the rollout context: (1) no noise, (2) fixed-scale noise, and (3) random timestep sampling (LIVE). Without noise, the model shows acceptable shorthorizon performance but degrades at longer sequences. Fixed-scale noise provides marginal improvement, while our random timestep sampling achieves the best results. This validates the analysis in Sec 4.1 Step 2: after reversing, context quality improves monotonically, allowing trivial recovery by attending to higher-quality neighboring frame. Random per-frame noise breaks this pattern, forcing model to learn robust recovery from diverse error distributions.

Effect of Progressive Training Curriculum. Directly set-

#### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

#### References

- [1] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024. 3
- [2] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, 2024. 3
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023. 3
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 2024. 3
- [6] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. arXiv preprint arXiv:2411.00769, 2024. 3
- [7] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems,

2024. 2, 3

- [8] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025. 2, 3
- [9] Xinle Cheng, Tianyu He, Jiayi Xu, Junliang Guo, Di He, and Jiang Bian. Playing with transformer at 30+ fps via nextframe diffusion. arXiv preprint arXiv:2506.01380, 2025. 3, 6, 7, 11
- [10] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 3
- [11] Decart, Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. 2024. 3

- [12] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. 6, 11
- [13] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568, 2024. 3
- [14] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375, 2024. 2, 3
- [15] Google. Veo 3. https://deepmind.google/ models/veo/, 2025. 2
- [16] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Longcontext autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025. 3
- [17] Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, and Jiang Bian. Mineworld: a real-time and open-source interactive world model on minecraft. arXiv preprint arXiv:2504.08388, 2025. 3
- [18] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [19] Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, and Dahua Lin. Endto-end training for autoregressive video diffusion via selfresampling, 2025. 3
- [20] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. In European Conference on Computer Vision. Springer, 2024. 3
- [21] William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. Advances in neural information processing systems, 2022. 3
- [22] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 6, 7
- [23] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source realtime and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025. 3
- [24] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 3

- [25] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3
- [26] Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720, 2024. 2
- [27] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft, 2025. 3
- [28] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3, 6
- [29] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 2, 3

- [30] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 3
- [31] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3
- [32] Zongyi Li, Shujie Hu, Shujie Liu, Long Zhou, Jeongsoo Choi, Lingwei Meng, Xun Guo, Jinyu Li, Hefei Ling, and Furu Wei. Arlon: Boosting diffusion transformers with autoregressive models for long video generation. arXiv preprint arXiv:2410.20502, 2024. 3
- [33] OpenAI. Sora, 2024. 2, 3
- [34] J Parker-Holder, P Ball, J Bruce, V Dasagi, K Holsheimer, C Kaplanis, A Moufarek, G Scully, J Shar, J Shi, et al. Genie 2: A large-scale foundation world model. URL: https://deepmind. google/discover/blog/genie2-a-large-scale-foundation-world-model, 2024. 3
- [35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 2023. 3
- [36] Ryan Po, Eric Ryan Chan, Changan Chen, and Gordon Wetzstein. Bagger: Backwards aggregation for mitigating drift in autoregressive video diffusion models, 2025. 3
- [37] A Polyak, A Zohar, A Brown, A Tjandra, A Sinha, A Lee, A Vyas, B Shi, CY Ma, CY Chuang, et al. Movie gen: A cast of media foundation models. 2024a. arXiv preprint arXiv:2410.13720, 2024. 2
- [38] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025. 3

- [39] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025. 2, 3, 6, 7
- [40] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 3
- [41] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024. 3
- [42] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3
- [43] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling. arXiv preprint arXiv:2507.07982, 2025. 6, 7
- [44] Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. ivideogpt: Interactive videogpts are scalable world models. Advances in Neural Information Processing Systems, 2024. 3
- [45] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Longterm consistent world simulation with memory, 2025. 6, 11
- [46] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 3
- [47] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622,

2025. 3

- [48] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [49] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 3
- [50] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141,

2025. 6, 11

- [51] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025. 2, 3
- [52] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025. 3

#### 7. Appendix

##### 7.1. Implementation Training Details

- 7.1.1. RealEstate10K

Our model has 774M parameters, sharing identical architecture (DiTs) and inference procedures with NFD [9]. Specifically, we use 18-step ODE sampling during inference for all methods. The model operates at 256 × 256 resolution with a frame skip of 2 during both training and inference, employing the same VAE as NFD for 16× spatial downsampling to the latent space.

All experiments are conducted on a cluster of 32 NVIDIA H100 GPUs with a batch size of 64. We use the Adam optimizer with a learning rate of 4 × 10−5. Both NFD-TF and NFD-DF are trained from scratch for over 200k iterations until convergence. Our method (LIVE) is initialized from the converged NFD-DF checkpoint (200k steps) and trained for an additional 20k iterations until convergence.

We use a fixed context window of 32 frames during both training and evaluation. The training set follows DFoT, containing approximately 50-60k videos. All metrics are reported on the complete RealEstate10K test set with over 7k videos.

- 7.1.2. UE Engine Videos

For UE Engine Videos dataset [50], we use the same model configuration as RealEstate10K (774M parameters). We initialize from RealEstate10K (256×256) pre-trained weights and fine-tune at 352×640 resolution with a frame skip of 2. The dataset contains 100 videos totaling 7,601 frames across 12 scenes with camera pose annotations. We randomly select 12 videos (one per scene) for testing and use the remaining 88 videos for training. For evaluation, we uniformly sample 50 starting frames from each test video, resulting in 600 test sequences in total.

All experiments are conducted on 32 NVIDIA H100 GPUs with a batch size of 64. We use the Adam optimizer with a learning rate of 4×10−5. NFD-TF and NFD-DF are initialized from their respective RealEstate10K checkpoints (TF and DF) and fine-tuned for 10k iterations. Our method (LIVE) is initialized from the RealEstate10K DF checkpoint, first fine-tuned with DF for 10k iterations, then further trained with LIVE for 6.5k iterations. We use the same VAE as RealEstate10K for 16× spatial downsampling to the latent space.

- 7.1.3. Minecraft

For Minecraft, we use the same model configuration as RealEstate10K (774M parameters) and operate at 224×384 resolution with a frame skip of 1. The WorldMem [45] training dataset contains approximately 10k interactive gameplay videos of 1500 frames each, collected through MineDojo [12], where each frame is accompanied by a 25-dimensional action vector. Since WorldMem does not provide an official test set, we collect 300 action trajectories from MineDojo for evaluation, with each trajectory representing randomly generated gameplay data.

All experiments are conducted on 32 NVIDIA H100 GPUs with a batch size of 64. We use the Adam optimizer with a learning rate of 4×10−5. NFD-DF is initialized from the original NFD checkpoint (200k steps) and fine-tuned on WorldMem for 30k iterations. NFD-TF is trained from scratch on WorldMem for 100k iterations. Our method (LIVE) is initialized from the converged NFD-DF checkpoint (after 30k iterations on WorldMem) and further trained with LIVE for 3k iterations. We use the same VAE as NFD with the decoder fine-tuned on Minecraft scenarios for 16× spatial downsampling to the latent space.

##### 7.2. Additional Qualitative Results

We provide additional qualitative examples across different datasets. Through these examples, we observe that different models exhibit distinct failure patterns during long rollouts. For instance, TF models tend to develop color distortion and semantic inconsistency, while DF models show exposure problems with overexposed or underexposed regions. Our method addresses these issues by training the model to recover from its own generated errors, thereby achieving stable generation quality even over extended sequences. The reversibility constraint ensures that the model learns to maintain quality within a bounded range throughout the generation process.

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

GTDFoTGFDFTFOurs

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

1!" 256#$

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

GTDFoTGFDFTFOurs

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

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

1!" 256#$

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

GTDFoTGFDFTFOurs

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

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

[Figure 475]

[Figure 476]

1!" 256#$

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

GTDFoTGFDFTFOurs

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

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

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

1!" 256#$

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

GTDFoTGFDFTFOurs

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

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

1!" 256#$

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

GTDFoTGFDFTFOurs

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

1!" 256#$

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

GTDFoTGFDFTFOurs

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

1!" 256#$

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

GTDFoTGFDFTFOurs

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

1!" 256#$

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

GTDFoTGFDFTFOurs

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

1!" 256#$

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

GTDFoTGFDFTFOurs

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

1!" 256#$

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

GTDFoTGFDFTFOurs

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

1!" 256#$

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

GTDFoTGFDFTFOurs

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

1!" 256#$

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

GTDFoTGFDFTFOurs

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

1!" 256#$

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

OursDFGTTF

[Figure 1095]

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

1𝑠𝑡 256𝑡ℎ

[Figure 1116]

