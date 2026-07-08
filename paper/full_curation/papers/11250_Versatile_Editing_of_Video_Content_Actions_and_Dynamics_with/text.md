# arXiv:2603.17989v1[cs.CV]18Mar2026

## Versatile Editing of Video Content, Actions, and Dynamics without Training

Vladimir Kulikov⋆1,2, Roni Paiss1, Andrey Voynov1, Inbar Mosseri1, Tali Dekel1,3, and Tomer Michaeli2

1 Google DeepMind 2 Technion – Israel Institute of Technology 3 The Weizmann Institute of Science

Abstract. Controlled video generation has seen drastic improvements in recent years. However, editing actions and dynamic events, or inserting contents that should affect the behaviors of other objects in realworld videos, remains a major challenge. Existing trained models struggle with complex edits, likely due to the difficulty of collecting relevant training data. Similarly, existing training-free methods are inherently restricted to structure- and motion-preserving edits and do not support modification of motion or interactions. Here, we introduce DynaEdit, a training-free editing method that unlocks versatile video editing capabilities with pretrained text-to-video flow models. Our method relies on the recently introduced inversion-free approach, which does not intervene in the model internals, and is thus model-agnostic. We show that naively attempting to adapt this approach to general unconstrained editing results in severe low-frequency misalignment and high-frequency jitter. We explain the sources for these phenomena and introduce novel mechanisms for overcoming them. Through extensive experiments, we show that DynaEdit achieves state-of-the-art results on complex textbased video editing tasks, including modifying actions, inserting objects that interact with the scene, and introducing global effects (see website).

### 1 Introduction

Generative video models have advanced to a point where synthesized content is increasingly indistinguishable from reality in its adherence to physics, causality, and complex dynamics [3,4,13–15,23,46,53]. Modern text-to-video models are now often regarded as “world models” – foundation models that possess an inherent understanding of our physical and dynamic world [33,49]. Given this progress, a natural question arises – can we tap into the immense knowledge of these models to alter a real-world video rather than generating one from scratch? For example, can we change the actions and movements of a subject, insert or swap an existing object to facilitate meaningful interaction with the scene, or create global effects that integrate naturally with the world?

⋆ Work done during an internship at Google DeepMind.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Input

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Edited

Cat playing with a toy marshmallow → …+ runs off

Horse circling… → … + jumping over obstacle

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

InputEdited

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Sunny beach… → Nighttime beach with a campfire

Billiard ball missing … → … + white ball enters pocket

- Fig. 1: Training-free versatile editing of actions and dynamics in videos. We present DynaEdit, a training-free flow-based method for video editing, which is the first to enable manipulation of dynamics and contents in videos using textual descriptions. DynaEdit supports the modification of actions and the insertion of objects that interact with the scene (e.g. causing a horse to jump due to a newly inserted obstacle, a cat to run off due to a toy edited to become a burning marshmallow, or a billiard ball to enter the pocket). It also allows global and stylistic modifications, like changing daytime to nighttime, all while avoiding unnecessary changes to the video (see SM for the videos).

Despite the remarkable progress in video editing [7–9, 12, 21, 26, 28, 40, 42, 47,48,50,52], the task of non-rigid, dynamic manipulation in real-world videos remains an open challenge. This stems from a fundamental tension in the editing objective: the model must possess enough flexibility to fundamentally alter motion or object interactions, yet simultaneously remain strictly faithful to the original objects’ identities and environmental context. A data-driven approach to this problem is hindered by the difficulty of obtaining high-quality training data. Specifically, non-rigid editing requires precisely paired source-target example videos that demonstrate the same scene under different physical outcomes, data that is exceptionally difficult to collect or simulate at scale. Currently, RunwayML’s Gen-4 Aleph [41] is the only publicly available trained model that provides a general prompt-based framework for manipulation of video. While constituting a significant advancement, this model still struggles with complex nonrigid action-altering edit requirements. Several works proposed training-free editing methods that harness a pre-trained text-to-video model [8,9,12,22,24–26,35]. Yet, these methods are constrained to structurally aligned transformations, or to layer-like object insertion, where the inserted object can one-sidedly react to the rest of the content in the video, but cannot affect it.

In this paper, we introduce DynaEdit, a training-free method for in-thewild unconstrained video editing. Given an input source video and a target text prompt that describes the edit, our method steers the generation process of a pre-trained text-to-video flow model towards the desired solution – altering

the scene’s dynamics, while preserving the properties of the original video that should not be affected by the edit. As shown in Fig. 1, DynaEdit supports modification of dynamic events, like causing a horse to jump over a newly inserted obstacle, a billiard ball to enter the pocket, or a cat to run off due to interaction with a toy that was edited to become a burning marshmallow. It also allows global modifications, like changing a sunny scene into a nighttime setting.

DynaEdit relies on the recently introduced inversion-free approach [25]. We show that the naive adaptation of this approach to support significant spatiotemporal modifications leads to severe low-frequency misalignment with the source video and high-frequency jitter. We explain why these phenomena arise and introduce two novel components for mitigating them: a Similarity Guided Aggregation (SGA) mechanism and an Annealed Noise Correlation (ANC) schedule. Extensive evaluations demonstrate that DynaEdit not only outperforms all existing training-free methods but effectively closes the performance gap with the proprietary trained Aleph model on a wide variety of complex editing tasks.

### 2 Related Work

Text-to-video generative models [3,4,13–15,23,34,46,53] have seen tremendous recent progress, with the most advanced open-source models [15,23,46] relying on the flow matching framework [27,29]. This progress has given rise to numerous video editing methods. Many methods target specific types of edits, such as motion transfer [19,32,39,55], effect transfer [20], object insertion [2,44,45,54], optical-flow or keypoint-controlled motion editing [5,6], re-angling [51,57] or style transfer [30,56]. Here, we focus on general-purpose video editing using only text. Existing methods in this category focus on the sub-task of structurally-aligned editing [22,24,26,35], leaving general editing an open challenge.

Training-based video editing. Training a model for general video editing requires non-trivial data collection and an extensive computational budget. Some methods propose lightweight inference-time training [10, 38]. The only trained model that currently supports in-the-wild video editing is RunwayML’s Gen-4 Aleph [41], which is not open source. This model was the first to allow text-based editing of actions and dynamics, however it still struggles to perform complex manipulations, attesting to the difficulty of the task.

Training-free video editing. Opting for open source video editing solutions, several works proposed training-free methods which utilize pre-trained video flow models. These works can be broadly categorized into inversion-based and inversionfree approaches. Inversion-based methods, such as [32, 47, 54], start by finding a noise initialization that reconstructs the input video when conditioned on a source prompt describing that video. They then use that noise for sampling a new video by conditioning on a target prompt that describes the desired edit. This approach by itself often leads to poor results [18] and therefore many works proposed additional model-specific interventions. For example, DynVFX [54] employ attention-based manipulations during the inversion-and-sampling process

to perform object insertion. This method can incorporate objects in a natural harmonious manner, however the inserted objects cannot dynamically interact with the surrounding scene or alter the video’s outcomes. Inversion-free approaches [22,25,26] traverse a noise-free path between the source and target domains, without relying on inversion. FlowEdit [25] first proposed and implemented this paradigm for flow-based image editing. FlowAlign [22] introduced an improved variant of this approach and exemplified its effectiveness in the video domain as well. FlowDirector [26] proposed an ad-hoc solution for swapping objects in videos by leveraging an attention-based mask construction to constrain the edits to desired regions. While these approaches can achieve better quality than inversion-based methods, both approaches are constrained to strong structure-preserving edits, with limited ability to change the coarsest features of the source video. In this work, we build upon the inversion-free editing approach, but make key adaptations to allow it to support structurally unrestricted editing.

### 3 Preliminaries

We use upper-case and lower-case letters to denote random variables and their realizations (samples from the corresponding distribution), respectively.

#### 3.1 Rectified Flow Models

Flow models learn a velocity field V , parameterized by a neural network, with which they generate samples by solving the ODE

dZt = V (Zt,t)dt (1)

over t ∈ [0,1]. The core objective is to have the ODE transport from a simple prior at t = 1, typically N(0,I), to the data distribution at t = 0. Sampling thus involves initializing the ODE at t = 1 with a sample of Gaussian noise and numerically solving it in reverse down to t = 0. In practice, the integration is performed over N discrete steps {ti}Ni=0. Rectified flows [1, 27, 29] represent a specific class of these models, where Zt is distributed like

Xt = (1 − t)X0 + tX1, (2)

where X0,X1 are statistically independent. This choice leads to low path curvatures and thus enables sampling with a small number of discretization steps.

Image-to-video (I2V) flow models employ a velocity field V (xt,t,c,f) that is conditioned on a text prompt c and an image f depicting the first frame. Such models are trained on triplets of text, first frame, and video data, {c,f,x0}, and thus enable sampling from the conditional distribution of X0 given C,F. Throughout this work we use an I2V model, which is beneficial for our task of general video editing. Specifically, when the edited video is required to lose spatio-temporal alignment with the source video, the first frame conditioning helps in maintaining scene, object, and color-palette consistency (see App. C.6).

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Realvideo

Horse circling → … + jumping over obstacle

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

𝑛 =𝑁−1

𝑛 =1 𝑛=𝑁max

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

𝑛=1avg

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

𝑛 =100

𝑛 =𝑁

- Fig. 2: Limitations of state-of-the-art inversion-free editing methods. Existing inversion-free methods [22,25,26] suffer from a tradeoff between edit expressivity and visual quality, illustrated here with FlowEdit [25] using an I2V model. When starting the generation at timestep nmax = N − 1, the method struggles to modify motion (second row). Using nmax = N allows making more significant spatio-temporal modifications and thus to better adhere to the prompt, but results in severe jitter artifacts and illogical motions (third row). Attempting to reduce artifacts by averaging over navg = 100 edit directions in each step leads to blur (fourth row).

#### 3.2 Inversion-Free Editing With Pretrained Flow Models

In text-based video editing, the user provides an input video xsrc, a source prompt describing it csrc, a target prompt ctar that describes the desired edit, and optionally an edited first frame ftar for added conditioning. Several approaches exist for inversion-free editing [22,25,26]. Here we focus on FlowEdit [25].

The idea in this approach is to construct an ODE that directly transforms the source video into an edited video, such that all intermediate videos along the path are noise-free. To simplify notations we denote the source- and target-conditioned velocities by V src(xt,t) = V (xt,t,csrc,fsrc) and V tar(xt,t) = V (xt,t,ctar,ftar), respectively. In FlowEdit, the noise-free path is traced by the ODE

dZtedit = E Vt∆(Ztsrc,Zttar) dt, (3)

where Vt∆(Ztsrc,Zttar) = V tar(Zttar,t)−V src(Ztsrc,t). Here, Ztsrc = (1−t)xsrc+tWt is a noisy version of the source video obtained with Wt ∼ N(0,I), and Zttar = Ztedit + Ztsrc − xsrc is a noisy version of the target video being constructed. The expectation is over Wt. The ODE is initialized at t = 1 with the source video Z1edit = xsrc and solved backwards down to t = 0 to obtain an edited video Z0edit.

In practice, the expectation in (3) is approximated by averaging over navg independent noise samples in each timestep. These samples are taken to be independent also across timesteps, a fact that turns out to play an important

- Algorithm 1 Inversion-free editing (FlowEdit)

- 1: Input: real video xsrc, source prompt csrc, target prompt ctar
- 2: Output: edited video zedit
- 3: Initialize: zedit = xsrc
- 4: for i = nmax to 1 do
- 5: {wj}nj=1avg ∼ N(0, I) // Sample random noise vectors
- 6: zjsrc ← (1 − ti)xsrc + tiwj // Construct noisy source samples
- 7: zjtar ← zedit + zjsrc − xsrc // Construct noisy target samples
- 8: Vj∆ ← V (zjtar, ti, ctar) − V (zjsrc, ti, csrc) //Calculate velocity differences
- 9: V¯∆ ← n1

avg

navg j=1 Vj∆ // Average directions

- 10: zedit ← zedit + (ti−1 − ti)V¯∆ // Propagate ODE
- 11: end for
- 12: Return: zedit

role, as we illustrate in Secs. 4 and 5. The hyperparameter navg is often set to 1,

- as averaging naturally occurs also across timesteps. To control the amount of deviation from the source video, FlowEdit can be

initialized at a timestep nmax ≤ N. This hyperparameter effectively controls the maximum amount of noise that is added to the source video and thus implicitly determines the coarsest spatio-temporal features that can get modified. It therefore controls the tradeoff between edit expressivity and structural adherence to the source video. A pseudo-code for this method is given in Alg. 1, where subscripts are used to index samples within the batch rather than time.

### 4 Roadblocks towards motion and interaction editing

Existing inversion-free methods struggle with complex edits that require significant spatio-temporal modifications. For example, while FlowEdit should in theory be able to perform arbitrary edits given a large enough nmax, it is practically impossible to select a value for nmax that strikes a good balance between output quality, prompt adherence, and loyalty to the source video. This is illustrated in Fig. 2, which shows FlowEdit results using nmax = N − 1 and nmax = N. Here, the goal is to insert an obstacle and have the horse jump over it. As seen, setting nmax = N − 1 is too restrictive for the requested edit (the horse fails to perform the requested jump). On the other hand, setting nmax = N results in a video that adheres to the edit prompt, but exhibits extraneous low frequency changes (the horse’s trajectory needlessly deviates from the source motion), and suffers from severe high frequency jitter artifacts (evident by the blurry obstacle). Note that while FlowEdit’s velocity averaging usually improves quality, in the case of structurally-unaligned video editing, it results in blurry edits, as seen in the last row, where navg = 100 is used. We next analyze the causes for the low frequency misalignment and high frequency jitter that emerge when nmax = N.

Low frequency misalignment. When using nmax = N, the noisy marginals Ztsrc

N

and Zttar

in Eqn. (3) contain pure noise (both equal Wt

). This means that the

N

N

###### (a) Original Video

(b) Effect of initial noise on final edit

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

- #1Result

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

frame

Different motion 1

[Figure 53]

[Figure 54]

[Figure 55]

- #2Result

[Figure 56]

frame

Different motion 2

Different motion 2

[Figure 57]

[Figure 58]

[Figure 59]

- #3Result

Input

frame

Original motion

Original motion

Train passing through the forest → … + hitting a yellow paint bucket

(c) Effect of noise correlations on final edit

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

iidnoiseConstnoise

frameframe

Jitter

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

frame

Different motion 3

Gap

- Fig. 3: Effects of noise in inversion-free editing. (a) A source video (three frames and a spatio-temporal slice corresponding to the dashed line). (b) Three inversion-free

editing results (FlowEdit) differing only in the noise sample at timestep tN. As seen, the initial noise strongly affects coarse spatio-temporal features, e.g. modifying the camera motion and train position across edits, although those features are not required to change to adhere to the prompt. (c) Using independent noise samples across timesteps (top) leads to high-frequency jitter, e.g. the blurry bucket and paint drops. This can be alleviated by using the same noise sample for all timesteps (bottom) but at the cost of worse alignment with the source video, e.g. causing the bucket to levitate.

has no connection to the source video beyond the first frame conditioning. The effect that this has is visualized in Fig. 3(a),(b). Here, the goal is to insert a bucket of paint to the train tracks and have the train collide with it. The figure shows the input video and three different edit results, each obtained with a different noise realization for the initial timestep, but the same set of noise maps for all subsequent timesteps. As seen, the resulting videos have different camera motions, train speeds, and bucket explosion times. This reveals that the initial edit step has an immense impact on the coarse spatio-temporal features of the edited video. Importantly, the resulting edits are not well-aligned with the source video, as seen by the spatio-temporal slices (e.g. the curves caused by the camera motion do not align). This suggests that while using nmax = N is crucial for modifying coarse spatio-temporal features, the noise realizations in the initial timesteps should be carefully selected to allow maintaining adherence to the source video. We explore this in Sec. 5.1.

edit velocity Vt∆

N

High frequency jitter. When the edited video contains assets that are not spatiotemporally aligned with the source (e.g. an inserted object or edited dynamics), severe high frequency jitter emerges. This is evident in the first row of Fig. 3(c), where the high frequencies of the bucket and the paint drops are fuzzy. We hypothesize that this stems from the fact that the noises {Wt

n} are uncorrelated across timesteps. This causes the edit velocities Vt∆ to point to different directions that accumulate to the visible jitter artifacts. To test this hypothesis, the second row of Fig. 3(c) shows the result obtained when using the same noise realization Wt

##### = Wconst for all timesteps. As can be seen, this indeed eliminates the high-frequency jitter. Unfortunately, however, it worsens the alignment with the input video’s coarse features, resulting in unnatural interactions (notice the

n

Inversion-Free Edit Path

Source video Edited video

[Figure 69]

[Figure 70]

[Figure 71]

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

… …

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Edit progression

𝑧

𝑧

𝑉

𝑛

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

###### VEP EPV ⋅ 𝑑𝑡

|Velocity calculation (Alg. 2. L7-L9)|
|---|

+

SGA

𝑗 = 1

𝑛

𝑛

Inputs: 𝑥 𝑐 :

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

[Figure 125]

[Figure 126]

ANC

###### 𝑐 :

[Figure 127]

[Figure 128]

“horses in a pasture.”

+ “right horse leaves.”

𝑗 = 1

𝑗 = 1

𝑤

𝑤

Similarity Guided Aggregation

Velocity to Edit Prediction

Edit Prediction to Velocity

𝑧 𝑧

𝑉 = 𝑧 − 𝑧̅ /𝑡

𝑧 = 𝑧 − 𝑡 ⋅ 𝑉

𝑥

𝑧

𝑉 𝑧

𝑧 𝑉

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

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

… …

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

|Cosine similarity| | | | | |
|---|---|---|---|---|---|
|×|…|×|…|×| |

Annealing Noise Correlations

C

𝑤

𝑤

𝑤

𝑧̅

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

𝑎 ×

+ 1 − 𝑎 ×

Softmax

- Fig. 4: DynaEdit. Our method constructs a noise free path from the source video to the edited one (top pane). The middle pane shows one step in this process, with our key contributions colored. Our SGA module (bottom left) aggregates several noise-free velocities based on the similarities between the edits they induce and the source video. Our ANC mechanism (bottom right) induces gradually increasing correlations between the noises of consecutive timesteps.

levitating bucket). This suggests that introducing some amount of correlation between the noises of different timesteps may allow to strike a good balance between visual quality and low-frequency alignment. We explore this in Sec. 5.2.

- 5 Method

We now present DynaEdit, an inversion-free method that overcomes the limitations discussed in Sec. 4. DynaEdit relies on two new components, as we detail next. The method is illustrated in Fig. 4, and pseudo-code is provided in Alg. 2.

#### 5.1 Similarity Guided Aggregation

In Fig. 3(b), we saw that the initial edit steps facilitate the most significant changes to the low spatio-temporal frequencies, but vary significantly depending on the noise seed. To achieve edits that are better aligned to the source

#### Algorithm 2 DynaEdit

- 1: Input: real video xsrc, source prompt csrc, target prompt ctar
- 2: Output: edited video xedit
- 3: Initialize: zedit = xsrc, {w˜j}n

SGA N

j=1 = 0

- 4: for i = N to 1 do
- 5: {wj}n

SGA

- i
- j=1 ∼ N(0, I) // Sample random noise vectors

- 6: w˜j ←

√aiw˜j + √1 − atiwj // Construct correlated noise with ANC (Eq. (7))

- 7: zjsrc ← (1 − ti)xsrc + tiw˜j // Construct noisy source samples
- 8: zjtar ← zedit + zjsrc − xsrc // Construct noisy target samples
- 9: Vj∆ ← V (zjtar, ti, ctar) − V (zjsrc, ti, csrc) // Calculate velocity differences
- 10: V¯∆ ← SGA({Vj∆}n

SGA

- i
- j=1 ) // Aggregate velocities using SGA (Eqs. (4)-(6))

- 11: zedit ← zedit + (ti−1 − ti)V¯∆ // Propagate ODE
- 12: end for
- 13: Return: zedit

frequencies, we propose similarity guided aggregation (SGA), a mechanism for soft selection of edit velocities based on their similarity to the source video. In each edit step i, we use nSGAi noise samples to obtain random edit directions V1∆,...,Vn∆SGA

. We predict the final edit that would be obtained with each of them by using ∆t = t0 − ti = −ti, namely we construct the projected edits

i

zjedit-proj = zedit − tiVj∆. (4) We calculate the cosine similarity between each prediction and the source video to obtain coefficients sj = sim(xsrc,zjedit-proj) (see App. C.4 for the effect of other similarity metrics) and normalize them using softmax with temperature τ. The resulting weights are used to construct the combined edit prediction as

z¯edit =

nSGAi

sjzjedit-proj. (5)

j=1

This prediction is transformed back to a velocity to obtain the edit direction

V¯∆ = (zedit − z¯edit-proj)/ti. (6) The SGA module is depicted in the bottom-left pane of Fig. 4. We find that to save computation, it is enough to use nSGAi > 1 only for the first few timesteps (see Sec. 6.1 for details). The softmax temperature τ controls the degree of alignment between the edited video and the source video. When τ is small Eq. (6) collapses to a hard-selection rule, retaining only the edit path that best matches the source video and thus leading to stronger alignment. We demonstrate the advantage of SGA over the simple velocity averaging of FlowEdit [25] in App. C.1.

#### 5.2 Annealed Noise Correlation

As discussed in Sec. 4, when setting nmax = N, the use of i.i.d. noise leads to high-frequency jitter. We attribute this to the fact that uncorrelated noise sam-

ples in consecutive timesteps steer the process towards different edit directions. In our case, where the spatio-temporal structure of the edited video may significantly deviate from the source, this stochasticity can cause fuzziness and visible jitter. We saw in Fig. 3(c) that using the same noise realization for all timesteps mitigates the high frequency jitter, but worsens the low-frequency misalignment. This is because, as discussed in Sec. 5.1, improving the low-frequency alignment requires a diverse set of noise realization to choose from. Therefore, to reduce the high-frequency jitter without worsening low-frequency misalignment, we propose an Annealed Noise Correlation (ANC) scheduler, which introduces noise correlations that grow towards the end of the sampling process. Specifically, assuming w˜j is the jth noise sample at timestep ti, then at timestep ti−1 we set

√at

wj, (7) where {wj} are i.i.d noise samples and {at

w˜j ←

w˜j + 1 − at

i

i

i} is an increasing sequence such that

= 1. This guarantees that the correlation increases towards the last sampling steps, where the high frequency jitter is most prominent. The ANC module is depicted in the bottom-right part of Fig. 4. In App. C.3 we demonstrate the effect of the noise correlation schedule on the edit path.

= 0 and at

- at

1

N

### 6 Experiments

#### 6.1 Implementation details

We conduct our experiments using the WAN2.1 14B 480p I2V model [46] and provide additional qualitative results using Hunyuan 1.5 I2V [23] in App. A.2. For the SGA module (Sec. 5.1), we use a fixed schedule of nSGAi = 5 for the initial three steps (i > N −3) and nSGAi = 1 thereafter. For the ANC module (Sec. 5.2), we choose at to be linearly increasing from 0 to 1, starting from zero correlation at t = 1, reaching a correlation of 1 at t = 0.25, and remaining constant until t = 0. We experiment with four hyperparameter configurations, which correspond to two options for the CFG scale parameter [16] and two options for the SGA temperature τ. Specifically, for the CFG parameter we use either 4.5 and 8.5 or 2.5 and 4.5 for the source and target velocities, respectively. For the temperature we use either τ = 0.01 or τ = 1. We quantitatively evaluate each of these four configurations in Sec. 6.6 and illustrate their effects in App. D.1.

#### 6.2 Evaluation set

There are no existing evaluation sets for diverse video editing tasks that require significant spatio-temporal modification. We therefore curate a dataset of 71 tuples of {source video, source text, target text, edited first frame} in four different categories: (a) insertion of objects that require two-sided interaction with the original content of the video, (b) swapping of objects with implications on the outcome of events, (c) modifications of motion and action of objects, and (d) global spatio-temporal effects. We manually curated the source videos

###### Interactive Insertion

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

EditedRealvideo

Two astronauts… → … + flag picked up Aerial night view of a town… → … + Heli with light

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Swap & Action

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

EditedRealvideoEditedRealvideoEditedRealvideo

Orange stand → … + red apple picked up Pitcher with red juice… → …green juice + reaction

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

###### Action change

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Fisherman fishes… → … + catches fish Pizza rotated in oven → Pizza taken out

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Global effect

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Man hiking… → … + sandstorm + hat off Promenade at night → … + fireworks

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

- Fig. 5: DynaEdit Results. Our method supports a wide range of edits, including motion manipulation (swans), interactive object addition (horse, barrier, dinosaur), and global style changes (magma, Manga).

from Pexels [37], and selected target prompts to maximize diversity and ensure that each category contains at least 15 edits. We verify in App. C.5 that the particular phrasings we chose for the prompts do not affect the performance of the method. The videos are 49-81 frames long, with resolution of 832 × 480 and 16fps, as expected by the WAN model. For edits that require changes to the first frame (such as style change or object insertion), we obtain the edited first frame by querying Google’s Gemini 2.5 Flash Image (Nano-Banana Pro). For the rest of the edits, the first frame is taken to be that of the source video.

#### 6.3 Qualitative results

Figures 1, 5, 6, and App. Fig. 1 present diverse edits obtained with our method (see videos in the SM). As seen, DynaEdit can realistically manipulate motion (e.g. making the ball enter the pocket in Fig. 1, bottom left), actions (e.g. making the horse jump in Fig. 1, top left) and motion (making a fisherman catch a fish in Fig. 5, third row). Importantly, our method keeps the output video as similar as possible to the source video given the edit request. For example, unless specified otherwise in the text, camera motion remains similar, objects not related to the edit keep their original action, and motions follow the same patterns. We report the hyperparameters used for each of the figures in the paper in App. D.2.

#### 6.4 Competing methods

We compare our method against FlowEdit [25], FlowAlign [22], editing by ODE inversion [43], I2V sampling, SDEdit [31] and Aleph [41] (a trained instructionbased video editing model). In Apps. B.3,B.4 we provide additional comparisons to FlowDirector [26] and DynVFX [54], which support only object swapping and object insertion, respectively. These are evaluated only on the relevant categories.

For Runway Aleph, we attach the target frame as a reference image for the instruction and specify in the prompt to use it as reference for the first frame. Runway Aleph expects instruction-style prompts rather than source and target text captions, so we use Gemini 3 Pro to convert the pairs into an instruction prompt (see App. E). For FlowEdit, FlowAlign, SDEdit, and I2V sampling, which are training-free methods, we use the same base I2V model as we use for our method. We tested all these methods with multiple hyperparameter configurations (including the reported default, where applicable) and chose the best performing option. For Runway Aleph we used the default settings in their web API. We report the parameters in App. Tab. 1.

#### 6.5 Qualitative comparisons

Figure 6 and App. Fig. 3 provide qualitative comparisons against competing baselines.

In Fig. 6, a video of two strawberries falling into water is edited to replace the right strawberry with a feather. The edit requires a significant change to the dynamics of the scene due to the different physical properties of the feather. As can be seen, DynaEdit is the only method that manages to successfully generate a plausible edit, where the feather slowly descends until reaching the water level and then floats on the water surface. Importantly, DynaEdit does not modify the dynamics of the strawberry on the left. This is in contrast to all other methods, which either cause it to disappear/fade (Aleph, FlowAlign) or change its velocity (FlowEdit, I2V). Appendix B.1 provides additional examples, where the competing methods fail to generate a high-quality result.

Frames ST Slice

[Figure 237]

[Figure 238]

RealvideoFlowEditI2V

[Figure 239]

[Figure 240]

[Figure 241]

Original motion

Two strawberries fall → A strawberry and a feather…

frame

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

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

OursAlephFlowAlign

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

- Fig. 6: Qualitative comparison. A video depicting two strawberries falling into a water tank is edited such that the right strawberry is replaced by a feather. In contrast to competing methods, DynaEdit generates a physically plausible video, where the feather slowly descends and then floats on the water, while keeping the dynamics of the left strawberry unchanged (see App. 3 for more comparisons).

#### 6.6 Quantitative comparisons

VLM-based evaluation. Similar to [11,17,54], we utilize a vision-language model (VLM) to rate the edited results with a score of 1 to 5 for each of the following three aspects: (a) adherence to the source video (b) adherence to the target text (c) overall visual quality. We use Gemini 3 Pro as our VLM and base the evaluation instructions on the ones used in [36]. Additional details are provided in App. F. Figure 7 (tabular version in App. Tab. 1) shows this comparison, where our method is evaluated across all four hyperparameter configurations mentioned in Sec. 6.1. As can be seen, our method strikes the best balance between the preservation of the source video and adherence to the target text, while also achieving superior visual quality. In Fig. 8(a) in the appendix, we provide a breakdown of this comparison according to the four edit categories in our dataset. In all categories, our method significantly outperforms the trainingfree baselines, and is typically at least comparable to the trained Aleph model.

(a) VLM Evaluation (b) User Study

Content Preservation vs. Text Adherence

Content Preservation vs. Visual Quality

Visual Quality

Content Preservation

Adherence to Text

ContentPreservationScore →

[Figure 267]

[Figure 268]

59.2%

77.7%

79.9%

56.4%

82.1%

80.2%

58.3%

73.4%

77.2%

Visual Quality Score →

Adherence to Text Score →

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Ours Aleph FlowAlign FlowEdit I2V Sample ODE Inversion SDEdit

Vs. Aleph Vs. I2V Sample Vs. FlowEdit

- Fig. 7: Quantitative comparison. We compare DynaEdit to existing methods on content preservation, text adherence and visual quality. (a) VLM ratings of these criteria show that DynaEdit ranks best in content preservation, while achieving comparable text adherence and visual quality to the trained Aleph model. (b) A user study comparing DynaEdit against the top three competitors shows higher preference rates for DynaEdit, with the trained Aleph model being the closest competitor.

User study. To complement the automatic evaluation, we conducted a user study comparing our method against each of the top competing methods: Runway Aleph, FlowEdit, and I2V sampling. Here, we used a different hyperparameter configuration (among the four mentioned in Sec. 6.1) for each edit according to the extent of the required modification. For changes to larger parts of the scene we use the higher CFG configuration, and for edits requiring stronger motion changes we use the higher SGA temperature configuration (see App. D.1). In each question, the participants were presented a source video, a prompt depicting the desired edit, and two edit results, one of them being ours. The participants were asked three questions: (1) which video best preserves the source content? (2) which video best adheres to the target text? and (3) which video has the best visual quality? We collected over 2400 responses from 32 unique participants. Full details on the user study can be found in App. G. The results are reported in Fig. 7, right pane. As seen, DynaEdit’s results are preferred by most users over those of the leading methods, including the trained Aleph model, in all three aspects. Per-category comparisons are provided in Fig. 8(b) in the appendix.

### 7 Conclusion

We presented DynaEdit, a versatile video editing framework that allows significant modification to dynamics and contents. Our method is the first training-free method that tackles this task, achieving performance that is at least comparable to the only existing trained model. DynaEdit leverages the inversion-free paradigm, a technique that transforms the source video into its edited version through a noise-free path, but has only been explored for structure-preserving edits. We introduced mechanisms to unlock the untapped potential of this approach for general video editing. Extensive experiments demonstrated that Dy-

naEdit achieves state-of-the-art results on complex editing tasks. Our method is not free of limitations, as we discuss in App. H. In particular, it inherits the limitations of the underlying I2V model, which often struggles with physics and leads to artifacts. Additionally, it often fails to make very large spatio-temporal modifications and simultaneously preserve regions that should not be affected by the edit. We leave improvements on these fronts to future work.

### References

- 1. Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E.: Stochastic interpolants: A unifying framework for flows and diffusions (2025), https://arxiv.org/abs/2303.08797
- 2. Bai, C., Shao, Z., Zhang, G., Liang, D., Yang, J., Zhang, Z., Guo, Y., Zhong, C., Qiu, Y., Wang, Z., Guan, Y., Zheng, X., Wang, T., Lu, C.: Anything in any scene: Photorealistic video object insertion (2024), https://arxiv.org/abs/2401.17509
- 3. Bar-Tal, O., Chefer, H., Tov, O., Herrmann, C., Paiss, R., Zada, S., Ephrat, A., Hur, J., Liu, G., Raj, A., Li, Y., Rubinstein, M., Michaeli, T., Wang, O., Sun, D., Dekel, T., Mosseri, I.: Lumiere: A space-time diffusion model for video generation

(2024), https://arxiv.org/abs/2401.12945

- 4. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models

(2023), https://arxiv.org/abs/2304.08818

- 5. Burgert, R., Herrmann, C., Cole, F., Ryoo, M.S., Wadhwa, N., Voynov, A., Ruiz, N.: Motionv2v: Editing motion in a video (2025), https://arxiv.org/abs/2511. 20640
- 6. Burgert, R., Xu, Y., Xian, W., Pilarski, O., Clausen, P., He, M., Ma, L., Deng, Y., Li, L., Mousavi, M., Ryoo, M., Debevec, P., Yu, N.: Go-with-the-flow: Motioncontrollable video diffusion models using real-time warped noise (2025), https: //arxiv.org/abs/2501.08331
- 7. Ceylan, D., Huang, C.H.P., Mitra, N.J.: Pix2video: Video editing using image diffusion (2023), https://arxiv.org/abs/2303.12688
- 8. Cohen, N., Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., Michaeli, T.: Slicedit: Zero-shot video editing with text-to-image diffusion models using spatiotemporal slices (2024), https://arxiv.org/abs/2405.12211
- 9. Cong, Y., Xu, M., Simon, C., Chen, S., Ren, J., Xie, Y., Perez-Rua, J.M., Rosenhahn, B., Xiang, T., He, S.: Flatten: optical flow-guided attention for consistent text-to-video editing (2024), https://arxiv.org/abs/2310.05922
- 10. Gao, C., Ding, L., Cai, X., Huang, Z., Wang, Z., Xue, T.: Lora-edit: Controllable first-frame-guided video editing via mask-aware lora fine-tuning. arXiv preprint arXiv:2506.10082 (2025)
- 11. Garibi, D., Yadin, S., Paiss, R., Tov, O., Zada, S., Ephrat, A., Michaeli, T., Mosseri,

I., Dekel, T.: Tokenverse: Versatile multi-concept personalization in token modulation space (2025), https://arxiv.org/abs/2501.12224

- 12. Geyer, M., Bar-Tal, O., Bagon, S., Dekel, T.: Tokenflow: Consistent diffusion features for consistent video editing (2023), https://arxiv.org/abs/2307.10373
- 13. Google DeepMind: Veo: Our most capable generative video model (2024), https: //deepmind.google, accessed: 2026-02-18
- 14. HaCohen, Y., Brazowski, B., Chiprut, N., Bitterman, Y., Kvochko, A., Berkowitz, A., Shalem, D., Lifschitz, D., Moshe, D., Porat, E., Richardson, E., Shiran, G., Chachy, I., Chetboun, J., Finkelson, M., Kupchick, M., Zabari, N., Guetta, N., Kotler, N., Bibi, O., Gordon, O., Panet, P., Benita, R., Armon, S., Kulikov, V., Inger, Y., Shiftan, Y., Melumian, Z., Farbman, Z.: Ltx-2: Efficient joint audio-visual foundation model (2026), https://arxiv.org/abs/2601.03233
- 15. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., Panet, P., Weissbuch, S., Kulikov, V., Bitterman, Y., Melumian, Z., Bibi, O.: Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103 (2024)

- 16. Ho, J., Salimans, T.: Classifier-free diffusion guidance (2022), https://arxiv.org/ abs/2207.12598
- 17. Hsu, H.Y., Lin, Z.H., Zhai, A., Xia, H., Wang, S.: Autovfx: Physically realistic video editing from natural language instructions (2024), https://arxiv.org/abs/ 2411.02394
- 18. Huberman-Spiegelglas, I., Kulikov, V., Michaeli, T.: An edit friendly ddpm noise space: Inversion and manipulations (2024), https://arxiv.org/abs/2304.06140
- 19. Jiang, Z., Han, Z., Mao, C., Zhang, J., Pan, Y., Liu, Y.: Vace: All-in-one video creation and editing (2025), https://arxiv.org/abs/2503.07598
- 20. Jones, M., Abdal, R., Patashnik, O., Salakhutdinov, R., Tulyakov, S., Zhu, J.Y., Wang, K.C.J.: Tuning-free visual effect transfer across videos (2026), https:// arxiv.org/abs/2601.07833
- 21. Kara, O., Kurtkaya, B., Yesiltepe, H., Rehg, J.M., Yanardag, P.: Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models (2023), https://arxiv.org/abs/2312.04524
- 22. Kim, J., Hong, Y., Park, J., Ye, J.C.: Flowalign: Trajectory-regularized, inversionfree flow-based image editing (2025), https://arxiv.org/abs/2505.23145
- 23. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)
- 24. Ku, M., Wei, C., Ren, W., Yang, H., Chen, W.: Anyv2v: A tuning-free framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468 (2024)
- 25. Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., Michaeli, T.: Flowedit: Inversion-free text-based editing using pre-trained flow models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19721–19730

(2025)

- 26. Li, G., Yang, Y., Song, C., Zhang, C.: Flowdirector: Training-free flow steering for precise text-to-video editing. arXiv preprint arXiv: 2506.05046 (2025)
- 27. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling (2023), https://arxiv.org/abs/2210.02747
- 28. Liu, S., Zhang, Y., Li, W., Lin, Z., Jia, J.: Video-p2p: Video editing with crossattention control (2023), https://arxiv.org/abs/2303.04761
- 29. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow (2022), https://arxiv.org/abs/2209.03003
- 30. Mehraban, S., Adeli, V., Rommann, J., Taati, B., Truskovskyi, K.: Pickstyle: Videoto-video style transfer with context-style adapters (2025), https://arxiv.org/ abs/2510.07546
- 31. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic differential equations (2022), https: //arxiv.org/abs/2108.01073
- 32. Meral, T.H.S., Yesiltepe, H., Dunlop, C., Yanardag, P.: Motionflow: Attentiondriven motion transfer in video diffusion models (2024), https://arxiv.org/abs/ 2412.05275
- 33. Motamed, S., Culp, L., Swersky, K., Jaini, P., Geirhos, R.: Do generative video models understand physical principles? (2025), https://arxiv.org/abs/2501. 09038
- 34. OpenAI: Sora 2: Our flagship video and audio generation model. https://openai. com (2025), accessed: 2026-02-25
- 35. Ouyang, W., Dong, Y., Yang, L., Si, J., Pan, X.: I2vedit: First-frame-guided video editing via image-to-video diffusion models. arXiv preprint arXiv:2405.16537

(2024)

- 36. Peng, Y., Cui, Y., Tang, H., Qi, Z., Dong, R., Bai, J., Han, C., Ge, Z., Zhang, X., Xia, S.T.: Dreambench++: A human-aligned benchmark for personalized image generation. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=4GSOESJrk6
- 37. Pexels GmbH: Pexels: Free stock photos, royalty free images & videos. https: //www.pexels.com (2024), accessed: 2024-05-22
- 38. Polaczek, S., Patashnik, O., Mahdavi-Amiri, A., Cohen-Or, D.: In-context synclora for portrait video editing (2025), https://arxiv.org/abs/2512.03013
- 39. Pondaven, A., Siarohin, A., Tulyakov, S., Torr, P., Pizzati, F.: Video motion transfer with diffusion transformers (2025), https://arxiv.org/abs/2412.07776
- 40. Qi, C., Cun, X., Zhang, Y., Lei, C., Wang, X., Shan, Y., Chen, Q.: Fatezero: Fusing attentions for zero-shot text-based video editing (2023), https://arxiv.org/abs/ 2303.09535
- 41. Runway AI, Inc.: Introducing Runway Aleph: A state-of-the-art in-context video model (2025), https://runwayml.com/research/introducing-runway-aleph, accessed: 2026-02-18
- 42. Singer, U., Zohar, A., Kirstain, Y., Sheynin, S., Polyak, A., Parikh, D., Taigman, Y.: Video editing via factorized diffusion distillation (2024), https://arxiv.org/ abs/2403.09334
- 43. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models (2022), https: //arxiv.org/abs/2010.02502
- 44. Tewel, Y., Gal, R., Samuel, D., Atzmon, Y., Wolf, L., Chechik, G.: Add-it: Trainingfree object insertion in images with pretrained diffusion models (2024), https: //arxiv.org/abs/2411.07232
- 45. Tu, Y., Luo, H., Chen, X., Ji, S., Bai, X., Zhao, H.: Videoanydoor: High-fidelity video object insertion with precise motion control (2025), https://arxiv.org/ abs/2501.01427
- 46. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang,

- X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li,
- Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.F., Liu, Z.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)

- 47. Wang, J., Pu, J., Qi, Z., Guo, J., Ma, Y., Huang, N., Chen, Y., Li, X., Shan, Y.: Taming rectified flow for inversion and editing (2025), https://arxiv.org/abs/ 2411.04746
- 48. Wang, Y., Wang, L., Ma, Z., Hu, Q., Xu, K., Guo, Y.: Videodirector: Precise video editing via text-to-video models (2025), https://arxiv.org/abs/2411.17592
- 49. Wiedemer, T., Li, Y., Vicol, P., Gu, S.S., Matarese, N., Swersky, K., Kim, B., Jaini, P., Geirhos, R.: Video models are zero-shot learners and reasoners (2025), https://arxiv.org/abs/2509.20328
- 50. Wu, J.Z., Ge, Y., Wang, X., Lei, W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-a-video: One-shot tuning of image diffusion models for text-tovideo generation (2023), https://arxiv.org/abs/2212.11565
- 51. Wu, R., Gao, R., Poole, B., Trevithick, A., Zheng, C., Barron, J.T., Holynski, A.: Cat4d: Create anything in 4d with multi-view video diffusion models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 26057–26068 (2024), https://api.semanticscholar.org/CorpusID:274305973

- 52. Yang, S., Zhou, Y., Liu, Z., Loy, C.C.: Rerender a video: Zero-shot text-guided video-to-video translation (2023), https://arxiv.org/abs/2306.07954
- 53. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong,

- W., Zhang, X., Feng, G., Yin, D., Zhang, Y., Wang, W., Cheng, Y., Xu, B., Gu,
- X., Dong, Y., Tang, J.: Cogvideox: Text-to-video diffusion models with an expert transformer (2025), https://arxiv.org/abs/2408.06072

- 54. Yatim, D., Fridman, R., Bar-Tal, O., Dekel, T.: Dynvfx: Augmenting real videos with dynamic content (2025), https://arxiv.org/abs/2502.03621
- 55. Yatim, D., Fridman, R., Bar-Tal, O., Kasten, Y., Dekel, T.: Space-time diffusion features for zero-shot text-driven motion transfer (2023), https://arxiv.org/ abs/2311.17009
- 56. Ye, Z., Huang, H., Wang, X., Wan, P., Zhang, D., Luo, W.: Stylemaster: Stylize your video with artistic generation and translation (2024), https://arxiv.org/ abs/2412.07744
- 57. Zhang, D.J., Paiss, R., Zada, S., Karnad, N., Jacobs, D.E., Pritch, Y., Mosseri, I., Shou, M.Z., Wadhwa, N., Ruiz, N.: Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 2050–2062 (2024), https://api.semanticscholar.org/CorpusID:273877842

## Appendices

We highly recommend the reader to refer to the project’s website containing video data, including results and comparison videos.

### A Additional Results

- A.1 Additional results with the WAN model

Figure 1 presents additional results obtained by our method with the WAN 2.1 14B I2V model, divided into four general editing categories: Interactive insertion, interactive swap, action change, and global effects. The videos cover diverse scenarios.

- A.2 Additional results with the Hunyuan model

As DynaEdit is model-agnostic, it can be easily adapted to work with arbitrary I2V models. To illustrate its versatility, Fig. 2 shows its use with the Hunyuan 1.5 I2V model. As can be seen, DynaEdit successfully performs dynamic edits with that model, achieving good adherence to the source video as well as to the target prompt.

### B Additional Comparisons

#### B.1 Additional qualitative comparisons

In Fig. 3 we present additional qualitative comparisons against the competing methods: FlowEdit, I2V sampling, SDEdit, Editing-by-ODE-Inversion, FlowAlign and Runway Aleph.

#### B.2 VLM evaluation - table

In Tab. 1 we present the tabular version of the quantitative VLM evaluation of Fig. 7, add report several additional hyper-parameter configurations for the competing methods. The second column reports the CFG hyper-parameter configurations used for each method, and the SGA temperature parameter τ for our method (Sec. 5.1). Note that, as discussed in the main text, we employ nmax = N for all inversion-free approaches, as well as for the editing-by-inversion baseline, as this configuration is crucial for enabling the spatio-temporal modifications required by the edits.

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

RealvideoEditedEditedRealvideoEditedRealvideo

Horse circling → … + person riding

Forest… → … + mirror

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

###### Interactive Insertion – (first frame edit + text conditioned)

###### Interactive insertion – (first frame edit + text conditioned)

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Woman at the beach → … + kite and umbrella

Tulips swaying → … + dandelion blown away

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Swap & Action – (first frame edit + text conditioned)

###### Interactive insertion – (first frame edit + text conditioned)

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

EditedRealvideo

Man with backpack… → …with sprinkling waterbag

Bridge with train… → …with giant snail

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Swap & Action – (first frame edit + text conditioned)

###### Swap & Action – (first frame edit + text conditioned)

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

Three swans → … + leftmost dives

Pizza rotated in oven → Pizza catches fire

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

(Action change – prompt conditioned)

###### (Action change– text conditioned)

EditedRealvideoEditedRealvideo

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

Waterfall → … Magma waterfall

Turtle and fish… → … + fish swims away

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

(Action change – prompt conditioned)

Global effect – (first frame edit + text conditioned)

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

Bridge with train… → … during earthquake

Train passing… → … + hit by lightning at night

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

Global effect– (text conditioned)

Global effect– (first frame edit + text conditioned)

- Fig. 1: Additional Results. DynaEdit is capable of diverse dynamic editing effects. For example, on the top left we insert a mirror into the forest video, showcasing reflections reactive to the scene’s content. In the fourth row (left), we prompt the leftmost swan to dive, while preserving the other swans’ motions. On the bottom left row, a global earthquake effect is added to the bridge scene causing the moving train to collapse, all while preserving the original camera motion.

###### Realvideo

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

Woman at the beach → … + kite and umbrella

Two astronauts… → … + flag picked up

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

RealvideoEditedEdited

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Strawberries falling into water → … into viscous gel

Promenade at night→ … + fireworks

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

- Fig. 2: Additional results with the Hunyuan I2V model. DynaEdit is model agnostic and can thus potentially leverage any I2V model. The figure illustrates diverse results obtained by DynaEdit with the Hunyuan I2V model, like interactive insertion with of a Mars flag and reactive material swap (strawberries fall into gel vs. water).

Table 1: Comparison vs. competing methods (with reported hyperparameters) using VLM evaluation. Best scores are highlighted in bold, second best are underlined.

Method CFG γ, SGA Temp. τ Text Adherence Content Pres. Visual Quality

- γsrc = 2.5, γtar = 4.5, τ = 0.01 4.08 4.50 3.83

- γsrc = 2.5, γtar = 4.5, τ = 1.0 4.04 4.38 3.60

DynaEdit

- γsrc = 4.5, γtar = 8.5, τ = 0.01 4.21 4.36 3.35
- γsrc = 4.5, γtar = 8.5, τ = 1.0 4.14 4.25 3.54

γ = 18.5 3.76 3.40 1.95 γ = 7.5 3.84 3.59 2.47 γ = 13.5 3.76 3.69 2.33

FlowAlign

γsrc = 2.5, γtar = 4.5 3.85 4.07 2.87 γsrc = 4.5, γtar = 8.5 3.84 4.01 2.90

FlowEdit

γ = 4.5 3.95 3.80 3.66 γ = 8.5 3.95 3.98 3.88

I2V samp.

γsrc = 2.5, γtar = 4.5 3.53 3.38 2.59 γsrc = 4.5, γtar = 8.5 3.42 2.80 1.84

ODE Inv.

Aleph - 4.18 4.18 3.61

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

###### Realvideo

Living room zoom in → … + Cat jumping on sofa

Orange stand → … + red apple picked up

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

I2VFlowEditOursFlowAlignSDEditInversion

Content preserved ✘ Edit Adherence ✔ Quality ✘

Content preserved ✔ Edit Adherence ✘ Quality ✘

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Content preserved ✘ Edit Adherence ✔ Quality ✔

Content preserved ✘ Edit Adherence ✔ Quality ✔

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

Content preserved ✔ Edit Adherence ✘ Quality ✔

Content preserved ✔ Edit Adherence ✘ Quality ✘

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

Content preserved ✔ Edit Adherence ✘ Quality ✘

Content preserved ✔ Edit Adherence ✔ Quality ✘

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

Content preserved ✘ Edit Adherence ✔ Quality ✘

Content preserved ✘ Edit Adherence ✔ Quality ✘

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

| |
|---|

###### Aleph

[Figure 415]

Content preserved ✔ Edit Adherence ✔ Quality ✘

Content preserved ✔ Edit Adherence ✔ Quality ✘

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

Content preserved ✔ Edit Adherence ✔ Quality ✔

Content preservation ✔ Edit Adherence ✔ Quality ✔

- Fig. 3: Additional comparisons against the competing methods. On the left, a cat is inserted into a zooming-in video of a living room. It is prompted to jump on the sofa. The DynaEdit result is the only one to simultaneously adhere to the original camera motion, to add a cat that performs the proper action, and to exhibit no visible artifacts. On the right, an orange in a market stand is swapped into a red apple, and a person walks in to grab it. DynaEdit performs the required edit without visible artifacts while preserving the background (people passing by). In both examples, the other edits do not exhibit all three properties simultaneously.

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

###### Realvideo

Aerial night view of a town… → … + Heli with light

Woman at the beach → … + kite and umbrella

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

OursDynVFX

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

- Fig. 4: Qualitative comparison against DynVFX on object insertion subset. Our method meaninfgully integrates inserted objects into the scene. As visible by the inserted helicopter that shines light on the the city at night, or the kite and umbrella that react to the wind in the scene. This is in contrast to DynVFX, which showcases limited dynamical interactions between the inserted objects and the scene.

#### B.3 Comparison against DynVFX on object insertion

Here we compare our method to DynVFX, which is an object insertion method. Since it supports only insertion based effects, we compare to it only on the relevant subset of our evaluation set (20 out of 71 videos). We use the default hyperparameters in the official implementation. Note that DynVFX is based on the CogVideoX model, which processes videos of length 49 frames at 8 fps (∼6 sec. long). The videos in our dataset contain between 49-82 frames at 16 fps (∼3-5 sec. long). Therefore, for DynVFX, we go back to the original videos, from which our clips were extracted, and re-extract 49 frames at 8 fps (this results in slightly longer versions of the videos on which DynaEdit was run). Figure 4 qualitatively demonstrates that our method is superior in terms of interactive object insertion. For instance, when inserting a helicopter that shines light on a town at night DynVFX fails to model proper light interactions, while our method does so successfully. Quantitative VLM evaluations are presented in Fig. 5. These demonstrate that our method achieves a better tradeoff between text adherence, visual quality, and content preservation.

#### B.4 Comparison against FlowDirector on object swapping

Here we compare against FlowDirector, which is a localized object-level editing method, using their default reported hyperparameters. We report results only on the subset of our evaluation set that includes object swap edit instructions (17 out of 71 videos). FlowDirector’s editing pipeline leverages attention-based local object masking to localize the edits only to the desired object regions. While this improves fidelity to the source video, it naturally restricts the method to perform only structurally-aligned edits. As demonstrated in Fig. 6, our object swaps are interactive with the scene, while FlowDirector’s edit interactions are limited to

Content Preservation vs. Text Adherence

Content Preservation vs. Visual Quality

ContentPreservationScore →

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Adherence to Text Score → Visual Quality Score →

- Fig. 5: Comparison against DynVFX on object insertion subset. While DynVFX achieves a good visual quality score, the inserted objects are not able to affect the scene’s outcomes. Due to this, they receive a lower text adherence score.

the close vicinity of the swapped object. Quantitative results are presented in Fig. 7, where we show a more favorable balance between quality, text adherence, and loyalty to the source video.

#### B.5 Per-category quantitative evaluation

- In Fig. 8 we present a per-edit-category split of the VLM and user study results that were reported in Fig. 7 of the main text, across the four categories in our evaluation set: Insertion, Swap, Action change, and Global effects. As seen, DynaEdit outperforms the training-free methods on all categories across the three evaluation criteria: visual quality, content preservation, and adherence to text. As for the trained Runway Aleph model, DynaEdit exhibits better performance in the categories of dynamic insertion and object swap, comparable performance in the category of action change, and worse performance only in the category of global effects.

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

###### Realvideo

Pitcher with red juice… → …green juice + reaction

A train traversing a bridge → A giant snail …

FlowDirectorOurs

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

- Fig. 6: Qualitative comparison against FlowDirector on object swap subset. On the left example, the source video depicts a person squeezing a lemon into a pitcher with red juice. The edit requires changing the juice’s color to green and adding a sudden chemical reaction with the lemon. Our method performs the edit successfully, depicting a realistic reaction, while FlowDirector fails to add a meaningful reaction effect due to the localized nature of it’s edits. On the right, a train is edited into a giant snail. The edit prompt requires integrating an object that is not strictly bound to the source object’s geometry (the train), which our method successfully follows. However, FlowDirector struggles in performing the required edit, due to the limitations imposed by it’s locality.

Content Preservation vs. Text Adherence

Content Preservation vs. Visual Quality

Adherence to Text Score → Visual Quality Score →

ContentPreservationScore →

[Figure 458]

- Fig. 7: Comparison against FlowDirector on object swap subset. Due to the local nature of FlowDirector’s edits, it struggles to follow prompts that require dynamic interaction with the scene, and thus generates unnatural videos. This is evident by the low text and visual quality scores. FlowDirector also performs worse in terms of content preservation, as oftentimes it struggles to maintain the swapped object’s identity, or the background in close proximity to the object (see qualitative comparisons in Fig. 6).

###### (a) VLM Evaluation (b) User Study

###### Category: Insertion

[Figure 459]

Content Preservation vs. Text Adherence

Content Preservation vs. Visual Quality

Visual Quality

Content Preservation

Adherence to Text

[Figure 460]

ContentPreservationScore →

71.0%

82.1%

74.2%

67.1%

85.8%

79.7%

73.3%

79.4%

77.9%

Adherence to Text Score → Visual Quality Score →

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

Ours Aleph FlowAlign FlowEdit I2V Sample ODE Inversion SDEdit Vs. Aleph Vs. I2V Sample Vs. FlowEdit

[Figure 468]

Visual Quality

Content Preservation

Adherence to Text

Category: Swap

[Figure 469]

ContentPreservationScoreContentPreservationScoreContentPreservationScore → → →

90.0%

82.8%

77.8%

73.9%

84.4%

75.0%

81.1%

83.3%

67.2%

Adherence to Text Score → Visual Quality Score →

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

Ours Aleph FlowAlign FlowEdit I2V Sample ODE Inversion SDEdit Vs. Aleph Vs. I2V Sample Vs. FlowEdit

[Figure 477]

Category: Action

Visual Quality

Content Preservation

Adherence to Text

[Figure 478]

50.9%

70.9%

93.9%

46.5%

77.5%

94.2%

58.8%

71.2%

91.2%

Adherence to Text Score → Visual Quality Score →

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

Ours Aleph FlowAlign FlowEdit I2V Sample ODE Inversion SDEdit Vs. Aleph Vs. I2V Sample Vs. FlowEdit

[Figure 486]

Category: Global effects

Visual Quality

Content Preservation

Adherence to Text

[Figure 487]

40.4%

85.7%

75.3%

46.3%

85.4%

73.2%

28.0%

65.9%

69.0%

Adherence to Text Score → Visual Quality Score →

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

Ours Aleph FlowAlign FlowEdit I2V Sample ODE Inversion SDEdit Vs. Aleph Vs. I2V Sample Vs. FlowEdit

- Fig. 8: VLM Evaluation and user study results per-edit-category. (a) Percategory VLM evaluations reveal that DynaEdit achieves a favorable balance between source content preservation, adherence to target text, and visual quality compared to training-free methods. Against the trained Aleph model, we achieve comparable results on most categories, with a disadvantage only in visual quality for the global effects category. (b) Per-category user study results show that on the insertion and swap categories DynaEdit is favorable compared to the leading competing methods, including Aleph. For the action and global effects categories, DynaEdit is preferred over the training-free approaches, and is mostly comparable to Runway Aleph, losing only in text adherence on the global effects category.

[Figure 495]

###### 28 V. Kulikov et al.

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

###### Realvideo

[Figure 502]

A turtle and a fish … → … + fish swims away

A forest fire → … + a fire tornado begins

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

w/oSGA

w/ANC w/SGA

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

w/ANC

- Fig. 9: Ablation of the SGA mechanism. On the left, we edit a source video depicting synchronous motion of a sea turtle and a fish under it. The edit prompts the fish to leave the scene without changing the turtle’s motion. When using the SGA mechanism, the motion of the turtle stays the same as in the source, while with regular averaging the motion changes. On the right, the edit requires adding a fire-tornado to a video of a forest fire. While strong changes are required to achieve this manipulation, the SGA mechanism keeps the original camera motion intact, as opposed to the naive averaging approach (white arrows).

### C Ablations

#### C.1 Importance of Similarity Guided Aggregation (SGA)

In Fig. 9 we qualitatively ablate the effectiveness of our SGA mechanism by comparing it to the regular averaging mechanism of FlowEdit. In both experiments the number of aggregated velocities is dictated by the same nSGAi scheduler (as reported in Sec. 6.1). As seen, the SGA mechanism is critical for keeping good alignment with the source video in terms of both object dynamics and camera motion.

#### C.2 Importance of Annealed Noise Correlation (ANC)

- In Fig. 10 we qualitatively ablate the effectiveness of ANC. We do so by comparing our full method with a version that employs i.i.d. noise (as in FlowEdit) instead of ANC. As can be seen, the vanilla i.i.d. scheduler leads to prominent high frequency jitter in the edited videos, whereas our ANC allows for better high-frequency fidelity.

In Fig. 11, we show the effect that the noise schedule has on the correlations between consecutive edit velocities Vt∆. The plots show the cosine similarities between each pair of consecutive noise maps and each pair of consecutive velocity difference. As demonstrated, when our noise correlation (orange lines) is employed, the resulting velocities (blue lines) are more correlated compared to the i.i.d. case. This behavior becomes more prominent towards the later timesteps of the generation.

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

###### Realvideo

Train passing through the forest → … + hitting a yellow paint bucket

Two astronauts → … + flag picked up

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

w/oANC w/SGA

w/SGA

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

w/ANC

- Fig. 10: Ablation of the ANC mechanism. On the left, a source video of a train passing through a forest is manipulated by adding a yellow paint bucket to the train’s path. In the i.i.d. noise case, while the result adheres to the original motion (due to the SGA mechanism) it exhibits severe high-frequency jitter artifacts. In contrast, with ANC the edited video exhibits good fidelity in the high frequency details, evident by the visible paint particles. On the right, a flag is inserted into a video depicting astronauts walking on Mars. The edit prompt requires one astronaut to pick up the flag, while the other one to continue walking. In the i.i.d. noise case, the flag dissolves towards the end of the video. On the other hand, our method maintains the high-fidelity dynamics of the flag throughout the whole video.

#### C.3 Annealed noise correlation schedules

- In Fig. 12 we ablate the effect of different noise correlation schedule choices (see right side of the figure). The first schedule is a non-Markovian increasing correlation schedule, where a different random noise map sampled at each timestep is

mixed with a fixed noise map wconst. The correlation coefficients at

i

are chosen to be monotonically increasing as described in Sec. 6.1. The schedule is nonMarkovian since a global map is used to correlate all steps together (to an extent dictated by at

i

). In this case the results showcase ghosting artifacts. The second schedule is a Markovian Increasing correlation schedule, where each noise map is mixed with the previously sampled noise, but with a monotonically decreasing schedule (the exact opposite of the decreasing schedule we use in our default setting). The edited video contains jitter artifacts, showcasing the importance of introducing correlations in the later steps of the edit path, and strengthening our claim that high-frequency jitter stems from low edit correlations at later timesteps. Finally, we present the result obtained with the Markovian increasing schedule we use in our method. As evident, the result is free of jitter or any other visual artifacts.

C.4 SGA similarity function ablation

- In Fig. 13 we discuss the effect of different similarity functions on the SGA mechanism (Sec. 5.1). Specifically we compare the cosine similarity loss with the MSE loss (see right side of figure). As evident in the edits, sometimes the

i.i.d. 𝑤 Our 𝑤 (ANC)

[Figure 534]

[Figure 535]

- Fig. 11: Effect of noise correlations on edit velocity. The left pane shows the effect of using i.i.d. noise on the correlations between consecutive edit velocities. On the right, our annealed noise correlation schedule (ANC) is employed, which induces stronger correlations between consecutive noises (orange), which leads to higher correlations between consecutive edit velocities (blue) compared to the i.i.d. case.

cosine similarity helps with adherence to the source when delicate objects with fine motion are present. For example, it allows preserving the paintbrush in the painting example when turning it into a pencil. However, in some cases, there is no significant difference between these loss functions. This is evident by the edit result showcasing the transformation of a bird into a phoenix.

#### C.5 Robustness to edit prompt choice

In this section we show that the precise phrasing of the user-provided sourceand target-prompts has a marginal effect on the results (as long as the phrasing conveys the same meaning). This shows that our proposed method is robust to the choice of prompt, and does not require tiresome prompt tuning. In Fig. 14 we show two different edit scenarios, with five different source-target pairs describing the same edit in different ways. As can be seen, the edit outcomes look similar regardless of the different prompting styles and lengths, strengthening our claim. We obtained these edits using Gemini 3 pro, by feeding it one source-target pair and asking it to create several different pairs with the same meaning, but with varying text styles and levels of conciseness. The source-target pairs are given in the following table.

Table 2: Prompts used for prompt robustness ablation

Source Prompt Target Prompt

- Scenario 1: Astronauts Continued on next page

Table 2 – continued from previous page Source Prompt Target Prompt

Tracking shot of two astronauts traversing the Martian landscape. The terrain is a desert with mountains in the distance.

Tracking shot of two astronauts traversing the Martian landscape. The terrain is a desert with mountains in the distance. The astronaut on the right hand side grabs a flag labeled ’MARS’ mid-stride. The second astronaut remains oblivious to this action.

Show two astronauts walking together on Mars. There are mountains and red sand in the background, and the camera is following behind them.

Show two astronauts walking together on Mars. There are mountains and red sand in the background, and the camera is following behind them. Suddenly, the astronaut on the right spots a flag that says ’MARS’ and picks it up without stopping. The other astronaut doesn’t notice it happening.

Two astronauts walking on Mars. Desert setting, mountain backdrop. Camera follows subjects.

Two astronauts walking on Mars. Desert setting, mountain backdrop. Camera follows subjects. Right astronaut picks up a ’MARS’ flag while walking. Left astronaut is unaware.

A pair of space travelers stroll across the dusty red dunes of Mars, flanked by towering mountains. The viewpoint moves with them as they walk.

A pair of space travelers stroll across the dusty red dunes of Mars, flanked by towering mountains. The viewpoint moves with them. The traveler on the right reaches out and snatches a flag reading ’MARS’ from the ground without halting. Their companion continues walking, completely unaware.

Generate a clip of two astronauts on Mars. Background: Mountains/Desert. Camera: Follow movement.

Generate a clip of two astronauts on Mars. Background: Mountains/Desert. Camera: Follow movement. Action: The astronaut on the right must pick up a ’MARS’ flag midwalk. Condition: The other astronaut must not react to the flag.

###### Scenario 2: Aerial Night View

A bird’s-eye view of a city at night, dotted with faint orange streetlights.

A bird’s-eye view of a city at night, dotted with faint orange streetlights. A chopper hovers overhead, projecting a powerful spotlight onto the town center that illuminates the roofs.

Looking straight down on a dark town, illuminated only by scattered, soft orange glows.

Looking straight down on a dark town, illuminated only by scattered, soft orange glows. A helicopter is present in the air, beaming a bright light down into the middle of the town, making the rooftops visible.

An overhead perspective of a village during the night; the area is speckled with weak orange lights.

An overhead perspective of a village during the night; the area is speckled with weak orange lights. There is a helicopter flying above, directing a harsh beam of light at the center of the village, lighting up the tops of the buildings.

Top-down aerial shot of a town at night, with dim orange lamps dispersed throughout the streets.

Top-down aerial shot of a town at night, with dim orange lamps dispersed throughout the streets. A helicopter flies through the sky, shining a brilliant searchlight on the town square, brightening the rooftops.

A high-altitude view looking down at a residential area at night, where dim orange lighting is spread across the town.

A high-altitude view looking down at a residential area at night, where dim orange lighting is spread across the town. A helicopter circles in the sky, casting a bright light in the middle of the town which illuminates the building roofs.

[Figure 536]

[Figure 537]

[Figure 538]

Realvideo

A camel eating a bush-> … + giraffe interrupts Noise Correlation schedule

Markovian

[Figure 539]

[Figure 540]

[Figure 541]

Increasing Markovian

Non-

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

Decreasing Markovian

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

Increasing

(Ours)

[Figure 550]

- Fig. 12: Ablation of different annealing noise correlation schedules. Three different noise correlation schedulers are compared on a video of a camel in a desert (first row). The edit prompt requires inserting a giraffe that interacts with the camel during its meal. The second row shows the result of employing non-Markovian correlations,

with the same increasing coefficients ati as used in our method (see Sec. 6.1). The result showcases visible ghosting artifacts, as seen by the two giraffe necks. The third row shows the result of Markovian correlations (as in our method) but with a mirrored correlation schedule (lower correlations at later timesteps). The result exhibits high-frequency jitter artifacts. The fourth row shows Markovian correlations with an increasing schedule, which is the one employed by our method. As seen, the results are artifact- and jitter-free.

#### C.6 Image-to-video conditioning

Image conditioning during video generation is a highly effective complement to the regular text conditioning, allowing more user control over the initial scene configuration, subject identities, lighting, and other things that are hard to convey with text alone. The same holds when performing editing. With I2V based editing the user can specify a target edited first frame, obtained via any image editing method that can be utilized to enforce it in the edited video. We find that this mechanism is crucial for the success of our method, where we allow edits starting from very coarse features that correspond to features like color and lighting. This can be seen in Fig. 15, where we prompt a horse to jump over an obstacle instead of running in a straight line. We employ FlowEdit with the WAN2.1 T2V model, and also with the I2V model conditioned on the first frame of the source video for both source and target image conditionings. As can be seen, when visualizing intermediate edit steps V ∆, the T2V variant changes colors and background, while the I2V variant confines changes mainly to the horse itself without modifying colors or much of the background. This is due to the reduced edit uncertainty provided by the information in the first frame conditioning.

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

Realvideo

SGA similarity function

A brush watercolor painting … -> A pencil sketch…

A bird on a branch… -> A phoenix setting fire to a branch…

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

OursL2sim

sim 𝑋,𝑌 = −MSE 𝑋,𝑌

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

sim 𝑋,𝑌 = cosim 𝑋,𝑌

- Fig. 13: Ablation of SGA similarity function choices. Two SGA similarity functions are compared: negative MSE loss and our cosine similarity loss (see right pane). Edits are performed on a video of a paintbrush painting edited into a pencil sketch, and a video of a bird swapped into a flaming phoenix that sets fire to a branch. For the painting example, the cosine similarity function performs better, capturing the fine spatio-temporal motion of the paintbrush, as evident by the aligned motion of the pencil in the edit. This is in contrast to the MSE variant, where the pencil’s location is not aligned with the paintbrush. For the phoenix example, there is little difference between the two loss functions.

### D DynaEdit Hyperparameter Choices

#### D.1 Hyperparameter groups

Following the discussion in Sec. 6.1, here we detail the four hyperparameter configurations we explored. In Fig. 16 we qualitatively demonstrate the effects of these hyperparameter groups on the final edit. Importantly, we find that for the task of general text-based editing, different edit strengths lead to different plausible text-adherent outcomes, and choosing the extent of deviation from the source is for the user to decide. As reported in the main text, we propose four sets of configurations for the source and target CFG parameters (denoted here γsrc and γtar) and for the SGA temperature τ on the WAN 2.1 14B I2V model:

- (1) γsrc = 2.5,γtar = 4.5,τ = 1.0: This small CFG configuration supports edits with smaller, more precise, modification requirements. The higher temperature allows more deviation from the source’s coarse features (allowing more structural deviations, as well as camera motion).
- (2) γsrc = 2.5,γtar = 4.5,τ = 0.001: This small CFG configuration supports edits with smaller, more precise modification requirements. The lower temperature enforces stronger alignment with the source video (e.g. limiting motion changes of big objects, or camera motion).
- (3) γsrc = 4.5,γtar = 8.5,τ = 1.0: This higher CFG configuration supports edits that require larger deviation from the source video to adhere to the target prompt (such as manipulating trajectories of big objects) at the cost of weaker alignment with the input. The higher temperature allows more deviation from the source video (enabling e.g. modification of camera motion).
- (4) γsrc = 4.5,γtar = 8.5,τ = 0.001: This higher CFG configuration supports edits that require larger deviation from the source video to adhere to the

Realvideo Prompt

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

Astronauts walking…-> … + flag picked up Aerial view of town -> … + Heli illuminating

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

Prompt

iter.#1 Prompt

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

iter.#3 Prompt

iter.#2

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

iter.#5 Prompt

iter.#4

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

- Fig. 14: Ablation of prompt robustness. Each row depicts the editing result obtained with a different variant of the text prompts (see Sec. C.5). As can be seen, the precise phrasing of the prompt has little effect on the result.

target prompt (such as changing trajectories of big objects) at the cost of weaker alignment with the input. The lower temperature enforces stronger alignment with the source video (e.g. limiting motion changes of big objects, or camera motion).

For the qualitative results in the paper, as well as the user study, we use the four hyperparameter sets in the following ways. For edits that require subtle global effects that can alter the source’s coarse features (like color or motion) we found that hyperparameter set (1) is most favorable. For example, set (1) is used in the train-hit-by-lightning example in Fig. 1. For object insertion, action change, or global effects, when the objects affected are small and a strong alignment with the source video is required, we use hyperparameter set (2). An example is the insertion of the Mars flag in Fig. 5. For edits that prompt for big action changes, or insertion of objects that strongly influence the outcome of the video, we found it useful to use hyperparameter set (3). An example is the horse-jumping-over-obstacle edit in Fig. 1. For objects that affect many pixels, but do not require strong deviation from source motion, we found it beneficial to use hyperparameter set (4). An example is the fireworks edit in Fig. 5, where the people in the background preserve their motion. All in all, different hyperparamters could lead to plausible results on the same edit prompt,

[Figure 605]

[Figure 606]

[Figure 607]

Input FlowEdit(T2V)

[Figure 608]

Horse circling → … + jumping over obstacle

[Figure 609]

𝑛 =𝑁 FlowEdit(I2V)

[Figure 610]

𝑛 =𝑁

- Fig. 15: Importance of I2V conditioning. Performing strong edits (nmax = N) without an image condition for the source and target branches results in random color changes, which are evident in the edit result in the middle row. Using an Image-toVideo model conditioned on the first frame of the input video results in preserved colors, as seen in the bottom row.

and are usually up to the user’s personal preference on the tradeoff between loyalty to the source and adherence to the target text. For the user study, we used the same hyperparameter groups as for the qualitative results (see Tab. 3).

#### D.2 Parameters used for the qualitative results

In Tab. 3 we report the hyperparameter groups (from Sec. D.1) used to obtain the qualitative results for the figures in the paper.

### E Creating Instruction Prompts for DynVFX and Runway Aleph

As discussed in Sec. 6, since our evaluation set consists of source-target prompt pairs, and DynVFX and Runway Aleph require an instruction prompt, we create an analogous instruction prompt set using a VLM. Specifically, we query Gemini 3 Pro with the prompt “Given a source prompt depicting a source video and a target prompt describing a desired edited video, analyze the differences between them, and provide an analogous instruction prompt that depicts the edit that needs to be performed.”. We then manually go over the resulting 71 instruction prompts and make sure that they faithfully convey the desired edit. Several examples are provided in Tab. 4

### F VLM Evaluation Protocol

For the VLM evaluation reported in Sec. 6, we use Google’s Gemini 3 Pro. We provide the VLM the source and edited videos and ask it to rate the edited

Table 3: Hyperparameter groups for qualitative results.

|Location<br><br>|Video description<br><br>|Parameter group|
|---|---|---|
|Fig. 1<br><br>|“Horse jumping”<br><br>|large object, strict motion (3)|
|Fig. 1<br><br>|“Cat playing”|action change, large motion change (3)|
|Fig. 1<br><br>|“Billiard ball entering”<br><br>|action change, limited motion change (4)|
|Fig. 1<br><br>|“Nighttime beach”<br><br>|global effect, limited motion change (4)|
|Fig. 5<br><br>|“Mars flag”|small object, limited motion change (2)|
|Fig. 5<br><br>|“Heli with light”|small object, limited motion change (2)|
|Fig. 5<br><br>|“Red apple picked up”<br><br>|large object, large motion change (3)|
|Fig. 5<br><br>|“Juice chemical reaction”|small object, limited motion change (2)|
|Fig. 5<br><br>|“Fisherman catches fish”|action change, limited motion change (4)|
|Fig. 5<br><br>|“Pizza taken out”|action change, limited motion change (4)|
|Fig. 5<br><br>|“Hiking during sandstorm”<br><br>|global effect, large motion change (3)|
|Fig. 5<br><br>|“Fireworks”<br><br>|global effect, limited motion change (4)|
|Fig. 1|“Forest with mirror”|small object, limited motion change (2)|
|Fig. 1<br><br>|“Woman on horse”|small object, limited motion change (2)|
|Fig. 1<br><br>|“Beach with kite and umbrella”<br><br>|medium objects, limited motion change (4)|
|Fig. 1<br><br>|“Dandelion blown away”|small object, limited motion change (2)|
|Fig. 1|“Sprinkling waterbag”<br><br>|small object swap, limited motion change (2)|
|Fig. 1|“Giant snail”<br><br>|small object swap, limited motion change (2)|
|Fig. 1|“leftmost swan dives”|subtle action change, limited motion change (2)|
|Fig. 1<br><br>|“Pizza catches fire”<br><br>|action change, motion change (3)|
|Fig. 1|“Fish swims away”<br><br>|subtle action change, limited motion change (2)|
|Fig. 1<br><br>|“Magma waterfall”<br><br>|global effect, motion change (3)|
|Fig. 1<br><br>|“Bridge earthquake”|global effect, motion change (3)|
|Fig. 1<br><br>|“Train hit by lightning”|subtle global effect, motion change (1)|
|Fig. 6<br><br>|“Strawberry and feather”<br><br>|small object, high S-T preserv. (2)|
|Fig. 3<br><br>|“Cat jumping on sofa”|small object, high S-T preserv. (2)|
|Fig. 3<br><br>|“Red apple picked up”|large object, large motion change (3)|

result on a scale of 1 to 5 in three aspects. The first is adherence to the source video – the VLM should analyze the edit requirement (what should not change based on the source-target pair or the edit instruction) and see that unnecessary changes are kept to a minimum. This includes changes in camera trajectories, motion of objects, etc. The second aspect is loyalty to the edit – the VLM should understand what is necessary to fulfill the edit requirement, and check if the result indeed does so. This includes checking for logical interactions in case of object insertion, or logical outcomes if an action change is required, etc. The third aspect is visual quality. This includes checking for visual artifacts, like blur, flickeriness, etc. There is a slight difference when querying the VLM for the edit’s adherence to the edit prompt between methods that accept a sourcetarget text pair, and methods that require an edit instruction. All compared methods except for DynVFX and Runway Aleph fall into the first category. For the source-target pairs we first ask the VLM to analyze the differences between the prompts to understand the edit requirement, and then use it to evaluate the edit. For the instruction based methods, we ask the VLM to evaluate the result based on the stated edit requirement. In the following, we give a snippet of the evaluation protocol fed to the VLM in the case of source-target (the instruction setting is similar):

Table 4: Examples for source-target prompt to instruction prompt conversion.

|Source prompt|Target prompt<br><br>|Instruction prompt|
|---|---|---|
|“Two astronauts are walking on Mars. Desert, mountains in the background. The camera follows them.”<br><br>|“Two astronauts are walking on Mars. Desert, mountains in the background. The camera follows them. The astonaut on the right approaches a flag spelling MARS and picks it up mid walk. The other astonaut is unaware of the flag.”|“Make the astronaut on the right pick up a flag spelling MARS while walking. Do not change the other astronaut.”|
|“A tracking camera shot of a forest filled with trees. The camera is moving to the right.”<br><br>|“A tracking camera shot of a forest filled with trees. The camera is moving to the right. A big mirror is in the middle of the forest.”<br><br>|“Place a large mirror in the middle of the forest.”|
|“A scenic shot of a stone train bridge in the mountains. A red train is crossing the bridge from left to right. The camera is slowly moving forward.”<br><br>|“A scenic shot of a stone train bridge in the mountains. A giant snail is crossing the bridge from left to right, leaving a trail of ooze. The camera is slowly moving forward.”<br><br>|“Replace the train with a giant snail crossing the bridge and leaving a trail of ooze.”|
|“A static camera shot of yellow tulips swaying in the wind. Close-up shot.”|“A static camera shot of yellow tulips and a dandelion swaying in the wind. Close-up shot. Suddenly a hand reaches out and picks the middle tulip, leaving the others intact.”<br><br>|“Have a hand appear and pick the middle tulip.”|
|“A static shot of a Ferris wheel in motion. The wheel is spinning, and the spokes are moving. Sky in the background.”|“A static shot of a Ferris wheel in motion. The wheel is spinning, and the spokes are moving. Sky in the background. Suddenly, gray clouds begin to form in the sky, turning the sky gray.”<br><br>|“Make gray storm clouds gather in the sky.”|

#### System Prompt: Instructions

CONTENT_PRESERVATION_INSTRUCTIONS = """ ### Task Definition You will be provided with: SOURCE_VIDEO SOURCE_PROMPT - a text description of SOURCE_VIDEO TARGET_VIDEO TARGET_PROMPT - a text description of TARGET_VIDEO

You need to carefully analyze the difference between SOURCE_PROMPT and TARGET_PROMPT

to understand what should be the difference between SOURCE_VIDEO and TARGET_VIDEO.

TARGET_VIDEO should be a minimal edit of SOURCE_VIDEO to make it adhere to TARGET_PROMPT. Elements in SOURCE_VIDEO that are not required to change to match TARGET_PROMPT should remain the same in TARGET_VIDEO.

As an experienced evaluator, your task is to evaluate how well the edited video TARGET_VIDEO is consistent with SOURCE_VIDEO, according to the scoring criteria

.

### Scoring Criteria Unless a modification is necessary to align with TARGET_PROMPT, the following

criteria should be met:

- 1. Objects: Determine if the elements and subjects in SOURCE_VIDEO are presented in TARGET_VIDEO.
- 2. Alignment - the content of TARGET_VIDEO should align with the content of SOURCE_VIDEO both spatially and temporally.

- 3. Motions and Actions: The motions and actions in TARGET_VIDEO should be as similar to the ones in SOURCE_VIDEO as possible.
- 4. Camera Motion: The camera trajectory in should be identical to the one in SOURCE_VIDEO.

Do not take into consideration visual quality. Ignore visual artifacts. ### Scoring Range Based on these criteria, a specific integer score from 1 to 5 can be assigned to

determine the level of semantic consistency:

Very Poor (1): No consistency. TARGET_VIDEO completely and unneededly deviates from SOURCE_VIDEO, one could not understand from watching TARGET_VIDEO that its source was SOURCE_VIDEO.

Poor (2): Weak consistency. TARGET_VIDEO resembles SOURCE_VIDEO semantically (same types of objects, motion, etc) but the visual content is not consistent. Fair (3): Moderate consistency. TARGET_VIDEO resembles SOURCE_VIDEO to an extent but lacks several important details or contains some inaccuracies. Good (4): Strong consistency. TARGET_VIDEO accurately depicts most of the information from the SOURCE_VIDEO with only minor omissions or inaccuracies.

Excellent (5): Near-perfect consistency. TARGET_VIDEO remains similar to SOURCE_VIDEO with high precision and detail, with no modification other that what is necessary to adhere to TARGET_PROMPT.

### Output format Provide only a score in the range of 1-5, and nothing else.

## Inputs SOURCE_PROMPT: {source_prompt} TARGET_PROMPT: {target_prompt} """

TEXT_ADHERENCE_INSTRUCTIONS = """ ### Task Definition You will be provided with a VIDEO, and text PROMPT describing its content. As an experienced evaluator, your task is to evaluate the semantic consistency

between VIDEO and PROMPT, according to the scoring criteria.

### Scoring Criteria When assessing the semantic consistency between an image and its accompanying PROMPT

, it is crucial to consider how well the visual content of the VIDEO aligns with the textual description in PROMPT. This evaluation can be based on several

key aspects:

- 1. Relevance: Determine if the elements and subjects presented in the VIDEO directly relate to the core topics and concepts mentioned in the PROMPT. The VIDEO

should reflect the main ideas or narratives described.

- 2. Accuracy: Examine the VIDEO for the presence and correctness of specific details mentioned in the PROMPT. This includes the depiction of particular objects, settings, actions, or characteristics that the PROMPT describes.
- 3. Completeness: Evaluate whether the VIDEO captures all the critical elements of the PROMPT. The VIDEO should not omit significant details that are necessary for the full understanding of the PROMPT’s message.
- 4. Context: Consider the context in which the PROMPT places the subject and whether the VIDEO accurately represents this setting. This includes the portrayal of the appropriate environment, interactions, and background elements that align with the PROMPT.

### Scoring Range Based on these criteria, a specific integer score from 1 to 5 can be assigned to

determine the level of semantic consistency: Very Poor (1): No correlation. The VIDEO does not reflect any of the key points or details of the PROMPT. Poor (2): Weak correlation. The VIDEO addresses the PROMPT in a very general sense but misses most details and nuances. Fair (3): Moderate correlation. The VIDEO represents the PROMPT to an extent but lacks several important details or contains some inaccuracies. Good (4): Strong correlation. The VIDEO accurately depicts most of the information from the PROMPT with only minor omissions or inaccuracies.

Excellent (5): Near-perfect correlation. The VIDEO captures the PROMPT’s content with high precision and detail, leaving out no significant information.

### Output format Provide only a score in the range of 1-5, and nothing else.

## Inputs PROMPT: {target_prompt} """

VISUAL_QUALITY_INSTRUCTIONS = """" ### Task Definition You will be provided with a VIDEO. As an experienced evaluator, your task is to evaluate the visual quality of the

video, according to the scoring criteria. ### Scoring Criteria

- 1. Bluriness: Is the video sharp or blurry?
- 2. artifacts: Are there artifacts in the video?
- 3. Flickerness: Is the video temporally smooth or exhibit temporal flickering?
- 4. temporal consistency: Is the video temporally consistent or do objects unreasonably change in time? unnatural motions are ok but not changes to the objects.

### Scoring Range Based on these criteria, a specific integer score from 1 to 5 can be assigned to

determine the level of visual quality: Very Poor (1): the video suffers from severe artifacts, blurriness, flickeriness, or temporal inconsistencies. Poor (2): the video suffers from many artifacts, blurriness, flickeriness, or temporal inconsistencies. Fair (3): the video has noticable artifacts, blurriness, flickeriness, or temporal inconsistencies but overall quality is ok. Good (4): the video has minor artifacts, blurriness, flickeriness, or temporal inconsistencies but overall quality is good. Excellent (5): the video has no artifacts, blurriness, flickeriness, or temporal inconsistencies. It can pass as a legit video.

### Output format Provide only a score in the range of 1-5, and nothing else.

### G User Study

As reported in the main text, to further evaluate the effectiveness of our proposed video editing method, we conducted a comprehensive user study comparing our approach against leading competing methods. The study was conducted via Google Forms. A snippet of the user study is shown in Fig. 17.

#### G.1 Study design and protocol

The study was designed as a side-by-side comparison to assess the subjective quality of the edited videos. Participants were presented with 18 distinct video sets. Each set consisted of a Source Video as a reference, followed by two edited results generated by our method and the baseline, labeled randomly as Method A and Method B to avoid bias. Importantly, the user is given an edit prompt that depicts the required difference between the source and target videos, instead

of the full source-target prompt pairs. This was done to reduce the amount of redundant text that the user has to go through, and allow focusing on the edit requirement itself. We obtain the edit prompt by stripping the target prompt from text that is present in both source and target prompts. For example, for the source-target pair: source “Two astronauts are walking on Mars. Desert, mountains in the background. The camera follows them.”, target “Two astronauts are walking on Mars. Desert, mountains in the background. The camera follows them. The astonaut on the right approaches a flag spelling MARS and picks it up mid walk. The other astonaut is unaware of the flag.”, the edit prompt would be “The astonaut on the right approaches a flag spelling MARS and picks it up mid walk. The other astonaut is unaware of the flag.”.

#### G.2 Evaluation criteria

For each comparison, participants were asked to evaluate the results based on three key dimensions:

- – Visual Quality: Assessment of technical fidelity, including the presence of artifacts, flickering, and overall realism.
- – Source Loyalty: The degree to which the edited video preserves the background, structure, and original motion of the source.
- – Target Adherence: The effectiveness of the method in executing the specific edit described in the provided ‘Target Prompt’.

### H Limitations

Open source video models, like the WAN model we used in our experiments, have limited capacity, which sometimes causes our edits to exhibit sub-optimal visual quality or unrealistic temporal interactions. This is exemplified Fig. 18. The top row shows results with limited visual quality. For example, on the left, the man is being chased by people with distrorted facial features, and on the right, the monster emerging from the water looks more like a giant plant. The bottom row shows examples of unrealistic interactions. For example, on the left, the man is taking his hat off during the sandstorm and reveals another hat underneath, and on the right, the palm trees remain static as in the source video despite the introduction of stormy weather.

An additional limitation of our method is that, in some cases, achieving a favorable edit depends on the choice of hyperparameter configuration (as discussed in App. D.1), as these parameters control a tradeoff between source preservation, edit adherence, and sometimes visual quality. Figure 16 shows several examples.

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

Input

A turtle and a fish swimming→ … fish leaves

A horse circling → … jumping over an obstacle

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

(2)(1)(3)

|[Figure 623]<br><br>[Figure 624]<br><br>[Figure 625]|
|---|

[Figure 626]

[Figure 627]

[Figure 628]

|[Figure 629]<br><br>[Figure 630]<br><br>[Figure 631]|
|---|

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

(4)

- Fig. 16: Effect of Hyperparameters. Depending on the prompt requirement, hyperparameter configurations can sometimes be crucial to strike a favorable balance between edit adherence and loyalty to the source. On the left, the horse is prompted to jump over an obstacle. Since the horse takes up many pixels in each frame, a strong CFG is preferred (parameter sets (3),(4)). In the other cases, the edit is too weak to properly edit the horse, resulting in visual artifacts. Among parameter sets (3),(4), it can be seen that parameter set (4) (with the lower temperature) induces a strong adherence to the input video’s coarse features, hindering the ability to change the horse’s original trajectory. So in this case parameter set (3) (marked with green frame), which uses a higher temperature, is preferable. For the turtle and fish example, where the prompt requests to change the action of the smaller fish, a weaker edit is enough to adhere to the target. So here parameter set (2) (marked with green frame), which uses lower CFG and lower SGA temperature, suffices. In the other cases, the motion of the turtle changes as well. While these options are plausible in terms of text adherence, they needlessly deviate from the input video compared to the low CFG settings (parameter sets (2),(3)).

Instruction Page

Video Editing User Study 25/02/2026, 12:04

Video Editing User Study

Welcome! The purpose of this study is to evaluate new video editing methods Total time is approximately 10-20 minutes. Thank you for your help!

Given a 𝙎𝙤𝙪𝙧𝙘𝙚 𝙫𝙞𝙙𝙚𝙤 and a 𝗣𝙧𝙤𝙢𝙥𝙩 depicting a desired modiHcation to the video, the goal is to generate a new video that follows the prompt while remaining as similar as possible to the source video (no unneeded changes).

- 1. 𝙎𝙤𝙪𝙧𝙘𝙚 𝙫𝙞𝙙𝙚𝙤 (Top)
- 2. 𝗣𝙧𝙤𝙢𝙥𝙩 (Top)
- 3. 𝘼 (Middle)
- 4. 𝙊𝙪𝙩𝙥𝙪𝙩 𝙑𝙞𝙙𝙚𝙤 𝘽 (Bottom)

You will be presented with 18 video comparisons. Each comparison consists of:

GUIDELINES: For each comparison, you will be asked to select which output video is preferred in terms of:

- • Visual Quality: Which video has better visual quality? (less artifacts, less ]ickering, etc...)
- • Adherence to the source video: Which video better preserves the content of the source video?
- • Adherence to target prompt: Which video better executes the speciHc edit described in the 'Target Prompt'?

###### vladimir.haifa@gmail.com Switch account

Questions

Not shared

[Figure 641]

[Figure 642]

Next Clear form

This content is neither created nor endorsed by Google. - Contact form owner - Terms of Service - Privacy Policy Does this form look suspicious? Report

https://docs.google.com/forms/d/e/1FAIpQLScdkECNtg67XyvAqC9z3Lifs64E2S5bFY1qhxUc_ZOFQltjQg/viewform Page 1 of 1

- Fig. 17: User study snippet. The first-page instruction is given at the top. The videos and following question structure are given in the bottom.

Suboptimal quality

Realvideo

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

Person hiking…-> … + chased by people

Waterfall -> … + monster emerges

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

RealvideoEditedEdited

Unrealistic interactions

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

Man hiking… -> … + sandstorm

Palms -> … + storm begins

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

- Fig. 18: Limitations arising from the base I2V model. Top row examples showcase limited visual quality. For instance, on the right, the person’s face takes up a small region in the video, so the I2V model struggles to properly generate such small details. On the top right, a video of a waterfall is edited to have a monster suddenly emerge from the water. The monster’s details are not of high quality due to the model’s generative capacity. The bottom examples show limitations due to the I2V model’s limits in visual reasoning. For example, on the left, the man is prompted to remove his hat due to the sandstorm. After the man successfully removes the hat, another hat appears under it, which is illogical. In the bottom right, a video of palm trees is edited such that a sudden storm begins. However, the trees remain static, showcasing the model’s limited understanding of wind dynamics.

