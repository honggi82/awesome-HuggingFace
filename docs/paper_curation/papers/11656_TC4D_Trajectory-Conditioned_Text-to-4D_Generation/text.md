# arXiv:2403.17920v3[cs.CV]14Oct2024

## TC4D: Trajectory-Conditioned Text-to-4D Generation

Sherwin Bahmani∗1,2,3, Xian Liu∗4, Wang Yifan∗5, Ivan Skorokhodov3, Victor Rong1,2, Ziwei Liu6, Xihui Liu7, Jeong Joon Park8, Sergey Tulyakov3, Gordon Wetzstein5, Andrea Tagliasacchi1,9,10, and David B. Lindell1,2

1University of Toronto 2Vector Institute 3Snap Inc. 4CUHK 5Stanford 6NTU 7HKU 8University of Michigan 9SFU 10Google DeepMind *equal contribution

###### https://sherwinbahmani.github.io/tc4d

Abstract. Recent techniques for text-to-4D generation synthesize dynamic 3D scenes using supervision from pre-trained text-to-video models. However, existing representations, such as deformation models or timedependent neural representations, are limited in the amount of motion they can generate—they cannot synthesize motion extending far beyond the bounding box used for volume rendering. The lack of a more flexible motion model contributes to the gap in realism between 4D generation methods and recent, near-photorealistic video generation models. Here, we propose TC4D: trajectory-conditioned text-to-4D generation, an approach that factors motion into global and local components. We represent the global motion of a scene’s bounding box using rigid transformation along a trajectory parameterized by a spline. We learn local deformations that conform to the global trajectory using supervision from a text-to-video model. Our approach enables synthesis of scenes animated along arbitrary trajectories, compositional scene generation, and significant improvements to the realism and amount of generated motion, which we evaluate qualitatively and through a user study. Video results can be viewed on our website: https://sherwinbahmani.github.io/tc4d.

#### 1 Introduction

Recent advances in video generation models [9–11,31,74,90,91,95] enable the synthesis of dynamic visual content with unprecedented fidelity. These advances have prompted the rapid development of techniques for 3D video generation (see Fig. 1), which use supervision from video generation models to produce animated 3D scenes, conditioned on an input video [37], image [68], or text prompt [75]. With applications including visual effects, virtual reality, and industrial design, realistic and controllable generation of 4D content has broad potential impact.

Although methods for 4D generation show immense promise, they are yet limited in the quality and realism of synthesized motion effects. Specifically, these methods animate a volumetric scene representation via learned deformation [48,68,104] or appearance changes predicted by a neural representation [4,

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

viewpoint 1

[Figure 7]

[Figure 8]

time

[Figure 9]

[Figure 10]

viewpoint 2 viewpoint 3

trajectory

"Deadpool riding a cow"

"an astronaut riding a horse"

"a bear walking"

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

time

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

"a giraffe walking"

trajectories

"an elephant walking"

- Fig. 1: Scenes generated using trajectory-conditioned 4D generation (TC4D). We show scenes consisting of multiple dynamic objects generated with text prompts and composited together. The scene is shown for different viewpoints (panels) and at different time steps (horizontal dimension). Motion is synthesized by animating the scene bounding box along a provided trajectory using a rigid transformation, and we optimize for local deformations that are consistent with the trajectory using supervision from a video diffusion model. Overall, our approach improves the amount and realism of motion in generated 4D scenes.

37,111,112] or generative model [60]. Since the scene is confined to a particular region—e.g., a 3D bounding box containing the volumetric representation—the synthesized motion is highly local, and the magnitude of motion is typically much smaller than the scale of the scene itself. For example, existing techniques can synthesize characters that walk in place, gesticulate, or sway, but synthesized outputs cannot move around a scene or exhibit global motion effects in general.

Overall, there is a stark contrast in realism between motion effects created with video generation models and existing techniques for 4D generation (see Fig. 2).

To address this limitation, we consider the specific problem of text-to-4D generation with both local and global motion, where global motions occur at scales larger than the bounding box of an object. As existing architectures are incompatible with synthesizing global motion, we require a new motion representation that facilitates learning motion at coarse and fine scales. Additionally, depicting global motion requires a large number of video frames, but current open-source video models typically output only a few tens of frames [9]. Thus, we require a solution that decouples the temporal resolution and duration of generated 4D scenes from the output size constraints of the video generation model used for supervision.

Our solution to this problem uses trajectory conditioning for 4D generation (TC4D)—a form of coarse-scale motion guidance that enables a novel decomposition of global and local motion effects. Specifically, we first use a text-to-3D model [4] to generate a static scene. Then, we model global motion by animating the bounding box of an object using a rigid transformation along a 3D trajectory. We parameterize the trajectory using a spline specified by a user. After fixing the global motion, we use supervision from a video diffusion model to learn local motion that is consistent with the global motion along the trajectory.

We model local motion along the trajectory by optimizing a time-varying deformation field based on a trajectory-aware version of video score distillation sampling (VSDS) [75]. That is, we apply VSDS to arbitrarily long trajectories by stochastically sampling segments of the trajectory and supervising a deformation model that is used to animate the scene. We observe that the quality of synthesized motion effects using the deformation model is highly dependent on the diffusion time steps used with VSDS; sampling early timesteps introduces relatively no motion, but sampling later time steps leads to unrealistic, highfrequency jitter. To address this issue, we introduce an annealing procedure that gradually modifies the interval of sampled diffusion time steps during optimization to achieve temporally consistent motion. Altogether, trajectory conditioning enables coherent synthesis of local and global scene animation, dramatically improving the amount and realism of motion in generated 4D scenes.

In summary, we make the following contributions.

- – We introduce the task of 4D synthesis with global motion and propose a novel solution based on trajectory conditioning that decouples motion into global and local components.
- – We demonstrate novel applications in text-to-4D generation with global motion, including 4D generation from arbitrary trajectories and compositional 4D scene synthesis.
- – We conduct an extensive evaluation of the approach, including ablation and user studies, and we demonstrate state-of-the-art results for text-to-4D generation.

4D-fy VideoCrafter2

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

"aunicornrunning""acorgirunning"

[Figure 25]

[Figure 26]

[Figure 27]

time

[Figure 28]

[Figure 29]

- Fig. 2: Comparing text-to-video and text-to-4D generation. Existing methods for video generation (e.g., VideoCrafter2 [14]) create realistic objects with motion at both global and local scales. For example, a generated unicorn or corgi moves across the camera field of view (global motion) in concert with the animation of its legs and body (local motion). However, existing text-to-4D methods [4] only generate a limited amount of local motion and do not learn global motion in 3D space.

#### 2 Related Work

Our work is most directly connected to text-to-video, text-to-3D, and text-to-4D generative models. For a comprehensive overview of recent advances in diffusion models and 4D generation, we refer readers to [63] and [107].

Text-to-video generation. We build on recent advances in 2D image and video generative models, with methods for text-to-video generation being especially relevant to our approach [8,63]. While earlier methods for image and video generation favor GANs [2,76], recent work shifts towards using diffusion models for their superior performance in video generation [11]. Still, most existing textto-video models exhibit a significant quality gap compared to text-to-image models due to dataset and computational constraints. Hybrid training approaches combining image and video datasets have been explored to overcome these limitations, employing techniques for enhancing spatial and temporal resolution through pixel-space and latent space upsampling [7, 27, 30, 31, 91, 92, 100, 113]. The emergence of transformer-based autoregressive models provides a leap forward, enabling the creation of videos of arbitrary lengths [11,54,56,83]. Recent efforts have also focused on fine-tuning text-to-image models for video data and decoupling content and motion generation for enhanced quality and flexibility [9,10,27,74,95]. Additionally, research aimed at increasing video generation controllability has shown promising results, allowing certain properties of an output video to be explicitly manipulated [6,72,89,94,102].

Despite recent advances, high-quality video synthesis remains challenging, with a notable quality disparity between open-source models and top-performing proprietary ones [1,90]. Although the recent open-source method Stable Video

Diffusion [9] can synthesize high-quality videos from an input image, we use VideoCrafter2 [14] for our experiments due to its support for text conditioning.

Text-to-3D generation. Early work on 3D generation leveraged the development of GAN algorithms to create realistic 3D objects of single categories, e.g., human or animal faces, cars, etc. These 3D GAN approaches [3,12,19,59,71], however, were limited in the class of scenes they could generate and thus were not suitable for text-based generation. Initial attempts at data-driven, textconditioned methods used CLIP-based supervision [67], which enabled synthesis or editing of 3D assets [15, 24, 34, 35, 70, 87]. These techniques evolved into the most recent approaches, which optimize a mesh or radiance field based on Score Distillation Sampling (SDS) and its variants using pre-trained diffusion models [16, 29, 38, 41, 44–46, 51, 64, 77, 88, 93, 103, 106]. These SDS-based methods are improving, generating increasingly high-quality 3D assets. The quality of generated 3D structures has been further improved by applying diffusion models that consider multiple viewpoints [21, 32, 39, 47, 49, 50, 73, 78, 85]. These models alleviate the so-called ”Janus” problem, where objects are generated with extra heads, limbs, etc. Recent methods have shifted toward using diffusion or transformer models to lift an input 2D image into a 3D representation for novel-view synthesis [13, 25, 52, 53, 66, 80, 81, 86, 105]. Other recent work [28, 33, 36, 42, 65, 79, 82, 97–99, 109] considers feed-forward 3D generation methods by first generating novel view images and converting them into 3D using transformer architectures. Another line of work models compositional 3D scenes [5, 17, 18, 20, 22, 62, 84, 110, 114] by decomposing the scene into objects. While these techniques for 3D generation produce impressive results, they do not support generating 4D scenes, i.e., their generated objects/scenes are static. We adapt MVDream [73] and ProlificDreamer [93] to obtain an initial high-quality and static 3D model, which we animate along a trajectory using supervision from a video diffusion model.

4D generation. Our research aligns closely with SDS-based 4D generation methodologies that are initialized with a 3D model and subsequently animated using text-to-video supervision. Within the last year, significant progress from initial results in 4D generation [75] has been made by a slew of recent methods [4,23,37,48,68,101,101,112]. These techniques have notably enhanced both geometry and appearance by integrating supervision from various text-to-image models. More recent endeavors [23, 48, 60, 68, 104, 108, 111, 112] have employed similar SDS-based strategies for image or video-conditioned 4D generation. Still, all of these approaches yield 4D scenes that remain stationary at a single location and are limited to short and small motions, such as swaying or turning. Our objective, however, is to generate large-scale motions that more accurately mimic real-world movements.

[Figure 30]

###### initialization trajectory video supervision

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

canonical deformation field NeRF

[Figure 35]

| | |
|---|---|
| | |

|[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]|
|---|

video diffusion guidance

spline control points

2D/3D diffusion guidance

[Figure 41]

- Fig. 3: Overview of TC4D. Our method takes as input a pre-trained 3D scene generated using supervision from a 2D and/or 3D diffusion model. We animate the scene through a decomposition of motion at global and local scales. Motion at the global scale is incorporated via rigid transformation of the bounding box containing the object representation along a given spline trajectory T at steps t. We align local motion to the trajectory by optimizing a separate deformation model that warps the underlying volumetric representation based on supervision from a text-to-video model. The output is an animated 3D scene with motion that is more realistic and greater in magnitude than previous techniques.

#### 3 Trajectory-Conditioned Text-to-4D Generation

Our method generates a 4D scene given an input text prompt and trajectory. We assume that an initial static 3D scene can be generated by adapting existing approaches for text-to-3D generation [4, 73, 93], and we incorporate global and local motion effects using trajectory conditioning and trajectory-aware video score distillation sampling (VSDS). Trajectory conditioning rigidly transforms the scene along a given path, and we parameterize local motions using a deformation model conditioned on the trajectory position. An overview of the method is shown in Fig. 3, and we describe each component as follows.

- 3.1 4D Scene Representation

We represent a 4D scene using a deformable 3D neural radiance field (NeRF) [43, 57,58] parameterized by a neural representation N3D. This representation models appearance and geometry within a unit-length bounding box B = [0,1]3 as

(c,σ) = N3D(xc), xc ∈ B, (1)

where c and σ are the radiance and density used for volume rendering [57]. The representation captures the canonical pose at t = t0 (without loss of generality we assume t0 = 0), and it models motion of the scene at subsequent steps. The

###### 4D scene at any step 0 < t ≤ 1 can be represented by warping the sampling rays with a time-varying deformation field F(·,t) to the canonical pose, i.e.,

N4D (x,t) = N3D (F (x,t)).

In this paper, we decompose the deformation F into a global and local term. The global motion is parameterized using a time-dependent rigid transformation

R : (x,t)  → xd that applies rotation, translation and scaling at time t. The local motion is given by a deformation field D : (xd,t)  → xc that maps an input coordinate xd ∈ B in a deformed space to the coordinate in canonical (i.e., nondeformed) space xc ∈ B. Composing the global and local motions, the 4D scene can be represented as

N4D (x,t) = N3D (D ◦ R(x,t)). (2) We describe the implementation of global and local motion operators in the

next sections.

##### 3.2 Modeling Global Motion with Trajectory Conditioning

Our proposed motion decomposition enables using a straightforward model for global motion R, by simply applying a rigid transformation to the bounding box B along a trajectory T . A holistic model of motion without this decompositione.g. through deformation alone—would be intractable. The scene bounding box would need to encompass the entire range of motion (inefficient in terms of computation and memory), and we would need to optimize long-range deformations, which is highly challenging.

In practice, the trajectory is parameterized as a cubic spline {T (t)|t ∈ [0,1]} and T (0) = 0 (i.e., the trajectory starts at the origin). We assume the 3D control points are provided by a user. This parametric curve defines the translation and the rotation of the bounding box B, and the resulting global motion can be expressed as

R(x,t) = R(nˆ,T ′)x + T (t). (3)

Here, R is the rotation matrix that maps from the trajectory’s tangent vector to the canonical bounding box orientation. Specifically, it maps the gradient of the trajectory T ′ = dT /dt to the normal of the bounding box’s “front” face nˆ while keeping the vertical component of the gradient perpendicular to the ground plane. Then, T (t) is the translation vector that maps back to the canonical bounding box location.

##### 3.3 Modeling Local Motion with Trajectory-Aware VSDS

We capture local motion by optimizing the deformation model with guidance from a trajectory-aware version of VSDS using a text-to-video diffusion model. To regularize the generated deformation field we use a smoothness penalty, and we introduce a motion annealing technique, which empirically improves realism and temporal consistency.

Video score distillation sampling (VSDS). For completeness, we briefly review the principle of VSDS, and refer the interested reader to Singer et al. [75] for a more detailed description. Intuitively this procedure queries a video diffusion

model to see how it adds structure to a video rendered from our representation. Then, this signal is used to backpropagate gradients into the model to reconstruct the 4D scene.

Specifically, a video sequence v is rendered from the 4D representation parameterized with θ. Then, noise ϵ is added to the video according to a diffusion process; the amount of noise is governed by the diffusion timestep td such that td = 0 yields a noise-free sequence and the maximum timestep (td = 1) gives zero-mean Gaussian noise. The diffusion model, conditioned on a text embedding y, predicts the noise added to the video ϵˆ(vt

;td,y), and this is used to calculate the score distillation sampling gradient ∇θL4D used to optimize the

d

- 4D representation [64].

∂v ∂θ

∇θL4D = Et

;td,y) − ϵ)

##### d,ϵ (ϵˆ(vt

d

. (4)

Here, the expectation E is taken over all timesteps and noise realizations, and ∂∂θv is the gradient of the generated video w.r.t. the 4D representation parameters

θ. In our case, we freeze the parameters of N3D and optimize the parameters of the deformation model D to learn local motion.

Incorporating the trajectory. One challenge with using VSDS to generate dynamic objects along a trajectory is that the pre-trained video diffusion models generate a finite number of N frames and thus can only represent limited range of motion. Naively sampling N timesteps across a long trajectory would require the diffusion model to represent long-range motion, different from its training data.

Instead, we divide the trajectory into segments of length ∆t that roughly align with the intrinsic motion scale of the video diffusion model—that is, the range of motion typically exhibited in generated video samples. Assuming the intrinsic motion scale Mv of the video model is known, and given the total length of the trajectory L, we approximate the segment length as

Mv L

,1 . (5)

∆t = max

In practice, we determine Mv based on an empirical evaluation across multiple scene categories (see supplement for additional details).

Our trajectory-aware VSDS thus consists of three steps: (1) we sample random start times t0 ∈ [0,1−∆t] along the trajectory, and evenly sample N frame times as tn = t0 + n∆t/(N − 1) for 0 ≤ n ≤ N − 1; (2) we compute the corresponding scene representation from N4D using Eq. 2; (3) we render videos and optimize the deformation model using Eq. 4. We summarize the entire optimization procedure Algorithm 1.

##### 3.4 Implementation details

Following prior work [4,112], we use hash-encoded feature grids to encode the 3D NeRF N3D and the local deformation field D. N3D, is optimized using the

Algorithm 1 TC4D

Require:

N3D ▷ optimized 3D neural representation T ▷ trajectory parameterized with a spline D ▷ initial local deformation model parameterized with θ

###### Output:

D∗ ▷ optimized local deformation model

- 1: determine ∆t
- 2: sample t0 and trajectory steps tn, 0 ≤ n ≤ N − 1
- 3: sample points along rendering rays x
- 4: apply rigid transform and local deformation (Eq. 2) xc = D ◦ R(x, tn)
- 5: volume render video from N3D(xc) and render deformation video from D(x, t)
- 6: calculate ∇θL4D and ∇θLsmooth from Eq. 4 and Eq. 6
- 7: update D
- 8: repeat 1-7

hybrid SDS proposed by Bahmani et al. [4], which incorporates supervision from a combination of pre-trained image diffusion models [73,93] to generate a static

- 3D scene with high-quality appearance and 3D structure.

Once trained, we freeze N3D and optimize the local deformation field D using the trajectory-aware VSDS gradient and an additional smoothness loss. For ease of optimization, we let D represent the offset d = xd − xc and initialize it to a small value. Following Zheng et al. [112], we apply volume rendering to create videos of the 3D displacement and compute the penalty as

Lsmooth = λ

∥dx∥22 + ∥dy∥22 + ∥dt∥22, (6)

p,n

where the summation is over all pixels p, video frames n, λ is a hyperparameter, and dx/y/t represents a finite-difference approximation to spatial and temporal derivatives in the 3D displacement video sequence.

Additionally, we find that VSDS gradients for diffusion timesteps where td ≈ 1 introduce jittery, high-frequency motion into the deformation field. We address this issue by annealing the sampling interval of the diffusion timesteps during the course of optimization, which improves the temporal coherence of local motion. At the start of training we compute ∇L4D by uniformly sampling td ∈ [0.02,0.98]. This interval is linearly decayed during training to td ∈ [0.02,0.5].

The model is trained using the Adam optimizer [40] for 10000 iterations, which we find is sufficient to ensure robust and repeatable convergence.

#### 4 Experiments

Motion quality is extremely difficult to visualize. In this manuscript, we focus on presenting a detailed analysis of our quantitative evaluation and include only the most representative visual examples. We highly encourage the readers to check the videos in the supplemental material for a conclusive quality assessment.

##### 4.1 Metrics

Since there is no automated metric for 4D generation, we evaluate TC4D with a user study. We refer to Sec. 3 of the appendix for additional CLIP score evaluations. The method is compared quantitatively and qualitatively to a modified version of 4D-fy [4]. That is, we animate the output of 4D-fy by rigidly transforming it to follow the same trajectory as used for TC4D. We further compare to the concurrent work DreamGaussian4D [68] in Sec. 3 of the appendix. In total we provide comparisons using 33 scenes generated from different text prompts, and we conduct an extensive ablation study on a subset of these text prompts. Our evaluation uses a similar number of text prompts as previous work (e.g., 28 [75], or 17 [4]).

We recruit a total of 20 human evaluators and show the evaluators two video clips generated using the same text prompt with TC4D and 4D-fy. We ask the evaluators to specify their overall preference and their preference in terms of appearance quality (AQ), 3D structure quality (SQ), motion quality (MQ), motion amount (MA), text alignment (TA) (see the supplement for the detailed protocol for the study). The results of the user study are shown in Table 1 and we include further details in the supplement. Each row of the table reports the percentage of evaluations that ranked T4CD over 4D-fy or the ablated approach, i.e., a percentage above 50% indicates T4CD is preferred. Statistically significant results (p < 0.05) are indicated based on a χ2 test.

##### 4.2 Results

We visualize the generated 4D scenes by rendering and compositing images along the trajectory from multiple camera viewpoints in Fig. 4. TC4D shows significantly more motion and also more realistic motion than 4D-fy—this is especially noticeable for scenes in which TC4D synthesizes walking animations that are coherent with the trajectory, but 4D-fy produces only a small amount of relatively incoherent motion. In the user study, participants indicated a statistically significant preference for results from TC4D for all metrics except structure quality. In terms of overall preference, 85% of comparisons were in favor of T4CD. We refer the reader to the video clips included in the supplement, which provide a clear visualization of the advantages of the proposed approach. In addition to these results based on user-defined trajectories, we show results for automated end-to-end generation using large language models in Sec. 4 of the appendix. Moreover, we provide geometry visualizations in Sec. 5 of the appendix

TC4D 4D-fy

|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>"batman riding a camel"|
|---|

|[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]|
|---|

viewpoint2viewpoint1viewpoint2viewpoint1

|[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]|
|---|

|[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]|
|---|

|[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>"a deer walking"|
|---|

|[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]|
|---|

|[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]|
|---|

|[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]|
|---|

- Fig. 4: Comparison of TC4D and 4D-fy. We show two generated scenes of “batman riding a camel” and “a deer walking”. Each panel contains images rendered from a single viewpoint from two steps along the same trajectory. While 4D-fy produces mostly static results (which we rigidly transform along the trajectory), TC4D learns coherent motion and produces a walking animation. Please also refer to the video results in the supplement.

##### 4.3 Ablation Study

We assess the impact of various design choices for TC4D through an extensive ablation study. The results are evaluated in the user study (Table 1), and we show qualitative results for the ablated methods on multiple scenes in video clips included in the supplement. To allow for fairer comparison between the methods, we initialize each method with the same static model for a given prompt. This generally led to similar appearance and 3D structure across all methods, which is why many of the quantitative results for appearance quality and 3D structure

Human Preference (Ours vs. Method) Method AQ SQ MQ MA TA Overall 4D-fy [4] 55% 52*% 89% 92% 81% 85% Ablation Study

w/o local deformation 59%* 57%* 98% 98% 86% 97% w/o trajectory 56%* 61% 80% 85% 79% 84% w/o traj. training 61% 56%* 87% 92% 79% 89% w/o traj.-aware VSDS 62% 59%* 84% 84% 74% 85% w/o global transformations 65% 58%* 94% 94% 85% 97% w/o smooth. penalty 58%* 60%* 73% 69% 63% 74% w/o timestep annealing 63% 65% 84% 83% 73% 84%

Table 1: Quantitative results. We compare TC4D to the output of 4D-fy animated along the trajectory using a rigid transformation. The methods are evaluated in a user study in which participants indicate their preference (or no preference) based on appearance quality (AQ), 3D structure quality (SQ), motion quality (MQ), motion amount (MA), text alignment (TA), and overall preference (Overall). We also conduct ablations using a subset of the text prompts. The percentages indicate preference for TC4D vs. the alternative method (in each row), i.e., >50% indicates ours is preferred. All results are statistically significant (p < 0.05) except those indicated with an asterisk.

quality in Table 1 do not indicate a statistically significant preference. We discuss each of the ablations in the following.

Local deformation. Removing the local deformation model reduces the method to rigid transformation of a static pre-trained 3D scene along a spline (i.e., with N4D(x,t) = N3D(R(x,t))). This ablated method produces unrealistic and completely rigid motion. It is indicated in Table 1 as “w/o local deformation”. The results show that the participants perceive less and lower quality motion (see MA and MQ columns respectively), and they overwhelmingly prefer including the deformation motion component.

Trajectory (training and rendering). Alternatively, entirely removing the trajectory reduces the method to the conventional text-to-4D generation setting with deformation-based motion. That is, we have N4D = N3D(D(x,t)). This experiment is indicated in Table 1 as “w/o trajectory”. While the method still learns to incorporate some motion without the trajectory, we find that the trajectory is key to producing more complex animations such as walking.

Trajectory (training). Similar to the 4D-fy baseline, we can omit the trajectory at the training stage and animate the scene via rigid transformation after training. This experiment is indicated in Table 1 as “w/o traj. training”. In this experiment, since the synthesized motion is agnostic to the trajectory, the resulting animations are less coherent than when incorporating the trajectory into the training procedure. Specifically, only a smaller range of motion is visible. User study participants prefer TC4D over this ablation by a significant margin.

Trajectory-aware VSDS. We use trajectory-aware VSDS to ensure that all regions of the trajectory receive adequate supervision from the text-to-video model. Specifically, this procedure involves breaking the trajectory down into segments, and then sampling one segment for which we render the video clips used to compute the VSDS gradients. Then the VSDS gradients are computed using the resulting video. Overall, we find this approach results in too coarse a sampling of the trajectory and fails to synthesize smooth motion. In Table 1, this method is indicated as “w/o traj.-aware VSDS”.

Global transformations. To demonstrate that VSDS is trajectory dependent, we conduct an ablation where we remove global transformations during training, but still sample the time dimension in a segmented way as in trajectory-aware VSDS. The local deformations in this approach are significantly reduced.

Smoothness penalty. We assess the impact of removing the smoothness penalty from the deformation field (Eq. 6). This regularization term, proposed by Zheng et al. [112], helps to dampen high-frequency displacements in the deformation field and leads to smoother motion. As the row “w/o smooth penalty” shows in Table 1, the responses from user study participants indicate that removing this penalty has only a modest negative effect.

Diffusion timestep annealing. Scheduling the sampling of the diffusion timesteps during training with VSDS provides another form of motion regularization in our method. Specifically, we reduce the sampling of large diffusion timesteps as we train, and we find that this produces smoother, more realistic motion. As we show in the last row of Table 4.2, “w/o timestep annealing”, removing this procedure entirely (and keeping the smoothness penalty) negatively impacts the users’ preference. In the supplement, we show that this method produces jittery animations and noticeable high-frequency deformation.

Effect of scale. Finally, we evaluate the effect of combining scaling with the rigid transformation used to define the trajectory. That is, we modify Eq. 3 to incorporate a time-dependent term S(t) ∈ R3+, which scales the bounding box along each axis:

RS(x,t) = S(t) ⊙ R(nˆ,T ′)x + T (t), (7) where ⊙ indicates Hadamard product.

We demonstrate this approach on a scene consisting of a column of fire that is generated with the text prompt, “a flame getting larger” (Fig. 5). Generating this scene without the trajectory and without bounding box scaling results in a flame that has little variation in size. Alternatively, we animate the flame along an upwards trajectory, but this approach simply translates the generated scene rather than changing the size. Combining translation and scaling in the vertical dimension allows us to keep the center of the bounding box at a consistent location, and with this technique, the generated flame appears to grow in size.

w/o trajectory, w/o scale

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

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

w/ trajectory, w/o scale

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

[Figure 118]

[Figure 119]

w/ trajectory, w/ scale

time

Fig. 5: Trajectory with scale. We demonstrate adding a scale term to the trajectory for a scene generated with the text prompt “a flame getting larger”. We compare generating the scene without a trajectory or scale (top), with an upwards trajectory only (middle), and with both a trajectory and scale (bottom), the last of which yields a convincing result.

#### 5 Conclusion

Overall our work takes important steps in making motion synthesis for textto-4D models more realistic and expressive. Our framework enables scene-scale motion of entities within compositional 4D scenes with trajectory conditioning, achieving improvement over prior work that shows motion at fixed locations.

The main limitations of our results are imperfect leg motion, foot sliding, or asymmetric and not fully cyclic motion. The motion imperfections stem from the quality of publicly available pre-trained text-to-video models used for VSDS. Also, VideoCrafter2 outputs 16-frame videos that may be too short to capture fully cyclic motion (e.g., a complete walking cycle), contributing to this limitation. We expect that TC4D will benefit from rapid improvements in 2D video generation, and we open source our code to spur follow-on work.

The proposed factorized motion model opens several avenues for future work. In particular, we envision extensions to generate scenes with multiple interacting objects, adding motion to existing in-the-wild 3D assets, and synthesis of motion across unbounded, large-scale scenes. Finally, designing automated text-to-4D metrics by temporally extending text-to-3D metrics [96] is an important direction for future work.

Ethics statement. Our approach can automatically generate realistic, animated

- 3D scenes; such techniques can be misused to promulgate realistic fake content for the purposes of misinformation. We condemn misuse of generative models in this fashion and emphasize the importance of research to thwart such efforts (see, e.g., Masood et al. [55] for an extended discussion of this topic).

#### Acknowledgements

This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant program, the Digital Research Alliance of Canada, and by the Advanced Research Computing at Simon Fraser University. It was also supported by Snap Inc. and a Swiss Postdoc Mobility fellowship.

#### References

- 1. Zeroscope text-to-video model. https : / / huggingface . co / cerspense / zeroscope_v2_576w, accessed: 2023-10-31
- 2. Bahmani, S., Park, J.J., Paschalidou, D., Tang, H., Wetzstein, G., Guibas, L., Van Gool, L., Timofte, R.: 3D-aware video generation. TMLR (2023)
- 3. Bahmani, S., Park, J.J., Paschalidou, D., Yan, X., Wetzstein, G., Guibas, L., Tagliasacchi, A.: CC3D: Layout-conditioned generation of compositional 3D scenes. Proc. ICCV (2023)
- 4. Bahmani, S., Skorokhodov, I., Rong, V., Wetzstein, G., Guibas, L., Wonka, P., Tulyakov, S., Park, J.J., Tagliasacchi, A., Lindell, D.B.: 4D-fy: Text-to-4D generation using hybrid score distillation sampling. Proc. CVPR (2024)
- 5. Bai, H., Lyu, Y., Jiang, L., Li, S., Lu, H., Lin, X., Wang, L.: CompoNeRF: Textguided multi-object compositional NeRF with editable 3D scene layout. arXiv preprint arXiv:2303.13843 (2023)
- 6. Bai, J., He, T., Wang, Y., Guo, J., Hu, H., Liu, Z., Bian, J.: Uniedit: A unified tuning-free framework for video motion and appearance editing. arXiv preprint arXiv:2402.13185 (2024)
- 7. Bain, M., Nagrani, A., Varol, G., Zisserman, A.: Frozen in time: A joint video and image encoder for end-to-end retrieval. Proc. ICCV (2021)
- 8. Bie, F., Yang, Y., Zhou, Z., Ghanem, A., Zhang, M., Yao, Z., Wu, X., Holmes, C., Golnari, P., Clifton, D.A., et al.: RenAIssance: A survey into AI text-to-image generation in the era of large model. arXiv preprint arXiv:2309.00810 (2023)
- 9. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127

(2023)

- 10. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. Proc. CVPR (2023)
- 11. Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world simulators (2024), https://openai.com/research/ video-generation-models-as-world-simulators
- 12. Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., De Mello, S., Gallo, O., Guibas, L.J., Tremblay, J., Khamis, S., et al.: Efficient geometry-aware 3D generative adversarial networks. Proc. CVPR (2022)
- 13. Chan, E.R., Nagano, K., Chan, M.A., Bergman, A.W., Park, J.J., Levy, A., Aittala, M., De Mello, S., Karras, T., Wetzstein, G.: Generative novel view synthesis with 3D-aware diffusion models. Proc. ICCV (2023)
- 14. Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C., Shan, Y.: VideoCrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047 (2024)
- 15. Chen, K., Choy, C.B., Savva, M., Chang, A.X., Funkhouser, T., Savarese, S.: Text2Shape: Generating shapes from natural language by learning joint embeddings. Proc. ACCV (2018)
- 16. Chen, R., Chen, Y., Jiao, N., Jia, K.: Fantasia3D: Disentangling geometry and appearance for high-quality text-to-3D content creation. arXiv preprint arXiv:2303.13873 (2023)

- 17. Chen, Y., Wang, T., Wu, T., Pan, X., Jia, K., Liu, Z.: ComboVerse: Compositional 3D assets creation using spatially-aware diffusion guidance. arXiv preprint arXiv:2403.12409 (2024)
- 18. Cohen-Bar, D., Richardson, E., Metzer, G., Giryes, R., Cohen-Or, D.: Set-thescene: Global-local training for generating controllable NeRF scenes. Proc. ICCV Workshops (2023)
- 19. DeVries, T., Bautista, M.A., Srivastava, N., Taylor, G.W., Susskind, J.M.: Unconstrained scene generation with locally conditioned radiance fields. Proc. ICCV

(2021)

- 20. Epstein, D., Poole, B., Mildenhall, B., Efros, A.A., Holynski, A.: Disentangled 3D scene generation with layout learning. Proc. ICML (2024)
- 21. Feng, Q., Xing, Z., Wu, Z., Jiang, Y.G.: FDGaussian: Fast Gaussian splatting from single image via geometric-aware diffusion model. arXiv preprint arXiv:2403.10242 (2024)
- 22. Gao, G., Liu, W., Chen, A., Geiger, A., Schölkopf, B.: GraphDreamer: Compositional 3D scene synthesis from scene graphs. Proc. CVPR (2024)
- 23. Gao, Q., Xu, Q., Cao, Z., Mildenhall, B., Ma, W., Chen, L., Tang, D., Neumann, U.: GaussianFlow: Splatting Gaussian dynamics for 4D content creation. arXiv preprint arXiv:2403.12365 (2024)
- 24. Gao, W., Aigerman, N., Groueix, T., Kim, V., Hanocka, R.: TextDeformer: Geometry manipulation using text guidance. Proc. SIGGRAPH (2023)
- 25. Gu, J., Trevithick, A., Lin, K.E., Susskind, J.M., Theobalt, C., Liu, L., Ramamoorthi, R.: NerfDiff: Single-image view synthesis with NeRF-guided distillation from 3D-aware diffusion. Proc. ICML (2023)
- 26. Guo, Y.C., Liu, Y.T., Shao, R., Laforte, C., Voleti, V., Luo, G., Chen, C.H., Zou, Z.X., Wang, C., Cao, Y.P., Zhang, S.H.: threestudio: A unified framework for 3D content generation. https://github.com/threestudio-project/threestudio

(2023)

- 27. Guo, Y., Yang, C., Rao, A., Wang, Y., Qiao, Y., Lin, D., Dai, B.: AnimateDiff: Animate your personalized text-to-image diffusion models without specific tuning. Proc. ICLR (2024)
- 28. Han, J., Kokkinos, F., Torr, P.: VFusion3D: Learning scalable 3D generative models from video diffusion models. arXiv preprint arXiv:2403.12034 (2024)
- 29. He, X., Chen, J., Peng, S., Huang, D., Li, Y., Huang, X., Yuan, C., Ouyang, W., He, T.: GVGEN: Text-to-3D generation with volumetric representation. arXiv preprint arXiv:2403.12957 (2024)
- 30. He, Y., Yang, T., Zhang, Y., Shan, Y., Chen, Q.: Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221 (2022)
- 31. Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., et al.: Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022)
- 32. Höllein, L., Božič, A., Müller, N., Novotny, D., Tseng, H.Y., Richardt, C., Zollhöfer, M., Nießner, M.: ViewDiff: 3D-consistent image generation with text-toimage models. Proc. CVPR (2024)
- 33. Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., Tan, H.: LRM: Large reconstruction model for single image to 3D. Proc. ICLR (2024)
- 34. Jain, A., Mildenhall, B., Barron, J.T., Abbeel, P., Poole, B.: Zero-shot text-guided object generation with dream fields. Proc. CVPR (2022)

- 35. Jetchev, N.: ClipMatrix: Text-controlled creation of 3D textured meshes. arXiv preprint arXiv:2109.12922 (2021)
- 36. Jiang, L., Wang, L.: Brightdreamer: Generic 3D Gaussian generative framework for fast text-to-3D synthesis. arXiv preprint arXiv:2403.11273 (2024)
- 37. Jiang, Y., Zhang, L., Gao, J., Hu, W., Yao, Y.: Consistent4D: Consistent 360◦ dynamic object generation from monocular video. arXiv preprint arXiv:2311.02848

(2023)

- 38. Katzir, O., Patashnik, O., Cohen-Or, D., Lischinski, D.: Noise-free score distillation. Proc. ICLR (2024)
- 39. Kim, S.W., Brown, B., Yin, K., Kreis, K., Schwarz, K., Li, D., Rombach, R., Torralba, A., Fidler, S.: NeuralField-LDM: Scene generation with hierarchical latent diffusion models. Proc. CVPR (2023)
- 40. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. Proc. ICLR

(2015)

- 41. Lee, K., Sohn, K., Shin, J.: DreamFlow: High-quality text-to-3D generation by approximating probability flow. Proc. ICLR (2024)
- 42. Li, J., Tan, H., Zhang, K., Xu, Z., Luan, F., Xu, Y., Hong, Y., Sunkavalli, K., Shakhnarovich, G., Bi, S.: Instant3D: Fast text-to-3D with sparse-view generation and large reconstruction model. Proc. ICLR (2024)
- 43. Li, R., Tancik, M., Kanazawa, A.: NerfAcc: A general NeRF acceleration toolbox. Proc. ICCV (2023)
- 44. Li, Z., Chen, Y., Zhao, L., Liu, P.: Controllable text-to-3D generation via surfacealigned Gaussian splatting. arXiv preprint arXiv:2403.09981 (2024)
- 45. Liang, Y., Yang, X., Lin, J., Li, H., Xu, X., Chen, Y.: Luciddreamer: Towards high-fidelity text-to-3D generation via interval score matching. arXiv preprint arXiv:2311.11284 (2023)
- 46. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3D: High-resolution text-to-3D content creation. Proc. CVPR (2023)
- 47. Lin, Y., Han, H., Gong, C., Xu, Z., Zhang, Y., Li, X.: Consistent123: One image to highly consistent 3D asset using case-aware diffusion priors. arXiv preprint arXiv:2309.17261 (2023)
- 48. Ling, H., Kim, S.W., Torralba, A., Fidler, S., Kreis, K.: Align your Gaussians: Text-to-4D with dynamic 3D Gaussians and composed diffusion models. Proc. CVPR (2024)
- 49. Liu, P., Wang, Y., Sun, F., Li, J., Xiao, H., Xue, H., Wang, X.: Isotropic3D: Image-to-3D generation based on a single clip embedding. arXiv preprint arXiv:2403.10395 (2024)
- 50. Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1-to-3: Zero-shot one image to 3D object. Proc. ICCV (2023)
- 51. Liu, X., Zhan, X., Tang, J., Shan, Y., Zeng, G., Lin, D., Liu, X., Liu, Z.: HumanGaussian: Text-driven 3D human generation with Gaussian splatting. Proc. CVPR (2024)
- 52. Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., Wang, W.: SyncDreamer: Generating multiview-consistent images from a single-view image. Proc. ICLR

(2024)

- 53. Long, X., Guo, Y.C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.H., Habermann, M., Theobalt, C., et al.: Wonder3D: Single image to 3D using crossdomain diffusion. Proc. CVPR (2024)

- 54. Ma, X., Wang, Y., Jia, G., Chen, X., Liu, Z., Li, Y.F., Chen, C., Qiao, Y.: Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048 (2024)
- 55. Masood, M., Nawaz, M., Malik, K.M., Javed, A., Irtaza, A., Malik, H.: Deepfakes generation and detection: State-of-the-art, open challenges, countermeasures, and way forward. Applied Intelligence 53(4), 3974–4026 (2023)
- 56. Menapace, W., Siarohin, A., Skorokhodov, I., Deyneka, E., Chen, T.S., Kag, A., Fang, Y., Stoliar, A., Ricci, E., Ren, J., et al.: Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. Proc. CVPR (2024)
- 57. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: Representing scenes as neural radiance fields for view synthesis. Proc. ECCV (2020)
- 58. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. 41(4), 1–15 (2022)
- 59. Or-El, R., Luo, X., Shan, M., Shechtman, E., Park, J.J., KemelmacherShlizerman, I.: StyleSDF: High-resolution 3D-consistent image and geometry generation. Proc. CVPR (2022)
- 60. Pan, Z., Yang, Z., Zhu, X., Zhang, L.: Fast dynamic 3D object generation from a single-view video. arXiv preprint arXiv:2401.08742 (2024)
- 61. Park, D.H., Azadi, S., Liu, X., Darrell, T., Rohrbach, A.: Benchmark for compositional text-to-image synthesis. In: Proc. NeurIPS (2021)
- 62. Po, R., Wetzstein, G.: Compositional 3D scene generation using locally conditioned diffusion. Proc. 3DV (2024)
- 63. Po, R., Yifan, W., Golyanik, V., Aberman, K., Barron, J.T., Bermano, A.H., Chan, E.R., Dekel, T., Holynski, A., Kanazawa, A., et al.: State of the art on diffusion models for visual computing. arXiv preprint arXiv:2310.07204 (2023)
- 64. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: DreamFusion: Text-to-3D using 2D diffusion. In: Proc. ICLR (2023)
- 65. Qian, G., Cao, J., Siarohin, A., Kant, Y., Wang, C., Vasilkovsky, M., Lee, H.Y., Fang, Y., Skorokhodov, I., Zhuang, P., et al.: Atom: Amortized text-to-mesh using 2d diffusion. arXiv preprint arXiv:2402.00867 (2024)
- 66. Qian, G., Mai, J., Hamdi, A., Ren, J., Siarohin, A., Li, B., Lee, H.Y., Skorokhodov,

I., Wonka, P., Tulyakov, S., et al.: Magic123: One image to high-quality 3D object generation using both 2D and 3D diffusion priors. Proc. ICLR (2024)

- 67. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. Proc. ICML (2021)
- 68. Ren, J., Pan, L., Tang, J., Zhang, C., Cao, A., Zeng, G., Liu, Z.: DreamGaussian4D: Generative 4D Gaussian splatting. arXiv preprint arXiv:2312.17142

(2023)

- 69. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proc. CVPR (2022)
- 70. Sanghi, A., Chu, H., Lambourne, J.G., Wang, Y., Cheng, C.Y., Fumero, M., Malekshan, K.R.: CLIP-Forge: Towards zero-shot text-to-shape generation. Proc. CVPR (2022)
- 71. Schwarz, K., Sauer, A., Niemeyer, M., Liao, Y., Geiger, A.: VoxGRAF: Fast 3Daware image synthesis with sparse voxel grids. Proc. NeurIPS (2022)
- 72. Shi, X., Huang, Z., Wang, F.Y., Bian, W., Li, D., Zhang, Y., Zhang, M., Cheung, K.C., See, S., Qin, H., et al.: Motion-I2V: Consistent and controllable image-tovideo generation with explicit motion modeling. arXiv preprint arXiv:2401.15977

(2024)

- 73. Shi, Y., Wang, P., Ye, J., Mai, L., Li, K., Yang, X.: MVDream: Multi-view diffusion for 3D generation. Proc. ICLR (2024)
- 74. Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al.: Make-a-video: Text-to-video generation without text-video data. Proc. ICLR (2023)
- 75. Singer, U., Sheynin, S., Polyak, A., Ashual, O., Makarov, I., Kokkinos, F., Goyal, N., Vedaldi, A., Parikh, D., Johnson, J., et al.: Text-to-4D dynamic scene generation. Proc. ICML (2023)
- 76. Skorokhodov, I., Tulyakov, S., Elhoseiny, M.: StyleGAN-V: A continuous video generator with the price, image quality and perks of StyleGAN2. Proc. CVPR

(2022)

- 77. Sun, J., Zhang, B., Shao, R., Wang, L., Liu, W., Xie, Z., Liu, Y.: DreamCraft3D: Hierarchical 3D generation with bootstrapped diffusion prior. Proc. ICLR (2024)
- 78. Szymanowicz, S., Rupprecht, C., Vedaldi, A.: Splatter image: Ultra-fast singleview 3D reconstruction. Proc. CVPR (2024)
- 79. Tang, J., Chen, Z., Chen, X., Wang, T., Zeng, G., Liu, Z.: LGM: Large multi-view gaussian model for high-resolution 3d content creation. Proc. ECCV (2024)
- 80. Tang, J., Wang, T., Zhang, B., Zhang, T., Yi, R., Ma, L., Chen, D.: Make-it-3D: High-fidelity 3D creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184 (2023)
- 81. Tewari, A., Yin, T., Cazenavette, G., Rezchikov, S., Tenenbaum, J., Durand, F., Freeman, B., Sitzmann, V.: Diffusion with forward models: Solving stochastic inverse problems without direct supervision. Proc. NeurIPS (2023)
- 82. Tochilkin, D., Pankratz, D., Liu, Z., Huang, Z., Letts, A., Li, Y., Liang, D., Laforte, C., Jampani, V., Cao, Y.P.: Triposr: Fast 3D object reconstruction from a single image. arXiv preprint arXiv:2403.02151 (2024)
- 83. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Proc. NeurIPS (2017)
- 84. Vilesov, A., Chari, P., Kadambi, A.: CG3D: Compositional generation for textto-3D via Gaussian splatting. arXiv preprint arXiv:2311.17907 (2023)
- 85. Voleti, V., Yao, C.H., Boss, M., Letts, A., Pankratz, D., Tochilkin, D., Laforte, C., Rombach, R., Jampani, V.: SV3D: Novel multi-view synthesis and 3D generation from a single image using latent video diffusion. arXiv preprint arXiv:2403.12008

(2024)

- 86. Wan, Z., Paschalidou, D., Huang, I., Liu, H., Shen, B., Xiang, X., Liao, J., Guibas, L.: CAD: Photorealistic 3D generation via adversarial distillation. Proc. CVPR

(2024)

- 87. Wang, C., Chai, M., He, M., Chen, D., Liao, J.: Clip-NeRF: Text-and-image driven manipulation of neural radiance fields. Proc. CVPR (2022)
- 88. Wang, H., Du, X., Li, J., Yeh, R.A., Shakhnarovich, G.: Score Jacobian chaining: Lifting pretrained 2d diffusion models for 3D generation. Proc. CVPR (2023)
- 89. Wang, J., Zhang, Y., Zou, J., Zeng, Y., Wei, G., Yuan, L., Li, H.: Boximator: Generating rich and controllable motions for video synthesis. arXiv preprint arXiv:2402.01566 (2024)
- 90. Wang, J., Yuan, H., Chen, D., Zhang, Y., Wang, X., Zhang, S.: Modelscope textto-video technical report. arXiv preprint arXiv:2308.06571 (2023)
- 91. Wang, W., Yang, H., Tuo, Z., He, H., Zhu, J., Fu, J., Liu, J.: Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874 (2023)

- 92. Wang, X., Yuan, H., Zhang, S., Chen, D., Wang, J., Zhang, Y., Shen, Y., Zhao, D., Zhou, J.: Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018 (2023)
- 93. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: ProlificDreamer: High-fidelity and diverse text-to-3D generation with variational score distillation. Proc. NeurIPS (2023)
- 94. Wang, Z., Yuan, Z., Wang, X., Chen, T., Xia, M., Luo, P., Shan, Y.: MotionCtrl: A unified and flexible motion controller for video generation. arXiv preprint arXiv:2312.03641 (2023)
- 95. Wu, R., , Chen, L., Yang, T., Guo, C., Li, C., Zhang, X.: LAMP: Learn a motion pattern for few-shot-based video generation. arXiv preprint arXiv:2310.10769

(2023)

- 96. Wu, T., Yang, G., Li, Z., Zhang, K., Liu, Z., Guibas, L., Lin, D., Wetzstein, G.: GPT-4V(ision) is a human-aligned evaluator for text-to-3D generation. Proc. CVPR (2024)
- 97. Xie, K., Lorraine, J., Cao, T., Gao, J., Lucas, J., Torralba, A., Fidler, S., Zeng, X.: LATTE3D: Large-scale amortized text-to-enhanced3D synthesis. Proc. ECCV

(2024)

- 98. Xu, Y., Shi, Z., Yifan, W., Chen, H., Yang, C., Peng, S., Shen, Y., Wetzstein, G.: GRM: Large Gaussian reconstruction model for efficient 3D reconstruction and generation. Proc. ECCV (2024)
- 99. Xu, Y., Tan, H., Luan, F., Bi, S., Wang, P., Li, J., Shi, Z., Sunkavalli, K., Wetzstein, G., Xu, Z., et al.: DMV3D: Denoising multi-view diffusion using 3D large reconstruction model. Proc. ICLR (2024)
- 100. Xue, H., Hang, T., Zeng, Y., Sun, Y., Liu, B., Yang, H., Fu, J., Guo, B.: Advancing high-resolution video-language representation with large-scale video transcriptions. In: Proc. CVPR (2022)
- 101. Yang, Q., Feng, M., Wu, Z., Sun, S., Dong, W., Wang, Y., Mian, A.: Beyond skeletons: Integrative latent mapping for coherent 4D sequence generation. arXiv preprint arXiv:2403.13238 (2024)
- 102. Yang, S., Hou, L., Huang, H., Ma, C., Wan, P., Zhang, D., Chen, X., Liao, J.: Direct-a-video: Customized video generation with user-directed camera movement and object motion. arXiv preprint arXiv:2402.03162 (2024)
- 103. Ye, J., Liu, F., Li, Q., Wang, Z., Wang, Y., Wang, X., Duan, Y., Zhu, J.: DreamReward: Text-to-3D generation with human preference. arXiv preprint arXiv:2403.14613 (2024)
- 104. Yin, Y., Xu, D., Wang, Z., Zhao, Y., Wei, Y.: 4DGen: Grounded 4D content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225

(2023)

- 105. Yoo, P., Guo, J., Matsuo, Y., Gu, S.S.: DreamSparse: Escaping from Plato’s cave with 2D diffusion model given sparse views. arXiv preprint arXiv:2306.03414

(2023)

- 106. Yu, X., Guo, Y.C., Li, Y., Liang, D., Zhang, S.H., Qi, X.: Text-to-3D with classifier score distillation. arXiv preprint arXiv:2310.19415 (2023)
- 107. Yunus, R., Lenssen, J.E., Niemeyer, M., Liao, Y., Rupprecht, C., Theobalt, C., Pons-Moll, G., Huang, J.B., Golyanik, V., Ilg, E.: Recent trends in 3D reconstruction of general non-rigid scenes. Computer Graphics Forum (2024)
- 108. Zeng, Y., Jiang, Y., Zhu, S., Lu, Y., Lin, Y., Zhu, H., Hu, W., Cao, X., Yao, Y.: STAG4D: Spatial-temporal anchored generative 4D Gaussians. arXiv preprint arXiv:2403.14939 (2024)

- 109. Zhang, B., Yang, T., Li, Y., Zhang, L., Zhao, X.: Compress3D: a compressed latent space for 3D generation from a single image. arXiv preprint arXiv:2403.13524

(2024)

- 110. Zhang, Q., Wang, C., Siarohin, A., Zhuang, P., Xu, Y., Yang, C., Lin, D., Zhou, B., Tulyakov, S., Lee, H.Y.: SceneWiz3D: Towards text-guided 3D scene composition. Proc. CVPR (2024)
- 111. Zhao, Y., Yan, Z., Xie, E., Hong, L., Li, Z., Lee, G.H.: Animate124: Animating one image to 4D dynamic scene. arXiv preprint arXiv:2311.14603 (2023)
- 112. Zheng, Y., Li, X., Nagano, K., Liu, S., Hilliges, O., De Mello, S.: A unified approach for text-and image-guided 4D scene generation. Proc. CVPR (2024)
- 113. Zhou, D., Wang, W., Yan, H., Lv, W., Zhu, Y., Feng, J.: Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018

(2022)

- 114. Zhou, X., Ran, X., Xiong, Y., He, J., Lin, Z., Wang, Y., Sun, D., Yang, M.H.: GALA3D: Towards text-to-3D complex scene generation via layout-guided generative Gaussian splatting. arXiv preprint arXiv:2402.07207 (2024)

#### A Video Results

We include an extensive set of 4D generation results in video format in the supplementary website https://sherwinbahmani.github.io/tc4d, best viewed with Google Chrome. There, we showcase compositional 4D scenes, a diverse set of trajectory-conditioned 4D results and comparisons to 4D-fy [4].

#### B Implementation Details

SDS guidance. We implement our approach based on the threestudio framework [26], which includes implementations of MVDream [73] for 3D-aware textto-image diffusion and score distillation sampling (SDS), ProlificDreamer [93] with Stable Diffusion [69] (text-to-image diffusion and VSD), and we implement video SDS using VideoCrafter2 [14].

Hyperparameter values. We initialize the 4D neural representation following [46, 64] and add an offset to the density predicted by the network in the center of the scene to promote object-centric reconstructions in the static stage. For the deformation network, we use a 4D hash grid with 8 levels, 2 features per level, and a base resolution of 4. We decode the dynamic hash grid features into a 3D deformation vector with a small two layer multi-layer perceptron (MLP) with 64 hidden features. We set the learning rates to 0.01 for the static hash map, 0.001 for the deformation hash map, and 0.0001 for the deformation MLP. The number of training iterations for the two static stages of hybrid SDS [4] as well as the dynamic stage are set to 10000, 10000, and 10000, respectively. We set the probability for hybrid SDS in the static stage to 0.5, following 4D-fy [4].

The intrinsic motion scale Mv is set to 0.3 for all text prompts. Increasing Mv leads to larger jumps between each timestep and a tendency towards lowfrequency motion; smaller values of Mv lead to high motion variation between nearby samples on the trajectory and high-frequency motion. Overall, this hyperparameter depends on the specific text prompt, and per prompt tuning could yield improved results. For simplicity and a fair comparison to 4D-fy, which does not tune any hyperparameters per text prompt, we set Mv to be the same across all text prompts.

Rendering. Each of the pretrained diffusion models has a different native resolution, so we render images from network accordingly. We render four images from different camera positions for the 3D-aware SDS update at the native (256×256 pixel) resolution of the 3D-aware text-to-image model. The VSD update is computed by rendering a 256×256 image and bilinearly upsampling the image to the native resolution of Stable Diffusion (512×512). Finally, the video SDS update is computed by rendering 16 video frames at 128×80 resolution and upsampling to the native 512×320 resolution of VideoCrafter2. To render a background for the image, we optimize a second small MLP that takes in a ray direction and returns a background color. At inference time, we render all videos on the same grey background.

Camera pose sampling. For training, we sample one random view per VSDS update shared across all trajectory timesteps. For each iteration, we randomly sample a camera on a sphere with radius in [1.5, 2.0], field of view in [15◦, 60◦], elevation in [0◦, 30◦], and azimuth in [-180◦, 180◦]. For inference, we render videos with radius 1.75, field of view 40◦, elevation 15◦, and equidistant azimuth samples in [-180◦, 180◦]. The radius and field of view are selected to ensure that objects largely stay within the image frames.

Computation. We optimized the model on a single NVIDIA A100 GPU. The entire procedure requires roughly 80 GB of VRAM and the three optimization stages require approximately 1, 2, and 7 hours of compute, respectively. The first two optimization stages follow the static optimization procedure of 4D-fy, and the third optimization stage involves training our deformation model. In comparison to the baseline 4D-fy, which takes 23 hours in total, our third stage training takes significantly less time to converge while demonstrating more motion.

- 4D-fy baseline. While 4D-fy [4] is an object-centric text-to-4D method, we extend it to a trajectory-conditioned generation method for a fair comparison. Concretely, we use identical trajectories as for our methods as input and apply the trajectory post-training as a naive approach to combining text-to-4D generation with a trajectory. We use the same video diffusion model VideoCrafter2 [14] as guidance for TC4D and 4D-fy for a fair comparison.

#### C Supplemental Experiments

Comparison with DreamGaussian4D. We further compare our method with the concurrent work DreamGaussian4D [68]. Similar to the 4D-fy baseline, we naively extend DreamGaussian4D by animating the generated models post-hoc using same trajectories as TC4D for fair comparison. Note that DreamGaussian4D is an image-to-4D method, hence we generate the input image with StableDiffusion 2.1 [69]. We notice that the 3D structure of DreamGaussian4D is sensitive to the input image being perfectly aligned in a front facing view. While re-running StableDiffusion with multiple seeds can minimize the issue, this does not provide a fully automated approach. We believe that some sort of initial pose estimation as part of the DreamGaussian4D pipeline could further improve the results; however this is out of the scope of our comparison as it requires architectural adjustments to the DreamGaussian4D baseline. In contrast, our method is robust to varying random seeds as we do not rely on front facing input images. We show a visual comparison in Fig. 6 and quantitative results in Tab. 2. We additionally ran a separate user study with 17 participants who each evaluated 33 comparisons between our method and the DreamGaussian4D baseline and obtained results overwhelmingly in our favor, which we show in Tab. 3.

CLIP Score. In addition to the user study results, we evaluate all methods using CLIP score following the protocol in 4D-fy [4]. CLIP Score [61] evaluates the

TC4D DreamGaussian4D

|[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>"an astronaut riding a horse"<br><br>time<br><br>|
|---|

|[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]|
|---|

viewpoint1viewpoint2viewpoint1viewpoint2

|[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]|
|---|

|[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]|
|---|

|[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>"a sheep running"|
|---|

|[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]|
|---|

|[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]|
|---|

|[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]|
|---|

###### Fig. 6: Comparison to DreamGaussian4D. We compare 4D scenes generated with TC4D and DreamGaussian4D [68]. The proposed approach generates more realistic scenes with greater amounts of motion (see supplemental video).

Method CLIP Ours 32.01 4D-fy 31.90 DreamGaussian4D 28.70 Ablation Study

Ours 32.69 w/o local deformation 32.11 w/o trajectory 32.16 w/o traj. training 32.16 w/o traj.-aware VSDS 32.62 w/o smooth. penalty 32.68 w/o timestep annealing 32.62

- Table 2: CLIP Score results. Note that the ablations were evaluated with a subset of the prompts to save computational costs, hence the ablation CLIP score of our method differs from the main CLIP score.

correlation between a text prompt and an image. Specifically, this corresponds to the cosine similarity between textual CLIP [67] embedding and visual CLIP [67] embedding. For all text prompts, we render a video of the generated 4D scene following the camera trajectory used 4D-fy (i.e., the camera moves in azimuth at a fixed elevation angle). All rendered video frames are scored using CLIP ViTB/32, and we report the average CLIP score over all frames and text prompts.

We show CLIP scores in Tab. 2. Note that except for DreamGaussian4D, which generally shows low per-frame quality, the scores are very similar across methods and ablations. The CLIP model only takes a single frame as input, and so it cannot be expected to assess motion quality. Since the main difference between our method and the baselines relates to motion along the trajectory, we find that the CLIP score is a relatively weak evaluation metric. Nevertheless, for the sake of completeness, we include CLIP scores below. We note that the user study and supplementary video results clearly show the motion benefits of our method. Developing automated video metrics for text-to-4D generation is an important direction for future work.

User Study. We give additional details of our user study. Following the protocol in 4D-fy [4], users were prompted with 63 pairs of videos. Each pair contains a result generated from our method and from a comparison method, with the same prompt used to generate both. The evaluator is then asked to select their preference between each video, or to indicate equal preference. In the main paper, we aggregate selections for ours and ties, where ties count equally towards ours and theirs. In Tab. 3, we show the full breakdown between ours, ties, and theirs. As can be seen, the majority of evaluations for appearance quality and 3D structure quality are ties, while the majority of evaluations for motion quality and motion amount prefer our the proposed method.

Human Preference (Ours vs. Method) Method AQ SQ MQ

4D-fy 31%/48%/21% 28%/49%/23% 82%/11%/7% DreamGaussian4D 99.8%/0.0%/0.2% 99.8%/0.0%/0.2% 99.6%/0.2%/0.2%

Ablation Study

w/o local deformation 29%/58%/13% 29%/56%/15% 94%/6%/0% w/o trajectory 32%/52%/16% 34%/55%/11% 72%/20%/8% w/o traj. training 25%/68%/7% 22%/66%/12% 78%/16%/6% w/o traj.-aware VSDS 26%/69%/5% 23%/70%/7% 72%/23%/5% w/o global transformations 34%/62%/4% 24%/68%/8% 94%/5%/1% w/o smooth. penalty 23%/69%/8% 24%/69%/7% 48%/44%/8% w/o timestep annealing 29%/63%/8% 28%/69%/3% 64%/34%/2%

Human Preference (Ours vs. Method) Method MA TA Overall

4D-fy 88%/6%/6% 64%/33%/3% 78%/13%/9% DreamGaussian4D 99.3%/0.5%/0.2% 98.5%/1.3%/0.2% 99.8%/0.0%/0.2%

Ablation Study

w/o local deformation 96%/3%/1% 75%/24%/1% 93%/6%/1% w/o trajectory 78%/15%/7% 66%/30%/4% 78%/14%/8% w/o traj. training 87%/9%/4% 60%/38%/2% 82%/13%/5% w/o traj.-aware VSDS 72%/22%/6% 48%/49%/3% 73%/22%/5% w/o global transformations 93%/2%/5% 71%/29%/0% 94%/6%/0% w/o smooth. penalty 48%/38%/14% 30%/64%/6% 54%/34%/12% w/o timestep annealing 65%/32%/3% 44%/54%/2% 68%/28%/4%

- Table 3: Full user study results. We compare TC4D to modified versions of 4D-fy and DreamGaussian4D. The methods are evaluated in a user study in which participants indicate their preference (or no preference) based on appearance quality (AQ),

- 3D structure quality (SQ), motion quality (MQ), motion amount (MA), text alignment (TA), and overall preference (Overall). We also conduct ablations using a subset of the text prompts. The percentages indicate user selections for ours/ties/theirs.

#### D Automated Generation

The trajectory control points used to generate results shown in the main text are set manually—this facilitates generating a diverse set of results. Moreover, using a sparse set of control points facilitates user-controllable motion in 4D generation.

However, to demonstrate that our method can be automated, we explore the potential of large language models to fully automate the trajectory-conditioned text-to-4D generation process. Specifically, we use ChatGPT to specify the scene layout, including all object sizes and trajectory control points. The output of ChatGPT is a configuration file that can be input to our code to instantiate the 4D scene. We visualize the results of this procedure in Fig. 7. While the result has some imperfections, e.g., the fire hydrant is too small compared to the other objects, we envision future work in this direction could enable high-fidelity

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

"a cat walking"

"a goat walking"

[Figure 156]

[Figure 157]

trajectory

t = 0.0

t = 0.25

[Figure 158]

[Figure 159]

"a corgi running"

"a sheep walking"

"a fire hydrant spraying water"

t = 0.5 t = 0.75

- Fig. 7: Example of automated scene generation. We use ChatGPT to automatically configure the layout of a scene from a text prompt. Specifically, ChatGPT takes the object descriptions as input and generates a configuration file that contains the trajectory for each object.

scene layout and complex trajectory modeling for fully automated trajectoryconditioned text-to-4D generation. We show the conversation with ChatGPT used to generate the trajectories in Figs. 8 and 9.

Moreover, we show single-object automated results in Fig. 10. Improving automated trajectory generation and multi-object text-to-4D generation using, e.g., scene graphs [22], is an important direction for future work.

#### E Geometry Visualization

We show that TC4D generates scenes with comparable geometry to 4D-fy while introducing significantly more motion in Figs. 11 and 12. Note that the focus of this work is not to improve or provide high-quality geometry but introduce more realistic and larger-scale motion.

[Figure 160]

Suppose I have a list of text prompts and file paths, please summarize below variables based on my input:

- - the number of prompts as ${N};
- - the subject of each prompt as ${obj1}, ${obj2}, ..., ${objN}. For example, the subject of prompt "a dog is running" is "dog";
- - a list of text prompts, ${prompt1}, ${prompt2}, ..., ${promptN};
- - a list of file paths, ${path1}, ${path2}, ..., ${pathN}.
- - Based on the real-world common knowledge, give me the order of these subjects based on their size.

Now, my input text prompts are ["a firehydrant spraying water", "a cat walking", "a corgi running", "a sheep walking", "a goat walking"], my input file paths are [ckpt1, ckpt2, ckpt3, ckpt4, ckpt5].

Based on the provided input: ${N} = 5 (there are 5 prompts) ${obj1} = firehydrant, ${obj2} = cat, ${obj3} = corgi, ${obj4} = sheep, ${obj5} = goat ${prompt1} = "a firehydrant spraying water", ${prompt2} = "a cat walking", ${prompt3} = "a corgi running", ${prompt4} = "a sheep walking", ${prompt5} = "a goat walking" ${path1} = ckpt1, ${path2} = ckpt2, ${path3} = ckpt3, ${path4} = ckpt4, ${path5}

= ckpt5 Ordering the subjects based on their typical sizes: Sheep Goat Corgi Cat Firehydrant This order is based on common knowledge of typical sizes of these objects. Now based the variables you summarized, please sample the variables ${radius} and ${size1}, ${size2}, ..., ${sizeN} for me. They should satisfy all the below requirements, please consider them jointly:

[Figure 161]

- - ${radius} is within 0.2 to 0.3;
- - ${size1} is within 0.05 to 0.15;
- - The size variables ${size2}, ..., ${sizeN} of each subject should follow their size order (descending) in the previous answer. If the x-th subject ${objx} is usually larger than the y-th subject ${objy} in object size, then ${sizex} should be larger than ${sizey};
- - The ${size2}, ..., ${sizeN} are all smaller than 2 * ${radius};
- - The ${size2}, ..., ${sizeN} are all within 0.3 to 0.4;

[Figure 162]

Now based the variables you summarized and sampled in the above answers, please summarize below variables we have:

Variable:

- - ${N}, as in previous answer;
- - multiple subjects ${obj1}, ${obj2}, ..., ${objN}, as in previous answer;
- - a list of text prompts, ${prompt1}, ${prompt2}, ..., ${promptN}, as in previous answer;
- - a list of file paths, ${path1}, ${path2}, ..., ${pathN}, as in previous answer;
- - ${radius}, as in previous answer;

[Figure 163]

[Figure 164]

Here are the sampled values that satisfy all the requirements: ${radius} = 0.25

- ${size1} = 0.1
- ${size2} = 0.35
- ${size3} = 0.34
- ${size4} = 0.33
- ${size5} = 0.32 These values satisfy all the given constraints. ${radius} is between 0.2 and 0.3,

- ${size1} is between 0.05 and 0.15, ${size2} to ${size5} follow the size order of subjects, all ${size2} to ${size5} are smaller than 2 * ${radius}, and all
- ${size2} to ${size5} are within 0.3 to 0.4.

[Figure 165]

Based on the provided information and sampled values: Variables: ${N} = 5 ${obj1} = firehydrant, ${obj2} = cat, ${obj3} = corgi, ${obj4} = sheep, ${obj5} = goat ${prompt1} = "a firehydrant spraying water", ${prompt2} = "a cat walking", ${prompt3} = "a corgi running", ${prompt4} = "a sheep walking", ${prompt5} = "a goat walking" ${path1} = ckpt1, ${path2} = ckpt2, ${path3} = ckpt3, ${path4} = ckpt4, ${path5}

= ckpt5 ${radius} = 0.25 ${size1} = 0.1, ${size2} = 0.35, ${size3} = 0.34, ${size4} = 0.33, ${size5} = 0.32 ${circle_low} = 0.5 - ${radius} = 0.5 - 0.25 = 0.25 ${circle_high} = 0.5 + ${radius} = 0.5 + 0.25 = 0.75

These variables represent the necessary information as per the given requirements.

###### Fig. 8: Automatic scene generation using ChatGPT (part 1). We provide a listing with the ChatGPT conversation used to generate the scene trajectories (continues in the next listing).

[Figure 166]

Please generate a configuration yaml file by filling in the missing information with the variable values. All the variables should be consistent in your response:

- Step 1: Put the text prompts ${prompt1}, ${prompt2}, ..., ${promptN} into the `config_scene.prompt_processor.prompt` field.
- Step 2: Put the file paths ${path1}, ${path2}, ..., ${pathN} into the `config_scene.checkpoints` field.
- Step 3: Put each subject into the corresponding ${obj1}, ${obj2}, ..., ${objN}.
- Step 4: Put ${size1}, ${size2}, ..., ${sizeN} into the corresponding `config_scene.traj_kwargs.proxy_size` field.
- Step 5: Put ${circle_low} and ${circle_high} into the `config_scene.traj_kwargs.coords` field.
- Step 6: Calculate `time_offset` for ${obj2}, ..., ${objN}, where the `time_offset` for the i-th objects is (i-1)/${N}.
- Step 7: Output the final configuration yaml file to me.

[Figure 167]

name: "firehydrant_cat_corgi_sheep_goat" config_scene:

traj_kwargs: # firehydrant (static)

-

proxy_size: [0.1, 0.1, 0.1] trajs:

- traj_type: static-trajectory pos_start: [0.5, 0.5, 0.5] angle_offset: 90.

# cat (dynamic)

-

proxy_size: [0.35, 0.35, 0.35] trajs:

- traj_type: spline-trajectory coords: [[0.25, 0.5, 0.5], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5],

[0.5, 0.25, 0.5], [0.25, 0.5, 0.5]] time_offset: 0.0 angle_offset: 0.vv

# corgi (dynamic)

-

proxy_size: [0.34, 0.34, 0.34] trajs:

- traj_type: spline-trajectory coords: [[0.25, 0.5, 0.5], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5],

[0.5, 0.25, 0.5], [0.25, 0.5, 0.5]] time_offset: 0.25 angle_offset: 0.

# sheep (dynamic)

-

proxy_size: [0.33, 0.33, 0.33] trajs:

- traj_type: spline-trajectory coords: [[0.25, 0.5, 0.5], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5], [0.5, 0.25, 0.5], [0.25, 0.5, 0.5]]

time_offset: 0.5 angle_offset: 0.

# goat (dynamic)

-

proxy_size: [0.32, 0.32, 0.32] trajs:

- traj_type: spline-trajectory coords: [[0.25, 0.5, 0.5], [0.5, 0.75, 0.5], [0.75, 0.5, 0.5],

[0.5, 0.25, 0.5], [0.25, 0.5, 0.5]] time_offset: 0.75 angle_offset: 0.

prompt_processor:

prompt: [ "a firehydrant spraying water", "a cat walking", "a corgi running", "a sheep walking", "a goat walking"

] checkpoints: [

- ckpt1,
- ckpt2,
- ckpt3,
- ckpt4,
- ckpt5,

]

###### Fig. 9: Example of automated scene generation (part 2). We provide a listing with the ChatGPT conversation used to generate the scene trajectories (continued from previous listing).

### "a deer walking" "deadpool riding a cow"

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Fig. 10: Results using automated trajectory generation.

rendered scene depth normals

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

- 4D-fy4D-fy4D-fyTC4DTC4DTC4D

time

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

"a bear walking"

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

"a camel walking"

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

"a cat walking"

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

###### Fig. 11: Geometry visualization. We show additional results comparing rendered images, depth maps, and normals from 4D-fy [4] and TC4D. The results from TC4D show improved motion with depth maps and normals that are of similar quality to 4D-fy.

rendered scene depth normals

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

4D-fy4D-fy4D-fyTC4DTC4DTC4D

time

"a deer walking"

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

"son goku riding an elephant"

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

"an assassin riding a cow"

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

###### Fig. 12: Geometry visualization. We show additional results comparing rendered images, depth maps, and normals from 4D-fy [4] and TC4D. The results from TC4D show improved motion with depth maps and normals that are similar to 4D-fy.

