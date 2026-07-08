## Action Images: End-to-End Policy Learning via Multiview Video Generation

Haoyu Zhen1∗, Zixian Gao1∗†, Qiao Sun1†, Yilin Zhao2, Yuncong Yang1, Yilun Du3, Pengsheng Guo4, Tsun-Hsuan Wang4, Yi-Ling Qiao4, and Chuang Gan1

# arXiv:2604.06168v2[cs.CV]15Apr2026

1UMass Amherst 2NVIDIA 3Harvard University 4Genesis AI

https://ActionImages.github.io

[Figure 1]

|[Figure 2]<br><br>[Figure 3]<br><br>2. Pixel-grounded Action Images|
|---|

[Figure 4]

View 1 View 2

1. Observations from Any View

|[Figure 5]<br><br>|[Figure 6]|
|---|
<br><br>[Figure 7]<br><br>|[Figure 8]|
|---|
<br><br>|[Figure 9]|
|---|
<br><br>|[Figure 10]|
|---|
<br><br>[Figure 11]<br><br>|[Figure 12]|
|---|
<br><br>|[Figure 13]|
|---|
<br><br>Text prompt: place blue chip bag in grey bowl<br><br>Robot Videos<br><br>|[Figure 14]|
|---|
<br><br>|[Figure 15]|
|---|
<br><br>3. Video-action Joint Generation 4. Zero-shot 3D Policy<br><br>Action Videos|
|---|

Fig. 1: Action Images turns policy learning as multiview video generation: 7-DoF actions are translated into pixel-grounded action images that explicitly track robot-arm motion, enabling a zero-shot policy directly from a unified video backbone

Abstract. World action models (WAMs) have emerged as a promising direction for robot policy learning, as they can leverage powerful video backbones to model the future states. However, existing approaches often rely on separate action modules, or use action representations that are not pixel-grounded, making it difficult to fully exploit the pretrained knowledge of video models and limiting transfer across viewpoints and environments. In this work, we present Action Images, a unified world action model that formulates policy learning as multiview video generation. Instead of encoding control as low-dimensional tokens, we translate 7-DoF robot actions into interpretable action images: multi-view

* Equal contribution. † This work was done when two of the authors were remote interns at UMass.

action videos that are grounded in 2D pixels and explicitly track robotarm motion. This pixel-grounded action representation allows the video backbone itself to act as a zero-shot policy, without a separate policy head or action module. Beyond control, the same unified model supports video-action joint generation, action-conditioned video generation, and action labeling under a shared representation. On RLBench and real-world evaluations, our model achieves the strongest zero-shot success rates and improves video–action joint generation quality over prior video-space world models, suggesting that interpretable action images are a promising route to policy learning.

### 1 Introduction

World action models [20, 27, 34, 65, 70] have made rapid progress in predicting future observations, but turning this predictive ability into policy generalization remains an open challenge. In particular, strong video generation does not automatically produce a strong policy: a model may successfully synthesize plausible future frames, yet still fail to decide how to act in unseen environments. This gap between video generalization and policy generalization is a central bottleneck for world models.

A key reason is that action is still not represented in a form that world models can naturally generalize. Existing approaches typically follow one of two paths. Some [12, 34, 65, 70, 72] attach a separate policy head or action module on top of a world model, asking an additional network to decode control from learned video features. Others [27] adapt video models to action generation using representations that are not spatially grounded in image space. In both cases, the model’s predictive knowledge of the world is only indirectly connected to acting. As a result, the burden of generalization is shifted to a specialized control moduel, which is often exactly where transfer breaks down.

In this work, we formulate policy learning as video generation and address policy generalization at the representation level. We propose multi-view action videos, a robotics world modeling framework that translates robot actions into interpretable action images and models them together with observations in a unified video-space representation of observation and action. Instead of treating 7-DoF control as low-dimensional signals or latent action codes, we convert each action into a pixel-grounded action representation that explicitly tracks robotarm motion in image space across multiple views. This design makes action native to the video model itself: the same video backbone can observe, predict, condition on, and generate action, enabling a zero-shot policy. By grounding action in pixels rather than in an external interface, we obtain a more generalizable policy model that transfers more naturally across viewpoints and embodiments.

A key design choice is to represent these action images as multi-view videos. The motivation is not merely to add more visual observations, but to bridge the gap between 2D image and the 7-DoF robot action in the 3D space. A single view often provides only a ambiguous projection of motion, making it difficult for the model to infer the full action consistently from pixels alone [70, 73]. Using

multiple views makes the pixel-grounded action image more reconstructable, while also improving robustness when some motion is partially occluded.

Beyond control, the same unified video-space representation of observation and action supports multiple tasks within a single model. Because observation and action share the same generative space, the model can perform video-action joint generation, action-conditioned video generation, and action labeling under one backbone and one training objective. These capabilities emerge without a separate policy head or action module, showing that a robotics world model can be trained not only to predict the world, but also to act in it through a common visual representation.

In summary, our contributions are as follows:

- – We identify the gap between video generalization and policy generalization as a central limitation of current robotics world models, and argue that this gap can be addressed at the level of action representation.
- – We propose multi-view action representation, which translate robot control into interpretable action images forming a pixel-grounded action representation, and use this representation to build a zero-shot policy without a separate policy head or action module.
- – We show that this design yields a more generalizable policy model and provides a unified video-space representation of observation and action that supports video-action joint generation, action-conditioned video generation, and action labeling within a single robotics world model.

### 2 Related Work

#### 2.1 Robotics World Models.

Originating from Reinforcement Learning [41, 53], world models typically take actions and the current state as input and predict future states [2, 9]. In recent years, learning world models for diverse robotic applications [5, 17, 32, 60, 69] has garnered significant interest. With the success of video generation models, lots of work has developed robotics world models based on video generation [8, 12, 16, 30, 52, 70, 72]. These video-based approaches typically adopt a two-stage pipeline, where future observations are first predicted and actions are then generated based on these predictions. More recently, joint video-action generation has been explored to unify modeling and control [27, 34]. In particular, DreamZero [65] demonstrates strong zero-shot generalization and crossembodiment transfer. However, these methods encode actions with additional action modules, leaving much of the pretrained video knowledge underused; we instead use multi-view action images so the backbone itself is a zero-shot policy. Concurrent work [33] also investigates video-based formulations for robot policy learning. Our approach differs in representing actions as pixel-grounded multiview images that encode full 7-DoF control, enabling a unified video-action space and eliminating the need for separate modules.

#### 2.2 Generalist Robot Policy Models.

Policy models map current states to future actions [46, 58]. Developing generalist control policies that can succeed in diverse tasks and can be lightweightly finetuned to adapt to downstream tasks has long been a central goal [7, 18, 35, 42, 54, 63, 67]. While multiple advances in Vision-Language-Action (VLA) models [6, 22, 28, 74], Diffusion Policy [10, 44], and Reinforcement Learning [21, 42] have greatly promoted the generalizability of policy models, their diversity is still limited to relatively narrow task distributions and they struggle to zero-shot generalize to new environments [13, 71]. In parallel, strong capabilities of video generation foundation models in predicting future frames and modeling physical dynamics have inspired policy learning approaches [20, 34, 37]. However, how to turn video prediction into transferable control remains nontrivial; our actionframe representation bridge this gap by making action native to the video space.

#### 2.3 4D Generation Models.

“4D” here refers to 3D plus time. Optimization-based methods employ Score Distillation Sampling, which distills pre-trained diffusion models into specific

##### 4D representations [3, 49, 51, 64]. Recent work [36] explores native 4D generation, which is trained directly on 4D datasets. Due to the lack of large-scale pretraining assets, a branch of research leverages the rich semantic priors in pretrained video generation models and integrates reconstruction methods to lift

- 2D frame sequences into 4D results [24, 43, 59, 62, 66]. However, these contributions mostly focus on single-avatar or simple scene generation. Close to our method, [4, 61] leverage multiview generation to produce complex dynamic 4D scenes that can be replayed at any specified camera pose and timestamp. However, for robotic tasks, 4D generation is typically limited to a fixed single view [16, 70, 73]. Although [40] has leveraged multi-view inputs and introduced a geometry-consistent supervision, they still do not generalize well beyond their training scenes.
- 3 Method

Robotics world models have recently shown strong capability in modeling dynamics, especially when built on large pretrained video backbones. However, these advances in video prediction do not directly translate into strong policy generalization. To address this limitation, we build a unified video-space representation of observation and action, where robot control is translated into interpretable action images that form a pixel-grounded representation. We first introduce how 7-DoF robot actions are converted into multi-view action videos (Sec. 3.1), then describe how this representation can be decoded back into continuous control with only minor information loss (Sec. 3.2), and finally present the training of a unified world-action model that enables a zero-shot policy (Sec. 3.3).

[Figure 16]

2D Gaussian Point

[Figure 17]

[Figure 18]

[Figure 19]

Up Point & Gripper Openness Normal Point

[Figure 20]

Position Point

Normal Point

- Fig. 2: Action as image. We convert each 7-DoF robot action into three semantic

- 3D points (position, normal, and up), project them into image space, and render them as RGB Gaussian heatmaps. The blue channel further encodes gripper openness in the low-response background, producing a pixel-grounded action representation.

#### 3.1 Action as Images

Our central idea is to represent robot action in the same modality as robot observation. Instead of treating action as a low-dimensional control vector that must be interpreted by a separate policy head, we convert each action into interpretable action images and model it directly in video space. This yields a pixel-grounded action representation that is aligned with the robot RGB stream and can therefore be processed by the same video backbone. As illustrated in Fig. 2, this design turns action modeling into a tracking-like visual prediction problem: the model does not need to infer control from abstract tokens, but instead learns to localize and reason about robot-arm motion.

From 7-DoF action to semantic 3D points. At each time step t, the robot action is at = [pt, θt, gt] ∈ R7, where pt ∈ R3 is the end-effector position, θt ∈ R3 denotes its orientation, and gt ∈ R is the gripper openness. We convert this 7-DoF action into three semantic 3D points: a position point, a normal point, and an up point. The position point is the end-effector position, qpost = pt. The other two points are defined by rotating two canonical axes attached to the end-effector and extending them by a small length ℓ:

qupt = pt + ℓR(θt)ex, qnormalt = pt + ℓR(θt)(−ez), (1)

where R(θt) ∈ SO(3) is the rotation matrix derived from the action orientation. Here, the up point follows a canonical in-plane direction of the gripper, while

the normal point follows the direction normal to the robot gripper plane. Together, these three points capture end-effector pose in a form that can be directly projected into image space.

Multi-view action image rendering. Given a camera view v, we project the three semantic 3D points into image space using the camera intrinsics and

extrinsics. Denoting the corresponding projection function by πt(v)(·), we obtain

upost ,(v) = πt(v)(qpost ), unormalt ,(v) = πt(v)(qnormalt ), uupt ,(v) = πt(v)(qupt ). (2)

We then render these projected points into an action image A(tv) ∈ RH×W×3 using 2D Gaussian. The red channel encodes the position point, the green channel encodes the normal point, and the blue channel encodes the up point together with the gripper openness, as shown in Fig. 2. Let G(x;u,σ) denote a 2D Gaussian centered at pixel u. The red and green channels are defined as

A(tv)(:,:,1) = G(·;upost ,(v),σ), A(tv)(:,:,2) = G(·;unormalt ,(v),σ). (3) For the blue channel, we first render the up point as a Gaussian map,

A˜ (tv)(:,:,3) = G(·;uupt ,(v),σ), (4) and then inject the binary gripper openness signal into low-response regions:

A ˜ (tv)(i,j,3), A˜ (tv)(i,j,3) > 0.25, 0.25 · gt, otherwise,

A(tv)(i,j,3) =

(5)

In this way, the blue channel preserves the projected up point while also encoding gripper openness in a simple and spatially consistent form. The resulting image is an interpretable action image. Stacking these frames over time yields an action video for each view,

A(v) = {A(1v),...,A(Tv)} ∈ RT×H×W×3. (6)

Since these action videos have the same spatial and temporal structure as the corresponding robot RGB observations O(v) ∈ RT×H×W×3, they naturally form a unified video-space representation of observation and action.

Benefits. Representing actions as interpretable action images provides two key benefits. First, it makes action prediction spatially grounded: the model learns control through visible robot-arm motion rather than through abstract action tokens. Second, it is naturally compatible with pretrained video backbones, allowing the same model to reason over observation and action without an action module. In this way, our zero-shot policy is obtained by turning the robot action into a visual prediction problem. Because the representation is pixel-grounded and multi-view, it transfers more naturally across viewpoints, motion patterns, and robot embodiments, leading to a more generalizable policy model.

[Figure 21]

[Figure 22]

Main View Action Frame A

Side View Action Frame A

Action 𝒂[𝟏…𝑻]

[Figure 23]

1. Select point from Heatmap

𝑎

3.BackProjection

4. Select Best Match

2. Ray Casting

- Fig. 3: Action images decoding. A 2D heatmap point is selected in the main view, lifted to 3D by ray casting and side-view matching, and repeated for all semantic points to recover the original 7-DoF action.

#### 3.2 Action Images Decoding

A useful action representation should not only be easy to generate, but easy to decode back into continuous robot control. We therefore design a simple decoding method that maps the generated multi-view action videos back to the original 7-DoF action. The decoder first reads the gripper state directly from the blue channel, then reconstructs the underlying 3D semantic points from multi-view heatmaps, and finally converts them back into the action vector. In this way, the same unified video-space representation of observation and action can be used both for generation and for control.

Decoding gripper openness. The blue channel stores both one projected semantic point and the gripper openness, where the latter is written into low-

response background regions. Let A(tv)(:,:,3) denote the blue channel of the action image at time t and view v. We estimate gripper openness by averaging only the low-response pixels:

1 0.25 ·

1 |Ωt|

A(tv)(i,j,3), Ωt = {(i,j,v) | A(tv)(i,j,3) < 0.25}. (7)

gˆt =

(i,j,v)∈Ωt

Reconstructing 3D semantic points from multi-view heatmaps. For the remaining action information, we decode each semantic point from its corresponding heatmap using a simple multi-view geometric procedure. As illustrated in Fig. 3, we first select a 2D point from the heatmap in the main view by weighted averaging:

H(1)t (i,j) i + 0.5, j + 0.5 ⊤

uˆ(1)t = i,j

. (8)

i,j H(1)t (i,j)

where H(1)t ∈ [0,1]H×W is the heatmap in the main view. This gives the centroid of the heat distribution and serves as the 2D anchor point for decoding.

Starting from this point, we cast a ray from the main-view camera center

through uˆ(1)t , and sample a set of candidate 3D points along the ray between a near plane and a far plane. Each candidate is then projected into the side view, where it is scored against the corresponding side-view heatmap. We choose the

- 3D point whose projection best matches the side-view response. Concretely, if {xt,k}Kk=1 denotes the sampled 3D candidates along the ray, then we select

H(2)t πt(2)(xt,k) , (9)

xˆt = arg max

xt,k

where πt(2)(·) is the side-view projection and H(2)t is the side-view heatmap. In practice, this procedure is repeated for each semantic point heatmap in the action image, yielding a set of reconstructed 3D points. The main view provides the image-space anchor for ray casting, while the side view resolves the depth ambiguity by selecting the best match along the ray.

From reconstructed points back to 7-DoF action. Once the semantic 3D points are reconstructed, the original action can be recovered directly. Let qˆpost , qˆupt , and qˆnormalt denote the decoded 3D points. We recover the position as pˆt = qˆpost , define eˆxt = norm(qˆupt − qˆpost ) and eˆzt = norm(qˆpost − qˆnormalt ), then obtain eˆyt = eˆzt × eˆxt , from which the end-effector orientation θˆt is determined. The final decoded action is aˆt = [pˆt,θˆt,gˆt].

Discussion. When the predicted heatmaps are accurate, the remaining decoding error is dominated not by representation mismatch, but by discretization. In particular, the 3D reconstruction accuracy is mainly determined by (i) the sampling interval along the ray, which controls depth precision, and (ii) the spatial resolution of the heatmaps, which controls localization precision in image space. As a result, the information loss introduced by the action-frame parameterization is minor and predictable: finer ray sampling and higher image resolution directly improve the fidelity of the decoded action.

#### 3.3 Training Unified World Action Model

With robot actions represented as interpretable action images, control becomes a pixel-grounded visual signal rather than an abstract low-dimensional vector. This converts action modeling into the same video-space problem as observation modeling, yielding a unified video-space representation of observation and action. As shown in Fig. 4, we build a unified world action model by fine-tuning a large pretrained video generator (Wan 2.2 [56]) to jointly model multi-view robot videos and multi-view action videos under diverse supervision patterns.

Multi-view video-action tokenization and packing. For each camera view v, we have an RGB observation clip V1:(vT) ∈ [0,1]T×H×W×3 and the aligned action-frame clip A(1:vT) ∈ [0,1]T×H×W×3. We first encode both streams into the backbone latent space by the 3D-VAE [29, 56], and then concatenate them temporally to form a single input sequence

Xv = V1:(vT) , A(1:vT) ∈ R(2T)×h×w×c, (10)

[Figure 24]

Diverse Conditions Time

Latent Input under Diverse Tasks and Mask Strategies

Batch 1: Action & Video Joint Gen. Batch 2: Action-Conditioned Video Gen Batch 3: Video-to-Action Labeling

Camera Poses

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

[Figure 36]

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

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

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

[Figure 220]

[Figure 221]

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

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

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

[Figure 293]

[Figure 294]

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

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

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

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

[Figure 1302]

[Figure 1303]

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

[Figure 1352]

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

[Figure 1376]

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

[Figure 1381]

[Figure 1382]

[Figure 1383]

[Figure 1384]

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

[Figure 1391]

[Figure 1392]

[Figure 1393]

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

[Figure 1400]

[Figure 1401]

[Figure 1402]

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

[Figure 1425]

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

[Figure 1432]

[Figure 1433]

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

[Figure 1442]

[Figure 1443]

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

[Figure 1490]

[Figure 1491]

[Figure 1492]

[Figure 1493]

[Figure 1494]

[Figure 1495]

[Figure 1496]

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

[Figure 1529]

[Figure 1530]

[Figure 1531]

[Figure 1532]

[Figure 1533]

[Figure 1534]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

[Figure 1542]

[Figure 1543]

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

[Figure 1551]

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

[Figure 1577]

[Figure 1578]

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

[Figure 1593]

[Figure 1594]

[Figure 1595]

[Figure 1596]

[Figure 1597]

[Figure 1598]

[Figure 1599]

[Figure 1600]

[Figure 1601]

[Figure 1602]

[Figure 1603]

[Figure 1604]

[Figure 1605]

[Figure 1606]

[Figure 1607]

[Figure 1608]

[Figure 1609]

[Figure 1610]

[Figure 1611]

[Figure 1612]

[Figure 1613]

[Figure 1614]

[Figure 1615]

[Figure 1616]

[Figure 1617]

[Figure 1618]

[Figure 1619]

[Figure 1620]

[Figure 1621]

[Figure 1622]

[Figure 1623]

[Figure 1624]

[Figure 1625]

[Figure 1626]

[Figure 1627]

[Figure 1628]

[Figure 1629]

[Figure 1630]

[Figure 1631]

[Figure 1632]

[Figure 1633]

[Figure 1634]

[Figure 1635]

[Figure 1636]

[Figure 1637]

[Figure 1638]

[Figure 1639]

[Figure 1640]

[Figure 1641]

[Figure 1642]

[Figure 1643]

[Figure 1644]

[Figure 1645]

[Figure 1646]

[Figure 1647]

[Figure 1648]

[Figure 1649]

[Figure 1650]

[Figure 1651]

[Figure 1652]

[Figure 1653]

[Figure 1654]

[Figure 1655]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

[Figure 1660]

[Figure 1661]

[Figure 1662]

[Figure 1663]

[Figure 1664]

[Figure 1665]

[Figure 1666]

[Figure 1667]

[Figure 1668]

[Figure 1669]

[Figure 1670]

[Figure 1671]

[Figure 1672]

[Figure 1673]

[Figure 1674]

[Figure 1675]

[Figure 1676]

[Figure 1677]

[Figure 1678]

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

[Figure 1683]

[Figure 1684]

[Figure 1685]

[Figure 1686]

[Figure 1687]

[Figure 1688]

[Figure 1689]

[Figure 1690]

[Figure 1691]

[Figure 1692]

[Figure 1693]

[Figure 1694]

[Figure 1695]

[Figure 1696]

[Figure 1697]

[Figure 1698]

[Figure 1699]

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

[Figure 1704]

[Figure 1705]

[Figure 1706]

[Figure 1707]

[Figure 1708]

[Figure 1709]

[Figure 1710]

[Figure 1711]

[Figure 1712]

[Figure 1713]

[Figure 1714]

[Figure 1715]

[Figure 1716]

[Figure 1717]

[Figure 1718]

[Figure 1719]

[Figure 1720]

[Figure 1721]

[Figure 1722]

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

[Figure 1727]

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

[Figure 1732]

[Figure 1733]

[Figure 1734]

[Figure 1735]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

[Figure 1740]

[Figure 1741]

[Figure 1742]

[Figure 1743]

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

[Figure 1748]

[Figure 1749]

[Figure 1750]

[Figure 1751]

[Figure 1752]

[Figure 1753]

[Figure 1754]

[Figure 1755]

[Figure 1756]

[Figure 1757]

[Figure 1758]

[Figure 1759]

[Figure 1760]

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

[Figure 1765]

[Figure 1766]

[Figure 1767]

[Figure 1768]

[Figure 1769]

[Figure 1770]

[Figure 1771]

[Figure 1772]

[Figure 1773]

[Figure 1774]

[Figure 1775]

[Figure 1776]

[Figure 1777]

[Figure 1778]

[Figure 1779]

[Figure 1780]

[Figure 1781]

[Figure 1782]

[Figure 1783]

[Figure 1784]

[Figure 1785]

[Figure 1786]

[Figure 1787]

[Figure 1788]

[Figure 1789]

[Figure 1790]

[Figure 1791]

[Figure 1792]

[Figure 1793]

[Figure 1794]

[Figure 1795]

[Figure 1796]

[Figure 1797]

[Figure 1798]

[Figure 1799]

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

[Figure 1809]

[Figure 1810]

[Figure 1811]

[Figure 1812]

[Figure 1813]

[Figure 1814]

[Figure 1815]

[Figure 1816]

[Figure 1817]

[Figure 1818]

[Figure 1819]

[Figure 1820]

[Figure 1821]

[Figure 1822]

[Figure 1823]

[Figure 1824]

[Figure 1825]

[Figure 1826]

[Figure 1827]

[Figure 1828]

[Figure 1829]

[Figure 1830]

[Figure 1831]

[Figure 1832]

[Figure 1833]

[Figure 1834]

[Figure 1835]

[Figure 1836]

[Figure 1837]

[Figure 1838]

[Figure 1839]

[Figure 1840]

[Figure 1841]

[Figure 1842]

[Figure 1843]

[Figure 1844]

[Figure 1845]

[Figure 1846]

[Figure 1847]

[Figure 1848]

[Figure 1849]

[Figure 1850]

[Figure 1851]

[Figure 1852]

[Figure 1853]

[Figure 1854]

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

[Figure 1859]

[Figure 1860]

[Figure 1861]

[Figure 1862]

[Figure 1863]

[Figure 1864]

[Figure 1865]

[Figure 1866]

[Figure 1867]

[Figure 1868]

[Figure 1869]

[Figure 1870]

[Figure 1871]

[Figure 1872]

[Figure 1873]

[Figure 1874]

[Figure 1875]

[Figure 1876]

[Figure 1877]

[Figure 1878]

[Figure 1879]

[Figure 1880]

[Figure 1881]

[Figure 1882]

[Figure 1883]

[Figure 1884]

[Figure 1885]

[Figure 1886]

[Figure 1887]

[Figure 1888]

[Figure 1889]

[Figure 1890]

[Figure 1891]

[Figure 1892]

[Figure 1893]

[Figure 1894]

[Figure 1895]

[Figure 1896]

[Figure 1897]

[Figure 1898]

[Figure 1899]

[Figure 1900]

[Figure 1901]

[Figure 1902]

[Figure 1903]

[Figure 1904]

[Figure 1905]

[Figure 1906]

[Figure 1907]

[Figure 1908]

[Figure 1909]

[Figure 1910]

[Figure 1911]

[Figure 1912]

[Figure 1913]

[Figure 1914]

[Figure 1915]

[Figure 1916]

[Figure 1917]

[Figure 1918]

[Figure 1919]

[Figure 1920]

[Figure 1921]

[Figure 1922]

[Figure 1923]

[Figure 1924]

[Figure 1925]

[Figure 1926]

[Figure 1927]

[Figure 1928]

[Figure 1929]

[Figure 1930]

[Figure 1931]

[Figure 1932]

[Figure 1933]

[Figure 1934]

[Figure 1935]

[Figure 1936]

[Figure 1937]

[Figure 1938]

[Figure 1939]

[Figure 1940]

[Figure 1941]

[Figure 1942]

[Figure 1943]

[Figure 1944]

[Figure 1945]

[Figure 1946]

[Figure 1947]

[Figure 1948]

[Figure 1949]

[Figure 1950]

[Figure 1951]

[Figure 1952]

[Figure 1953]

[Figure 1954]

[Figure 1955]

[Figure 1956]

[Figure 1957]

[Figure 1958]

[Figure 1959]

[Figure 1960]

[Figure 1961]

[Figure 1962]

[Figure 1963]

[Figure 1964]

[Figure 1965]

[Figure 1966]

[Figure 1967]

[Figure 1968]

[Figure 1969]

[Figure 1970]

[Figure 1971]

[Figure 1972]

[Figure 1973]

[Figure 1974]

[Figure 1975]

[Figure 1976]

[Figure 1977]

[Figure 1978]

[Figure 1979]

[Figure 1980]

[Figure 1981]

[Figure 1982]

[Figure 1983]

[Figure 1984]

[Figure 1985]

[Figure 1986]

[Figure 1987]

[Figure 1988]

[Figure 1989]

[Figure 1990]

[Figure 1991]

[Figure 1992]

[Figure 1993]

[Figure 1994]

[Figure 1995]

[Figure 1996]

[Figure 1997]

[Figure 1998]

[Figure 1999]

[Figure 2000]

[Figure 2001]

[Figure 2002]

[Figure 2003]

[Figure 2004]

[Figure 2005]

[Figure 2006]

[Figure 2007]

[Figure 2008]

[Figure 2009]

[Figure 2010]

[Figure 2011]

[Figure 2012]

[Figure 2013]

[Figure 2014]

[Figure 2015]

[Figure 2016]

[Figure 2017]

[Figure 2018]

[Figure 2019]

[Figure 2020]

[Figure 2021]

[Figure 2022]

[Figure 2023]

[Figure 2024]

[Figure 2025]

[Figure 2026]

[Figure 2027]

[Figure 2028]

[Figure 2029]

[Figure 2030]

[Figure 2031]

[Figure 2032]

[Figure 2033]

[Figure 2034]

[Figure 2035]

[Figure 2036]

[Figure 2037]

[Figure 2038]

[Figure 2039]

[Figure 2040]

[Figure 2041]

[Figure 2042]

[Figure 2043]

[Figure 2044]

[Figure 2045]

[Figure 2046]

[Figure 2047]

[Figure 2048]

[Figure 2049]

[Figure 2050]

[Figure 2051]

[Figure 2052]

[Figure 2053]

[Figure 2054]

[Figure 2055]

[Figure 2056]

[Figure 2057]

[Figure 2058]

[Figure 2059]

[Figure 2060]

[Figure 2061]

[Figure 2062]

[Figure 2063]

[Figure 2064]

[Figure 2065]

[Figure 2066]

[Figure 2067]

[Figure 2068]

[Figure 2069]

[Figure 2070]

[Figure 2071]

[Figure 2072]

[Figure 2073]

[Figure 2074]

[Figure 2075]

[Figure 2076]

[Figure 2077]

[Figure 2078]

[Figure 2079]

[Figure 2080]

[Figure 2081]

[Figure 2082]

[Figure 2083]

[Figure 2084]

[Figure 2085]

[Figure 2086]

[Figure 2087]

[Figure 2088]

- View 1
- View 2

Text Prompt

[Figure 2089]

[Figure 2090]

[Figure 2091]

[Figure 2092]

[Figure 2093]

[Figure 2094]

[Figure 2095]

[Figure 2096]

[Figure 2097]

[Figure 2098]

[Figure 2099]

[Figure 2100]

[Figure 2101]

[Figure 2102]

[Figure 2103]

[Figure 2104]

[Figure 2105]

[Figure 2106]

[Figure 2107]

[Figure 2108]

[Figure 2109]

[Figure 2110]

[Figure 2111]

[Figure 2112]

[Figure 2113]

[Figure 2114]

[Figure 2115]

[Figure 2116]

[Figure 2117]

[Figure 2118]

[Figure 2119]

[Figure 2120]

[Figure 2121]

[Figure 2122]

[Figure 2123]

[Figure 2124]

[Figure 2125]

[Figure 2126]

[Figure 2127]

[Figure 2128]

[Figure 2129]

[Figure 2130]

[Figure 2131]

[Figure 2132]

[Figure 2133]

[Figure 2134]

[Figure 2135]

[Figure 2136]

[Figure 2137]

[Figure 2138]

[Figure 2139]

[Figure 2140]

[Figure 2141]

[Figure 2142]

[Figure 2143]

[Figure 2144]

[Figure 2145]

[Figure 2146]

[Figure 2147]

[Figure 2148]

[Figure 2149]

[Figure 2150]

[Figure 2151]

[Figure 2152]

[Figure 2153]

[Figure 2154]

[Figure 2155]

[Figure 2156]

[Figure 2157]

[Figure 2158]

[Figure 2159]

[Figure 2160]

[Figure 2161]

[Figure 2162]

[Figure 2163]

[Figure 2164]

[Figure 2165]

[Figure 2166]

[Figure 2167]

[Figure 2168]

[Figure 2169]

[Figure 2170]

[Figure 2171]

[Figure 2172]

[Figure 2173]

[Figure 2174]

[Figure 2175]

[Figure 2176]

[Figure 2177]

[Figure 2178]

[Figure 2179]

[Figure 2180]

[Figure 2181]

[Figure 2182]

[Figure 2183]

[Figure 2184]

[Figure 2185]

[Figure 2186]

[Figure 2187]

[Figure 2188]

[Figure 2189]

[Figure 2190]

[Figure 2191]

[Figure 2192]

[Figure 2193]

[Figure 2194]

[Figure 2195]

[Figure 2196]

[Figure 2197]

[Figure 2198]

[Figure 2199]

[Figure 2200]

[Figure 2201]

[Figure 2202]

[Figure 2203]

[Figure 2204]

[Figure 2205]

[Figure 2206]

[Figure 2207]

[Figure 2208]

[Figure 2209]

[Figure 2210]

[Figure 2211]

[Figure 2212]

[Figure 2213]

[Figure 2214]

[Figure 2215]

[Figure 2216]

[Figure 2217]

[Figure 2218]

[Figure 2219]

[Figure 2220]

[Figure 2221]

[Figure 2222]

[Figure 2223]

[Figure 2224]

[Figure 2225]

[Figure 2226]

[Figure 2227]

[Figure 2228]

[Figure 2229]

[Figure 2230]

[Figure 2231]

[Figure 2232]

[Figure 2233]

[Figure 2234]

[Figure 2235]

[Figure 2236]

[Figure 2237]

[Figure 2238]

[Figure 2239]

[Figure 2240]

[Figure 2241]

[Figure 2242]

[Figure 2243]

[Figure 2244]

[Figure 2245]

[Figure 2246]

[Figure 2247]

[Figure 2248]

[Figure 2249]

[Figure 2250]

[Figure 2251]

[Figure 2252]

[Figure 2253]

[Figure 2254]

[Figure 2255]

[Figure 2256]

[Figure 2257]

[Figure 2258]

[Figure 2259]

[Figure 2260]

[Figure 2261]

[Figure 2262]

[Figure 2263]

[Figure 2264]

[Figure 2265]

[Figure 2266]

[Figure 2267]

[Figure 2268]

[Figure 2269]

[Figure 2270]

[Figure 2271]

[Figure 2272]

[Figure 2273]

[Figure 2274]

[Figure 2275]

[Figure 2276]

[Figure 2277]

[Figure 2278]

[Figure 2279]

[Figure 2280]

[Figure 2281]

[Figure 2282]

[Figure 2283]

[Figure 2284]

[Figure 2285]

[Figure 2286]

[Figure 2287]

[Figure 2288]

[Figure 2289]

[Figure 2290]

[Figure 2291]

[Figure 2292]

[Figure 2293]

[Figure 2294]

[Figure 2295]

[Figure 2296]

[Figure 2297]

[Figure 2298]

[Figure 2299]

[Figure 2300]

[Figure 2301]

[Figure 2302]

[Figure 2303]

[Figure 2304]

[Figure 2305]

[Figure 2306]

[Figure 2307]

[Figure 2308]

[Figure 2309]

[Figure 2310]

[Figure 2311]

[Figure 2312]

[Figure 2313]

[Figure 2314]

[Figure 2315]

[Figure 2316]

[Figure 2317]

[Figure 2318]

[Figure 2319]

[Figure 2320]

[Figure 2321]

[Figure 2322]

[Figure 2323]

[Figure 2324]

[Figure 2325]

[Figure 2326]

[Figure 2327]

[Figure 2328]

[Figure 2329]

[Figure 2330]

[Figure 2331]

[Figure 2332]

[Figure 2333]

[Figure 2334]

[Figure 2335]

[Figure 2336]

[Figure 2337]

[Figure 2338]

[Figure 2339]

[Figure 2340]

[Figure 2341]

[Figure 2342]

[Figure 2343]

[Figure 2344]

[Figure 2345]

[Figure 2346]

[Figure 2347]

[Figure 2348]

[Figure 2349]

[Figure 2350]

[Figure 2351]

[Figure 2352]

[Figure 2353]

[Figure 2354]

[Figure 2355]

[Figure 2356]

[Figure 2357]

[Figure 2358]

[Figure 2359]

[Figure 2360]

[Figure 2361]

[Figure 2362]

[Figure 2363]

[Figure 2364]

[Figure 2365]

[Figure 2366]

[Figure 2367]

[Figure 2368]

[Figure 2369]

[Figure 2370]

[Figure 2371]

[Figure 2372]

[Figure 2373]

[Figure 2374]

[Figure 2375]

[Figure 2376]

[Figure 2377]

[Figure 2378]

[Figure 2379]

[Figure 2380]

[Figure 2381]

[Figure 2382]

[Figure 2383]

[Figure 2384]

[Figure 2385]

[Figure 2386]

[Figure 2387]

[Figure 2388]

[Figure 2389]

[Figure 2390]

[Figure 2391]

[Figure 2392]

[Figure 2393]

[Figure 2394]

[Figure 2395]

[Figure 2396]

[Figure 2397]

[Figure 2398]

[Figure 2399]

[Figure 2400]

[Figure 2401]

[Figure 2402]

[Figure 2403]

[Figure 2404]

[Figure 2405]

[Figure 2406]

[Figure 2407]

[Figure 2408]

[Figure 2409]

[Figure 2410]

[Figure 2411]

[Figure 2412]

[Figure 2413]

[Figure 2414]

[Figure 2415]

[Figure 2416]

[Figure 2417]

[Figure 2418]

[Figure 2419]

[Figure 2420]

[Figure 2421]

[Figure 2422]

[Figure 2423]

[Figure 2424]

[Figure 2425]

[Figure 2426]

[Figure 2427]

[Figure 2428]

[Figure 2429]

[Figure 2430]

[Figure 2431]

[Figure 2432]

[Figure 2433]

[Figure 2434]

[Figure 2435]

[Figure 2436]

[Figure 2437]

[Figure 2438]

[Figure 2439]

[Figure 2440]

[Figure 2441]

[Figure 2442]

[Figure 2443]

[Figure 2444]

[Figure 2445]

[Figure 2446]

[Figure 2447]

[Figure 2448]

[Figure 2449]

[Figure 2450]

[Figure 2451]

[Figure 2452]

[Figure 2453]

[Figure 2454]

[Figure 2455]

[Figure 2456]

[Figure 2457]

[Figure 2458]

[Figure 2459]

[Figure 2460]

[Figure 2461]

[Figure 2462]

[Figure 2463]

[Figure 2464]

[Figure 2465]

[Figure 2466]

[Figure 2467]

[Figure 2468]

[Figure 2469]

[Figure 2470]

[Figure 2471]

[Figure 2472]

[Figure 2473]

[Figure 2474]

[Figure 2475]

[Figure 2476]

[Figure 2477]

[Figure 2478]

[Figure 2479]

[Figure 2480]

[Figure 2481]

[Figure 2482]

[Figure 2483]

[Figure 2484]

[Figure 2485]

[Figure 2486]

[Figure 2487]

[Figure 2488]

[Figure 2489]

[Figure 2490]

[Figure 2491]

[Figure 2492]

[Figure 2493]

[Figure 2494]

[Figure 2495]

[Figure 2496]

[Figure 2497]

[Figure 2498]

[Figure 2499]

[Figure 2500]

[Figure 2501]

[Figure 2502]

[Figure 2503]

[Figure 2504]

[Figure 2505]

[Figure 2506]

[Figure 2507]

[Figure 2508]

[Figure 2509]

[Figure 2510]

[Figure 2511]

[Figure 2512]

[Figure 2513]

[Figure 2514]

[Figure 2515]

[Figure 2516]

[Figure 2517]

[Figure 2518]

[Figure 2519]

[Figure 2520]

[Figure 2521]

[Figure 2522]

[Figure 2523]

[Figure 2524]

[Figure 2525]

[Figure 2526]

[Figure 2527]

[Figure 2528]

[Figure 2529]

[Figure 2530]

[Figure 2531]

[Figure 2532]

[Figure 2533]

[Figure 2534]

[Figure 2535]

[Figure 2536]

[Figure 2537]

[Figure 2538]

[Figure 2539]

[Figure 2540]

[Figure 2541]

[Figure 2542]

[Figure 2543]

[Figure 2544]

[Figure 2545]

[Figure 2546]

[Figure 2547]

[Figure 2548]

[Figure 2549]

[Figure 2550]

[Figure 2551]

[Figure 2552]

[Figure 2553]

[Figure 2554]

[Figure 2555]

[Figure 2556]

[Figure 2557]

[Figure 2558]

[Figure 2559]

[Figure 2560]

[Figure 2561]

[Figure 2562]

[Figure 2563]

[Figure 2564]

[Figure 2565]

[Figure 2566]

[Figure 2567]

[Figure 2568]

[Figure 2569]

[Figure 2570]

[Figure 2571]

[Figure 2572]

[Figure 2573]

[Figure 2574]

[Figure 2575]

[Figure 2576]

[Figure 2577]

[Figure 2578]

[Figure 2579]

[Figure 2580]

[Figure 2581]

[Figure 2582]

[Figure 2583]

[Figure 2584]

[Figure 2585]

[Figure 2586]

[Figure 2587]

[Figure 2588]

[Figure 2589]

[Figure 2590]

[Figure 2591]

[Figure 2592]

[Figure 2593]

[Figure 2594]

[Figure 2595]

[Figure 2596]

[Figure 2597]

[Figure 2598]

[Figure 2599]

[Figure 2600]

[Figure 2601]

[Figure 2602]

[Figure 2603]

[Figure 2604]

[Figure 2605]

[Figure 2606]

[Figure 2607]

[Figure 2608]

[Figure 2609]

[Figure 2610]

[Figure 2611]

[Figure 2612]

[Figure 2613]

[Figure 2614]

[Figure 2615]

[Figure 2616]

[Figure 2617]

[Figure 2618]

[Figure 2619]

[Figure 2620]

[Figure 2621]

[Figure 2622]

[Figure 2623]

[Figure 2624]

[Figure 2625]

[Figure 2626]

[Figure 2627]

[Figure 2628]

[Figure 2629]

[Figure 2630]

[Figure 2631]

[Figure 2632]

[Figure 2633]

[Figure 2634]

[Figure 2635]

[Figure 2636]

[Figure 2637]

[Figure 2638]

[Figure 2639]

[Figure 2640]

[Figure 2641]

[Figure 2642]

[Figure 2643]

[Figure 2644]

[Figure 2645]

[Figure 2646]

[Figure 2647]

[Figure 2648]

[Figure 2649]

[Figure 2650]

[Figure 2651]

[Figure 2652]

[Figure 2653]

[Figure 2654]

[Figure 2655]

[Figure 2656]

[Figure 2657]

[Figure 2658]

[Figure 2659]

[Figure 2660]

[Figure 2661]

[Figure 2662]

[Figure 2663]

[Figure 2664]

[Figure 2665]

[Figure 2666]

[Figure 2667]

[Figure 2668]

[Figure 2669]

[Figure 2670]

[Figure 2671]

[Figure 2672]

[Figure 2673]

[Figure 2674]

[Figure 2675]

[Figure 2676]

[Figure 2677]

[Figure 2678]

[Figure 2679]

[Figure 2680]

[Figure 2681]

[Figure 2682]

[Figure 2683]

[Figure 2684]

[Figure 2685]

[Figure 2686]

[Figure 2687]

[Figure 2688]

[Figure 2689]

[Figure 2690]

[Figure 2691]

[Figure 2692]

[Figure 2693]

[Figure 2694]

[Figure 2695]

[Figure 2696]

[Figure 2697]

[Figure 2698]

[Figure 2699]

[Figure 2700]

[Figure 2701]

[Figure 2702]

[Figure 2703]

[Figure 2704]

[Figure 2705]

[Figure 2706]

[Figure 2707]

[Figure 2708]

[Figure 2709]

[Figure 2710]

[Figure 2711]

[Figure 2712]

[Figure 2713]

[Figure 2714]

[Figure 2715]

[Figure 2716]

[Figure 2717]

[Figure 2718]

[Figure 2719]

[Figure 2720]

[Figure 2721]

[Figure 2722]

[Figure 2723]

[Figure 2724]

[Figure 2725]

[Figure 2726]

[Figure 2727]

[Figure 2728]

[Figure 2729]

[Figure 2730]

[Figure 2731]

[Figure 2732]

[Figure 2733]

[Figure 2734]

[Figure 2735]

[Figure 2736]

[Figure 2737]

[Figure 2738]

[Figure 2739]

[Figure 2740]

[Figure 2741]

[Figure 2742]

[Figure 2743]

[Figure 2744]

[Figure 2745]

[Figure 2746]

[Figure 2747]

[Figure 2748]

[Figure 2749]

[Figure 2750]

[Figure 2751]

[Figure 2752]

[Figure 2753]

[Figure 2754]

[Figure 2755]

[Figure 2756]

[Figure 2757]

[Figure 2758]

[Figure 2759]

[Figure 2760]

[Figure 2761]

[Figure 2762]

[Figure 2763]

[Figure 2764]

[Figure 2765]

[Figure 2766]

[Figure 2767]

[Figure 2768]

[Figure 2769]

[Figure 2770]

[Figure 2771]

[Figure 2772]

[Figure 2773]

[Figure 2774]

[Figure 2775]

[Figure 2776]

[Figure 2777]

[Figure 2778]

[Figure 2779]

[Figure 2780]

[Figure 2781]

[Figure 2782]

[Figure 2783]

[Figure 2784]

[Figure 2785]

[Figure 2786]

[Figure 2787]

[Figure 2788]

[Figure 2789]

[Figure 2790]

[Figure 2791]

[Figure 2792]

[Figure 2793]

[Figure 2794]

[Figure 2795]

[Figure 2796]

[Figure 2797]

[Figure 2798]

[Figure 2799]

[Figure 2800]

[Figure 2801]

[Figure 2802]

[Figure 2803]

[Figure 2804]

[Figure 2805]

[Figure 2806]

[Figure 2807]

[Figure 2808]

[Figure 2809]

[Figure 2810]

[Figure 2811]

[Figure 2812]

[Figure 2813]

[Figure 2814]

[Figure 2815]

[Figure 2816]

[Figure 2817]

[Figure 2818]

[Figure 2819]

[Figure 2820]

[Figure 2821]

[Figure 2822]

[Figure 2823]

[Figure 2824]

[Figure 2825]

[Figure 2826]

[Figure 2827]

[Figure 2828]

[Figure 2829]

[Figure 2830]

[Figure 2831]

[Figure 2832]

[Figure 2833]

[Figure 2834]

[Figure 2835]

[Figure 2836]

[Figure 2837]

[Figure 2838]

[Figure 2839]

[Figure 2840]

[Figure 2841]

[Figure 2842]

[Figure 2843]

[Figure 2844]

[Figure 2845]

[Figure 2846]

[Figure 2847]

[Figure 2848]

[Figure 2849]

[Figure 2850]

[Figure 2851]

[Figure 2852]

[Figure 2853]

[Figure 2854]

[Figure 2855]

[Figure 2856]

[Figure 2857]

[Figure 2858]

[Figure 2859]

[Figure 2860]

[Figure 2861]

[Figure 2862]

[Figure 2863]

[Figure 2864]

[Figure 2865]

[Figure 2866]

[Figure 2867]

[Figure 2868]

[Figure 2869]

[Figure 2870]

[Figure 2871]

[Figure 2872]

[Figure 2873]

[Figure 2874]

[Figure 2875]

[Figure 2876]

[Figure 2877]

[Figure 2878]

[Figure 2879]

[Figure 2880]

[Figure 2881]

[Figure 2882]

[Figure 2883]

[Figure 2884]

[Figure 2885]

[Figure 2886]

[Figure 2887]

[Figure 2888]

[Figure 2889]

[Figure 2890]

[Figure 2891]

[Figure 2892]

[Figure 2893]

[Figure 2894]

[Figure 2895]

[Figure 2896]

[Figure 2897]

[Figure 2898]

[Figure 2899]

[Figure 2900]

[Figure 2901]

[Figure 2902]

[Figure 2903]

[Figure 2904]

[Figure 2905]

[Figure 2906]

[Figure 2907]

[Figure 2908]

[Figure 2909]

[Figure 2910]

[Figure 2911]

[Figure 2912]

[Figure 2913]

[Figure 2914]

[Figure 2915]

[Figure 2916]

[Figure 2917]

[Figure 2918]

[Figure 2919]

[Figure 2920]

[Figure 2921]

[Figure 2922]

[Figure 2923]

[Figure 2924]

[Figure 2925]

[Figure 2926]

[Figure 2927]

[Figure 2928]

[Figure 2929]

[Figure 2930]

[Figure 2931]

[Figure 2932]

[Figure 2933]

[Figure 2934]

[Figure 2935]

[Figure 2936]

[Figure 2937]

[Figure 2938]

[Figure 2939]

[Figure 2940]

[Figure 2941]

[Figure 2942]

[Figure 2943]

[Figure 2944]

[Figure 2945]

[Figure 2946]

[Figure 2947]

[Figure 2948]

[Figure 2949]

[Figure 2950]

[Figure 2951]

[Figure 2952]

[Figure 2953]

[Figure 2954]

[Figure 2955]

[Figure 2956]

[Figure 2957]

[Figure 2958]

[Figure 2959]

[Figure 2960]

[Figure 2961]

[Figure 2962]

[Figure 2963]

[Figure 2964]

[Figure 2965]

[Figure 2966]

[Figure 2967]

[Figure 2968]

[Figure 2969]

[Figure 2970]

[Figure 2971]

[Figure 2972]

[Figure 2973]

[Figure 2974]

[Figure 2975]

[Figure 2976]

[Figure 2977]

[Figure 2978]

[Figure 2979]

[Figure 2980]

[Figure 2981]

[Figure 2982]

[Figure 2983]

[Figure 2984]

[Figure 2985]

[Figure 2986]

[Figure 2987]

[Figure 2988]

[Figure 2989]

[Figure 2990]

[Figure 2991]

[Figure 2992]

[Figure 2993]

[Figure 2994]

[Figure 2995]

[Figure 2996]

[Figure 2997]

[Figure 2998]

[Figure 2999]

[Figure 3000]

[Figure 3001]

[Figure 3002]

[Figure 3003]

[Figure 3004]

[Figure 3005]

[Figure 3006]

[Figure 3007]

[Figure 3008]

[Figure 3009]

[Figure 3010]

[Figure 3011]

[Figure 3012]

[Figure 3013]

[Figure 3014]

[Figure 3015]

[Figure 3016]

[Figure 3017]

[Figure 3018]

[Figure 3019]

[Figure 3020]

[Figure 3021]

[Figure 3022]

[Figure 3023]

[Figure 3024]

[Figure 3025]

[Figure 3026]

[Figure 3027]

[Figure 3028]

[Figure 3029]

[Figure 3030]

[Figure 3031]

[Figure 3032]

[Figure 3033]

[Figure 3034]

[Figure 3035]

[Figure 3036]

[Figure 3037]

[Figure 3038]

[Figure 3039]

[Figure 3040]

[Figure 3041]

[Figure 3042]

[Figure 3043]

[Figure 3044]

[Figure 3045]

[Figure 3046]

[Figure 3047]

[Figure 3048]

[Figure 3049]

[Figure 3050]

[Figure 3051]

[Figure 3052]

[Figure 3053]

[Figure 3054]

[Figure 3055]

[Figure 3056]

[Figure 3057]

[Figure 3058]

[Figure 3059]

[Figure 3060]

[Figure 3061]

[Figure 3062]

[Figure 3063]

[Figure 3064]

[Figure 3065]

[Figure 3066]

[Figure 3067]

[Figure 3068]

[Figure 3069]

[Figure 3070]

[Figure 3071]

[Figure 3072]

[Figure 3073]

[Figure 3074]

[Figure 3075]

[Figure 3076]

[Figure 3077]

[Figure 3078]

[Figure 3079]

[Figure 3080]

[Figure 3081]

[Figure 3082]

[Figure 3083]

[Figure 3084]

[Figure 3085]

[Figure 3086]

[Figure 3087]

[Figure 3088]

[Figure 3089]

[Figure 3090]

[Figure 3091]

[Figure 3092]

[Figure 3093]

[Figure 3094]

[Figure 3095]

[Figure 3096]

[Figure 3097]

[Figure 3098]

[Figure 3099]

[Figure 3100]

[Figure 3101]

[Figure 3102]

[Figure 3103]

[Figure 3104]

[Figure 3105]

[Figure 3106]

[Figure 3107]

[Figure 3108]

[Figure 3109]

[Figure 3110]

[Figure 3111]

[Figure 3112]

[Figure 3113]

[Figure 3114]

[Figure 3115]

[Figure 3116]

[Figure 3117]

[Figure 3118]

[Figure 3119]

[Figure 3120]

[Figure 3121]

[Figure 3122]

[Figure 3123]

[Figure 3124]

[Figure 3125]

[Figure 3126]

[Figure 3127]

[Figure 3128]

[Figure 3129]

[Figure 3130]

[Figure 3131]

[Figure 3132]

[Figure 3133]

[Figure 3134]

[Figure 3135]

[Figure 3136]

[Figure 3137]

[Figure 3138]

[Figure 3139]

[Figure 3140]

[Figure 3141]

[Figure 3142]

[Figure 3143]

[Figure 3144]

[Figure 3145]

[Figure 3146]

[Figure 3147]

[Figure 3148]

[Figure 3149]

[Figure 3150]

[Figure 3151]

[Figure 3152]

[Figure 3153]

[Figure 3154]

[Figure 3155]

[Figure 3156]

[Figure 3157]

[Figure 3158]

[Figure 3159]

[Figure 3160]

[Figure 3161]

[Figure 3162]

[Figure 3163]

[Figure 3164]

[Figure 3165]

[Figure 3166]

[Figure 3167]

[Figure 3168]

[Figure 3169]

[Figure 3170]

[Figure 3171]

[Figure 3172]

[Figure 3173]

[Figure 3174]

[Figure 3175]

[Figure 3176]

[Figure 3177]

[Figure 3178]

[Figure 3179]

[Figure 3180]

[Figure 3181]

[Figure 3182]

[Figure 3183]

[Figure 3184]

[Figure 3185]

[Figure 3186]

[Figure 3187]

[Figure 3188]

[Figure 3189]

[Figure 3190]

[Figure 3191]

[Figure 3192]

[Figure 3193]

[Figure 3194]

[Figure 3195]

[Figure 3196]

[Figure 3197]

[Figure 3198]

[Figure 3199]

[Figure 3200]

[Figure 3201]

[Figure 3202]

[Figure 3203]

[Figure 3204]

[Figure 3205]

[Figure 3206]

[Figure 3207]

[Figure 3208]

[Figure 3209]

[Figure 3210]

[Figure 3211]

[Figure 3212]

[Figure 3213]

[Figure 3214]

[Figure 3215]

[Figure 3216]

[Figure 3217]

[Figure 3218]

[Figure 3219]

[Figure 3220]

[Figure 3221]

[Figure 3222]

[Figure 3223]

[Figure 3224]

[Figure 3225]

[Figure 3226]

[Figure 3227]

[Figure 3228]

[Figure 3229]

[Figure 3230]

[Figure 3231]

[Figure 3232]

[Figure 3233]

[Figure 3234]

[Figure 3235]

[Figure 3236]

[Figure 3237]

[Figure 3238]

[Figure 3239]

[Figure 3240]

[Figure 3241]

[Figure 3242]

[Figure 3243]

[Figure 3244]

[Figure 3245]

[Figure 3246]

[Figure 3247]

[Figure 3248]

[Figure 3249]

[Figure 3250]

[Figure 3251]

[Figure 3252]

[Figure 3253]

[Figure 3254]

[Figure 3255]

[Figure 3256]

[Figure 3257]

[Figure 3258]

[Figure 3259]

[Figure 3260]

[Figure 3261]

[Figure 3262]

[Figure 3263]

[Figure 3264]

[Figure 3265]

[Figure 3266]

[Figure 3267]

[Figure 3268]

[Figure 3269]

[Figure 3270]

[Figure 3271]

[Figure 3272]

[Figure 3273]

[Figure 3274]

[Figure 3275]

[Figure 3276]

[Figure 3277]

[Figure 3278]

[Figure 3279]

[Figure 3280]

[Figure 3281]

[Figure 3282]

[Figure 3283]

[Figure 3284]

[Figure 3285]

[Figure 3286]

[Figure 3287]

[Figure 3288]

[Figure 3289]

[Figure 3290]

[Figure 3291]

[Figure 3292]

[Figure 3293]

[Figure 3294]

[Figure 3295]

[Figure 3296]

[Figure 3297]

[Figure 3298]

[Figure 3299]

[Figure 3300]

[Figure 3301]

[Figure 3302]

[Figure 3303]

[Figure 3304]

[Figure 3305]

[Figure 3306]

[Figure 3307]

[Figure 3308]

[Figure 3309]

[Figure 3310]

[Figure 3311]

[Figure 3312]

[Figure 3313]

[Figure 3314]

[Figure 3315]

[Figure 3316]

[Figure 3317]

[Figure 3318]

[Figure 3319]

[Figure 3320]

[Figure 3321]

[Figure 3322]

[Figure 3323]

[Figure 3324]

[Figure 3325]

[Figure 3326]

[Figure 3327]

[Figure 3328]

[Figure 3329]

[Figure 3330]

[Figure 3331]

[Figure 3332]

[Figure 3333]

[Figure 3334]

[Figure 3335]

[Figure 3336]

[Figure 3337]

[Figure 3338]

[Figure 3339]

[Figure 3340]

[Figure 3341]

[Figure 3342]

[Figure 3343]

[Figure 3344]

[Figure 3345]

[Figure 3346]

[Figure 3347]

[Figure 3348]

[Figure 3349]

[Figure 3350]

[Figure 3351]

[Figure 3352]

[Figure 3353]

[Figure 3354]

[Figure 3355]

[Figure 3356]

[Figure 3357]

[Figure 3358]

[Figure 3359]

[Figure 3360]

[Figure 3361]

[Figure 3362]

[Figure 3363]

[Figure 3364]

[Figure 3365]

[Figure 3366]

[Figure 3367]

[Figure 3368]

[Figure 3369]

[Figure 3370]

[Figure 3371]

[Figure 3372]

[Figure 3373]

[Figure 3374]

[Figure 3375]

[Figure 3376]

[Figure 3377]

[Figure 3378]

[Figure 3379]

[Figure 3380]

[Figure 3381]

[Figure 3382]

[Figure 3383]

[Figure 3384]

[Figure 3385]

[Figure 3386]

[Figure 3387]

[Figure 3388]

[Figure 3389]

[Figure 3390]

[Figure 3391]

[Figure 3392]

[Figure 3393]

[Figure 3394]

[Figure 3395]

[Figure 3396]

[Figure 3397]

[Figure 3398]

[Figure 3399]

[Figure 3400]

[Figure 3401]

[Figure 3402]

[Figure 3403]

[Figure 3404]

[Figure 3405]

[Figure 3406]

[Figure 3407]

[Figure 3408]

[Figure 3409]

[Figure 3410]

[Figure 3411]

[Figure 3412]

[Figure 3413]

[Figure 3414]

[Figure 3415]

[Figure 3416]

[Figure 3417]

[Figure 3418]

[Figure 3419]

[Figure 3420]

[Figure 3421]

[Figure 3422]

[Figure 3423]

[Figure 3424]

[Figure 3425]

[Figure 3426]

[Figure 3427]

[Figure 3428]

[Figure 3429]

[Figure 3430]

[Figure 3431]

[Figure 3432]

[Figure 3433]

[Figure 3434]

[Figure 3435]

[Figure 3436]

[Figure 3437]

[Figure 3438]

[Figure 3439]

[Figure 3440]

[Figure 3441]

[Figure 3442]

[Figure 3443]

[Figure 3444]

[Figure 3445]

[Figure 3446]

[Figure 3447]

[Figure 3448]

[Figure 3449]

[Figure 3450]

[Figure 3451]

[Figure 3452]

[Figure 3453]

[Figure 3454]

[Figure 3455]

[Figure 3456]

[Figure 3457]

[Figure 3458]

[Figure 3459]

[Figure 3460]

[Figure 3461]

[Figure 3462]

[Figure 3463]

[Figure 3464]

[Figure 3465]

[Figure 3466]

[Figure 3467]

[Figure 3468]

[Figure 3469]

[Figure 3470]

[Figure 3471]

[Figure 3472]

[Figure 3473]

[Figure 3474]

[Figure 3475]

[Figure 3476]

[Figure 3477]

[Figure 3478]

[Figure 3479]

[Figure 3480]

[Figure 3481]

[Figure 3482]

[Figure 3483]

[Figure 3484]

[Figure 3485]

[Figure 3486]

[Figure 3487]

[Figure 3488]

[Figure 3489]

[Figure 3490]

[Figure 3491]

[Figure 3492]

[Figure 3493]

[Figure 3494]

[Figure 3495]

[Figure 3496]

[Figure 3497]

[Figure 3498]

[Figure 3499]

[Figure 3500]

[Figure 3501]

[Figure 3502]

[Figure 3503]

[Figure 3504]

[Figure 3505]

[Figure 3506]

[Figure 3507]

[Figure 3508]

[Figure 3509]

[Figure 3510]

[Figure 3511]

[Figure 3512]

[Figure 3513]

[Figure 3514]

[Figure 3515]

[Figure 3516]

[Figure 3517]

[Figure 3518]

[Figure 3519]

[Figure 3520]

[Figure 3521]

[Figure 3522]

[Figure 3523]

[Figure 3524]

[Figure 3525]

[Figure 3526]

[Figure 3527]

[Figure 3528]

[Figure 3529]

[Figure 3530]

[Figure 3531]

[Figure 3532]

[Figure 3533]

[Figure 3534]

[Figure 3535]

[Figure 3536]

[Figure 3537]

[Figure 3538]

[Figure 3539]

[Figure 3540]

[Figure 3541]

[Figure 3542]

[Figure 3543]

[Figure 3544]

[Figure 3545]

[Figure 3546]

[Figure 3547]

[Figure 3548]

[Figure 3549]

[Figure 3550]

[Figure 3551]

[Figure 3552]

[Figure 3553]

[Figure 3554]

[Figure 3555]

[Figure 3556]

[Figure 3557]

[Figure 3558]

[Figure 3559]

[Figure 3560]

[Figure 3561]

[Figure 3562]

[Figure 3563]

[Figure 3564]

[Figure 3565]

[Figure 3566]

[Figure 3567]

[Figure 3568]

[Figure 3569]

[Figure 3570]

[Figure 3571]

[Figure 3572]

[Figure 3573]

[Figure 3574]

[Figure 3575]

[Figure 3576]

[Figure 3577]

[Figure 3578]

[Figure 3579]

[Figure 3580]

[Figure 3581]

[Figure 3582]

[Figure 3583]

[Figure 3584]

[Figure 3585]

[Figure 3586]

[Figure 3587]

[Figure 3588]

[Figure 3589]

[Figure 3590]

[Figure 3591]

[Figure 3592]

[Figure 3593]

[Figure 3594]

[Figure 3595]

[Figure 3596]

[Figure 3597]

[Figure 3598]

[Figure 3599]

[Figure 3600]

[Figure 3601]

[Figure 3602]

[Figure 3603]

[Figure 3604]

[Figure 3605]

[Figure 3606]

[Figure 3607]

[Figure 3608]

[Figure 3609]

[Figure 3610]

[Figure 3611]

[Figure 3612]

[Figure 3613]

[Figure 3614]

[Figure 3615]

[Figure 3616]

[Figure 3617]

[Figure 3618]

[Figure 3619]

[Figure 3620]

[Figure 3621]

[Figure 3622]

[Figure 3623]

[Figure 3624]

[Figure 3625]

[Figure 3626]

[Figure 3627]

[Figure 3628]

[Figure 3629]

[Figure 3630]

[Figure 3631]

[Figure 3632]

[Figure 3633]

[Figure 3634]

[Figure 3635]

[Figure 3636]

[Figure 3637]

[Figure 3638]

[Figure 3639]

[Figure 3640]

[Figure 3641]

[Figure 3642]

[Figure 3643]

[Figure 3644]

[Figure 3645]

[Figure 3646]

[Figure 3647]

[Figure 3648]

[Figure 3649]

[Figure 3650]

[Figure 3651]

[Figure 3652]

[Figure 3653]

[Figure 3654]

[Figure 3655]

[Figure 3656]

[Figure 3657]

[Figure 3658]

[Figure 3659]

[Figure 3660]

[Figure 3661]

[Figure 3662]

[Figure 3663]

[Figure 3664]

[Figure 3665]

[Figure 3666]

[Figure 3667]

[Figure 3668]

[Figure 3669]

[Figure 3670]

[Figure 3671]

[Figure 3672]

[Figure 3673]

[Figure 3674]

[Figure 3675]

[Figure 3676]

[Figure 3677]

[Figure 3678]

[Figure 3679]

[Figure 3680]

[Figure 3681]

[Figure 3682]

[Figure 3683]

[Figure 3684]

[Figure 3685]

[Figure 3686]

[Figure 3687]

[Figure 3688]

[Figure 3689]

[Figure 3690]

[Figure 3691]

[Figure 3692]

[Figure 3693]

[Figure 3694]

[Figure 3695]

[Figure 3696]

[Figure 3697]

[Figure 3698]

[Figure 3699]

[Figure 3700]

[Figure 3701]

[Figure 3702]

[Figure 3703]

[Figure 3704]

[Figure 3705]

[Figure 3706]

[Figure 3707]

[Figure 3708]

[Figure 3709]

[Figure 3710]

[Figure 3711]

[Figure 3712]

[Figure 3713]

[Figure 3714]

[Figure 3715]

[Figure 3716]

[Figure 3717]

[Figure 3718]

[Figure 3719]

[Figure 3720]

[Figure 3721]

[Figure 3722]

[Figure 3723]

[Figure 3724]

[Figure 3725]

[Figure 3726]

[Figure 3727]

[Figure 3728]

[Figure 3729]

[Figure 3730]

[Figure 3731]

[Figure 3732]

[Figure 3733]

[Figure 3734]

[Figure 3735]

[Figure 3736]

[Figure 3737]

[Figure 3738]

[Figure 3739]

[Figure 3740]

[Figure 3741]

[Figure 3742]

[Figure 3743]

[Figure 3744]

[Figure 3745]

[Figure 3746]

[Figure 3747]

[Figure 3748]

[Figure 3749]

[Figure 3750]

[Figure 3751]

[Figure 3752]

[Figure 3753]

[Figure 3754]

[Figure 3755]

[Figure 3756]

[Figure 3757]

[Figure 3758]

[Figure 3759]

[Figure 3760]

[Figure 3761]

[Figure 3762]

[Figure 3763]

[Figure 3764]

[Figure 3765]

[Figure 3766]

[Figure 3767]

[Figure 3768]

[Figure 3769]

[Figure 3770]

[Figure 3771]

[Figure 3772]

[Figure 3773]

[Figure 3774]

[Figure 3775]

[Figure 3776]

[Figure 3777]

[Figure 3778]

[Figure 3779]

[Figure 3780]

[Figure 3781]

[Figure 3782]

[Figure 3783]

[Figure 3784]

[Figure 3785]

[Figure 3786]

[Figure 3787]

[Figure 3788]

[Figure 3789]

[Figure 3790]

[Figure 3791]

[Figure 3792]

[Figure 3793]

[Figure 3794]

[Figure 3795]

[Figure 3796]

[Figure 3797]

[Figure 3798]

[Figure 3799]

[Figure 3800]

[Figure 3801]

[Figure 3802]

[Figure 3803]

[Figure 3804]

[Figure 3805]

[Figure 3806]

[Figure 3807]

[Figure 3808]

[Figure 3809]

[Figure 3810]

[Figure 3811]

[Figure 3812]

[Figure 3813]

[Figure 3814]

[Figure 3815]

[Figure 3816]

[Figure 3817]

[Figure 3818]

[Figure 3819]

[Figure 3820]

[Figure 3821]

[Figure 3822]

[Figure 3823]

[Figure 3824]

[Figure 3825]

[Figure 3826]

[Figure 3827]

[Figure 3828]

[Figure 3829]

[Figure 3830]

[Figure 3831]

[Figure 3832]

[Figure 3833]

[Figure 3834]

[Figure 3835]

[Figure 3836]

[Figure 3837]

[Figure 3838]

[Figure 3839]

[Figure 3840]

[Figure 3841]

[Figure 3842]

[Figure 3843]

[Figure 3844]

[Figure 3845]

[Figure 3846]

[Figure 3847]

[Figure 3848]

[Figure 3849]

[Figure 3850]

[Figure 3851]

[Figure 3852]

[Figure 3853]

[Figure 3854]

[Figure 3855]

[Figure 3856]

[Figure 3857]

[Figure 3858]

[Figure 3859]

[Figure 3860]

[Figure 3861]

[Figure 3862]

[Figure 3863]

[Figure 3864]

[Figure 3865]

[Figure 3866]

[Figure 3867]

[Figure 3868]

[Figure 3869]

[Figure 3870]

[Figure 3871]

[Figure 3872]

[Figure 3873]

[Figure 3874]

[Figure 3875]

[Figure 3876]

[Figure 3877]

[Figure 3878]

[Figure 3879]

[Figure 3880]

[Figure 3881]

[Figure 3882]

[Figure 3883]

[Figure 3884]

[Figure 3885]

[Figure 3886]

[Figure 3887]

[Figure 3888]

[Figure 3889]

[Figure 3890]

[Figure 3891]

[Figure 3892]

[Figure 3893]

[Figure 3894]

[Figure 3895]

[Figure 3896]

[Figure 3897]

[Figure 3898]

[Figure 3899]

[Figure 3900]

[Figure 3901]

[Figure 3902]

[Figure 3903]

[Figure 3904]

[Figure 3905]

[Figure 3906]

[Figure 3907]

[Figure 3908]

[Figure 3909]

[Figure 3910]

[Figure 3911]

[Figure 3912]

[Figure 3913]

[Figure 3914]

[Figure 3915]

[Figure 3916]

[Figure 3917]

[Figure 3918]

[Figure 3919]

[Figure 3920]

[Figure 3921]

[Figure 3922]

[Figure 3923]

[Figure 3924]

[Figure 3925]

[Figure 3926]

[Figure 3927]

[Figure 3928]

[Figure 3929]

[Figure 3930]

[Figure 3931]

[Figure 3932]

[Figure 3933]

[Figure 3934]

[Figure 3935]

[Figure 3936]

[Figure 3937]

[Figure 3938]

[Figure 3939]

[Figure 3940]

[Figure 3941]

[Figure 3942]

[Figure 3943]

[Figure 3944]

[Figure 3945]

[Figure 3946]

[Figure 3947]

[Figure 3948]

[Figure 3949]

[Figure 3950]

[Figure 3951]

[Figure 3952]

[Figure 3953]

[Figure 3954]

[Figure 3955]

[Figure 3956]

[Figure 3957]

[Figure 3958]

[Figure 3959]

[Figure 3960]

[Figure 3961]

[Figure 3962]

[Figure 3963]

[Figure 3964]

[Figure 3965]

[Figure 3966]

[Figure 3967]

[Figure 3968]

[Figure 3969]

[Figure 3970]

[Figure 3971]

[Figure 3972]

[Figure 3973]

[Figure 3974]

[Figure 3975]

[Figure 3976]

[Figure 3977]

[Figure 3978]

[Figure 3979]

[Figure 3980]

[Figure 3981]

[Figure 3982]

[Figure 3983]

[Figure 3984]

[Figure 3985]

[Figure 3986]

[Figure 3987]

[Figure 3988]

[Figure 3989]

[Figure 3990]

[Figure 3991]

[Figure 3992]

[Figure 3993]

[Figure 3994]

[Figure 3995]

[Figure 3996]

[Figure 3997]

[Figure 3998]

[Figure 3999]

[Figure 4000]

[Figure 4001]

[Figure 4002]

[Figure 4003]

[Figure 4004]

[Figure 4005]

[Figure 4006]

[Figure 4007]

[Figure 4008]

[Figure 4009]

[Figure 4010]

[Figure 4011]

[Figure 4012]

[Figure 4013]

[Figure 4014]

[Figure 4015]

[Figure 4016]

[Figure 4017]

[Figure 4018]

[Figure 4019]

[Figure 4020]

[Figure 4021]

[Figure 4022]

[Figure 4023]

[Figure 4024]

[Figure 4025]

[Figure 4026]

[Figure 4027]

[Figure 4028]

[Figure 4029]

[Figure 4030]

[Figure 4031]

[Figure 4032]

[Figure 4033]

[Figure 4034]

[Figure 4035]

[Figure 4036]

[Figure 4037]

[Figure 4038]

[Figure 4039]

[Figure 4040]

[Figure 4041]

[Figure 4042]

[Figure 4043]

[Figure 4044]

[Figure 4045]

[Figure 4046]

[Figure 4047]

[Figure 4048]

[Figure 4049]

[Figure 4050]

[Figure 4051]

[Figure 4052]

[Figure 4053]

[Figure 4054]

[Figure 4055]

[Figure 4056]

[Figure 4057]

[Figure 4058]

[Figure 4059]

[Figure 4060]

[Figure 4061]

[Figure 4062]

[Figure 4063]

[Figure 4064]

[Figure 4065]

[Figure 4066]

[Figure 4067]

[Figure 4068]

[Figure 4069]

[Figure 4070]

[Figure 4071]

[Figure 4072]

[Figure 4073]

[Figure 4074]

[Figure 4075]

[Figure 4076]

[Figure 4077]

[Figure 4078]

[Figure 4079]

[Figure 4080]

[Figure 4081]

[Figure 4082]

[Figure 4083]

[Figure 4084]

[Figure 4085]

[Figure 4086]

[Figure 4087]

[Figure 4088]

[Figure 4089]

[Figure 4090]

[Figure 4091]

[Figure 4092]

[Figure 4093]

[Figure 4094]

[Figure 4095]

[Figure 4096]

[Figure 4097]

[Figure 4098]

[Figure 4099]

[Figure 4100]

[Figure 4101]

[Figure 4102]

[Figure 4103]

[Figure 4104]

[Figure 4105]

[Figure 4106]

[Figure 4107]

[Figure 4108]

[Figure 4109]

[Figure 4110]

[Figure 4111]

[Figure 4112]

[Figure 4113]

[Figure 4114]

[Figure 4115]

[Figure 4116]

[Figure 4117]

[Figure 4118]

[Figure 4119]

[Figure 4120]

[Figure 4121]

[Figure 4122]

[Figure 4123]

[Figure 4124]

[Figure 4125]

[Figure 4126]

[Figure 4127]

[Figure 4128]

[Figure 4129]

[Figure 4130]

[Figure 4131]

[Figure 4132]

[Figure 4133]

[Figure 4134]

[Figure 4135]

[Figure 4136]

[Figure 4137]

[Figure 4138]

[Figure 4139]

[Figure 4140]

[Figure 4141]

[Figure 4142]

[Figure 4143]

[Figure 4144]

[Figure 4145]

[Figure 4146]

[Figure 4147]

[Figure 4148]

[Figure 4149]

[Figure 4150]

[Figure 4151]

[Figure 4152]

[Figure 4153]

Large Video Generation Backbone (Pre-trained Wan 2.2)

Feature Encode

[Figure 4154]

||[Figure 4155]<br><br>[Figure 4156]<br><br>[Figure 4157]|
|---|
<br><br>Video<br><br>Action<br><br>[Figure 4158]<br><br>Noise<br><br>Text<br><br>Camera|
|---|

View 1 View 2

L2 Loss on each batch

[Figure 4159]

[Figure 4160]

[Figure 4161]

[Figure 4162]

2 Predicted

Ground Truth

- Fig. 4: Unified world-action model training. Multi-view video and action latents are packed with text and camera conditions, and trained under diverse mask strategies.

so that the model observes, for each view, a unified timeline of (robot video → action video). Multi-view data are processed with shared weights across views, enabling consistent cross-view learning while preserving per-view conditioning.

Unified training via multiple mask strategies. To support multiple tasks with a single model, we adopt a multiple mask strategy in the latent token space (Figure 4). Concretely, we randomly sample masks over the concatenated latent sequence Xk to instantiate different training objectives within the same diffusion-style denoising framework: 1) Action & video joint generation. We mask both V and A tokens except for the first observation frame, and ask the model to generate them jointly conditioned on text and camera inputs. 2) Actionconditioned video generation. We keep A visible while masking V, training the model to synthesize future visual observations consistent with provided actions.

- 3) Video-to-action labeling. We keep V visible while masking A, training the model to infer action images from the input video. 4) Video-only generation. For data without usable action, we train the model with video tokens only, using the same denoising objective to model future observations. This masking scheme turns the same backbone into a unified world model that can switch behaviors by changing which token subsets are observed vs. predicted, improving generalization across settings and downstream usages.

Beyond masking-based supervision, our unified model also supports cameracontrolled generation, which also helps maintain multi-view consistency. Following ReCamMaster [4], we inject camera plucker embedding [45] into the backbone as Fi = Fo + Ec(camt), where Ec is a lightweight convolutional encoder, Fo is the output of the spatial-attention layer, and Fi is the input to the subsequent 3D-attention layer.

Optimization objective. We fine-tune the pretrained backbone using a flow matching [39] objective on the masked latent tokens. The target velocity is defined as v = ϵ − X. We then minimize an L2 loss between the predicted and

Table 1: Summary of dataset for unified world action model training.

###### Dataset #Traj. #Views Real Action Ann. Cam. Calib. Cam. Motion

DROID 80k 2 ✓ ✓ ✓ Static RLBench 180k 4 ✗ ✓ ✓ Diverse BridgeV2 30k 1-4 ✓ ✓ ✗ Static

target velocities over the masked tokens:

L = E M ⊙ v − vθ(X,T ,cam) 22 , (11)

where M is the mask, T is the text input, and cam is the camera condition. This yields a single unified model that learns the coupled dynamics of visual observations and actions across multiple views.

Training datasets. Training a unified world action model requires largescale data, but this is challenging in robotics: multi-view datasets are limited, and datasets with well-aligned action and camera annotations are even rarer. We therefore train on a mixture of RLBench [23], DROID [26], and BridgeV2 [55], which provide complementary supervision as shown in Tab. 1. DROID offers the most complete real-robot annotations, but its camera calibration is often noisy or incomplete in practice, so we filter out low-quality samples. RLBench, although more toy-like than real-world data, provides highly accurate action and camera signals from simulation; we improve its visual diversity with RobotColosseum [47] background augmentation. BridgeV2 contains high-quality realworld videos, but lacks camera labels and action-camera alignment. We estimate camera annotations with VGGT [57] and use BridgeV2 for video-only generation.

### 4 Experiments

#### 4.1 Text-Controlled Action & Video Joint Generation

We treat text-controlled action and video joint generation as the primary evaluation setting of this paper. Given a language instruction and the initial multi-view observations, the model jointly generates future robot videos and corresponding multi-view action videos, from which executable controls are obtained by decoding the predicted action images. Unless otherwise specified, all experiments in this subsection are conducted in the multi-view setting under one-trial openloop evaluation. This is a particularly challenging setting, since the model must complete the task from a single forward prediction without online replanning, making the results directly reflect the quality and generalization ability of the learned pixel-grounded action representation.

###### Table 2: Zero-shot evaluation results on RLBench and Real-world settings.

RLBench Real

Methods

pick cup

reach target

close drawer

close laptop

Place Cup

Pick Unseen Toy

Pick Tissue

Close Drawer

Close Box

MV-Policy 0 0 0 0 0 0 0 0 0 π0.5 0 5 35 20 5 0 0 0 0 MolmoAct 20 5 10 0 10 5 5 5 0 TesserAct 0 0 0 0 0 0 0 0 0 Cosmos-Policy 0 5 20 0 0 0 0 0 0 Ours 30 60 50 15 40 20 15 45 10

Zero-shot policy results. We compare against several representative robot policy baselines, including MV-Policy [10], π0.5 [22], MolmoAct [31], TesserAct [70], and Cosmos-Policy [27]. MV-Policy is a multi-view extension of Diffusion Policy that encodes images from multiple camera views. π0.5 and MolmoAct are VLA-style baselines. For π0.5, we use the base checkpoint and augment the model with an MLP that injects camera parameters into the VLM. MolmoAct is a reasoning-based model that can predict 2D trajectories on images; we leverage this capability by querying trajectories in multiple views and lifting them into 3D motion. TesserAct and Cosmos-Policy are world-model-based baselines. For fair comparison, we reproduce both by fine-tuning the same Wan 2.2 [56] video backbone on our training set.

For evaluation, we use task success rate as the metric in both simulation and real-robot settings. The zero-shot setting differs across environments. In RLBench [23], the evaluated tasks may appear in other datasets, but these specific tasks are fully removed from the RLBench training split; the robot arm and environment are seen. In the real-world setting, the objects, environments, and robot arm (xArm) are all unseen. Across all settings, the language instructions are similar in form to those seen during training. As shown in Tab. 2, our method delivers the best overall zero-shot performance across simulation and real-world tasks. The improvement is most evident under strong distribution shift, supporting our claim that interpretable action images and a pixel-grounded action representation lead to a more generalizable zero-shot policy.

RLBench in-domain results. We next evaluate the same model on indomain RLBench tasks, using the same baselines and the metrics as above. Besides the reconstruction-based decoder that recovers actions from generated action images, we also consider an optional learned action head on top of the unified backbone. Specifically, we attach a lightweight MLP that takes as input the output video latents, camera parameters, and decoded actions and observations, and train it to directly regress the continuous 7-DoF action sequence. This head is not required for our main zero-shot policy claim; rather, it is introduced to test whether the learned representation can support improved decoding.

As shown in Tab. 3, our method remains competitive on in-domain RLBench tasks even under the same challenging setting. Moreover, adding the optional ac-

###### Table 3: RLBench in-domain tasks evaluation

close box

close door

open door

phone base

open bottle

close drawer

open oven

open jar

wipe desk

Methods

Avg.

MV-Diffusion Policy 20 40 15 20 5 50 10 0 0 17.8 MomolAct (zeroshot) 5 10 0 0 5 10 0 0 0 3.3

π0.5 10 0 5 5 45 65 0 0 0 14.4 TesserAct 40 25 5 15 20 70 5 5 0 20.6 Cosmos-Policy 40 15 0 15 30 80 0 0 0 20.0

Ours 55 60 0 0 5 60 5 0 0 20.6 w/ action head 80 65 15 20 40 80 15 5 10 36.7

###### Table 4: Video-and-Action Joint Generation Quality. († denotes a zero-shot model.).

Video Action

Models

PSNR ↑ SSIM (%) ↑ FVD ↓ LPIPS ↓ 2DErr ↓ 3DErr ×103 ↓

Cosmos-Predict2.5-14B† 17.92 50.77 208.65 0.409 - Cosmos-Policy 18.29 53.41 192.58 0.418 2.11 19.4 TesserAct 20.83 59.20 154.38 0.351 1.84 19.0 TesserAct-RGB 20.31 60.19 147.83 0.372 1.55 14.2

Ours 23.48 78.62 143.74 0.209 1.61 12.2

tion head brings substantial gains, especially on precision-sensitive tasks, showing that the action images can support stronger action decoding when additional supervision is available.

Joint generation quality. Unlike the policy evaluations above, this experiment focuses on how accurately the model predicts both future robot videos and the corresponding actions. We compare against world models, including Cosmos-Predict [1], Cosmos-Policy [27] and TesserAct [70]. For video quality, we use PSNR and SSIM to measure pixel-level fidelity and structural similarity, with FVD and LPIPS to evaluate perceptual and temporal realism. For action quality, we report both 2D and 3D trajectory error. Since all compared models, except ours, directly predict 3D actions, we additionally project the outputs using camera parameters to obtain 2D errors. Video generation is evaluated on in-domain RLBench, Bridge, and DROID, while action metrics are evaluated on RLBench only. As shown in Tab. 4, our method outperforms prior world-model baselines on all video metrics while maintaining action accuracy.

#### 4.2 Additional Unified-Model Capabilities

Action-conditioned video generation. This task tests whether the model can generate future robot videos when the action sequence is given. We compare with Tora [68], a 2D trajectory-conditioned video generation baseline. We evaluate generation quality using standard video metrics, including PSNR, SSIM, FVD, and LPIPS. As shown in Tab. 5, our method achieves better results on all metrics,

###### Table 5: Action-cond. video quality.

###### Table 6: Video-to-action labeling results.

Models PSNR ↑ SSIM (%) ↑ LVD ↓ LPIPS (%) ↓

Tora 19.76 52.43 187.41 39.62 Ours 31.35 67.16 115.02 21.78

Models Traj Err ↓ Jaccard @ ↑ 4 Avg. Jaccard ↑

TAPIR 14.80 40.26 29.77 CoTracker 12.91 46.15 31.20 Ours 5.785 64.92 46.71

suggesting that the unified video-space representation of observation and action can use action inputs more effectively for future video prediction.

Video-to-action labeling. This task tests whether the model can infer action-related motion directly from input videos. We compare with two pointtracking baselines, TAPIR [11] and CoTracker3 [25]. We use standard tracking metrics, including trajectory error, Jaccard@4, and average Jaccard. As shown in Tab. 6, our method outperforms both baselines by a clear margin. This result shows that the pixel-grounded action representation is not only useful for control and generation, but also provides a simple way to label action from video.

#### 4.3 Qualitative Results

We first evaluate zero-shot rollouts on an xArm platform, where the objects and environment are unseen. As shown in Fig. 6, we visualize the input image with the predicted tracking trajectories on the left. Our model first generates future observations and multi-view action images (middle), and we then decode the predicted 2D action into a 3D trajectory for point-cloud visualization (right), where the scene geometry is reconstructed by VGGT [57]. The 3D trajectory is colored by time, from blue (earlier) to red (later). We replay the decoded trajectory on the real robot to validate executability. Separately, we include results from a strong video-generation baseline (Veo3.1 [15]) for qualitative comparison. The execution matches the generated motion, indicating that the predicted action images decode into plausible trajectories.

To further test generalization, we sample two images from the FR3M [50] room dataset and prompt the model to perform unseen task. Fig. 5 compares our generations with LTX-2-Fast [38]. Our model produces videos with more accurate localization of targets. Notably, despite lacking action supervision on BridgeV2 during training, the model still generates coherent action images, indicating that the learned action-generation capability transfers across datasets and domains.

### 5 Conclusion

We presented a world action model that formulates policy learning as video generation through a unified video-space representation of observation and action. Our key idea is to translate 7-DoF robot control into interpretable action images, yielding a pixel-grounded action in the form of multi-view videos. This design allows the video backbone itself to serve as a zero-shot policy model, without

|Input Image|
|---|

|3D Visualization|
|---|

|Video Generation Results|
|---|

[Figure 4163]

[Figure 4164]

[Figure 4165]

[Figure 4166]

[Figure 4167]

[Figure 4168]

|[Figure 4169]|
|---|

|[Figure 4170]<br><br>[Figure 4171]|[Figure 4172]<br><br>[Figure 4173]|[Figure 4174]<br><br>[Figure 4175]|[Figure 4176]<br><br>[Figure 4177]|[Figure 4178]<br><br>[Figure 4179]|
|---|---|---|---|---|

[Figure 4180]

[Figure 4181]

[Figure 4182]

[Figure 4183]

[Figure 4184]

[Figure 4185]

[Figure 4186]

[Figure 4187]

|[Figure 4188]|
|---|

|[Figure 4189]<br><br>[Figure 4190]|[Figure 4191]<br><br>[Figure 4192]|[Figure 4193]<br><br>[Figure 4194]|[Figure 4195]<br><br>[Figure 4196]|[Figure 4197]<br><br>[Figure 4198]|
|---|---|---|---|---|

[Figure 4199]

[Figure 4200]

[Figure 4201]

[Figure 4202]

[Figure 4203]

[Figure 4204]

[Figure 4205]

[Figure 4206]

[Figure 4207]

[Figure 4208]

Real Exec

[Figure 4209]

[Figure 4210]

[Figure 4211]

[Figure 4212]

[Figure 4213]

[Figure 4214]

[Figure 4215]

[Figure 4216]

Veo3.1 Results

[Figure 4217]

[Figure 4218]

[Figure 4219]

[Figure 4220]

[Figure 4221]

[Figure 4222]

[Figure 4223]

[Figure 4224]

Veo3.1 Results

|Prompt: Close the paper box (unseen object and environment)|
|---|

- Fig. 5: Real-world zero-shot rollouts on xArm robot. From left to right, we show the input observation, generated future video frames with predicted action-image trajectories, and the reconstructed 3D visualization. The results demonstrate that our model can generalize to unseen real-world objects and environments, while producing executable action predictions that are consistent with the generated visual outcomes.

requiring a separate policy head or action module. The same model supports video-action joint generation, action-conditioned video generation, and action labeling under a shared generative framework. We hope this work suggests that grounding action in pixels provides a promising path toward more generalizable policy learning and robotics world modeling in a common video space.

Limitations. Our current system demonstrates strong open-loop results, but has not yet been fully developed into a closed-loop policy. Fortunately, recent progress on diffusion acceleration and distillation provides a promising path to address this issue. In future work, we plan to distill our model for faster inference and integrate it into a closed-loop control pipeline.

|Input Image|
|---|

|3D Visualization|
|---|

|Video Generation Results|
|---|

|[Figure 4225]|
|---|

[Figure 4226]

[Figure 4227]

[Figure 4228]

[Figure 4229]

[Figure 4230]

[Figure 4231]

|[Figure 4232]<br><br>[Figure 4233]|[Figure 4234]<br><br>[Figure 4235]|[Figure 4236]<br><br>[Figure 4237]|[Figure 4238]<br><br>[Figure 4239]|[Figure 4240]<br><br>[Figure 4241]|
|---|---|---|---|---|

[Figure 4242]

[Figure 4243]

|[Figure 4244]|
|---|

[Figure 4245]

[Figure 4246]

[Figure 4247]

[Figure 4248]

[Figure 4249]

[Figure 4250]

|[Figure 4251]<br><br>[Figure 4252]|[Figure 4253]<br><br>[Figure 4254]|[Figure 4255]<br><br>[Figure 4256]|[Figure 4257]<br><br>[Figure 4258]|[Figure 4259]<br><br>[Figure 4260]|
|---|---|---|---|---|

[Figure 4261]

[Figure 4262]

[Figure 4263]

[Figure 4264]

[Figure 4265]

[Figure 4266]

[Figure 4267]

[Figure 4268]

[Figure 4269]

[Figure 4270]

LTX-2 Results

[Figure 4271]

[Figure 4272]

[Figure 4273]

[Figure 4274]

[Figure 4275]

[Figure 4276]

[Figure 4277]

[Figure 4278]

LTX-2 Results

|Prompt: Pick up the mouse (unseen object, task and environment)|
|---|

- Fig. 6: Zero-shot video and action-image generation on FR3M [50] rooms. This example illustrates a challenging setting with an unseen object, unseen task, and unseen environment (pick up the mouse). The predicted action trajectories stay aligned with the scene geometry.

### 6 Acknowledgement

We are extremely grateful to Zeyuan Yang, Jiaben Chen, Ziqiao Ma, Sriram Krishna, Hongxin Zhang, Zhou Xian, and Theophile Gervet for their helpful feedback and insightful discussions.

## Action Images: End-to-End Policy Learning via Multiview Video Generation

Supplementary Material

### 1 Implementation Details

Training Details. We trained our unified world-action model by fine-tuning a pretrained Wan2.1-I2V-14B-480P [56] backbone. The training data comprised Bridge [55], RLBench [23], and DROID [26], sampled with mixture ratios of 0.2, 0.5, and 0.3, respectively. Each training sample contained 41 frames for a single view and a single modality; under the full two-view setting with both robot videos and action videos, this corresponds to 164 frames in total. We used a task mixture in which 85% of samples were used for joint generation, while video-only, action-label, and action-conditioned generation each accounted for 5%. Training was conducted on 32 A100 GPUs using DeepSpeed ZeRO [48], bfloat16 mixed precision, and gradient checkpointing. We used a per-device batch size of 1. The optimizer used a constant-with-warmup schedule with a learning rate of 5×10−7, a warmup of 1000 steps, and gradient clipping with a maximum norm of 1.0. We trained the model for 100,000 optimization steps.

For camera conditioning, we followed the design of ReCamMaster [4], except that we used Plücker embeddings [45] as the camera representation. The camera encoder first pooled the spatiotemporal camera features to a fixed resolution, then flattened and projected them into the model hidden dimension by a linear projector. We initialized the encoder projection to zeros and the final projector as an identity mapping, which stabilizes optimization at the beginning of training.

Inference Details. At inference time, we keep the input formatting and spatial-temporal configuration same with training. We use classifier-free guidance [19] with a scale of 10.0, and perform sampling for 50 denoising steps. In all experiments, inference is executed with 4-GPU Unified Sequence Parallelism [14]. For in-the-wild images without camera annotations, we estimate camera extrinsics and intrinsics using VGGT [57]. As shown in Table 7, we further improve inference throughput by introducing several system-level optimizations, including CFG parallelism, VAE parallelism, caching, and torch.compile. With these optimizations, the video backbone reaches up to 71 FPS. We also note that although DreamZero-Flash achieves extremely fast inference, it relies on highly aggressive denoising steps, which leads to a severe degradation in video quality.

Action Images Details. Following Sec. 3.1, each robot action is converted into three semantic 3D points (position, normal, and up) and projected into image space. The normal and up points are placed at a fixed distance of 0.1 from the position point along their directions. The projected 2D points are then rasterized as Gaussian heatmaps with a standard deviation σ = 0.05 relative to the image resolution. In practice, we observe that moderate changes to these hyperparameters do not noticeably affect performance, as long as the projected points remain within the image plane.

###### Table 7: Inference efficiency.

Models Size GPU Steps #Frames Image Res. Inference Time (s) TesserAct 5B 1 H100 50 49 (480, 640) 137.5 DreamZero 14B 1 H100 16 48 (176, 320) 5.7 DreamZero-Flash 14B 2 GB200 1 48 (176, 320) 0.15 Ours 5B 1 H100 50 164 (512, 512) 49.1 + Parallelism 5B 8 H100 50 164 (512, 512) 11.8 + Caching 5B 8 H100 16 164 (512, 512) 2.3

### 2 More Zero-shot Qualitative Results

##### First, Fig. 7 shows action labeling results given input videos, including one π0 [6] robot video and one Genie 3 [9] human-hand video, demonstrating that our model can handle both.

|Input Video|
|---|

[Figure 4279]

[Figure 4280]

[Figure 4281]

[Figure 4282]

[Figure 4283]

[Figure 4284]

[Figure 4285]

[Figure 4286]

|Output Action Video Results|
|---|

[Figure 4287]

[Figure 4288]

[Figure 4289]

[Figure 4290]

[Figure 4291]

[Figure 4292]

[Figure 4293]

[Figure 4294]

[Figure 4295]

[Figure 4296]

[Figure 4297]

[Figure 4298]

[Figure 4299]

[Figure 4300]

[Figure 4301]

[Figure 4302]

|Input Video|
|---|

[Figure 4303]

[Figure 4304]

[Figure 4305]

[Figure 4306]

[Figure 4307]

[Figure 4308]

[Figure 4309]

[Figure 4310]

|Output Action Video Results|
|---|

[Figure 4311]

[Figure 4312]

[Figure 4313]

[Figure 4314]

[Figure 4315]

[Figure 4316]

[Figure 4317]

[Figure 4318]

[Figure 4319]

[Figure 4320]

[Figure 4321]

[Figure 4322]

[Figure 4323]

[Figure 4324]

[Figure 4325]

[Figure 4326]

###### Fig. 7: Action labeling results.

##### Fig. 8 provides more qualitative robot manipulation results, mainly on grasping tasks across diverse objects and scenes.

|Input Image|
|---|

|3D Visualization|
|---|

|Video Generation Results|
|---|

[Figure 4327]

[Figure 4328]

[Figure 4329]

[Figure 4330]

[Figure 4331]

[Figure 4332]

|[Figure 4333]|
|---|

|[Figure 4334]<br><br>[Figure 4335]|[Figure 4336]<br><br>[Figure 4337]|[Figure 4338]<br><br>[Figure 4339]|[Figure 4340]<br><br>[Figure 4341]|[Figure 4342]<br><br>[Figure 4343]|
|---|---|---|---|---|

[Figure 4344]

[Figure 4345]

[Figure 4346]

[Figure 4347]

[Figure 4348]

[Figure 4349]

[Figure 4350]

[Figure 4351]

|[Figure 4352]|
|---|

[Figure 4353]

|[Figure 4354]<br><br>[Figure 4355]|[Figure 4356]<br><br>[Figure 4357]<br><br>[Figure 4358]|[Figure 4359]|[Figure 4360]<br><br>[Figure 4361]|[Figure 4362]<br><br>[Figure 4363]|
|---|---|---|---|---|

[Figure 4364]

[Figure 4365]

[Figure 4366]

[Figure 4367]

[Figure 4368]

[Figure 4369]

[Figure 4370]

[Figure 4371]

[Figure 4372]

Real Exec

|Prompt: Place the black bowl in the paper box|
|---|

[Figure 4373]

[Figure 4374]

[Figure 4375]

[Figure 4376]

[Figure 4377]

[Figure 4378]

[Figure 4379]

|[Figure 4380]<br><br>[Figure 4381]|[Figure 4382]|[Figure 4383]<br><br>[Figure 4384]<br><br>[Figure 4385]|[Figure 4386]|[Figure 4387]<br><br>[Figure 4388]<br><br>[Figure 4389]|
|---|---|---|---|---|

[Figure 4390]

[Figure 4391]

[Figure 4392]

[Figure 4393]

[Figure 4394]

[Figure 4395]

[Figure 4396]

[Figure 4397]

[Figure 4398]

|[Figure 4399]<br><br>[Figure 4400]|[Figure 4401]<br><br>[Figure 4402]|[Figure 4403]<br><br>[Figure 4404]|[Figure 4405]<br><br>[Figure 4406]|[Figure 4407]<br><br>[Figure 4408]|
|---|---|---|---|---|

[Figure 4409]

[Figure 4410]

[Figure 4411]

[Figure 4412]

[Figure 4413]

[Figure 4414]

[Figure 4415]

[Figure 4416]

[Figure 4417]

[Figure 4418]

Real Exec

|Prompt: Close the middle drawer|
|---|

###### Fig. 8: Additional robot manipulation results, mainly on grasping tasks.

##### We then show camera control results in Fig. 9, where the model is given an input image and a task, and generates videos with controlled viewpoint changes in complex scenes from the Pi0 website.

|Input Image|
|---|

|Video Generation Results|
|---|

[Figure 4419]

[Figure 4420]

[Figure 4421]

[Figure 4422]

[Figure 4423]

[Figure 4424]

[Figure 4425]

[Figure 4426]

Fig. 9: Camera control results in complex scenes from the π0 website.

##### Finally, we show action-conditioned generation results in Fig. 10, where we use the first frame from π0 demo videos as input to generate future videos conditioned on actions.

|Input Image|
|---|

|Video Generation Results|
|---|

[Figure 4427]

[Figure 4428]

[Figure 4429]

[Figure 4430]

[Figure 4431]

[Figure 4432]

[Figure 4433]

[Figure 4434]

|Input action condition|
|---|

[Figure 4435]

[Figure 4436]

[Figure 4437]

[Figure 4438]

[Figure 4439]

[Figure 4440]

[Figure 4441]

[Figure 4442]

[Figure 4443]

[Figure 4444]

[Figure 4445]

[Figure 4446]

[Figure 4447]

[Figure 4448]

[Figure 4449]

[Figure 4450]

|Prompt: Place the yellow cup on the green plate|
|---|

###### Fig. 10: Action-conditioned generation results.

## Bibliography

- [1] Ali, A., Bai, J., Bala, M., Balaji, Y., Blakeman, A., Cai, T., Cao, J., Cao, T., Cha, E., Chao, Y.W., et al.: World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062 (2025)
- [2] Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., et al.: V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985 (2025)
- [3] Bahmani, S., Skorokhodov, I., Rong, V., Wetzstein, G., Guibas, L., Wonka, P., Tulyakov, S., Park, J.J., Tagliasacchi, A., Lindell, D.B.: 4d-fy: Text-to4d generation using hybrid score distillation sampling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7996–8006 (2024)
- [4] Bai, J., Xia, M., Fu, X., Wang, X., Mu, L., Cao, J., Liu, Z., Hu, H., Bai, X., Wan, P., et al.: Recammaster: Camera-controlled generative rendering from a single video. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14834–14844 (2025)
- [5] Bar, A., Zhou, G., Tran, D., Darrell, T., LeCun, Y.: Navigation world models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15791–15801 (2025)
- [6] Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al.: pi_0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164 (2024)
- [7] Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al.: Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817

(2022)

- [8] Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world simulators (2024), https://openai.com/ research/video-generation-models-as-world-simulators
- [9] Bruce, J., Dennis, M.D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al.: Genie: Generative interactive environments. In: Forty-first International Conference on Machine Learning (2024)
- [10] Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., Song, S.: Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research 44(10-11), 1684–1704

(2025)

- [11] Doersch, C., Yang, Y., Vecerik, M., Gokay, D., Gupta, A., Aytar, Y., Carreira, J., Zisserman, A.: Tapir: Tracking any point with per-frame initialization and temporal refinement. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10061–10072 (2023)

- [12] Du, Y., Yang, S., Dai, B., Dai, H., Nachum, O., Tenenbaum, J., Schuurmans, D., Abbeel, P.: Learning universal policies via text-guided video generation. Advances in neural information processing systems 36, 9156–9172 (2023)
- [13] Etukuru, H., Naka, N., Hu, Z., Lee, S., Mehu, J., Edsinger, A., Paxton, C., Chintala, S., Pinto, L., Shafiullah, N.M.M.: Robot utility models: General policies for zero-shot deployment in new environments. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 8275–8283. IEEE (2025)
- [14] Fang, J., Zhao, S.: Usp: A unified sequence parallelism approach for long context generative ai. arXiv preprint arXiv:2405.07719 (2024)
- [15] Google DeepMind: Veo: a text-to-video generation system. PDF (2025), https://storage.googleapis.com/deepmind-media/veo/Veo-3-TechReport.pdf, accessed: 2026-03-05
- [16] Guo, J., Ma, X., Wang, Y., Yang, M., Liu, H., Li, Q.: Flowdreamer: A rgb-d world model with flow-based motion representations for robot manipulation. arXiv preprint arXiv:2505.10075 (2025)
- [17] Guo, Y., Shi, L.X., Chen, J., Finn, C.: Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125

(2025)

- [18] Gupta, A., Murali, A., Gandhi, D.P., Pinto, L.: Robot learning in homes: Improving generalization and reducing dataset bias. Advances in neural information processing systems 31 (2018)
- [19] Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)
- [20] Hu, Y., Guo, Y., Wang, P., Chen, X., Wang, Y.J., Zhang, J., Sreenath, K., Lu, C., Chen, J.: Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803 (2024)
- [21] Intelligence, P., Amin, A., Aniceto, R., Balakrishna, A., Black, K., Conley, K., Connors, G., Darpinian, J., Dhabalia, K., DiCarlo, J., et al.: π∗0.6: a vla that learns from experience. arXiv preprint arXiv:2511.14759 (2025)
- [22] Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al.: π0.5: a visionlanguage-action model with open-world generalization. arXiv preprint arXiv:2504.16054 (2025)
- [23] James, S., Ma, Z., Arrojo, D.R., Davison, A.J.: Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters 5(2), 3019–3026 (2020)
- [24] Jin, Y., Peng, S., Wang, X., Xie, T., Xu, Z., Yang, Y., Shen, Y., Bao, H., Zhou, X.: Diffuman4d: 4d consistent human view synthesis from sparseview videos with spatio-temporal diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 11047–11057

(2025)

- [25] Karaev, N., Makarov, Y., Wang, J., Neverova, N., Vedaldi, A., Rupprecht, C.: Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6013–6022 (2025)

- [26] Khazatsky, A., Pertsch, K., Nair, S., Balakrishna, A., Dasari, S., Karamcheti, S., Nasiriany, S., Srirama, M.K., Chen, L.Y., Ellis, K., et al.: Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945 (2024)
- [27] Kim, M.J., Gao, Y., Lin, T.Y., Lin, Y.C., Ge, Y., Lam, G., Liang, P., Song, S., Liu, M.Y., Finn, C., et al.: Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163 (2026)
- [28] Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al.: Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246

(2024)

- [29] Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)
- [30] Ko, P.C., Mao, J., Du, Y., Sun, S.H., Tenenbaum, J.B.: Learning to act from actionless videos through dense correspondences. arXiv preprint arXiv:2310.08576 (2023)
- [31] Lee, J., Duan, J., Fang, H., Deng, Y., Liu, S., Li, B., Fang, B., Zhang, J., Wang, Y.R., Lee, S., et al.: Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917 (2025)
- [32] Li, C., Krause, A., Hutter, M.: Robotic world model: A neural network simulator for robust policy optimization in robotics. arXiv preprint arXiv:2501.10100 (2025)
- [33] Li, P., Chen, Y., Xu, Y., Yang, J., Wu, X., Guo, J., Sun, N., Qian, L., Li, X., Xiao, X., Liu, J., Liu, N., Kong, T., Huang, Y., Wang, L., Tan, T.: Multi-view video diffusion policy: A 3d spatio-temporal-aware video action model (2026), https://arxiv.org/abs/2604.03181
- [34] Li, S., Gao, Y., Sadigh, D., Song, S.: Unified video action model. arXiv preprint arXiv:2503.00200 (2025)
- [35] Li, Y., Luo, Z., Zhang, T., Dai, C., Kanervisto, A., Tirinzoni, A., Weng, H., Kitani, K., Guzek, M., Touati, A., et al.: Bfm-zero: A promptable behavioral foundation model for humanoid control using unsupervised reinforcement learning. arXiv preprint arXiv:2511.04131 (2025)
- [36] Li, Z., Zhang, M., Wu, T., Tan, J., Wang, J., Lin, D.: Ss4d: Native 4d generative model via structured spacetime latents. ACM Transactions on Graphics (TOG) 44(6), 1–12 (2025)
- [37] Liang, J., Tokmakov, P., Liu, R., Sudhakar, S., Shah, P., Ambrus, R., Vondrick, C.: Video generators are robot policies. arXiv preprint arXiv:2508.00795 (2025)
- [38] Lightricks: Ltx studio. Online (2024), https://app.ltx.studio/, accessed: 2026-02
- [39] Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)
- [40] Liu, Z., Li, S., Cousineau, E., Feng, S., Burchfiel, B., Song, S.: Geometry-aware 4d video generation for robot manipulation. arXiv preprint arXiv:2507.01099 (2025)
- [41] Ljung, L., Glad, T.: Modeling of dynamic systems. Prentice-Hall, Inc. (1994)

- [42] Nakamoto, M., Mees, O., Kumar, A., Levine, S.: Steering your generalists: Improving robotic foundation models via value guidance. arXiv preprint arXiv:2410.13816 (2024)
- [43] Pan, Z., Yang, Z., Zhu, X., Zhang, L.: Efficient4d: Fast dynamic 3d object generation from a single-view video. arXiv preprint arXiv:2401.08742 (2024)
- [44] Pearce, T., Rashid, T., Kanervisto, A., Bignell, D., Sun, M., Georgescu, R., Macua, S.V., Tan, S.Z., Momennejad, I., Hofmann, K., et al.: Imitating human behaviour with diffusion models. arXiv preprint arXiv:2301.10677

(2023)

- [45] Plucker, J.: Xvii. on a new geometry of space. Philosophical Transactions of the Royal Society of London (155), 725–791 (1865)
- [46] Polydoros, A.S., Nalpantidis, L.: Survey of model-based reinforcement learning: Applications on robotics. Journal of Intelligent & Robotic Systems 86(2), 153–173 (2017)
- [47] Pumacay, W., Singh, I., Duan, J., Krishna, R., Thomason, J., Fox, D.: The colosseum: A benchmark for evaluating generalization for robotic manipulation. arXiv preprint arXiv:2402.08191 (2024)
- [48] Rajbhandari, S., Rasley, J., Ruwase, O., He, Y.: Zero: Memory optimizations toward training trillion parameter models. In: SC20: international conference for high performance computing, networking, storage and analysis. pp. 1–16. IEEE (2020)
- [49] Ren, J., Pan, L., Tang, J., Zhang, C., Cao, A., Zeng, G., Liu, Z.: Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142

(2023)

- [50] Shen, W., Yang, G., Yu, A., Wong, J., Kaelbling, L.P., Isola, P.: Distilled feature fields enable few-shot language-guided manipulation. arXiv preprint arXiv:2308.07931 (2023)
- [51] Singer, U., Sheynin, S., Polyak, A., Ashual, O., Makarov, I., Kokkinos, F., Goyal, N., Vedaldi, A., Parikh, D., Johnson, J., et al.: Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280 (2023)
- [52] Sun, Q., Yang, L., Tang, W., Huang, W., Xu, K., Chen, Y., Liu, M., Yang, J., Zhu, H., Wang, Y., et al.: Learning primitive embodied world models: Towards scalable robotic learning. arXiv preprint arXiv:2508.20840 (2025)
- [53] Sutton, R.S.: Dyna, an integrated architecture for learning, planning, and reacting. ACM Sigart Bulletin 2(4), 160–163 (1991)
- [54] Team, O.M., Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., et al.: Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213 (2024)
- [55] Walke, H.R., Black, K., Zhao, T.Z., Vuong, Q., Zheng, C., Hansen-Estruch, P., He, A.W., Myers, V., Kim, M.J., Du, M., et al.: Bridgedata v2: A dataset for robot learning at scale. In: Conference on Robot Learning. pp. 1723–

1736. PMLR (2023)

- [56] Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 3(4), 6 (2025)

- [57] Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)
- [58] Watkins, O., Huang, S., Frost, J., Bhatia, K., Weiner, E., Abbeel, P., Darrell, T., Plummer, B., Saenko, K., Dragan, A.: Explaining robot policies. Applied AI Letters 2(4), e52 (2021)
- [59] Wu, H., Wu, D., He, T., Guo, J., Ye, Y., Duan, Y., Bian, J.: Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling. arXiv preprint arXiv:2507.07982 (2025)
- [60] Wu, P., Escontrela, A., Hafner, D., Abbeel, P., Goldberg, K.: Daydreamer: World models for physical robot learning. In: Conference on robot learning. pp. 2226–2240. PMLR (2023)
- [61] Wu, R., Gao, R., Poole, B., Trevithick, A., Zheng, C., Barron, J.T., Holynski, A.: CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models. arXiv:2411.18613 (2024)
- [62] Xie, Y., Yao, C.H., Voleti, V., Jiang, H., Jampani, V.: Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470 (2024)
- [63] Xing, Y., Luo, X., Xie, J., Gao, L., Shen, H., Song, J.: Shortcut learning in generalist robot policies: The role of dataset diversity and fragmentation. arXiv preprint arXiv:2508.06426 (2025)
- [64] Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20331–20341 (2024)
- [65] Ye, S., Ge, Y., Zheng, K., Gao, S., Yu, S., Kurian, G., Indupuru, S., Tan, Y.L., Zhu, C., Xiang, J., et al.: World action models are zero-shot policies. arXiv preprint arXiv:2602.15922 (2026)
- [66] Zhang, H., Chen, X., Wang, Y., Liu, X., Wang, Y., Qiao, Y.: 4diffusion: Multi-view video diffusion model for 4d generation. Advances in Neural Information Processing Systems 37, 15272–15295 (2024)
- [67] Zhang, W., Li, Y., Qiao, Y., Huang, S., Liu, J., Dayoub, F., Ma, X., Liu, L.: Effective tuning strategies for generalist robot manipulation policies. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 7255–7262. IEEE (2025)
- [68] Zhang, Z., Liao, J., Li, M., Dai, Z., Qiu, B., Zhu, S., Qin, L., Wang, W.: Tora: Trajectory-oriented diffusion transformer for video generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2063– 2073 (2025)
- [69] Zhen, H., Qiu, X., Chen, P., Yang, J., Yan, X., Du, Y., Hong, Y., Gan, C.: 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631 (2024)
- [70] Zhen, H., Sun, Q., Zhang, H., Li, J., Zhou, S., Du, Y., Gan, C.: Tesseract: learning 4d embodied world models. arXiv preprint arXiv:2504.20995 (2025)
- [71] Zheng, D., Huang, S., Zhao, L., Zhong, Y., Wang, L.: Towards learning a generalist model for embodied navigation. In: Proceedings of the

- IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13624–13634 (2024)
- [72] Zhou, S., Du, Y., Chen, J., Li, Y., Yeung, D.Y., Gan, C.: Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377 (2024)
- [73] Zhu, H., Wang, Y., Zhou, J., Chang, W., Zhou, Y., Li, Z., Chen, J., Shen, C., Pang, J., He, T.: Aether: Geometric-aware unified world modeling. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8535–8546 (2025)
- [74] Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al.: Rt-2: Vision-language-action models transfer web knowledge to robotic control. In: Conference on Robot Learning. pp. 2165–2183. PMLR (2023)

