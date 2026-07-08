# arXiv:2409.00558v1[cs.CV]31Aug2024

## Compositional 3D-aware Video Generation with LLM Director

Hanxin Zhu1∗, Tianyu He2, Anni Tang3, Junliang Guo2, Zhibo Chen1, Jiang Bian2 1University of Science and Technology of China 2Microsoft Research Asia 3Shanghai Jiao Tong University

hanxinzhu@mail.ustc.edu.cn, tianyuhe@microsoft.com, memory97@sjtu.edu.cn, junliangguo@microsoft.com, chenzhibo@ustc.edu.cn, jiang.bian@microsoft.com

### Abstract

Significant progress has been made in text-to-video generation through the use of powerful generative models and large-scale internet data. However, substantial challenges remain in precisely controlling individual concepts within the generated video, such as the motion and appearance of specific characters and the movement of viewpoints. In this work, we propose a novel paradigm that generates each concept in 3D representation separately and then composes them with priors from Large Language Models (LLM) and 2D diffusion models. Specifically, given an input textual prompt, our scheme consists of three stages: 1) We leverage LLM as the director to first decompose the complex query into several sub-prompts that indicate individual concepts within the video (e.g., scene, objects, motions), then we let LLM to invoke pre-trained expert models to obtain corresponding 3D representations of concepts. 2) To compose these representations, we prompt multi-modal LLM to produce coarse guidance on the scales and coordinates of trajectories for the objects. 3) To make the generated frames adhere to natural image distribution, we further leverage 2D diffusion priors and use Score Distillation Sampling to refine the composition. Extensive experiments demonstrate that our method can generate high-fidelity videos from text with diverse motion and flexible control over each concept. Project page: https://aka.ms/c3v.

### 1 Introduction

Benefitting from large-scale data and the advancement of the generative models [1, 2], we have witnessed plenty of astonishing results across a wide array of tasks. For example, Large Language Models (LLM) pre-trained on web-scale datasets are revolutionizing machine learning with strong capability of zero-shot learning [3] and planning [4, 5], while diffusion models [6] empower text-toimage generation with a rapid surge in both quality and diversity [7–9].

To harness the power of text-to-image models in text-to-video generation, modern solutions directly view video as multiple images. In this way, tremendous efforts have been dedicated to extending text-to-image models with temporal interaction to ensure consistency between frames [10–17]. However, generating visual content conditioned on the textual prompt alone struggles to express multiple concepts with precise spatial layout control [18–20]. To tackle this issue, LVD [21] and VideoDirectorGPT [22] propose to first generate spatiotemporal bounding boxes of each object based on the textual prompt with LLM, and then condition the video generation on the obtained layouts. Although rough layout control can be realized, they still have inherent limitations for detailed concept control, e.g., the motion and appearance of specific characters, and the movement of viewpoints.

∗This work is accomplished in Microsoft, April 2024.

Preprint. Under review.

In nature, our understanding of the world is compositional [23, 24, 20], and the interaction with the world takes place in a 3D. Motivated by this, in contrast to the prior endeavors that implicitly learn different concepts in 2D space, we are interested in exploring an alternative solution that explicitly composes concepts in 3D space for video generation. To this end, we in particular identify two key technical challenges: 1) Since a textual prompt contains multiple concepts, how to coordinate the generation of various concepts? 2) Given the generated concepts, how to compose them to follow common sense in the real world?

In this work, we introduce text-guided compositional 3D-aware video generation (C3V), a novel paradigm that regards LLM as director and 3D as structural representation for video generation. C3V consists of three main stages: 1) Given a textual prompt, to coordinate the generation of various concepts, we leverage LLM to disassemble the input prompt into sub-prompts, where each subprompt describes an individual concept, e.g., the scene, objects, and motion. For each concept, a pre-trained expert model is assigned by LLM to generate its corresponding 3D representation (e.g., 3D Gaussians [25], SMPL parameters [26]) according to the textual description. 2) To provide coarse instruction for composition (i.e., the scale and trajectory of each object in the scene), we further resort to the priors in multi-modal LLM by querying it with the rendered scene image and the textual goals. However, directly instructing multi-modal LLM to return the scale and trajectory of each object leads to unexpected results, as it is challenging for LLM to estimate visual dynamics. Therefore, we follow a step-by-step reasoning philosophy [27] by representing the object with the bounding boxes and dividing the trajectory estimation into sub-tasks, i.e., estimating the starting points, ending points, and trajectories step-by-step. 3) After obtaining the coarse trajectories from the language space, we also propose to refine the scales, rotations, and exact locations with priors from large-scale visual data. Specifically, taking inspiration from DreamFusion [28], which proposes to distill generative priors from pre-trained image diffusion models into 3D objects, we employ Score Distillation Sampling (SDS) [28] to optimize the transformation matrix of each object in 3D space.

Our system has three main advantages: 1) Because each concept is represented by individual 3D representations, it naturally supports flexible control and interaction of each concept. 2) It inherently excels at synthesizing complex and long videos such as drama, etc. 3) The viewpoint is controllable.

Extensive experiments demonstrate that our proposed method can generate 3D-aware videos with diverse motion and high visual quality, even from complex queries that contain multiple concepts and relationships. We also illustrate the flexibility of C3V by editing various concepts of the generated videos. The generated videos are presented on our project page. To the best of our knowledge, we make the first attempt towards text-guided compositional 3D-aware video generation. We hope it can inspire further explorations on the interplay between video and 3D generation.

### 2 Related Works

#### 2.1 Video Generation with LLM

Recently, there have been substantial efforts in training text-to-video models on large-scale datasets with autoregressive Transformer [29, 30, 17] or diffusion models [10–13, 16]. A prominent approach for text-to-video generation is to extend a pre-trained text-to-image model by inserting temporal layers into its architecture, and fine-tuning models on video data. However, although effective, it remains challenging to generate objects with specific attributes or positions. To address this challenge, a series of studies proposed to exploit knowledge from LLM [31, 32] to achieve controllable generation [21, 19, 22, 33–35], zero-shot generation [36–39], or long video generation [40]. For example, Free-Bloom [36] and DirecT2V [38] used LLM to transform the input textual prompt into a sequence of sub-prompts that describe each frame. LVD [21] and VideoDirectorGPT [22] employed LLM to generate spatiotemporal bounding boxes to control the object-level dynamics in video generation.

In light of the above success of exploiting LLM to direct video generation in 2D space, we view LLM as a director in 3D, which differs from previous methods not only in terms of technical route but also in benefits: providing free interaction with individual concepts and flexible viewpoint control.

#### 2.2 Compositional 3D Generation

Generating 3D assets from textual prompt has garnered significant attention owing to its promising applications in various fields such as AR [41], VR [42], and autonomous driving [43]. However, due to the lack of large-scale 3D data, it is challenging to apply 2D generative models to 3D directly. Therefore, building upon Dream Fields [44], DreamFusion introduced the Score Distillation Sampling (SDS) [28], a technique enhancing 3D generation by distilling 2D diffusion priors from pre-trained text-to-image generative models. Motivated by the success of DreamFusion [28], dedicated efforts have been made to improve SDS [45–47]. Though achieving remarkable results, these methods struggle to generate scenes with multiple distinct elements. To mitigate this issue, several techniques was proposed to guide 3D generation with additional conditions like layout priors, which we refer to as compositional 3D generation [48–50]. However, these works still focus on static compositional

###### 3D generation and lack visual dynamic modeling.

Recently, two concurrent works Comp4D [51] and TC4D [52] also achieved compositional 4D generation (i.e., dynamic 3D generation). However, they only considered composition between objects, and the trajectory of these methods is either formulated by kinematics-based equations [51] or pre-defined by users [52]. Differently, we explore 3D-aware video generation with integrated 3D scenes and compose various concepts with priors from both LLM and 2D diffusion models.

### 3 Preliminaries

#### 3.1 3D Gaussian Splatting

- 3D Gaussian Splatting (3DGS) [25] has been attracting a lot of interest for novel view synthesis, due to its photorealistic visual quality and real-time rendering. 3DGS utilizes a set of anisotropic ellipsoids (i.e., 3D Gaussians) to encode 3D properties, in which each Gaussian is parameterized by position µ ∈ R3, covariance Σ ∈ R3×3 (obtained from scale s ∈ R3 and rotation r ∈ R3), opacity α ∈ R, and color c ∈ R3.

To render a novel view, 3DGS adopts a tile-based rasterization, where 3D Gaussians are projected onto the image plane as 2D Gaussians. The final color c(p) of pixel p is denoted as:

c(p) = c ˆσˆ (1 − σˆ), (1)

where ˆc and σˆ represent the individual color and opacity values of a series of 2D Gaussians contributing to this pixel. 3DGS are then optimized using L1 loss and SSIM [53] loss in a per-view optimization manner. Thanks to the nature of modeling 3D scenes explicitly, optimized 3D Gaussians can be easily controlled and edited.

3.2 Score Distillation Sampling

Different from text-to-image generation which benefits from a large number of text-image pairs available, text-to-3D generation suffers from a severe lack of data. To mitigate this issue, Score Distillation Sampling (SDS) [28] was proposed to distill generative priors from pretrained diffusionbased text-to-image models ϕ. Specifically, for a 3D representation parameterized by θ, SDS is served as a way to measure the similarity between the rendered images x = g(θ) and the given textual prompts y, where g represents the rendering operation. As a result, the gradients used to update θ are computed as follows:

∇θLSDS(ϕ,x = g(θ)) = Et,ϵ[w(t)(ˆϵϕ(xt;y,t) − ϵ)], (2)

where t is the noise level, ϵ is the ground-truth noise, w(t) is a weighting function, ϵˆϕ is the estimated noise given noised images xt with text embeddings y. Please refer to DreamFusion [28] for details.

- 4 Method

Overview. To achieve text-guided compositional 3D-aware video generation (C3V), we regard LLM as director and 3D as structural representation. To this end, our method consists of three stages. To begin with, we utilize LLMs to decompose the input textual prompts into three sub-prompts, each

[Figure 1]

In a Magician's magical cabin alone in a serene forest, an alien walking on the floor, starting from the cabin’s door to the mow near the bottom right corner of this image.

###### Multimodal Large Language Model

Trajectory Generator

Task Decomposer

Scale and trajectory estimation

Complex query into sub-prompts

Composition with 2D Diffusion Priors

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Q: Please give me a trajectory represents that an alien walking on the floor, starting from the cabin’s door to the mow near the bottom right corner of this image.

[Figure 9]

[Figure 10]

a Magician’s ... forest

[Figure 11]

[Figure 12]

Splatting

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

3D Scene

Generation

[Figure 20]

[Figure 21]

"In a Magician’s ...of this image."

[Figure 22]

2D Diffusion

A: Scale: 0.33 Start:

[Figure 23]

Priors

Step-by-Step

[Figure 24]

an alien

[Figure 25]

Estimation

|Scale|
|---|

(380,450) End:

Score Distillation Sampling

[Figure 26]

|Location|
|---|

Object Generation

[Figure 27]

(704,652)

Refinement

|Rotation|
|---|

Trajectory:

(380,450) (408,461) (439,475)

[Figure 28]

[Figure 29]

[Figure 30]

walking on the floor

[Figure 31]

Splatting

[Figure 32]

[Figure 33]

… (624,580)

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Motion Generation

[Figure 41]

[Figure 42]

(666,608)

[Figure 43]

(704,652)

- Figure 1: Illustration of our method. It consists of three stages: 1) The input textual prompt is decomposed into individual concepts by the LLM. Then we generate each concept in the form of

- 3D with the corresponding pre-trained expert model (left & Sec. 4.1). 2) We leverage knowledge in multi-modal LLM to estimate the 2D trajectory of objects step-by-step (middle & Sec. 4.2). 3) After lifting the estimated 2D trajectory into 3D as initialization, we refine the scales, locations, and rotations of objects within the 3D scene using 2D diffusion priors (right & Sec. 4.3).

of which provides a description for generating a corresponding concept (i.e., scene, object, motion, etc.) respectively (Sec. 4.1). Subsequently, we leverage multi-modal LLM to obtain coarse-grained scales and trajectories for each animatable object (Sec. 4.2). Finally, we employ 2D diffusion priors to refine the objects’ location, scale, and rotation for a fine-grained composition (Sec. 4.3).

#### 4.1 Task Decomposition with LLM

Task Instructions. Given a textual prompt, we invoke LLM (e.g., GPT-4V [32]) to decompose it into several sub-prompts. Each sub-prompt describes an individual concept such as the scene, object, and motion. Specifically, for an input prompt y, we query LLM with the instruction like: "Please decompose this prompt into several sub-prompts, each describing the scene, objects in the scene, and the objects’ motion.", from which we obtain the corresponding sub-prompts.

3D Representation. After obtaining the sub-prompt for each concept, we aim to generate its corresponding 3D representations using the pre-trained expert models. In this work, we build structural representation on 3DGS [25], which is an explicit form and therefore flexible enough to compose or animate. Concerning concepts like motion, our framework can generalize to arbitrary animatable 3D Gaussian-based objects. For simplicity, we take human motion as an instantiation because it is general for various scenarios. In order to obtain diverse human motions, we take a retrieval-augmented approach [54] to acquire motion in the form of SMPL-X parameters [55] from large motion libraries [56] according to the motion-related sub-prompt.

Instantiation. To illustrate the scheme formally, consider the following example. We have subprompts y1,y2 and y3 that describe scene, object, and motion respectively. Additionally, we have corresponding pre-trained text-guided expert models ϕ1, ϕ2, and ϕ3 that are selected by the LLM. The concept generation can be formulated as follows:

##### G1 = ϕ1(y1), G2 = ϕ2(y2,M), M = ϕ3(y3), (3)

[Figure 44]

- Figure 2: Illustration of coarse-grained trajectory generation with LLM. Instead of querying multimodal LLM to estimate dynamic trajectory directly, we generate trajectory in a step-by-step manner: estimating the locations of starting and ending points first, then reasoning the path between them.

where G1 and G2 represent the 3D Gaussians, and M means the motions used to drive G2. In the following sections, we will provide details on the composition of the generated concepts.

#### 4.2 Coarse-grained Trajectory Generation with LLM

Given the generated concepts, we aim to compose them into a dynamic 3D representation to render videos that align with the input textual prompt. Achieving this requires scales and trajectories of the objects to indicate their sizes and locations within the scene. To this end, we propose to leverage knowledge encoded in multi-modal LLMs (i.e., GPT-4V [32]) to provide priors.

For the scale of the object, we find that directly querying GPT-4V with the input prompt and rendered scene image can yield a reasonable estimation of its resolution (H2D and W2D). However, this is not the case for trajectory estimation. As demonstrated in Fig. 2, directly querying GPT-4V for trajectory will lead to a result that deviates conspicuously from common sense. Based on this observation, we conclude two issues: 1) it is too difficult for GPT-4V to generate the trajectory within a single query, as it lacks priors on visual dynamics; 2) since GPT-4V is trained to generate text, it has limitations on imagining visual content.

To mitigate this, we introduce two simple yet effective techniques. 1) Although GPT-4V lacks visual knowledge of the object, we can alleviate this by representing the object as a bounding box with the estimated resolution. 2) We follow a step-by-step reasoning philosophy [27] and propose to let GPT-4V estimate the locations of starting and ending points first, then reason the path between them.

Overall, we can formulate the above process as follows:

{Lip}Ni=1 = Φ(yp,I,S,Ls,Le), S =Φ(y,I),Ls = Φ(ys,I),Le = Φ(ye,I),

(4)

where Φ represents the multi-modal LLM (i.e., GPT-4V), I denotes the rendered scene image, S represents the estimated scale of the object given textual prompt y and I, Ls and Le represent the locations of starting and ending points respectively, {Lip}Ni=1 represent N locations indicating the path between Ls and Le. Notably, all locations (i.e., Ls, Le,{Lip}Ni=1 ) are represented by 2D pixel coordinates on I.

#### 4.3 Fine-grained Composition with 2D Diffusion Priors

Lift Trajectory from 2D to 3D. In Sec. 4.2, we obtain the 2D pixel coordinates Lip = (pix,piy) of the estimated trajectory. However, 2D trajectory is not enough for composition in 3D space.

Therefore, we convert it into corresponding 3D world coordinate Li3D = (xi,yi,zi). Specifically, we first predict the depth map D of the rendered scene image with a monocular depth estimator [57].

Then, we use the depth value of the center point of the lower boundary of the bounding box as the trajectory’s depth. As a result, we can transform 2D trajectory into 3D:

H2D 2

H2D 2

H3D 2

(xi,yi,zi,1)T = R−1K−1[(pix +

,piy,1)T · D(pix +

,piy)] − (

,0,0,0)T, (5)

- where R and K represent camera extrinsic and intrinsic respectively, H2D and W2D represent the resolution of the 2D bounding box. H3D represent the actual height of the 3D bounding box of this object within the scene.

Composition Refinement with 2D Diffusion Priors. With the lifted 3D trajectory, we then integrate the object into the scene. However, the trajectory estimated by LLM is still rough and may not obey natural image distribution. To address this, we propose to further refine the object’s scale, location, and rotation by distilling generative priors from pre-trained image diffusion models [7] into 3D space. Specifically, we treat the parameters for these attributes as optimizable variables and use SDS (Eq. 2) to improve the fidelity of rendered images. As a result, scale refinement can be formulated as follows:

∇SˆLScaleSDS = Et,ϵ[w(t)(ˆϵϕ(xt(L13D,(S + σ(Sˆ) · τs −

τs 2

) · G2);y,t) − ϵ)], (6)

- where Sˆ represents the optimizable variable, S represents the scale estimated by GPT-4(V), σ means the Sigmoid function, τs is a threshold, G2 represents the 3D gaussians of the object, and xt is the noised 2D image given Li3D and scaled G2.

After obtaining a more precise scale, we then refine the locations of the estimated 3D trajectory similarly, where the location refinement is denoted as:

τs 2

τL 2

∇LˆiLLocationSDS = Et,ϵ[w(t)(ˆϵϕ(xt(Li3D +σ(Lˆi)·τL−

,(S+σ(Sˆ)·τs−

)·G2);y,t)−ϵ)], (7)

where Lˆi represents the optimizable variable, τL is a threshold.

For the rotation of the object at different timesteps, we can directly compute the corresponding rotation matrix, based on the assumption that the object at the current time step should face the location of the object at the next time step. As a result, the rotation matrix Rˆi at time step i can be computed using the following equation:

 

 ,

tx2 + c txy − zs txz + ys txy + zs ty2 + c tyz − xs txz − ys tyz + xs tz2 + c

Rˆi =

(8)

t = 1 − c,c = cos(θ),s = sin(θ),u = (x,y,z)T

where θ and u represent the rotation angle and axis obtained through the cross product of (Li3+1D + σ(Liˆ+1) · τL − Li3D − σ(Lˆi) · τL) and (0,0,1)T.

Inference. After obtaining individual concepts in the form of 3D and the optimized parameters that indicate how to compose various concepts, we can render the 3D representation into 2D video with flexible camera control in real time [25].

### 5 Experiments

In this section, we instantiate C3V with three concepts: scene, humanoid object, and human motion, to generate 3D-aware video from text. We compare our proposed method with state-of-the-art text-to-

- 4D models (4D-FY [58]), compositional 4D generation models (Comp4D [51]) and text-to-video models (VideoCrafter2 [59]). Videos are available on our anonymous project page.

Implementation Details. We use LucidDreamer [60], HumanGaussian [61] and Motion-X [56] to generate 3D scenes, humanoid objects and motions respectively. To realize SDS, we utilize Stable Diffusion [7] as the image diffusion model. All the videos of our proposed method are rendered at a resolution of 512 × 512 in real time. Please refer to the appendix for more details.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

VideoCrafter2Comp4DOurs4D-FY

|[Figure 52]|
|---|

|[Figure 53]|[Figure 54]|[Figure 55]|[Figure 56]|[Figure 57]|[Figure 58]|
|---|---|---|---|---|---|

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

- (a) Text prompt: "In a Magician’s magical cabin alone in a serene forest, an alien walking on the floor, starting from the cabin’s door to the mow near the bottom right corner of this image".

|[Figure 72]|[Figure 73]|[Figure 74]|
|---|---|---|

|[Figure 75]|[Figure 76]|[Figure 77]|[Figure 78]|
|---|---|---|---|

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

VideoCrafter2Comp4DOurs4D-FY

- (b) Text prompt: "Four characters stood on the stage. In front of the stage, a man and a woman are performing Kung Fu and dancing respectively. On the right side of the stage, a skeleton man is dancing, and behind them, a clown is performing".

- Figure 3: Qualitative comparisons with baselines. When prompting complex queries, the baseline methods fail to follow the queries in terms of the number of objects and the corresponding motion. In contrast, our method excels in yielding both diverse motion and high visual quality.

Metrics. Following Comp4D [51], we choose Q-Align [62] as the referee to measure the quality and aesthetics of the video. The Q-Align score is a number ranging from 1 (worst) to 5 (best) where a higher score indicates a better performance. We also report the CLIP score [63] to measure the alignment between the generated videos and the input texts.

#### 5.1 Comparison with Competitors

In Fig. 3, we conduct a comparative analysis of our method against 4D-FY [58], Comp4D [51], and VideoCrafter2 [59] with the same textual prompt. It can be observed that all three baselines fail to provide diverse motion from the textual prompt, while our method excels in yielding large motion and high visual quality. For example, our scheme successfully obeys the complex query in terms of

Table 1: Quantative comparisons with competitors. Our method consistently outperforms all baseline methods in terms of both the video quality and the alignment with textual prompts.

Metric 4D-FY [58] Comp4D [51] VideoCrafter2 [59] Ours QAlign-img-quality ↑ [62] 1.681 1.687 3.839 4.030 QAlign-img-aesthetic↑ [62] 1.475 1.258 3.199 3.471 QAlign-vid-quality↑ [62] 2.154 2.142 3.868 4.112 QAlign-vid-aesthetic↑ [62] 1.580 1.425 3.159 3.723 CLIP Score↑ [63] 30.47 27.50 35.20 38.36

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

(II) Trajectory estimation with bounding box indicating objects.

(III) Trajectory estimation in a step-by-step manner.

(I) Direct trajectory estimation.

(IV) Ours.

- (a) Ablation studies on trajectory estimation with multi-modal LLM.
- (b) Ablation studies on composition with 2D diffusion models.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

(I) Without SDS refinement. (II) With scale refinement. (III) With trajectory refinement. (IV) With rotation refinement.

- Figure 4: Ablation studies on framework design. Each ablation is prompted with the same text.

the number of objects and the corresponding motion. In addition, since 4D-FY and Comp4D focus on object-centric generation, they fail to generate videos with natural backgrounds. In Tab. 1, we perform quantitative comparisons by utilizing Q-Align Score [62] and CLIP Score [63] to assess the quality of generated videos. Our method consistently outperforms the baseline models in terms of both the video quality and the alignment with textual prompts. More results are available in the appendix.

#### 5.2 Ablation Studies

Ablations on Trajectory Estimation with Multi-modal LLM. As shown in Fig. 4(a)(I), a direct prompt of GPT-4V will lead to obvious unsatisfactory trajectory estimation. When only depending on bounding boxes to indicate the location of objects within the scene (Fig. 4(a)(II)), though a roughly better trajectory can be achieved, it still leads to unreasonable results, such as several floating bounding boxes. Similarly, using only the step-by-step estimation strategy described in Sec. 4.2 typically results in a trajectory that is merely a simple straight line connecting the starting and ending points (Fig. 4(a)(III)). With both of the two techniques, we can achieve the best performance, with a more reasonable and smooth trajectory (Fig. 4(a)(IV)).

Ablations on Composition with 2D Diffusion Models. To figure out whether it is necessary to conduct fine-grained composition with 2D generative priors, we gradually refine the scales, locations, and rotations with SDS and visualize the results in Fig. 4(b). All results are generated with the same textual prompt: "An alien walking on the floor in front of the cabin’s door.". It shows that when we optimize the attributes with SDS, we can obtain consistently improved performance with a reasonable scale (Fig. 4(b)(II), accurate locations that are aligned with the input prompt (Fig. 4(b)(III), and orientation that accords with common sense (Fig. 4(b)(IV)).

[Figure 108]

- Figure 5: Our method offers flexible control of individual concepts. We demonstrate this by editing different concepts: the appearance and motion of the actors, and the scenes.

5.3 Applications on Controllable Generation

Due to our underlying 3D structural representation, our scheme has the natural merits of editing individual concepts. We illustrate this character in Fig. 5 by editing three different concepts: the appearance and motion of the actors, and the scenes. For the appearance and motion of the actor, we can seamlessly replace them in a zero-shot manner according to the textual prompt (Fig.5(a)(b)), while this is still challenging for implicit models [64, 65]. For scene editing, to ensure a smooth composition of objects within the target scene, we re-estimate the trajectory of the objects given the target scene. Kindly refer to appendix for more results.

- 6 Conclusion

In this paper, we present a novel paradigm for 3D-aware video generation by conceptualizing videos as compositions of independent concepts represented in 3D space. To this end, we leverage LLM as director to decompose the input textual prompts into individual concepts and then invoke pre-trained expert models to generate them separately. To compose various concepts, we first prompt multi-modal LLM in a step-by-step manner to provide coarse guidance on the scale and trajectory of objects, then refine the composition with 2D generative priors. We verify our scheme in different scenarios, demonstrating its superiority over the baseline methods.

Limitations and Future Works. Although we demonstrate promising results in 3D-aware video generation, there still are limitations to be improved in the future. First, our framework is instantiated with limited concepts in this work, i.e., scene, humanoid object, and human motion. It is exciting to generalize the framework to more concepts like animals, vehicles, etc. Second, the composition between concepts is conducted with priors from LLM and 2D diffusion priors in our method. However, it is still interesting to introduce physically grounded dynamics into 3D representation [66]. Third, though our method is naturally suitable for maintaining the consistency of actors across different

scenes, it still needs further exploration on long video generation with multiple scenes, e.g., a full-length film.

Ethics Statement. C3V is exclusively a research initiative with no current plans for product integration or public access. We are committed to adhering to Microsoft AI principles during the ongoing development of our models. The model is trained on AI-generated content, which has been thoroughly reviewed to ensure that they do not include personally identifiable information or offensive content. Nonetheless, as these generated data are sourced from the Internet, there may still be inherent biases. To address this, we have implemented a rigorous filtering process on the data to minimize the potential for the model to generate inappropriate content.

### References

- [1] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 33:6840–6851, 2020.
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems (NeurIPS), 33:1877– 1901, 2020.
- [4] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024.
- [5] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. In Conference on Robot Learning, pages 540–562. PMLR, 2023.
- [6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems (NeurIPS), 34:8780–8794, 2021.
- [7] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022.
- [8] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [9] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems (NeurIPS), 35:36479–36494, 2022.
- [10] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv:2204.03458, 2022.
- [11] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [12] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22563–22575, 2023.
- [13] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023.

- [14] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [15] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023.
- [16] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662, 2023.
- [17] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.
- [18] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.
- [19] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36, 2024.
- [20] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022.
- [21] Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. Llm-grounded video diffusion models. In International Conference on Learning Representations, 2024.
- [22] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091, 2023.
- [23] Noam Chomsky. Aspects of the Theory of Syntax. Number 11. MIT press, 2014.
- [24] Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. Human-level concept learning through probabilistic program induction. Science, 350(6266):1332–1338, 2015.
- [25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.
- [26] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: a skinned multi-person linear model. ACM Transactions on Graphics (TOG), 34(6):1–16, 2015.
- [27] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024.
- [28] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2023.
- [29] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2023.
- [30] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In International Conference on Learning Representations, 2023.

- [31] OpenAI. Chatgpt. https://openai.com/chatgpt, 2022.
- [32] OpenAI. Gpt-4v(ision) system card. https://openai.com/index/gpt-4v-system-card, 2023.
- [33] Sitong Su, Litao Guo, Lianli Gao, Hengtao Shen, and Jingkuan Song. Motionzero: Exploiting motion priors for zero-shot text-to-video generation. arXiv preprint arXiv:2311.16635, 2023.
- [34] Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. Peekaboo: Interactive video generation via masked-diffusion. arXiv preprint arXiv:2312.07509, 2023.
- [35] Sixiao Zheng, Jingyang Huo, Yu Wang, and Yanwei Fu. Intelligent director: An automatic framework for dynamic visual composition using chatgpt. arXiv preprint arXiv:2402.15746, 2024.
- [36] Hanzhuo Huang, Yufan Feng, Cheng Shi, Lan Xu, Jingyi Yu, and Sibei Yang. Free-bloom: Zero-shot text-to-video generator with llm director and ldm animator. Advances in Neural Information Processing Systems, 36, 2024.
- [37] Yu Lu, Linchao Zhu, Hehe Fan, and Yi Yang. Flowzero: Zero-shot text-to-video synthesis with llm-driven dynamic scene syntax. arXiv preprint arXiv:2311.15813, 2023.
- [38] Susung Hong, Junyoung Seo, Heeseong Shin, Sunghwan Hong, and Seungryong Kim. Direct2v: Large language models are frame-level directors for zero-shot text-to-video generation. arXiv preprint arXiv:2305.14330, 2023.
- [39] Gyeongrok Oh, Jaehwan Jeong, Sieun Kim, Wonmin Byeon, Jinkyu Kim, Sungwoong Kim, Hyeokmin Kwon, and Sangpil Kim. Mtvg: Multi-text video generation with text-to-video models. arXiv preprint arXiv:2312.04086, 2023.
- [40] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. Vlogger: Make your dream a vlog. arXiv preprint arXiv:2401.09414, 2024.
- [41] Chenliang Chang, Kiseung Bang, Gordon Wetzstein, Byoungho Lee, and Liang Gao. Toward the next-generation vr/ar optics: a review of holographic near-eye displays from a human-centric perspective. Optica, 7(11):1563–1578, 2020.
- [42] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multiview diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023.
- [43] Xiangyu Yue, Bichen Wu, Sanjit A Seshia, Kurt Keutzer, and Alberto L Sangiovanni-Vincentelli. A lidar point cloud generator: from a virtual world to autonomous driving. In Proceedings of the 2018 ACM on International Conference on Multimedia Retrieval, pages 458–464, 2018.
- [44] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 867–876, 2022.
- [45] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024.
- [46] Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, Zheng-Jun Zha, and Lei Zhang. Dreamtime: An improved optimization strategy for text-to-3d content creation. arXiv preprint arXiv:2306.12422, 2023.
- [47] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. arXiv preprint arXiv:2311.11284, 2023.
- [48] Ryan Po and Gordon Wetzstein. Compositional 3d scene generation using locally conditioned diffusion. arXiv preprint arXiv:2303.12218, 2023.
- [49] Gege Gao, Weiyang Liu, Anpei Chen, Andreas Geiger, and Bernhard Schölkopf. Graphdreamer: Compositional 3d scene synthesis from scene graphs. arXiv preprint arXiv:2312.00093, 2023.

- [50] Xiaoyu Zhou, Xingjian Ran, Yajiao Xiong, Jinlin He, Zhiwei Lin, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Gala3d: Towards text-to-3d complex scene generation via layout-guided generative gaussian splatting. arXiv preprint arXiv:2402.07207, 2024.
- [51] Dejia Xu, Hanwen Liang, Neel P Bhatt, Hezhen Hu, Hanxue Liang, Konstantinos N Plataniotis, and Zhangyang Wang. Comp4d: Llm-guided compositional 4d scene generation. arXiv preprint arXiv:2403.16993, 2024.
- [52] Sherwin Bahmani, Xian Liu, Yifan Wang, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. Tc4d: Trajectory-conditioned text-to-4d generation. arXiv preprint arXiv:2403.17920, 2024.
- [53] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600– 612, 2004.
- [54] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations, 2020.
- [55] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10975–10985, 2019.
- [56] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems, 36, 2024.
- [57] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179– 12188, 2021.
- [58] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. arXiv preprint arXiv:2311.17984, 2023.
- [59] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047, 2024.
- [60] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023.
- [61] Xian Liu, Xiaohang Zhan, Jiaxiang Tang, Ying Shan, Gang Zeng, Dahua Lin, Xihui Liu, and Ziwei Liu. Humangaussian: Text-driven 3d human generation with gaussian splatting. arXiv preprint arXiv:2311.17061, 2023.
- [62] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023.
- [63] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [64] Wilson Yan, Andrew Brown, Pieter Abbeel, Rohit Girdhar, and Samaneh Azadi. Motionconditioned image animation for video editing. arXiv preprint arXiv:2311.18827, 2023.

- [65] Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. Uniedit: A unified tuning-free framework for video motion and appearance editing. arXiv preprint arXiv:2402.13185, 2024.
- [66] Tianyi Xie, Zeshun Zong, Yuxin Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physics-integrated 3d gaussians for generative dynamics. arXiv preprint arXiv:2311.12198, 2023.

