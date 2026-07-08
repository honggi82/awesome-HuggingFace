# arXiv:2504.08366v3[cs.GR]27Sep2025

## In-2-4D: Inbetweening from Two Single-View Images to 4D Generation

SAURADIP NAG, Simon Fraser University, Canada DANIEL COHEN-OR, Tel Aviv University, Israel HAO ZHANG, Simon Fraser University, Canada ALI MAHDAVI-AMIRI, Simon Fraser University, Canada

We pose a new problem, In-2-4D, for generative 4D (i.e., 3D + motion) inbetweening to interpolate two single-view images. In contrast to video/4D generation from only text or a single image, our interpolative task can leverage more precise motion control to better constrain the generation. Given two monocular RGB images representing the start and end states of an object in motion, our goal is to generate and reconstruct the motion in 4D, without making assumptions on the object category, motion type, length, or complexity. To handle such arbitrary and diverse motions, we utilize a foundational video interpolation model for motion prediction. However, large frame-to-frame motion gaps can lead to ambiguous interpretations. To this end, we employ a hierarchical approach to identify keyframes that are visually close to the input states while exhibiting significant motions, then generate smooth fragments between them. For each fragment, we construct a 3D representation of the keyframe using Gaussian Splatting (3DGS). The temporal frames within the fragment guide the motion, enabling their transformation into dynamic 3DGS through a deformation field. To improve temporal consistency and refine the 3D motion, we expand the self-attention of multi-view diffusion across timesteps and apply rigid transformation regularization. Finally, we merge the independently generated 3D motion segments by interpolating boundary deformation fields and optimizing them to align with the guiding video, ensuring smooth and flicker-free transitions. Through extensive qualitative and quantitive experiments as well as a user study, we demonstrate the effectiveness of our method and design choices. Project Page & Source Code: https://in-2-4d.github.io/

Additional Key Words and Phrases: 3D motion interpolation, diffusion models, 4D reconstruction and synthesis from images

#### ACM Reference Format:

Sauradip Nag, Daniel Cohen-Or, Hao Zhang, and Ali Mahdavi-Amiri. 2025. In-2-4D: Inbetweening from Two Single-View Images to 4D Generation. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3757377.3763904

1 INTRODUCTION

Motion inbetweening is a classic animation problem. When generating motions of 3D objects, the typical input consists of a 3D object in two distinct motion states, e.g., as in point cloud interpolation [Peng et al. 2024; Zheng et al. 2023]. With significant progresses in 3D generative AI in recent years, many recent attempts have been made on “video-to-4D" [Jiang et al. 2023; Ren et al. 2023; Wu et al. 2024; Yin et al. 2023; Zeng et al. 2024], whose task is to “lift” an object captured in a video into the 3D space so its motion from the video can be viewed from all angles. However, videos can be unrealistic (e.g., for very slow motions such as the flower blossoming in Fig. 1)

Please use nonacm option or ACM Engage class to enable CC licenses This work is licensed under a Creative Commons Attribution 4.0 International License. SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2137-3/2025/12 https://doi.org/10.1145/3757377.3763904

[Figure 1]

Fig. 1. In-2-4D: 4D motion inbetweening from 2 single-view images, a minimalistic input setup while still offering visual motion control. Given two monocular RGB images of an object at two distinct motion states (start and end), our method generates a smooth, natural, and seamless 4D (3D object + motion) interpolation between them. We make no assumptions on the object category or motion type. Piecewise rigid, freeform, and long-range motions, even with topology changes (e.g., egg dropping in liquid causing a splash), are all allowed.

or at least expensive (e.g., for very fast motions) to acquire. In other application settings such as motion planning and visual storyboarding [Zhong et al. 2025], the motions produced are the outcomes to be explored; they were not acquirable in the first place.

An intriguing question is whether the inbetweening and lifting problems above can be “fused" to produce 4D contents, i.e., 3D object with motion, from a minimalistic and easy-to-acquire input setup. Indeed, with rapid advances in video foundation models and a push for controllable motion generation, there has been an emergence of works on video frame interpolation [Danier et al. 2024; Feng et al. 2024; Reda et al. 2022; Wang et al. 2025; Xing et al. 2024] which produce video, rather than 4D, outputs from two images.

In this paper, we seek a solution to the novel task of generating 4D interpolative contents from merely two single-view images capturing an object in two distinct motion states. Compared to video or 4D

[Figure 2]

- Fig. 2. Comparing to text-to-4D and single-image-to-4D. Text-only and single-image inputs are insufficient to constrain the 4D generation by state-of-the-art models (L4GM [Ren et al. 2025; Zeng et al. 2024] , leading to unnatural results and strong artifacts, especially for complex motions such as the one shown. Results from our method can be found in Fig. 1.

generation from text only [Bahmani et al. 2024; Singer et al. 2023] or from a single image [Ren et al. 2023], our interpolative task anchored on two images can leverage more precise motion control to better constrain the generative process and lead to more plausible and cleaner results, as shown in Figs. 1 and 2.

We call our task and the proposed method both as In-2-4D, for Inbetweening from two (2) single-view images to 4D generation. Aside from the sparse inputs, we aim to tackle additional challenges related to the diversity and complexity of the generated motions: a) no particular assumptions are made on the object or motion categories; b) arbitrary freeform motions are permissable, without assumptions on rigidity or volume/topology preservation, e.g., see Fig. 1 for a floral motion that is non-rigid and quite intricate, and an egg dropping into liquid, causing a splash and a topology change; c) moderately complex and long-range motions are allowed, where the two input motion states are not assumed to be close in time, pose, or structure. Our goal is to synthesize a smooth and believable 3D transition between them.

At the high level, our method operates in two phases: 2D still images to video via interpolation, and then video-to-4D via lifting, as illustrated in Fig. 3. To handle arbitrary and diverse motions, we leverage video foundational models. However, most such models are built on video diffusion [Blattmann et al. 2023b], which has been trained predominantly by short videos. As such, they can be ineffective for motion inbetweening when the input states span large geometry or structural changes, resulting in large motion “jumps” and absence of detailed and intricate object movements. When such video models are used to model complex motion, they often suffer from visible artifacts which further detoriate the novel views when lifted to 4D space.

To this end, we develop a divide-and-conquer approach to adaptively and recursively generate a set of keyframes to form a coarse outline of the motion, breaking the long-range and nonlinear motion into simpler, temporally-local transitions that are easier to model and interpolate. To start, we employ a foundational video interpolation model such as DynamiCrafter [Xing et al. 2023] to generate an initial set of intermediate frames between the two input states.

Then we perform motion and appearance feature analyses over these frames to select one or more keyframes that are visually close to the input states and show significant motion jumps. Consecutive keyframes that incur a large motion will anchor a new video interpolation to generate more immediate frames, effectively “magnifying” the said motion. This process is carried out hierarchically until all motions between consecutive keyframes are sufficiently small. Notably, our video fragment generation is training-free and requires no fine-tuning of video diffusion models, making it both accessible and effective for 4D generation. For each video fragment, and in parallel with other fragments, we first learn a distinct static 3D Gaussian splatting (3DGS) model to capture the object geometry. We then apply a deformation field to convert this 3DGS into a dynamic, i.e., 4D, model by utilizing multi-view diffusion priors to refine the warping, geometry, and textures over unseen areas. By construction, the fragment contains relatively simple motions, hence multi-view generation can effectively mitigate texture and geometry degradation.

Finally, we merge the independently generated 4D fragments where we first linearly interpolate and then optimize the deformation fields over an overlapping frame and regularize the geometry of novel views in a cascading sliding window fashion to smooth the orientation of the dynamic 3DGS based on the neighboring frames. Fig. 3 overviews our pipeline.

Our main contributions are summarized below:

- • To the best of our knowledge, In-2-4D is the first method for generative 4D inbetweening over two distant monocular frames spanning arbitrary motions.
- • Our novel hierarchical approach breaks the complex inbetweening into a series of simpler motion estimations via video, and then 4D (i.e., dynamic 3DGS) generation.
- • To generate smooth 3D object and motion transitions, we further optimize the 3D trajectories using a bottom-up merging strategy with smoothing regularization.
- • We contribute a new 4D interpolation benchmark I4D-15 on challenging real world object motions.

We conduct extensive experiments on I4D-15 and Consistent4D [Jiang et al. 2023] benchmark for evaluation. Quantitative and qualitative comparisons are made to methods and baselines to demonstrate the effectiveness of our method in terms of the quality of generated results, generalizability, and handling of a variety of motions; see Fig. 1. While achieving superior generation quality and remaining faithful to the input images than all current alternatives, our solution is far from artifact-free. This highlights the significant challenges that still lie ahead with 4D content generation from highly sparse inputs which still aim to exert intricate control. As a first attempt at tackling such a complex problem, we hope it can serve as a promising start to stimulate future work.

2 RELATED WORK

In computer graphics, most works on motion inbetweening have been developed for character animation, e.g., [Cohan et al. 2024; Qin et al. 2022; Zhong et al. 2025; Zhou et al. 2020]. Our work aims to handle arbitrary and diverse motions, leveraging foundational

[Figure 3]

- Fig. 3. Illustration of our In-2-4D solution pipeline. Given two single-view images as input (a), we first generate keyframes inbetween them to bridge the motion gaps so as to avoid abrupt motions between consecutive keyframes to generate video fragments (b) . These keyframes are then utilized to learn static

- 3D geometries per fragment which are then deformed using a deformation field (e.g., Hexplane) to obtain a 4D scene in the form of 3D Gaussian splats (3DGS) per fragment. Finally, we obtain a 4D scene in the form of 3D Gaussian splats (3DGS) per fragment. (c) To aggregate the deformations, we linearly interpolate the deformation fields (d) in a cascading fashion and apply smoothing constraints on the 3DGS to improve geometric and structural consistency between the generated novel views.

models, for the novel task of In-2-4D. In this section, we mainly cover related works on 4D generation and video/4D interpolation.

- 4D dynamic scene generation. Recent works [Guo et al. 2024a; Liang et al. 2023; Ren et al. 2023; Wu et al. 2023b, 2024; Yang et al. 2023; Zeng et al. 2024] extend 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023a] to 4D using time-conditioned deformation networks with Score Distillation Sampling (SDS) and multi-view geometry. MAV3D [Singer et al. 2023] pioneered text-to-4D via NeRF and score distillation, followed by similar approaches [Bahmani et al. 2024]. Consistent4D [Jiang et al. 2023] introduces video-to-4D with pretrained image diffusion models, extending to video-conditional 4D generation [Shi et al. 2023; Wang and Shi 2023; Yang et al. 2024b]. STAG4D [Zeng et al. 2024] and 4DGen [Yin et al. 2023] refine diffusion with pseudo-labels, while SC4D [Wu et al. 2024] employs sparse Gaussians and Linear Binding Skinning (LBS) for dynamic

Video inbetweening or frame interpolation. Motion inbetweening allows finer-grained control compared to textual inputs or motion extrapolation from a single frame. Recent methods have extended pre-trained diffusion-based text-to-image models to generate motion from static images by adapting UNets to temporal data [Khachatryan et al. 2023; Singer et al. 2022; Wu et al. 2023a]. One notable model is AnimateDiff [Guo et al. 2024b], which learns low-rank adapters for diverse motion patterns. More recent approaches condition pre-trained text-to-video models on input images. VideoCrafter1 [Chen et al. 2023] uses dual cross-attention layers to combine image features with text prompts, while DynamiCrafter [Xing et al. 2023] further refines this by concatenating the input image with noisy latent features. Our method builds on DynamiCrafter to enhance its outputs through recursive video magnification. While several video magnification techniques exist [Le Ngo and Phan 2019; Liu et al. 2005], we leverage a video inbetweening network (e.g., DynamiCrafter) to interpolate frames and amplify motion when large displacements are present. Decomposing large motions into smaller fragments with smoother transitions reduces geometric ambiguities between consecutive frames, producing 4D results with fewer artifacts and improved visual quality.

- 3D generation. Recently, research has shifted towards SDS-free 4D foundation models [Jiang et al. 2024; Liang et al. 2024b; Ren et al. 2025; Sun et al. 2024; Wu et al. 2025; Xie et al. 2024; Zhang et al. 2024] that synthesize synchronized multi-view videos as intermediates for 4D reconstruction, bypassing SDS optimization. While these models produce plausible 4D motions, they remain limited by the inductive bias of their training data and fail to generalize to scenes with complex dynamics. Despite progress in video diffusion, current video-to-4D methods still struggle with highly dynamic motions and accumulate errors over long sequences due to reliance on a single canonical model. To address this, our framework segments videos into shorter fragments, via keyframe generation, each associated with its own 3D canonical model, thereby improving geometry.

4D scene interpolation. Lately, dynamic 3D scene interpolation, or 4D Interpolation, has become more popular. Earlier works [Park et al. 2023] leverage neural radiance fields (NeRF) for temporally coherent 3D reconstructions, while NeuralPCI [Zheng et al. 2023] employs neural fields for multi-frame, non-linear 3D point cloud interpolation. PAPR [Peng et al. 2024] estimates motion via point-based rendering and local displacement optimization. Recent methods [Jiang et al. 2023; Ling et al. 2024; Ren et al. 2025] use frame motions from

[Figure 4]

Diffusion-based Video Interpolation models [Blattmann et al. 2023a; Xing et al. 2023] to infer 3D deformation. However, Video Diffusion models [Blattmann et al. 2023b], trained on short clips (e.g., 16 frames), struggle with long sequences, causing artifacts from large per-frame motion jumps. To address this, we use generative Video Interpolation models (e.g., DynamiCrafter [Xing et al. 2023]) hierarchically for longer 3D trajectory estimation without extra training. In addition, In-2-4D represents the first attempt at general

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Start Generated Intermediates End

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

DIFT Heatmap

[Figure 16]

[Figure 17]

[Figure 18]

- 4D interpolation from two single-view motion frames.

Large Change = Split Here

3 METHODOLOGY

Visual Artifacts (Regenerate Again)

Keyframe

An overview of our method is shown in Fig. 3. Given two images representing the start and end states of an object in motion, we aim to generate and reconstruct the motion in 4D (3D+motion). To predict the motion, we use a video interpolation model, but large motions between frames can lead to ambiguous interpretations and results with artifacts. To solve this, we employ a hierarchical approach to identify keyframes that are visually close to the input states and exhibit significant motion, then generate smooth fragments between them. For each fragment, the 3D representation of the keyframe is first constructed using Gaussian Splatting. The temporal frames within the fragment serve as motion guidance, enabling their transformation into dynamic Gaussians through a deformation field. To enhance temporal consistency and refine 3D motion of the fragment, we expand the self-attention of multi-view diffusion across time steps and introduce rigid transformation regularization. Finally, the independently generated 3D motion segments are merged by interpolating the boundary deformation fields and optimizing them to align with the guiding video. This ensures flicker-free transitions.

[Figure 19]

[Figure 20]

No Visual Artifacts (No Regeneration)

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Quasi-Static Motion

Fragment 1 Fragment 2

Fig. 4. Illustration of Hierarchal Fragment Generation. At each generation step, a keyframe is selected by finding the largest motion from the DIFT heatmap and FID score. New frames are re-generated using the keyframe to minimize large motion changes. Selection of the keyframes are done in a hierarchial manner to generate fragments having simple motions

Hierarchical key-frame generator. To generate keyframes for the intermediate motion between two initial states, we employ a Video Interpolation Model (e.g., DynamiCrafter), denoted as 𝜓(.). Given input images 𝐼𝑠 and 𝐼𝑒 along with a motion prompt 𝑝 (extracted using BLIP [Li et al. 2022]), we generate a sequence of latent frames 𝑍. The pairwise DIFT [Tang et al. 2023] features quantify framewise similarity, enabling a rapid assessment of motion changes. As illustrated in Fig. 4, a heatmap visualizes temporal variations, where significant object movements or new appearances are represented as bright regions. The heatmap between frames 𝐼𝑖 and 𝐼𝑗 is computed as:

- 3.1 Problem Setup Task description. Given a pair of start and end single view images

𝐼𝑠 and 𝐼𝑒 ∈ R𝐻×𝑊 ×3 representing a dynamic scene possibly having a complex and large motion, our task is to generate a 4D interpolated scene that can be observed at any point of time or view.

Our framework. Our objective is to generate smooth motion while minimizing 3D artifacts in novel views. To achieve this, we introduce gradual local displacements and insert frames in regions with complex motion to prevent abrupt transitions. First, keyframes are adaptively generated by analyzing motion differences in feature space, segmenting fragments with simple motion (Sec.3.2). These fragments are then individually lifted to 4D space using their respective motion (Sec.3.3). Finally, local deformations are integrated into a globally smooth 4D motion with regularization (Sec. 3.4).

- 3.2 Temporal Fragment Hierarchy

𝐻𝑖,𝑗𝑝 = CS(𝑓𝑖𝑝, 𝑓𝑗𝑞∗), where 𝑞∗ = argmaxCS(𝑓𝑖𝑝, 𝑓𝑗𝑞) where CS(.) represents cosine similarity, and 𝑝,𝑞 denote tokens of DIFT feature 𝑓 . A frame is marked as a keyframe if the mean heatmap value of 𝐻𝑖,𝑗 between frame pairs 𝐼𝑖,𝐼𝑗 falls below a predefined threshold. To sample the best keyframe in terms of visual fidelity we further assess its consistency with the initial inputs using FID metric. The keyframe latent 𝑧𝑚 at timestep 𝑚 is selected based on the highest FID against the input states to remain faithful to inputs. For instance, in Fig 4, the chosen keyframe exhibits the highest fidelity to the input states of the eagle. Once identified, the keyframe divides the motion trajectory into two segments: 𝑧𝑠,𝑧𝑚 and 𝑧𝑚,𝑧𝑒. The interpolation model 𝜓(.) then utilizes these fragments iteratively in a "divide-and-conquer" fashion, identifying further keyframes until the full video is processed. This hierarchical approach ensures adaptive keyframe density, reducing redundant intermediate frames in low-motion areas while preserving complex motion details. Therefore, the hierarchical keyframe selection is performed recursively based on prior selections:

We propose a method for identifying keyframes in fragments with significant deformations and adaptively expanding them. Large deformations between start and end states induce rapid intermediate motion changes, which hinder 3D deformation learning [Liang et al. 2024a] as shown in Fig. 5. To mitigate this, we partition the motion trajectory into fragments with smoother quasi-static motions, selecting keyframes densely in dynamic regions and sparsely in static regions. This balances training overhead, model size, and performance and enhances temporal consistency.

K = 𝐾(𝑠)(1),𝐾(1)(2),𝐾(2)(3), ...,𝐾(𝑐)(𝑒) (1)

where 𝐾(𝑖)(𝑗) denotes the keyframe between states 𝑖 and 𝑗.

Temporal fragment generation. Having keyframes K, we reuse the video interpolation module 𝜓(.) to perform inebetweening for consecutive keyframes 𝐾(𝑖)(𝑗) and 𝐾(𝑗)(𝑘). Since 𝜓(.) receives latents, we interpolate the latents and decode them using a VAE decoder to insert new RGB frames. Since the consecutive keyframes represent simple quasi-static motions, this interpolation generates smooth fragments with fewer artifacts. We generate 𝑇 such fragments denoted by V𝑖 each having fixed number of frames 𝑓 (e.g., 16) representing the motion between keyframes:

V = {V𝑠(1𝑓 ), V(1𝑓 )(2𝑓 ), ..., V((𝑐−1)𝑓 )𝑒}, (2) where V𝑠(1𝑓 ) = D(𝜓(𝑧(𝑠)(1),𝑧(1)(2))); D is VAE decoder.

- 3.3 Modelling Intra-Fragment Geometry

We lift individual video fragments to 4D by generating multi-view videos of the object. Existing video-to-4D methods [Ren et al. 2023; Wu et al. 2024] use multi-view diffusion models [Liu et al. 2023] to synthesize multi-view videos by independently processing each frame. However, this approach ensures cross-view consistency but leads to temporally inconsistent geometry Fig 7. Moreover, due to sole reliance on multi-view video supervision, Gaussian splatting often produces flickering and texture variations [Luiten et al. 2024] due to its high degrees of freedom per point and lack of motion constraints. We address these issues by generating temporally consistent multi-view videos and regularizing motion with rigid constraints within each fragment.

Learning canonical 3D. We start by estimating a canonical Gaussian representation and then add motion to it from the multi-view videos. For each temporal fragment V𝑖, we designate the keyframe 𝐾(𝑖)(𝑗) as the canonical reference and reconstruct its 3D structure via multi-view synthesis. Specifically, we employ the multiview diffusion model Era3D [Li et al. 2024a] to generate multi-view images from 𝐾(𝑖)(𝑗), followed by 3DGS [Kerbl et al. 2023a] for coarse geometry reconstruction. As each fragment is processed independently, parallel execution reduces computation time. The resulting coarse geometry provides an effective initialization for learning texture and geometry across the remaining temporal frames.

Dynamic 3D fragment generation. After learning 3D static Gaussians, we leverage motion priors from the video fragment to transform them into dynamic Gaussians. Since single-view videos cannot provide diverse observations of the scene from different viewpoints, we use multi-view videos. To promote temporal consistency, rather than generating multiple-view of the frames independently at each timestep, we propagate the self-attention features of the multi-view diffusion model [Li et al. 2024a] from the canonical frame across the entire frames of the fragment as follows:

𝑧𝑡 ← 𝛾.𝑧𝑐 + (1 −𝛾).𝑧𝑡, Q = W𝑞.𝑧𝑡, K = W𝑘.𝑧𝑡, V = W𝑣.𝑧𝑡, Attention(Q, K, V) = 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥(

QK𝑇 √𝑑𝑘

.V).

where𝑧𝑐 is the multi-view latent of the canonical frame,𝑡 is timestep of the video fragment, 𝛾 is the blending weight and 𝑑𝑘 is the key dimension. With quasi-static motions in each video fragment, the

[Figure 25]

Fig. 5. Effect of inbetweening on geometry. When the input states are significantly different, the 3D deformation module undergoes large movements (fast motion) leading to artifacts in novel views, whereas generating intermediate frames between the states (slow motion) enhances the geometry using smaller deformations.

generated multi-view videos have minimal variation in viewpoints, making it easier for the model to capture accurate and consistent geometry (Fig. 7). With the synthesized multi-view videos of the dynamic object, we optimize a 3D deformation field (denoted by Δ𝚽𝑖) to enable free-viewpoint rendering. We chose Hexplanes [Cao and Johnson 2023] as our deformation field due to modeling efficiency. The deformation field predicts each Gaussian’s geometric offsets at a given timestamp relative to the mean canonical state (keyframe). For each timestamp𝜏 of video and 3D Gaussian 𝑝, Hexplanes predict displacement, rotation, and scaling for the 3D gaussian points.

Optimization objective. To respect the driving video and optimize the deformation field, we fix the camera to a view and minimize the Mean Squared Error (MSE) between the rendered image and each video frame:

∑︁T

1 T

||𝑓 (𝜙(𝑆,𝜏),𝑜Ref) − 𝐼Ref𝜏 ||22, (3)

LRef =

𝜏=1

where 𝐼Ref𝜏 is the 𝜏-th frame, 𝑜Ref is the reference viewpoint, and 𝑓 is the rendering function. Dynamic Gaussians tend to move freely across regions of similar color [Luiten et al. 2024] without constraints, causing flickering and floating artifacts that degrade 4D motion realism. As motion within each fragment is minimal, we enforce rigid assumptions on point movements relative to the canonical state by regularization:

L𝑟𝑖𝑔𝑖𝑑 = ||𝑑(𝜇𝑖𝑐, 𝜇𝑐𝑗) − 𝑑(𝜇𝑖𝜏, 𝜇𝜏𝑗 )||1 (4) where 𝑑(𝑥,𝑦) = ||𝑥 − 𝑦||2 is the distance function, and 𝜇 denotes the Gaussian center of neighboring clusters N. 𝜇𝑐 and 𝜇𝜏 represent Gaussian centers in the canonical and arbitrary timestep frames, respectively. This regularization permits non-rigid deformations (like bending) while minimizing local rigid distortions. In addition to this, we also use a random view at each timestep and apply foreground mask loss L𝑚𝑎𝑠𝑘, resulting in a total training objective:

L = 𝜆1L𝑅𝑒𝑓 + 𝜆2L𝑚𝑎𝑠𝑘 + 𝜆3L𝑟𝑖𝑔𝑖𝑑, (5)

(a) Temporal Fragment Hierarchy

- (b) Modelling Intra-Fragment Geometry

- (c) Cascaded 3D Motion Aggregration

3DGS

Generating 3D Keyframe

Model Input

Start State End State

MV-3D Reconstruction

Multi-View Generator

Text Prompt

RGm

"A robot is dancing with its arm raised"

Generated Views

Static 3D Canonical

2D Keyframe(Canonical)

Views

Dynamic 3D Fragment

|<br><br>||
|---|---|
|<br><br><br><br><br><br><br><br>T|<br><br>|

Optional

D-3DGS

Hierarchial Keyframe Generator

m

MV Gen

T

Keyframe

Static 3D

Video Fragment

MV Consistent Fragment Dynamic 3D

Keyframe

Temporal Fragment Generator

Quasi-Static Motion

Locally Rigid 3D Motion

3D Motion Fragment

Defor. Field Concat

Defor. Field Concat

Generated 4D

t=0 t=64

Frame Overlap

Generated Keyframe

Fast/More Motion = Large Deformation

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Artefacts

No Intermediate State

Slow/Less Motion = Small Deformation

[Figure 85]

[Figure 86]

[Figure 87]

No Artefacts

[Figure 88]

[Figure 89]

[Figure 90]

Generated Intermediate State

Timestep = T2

Timestep = T1

SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong , Sauradip Nag, Daniel Cohen-Or, Hao Zhang, and Ali Mahdavi-Amiri

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Timestep = T1 Timestep = T2

Timestep = T3

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

- (a) Frame-by-Frame Multi-View 3D reconstruction

[Figure 104]

- (b) Rigid Constrainted Multi-View 3D reconstruction

Fig. 6. Trajectory smoothing of fragments leads to correction of Gaussians and help render better novel views.

[Figure 105]

where 𝜆1, 𝜆2 and 𝜆3 are loss weights which are set as 0.4, 0.3, 0.3 respectively. In training, for each fragment V𝑖, we first use L to supervise the static 3D Gaussian, then train the dynamic 4D Gaussian with all reference frames.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

- 3.4 Cascaded 3D Motion Aggregation

Since we learn each fragment’s deformation independently, the entire video may lack consistency over the global geometry and motion. The overall 3D deformation field Δ consists of mini-deformations per fragment:

(c) Temporaly Conistent and Rigid Constrained Multi-View 3D reconstruction

Fig. 7. Effect of Inter-Fragment Consistency. Without using any consistency or regularization, blurriness and oversaturated artifacts are produced. Rigid consistency improves the structure and when combined with temporalaware multi-view generation, better geometry and texture are obtained.

Δ = [𝚫𝚽1, 𝚫𝚽2, ..., 𝚫𝚽𝑲 ], (6)

where each 𝚫𝚽𝒊 is optimized separately. To achieve smooth, flickerfree motion in novel views, we need to merge these fragment deformations.

where 𝜃 is the angle between consecutive rotations, and 𝛼 is the EMA decay factor (set as 0.6). The process is repeated iteratively until all disoriented Gaussians are corrected, yielding stable and flicker-free 3D reconstructions.

Motion merging. With overlapping frames between adjacent fragments, we linearly interpolate the deformation fields for these overlaps. Specifically, we define the interpolated deformation field as:

𝚫merge𝚽𝑖𝑗 = 𝜆𝚫𝚽𝑖 + (1 − 𝜆)𝚫∗𝚽𝑗, (7)

4 EXPERIMENTS AND RESULTS

where𝜆 = 0.5, and only 𝚫∗𝚽𝑗 is learnable. With intra-frame motions already smooth, we freeze 𝚫𝚽𝑖 and 𝚫𝚽𝑗, optimizing only 𝚫merge𝚽𝑖𝑗 for a few iterations (e.g., 1,000) at a low learning rate using Eq.5 to ensure smooth inter-frame motion between fragments. Starting with 𝚫𝚽1, we progressively merge all deformation fields in a cumulative fashion, resulting in a smooth and globally coherent 3D motionwithout abrupt transitions or flickers.

Dataset. We evaluate our method on 4D sequences with large object motions, defined when multiple object parts move. We introduce the I4D-15 benchmark, comprising 15 articulated objects across categories like Vehicles, Flowers, Humans, Animals, and Daily Life scenes. The dataset includes 64-frame sequences at 16 fps from Objaverse1.0 [Deitke et al. 2023] and SketchFab [Spiess et al. 2024], rendered from 5 evenly spaced views at 0◦ elevation. We select the first and last frames of the front view as input states and reserve the remaining video for evaluation. The camera radius is 1.5, and the FOV is 49.1◦, consistent with [Jiang et al. 2023]. Motion filtering [Li et al. 2024b] ensures large motion selection. Evaluation uses appearance metrics (LPIPS, FVD)[Ren et al. 2023] and geometry metrics (SI-CD, CD)[Peng et al. 2024].In addition to this, we have also evaluated our method along with the baselines on Consistent-4D [Jiang et al. 2023] dataset for more extensive validation.

### Cascadedtrajectorysmoothing.Despite smoothtransitionsacross

fragments, minor 3D inconsistencies may persist (see 6), often due to disoriented Gaussians causing blurry or over-reconstructed artifacts [Kerbl et al. 2023b]. Since the 3D Gaussian geometry is controlled by the covariance matrix (i.e., rotation 𝑞 and scaling 𝑠), we regularize 𝑞 and 𝑠 over a fixed window with neighboring frame constraints. We adopt off-the-shelf tracking (e.g., CoTracker [Karaev et al. 2023]) and depth models (e.g., DepthAnything [Yang et al. 2024a]) in a sliding window manner for post-processing. Given a window 𝑤, we estimate depth D and trajectory T on the video V (Eq. 2) and lift 𝑁 randomly selected trajectories to 3D using camera intrinsics. Points visible for at least 80% frames are smoothed with Exponential Moving Average (EMA) on rotation and scale:

Baselines. As we introduce a novel task, no existing method can be directly applied to our setting. Thus, we establish the following baselines for quantitative comparison: For 4D baseline generation, we first generate videos from two frames and subsequently lift it to 4D using a Video-to-4D approach. (a) Baseline-I employs a image based interpolation DreamMover [Shen et al. 2024] to generate keyframes followed by [Xing et al. 2024] to generate video fragments and then use a latest Video-to-4D method [Ren et al. 2025] (b) Baseline-II

𝑠𝑖𝑛((1 − 𝛼).𝜃) 𝑠𝑖𝑛𝜃

𝑠𝑖𝑛(𝛼.𝜃) 𝑠𝑖𝑛𝜃

𝑞𝑡 =

𝑞𝑡 +

𝑞𝑡−1, 𝑠𝑡 = 𝛼𝑠𝑡 + (1 − 𝛼)𝑠𝑡−1,

(8)

[Figure 113]

Fig. 8. Qualitative comparison of our approach with the baselines on challenging synthetic and real-world scenes (e.g, water splash , weightlifting).

utilizes latest video interpolation approach [Xing et al. 2024] along with our temporal fragment generation module for 2D interpolation and a recent Video-to-4D method [Xie et al. 2024]. (c) Baseline-III is an improved version of Baseline I in two ways: first, it incorporates the output of our Temporal Fragment Hierarchy Module (Sec. 3.2) for video generation and second, we replace L4GM [Ren et al. 2025] with a more recent video-to-4D model, SV4D [Xie et al. 2024] to lift the video to 4D. Additionally, we evaluate a single-image-to-4D task on our dataset, with further analysis provided in the supplementary. Baseline results are obtained using official GitHub implementations. Quantitative comparisons. We quantitatively evaluate our approach on our I4D-15 benchmark. Two images from one view are used as input and 4 videos (each 64 frames) from other viewpoints and their corresponding timsetep point clouds are used for evaluation. As shown in Tab. 1, our method outperforms the baseline across all metrics in appearance and geometry. To compare our method on the Consistent4D benchmark, we use the first and the last frame of the input test video and use the novel view videos of the test set for evaluation. Similar observations can be drawn from Tab. 2, where our method is much superior than the baselines which is expected as the dataset contains simple motions. This shows the robustness of our model design and effectiveness of our method in handling complex motions in 4D by dividing it into quasi-static temporal fragments.

Table 1. Quantitative Analysis on proposed I4D-15 Dataset.

|Method|Appearance CLIP LPIPS FVD<br><br>|Geometry SI-CD CD|
|---|---|---|
| |↑ ↓ ↓|↓ ↓|
|Baseline-I<br><br>Baseline-II<br><br>Baseline-III<br><br><br>|0.81 0.143 992.23 0.84 0.136 729.32 0.83 0.138 811.26|33.58 0.76<br><br>31.79 0.73<br><br>32.46 0.74<br>|
|Ours|0.91 0.103 679.23<br><br>|22.67 0.59<br><br>|

Table 2. Quantitative Analysis on proposed Consistent-4D Dataset.

|Method<br><br>|LPIPS ↓ CLIP-S ↑ FVD ↓|
|---|---|
|Baseline-I<br><br>Baseline-II<br><br>Baseline-III<br><br><br>|0.18 0.84 925.46 0.14 0.87 810.24 0.16 0.86 878.13|
|Ours|0.13 0.90 741.88<br><br>|

Qualitative comparisons. Fig. 8 and Fig 11 provides some qualitative comparisons with the baseline on both real-world scenes and synthetic objects. It is apparent that our method produces fewer artifacts in the novel views and also preserve the geometry better in complex motions. In addition to this, we also present a gallery visualization of scenes in Fig 10 for different motions and object categories from our proposed I4D-15 benchmark and additional real-world scenes. More videos are provided in supplementary.

Ablation Study. This study evaluates the contribution of key components in our method on the I4D-15 benchmark. As shown in

- Table 3. Ablation on the impact of temporal fragment generation.

|# Fragments<br><br>|LPIPS ↓ FVD ↓|SI-CD ↓ CD ↓<br><br>|Time ↓|
|---|---|---|---|
|w/o TFH<br><br>|0.137 922.16|32.56 0.74<br><br>|5 mins|
|2 8|0.124 898.23 0.101 680.11<br><br>|29.68 0.70 22.59 0.60<br><br>|8 mins 36 mins|
|4 (Ours)|0.103 679.23<br><br>|22.67 0.59<br><br>|17 mins|

- Table 4. Ablation on the components of motion aggregation.

|Motion Merging<br><br>Trajectory Smoothing<br><br>|Appearance LPIPS FVD|Geometry SI-CD CD|
|---|---|---|
| |↓ ↓|↓ ↓|
|✓ ✓<br><br>|0.103 679.23<br><br>|22.67 0.59<br><br>|
|✓ ✗ ✗ ✗<br><br>|0.116 783.28 0.137 922.16|25.40 0.71 32.56 0.74<br><br>|

Tab 3, using four segments enables our model to decompose complex motion into finer details using our Temporal Fragment Hierarchy (TFH) module, achieving the best cost-performance balance. Additionally, we visually analyze the effect of Intra-Fragment consistency (Sec. 3.3) in Fig. 7, revealing that mv-consistency significantly enhances novel view synthesis, while rigid consistency mitigates deformation artifacts. Tab. 4 further highlights the impact of 3D motion aggregation. The combination of merging and smoothing improves both appearance and geometry metrics, except for a slight decline in FVD when trajectory smoothing is applied. Moreover, we benchmark runtime against all baselines, as shown in Tab. 5, demonstrating that our approach outperforms the fastest baseline (B-I) by 25% on an NVIDIA A100 GPU. Additional ablations are provided in the supplementary material.

User study. A user study was conducted, as human judgment is most effective for assessing 3D generation and motion quality. Since no standard metrics exist for evaluating 4D performance holistically, human judgment becomes a viable option. The study involved 20 graduate students with a background in computer science, who were shown both the start/end state images and the output rendered videos from all the methods. Each object motion was presented from four canonical views (front, back, left, and right) and the participants were asked to rank four methods (1 = best, 4 = worst) based on how well the start/end state images align with the generated 3D geometry and how consistent is the motion (reduced flicker). In case of ties in motion consistency, 3D generation quality was prioritized. As shown in Tab. 6, our method achieves the best average rank (1.46), demonstrating the superiority of our model for overall 4D generation quality.

Table 5. Ablation on runtime

|Methods|FVD ↓ Time↓<br><br>|
|---|---|
|B-I<br><br>B-II<br><br>B-III<br><br><br>|992 2.25 hr 729 1.10 hr 811 1.20 hr|
|Ours|679 35 min<br><br>|

Table 6. User study

|Methods|Gen. Quality ↓<br><br>|
|---|---|
|B-I<br><br>B-II<br><br>B-III<br><br><br>|3.14 2.35 2.85|
|Ours|1.46|

Application: Customized 4D Motion. In contrast to most existing 4D generation methods [Ren et al. 2023; Zeng et al. 2024] that depend on SDS [Poole et al. 2022], our approach improves controllability and motion diversity. While BLIP [Li et al. 2022] is used by default to extract motion prompts, users can input custom prompts

[Figure 114]

Fig. 9. Controllable Motions. In-2-4D allows generation of diverse motions for the same start and end states

to generate 4D motions for the same initial and final states. As shown in Fig. 9, both jumping and walking motions of a dog are synthesized under identical start and end conditions. Despite motion complexity, our bottom-up 3D optimization ensures artifact-free novel view generation.

5 CONCLUSION, LIMITATIONS, FUTURE WORK

We introduce the novel task of generative 4D inbetweening from two single view images at distinct motion states. To address this challenging task, we leverage the capabilities of foundational video diffusion models to extract motion in between the states. We identify complex and large motions and divide them into fragments with simpler and smoother motions through a divide and conquer approach. Using multi-view priors, we lift the object at different states to 3D and merge these simple 3D motions in a bottom-up fashion with smoothness constraints into a flicker-free 4D motion. Although our work is able to outperform baselines but it is still a strong baseline on this challenging task and paves the way for further exploration and advancement.

Our method has some limitations. First, our method produces un-natural deformations when the in-between motion is extreme. Since the resulting videos are used to lift the object motion to the 3D space, the subtle movements may not look natural in 4D space. A promising direction for future work would be to extend this approach to incorporate specific motion trajectories or other 2D or 3D conditional signals in 4D motion generation to provide more realistic dynamism. Additionally, the 3D and 2D components do not currently interact in a way that allows mutual correction. Finally, since our pipeline relies on an image-to-3D module, we also inherit its limitations.If the module fails to accurately reconstruct an object, the resulting 4D output will exhibit corresponding artifacts.

Acknowledgement We thank all the anonymous reviewers for their insightful comments and constructive feedback. This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant.

REFERENCES

Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 2024. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7996–8006.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023a. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023b. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.

- Ang Cao and Justin Johnson. 2023. HexPlane: A Fast Representation for Dynamic Scenes. CVPR (2023).

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. 2023. VideoCrafter1: Open Diffusion Models for High-Quality Video Generation. arXiv:2310.19512 [cs.CV] https://arxiv.org/abs/2310.19512

Setareh Cohan, Guy Tevet, Daniele Reda, Xue Bin Peng, and Michiel van de Panne.

2024. Flexible Motion In-betweening with Diffusion Models. In ACM SIGGRAPH.

Duolikun Danier, Fan Zhang, and David Bull. 2024. Ldmvfi: Video frame interpolation with latent diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 1472–1480.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13142–13153.

Haiwen Feng, Zheng Ding, Zhihao Xia, Simon Niklaus, Victoria Abrevaya, Michael J Black, and Xuaner Zhang. 2024. Explorative Inbetweening of Time and Space. In 2024.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2024b. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. In ICLR.

Zhiyang Guo, Wen gang Zhou, Li Li, Min Wang, and Houqiang Li. 2024a. Motionaware 3D Gaussian Splatting for Efficient Dynamic Scene Reconstruction. ArXiv abs/2403.11447 (2024). https://api.semanticscholar.org/CorpusID:268512916

Yanqin Jiang, Chaohui Yu, Chenjie Cao, Fan Wang, Weiming Hu, and Jin Gao. 2024. Animate3d: Animating any 3d model with multi-view video diffusion. Advances in Neural Information Processing Systems 37 (2024), 125879–125906.

Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. 2023. Consistent4d: Consistent 360 {\deg} dynamic object generation from monocular video. arXiv preprint arXiv:2311.02848 (2023).

Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. 2023. CoTracker: It is Better to Track Together. arXiv:2307.07635

(2023).

- Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023a. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics 42, 4 (July 2023). https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023b. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (ToG) 42, 4 (2023), 1–14.

Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023. Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators. arXiv preprint arXiv:2303.13439 (2023).

- Anh Cat Le Ngo and Raphael C.-W. Phan. 2019. Seeing the Invisible: Survey of Video Motion Magnification and Small Motion Analysis. ACM Comput. Surv. 52, 6 (2019).

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning. PMLR, 12888–12900.

Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang Zhang, Wei Xue, Wenhan Luo, et al. 2024a. Era3d: high-resolution multiview diffusion using efficient row-wise attention. Advances in Neural Information Processing Systems 37 (2024), 55975–56000.

Ruining Li, Chuanxia Zheng, Christian Rupprecht, and Andrea Vedaldi. 2024b. Puppetmaster: Scaling interactive video generation as a motion prior for part-level dynamics. arXiv preprint arXiv:2408.04631 (2024).

Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. 2024b. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645 (2024).

Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-Phuoc, Douglas Lanman, James Tompkin, and Lei Xiao. 2023. GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis. ArXiv abs/2312.11458 (2023). https:

//api.semanticscholar.org/CorpusID:266359262

Yiqing Liang, Mikhail Okunev, Mikaela Angelina Uy, Runfeng Li, Leonidas Guibas, James Tompkin, and Adam W Harley. 2024a. Monocular Dynamic Gaussian Splatting is Fast and Brittle but Smooth Motion Helps. arXiv preprint arXiv:2412.04457 (2024).

Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. 2024. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8576–8588.

Ce Liu, Antonio Torralba, William T. Freeman, Frédo Durand, and Edward H. Adelson.

2005. Motion magnification. TOG 24, 3 (2005).

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9298–9309.

Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. 2024. Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis. In 3DV.

Sungheon Park, Minjung Son, Seokhwan Jang, Young Chun Ahn, Ji-Yeon Kim, and Nahyup Kang. 2023. Temporal interpolation is all you need for dynamic neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4212–4221.

Shichong Peng, Yanshu Zhang, and Ke Li. 2024. PAPR in Motion: Seamless Point-level 3D Scene Interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21007–21016.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022). Jia Qin, Youyi Zheng, and Kun Zhou. 2022. Motion In-Betweening via Two-Stage Transformers. ACM Transactions on Graphics 41 (2022). Issue 6.

Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, and Brian Curless. 2022. Film: Frame interpolation for large motion. In European Conference on Computer Vision. Springer, 250–266.

Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. 2023. DreamGaussian4D: Generative 4D Gaussian Splatting. arXiv preprint arXiv:2312.17142 (2023).

Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. 2025. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems 37 (2025), 56828–56858.

Liao Shen, Tianqi Liu, Huiqiang Sun, Xinyi Ye, Baopu Li, Jianming Zhang, and Zhiguo Cao. 2024. Dreammover: Leveraging the prior of diffusion models for image interpolation with large motion. In European Conference on Computer Vision. Springer, 336–353.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023. Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model. arXiv preprint arXiv:2310.15110 (2023).

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. 2022. Make-A-Video: Text-to-Video Generation without Text-Video Data. arXiv:2209.14792 [cs.CV] https://arxiv.org/abs/2209.14792

Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. 2023. Textto-4d dynamic scene generation. arXiv preprint arXiv:2301.11280 (2023).

Florian Spiess, Raphael Waltenspül, and Heiko Schuldt. 2024. The sketchfab 3d creative commons collection (s3d3c). arXiv preprint arXiv:2407.17205 (2024).

Qi Sun, Zhiyang Guo, Ziyu Wan, Jing Nathan Yan, Shengming Yin, Wengang Zhou, Jing Liao, and Houqiang Li. 2024. Eg4d: Explicit generation of 4d object without score distillation. arXiv preprint arXiv:2405.18132 (2024).

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. 2023. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems 36 (2023), 1363–1389.

Peng Wang and Yichun Shi. 2023. ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation. arXiv:2312.02201 [cs.CV] https://arxiv.org/abs/2312.02201

Wen Wang, Qiuyu Wang, Kecheng Zheng, Hao Ouyang, Zhekai Chen, Biao Gong, Hao Chen, Yujun Shen, and Chunhua Shen. 2025. Framer: Interactive frame interpolation. In ICLR.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Wang Xinggang. 2023b. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. arXiv preprint arXiv:2310.08528 (2023).

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023a. Tune-A-Video: OneShot Tuning of Image Diffusion Models for Text-to-Video Generation. In ICCV. https://arxiv.org/abs/2212.11565

Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. 2025. Cat4d: Create anything in 4d with multi-view video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 26057–26068.

Zijie Wu, Chaohui Yu, Yanqin Jiang, Chenjie Cao, Fan Wang, and Xiang Bai. 2024. SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer. arXiv preprint

arXiv:2404.03736 (2024).

Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. 2024. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470 (2024).

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. 2023. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190 (2023).

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. 2024. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision. Springer, 399–417.

Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024a. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10371–10381.

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. 2023. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. arXiv preprint arXiv:2309.13101 (2023).

Zeyu Yang, Zijie Pan, Chun Gu, and Li Zhang. 2024b. Diffusion2: Dynamic 3d content generation via score composition of video and multi-view diffusion models. arXiv

preprint arXiv:2404.02148 (2024).

Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 2023. 4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency. arXiv preprint arXiv:2312.17225 (2023).

Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. 2024. Stag4d: Spatial-temporal anchored generative 4d gaussians. arXiv preprint arXiv:2403.14939 (2024).

Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao.

2024. 4diffusion: Multi-view video diffusion model for 4d generation. Advances in Neural Information Processing Systems 37 (2024), 15272–15295.

Zehan Zheng, Danni Wu, Ruisi Lu, Fan Lu, Guang Chen, and Changjun Jiang. 2023. Neuralpci: Spatio-temporal neural field for 3d point cloud multi-frame non-linear interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 909–918.

Lei Zhong, Chuan Guo, Yiming Xie, Jiawei Wang, and Changjian Li. 2025. Sketch2Anim: Towards Transferring Sketch Storyboards into 3D Animation. In ACM SIGGRAPH. Yi Zhou, Jingwan Lu, Connelly Barnes, Jimei Yang, Sitao Xiang, and Hao li.

2020. Generative Tweening: Long-term Inbetweening of 3D Human Motions. arXiv:2005.08891 [cs.CV] https://arxiv.org/abs/2005.08891

Novel Views Novel Views

[Figure 115]

[Figure 116]

Input

Input

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Text

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Time

[Figure 130]

Input

[Figure 131]

[Figure 132]

Input

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

Input Input

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

Time

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Input Input

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

Input

Input

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

- Fig. 10. Results Gallery. We illustrate our method’s performance on a wide variety of objects including furnitures, animals, humans, vehicles and daily life objects. Our model produces consistent textures and preserves the geometry in novel views without any direct 3D supervision.

Input Images Baseline-I Baseline-II Baseline-III Ours Input Images Baseline-I Baseline-II Baseline-III Ours

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

NovelViewsNovelViewsNovelViewsNovelViews

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

- Fig. 11. Qualitative Comparison on Consistent4D benchmark. Having only the first and last frames of a motion, we are able to generate moving 3D objects that can be seen at different views. Objects seen from different view directions are still plausible although no direct supervision signal is available.

