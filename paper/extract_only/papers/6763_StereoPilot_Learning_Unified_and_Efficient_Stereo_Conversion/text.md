## StereoPilot: Learning Unified and Efficient Stereo Conversion via Generative Priors

Guibao Shen1,3*† Yihua Du1* Wenhang Ge1,3*† Jing He1 Chirui Chang3 Donghao Zhou4 Zhen Yang1 Luozhou Wang1 Xin Tao3 Ying-Cong Chen1,2 ‡

1 HKUST(GZ) 2 HKUST 3 Kling Team, Kuaishou Technology 4 CUHK

https://github.com/KlingTeam/StereoPilot

# arXiv:2512.16915v1[cs.CV]18Dec2025

#### Abstract

GT-Left / Ours-Right

GT-Left / GT-Right

GT-Left / DWI-Right

Mirrored Object

Mirror

| |[Figure 1]|[Figure 2]|
|---|---|---|

|[Figure 3]<br><br>[Figure 4]| | |
|---|---|---|

| |[Figure 5]<br><br>[Figure 6]| |
|---|---|---|

###### LDepthAmbiguity!

Object

Reflection Depth

The rapid growth of stereoscopic displays, including VR headsets and 3D cinemas, has led to increasing demand for high-quality stereo video content. However, producing 3D videos remains costly and complex, while automatic Monocular-to-Stereo conversion is hindered by the limitations of the multi-stage “Depth-Warp-Inpaint” (DWI) pipeline. This paradigm suffers from error propagation, depth ambiguity, and format inconsistency between parallel and converged stereo configurations. To address these challenges, we introduce UniStereo, the first large-scale unified dataset for stereo video conversion, covering both stereo formats to enable fair benchmarking and robust model training. Building upon this dataset, we propose StereoPilot, an efficient feed-forward model that directly synthesizes the target view without relying on explicit depth maps or iterative diffusion sampling. Equipped with a learnable domain switcher and a cycle consistency loss, StereoPilot adapts seamlessly to different stereo formats and achieves improved consistency. Extensive experiments demonstrate that StereoPilot significantly outperforms state-of-the-art methods in both visual fidelity and computational efficiency. Project page: StereoPilot Page.

Surface Depth

Left)

DepthEstimate(

[Figure 7]

Viewpoint

RAFT (GT-Left, DWI-Right)

RAFT (GT-Left, Ours-Right)

RAFT (GT-Left, GT-Right)

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Inverse

[Figure 12]

❌

Depth of Left View Disparity of GT Pair

Disparity of DWI

Disparity of Ours

Figure 1. Depth Ambiguity Issue. As shown in the legend in the upper left corner of the figure, when there are specular reflections, there will be two depths at the mirror position: the depth of the mirror surface dS and the depth of the object’s reflection dR. In the real physical world, these two points are warped separately according to their respective disentangled depths. However, depth estimation algorithms cannot predict multiple depths at the same position. Therefore, the inverse relationship between depth and disparity breaks down. This will cause Depth-Warp-Inpaint (DWI) type methods to predict results with incorrect disparity.

quality stereo content remains a formidable challenge.

Current immersive stereo video creation is constrained by two primary factors. First, native stereo video production necessitates specialized and expensive stereoscopic camera equipment coupled with complex post-processing workflows. Second, converting the vast repository of existing 2D video into a stereo format is a non-trivial task. For instance, the renowned manual conversion of the film Titanic required a team of 300 engineers, took 60 weeks to complete, and resulted in a cost of 18 million dollars. The prohibitive costs and labor associated with these methods underscore the urgent need for robust and efficient automatic stereo video conversion techniques.

#### 1. Introduction

The rapidly improving stereoscopic displays, including virtual reality (VR) headsets, smart glasses, and 3D cinemas, have driven a significant and growing demand for immersive stereo video content. This technology offers a more immersive user experience compared to traditional 2D media, with widespread applications in the film, gaming, and education industries. Despite its potential, generating high-

Recent advancements in video generation [7, 10, 42, 52] and depth estimation [21, 28, 50, 51] have propelled significant progress in automated stereo video conversion. A predominant category of these approaches [12, 34, 38, 55, 59] adopts a multi-stage “Depth-Warp-Inpaint” (DWI) pipeline.

*Equal contribution †This work was conducted during the author’s internship at Kling. ‡Corresponding author

[Figure 13]

This involves first estimating depth from the monocular video, then warping the image based on this depth map to create a second viewpoint, and finally inpainting any resulting occlusions. However, this paradigm suffers from a critical limitation in its sequential architecture: it creates a strong dependency on the initial depth estimation, where inaccuracies from the depth predictor propagate and compound through the pipeline, leading to substantial degradation in the final synthesized view.

More fundamentally, the depth-based approach suffers from inherent conceptual limitations. First, the depth-warp mechanism cannot adequately resolve scenes with depth ambiguity. As demonstrated in Figure 1, in scenarios involving reflective surfaces such as mirrors, a single source pixel may correspond to multiple depth values—both the depth of the reflective surface itself and the depth of the reflected object. A conventional depth-warp operation cannot model this one-to-many mapping, causing incorrect disparity in the synthesized view. Second, the warping stage relies on a simple inverse relationship between depth and disparity, which is only valid for parallel camera setups, as shown in Figure 2. However, this assumption breaks down for converged (toe-in) configurations—the standard for 3D cinema [46, 49]—where the geometric relationship becomes more complex.

Figure 2. Parallel vs. Converged. In the parallel setup, when both eyes observe the same subject, the projected image points on the left and right views are denoted as XL and XR, and their absolute difference |XL − XR| defines the disparity s. According to geometric relationships derived from similar triangles, b, f, d, and s satisfy an inverse proportionality between disparity and depth when the baseline b and focal length f remain constant. In the converged configuration, a Zero-disparity Projection Plane is present—objects in front of this plane yield positive disparity, while those behind it produce negative disparity.

release UniStereo publicly, hoping to contribute to the development of stereo video conversion.

To address these challenges, we introduce StereoPilot, an efficient model for high-quality stereo video conversion built upon our UniStereo dataset. To overcome error propagation in multi-stage pipelines, StereoPilot adopts an endto-end architecture that directly synthesizes the novel view. Instead of relying on explicit depth maps that struggle with ambiguous scenes, our approach leverages the rich generative priors from a pretrained video diffusion transformer. Rather than depending on format-specific geometric assumptions, our method employs a data-driven approach that adapts to diverse stereo formats. We adopt a feed-forward architecture that performs direct regression to the target view, avoiding the randomness and computational overhead of iterative sampling while ensuring consistency and efficiency. To handle both formats within a unified framework, we integrate a learnable domain switcher and a cycle consistency loss to ensure precise view alignment. Comprehensive quantitative and qualitative experiments demonstrate that StereoPilot significantly outperforms state-of-the-art methods in both visual fidelity and processing speed.

This geometric limitation highlights a broader issue: the diversity of stereo video formats and the lack of unified treatment in existing methods. Based on our analysis of prior methods and a review of 3D film industry literature [14, 37, 41, 46, 49], we identify two primary categories of stereo video with distinct characteristics, shown in Figure 2. The parallel format uses cameras with parallel axes, establishing a simple inverse relationship between disparity and depth [6, 9, 24, 27, 45]. In contrast, the converged format employs cameras in a “toe-in” configuration, creating a zero-disparity plane that is standard for 3D cinema [46, 49]. Several DWI methods assume parallel geometry, yet training data is in converged format [55, 59], creating a fundamental mismatch. Moreover, previous methods [12, 18, 34, 38, 48, 55, 59] were mostly trained on private data belonging to only one format, without explicitly distinguishing or providing unified treatment for both formats. This has led to two critical consequences: first, the applicability of each method remains unclear; second, inappropriate evaluation protocols have emerged, such as benchmarking a model trained on parallel data against one trained on converged data [18, 38], resulting in unfair comparisons. To address this critical gap, we introduce UniStereo, the first large-scale, unified dataset for stereo video conversion that incorporates both parallel and converged stereo data. Our dataset comprises approximately 103,000 stereo video pairs with corresponding textual captions for training, along with a dedicated test set for standardized evaluation. We will

In summary, our contributions are listed as follows:

- • We introduce UniStereo, the first large-scale, unified dataset for stereo video conversion, featuring both parallel and converged formats to enable fair benchmarking and model comparisons.
- • We propose StereoPilot, an efficient feed-forward architecture that leverages a pretrained video diffusion transformer to directly synthesize the novel view. It overcomes the limitations of “Depth-Warp-Inpaint” methods (error propagation, depth ambiguity, and format-specific

assumptions) without iterative denoising overhead, while integrating a domain switcher and cycle consistency loss for robust multi-format processing.

• Extensive experiments show StereoPilot significantly outperforms state-of-the-art methods on our UniStereo benchmark in both visual quality and efficiency.

#### 2. Related Works

##### 2.1. Novel View Synthesis

Novel view synthesis is a long-standing problem in computer vision and graphics. With advancements in 3D representation, such as Nerual Radiance Fields [35] and 3D Gaussian Splatting [29], static multi-view optimizationbased novel view synthesis [4, 5, 17, 31, 40, 57] has made significant progress. However, these methods typically require per-scene optimization with static, calibrated, multiview dense images, which is computationally expensive, lacks generalization, and can only handle static synthesis.

To address the issues of generalization and the need of multi-view dense images, recent methods have explored camera-controlled video generation [1, 2, 15, 19, 22, 36, 39, 44, 56] for single-image generalizable novel view synthesis. Given a target camera trajectory and a reference image, these methods can generate novel views that match the specified camera trajectory. More recent works [3, 20, 47] focus on dynamic novel view synthesis. For example, ReCamMaster [3] is able to reproduce the dynamic scene of an input video at novel camera trajectories. However, training such a video diffusion model requires multi-view calibrated videos, which are difficult to acquire in the real world. Moreover, the camera controllability is still limited.

In this work, we explore the task of stereo video synthesis, which is essentially a sub-problem of novel view synthesis with a fixed target camera. The training data can be more easily constructed using real-world videos with existing algorithms [23]. Since we focus on fixed novel-view synthesis, we do not explicitly require camera pose and can achieve better camera controllability.

##### 2.2. Stereo Video Generation

Monocular-to-stereo video conversion, a specialized task within novel view synthesis, aims to generate a stereoscopic video given only the source viewpoint. Current approaches can be broadly classified into two families: multi-stage pipelines and end-to-end models.

Multi-Stage Depth-Warp-Inpaint Pipelines. The dominant paradigm follows a “depth-warp-inpaint” strategy [12, 34, 38, 55, 59]. These methods first estimate a per-pixel depth map from the monocular input video. This depth map is then used to explicitly warp the source view to the target view, which inevitably creates occlusions and artifacts. Finally, a generative model is employed to inpaint these

Left View Input ReCamMaster Ours GT Right View

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Figure 3. The inherent stochasticity of generative models can cause them to fabricate objects not present in the source view. As this figure illustrates, the right view generated by ReCamMaster erroneously introduces new artifacts, e.g., a car and a man (highlighted in red bounding box), that do not exist in the original input.

missing regions. While popular, this multi-stage approach suffers from two fundamental limitations. First, the entire pipeline’s performance is heavily dependent on the quality of the initial depth estimation. Inaccuracies in the predicted depth cascade through the subsequent stages, leading to severe geometric distortions and artifacts. Second, these methods are unable to handle scenes with depth ambiguity, such as those containing reflections or transparent materials, as shown in Figure 1. In these scenarios, a single pixel location corresponds to multiple depth values (e.g., the glass surface and the object behind it). By relying on a single depth value per pixel, this pipeline fails to model the correct physical optics, resulting in artifacts such as virtual images appearing “baked” onto the mirror surface. Third, warping relies on the assumption that depth and disparity follow an inverse relationship, which does not hold for converged camera configurations.

End-to-End Synthesis Methods. A second category of methods attempts to synthesize the target view directly in an end-to-end (E2E) fashion. Early E2E works, such as Deep3D [48], used CNNs to implicitly learn a soft disparity field for direct image prediction. However, it is trained on single images, failing to guarantee temporal consistency across video frames, and its CNN-based backbone limits scalability. More recently, methods like Eye2Eye [18] have framed the task as conditional video generation using diffusion transformers. By iteratively denoising a noiseconditioned input view, the DiT can generate the other view directly. However, this iterative diffusion process requires dozens of sampling steps, making it computationally expensive and far too slow for practical applications. Moreover, due to the inherent stochasticity of video diffusion models, the diffusion process often produces hallucinated content that is misaligned with the source view, as shown in Figure 3.

Our method, StereoPilot, addresses these limitations. We propose a “diffusion as feed-forward” architecture that predict the target view in one step, efficiently leveraging diffusion priors while fundamentally avoiding the depth ambigu-

[Figure 22]

- Figure 4. UniStereo processing pipeline. We use green icons with numbered steps to depict the Stereo4D pipeline: starting from the raw VR180 videos, we set hfov = 90° and specify the projection resolution to produce the final left- and right-eye monocular videos. Simultaneously, blue icons with numbered steps denote the 3DMovie pipeline: we segment the source films into clips, filter out non-informative segments, convert from side-by-side (SBS) to left/right monocular views, and remove black borders. All resulting videos are captioned using ShareGPT4Video [11].

ity problem. Unlike prior works trained on a single stereo format (converged or parallel), we introduce a learnable domain switcher that makes our model compatible with both, ensuring strong generalization. Finally, a cycle consistency loss is adopted to guide the model to generate the other view that aligns with the input view. StereoPilot thus achieves efficient, accurate, and robust mono-to-stereo conversion.

#### 3. Dataset Construction — UniStereo

As mentioned earlier, to unify the generation of both converged and parallel stereo data, and to address the issues of inappropriate evaluation protocols and unfair comparisons between models trained on parallel versus converged datasets, we propose UniStereo — the first large-scale unified dataset for 2D-to-3D video conversion that integrates both parallel and converged stereo data.

Specifically, UniStereo integrates two complementary components. The Stereo4D [23] subset provides large-scale parallel stereo pairs, as detailed in Section 3.1. To complement this, we construct 3DMovie, a new converged stereo video dataset derived from 3D cinematic productions, addressing the absence of publicly available converged-view data, as detailed in Section 3.2.

##### 3.1. Stereo4D

The Stereo4D subset serves as a large-scale collection of parallel stereo data designed to capture diverse real-world dynamics. It contains over 100K short clips depicting everyday scenes and activities, including both moving and static camera shots, sourced from approximately 7K online videos. Each clip spans 1–6 seconds and provides the camera-to-world (C2W) matrices for the left-eye view. The dataset contains a broad spectrum of indoor and outdoor en-

vironments and includes challenging scenes with reflective and transparent surfaces, ensuring rich visual diversity.

As illustrated in Figure 4, we follow the official preprocessing pipeline of Stereo4D [23] to generate consistent and rectified stereo pairs. Specifically, we convert VR180 videos with a horizontal field of view of 90° into rectified perspective videos at a resolution of 832 × 480. All videos are resampled to 16 fps and trimmed to a fixed length of 81 frames to ensure temporal uniformity across samples. Finally, we employ ShareGPT4Video [11] to automatically generate captions for each video, resulting in approximately 60K high-quality stereo video pairs.

##### 3.2. 3DMovie

To address the lack of publicly available converged-stereo datasets, we constructed a large-scale Converged 3D Movie Dataset (Figure 4). We curated 142 high quality 3D films and manually verified the consistency across both views to ensure valid stereo correspondence.

For temporal standardization, all movies were resampled to 16 fps. We then applied PySceneDetect [8] for shot boundary detection, automatically segmenting each film into coherent stereo clips. Each clip was further divided into uniform segments of 81 frames to facilitate model training. Non-informative content, including opening logos, textual overlays, and ending credits, was removed to maintain visual relevance. After preprocessing, each video was converted from side-by-side (SBS) format into left–right paired monocular streams. To eliminate visual artifacts, we cropped edge black borders and resized all videos to 832 × 480 resolution. Finally, we employed ShareGPT4Video [11] to generate descriptive captions for every stereo pair. The resulting dataset contains approximately 48K high-quality stereo video pairs, providing a reliable foundation for research on converged-stereo understanding and generation.

#### 4. Method

We begin with a brief overview of the conditional video diffusion models in Section 4.1. Section 4.2 describes diffusion as feed-forward model in our method. We then introduce the domain switcher to include both converged and parallel formats in Section 4.3, followed by the designed cycle consistent learning objective in Section 4.4. The overall is illustrated in Figure 5.

##### 4.1. Preliminary

Task Formulation. The goal of monocular-to-stereoscopic video conversion is to synthesize a 3D stereoscopic video from a 2D video. We define this task as a conditional view synthesis problem. Given a ground-truth stereoscopic video pair consisting of a left view Vl and a right view Vr, where Vl,Vr ∈ RN×H×W×3, the objective is to predict one view

Right-to-Left

𝒛

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

𝒛

[Figure 29]

🔥

[Figure 30]

Rec. loss

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | |[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]| |
| | | | | | | |

WanEncoder

Transformer 𝑣 → , 

| | | | | | | |
|---|---|---|---|---|---|---|
|[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]| | | | | | |
| | | | | | | |

Cyc. loss

Left View 𝑉

[Figure 41]

[Figure 42]

𝒛

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

🔥

[Figure 50]

[Figure 51]

Time-Step 𝑡 Switcher 𝑠

[Figure 52]

[Figure 53]

🔥

[Figure 54]

Right View 𝑉

| | | | | | | |
|---|---|---|---|---|---|---|
|[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]| | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]| | |
| | | | | | | |

Transformer 𝑣 → , 

WanEncoder

Rec. loss

𝒛 𝒛

Left-to-Right

- Figure 5. The training framework of the proposed StereoPilot. StereoPilot uses a single-step feed-forward architecture (Diffusion as Feed-Forward) that incorporates a learnable domain switcher s to unify conversion for both parallel and converged stereo formats. The entire model is optimized using a cycle-consistent training strategy, combining reconstruction and cycle-consistency losses to ensure high fidelity and precise geometric alignment. The blue and orange lines represent the Left-to-Right and Right-to-Left reconstruction processes, and the orange dashed line denotes the L → R → L cycle-consistency path.

where tnext < tcurr and η (0 < η ≤ 1) denotes the step size, which is determined by the total number of inference time-steps.

conditioned on the other. Following prior works, we formulate our approach to take the left view video Vl as input and the right view video Vr as output.

Conditional Video Diffusion Models. Recent advanced video generation models [26, 30, 42] are composed of a VAE and a Transformer network. The video diffusion transformers usually adopt Rectified Flow framework. It learns to map samples from a simple prior distribution to a complex data distribution (e.g., videos) by modeling the transport via an Ordinary Differential Equation (ODE). Let

Generative Novel View Video Synthesis. To enable a textto-video generation model to generate a novel view Vtarget conditioning on the input view Vsource, some works try to inject the condition video Vsource to the generation process by concatenating Vsource with the noised zt, either on the channel dimension [18] or the frame dimension [3]. However, applying a generative paradigm to the mono-to-stereo task suffers from two significant limitations. First, a fundamental mismatch exists between stochastic generative models and the task’s highly deterministic nature. Since occlusions between viewpoints typically constitute only a small fraction of the total image, the mapping for most pixels is deterministic. The model’s inherent uncertainty is ill-suited for this, often introducing artifacts in non-occluded regions, as shown in Fig 3. Second, the iterative inference process incurs substantial computational overhead. Generating even a few seconds of video can require tens of minutes, restricting practical application in real-world scenarios.

- z0 ∼ pdata be a sample from the real data distribution and
- z1 ∼ p1(z) = N(0,I) be a Gaussian noise. Rectified Flow defines a simple, straight-line path between these two points:

zt = (1 − t)z0 + tz1, t ∈ [0,1]. (1) The velocity vector field along this path is constant and given by v(zt) = z1 − z0. The transformer network vθ(zt,t,c) is then trained to approximate this vector field, often conditioned on context c (e.g., a text embedding). The model is optimized using the flow-matching objective, which is an L2 regression loss:

0,z1,c ∥vθ(zt,t,c) − (z1 − z0)∥22 ,

LFM = Et∼U(0,1),z

##### 4.2. Diffusion as Feed-Forward Model

(2) where t is sampled uniformly from [0,1]. During inference, a sample is generated by first sampling noise z0 ∼ N(0,I) and a condition c. The corresponding generative ODE is then solved from t = 1 to t = 0 using a numerical solver (e.g., Euler’s method):

As we mentioned above, the stereo conversion task is highly deterministic, which contrasts with standard probabilistic diffusion models. To adapt a diffusion model for this task, we formulate it as a feed-forward network, which has shown success in some deterministic tasks like depth and normal estimation [16, 21]. We achieve this by modifying the standard training procedure: instead of sampling t randomly, we

,t,c), (3)

curr − η · vθ(zt

###### zt

= zt

next

curr

fix the timestep to a small constant t0 = 0.001. This value is chosen since the input state at this near-zero timestep closely approximates our task’s input data distribution, allowing us to effectively harness the model’s generative priors. The process is thus reduced to a single-step deterministic prediction:

###### zr = vθ(zl,t0,c), (4)

where vθ predicts the right view zr from the left view zl and text conditioning c in a single forward pass. This feedforward methodology is computationally efficient (one inference step), and its deterministic nature fits the task. Furthermore, the pretrained knowledge in the video diffusion model also provides rich generation prior to completion of the occluded regions.

##### 4.3. Unified Conversion by Domain Switcher

Converged stereo data Dc and parallel stereo data Dp stereo data are distinct, and applications often require conversion between them. A naive approach trains two separate models, but this has key limitations. First, domain biases—such as parallel public datasets lacking synthetic content—cause models trained only on Dp to fail on those styles. Second, separate training limits the data scale for each model, restricting generalization.

To address this, we propose a unified model that learns from both formats simultaneously. We use a conditional switcher, s ∈ {sc,sp}, to direct the model to predict either the converged or parallel target view. The switcher s is a learnable one-dimensional vector that is added to the time embeddings of the diffusion model, enabling seamless domain switching without mutual interference. Formally, Equation. 4 becomes

vθ(zl,t0,c,sc), if (zl,zr) ∈ Dc vθ(zl,t0,c,sp), if (zl,zr) ∈ Dp

. (5)

zr =

This unified strategy provides stronger generalization. It successfully resolves the challenge of processing synthetic animation styles in the parallel format, a key failure point for separately trained models.

##### 4.4. Cycle Consistent Loss

Maintaining consistency between the left and right views is a critical objective in stereo video conversion. To improve the consistency, we introduce a cycle-consistency training objective, which is designed to enforce alignment between the generated target view and the source view. We leverage the inherent symmetry of the stereo conversion task, which can be formulated as both a left-to-right (L → R) transformation and a right-to-left (R → L) transformation. Our framework employs two distinct models: a generator

that translates the left view to the right, and a generator vr→l,θ

vl→r,θ

l

that translates the right view to the left. Given a ground-truth stereo pair (zl,zr), we first compute the synthesized target views zˆr (from left) and zˆl (from right):

r

(zr,t0,c,s). (6)

zˆr = vl→r,θ

(zl,t0,c,s), zˆl = vr→l,θ

r

l

The total training objective L combines two reconstruction losses with a cycle-consistency loss. The reconstruction losses ensure that the generated images are faithful to their respective ground-truth targets:

Lrecon = ∥zˆr − zr∥22 + ∥zˆl − zl∥22. (7)

The cycle-consistency loss enforces that if we translate a view to the other domain and back again, we should recover the original view. Here, we apply the L → R → L cycle:

Lcycle = ∥zl − zˆlc∥22, (8)

(zˆr,t0,c,s)∥22, (9)

= ∥zl − vr→l,θ

r

where zˆlc denotes the synthesized left view from the synthesized right view zˆr. The final loss function is the sum of these components, which are jointly optimized:

L = Lrecon + λ · Lcycle. (10)

#### 5. Experiments

##### 5.1. Datasets and Evaluation Protocol

Datasets. Our evaluation benchmark consists of two subsets: Stereo4D, which follows a parallel format, and 3DMovie, which follows a converged format. We process all videos into 5-second clips sampled at 16 fps, resulting in a total of 81 frames per clip.

We extract 58,000 clips from the official training split of the Stereo4D dataset. For evaluation, and to ensure a fair comparison with previous methods, we follow prior work and randomly sample 400 video clips from the official Stereo4D test set [23]. For 3DMoive, we collect 44,879 video clips from a total of 126 3D movies for training. This collection is diverse, comprising 48 animated films and 78 live-action films. For the test set, we randomly sample 400 clips from 16 distinct 3D movies that are not included in the training set.

Evaluation Protocol. We evaluate the performance of our method using metrics that assess both conversion quality and processing efficiency. Unlike general visual generation tasks, our stereo conversion task provides corresponding ground-truth data, enabling direct measurement of the fidelity and alignment of the generated results. Following prior work [38, 43, 55], we adopt a comprehensive set of widely used metrics to evaluate the quality of the generated stereo pairs, including PSNR, SSIM, MS-SSIM [58], and

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

###### Converged Parallel

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

[Figure 73]

[Figure 74]

[Figure 75]

| |
|---|
| |

[Figure 76]

[Figure 77]

[Figure 78]

StereoCrafterStereoDiffusionOursSVGLeftViewInputGTRightViewMono2Stereo

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

------

------

------

------

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

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

PSNR 12.89

PSNR 19.17

PSNR 16.59

- PSNR 15.35
- PSNR 16.93

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

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

PSNR 22.11

PSNR 19.32

PSNR 24.29

| |
|---|
| |

| |
|---|
| |

[Figure 109]

[Figure 110]

| |
|---|
| |

| |
|---|
| |

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

PSNR 16.83

PSNR 16.99

PSNR 20.32

PSNR 14.50

| |
|---|
| |

| |
|---|
| |

[Figure 121]

[Figure 122]

| |
|---|
| |

| |
|---|
| |

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

PSNR 26.89

PSNR 20.90

PSNR 25.56

PSNR 22.85

| |
|---|
| |

| |
|---|
| |

[Figure 133]

[Figure 134]

| |
|---|
| |

| |
|---|
| |

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

------

------

------

------

| |
|---|
| |

[Figure 145]

| |
|---|
| |

| |
|---|
| |

| |
|---|
| |

[Figure 146]

[Figure 147]

[Figure 148]

PSNR 33.90

PSNR 23.70

PSNR 37.67

PSNR 31.85

- Figure 6. Qualitative Results. Our method achieves more accurate disparity estimation and preserves finer visual details on both Parallel and Converged data compared with existing baselines.

the perceptual metric LPIPS. Furthermore, to better capture perceptual quality as judged by humans, we also include the SIOU metric, which was introduced in Mono2Stereo [55] and has been shown to exhibit a strong correlation with human perception. To measure computational performance, we report latency. All latency values presented in this paper refer to the time required to process a single 81-frame video clip (5 seconds at 16 fps) on a single GPU.

##### 5.2. Implementation Details

We use Wan2.1-1.3B [42] as our backbone model. During training, we adopt AdamW [32] to optimize the model parameters for approximately 30K iterations with a learning rate of 3 × 10−4. The coefficient λ is set to 0.5 to ensure training stability. Please refer to the Appendix for additional implementation details.

##### 5.3. Main Results

Baselines. We compare StereoPilot with several state-ofthe-art methods, including StereoDiffusion [43], SVG [12],

ReCamMaster [3], and Mono2Stereo [55]. In addition, we also include two influential works from the community, StereoCrafter [59] and M2SVid [38], for comprehensive comparison. We use the default parameter settings provided in each baseline method to ensure a fair comparison.

Quantitative Results. As demonstrated in Table 1, our StereoPilot outperforms all the baselines in all 5 quantitative metrics. Furthermore, our method requires only 11 seconds to complete an 81-frame video’s stereo conversion, showing significant superiority on computational efficiency. Note that we do not report results of M2SVid on 3DMovies, since code of M2SVid has not been released yet. We adopt the performance on 400 test samples on Stereo4D they reported in the paper, aligning with our test setting. For ReCamMaster, we first compare it on the Stereo4D, as Stereo4D provides ground truth camera tracks of the target view, which is necessary for ReCamMaster to generate the target view video. However, ReCamMaster shows weak camera control ability and tends to generate new objects that never ap-

Table 1. Quantitative comparison on the Stereo4D and 3D Movie datasets. ↑ and ↓ indicate that higher is better and lower is better.

Stereo4D-Parallel Format 3D Movie-Converged Format

Method Venues

Latency↓ SSIM↑ MS-SSIM↑ PSNR↑ LPIPS↓ SIOU↑ SSIM↑ MS-SSIM↑ PSNR↑ LPIPS↓ SIOU↑

StereoDiffusion [43] CVPR’24 0.642 0.711 20.541 0.245 0.252 0.678 0.612 20.695 0.341 0.181 60 min StereoCrafter [59] arXiv’24 0.553 0.562 17.673 0.298 0.226 0.706 0.799 23.794 0.203 0.213 1 min SVG [12] ICLR’25 0.561 0.543 17.971 0.368 0.220 0.653 0.553 19.059 0.426 0.166 70 min ReCamMaster [3] ICCV’25 0.542 0.525 17.229 0.312 0.239 – – – – – 15 min M2SVid [38] 3DV’26 – 0.915 26.200 0.180 – – – – – – – Mono2Stereo [55] CVPR’25 0.649 0.721 20.894 0.222 0.241 0.795 0.810 25.756 0.191 0.201 15 min

StereoPilot (Ours) – 0.861 0.937 27.735 0.087 0.408 0.837 0.872 27.856 0.122 0.260 11 s

[Figure 149]

[Figure 150]

[Figure 151]

Table 2. Ablation results of our proposed method

Left View Input w/o domain switcher w/ domain switcher GT Right View

| |
|---|

| |
|---|

| |
|---|

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Method SSIM↑ MS-SSIM↑ PSNR↑ LPIPS↓ SIOU↑

| |
|---|

| |
|---|

| |
|---|

Baseline 0.833 0.891 26.954 0.143 0.319 Baseline w/ switcher 0.845 0.895 27.332 0.118 0.323 Baseline w/ switcher + Lcycle 0.849 0.905 27.796 0.105 0.334

| |
|---|

| |
|---|

| |
|---|

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

| |
|---|

| |
|---|

| |
|---|

Table 3. Performance on UE5 synthetic style parallel format data.

Method SSIM↑ MS-SSIM↑ PSNR↑ LPIPS↓ SIOU↑

Figure 7. Results w/ and w/o domain switcher. Our proposed domain switcher enhances the model’s generalization ability.

Baseline 0.791 0.881 28.377 0.150 0.291 Baseline w/ switcher 0.824 0.905 29.614 0.144 0.305

pear in the source view. Considering ReCamMaster’s performance on Stereo4D, which has an accurate target view camera track, was already poor, we did not further test it on 3DMovie. Because extracting the camera track for the target view’s video is difficult and will also introduce additional cumulative errors, which have been discussed in [33].

Qualitative Results. As illustrated in Figure 6, our method consistently outperforms all the baselines in both disparity accuracy and visual quality on the Converged and Parallel test sets. In the Converged cases, the baselines—StereoDiffusion, StereoCrafter, SVG, and Mono2Stereo—frequently produce noticeable disparity errors, which are clearly reflected in the highlighted regions. While Mono2Stereo performs comparatively better, it still exhibits misalignment with the ground truth and introduces visible artifacts. The remaining methods suffer from severe blur in repainted areas, with StereoCrafter additionally showing a persistent yellow color shift. In the second Converged example, facial and shoulder close-ups further reveal consistent disparity mistakes and detail loss among all baselines, resulting in noticeably blurrier outputs than ours. Similar observations hold for the Parallel set: all four baselines fail to predict correct disparity, and their reconstructed textures appear soft and lacking detail, whereas our method maintains sharpness and stable geometry. Finally, under challenging mirror conditions, all DWI-based approaches incorrectly estimate the reflected disparity, with StereoDiffusion and SVG even producing distortions. The PSNR annotations verify that our results remain significantly better aligned with the ground truth than all competing methods.

##### 5.4. Ablation Study

To validate our design within StereoPilot, we conduct ablation studies in this section. Detailed results are provided in Table 2. Initially, we train two models in a Diffusion FeedForward manner on the Stereo4D and 3DMovie datasets, respectively, to establish our baseline. Then the domain switcher is injected to enable unified conversion and strong generalization performance, as shown in Figure 7. Finally, we ablate the cycle consistent loss on the model with the switcher. The findings from these ablations validate the effectiveness of our proposed module and training target, demonstrating that each design plays a vital role in optimizing the StereoPilot for stereo conversion tasks. Note that we average the performance in two datasets for simplicity.

Furthermore, in order to verify the effectiveness of the domain switcher in solving the domain bias in data distribution, that is, the model’s generalization ability on parallelstyle synthetic/anime-style stereo videos that have not appeared in the training set. We used Unreal Engine 5 [13] to render 200 synthetic data style parallel 3D videos for testing. The performance improvement in Table 3 demonstrates the effectiveness of our designed domain switcher. Please refer to the Appendix for details.

##### 5.5. Conclusion and Limitation

Conclusion. We introduce UniStereo, the first large, unified dataset with parallel and converged formats for standardized evaluation. We also propose StereoPilot, an efficient feed-forward model using a pretrained video diffusion transformer to directly synthesize the second view, avoid-

ing “Depth-Warp-Inpaint” limitations like depth ambiguity. We integrate a domain switcher for format flexibility and a cycle consistency loss for high fidelity. Experiments show StereoPilot outperforms SOTA methods in visual quality and efficiency.

Limitation. Our model’s 11-second conversion time for a 5-second video is not yet sufficient for real-time applications such as live streaming. We plan to address this limitation in future work by exploring autoregressive conversion.

#### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. arXiv preprint arXiv:2411.18673, 2024. 3
- [2] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffusion transformers for 3d camera control. arXiv preprint arXiv:2407.12781, 2024. 3
- [3] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025. 3, 5, 7, 8, 1
- [4] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 3
- [5] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3
- [6] Stan Birchfield and Carlo Tomasi. Depth discontinuities by pixel-to-pixel stereo. International Journal of Computer Vision, 35(3):269–293, 1999. 2
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1
- [8] Brandon Castellano. PySceneDetect. 4, 2
- [9] Jia-Ren Chang and Yong-Sheng Chen. Pyramid stereo matching network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5410–5418,

2018. 2

- [10] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310– 7320, 2024. 1

- [11] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems, 37:19472–19495, 2024. 4, 1, 2
- [12] Peng Dai, Feitong Tan, Qiangeng Xu, David Futschik, Ruofei Du, Sean Fanello, Xiaojuan Qi, and Yinda Zhang. Svg: 3d stereoscopic video generation via denoising frame matrix. arXiv preprint arXiv:2407.00367, 2024. 1, 2, 3, 7, 8
- [13] Epic Games. Unreal engine 5. https://www. unrealengine.com/zh-CN/unreal-engine-5,

2025. Accessed: 2025-04-02. 8, 3

- [14] Fiveable. 9.3 interaxial distance and convergence – advanced cinematography. https : / / fiveable . me / advanced - cinematography / unit - 9 / interaxial- distance- convergence/studyguide/D3DKeJlVMxbTpG64, 2024. Accessed: 202510-12. 2
- [15] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 3
- [16] Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan De Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 753–

762. IEEE, 2025. 5

- [17] Wenhang Ge, Tao Hu, Haoyu Zhao, Shu Liu, and Ying-Cong Chen. Ref-neus: Ambiguity-reduced neural implicit surface learning for multi-view reconstruction with reflection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 3
- [18] Michal Geyer, Omer Tov, Linyi Jin, Richard Tucker, Inbar Mosseri, Tali Dekel, and Noah Snavely. Eye2eye: A simple approach for monocular-to-stereo video synthesis. arXiv preprint arXiv:2505.00135, 2025. 2, 3, 5
- [19] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3
- [20] Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng Wei, Liangke Gui, Qi Zhao, Gordon Wetzstein, Lu Jiang, and Hongsheng Li. Cameractrl ii: Dynamic scene exploration via camera-controlled video diffusion models. arXiv preprint arXiv:2503.10592, 2025. 3
- [21] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and YingCong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024. 1, 5
- [22] Tao Hu, Haoyang Peng, Xiao Liu, and Yuewen Ma. Ex-4d: Extreme viewpoint 4d video synthesis via depth watertight mesh. arXiv preprint arXiv:2506.05554, 2025. 3
- [23] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4d: Learning how

- things move in 3d from internet stereo videos. arXiv preprint arXiv:2412.09621, 2024. 3, 4, 6, 1
- [24] Junpeng Jing, Ye Mao, and Krystian Mikolajczyk. Matchstereo-videos: Bidirectional alignment for consistent dynamic stereo matching. In European Conference on Computer Vision, pages 415–432. Springer, 2024. 2
- [25] jinotter3. Stereo4d downloader. https://github. com/jinotter3/stereo4d_downloader. Accessed: 2025-11-19. 1
- [26] K. Team. Kling. https://klingai.kuaishou. com/, 2024. Accessed: December 9, 2024. 5
- [27] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13229–13239, 2023. 2
- [28] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9492–9502,

2024. 1

- [29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (ToG), 2023. 3
- [30] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 5
- [31] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng Wang, Lingjie Liu, Taku Komura, and Wenping Wang. Nero: Neural geometry and brdf reconstruction of reflective objects from multiview images. In SIGGRAPH, 2023. 3
- [32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [33] Yawen Luo, Jianhong Bai, Xiaoyu Shi, Menghan Xia, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Tianfan Xue. Camclonemaster: Enabling reference-based camera control for video generation. arXiv preprint arXiv:2506.03140,

2025. 8

- [34] Zhen Lv, Yangqi Long, Congzhentao Huang, Cao Li, Chengfei Lv, Hao Ren, and Dian Zheng. Spatialdreamer: Self-supervised stereo video synthesis from monocular input. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 811–821, 2025. 1, 2, 3
- [35] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021. 3
- [36] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. arXiv preprint arXiv:2503.03751, 2025. 3

- [37] Mike Seymour. Art of stereo conversion: 2d to 3d – 2012,

2012. 2

- [38] Nina Shvetsova, Goutam Bhat, Prune Truong, Hilde Kuehne, and Federico Tombari. M2svid: End-to-end inpainting and refinement for monocular-to-stereo video conversion. arXiv preprint arXiv:2505.16565, 2025. 1, 2, 3, 6, 7, 8
- [39] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928, 2024. 3
- [40] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3
- [41] Anthony Vetro, Sehoon Yea, and Aljoscha Smolic. Toward a 3d video format for auto-stereoscopic displays. In Applications of Digital Image Processing XXXI, pages 113–122. SPIE, 2008. 2
- [42] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 5, 7, 2
- [43] Lezhong Wang, Jeppe Revall Frisvad, Mark Bo Jensen, and Siavash Arjomand Bigdeli. Stereodiffusion: Training-free stereo image generation using latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7416–7425, 2024. 6, 7, 8
- [44] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers (SIGGRAPH), 2024. 3
- [45] Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, and Stan Birchfield. Foundationstereo: Zeroshot stereo matching. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5249–5260,

2025. 2

- [46] Steve Wright. Parallel vs converged. https://vfxio. com/PDFs/Parallel_vs_Converged.pdf, 2011. Accessed: 2024-10-12. 2
- [47] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26057–26068, 2025. 3
- [48] Junyuan Xie, Ross Girshick, and Ali Farhadi. Deep3d: Fully automatic 2d-to-3d video conversion with deep convolutional neural networks. In European conference on computer vision, pages 842–857. Springer, 2016. 2, 3
- [49] Hirokazu Yamanoue. The differences between toed-in camera configurations and parallel camera configurations in shooting stereoscopic images. In 2006 IEEE International Conference on Multimedia and Expo, pages 1701–1704. IEEE, 2006. 2

- [50] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv preprint arXiv:2406.09414, 2024. 1
- [51] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024. 1
- [52] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1
- [53] YouTube. Youtube. https://www.youtube.com/. Accessed: 2025-11-19. 1
- [54] yt-dlp Contributors. yt-dlp. https://github.com/ yt-dlp/yt-dlp. Accessed: 2025-11-19. 1
- [55] Songsong Yu, Yuxin Chen, Zhongang Qi, Zeke Xie, Yifan Wang, Lijun Wang, Ying Shan, and Huchuan Lu. Mono2stereo: A benchmark and empirical study for stereo conversion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21847–21856, 2025. 1, 2, 3, 6, 7, 8
- [56] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 3
- [57] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492, 2020. 3
- [58] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2018. 6
- [59] Sijie Zhao, Wenbo Hu, Xiaodong Cun, Yong Zhang, Xiaoyu Li, Zhe Kong, Xiangjun Gao, Muyao Niu, and Ying Shan. Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos. arXiv preprint arXiv:2409.07447, 2024. 1, 2, 3, 7, 8

## StereoPilot: Learning Unified and Efficient Stereo Conversion via Generative Priors

[Figure 163]

### Supplementary Material

#### 6. Depth Ambiguity Details

Left View Left Crop

[Figure 164]

To demonstrate the challenge of depth ambiguity, we provide an in-depth analysis in Figure 8 by tracking two representative keypoints across the stereo views. Let P denote a pixel corresponding to a specular reflection (a utility pole) on the mirror surface, and Q denote the position of the topright outer frame outside the mirror.

𝑄

𝑃

[Figure 165]

| |
|---|

While the physical depth of the mirror surface at P (DP

) is proximate to the frame at Q (DQ), satisfying DP

Right Crop

Right View

S

[Figure 166]

S ≈ DQ, their optical characteristics differ significantly. Specifically, the virtual imaging depth of the reflection at P (DP

𝑄 Δ𝑥

𝑃

| |
|---|

) is substantially larger than the physical depth of the frame (DP

R

R ≫ DQ). Consequently, this discrepancy leads to distinct disparity behaviors: the reflected content at P exhibits negligible parallax (∆xP ≈ 0), whereas the physical frame at Q shows significant displacement (∆xQ = 11 px).

This manifests as a counter-intuitive visual phenomenon where the mirror surface and the reflected content appear disentangled: in the transition between views, the mirror frame undergoes significant horizontal translation, while the specular reflection remains virtually stationary.

Δ𝑥 = 11px Δ𝑥 < 1px

𝐷𝑒𝑝𝑡ℎ ≈ 𝐷𝑒𝑝𝑡ℎ

Figure 8. Detailed Analysis of Depth Ambiguity.

#### 7. UniStereo Construction Details

cretely, we set the output resolution in the rectification code to 832 × 480 and the horizontal field-of-view (hfov) to 90◦ to convert each VR180 video into a pair of per-eye monocular videos. We run this rectification process on a cluster of 32 machines, each equipped with 32 CPU cores, for about two to three weeks, which yields on the order of 100K clips with durations between 1 and 6 seconds. To make the data compatible with Wan2.1-1.3B [42] and to standardize the training inputs, we then resample all clips to 16 fps and retain only those whose length is at least 81 frames. From these filtered clips, we further process each one into an 81frame sequence at a fixed resolution of 832 × 480. Finally, we apply ShareGPT4Video [11] to generate captions for the left-eye video of each stereo pair, obtaining approximately 60K stereo video–caption pairs for training and evaluation.

We provide a detailed description of the data construction pipelines for both the Stereo4D (Section 7.1) and 3DMovie (Section 7.2) datasets, outlining each processing step from raw videos to the final training-ready stereo pairs.

##### 7.1. Stereo4D Construction Details

As illustrated in Figure 9, we now describe the construction pipeline for the Stereo4D subset. We start from the official Stereo4D release [23], which defines a corpus that links to about 4 TB of raw VR180 videos on YouTube and provides about 4 TB of accompanying NPZ metadata. The raw videos are VR180 clips on YouTube [53] that we download using yt-dlp [54] together with the Stereo4D downloader [25], resulting in roughly 7K source videos. Downloading and uploading this volume of data on a single machine typically takes three to four weeks. The NPZ files record, for each video, its video id for downloading, as well as the rectification parameters and camera2world matrices, which we later use as camera pose inputs when running ReCamMaster [3] for inference. We obtain these NPZ files by following the instructions provided by Stereo4D [23].

##### 7.2. 3DMovie Construction Details

As illustrated in Figure 10, we now describe the construction pipeline of the 3DMovie dataset in detail. We start by collecting 159 convergent 3D movies and manually inspecting the stereo layout of each title. We retain only those movies encoded in left–right side-by-side (SBS) format and discard those using top–bottom layouts, so that all

With the raw videos and metadata in place, we then follow the rectification pipeline specified in Stereo4D. Con-

[Figure 167]

###### Figure 9. Stereo4D processing pipeline. The detailed process of Stereo4D data processing.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Filter out the beginning and the end of the movie

Slice into 81frames segments

Sample fps16

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Filter out clips with a length of less than 81 frames

Original Movie

Split the movie and filter the Clips

SBS Format converted to left–right monocular format Remove black borders and produce 832×480 video with its caption

Figure 10. 3DMovie processing pipeline. The detailed process of 3DMovie data processing.

data share a consistent arrangement. Next, we examine the disparity pattern of every remaining movie. We find that some titles do not follow a true convergent stereo setup: switching between the left and right views only produces a horizontal shift in the image, rather than the slight rotational change typical of convergent cameras with a welldefined zero-disparity plane. Because such “pseudo-stereo” content may introduce harmful biases into model training, we remove these movies and finally keep 142 high-quality convergent 3D films.

We then standardize the data format. We first resample each movie to 16 fps, so that all videos share a unified temporal resolution that satisfies the stability requirements of the Wan2.1-1.3B [42] model. Building on this, we apply PySceneDetect [8] to segment each movie into short clips, ensuring that frames within the same clip mostly belong to a single scene and therefore exhibit a relatively consistent disparity pattern, which in turn makes stereo learning easier and accelerates training convergence. These scene clips have varying durations, so for training convenience and to match the temporal window of our model, we discard clips shorter than 81 frames. In addition, we trim the beginning and end of each movie to remove segments dominated by credits and production subtitles, which do not provide useful visual content. From this refined pool of clips, we generate fixed-length training samples by further slicing them into 81-frame segments. When a single clip yields multiple

81-frame segments, we index them in temporal order and retain only the segments with odd indices, thereby reducing redundancy while preserving temporal diversity.

Next, we process each 81-frame SBS segment into pereye videos. We first split the SBS format into separate monocular sequences for the left and right eyes, and at the same time restore the original per-eye width so that each view has the correct spatial resolution. Some movies contain black borders around the image, which would otherwise introduce artificial structures into the data. To remove them in a consistent manner, we detect black borders on the left-eye video and then apply the same symmetric cropping to both left and right views. After this step, each retained segment becomes a pair of 81-frame, border-free left/right monocular videos. We then adjust the spatial resolution to match the input requirements of Wan2.1-1.3B [42]. Based on the aspect ratio, we select video pairs whose size is close to 832 × 480 and reshape them to exactly 832 × 480. This results in a collection of stereo video pairs, each with 81 frames at a resolution of 832 × 480. Finally, we use ShareGPT4Video [11] to generate captions for the left-eye video of each pair, yielding approximately 48K high-quality stereo video–caption pairs for training and evaluation.

#### 8. UE5 Synthetic Dataset

To validate the generalization on the synthetic style paralleled format stereo video of StereoPilot, we construct a

GT Left View GT Right View Right View StereoPilot

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Figure 11. Examples of UE5-Rendered Stereo Video Data.

synthetic benchmark based on 6 randomly selected Unreal Engine [13] scenes under both daytime and nighttime conditions. To approximate human binocular vision and obtain paired stereo videos, we place two virtual cameras with a fixed baseline of 6.3 cm [23]. The scenes feature 28 categories of dynamically animated animals as the primary foreground subjects, and in total, we render 200 stereo video pairs, which are used to rigorously evaluate the effectiveness of our proposed model. As shown in Figure 11, our model still demonstrates stable generalization ability on synthetic style parallel stereo datasets.

#### 9. Geometry Relationship in Stereo Vision

In this section, we first define the following common parameters for the stereo rig in stereoscopic vision for both parallel and converged camera configuration:

- • Baseline (B): The distance between the optical centers of the left (OL) and right (OR) cameras.
- • Focal Length (f): The distance from the optical center to the image plane (assumed identical for both cameras).
- • Point P(X,Y,Z): A target point in 3D space located at depth Z.
- • Projections (xl,xr): The horizontal coordinates of point P projected onto the left and right image planes, respectively.

##### 9.1. Parallel Stereo Video

In a parallel stereo configuration, the two cameras are aligned such that their optical axes are strictly parallel. This setup is mathematically equivalent to standard epipolar geometry.

###### 9.1.1. Visual Representation

The geometry is illustrated in Figure 12. We assume the world coordinate system origin is located at the optical center of the left camera, OL = (0,0,0).

###### 9.1.2. Mathematical Derivation

Using similar triangles, we relate the physical coordinates to the image plane coordinates.

P(X,Z)

Z xl xr

Image Plane

f

Baseline B

OL OR

- Figure 12. Geometry of parallel stereo vision. By convention, xl is positive and xr is negative for a point located between the camera axes.

Left Camera: With the camera at the origin (0,0,0), the projection is:

xl =

f · X Z

(11)

Right Camera: With the camera shifted by B along the X-axis, the local X-coordinate is (X − B). The projection is:

xr =

f · (X − B) Z

(12)

Disparity and Depth: Disparity d = xl−xr is calculated as:

d =

fX Z −

f(X − B) Z d =

f · B Z

(13)

Rearranging for depth Z:

Z =

f · B d

(14)

9.2. Converged Stereo Video

In stereoscopic 3D video creation, a toed-in (or converged) configuration is often used. The optical axes of the two cameras are rotated to intersect at a specific convergence distance Zc.

9.2.1. Visual Representation

- Figure 13 illustrates the converged geometry. The cameras are rotated inward by an angle θ such that their optical axes intersect at Zc.

Zworld

P(X,Z)

Zc

θ

Xworld

B

Figure 13. Geometry of a converged stereoscopic system where optical axes intersect at Zc.

###### 9.2.2. Mathematical Derivation

We define the world coordinate origin (0,0,0) at the midpoint of the baseline.

Coordinate Transformation: The cameras are translated by ±B/2 and rotated by ∓θ. We first transform the world point P into the local camera coordinate systems (Xcam,Ycam,Zcam).

###### Left Camera (L):

 

 

 

  (15)

  =

 

cosθ 0 −sinθ 0 1 0 sinθ 0 cosθ

XL YL ZL

X + B/2

- Y
- Z

###### Right Camera (R):

 

  =

 

 

 

  (16)

X − B/2

XR YR ZR

cosθ 0 sinθ 0 1 0

- Y
- Z

−sinθ 0 cosθ

Projection Equations: Applying perspective projection x = f · Xcam/Zcam to the transformed coordinates:

(X + B/2)cosθ − Z sinθ (X + B/2)sinθ + Z cosθ

(17)

xl = f

(X − B/2)cosθ + Z sinθ −(X − B/2)sinθ + Z cosθ

(18)

xr = f

These equations account for the keystone distortion inherent in toed-in camera configurations.

