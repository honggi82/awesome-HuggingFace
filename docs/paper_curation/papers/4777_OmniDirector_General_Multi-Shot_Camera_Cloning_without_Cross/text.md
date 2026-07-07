## OmniDirector: General Multi-Shot Camera Cloning without Cross-Paired Data

### Jiwen Liu1∗, Shujuan Li1,2∗, Zhixue Fang1, Xiaohan Li1, Yan Zhou1†, Zijie Meng1,3, Zhimin Zhang1,3, Yawen Luo1, Guoxin Zhang1, Yu-Shen Liu2, Pengfei Wan1

1Kling Team, Kuaishou Technology, 2Tsinghua University, 3Peking University

# arXiv:2606.13432v1[cs.CV]11Jun2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

shot1 shot2 shot3

Synth.Ref.

Multi-ShotArcShot

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

shot1 shot2 shot3

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Invertedperspective

Synth.Ref.Ref.Synth.

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

Figure 1: OmniDirector precisely clones diverse camera motions from reference videos to animate source images, remaining invariant to content discrepancies, aspect ratios, and spatial scales.

###### Abstract

camera parameters visually and supports the integration of diverse trajectories for multi-shot video generation. Building upon this, we propose OmniDirector, a unified framework trained on a million-scale camera grid-video pairs that coordinates characters, actions, and cameras to provide director-level control for multimodal diffusion transformers. Furthermore, we design a novel hierarchical prompt expansion agent that harmoniously integrates different control signals by systematically describing camera motion and visual content through understanding signal relationships. Extensive experiments demonstrate the superior performance and outstanding controllability of our framework. Project page: https://ymlinfeng.github.io/OmniDirector.github.io/

Cloning camera motion from reference videos is an important task in video generation, as videos provide intuitive and precise control. Existing methods either directly use parametric representations that fail to handle multi-shot generation or synthesize cross-paired data, which suffer from data scarcity, resulting in poor performance in complicated camera motion cloning. To address these issues, we introduce a general camera motion representation that encodes cameras as grid motion videos. This camera grid represents the

∗These authors contributed equally. †Corresponding author.

### Introduction

Camera motion is a vital component of video generation, which transcends spatial composition to shape the atmosphere, emotion, and narrative depth. Conventional approaches to camera control in video generation typically employ either textual descriptions (Wang et al. 2026; Luo et al. 2026) or explicit camera parameters (He et al. 2024; Bai et al. 2025a; Wang et al. 2025; Meng et al. 2026a). However, these modalities present an inherent trade-off: textual specifications fail to precisely define nuanced cinematographic attributes, while parametric representations impose significant user barriers to accessibility. Consequently, directly cloning camera motion from reference videos emerges as an optimal representation that reconciles both precision and accessibility requirements. Particularly, multi-shot camera cloning remains an under-explored problem.

Video reference-based camera control methods for video generation can be primarily categorized into explicit and implicit approaches. Explicit methods (Bai et al. 2025a; He et al. 2024; Wu et al. 2025; Meng et al. 2026b) typically inject camera motion into video generation models using matrices (Bai et al. 2025a) or Plücker coordinates (He et al.

- 2024). However, these frameworks are limited to representing basic camera motions and fail to handle shot transitions in multi-shot videos. Furthermore, the inherent semantic gap between these parameterized representations and standard visual signals (such as images and videos) complicates the model optimization. Implicit methods (Luo et al. 2025) directly encode camera information from paired videos that differ solely in camera motion, yet such strictly controlled data is exceedingly scarce in real-world scenarios. Although recent methods (Bai et al. 2025a; Luo et al. 2025) alleviate this issue with synthetic data, these methods fail to handle complicated cameras due to the limited diversity. Additionally, these videos inherently contain substantial extraneous information (such as appearance, character motion), frequently resulting in information leakage during practical deployment.

To address these challenges, we introduce the camera grid, a general camera representation that handles complex camera motions for single or multi-shot videos in a unified way. Specifically, we first extract camera parameters from reference videos, then render them as a grid motion video which showsthemovementwithina3Demptyscene.Thisrepresentation offers three key advantages: (1) Generality: The camera grid enables unified handling of diverse camera motion for multi-shot cloning. (2) Decoupling: The empty scene decouples camera motion from other signals, preventing interference information from being injected into the model. (3) Scalability and Compatibility: The camera grid facilitates diffusion models’ learning of camera motion from both data and optimization perspectives. For data, any video sample can generate its corresponding camera grid, facilitating construction from massive internet-scale datasets. For optimization, the camera grid is modality-compatible with other visual signals (e.g., images and videos), making it easier for video diffusion models to interpret.

Based on the camera grid, we propose a novel framework OmniDirector, a unified framework designed to pro-

vide director-level control for Multi-Modal Diffusion Transformers (MMDiTs) (Esser et al. 2024). We train the model on a newly curated dataset of million-scale camera gridvideo pairs. By utilizing the camera grid as an optimizationfriendly visual-semantic representation, our approach effectively bridges the modality gap and facilitates camera motion learning from both data and optimization perspectives.

Furthermore, to harmoniously integrate camera control with other control signals into a unified whole, we design a novel hierarchical Prompt Expansion (PE) Agent mechanism during the inference stage. First, we design a camera prompt generator to describe the camera motion. We decompose the camerapromptintotwohierarchies:inter-shotandintra-shot. The inter-level prompt handles the shot relationships across shot transitions to ensure semantic coherence in multi-shot scenarios. The intra-level focuses on the camera clone descriptions within individual shots. Finally, through semantic fusion, we integrate camera motion, subjects, and object motion into a unified representation. Overall, our work makes the following key contributions: (1) We propose OmniDirector, a unified video generation framework that achieves general multi-shot camera cloning without cross-paired data, empowering MMDiTs with director-level control. (2) We introduce the camera grid, a general representation of various camera movements for multi-shot camera cloning, and construct a million-scale camera grid–video dataset to scale up the training of our model. (3) At inference, the hierarchical prompt expansion agent integrates camera motion with other control signals into a harmonious text, enabling collaborative creation with multimodal signals.

### Related Work

Video Generation. Video generation has advanced significantly, driven by the success of Diffusion Models (Ho, Jain, and Abbeel 2020; Ho et al. 2022b,a). Based on input conditions, the field is primarily categorized into Text-toVideo (T2V) (Singer et al. 2022; Villegas et al. 2023; Bar-Tal et al. 2024; Polyak et al. 2024), Image-to-Video (I2V) (Zhang et al. 2023; Xing et al. 2024; Chen et al. 2023b; Ren et al. 2024), and Video-to-Video (V2V) (Wang et al. 2023) tasks. T2V focuses on complex semantic-to-motion translation; I2V emphasizes temporal extrapolation while preserving spatial identity; and V2V targets attribute manipulation while maintaining structural priors. Architecturally, early diffusion-based video generation models predominantly relied on 3D U-Nets (Ho et al. 2022b,a; Singer et al. 2022; Blattmann et al. 2023) equipped with factorized spatialtemporal attention to model inter-frame dynamics. However, driven by scaling laws, the paradigm has rapidly shifted towards Diffusion Transformers (DiT) (Peebles and Xie 2023; Ma et al. 2025). To further enhance condition adherence, the Multi-Modal Diffusion Transformer (MMDiT) (Esser et al. 2024) has emerged as a leading framework. By decoupling text and visual tokens into separate streams and enabling interaction via joint attention blocks, MMDiT achieves superior cross-modal alignment (Polyak et al. 2024). This architectural evolution significantly improves scalability, enabling the generation of longer, high-fidelity, and physically plausible videos.

Camera Controllable Video Generation. Despite the remarkable success of text-to-video generation models, relying solely on text prompts falls short in providing the precise spatial-temporal control required for real-world applications (Zhang, Rao, and Agrawala 2023; Mou et al. 2024; Chenetal.2023a).Consequently,introducingadditionalconditional signals for controllable generation has been widely studied (Zhao et al. 2024; Guo et al. 2023; Hu 2024; Yin et al. 2023; Chen et al. 2024; Girdhar et al. 2024). Among various control modalities, camera movement stands out as a fundamental cinematic language. Existing camera-controllable methods (Xu et al. 2024; Zheng et al. 2024; Bahmani et al.

- 2025) can be broadly categorized by their reliance on explicit camera parameters. The first category requires explicit parameters to dictate camera dynamics. For instance, MotionCtrl (Wang et al. 2024) encodes 6DoF camera extrinsics and injects them into temporal attention layers, while CameraCtrl (He et al. 2024) employs Plücker embeddings and a dedicated camera encoder to capture richer geometric information. To further enforce 3D geometric constraints, methods like CamCo (Xu et al. 2024) and CamI2V (Zheng et al. 2024) utilize epipolar attention. Recently, AC3D (Bahmani et al. 2025) conducted an in-depth investigation into camera motion within diffusion transformers, achieving enhanced visual quality. While these methods achieve precise viewpoint control, a key limitation is that obtaining and specifying explicit camera trajectories can be cumbersome for general users. To alleviate this, the second category explores parameter-free or training-free approaches. AnimateDiff (Guo et al. 2023) introduces various motion LoRAs to learn specific patterns of camera movements without requiring frame-wise parameters. Furthermore, methods like MotionMaster (Hu et al. 2024) and MotionClone (Ling et al.

2024) employ an inversion process to derive motion representations directly from temporal attention maps. Although these parameter-free methods offer greater user convenience, they often exhibit limited generalization and can struggle to maintain robust control in complex scenarios.

Video-referenced Camera Controllable Video Generation. To bridge the gap between the precise but cumbersome explicit parameter control and the user-friendly but coarse text-driven generation, video-referenced camera control has emerged as a highly practical paradigm. By leveraging a source video as a motion exemplar, this approach offers superior precision over pure text prompts while remaining significantly more intuitive for users than manually specifyingcomplextrajectories.Existingmethodsinthisdomaincan be broadly classified into two categories. The first category adopts a parameter-extraction pipeline (Wang et al. 2024; He et al. 2024; Xu et al. 2024; Zheng et al. 2024), where explicit camera parameters are estimated from the reference video and subsequently injected into the generation model as conditions. However, this paradigm is fundamentally bottlenecked by inherent scale ambiguities across different scenes. Applying camera translations extracted from a reference environment to a generated scene with a disparate spatial scale often leads to severe geometric distortions and unnatural dynamics. The second category bypasses explicit parameter extraction by directly training models on cross-paired

3D Empty Scene-Camera Trajectory

|[Figure 35]| | | |
|---|---|---|---|
|[Figure 36]|[Figure 37]|[Figure 38]|[Figure 39]|
|[Figure 40]|[Figure 41]|[Figure 42]|[Figure 43]|
|[Figure 44]|[Figure 45]|[Figure 46]|[Figure 47]|
|[Figure 48]|[Figure 49]|[Figure 50]|[Figure 51]|

Camera GridSynth.2VideoRef.Synth.1

Figure 2: 3D Scene Modeling in Camera Grid. Top: Given reference camera poses, we simulate spatial motion within an empty 3D scene. Orthogonal lines represent the ceiling and floor (red and blue), while vertical lines (yellow) denote the walls. Bottom: Rendering this grid scene from varying viewpoints yields the camera grid, providing a visual representation of camera motion that conditions the model to generate videos with similar trajectories.

data—pairs of videos that share identical camera movements but depict different contents. While this end-to-end approach (Luo et al. 2025; Ling et al. 2024; Hu et al. 2024) mitigates scale mismatch, it is severely constrained by the scarcity of real-worldpaireddata,hinderingtheabilitytoscaleupmodel training. To overcome this data scarcity, some recent methods (Li et al. 2024; He et al. 2025) resort to synthetic datasets rendered via game engines like Unreal Engine. Nevertheless, these synthetic videos typically lack the rich narrative contexts found in real-world footage, making it exceedingly difficult for models to comprehend and handle complex cinematic signals, such as abrupt shot cuts and intricate scene

transitions. Method

In this section, we detail the design of our proposed OmniDirector in Figure 4. We first describe the core representation camera grid. Next, we introduce OmniDirector’s architecture and training strategy. Finally, we introduce the hierarchical prompt expansion agent and inference strategies.

##### Camera Grid

Camera movement is the dynamic change of viewpoint over time, which reveals the spatial relationships among objects

Tile + Fisheye

[Figure 52]

[Figure 53]

[Figure 54]

Dolly Zoom

[Figure 55]

[Figure 56]

[Figure 57]

- Figure 3: Extension of the camera grid to special camera effects. First row: fisheye distortion, where straight lines are rendered as smooth curves via the Kannala–Brandt model. Second row: dolly zoom (Hitchcock zoom), where the subject remains fixed in size while the background undergoes pronounced perspective stretching.

within a scene. To accurately simulate the spatial transformation with camera movement, we adopt a simplified modeling approach: we abstract the complex real world into an empty room as shown in Figure 2, where no objects or scene elements are placed. Only 3D grid lines are used to indicate the directions of spatial coordinate axes, thereby clearly presenting the geometric structure of the space and the camera motion trajectory.

Visualizingthecameramovinginanemptyscene. Given a sequence of camera matrix parameters P = {Ri,ti}Ti=1 from the reference video, where Ri ∈ SO(3) denotes the rotation matrix, ti ∈ R3 represents the translation vector, and T denotes the total number of video frames. First, we compute the spatial bounding box of the scene by analyzing the camera translation trajectory {ti}. In the scene box, we use two planes to represent the floor and ceiling. Specifically, uniformly sampled grid points {p} are generated on the X-Z plane with two fixed heights. We define the heights as relative to the average scene height y along the Y-axis with an offset ∆h as follows:

yfloor = y − ∆h (1) yceiling = y + ∆h (2)

In our experiments, ∆h is defined proportionally to the median distance between consecutive camera poses, with a lower bound to ensure robustness. Orthogonal grid lines are then rendered on these two planes to form a spatial reference framework, as shown in Figure 2.

To enhance the spatial perception of camera motion, we use lines along the Y-axis to present the relative movement of objects. For the sake of computational efficiency and visual simplicity, rather than connecting all grid points between the two X-Z planes, we construct a tubular boundary structure surrounding the trajectory. Specifically, we project the camera trajectory onto the X-Z plane, yielding projected points cproj. A KD-tree is employed to compute the distance dtraj from each grid point p on the X-Z plane to the projected points cproj, and vertical line segments are generated within an annular region:

W = {(x,z) | r < dtraj(x,z) < r + δ} (3)

where r denotes the inner tunnel radius and δ represents the wall thickness. These vertical lines connect the floor and ceiling planes, creating a visual “tunnel wall” effect.

After completing the spatial scene modeling, spatial variations are achieved by rendering views under different camera poses. For camera movements (such as dolly, zoom, pan, and tilt), during the rendering of each frame, the grid line segment endpoints {Pw} in the world coordinate system are first transformed to the camera coordinate system via the camera extrinsic matrix [Ri | ti] as Pc = RiPw + ti, and then mappedtotheimageplanethroughprojectiontransformation using the camera intrinsic parameters. Finally, we synthesize temporal camera motion by rendering the per-frame poses.

Extendingthecameragridtorepresentspecialcameraeffects. The camera grid can be extended to represent certain special camera effects by modifying its rendering scheme. As illustrated in the Figure 3, we take fisheye distortion and dolly zoom as two representative examples.

Fisheye Distortion. For fisheye distortion, we adopt a distortion formulation based on the Kannala–Brandt model (Kannala and Brandt 2006). The core procedure first computes the incident angle of a spatial point with respect to the camera’s optical axis,

###### θ = arctan(r′/ζ), (4)

where θ denotes the incident angle between the line of sight to the spatial point and the optical axis, ζ is the depth of the point along the optical axis (i.e., its z-coordinate in the camera coordinate system), and r′ is the radial distance of the point from the optical axis on the plane perpendicular to it. A fourth-order polynomial is then applied to perform radial nonlinear scaling of this angle, yielding the distorted angle

###### θd = θ 1 + k1θ2 + k2θ4 + k3θ6 + k4θ8 , (5)

where θd is the distorted (mapped) angle, and kj,j ∈ {1,2,3,4} are the radial distortion coefficients of the Kannala–Brandt model that governs the nonlinear projection behavior of the fisheye lens. In the rendering implementation, each 3D spatial line is densely subdivided into a set of discrete points, and the aforementioned nonlinear projection mapping is applied to each point individually. In this way, the originally straight physical lines are fitted into smooth curves on the pixel plane, thereby faithfully reproducing the characteristic “straight-line-to-curve” visual distortion inherent to fisheye lenses.

Dolly Zoom. The dolly zoom effect is represented by constructing a 3D cube at the subject’s location in conjunction with a picture-in-picture (PIP) tracking view. Its core mechanism exploits the principle that the focal length is proportional to the distance,

φ ∝ ρ, (6) where φ denotes the focal length of the camera and ρ is the distance from the camera to the subject. This proportionality ensures that the projected visual size of the 3D cube remains constant. Simultaneously, a 2D bounding frame scaled by the focal-length ratio φ/φref is rendered to quantify the change in field of view, where φref denotes the reference

| |
|---|

|ReferenceVideo𝑉 Camera Grid Rendering<br><br>Focal Length<br><br>𝑓 = 𝑎 × max(𝐻,𝑊)<br><br>Camera Grid 𝐺<br><br>… …<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>Camera Extrinsics<br><br>[Figure 66]<br><br>Pose<br><br>Estimator<br><br>|
|---|
|Referencing camera motion<br><br>from the video<br><br>Text 𝑇′<br><br>Patchify<br><br>MMDiTBlock<br><br>× 𝑵<br><br>[Figure 67]<br><br>[Figure 68]<br><br>……<br><br>[Figure 69]<br><br>𝒁𝑣𝑖𝑠<br><br>[Figure 70]<br><br>[Figure 71]<br><br>……<br><br>[Figure 72]<br><br>𝒁𝑡<br><br>Camera Grid 𝐺<br><br>[Figure 73]<br><br>ReferenceImage𝐼 Visual Encoding<br><br>[Figure 74]<br><br>3D VAE<br><br>𝒛𝑐 ∈ 𝑹𝑇×𝐻×𝑊×𝐶<br><br>𝒛𝑣 ∈ 𝑹𝑇×𝐻×𝑊×𝐶<br><br>𝒛𝐼 ∈ 𝑹1×𝐻×𝑊×𝐶<br><br>| | | |
|---|---|---|
<br><br>| | | |
|---|---|---|
| | | |
<br><br>| | | |
|---|---|---|
| | | |
<br><br>Frame<br><br>Concatenation<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>Output<br><br>Training<br><br>Noisy Latent<br><br>[Figure 79]<br><br>3D VAE<br><br>|
|[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>…<br><br>Output<br><br>OmniDirector<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Inference<br><br>Reference<br><br>Video 𝑉<br><br>Camera Grid 𝐺<br><br>[Figure 84]<br><br>|Camera<br><br>Prompt Generator<br><br>Semantic<br><br>Fusion<br><br>User<br><br>Prompt 𝑇𝑢<br><br>Reference<br><br>Image 𝐼<br><br>Final Prompt 𝑇𝑓<br><br>|
|---|
<br><br>User<br><br>Prompt 𝑇𝑢<br><br>Reference<br><br>Image 𝐼<br><br>[Figure 85]<br><br>Camera<br><br>Poses<br><br>Transition Frames<br><br>Pose<br><br>Estimation<br><br>Transition<br><br>Detection<br><br>Camera Prompt Generator Training<br><br>Transition<br><br>Frames 𝐹<br><br>+<br><br>Camera Poses 𝑃<br><br>MLLM<br><br>1. Multi-shot relationship analyzing<br>2. Pose-to-prompt translation<br>3. Video-based correction<br>4. Human annotation<br><br><br>Camera<br><br>Prompt 𝑇𝑐<br><br>Input Label<br><br>|

MMDiTBlock

#### ……

- Figure 4: Overview of OmniDirector. Top: OmniDirector represents camera motion via a camera grid G, which is obtained by rendering the camera poses of a reference video V as movement within an empty 3D space. Middle: During training, the camera grid is injected into the MMDiT alongside other control signals via token concatenation. Bottom: At inference, a PE Agent harmoniously integrates various signals into the text prompt, achieving unified multi-signal control.

focal length serving as the baseline for the scaling. In parallel, a virtual tracking camera generates the picture-in-picture view in which the subject is always kept centered; by leveraging the drastic perspective deformation of the background tunnel grid during zooming, this design precisely reproduces the signature “static subject, stretching background” visual characteristic of the dolly zoom effect.

video into distinct sub-clips based on F to ensure scene consistency within each segment. We render the transition frames as special pure white frames to construct a transition signal.

Then we treat each sub-clip as a single shot video. We leverage DPA-V3 (Lin et al. 2025) to estimate camera extrinsic parameters, and approximate the camera focal length with f = a×max(H,W) to enhance the stability of camera control and ensure resolution-agnostic generalization, where a = 0.8 is an empirical scale factor. Then we render the camera grid G with the process in the previous section.

General camera representation for multi-shot camera cloning. As a purely visual representation, the camera grid readily generalizes to encode camera motion across multishot video sequences.

By naturally extracting a corresponding camera grid from arbitrary video samples, we can automatically construct paired training data consisting of reference videos and their camera grids. This inherent capability means that the generation of these data pairs can be seamlessly scaled up, facilitat-

Given a reference video V , we first adopt the transition detection model TransNet-V2 (Soucek and Lokoc 2024) to detect the shot transition frames F = {fi}Ki=1, where K is the total number of transition frames. We segment the

ing the efficient curation of massive, internet-scale datasets for robust model training.

##### OmniDirector

Injecting camera grid via token concatenation. The proposed camera grid representation achieves complete disentanglement of camera control from other visual attributes, enabling interference-free injection into the transformer architecture. Benefiting from the fact that the camera grid is essentially a spatiotemporal signal similar to video, it can be encoded following the standard video processing pipeline.

Specifically, we first encode the camera grid G and reference image I into latents using a pre-trained 3D-VAE zc = ε(G) ∈ RT×H×W×C and zI = ε(I) ∈ R1×H×W×C, where H × W denotes the frame resolution, and C denotes the latent channel. We concatenate these visual modalities with the video noisy latent Zv along the frame dimension to form a unified spatiotemporal representation zvis = Concat(zI,zv,zc) ∈ R(2T+1)×H×W×C, which is then mapped into token sequences through 3D convolutional layers:

Zvis = Patchify(zvis) ∈ RN

vis×D (7)

where Nvis denotes the number of video tokens and D is the hidden dimension. For the text condition T′, we process it through a pre-trained text encoder (e.g., T5): Zt = TextEncoder(T′) ∈ RN

t×D.

Subsequently, we perform joint attention between the visual tokens and text tokens. The visual and text modalities are processed through separate attention pathways within the MMDiT architecture, where they interact through attention. Within each Transformer block, tokens from both modalities exchange information:

Z(visl+1) = FFN(Attention(LN(Z(visl) ),Z(tl))) + Z(visl) (8)

where l denotes the layer and FFN denotes Feed-Forward Network.

This hierarchical fusion design first unifies all visual modalities (reference image, video, and camera grid) into a coherent spatiotemporal representation, then enables text semantics to modulate the visual generation process through joint attention. This architecture maintains the natural alignment between visual signals while allowing flexible textbased control, supporting seamless injection of multimodal conditions in an end-to-end manner.

Training strategy. To balance controllability and generation capability, we incorporate a self-reconstruction objective into 30% of the training samples. For these samples, instead of pairing the camera grid G with a natural video, we replace the target video with the camera grid itself, i.e., the modelisconditionedonzc = ε(G)andtrainedtoreconstruct zv = ε(G). Since the camera grid contains no appearance or content cues, faithfully reconstructing it forces the model to parse the geometric structure and temporal dynamics encoded in the grid, rather than treating it as a weak auxiliary hint that can be partially ignored. This auxiliary objective thus strengthens the model’s understanding of the camera trajectory representation and prevents it from overfitting to

spurious correlations in the camera-to-video mapping. The remaining follows the standard camera-conditioned video generation task, where the target video depicts natural content consistent with the conditioning camera grid.

##### Unified Multimodal Control Signals for Inference

Hierarchical prompt expansion agent. Video generation models are commonly pre-trained on text-to-video (T2V) objectives, making textual conditioning a primary factor that governs the generated content. Consequently, consolidating heterogeneous control signals (e.g., user prompts, referenceimage cues, and camera grids) into a unified textual condition is a crucial step for eliciting reliable camera controllability.

A straightforward approach to obtain camera-related text from a reference video is to directly ask a multimodal large language model (MLLM), to describe the camera motion. However, camera motion in real videos is tightly entangled with other factors such as subjects, actions, and backgrounds. The MLLM tends to incorporate these non-camera semantics into the description. When used at inference time, such entangled signals can conflict with the intended conditioning, thereby weakening the effective camera control. To address this issue, we propose a hierarchical prompt expansion agent capable of generating semantically coherent prompts free from background leakage, conditioned on the given inputs as shown at the bottom of Figure 4. We divide the text expansion process into two parts: camera prompt generation and multimodal signal fusion.

###### Camera prompt generation. We first train a generator

to describe the camera motions with prompt Tc directly from camera parameters, including the transition frames

F = {fi}Ki=1 and camera poses P. To achieve this, we decompose the camera prompt into inter-shot and intra-shot

to ensure consistency across the whole video. Specifically, we prepare the training data with four fundamental stages: (1) analyzing the relationship between adjacent shots and generating an inter-shot prompt Ts; (2) deriving the pose prompt descriptions Tp through segment-wise textual matching based on the camera poses; (3) correcting Tp by leveraging visual cues from reference videos; and (4) performing rigorous manual refinement to ensure high data quality.

Given the transition frames F = {fi}Ki=1, we first extract several frames immediately before and after the transition frame fi, and prompt an MLLM to analyze the relative relational changes. Finally, we aggregate all the shot transition relationships into a comprehensive description Ts.

Then we segment the video into distinct sub-clips based on transition frames. For each clip, we then perform camera pose analysis to derive camera motion descriptions via a textmatching approach. Specifically, we segment the full camera pose trajectory into multiple sub-clips. For each sub-clip, we compute the relative pose between the first and last frames: ∆P = [∆R | ∆t] = P0−1Pt, where ∆t and ∆R denote the translational and rotational increments, respectively. For translation, we identify the dominant motion axis with the largest absolute displacement in ∆t, which determines the motion direction; we further discretize the motion speed into fast, normal, or slow based on the translation distance. For rotation, we convert ∆R into Euler angles and determine the

rotation type by the dominant angular component. Additionally, we define arc shot with an explicit rule: a sub-clip is classified as arc shot if the dominant translation axis is x and the dominant rotation component is yaw:

Left Arc Shot : ∆θyaw > 0 and ∆x < 0 (9)

Right Arc Shot : ∆θyaw < 0 and ∆x > 0 (10) Finally, we merge consecutive sub-clips with redundant or equivalent motion labels to form a compact, complete camera-motion description Tp.

Recognizing that pose estimation often yields inaccurate results in complex videos, we leverage reference videos for further rectification. Specifically, we extract keyframes from the reference video V and feed them alongside the pose prompt Tp into Qwen3-VL (Bai et al. 2025b). The model is then instructed to synthesize both modalities to generate an accurate and refined camera motion description. After combining the inter-shot prompt Ts and refined intra-shot prompt with Qwen3-VL (Bai et al. 2025b), we incorporate manual annotation to further enhance the accuracy and reliability of the generated camera motion descriptions Tc.

Based on the aforementioned data construction pipeline, we fine-tune the Qwen3-VL model using camera parameters as inputs and the final camera motion descriptions as labels.

Semantic Fusion. To effectively transfer the camera motions acquired in previous stages to novel scenes, we must consolidate multiple control signals into a unified entity. Specifically, we leverage Qwen3-VL as our multimodal engine to seamlessly integrate the existing pose descriptions Tc, the reference image I, and the user prompt Tu into a comprehensive and cohesive representation.

Adaptive Classifier-Free Guidance Strategy. In video generation with multi-modal control signals, providing informative conditional guidance is crucial for fully unleashing the model’s generative capabilities. Standard Classifier-Free Guidance (CFG) typically interpolates between conditional and unconditional predictions. Considering the unique visual characteristics of the camera grid representation, we specifically tailor its unconditional branch: we set the visual unconditional input to a completely black background and explicitly incorporate the description of a “completely static camera” into the negative text prompt.

Furthermore, camera motion is fundamentally a global spatial transformation that dictates the macroscopic geometric outlines and scene layout of each frame. Consequently, we introduce a coarse-to-fine denoising scheduling strategy. Specifically, we inject the camera grid features during the high-noise regime of the diffusion process to establish the global spatial structure, while introducing other control signals during the low-noise regime to refine local contents and semantic details.

### Experiments

##### Experimental Setups

OmniDirector is trained on top of an in-house image-tovideo diffusion backbone. For training, we curate a largescale dataset of 1.8M internet videos spanning diverse do-

mains (e.g., movies and advertisements). Each video is preprocessed by resizing to 480p. During training, the model is conditioned on the triplet {G,I,T′}. We train the model for 10k optimization steps using a learning rate of 5 × 10−5 and a batch size of 64. During the training phase, data augmentation is performed by applying random colors to the camera grid, and pose jitter is introduced to ensure robust performance.

Evaluation Metrics. To comprehensively evaluate the performance of OmniDirector, we conduct extensive experiments focusing on its camera control capabilities and overall visual quality. Regarding camera control, we employ DPAV3 (Lin et al. 2025) to extract the camera pose trajectories from both the reference and generated videos. We then compute the scale-invariant relative pose errors, namely Relative Rotation Error (RRE) and Relative Translation Error (RTE). RRE is defined as the angular difference between the poses of reference videos and generated ones. RTE measures the directional error of the translation. To further evaluate the robustness of the predictions, we report R-Pre and T-Pre. R-Pre is defined as the percentage of predictions with a relative rotation error below 4◦. Similarly, T-Pre denotes the proportion of predictions with a relative translation error below 20◦.

Since our data contains a large number of complex realworld scenes,poseestimationmethodsinherently sufferfrom errors. To evaluate the performance of OmniDirector more comprehensively and accurately, we utilize Gemini 3.1 Pro and GSB (Good / Same / Bad) to quantify the results.

To evaluate the accuracy of shot transitions, we define the success rate from two dimensions: (1) Tem-Pre (Temporal): Measures the temporal alignment of the transitions. A predicted transition is considered successful if its temporal error relative to the reference transition is less than 3 frames. This metric is evaluated using the TransNet-V2 (Soucek and Lokoc 2024) shot boundary detection model. (2) Sem-Pre (Semantic): Evaluates the semantic consistency of the transitions. For temporally aligned samples, a transitionisdeemedsuccessfuliftheGemini3.1Promodelverifies that its transition type matches the reference.

Furthermore, we utilize the Gemini 3.1 Pro model to quantify the extent of reference video leakage at both the frame and shot levels. And we conducted a GSB pairwise comparison with CamCloneMaster (Luo et al. 2025) to analyze performance across three key areas: camera, quality, and narrative to conduct a more comprehensive assessment.

Evaluation Set. We construct a validation set comprising 1,094 carefully curated samples, which are collected from the web, covering a wide range of domains including advertising, cinematic content, and several complex visual effects. This evaluation set encompasses a diverse range of scenarios, including in-domain and cross-domain reference videoimage pairs, varying resolutions, single-shot and multi-shot sequences, as well as simple and complex camera trajectories. The detailed data distribution is shown in Figure 5.

##### Comparisons with State-of-the-Art Methods

Baselines. We compare the proposed OmniDirector with state-of-the-art camera cloning method CamCloneMas-

###### Detailed Evaluation Set Distribution

Hard 9.3%

Special Effects 8.5%

Medium (Multi-shot) 18.1%

Easy (Single-shot) 23.1%

Medium (Single-shot) 22.6%

Easy (Multi-shot) 18.4%

Easy (Single-shot)

Medium (Single-shot)

Hard

Special Effects

Easy (Multi-shot)

Medium (Multi-shot)

###### Total samples = 1,094

Figure 5: The evaluation set distribution.

ter (Luo et al. 2025), commercial models Seedance2.0 (Seedance et al. 2026) and LTX-LoRA (Cseti 2024). CamCloneMaster trains a Diffusion Transformer (DiT) architecture utilizing cross-paired data. Seedance2.0, representing the latest state-of-the-art commercial model, is evaluated under its omni-reference mode. Additionally, LTX-LoRA builds upon LTX-Video 2.3 (HaCohen et al. 2025), employing Low-Rank Adaptation (LoRA) fine-tuning specifically for camera motion control.

Qualitative comparison. We illustrate the camera motion cloning capabilities of OmniDirector in Figures 1 and 2. As demonstrated,ourmodelfaithfullyclonescameratrajectories while exhibiting strong robustness against variations in scene scale, image resolution, and content discrepancy. Notably, in multi-shot scenarios, OmniDirector not only achieves accurate shot transitions but also strictly preserves the semantic coherence of the transitions.

We provide a qualitative comparison between OmniDirector and baseline approaches. As shown in Figure 6, Seedance2.0 (Seedance et al. 2026) and CamCloneMaster (Luo et al. 2025) struggle with multi-shot videos, yielding motion amplitudes that contradict human visual perception. While LTX-LoRA (Cseti 2024) manages to execute shot transitions, it suffers from substantial content leakage. Conversely, our proposed method seamlessly adapts to various complex scenarios, synthesizing high-quality videos that are highly consistent with the reference camera dynamics.

Quantitative comparison. We report the quantitative comparison results of camera control accuracy in Table 1. Our method outperforms all baselines across all evaluation metrics, demonstrating its effectiveness and its ability to clone cameras from reference videos accurately. Notably, we achieve a relative improvement of 39.3% in translation precision (T-Pre) over the second-best method CamCloneMaster (Luo et al. 2025). This significant gain is because previ-

ous methods struggle with scale inconsistencies between the reference video and the source view, whereas our visual representation provides superior relative scale generalization. Furthermore, most existing methods fail to respond to reference videos with multiple shots, primarily due to the absence of such data in their training sets. While LTX-LoRA (Cseti 2024) can generate occasional shot transitions, the results in Table 1 indicate that this capability is actually an artifact of information leakage rather than genuine camera control.

We quantify the leakage rates of different methods and present the results in Table 1. The results demonstrate that our approach achieves the lowest leakage rate compared to the baselines. This is because baseline methods rely heavily on the visual content of the reference video, whereas our camera grid representation and Prompt Expansion agent completely decouple the camera signal from the reference video. Notably, LTX-LoRA (Cseti 2024) exhibits the most severe leakage, which can be attributed to its insufficient number of fine-tuning parameters.

We report the GSB (Good/Same/Bad) comparison against CamCloneMaster (Luo et al. 2025) in Table 3. As illustrated, our approach demonstrates clear advantages across all three evaluated dimensions. These results strongly indicate that our method achieves superior performance in terms of both overall effectiveness and generation stability.

##### Ablation study

The effect of PE agent design. To validate the effectiveness of our design, we evaluate the prompts generated at different stages of the PE agent module during inference. As shown in Table 2, comparing the “w/o PE Semantic Fusion” and “Full” configurations reveals that fusing multiple signals via the MLLM leads to comprehensive improvements across all metrics. Supported by the visual results in Figure 7, this multimodal signal fusion seamlessly integrates the camera motion with the reference image, yielding highly plausible visual content and camera trajectories. Furthermore, removing the fusion significantly increases the leakage rate. This is becausethefusioneffectivelyreducesconflictsbyintegrating multiple control signals.

The role of the inter-shot prompt is demonstrated in the “w/o Trans PE” row. Compared to the “Full” model, the absence of inter-shot relationship modeling degrades all metrics, with a particularly severe drop in the semantic precision of shot transitions (Sem-Pre). This occurs because, without guidance after a transition, the generated video shifts to a random scene that fails to correlate with the reference camera motion. Ultimately, our multi-stage fusion strategy ensures accurate and harmonious camera cloning.

The effect of Adaptive CFG. The row labeled “w/o AdaCFG” in the table presents the results of replacing adaptive CFG with a full-stage injection of the camera motion signal. As shown, this leads to a significant drop in camera accuracy. Furthermore, the visual results in Figure 7 demonstrate that without AdaCFG, the camera rotates noticeably more slowly. This occurs because the model is forced to process multiple signals simultaneously, resulting in an insufficient response to the camera motion guidance.

###### Single-Shot Video Multi-Shot Video

[Figure 86]

shot1 shot2 shot3

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

LTX+LoRACamCloneSeedance2.0OursRef. Video

OursSeedance2.0Ref. VideoLTX+LoRACamClone

[Figure 100]

[Figure 101]

<shot1> Pull back while panning right. <shot2> Move left while pulling back. <shot3> Descend while pulling back

Push in while ascending, then push in while rolling right, then orbit right while ascending and pushing in.

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

Figure 6: Qualitative Evaluations. The results demonstrate that OmniDirector accurately clones the camera motion and shot transition semantics of the reference video.

[Figure 150]

[Figure 151]

[Figure 152]

| |
|---|
|[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>wide shot|
| |

|[Figure 156]|
|---|

|[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]|
|---|

| | |
|---|---|
|[Figure 160]|[Figure 161]<br><br>[Figure 162]<br><br>medium shot|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

| |
|---|
|[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>first-person view|
| |

Tran. PEw/oVideoRef.Full

Sem.FusionVideow/oRef.Full

|[Figure 168]<br><br>| |[Figure 169]|
|---|---|---|
| |[Figure 170]| |
| | | |

| |[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]| | |
|---|---|---|---|

[Figure 174]

|[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]|
|---|
| |

[Figure 178]

[Figure 179]

|[Figure 180]|
|---|

[Figure 181]

[Figure 182]

|[Figure 183]|[Figure 184]| | |[Figure 185]|
|---|---|---|---|---|

|[Figure 186]<br><br>[Figure 187]|[Figure 188]| |
|---|---|---|

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

w/oAdaCFGVideoRef.Full

|[Figure 198]|
|---|

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Figure 7: Visualization of ablation studies.

|Camera Accuracy| | | |Transition Accuracy| |Leakage Rate| |
|---|---|---|---|---|---|---|---|
|RRE ◦ ↓|R-Pre % ↑|RTE ◦ ↓<br><br>|T-Pre % ↑|Tem-Pre % ↑<br><br>|Sem-Pre % ↑|Frame % ↓<br><br>|Shot % ↓|

Method

Seedance2.0 8.33 56.49 49.98 29.07 4.17 − 4.43 20.90 CamCloneMaster 4.11 74.14 27.45 52.21 2.20 − 1.60 11.59 LTX-LoRA 5.67 66.34 26.96 52.07 38.94 29.55 15.04 56.52 Ours 2.64 83.18 16.84 72.74 96.52 83.79 0.51 3.38

Table 1: Quantitative Evaluations. − denotes that this feature is not supported. In comparison, we achieve superior performance across all evaluation metrics, including camera accuracy, transition accuracy, and leakage rate, which demonstrates the effectiveness of OmniDirector.

|Camera Accuracy<br><br>| | | |Transition Accuracy| |Leakage Rate|
|---|---|---|---|---|---|---|
|RRE ◦ ↓<br><br>|R-Pre % ↑<br><br>|RTE ◦ ↓|T-Pre % ↑|Tem-Pre % ↑<br><br>|Sem-Pre % ↑<br><br>|Shot ↓|

Settings

w/o Semantic Fusion 3.85 78.20 19.90 67.45 94.40 78.30 4.10 w/o Trans PE 2.71 81.50 17.10 71.25 93.35 38.45 3.45 w/o AdaCFG 4.15 74.55 21.41 62.30 94.10 80.20 3.83 Full 2.64 83.18 16.84 72.74 96.52 83.79 3.38

###### Table 2: Ablation studies on our strategies

##### Emergent Camera Understanding

| |(G+S)/T %<br><br>|G/(G+B) %|(G+S)/(B+S)<br><br>|
|---|---|---|---|
|Camera Quality Narrative<br><br>|88.52 95.69 94.26|86.29 90.82 85.71<br><br>|3.19 1.67 1.44|
|Average<br><br>|91.10|87.08<br><br>|2.54|

Fundamentally, the camera grid functions as a visual proxy for spatial movement, sharing highly similar spatiotemporal semantics with RGB videos. Empirically, we discover that this representation effectively unlocks an emergent capability within the video generation model for comprehending camera dynamics. As illustrated in Figure 8, with model parameters frozen during inference, the model robustly deduces and executes camera motions when conditioned on diverse signals, including the raw reference video or a Canny edge video. This strong cross-domain generalization not only mitigates potential errors stemming from inaccurate pose estimation methods (Lin et al. 2025; Keetha et al. 2026), but also enables models to clone complex cinematographic techniques—such as the Hitchcock zoom and distortion depicted in the figure without requiring explicitly curated training data for such specific trajectories.

Table 3: GSB comparisons of ours vs. CamCloneMaster. Notice that the average is weighted with 3 : 1 : 1.

###### Reference Video Condition

Condition Synth Condition Synth

|[Figure 207]|
|---|

[Figure 208]

[Figure 209]

|[Figure 210]|
|---|
|[Figure 211]|

|[Figure 212]|
|---|
|[Figure 213]|

[Figure 214]

[Figure 215]

### Conclusion

Canny Edge Condition

In this paper, we propose OmniDirector, which achieves general multi-shot camera cloning. By rendering the camera parameters within a 3D empty room, we represent the camera motion as grid videos. Based on this, we construct millionscale camera grid-video training pairs to empower OmniDirector with camera control. Furthermore, we design a Hierarchical Prompt Expansion Agent during inference to harmoniously integrate camera motion with other multimodal control signals. Overall, OmniDirector provides an efficient and accessible paradigm for multi-shot camera cloning.

Ref Condition Synth Ref Synth

Condition

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

LimitationsandFutureWork.OmniDirectorcurrentlyemploys direct token concatenation to integrate the multimodal control signals, which struggles to maintain long-term memory and temporal consistency when scaling to significantly longervideosequences.Infuturework,weplantoexploreadvanced temporal memory mechanisms (such as long-context cross-attention modules or memory banks) to enhance the model’s ability.

Figure 8: Emergent zero-shot camera control in OmniDirector. During inference, substituting the camera grid with raw RGB videos or Canny edge sequences effectively drives camera motion, demonstrating robust generalization without any retraining.

### Acknowledgments

We sincerely thank Mingyang Shan, Fanqi Meng, Wanqi Shi and Jiaxin Hu for contributing to the evaluation part.

### References

Bahmani, S.; Skorokhodov, I.; Qian, G.; Siarohin, A.; Menapace, W.; Tagliasacchi, A.; Lindell, D. B.; and Tulyakov, S. 2025. AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, 22875–22889.

Bai,J.;Xia,M.;Fu,X.;Wang,X.;Mu,L.;Cao,J.;Liu,Z.;Hu,

- H.; Bai, X.; Wan, P.; et al. 2025a. ReCamMaster: CameraControlled Generative Rendering from A Single Video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14834–14844.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025b. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Bar-Tal, O.; Chefer, H.; Tov, O.; Herrmann, C.; Paiss, R.; Zada, S.; Ephrat, A.; Hur, J.; Liu, G.; Raj, A.; et al. 2024. Lumiere: A Space-Time Diffusion Model for Video Generation. In SIGGRAPH Asia 2024 Conference Papers, 1–11.

Blattmann, A.; Dockhorn, T.; Kulal, S.; Mendelevitch, D.; Kilian, M.; Lorenz, D.; Levi, Y.; English, Z.; Voleti, V.; Letts, A.; et al. 2023. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv preprint arXiv:2311.15127.

Chen, H.; Zhang, Y.; Cun, X.; Xia, M.; Wang, X.; Weng, C.; and Shan, Y. 2024. VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models. In ProceedingsoftheIEEE/CVFconferenceoncomputervision and pattern recognition, 7310–7320.

- Chen, W.; Ji, Y.; Wu, J.; Wu, H.; Xie, P.; Li, J.; Xia, X.; Xiao, X.; and Lin, L. 2023a. Control-A-Video: Controllable Textto-Video Diffusion Models with Motion Prior and Reward Feedback Learning. arXiv preprint arXiv:2305.13840.
- Chen, X.; Wang, Y.; Zhang, L.; Zhuang, S.; Ma, X.; Yu, J.; Wang, Y.; Lin, D.; Qiao, Y.; and Liu, Z. 2023b. Seine: Shortto-Long Video Diffusion Model for Generative Transition and Prediction. In The Twelfth International Conference on Learning Representations.

Cseti. 2024. LTX2.3-22B_IC-LoRA-Cameraman_v1. https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRACameraman_v1. Hugging Face Model Repository.

Esser, P.; Kulal, S.; Blattmann, A.; Entezari, R.; Müller, J.; Saini, H.; Levi, Y.; Lorenz, D.; Sauer, A.; Boesel, F.; et al. 2024. Scaling Rectified Flow Transformers for HighResolution Image Synthesis. In Forty-first international conference on machine learning.

Girdhar, R.; Singh, M.; Brown, A.; Duval, Q.; Azadi, S.; Rambhatla, S. S.; Shah, A.; Yin, X.; Parikh, D.; and Misra,

- I. 2024. Factorizing Text-to-Video Generation by Explicit Image Conditioning. In European Conference on Computer Vision, 205–224. Springer.

Guo, Y.; Yang, C.; Rao, A.; Liang, Z.; Wang, Y.; Qiao, Y.; Agrawala, M.; Lin, D.; and Dai, B. 2023. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. arXiv preprint arXiv:2307.04725.

HaCohen, Y.; Brazowski, B.; Chiprut, N.; Bitterman, Y.; Kvochko, A.; Berkowitz, A.; Shalem, D.; Lifschitz, D.; Moshe, D.; Porat, E.; Richardson, E.; Shiran, G.; Chachy,

- I.; Chetboun, J.; Finkelson, M.; Kupchick, M.; Zabari, N.; Guetta, N.; Kotler, N.; Bibi, O.; Gordon, O.; Panet, P.; Benita, R.; Armon, S.; Kulikov, V.; Inger, Y.; Shiftan, Y.; Melumian, Z.; and Farbman, Z. 2025. LTX-2: Efficient Joint AudioVisual Foundation Model. arXiv preprint arXiv:2601.03233.

He, H.; Xu, Y.; Guo, Y.; Wetzstein, G.; Dai, B.; Li, H.; and Yang, C. 2024. CameraCtrl: Enabling Camera Control for Text-to-Video Generation. arXiv preprint arXiv:2404.02101.

He, H.; Yang, C.; Lin, S.; Xu, Y.; Wei, M.; Gui, L.; Zhao, Q.; Wetzstein, G.; Jiang, L.; and Li, H. 2025. CameraCtrl II: Dynamic Scene Exploration via Camera-Controlled Video Diffusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 13416–13426.

Ho, J.; Chan, W.; Saharia, C.; Whang, J.; Gao, R.; Gritsenko, A.; Kingma, D. P.; Poole, B.; Norouzi, M.; Fleet, D. J.; et al. 2022a. Imagen Video: High Definition Video Generation with Diffusion Models. arXiv preprint arXiv:2210.02303.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Diffusion Probabilistic Models. Advances in neural information processing systems, 33: 6840–6851.

Ho, J.; Salimans, T.; Gritsenko, A.; Chan, W.; Norouzi, M.; and Fleet, D. J. 2022b. Video Diffusion Models. Advances in neural information processing systems, 35: 8633–8646.

Hu, L. 2024. Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8153–8163.

Hu, T.; Zhang, J.; Yi, R.; Wang, Y.; Huang, H.; Weng,

- J.; Wang, Y.; and Ma, L. 2024. MotionMaster: TrainingFree Camera Motion Transfer for Video Generation. arXiv preprint arXiv:2404.15789.

Kannala, J.; and Brandt, S. S. 2006. A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses. IEEE transactions on pattern analysis and machine intelligence, 28(8): 1335–1340.

Keetha, N.; Müller, N.; Schönberger, J.; Porzi, L.; Zhang, Y.; Fischer, T.; Knapitsch, A.; Zauss, D.; Weber, E.; Antunes, N.; et al. 2026. MapAnything: Universal Feed-Forward Metric 3D Reconstruction. In 2026 International Conference on 3D Vision (3DV), 499–509. IEEE.

Li, X.; Lai, Z.; Xu, L.; Qu, Y.; Cao, L.; Zhang, S.; Dai, B.; and Ji, R. 2024. Director3D: Real-World Camera Trajectory and 3D Scene Generation from Text. Advances in neural information processing systems, 37: 75125–75151.

Lin, H.; Chen, S.; Liew, J.; Chen, D. Y.; Li, Z.; Shi, G.; Feng, J.; and Kang, B. 2025. Depth Anything 3: Recovering the Visual Space from Any Views. arXiv preprint arXiv:2511.10647.

Ling, P.; Bu, J.; Zhang, P.; Dong, X.; Zang, Y.; Wu, T.; Chen, H.; Wang, J.; and Jin, Y. 2024. MotionClone: Training-Free Motion Cloning for Controllable Video Generation. arXiv e-prints, arXiv–2406.

Luo, Y.; Shi, X.; Bai, J.; Xia, M.; Xue, T.; Wang, X.; Wan, P.; Zhang, D.; and Gai, K. 2025. CamCloneMaster: Enabling Reference-Based Camera Control for Video Generation. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, 1–10.

Luo, Y.; Shi, X.; Zhuang, J.; Chen, Y.; Liu, Q.; Wang, X.; Wan, P.; and Xue, T. 2026. ShotStream: Streaming MultiShot Video Generation for Interactive Storytelling. arXiv preprint arXiv:2603.25746.

Ma, X.; Wang, Y.; Chen, X.; Jia, G.; Liu, Z.; Li, Y.-F.; Chen, C.; and Qiao, Y. 2025. Latte: Latent Diffusion Transformer for Video Generation. Transactions on Machine Learning Research.

Meng, Z.; Che, J.; Wei, B.; and Cao, X. 2026a. Make a Game: A Novel Paradigm for Interactive Game Rendering. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1026– 1030. IEEE.

Meng, Z.; Liu, J.; Liu, Y.; Tong, C.; Liu, X.; Zhang, Y.; Xu,

- Y.;andWan,P.2026b. ARGUS:StackedMulti-ViewIdentity Mosaic Injection for Subject-Preserving Video Generation. arXiv:2606.11670.

Mou, C.; Wang, X.; Xie, L.; Wu, Y.; Zhang, J.; Qi, Z.; and Shan, Y. 2024. T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, 4296–4304.

Peebles, W.; and Xie, S. 2023. Scalable Diffusion Models with Transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 4195–4205.

Polyak, A.; Zohar, A.; Brown, A.; Tjandra, A.; Sinha, A.; Lee, A.; Vyas, A.; Shi, B.; Ma, C.-Y.; Chuang, C.-Y.; et al. 2024. Movie Gen: A Cast of Media Foundation Models. arXiv preprint arXiv:2410.13720.

Ren, W.; Yang, H.; Zhang, G.; Wei, C.; Du, X.; Huang, W.; and Chen, W. 2024. ConsistI2V: Enhancing Visual Consistency for Image-to-Video Generation. Transactions on Machine Learning Research.

Seedance, T.; Chen, D.; Chen, L.; Chen, X.; Chen, Y.; Chen,

- Z.; Chen, Z.; Cheng, F.; Cheng, T.; Cheng, Y.; et al. 2026. Seedance 2.0: Advancing Video Generation for World Complexity. arXiv preprint arXiv:2604.14148.

Singer, U.; Polyak, A.; Hayes, T.; Yin, X.; An, J.; Zhang, S.; Hu, Q.; Yang, H.; Ashual, O.; Gafni, O.; et al. 2022. Makea-Video: Text-to-Video Generation without Text-Video Data. arXiv preprint arXiv:2209.14792.

Soucek, T.; and Lokoc, J. 2024. TransNet v2: An Effective Deep Network Architecture for Fast Shot Transition Detection. In Proceedings of the 32nd ACM International Conference on Multimedia, 11218–11221.

Villegas, R.; Moraldo, H.; Castro, S.; Babaeizadeh, M.; Zhang, H.; Kunze, J.; Kindermans, P.; Saffar, M.; and Erhan,

D. 2023. Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions. In 11th International Conference on Learning Representations, ICLR 2023. International Conference on Learning Representations (ICLR).

Wang, Q.; Luo, Y.; Shi, X.; Jia, X.; Lu, H.; Xue, T.; Wang, X.; Wan, P.; Zhang, D.; and Gai, K. 2025. CineMaster: A 3D-Aware and Controllable Framework for Cinematic Textto-Video Generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 1–10.

Wang, Q.; Shi, X.; Li, B.; Bian, W.; Liu, Q.; Lu, H.; Wang,

- X.; Wan, P.; Gai, K.; and Jia, X. 2026. MultishotMaster: A Controllable Multi-shot Video Generation Framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16268–16278. Wang, X.; Yuan, H.; Zhang, S.; Chen, D.; Wang, J.; Zhang,
- Y.; Shen, Y.; Zhao, D.; and Zhou, J. 2023. Videocomposer: Compositional Video Synthesis with Motion Controllability. Advances in Neural Information Processing Systems, 36: 7594–7611.

Wang, Z.; Yuan, Z.; Wang, X.; Li, Y.; Chen, T.; Xia, M.; Luo, P.; and Shan, Y. 2024. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. In ACM SIGGRAPH 2024 Conference Papers, 1–11.

Wu, X.; Gao, B.; Qiao, Y.; Wang, Y.; and Chen, X. 2025. CineTrans: Learning to Generate Videos with Cinematic Transitions via Masked Diffusion Models. arXiv preprint arXiv:2508.11484.

Xing, J.; Xia, M.; Zhang, Y.; Chen, H.; Yu, W.; Liu, H.; Liu, G.; Wang, X.; Shan, Y.; and Wong, T.-T. 2024. Dynamicrafter: Animating Open-Domain Images with Video Diffusion Priors. In European Conference on Computer Vision, 399–417. Springer.

Xu, D.; Nie, W.; Liu, C.; Liu, S.; Kautz, J.; Wang, Z.; and Vahdat, A. 2024. CamCo: Camera-Controllable 3DConsistent Image-to-Video Generation. arXiv preprint arXiv:2406.02509.

Yin, S.; Wu, C.; Liang, J.; Shi, J.; Li, H.; Ming, G.; and Duan, N. 2023. Dragnuwa: Fine-Grained Control in Video Generation by Integrating Text, Image, and Trajectory. arXiv preprint arXiv:2308.08089.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding Conditional Control to Text-to-Image Diffusion Models. In Proceedings of the IEEE/CVF international conference on computer vision, 3836–3847.

Zhang, S.; Wang, J.; Zhang, Y.; Zhao, K.; Yuan, H.; Qin, Z.; Wang, X.; Zhao, D.; and Zhou, J. 2023. I2VGen-XL: HighQuality Image-to-Video Synthesis via Cascaded Diffusion Models. arXiv preprint arXiv:2311.04145.

Zhao, R.; Gu, Y.; Wu, J. Z.; Zhang, D. J.; Liu, J.-W.; Wu, W.; Keppo, J.; and Shou, M. Z. 2024. MotionDirector: Motion Customization of Text-to-Video Diffusion Models. In European Conference on Computer Vision, 273–290. Springer.

Zheng, G.; Li, T.; Jiang, R.; Lu, Y.; Wu, T.; and Li, X. 2024. CamT2V: Camera-Controlled Image-to-Video Diffusion Model. arXiv preprint arXiv:2410.15957.

