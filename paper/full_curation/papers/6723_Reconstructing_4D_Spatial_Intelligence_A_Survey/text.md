## Reconstructing 4D Spatial Intelligence: A Survey

Yukang Cao, Jiahao Lu, Zhisheng Huang, Zhuowen Shen, Chengfeng Zhao, Fangzhou Hong, Zhaoxi Chen, Xin Li, Wenping Wang, Yuan Liu, Ziwei Liu

### arXiv:2507.21045v2[cs.CV]3Aug2025

Abstract—Reconstructing 4D spatial intelligence from visual observations has long been a central yet challenging task in computer vision, with broad real-world applications. These range from entertainment domains like movies, where the focus is often on reconstructing fundamental visual elements, to embodied AI, which emphasizes interaction modeling and physical realism. Fueled by rapid advances in 3D representations and deep learning architectures, the field has evolved quickly, outpacing the scope of previous surveys. Additionally, existing surveys rarely offer a comprehensive analysis of the hierarchical structure of 4D scene reconstruction. To address this gap, we present a new perspective that organizes existing methods into five progressive levels of 4D spatial intelligence:

(1) Level 1 – reconstruction of low-level 3D attributes (e.g., depth, pose, and point maps); (2) Level 2 – reconstruction of 3D scene components (e.g., objects, humans, structures); (3) Level 3 – reconstruction of 4D dynamic scenes; (4) Level 4 – modeling of interactions among scene components; and (5) Level 5 – incorporation of physical laws and constraints. We conclude the survey by discussing the key challenges at each level and highlighting promising directions for advancing toward even richer levels of 4D spatial intelligence. To track ongoing developments, we maintain an up-to-date project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence.

Index Terms—4D spatial intelligence, low-level cues, scene reconstruction, dynamics modeling, interactions, physics, video

✦

1 INTRODUCTION

# T

HE automatic reconstruction of 4D spatial intelligence using machine learning or deep learning techniques

has long been a crucial and challenging problem in computer vision. By capturing both the static configurations and dynamic changes over time, 4D spatial intelligence shall provide a comprehensive representation and understanding of the spatial environments that integrate the threedimensional geometric structures with their temporal evolution. This field has attracted significant attention due to its wide range of applications in video games [1], movies [2], and immersive experiences (AR/VR) [3], [4], where highfidelity 4D scenes serve as the foundation for delivering realistic user experiences. Beyond these applications that primarily focus on the fundamental components of 4D spatial intelligence – namely low-level cues such as depth, camera pose, point map, and 3D tracking, as well as scene composing elements and dynamics – spatial intelligence also plays a pivotal role in advancing embodied AI [5], [6], [7] and world models [8]. These latter domains place a strong emphasis on the interactions among scene components and the physical plausibility of the reconstructed

- • This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP20221-0012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). (Corresponding author: Ziwei Liu.)
- • Y. Cao, F. Hong, Z. Chen, and Z. Liu are with S-Lab, College of Computing and Data Science, Nanyang Technological University, Singapore

639798. E-mail: yukang.cao@ntu.edu.sg, fangzhou.hong@ntu.edu.sg, frozen.burning@gmail.com, ziwei.liu@ntu.edu.sg J. Lu, C. Zhao, Y. Liu are with Intelligent Graphics Lab, The Hong Kong University of Science and Technology. E-mail: lujiahao@mail.ustc.edu.cn, zhaochf.afterjourney@gmail.com, yuanly@ust.hk Z. Shen, Z. Huang, X. Li, and W. Wang are with Texas A&M University. E-mail: mickshen@tamu.edu, hzs@tamu.edu, xinli@tamu.edu, wenping@tamu.edu

environments.

In recent years, techniques for reconstructing 4D spatial intelligence have seen rapid advancements. Several surveys [9], [10] have provided valuable perspectives from various angles and have highlighted persistent challenges in the field. For example, [11], [12], [13] reviewed the recent process in deep stereo matching to obtain the low-level scene information; [14], [15], [16] offered a comprehensive overview of advances in 3D scene reconstruction, covering a range of input modalities and diverse 3D representations; [9], [10] classified dynamic 4D scene reconstruction methods into categories based on their core architectural principles. However, the field has advanced considerably, driven by the emergence of novel 3D representations [17], [18], [19], high-quality video generation techniques [20], [21], [22] that provide richer input data, and more efficient models capable of delivering superior reconstruction quality. Despite these strides, additionally, none of the existing surveys thoroughly examines the different compositional levels of the dynamic 4D scenes, nor do they offer a detailed analysis of their respective developments and open challenges. This would potentially lead to a fragmented understanding that overlooks critical components. These gaps highlight the need for a comprehensive, up-to-date survey that systematically categorizes 4D spatial intelligence into distinct levels, consolidates recent advancements, and maps the evolving landscape of 4D scene reconstruction.

Driven by this urgent situation, we categorize the existing methods for reconstructing 4D spatial intelligence into five levels and provide a structured overview of their respective advances:

• Level 1 – reconstruction of low-level 3D cues.

At Level 1, the system targets the reconstruction of fundamental 3D cues – namely, depth, camera pose, point maps, and 3D tracking. These low-level cues define the

[Figure 1]

- Fig. 1. Classification of 4D spatial intelligence by level. Specifically, in this survey, we categorize the methods of reconstructing 3D spatial intelligence from video into five levels: (1) low-level 3D cues, (2) 3D scene components, (3) 4D dynamic scenes, (4) modeling of interactions among scene components, and (5) incorporation of physical laws and constraints.

core structure of a 3D scene. Traditionally, this task has been broken down into separate subfields such as keypoint detection [23], [24], [25] and matching [26], [27], [28], [29], robust estimation [28], [30], Structure-from-Motion (SfM) [31], [32], [33], [34], Bundle Adjustment (BA) [35], [36], [37], [38], and dense Multi-View Stereo (MVS) [39], [40], [41], [42], [43]. Recent approaches like DUSt3R [44] and its followups [45], [46], [47], [48] aim to jointly solve these subproblems, enabling more integrated and collaborative reasoning. Building on transformer-based advances [49], [50], [51], [52], [53], VGGT [54] further introduces an end-to-end

framework that rapidly estimates these low-level 3D cues within seconds.

• Level 2 – reconstruction of 3D scene components.

On top of level 1, level 2 methods move beyond basic 3D cues to reconstruct individual scene elements such as humans, objects, and buildings. While some methods may involve the composition or spatial arrangement of these elements within a scene, they generally do not model or enforce the correctness of their interactions. Recent methods for this level leverage the innovations in 3D representations

like NeRF [55], 3D Gaussians [56], and meshes (DMTET [18], FlexiCube [57]) to improve the reconstructed fine-scale details, rendering efficiency, and global structural coherence, making the results ideal for photorealistic scene reconstruction and immersive virtual experiences.

• Level 3 – reconstruction of 4D dynamic scenes.

Level 3 incorporates dynamics into reconstructed 4D scenes, marking a key step toward enabling the “bullet time” experience of 4D spatial intelligence and delivering more immersive visual content. Existing approaches can generally be categorized into two main directions. The first line of work [58], [59], [60], [61], [62] reconstructs a static canonical radiance field and models temporal changes through learned deformations over time. In contrast, the other type of techniques [63], [64], [65], [66], [67], [68], [69] encode time directly as an additional parameter within the 3D representation, allowing for continuous modeling of scene dynamics.

• Level 4 – modeling of interactions among scene

components.

Advancing beyond the reconstruction of low-level cues, scene elements, and dynamics, Level 4 of spatial intelligence enters a more mature phase – focusing on modeling interactions between different components within a scene. Given that humans are often the central agents of interaction, early works [70], [71], [72], [73], [74] primarily concentrated on capturing the motion of humans and manipulated objects observable in video inputs. With recent progress in 3D representations, recent methods [75], [76], [77], [78], [79], [80] have achieved more accurate reconstructions of both human and object appearances. Furthermore, the study of humanscene interactions [81], [82], [83], [84], [85] has gained traction, serving as a foundational step toward constructing comprehensive world models.

• Level 5 – Incorporation of physical laws and con-

straints.

While Level 4 systems are capable of modeling interactions between different scene components, they typically overlook underlying physical principles such as gravity, friction, and pressure. As a result, these methods may fall short in applications like embodied AI [5], [6], [7], where the goal is often to enable real-world robots to imitate actions and interactions observed in videos. Level 5 systems address this limitation by focusing on enforcing physical plausibility within reconstructed 4D scenes. Recent approaches [86], [87], [88], leveraging platforms such as IsaacGym [89] and reinforcement learning techniques [90], [91], [92], have demonstrated the ability to learn and replicate human-like skills directly from video data, marking a significant step toward physically grounded spatial intelligence. Beyond human-related applications, the physical modeling of general 3D objects, such as simulating object deformation, collisions, and dynamics, as well as physical scenes, has also become an active area of research [93], [94], [95], expanding the scope and applicability of Level 5 reconstruction systems.

Scope. This survey primarily focuses on approaches for reconstructing 4D scenes from video inputs. Specifically,

we examine key developments and representative works across our defined Levels 1 through 5 of the 4D sptial intelligence. The papers reviewed are predominantly drawn from leading conferences and journals in computer vision and computer graphics, along with select preprints released on arXiv in 2025. Our selection criteria emphasize relevance to the scope of this survey, with the goal of providing a comprehensive overview of recent rapid progress in the field. We do not include the 3D generation methods [96], [97], [98] and 4D generation approaches [99], [100], [101], [102], [103], [104] based on generative video diffusion models [20], [21], [22], as these methods typically yield a single type of input and have limited direct relevance to 4D reconstruction techniques. Additionally, this survey does not delve into a detailed analysis of various 3D representations. Readers interested in these complementary areas are encouraged to read existing surveys on 4D generation [105], [106], [107], [108] and the evolution of 3D representation methods [10], [15], [109], [110].

Organization. An overview of the different levels of 4D spatial intelligence is illustrated in Fig. 1. In the following sections, we introduce a taxonomy that organizes recent research efforts according to the evolving process of reconstructing five key levels from video inputs: low-level 3D cues (Sec. 2), basic 3D scene components (Sec. 3), dynamic 4D scenes (Sec. 4), interaction between scene components (Sec. 5), and physics modeling (Sec. 6). The overall structure of the survey is summarized in Fig. 1. Finally, in Sec. 7, we critically reflect on current methodologies, identify open challenges at each level of spatial intelligence, and discuss future directions for advancing 4D spatial intelligence beyond these five defined levels.

#### 2 LEVEL 1 – LOW-LEVEL 3D CUES

Depth, camera pose, and 3D tracking are commonly regarded as low-level cues in 3D scene modeling. These parameters capture the fundamental geometric and positional structure of the environment, forming the basis for higherlevel tasks such as object reconstruction, scene composition, and physical interaction modeling. In this sense, they function similarly to pixels and edges in 2D vision. As such, we define the reconstruction of these elements as level 1 of 4D spatial intelligence. The paradigms of the methods for obtaining these low-level cues from videos are illustrated in Fig. 3. They can be further categorized according to their respective objectives and the type of input videos.

###### 2.1 Depth estimation

Video-based depth estimation aims to generate accurate and temporally consistent depth maps from RGB video sequences. Early approaches typically relied on inferencetime optimization to align depth across frames [111], or employed self-supervised warping using estimated egomotion and optical flow [112], [113], [114], often further enhanced by test-time refinement [115], [116]. While effective, these methods are computationally expensive and heavily dependent on the accuracy of pose and flow estimations. To address these challenges, feed-forward architectures have been introduced. Cost-volume–based methods construct 3D

[Figure 2]

- Fig. 2. The paradigms of methods for reconstructing low-level cues from video input. (I) Video-based depth reconstruction methods recently leverage the diffusion model to obtain the depth maps; (II) Methods for reconstructing camera pose from video input typically employ the neural network to infer the camera pose based on the encoded image features; (III) 3D tracking methods uses point tracker and transformers to achieve 3D tracking from video input; (IV) Recent methods, such as VGGT, apply DINO to extract the features and then train transformerbased DPT heads to infer the unified 3D attributes. “Enc.”, “Dec.”, “Spt. Grid”, “Qry. Points”, and “Cam.” denote “Encoder”, “Decoder”, “Supporting Grid”, “Query Points”, and “Camera Head” correspondingly.

matching volumes to enforce temporal coherence [117], [118], [119], [120], while flow-guided approaches integrate optical flow cues directly [121], [122]. Recurrent models leverage temporal recurrence to iteratively refine predictions across frames [123], [124], and attention-based mechanisms dynamically reweight temporal features [125], [126],

- [127]. Other notable feed-forward systems include [126],
- [128], [129], [130], [131]. More recently, large-scale pretraining and diffusion-based frameworks have pushed the frontier further. DepthCrafter [132], ChronoDepth [133], and DepthAnyVideo [134] leverage video diffusion models to generate depth sequences directly, while [135] extends the ViT-based Depth Anything V2 [136] for video depth estimation. These models exhibit strong temporal consistency and robust generalization across diverse scenes. Overall, the field has progressed from optimization-heavy, posedependent pipelines to efficient feed-forward networks, and most recently, to pretrained, diffusion-driven models that achieve both high accuracy and temporal coherence.

###### 2.2 Camera pose estimation

Camera pose estimation from RGB videos can be generally solved by Visual Odometry (VO) algorithms, which are widely applied in robotics applications. Classical geometrybased VO methods are typically categorized into two groups: feature-based and direct approaches. Feature-based VO [137], [138], [139], [140], [141] estimates camera motion by detecting and tracking visual features across frames, while direct VO [142], [143], [144], [145] infers motion by minimizing photometric error or applying feature warping [146]. With the advent of deep learning, learning-based VO methods [147], [148], [149] have gained prominence, often outperforming traditional approaches in controlled settings but facing challenges in generalizing to unseen environments. To overcome these limitations, hybrid meth-

ods [150], [151], [152], [153], [154], [155], [156] have been proposed to combine learning-based techniques with geometric insights, leveraging the strengths of both paradigms. More recently, to reduce reliance on manual hyperparameter tuning required by these hybrid methods, further studies [157], [158] have explored reinforcement learning for adaptive decision-making in VO systems. It is also worth noting that VO is closely related to Visual Simultaneous Localization and Mapping (VSLAM), which extends VO by concurrently constructing a map of the environment. Methods that jointly estimate camera pose and dense depth for mapping purposes will be discussed in a later section on unified camera pose and depth estimation from video.

###### 2.3 3D tracking

3D tracking estimation aims to recover the motion of scene elements in dynamic videos, providing temporally coherent correspondences in 3D space. A notable approach in this area is OmniMotion [159], which represents an input video using a quasi-3D canonical volume and performs dense, pixel-wise tracking by establishing bijective mappings between the local input space and the canonical space. Through per-video optimization, it jointly estimates the motion trajectories across the entire sequence, enabling consistent tracking over time. Building upon this framework, OmniTrackFast [160] enhances both computational efficiency and robustness by factorizing the underlying function representation into a local spatiotemporal feature grid, and further improves the model’s expressiveness by introducing non-linear functions into the coupling blocks. In contrast to these optimization-heavy methods, SpatialTracker [161] proposes a feed-forward architecture that supports longrange 3D tracking across videos without the need for testtime optimization, offering a more scalable and efficient alternative. SceneTracker [162] employs an iterative strategy to approximate the optimal 3D trajectory, dynamically indexing and constructing both appearance correlation and depth residual features in parallel. DELTA [163] introduces a coarse-to-fine trajectory estimation strategy, allowing for efficient dense tracking across the entire frame rather than being limited to a sparse set of locations. Seurat [164] derives depth directly from 2D tracking inputs to recover 3D trajectories. TAPIP3D [165] constructs spatio-temporal feature clouds from videos by utilizing depth and camera motion information to project 2D video features into a 3D world space, where the effects of camera movement are effectively neutralized. Recent methods, such as EgoPoints [166], introduces a new benchmark and new metrics for point tracking from egocentric videos. It opens the door for future works. Together, these methods illustrate the evolving landscape of 3D tracking, spanning from optimization-based pipelines to fully end-to-end learning systems.

###### 2.4 Unified 3D cues modeling

Accurate 4D scene reconstruction requires not only highquality geometry but also coherent spatial-temporal understanding across frames. To achieve this, recent research has moved toward unified frameworks that jointly estimate depth, camera pose, and even 3D tracking from video. These joint approaches reduce the inconsistencies and ambiguities

that often arise when these components are estimated independently, leading to more robust and temporally consistent reconstructions. The field has seen a wide range of solutions—from optimization-based pipelines that refine predictions per video, to end-to-end feed-forward architectures designed for efficiency and scalability. In this section, we review representative methods that integrate one or more of these key components, highlighting their methodological designs, trade-offs, and contributions to the broader goal of learning-based 4D spatial intelligence.

Unifying depth and camera pose estimation Jointly predicting depth and camera pose from monocular video is a foundational step toward full 4D scene reconstruction. Many recent methods tackle this challenge by leveraging monocular depth priors and applying per-video optimization to enforce temporal and geometric consistency. For example, Robust-CVD [167] applies flexible deformation splines for large-scale geometric alignment and introduces geometry-aware filtering to refine high-frequency depth details; CasualSAM [168] fine-tunes a monocular depth model on individual video sequences to jointly optimize both depth and camera pose. MegaSaM [169] adapts traditional visual SLAM paradigms to handle dynamic scenes, maintaining dense map construction and pose estimation. More recently, DUSt3R [44] proposes a unified framework that simultaneously predicts depth, camera pose, and a dense point map, allowing mutual refinement across these outputs. Building on this idea, MonST3R [46] and Align3R [47] extend the approach to dynamic scenes by fine-tuning on motion-rich datasets and producing temporally consistent point trajectories. Align3R further integrates monocular depth priors to enhance reconstruction quality, although both methods still rely on global alignment during postprocessing. Easi3R [170] is a simple yet efficient trainingfree method for 4D reconstruction. It adapts attention during inference, eliminating the need for pre-training from scratch or network fine-tuning. GeometryCrafter [171] introduces a pointmap Variational Autoencoder (VAE) that learns a latent space independent of video-specific distributions, enabling effective pointmap encoding and decoding for accurate 3D/4D reconstruction and camera parameter estimation. In contrast to optimization-heavy pipelines, Spann3R [172] adopts a feed-forward approach that enables continuous 4D reconstruction via a spatial memory mechanism. CUT3R [173], on the other hand, leverages a compressed state representation that not only encodes observed information but also supports the inference of unobserved structures. Point3R [174] and StreamVGGT [175] further enhance the capabilities of streaming 3D/4D reconstruction. Pi3 [176] introduces an innovative feed-forward neural network that fundamentally changes visual geometry reconstruction by removing the dependency on a fixed reference view. Several diffusion-based methods [177], [178], [179] have also demonstrated strong performance on this task. Aether [177], Geo4D [178], and UniGeo [179] can simultaneously predict high-quality depth and accurate camera poses, benefiting from their denoising-based designs.

Unifying depth, camera pose, and 3D tracking Recent methods have made significant progress toward jointly estimating video depth, camera pose, and 3D tracking. Uni4D [180] adopts a multi-stage optimization framework

that integrates multiple pretrained models to handle both static and dynamic 3D reconstruction. BA-Track [181] disentangles camera-induced motion from object motion using a 3D point tracker, enabling robust bundle adjustment across the entire scene, and further enforces temporal depth consistency through scale-map-based post-processing. Built upon DUSt3R [44], several recent works extend its capability toward dynamic 3D reconstruction and tracking. Stereo4D [182] leverages large-scale training on temporally consistent 3D point clouds to recover long-term pseudomotion trajectories along with depth and camera pose. DPM [183] introduces time into the representation, leading to multiple possible spatial-temporal reference frames for defining point maps. The authors identify a minimal and sufficient subset of these reference combinations that can be regressed by a network to address the aforementioned sub-tasks. St4RTrack [184] jointly predicts point maps at a single timestamp within a unified world coordinate system and chains these predictions across time to reconstruct longrange trajectories, effectively integrating reconstruction and tracking. POMATO [185] combines pointmap matching with temporal motion modeling, establishing cross-view pixelto-pointmap correspondences and introducing a temporal module to enforce scale consistency and improve 3D point tracking. D2USt3R [186] directly regresses 4D pointmaps that represent both static and dynamic scene geometry, explicitly modeling spatial and temporal aspects to provide dense spatio-temporal correspondences beneficial for downstream tasks. Zero-MSF [187] presents a joint geometry-andmotion estimation framework supported by a large-scale data generation pipeline, which produces 1M annotated samples from diverse synthetic scenes. It also identifies and adopts an effective parameterization for scene flow through systematic evaluation. In contrast to the optimization-heavy methods above, TracksTo4D [188] operates in an unsupervised and feed-forward manner. It takes 2D point tracks as input and predicts full 4D structures from casually captured videos without requiring ground truth or supervision. Leveraging recent advancements in transformers [189], VGGT [54] introduces an end-to-end architecture capable of efficiently predicting low-level 3D cues within seconds. SpatialTrackerV2 [190] proposes a unified, feedforward 3D point tracker that integrates point tracking, monocular depth, and camera pose estimation. It decomposes worldspace motion into scene geometry, camera ego-motion, and pixel-wise object motion, with a fully differentiable, endto-end architecture that scales across synthetic, RGB-D, and in-the-wild data.

On the other hand, jointly estimating depth and camera pose from dynamic RGB video remains an open challenge due to occlusions, object motion, and scene complexity. Several methods address this by enforcing temporal and geometric consistency within short frame windows [150], [191], [192], [193], [194], [195], yielding locally consistent results. However, these methods often suffer from accumulated errors over longer sequences. In response, dense visual SLAM methods [196], [197], [198], [199] extend traditional SLAM pipelines to produce globally consistent and dense depth maps instead of sparse point clouds. Notably, recent SLAM systems [200], [201], [202], [203] adopt 3D Gaussian Splatting [19] as a scene representation, benefiting from its

real-time rendering and high-fidelity geometry. Please refer to the survey in [204] for a comprehensive overview.

Recent studies have also revisited classical Structurefrom-Motion (SfM) techniques within a differentiable framework. FlowMap [205], for example, presents an end-toend model that jointly estimates camera intrinsics, extrinsics, and depth maps from monocular videos. Similarly, DUSt3R [44] and MASt3R [206] reformulate pairwise reconstruction as point map regression, simultaneously inferring depth, pose, intrinsics, and pixel correspondences. Extensions such as MASt3R++ [45], [207] further improve dense multi-view stereo pipelines. Building on these ideas, recent works [48], [172], [208], [209], [210], [211] have demonstrated fast and accurate joint estimation of camera pose and depth across entire video sequences, signaling significant progress toward scalable 4D reconstruction systems.

#### 3 LEVEL 2 – 3D SCENE COMPONENTS

While low-level cues provide the geometric and positional foundations necessary for understanding the scene’s layout, they are typically insufficient for capturing higherlevel semantics and object-level structures. Moving forward from this, level 2 methods focus on recovering the detailed representations of individual elements, such as objects, humans, and architectural structures, as well as their spatial arrangement within a scene. An overview of representative approaches in this category is shown in Fig. 2. We specifically categorize the approaches into two types: (1) smallscale 3D object/scene reconstruction and (2) large-scale 3D scene reconstruction. In the following subsection, we begin by reviewing the key 3D representations that underpin these methods.

###### 3.1 Scene representations

In recent years, a variety of 3D scene representations have been developed and adopted for static surface reconstruction. In this subsection, we first highlight the most commonly used representations and explain how they are typically integrated into modern neural network architectures.

Point cloud. Point clouds, composed of discrete 3D points often enriched with attributes like color and normals, are a fundamental representation for surface geometry. Beyond basic points, surfels, points with orientation and radius, offer a richer representation and support differentiable rendering [19], [212], [213]. This enables optimization of point properties such as position, color, and size. Recent methods like Neural Point-based Rendering [1], [214], SynSin [215], Pulsar [216], [217], and ADOP [218] incorporate learnable features to better capture appearance and shape. Others, including FVS [219], SVS [220], and FWD-Transformer [221], further improve rendering by warping point-based features to novel views for color prediction, enhancing reconstruction quality.

Meshes. Meshes, formed by connecting vertices with edges and polygons (typically triangles or quads), are widely used for representing complex 3D shapes due to their flexibility and computational efficiency [222], [223]. To utilize this kind of 3D representation, neural networks are generally designed to predict vertex positions [224], [225], while textured appearances are commonly achieved using per-vertex

colors or UV-mapped textures. To integrate meshes into 3D reconstruction pipelines, differentiable mesh rendering is essential. Techniques such as OpenDR [226], Neural Mesh Renderer [227], Soft Rasterizer [228], and Paparazzi [229] support gradient-based learning, while general-purpose renderers like Mitsuba [230] and Taichi [231] enable meshbased differentiable rendering via automatic differentiation. Neural radiance field (NeRF). Neural Radiance Fields (NeRF) [17] represent 3D scenes as continuous volumetric fields instead of discrete geometry like point clouds or meshes. By using a neural network (typically an MLP), NeRF maps a 3D point and viewing direction to color and density values. Rendering is achieved through volumetric integration along camera rays, where sampled densities and colors are accumulated to produce the final pixel color using differentiable volume rendering [232]. This implicit representation has enabled high-quality novel view synthesis and has seen widespread applications in editing [233], inverse rendering [234], camera pose estimation [235], and avatar reconstruction [236].

Extensions like NeuS [237] and VolSDF [238] integrate signed distance functions (SDFs) into NeRF to provide sharper surface definitions. These methods convert signed distances into opacities using differentiable mappings and optimize the scene by minimizing photometric losses. NeRFs [17], [239], [240], [241], [242], [243] have become a versatile tool across a wide range of tasks in computer vision and graphics. They have been successfully applied to scene editing [233], [244], camera pose optimization [235], [245], inverse rendering [234], [246], generalization to unseen scenes [247], [248], acceleration [249], and free-viewpoint video generation [250], [251]. NeRFs also support avatar modeling tasks such as face and body reenactment [236], [252]. Beyond graphics, their adaptability has also extended to fields like robotics [253], medical imaging [254], and even astronomy [255], highlighting the broad applicability of neural volumetric representations.

3D Gaussian splatting (3DGS). 3D Gaussian Splatting (3DGS) [19] offers an efficient alternative to NeRFs by directly optimizing a set of 3D Gaussians, each defined by a position µ, opacity α, anisotropic covariance Σ ∈ R3×3, and spherical harmonic (SH) coefficients SH to model viewdependent color c:

##### G = {(µi,Σi,ci,αi)}Ni=1, (1)

where N is the number of 3D Gaussian primitives. Note that Spherical Harmonic (SH) is used for controlling the color of each Gaussian to accurately capture the view-dependent appearance of the scene. Unlike NeRFs that rely on MLPs, 3DGS represents scenes with explicit primitives, enabling high-resolution rendering with significantly faster training.

###### 3.2 Small-scale 3D object/scene reconstruction

Given the significant advantages of 3D reconstruction from video sequences, early research efforts focused substantially on regressing a mesh as a unified representation for geometric surface reconstruction. A primary task within this domain involves reconstructing the surface given a fixed, bounded video input. Pioneering methods typically relied on Structure-from-Motion (SfM) [31], [32], [33], [34] and

[Figure 3]

Fig. 3. The paradigms of methods for reconstructing 3D scene components from video input. 3D reconstruction methods for smallscale and large-scale scenes often share similar architectures, differing primarily in the spatial extent they handle. As shown in the left panel (Image source: MipNeRF360 [256]), small-scale scenes correspond to the unaffected domain. large-scale scenes additionally incorporate a contracted domain. Examples illustrating both scene types are provided in the right panel.

Multi-View Stereo (MVS) [39], [40], [41], [42], [43] pipelines to predict dense depth maps, which were subsequently fused into surfaces using techniques like Poisson reconstruction [257] or Delaunay triangulation [258]. For instance, COLMAP [39], [259] employs pixel-level SIFT features for matching and depth prediction. MVSNet [260] leverages deep neural networks to extract latent features and aggregate depth predictions across multiple frames. However, the depth maps generated by these approaches are frequently noisy, and both Poisson reconstruction [257] and Delaunaybased methods exhibit high sensitivity to noise within the underlying point clouds. This sensitivity often results in reconstructed surfaces of compromised quality.

Building upon these foundations and driven by the development of more efficient 3D representations, recent reconstruction methods derive surfaces directly from implicit volumetric representations, notably Neural Radiance Fields (NeRF) [55] and 3D Gaussian Splatting (3DGS) [56]. These approaches are capable of yielding high-quality geometry and superior view synthesis. Specifically, NeRFbased surface reconstruction techniques, such as NeuS [237], VolSDF [238], NeAT [261], and Neuralangelo [262], jointly optimize a Signed Distance Function (SDF) field alongside the radiance field, subsequently extracting a mesh via the Marching Cubes algorithm [263]. Later on, with the development of 3DGS, 2DGS [264], GOF [265], and PGSR [266], propose to obtain a mesh by applying Truncated Signed Distance Function (TSDF) fusion to multi-view depth renderings. SuGaR [267] integrates Gaussians within an SDF field to provide high-quality reconstructions. The field continues to advance rapidly, with several recent techniques further enhancing Gaussian Splatting for surface reconstruction by introducing diverse innovations: recovering high-frequency details at scale [268]; enforcing geometric consistency across multiple viewpoints [269]; utilizing triangulation constraints specifically for indoor scenes (Tri2plane [270]); applying elongation splitting and assimilation strategies for improved accuracy (ESA-GS [271]); inferring Unsigned Distance Functions (UDFs) (GaussianUDF [272]); introducing sorted opacity fields (SOF [273]); leveraging photometric

and reflectance priors [274]; employing learned initialization strategies (QuickSplat [275]); and modifying Gaussian primitive representations [276], [277].

In contrast to these optimization-based methods, feedforward approaches have emerged to directly predict feature volumes via an end-to-end network. From these predicted feature volumes, representations such as signed distance functions (SDF) coupled with radiance fields (e.g., SparseNeuS [278], GenS [279], C2F2NeuS [280], UFORecon [281], SuRF [282], ReTR [283]) or 2D Gaussian representations (LaRa [284]) can be efficiently extracted. These methods demonstrate considerable potential for achieving generalized and real-time surface reconstruction by directly regressing both geometry and appearance in a single forward pass. However, the substantial memory requirements inherent in constructing and processing large feature volumes can impose practical limitations on reconstruction fidelity and scalability.

Reconstructing the environment from a first-person view is also an important application of spatial intelligence from video [285]. Recently, with the advancements of world models and embodied AI, there have been a few attempts in static scene reconstruction from egocentric videos [286]. Specifically, SceneScript [287] utilizes a large language model to predict scene description codes from point clouds, which are generated from visual SLAM. EgoLifter [288] and Photoreal Reconstruction [289] leverage 3DGS to optimize photo-realistic static scenes directly from egocentric videos. OSNOM [290] proposes to achieve scene reconstruction by first lifting 2D observations to 3D, and then mathcing them over various timesteps.

###### 3.3 Large-scale 3D scene reconstruction

Modeling large-scale 3D scenes presents inherent challenges due to the complex, multi-scale nature of the required parameterization and the absence of predefined spatial boundaries. NeRF++ [291] pioneered solutions for this domain by decomposing the radiance field into distinct bounded foreground and inverse sphere-based background components. This foundational approach enabled photorealistic rendering, extending far beyond the immediate camera frustum. Subsequently, Mip-NeRF360 [256] addressed critical issues of aliasing and scale imbalance through integrated cone sampling, a non-linear distortion field, and online distillation, achieving high-fidelity 360° reconstructions. Building upon this, Zip-NeRF [292] effectively integrated MipNeRF360’s anti-aliasing capabilities with accelerated hashgrid representations, achieving comparable quality while training an order of magnitude faster; CityGS [293], [294] and OctreeGS [295] proposed a novel divide-and-conquer training and Level-of-Detail (LoD) strategy to achieve efficiency large-scale training and rendering; CityGS-X [296] further adopted a batch-level multi-task rendering process to achieve more efficient modeling; LODGE [297] recently introduced a hierarchical LoD representation, which iteratively selects optimal subsets of Gaussians based on the camera distance, to make real-time rendering feasible even on memory-constrained devices.

These foundational advances enabled further scaling to vast environments. Partitioning strategies emerged

as a key solution, with Block-NeRF [298] and MegaNeRF [299] decomposing scenes into independent local networks to support neighborhood and city-block navigation. City-NeRF [300] extended this concept through progressive network and dataset expansion, seamlessly integrating satellite-to-street-level perspectives. Concurrently, F2NeRF [301] accommodated arbitrary camera trajectories via perspective warping, while Gaussian splatting approaches like Scaffold-GS [302] achieved efficient view-adaptive rendering by anchoring sparse Gaussians on learned scaffolds.

Complementing these representation and partitioning frameworks, regularization techniques are also proposed to enhance the reconstruction robustness. MonoSDF [303] strengthened outdoor geometry by integrating monocular depth and normal cues within a signed distance function (SDF) formulation. Mixture-of-experts systems like SCALAR-NeRF [304] and Switch-NeRF [305] employ shared decoders or learned gating mechanisms to fuse predictions from multiple local models, establishing a critical pathway toward real-time, appearance-consistent reconstruction at city scales.

Another major advancement in surface reconstruction from static videos lies in the development of online, endto-end systems that support real-time, interactive, and scalable applications. A notable early contribution is NeuralRecon [306], which introduced real-time reconstruction based on TSDF volumes using 3D convolutional GRUs [307]. Building on this foundation, later works incorporated more advanced techniques: TransformerFusion [308] leveraged transformer architectures; Flora [309] improved feature aggregation; and VisFusion [310] introduced visibility-aware fusion and ray-based sparsification. To further reduce reliance on strict photometric consistency, several methods (SimpleRecon [120], DG-Recon [311], CVRecon [312], and FineRecon [313]) begin integrating geometric priors from foundation models, enabling better use of both global and local contextual information. Self-supervised techniques, like MonoSelfRecon [314], have also contributed to this trend. More recently, GeoRecon [315] and DetailRecon [316] have advanced the field further by improving global structural consistency and preserving fine-grained surface details.

#### 4 LEVEL 3 – 4D DYNAMIC SCENES

The static nature of reconstructions from Level 2 methods limits their applicability in real-world, dynamic environments. To address this, Level 3 methods focus on introducing temporal dynamics into the scene, enabling the reconstruction of 4D representations that capture motion and changes over time. There are two popular directions (as illustrated in Fig. 4), which are: (1) reconstruct a canonical space while learning its deformation over time, and (2) extend the original 3D representation by explicitly incorporating time as an additional input. Typically, these approaches can be broadly categorized into two groups based on their primary subjects: general 4D scene reconstruction and human-centric dynamic modeling.

[Figure 4]

Fig. 4. The paradigms of methods for reconstructing dynamic scenes from video input. Methods in this domain typically adopt one of two strategies for temporal modeling: (I) explicitly incorporating time as an additional input to extend a static 3D representation, or (II) reconstructing a canonical 3D space and learning its deformation over time. “Def.” denotes “Deformation module”.

###### 4.1 General 4D scene reconstruction

Surface reconstruction from dynamic videos. Dynamic surface reconstruction from video is a vital area of research with wide-ranging applications in fields such as robotics, virtual reality, and autonomous systems. Early methods [317], [318], [319], [320] typically relied on deforming predefined object templates, which limited their ability to handle complex motions or occlusions. The advent of differentiable rendering [228] has enabled more flexible reconstruction pipelines, allowing systems like LASR [321] and ViSER [322] to model articulated shapes directly from video. Building on this, several approaches have extended NeRF frameworks to dynamic and articulated objects, including BANMo [323], PPR [324], and REACTO [325]. The use of neural implicit 3D representations [55], [326] has further removed the need for template constraints, enabling fully unconstrained reconstruction of dynamic scenes [327], [328], [329], [330]. More recently, with the development of 3DGS, methods such as [331], [332], [333], [334], [335], [336], [337] have incorporated 3DGS, significantly improving reconstruction speed, temporal coherence, and robustness to challenging motion.

Novel view synthesis from dynamic videos. Beyond reconstructing dynamic 4D scenes, both NeRF and 3D Gaussian Splatting (3DGS) have been widely adopted for generating novel viewpoints. This free-viewpoint video delivers immersive visual experiences while enabling the creation of freeze-frame (bullet time) effects [338]. One popular direction is to reconstruct a static canonical radiance field and learn its deformation with time, as introduced by DNeRF [339]. Building upon this idea, several NeRF-based approaches [58], [59], [60], [61], [62] employ scene-specific optimization to model non-rigid motions, dynamic appearances, and complex specular effects. In parallel, recent methods [340], [341], [342], [343], [344], [345], [346], [347], [348] based on 3D Gaussian splatting leverage explicit pointbased representations to directly encode dynamic geometry and appearance, offering improved computational efficiency and easier editing. Rather than relying solely on deformation fields, some approaches [349], [350], [351] model motion using vector fields derived from optical flow [352], providing an alternative and interpretable way to describe temporal dynamics.

Another line of work extends radiance field representations by explicitly incorporating time as an additional

input, enabling true 4D reconstruction. NeRF-based approaches [63], [64], [65], [66], [67], [68], [69] treat time as a learnable parameter, allowing the model to capture temporal changes in geometry and appearance. Some approaches [63], [250] use temporal flow to regularize training, while others [353] apply depth-based warping to synthesize novel views, even under inconsistent depth estimates. Additional efforts [354], [355], [356], [357], [358] enhance temporal modeling by embedding explicit timeaware features, improving both coherence and efficiency in capturing scene dynamics. On the other hand, 3DGS–based frameworks [359], [360], [361] incorporate timestamps as additional Gaussian attributes, enabling real-time and highfidelity dynamic view synthesis through explicit pointbased rendering. However, these methods can struggle with maintaining geometric consistency across time. To overcome this, more recent GS-based approaches [347], [362], [363], [364], [365], [366], [367] represent 4D scenes as temporally evolving trajectories of 3D Gaussians. This formulation improves temporal coherence, enhances tracking robustness, and provides more accurate reconstruction of dynamic geometry.

Feed-forward approaches to dynamic scene reconstruction have opened new possibilities for real-time 4D modeling. As the first approach, MonoNeRF [368] pioneered this direction by introducing a generalizable dynamic radiance field that jointly encodes spatial and temporal features. Similarly, [369] tackles occlusion in dynamic settings without per-scene optimization. FlowIBR [370] reduces optimization time by leveraging pre-training on large static datasets, while [371] shows that strong temporal and geometric consistency can be achieved without tuning appearance for each scene. Most recently, [372] advances the paradigm by aggregating information from all context frames to reconstruct target frames directly, further improving efficiency and generalization. DAS3R [373] combines 3D Gaussian Splatting with DUSt3R to reduce the optimization difficulty typically associated with 3D Gaussian Splatting, and achieves more accurate background reconstruction results. LINO-UniPS [374], built upon VGGT, leverages learnable light register tokens to decouple illumination and normal features. It further introduces a global cross-image attention mechanism to enhance multi-view lighting representation and normal consistency.

###### 4.2 Human-centric dynamic modeling

As illustrated in Fig. 5, human-centric dynamic modeling methods can be grouped into two main categories based on their underlying 3D representation: SMPL-based human mesh recovery and appearance-rich dynamic human modeling. In the following, we will first illustrate SMPL [378], a parametric 3D human template, which forms the basis for human modeling.

SMPL [378]. The parametric human model SMPL [378] represents the 3D shape by incorporating body vertices, joints, face and hands landmarks, and expression parameters. Formally, given the pose parameter θ and shape parameter β, SMPL can map the canonical model with nS vertices to

[Figure 5]

Fig. 5. The illustrations of methods for reconstructing 4D dynamic humans from video input. Human-centric dynamic modeling approaches are generally categorized based on their representations: (I) methods that apply SMPL parametric model as their representation to derive the human pose and shape parameters (image source: Neural Body Fitting [375]), (II) methods that similarly apply SMPL but focus more on the prediction based on egocentric videos (image source: EgoAllo [376]), and (III) appearance-rich non-parametric methods that are capable of reconstructing the textured topologies, such as garments and accessories, from video data (image source: Neural Body [377]).

observation space:

##### M(β,θ) = lbs(T(β,θ),J(β),θ,W), T(β,θ) = T + Bs(β) + Bp(θ),

(2)

where M is the function representing the SMPL model in the observation space, and T gives the transformed vertices. W is the blend weight, Bs and Bp are the shape blend shape function and pose blend shape function, respectively. lbs(·) denotes the linear blend skinning function, corresponding to articulated deformation. It poses T(·) based on the pose parameters θ and joint locations J(β), using the blend weights W, individually for each body vertex:

K

wkGk(θ,jk), (3)

vo = G · vc, G =

k=1

where vc and vo respectively are SMPL vertices under the canonical pose and observation space, wk is the skinning weight, Gk(θ,jk) is the affine deformation that transforms the k-th joint jk from the canonical space to observation space, and K is the number of neighboring joints.

SMPL-X [379] evolves from SMPL to include more face vertices, expression parameters ϕ, and the expression blend shape function Be into the model:

##### M(β,θ,ϕ) = lbs(T(β,θ,ϕ),J(β),θ,W), T(β,θ,ϕ) = T + Bs(β) + Be(ϕ) + Bp(θ).

(4)

SMPL-based human mesh recovery and tracking Human mesh recovery (HMR) from dynamic videos has gar-

nered significant research attention in recent years, facilitated by the use of parametric models like SMPL [378]. Early work approached the problem frame-by-frame using optimization [380], [381] or deep learning [382], [383] to estimate human pose and shape. Progress in this area includes enhancements to camera modeling [384], graphbased or location-aware estimators [385], [386], [387], hybrid optimization and regression [388], [389], kinematic parts and dense correspondences [375], [390], [391], [392], [393], [394], [395], image-aligned features [396], [397], [398], and physical constraints [399], [400]. Building on the success of transformer architectures [401], [402], recent innovations emphasize tokenized representation [403], [404], [405], [406] and scaling up models for human pose and shape estimation to millions of training instances [407]. To capture motion over time, video-based HMR methods integrate temporal information using recurrent networks [408], [409], [410], VAEs [411], and optical flow [412]. Moreover, to tackle the challenges in estimating globally consistent human motion from unconditioned and dynamic cameras, researchers have adopted BERT-like [413] masked pretraining for motion sequence modeling [414], structure-from-motion (SfM) [415], optical flow [416], and simultaneous localization and mapping (SLAM) [417], [418] techniques to holistically model human motion and camera movements in a shared world coordinate.

Egocentric motion tracking. Different from exocentric videos, egocentric videos are captured with head-mounted devices, e.g., smart glasses [419] or VR/AR devices, from the first-person view, carrying rich information of the dynamic world, including wearers themselves. The camera motion in egocentric videos reflects the head movement and can be used as a conditional signal for full-body motion generation [420], [421], typically referred to as one-point body tracking. Additional environmental cues, such as explicit hand detection [376] or implicit video features [422], [423], [424], further enhance the performance of one-point tracking. In VR applications, handheld controllers offer additional hand trajectories, providing extra constraints for “three-point body tracking”. This would make the fullbody motion tracking model grounded [423], [425], [426], [427]. However, both one-point and three-point tracking remain ill-posed problems due to the lack of lower-body observations. To mitigate this, wide field-of-view (FoV) cameras mounted on the head and angled downward are often employed to capture more of the body, improving the accuracy in full-body motion reconstruction [428], [429].

Appearance-rich dynamic human modeling Beyond HMR and interactive behavior capture, reconstructing animatable human avatars from RGB videos has emerged as a prominent research direction [430], encapsulating novel pose animation and novel view synthesis. Early methods relied on explicit geometric representations such as meshes, skeletons and silhouettes to model articulated motions and non-rigid surface deformations [431]. A notable milestone was VideoAvatar [432], which introduced canonical space mapping to decouple pose estimation from geometry and texture learning. Built upon this network, LiveCap [433] further improves the efficiency to achieve real-time performance. However, explicit representations faced limitations due to their dependency on pre-defined avatar templates,

prompting a shift toward implicit neural representations. Pioneering approaches in this domain include regression models for avatar template prediction [434], latent code aggregation across sequential observations [377], and Fourier occupancy fields (FOF) for rapid reconstruction [435]. Further developments extended NeRF to articulated human structures via generative models [436], motion field-based canonical warping [236], and hybrid frameworks integrating volumetric rendering with background reconstruction [85], [437]. Additional strategies explored disentangled representations via graph neural networks [438] and hybrid implicit-explicit representations for spatio-temporally coherent avatars [439]. More recently, 3D Gaussian Splatting (3DGS) [56] has enabled high-fidelity, animatable avatars with fast and flexible rendering [363], [440], [441], [442], [443], setting new standards in both reconstruction quality and view synthesis.

#### 5 LEVEL 4 – INTERACTIONS AMONG SCENE COMPONENTS

Advancing from previous levels, level 4 of spatial intelligence enters a more mature phase. At this level, the methods focus on modeling interactions between different components within a scene. Considering human is often the central subject of interaction, our following discussion emphasizes human-centric interaction modeling. These approaches are also categorized into three groups based on their input types and the underlying representations: SMPL-based humancentric interaction, appearance-rich human-centric interaction, and egocentric human-centric interaction.

###### 5.1 SMPL-based human-centric interaction

Building on human mesh recovery (HMR), recent research has progressed toward capturing 3D interactive behaviors from videos, which can be broadly classified into three categories: human-object interaction (HOI), human-scene interaction (HSI), and human-human interaction (HHI). Examples of different categories is provided in Fig. 6

Human-object-interaction (HOI). Considering the lack of high-quality 3D HOI data with videos [447], early methods tackled 3D HOI by utilizing traditional optimization frameworks to reconstruct human-object spatial arrangements through heuristic contact priors [448], [449].

The emergence of scalable methods for collecting 3D HOI data involving videos [70], [71], [72], [73], [74], [421] prompted learning-based approaches, which present significant advantages in generalization and inference efficiency. Pioneering works proposed to model object proximity relative to the human by learning signed distance fields (SDFs) from data [450], [451], and then operate post-optimization based on the learned field. To enhance robustness and accuracy, particularly under occlusions, methods based on generative models such as normalizing flows [75], [76] learn the distribution of human-object spatial arrangements conditioned on input video, thereby mitigating outlier predictions. A separate strategy augments auxiliary input modality, such as IMU, to facilitate object tracking [77].

However, a persistent limitation of these methods is their reliance on well-reconstructed object template geometries,

[Figure 6]

- Fig. 6. Examples of methods for modeling SMPL-based humancentric interaction. Image source: InterDreamer [444], CIRCLE [445], and BUDDI [446].

restricting their feasibility and applicability across diverse scenarios. To overcome this challenge, recent approaches like HDM [452] and InterTrack [79] proposed to learn geometric correspondences within families of similar object categories using diffusion models [453], [454]. This enabled geometry-agnostic reconstruction of 3D HOI point clouds directly from image frames and facilitated the construction of large-scale synthetic 3D HOI datasets.

Human-scene-interaction (HSI). Extending to full scenes including movable objects and fixed contextual layout, early methods focused on estimating contacts between human and static scenes from image frames [455], [456], which are trained on data with sparse labels and inaccurate 3D scene geometries.

To address the limitations of scale and quality of comprehensive 3D scene modeling in HSI data, GTA-IM [457] constructed synthetic data comprising videos alongside pseudo 3D HSI labels obtained from 3D assets within the game engine. Similarly, CIRCLES [445] integrated real-world motion capture with digital environments via VR applications, while TRUMANS [458] replicated 3D scene assets in reality. These methods provided richer and more accurate 3D labels, enabling reconstruction of HSI from videos to advance into diverse indoor and outdoor contexts involving dynamic objects.

Nevertheless, a significant gap persists between 3D

assets and real-world environments. Jointly reconstructing both humans and dynamic scenes from casual, realworld videos like web footage remains highly desirable. SitComs3D [81] targeted on television show with multiple shots of the same scene. By disentangling human and scene using different representations, SitComs3D [81] expressed the scene as NeRF [17] and estimated human motion within this context. More recently, leveraging advanced low-level 3D attributes prediction models (introduced in Level 1), JOSH [82] jointly recovered human motion, 3D scene structure, camera poses, contacts, and optimized them contextually with physics-based constraints. While ODHSR [83] achieved the same target by holistically representing the human and scene as 3DGS for each frame.

Human-human-interaction (HHI). In the case of multiperson interactions, earlier monocular and sparse multiview systems utilized 3D keypoint heatmaps for multihuman pose estimation [459], [460]. However, these methods ignore geometric constraints and physical contacts, leading to unrealistic results.

To address this issue, datasets [461], [462] and methods [463] introduced instance-level prior and geometric collision loss to obtain physically plausible multi-human interactions from multi-view videos. While BUDDI [446] and HumanInteraction [464] leveraged generative models such as diffusion models [453], [454] and VQ-VAE [465] to model interaction prior, which effectively provides a desirable initial estimation for following optimization iterations. MultiPhys [466] take another strategy to incorporate a physical simulator to search for the optimal policy in physically correct motion space via an imitation learning framework.

###### 5.2 Appearance-rich human-centric interaction

Earlier attempts at reconstructing textured humancentric interactions from videos were limited by the lack of high-quality 3D datasets, making it difficult to model complex interactions between humans and objects. Fortunately, recent progress in differentiable 3D representations, particularly NeRF and 3D Gaussian Splatting (3DGS), has opened up new possibilities for capturing these interactions without relying on expensive 4D datasets.

A representative example is HOSNeRF [84], which enables joint reconstruction of humans and their interacted objects (e.g., backpacks) from RGB videos. Specifically, it extends the human skeleton with object bones, allowing the model to account for deformations introduced by contact. The process to obtain color c and density value σ for each point xc can be then written as:

F(γ(xc),Oci)  → (c,σ), (5) where F(·) is the NeRF module, γ(·) denotes the positional embedding, and Oci is the learnable state embedding representing object states in the canonical space at frame i. These embeddings allow the model to conditionally represent different interaction configurations.

Following HOSNeRF, other recent methods further extend this direction: NeuMan [85] decouples the human and scene by training separate NeRFs for each, improving flexibility and scene composition; PPR [324] combines

[Figure 7]

- Fig. 7. The paradigms of methods for reconstructing appearancerich human-centric interaction. These methods generally build on SMPL-based linear blend skinning (LBS) deformation, extending the human body skeleton to include interacted objects. An example result is shown in the figure below. (image source: HOSNeRF [84]). “H. Def.”, “O. Def.”, and “Ext. LBS” denote “Human Deformation”, “Object Deformation”, and “Extended SMPL-based LBS” correspondingly.

differentiable physics simulation with differentiable rendering, optimizing the reconstruction via coordinate descent to improve realism; RAC [467] generalizes the approach to animals and humans by learning consistent skeletons with fixed bone lengths.

###### 5.3 Egocentric human-centric interaction

Egocentric videos, captured from a first-person perspective, uniquely record the wearer’s interactions with objects, environments, and other people, offering rich context for reconstructing and understanding human-centric interactions. Most existing benchmarks and models primarily focus on hand-object interactions [468]. An early effort in this domain, H2O [469], captured egocentric hand-object interactions using a head-mounted RGB-D camera along with multiple third-person cameras. HOI4D [470] further scales up the egocentric hand-object interaction capture with more objects. HOT3D [471] leverages Project Aria glasses [419] and Quest 3 headsets in a multi-camera rig to enable more precise annotation of hand and object poses. Beyond household scenarios, egocentric hand-object interaction also plays a crucial role in other domains. For instance, POVSurgery [472] introduces a synthetic dataset tailored for estimating hand and instrument poses in surgical settings; HOI-Ref [473] curates an HOI-QA dataset that consists of 3.9M question-answer pairs and achieves good performance for retrieving human-object-interactions from videos; AMEGO [474] introduces a new Active Memories Benchmark (AMB) while achieving SOTA performance for capturing key locations and object interactions; Ego-Exo4D [475], Nymeria [421], HD-EPIC [476], and EPIC-Fusion [477], [478] collect the first large-scale 4D datasets for human scene interactions. More recently, works like EPIC-Fields [479] present the first trial to leverage the 3D priors for understanding the videos with human-centric interactions.

#### 6 LEVEL 5 – INCORPORATION OF PHYSICAL LAWS AND CONSTRAINTS

Recent advancements in 4D scene reconstruction and embodied AI have paved the way for a shift in research focus, i.e., modeling the underlying physics of the environment to achieve a more comprehensive representation of 4D spatial intelligence that imitates real-world human

[Figure 8]

Fig. 8. The paradigms of methods for inferring physically grounded 3D spatial understanding from videos. (I) Physical dynamic human modeling methods learn motion policies from real-world captures of human-object interactions, enabling deployment in simulators and transfer to humanoid robotics (image source: SkillMimic [481]). (II) Physically plausible 3D scene reconstruction mitigates missing geometry artifacts prevalent in traditional approaches, producing simulation-ready environments (image source: PhyRecon [482]).

experiences. This evolution in research aims not only to capture dynamic human-scene interactions over time but also to embed physical plausibility and reasoning into these reconstructions. Such developments are crucial for enabling intelligent robotic systems to operate effectively in complex, unstructured environments. As illustrated in Fig. 8, current efforts in this area primarily focus on 4D reconstruction of dynamic humans and 3D scenes. While some recent methods [480] have incorporated physics to improve 3D object generation, we do not cover this line of work in our illustration.

###### 6.1 Dynamic 4D human simulation with physics

Recent advancements in physical human modeling have focused on generating realistic, physics-based character animation using motion capture data, imitation learning, and reinforcement learning. This body of work spans two key areas: physics-driven character animation and human-object interaction (HOI).

Physics-based character animation Generating physically accurate motion for human and animal characters has long been a central challenge in animation and control research [483], [484], [485], [486], [487], [488], [489], [490]. Recent advances in physics-based character animation emphasize learning from large MoCap datasets using reinforcement learning (RL) [158] and imitation learning [87], [491], [492], [493], [494], [495]. Among them, a foundational approach is DeepMimic [491], which learns to reproduce dynamic motions through direct trajectory tracking. AMP [496], based on GAIL [497], improves realism by using a generative adversarial framework, where a discriminator judges the realism of the motion, guiding the controller during training. However, AMP demands

training a separate policy for each task. Extensions like ASE [498], CALM [499], ControlVAE [500], PULSE [88], OmniGrasp [501], HOVER [502], ASAP [503], and UniPhys [504] aim to extract more general motion priors that can be reused across tasks. MaskedMimic [505] introduces a masked conditional VAE for multi-task learning, but still struggles with generalizing to unseen control signals.

In parallel, there has been a rise in text-driven control approaches [499], [506], [507], [508], where high-level natural language is used to direct character behavior. For instance, SuperPADL [507] employs a multi-stage training pipeline combining reinforcement learning and behavior cloning. PDP [508] uses diffusion models to create multimodal controllers that interpret text commands, improving robustness by injecting noise during training. Despite these innovations, text-driven physical controllers still lag behind kinematic methods in expressiveness and diversity due to difficulties in distilling controllable and reliable multimodal behaviors. To bridge this gap, researchers have turned to hierarchical control frameworks [509], [510], [511], [512]. These methods split the problem into a high-level planning stage and a low-level controller. The planner might generate trajectories [509], waypoints [511], or partial-body targets [510], which are tracked by an RL policy. For example, CLoSD [509] combines a kinematic diffusion-based planner with a physics-based tracker. However, mismatches between high-level kinematic plans and low-level physical feasibility often lead to artifacts like foot sliding or jitter, which require task-specific fine-tuning [509], [510].

Learning human-object interaction (HOI) Human-object interaction presents additional complexity due to the need for fine contact control, multi-body coordination, and realistic physical responses. Early systems approached HOI using handcrafted state machines or models like inverted pendulums to simulate behaviors such as running or jumping [513], [514], [515]. More recent works use deep RL [516] to model more diverse interactions, including sports and tool use [517], [518], [519]. Imitation learning is a natural fit for HOI, and methods like [520], [521], [522] attempt to model whole-body interactions using motion priors or grasping models. However, directly adapting locomotion-based imitation frameworks (e.g., DeepMimic [491], AMP [496]) to HOI has proven unstable. These methods often fail to address unbalanced reward structures, crucial contact dynamics, or the importance of relative positioning, leading to poor performance. Some approaches use interaction graphs to model spatial dependencies between body and object [523], [524], but these primarily focus on kinematics and often lack physical realism. In contrast, more recent frameworks introduce contact-aware rewards that explicitly encourage correct and stable contact, significantly improving performance on complex HOI tasks. This contact-driven perspective, introduced in early work like [86], [481], allows for unified training across a wide range of interaction scenarios without the need for handcrafted rewards or separate pipelines.

###### 6.2 3D scene reconstruction with physical plausibility

As we move toward building the 3D virtual worlds that can mimic real-world actions, physically plausible 3D scene

reconstruction is gradually becoming a central focus in 3D scene modeling [482], [525]. Addressing this challenge, PhysicsNeRF [93] injects explicit physics guidance– specifically, depth-ranking, sparsity, and cross-view alignment losses–to achieve stable and physically consistent geometry even from extremely sparse multi-view inputs. Building on this foundation, inverse-rendering pipelines such as PBR-NeRF [94] integrate neural radiance fields with physics-based rendering priors. This coupling enables the joint optimization of geometry, illumination, and spatially varying materials, effectively mitigating the physically impossible albedo–lighting entanglement inherent in vanilla NeRFs. Progressing to the scene level, CAST [95] first retrieves CAD proxies from a single RGB image and subsequently applies a physics-aware correction step. This step rigorously enforces support, non-penetration, and objectrelation constraints, resulting in contact-consistent layouts. PhyRecon [482] proposes to leverage the differentiable gradients from the simulator [231], [526] to improve the physical plausibility of the reconstruction scene components. Orthogonal to explicit simulators, Aug-NeRF [527] employs triple-level, physically grounded augmentations as a regularization strategy during training. This technique dramatically reduces view-inconsistent floaters and enhances generalization capabilities. Finally, specialized methodologies are emerging for targeted phenomena; for instance, the Planar Reflection-Aware NeRF [528] explicitly models secondary reflected rays. This advancement eliminates the floaters frequently hallucinated behind reflective surfaces like glass and mirrors, thereby further improving the physical plausibility of reconstructions in everyday indoor scenes.

#### 7 CHALLENGES AND FUTURE DIRECTIONS

While notable advancements have been made for methods from level 1 to level 5, current techniques still encounter major challenges.

###### 7.1 Level 1 – low-level 3D attributes

Challenges Despite remarkable advances, reconstructing depth, camera, and 3D tracking from video remains challenging, due to its inherent ill-posed nature, especially for unconstrained, dynamic inputs. (1) A core challenge lies in handling occlusions, dynamic object motion, and nonLambertian surfaces, which violate many of the assumptions underlying existing methods. (2) Many methods still require post-processing steps, global alignment, or manual hyperparameter tuning, limiting their automation and applicability. (3) Ensuring robustness to sensor noise, and generalization to diverse viewpoints and motion patterns (e.g., handheld, drone, egocentric) still remain open concerns.

Future directions Given the key challenges outlined above, several promising directions can be explored. (1) Developing world models that jointly represent geometry, motion, semantics, and uncertainty could offer a principled approach to reducing ambiguity in 4D reconstruction. These models can also take cues from vision-language foundation models, using large-scale pretraining on both synthetic and real-world video to learn strong inductive biases. (2) Additionally, further progress can also be made to achieve in-

teractive annotation tools and multi-agent data collection, to provide richer supervision for training more robust systems.

###### 7.2 Level 2 – 3D scene components

Challenges While 3D scene reconstruction from video has seen remarkable progress, multiple technical challenges remain. (1) There is no universally optimal scene representation. Point clouds, meshes, NeRFs, and 3D Gaussian Splatting each present trade-offs between fidelity, efficiency, and expressiveness. (2) Recovering fine-scale geometry in unbounded or textureless regions, particularly under challenging conditions such as motion blur, lighting changes, or sparse viewpoints, remains difficult. (3) Reconstructions from egocentric videos are especially prone to degradation due to rapid motion and limited field of view (FoV), often resulting in incomplete or distorted outputs.

Future directions In light of these challenges, future directions could potentially focus on: (1) Developing hierarchical and scalable architectures, such as mixture-of-experts models, sparse voxel grids, and scaffolded splatting, might be key to enabling efficient reconstruction and rendering across large-scale environments. (2) Advancing egocentric and dynamic scene understanding through cross-modal learning, by integrating signals such as IMU data, audio, and textual descriptions, may improve robustness in complex and motion-heavy scenarios.

###### 7.3 Level 3 – 4D dynamic scenes

Challenges Despite rapid progress in 4D scene reconstruction and human-centric dynamic modeling, several critical challenges remain unresolved: (1) Feed-forward methods accelerate reconstruction but suffer speed-generalizationquality trade-offs, often relying on per-scene optimization or massive training data that hinder scalability. (2) Complex dynamic phenomena, including fluids, smoke, semi-rigid objects, and topological changes (e.g., object splitting/merging), remain largely unsolved. (3) Similarly, egocentric reconstruction is severely challenged by selfocclusions and limited FoV in head-mounted fisheye captures, complicating dynamic scene recovery.

Future directions Based on these issues, several directions deserve to be considered. (1) Hybrid implicit-explicit representations could balance reconstruction speed and fidelity, while physics-informed priors (biomechanics, fluid dynamics) may enforce physically plausible motion. (2) Benchmarks and collaborative exo-ego capture frameworks are critical to evaluate temporal coherence, overcome occlusion limitations, and enable scalable AR/VR/robotics applications.

###### 7.4 Level 4 – interactions among scene components

Challenges Reconstructing human-centric interactions from video remains a highly challenging task due to several technical limitations. Human-object interaction (HOI) methods often require accurate object templates, which restricts their ability to generalize across diverse object categories and deformable instances. Human-scene interaction (HSI) models struggle to consistently align humans with dynamic environments over time, particularly in real-world videos

where spatial and temporal cues are sparse. For humanhuman interaction (HHI), monocular setups frequently face issues such as occlusion, depth ambiguity, and unrealistic contact modeling. Across all domains, maintaining physical plausibility, preventing interpenetration, and ensuring temporal coherence remain persistent challenges. Moreover, the lack of large-scale, high-quality datasets that capture textured interactions or egocentric viewpoints continues to impede the method’s generalization and real-world deployment.

Future directions To address these challenges, future research could focus on category-agnostic modeling through geometry-aware or generative approaches such as diffusion models, to enable broader generalization across interaction types. Integrating differentiable physics, learned contact priors, or imitation learning frameworks may improve physical realism in interactions. For egocentric interactions, future progress may be able to come from multimodal fusion (e.g., video, IMU, gaze) and learning from large-scale benchmarks. Last but not least, achieving real-time and interactive simulation of human-centric behaviors might require combining video-based reconstruction with embodied reasoning and action modeling.

7.5 Level 5 – incorporation of physical laws and constraints

Challenges Despite significant progress, physics-based reconstruction and simulation from video still face some challenges. Reinforcement learning based character animation often suffers from sample inefficiency, high computational cost, and unstable optimization, particularly for complex or contact-rich motions. Generalization across diverse motion types remains limited; most policies are task-specific, and hierarchical controllers frequently introduce artifacts due to mismatches between high-level kinematic planning and low-level physical feasibility. For HOI situations, achieving stable contact behaviors and precise coordination between human and object dynamics remains difficult, especially under limited supervision or with diverse object geometries. In 3D scene reconstruction, enforcing physical plausibility, such as support, friction, and non-penetration, is difficult when scene geometry is incomplete or inferred from sparse views. As a result, many methods continue to produce floating artifacts, interpenetrations, or physically implausible contacts due to insufficient physical priors.

Future directions To overcome these limitations, future research could first explore multimodal representations that integrate video cues, physical constraints, and motion priors from different tasks and domains. Secondly, differentiable physics engines may offer a promising foundation for jointly optimizing geometry, dynamics, and interaction constraints within a fully learnable pipeline. Thirdly, developing hierarchical and diffusion-based motion planning, particularly when conditioned on text or scene context, may also lead to more expressive, controllable, and physically plausible character behaviors. Finally, better integration of perception and control, allowing characters to adapt their behavior based on inferred scene dynamics, may unlock more interactive and physically realistic virtual humans.

#### 8 CONCLUSION

In this report, we begin by categorizing 4D spatial intelligence into five distinct levels: Level 1 — basic 3D cues; Level 2 — components of 3D scenes; Level 3 — dynamic 4D scenes; Level 4 — interactions among scene components; and Level 5 — integration of physical laws and constraints. We then provide a thorough review of methods corresponding to each level. Additionally, we discuss the remaining challenges faced by current techniques and explore promising future directions to overcome these issues. As this is a rapidly evolving field with new papers published weekly or even daily, we hope this survey offers an accessible entry point for interested readers and inspires progress toward a potential Level 6 in 4D spatial intelligence.

#### REFERENCES

- [1] K.-A. Aliev, A. Sevastopolsky, M. Kolos, D. Ulyanov, and V. Lempitsky, “Neural point-based graphics,” in European conference on computer vision. Springer, 2020, pp. 696–712.
- [2] H. Kim, P. Garrido, A. Tewari, W. Xu, J. Thies, M. Niessner, P. P´erez, C. Richardt, M. Zollh¨ofer, and C. Theobalt, “Deep video portraits,” ACM transactions on graphics (TOG), vol. 37, no. 4, pp. 1–14, 2018.
- [3] N. Deng, Z. He, J. Ye, B. Duinkharjav, P. Chakravarthula, X. Yang, and Q. Sun, “Fov-nerf: Foveated neural radiance fields for virtual reality,” IEEE Transactions on Visualization and Computer Graphics, vol. 28, no. 11, pp. 3854–3864, 2022.
- [4] S. Li, C. Li, W. Zhu, B. Yu, Y. Zhao, C. Wan, H. You, H. Shi, and Y. Lin, “Instant-3d: Instant neural radiance field training towards on-device ar/vr 3d reconstruction,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, 2023, pp. 1–13.
- [5] Y. Liu, W. Chen, Y. Bai, X. Liang, G. Li, W. Gao, and L. Lin, “Aligning cyber space with physical world: A comprehensive survey on embodied ai,” arXiv preprint arXiv:2407.06886, 2024.
- [6] T. Gupta, W. Gong, C. Ma, N. Pawlowski, A. Hilmkil, M. Scetbon, M. Rigter, A. Famoti, A. J. Llorens, J. Gao et al., “The essential role of causality in foundation world models for embodied ai,” arXiv preprint arXiv:2402.06665, 2024.
- [7] J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S.-C. Zhu, B. Jia, and S. Huang, “An embodied generalist agent in 3d world,” arXiv preprint arXiv:2311.12871, 2023.
- [8] H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan, “3d-vla: A 3d vision-language-action generative world model,” arXiv preprint arXiv:2403.09631, 2024.
- [9] T. Wu, Y.-J. Yuan, L.-X. Zhang, J. Yang, Y.-P. Cao, L.-Q. Yan, and L. Gao, “Recent advances in 3d gaussian splatting,” Computational Visual Media, vol. 10, no. 4, pp. 613–642, 2024.
- [10] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, and Y. He, “3d gaussian splatting as new era: A survey,” IEEE Transactions on Visualization and Computer Graphics, 2024.
- [11] M. S. Hamid, N. Abd Manap, R. A. Hamzah, and A. F. Kadmin, “Stereo matching algorithm based on deep learning: A survey,” Journal of King Saud University-Computer and Information Sciences, vol. 34, no. 5, pp. 1663–1673, 2022.
- [12] K. Zhou, X. Meng, and B. Cheng, “Review of stereo matching algorithms based on deep learning,” Computational intelligence and neuroscience, vol. 2020, no. 1, p. 8562323, 2020.
- [13] H. Laga, L. V. Jospin, F. Boussaid, and M. Bennamoun, “A survey on deep learning techniques for stereo-based depth estimation,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 4, pp. 1738–1764, 2020.
- [14] K. Gao, Y. Gao, H. He, D. Lu, L. Xu, and J. Li, “Nerf: Neural radiance field in 3d vision, a comprehensive review,” arXiv preprint arXiv:2210.00379, 2022.
- [15] Y. Bao, T. Ding, J. Huo, Y. Liu, Y. Li, W. Li, Y. Gao, and J. Luo, “3d gaussian splatting: Survey, technologies, challenges, and opportunities,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.
- [16] G. Wang, L. Pan, S. Peng, S. Liu, C. Xu, Y. Miao, W. Zhan, M. Tomizuka, M. Pollefeys, and H. Wang, “Nerf in robotics: A survey,” arXiv preprint arXiv:2405.01333, 2024.

- [17] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in European conference on computer vision. Springer, 2020, pp. 405–421.
- [18] T. Shen, J. Gao, K. Yin, M.-Y. Liu, and S. Fidler, “Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis,” 2021.
- [19] B. Kerbl, G. Kopanas, T. Leimkuhler,¨ and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics (ToG), vol. 42, no. 4, pp. 1–14, 2023.
- [20] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” Advances in Neural Information Processing Systems, vol. 35, pp. 8633–8646, 2022.
- [21] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet et al., “Imagen video: High definition video generation with diffusion models,” arXiv preprint arXiv:2210.02303, 2022.
- [22] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.
- [23] P. C. Ng and S. Henikoff, “Sift: Predicting amino acid changes that affect protein function,” Nucleic acids research, vol. 31, no. 13, pp. 3812–3814, 2003.
- [24] J. Revaud, C. De Souza, M. Humenberger, and P. Weinzaepfel, “R2d2: Reliable and repeatable detector and descriptor,” Advances in neural information processing systems, vol. 32, 2019.
- [25] D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superpoint: Selfsupervised interest point detection and description,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 224–236.
- [26] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superglue: Learning feature matching with graph neural networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4938–4947.
- [27] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, “Loftr: Detectorfree local feature matching with transformers,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 8922–8931.
- [28] E. Brachmann and C. Rother, “Neural-guided ransac: Learning where to sample model hypotheses,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 4322–4331.
- [29] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “Lightglue: Local feature matching at light speed,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 17627–17638.
- [30] D. Barath, D. Mishkin, L. Cavalli, P.-E. Sarlin, P. Hruby, and M. Pollefeys, “Affineglue: Joint matching and robust estimation,” arXiv preprint arXiv:2307.15381, 2023.
- [31] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4104–4113.
- [32] O. Ozye¸sil,¨ V. Voroninski, R. Basri, and A. Singer, “A survey of structure from motion*.” Acta Numerica, vol. 26, pp. 305–364, 2017.
- [33] J. Iglhaut, C. Cabo, S. Puliti, L. Piermattei, J. O’Connor, and J. Rosette, “Structure from motion photogrammetry in forestry: A review,” Current Forestry Reports, vol. 5, no. 3, pp. 155–168, 2019.
- [34] P. Lindenberger, P.-E. Sarlin, V. Larsson, and M. Pollefeys, “Pixelperfect structure-from-motion with featuremetric refinement,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5987–5997.
- [35] S. Agarwal, N. Snavely, S. M. Seitz, and R. Szeliski, “Bundle adjustment in the large,” in European conference on computer vision. Springer, 2010, pp. 29–42.
- [36] C. Engels, H. Stew´enius, and D. Nist´er, “Bundle adjustment rules,” Photogrammetric computer vision, vol. 2, no. 32, 2006.
- [37] C. Zach, “Robust bundle adjustment revisited,” in European Conference on Computer Vision. Springer, 2014, pp. 772–787.
- [38] B. Triggs, P. F. McLauchlan, R. I. Hartley, and A. W. Fitzgibbon, “Bundle adjustment—a modern synthesis,” in International workshop on vision algorithms. Springer, 1999, pp. 298–372.
- [39] J. L. Sch¨onberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, “Pixelwise view selection for unstructured multi-view stereo,” in European Conference on Computer Vision (ECCV), 2016.

- [40] J. Zhang, Y. Yao, S. Li, Z. Luo, and T. Fang, “Visibility-aware multi-view stereo network,” British Machine Vision Conference (BMVC), 2020.
- [41] J. Yang, W. Mao, J. M. Alvarez, and M. Liu, “Cost volume pyramid based depth inference for multi-view stereo,” in The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.
- [42] F. Wang, S. Galliani, C. Vogel, P. Speciale, and M. Pollefeys, “Patchmatchnet: Learned multi-view patchmatch stereo,” 2021.
- [43] X. Gu, Z. Fan, S. Zhu, Z. Dai, F. Tan, and P. Tan, “Cascade cost volume for high-resolution multi-view stereo and stereo matching,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 2495–2504.
- [44] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r: Geometric 3d vision made easy,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20697–20709.
- [45] R. Murai, E. Dexheimer, and A. J. Davison, “Mast3r-slam: Realtime dense slam with 3d reconstruction priors,” arXiv preprint arXiv:2412.12392, 2024.
- [46] J. Zhang, C. Herrmann, J. Hur, V. Jampani, T. Darrell, F. Cole, D. Sun, and M.-H. Yang, “Monst3r: A simple approach for estimating geometry in the presence of motion,” arXiv preprint arXiv:2410.03825, 2024.
- [47] J. Lu, T. Huang, P. Li, Z. Dou, C. Lin, Z. Cui, Z. Dong, S.-K. Yeung, W. Wang, and Y. Liu, “Align3r: Aligned monocular depth estimation for dynamic videos,” arXiv preprint arXiv:2412.03079, 2024.
- [48] J. Yang, A. Sax, K. J. Liang, M. Henaff, H. Tang, A. Cao, J. Chai, F. Meier, and M. Feiszli, “Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass,” arXiv preprint arXiv:2501.13928, 2025.
- [49] K. Han, A. Xiao, E. Wu, J. Guo, C. Xu, and Y. Wang, “Transformer in transformer,” Advances in neural information processing systems, vol. 34, pp. 15908–15919, 2021.
- [50] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu et al., “A survey on vision transformer,” IEEE transactions on pattern analysis and machine intelligence, vol. 45, no. 1, pp. 87–110, 2022.
- [51] N. Kitaev, Ł. Kaiser, and A. Levskaya, “Reformer: The efficient transformer,” arXiv preprint arXiv:2001.04451, 2020.
- [52] H. Zhao, L. Jiang, J. Jia, P. H. Torr, and V. Koltun, “Point transformer,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 16259–16268.
- [53] N. Parmar, A. Vaswani, J. Uszkoreit, L. Kaiser, N. Shazeer, A. Ku, and D. Tran, “Image transformer,” in International conference on machine learning. PMLR, 2018, pp. 4055–4064.
- [54] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294–5306.
- [55] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.
- [56] B. Kerbl, G. Kopanas, T. Leimkuhler,¨ and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- [57] T. Shen, J. Munkberg, J. Hasselgren, K. Yin, Z. Wang, W. Chen, Z. Gojcic, S. Fidler, N. Sharp, and J. Gao, “Flexible isosurface extraction for gradient-based mesh optimization,” ACM Trans. Graph., vol. 42, no. 4, jul 2023. [Online]. Available: https://doi.org/10.1145/3592430
- [58] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla, “Nerfies: Deformable neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5865–5874.
- [59] E. Tretschk, A. Tewari, V. Golyanik, M. Zollh¨ofer, C. Lassner, and C. Theobalt, “Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12959–12970.
- [60] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, and S. M. Seitz, “Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields,” arXiv preprint arXiv:2106.13228, 2021.

- [61] H. Yu, J. Julin, Z. A. Milacski, K. Niinuma, and L. A. Jeni, “Dylin: Making light field networks dynamic,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12397–12406.
- [62] C.-D. Fan, C.-W. Chang, Y.-R. Liu, J.-Y. Lee, J.-L. Huang, Y.-C. Tseng, and Y.-L. Liu, “Spectromotion: Dynamic 3d reconstruction of specular scenes,” arXiv preprint arXiv:2410.17249, 2024.
- [63] Z. Li, S. Niklaus, N. Snavely, and O. Wang, “Neural scene flow fields for space-time view synthesis of dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 6498–6508.
- [64] C. Gao, A. Saraf, J. Kopf, and J.-B. Huang, “Dynamic view synthesis from dynamic monocular video,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5712–5721.
- [65] M. You and J. Hou, “Decoupling dynamic monocular videos for dynamic view synthesis,” IEEE Transactions on Visualization and Computer Graphics, 2024.
- [66] C. Wang, B. Eckart, S. Lucey, and O. Gallo, “Neural trajectory fields for dynamic novel view synthesis,” arXiv preprint arXiv:2105.05994, 2021.
- [67] H. Lin, Q. Wang, R. Cai, S. Peng, H. Averbuch-Elor, X. Zhou, and N. Snavely, “Neural scene chronology,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 20752–20761.
- [68] A. Lou, B. Planche, Z. Gao, Y. Li, T. Luan, H. Ding, T. Chen, J. Noble, and Z. Wu, “Darenerf: Direction-aware representation for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5031–5042.
- [69] J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che, D. Xu, S. Fidler, M. Pavone et al., “Emernerf: Emergent spatial-temporal scene decomposition via self-supervision,” arXiv preprint arXiv:2311.02077, 2023.
- [70] R. Dabral, S. Shimada, A. Jain, C. Theobalt, and V. Golyanik, “Gravity-aware monocular 3d human-object reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12365–12374.
- [71] X. Xu, H. Joo, G. Mori, and M. Savva, “D3d-hoi: Dynamic 3d human-object interactions from videos,” arXiv preprint arXiv:2108.08420, 2021.
- [72] B. L. Bhatnagar, X. Xie, I. A. Petrov, C. Sminchisescu, C. Theobalt, and G. Pons-Moll, “Behave: Dataset and method for tracking human object interactions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 15935– 15946.
- [73] Y. Huang, O. Taheri, M. J. Black, and D. Tzionas, “Intercap: Joint markerless 3d tracking of humans and objects in interaction,” in DAGM German Conference on Pattern Recognition. Springer, 2022, pp. 281–299.
- [74] N. Jiang, T. Liu, Z. Cao, J. Cui, Z. Zhang, Y. Chen, H. Wang, Y. Zhu, and S. Huang, “Full-body articulated human-object interaction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9365–9376.
- [75] C. Huo, Y. Shi, Y. Ma, L. Xu, J. Yu, and J. Wang, “Stackflow: Monocular human-object reconstruction by stacked normalizing flow with offset,” arXiv preprint arXiv:2407.20545, 2024.
- [76] C. Huo, Y. Shi, and J. Wang, “Monocular human-object reconstruction in the wild,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 5547–5555.
- [77] C. Zhao, J. Zhang, J. Du, Z. Shan, J. Wang, J. Yu, J. Wang, and L. Xu, “I’m hoi: Inertia-aware monocular capture of 3d humanobject interactions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 729–741.
- [78] Y. Xie, C.-H. Yao, V. Voleti, H. Jiang, and V. Jampani, “Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency,” arXiv preprint arXiv:2407.17470, 2024.
- [79] X. Xie, J. E. Lenssen, and G. Pons-Moll, “Intertrack: Tracking human object interaction without object templates,” arXiv preprint arXiv:2408.13953, 2024.
- [80] Y. Cao, L. Pan, K. Han, K.-Y. K. Wong, and Z. Liu, “Avatargo: Zero-shot 4d human-object interaction generation and animation,” arXiv preprint arXiv:2410.07164, 2024.
- [81] G. Pavlakos, E. Weber, M. Tancik, and A. Kanazawa, “The one where they reconstructed 3d humans and environments in tv shows,” in European Conference on Computer Vision. Springer, 2022, pp. 732–749.

- [82] Z. Liu, J. Lin, W. Wu, and B. Zhou, “Joint optimization for 4d human-scene reconstruction in the wild,” arXiv preprint arXiv:2501.02158, 2025.
- [83] Z. Zhang, M. Kaufmann, L. Xue, J. Song, and M. R. Oswald, “Odhsr: Online dense 3d reconstruction of humans and scenes from monocular videos,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21824–21835.
- [84] J.-W. Liu, Y.-P. Cao, T. Yang, Z. Xu, J. Keppo, Y. Shan, X. Qie, and M. Z. Shou, “Hosnerf: Dynamic human-object-scene neural radiance fields from a single video,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18483–18494.
- [85] W. Jiang, K. M. Yi, G. Samei, O. Tuzel, and A. Ranjan, “Neuman: Neural human radiance field from a single video,” in European Conference on Computer Vision. Springer, 2022, pp. 402–418.
- [86] Y. Wang, J. Lin, A. Zeng, Z. Luo, J. Zhang, and L. Zhang, “Physhoi: Physics-based imitation of dynamic human-object interaction,” arXiv preprint arXiv:2312.04393, 2023.
- [87] Z. Luo, J. Cao, A. W. Winkler, K. Kitani, and W. Xu, “Perpetual humanoid control for real-time simulated avatars,” in International Conference on Computer Vision (ICCV), 2023.
- [88] Z. Luo, J. Cao, J. Merel, A. Winkler, J. Huang, K. M. Kitani, and W. Xu, “Universal humanoid motion representations for physics-based control,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https: //openreview.net/forum?id=OrOd8PxOO2
- [89] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa et al., “Isaac gym: High performance gpu-based physics simulation for robot learning,” arXiv preprint arXiv:2108.10470, 2021.
- [90] M. A. Wiering and M. Van Otterlo, “Reinforcement learning,” Adaptation, learning, and optimization, vol. 12, no. 3, p. 729, 2012.
- [91] L. P. Kaelbling, M. L. Littman, and A. W. Moore, “Reinforcement learning: A survey,” Journal of artificial intelligence research, vol. 4, pp. 237–285, 1996.
- [92] R. S. Sutton, A. G. Barto et al., “Reinforcement learning,” Journal of Cognitive Neuroscience, vol. 11, no. 1, pp. 126–134, 1999.
- [93] M. R. Barhdadi, H. Kurban, and H. Alnuweiri, “Physicsnerf: Physics-guided 3d reconstruction from sparse views,” 2025.
- [94] S. Wu, S. Basu, T. Broedermann, L. V. Gool, and C. Sakaridis, “Pbr-nerf: Inverse rendering with physics-based neural fields,” 2025.
- [95] K. Yao, L. Zhang, X. Yan, Y. Zeng, Q. Zhang, W. Yang, L. Xu, J. Gu, and J. Yu, “Cast: Component-aligned 3d scene reconstruction from an rgb image,” 2025.
- [96] V. Voleti, C.-H. Yao, M. Boss, A. Letts, D. Pankratz, D. Tochilkin, C. Laforte, R. Rombach, and V. Jampani, “Sv3d: Novel multiview synthesis and 3d generation from a single image using latent video diffusion,” in European Conference on Computer Vision. Springer, 2025, pp. 439–457.
- [97] Z. Chen, Y. Wang, F. Wang, Z. Wang, and H. Liu, “V3d: Video diffusion models are effective 3d generators,” arXiv preprint arXiv:2403.06738, 2024.
- [98] Y. Cao, Y.-P. Cao, K. Han, Y. Shan, and K.-Y. K. Wong, “Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 958–968.
- [99] S. Bahmani, I. Skorokhodov, V. Rong, G. Wetzstein, L. Guibas, P. Wonka, S. Tulyakov, J. J. Park, A. Tagliasacchi, and D. B. Lindell, “4d-fy: Text-to-4d generation using hybrid score distillation sampling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7996–8006.
- [100] S. Bahmani, X. Liu, W. Yifan, I. Skorokhodov, V. Rong, Z. Liu, X. Liu, J. J. Park, S. Tulyakov, G. Wetzstein et al., “Tc4d: Trajectoryconditioned text-to-4d generation,” in European Conference on Computer Vision. Springer, 2024, pp. 53–72.
- [101] H. Zhang, X. Chen, Y. Wang, X. Liu, Y. Wang, and Y. Qiao, “4diffusion: Multi-view video diffusion model for 4d generation,” Advances in Neural Information Processing Systems, vol. 37, pp. 15272–15295, 2024.
- [102] H. Liang, Y. Yin, D. Xu, H. Liang, Z. Wang, K. N. Plataniotis, Y. Zhao, and Y. Wei, “Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models,” arXiv preprint arXiv:2405.16645, 2024.
- [103] Y. Cao, X. Guo, M. Zhang, H. Xie, C. Gu, and Z. Liu, “Crowdmogen: Zero-shot text-driven collective motion generation,” arXiv preprint arXiv:2407.06188, 2024.

- [104] Y. Cao, Y.-P. Cao, K. Han, Y. Shan, and K.-Y. K. Wong, “Guide3d: Create 3d avatars from text and image guidance,” arXiv preprint arXiv:2308.09705, 2023.
- [105] Q. Miao, K. Li, J. Quan, Z. Min, S. Ma, Y. Xu, Y. Yang, and Y. Luo, “Advances in 4d generation: A survey,” arXiv preprint arXiv:2503.14501, 2025.
- [106] C. Li, C. Zhang, A. Waghwase, L.-H. Lee, F. Rameau, Y. Yang, S.-H. Bae, and C. S. Hong, “Generative ai meets 3d: A survey on text-to-3d in aigc era,” arXiv preprint arXiv:2305.06131, 2023.
- [107] J. Liu, X. Huang, T. Huang, L. Chen, Y. Hou, S. Tang, Z. Liu, W. Ouyang, W. Zuo, J. Jiang et al., “A comprehensive survey on 3d content generation,” arXiv preprint arXiv:2402.01166, 2024.
- [108] X. Li, Q. Zhang, D. Kang, W. Cheng, Y. Gao, J. Zhang, Z. Liang, J. Liao, Y.-P. Cao, and Y. Shan, “Advances in 3d generation: A survey,” arXiv preprint arXiv:2401.17807, 2024.
- [109] H. Kato, D. Beker, M. Morariu, T. Ando, T. Matsuoka, W. Kehl, and A. Gaidon, “Differentiable rendering: A survey,” arXiv preprint arXiv:2006.12057, 2020.
- [110] B. Deng, Y. Yao, R. M. Dyke, and J. Zhang, “A survey of nonrigid 3d registration,” in Computer Graphics Forum, vol. 41, no. 2. Wiley Online Library, 2022, pp. 559–589.
- [111] R. G. Pacheco and R. S. Couto, “Inference time optimization using branchynet partitioning,” in 2020 IEEE Symposium on Computers and Communications (ISCC). IEEE, 2020, pp. 1–6.
- [112] Z. Yin and J. Shi, “Geonet: Unsupervised learning of dense depth, optical flow and camera pose,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 1983–1992.
- [113] A. Gordon, H. Li, R. Jonschkowski, and A. Angelova, “Depth from videos in the wild: Unsupervised monocular depth learning from unknown cameras,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 8977–8986.
- [114] J. Bian, Z. Li, N. Wang, H. Zhan, C. Shen, M.-M. Cheng, and

I. Reid, “Unsupervised scale-consistent depth and ego-motion learning from monocular video,” Advances in neural information processing systems, vol. 32, 2019.

- [115] X. Luo, J.-B. Huang, R. Szeliski, K. Matzen, and J. Kopf, “Consistent video depth estimation,” ACM Transactions on Graphics (ToG), vol. 39, no. 4, pp. 71–1, 2020.
- [116] Y. Chen, C. Schmid, and C. Sminchisescu, “Self-supervised learning with geometric constraints in monocular video: Connecting flow, depth, and camera,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 7063–7072.
- [117] X. Long, L. Liu, W. Li, C. Theobalt, and W. Wang, “Multi-view depth estimation using epipolar spatio-temporal networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 8258–8267.
- [118] V. Guizilini, R. Ambrus,, D. Chen, S. Zakharov, and A. Gaidon, “Multi-frame self-supervised depth with transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 160–170.
- [119] J. Watson, O. Mac Aodha, V. Prisacariu, G. Brostow, and M. Firman, “The temporal opportunist: Self-supervised multi-frame monocular depth,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 1164–1174.
- [120] M. Sayed, J. Gibson, J. Watson, V. Prisacariu, M. Firman, and C. Godard, “Simplerecon: 3d reconstruction without 3d convolutions,” in European Conference on Computer Vision. Springer, 2022, pp. 1–19.
- [121] J. Xie, C. Lei, Z. Li, L. E. Li, and Q. Chen, “Video depth estimation by fusing flow-to-depth proposals,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020, pp. 10100–10107.
- [122] C. Eom, H. Park, and B. Ham, “Temporally consistent depth prediction with flow-guided memory units,” IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 11, pp. 4626–4636,

- 2019.

[123] V. Patil, W. Van Gansbeke, D. Dai, and L. Van Gool, “Don’t forget the past: Recurrent depth estimation from monocular video,” IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 6813–6820,

- 2020.

- [124] H. Zhang, C. Shen, Y. Li, Y. Cao, Y. Liu, and Y. Yan, “Exploiting temporal consistency for real-time video depth estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1725–1734.
- [125] Y. Wang, Z. Pan, X. Li, Z. Cao, K. Xian, and J. Zhang, “Less is more: Consistent video depth estimation with masked frames

- modeling,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 6347–6358.
- [126] R. Yasarla, H. Cai, J. Jeong, Y. Shi, R. Garrepalli, and F. Porikli, “Mamo: Leveraging memory and attention for monocular video depth estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 8754–8764.
- [127] C. Zhao, Y. Zhang, M. Poggi, F. Tosi, X. Guo, Z. Zhu, G. Huang, Y. Tang, and S. Mattoccia, “Monovit: Self-supervised monocular depth estimation with a vision transformer,” in 2022 international conference on 3D vision (3DV). IEEE, 2022, pp. 668–678.
- [128] Z. Li, W. Ye, D. Wang, F. X. Creighton, R. H. Taylor, G. Venkatesh, and M. Unberath, “Temporally consistent online depth estimation in dynamic scenes,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2023, pp. 3018–3027.
- [129] Z. Teed and J. Deng, “Deepv2d: Video to depth with differentiable structure from motion,” in International Conference on Learning Representations, 2020.
- [130] Y. Wang, M. Shi, J. Li, Z. Huang, Z. Cao, J. Zhang, K. Xian, and G. Lin, “Neural video depth stabilizer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 9466–9476.
- [131] Y. Wang, M. Shi, J. Li, C. Hong, Z. Huang, J. Peng, Z. Cao, J. Zhang, K. Xian, and G. Lin, “Nvds+: Towards efficient and versatile neural stabilizer for video depth estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–18, 2024.
- [132] W. Hu, X. Gao, X. Li, S. Zhao, X. Cun, Y. Zhang, L. Quan, and Y. Shan, “Depthcrafter: Generating consistent long depth sequences for open-world videos,” arXiv preprint arXiv:2409.02095, 2024.
- [133] J. Shao, Y. Yang, H. Zhou, Y. Zhang, Y. Shen, M. Poggi, and Y. Liao, “Learning temporally consistent video depth from video diffusion priors,” arXiv preprint arXiv:2406.01493, 2024.
- [134] H. Yang, D. Huang, W. Yin, C. Shen, H. Liu, X. He, B. Lin, W. Ouyang, and T. He, “Depth any video with scalable synthetic data,” arXiv preprint arXiv:2410.10815, 2024.
- [135] S. Chen, H. Guo, S. Zhu, F. Zhang, Z. Huang, J. Feng, and B. Kang, “Video depth anything: Consistent depth estimation for superlong videos,” arXiv preprint arXiv:2501.12375, 2025.
- [136] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” arXiv preprint arXiv:2406.09414, 2024.
- [137] R. Mur-Artal and J. D. Tard´os, “Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras,” IEEE transactions on robotics, vol. 33, no. 5, pp. 1255–1262, 2017.
- [138] Q. Wang, Z. Yan, J. Wang, F. Xue, W. Ma, and H. Zha, “Line flow based simultaneous localization and mapping,” IEEE Transactions on Robotics, vol. 37, no. 5, pp. 1416–1432, 2021.
- [139] S. Kannapiran, N. Bendapudi, M.-Y. Yu, D. Parikh, S. Berman, A. Vora, and G. Pandey, “Stereo visual odometry with deep learning-based point and line feature matching using an attention graph neural network,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 3491– 3498.
- [140] F. Shu, J. Wang, A. Pagani, and D. Stricker, “Structure plp-slam: Efficient sparse mapping and localization using point, line and plane for monocular, rgb-d and stereo cameras,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 2105–2112.
- [141] H. Jiang, R. Qian, L. Du, J. Pu, and J. Feng, “Ul-slam: A universal monocular line-based slam via unifying structural and nonstructural constraints,” IEEE Transactions on Automation Science and Engineering, 2024.
- [142] J. Engel, T. Sch¨ops, and D. Cremers, “Lsd-slam: Large-scale direct monocular slam,” in European conference on computer vision. Springer, 2014, pp. 834–849.
- [143] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 3, pp. 611–625, 2017.
- [144] L. Zhou, G. Huang, Y. Mao, S. Wang, and M. Kaess, “Edplvo: Efficient direct point-line visual odometry,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 7559–7565.
- [145] N. Yang, L. v. Stumberg, R. Wang, and D. Cremers, “D3vo: Deep depth, deep pose and deep uncertainty for monocular visual odometry,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 1281–1292.

- [146] J. Pelecanos and S. Sridharan, “Feature warping for robust speaker verification,” in Proceedings of 2001 A speaker Odyssey: the speaker recognition workshop. European Speech Communication Association, 2001, pp. 213–218.
- [147] W. Wang, Y. Hu, and S. Scherer, “Tartanvo: A generalizable learning-based vo,” in Conference on Robot Learning. PMLR, 2021, pp. 1761–1772.
- [148] S. Wang, R. Clark, H. Wen, and N. Trigoni, “Deepvo: Towards end-to-end visual odometry with deep recurrent convolutional neural networks,” in 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017, pp. 2043–2050.
- [149] S. Shen, Y. Cai, W. Wang, and S. Scherer, “Dytanvo: Joint refinement of visual odometry and motion segmentation in dynamic environments,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 4048–4055.
- [150] C. Zhao, Y. Tang, Q. Sun, and A. V. Vasilakos, “Deep direct visual odometry,” IEEE transactions on intelligent transportation systems, vol. 23, no. 7, pp. 7733–7742, 2021.
- [151] Z. Teed, L. Lipson, and J. Deng, “Deep patch visual odometry,” Advances in Neural Information Processing Systems, vol. 36, pp. 39033–39051, 2023.
- [152] K. Xu, Y. Hao, S. Yuan, C. Wang, and L. Xie, “Airslam: An efficient and illumination-robust point-line visual slam system,” IEEE Transactions on Robotics, 2025.
- [153] L. Lipson, Z. Teed, and J. Deng, “Deep patch visual slam,” in European Conference on Computer Vision. Springer, 2024, pp. 424– 440.
- [154] L. Lai, Z. Shangguan, J. Zhang, and E. Ohn-Bar, “Xvo: Generalized visual odometry via cross-modal self-training,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 10094–10105.
- [155] F. Wimbauer, W. Chen, D. Muhle, C. Rupprecht, and D. Cremers, “Anycam: Learning to recover camera poses and intrinsics from casual videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [156] C. Rockwell, J. Tung, T.-Y. Lin, M.-Y. Liu, D. F. Fouhey, and C.-H. Lin, “Dynamic camera poses and where to find them,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 12444–12455.
- [157] S. Zhang, J. He, Y. Zhu, J. Wu, and J. Yuan, “Efficient camera exposure control for visual odometry via deep reinforcement learning,” IEEE Robotics and Automation Letters, 2024.
- [158] N. Messikommer, G. Cioffi, M. Gehrig, and D. Scaramuzza, “Reinforcement learning meets visual odometry,” in European Conference on Computer Vision. Springer, 2024, pp. 76–92.
- [159] Q. Wang, Y.-Y. Chang, R. Cai, Z. Li, B. Hariharan, A. Holynski, and N. Snavely, “Tracking everything everywhere all at once,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19795–19806.
- [160] Y. Song, J. Lei, Z. Wang, L. Liu, and K. Daniilidis, “Track everything everywhere fast and robustly,” in European Conference on Computer Vision. Springer, 2024, pp. 343–359.
- [161] Y. Xiao, Q. Wang, S. Zhang, N. Xue, S. Peng, Y. Shen, and X. Zhou, “Spatialtracker: Tracking any 2d pixels in 3d space,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20406–20417.
- [162] B. Wang, J. Li, Y. Yu, L. Liu, Z. Sun, and D. Hu, “Scenetracker: Long-term scene flow estimation network,” arXiv preprint arXiv:2403.19924, 2024.
- [163] T. D. Ngo, P. Zhuang, C. Gan, E. Kalogerakis, S. Tulyakov, H.-Y. Lee, and C. Wang, “Delta: Dense efficient long-range 3d tracking for any video,” arXiv preprint arXiv:2410.24211, 2024.
- [164] S. Cho, J. Huang, S. Kim, and J.-Y. Lee, “Seurat: From moving points to depth,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 7211–7221.
- [165] B. Zhang, L. Ke, A. W. Harley, and K. Fragkiadaki, “Tapip3d: Tracking any point in persistent 3d geometry,” arXiv preprint arXiv:2504.14717, 2025.
- [166] A. Darkhalil, R. Guerrier, A. W. Harley, and D. Damen, “Egopoints: Advancing point tracking for egocentric videos,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 8556–8565.
- [167] J. Kopf, X. Rong, and J.-B. Huang, “Robust consistent video depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 1611–1621.

- [168] Z. Zhang, F. Cole, Z. Li, M. Rubinstein, N. Snavely, and W. T. Freeman, “Structure and motion from casual videos,” in European Conference on Computer Vision. Springer, 2022, pp. 20–37.
- [169] Z. Li, R. Tucker, F. Cole, Q. Wang, L. Jin, V. Ye, A. Kanazawa, A. Holynski, and N. Snavely, “Megasam: Accurate, fast, and robust structure and motion from casual dynamic videos,” arXiv preprint arXiv:2412.04463, 2024.
- [170] X. Chen, Y. Chen, Y. Xiu, A. Geiger, and A. Chen, “Easi3r: Estimating disentangled motion from dust3r without training,” arXiv preprint arXiv:2503.24391, 2025.
- [171] T.-X. Xu, X. Gao, W. Hu, X. Li, S.-H. Zhang, and Y. Shan, “Geometrycrafter: Consistent geometry estimation for open-world videos with diffusion priors,” arXiv preprint arXiv:2504.01016, 2025.
- [172] H. Wang and L. Agapito, “3d reconstruction with spatial memory,” arXiv preprint arXiv:2408.16061, 2024.
- [173] Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa, “Continuous 3d perception model with persistent state,” arXiv preprint arXiv:2501.12387, 2025.
- [174] Y. Wu, W. Zheng, J. Zhou, and J. Lu, “Point3r: Streaming 3d reconstruction with explicit spatial pointer memory,” arXiv preprint arXiv:2507.02863, 2025.
- [175] D. Zhuo, W. Zheng, J. Guo, Y. Wu, J. Zhou, and J. Lu, “Streaming 4d visual geometry transformer,” arXiv preprint arXiv:2507.11539, 2025.
- [176] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “π3: Scalable permutationequivariant visual geometry learning,” 2025. [Online]. Available: https://arxiv.org/abs/2507.13347
- [177] A. Team, H. Zhu, Y. Wang, J. Zhou, W. Chang, Y. Zhou, Z. Li, J. Chen, C. Shen, J. Pang et al., “Aether: Geometric-aware unified world modeling,” arXiv preprint arXiv:2503.18945, 2025.
- [178] Z. Jiang, C. Zheng, I. Laina, D. Larlus, and A. Vedaldi, “Geo4d: Leveraging video generators for geometric 4d scene reconstruction,” arXiv preprint arXiv:2504.07961, 2025.
- [179] Y.-T. Sun, X. Yu, Z. Huang, Y.-H. Huang, Y.-C. Guo, Z. Yang, Y.P. Cao, and X. Qi, “Unigeo: Taming video diffusion for unified consistent geometry estimation,” arXiv preprint arXiv:2505.24521, 2025.
- [180] D. Y. Yao, A. J. Zhai, and S. Wang, “Uni4d: Unifying visual foundation models for 4d modeling from a single video,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1116–1126.
- [181] W. Chen, G. Zhang, F. Wimbauer, R. Wang, N. Araslanov, A. Vedaldi, and D. Cremers, “Back on track: Bundle adjustment for dynamic scene reconstruction,” arXiv preprint arXiv:2504.14516, 2025.
- [182] L. Jin, R. Tucker, Z. Li, D. Fouhey, N. Snavely, and A. Holynski, “Stereo4d: Learning how things move in 3d from internet stereo videos,” arXiv preprint arXiv:2412.09621, 2024.
- [183] E. Sucar, Z. Lai, E. Insafutdinov, and A. Vedaldi, “Dynamic point maps: A versatile representation for dynamic 3d reconstruction,” arXiv preprint arXiv:2503.16318, 2025.
- [184] H. Feng, J. Zhang, Q. Wang, Y. Ye, P. Yu, M. J. Black, T. Darrell, and A. Kanazawa, “St4rtrack: Simultaneous 4d reconstruction and tracking in the world,” arXiv preprint arXiv:2504.13152, 2025.
- [185] S. Zhang, Y. Ge, J. Tian, G. Xu, H. Chen, C. Lv, and C. Shen, “Pomato: Marrying pointmap matching with temporal motion for dynamic 3d reconstruction,” arXiv preprint arXiv:2504.05692, 2025.
- [186] J. Han, H. An, J. Jung, T. Narihira, J. Seo, K. Fukuda, C. Kim, S. Hong, Y. Mitsufuji, and S. Kim, “Dˆ 2ust3r: Enhancing 3d reconstruction with 4d pointmaps for dynamic scenes,” arXiv preprint arXiv:2504.06264, 2025.
- [187] Y. Liang, A. Badki, H. Su, J. Tompkin, and O. Gallo, “Zeroshot monocular scene flow estimation in the wild,” arXiv preprint arXiv:2501.10357, 2025.
- [188] Y. Kasten, W. Lu, and H. Maron, “Fast encoder-based 3d from casual videos via point track processing,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems.
- [189] S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah, “Transformers in vision: A survey,” ACM computing surveys (CSUR), vol. 54, no. 10s, pp. 1–41, 2022.
- [190] Y. Xiao, J. Wang, N. Xue, N. Karaev, Y. Makarov, B. Kang, X. Zhu, H. Bao, Y. Shen, and X. Zhou, “Spatialtrackerv2: 3d point tracking made easy,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. [Online]. Available: https://arxiv.org/abs/2507.12462

- [191] R. Li, S. Wang, Z. Long, and D. Gu, “Undeepvo: Monocular visual odometry through unsupervised deep learning,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 7286–7291.
- [192] Z. Teed and J. Deng, “Deepv2d: Video to depth with differentiable structure from motion,” arXiv preprint arXiv:1812.04605, 2018.
- [193] S. Li, X. Wu, Y. Cao, and H. Zha, “Generalizing to the open world: Deep visual odometry with online adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 13184–13193.
- [194] L. Sun, W. Yin, E. Xie, Z. Li, C. Sun, and C. Shen, “Improving monocular visual odometry using learned depth,” IEEE Transactions on Robotics, vol. 38, no. 5, pp. 3173–3186, 2022.
- [195] W. Zhao, S. Liu, Y. Shu, and Y.-J. Liu, “Towards better generalization: Joint depth-pose learning without posenet,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 9151–9161.
- [196] G. Lu, “Deep unsupervised visual odometry via bundle adjusted pose graph optimization,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 6131–6137.
- [197] Z. Teed and J. Deng, “Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras,” Advances in neural information processing systems, vol. 34, pp. 16558–16569, 2021.
- [198] Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and M. Pollefeys, “Nicer-slam: Neural implicit scene encoding for rgb slam,” in 2024 International Conference on 3D Vision (3DV). IEEE, 2024, pp. 42–52.
- [199] H. Li, X. Gu, W. Yuan, L. Yang, Z. Dong, and P. Tan, “Dense rgb slam with neural implicit maps,” arXiv preprint arXiv:2301.08930, 2023.
- [200] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, “Go-slam: Global optimization for consistent 3d instant reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3727–3737.
- [201] G. Zhang, E. Sandstr¨om, Y. Zhang, M. Patel, L. Van Gool, and M. R. Oswald, “Glorie-slam: Globally optimized rgb-only implicit encoding point cloud slam,” arXiv preprint arXiv:2403.19549, 2024.
- [202] E. Sandstr¨om, K. Tateno, M. Oechsle, M. Niemeyer, L. Van Gool, M. R. Oswald, and F. Tombari, “Splat-slam: Globally optimized rgb-only slam with 3d gaussians,” arXiv preprint arXiv:2405.16544, 2024.
- [203] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian splatting slam,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18039–18048.
- [204] F. Tosi, Y. Zhang, Z. Gong, E. Sandstr¨om, S. Mattoccia, M. Oswald, and M. Poggi, “How nerfs and 3d gaussian splatting are reshaping slam: A survey. arxiv 2024,” arXiv preprint arXiv:2402.13255, 2024.
- [205] C. Smith, D. Charatan, A. Tewari, and V. Sitzmann, “Flowmap: High-quality camera poses, intrinsics, and depth via gradient descent,” arXiv preprint arXiv:2404.15259, 2024.
- [206] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3d with mast3r,” in European Conference on Computer Vision. Springer, 2024, pp. 71–91.
- [207] B. Duisterhof, L. Zust, P. Weinzaepfel, V. Leroy, Y. Cabon, and J. Revaud, “Mast3r-sfm: a fully-integrated solution for unconstrained structure-from-motion,” arXiv preprint arXiv:2409.19152, 2024.
- [208] S. Elflein, Q. Zhou, S. Agostinho, and L. Leal-Taix´e, “Light3r-sfm: Towards feed-forward structure-from-motion,” arXiv preprint arXiv:2501.14914, 2025.
- [209] Y. Cabon, L. Stoffl, L. Antsfeld, G. Csurka, B. Chidlovskii, J. Revaud, and V. Leroy, “Must3r: Multi-view network for stereo 3d reconstruction,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1050–1060.
- [210] W. Jang, P. Weinzaepfel, V. Leroy, L. Agapito, and J. Revaud, “Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1071–1081.
- [211] S. Liu, W. Li, P. Qiao, and Y. Dou, “Regist3r: Incremental registration with stereo foundation model,” arXiv preprint arXiv:2504.12356, 2025.
- [212] H. Pfister, M. Zwicker, J. Van Baar, and M. Gross, “Surfels: Surface elements as rendering primitives,” in Proceedings of the 27th annual

- conference on Computer graphics and interactive techniques, 2000, pp. 335–342.
- [213] W. Yifan, F. Serena, S. Wu, C. Oztireli,¨ and O. Sorkine-Hornung, “Differentiable surface splatting for point-based geometry processing,” ACM Transactions on Graphics (TOG), vol. 38, no. 6, pp. 1–14, 2019.
- [214] P. Dai, Y. Zhang, Z. Li, S. Liu, and B. Zeng, “Neural point cloud rendering via multi-plane projection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7830–7839.
- [215] O. Wiles, G. Gkioxari, R. Szeliski, and J. Johnson, “Synsin: Endto-end view synthesis from a single image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7467–7477.
- [216] C. Lassner and M. Zollhofer, “Pulsar: Efficient sphere-based neural rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 1440–1449.
- [217] G. Kopanas, J. Philip, T. Leimkuhler,¨ and G. Drettakis, “Pointbased neural rendering with per-view optimization,” in Computer Graphics Forum, vol. 40, no. 4. Wiley Online Library, 2021, pp. 29–43.
- [218] D. Ruckert,¨ L. Franke, and M. Stamminger, “Adop: Approximate differentiable one-pixel point rendering,” ACM Transactions on Graphics (ToG), vol. 41, no. 4, pp. 1–14, 2022.
- [219] G. Riegler and V. Koltun, “Free view synthesis,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIX 16. Springer, 2020, pp. 623–640.
- [220] ——, “Stable view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 12216–12225.
- [221] A. Cao, C. Rockwell, and J. Johnson, “Fwd: Real-time novel view synthesis with forward warping and depth,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 15713–15724.
- [222] M. Botsch, L. Kobbelt, M. Pauly, P. Alliez, and B. L´evy, Polygon mesh processing. CRC press, 2010.
- [223] L. A. Shirman and C. H. Sequin, “Local surface interpolation with b´ezier patches,” Computer Aided Geometric Design, vol. 4, no. 4, pp. 279–295, 1987.
- [224] A. Burov, M. Nießner, and J. Thies, “Dynamic surface function networks for clothed human bodies,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10754–10764.
- [225] J. Thies, M. Zollh¨ofer, and M. Nießner, “Deferred neural rendering: Image synthesis using neural textures,” Acm Transactions on Graphics (TOG), vol. 38, no. 4, pp. 1–12, 2019.
- [226] M. M. Loper and M. J. Black, “Opendr: An approximate differentiable renderer,” in Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VII 13. Springer, 2014, pp. 154–169.
- [227] H. Kato, Y. Ushiku, and T. Harada, “Neural 3d mesh renderer,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3907–3916.
- [228] S. Liu, T. Li, W. Chen, and H. Li, “Soft rasterizer: A differentiable renderer for image-based 3d reasoning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 7708–7717.
- [229] H.-T. D. Liu, M. Tao, and A. Jacobson, “Paparazzi: surface editing by way of multi-view image processing.” ACM Trans. Graph., vol. 37, no. 6, pp. 221–1, 2018.
- [230] M. Nimier-David, D. Vicini, T. Zeltner, and W. Jakob, “Mitsuba 2: A retargetable forward and inverse renderer,” ACM Transactions on Graphics (TOG), vol. 38, no. 6, pp. 1–17, 2019.
- [231] Y. Hu, T.-M. Li, L. Anderson, J. Ragan-Kelley, and F. Durand, “Taichi: a language for high-performance computation on spatially sparse data structures,” ACM Transactions on Graphics (TOG), vol. 38, no. 6, p. 201, 2019.
- [232] N. Max, “Optical models for direct volume rendering,” IEEE Transactions on Visualization and Computer Graphics, vol. 1, no. 2, pp. 99–108, 1995.
- [233] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, “Nerf in the wild: Neural radiance fields for unconstrained photo collections,” in CVPR, 2021, pp. 7210–7219.
- [234] K. Zhang, F. Luan, Q. Wang, K. Bala, and N. Snavely, “Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting,” in Proceedings of the IEEE/CVF

- Conference on Computer Vision and Pattern Recognition, 2021, pp. 5453–5462.
- [235] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, “Barf: Bundleadjusting neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5741–5751.
- [236] C.-Y. Weng, B. Curless, P. P. Srinivasan, J. T. Barron, and

I. Kemelmacher-Shlizerman, “Humannerf: Free-viewpoint rendering of moving people from monocular video,” in Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, 2022, pp. 16210–16220.

- [237] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction,” Advances in Neural Information Processing Systems, vol. 34, pp. 27171–27183, 2021.
- [238] L. Yariv, J. Gu, Y. Kasten, and Y. Lipman, “Volume rendering of neural implicit surfaces,” Advances in Neural Information Processing Systems, vol. 34, pp. 4805–4815, 2021.
- [239] M. Niemeyer and A. Geiger, “GIRAFFE: Representing scenes as compositional generative neural feature fields,” 2021.
- [240] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. MartinBrualla, and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5855–5864.
- [241] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5470–5479.
- [242] D. Verbin, P. Hedman, B. Mildenhall, T. Zickler, J. T. Barron, and P. P. Srinivasan, “Ref-nerf: Structured view-dependent appearance for neural radiance fields,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 5481–5490.
- [243] Z. Li, Q. Wang, F. Cole, R. Tucker, and N. Snavely, “Dynibar: Neural dynamic image-based rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4273–4284.
- [244] S. Zhi, T. Laidlow, S. Leutenegger, and A. J. Davison, “In-place scene labelling and understanding with implicit scene representation,” in ICCV, 2021, pp. 15838–15847.
- [245] P. Truong, M.-J. Rakotosaona, F. Manhardt, and F. Tombari, “Sparf: Neural radiance fields from sparse and noisy poses,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4190–4200.
- [246] M. Boss, R. Braun, V. Jampani, J. T. Barron, C. Liu, and H. Lensch, “Nerd: Neural reflectance decomposition from image collections,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12684–12694.
- [247] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural radiance fields from one or few images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4578–4587.
- [248] Q. Wang, Z. Wang, K. Genova, P. P. Srinivasan, H. Zhou, J. T. Barron, R. Martin-Brualla, N. Snavely, and T. Funkhouser, “IBRNet: Learning multi-view image-based rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4690–4699.
- [249] C. Reiser, S. Peng, Y. Liao, and A. Geiger, “Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 14335–14345.
- [250] Y. Du, Y. Zhang, H.-X. Yu, J. B. Tenenbaum, and J. Wu, “Neural radiance flow for 4d view synthesis and video processing,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society, 2021, pp. 14304–14314.
- [251] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim, T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe et al., “Neural 3d video synthesis from multi-view video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5521–5531.
- [252] S. Peng, J. Dong, Q. Wang, S. Zhang, Q. Shuai, X. Zhou, and H. Bao, “Animatable neural radiance fields for modeling dynamic human bodies,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 14314–14323.
- [253] A. Zhou, M. J. Kim, L. Wang, P. Florence, and C. Finn, “Nerf in the palm of your hand: Corrective augmentation for robotics via novel-view synthesis,” in Proceedings of the IEEE/CVF Conference

- on Computer Vision and Pattern Recognition, 2023, pp. 17907– 17917.
- [254] D. Ruckert,¨ Y. Wang, R. Li, R. Idoughi, and W. Heidrich, “Neat: Neural adaptive tomography,” ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 1–13, 2022.
- [255] A. Levis, P. P. Srinivasan, A. A. Chael, R. Ng, and K. L. Bouman, “Gravitationally lensed black hole emission tomography,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 19841–19850.
- [256] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” CoRR, vol. abs/2111.12077, 2022.
- [257] M. Kazhdan and H. Hoppe, “Screened poisson surface reconstruction,” ACM Transactions on Graphics (ToG), vol. 32, no. 3, pp. 1–13, 2013.
- [258] D.-T. Lee and B. J. Schachter, “Two algorithms for constructing a delaunay triangulation,” International Journal of Computer & Information Sciences, vol. 9, no. 3, pp. 219–242, 1980.
- [259] J. L. Sch¨onberger and J.-M. Frahm, “Structure-from-motion revisited,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [260] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “Mvsnet: Depth inference for unstructured multi-view stereo,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 767– 783.
- [261] X. Meng, W. Chen, and B. Yang, “Neat: Learning neural implicit surfaces with arbitrary topologies from multi-view images,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 248–258.
- [262] Z. Li, T. Muller,¨ A. Evans, R. H. Taylor, M. Unberath, M.-Y. Liu, and C.-H. Lin, “Neuralangelo: High-fidelity neural surface reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8456–8465.
- [263] W. E. Lorensen and H. E. Cline, “Marching cubes: A high resolution 3d surface construction algorithm,” in Seminal graphics: pioneering efforts that shaped the field, 1998, pp. 347–353.
- [264] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024 conference papers, 2024, pp. 1–11.
- [265] Z. Yu, T. Sattler, and A. Geiger, “Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes,” ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp. 1–13, 2024.
- [266] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, “Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction,” IEEE Transactions on Visualization and Computer Graphics, 2024.
- [267] A. Gu´edon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5354–5363.
- [268] S. Chen, Z. Li, Z. Chen, Q. Yan, G. Shen, and R. Duan, “3d gaussian splatting for fine-detailed surface reconstruction in largescale scene,” arXiv preprint arXiv:2506.17636, 2025.
- [269] J. Kim, G. Park, and S. Lee, “Multiview geometric regularization of gaussian splatting for accurate radiance fields,” arXiv preprint arXiv:2506.13508, 2025.
- [270] Y. Xie, H. Xiao, and W. Kang, “Tri 2 plane: Advancing neural implicit surface reconstruction for indoor scenes,” IEEE Transactions on Multimedia, 2025.
- [271] Y. Chen, W. Wu, Y. Peng, Y. Fei, and L. Zheng, “Esa-gs: Elongation splitting and assimilation in gaussian splatting for accurate surface reconstruction,” Computer Aided Geometric Design, p. 102434, 2025.
- [272] S. Li, Y.-S. Liu, and Z. Han, “Gaussianudf: Inferring unsigned distance functions through 3d gaussian splatting,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 27113–27123.
- [273] L. Radl, F. Windisch, T. Deixelberger, J. Hladky, M. Steiner, D. Schmalstieg, and M. Steinberger, “Sof: Sorted opacity fields for fast unbounded surface reconstruction,” arXiv preprint arXiv:2506.19139, 2025.
- [274] R. Bruneau, B. Brument, Y. Qu´eau, J. M´elou, F. B. Lauze, J.-D. Durou, and L. Calvet, “Multi-view surface reconstruction using normal and reflectance cues,” arXiv preprint arXiv:2506.04115, 2025.

- [275] Y.-C. Liu, L. H¨ollein, M. Nießner, and A. Dai, “Quicksplat: Fast 3d surface reconstruction via learned gaussian initialization,” arXiv preprint arXiv:2505.05591, 2025.
- [276] K. Jiang, V. Sivaram, C. Peng, and R. Ramamoorthi, “Geometry field splatting with gaussian surfels,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5752– 5762.
- [277] Z. Shen, Y. Liu, Z. Chen, Z. Li, J. Wang, Y. Liang, Z. Yu, J. Zhang, Y. Xu, S. Schaefer et al., “Solidgs: Consolidating gaussian surfel splatting for sparse-view surface reconstruction,” arXiv preprint arXiv:2412.15400, 2024.
- [278] X. Long, C. Lin, P. Wang, T. Komura, and W. Wang, “Sparseneus: Fast generalizable neural surface reconstruction from sparse views,” in European Conference on Computer Vision. Springer, 2022, pp. 210–227.
- [279] R. Peng, X. Gu, L. Tang, S. Shen, F. Yu, and R. Wang, “Gens: Generalizable neural surface reconstruction from multi-view images,” Advances in Neural Information Processing Systems, vol. 36, pp. 56932–56945, 2023.
- [280] L. Xu, T. Guan, Y. Wang, W. Liu, Z. Zeng, J. Wang, and W. Yang, “C2f2neus: Cascade cost frustum fusion for high fidelity and generalizable neural surface reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18291–18301.
- [281] Y. Na, W. J. Kim, K. B. Han, S. Ha, and S.-E. Yoon, “Uforecon: generalizable sparse-view surface reconstruction from arbitrary and unfavorable sets,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5094–5104.
- [282] R. Peng, S. Shen, K. Xiong, H. Gao, J. Jiao, X. Gu, and R. Wang, “Surface-centric modeling for high-fidelity generalizable neural surface reconstruction,” in European Conference on Computer Vision. Springer, 2024, pp. 183–200.
- [283] Y. Liang, H. He, and Y. Chen, “Retr: Modeling rendering via transformer for generalizable neural surface reconstruction,” Advances in neural information processing systems, vol. 36, pp. 62332– 62351, 2023.
- [284] A. Chen, H. Xu, S. Esposito, S. Tang, and A. Geiger, “Lara: Efficient large-baseline radiance fields,” in European Conference on Computer Vision. Springer, 2024, pp. 338–355.
- [285] A. Furnari, D. Crandall, D. Damen, K. Grauman, and G. M. Farinella, “Special section on egocentric perception,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 6, pp. 6602–6604, 2023.
- [286] C. Plizzari, G. Goletto, A. Furnari, S. Bansal, F. Ragusa, G. M. Farinella, D. Damen, and T. Tommasi, “An outlook into the future of egocentric vision,” International Journal of Computer Vision, vol. 132, no. 11, pp. 4880–4936, 2024.
- [287] A. Avetisyan, C. Xie, H. Howard-Jenkins, T.-Y. Yang, S. Aroudj, S. Patra, F. Zhang, D. Frost, L. Holland, C. Orme et al., “Scenescript: Reconstructing scenes with an autoregressive structured language model,” in European Conference on Computer Vision. Springer, 2024, pp. 247–263.
- [288] Q. Gu, Z. Lv, D. Frost, S. Green, J. Straub, and C. Sweeney, “Egolifter: Open-world 3d segmentation for egocentric perception,” in European Conference on Computer Vision. Springer, 2024, pp. 382–400.
- [289] Z. Lv, M. Monge, K. Chen, Y. Zhu, M. Goesele, J. Engel, Z. Dong, and R. Newcombe, “Photoreal scene reconstruction from an egocentric device,” in ACM SIGGRAPH, 2025.
- [290] C. Plizzari, S. Goel, T. Perrett, J. Chalk, A. Kanazawa, and D. Damen, “Spatial cognition from egocentric video: Out of sight, not out of mind,” arXiv preprint arXiv:2404.05072, 2024.
- [291] K. Zhang, G. Riegler, N. Snavely, and V. Koltun, “Nerf++: Analyzing and improving neural radiance fields,” CoRR, vol. abs/2010.07492, 2020.
- [292] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Zip-nerf: Anti-aliased grid-based neural radiance fields,” CoRR, vol. abs/2304.06706, 2023.
- [293] Y. Liu, C. Luo, L. Fan, N. Wang, J. Peng, and Z. Zhang, “Citygaussian: Real-time high-quality large-scale scene rendering with gaussians,” in European Conference on Computer Vision. Springer, 2024, pp. 265–282.
- [294] Y. Liu, C. Luo, Z. Mao, J. Peng, and Z. Zhang, “Citygaussianv2: Efficient and geometrically accurate reconstruction for large-scale scenes,” arXiv preprint arXiv:2411.00771, 2024.
- [295] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, “Octree-

- gs: Towards consistent real-time rendering with lod-structured 3d gaussians,” arXiv preprint arXiv:2403.17898, 2024.
- [296] Y. Gao, H. Li, J. Chen, Z. Zou, Z. Zhong, D. Zhang, X. Sun, and J. Han, “Citygs-x: A scalable architecture for efficient and geometrically accurate large-scale scene reconstruction,” arXiv preprint arXiv:2503.23044, 2025.
- [297] J. Kulhanek, M.-J. Rakotosaona, F. Manhardt, C. Tsalicoglou, M. Niemeyer, T. Sattler, S. Peng, and F. Tombari, “Lodge: Levelof-detail large-scale gaussian splatting with efficient rendering,” arXiv preprint arXiv:2505.23158, 2025.
- [298] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan, J. T. Barron, and H. Kretzschmar, “Block-nerf: Scalable large-scene neural view synthesis,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2022.
- [299] H. Turki, D. Ramanan, and M. Satyanarayanan, “Mega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs,” CoRR, vol. abs/2112.10703, 2022.
- [300] Y. Xiangli, L. Xu, X. Pan, N. Zhao, A. Rao, C. Theobalt, B. Dai, and D. Lin, “Bungeenerf (city-nerf): Progressive neural radiance field for extreme multi-scale scene rendering,” in European Conf. Computer Vision (ECCV), 2022, pp. 106–122.
- [301] P. Wang, Y. Liu, Z. Chen, L. Liu, Z. Liu, T. Komura, C. Theobalt, and W. Wang, “F2-nerf: Fast neural radiance field training with free camera trajectories,” Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2023.
- [302] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, “Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” CoRR, vol. abs/2312.00109, 2023.
- [303] Z. Yu, S. Peng, M. Niemeyer, T. Sattler, and A. Geiger, “Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction,” CoRR, vol. abs/2206.00665, 2022.
- [304] Y. Chen and G. H. Lee, “Scalar-nerf: Scalable large-scale neural radiance fields for scene reconstruction,” CoRR, vol. abs/2311.16657, 2023.
- [305] Z. Mi and D. Xu, “Switch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields,” in International Conference on Learning Representations (ICLR), 2023.
- [306] J. Sun, Y. Xie, L. Chen, X. Zhou, and H. Bao, “Neuralrecon: Real-time coherent 3d reconstruction from monocular video,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 15598–15607.
- [307] K. Cho, B. Van Merri¨enboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using rnn encoder-decoder for statistical machine translation,” arXiv preprint arXiv:1406.1078, 2014.
- [308] A. Bozic, P. Palafox, J. Thies, A. Dai, and M. Nießner, “Transformerfusion: Monocular rgb scene reconstruction using transformers,” Advances in Neural Information Processing Systems, vol. 34, pp. 1403–1414, 2021.
- [309] L. Wang, Y. Gong, Q. Wang, K. Zhou, and L. Chen, “Flora: dual-frequency loss-compensated real-time monocular 3d video reconstruction,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 2, 2023, pp. 2599–2607.
- [310] H. Gao, W. Mao, and M. Liu, “Visfusion: Visibility-aware online 3d scene reconstruction from videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17317–17326.
- [311] J. Ju, C. W. Tseng, O. Bailo, G. Dikov, and M. Ghafoorian, “Dgrecon: Depth-guided neural 3d scene reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18184–18194.
- [312] Z. Feng, L. Yang, P. Guo, and B. Li, “Cvrecon: Rethinking 3d geometric feature learning for neural reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17750–17760.
- [313] N. Stier, A. Ranjan, A. Colburn, Y. Yan, L. Yang, F. Ma, and B. Angles, “Finerecon: Depth-aware feed-forward network for detailed 3d reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18423–18432.
- [314] R. Li, U. Mahbub, V. Bhaskaran, and T. Nguyen, “Monoselfrecon: Purely self-supervised explicit generalizable 3d reconstruction of indoor scenes from monocular rgb views,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 656–666.
- [315] W. Bei, X. Fan, H. Jian, X. Du, D. Yan, J. Xu, and Q. Ge, “Georecon: a coarse-to-fine visual 3d reconstruction approach for high-

- resolution images with neural matching priors,” International Journal of Digital Earth, vol. 17, no. 1, p. 2421956, 2024.
- [316] F. Chu, Y. Cong, Y. Wang, and R. Chen, “Detailrecon: Focusing on detailed regions for online monocular 3d reconstruction,” IEEE Transactions on Multimedia, 2025.
- [317] D. Casillas-Perez, D. Pizarro, D. Fuentes-Jimenez, M. Mazo, and A. Bartoli, “The isowarp: the template-based visual geometry of isometric surfaces,” International Journal of Computer Vision, vol. 129, no. 7, pp. 2194–2222, 2021.
- [318] N. Kairanda, E. Tretschk, M. Elgharib, C. Theobalt, and V. Golyanik, “f-sft: Shape-from-template with a physics-based deformation model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3948–3958.
- [319] S. Zuffi, A. Kanazawa, D. W. Jacobs, and M. J. Black, “3d menagerie: Modeling the 3d shape and pose of animals,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 6365–6373.
- [320] X. Li, S. Liu, S. De Mello, K. Kim, X. Wang, M.-H. Yang, and J. Kautz, “Online adaptation for consistent mesh reconstruction in the wild,” Advances in Neural Information Processing Systems, vol. 33, pp. 15009–15019, 2020.
- [321] G. Yang, D. Sun, V. Jampani, D. Vlasic, F. Cole, H. Chang, D. Ramanan, W. T. Freeman, and C. Liu, “Lasr: Learning articulated shape reconstruction from a monocular video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 15980–15989.
- [322] G. Yang, D. Sun, V. Jampani, D. Vlasic, F. Cole, C. Liu, and D. Ramanan, “Viser: Video-specific surface embeddings for articulated 3d shape reconstruction,” Advances in Neural Information Processing Systems, vol. 34, pp. 19326–19338, 2021.
- [323] G. Yang, M. Vo, N. Neverova, D. Ramanan, A. Vedaldi, and H. Joo, “Banmo: Building animatable 3d neural models from many casual videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 2863–2873.
- [324] G. Yang, S. Yang, J. Z. Zhang, Z. Manchester, and D. Ramanan, “Ppr: Physically plausible reconstruction from monocular videos,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3914–3924.
- [325] C. Song, J. Wei, C. S. Foo, G. Lin, and F. Liu, “Reacto: Reconstructing articulated objects from a single video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5384–5395.
- [326] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, “Deepsdf: Learning continuous signed distance functions for shape representation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 165–174.
- [327] W. Mao, R. Hartley, M. Salzmann et al., “Neural sdf flow for 3d reconstruction of dynamic scenes,” in The Twelfth International Conference on Learning Representations.
- [328] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y. Liu, “Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 16632–16642.
- [329] E. Johnson, M. Habermann, S. Shimada, V. Golyanik, and C. Theobalt, “Unbiased 4d: Monocular 4d reconstruction with a neural deformation model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6598–6607.
- [330] J. Choe, C. Choy, J. Park, I. S. Kweon, and A. Anandkumar, “Spacetime surface regularization for neural dynamic scene reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17871–17881.
- [331] I. Liu, H. Su, and X. Wang, “Dynamic gaussians mesh: Consistent mesh reconstruction from monocular videos,” arXiv preprint arXiv:2404.12379, 2024.
- [332] S. Ma, Y. Luo, and Y. Yang, “Reconstructing and simulating dynamic 3d objects with mesh-adsorbed gaussian splatting,” arXiv preprint arXiv:2406.01593, 2024.
- [333] W. Cai, W. Ye, P. Ye, T. He, and T. Chen, “Dynasurfgs: Dynamic surface reconstruction with planar-based gaussian splatting,” arXiv preprint arXiv:2408.13972, 2024.
- [334] X. Li, J. Tong, J. Hong, V. Rolland, and L. Petersson, “Dgns: Deformable gaussian splatting and dynamic neural surface for monocular dynamic 3d reconstruction,” arXiv preprint arXiv:2412.03910, 2024.

- [335] S. Wang, B. Huang, R. Wang, and S. Gao, “Space-time 2d gaussian splatting for accurate surface reconstruction under complex dynamic scenes,” arXiv preprint arXiv:2409.18852, 2024.
- [336] C. Zheng, L. Xue, J. Zarate, and J. Song, “Gstar: Gaussian surface tracking and reconstruction,” arXiv preprint arXiv:2501.10283, 2025.
- [337] D. Chen, B. Oberson, I. Feldmann, O. Schreer, A. Hilsmann, and P. Eisert, “Adaptive and temporally consistent gaussian surfels for multi-view dynamic reconstruction,” arXiv preprint arXiv:2411.06602, 2024.
- [338] M. Magnor, M. Pollefeys, G. Cheung, W. Matusik, and C. Theobalt, “Video-based rendering,” in ACM SIGGRAPH 2005 Courses, 2005, pp. 1–es.
- [339] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “D-nerf: Neural radiance fields for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 10318–10327.
- [340] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20310–20320.
- [341] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20331–20341.
- [342] J. Bae, S. Kim, Y. Yun, H. Lee, G. Bang, and Y. Uh, “Per-gaussian embedding-based deformation for deformable 3d gaussian splatting,” in European Conference on Computer Vision. Springer, 2024, pp. 321–335.
- [343] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi, “Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4220–4230.
- [344] J. Lu, J. Deng, R. Zhu, Y. Liang, W. Yang, X. Zhou, and T. Zhang, “Dn-4dgs: Denoised deformable network with temporal-spatial aggregation for dynamic scene rendering,” Advances in Neural Information Processing Systems, vol. 37, pp. 84114–84138, 2024.
- [345] Z. Lu, X. Guo, L. Hui, T. Chen, M. Yang, X. Tang, F. Zhu, and Y. Dai, “3d geometry-aware deformable gaussian splatting for dynamic view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8900–8910.
- [346] Q. Liu, Y. Liu, J. Wang, X. Lv, P. Wang, W. Wang, and J. Hou, “Modgs: Dynamic gaussian splatting from causually-captured monocular videos,” arXiv preprint arXiv:2406.00434, 2024.
- [347] D. Das, C. Wewer, R. Yunus, E. Ilg, and J. E. Lenssen, “Neural parametric gaussians for monocular non-rigid object reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10715–10725.
- [348] J. Lei, Y. Weng, A. Harley, L. Guibas, and K. Daniilidis, “Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds,” arXiv preprint arXiv:2405.17421, 2024.
- [349] C. Wang, L. E. MacDonald, L. A. Jeni, and S. Lucey, “Flow supervision for deformable nerf,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21128–21137.
- [350] Q. Gao, Q. Xu, Z. Cao, B. Mildenhall, W. Ma, L. Chen, D. Tang, and U. Neumann, “Gaussianflow: Splatting gaussian dynamics for 4d content creation,” arXiv preprint arXiv:2403.12365, 2024.
- [351] R. Zhu, Y. Liang, H. Chang, J. Deng, J. Lu, W. Yang, T. Zhang, and Y. Zhang, “Motiongs: Exploring explicit motion guidance for deformable 3d gaussian splatting,” Advances in Neural Information Processing Systems, vol. 37, pp. 101790–101817, 2024.
- [352] S. S. Beauchemin and J. L. Barron, “The computation of optical flow,” ACM computing surveys (CSUR), vol. 27, no. 3, pp. 433–466, 1995.
- [353] J. S. Yoon, K. Kim, O. Gallo, H. S. Park, and J. Kautz, “Novel view synthesis of dynamic scenes with globally coherent depths from a monocular camera,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 5336–5345.
- [354] W. Xian, J.-B. Huang, J. Kopf, and C. Kim, “Space-time neural irradiance fields for free-viewpoint video,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 9421–9431.
- [355] Q. Wang, V. Ye, H. Gao, J. Austin, Z. Li, and A. Kanazawa, “Shape of motion: 4d reconstruction from a single video,” arXiv preprint arXiv:2407.13764, 2024.

- [356] Y.-L. Liu, C. Gao, A. Meuleman, H.-Y. Tseng, A. Saraf, C. Kim, Y.Y. Chuang, J. Kopf, and J.-B. Huang, “Robust dynamic radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13–23.
- [357] A. Cao and J. Johnson, “Hexplane: A fast representation for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 130–141.
- [358] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, “K-planes: Explicit radiance fields in space, time, and appearance,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12479–12488.
- [359] Z. Li, Z. Chen, Z. Li, and Y. Xu, “Spacetime gaussian feature splatting for real-time dynamic view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8508–8520.
- [360] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting,” arXiv preprint arXiv:2310.10642, 2023.
- [361] Y. Duan, F. Wei, Q. Dai, Y. He, W. Chen, and B. Chen, “4drotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.
- [362] C. Stearns, A. Harley, M. Uy, F. Dubost, F. Tombari, G. Wetzstein, and L. Guibas, “Dynamic gaussian marbles for novel view synthesis of casual monocular videos,” in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1–11.
- [363] M. Kocabas, J.-H. R. Chang, J. Gabriel, O. Tuzel, and A. Ranjan, “Hugs: Human gaussian splats,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 505– 515.
- [364] B. P. Duisterhof, Z. Mandi, Y. Yao, J.-W. Liu, M. Z. Shou, S. Song, and J. Ichnowski, “Md-splatting: Learning metric deformation from 4d gaussians in highly deformable scenes,” arXiv preprint arXiv:2312.00583, 2023.
- [365] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis,” arXiv preprint arXiv:2308.09713, 2023.
- [366] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20675–20685.
- [367] K. Katsumata, D. M. Vo, and H. Nakayama, “An efficient 3d gaussian representation for monocular/multi-view dynamic scenes,” arXiv preprint arXiv:2311.12897, 2023.
- [368] F. Tian, S. Du, and Y. Duan, “Mononerf: Learning a generalizable dynamic radiance field from monocular videos,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17903–17913.
- [369] B. Van Hoorick, P. Tendulkar, D. Sur´ıs, D. Park, S. Stent, and C. Vondrick, “Revealing occlusions with 4d neural fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 3011–3021.
- [370] M. Busching,¨ J. Bengtson, D. Nilsson, and M. Bj¨orkman, “Flowibr: Leveraging pre-training for efficient neural image-based rendering of dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8016–8026.
- [371] X. Zhao, R. A. Colburn, F. Ma, M. A.´ Bautista, J. M. Susskind, and A. Schwing, “Pseudo-generalized dynamic view synthesis from a video,” in The Twelfth International Conference on Learning Representations, 2024.
- [372] H. Liang, J. Ren, A. Mirzaei, A. Torralba, Z. Liu, I. Gilitschenski, S. Fidler, C. Oztireli, H. Ling, Z. Gojcic et al., “Feed-forward bullet-time reconstruction of dynamic scenes from monocular videos,” arXiv preprint arXiv:2412.03526, 2024.
- [373] K. Xu, T. H. E. Tse, J. Peng, and A. Yao, “Das3r: Dynamics-aware gaussian splatting for static scene reconstruction,” arXiv preprint arXiv:2412.19584, 2024.
- [374] H. Li, H. Chen, C. Ye, Z. Chen, B. Li, S. Xu, X. Guo, X. Liu, Y. Wang, B. Zhang et al., “Light of normals: Unified feature representation for universal photometric stereo,” arXiv preprint arXiv:2506.18882, 2025.
- [375] M. Omran, C. Lassner, G. Pons-Moll, P. Gehler, and B. Schiele, “Neural body fitting: Unifying deep learning and model based human pose and shape estimation,” in 2018 international conference on 3D vision (3DV). IEEE, 2018, pp. 484–494.

- [376] B. Yi, V. Ye, M. Zheng, Y. Li, L. Muller,¨ G. Pavlakos, Y. Ma, J. Malik, and A. Kanazawa, “Estimating body and hand motion in an ego-sensed world,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 7072–7084.
- [377] S. Peng, Y. Zhang, Y. Xu, Q. Wang, Q. Shuai, H. Bao, and X. Zhou, “Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 9054–9063.
- [378] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, “Smpl: A skinned multi-person linear model,” ACM Trans. Graph., vol. 34, no. 6, pp. 248:1–248:16, Oct. 2015.
- [379] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A. A. A. Osman, D. Tzionas, and M. J. Black, “Expressive body capture: 3D hands, face, and body from a single image,” in Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 10975– 10985.
- [380] F. Bogo, A. Kanazawa, C. Lassner, P. Gehler, J. Romero, and M. J. Black, “Keep it smpl: Automatic estimation of 3d human pose and shape from a single image,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 1114, 2016, Proceedings, Part V 14. Springer, 2016, pp. 561–578.
- [381] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A. A. Osman, D. Tzionas, and M. J. Black, “Expressive body capture: 3d hands, face, and body from a single image,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 10975–10985.
- [382] A. Kanazawa, M. J. Black, D. W. Jacobs, and J. Malik, “End-toend recovery of human shape and pose,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 7122–7131.
- [383] G. Pavlakos, L. Zhu, X. Zhou, and K. Daniilidis, “Learning to estimate 3d human pose and shape from a single color image,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 459–468.
- [384] M. Kocabas, C.-H. P. Huang, J. Tesch, L. Muller,¨ O. Hilliges, and M. J. Black, “SPEC: Seeing people in the wild with an estimated camera,” in Proc. International Conference on Computer Vision (ICCV), Oct. 2021, pp. 11035–11045.
- [385] N. Kolotouros, G. Pavlakos, and K. Daniilidis, “Convolutional mesh regression for single-image human shape reconstruction,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4501–4510.
- [386] Z. Li, J. Liu, Z. Zhang, S. Xu, and Y. Yan, “Cliff: Carrying location information in full frames into human pose and shape estimation,” in European Conference on Computer Vision. Springer, 2022, pp. 590–606.
- [387] I. S´ar´andi and G. Pons-Moll, “Neural localizer fields for continuous 3d human pose and shape estimation,” 2024.
- [388] N. Kolotouros, G. Pavlakos, M. J. Black, and K. Daniilidis, “Learning to reconstruct 3d human pose and shape via model-fitting in the loop,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 2252–2261.
- [389] J. Song, X. Chen, and O. Hilliges, “Human body model fitting by learned gradient descent,” in European Conference on Computer Vision. Springer, 2020, pp. 744–760.
- [390] R. A. Guler and I. Kokkinos, “Holopose: Holistic 3d human reconstruction in-the-wild,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 10884– 10894.
- [391] Y. Xu, S.-C. Zhu, and T. Tung, “Denserac: Joint 3d pose and shape estimation by dense render-and-compare,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 7760–7770.
- [392] H. Zhang, J. Cao, G. Lu, W. Ouyang, and Z. Sun, “Danet: Decompose-and-aggregate network for 3d human shape and pose estimation,” in Proceedings of the 27th ACM International Conference on Multimedia, 2019, pp. 935–944.
- [393] W. Zeng, W. Ouyang, P. Luo, W. Liu, and X. Wang, “3d human mesh regression with dense correspondence,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 7054–7063.
- [394] G. Georgakis, R. Li, S. Karanam, T. Chen, J. Koˇseck´a, and Z. Wu, “Hierarchical kinematic human mesh recovery,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16. Springer, 2020, pp. 768– 784.

- [395] M. Kocabas, C.-H. P. Huang, O. Hilliges, and M. J. Black, “Pare: Part attention regressor for 3d human body estimation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 11127–11137.
- [396] G. Moon and K. M. Lee, “I2l-meshnet: Image-to-lixel prediction network for accurate 3d human pose and mesh estimation from a single rgb image,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII

16. Springer, 2020, pp. 752–768.

- [397] H. Zhang, Y. Tian, X. Zhou, W. Ouyang, Y. Liu, L. Wang, and Z. Sun, “Pymaf: 3d human pose and shape regression with pyramidal mesh alignment feedback loop,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 11446–11456.
- [398] H. Zhang, Y. Tian, Y. Zhang, M. Li, L. An, Z. Sun, and Y. Liu, “Pymaf-x: Towards well-aligned full-body model regression from monocular images,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 10, pp. 12287–12303, 2023.
- [399] S. Shimada, V. Golyanik, W. Xu, and C. Theobalt, “Physcap: Physically plausible monocular 3d motion capture in real time,” ACM Transactions on Graphics (ToG), vol. 39, no. 6, pp. 1–16, 2020.
- [400] S. Tripathi, L. Muller,¨ C.-H. P. Huang, O. Taheri, M. J. Black, and D. Tzionas, “3d human pose estimation via intuitive physics,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 4713–4725.
- [401] A. Vaswani, “Attention is all you need,” Advances in Neural Information Processing Systems, 2017.
- [402] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” 2020.
- [403] K. Lin, L. Wang, and Z. Liu, “End-to-end human pose and mesh reconstruction with transformers,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 1954–1963.
- [404] S. Goel, G. Pavlakos, J. Rajasegaran, A. Kanazawa, and J. Malik, “Humans in 4d: Reconstructing and tracking humans with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 14783–14794.
- [405] S. K. Dwivedi, Y. Sun, P. Patel, Y. Feng, and M. J. Black, “Tokenhmr: Advancing human mesh recovery with a tokenized pose representation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1323–1333.
- [406] M. U. Saleem, E. Pinyoanuntapong, P. Wang, H. Xue, S. Das, and C. Chen, “Genhmr: Generative human mesh recovery,” arXiv preprint arXiv:2412.14444, 2024.
- [407] Z. Cai, W. Yin, A. Zeng, C. Wei, Q. Sun, W. Yanjun, H. E. Pang, H. Mei, M. Zhang, L. Zhang et al., “Smpler-x: Scaling up expressive human pose and shape estimation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [408] A. Kanazawa, J. Y. Zhang, P. Felsen, and J. Malik, “Learning 3d human dynamics from video,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 5614–5623.
- [409] M. Kocabas, N. Athanasiou, and M. J. Black, “Vibe: Video inference for human body pose and shape estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5253–5263.
- [410] Z. Wan, Z. Li, M. Tian, J. Liu, S. Yi, and H. Li, “Encoderdecoder with multi-level attention for 3d human shape and pose estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 13033–13042.
- [411] D. Rempe, T. Birdal, A. Hertzmann, J. Yang, S. Sridhar, and L. J. Guibas, “Humor: 3d human motion model for robust pose estimation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 11488–11499.
- [412] C. Doersch and A. Zisserman, “Sim2real transfer learning for 3d human pose estimation: motion to the rescue,” Advances in Neural Information Processing Systems, vol. 32, 2019.
- [413] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in NAACL-HLT, 2019.
- [414] Y. Yuan, U. Iqbal, P. Molchanov, K. Kitani, and J. Kautz, “Glamr: Global occlusion-aware human mesh recovery with dynamic cameras,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

- [415] V. Ye, G. Pavlakos, J. Malik, and A. Kanazawa, “Decoupling human and camera motion from videos in the wild,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 21222–21232.
- [416] Y. Sun, Q. Bao, W. Liu, T. Mei, and M. J. Black, “Trace: 5d temporal regression of avatars with dynamic cameras in 3d environments,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8856–8866.
- [417] S. Shin, J. Kim, E. Halilaj, and M. J. Black, “Wham: Reconstructing world-grounded humans with accurate 3d motion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2070–2080.
- [418] Y. Wang, Z. Wang, L. Liu, and K. Daniilidis, “Tram: Global trajectory and motion of 3d humans from in-the-wild videos,” in European Conference on Computer Vision. Springer, 2024, pp. 467–487.
- [419] J. Engel, K. Somasundaram, M. Goesele, A. Sun, A. Gamino, A. Turner, A. Talattof, A. Yuan, B. Souti, B. Meredith et al., “Project aria: A new tool for egocentric multi-modal ai research,” arXiv preprint arXiv:2308.13561, 2023.
- [420] J. Li, K. Liu, and J. Wu, “Ego-body pose estimation via ego-head pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17142–17151.
- [421] L. Ma, Y. Ye, F. Hong, V. Guzov, Y. Jiang, R. Postyeni, L. Pesqueira, A. Gamino, V. Baiyya, H. J. Kim et al., “Nymeria: A massive collection of multimodal egocentric daily motion in the wild,” in European Conference on Computer Vision. Springer, 2024, pp. 445–465.
- [422] V. Guzov, Y. Jiang, F. Hong, G. Pons-Moll, R. Newcombe, C. K. Liu, Y. Ye, and L. Ma, “Hmdˆ 2: Environment-aware motion generation from single egocentric head-mounted device,” arXiv preprint arXiv:2409.13426, 2024.
- [423] F. Hong, V. Guzov, H. J. Kim, Y. Ye, R. Newcombe, Z. Liu, and L. Ma, “Egolm: Multi-modal language model of egocentric motions,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5344–5354.
- [424] M. Hatano, Z. Zhu, H. Saito, and D. Damen, “The invisible egohand: 3d hand forecasting through egobody pose estimation,” arXiv preprint arXiv:2504.08654, 2025.
- [425] A. Castillo, M. Escobar, G. Jeanneret, A. Pumarola, P. Arbel´aez, A. Thabet, and A. Sanakoyeu, “Bodiffusion: Diffusing sparse observations for full-body human motion synthesis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4221–4231.
- [426] Y. Du, R. Kips, A. Pumarola, S. Starke, A. Thabet, and A. Sanakoyeu, “Avatars grow legs: Generating smooth human motion from sparse tracking inputs with diffusion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 481–490.
- [427] J. Jiang, P. Streli, H. Qiu, A. Fender, L. Laich, P. Snape, and C. Holz, “Avatarposer: Articulated full-body pose tracking from sparse motion sensing,” in European conference on computer vision. Springer, 2022, pp. 443–460.
- [428] J. Wang, Z. Cao, D. Luvizon, L. Liu, K. Sarkar, D. Tang, T. Beeler, and C. Theobalt, “Egocentric whole-body motion capture with fisheyevit and diffusion-based motion refinement,” 2023.
- [429] J. Wang, D. Luvizon, W. Xu, L. Liu, K. Sarkar, and C. Theobalt, “Scene-aware egocentric 3d human pose estimation,” CVPR, 2023.
- [430] R. Wang, Y. Cao, K. Han, and K.-Y. K. Wong, “A survey on 3d human avatar modeling–from reconstruction to generation,” arXiv preprint arXiv:2406.04253, 2024.
- [431] W. Xu, A. Chatterjee, M. Zollh¨ofer, H. Rhodin, D. Mehta, H.P. Seidel, and C. Theobalt, “Monoperfcap: Human performance capture from monocular video,” ACM Transactions on Graphics (ToG), vol. 37, no. 2, pp. 1–15, 2018.
- [432] T. Alldieck, M. Magnor, W. Xu, C. Theobalt, and G. Pons-Moll, “Video based reconstruction of 3d people models,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 8387–8397.
- [433] M. Habermann, W. Xu, M. Zollhoefer, G. Pons-Moll, and C. Theobalt, “Livecap: Real-time human performance capture from monocular video,” ACM Transactions On Graphics (TOG), vol. 38, no. 2, pp. 1–17, 2019.
- [434] C. Guo, X. Chen, J. Song, and O. Hilliges, “Human performance capture from monocular video in the wild,” in 2021 International Conference on 3D Vision (3DV). IEEE, 2021, pp. 889–898.

- [435] Q. Feng, Y. Liu, Y.-K. Lai, J. Yang, and K. Li, “Fof: Learning fourier occupancy field for monocular real-time human reconstruction,” Advances in Neural Information Processing Systems, vol. 35, pp. 7397–7409, 2022.
- [436] S.-Y. Su, F. Yu, M. Zollh¨ofer, and H. Rhodin, “A-nerf: Articulated neural radiance fields for learning human shape, appearance, and pose,” Advances in neural information processing systems, vol. 34, pp. 12278–12291, 2021.
- [437] C. Guo, T. Jiang, X. Chen, J. Song, and O. Hilliges, “Vid2avatar: 3d avatar reconstruction from videos in the wild via self-supervised scene decomposition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12858– 12868.
- [438] S.-Y. Su, T. Bagautdinov, and H. Rhodin, “Danbo: Disentangled articulated neural body representations via graph neural networks,” in European Conference on Computer Vision. Springer, 2022, pp. 107–124.
- [439] B. Jiang, Y. Hong, H. Bao, and J. Zhang, “Selfrecon: Self reconstruction your digital avatar from monocular video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5605–5615.
- [440] L. Hu, H. Zhang, Y. Zhang, B. Zhou, B. Liu, S. Zhang, and L. Nie, “Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 634–644.
- [441] Z. Li, Z. Zheng, L. Wang, and Y. Liu, “Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19711–19722.
- [442] S. Zheng, B. Zhou, R. Shao, B. Liu, S. Zhang, L. Nie, and Y. Liu, “Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19680–19690.
- [443] Z. Qian, S. Wang, M. Mihajlovic, A. Geiger, and S. Tang, “3dgsavatar: Animatable avatars via deformable 3d gaussian splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5020–5030.
- [444] S. Xu, Y.-X. Wang, L. Gui et al., “Interdreamer: Zero-shot text to 3d dynamic human-object interaction,” Advances in Neural Information Processing Systems, vol. 37, pp. 52858–52890, 2024.
- [445] J. P. Araujo,´ J. Li, K. Vetrivel, R. Agarwal, J. Wu, D. Gopinath, A. W. Clegg, and K. Liu, “Circle: Capture in rich contextual environments,” in Proceedings of the IEEE/CVF Conference on Computer

- Vision and Pattern Recognition, 2023, pp. 21211–21221.

[446] L. Muller,¨ V. Ye, G. Pavlakos, M. Black, and A. Kanazawa, “Generative proxemics: A prior for 3d social interaction from images,” in Proceedings of the IEEE/CVF Conference on Computer

- Vision and Pattern Recognition, 2024, pp. 9687–9697.

- [447] M. Savva, A. X. Chang, P. Hanrahan, M. Fisher, and M. Nießner, “Pigraphs: learning interaction snapshots from observations,” ACM Transactions On Graphics (TOG), vol. 35, no. 4, pp. 1–12, 2016.
- [448] J. Y. Zhang, S. Pepose, H. Joo, D. Ramanan, J. Malik, and A. Kanazawa, “Perceiving 3d human-object spatial arrangements from a single image in the wild,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16. Springer, 2020, pp. 34–51.
- [449] S. Goel, G. Pavlakos, J. Rajasegaran, A. Kanazawa*, and J. Malik*, “Humans in 4D: Reconstructing and tracking humans with transformers,” in International Conference on Computer Vision (ICCV), 2023.
- [450] X. Xie, B. L. Bhatnagar, and G. Pons-Moll, “Chore: Contact, human and object reconstruction from a single rgb image,” in European Conference on Computer Vision. Springer, 2022, pp. 125– 145.
- [451] ——, “Visibility aware human-object interaction tracking from single rgb camera,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4757–4768.
- [452] X. Xie, B. L. Bhatnagar, J. E. Lenssen, and G. Pons-Moll, “Template free reconstruction of human-object interaction with procedural interaction generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10003– 10015.
- [453] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic

- models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.
- [454] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.
- [455] M. Hassan, V. Choutas, D. Tzionas, and M. J. Black, “Resolving 3d human pose ambiguities with 3d scene constraints,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 2282–2292.
- [456] C.-H. P. Huang, H. Yi, M. H¨oschle, M. Safroshkin, T. Alexiadis, S. Polikovsky, D. Scharstein, and M. J. Black, “Capturing and inferring dense full-body human-scene contact,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 13274–13285.
- [457] Z. Cao, H. Gao, K. Mangalam, Q.-Z. Cai, M. Vo, and J. Malik, “Long-term human motion prediction with scene context,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16. Springer, 2020, pp. 387–404.
- [458] N. Jiang, Z. Zhang, H. Li, X. Ma, Z. Wang, Y. Chen, T. Liu, Y. Zhu, and S. Huang, “Scaling up dynamic human-scene interaction modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1737–1747.
- [459] Y. Sun, Q. Bao, W. Liu, Y. Fu, M. J. Black, and T. Mei, “Monocular, one-stage, regression of multiple 3d people,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 11179–11188.
- [460] Q. Shuai, Z. Yu, Z. Zhou, L. Fan, H. Yang, C. Yang, and X. Zhou, “Reconstructing close human interactions from multiple views,” ACM Transactions on Graphics (TOG), vol. 42, no. 6, pp. 1–14, 2023.
- [461] Y. Yin, C. Guo, M. Kaufmann, J. J. Zarate, J. Song, and O. Hilliges, “Hi4d: 4d instance segmentation of close human interaction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17016–17027.
- [462] R. Khirodkar, J.-T. Song, J. Cao, Z. Luo, and K. Kitani, “Harmony4d: A video dataset for in-the-wild close human interactions,” Advances in Neural Information Processing Systems, vol. 37, pp. 107270–107285, 2024.
- [463] F. Lu, Z. Dong, J. Song, and O. Hilliges, “Avatarpose: Avatarguided 3d pose estimation of close human interaction from sparse multi-view videos,” in European Conference on Computer Vision. Springer, 2024, pp. 215–233.
- [464] B. Huang, C. Li, C. Xu, L. Pan, Y. Wang, and G. H. Lee, “Closely interactive human reconstruction with proxemics and physicsguided adaption,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1011–1021.
- [465] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” 2017.
- [466] N. Ugrinovic, B. Pan, G. Pavlakos, D. Paschalidou, B. Shen, J. Sanchez-Riera, F. Moreno-Noguer, and L. Guibas, “Multiphys: Multi-person physics-aware 3d motion estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2331–2340.
- [467] G. Yang, C. Wang, N. D. Reddy, and D. Ramanan, “Reconstructing animatable categories from videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 16995–17005.
- [468] D. Damen, “Human-centric object interactions-a fine-grained perspective from egocentric videos,” in Proceedings of the 1st International Workshop on Human-centric Multimedia Analysis, 2020, pp. 1–1.
- [469] T. Kwon, B. Tekin, J. Stuhmer,¨ F. Bogo, and M. Pollefeys, “H2o: Two hands manipulating objects for first person interaction recognition,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10138–10148.
- [470] Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, and L. Yi, “Hoi4d: A 4d egocentric dataset for categorylevel human-object interaction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 21013–21022.
- [471] P. Banerjee, S. Shkodrani, P. Moulon, S. Hampali, F. Zhang, J. Fountain, E. Miller, S. Basol, R. Newcombe, R. Wang et al., “Introducing hot3d: An egocentric dataset for 3d hand and object tracking,” arXiv preprint arXiv:2406.09598, 2024.
- [472] R. Wang, S. Ktistakis, S. Zhang, M. Meboldt, and Q. Lohmeyer, “Pov-surgery: A dataset for egocentric hand and tool pose estimation during surgical activities,” in International Conference

- on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 440–450.
- [473] S. Bansal, M. Wray, and D. Damen, “Hoi-ref: Hand-object interaction referral in egocentric vision,” arXiv preprint arXiv:2404.09933, 2024.
- [474] G. Goletto, T. Nagarajan, G. Averta, and D. Damen, “Amego: Active memory from long egocentric videos,” in European Conference on Computer Vision. Springer, 2024, pp. 92–110.
- [475] K. Grauman, A. Westbury, L. Torresani, K. Kitani, J. Malik, T. Afouras, K. Ashutosh, V. Baiyya, S. Bansal, B. Boote et al., “Ego-exo4d: Understanding skilled human activity from firstand third-person perspectives,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19383–19400.
- [476] T. Perrett, A. Darkhalil, S. Sinha, O. Emara, S. Pollard, K. K. Parida, K. Liu, P. Gatti, S. Bansal, K. Flanagan et al., “Hd-epic: A highly-detailed egocentric video dataset,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 23901–23913.
- [477] A. Darkhalil, D. Shan, B. Zhu, J. Ma, A. Kar, R. Higgins, S. Fidler, D. Fouhey, and D. Damen, “Epic-kitchens visor benchmark: Video segmentations and object relations,” Advances in Neural Information Processing Systems, vol. 35, pp. 13745–13758, 2022.
- [478] E. Kazakos, A. Nagrani, A. Zisserman, and D. Damen, “Epicfusion: Audio-visual temporal binding for egocentric action recognition,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 5492–5501.
- [479] V. Tschernezki, A. Darkhalil, Z. Zhu, D. Fouhey, I. Laina, D. Larlus, D. Damen, and A. Vedaldi, “Epic fields: Marrying 3d geometry and video understanding,” Advances in Neural Information Processing Systems, vol. 36, pp. 26485–26500, 2023.
- [480] R. Li, C. Zheng, C. Rupprecht, and A. Vedaldi, “Dso: Aligning 3d generators with simulation feedback for physical soundness,” arXiv preprint arXiv:2503.22677, 2025.
- [481] Y. Wang, Q. Zhao, R. Yu, A. Zeng, J. Lin, Z. Luo, H. W. Tsui, J. Yu, X. Li, Q. Chen et al., “Skillmimic: Learning reusable basketball skills from demonstrations,” arXiv e-prints, pp. arXiv–2408, 2024.
- [482] J. Ni, Y. Chen, B. Jing, N. Jiang, B. Wang, B. Dai, P. Li, Y. Zhu, S.-C. Zhu, and S. Huang, “Phyrecon: Physically plausible neural scene reconstruction,” Advances in Neural Information Processing Systems, vol. 37, pp. 25747–25780, 2024.
- [483] J. Lee, J. Chai, P. S. Reitsma, J. K. Hodgins, and N. S. Pollard, “Interactive control of avatars animated with human motion data,” in Proceedings of the 29th annual conference on Computer graphics and interactive techniques, 2002, pp. 491–500.
- [484] Y. Lee, K. Wampler, G. Bernstein, J. Popovi´c, and Z. Popovi´c, “Motion fields for interactive character locomotion,” in ACM SIGGRAPH Asia 2010 papers, 2010, pp. 1–8.
- [485] A. Safonova and J. K. Hodgins, “Construction and optimal search of interpolated motion graphs,” in ACM SIGGRAPH 2007 papers, 2007, pp. 106–es.
- [486] A. Treuille, Y. Lee, and Z. Popovi´c, “Near-optimal character animation with continuous control,” in ACM SIGGRAPH 2007 papers, 2007, pp. 7–es.
- [487] S. Levine, J. M. Wang, A. Haraux, Z. Popovi´c, and V. Koltun, “Continuous character control with low-dimensional embeddings,” ACM Transactions on Graphics (TOG), vol. 31, no. 4, pp. 1–10, 2012.
- [488] H. Zhang, S. Starke, T. Komura, and J. Saito, “Mode-adaptive neural networks for quadruped motion control,” ACM Transactions on Graphics (ToG), vol. 37, no. 4, pp. 1–11, 2018.
- [489] D. Holden, T. Komura, and J. Saito, “Phase-functioned neural networks for character control,” ACM Transactions on Graphics (TOG), vol. 36, no. 4, pp. 1–13, 2017.
- [490] H. Y. Ling, F. Zinno, G. Cheng, and M. Van De Panne, “Character controllers using motion vaes,” ACM Transactions on Graphics (TOG), vol. 39, no. 4, pp. 40–1, 2020.
- [491] X. B. Peng, P. Abbeel, S. Levine, and M. van de Panne, “Deepmimic: Example-guided deep reinforcement learning of physics-based character skills,” ACM Trans. Graph., vol. 37, no. 4, pp. 143:1–143:14, Jul. 2018. [Online]. Available: http: //doi.acm.org/10.1145/3197517.3201311
- [492] T. Wang, Y. Guo, M. Shugrina, and S. Fidler, “Unicon: Universal neural controller for physics-based character motion,” arXiv preprint arXiv:2011.15119, 2020.

- [493] J. Won, D. Gopinath, and J. Hodgins, “A scalable approach to control diverse behaviors for physically simulated characters,” ACM Transactions on Graphics (TOG), vol. 39, no. 4, pp. 33–1, 2020.
- [494] ——, “A scalable approach to control diverse behaviors for physically simulated characters,” ACM Trans. Graph., vol. 39, no. 4, 2020. [Online]. Available: https://doi.org/10.1145/ 3386569.3392381
- [495] N. Wagener, A. Kolobov, F. V. Frujeri, R. Loynd, C.-A. Cheng, and M. Hausknecht, “MoCapAct: A multi-task dataset for simulated humanoid control,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 35418–35431.
- [496] X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa, “Amp: Adversarial motion priors for stylized physics-based character control,” ACM Transactions on Graphics (ToG), vol. 40, no. 4, pp. 1–20, 2021.
- [497] J. Ho and S. Ermon, “Generative adversarial imitation learning,” Advances in neural information processing systems, vol. 29, 2016.
- [498] X. B. Peng, Y. Guo, L. Halper, S. Levine, and S. Fidler, “Ase: Largescale reusable adversarial skill embeddings for physically simulated characters,” ACM Transactions On Graphics (TOG), vol. 41, no. 4, pp. 1–17, 2022.
- [499] C. Tessler, Y. Kasten, Y. Guo, S. Mannor, G. Chechik, and X. B. Peng, “Calm: Conditional adversarial latent models for directable virtual characters,” in ACM SIGGRAPH 2023 Conference Proceedings, 2023, pp. 1–9.
- [500] H. Yao, Z. Song, B. Chen, and L. Liu, “Controlvae: Model-based learning of generative controllers for physics-based characters,” ACM Trans. Graph., vol. 41, no. 6, 2022. [Online]. Available: https://doi.org/10.1145/3550454.3555434
- [501] Z. Luo, J. Cao, S. Christen, A. Winkler, K. Kitani, and W. Xu, “Omnigrasp: Grasping diverse objects with simulated humanoids,” Advances in Neural Information Processing Systems, vol. 37, pp. 2161–2184, 2024.
- [502] T. He, W. Xiao, T. Lin, Z. Luo, Z. Xu, Z. Jiang, J. Kautz, C. Liu, G. Shi, X. Wang et al., “Hover: Versatile neural whole-body controller for humanoid robots,” arXiv preprint arXiv:2410.21229, 2024.
- [503] T. He, J. Gao, W. Xiao, Y. Zhang, Z. Wang, J. Wang, Z. Luo, G. He, N. Sobanbab, C. Pan et al., “Asap: Aligning simulation and realworld physics for learning agile humanoid whole-body skills,” arXiv preprint arXiv:2502.01143, 2025.
- [504] Y. Wu, K. Karunratanakul, Z. Luo, and S. Tang, “Uniphys: Unified planner and controller with diffusion for flexible physics-based character control,” arXiv preprint arXiv:2504.12540, 2025.
- [505] C. Tessler, Y. Guo, O. Nabati, G. Chechik, and X. B. Peng, “Maskedmimic: Unified physics-based character control through masked motion,” in ACM Transactions On Graphics (TOG). ACM New York, NY, USA, 2024.
- [506] J. Juravsky, Y. Guo, S. Fidler, and X. B. Peng, “Padl: Languagedirected physics-based character control,” in SIGGRAPH Asia 2022 Conference Papers, 2022, pp. 1–9.
- [507] ——, “Superpadl: Scaling language-directed physics-based control with progressive supervised distillation,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.
- [508] T. E. Truong, M. Piseno, Z. Xie, and K. Liu, “Pdp: Physics-based character animation via diffusion policy,” in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1–10.
- [509] G. Tevet, S. Raab, S. Cohan, D. Reda, Z. Luo, X. B. Peng, A. H. Bermano, and M. van de Panne, “Closd: Closing the loop between simulation and diffusion for multi-task character control,” arXiv preprint arXiv:2410.03441, 2024.
- [510] N. Hansen, J. SV, V. Sobal, Y. LeCun, X. Wang, and H. Su, “Hierarchical world models as visual whole-body humanoid controllers,” 2025.
- [511] D. Rempe, Z. Luo, X. B. Peng, Y. Yuan, K. Kitani, K. Kreis, S. Fidler, and O. Litany, “Trace and pace: Controllable pedestrian animation via guided trajectory diffusion,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- [512] J. Wang, Z. Luo, Y. Yuan, Y. Li, and B. Dai, “Pacer+: On-demand pedestrian animation controller in driving scenarios,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [513] J. K. Hodgins, W. L. Wooten, D. C. Brogan, and J. F. O’Brien, “Animating human athletics,” in Proceedings of the 22nd annual conference on Computer graphics and interactive techniques, 1995, pp. 71–78.

- [514] K. Yin, K. Loken, and M. Van de Panne, “Simbicon: Simple biped locomotion control,” ACM Transactions on Graphics (TOG), vol. 26,

- no. 3, pp. 105–es, 2007.

[515] S. Coros, P. Beaudoin, and M. Van de Panne, “Generalized biped walking control,” ACM Transactions On Graphics (TOG), vol. 29,

- no. 4, pp. 1–9, 2010.

- [516] K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath, “Deep reinforcement learning: A brief survey,” IEEE Signal Processing Magazine, vol. 34, no. 6, pp. 26–38, 2017.
- [517] L. Liu and J. Hodgins, “Learning basketball dribbling skills using trajectory optimization and deep reinforcement learning,” ACM Transactions on Graphics (TOG), vol. 37, no. 4, pp. 1–14, 2018.
- [518] ——, “Learning to schedule control fragments for physics-based characters using deep q-learning,” ACM Transactions on Graphics (TOG), vol. 36, no. 3, pp. 1–14, 2017.
- [519] J. Tan, Y. Gu, C. K. Liu, and G. Turk, “Learning bicycle stunts,” ACM Transactions on Graphics (TOG), vol. 33, no. 4, pp. 1–12, 2014.
- [520] H. Zhang, Y. Yuan, V. Makoviychuk, Y. Guo, S. Fidler, X. B. Peng, and K. Fatahalian, “Learning physically simulated tennis skills from broadcast videos,” ACM Trans. Graph., 2023.
- [521] J. Bae, J. Won, D. Lim, C.-H. Min, and Y. M. Kim, “Pmp: Learning to physically interact with environments using part-wise motion priors,” in ACM SIGGRAPH 2023 Conference Proceedings, 2023, pp. 1–10.
- [522] J. Braun, S. Christen, M. Kocabas, E. Aksan, and O. Hilliges, “Physically plausible full-body hand-object interaction synthesis,” arXiv preprint arXiv:2309.07907, 2023.
- [523] E. S. Ho, T. Komura, and C.-L. Tai, “Spatial relationship preserving character motion adaptation,” in ACM SIGGRAPH 2010 papers, 2010, pp. 1–8.
- [524] Y. Zhang, D. Gopinath, Y. Ye, J. Hodgins, G. Turk, and J. Won, “Simulation and retargeting of complex multi-character interactions,” in ACM SIGGRAPH 2023 Conference Proceedings, 2023, pp. 1–11.
- [525] T. Zhang, H.-X. Yu, R. Wu, B. Y. Feng, C. Zheng, N. Snavely, J. Wu, and W. T. Freeman, “Physdreamer: Physics-based interaction with 3d objects via video generation,” 2024.
- [526] Y. Hu, L. Anderson, T.-M. Li, Q. Sun, N. Carr, J. Ragan-Kelley, and F. Durand, “Difftaichi: Differentiable programming for physical simulation,” ICLR, 2020.
- [527] T. Chen, P. Wang, Z. Fan, and Z. Wang, “Aug-nerf: Training stronger neural radiance fields with triple-level physicallygrounded augmentations,” 2022.
- [528] C. Gao, Y. Wang, C. Kim, J.-B. Huang, and J. Kopf, “Planar reflection-aware neural radiance fields,” 2024.

[Figure 9]

Yukang Cao is currently a Research Fellow at MMLab@NTU, Nanyang Technological University, supervised by Prof. Ziwei Liu. He received Ph.D degree from the Department of Computer Science, The University of Hong Kong (HKU) advised by Prof. Kwan-Yee K. Wong in 2024. He was the recipient of HKU-PS scholarship during Ph.D. He received my B.Eng from Zhejiang University in 2020. His research interests include computer vision and deep learning. Particularly, he is interested in 3D representation learning.

[Figure 10]

Jiahao Lu is currently pursuing a Ph.D. degree at The Hong Kong University of Science and Technology. He received his bachelor’s degree in Artificial Intelligence and Automation from Huazhong University of Science and Technology, Wuhan, Hubei, P. R. China, in 2022. His research interests include computer vision and machine learning, with a focus on 3D reconstruction, 3D perception, and 3D generation.

[Figure 11]

Zhisheng Huang is currently a PhD student under the co-supervision of Professor Wenping Wang and Professor Xin Li in the CSE Department at Texas A&M University (TAMU). He completed both his Bachelor’s and Master’s degrees at Wuhan University. His research interests lie in 3D computer vision and graphics.

[Figure 12]

Zhuowen Shen is currently pursuing a Ph.D. degree in Computer Science and Engineering at Texas A&M University, coadvised by Prof. Wenping Wang and Prof. Xin Li. He received his Master’s degree in Computer Science and Engineering from the University of Michigan, Ann Arbor, in 2023. His research interests lie in computer vision and machine learning, with a focus on 3D reconstruction and 3D representation learning.

[Figure 13]

Chengfeng Zhao is currently a first-year Ph.D student at Intelligent Graphics Lab in HKUST, supervised by Prof. Yuan Liu. Prior to this, He obtained his master and bachelor’s degree from ShanghaiTech University, advised by Prof. Lan Xu. He was also fortunate to work closely with Prof. Jingyi Yu and Prof. Yuexin Ma. His research interests are in Computer Graphics and 3D Computer Vision, specifically video generation, human motion synthesis, learning-based garment simulation, large models, etc.

[Figure 14]

Fangzhou Hong is currently a research fellow at MMLab@NTU, Nanyang Technological University, supervised by Prof. Ziwei Liu. He received Ph.D. degree from MMLab at Nanyang Technological University, supervised by Prof. Ziwei Liu in 2025. He received a B.Eng. degree in software engineering from Tsinghua University, China, in 2020. His research interests include computer vision and deep learning. Particularly, he is interested in 3D representation learning.

[Figure 15]

Zhaoxi Chen is currently a Ph.D. student at MMLab@NTU, Nanyang Technological University, supervised by Prof. Ziwei Liu. He received the bachelor’s degree from Tsinghua University, in 2021. He received the AISG PhD Fellowship in 2021. His research interests include inverse rendering and 3D generative models. He has published several papers in CVPR, ICCV, ECCV, ICLR, NeurIPS, TOG, and TPAMI. He also served as a reviewer for CVPR, ICCV, NeurIPS, TOG, and IJCV.

Xin Li is currently a Professor and Chair of the Section of Visual Computing and Computational Media, within the College of Performance, Visualization, and Fine Arts. He is an affiliated faculty member (courtesy appointment) of the Department of Computer Science and Engineering, College of Engineering. He is also affiliated with Aggie Computer Graphics Group . He got my B.S. degree in Computer Science in 2003 at University of Science and Technology of China (USTC) with a major in Computer Science, and

[Figure 16]

obtained his M.S. and Ph.D. degrees in Computer Science from State University of New York at Stony Brook in 2005 and 2008. Before He joined Texas A&M University, he was a faculty member at School of Electrical Engineering and Computer Science, Louisiana State University (from 2008 to 2022).

Wenping Wang is currently a Professor of Computer Science & Engineering at Texas A&M University. His research interests include computer graphics, computer visualization, computer vision, robotics, medical image processing, and geometric computing. He has published over 300 technical papers in these fields. He is journal associate editor of Computer Aided Geometric Design (CAGD) and IEEE Transactions on Visualization and Computer Graphics, and has chaired a number of international conferences,

[Figure 17]

including Pacific Graphics 2012, ACM Symposium on Physical and Solid Modeling (SPM) 2013, SIGGRAPH Asia 2013, and Geometry Summit 2019. He received the John Gregory Memorial Award for his contributions in geometric modeling. He is an ACM Fellow and IEEE Fellow.

[Figure 18]

Yuan Liu is an assistant professor at the Hong Kong University of Science and Technology (HKUST). Prior to that, Yuan worked in Nanyang Technological University (NTU) as a PostDoc researcher and obtained his PhD degree at the University of Hong Kong (HKU). His research mainly concentrates on 3D vision and graphics. He currently works on topics about 3D AIGC, including 3D neural representations, 3D generative models, and 3D-aware video generation.

Ziwei Liu is currently an associate professor at Nanyang Technological University, Singapore. His research revolves around computer vision, machine learning, and computer graphics. He has published extensively on top-tier conferences and journals in relevant fields, including CVPR, ICCV, ECCV, NeurlPS, ICLR, ICML, TPAMI, TOG, and Nature Machine Intelligence. He is the recipient of the Microsoft Young Fellowship, Hong Kong PhD Fellowship, ICCV Young Researcher Award, HKSTP Best Paper Award

[Figure 19]

and WAIC Yunfan Award. He serves as an Area Chair of CVPR, ICCV, NeurlPS, and ICLR, as well as an Associate Editor of IJCV.

