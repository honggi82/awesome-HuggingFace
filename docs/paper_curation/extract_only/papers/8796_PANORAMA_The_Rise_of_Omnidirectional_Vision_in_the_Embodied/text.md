# arXiv:2509.12989v1[cs.CV]16Sep2025

## PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era

### Xu Zheng1,2,†, Chenfei Liao1,3,*, Ziqiao Weng1,*, Kaiyu Lei1,*, Zihao Dongfang1, Haocong He3, Yuanhuiyi Lyu1, Lutao Jiang1, Lu Qi6,7, Li Chen1, Danda Pani Paudel2, Kailun Yang5, Linfeng Zhang3, Luc Van Gool2, Xuming Hu1,4,‡

1HKUST(GZ), 2INSAIT, Sofia University “St. Kliment Ohridski”, 3Shanghai Jiao Tong University, 4CSE, HKUST, 5Hunan University, 6Wuhan University, 7Insta360 †Presenter, *Core Contributor, ‡Corresponding Author {zhengxu128, lcfgreat624}@gmail.com Abstract

Problem Space Techniques

|Generation<br><br>Perception<br><br>Understanding<br><br>|
|---|

[Figure 1]

Data Bottlenecks

Omnidirectional vision—using 360◦ vision to understand the environment—has become increasingly critical across domains like robotics, industrial inspection, and environmental monitoring. Compared to traditional pinhole vision, omnidirectional vision provides holistic environmental awareness, significantly enhancing the completeness of scene perception and the reliability of decision-making. However, foundational research in this area has historically lagged behind traditional pinhole vision. This talk presents an emerging trend in the embodied AI era: the rapid development of omnidirectional vision, driven by growing industrial demand and academic interest. We highlight recent breakthroughs in omnidirectional generation, omnidirectional perception, omnidirectional understanding, and related datasets. Drawing on insights from both academia and industry, we propose an ideal panoramic system architecture in the embodied AI era: PANORAMA, which consists of four key subsystems. Moreover, we offer in-depth opinions related to emerging trends and crosscommunity impacts at the intersection of panoramic vision and embodied AI, along with the future roadmap and open challenges. This talk synthesizes state-of-the-art advancements and outlines challenges and opportunities for future research in building robust, general-purpose omnidirectional AI systems in the embodied AI era.

[Figure 2]

Model Capabilities

[Figure 3]

Application Blanks

Figure 1: Challenges & techniques of 360 vision.

ings (Xu et al. 2024, 2025). Therefore, omnidirectional vision has gradually become a more competitive solution than pinhole-based vision in such an embodied AI era (Ai, Cao, and Wang 2025).

The integration of panoramic visual technology with embodied intelligence faces fundamental gaps that define the core problem space. From a systematic perspective, these gaps can be categorized into three types as shown in Figure 1: data bottlenecks, model capabilities, and application blanks:

### Motivation & Problem Space

Data Bottlenecks: Panoramic images, which are often distorted due to equirectangular projection (ERP) and typically captured at high resolutions, are more costly to annotate manually compared to other image types, such as pinhole images. The geometric distortions inherent in these projections make conventional automated annotation tools, typically designed for pinhole images, ineffective. These challenges significantly hinder the development of large-scale, high-quality datasets (Li et al. 2024b; Zhang, Ye, and Zheng 2025; Jiang et al. 2025; Huang et al. 2024; Park et al. 2024; Liu et al. 2025), creating a data bottleneck that impedes the advancement of omnidirectional vision in the field of embodied AI.

The pursuit of artificial visual perception that rivals human capabilities has long been a cornerstone of computer vision (Zhao et al. 2024; Zheng et al. 2023a, 2025). Decades of research focused on pinhole-based visual perception, which offers a narrow, frustum-limited view of the world (Zheng et al. 2024; Lyu et al. 2024). In the past eras, pinholebased visual perception has driven significant advancements in fields like image classification (Chen et al. 2021), object detection (Kaur and Singh 2023), and semantic segmentation (Mo et al. 2022). While in the current era of embodied AI, more complex tasks, such as indoor/outdoor navigation (Wang et al. 2025), are emerging. Such tasks rely on the comprehensive and thorough perception of environments, requiring a holistic, 360◦ understanding of their surround-

Model Capabilities: From the modeling perspective, most existing pre-trained models encode inductive biases (e.g., translation invariance) through operations like convolution and pooling, which are applicable for pinhole images. However, these models are incapable of understanding the distor-

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

tion characteristics of panoramic images, leading to a significant decline in performance when the models are directly transferred to deal with panoramic images. Thus, how to efficiently bridge the domain gap of data at the model level has become a key issue in the research on panoramic vision (Coors, Condurache, and Geiger 2018; Shen et al. 2022; Su and Grauman 2019; Yun et al. 2023), especially in the era of embodied AI.

Application Blanks: When new sensors (360 cameras) meet a new era (embodied AI), many original scenarios, such as industrial scenarios, are likely to experience new technological iterations. However, different application scenarios exhibit distinct characteristics and priorities, which have spawned numerous scenario-specific research subfields. This has given rise to numerous sub-fields of research that are scenario-specific. Due to the lack of interdisciplinary talents and the insufficiency of existing panoramic data and models, these sub-fields, such as panoramic production safety inspection, panoramic forest fire detection, and so on, currently lack sufficient exploration (Yan et al. 2024; Jung et al. 2025; Fu, Lou, and Yu 2025).

In summary, only by systematically tackling the threefold challenges of data, models, and applications can we truly unlock the enormous potential of omnidirectional vision in the era of embodied AI, thereby laying a solid foundation for achieving general and robust embodied intelligence.

### Recent Technical Advances

Recent advances in omnidirectional vision can be categorized into three closely related areas: generation, perception, and understanding. In the area of generation, existing research focuses on creating structurally consistent panoramic images and addressing domain-specific challenges, such as those posed by spherical projections. In perception, the focus is on adapting models designed for pinhole image tasks to handle omnidirectional perception tasks. In the area of understanding, the emphasis is on enabling models, particularly multimodal large language models, to interpret panoramic images, with a particular focus on extracting and understanding the spatial information they contain.

#### Omnidirectional Generation

Researchers focus on the generative adversarial networkbased methods for omnidirectional generation at the early stage (Ai et al. 2024; Chen, Wang, and Liu 2022; Cheng et al. 2022; Wang et al. 2022; Oh et al. 2022). Typically, through a two-stage strategy, including both codebookbased panorama outpainting and frequency-aware refinement, Dream360 (Ai et al. 2024) successfully generates high-quality and high-resolution panorama images based on selected viewports. As diffusion models have become the mainstream method in the field of generation recently, related research has gradually emerged (Li and Bansal 2023; Li et al. 2024a; Wang et al. 2024b, 2023, 2024a; Ye et al. 2024; Li et al. 2024c; Wu, Zheng, and Cham 2023). Among these works, PanoDiffusion (Wu, Zheng, and Cham 2023) generates panorama images through a two-branch diffusion

structure, which allows RGB-D data as inputs during training. Thus, more spatial information can be injected into the generation model, improving the quality of the generated panoramic images. Meanwhile, OmniDrag (Li et al. 2024c) controls the generation of panoramic images based on trajectories, further enhancing the user-friendliness of panoramic image generation.

#### Omnidirectional Perception

Considering the data bottleneck problems of omnidirectional vision, the domain adaptation technique has become a popular solution that enables models to deal with panoramic images with unlabeled data (Zheng et al. 2024). The existing strategies can be primarily classified into three types: adversarial learning-based strategy, pseudo-label-based strategy, and prototype-based strategy (Zhong et al. 2025). Adversarial learning-based strategies introduce a discriminator to force the model to generate features that are difficult to distinguish between domains, thereby capturing domaininvariant representations. Pseudo-label-based strategies train models by generating self-supervised labels for the target domain data, where GoodSAM (Zhang et al. 2024b) and GoodSAM++ (Zhang et al. 2024b) refine pseudo-labels using the Segment Anything Model (SAM) (Kirillov et al.

- 2023) to provide more reliable pseudo-labels for the target model. Meanwhile, OmniSAM (Zhong et al. 2025) proposes a dynamic pseudo-label updating mechanism to improve the credibility of the pseudo-label. Prototype alignment strategies aim to align the centers of high-level features between the source and target domains to reduce domain discrepancies. Previous works, such as 360SFUDA++ (Zheng et al.
- 2024) and OmniSAM (Zhong et al. 2025), center on matching the distortion and abstracting the semantics with prototypes, yielding significant improvements.

#### Omnidirectional Understanding

Current multi-modal large language models tend to be trained through normal images, especially the pinhole images. Consequently, these models struggle to understand panoramic images, having never encountered them before. From the perspective of data, recent works focus on building omnidirectional understanding datasets and benchmarks (Dongfang et al. 2025; Zhang, Ye, and Zheng 2025; Song et al. 2024; Zhou et al. 2025; Chou et al. 2020). Especially, OSR-bench (Dongfang et al. 2025) creates a concept of cognitive maps, dividing the whole panoramic image into patches and labeling patches by the objects within them. Through this hierarchical approach, OSR-Bench achieves fast and effective data labeling and benchmarking. Meanwhile, OmniVQA (Zhang, Ye, and Zheng 2025) achieves efficient data labeling through agent collaboration. While from the perspective of models, current methods tend to apply the GRPO techniques (Zhang, Ye, and Zheng 2025; Zhou et al. 2025). However, the existing works prefer to directly fine-tune the multi-modal large language models based on the existing VQA datasets. Thus, works like ERPRoPE (Zhou et al. 2025) attempt to explore the internal features of panoramic images, which further enhance the model’s understanding of panoramic images.

###### Scene Source Dataset Year Annotations Scale

PanoContext (Dong et al. 2024) 2014 Layouts, object 3D bounding boxes 700 full-view panoramas Stanford 2D-3D-S (Armeni et al. 2017) 2017 RGB, depth, semantics 70,000 images

Real

Matterport3D (Chang et al. 2017) 2017 RGB-D panoramas, 2D/3D semantics & poses 10.8 K images, 90 buildings

ZInD (Cruz et al. 2021) 2021 360◦ panoramas, 3D layouts, floorplans 71 K panoramas, 1.5 K homes HM3D (Ramakrishnan et al. 2021) 2021 Textured 3D mesh reconstructions of interiors 1000 building-scale scenes

Indoor

ToF-360 (Kanayama et al. 2025) 2025 2D/3D Segmentation, Layout Estimation 207 total panoramas

3D60 (Zioulis et al. 2018) 2018 Rendered panoramas + depth 60 real-world scenes Structured3D (Zheng et al. 2020) 2020 CAD-rendered panoramas + geometry/semantics 3.5 K layouts DeepPanoContext (Zhang et al. 2021) 2021 Synthetic panoramas with layout, object shape, pose Rich 3D annotations SynPASS (Zhang et al. 2024a) 2024 22-class semantic synthetic panoramas 9 K images HM3DSem (Yadav et al. 2023) 2022 Dense object instance semantic annotations 142,646 object instances

Synthetic

VLN-RAM (Wei et al. 2025) 2024 Human-aware visual language navigation 90 scenes, 145 human activity descriptions OSR-Bench (Dongfang et al. 2025) 2025 QA of omnidirectional spatial reasoning 153,000+ diverse question-answer pairs

StreetLearn (Mirowski et al. 2019) 2019 StreetView panoramas + connectivity graph 143 K images CVIP360 (Mazzola et al. 2021) 2021 360◦ video + pedestrian bounding boxes 16 videos

360VOT (Huang et al. 2023) 2023 360◦ video, tracking + segmentation labels 120 sequences, 113 K frames

Real

360 in the Wild (Park et al. 2024) 2024 360◦ images + poses + depth 25 K real scenes 360Loc (Huang et al. 2024) 2024 360◦ images + 6DoF poses, cross-device Multi-device localization testbed Dense360 (Zhou et al. 2025) 2025 Entity-grounded panoramic scene descriptions 160K panoramas + 5M entity-level captions

Outdoor

OmniScape (Sekkat et al. 2020) 2020 Multiple projection panoramas + semantics 10,000 captures OmniHorizon (Bhanushali, Muniyandi, and Chakravarthula 2024) 2023 Synthetic outdoor panoramas + depth/normal 24,335 images

Synthetic

Real UAV-ERP (Zhang et al. 2022b) 2022 ERP panoramas captured 2.3 K images annotated Synthetic AirSim-360 (Shah et al. 2017) 2018 Unreal-simulated panoramas + depth Configurable generation via API

UAV (Flight)

Table 1: Overview of representative omnidirectional datasets.

#### Dataset

Moreover, datasets also serve as the solid foundation of the three above tasks. We categorize omnidirectional datasets into three domains: Indoor, Outdoor, and UAV/Flight, each further divided into real or synthetic sources. From this taxonomy, which is shown in Table 1, we identify 23 representative datasets, providing a structured basis for comparing scale, modalities, and annotations.

Across these directions, common modalities include RGB panoramas, depth, camera poses, and semantic labels. Datasets differ in granularity: indoor resources often offer dense semantics, scan-based or synthetic sets yield geometry and object metadata (Chang et al. 2017), while outdoor or aerial captures prioritize spatial coverage over instance detail (Huang et al. 2023). Temporal scope, viewpoint density, and annotation type further diversify the landscape.

Recent datasets increasingly combine geometric ground truth with downstream tasks. For example, Matterport3D (Chang et al. 2017), ReplicaPano (Dong et al. 2024), and HM3D (Ramakrishnan et al. 2021) illustrate datasets that pair panoramas/depth/layouts with instance or task-level labels. Temporal and embodied captures are becoming more common (Xia et al. 2018). However, fully synchronized multi-sensor panoptic collections that com-

bine panoramic RGB, LiDAR, spatial audio, and behavioral traces remain rare (Kim et al. 2024). Researchers are paying more attention to projection-induced distortions and adopting distortion-aware methods or losses to mitigate latitude or pole artifacts (Zhang et al. 2022a). Finally, QAstyle panoramic reasoning benchmarks have begun to appear (e.g., OSR-Bench (Dongfang et al. 2025) and OmniVQA (Zhang, Ye, and Zheng 2025)), indicating early interest in reasoning-oriented panoramic evaluation even though these resources are still nascent.

### PANORAMA System Architecture

The advent of embodied intelligence presents fundamental challenges and opportunities to omnidirectional vision. Therefore, a dedicated panoramic system is not merely beneficial but necessary to unlock the potential of omnidirectional vision for embodied AI. Thus, as shown in Figure 2, we propose an ideal panoramic system, PANORAMA, comprised of four integrated subsystems, which has the potential to be a key solution for the integration of panoramic vision and embodied AI.

#### Key Subsystems

Subsystem 1: Data Acquisition & Pre-processing This subsystem is responsible for capturing raw omnidirectional

[Figure 4]

PANORAMA System Architecture

scene awareness for more natural and context-aware interactions.

|Subsystem 1 Subsystem 2 Subsystem 3<br><br>Subsystem 4<br><br>Data Acquisition & Pre-processing<br><br>Acceleration & Employment<br><br>Perception Application<br><br>Synchronization & Calibration<br><br>Format Conversion<br><br>Data Capture<br><br>Feature Extraction<br><br>Environmental Perception<br><br>Navigation & SLAM<br><br>Human-Robot Interaction<br><br>Digital Twin & 3D Reconstruction<br><br>……<br><br>Hardware<br><br>Software Employment: How to really use the model? Acceleration: How to better use the model?<br><br>|
|---|

• Digital Twin & 3D Reconstruction: Creating immersive and accurate virtual models of real-world spaces for simulation, planning, and monitoring.

Subsystem 4: Acceleration & Employment This subsystem addresses the computational challenges of processing high-resolution panoramic data in real-world, often resource-constrained settings. It focuses on the practical implementation of the entire pipeline.

- • Software Acceleration: Optimizing the entire stack through techniques like model quantization and pruning to balance accuracy, latency, and power consumption for deployment on edge devices.
- • Hardware Employment: Employing edge computing platforms (e.g., NVIDIA Jetson, SOPHGO SE9) to achieve real-world applications.

Figure 2: The overview of the system architecture.

#### Workflow

data and converting it into a format suitable for computational processing. It primarily consists of hardware like cameras (e.g., those utilizing equirectangular projection or multi-fisheye lens rigs) and complementary sensors (e.g., IMUs, depth sensors). Its core functions include:

The whole system operates as a cohesive and integrated pipeline. The workflow begins with the Data Acquisition & Pre-processing Subsystem, where raw data from panoramic cameras and other sensors is captured, corrected, and synchronized. This clean, formatted data is then passed to the Perception Subsystem, where deep learning models perform feature extraction and environmental perception to generate a comprehensive understanding of the scene. These perceptual outputs are subsequently utilized by the Application Subsystem to execute specific embodied AI tasks, such as navigation or interaction. Throughout this process, the Acceleration & Deployment Subsystem ensures the computational feasibility of the pipeline, enabling low-latency, efficient operation on edge devices from raw sensor inputs to final embodied applications.

- • Data Capture: Acquiring high-resolution omnidirectional images and videos.
- • Format Conversion: Dynamically transforming data between different representations (e.g., ERP, Cubemap) to suit the needs of downstream processing tasks.
- • Synchronization & Calibration: Ensuring temporal alignment and spatial coordination between multiple/multi-modal sensors for accurate data fusion.

- Subsystem 2: Perception This subsystem performs fundamental scene perception on the preprocessed panoramic data. It employs deep learning models adapted for spherical geometry to extract rich, structured information from the omnidirectional input. Its key capabilities include:

- • Feature Extraction: Utilizing specialized architectures (e.g., Spherical CNNs, Transformers) to understand the omnidirectional context.
- • Environmental Perception: Simultaneously performing core perception tasks such as semantic segmentation, object detection, and depth estimation from a shared feature backbone for efficiency.

- Subsystem 3: Application This subsystem translates the perceptual insights into actions for embodied AI agents. It consumes the structured data (e.g., semantic maps, object lists, depth information) to serve specific downstream tasks. Example applications include:

### Emerging Trends & Future Roadmap

Recent advancements in omnidirectional vision have shown promising progress across various downstream applications, including visual perception (Zheng et al. 2023b), reasoning (Dongfang et al. 2025), and embodied tasks such as navigation (Wang et al. 2025) and grasping (Kerr et al. 2025). Representative model architectures often involve panoramic extensions of 2D CNNs or Transformer backbones, as well as unified multi-task encoders trained on 360-degree geometry and semantics (Sun, Sun, and Chen 2021; Jiang et al. 2021). Additionally, early multi-modal models have emerged that integrate vision and language for more comprehensive task understanding. Despite these advancements, many approaches remain task-specific, struggle with projection ambiguities, and lack large-scale multi-modal pretraining resources. These limitations hinder model generalization and pose significant challenges to the broader development of embodied AI (He et al. 2022; Zhang, Ye, and Zheng 2025). To overcome this gap and realize the PANORAMA system, we further propose a staged roadmap as shown in Figure 3 to build an ideal unified model for omnidirectional tasks. The timeline spans six stages:

- • Navigation & SLAM: Enabling autonomous movement and real-time spatial mapping in indoor and outdoor environments.
- • Human-Robot Interaction: Providing agents with full-

###### Dataset Integration

[Figure 6]

Bring together existing datasets into a single, consistent framework.

###### Multi-Modal Expansion

[Figure 7]

Synchronize signals from RGB, depth, LiDAR,

audio, and IMU sensors.

###### Unified Model Pretraining

[Figure 8]

###### Reasoning and Embodied Data

[Figure 9]

Pretrain unified, multi-task encoders for panoramic models.

Release reasoning-augmented datasets and benchmarks designed specifically for real-world

interactive environments.

###### Deployment and Robustness

Prepare models for real-world deployment through cross-domain transfer, continual learning, and test their robustness.

###### Evaluation and Benchmarking

Establish standardized splits, projection-aware metrics, and out-of-distribution evaluation protocols.

Figure 3: Roadmap for implementing an omnidirectional model in the Embodied AI area.

- Stage 1: Dataset Integration In the first stage, the focus is on bringing together existing datasets into a single, consistent framework for projections, along with standardized test splits. The data will be re-annotated with consistent labels, and flexible re-projection tools will help ensure fair comparisons of model performance across different formats, like ERP and cubemap. This stage will result in a well-organized benchmark suite, with careful human checks to reduce errors in the annotations and improve label accuracy.
- Stage 2: Multi-Modal Expansion Next, the focus shifts to synchronizing signals from RGB, depth, LiDAR, audio, and IMU sensors, enabling multi-modal and multi-task pretraining tailored for panoramic vision. Standardized rigs and calibration protocols will facilitate richer sensor fusion, enhancing the modeling of environments captured by panoramic cameras. A key milestone in this phase would be the creation of a public multi-sensor corpus with harmonized splits, enabling more effective benchmarking. To offset the cost of large-scale data collection, hybrid real–synthetic pipelines will be utilized, combining both real-world and simulated data for more robust sensor training.
- Stage 3: Reasoning and Embodied Data The third stage focuses on advancing reasoning capabilities in interactive, embodied tasks such as grounded visual question answering (VQA), instruction-following, navigation, and grasping. These tasks require robust spatial reasoning to understand and interact with the environment. To facilitate this, hybrid question generation methods—combining templates, large language models (LLMs), and human validation—will be employed to ensure both scale and diversity in the training data. Simulation environments will play a crucial role in providing varied and dynamic scenarios for tasks like navigation and grasping, where precise spatial awareness and decision-making are essential. The release of a reasoningaugmented dataset and a benchmark designed specifically for evaluating spatial reasoning, navigation, and grasping

- performance will establish standardized protocols for measuring the success rates of models in real-world interactive environments.
- Stage 4: Unified Model Pretraining Building on the integrated multi-modal corpora from earlier stages, this phase focuses on pretraining unified, multi-task encoders for panoramic models. These models jointly process 360◦ geometry, semantic labels, and synchronized sensor streams (RGB, depth, LiDAR, audio, and IMU). Based on weights from established 2D or 3D architectures, the key innovation is fine-tuning and post-training using panoramicspecific datasets and task objectives. Training incorporates cross-projection representations, multi-objective losses, and domain-mixing curricula (real–synthetic) to ensure transferability. This stage adapts traditional models to the challenges of panoramic vision and enhances generalization in real-world scenarios like navigation, grasping, and embodied tasks. Fine-tuning on tailored datasets ensures efficient learning for both structured and unstructured data in dynamic environments.
- Stage 5: Evaluation and Benchmarking This stage establishes a rigorous evaluation infrastructure consisting of standardized dataset splits, projection-consistent reprojection tools, and a harmonized metric suite covering per-task accuracy, cross-projection consistency, and success rates of reasoning and embodiment tasks. Protocols include explicit out-of-distribution splits, calibration and uncertainty measures, efficiency targets, and human-verified evaluation for critical tasks; together, these components enable reproducible comparison, ablation studies, and operational readiness assessments.
- Stage 6: Deployment and Generalization In the final stage, the focus is on preparing models for real-world deployment through cross-domain transfer, continual learning, and testing their robustness. Models are tested under real-world conditions, including changes in data distri-

butions, using out-of-distribution (OOD) splits. Evaluation will include measures like calibration, latency, and uncertainty. This stage also includes delivering a deployment kit with stress-test datasets, evaluation benchmarks for ongoing adaptation, and workflows to validate models’ uncertainty.

### Cross-Community Impacts & Open Challenges

In today’s era of embodied AI, the maturity of omnidirectional vision no longer merely refers to the gradual update of technology but rather its emergence as a fundamental enabling technology, promoting cross-community breakthroughs in practical application domains. The impact of omnidirectional vision, especially the PANORAMA system, in the current era extends far beyond a single community, providing cross-community impacts for researchers from diverse fields:

- • Robotics & Autonomous Navigation: For mobile robots and autonomous vehicles, omnidirectional perception is the cornerstone of complete situational awareness. It eliminates blind spots, making navigation in dense and dynamic environments—such as crowded public places—more accurate and safer (Ran et al. 2017; Huang et al. 2024), enhancing the robot’s perception ability by providing contextual information from different angles (Park et al. 2024).
- • Human-Robot Interaction: Omnidirectional vision enables robots to understand social and spatial information like humans. A robot equipped with an omnidirectional camera can simultaneously track multiple individuals (Huang et al. 2024; Li et al. 2024b), interpret group conversations, and comprehend social cues from any direction (Rai, Guti´errez, and Le Callet 2017; Martin et al. 2022), thereby facilitating more natural, seamless, and trustworthy human-robot interaction.
- • Cognitive AI & Virtual Agents: Omnidirectional vision provides a dense, information-rich perceptual stream that is fundamentally closer to a human’s egocentric vision of the world (Xia et al. 2018; Zhou et al. 2025). This highfidelity input is crucial for developing the foundation of high-level, human-like cognitive abilities, including spatial reasoning, long-horizon task planning, and commonsense understanding of environmental physics.

Despite the positive cross-community impacts of omnidirectional vision in the embodied AI era, several open challenges remain, providing new directions for future research.

• Generalization and Robustness: Most current models still focus on specific scenarios or projection methods (Ai et al. 2022). Developing models that generalize across diverse panoramic sensor specifications, application scenarios, and projection methods remains non-trivial (Ai, Cao, and Wang 2025). It is necessary for future works to focus on projection-agnostic representations and selfsupervised learning techniques that can learn invariant features from unlabeled omnidirectional information, including both images and video streams.

- • Dynamic Distortion Handling: While current approaches have made significant progress in handling the static distortion of panoramic images (Zhang et al. 2022a), they treat it as a frame-independent geometric problem. This represents a critical limitation, as the distortion is fundamentally dynamic in real-world scenarios. Future research should further explicitly consider the temporal consistency and evolution of distortions across omnidirectional video sequences.
- • Action-Aware Representation Learning: The ultimate goal of omnidirectional vision in the era of embodied AI is not merely to enable robots to observe better, but to empower them to take action more effectively (Xu et al. 2024, 2025). A key direction is to allow models to learn action-oriented representations within panoramic images (Mu et al. 2023; Fu, Lou, and Yu 2025). By integrating the unique advantages of omnidirectional visual features into downstream control strategies, we can achieve more effective and efficient decision-making in robotic behavior.
- • Scalable and Unified Architectures: An important challenge is the creation of unified, multi-task foundation models specifically designed for omnidirectional vision (Dong et al. 2024; Shen et al. 2022). Moving beyond the inefficiency of task-specific models, these models would be pre-trained on extensive panoramic data to capture a fundamental understanding of omnidirectional geometry and semantics (Sun, Sun, and Chen 2021; Jiang et al. 2021). This would yield a powerful visual backbone that could be rapidly specialized for numerous applications, improving performance and generalization while reducing the need for massive task-specific datasets.

In summary, the process towards achieving truly embodied intelligence is a collective effort. In the field of omnidirectional vision, we call upon researchers to:

- • For Dataset Creators: Plan and publish large-scale multi-task omnidirectional datasets that encompass the complexity of real-world scenes, including both indoor and outdoor scenarios, general scenarios, and embodied intelligent scenarios.
- • For Algorithm Researchers: Go beyond simple adaptations based on pinhole models and create novel architectures and dynamic learning paradigms that possess omnidirectional information, which is the key to embracing the unique challenges of omnidirectional vision.
- • For Application Engineers: Explore and demonstrate the benefits of omnidirectional perception in real-world robotics and interactive systems, as it bridges the gap between laboratory research and practical applications.

The era of embodied intelligence demands a paradigm shift from a narrow, forward-looking view to a comprehensive, spherical understanding of our world. By embracing omnidirectional vision, we can collectively unlock the next frontier of embodied AI, creating agents that not only see but also understand and interact with the entirety of their environment. The embodied AI’s future is omnidirectional.

### References

Ai, H.; Cao, Z.; Lu, H.; Chen, C.; Ma, J.; Zhou, P.; Kim, T.K.; Hui, P.; and Wang, L. 2024. Dream360: Diverse and immersive outdoor virtual scene creation via transformer-based 360 image outpainting. IEEE transactions on visualization and computer graphics, 30(5): 2734–2744.

Ai, H.; Cao, Z.; and Wang, L. 2025. A survey of representation learning, optimization strategies, and applications for omnidirectional vision. International Journal of Computer Vision, 1–40.

Ai, H.; Cao, Z.; Zhu, J.; Bai, H.; Chen, Y.; and Wang, L. 2022. Deep Learning for Omnidirectional Vision: A Survey and New Perspectives. arXiv:2205.10468.

Armeni, I.; Sax, S.; Zamir, A. R.; and Savarese, S. 2017. Joint 2d-3d-semantic data for indoor scene understanding. arXiv preprint arXiv:1702.01105.

Bhanushali, J.; Muniyandi, M.; and Chakravarthula, P. 2024. Cross-Domain Synthetic-to-Real In-the-Wild Depth and Normal Estimation for 3D Scene Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 1290–1300.

Chang, A.; Dai, A.; Funkhouser, T.; Halber, M.; Niessner, M.; Savva, M.; Song, S.; Zeng, A.; and Zhang, Y. 2017. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158.

Chen, L.; Li, S.; Bai, Q.; Yang, J.; Jiang, S.; and Miao,

- Y. 2021. Review of image classification algorithms based on convolutional neural networks. Remote Sensing, 13(22): 4712.

Chen, Z.; Wang, G.; and Liu, Z. 2022. Text2light: Zero-shot text-driven hdr panorama generation. ACM Transactions on Graphics (TOG), 41(6): 1–16.

Cheng, Y.-C.; Lin, C. H.; Lee, H.-Y.; Ren, J.; Tulyakov, S.; and Yang, M.-H. 2022. Inout: Diverse image outpainting via gan inversion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11431–11440.

Chou, S.-H.; Chao, W.-L.; Lai, W.-S.; Sun, M.; and Yang, M.-H. 2020. Visual question answering on 360deg images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, 1607–1616.

Coors, B.; Condurache, A. P.; and Geiger, A. 2018. Spherenet: Learning spherical representations for detection and classification in omnidirectional images. In Proceedings of the European conference on computer vision (ECCV), 518–533. Springer.

Cruz, S.; Hutchcroft, W.; Li, Y.; Khosravan, N.; Boyadzhiev, I.; and Kang, S. B. 2021. Zillow indoor dataset: Annotated floor plans with 360deg panoramas and 3d room layouts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2133–2143.

Dong, Y.; Fang, C.; Bo, L.; Dong, Z.; and Tan, P. 2024. PanoContext-Former: Panoramic total scene understanding with a transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 28087– 28097.

Dongfang, Z.; Zheng, X.; Weng, Z.; Lyu, Y.; Paudel, D. P.; Van Gool, L.; Yang, K.; and Hu, X. 2025. Are Multimodal Large Language Models Ready for Omnidirectional Spatial Reasoning? arXiv preprint arXiv:2505.11907.

Fu, Y.; Lou, M.; and Yu, Y. 2025. SegMAN: Omni-scale context modeling with state space models and local attention for semantic segmentation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 19077–19087.

He, L.; Jian, B.; Wen, Y.; Zhu, H.; Liu, K.; Feng, W.; and Liu, S. 2022. Rethinking supervised depth estimation for 360deg panoramic imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5173–5181.

Huang, H.; Liu, C.; Zhu, Y.; Cheng, H.; Braud, T.; and Yeung, S.-K. 2024. 360loc: A dataset and benchmark for omnidirectional visual localization with cross-device queries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22314–22324.

Huang, H.; Xu, Y.; Chen, Y.; and Yeung, S.-K. 2023. 360vot: A new benchmark dataset for omnidirectional visual object tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 20566–20576.

Jiang, H.; Sheng, Z.; Zhu, S.; Dong, Z.; and Huang, R. 2021. Unifuse: Unidirectional fusion for 360 panorama depth estimation. IEEE Robotics and Automation Letters, 6(2): 1519– 1526.

Jiang, J.; Zhao, S.; Zhu, J.; Tang, W.; Xu, Z.; Yang, J.; Liu, G.; Xing, T.; Xu, P.; and Yao, H. 2025. Multi-source domain adaptation for panoramic semantic segmentation. Information Fusion, 117: 102909.

Jung, D.; Choi, J.; Lee, Y.; Jeong, S.; Lee, T.; Manocha, D.; and Yeon, S. 2025. EDM: Equirectangular ProjectionOriented Dense Kernelized Feature Matching. In Proceedings of the Computer Vision and Pattern Recognition Conference, 6337–6347.

Kanayama, H.; Chamseddine, M.; Guttikonda, S.; Okumura, S.; Yokota, S.; Stricker, D.; and Rambach, J. 2025. ToF-360A Panoramic Time-of-flight RGB-D Dataset for Single Capture Indoor Semantic 3D Reconstruction. In Proceedings of the Computer Vision and Pattern Recognition Conference, 4442–4451.

Kaur, R.; and Singh, S. 2023. A comprehensive review of object detection with deep learning. Digital Signal Processing, 132: 103812.

Kerr, J.; Hari, K.; Weber, E.; Kim, C. M.; Yi, B.; Bonnen, T.; Goldberg, K.; and Kanazawa, A. 2025. Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop. arXiv preprint arXiv:2506.10968.

Kim, G.; Kim, D.; Jang, J.; and Hwang, H. 2024. PAIR360: A Paired Dataset of High-Resolution 360 Panoramic Images and LiDAR Scans. IEEE Robotics and Automation Letters.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.-Y.; et al. 2023. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, 4015–4026.

Li, J.; and Bansal, M. 2023. Panogen: Text-conditioned panoramic environment generation for vision-and-language navigation. Advances in neural information processing systems, 36: 21878–21894.

Li, R.; Pan, P.; Yang, B.; Xu, D.; Zhou, S.; Zhang, X.; Li,

- Z.; Kadambi, A.; Wang, Z.; Tu, Z.; et al. 2024a. 4k4dgen: Panoramic 4d generation at 4k resolution. arXiv preprint arXiv:2406.13527.

Li, R.; Sheng, X.; Li, W.; and Zhang, J. 2024b. OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model.

Li, W.; Zhao, S.; Mou, C.; Sheng, X.; Zhang, Z.; Wang, Q.; Li, J.; Zhang, L.; and Zhang, J. 2024c. OmniDrag: Enabling Motion Control for Omnidirectional Image-to-Video Generation. arXiv preprint arXiv:2412.09623.

Liu, X.; Wen, Z.; Wang, S.; Chen, J.; Tao, Z.; Wang, Y.; Jin, X.; Zou, C.; Wang, Y.; Liao, C.; et al. 2025. Shifting ai efficiency from model-centric to data-centric compression. arXiv preprint arXiv:2505.19147.

Lyu, Y.; Zheng, X.; Zhou, J.; and Wang, L. 2024. Unibind: Llm-augmented unified and balanced representation space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26752– 26762.

Martin, D.; Serrano, A.; Bergman, A. W.; Wetzstein, G.; and Masia, B. 2022. Scangan360: A generative model of realistic scanpaths for 360 images. IEEE Transactions on Visualization and Computer Graphics, 28: 2003–2013.

Mazzola, G.; Lo Presti, L.; Ardizzone, E.; and La Cascia, M. 2021. A dataset of annotated omnidirectional videos for distancing applications. Journal of Imaging, 7(8): 158.

Mirowski, P.; Banki-Horvath, A.; Anderson, K.; Teplyashin, D.; Hermann, K. M.; Malinowski, M.; Grimes, M. K.; Simonyan, K.; Kavukcuoglu, K.; Zisserman, A.; et al. 2019. The streetlearn environment and dataset. arXiv preprint arXiv:1903.01292.

Mo, Y.; Wu, Y.; Yang, X.; Liu, F.; and Liao, Y. 2022. Review the state-of-the-art technologies of semantic segmentation based on deep learning. Neurocomputing, 493: 626– 646.

Mu, Y.; Zhang, Q.; Hu, M.; Wang, W.; Ding, M.; Jin, J.; Wang, B.; Dai, J.; Qiao, Y.; and Luo, P. 2023. Embodiedgpt: Vision-language pre-training via embodied chain of thought. Advances in Neural Information Processing Systems, 36: 25081–25094.

Oh, C.; Cho, W.; Chae, Y.; Park, D.; Wang, L.; and Yoon, K.-J. 2022. Bips: Bi-modal indoor panorama synthesis via residual depth-aided adversarial learning. In European Conference on Computer Vision, 352–371. Springer.

Park, K.; Rameau, F.; Park, J.; and Kweon, I. S. 2024. 360 in the Wild: Dataset for Depth Prediction and View Synthesis. arXiv preprint arXiv:2406.18898.

Rai, Y.; Guti´errez, J.; and Le Callet, P. 2017. A Dataset of Head and Eye Movements for 360 Degree Images. In Proceedings of the 8th ACM on Multimedia Systems Conference, 205–210. Association for Computing Machinery.

Ramakrishnan, S. K.; Gokaslan, A.; Wijmans, E.; Maksymets, O.; Clegg, A.; Turner, J.; Undersander, E.; Galuba, W.; Westbury, A.; Chang, A. X.; et al.

- 2021. Habitat-matterport 3d dataset (hm3d): 1000 largescale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238.

Ran, L.; Zhang, Y.; Zhang, Q.; and Yang, T. 2017. Convolutional neural network-based robot navigation using uncalibrated spherical images. Sensors, 17: 1341.

Sekkat, A. R.; Dupuis, Y.; Vasseur, P.; and Honeine, P. 2020. The omniscape dataset. In 2020 IEEE International conference on robotics and automation (ICRA), 1603–1608. IEEE. Shah, S.; Dey, D.; Lovett, C.; and Kapoor, A. 2017. AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles. In Field and Service Robotics. Shen, Z.; Lin, C.; Liao, K.; Nie, L.; Zheng, Z.; and Zhao, Y.

- 2022. PanoFormer: panorama transformer for indoor 360° depth estimation. In European Conference on Computer Vision, 195–211. Springer.

Song, I.; Joo, M.; Kwon, J.; and Lee, J. 2024. Video question answering for people with visual impairments using an egocentric 360-degree camera. arXiv preprint arXiv:2405.19794.

Su, Y.-C.; and Grauman, K. 2019. Kernel transformer networks for compact spherical convolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9442–9451.

Sun, C.; Sun, M.; and Chen, H.-T. 2021. Hohonet: 360 indoor holistic understanding with latent horizontal features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2573–2582.

- Wang, G.; Yang, Y.; Loy, C. C.; and Liu, Z. 2022. Stylelight: Hdr panorama generation for lighting estimation and editing. In European conference on computer vision, 477–

492. Springer.

- Wang, H.; Xiang, X.; Fan, Y.; and Xue, J.-H. 2024a. Customizing 360-degree panoramas through text-to-image diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 4933– 4943.

Wang, J.; Chen, Z.; Ling, J.; Xie, R.; and Song, L. 2023. 360degree panorama generation from few unregistered nfov images. arXiv preprint arXiv:2308.14686.

Wang, Q.; Li, W.; Mou, C.; Cheng, X.; and Zhang, J.

- 2024b. 360dvd: Controllable panorama video generation with 360-degree video diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6913–6923. Wang, S.; Zhou, D.; Xie, L.; Xu, C.; Yan, Y.; and Yin, E.
- 2025. PanoGen++: Domain-adapted text-guided panoramic environment generation for vision-and-language navigation. Neural Networks, 187: 107320.

Wei, Z.; Lin, B.; Nie, Y.; Chen, J.; Ma, S.; Xu, H.; and Liang, X. 2025. Unseen from seen: Rewriting observationinstruction using foundation models for augmenting visionlanguage navigation. arXiv preprint arXiv:2503.18065.

Wu, T.; Zheng, C.; and Cham, T.-J. 2023. Panodiffusion: 360-degree panorama outpainting via diffusion. arXiv preprint arXiv:2307.03177.

Xia, F.; Zamir, A. R.; He, Z.; Sax, A.; Malik, J.; and Savarese, S. 2018. Gibson Env: Real-World Perception for Embodied Agents. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Xu, J.; Sun, Q.; Han, Q.-L.; and Tang, Y. 2025. When embodied AI meets Industry 5.0: Human-centered smart manufacturing. IEEE/CAA Journal of Automatica Sinica, 12(3): 485–501.

Xu, Z.; Wu, K.; Wen, J.; Li, J.; Liu, N.; Che, Z.; and Tang, J. 2024. A survey on robotics with foundation models: toward embodied ai. arXiv preprint arXiv:2402.02385.

Yadav, K.; Ramrakhya, R.; Ramakrishnan, S. K.; Gervet, T.; Turner, J.; Gokaslan, A.; Maestre, N.; Chang, A. X.; Batra, D.; Savva, M.; Clegg, A. W.; and Chaplot, D. S. 2023. Habitat-Matterport 3D Semantics Dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4927–4936.

Yan, S.; Xu, X.; Zhang, R.; Hong, L.; Chen, W.; Zhang, W.; and Zhang, W. 2024. Panovos: Bridging non-panoramic and panoramic views with transformer for video segmentation. In European Conference on Computer Vision, 346– 365. Springer.

Ye, W.; Ji, C.; Chen, Z.; Gao, J.; Huang, X.; Zhang, S.-H.; Ouyang, W.; He, T.; Zhao, C.; and Zhang, G. 2024. Diffpano: Scalable and consistent text to panorama generation with spherical epipolar-aware diffusion. Advances in Neural Information Processing Systems, 37: 1304–1332.

Yun, I.; Shin, C.; Lee, H.; Lee, H.-J.; and Rhee, C. E. 2023. Egformer: Equirectangular geometry-biased transformer for 360 depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 6101–6112.

Zhang, C.; Cui, Z.; Chen, C.; Liu, S.; Zeng, B.; Bao, H.; and Zhang, Y. 2021. Deeppanocontext: Panoramic 3d scene understanding with holistic scene context graph and relationbased optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 12632–12641.

Zhang, J.; Yang, K.; Ma, C.; Reiß, S.; Peng, K.; and Stiefelhagen, R. 2022a. Bending reality: Distortion-aware transformers for adapting to panoramic semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 16917–16927.

Zhang, J.; Yang, K.; Shi, H.; Reiß, S.; Peng, K.; Ma, C.; Fu, H.; Torr, P. H. S.; Wang, K.; and Stiefelhagen, R. 2024a. Behind Every Domain There is a Shift: Adapting DistortionAware Vision Transformers for Panoramic Semantic Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12): 8549–8567.

- Zhang, W.; Liu, Y.; Zheng, X.; and Wang, L. 2024b. Goodsam: Bridging domain and capacity gaps via segment anything model for distortion-aware panoramic semantic segmentation. arXiv preprint arXiv:2403.16370.
- Zhang, X.; Yang, D.; Song, T.; Ye, Y.; Zhou, J.; and Song,
- Y. 2022b. Classification and object detection of 360° omni-

directional images based on continuity-distortion processing and attention mechanism. Applied Sciences, 12(23): 12398. Zhang, X.; Ye, Z.; and Zheng, X. 2025. Towards Omnidirectional Reasoning with 360-R1: A Dataset, Benchmark, and GRPO-based Method. arXiv preprint arXiv:2505.14197. Zhao, X.; Wang, L.; Zhang, Y.; Han, X.; Deveci, M.; and Parmar, M. 2024. A review of convolutional neural networks in computer vision. Artificial Intelligence Review, 57(4): 99. Zheng, J.; Zhang, J.; Li, J.; Tang, R.; Gao, S.; and Zhou, Z. 2020. Structured3d: A large photo-realistic dataset for structured 3d modeling. In European Conference on Computer Vision, 519–535. Springer.

Zheng, X.; Liu, Y.; Lu, Y.; Hua, T.; Pan, T.; Zhang, W.; Tao, D.; and Wang, L. 2023a. Deep learning for event-based vision: A comprehensive survey and benchmarks. arXiv preprint arXiv:2302.08890.

Zheng, X.; Weng, Z.; Lyu, Y.; Jiang, L.; Xue, H.; Ren, B.; Paudel, D.; Sebe, N.; Van Gool, L.; and Hu, X. 2025. Retrieval augmented generation and understanding in vision: A survey and new outlook. arXiv preprint arXiv:2503.18016.

Zheng, X.; Zhou, P. Y.; Vasilakos, A. V.; and Wang, L. 2024. 360sfuda++: Towards source-free uda for panoramic segmentation by learning reliable category prototypes. IEEE Transactions on Pattern Analysis and Machine Intelligence. Zheng, X.; Zhu, J.; Liu, Y.; Cao, Z.; Fu, C.; and Wang, L. 2023b. Both style and distortion matter: Dual-path unsupervised domain adaptation for panoramic semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1285–1295.

Zhong, D.; Zheng, X.; Liao, C.; Lyu, Y.; Chen, J.; Wu, S.; Zhang, L.; and Hu, X. 2025. Omnisam: Omnidirectional segment anything model for uda in panoramic semantic segmentation. arXiv preprint arXiv:2503.07098.

Zhou, Y.; Zhang, T.; Zhang, D.; Ji, S.; Li, X.; and Qi, L. 2025. Dense360: Dense Understanding from Omnidirectional Panoramas. arXiv preprint arXiv:2506.14471.

Zioulis, N.; Karakottas, A.; Zarpalas, D.; and Daras, P. 2018. OmniDepth: Dense Depth Estimation for Indoors Spherical Panoramas. In Proceedings of the European Conference on Computer Vision (ECCV).

