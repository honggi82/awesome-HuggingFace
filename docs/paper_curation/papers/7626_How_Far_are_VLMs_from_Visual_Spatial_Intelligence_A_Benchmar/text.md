## How Far are VLMs from Visual Spatial Intelligence? A Benchmark-Driven Perspective

Songsong Yu∗, Yuxin Chen∗, Hao Ju∗, Lianjie Jia∗, Fuxi Zhang, Shaofei Huang, Yuhan Wu, Rundi Cui, Binghao Ran, Zaibin Zhang, Zhipeng Zhang, Yifan Wang, Lin Song, Zhedong Zheng, Lijun Wang, Yanwei Li†, Ying Shan, Huchuan Lu, Fellow, IEEE

### arXiv:2509.18905v2[cs.AI]11Nov2025

Abstract—Visual Spatial Reasoning (VSR) is a core human cognitive ability and a critical requirement for advancing embodied intelligence and autonomous systems. Despite recent progress in Vision-Language Models (VLMs), achieving human-level VSR remains highly challenging due to the complexity of representing and reasoning over three-dimensional space. In this paper, we present a systematic investigation of VSR in VLMs, encompassing a review of existing methodologies across input modalities, model architectures, training strategies, and reasoning mechanisms. Furthermore, we categorize spatial intelligence into three levels of capability, i.e., basic perception, spatial understanding, spatial planning, and curate SIBench, a spatial intelligence benchmark encompassing nearly 20 open-source datasets across 23 task settings. Experiments with state-of-the-art VLMs reveal a pronounced gap between perception and reasoning, as models show competence in basic perceptual tasks but consistently underperform in understanding and planning tasks, particularly in numerical estimation, multi-view reasoning, temporal dynamics, and spatial imagination. These findings underscore the substantial challenges that remain in achieving spatial intelligence, while providing both a systematic roadmap and a comprehensive benchmark to drive future research in the field. The related resources of this study are accessible at https://sibench.github.io/AwesomeVisual-Spatial-Reasoning/.

Index Terms—Visual spatial reasoning, vision-language models, multimodal large language models, benchmarking, deep learning, literature survey.

✦

1 INTRODUCTION

# V

ISUAL Spatial Reasoning represents a fundamental cognitive ability in humans [1], [2]. It enables us to derive

rich spatial information from observing the world, which is crucial for navigating and interacting with real-world environments. This capacity is not only vital for humans but is also indispensable for advancing key AI domains such as embodied intelligence and autonomous driving. Consequently, achieving human-level visual Spatial Reasoning in machines has been a long-standing pursuit in the research community [3]–[10]. However, this objective remains highly challenging, primarily due to the inherent complexity of representing and reasoning about three-dimensional (3D) space.

In recent years, Vision-Language Models (VLMs) [4], [11]–[22] attract significant attention due to the powerful visual understanding and reasoning capabilities, bringing widespread applications in various Visual Question Answering (VQA) tasks [23]–[25]. Concurrently, an emerging

- • ∗ Equal contribution, † Corresponding author: Yanwei Li. Work is done during the internship at Tencent ARCLab.
- • Songsong Yu is with Shanghai Jiao Tong University and an intern at ARC Lab, Tencent PCG.
- • Yuxin Chen, Lin Song, and Ying Shan are with ARC Lab, Tencent PCG.
- • Hao Ju, Zhedong Zheng, and Shaofei Huang are with the University of Macau, Macau SAR, China.
- • Lianjie Jia, Fuxi Zhang, Yuhan Wu, Rundi Cui, Binghao Ran, Zaibin Zhang, Yifan Wang, Lijun Wang, and Huchuan Lu are with Dalian University of Technology, China.
- • Zhipeng Zhang is with Shanghai Jiao Tong University, China.
- • Yanwei Li is with The Chinese University of Hong Kong, Hong Kong SAR, China. E-mail: liyanwei@link.cuhk.edu.hk

body of work [26]–[29] begins to apply VLMs to Visual Spatial Reasoning (VSR). However, VSR presents a distinct and more formidable set of challenges compared to generalpurpose VQA, which often focuses on semantic-level understanding, as illustrated in Fig 1. While general VQA targets object recognition or attribute identification, VSR demands intricate reasoning about complex spatial dynamics, such as the relative positions, orientations, distances, and object motion. This need for deep spatial awareness is particularly exacerbated in multi-view and video contexts, which not only amplify the reasoning complexity but also increase the model’s propensity for hallucinations. Furthermore, this difficulty extends to data curation. Building robust VSR datasets is substantially more demanding, requiring precise spatial annotations and complex scene analysis that vastly exceed the requirements of traditional VQA benchmarks.

Meanwhile, research in VSR advances rapidly, with current efforts concentrating on two primary fronts. On one hand, the community continues to drive innovation in input modalities [27], model architectures [28], training strategies [30], and reasoning mechanisms [29]. On the other hand, there is a dedicated push [31]–[34] to develop higherquality and more diverse datasets that enable comprehensive evaluation of VSR capabilities.

Despite a wealth of related research, there remains a lack of a systematic review, particularly in terms of methodologies and the specific task settings. Crucially, despite the recent proliferation of evaluation benchmarks, the landscape remains highly fragmented. Each existing dataset typically addresses only a narrow and specific set of tasks, thus failing to provide a comprehensive assessment of a model’s visual

[Figure 1]

[Figure 2]

[Figure 3]

###### General VQA

[Figure 4]

Q: What is shown in the

video?

[Figure 5]

[Figure 6]

A: The video shows an indoor scene.…

[Figure 7]

###### Visual Spatial Reasoning

Q: If the bed is moved to the center to the room, how far is it from the bookshelf?

Single-Image Occlusion 0.5k Distance 0.67k Coordinate Conversion 0.8k Height 0.6k Situation QA 0.9k Spatial Relation 0.5k Spatial Imagination 0.4k

Existence 0.1k Velocity Acceleration 0.3k Object Localization 0.12k Geometric Reasoning 0.25k

Reach 0.1k Compatibility 0.2k Spatial Grid 0.4k Maze Navigation 0.4k

Multi-Image Shape 0.13k Camera Pose 0.1k Multi-view Reasoning 0.95k

Video Counting 0.7k Localization 0.4k Order 0.4k

Size 0.8k Velocity Acceleration 0.3k Camera Pose 0.38k Route Planning 0.2k

- Fig. 1: Performance of SOTA Models on 23 Visual Spatial Reasoning Tasks (left). The evaluation reveals that the models have significant room for improvement, especially in tasks requiring precise numerical estimation, perspective taking, temporal information processing, and, particularly, spatial imagination. See Table 1 and Table 2 for detailed results. Comparison of Visual Spatial Reasoning and General VQA (Upper-right). While general VQA tasks primarily focus on extracting semantic information from images, VSR necessitates a deeper capacity to model and reason about spatial relationships. Data Formats and Task Settings for Visual Spatial Reasoning (Bottom-right). The evaluation includes 3 input formats and 23 task settings, covering three levels: Basic Perception, Spatial Understanding, and Planning.

spatial reasoning capabilities. Given these issues, this paper aims to provide a detailed review of the existing methods and datasets in the field, while also curating, consolidating, and organizing the current benchmarks into a comprehensive VSR evaluation dataset. It provides a convenient, objective, and comprehensive evaluation tool for assessing the spatial understanding capabilities of VLMs.

In summary, this paper makes the following key contributions:

- • Thorough Review of Visual Spatial Reasoning Methods. We review and analyze existing methods, focusing on input modalities, model architectures, training approaches, and reasoning mechanisms, providing a systematic reference for researchers.
- • A Hierarchical Task Categorization Based on Cognitive Levels. We introduce a systematic categorization of VSR tasks organized by cognitive levels, outlining their core objectives, inherent challenges, and the current research progress to guide future work.
- • Development of SIBench, a Comprehensive Evaluation Benchmark. We introduce SIBench, which curates nearly 20 open-source benchmarks covering 23 distinct VSR task settings. Beyond serving as a rigorous and comprehensive evaluation tool for VLMs, our analysis using SIBench reveals that current models exhibit significant deficiencies in VSR tasks. These shortcomings are particularly pronounced in areas such as precise numerical estimation, multi-view reasoning, temporal information processing, and spatial imagination.

In the following sections, we provide a comprehensive review of over 150 research papers related to VSR published since 2023. Firstly, we define VSR and outline its research scope in Section 2, followed by a thorough review of existing methodologies in Section 3. Next, we categorize the tasks

involved in VSR based on cognitive levels and discuss the primary challenges these tasks present in Section 4. We then introduce a comprehensive evaluation dataset for VSR and present evaluation results for several models in Section 5. Finally, we conclude by summarizing future research directions and potential applications in Section 6, highlighting key challenges that need to be addressed in the field.

2 BACKGROUND

2.1 Research Scope

In recent years, with the development of Vision-Language Models [3]–[5], [19], [35]–[37] (VLMs) and generative models [38]–[43], the realization of spatial intelligence has become increasingly promising. Spatial intelligence manifests in various aspects. First, agents perceive the 3D world through sensor inputs, gaining an understanding of its basic properties, followed by the comprehension of spatial relationships and physical laws. Furthermore, agents can interact with their environment, such as performing spatial navigation and manipulating objects. Additionally, the ability to create and imagine entirely new worlds is also a crucial aspect of spatial intelligence. The scope of spatial intelligence is vast, covering a wide range of tasks and applications, such as spatial reasoning, visual-language localization [44], [45], embodied artificial intelligence [46]– [49], and video world models [50]–[52].

For agents, image/video input is a readily accessible and cost-effective form of data. For biological organisms, vision is also a vital pathway for spatial modeling. Therefore, this paper focuses on the application of VLMs in spatial reasoning, which involves interpreting spatial information from images, multi-view inputs, or videos, including basic perception, spatial relationship understanding, and planning.

It should be noted that tasks involving point clouds [53]– [59] as input for multimodal spatial reasoning or puretext-based [56], [60]–[63] spatial reasoning are not within the scope of this research. Additionally, while generative models [64]–[69] also reflect spatial intelligence, they differ significantly in their modeling mechanisms and spatial representation form VLMs, and thus are not part of this discussion. Our focus is primarily on the general ability of VLMs in understanding spatial relationships, specific applications like vision-language action models [70]–[73] and vision-language navigation models are not the main focus of our study.

###### 2.2 Related Work

Several studies have provided comprehensive reviews of the development and evaluation frameworks of VisionLanguage Models (VLMs). For example, [74] systematically organizes the technological evolution, evaluation benchmarks, and application scenarios of VLMs, highlighting the potential performance fluctuations when handling complex tasks. The study further suggests introducing new evaluation metrics, emphasizing the importance of visual localization and multimodal understanding.

The study [75] investigates the integration of LLMs with 3D spatial understanding, analyzing challenges in data representation, model architecture, and evaluation metrics. It also recommends improvements in these three areas. It should be noted that the study in [75] primarily focuses on using 3D representations as input, rather than image/video input. Zha et al. [76] similarly explores the spatial reasoning capabilities of LLMs, categorizing tasks based on input modalities such as images, point clouds, and hybrid modalities. Additionally, some studies [77]–[79] have revealed shortcomings in the spatial reasoning abilities of VLMs, citing issues such as the lack of effective spatial attention mechanisms or the limited ability of existing attention mechanisms to align with object positions. Sapkot et al. [80] reviews the development of Vision-Language Action Models (VLA) and points out challenges in real-time control, dataset bias, and system integration when applying these models in practice. Researchers in [81] discusses the application of VLMs in autonomous driving scenarios, noting that the reliability of VLMs remains insufficient in complex traffic environments. Guo et al. [82] summarizes the technological evolution and applications of VLMs, highlighting challenges in data quality and complex tasks within the biomedical field.

#### 3 METHODOLOGIES

VSR requires that VLMs not only understand semantic information and localize targets but also reason about the spatial relationships between multiple objects, imagine 3D spatial structures from 2D images, and have the potential for dynamic prediction. In this chapter, we analyze existing research methods and summarize them into four areas for improvement: input modalities, model architecture, training strategy, and inference methods (see Fig. 2 for more details).

###### 3.1 Naive Solution

To better understand VSR methods, we begin by introducing the commonly used naive solutions. Similar to general VLMs, a VSR model takes an image I and a spatially-related question Q as input to produce an answer A. The overall process can be conceptualized as modeling the conditional probability P(A|I,Q). The architecture typically consists of three core components: a vision encoder fvision, a projection module gproj, and a Large Language Model (LLM) fLLM.

Generally, the vision encoder processes the image to extract a set of feature vectors Fv = fvision(I), where Fv ∈ RN×D

v is a sequence of N feature vectors, each with dimension Dv, capturing both semantic and spatial information. Concurrently, the text question Q is converted into embeddings Eq ∈ RM×D

t by text encoder. Given the potential mismatch in dimensions and feature spaces between the two modalities, the projection module is employed to align the visual features with the text embedding space. This module transforms Fv into a new representation Fv′ = gproj(Fv). This projection is usually implemented with a few linear layers or a cross-attention mechanism, mapping the visual features so that Fv′ ∈ RN×D

t, aligning their dimension with the text embeddings. Finally, the features from both modalities are concatenated and fed into the LLM, which then autoregressively generates the answer: A = fLLM(concat(Fv′,Eq)). Notable examples of this structure include SpatialVLM [83], LLaVA-VSD [17], and other related works [84]–[96]. This naive solution is both concise and efficient. Subsequent work, focusing on the specific characteristics of visual-spatial reasoning tasks, typically introduces improvements in four key aspects: input modalities, model architecture, training strategies, and inference methods.

###### 3.2 Assisted Input Modalities

RGB images are a planar projection of the 3D world, providing continuous texture information. However, during the projection process, real 3D structural information is lost, which is one of the challenges in visual spatial reasoning tasks [97]. 3D point clouds offer a better representation of structure, but their data scale, diversity, and quality fall far behind that of images. As a result, some approaches [27], [28] attempt to strike a balance between 3D and 2D. These methods, often referred to as 2.5D [7], utilize depth maps as an additional modality. The motivation is that combining depth maps with images can yield a representation similar to point clouds, enabling the recovery of 3D structure in the real world, while the input images retain complete complex texture. Furthermore, such RGB-D data is relatively easy to obtain, especially in embodied scenarios.

Due to the substantial distribution gap between depth maps and natural images, effectively encoding depth maps is by no means a trivial task. Additionally, the visual encoders of most VLMs are only trained on text-image pairs, and simply concatenating RGB and depth features may negatively affect performance [27].

To address these issues, SpatialRGPT [27] first replicates the single-channel depth map D to match the three channels of an RGB image. It then employs a shared vision

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

MM-Spatial

RoboRefer

Spatial-MLLM

RoboRefer

Svqa-R1

Pixel Reasoner

AdaptVis

3DLLM-Mem

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

SpatialRGPT

SpatialCLIP

VG-LLM

SpatialLLM

Embodied-R

Spatial Reasoner

VSI-Bench

VADAR

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

SSR

SpatialBot

ViCA2 VLM-3R

SpaceR

Videochat-R1

Visual CoT VoT

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

| |[Figure 32]| | |
|---|---|---|---|
| | |[Figure 33]| |
| |[Figure 34]| |[Figure 35]|
| |[Figure 36]<br><br>[Figure 37]| | |

Format Reward Caption Reward

[Figure 38]

: Reasoning step by step!

|Answer|
|---|

IoU Reward

Depth

[Figure 39]

Accuracy Reward GRPO

[Figure 40]

: Ok. Let me think…

|𝑓LLM|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

| | |
|---|---|
|[Figure 44]<br><br>[Figure 45]<br><br>Question| |

| | | | |
|---|---|---|---|
| |𝑔proj| |[Figure 46]|
| | | | |

[Figure 47]

CoT Cognitive Map

[Figure 48]

[Figure 49]

Policy Model

[Figure 50]

Proj. 𝑓vision

Single Image

[Figure 51]

[Figure 52]

KL loss

|[Figure 53]<br><br>VLM|
|---|

|[Figure 54]<br><br>VLM|
|---|

[Figure 55]

[Figure 56]

[Figure 57]

Question

[Figure 58]

Reference Model

Query

Multi-View

[Figure 59]

[Figure 60]

[Figure 61]

|[Figure 62]<br><br>Detect<br><br>[Figure 63]|Depth|
|---|---|

- Stage1:SFT

[Figure 64]

- Stage2:RL

|[Figure 65]<br><br>[Figure 66]|
|---|

Database

Spatial

Enc.

[Figure 67]

[Figure 68]

[Figure 69]

RAG API

Video

Ⅰ: Input Modality Ⅱ: Model Architecture Ⅲ: Training Strategy Ⅳ: Inference Strategy

- Fig. 2: An overview of the primary methods (Bottom). I: Incorporating an additional input modality, such as depth maps. II: An additional spatial encoder is incorporated into the model architecture to provide 3D information. III: Leveraging Reinforcement Learning to improve generalization. IV: The inference phase employs methods such as cognitive maps to perform structured reasoning. Representative methods for the four categories (Upper).

encoder, fvision to process both modalities, yielding features Fv = fvision(I) and Fd = fvision(D). These features are subsequently passed to different projectors, a design which ensures the model can still function normally without depth input and avoids the need for large-scale imagedepth data training. Building on this, RoboRefer [28] uses a completely separate Depth encoder fdepth to handle depth inputs, thereby avoiding modality interference. This architecture choice can be expressed as: Fv = fvision(I) and Fd = fdepth(D). This separation helps preserve general VQA performance while enhancing the perception of depth cues. SSR [98] also aims to integrate depth information in a similar manner. In addition to using modular designs for feature fusion, SpatialCLIP [97] is inspired by point cloud reconstruction and 3D point cloud architectures. It utilizes depth information to lift 2D tokens to 3D space, thereby better modeling the 3D spatial relationships. Specifically, for each transformer layer, it first lifts the 2D tokens to 3D voxels based on the average depth of each patch, and then introduces 3D depth convolutions to capture spatial patterns in the 3D space, thus providing enhanced spatial information.

In addition to processing at feature space level, some models introduce an explicit, symbolic reasoning step by interacting with external tools. SpatialBot [99] provides a method for querying depth values from a given depth map D through an API. The process begins when the model receives an image I and a question Q. Based on the question, SpatialBot first determines if depth information is required.

If so, it identifies the target coordinates (x,y) in the image relevant to the question. For instance, when asked about an object’s depth, it first computes the objects’s bounding box, Bobj, and then determines its center point: (xc,yc) = Center(Bobj). Subsequently, the model generates a symbolic API call, such as Depth(xc,yc). This triggers an external function that queries the depth map D to retrieve the corresponding depth value d: d = API Query(D,(xc,yc)). This numerical value d is then converted back into a natural language string, Td. Finally, SpatialBot generates the final answer A by processing a new augmented prompt that concatenates the original question with the retrieved depth information. This method of querying an API and re-inputting the information as text is further explored by MM-Spatial [100]. Their experiments also revealed that this “Query-and-Re-input” cycle yields significantly better results for spatial reasoning tasks in VLMs. While this class of methods often achieves high accuracy, its pipeline is also quite complicated.

###### 3.3 Model Architecture Optimization

Numerous existing Vision-Language Models, such as LLaVA [101] and its subsequent versions [18], [94], [102] employ contrastive learning for pretraining visual encoders, generating compact and expressive visual embeddings. These models align well with natural language, demonstrating strong performance in tasks like image captioning and general visual question answering. However, similar to the contrastive learning used in CLIP [103], they primarily

optimize for global semantic alignment and often neglect fine-grained spatial reasoning. Consequently, the visual embeddings produced by these encoders, while capturing an image’s overall semantic gist, provide only a coarse representation of its contents and are inherently limited in encoding precise spatial information.

To overcome this limitation, a powerful strategy has emerged: the dual-visual-encoder architecture. This design addresses the issue by creating two complementary visual pathways. While the original encoder provides the highlevel semantic context, a second, specialized encoder is introduced to supply the fine-grained details. For singleimage inputs, models like SpatialLLM [104] and ViCA2

- [105] incorporate additional encoders pre-trained on detailoriented tasks. The reason for choosing models like MAE
- [106], DINO v2 [107], and SAM [108] is that their training objectives, such as masked image reconstruction or large-scale segmentation, force them to learn rich, pixel-level features that the primary semantic encoder ignores. For inputs with inherent 3D information, like video or multi-view images, this approach is taken a step further. Standard fine-grained encoders may not be sufficient to interpret the geometric cues embedded in motion and parallax. Therefore, several studies use 3D reconstruction models as the second encoder. For instance, VLM-3R [109] incorporates CUT3R [110] to model the scene’s underlying 3D geometry from multiple views. By doing so, the model gains access to much richer representation that include depth and structure. Similarly, VG-LLM [111] and Spatial-MLLM [112] use advanced reconstruction networks like VGGT [113] to provide the language model with an even more sophisticated understanding of 3D space, significantly enhancing its spatial reasoning capabilities.

###### 3.4 Training Strategy Optimization

VSR goes beyond perceiving static structures, typically requiring dynamic, multi-step reasoning grounded in commonsense knowledge. This multi-step nature makes it a natural fit for reinforcement learning, especially Group Relative Policy Optimization (GRPO) [114], which has already shown success in enhancing the textual reasoning and generation capabilities of LLMs and VLMs. Inspired by this success, a growing body of work [30], [115]–[117] is now investigating RL’s potential for spatial reasoning tasks. However, unlike mathematical reasoning [24] or code generation [25] tasks, where correctness can be directly verified, spatial reasoning lacks a clear, well-defined reward signal. This makes it difficult for RL to acquire spatial reasoning abilities effectively, leaving the application of RL to VSR a challenging and open problem.

To deal with such challenges, current works generally adopt two strategies. The first category [28], [117]–[119] adopts a two-stage paradigm, in which a Supervised FineTuning (SFT) stage is introduced before Reinforcement Learning Fine-Tuning (RLFT) to provide explicit supervision for spatial reasoning. The SFT stage serves as a warm start for training, while the RLFT stage empowers models with strong generalization ability. For example, RoboRefer [118] first trains VLMs with depth, spatial understanding data, and instruction tuning at the SFT stage and further

fine-tunes VLMs with multi-step reasoning data at the RLFT stage. The second category adopts a one-stage paradigm, which focuses on designing task-specific reward functions tailored for spatial reasoning in RL training. Considering the invariance and variations between original and flipped images, Wang et al. [116] extend GRPO to Spatial-GRPO, comparing rewards between the original and flipped groups and penalizing the group with a higher score. SpaceR [30] explicitly constructs an object-centric map and defines a reward function based on this map to establish quantitative feedback for spatial understanding. In the video spatial reasoning domain, Li et al. [115] incorporate additional explicit IoU-based and recall-based rewards into GRPO. By training with different combinations of reward functions in GRPO, these methods achieve improved performance and generalization ability compared with SFT-trained variants.

###### 3.5 Inference Strategy Enhancements

VSR requires VLMs to conduct a series of steps, such as scene understanding, object grounding, and various forms of reasoning, including relationship and trajectory. Therefore, introducing intermediate procedures (e.g., building a map for the scene and integrating multimodal features) can effectively strengthen both visual and spatial capabilities. To provide a clear taxonomy, we categorize existing methods according to the source of their enhancement signals.

Internal enhancement. Internal enhancement refers to leveraging pretrained world knowledge inherent in LLMs without injecting additional external knowledge. According to the objects of enhancement, we divide this into three categories. (1) Multimodal Chain-of-Thought (CoT). Prompting LLMs to conduct several steps of linguistic reasoning has shown effectiveness [120], [121]. Inspired by this, multimodal CoT has been introduced for VSR tasks, enabling joint reasoning over visual and textual spaces and thereby enhancing spatial awareness during intermediate reasoning. To elicit and enhance the reasoning steps, previous works [63], [122] propose to visualize the intermediate steps of reasoning. For instance, Wu et al. [122] prompts the pretrained VLMs to generate reasoning traces z1....i and spatial visualizations v1....i in a separate manner as follows:

###### zi ∼ pθ(vi|x,z1....i−1,v1....i−1), (1) vi ∼ pθ(vi|x,z1....i,v1....i−1). (2)

Inspired by ReAct [123], Yao et al. [63] propose to generate intermediate spatial visualizations and reasoning traces simultaneously as follows:

###### vi,zi ∼ pθ(vi,zi|x,z1....i−1,v1....i−1). (3)

Both methods aim to make reasoning steps interpretable. To deal with the lack of intermediate supervision for spatial CoT, Visual CoT [124] collects a dataset consisting of detailed spatial reasoning steps and bounding boxes of Regions of Interest (RoI), serving as ground-truth for CoT supervision. Based on the dataset, the VLMs can be trained to extract visual tokens from the RoIs and enhance reasoning steps based on both RoIs and the entire input, following the ground-truth chain of thought. This design enables the model to dynamically focus on the relevant regions during

[Figure 70]

- Fig. 3: Taxonomy of visual spatial reasoning according to cognitive levels.

spatial reasoning steps. (2) Scene Representation. Inspired by the hippocampus’s central role in contextual memory [125], some methods [31], [126] propose to generate internal representations of spaces and conduct reasoning based on these. Given a video input and a question, Yang et al. [31] first construct a cognitive map consisting of object center positions in a grid format. Then the predicted map is leveraged to answer the question, achieving robust results regarding local distance awareness. Similarly, when exploring scenes in embodied tasks, a powerful internal representation is also useful to maintain past and current observations. For example, some works [127], [128] adopt 3D voxels as internal scene representation, on which the occupancy and exploration status are updated first. Afterwards, voxels are projected onto a 2D semantic map [127]. For the unexplored areas, this map is utilized to predict probabilities over three candidate directions and to estimate whether further exploration is worthwhile based on current observations. (3) Attention re-distribution. During reasoning, there exists an imbalance between visual and textual attention. Specifically, image tokens constitute 90% of the input sequence but only account for 10% of the total attention, leading to misalignment between the actual spatial layout of objects in images and instruction prior in texts [78]. To deal with this imbalance, AdaptVis [78] introduces a dynamic attention redistribution strategy that leverages output logits as guidance. When the guidance is low, the attention distribution is smoothed, encouraging the model to explore a broader range of inputs. In contrast, when the guidance signal is high, the attention distribution is sharpened, directing the model to focus on key objects.

External enhancement. External enhancement supplements VLMs with knowledge beyond their pretrained parameters to understand illogical or uncommon spatial relationships. One representative direction is the use of multi-agent system. Marsili et al. [129] designs a cooperative system where different agents interact in a code policy manner: API generation agents first decompose the query into sub-questions, while program synthesis agents receive these sub-questions and generate code to solve each problem. Another line of work is Retrieval Augmented Generation (RAG). For example, Yu et al. [130] retrieves templates for subjects-objects relationships as well as other spatial descriptions, and then integrates these templates into the inference stage via in-context learning.

- 4 TASK SETTINGS

While the systematic evaluation of VSR capabilities in VLMs has gained significant attention, current evaluation baselines

face several challenges. First, different benchmarks classify task settings at varying levels of granularity, lacking a systematic framework. Second, task settings are scattered across various benchmarks, making it difficult to assess VSR capabilities comprehensively.

In this section, we first provide a comprehensive overview of VSR task settings, organizing it into three cognitive levels: Basic Perception Section 4.1, Spatial Relation Reasoning Section 4.2, and Spatial Planning Section 4.3, as shown in Fig. 3. The classification is based on levels of reasoning, where basic perception concerns the attributes or states of a single object or category of objects, spatial understanding involves the spatial relationships among multiple objects, and planning refers to seeking satisfactory solutions under spatial constraints. For each level, we introduce the definition of the corresponding task and the challenges it presents.

###### 4.1 Basic Perception

Current Vision-Language Models (VLMs) are typically first pre-trained on vast corpora of text data, where fundamental object concepts from world knowledge are represented and associated within an abstract vocabulary. Subsequently, through multimodal training, connections are established between this abstract vocabulary and continuous visual representations or discretized image tokens. This process endows them with basic visual question answering (VQA) capabilities. However, compared to the goal of achieving spatial intelligence, which entails a fine-grained understanding of the 3D world, current VLMs still exhibit deficiencies in foundational object perception. These shortcomings manifest specifically in the perception of object attributes and states. (See Fig. 4 for an overview of basic perception tasks.)

4.1.1 Attributes

Attributes of objects refer to characteristics that remain relatively stable and are unlikely to change in the short term. These typically include shape, color, size, and quantity. The shape of an object is essential for distinguishing between different objects and understanding scenes. Object shape recognition tasks typically begin with simple 2D geometric figures, which include two types of task setups: whole-shape recognition [131], [132] and sub-shape recognition [133]. MM-Bench [134] increases the difficulty by requiring models to recognize object shapes in real images. Additionally, some datasets use multi-view images as input, such as MMSI-Bench [135], which asks the model to deduce an object’s shape from multiple views, synthesizing the information to infer the shape from a fixed viewpoint. Color also plays a critical role in object recognition, and colorrelated tasks are generally simpler, requiring the model to identify the color of a specific object [134], [136], [137]. In contrast, perceiving the size of an object is more complex. Some studies present two objects for comparison, asking the model to determine which is taller, shorter, wider, narrower, larger, or smaller. Other research requires the model to estimate an object’s size quantitatively, which involves using commonsense knowledge for direct measurement [27], [31], [138]. In OMNI3D-Bench [139], a reference object’s physical measurement is provided as a standard. Object quantity

|[Figure 71]<br><br>[Figure 72]|
|---|

Q: Are all four sinks the same size?

|[Figure 73]<br><br>[Figure 74]|
|---|

|[Figure 75]<br><br>[Figure 76]|
|---|

Q: Is the microwave door open?

Q: How many displays are to

the right of the keyboard? A: 1

A: Yes

A: No

Size

##### Counting Object Shape

State Direction

###### Attributes States

###### Color

Q: Which is wider, the

|[Figure 77]<br><br>[Figure 78]|
|---|

|[Figure 79]<br><br>[Figure 80]|
|---|

Q: What color is the sofa? (A)… (B) white (C) …

|[Figure 81]<br><br>[Figure 82]|
|---|

Q: Which direction is

photograph on the table

the man facing?

or the gramophone? A: gramophone

(A) … (B) left A: (B)

A: (B)

- Fig. 4: Task Settings for Basic Perception. Basic perception tasks are categorized into static and state attributes based on whether the attribute is subject to change.

recognition [34], [131], [136], [140]–[144] is one of the more challenging tasks in static attribute understanding. It depends on distinguishing between objects based on attributes such as shape, color, and size. This makes hallucination phenomena more pronounced in tasks involving object quantity recognition.

localization, distance measuring, and spatial compatibility reasoning.

Spatial Relationships. Early works in the paper VSR [148] identified 66 types of spatial relationships e.g., under, in front of, facing. The test data consists of image-text pairs, where the text includes a description and a label indicating correctness, such as: <Caption: The cow is ahead of the person. Label: False. >. This requires VLMs to first localize the two objects in the image and then determine whether the abstract relation ‘ahead of’ holds. Later studies introduced additional positional relationships, such as occlusion, reach, and containment. The difficulty in these tasks often lies not in object recognition but in understanding specific spatial relationships. The model must establish connections between the spatial structural relations between objects and the corresponding relation terms.

- 4.1.2 States

In contrast to attributes, states refer to characteristics of objects that are subject to change. Common state attributes include an object’s posture, orientation, and its open/closed status. In OpenEQA [137], the task requires the model to estimate the open/closed state of a microwave door, while other studies focus on determining an object’s front-facing orientation in space [32], [136], [145]. Additionally, some research [143], [146] introduces a temporal dimension, requiring the model to understand how states evolve over time. For instance, this might involve describing changes in an object’s rotational angle from the past to the present, or transformations in its shape resulting from specific actions [147]. While the perception of state attributes typically involves describing the properties of a single object or category of objects, and thus falls under basic perception, it is more complex than understanding static attributes. This complexity arises from the diversity of state attributes and the need for commonsense reasoning.

Object Localization. The object localization task requires the model to accurately identify the position of a specified target within an image. The specific localization methods include detecting the precise spatial coordinates of the target object, determining which bounding box best fits the target, or estimating the coordinates of key points on the target. The BLINK [149] test sample provides an image along with a labeled bounding box for the target object, where the model chooses the bounding box that best aligns with the target from multiple options. The SITE [142] involves the model locating a specific object based on a description or hint and estimating the coordinates of its 2D bounding box. Furthermore, the STI-Bench [150] requires the model to localize the 3D bounding box of the target object within a video. Object localization tasks are also included in other benchmarks [29], [34], [134], [137], [137]. The challenge in these tasks lies in the model’s need to possess strong object detection capabilities while also handling various complexities in the background and interference, such as partial occlusion and similar objects.

###### 4.2 Spatial Understanding

Compared to basic perception, spatial understanding requires the model not only to passively perceive the properties of individual objects or object categories, but also to comprehend the relationships between multiple objects. Furthermore, depending on whether the reasoning object includes a temporal dimension or viewpoint changes, we classify spatial understanding into static understanding and dynamic understanding. See Fig. 5 for an overview of spatial understanding tasks.

Distance Measuring. Distance measuring tasks aim to estimate the distance between objects in an image or the distance from an object to the camera, which is also referred to as depth estimation. Distance measuring typically involves two forms: qualitative judgment of proximity and quantitative numerical estimation. For instance, in the paper BLINK [149], the task requires the model to determine which of two specified points is closer to the camera; in

- 4.2.1 Static Understanding

The input to static understanding tasks consists of a single image and the corresponding question, with no temporal or viewpoint changes. The tasks primarily include understanding the spatial relationships between objects, object

Q: Is the car beneath the cat？

Q: What is the most appropriate description of the camera's movement trajectory?

Q: With cabinet (red point) at a depth of 4.2 meters, calculate the depth difference between windowframe (green point) and window (blue point).

Q: Which of the options is simply the original shape in a rotated orientation? (A) … (B).Center (C) … (D) …

A: No

|[Figure 83]<br><br>[Figure 84]|
|---|

A: (C) move forward 70m

[Figure 85]

|[Figure 86]<br><br>[Figure 87]|
|---|

A: 1.0

A: (B)

|[Figure 88]<br><br>[Figure 89]|
|---|

[Figure 90]

Spatial Relation Object localization Compatibility Distance Spatial Imagination Velocity

Dynamic Understanding

Static

Route Description

Understanding

Acceleration

|[Figure 91]<br><br>[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]<br><br>[Figure 95]|
|---|

|[Figure 96]<br><br>[Figure 97]|
|---|

What is the camera's instantaneous acceleration around t=15s? (unit: m/s²) (A) 0.11m/s²(B) … (C) … (D) …

Q: What is the average speed of the camera between 14s and 17s?

Q: Which bounding box more accurately localizes and encloses the vase?

Q: Can the box with the red outline

fit in the drawer?

(A) 0.11m/s (B) … (C) … (D) …

A: Yes

A: (A)

A: (A)

A: (B)

- Fig. 5: Categorization of Spatial Understanding Tasks. Spatial understanding tasks are divided into static and dynamic understanding. Dynamic understanding tasks are characterized by viewpoint shifts or a temporal component.

SPAR-Bench [151], the task asks the model to judge which object is closer to or farther from a reference point (such as another object or an observer) based on the 3D center point position of the objects. In addition to qualitative distance analysis, SpatialVLM [26] employs an automated data construction pipeline to obtain distance data, enabling quantitative distance estimation. In the paper SpatialBench [34], desktop-level distance estimation is introduced for use in embodied scenarios. Subsequently, many studies have incorporated distance measurement as a crucial task in spatial reasoning [27], [31], [109], [138]–[142], [150], [152]–[158]. The challenge of this task lies in the need for the model to infer spatial relationships using implicit cues such as structural information in the scene, object size ratios, and physical knowledge. Furthermore, varying camera parameters may introduce misguidance in distance perception.

Spatial Compatibility Reasoning. Spatial compatibility reasoning tasks require VLMs to determine whether one object can fit the spatial properties or another after thoroughly understanding the object’s shape, size, and spatial relationships. In real-world scenarios, this includes determining whether an object can fit into a given space [137] (’Can another cookie jar fit on the cookie jar shelf?’). This task assesses the model’s reasoning abilities concerning spatial constraints and physical adaptation relationships. Additionally, OmniSpatial [33] introduces reasoning about motion processes, such as determining whether an object can pass through a narrow passage. Some synthetic datasets also require the model to understand layout constraints and graphical combination rules in a 2D space, enabling it to provide fitting solutions. The difficulty of this task lies in the model’s need to not only perceive the size and shape of objects but also reason about physical world common sense, such as “an object too large cannot fit into a small space” or “some uniquely shaped objects cannot be stacked together”.

4.2.2 Dynamic Understanding.

Dynamic understanding typically involves inputs such as multi-view images or videos, introducing the time dimension or changes in viewpoint. Compared to static understanding, dynamic understanding is more complex and variable. The primary tasks include trajectory description, velocity and acceleration estimation, as well as spatial imagination.

Trajectory Description. The goal of the trajectory description task is to identify and describe the dynamic changes in the object or camera pose over a time sequence based on visual inputs, generating structured language descriptions. In this context, the trajectory refers to the continuous recording of the camera’s pose over time. The descriptions can be divided into two categories based on the observer (camera viewpoint) and the observed world (objective world): one describes the camera’s orientation and pose, and the other describes the objects within the observed world. In STIBench [150], the model is required to infer the position and orientation of the camera at a specific time based on video input and initial pose information. In the Ego-ST Bench [159], the test requires the model to qualitatively describe changes in direction, such as “turn left first and then turn right”. Additionally, multi-view reasoning tasks [33], [109], [135], [149], [150], [153], [158], [160], [161] involve numerous estimations of quantified camera parameters. Another category of trajectory description pertains to the observed world, involving tasks like recognizing changes in the direction of other objects in the video, such as “the car changes from moving straight to turning left”. In the VSIBench [31], a task is designed that requires the model to determine the order of appearance of objects in a spatiotemporal sequence. Similarly, in the SPACE [154], multiple landmarks are introduced, and the model is required to determine their sequence. The challenges in these tasks arise

from two key issues: first, the model must infer 3D spatial from 2D RGB inputs, which involves geometric principles such as perspective projection, a problem that cannot be resolved by simple image feature matching alone. Second, the estimation errors accumulate as the video sequence length increases, requiring the model to have dynamic error correction capabilities. This is a significant challenge in video pose estimation.

Velocity and Acceleration Estimation. In dynamic spatial reasoning tasks, models are required to estimate motion parameters, such as velocity and acceleration, which are crucial for subsequent planning and decision-making. The OmniSpatial [33] requires models to qualitatively assess the speed and acceleration of objects, like “Which car, A or B, is moving faster?” or “Is the acceleration of the ball on the slope increasing?”. In contrast, STI-Bench directly asks models to estimate the numerical values of velocity and acceleration from video, such as “What is the average speed of the camera?” or “How quickly is the ball accelerating?”. Additionally, STI-Bench [150] categorizes tests into three different environments: desktop, indoor, and outdoor, based on varying accuracy requirements for motion parameter estimations. In DynSuperCLEVR [162], the test introduces two additional settings: future prediction and counterfactual reasoning. Future prediction requires the model to infer the future motion state of an object based on its current speed and acceleration, such as “How fast will the blue sphere be moving after 2 seconds?”. Counterfactual reasoning, on the other hand, asks the model to reason about hypothetical scenarios where speed or acceleration changes, such as “If the green cylinder’s acceleration were doubled, when would it collide with the wall? The task of estimating velocity and acceleration requires the model to integrate displacement changes and temporal information within a 3D space while relying on an understanding of physical laws. The challenges of this task arise from two key issues: first, velocity and acceleration are multi-stage dynamic attributes. Estimating speed involves calculating displacement, followed by the estimation of acceleration, which creates a longer inference chain. Second, counterfactual reasoning and future prediction tasks require the model to reconstruct scenarios, involving complex “what-if” logic chains that are more difficult than simple factual queries.

Spatial Imagination. In this paper, we define spatial imagination tasks as those require reasoning from a hypothetical viewpoint different from the input perspective, given a visual input. It is important to note that some works [33], [138], [163] refer to the camera viewpoint as the “egocentric perspective”, while others [141], [164] refer to it as the “allocentric perspective.” To avoid ambiguity, we use the term “hypothetic perspective” to distinguish it from the camera viewpoint, which is the perspective directly provided to the model. The inputs can be a single image, requiring the model to reason from a hypothetical perspective. For example, “If the waiter is on the side of the vase, then who is on his left?” [33] The SQA3D requires the model to first locate a hypothetical viewpoint in the video based on the description and then perform a question-answering task, such as “Sitting at the edge of the bed and facing the sofa, can I go straight to the coffee table in front of me?” The MINDCUBE [126] considers reasoning from limited views, aiming

to assess the model’s ability to mentally model space, such as “If you move from view 1 to view 2, what is the furthest from you?” There is a significant body of research [31], [32], [142], [151] in this area, some of which also accounts for spatial understanding differences across various viewpoints and language conventions [164], or conducts reasoning from synthetic collections of images [165]. The naming of spatial imagination tasks varies across studies, SQA3D [166] labeling it as situational question-answering, and SRBench [165] referring to it as mental rotation. Spatial imagination tasks are not simply direct perception of static object attributes but require the model to construct spatial representations of objects in the mind and perform viewpoint transformations, involving dynamic operations on spatial structures. Thus, the challenge of spatial imagination lies in the need of the model to overcome the limitations of the input perspective, build accurate topological and spatial structural relationships from limited viewpoint inputs and imagine visual cues from the hypothetical perspective.

###### 4.3 Spatial Planning

Spatial planning aims to leverage previous observations of the environment to produce feasible actions and future predictions. Compared with spatial perception and understanding, spatial planning goes a step further by transforming static recognition and relational reasoning into dynamic decision-making processes. Given the different types of predictions, we divide spatial planning into three categories, i.e., environment planning, maze planning, and embodied planning. Four representative tasks are shown in Fig. 6.

4.3.1 Environment Planning

Environment planning requires VLMs to generate a spatial representation of the surrounding environment and subsequently reason over it. This process consists of two stages. In the first stage, VLMs receive a video walking through the environment and build a representation based on it. In the second stage, they perform reasoning based on the constructed representation. VLMs start to reason based on representation. Depending on the planning goals, this task can be divided into three settings.

Shortcut Discovery. After observing a video walkthrough from the start to the goal (with unnecessary steps), this task requires VLMs to plan a never-seen shortcut from the start to the goal [154]. The challenges of this task lie in (1) overall understanding of the environment, (2) interactive planning of actions given previous observations.

Route Retracing. In SPACE [154], this task requires VLMs to retrace the route from the start after observing a video from the start to the goal. The route in the video is always the shortest path to the goal. Similar to shortcut discovery, the challenge of this task also lies in the interactive planning of actions.

Map Sketching. After observing a video walkthrough from the start to the goal, map sketching [154] requires VLMs to pick the best map representation of the environment from multiple options. The map contains landmarks, including the start and goal, with different spatial relationships. The challenges of this task lie in (1) recognition of critical landmarks, (2) comprehension of spatial relationships among landmarks.

Q: You are a robot beginning at the door facing the bathtub. You want to navigate to

Q: How many right turns are there in the

the toilet. You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): 1. Go forward until the bathroom sink. 2. [please fill in] 3. Go forward until the toilet. You have reached the final destination.

provided path from S to E? Available

|[Figure 98]|
|---|

options: (A) 4 (B) 3 (C) 7 (D) 2

|[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]|
|---|

A: (B)

A: (B)

###### Maze Navigation Planning

Vision and Language Navigation Map Sketching

[Figure 103]

Shortcut Discovery

|[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]|
|---|

[Figure 108]

|[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]|
|---|

Q: You are shown a video walkthrough of some route to the goal from a start localtion. The route may be long with unneccesary

You are shown a video walkthrough

|[Figure 113]|
|---|

demonstrating an exploratory path through an environment. Sketch a map of the environment with the locations of thestart, goal and landmarks. Pickthe best option.

[Figure 114]

[Figure 115]

|[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]|
|---|

|[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]|
|---|

detours. You are placed at the start

location. Find a shortcut to the goal.

Fig. 6: Categorization of Spatial Planning Tasks.

- 4.3.2 Maze Navigation Planning Maze navigation planning [154] involves navigating in a 2D grid world, from the start point to the exit point. Compared with general environment planning, maze planning features a clearly defined start and exit point, as well as a simpler environment. The key challenge lies in multi-hop reasoning while avoiding impassable locations.

In VSI-Bench [31], the questions are instructions with a cloze (fill-in-the-blank) format, and the answer options correspond to possible actions during the navigation process. In OmniSpatial [33], each question includes the start point, end point, and orientation, while the options represent candidate navigation actions. OmniSpatial [33] also introduces a dynamic analysis setting, where question-answer pairs represent intermediate properties encountered during navigation, such as the number of doors passed.

Four types of elements are involved in this task, i.e., start point, exit point, impassable points, and passable points. To describe these elements for VLMs, two description modalities are adopted, i.e., visual and textual. In the visual description, elements are encoded in an image, with different colors indicating different element types. In the textual description, elements are encoded in ASCII format, with different characters representing different element types.

Mobile Manipulation. In VSR, mobile manipulation tasks ask agents to select the correct option that combines navigation with environment interaction, guided by the provided goals. A representative benchmark for this task is SpaCe-10 [170], which focuses on room tidying tasks and formulates each choice to reflect aspects of the task flow, including navigation paths, goals, goal attributes, and the corresponding actions to be executed.

Maze navigation problems are often formulated as multiple-choice questions. Depending on the input modality, maze navigation planning can be categorized into three settings: (1) text only, where mazes are represented in ASCII code [167], (2) visual-only, where mazes are depicted as colored images [154], [167], [168], and (3) text-visual, where both ASCII codes and colored images are provided [167].

#### 5 EXPERIMENT

In this chapter, we first introduce the process of constructing the test data in the Section 5.1, including data collection, sampling criteria, and organization methods. Then, we use this test data to evaluate existing models and provide analysis and discussion based on the results in the Section 5.2.

- 4.3.3 Embodied planning Embodied planning expects agents to leverage multi-modal data to perceive the surrounding environment, generate plans, and take actions in an autonomous manner. General embodied planning tasks are designed around completing a concrete goal. For example, in vision language navigation [169], agents are required to reach a specific location following given instructions, with metrics like success rate and average distance to goal used to measure planning accuracy. By contrast, instead of physically navigating to a goal, embodied planning tasks in VSR require VLMs to solve multi-choice questions, with each option corresponding to possible actions. Depending on the action space, embodied planning in VSR includes vision-and-language navigation and mobile manipulation, both formulated as multi-choice questions. Vision-and-Language Navigation Vision-and-Language Navigation (VLN) in VSR requires agents to answer multichoice questions while navigating through the environment.

###### 5.1 Benchmark Construction

The current benchmarks for VSR exhibit several issues, such as a lack of systematic and comprehensive task design, the presence of low-quality samples, a lack of human-annotated ground truth for some samples, and insufficient task rationale. To address these challenges, we survey existing opensource VSR benchmarks, aiming to integrate high-quality and diverse data. We ultimately develop SIBench, an evaluation benchmark that includes 23 task settings spanning 3 cognitive levels.

Quality. In our data quality assurance process, we prioritize datasets with manual annotations. For instance, when facing with a choice between manually annotated data and modelgenerated data for the same task, we consistently opt for the former. Only in the absence of manually annotated test data

[Figure 124]

- Fig. 7: Statistical data of SIBench. SIBench comprises VSR tasks across three cognitive levels, with a total of 8.8K data samples and 23 task settings.

do we consider using semi-automatically annotated data that has undergone human review. Additionally, we filter out “images” composed of emoticons and text to ensure that spatial information is derived from genuine visual input.

where yˆ is the model’s prediction, y is the ground truth, θ is a confidence threshold from the set C = {0.5,0.55,...,0.95}.

To streamline the evaluation process, we construct a mini version named SIBench-mini comprising 40 randomly selected samples from each task setting for evaluating Gemini2.5-Pro [21], Doubao-Seed-1.6-Vision [176], GLM4.5-V-106BA12B [22], and GPT-5 [20]. This subset is as comprehensive as the full benchmark in terms of task settings, but at a much lower testing cost.

Diversity. To construct a comprehensive evaluation dataset, we investigate nearly 20 open-source benchmarks, which we consolidate into 23 high-level task categories (see Fig. 1 for benchmark statistics). These categories span three cognitive levels: basic perception, spatial understanding, and planning. For each task category, we strive to enhance the diversity of the test data. For example, for object size estimation, the data is sourced from three different benchmarks: SPHERE-VLM [141], VSI-Bench [31], and STI-Bench [150]. Our test data includes three types of input formats: single images, multi-view images, and video data. The questions are presented in three formats: multiple-choice, true/false, and video data. Details regarding the SIBench are available in Fig. 1 and Fig. 7.

5.2.2 Main Results.

Comparison of Models. The experimental results indicate that leading proprietary models establish the highest tier of performance (see Table 1). Specifically, GPT-5 [20] achieves the best performance across the three levels of visual-spatial reasoning included in SIBench and SIBench-mini. Secondly, while a performance gap persists between open-source models and their leading proprietary counterparts, the former exhibit strong competitiveness against lightweight proprietary models. Although representative open-source models, such as InternVL-3.5-38B [174] (overall score: 0.5252) and Qwen2.5-VL-72B [171] (overall score: 0.5114), lag behind the leading models, their overall performance still outperforms lightweight models like GPT-4o-mini (0.4278) and Gemini2.5-Flash [21] (0.4389). Furthermore, we observe that GPT-5 and Gemini-2.5-Pro holds a significant lead in the dimensions of spatial understanding and planning, which may imply that these premier models possess superior abstract reasoning capabilities. On SIBench-mini, similar patterns are observed. Specifically, as shown in Table 2, the proprietary model GPT-5 [20] achieves the best performance, followed by Gemini-2.5-Pro and Doubao-Seed-1.6-Vision [176]. In contrast, other open-source models, such as GLM4.5-V106B-A12B [22] and InternVL-3.5-38B [11], still exhibit a noticeable performance gap.

###### 5.2 Evaluation

- 5.2.1 Setup. We perform a detailed comparative analysis of state-ofthe-art proprietary and open-source models. Our evaluation of proprietary models includes GPT-5 [20], Gemini2.5-Pro [21], Gemini-2.5-Flash [21] and GPT-4o-mini [172]. The primary open-source models under consideration are Qwen2.5-VL-72B [171], LLaVA-OneVision-72B [18] and InternVL-3-78B [173]. Consistency is maintained throughout our experimental design. All models process images (both single and multi-view) at an identical resolution. For video inputs, we apply a standard procedure of uniformly sampling 30 frames. Furthermore, we utilize specific prompt templates for each QA category; for example, multiplechoice questions are prefaced with ”Select from the following choices,” and numerical questions are prompted with ”Answer using a number.”

By default, we do not employ Chain-of-Thought (CoT) prompting. To handle any non-formatted outputs, we utilize Gemini-1.5-Flash [21] for post-processing to standardize the answers. The evaluation metric depends on the question type. For True/False (TF) and Multiple-Choice Questions (MCQ), we directly compare the post-processed predictions against the ground truth. For questions requiring a numerical answer, we adopt the Mean Relative Accuracy (MRA) metric from VSI-Bench [31], which is calculated as follows:

Comparison of Various Settings. Our evaluation reveals a stark contrast in the visual spatial capabilities of current models (see Table 3, Table 4 and Table 5). They exhibit high proficiency in fundamental perceptual tasks, such as identifying object existence, understanding occlusion, and judging qualitative spatial relationships, indicating strong static feature extraction. However, this proficiency does not extend to higher-level cognitive reasoning. Fig. 1 and Fig. 8 highlight four core deficiencies in the models, which are substantiated by detailed experimental data in the Appendix (see Table 4 and Table 5). First, the models exhibit

1 |yˆ − y| y

1 10 θ∈C

< 1 − θ , (4)

MRA =

- TABLE 1: Performance Evaluation of Different Models on SIBench. Bold is the best, and underlining is the second best.

Models Rank Overall Basic Perception Spatial Understanding Planning Random Choices - 0.3091 0.4182 0.2807 0.25 Qwen2.5-VL-7B [171] 9 0.4712 0.5280 0.4624 0.3442 Qwen2.5-VL-72B [171] 4 0.5447 0.5793 0.5430 0.4161 LLaVA-OneVision-7B [18] 8 0.4850 0.5640 0.4708 0.3355 LLaVA-OneVision-72B [18] 7 0.5207 0.6086 0.5022 0.3922 GPT-4o-mini [172] 11 0.4278 0.5505 0.3981 0.3050 Gemini-2.5-Flash [21] 10 0.4389 0.5422 0.3942 0.6100

- InternVL-2.5-78B-MPO [171] 5 0.5338 0.5886 0.5165 0.5338 InternVL-3-78B [173] 3 0.5481 0.6141 0.5344 0.4488
- InternVL-3.5-38B [174] 6 0.5252 0.5726 0.5134 0.4815 Gemini-2.5-Pro [21] 2 0.5883 0.6425 0.5559 0.8017 GPT-5 [20] 1 0.6341 0.6934 0.6158 0.6296

- TABLE 2: Performance Evaluation of Different Models on the SIBench-mini. Bold is the best, and underlining is the second best.

Models Rank Overall Basic Perception Spatial Understanding Planning LLaVA-OneVision-72B [18] 6 0.5252 0.6136 0.5065 0.3968 Qwen2.5-VL-72B [171] 7 0.5168 0.5930 0.4958 0.4743 InternVL-3.5-38B [11] 5 0.5355 0.6113 0.5089 0.4878 GLM4.5-V-106B-A12B [175] 4 0.5822 0.6936 0.5404 0.5125 Doubao-Seed-1.6-Vision [176] 3 0.6216 0.6963 0.5922 0.65 Gemini-2.5-Pro [21] 2 0.6295 0.7317 0.5827 0.675 GPT-5 [20] 1 0.6906 0.7248 0.6487 0.775

Distance Camera Pose

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

The images are taken

The cable duct (red point) is located at a depth of 3.3 meters. What is the

continuously from a

first-person perspective. In which direction is the camera rotating?

depth of the spray bottle (blue point)?

Calculate or judge based on the 3D center points of these objects.

The unit is meter. Type in exactly one number as your reply.

Options: A: Up, B: Down, C: Left, D: Right GPT-5: C ANSWER: A

GPT-5: 2.6 ANSWER: 5.5

Velocity Spatial Imagination

Option A Option B Option C Option D

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

What is the average speed

of the camera between 10s and 18s? (unit: m/s)

This image shows a 3D polycube shape. Which of the

(A) 0.07m/s (B) 0.09m/s (C) 0.02m/s (D) 0.06m/s (E) 0.03m/s GPT-5: D ANSWER: E

options is simply the original shape in a rotated orientation?

GPT-5: C ANSWER: A

- Fig. 8: Failure Cases in Spatial Reasoning Tasks, including quantitative reasoning (upper-left), multi-view reasoning (upper-right), non-static logical inference (bottom-left), and spatial imagination (bottom-right).

geometric constraints in training. Third, their performance degrades significantly on temporal tasks. This fragility in handling non-static inputs suggests their underlying logical inference is unreliable (see the bottom-left part of Fig. 8). A plausible reason is that these models primarily process frames independently, failing to capture coherent temporal dynamics. Finally, and perhaps most critically, the models exhibit a near-total absence of spatial imagination. All mod-

poor quantitative reasoning, often failing to accurately estimate physical quantities such as distance and size (see the upper-left part of Fig. 8). This limitation likely arises from their reliance on coarse visual heuristics rather than precise metric representations. Second, they struggle with multiview reasoning, particularly in tasks requiring the inference of camera pose transformations (see the upper-right part of Fig. 8). This weakness may stem from the lack of explicit

- TABLE 3: Performance evaluation of different models on visual spatial reasoning tasks. Bold is the best, and underlining is the second best.

Models Qwen2.5-VL-72B [171] LLaVA-OneVision-72B [18] GPT-5 [20] Gemini-2.5-Pro [21]

Settings

Reach Prediction 0.6750 0.6000 0.5750 0.5500 Height 0.6000 0.6466 0.6800 0.7133 Existence 1.0000 0.9500 0.9250 0.9000 Occlusion 0.5580 0.6660 0.6460 0.7300 Object Shape 0.3538 0.3077 0.5154 0.4154 Counting 0.5504 0.5084 0.7408 0.4664 Object Size Estimation 0.6085 0.6045 0.7645 0.5968

Basic Perception

Spatial Compatibility 0.6028 0.5841 0.5607 0.7103 Coordinate Conversion 0.6513 0.5950 0.7788 0.6900 Trajectory Description 0.6026 0.5513 0.3205 0.3333 Geometric Reasoning 0.2421 0.2579 0.5278 0.3730 Spatial Imagination 0.2900 0.2650 0.4575 0.2700 Spatial Grid 0.7875 0.5175 0.7900 0.9975 Temporal Appearance Order 0.5455 0.3636 0.2957 0.5455 Multi-View Reasoning 0.4072 0.4316 0.5854 0.4687 Situational QA 0.5295 0.5128 0.6055 0.5474 Velocity Acceleration 0.4605 0.4403 0.5746 0.4034 Relative Distance 0.5675 0.5037 0.6828 0.5459 Camera Pose 0.4738 0.3169 0.5000 0.4041 Spatial Relation 0.7978 0.7870 0.6929 0.6660 Object Localization 0.7134 0.7570 0.7727 0.7041

Spatial Understanding

Maze Navigation 0.4350 0.3975 0.7725 0.8625 Route Planning 0.2881 0.3559 0.3351 0.3898

Planning

Overall 0.5447 0.5207 0.6341 0.5883

els failed on tasks requiring the creation and manipulation of a mental model (see the bottom-right part of Fig. 8). This exposes a fundamental limitation: they excel at interpreting visible information but cannot conceive of what is absent, functioning as observers rather than reasoners. In short, while models have become adept at ”seeing,” their ability to perform cognitive ”thinking”, particularly spatial imagination and precise inference, remains nascent.

#### 6 DISCUSSION

In this section, we first summarize the challenges that VLMs face in Visual Spatial Reasoning tasks in Section 6.1, and then propose several potential solutions in Section 6.2.

###### 6.1 Challenges

While VLMs have advanced significantly in multimodal understanding, they face critical challenges in complex VSR. These challenges constrain their performance on benchmarks and impede real-world deployment in fields like robotics and autonomous driving. Based on the evaluation of existing models, we summarize these challenges into the following four main aspects (see Fig. 9).

Limited Robustness of Foundational Perception. Despite the strong overall perceptual capabilities of VLMs, their foundational perception capabilities often lack robustness, particularly in complex or atypical scenarios. These uncertainties at the perceptual level can directly undermine the

reliability of any subsequent spatial reasoning. For example, Counting remains a significant challenge, especially in dense or occluded scenes, while precise Object Localization and fine-grained Shape Recognition are also far from perfect. When deviations occur at these fundamental perceptual stages, such as misidentifying the number or location of objects, any reasoning based on this flawed information becomes invalid. This ”error accumulation” effect highlights that even if a model possesses the potential for advanced reasoning, its performance is ultimately constrained by the bottlenecks within its foundational perception capabilities.

Lack of Precise and Quantitative Capabilities. VLMs exhibit inherent shortcomings in precision and quantification, a limitation that may stem from their reliance on learning from human language, which is often ambiguous when describing the physical world. Consequently, while these models excel at processing qualitative spatial relationships, such as ”A is to the left of B,” they exhibit significant deficiencies in tasks requiring quantitative geometric computation. This deficiency is empirically validated by benchmark results. As illustrated in Fig. 1, existing models consistently underperform on tasks that demand precise numerical regression, including Coordinate Conversion, Distance Prediction, and Object Size Estimation. This difficulty in mapping visual features to precise physical coordinates or metric units severely restricts the application potential of VLMs in high-precision scenarios, such as robotic grasping and the accurate placement of content in Augmented Reality (AR).

###### Limited Foundational Perception

###### Lack of Quantitative Capabilities

###### Deficiency in Spatial Imagination

###### Insufficiency in Dynamics

[Figure 137]

[Figure 138]

|[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]|
|---|

|[Figure 143]<br><br>[Figure 144]|
|---|

[Figure 145]

[Figure 146]

[Figure 147]

Estimating Object Speed

[Figure 148]

[Figure 149]

Describing Motion Path

[Figure 150]

###### Identifying Order of Appearance

[Figure 151]

[Figure 152]

[Figure 153]

Banana Counting: 2 Banana Localization: Red box

Banana is on the left Banana–cup Distance Banana Length

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Camera Pose: estimate the camera pose change

Pathfinding: How to get from bedroom to bathroom?

[Figure 158]

[Figure 159]

Banana Shape: Cuboid

- Fig. 9: Challenges for VLMs in VSR tasks. From left to right, examples illustrate VLM deficiencies in foundational perception, quantitative reasoning, spatial imagination, and dynamic reasoning. Detailed examples can be found in the supplementary materials.

relational reasoning in VSR can train a model’s understanding of relative positions; maze navigation or route planning tasks can foster a global spatial-topological cognition; and simulated robotic manipulation tasks demand more finegrained localization and judgement of physical affordance. An ideal training strategy should therefore combine diverse task settings with high-quality structured annotations. Leveraging large-scale synthetic data with perfect ground truth from simulation engines like Blender and CARLA is also an essential component of this approach.

Deficiency in Spatial Imagination and 3D Reconstruction. Primarily relying on 2D images as input, VLMs generally lack the ability to ”imagine” or reconstruct a complete 3D world from limited perspectives. This deficiency hinders their ability to comprehend the physical properties of objects and the spatial relationships involving occluded parts. Evaluation results intuitively reflect this issue. For instance, in Situational Question and Answering tasks, models struggle to infer the complete geometry of unseen objects. Similarly, in tasks that require a mental simulation of spatial connectivity, such as Route Planning and Maze Navigation, VLMs without a robust 3D world model also perform poorly. This absence of capability prevents the models from constructing a coherent and global scene representation, which is a major obstacle to high-level spatial cognition.

Incorporating 3D-Aware and Fine-Grained Perception Tasks in Pre-training Stage. Incorporating 3D-aware and fine-grained perception tasks into the pre-training phase is crucial for endowing VLMs with robust spatial imagination and perceptual robustness. To facilitate a model’s understanding of the 3D world, large-scale synthetic 3D datasets (e.g., ShapeNet, Objaverse) can be utilized for tasks such as novel view synthesis or 3D shape reconstruction. These tasks compel the model to learn an internal, viewinvariant object representation. Concurrently, to enhance the robustness of foundational perception, a multi-task learning paradigm can be employed where the visual encoder learns auxiliary tasks like depth estimation, surface normal prediction, and instance segmentation alongside its primary vision-language alignment objective. This multi-task setup acts as a powerful regularizer, compelling the visual backbone to learn features that are generalizable across various geometries and semantics, thereby providing a more reliable input for higher-level reasoning and effectively mitigating the ”error accumulation” effect.

Insufficiency in Dynamic-Temporal and Cross-View Reasoning. The real world is inherently dynamic and variable, necessitating that an intelligent agent integrate information across time and viewpoints to form a coherent spatial cognition. However, current VLMs exhibit notable insufficiencies in this area. The Fig. 1 highlights the models’ weaknesses in tasks such as Multi-View Reasoning, Velocity & Acceleration, and Trajectory Description. This suggests that VLMs struggle both to track the spatio-temporal state of objects effectively and to align and fuse information from disparate camera viewpoints. This deficiency in processing dynamic and multi-view information poses a critical barrier to the advancement of VLMs into high-level applications requiring continuous interaction with dynamic environments, such as robotics and autonomous driving.

Towards Advanced Unified Spatiotemporal Architectures. Most existing VLM architectures are designed to process static, discrete images, creating a fundamental disconnect with the continuous, dynamic, and multi-view perception of an intelligent agent in the real world. Therefore, a key direction for architecture evolution is the development of a unified spatiotemporal architecture oriented towards Embodied AI. Such an architecture would no longer treat space and time as separate dimensions but would instead perceive the world as a continuous four-dimensional spacetime stream. Its core objective is to enable the model to integrate

###### 6.2 Potential Solutions

Constructing Higher-Quality and More Diverse Training Data. A foundational solution to the current challenges for VLMs is the construction of higher-quality and more diverse training data. The core of this strategy lies in increasing data diversity, not only in terms of sources (e.g., real-world images, videos, synthetic data) but more critically, in the breadth of task settings, as different tasks directly correspond to distinct foundational spatial abilities. For instance,

information from different times and viewpoints, forming a dynamic, consistent, and ego-centric understanding of the scene. This path aims to transcend the model’s limitation as a passive ’bystander’ and empower it with the ability to perceive and predict the world as an active ’agent.’ This evolution is an essential pathway toward advanced applications such as robotics and autonomous driving.

###### 6.3 Applications

As this review has elaborated, VSR represents not only a theoretical challenge but also the critical capability for translating digital intelligence into real-world action. The profound impact of VSR is most vividly exhibit across different frontier domains, such as autonomous driving, embodied robotics, and augmented reality.

Embodied Intelligence. For embodied AI and robotics, VSR acts as the crucial bridge connecting abstract instructions to concrete physical actions [31], [177]. It enables agents to ground natural language commands (e.g.“place the cup in the center of the table”) into precise 3D spatial coordinates and action sequences [169], thereby facilitating complex navigation [31], [33] and manipulation tasks [170]. Although current models still exhibit deficiencies in geometric and physical commonsense reasoning, the rise of Vision-Language-Action (VLA) models [178], [179], particularly those incorporating internal reasoning mechanisms like the “visual chain-of-thought” [180], is advancing robotic intelligence toward a ”think before you act” paradigm. This approach effectively decomposes complex tasks and improves the success rate of planning and execution.

Autonomous Driving. In the domain of autonomous driving, VSR serves as the cognitive cornerstone for safe navigation. It elevates a vehicle’s perceptual capabilities from simple object recognition to deep understanding of the 3D structure of dynamic traffic scenes [181]–[183], the relative relationships between objects [184], and their motion trajectories [185]. This capability is a prerequisite for all advanced planning and decision-making. A central challenge in this field is bridging the gap between the 2D imagebased reasoning abilities developed during model training and the need for precise, quantitative inference in the 3D physical world [186]. To this end, the field is shifting from imitation learning towards explicit reasoning, developing end-to-end models like EMMA [187] and neuro-symbolic hybrid frameworks such as Logic-RAG [188], to build more robust and interpretable driving intelligence.

#### 7 CONCLUSION

In this paper, we presents a systematic and comprehensive investigation into the domain of Visual Spatial Reasoning (VSR) for Vision-Language Models (VLMs). We begin by conducting a review of existing methodologies, analyzing them through input modalities, model architectures, training strategies, and inference mechanisms. Concurrently, we introduce a cognition-based hierarchical taxonomy that categorizes VSR tasks into three distinct levels: Basic Perception, Spatial Understanding, and Spatial Planning. Building upon this framework, we curate SIBench, a comprehensive benchmark that integrates nearly 20 open-source datasets

across 23 task settings to facilitate a holistic evaluation of VLM capabilities. Our experiments on state-of-the-art models using SIBench reveal a critical gap between perceptual and cognitive abilities. While models exhibit proficiency in basic perception, they exhibit significant deficiencies in higher-order reasoning, particularly in tasks requiring precise numerical estimation, multi-view reasoning, dynamic temporal understanding, and spatial imagination. These findings underscore the substantial challenges that remain on the path to achieving spatial intelligence in artificial agents.

#### REFERENCES

- [1] J. Piaget, Child’s Conception of Space: Selected Works vol 4. Routledge, 2013.
- [2] M.-B. Moser, D. C. Rowland, and E. I. Moser, “Place cells, grid cells, and memory,” Cold Spring Harbor perspectives in biology, vol. 7, no. 2, p. a021808, 2015.
- [3] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” NeurIPS, vol. 35, pp. 23716–23736, 2022.
- [4] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “Instructblip: Towards general-purpose visionlanguage models with instruction tuning,” NeurIPS, vol. 36, pp. 49250–49267, 2023.
- [5] Z. Liu, Y. Dong, J. Wang, Z. Liu, W. Hu, J. Lu, and Y. Rao, “Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment,” arXiv, 2025.
- [6] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li, “End-toend autonomous driving: Challenges and frontiers,” PAMI, 2024.
- [7] B. Jin and H. Liu, “Adapt: Action-aware driving caption transformer,” in CAAI, 2023, pp. 473–477.
- [8] M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang et al., “End to end learning for self-driving cars,” arXiv, 2016.
- [9] A. Kendall, J. Hawke, D. Janz, P. Mazur, D. Reda, J.-M. Allen, V.-D. Lam, A. Bewley, and A. Shah, “Learning to drive in a day,” in ICRA, 2019, pp. 8248–8254.
- [10] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, “Transfuser: Imitation with transformer-based sensor fusion for autonomous driving,” PAMI, vol. 45, no. 11, pp. 12878–12895, 2022.
- [11] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv, 2025.
- [12] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song et al., “Emerging properties in unified multimodal pretraining,” arXiv, 2025.
- [13] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv, 2025.
- [14] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation,” in ICML. PMLR, 2022, pp. 12888–12900.
- [15] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” NeurIPS, vol. 35, pp. 23716–23736, 2022.
- [16] D. Guo, F. Wu, F. Zhu, F. Leng, G. Shi, H. Chen, H. Fan, J. Wang, J. Jiang, J. Wang et al., “Seed1. 5-vl technical report,” arXiv, 2025.
- [17] Y. Jin, J. Li, J. Zhang, J. Hu, Z. Gan, X. Tan, Y. Liu, Y. Wang, C. Wang, and L. Ma, “Llava-vsd: Large language-and-vision assistant for visual spatial description,” in ACMMM, 2024, pp. 11420–11425.
- [18] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “Llava-onevision: Easy visual task transfer,” arXiv, 2024.
- [19] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese et al., “Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset,” arXiv, 2025.

- [20] OpenAI., “Gpt-5 system card,” Technical report, OpenAI, August 2025., Accessed: 2025-08-10.
- [21] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., “Gemini: a family of highly capable multimodal models,” arXiv, 2023.
- [22] A. Zeng, X. Lv, Q. Zheng, Z. Hou, B. Chen, C. Xie, C. Wang, D. Yin, H. Zeng, J. Zhang et al., “Glm-4.5: Agentic, reasoning, and coding (arc) foundation models,” arXiv, 2025.
- [23] J. Kuang, Y. Shen, J. Xie, H. Luo, Z. Xu, R. Li, Y. Li, X. Cheng, X. Lin, and Y. Han, “Natural language understanding and inference with mllm in visual question answering: A survey,” ACM Computing Surveys, vol. 57, no. 8, pp. 1–36, 2025.
- [24] A. Albalak, D. Phung, N. Lile, R. Rafailov, K. Gandhi, L. Castricato, A. Singh, C. Blagden, V. Xiang, D. Mahan, and N. Haber, “Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models,” 2025.
- [25] F. Pennino, B. Raimondi, M. Rondelli, A. Gurioli, and M. Gabbrielli, “From reasoning to code: Grpo optimization for underrepresented languages,” arXiv, 2025.
- [26] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Sadigh, L. Guibas, and F. Xia, “Spatialvlm: Endowing vision-language models with spatial reasoning capabilities,” in CVPR, 2024, pp. 14455–14465.
- [27] A.-C. Cheng, H. Yin, Y. Fu, Q. Guo, R. Yang, J. Kautz, X. Wang, and S. Liu, “Spatialrgpt: Grounded spatial reasoning in visionlanguage models,” NeurIPS, vol. 37, pp. 135062–135093, 2024.
- [28] E. Zhou, J. An, C. Chi, Y. Han, S. Rong, C. Zhang, P. Wang, Z. Wang, T. Huang, L. Sheng et al., “Roborefer: Towards spatial referring with reasoning in vision-language models for robotics,” arXiv, 2025.
- [29] H. Shao, S. Qian, H. Xiao, G. Song, Z. Zong, L. Wang, Y. Liu, and H. Li, “Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning,” NeurIPS, vol. 37, pp. 8612–8642, 2024.
- [30] K. Ouyang, Y. Liu, H. Wu, Y. Liu, H. Zhou, J. Zhou, F. Meng, and X. Sun, “Spacer: Reinforcing mllms in video spatial reasoning,” arXiv, 2025.
- [31] J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie, “Thinking in space: How multimodal large language models see, remember, and recall spaces,” in CVPR, 2025, pp. 10632–10643.
- [32] F. Shiri, X.-Y. Guo, M. G. Far, X. Yu, G. Haffari, and Y.-F. Li, “An empirical analysis on spatial reasoning capabilities of large multimodal models,” arXiv, 2024.
- [33] M. Jia, Z. Qi, S. Zhang, W. Zhang, X. Yu, J. He, H. Wang, and L. Yi, “Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models,” arXiv, 2025.
- [34] Z. Wang, S. Zhou, S. He, H. Huang, L. Yang, Z. Zhang, X. Cheng, S. Ji, T. Jin, H. Zhao et al., “Spatialclip: Learning 3d-aware image representations from spatially discriminative language,” in CVPR, 2025, pp. 29656–29666.
- [35] Z. Yang, L. Li, K. Lin, J. Wang, C.-C. Lin, Z. Liu, and L. Wang, “The dawn of lmms: Preliminary explorations with gpt-4v (ision),” arXiv, 2023.
- [36] C. Wu, P. Zheng, R. Yan, S. Xiao, X. Luo, Y. Wang, W. Li, X. Jiang, Y. Liu, J. Zhou et al., “Omnigen2: Exploration to advanced multimodal generation,” arXiv, 2025.
- [37] K. K. Team, B. Yang, B. Wen, C. Liu, C. Chu, C. Song, C. Rao, C. Yi, D. Li, D. Zang et al., “Kwai keye-vl technical report,” arXiv, 2025.
- [38] B. F. Labs, S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, S. Kulal, K. Lacey, Y. Levi, C. Li, D. Lorenz, J. Muller,¨ D. Podell, R. Rombach, H. Saini, A. Sauer, and L. Smith, “Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,” 2025.
- [39] Z. Li, J. Zhang, Q. Lin, J. Xiong, Y. Long, X. Deng, Y. Zhang, X. Liu, M. Huang, Z. Xiao et al., “Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding,” arXiv, 2024.
- [40] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” NeurIPS, vol. 34, pp. 8780–8794, 2021.
- [41] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv, 2020.
- [42] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” NeurIPS, vol. 33, pp. 6840–6851, 2020.
- [43] O. Katzir, O. Patashnik, D. Cohen-Or, and D. Lischinski, “Noisefree score distillation,” arXiv, 2023.

- [44] M. Chu, Z. Zheng, W. Ji, T. Wang, and T.-S. Chua, “Towards natural language-guided drones: Geotext-1652 benchmark with spatial relation matching,” in ECCV, 2024, pp. 213–231.
- [45] H. Ju, S. Huang, S. Liu, and Z. Zheng, “Video2bev: Transforming drone videos to bevs for video-based geo-localization,” ICCV, 2025.
- [46] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning, 2023, pp. 2165–2183.
- [47] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter et al., “pi 0: A visionlanguage-action flow model for general robot control,” arXiv, 2024.

- [48] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., “Openvla: An open-source vision-language-action model,” arXiv, 2024.
- [49] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, “Universal manipulation interface: Inthe-wild robot teaching without in-the-wild robots,” arXiv, 2024.
- [50] D. Gao, S. Cai, H. Zhou, H. Wang, I. Soltani, and J. Zhang, “Cardreamer: Open-source learning platform for world model based autonomous driving,” IEEE Internet of Things Journal, 2024.
- [51] Y. Yang, J. Mei, Y. Ma, S. Du, W. Chen, Y. Qian, Y. Feng, and Y. Liu, “Driving in the occupancy world: Vision-centric 4d occupancy forecasting and planning via world models for autonomous driving,” in AAAI, vol. 39, no. 9, 2025, pp. 9327–9335.
- [52] W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu, “Occworld: Learning a 3d occupancy world model for autonomous driving,” in ECCV, 2024, pp. 55–72.
- [53] M. Pan, J. Zhang, T. Wu, Y. Zhao, W. Gao, and H. Dong, “Omnimanip: Towards general robotic manipulation via object-centric interaction primitives as spatial constraints,” in CVPR, 2025, pp. 17359–17369.
- [54] X. Ding, J. Han, H. Xu, X. Liang, W. Zhang, and X. Li, “Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models,” in CVPR, 2024, pp. 13668–13677.
- [55] R. Li, T. Fischer, M. Segu, M. Pollefeys, L. Van Gool, and F. Tombari, “Know your neighbors: Improving single-view reconstruction via spatial vision-language reasoning,” in CVPR, 2024, pp. 9848–9858.
- [56] M. I. H. Rizvi, X. Zhu, and I. Gurevych, “Sparc and sparp: Spatial reasoning characterization and path generation for understanding spatial reasoning capability of large language models,” arXiv, 2024.
- [57] W. Zhou, M. Tao, C. Zhao, H. Guo, H. Dong, M. Tang, and J. Wang, “Physvlm: Enabling visual language models to understand robotic physical reachability,” in CVPR, 2025, pp. 6940– 6949.
- [58] Y. Yang, H. Yang, J. Zhou, P. Chen, H. Zhang, Y. Du, and C. Gan, “3d-mem: 3d scene memory for embodied exploration and reasoning,” in CVPR, 2025, pp. 17294–17303.
- [59] Z. Qi, R. Dong, S. Zhang, H. Geng, C. Han, Z. Ge, L. Yi, and K. Ma, “Shapellm: Universal 3d object understanding for embodied interaction,” in ECCV, 2024, pp. 214–238.
- [60] F. Li, D. C. Hogg, and A. G. Cohn, “Advancing spatial reasoning in large language models: An in-depth evaluation and enhancement using the stepgame benchmark,” in AAAI, vol. 38, no. 17, 2024, pp. 18500–18507.
- [61] S. Mouselinos, H. Michalewski, and M. Malinowski, “Beyond lines and circles: Unveiling the geometric reasoning gap in large language models,” arXiv, 2024.
- [62] R. Wang, K. Sun, and J. Kuhn, “Dspy-based neural-symbolic pipeline to enhance spatial reasoning in llms,” arXiv, 2024.
- [63] C. Li, W. Wu, H. Zhang, Y. Xia, S. Mao, L. Dong, I. Vuli´c, and F. Wei, “Imagine while reasoning in space: Multimodal visualization-of-thought,” arXiv, 2025.
- [64] D. Bogdoll, Y. Yang, and J. M. Z¨ollner, “Muvo: A multimodal generative world model for autonomous driving with geometric representations,” arXiv, vol. 3, no. 4, 2023.
- [65] S. Cai, E. R. Chan, S. Peng, M. Shahbazi, A. Obukhov, L. Van Gool, and G. Wetzstein, “Diffdreamer: Towards consistent unsupervised single-view scene extrapolation with conditional diffusion models,” in ICCV, 2023, pp. 2139–2150.
- [66] D. Yang, L. Hu, Y. Tian, Z. Li, C. Kelly, B. Yang, C. Yang, and Y. Zou, “Worldgpt: a sora-inspired video ai agent as rich world models from text and image inputs,” arXiv, 2024.

- [67] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps et al., “Genie: Generative interactive environments,” in Proc. ACM Int. Conf. Mach. Learn., 2024.
- [68] H. Liu, W. Yan, M. Zaharia, and P. Abbeel, “World model on million-length video and language with blockwise ringattention,” arXiv, 2024.
- [69] S. Yin, C. Wu, H. Yang, J. Wang, X. Wang, M. Ni, Z. Yang, L. Li, S. Liu, F. Yang et al., “Nuwa-xl: Diffusion over diffusion for extremely long video generation,” arXiv, 2023.
- [70] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang et al., “Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving,” in CVPR, 2025, pp. 12037–12047.
- [71] T. Li, H. Wang, X. Li, W. Liao, T. He, and P. Peng, “Generative planning with 3d-vision language pre-training for end-to-end autonomous driving,” in AAAI, vol. 39, no. 5, 2025, pp. 4950– 4958.
- [72] X. Jiang, Y. Ma, P. Li, L. Xu, X. Wen, K. Zhan, Z. Xia, P. Jia, X. Lang, and S. Sun, “Transdiffuser: End-to-end trajectory generation with decorrelated multi-modal representation for autonomous driving,” arXiv, 2025.
- [73] Y. Tang, Z. Xu, Z. Meng, and E. Cheng, “Hip-ad: Hierarchical and multi-granularity planning with deformable attention for autonomous driving in a single decoder,” arXiv, 2025.
- [74] Z. Li, X. Wu, H. Du, H. Nghiem, and G. Shi, “Benchmark evaluations, applications, and challenges of large vision language models: A survey,” arXiv, vol. 1, 2025.
- [75] X. Ma, Y. Bhalgat, B. Smart, S. Chen, X. Li, J. Ding, J. Gu, D. Z. Chen, S. Peng, J.-W. Bian et al., “When llms step into the 3d world: A survey and meta-analysis of 3d tasks via multi-modal large language models,” arXiv, 2024.
- [76] J. Zha, Y. Fan, X. Yang, C. Gao, and X. Chen, “How to enable llm with 3d capacity? a survey of spatial reasoning in llm,” arXiv, 2025.
- [77] J. Qi, J. Liu, H. Tang, and Z. Zhu, “Beyond semantics: Rediscovering spatial awareness in vision-language models,” arXiv, 2025.
- [78] S. Chen, T. Zhu, R. Zhou, J. Zhang, S. Gao, J. C. Niebles, M. Geva, J. He, J. Wu, and M. Li, “Why is spatial reasoning hard for vlms? an attention mechanism perspective on focus areas,” arXiv, 2025.
- [79] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” PAMI, vol. 46, no. 8, pp. 5625–5644, 2024.
- [80] R. Sapkota, Y. Cao, K. I. Roumeliotis, and M. Karkee, “Visionlanguage-action models: Concepts, progress, applications and challenges,” arXiv, 2025.
- [81] S. Xie, L. Kong, Y. Dong, C. Sima, W. Zhang, Q. A. Chen, Z. Liu, and L. Pan, “Are vlms ready for autonomous driving? an empirical study from the reliability, data, and metric perspectives,” arXiv, 2025.
- [82] R. Guo, J. Wei, L. Sun, B. Yu, G. Chang, D. Liu, S. Zhang, Z. Yao, M. Xu, and L. Bu, “A survey on advancements in image– text multimodal models: From general techniques to biomedical implementations,” Computers in biology and medicine, vol. 178, p. 108709, 2024.
- [83] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Sadigh, L. Guibas, and F. Xia, “Spatialvlm: Endowing vision-language models with spatial reasoning capabilities,” in CVPR, 2024, pp. 14455–14465.
- [84] B. Liu, Y. Dong, Y. Wang, Z. Ma, Y. Tang, L. Tang, Y. Rao, W.C. Ma, and R. Krishna, “Coarse correspondences boost spatialtemporal reasoning in multimodal language model,” in CVPR, 2025, pp. 3783–3792.
- [85] C. Wen, D. Jayaraman, and Y. Gao, “Can transformers capture spatial relations between objects?” arXiv, 2024.
- [86] Y. Tang, A. Qu, Z. Wang, D. Zhuang, Z. Wu, W. Ma, S. Wang, Y. Zheng, Z. Zhao, and J. Zhao, “Sparkle: Mastering basic spatial capabilities in vision language models elicits generalization to composite spatial reasoning,” arXiv, 2024.
- [87] X. Liu, D. Yin, Y. Feng, and D. Zhao, “Things not written in text: Exploring spatial commonsense from visual signals,” arXiv, 2022.
- [88] M. Ogezi and F. Shi, “Spare: Enhancing spatial reasoning in vision-language models with synthetic data,” arXiv, 2025.
- [89] Y. Wang, S.-Y. Chen, Z. Zhou, S. Li, H. Li, W. Zhou, and H. Li, “Root: Vlm based system for indoor scene understanding and beyond,” arXiv, 2024.
- [90] W. Yuan, J. Duan, V. Blukis, W. Pumacay, R. Krishna, A. Murali, A. Mousavian, and D. Fox, “Robopoint: A vision-language model for spatial affordance prediction for robotics,” arXiv, 2024.

- [91] Z. Liu, Y. Wang, S. Zheng, T. Pan, L. Liang, Y. Fu, and X. Xue, “Reasongrounder: Lvlm-guided hierarchical feature splatting for open-vocabulary 3d visual grounding and reasoning,” in CVPR, 2025, pp. 3718–3727.
- [92] B. R. Team, M. Cao, H. Tan, Y. Ji, M. Lin, Z. Li, Z. Cao, P. Wang, E. Zhou, Y. Han et al., “Robobrain 2.0 technical report,” arXiv, 2025.
- [93] R. Xu, W. Wang, H. Tang, X. Chen, X. Wang, F.-J. Chu, D. Lin, M. Feiszli, and K. J. Liang, “Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models,” arXiv, 2025.
- [94] M. Xu, M. Wu, Y. Zhao, J. C. L. Li, and W. Ou, “Llava-spacesgg: Visual instruct tuning for open-vocabulary scene graph generation with enhanced spatial relations,” in WACV. IEEE, 2025, pp. 6362–6372.
- [95] J. Li, X. Nan, M. Lu, L. Du, and S. Zhang, “Proximity qa: Unleashing the power of multi-modal large language models for spatial proximity analysis,” arXiv, 2024.
- [96] B. Zhao, L. P. Dirac, and P. Varshavskaya, “Can vision language models learn from visual demonstrations of ambiguous spatial reasoning?” arXiv, 2024.
- [97] Z. Wang, S. Zhou, S. He, H. Huang, L. Yang, Z. Zhang, X. Cheng, S. Ji, T. Jin, H. Zhao et al., “Spatialclip: Learning 3d-aware image representations from spatially discriminative language,” in CVPR, 2025, pp. 29656–29666.
- [98] Y. Liu, M. Ma, X. Yu, P. Ding, H. Zhao, M. Sun, S. Huang, and D. Wang, “Ssr: Enhancing depth perception in vision-language models via rationale-guided spatial reasoning,” arXiv, 2025.
- [99] W. Cai, I. Ponomarenko, J. Yuan, X. Li, W. Yang, H. Dong, and B. Zhao, “Spatialbot: Precise spatial understanding with vision language models,” arXiv, 2024.
- [100] E. Daxberger, N. Wenzel, D. Griffiths, H. Gang, J. Lazarow, G. Kohavi, K. Kang, M. Eichner, Y. Yang, A. Dehghan et al., “Mmspatial: Exploring 3d spatial understanding in multimodal llms,” arXiv, 2025.
- [101] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” NeurIPS, vol. 36, pp. 34892–34916, 2023.
- [102] Z. Guo, R. Xu, Y. Yao, J. Cui, Z. Ni, C. Ge, T.-S. Chua, Z. Liu, and G. Huang, “Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images,” in ECCV, 2024, pp. 390–406.
- [103] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICLR. PmLR, 2021, pp. 8748–8763.
- [104] W. Ma, L. Ye, C. M. de Melo, A. Yuille, and J. Chen, “Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models,” in CVPR, 2025, pp. 17249–17260.
- [105] Q. Feng, “Towards visuospatial cognition via hierarchical fusion of visual experts,” arXiv, 2025.
- [106] K. He, X. Chen, S. Xie, Y. Li, P. Doll´ar, and R. Girshick, “Masked autoencoders are scalable vision learners,” in CVPR, 2022, pp. 16000–16009.
- [107] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv, 2023.
- [108] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in ICCV, 2023, pp. 4015–4026.
- [109] Z. Fan, J. Zhang, R. Li, J. Zhang, R. Chen, H. Hu, K. Wang, H. Qu, D. Wang, Z. Yan et al., “Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction,” arXiv, 2025.
- [110] Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa, “Continuous 3d perception model with persistent state,” arXiv, 2025.
- [111] D. Zheng, S. Huang, Y. Li, and L. Wang, “Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors,” arXiv, 2025.
- [112] D. Wu, F. Liu, Y.-H. Hung, and Y. Duan, “Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence,” arXiv, 2025.
- [113] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in CVPR, 2025, pp. 5294–5306.

- [114] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv, 2025.
- [115] X. Li, Z. Yan, D. Meng, L. Dong, X. Zeng, Y. He, Y. Wang, Y. Qiao, Y. Wang, and L. Wang, “Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning,” arXiv, 2025.
- [116] P. Wang and H. Ling, “Svqa-r1: Reinforcing spatial reasoning in mllms via view-consistent reward optimization,” arXiv, 2025.
- [117] B. Zhao, Z. Wang, J. Fang, C. Gao, F. Man, J. Cui, X. Wang, X. Chen, Y. Li, and W. Zhu, “Embodied-r: Collaborative framework for activating embodied spatial reasoning in foundation models via reinforcement learning,” arXiv, 2025.
- [118] A. Su, H. Wang, W. Ren, F. Lin, and W. Chen, “Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning,” arXiv, 2025.
- [119] W. Ma, Y.-C. Chou, Q. Liu, X. Wang, C. de Melo, J. Xie, and A. Yuille, “Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning,” arXiv, 2025.
- [120] W. Huang, P. Abbeel, D. Pathak, and I. Mordatch, “Language models as zero-shot planners: Extracting actionable knowledge for embodied agents,” in ICML, 2022, pp. 9118–9147.
- [121] D. Sur´ıs, S. Menon, and C. Vondrick, “Vipergpt: Visual inference via python execution for reasoning,” in ICCV, 2023, pp. 11888– 11898.
- [122] W. Wu, S. Mao, Y. Zhang, Y. Xia, L. Dong, L. Cui, and F. Wei, “Mind’s eye of llms: visualization-of-thought elicits spatial reasoning in large language models,” NeurIPS, vol. 37, pp. 90277– 90317, 2024.
- [123] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in ICLR, 2023.
- [124] H. Shao, S. Qian, H. Xiao, G. Song, Z. Zong, L. Wang, Y. Liu, and H. Li, “Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning,” NeurIPS, pp. 8612–8642, 2024.
- [125] L. Nadel, “The hippocampus and context revisited.” 2008.
- [126] B. Yin, Q. Wang, P. Zhang, J. Zhang, K. Wang, Z. Wang, J. Zhang, K. Chandrasegaran, H. Liu, R. Krishna et al., “Spatial mental modeling from limited views,” arXiv, 2025.
- [127] A. Z. Ren, J. Clark, A. Dixit, M. Itkina, A. Majumdar, and D. Sadigh, “Explore until confident: Efficient exploration for embodied question answering,” arxiv, 2024.
- [128] W. Hu, Y. Hong, Y. Wang, L. Gao, Z. Wei, X. Yao, N. Peng, Y. Bitton, I. Szpektor, and K.-W. Chang, “3dllm-mem: Longterm spatial-temporal memory for embodied 3d large language model,” arXiv, 2025.
- [129] D. Marsili, R. Agrawal, Y. Yue, and G. Gkioxari, “Visual agentic ai for spatial reasoning with a dynamic api,” in CVPR, 2025, pp. 19446–19455.
- [130] J. Yu, Y. Zhang, Z. Zhang, Z. Yang, G. Zhao, F. Sun, F. Zhang, Q. Liu, J. Sun, J. Liang et al., “Rag-guided large language models for visual spatial description with adaptive hallucination corrector,” in ACMMM, 2024, pp. 11407–11413.
- [131] S. Azad, Y. Jain, R. Garg, V. Vineet, and Y. Rawat, “Understanding depth and height perception in large visual-language models,” in CVPR, 2025, pp. 3611–3620.
- [132] A. Kuhnle and A. Copestake, “Shapeworld-a new test methodology for multimodal language understanding,” arXiv, 2017.
- [133] R. Kamoi, Y. Zhang, S. S. S. Das, R. H. Zhang, and R. Zhang, “Visonlyqa: Large vision language models still struggle with visual perception of geometric information,” arXiv, 2024.
- [134] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu et al., “Mmbench: Is your multi-modal model an all-around player?” in ECCV, 2024, pp. 216–233.
- [135] S. Yang, R. Xu, Y. Xie, S. Yang, M. Li, J. Lin, C. Zhu, X. Chen, H. Duan, X. Yue et al., “Mmsi-bench: A benchmark for multiimage spatial intelligence,” arXiv, 2025.
- [136] X. Wang, W. Ma, T. Zhang, C. M. de Melo, J. Chen, and A. Yuille, “Spatial457: A diagnostic benchmark for 6d spatial reasoning of large mutimodal models,” in CVPR, 2025, pp. 24669–24679.
- [137] A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaff, S. Silwal, P. Mcvay, O. Maksymets, S. Arnaud et al., “Openeqa: Embodied question answering in the era of foundation models,” in CVPR, 2024, pp. 16488–16498.
- [138] W. Zhang, Z. Zhou, Z. Zheng, C. Gao, J. Cui, Y. Li, X. Chen, and X.-P. Zhang, “Open3dvqa: A benchmark for comprehensive

- spatial reasoning with multimodal large language model in open space,” arXiv, 2025.
- [139] D. Marsili, R. Agrawal, Y. Yue, and G. Gkioxari, “Visual agentic ai for spatial reasoning with a dynamic api,” in CVPR, 2025, pp. 19446–19455.
- [140] W. Chow, J. Mao, B. Li, D. Seita, V. Guizilini, and Y. Wang, “Physbench: Benchmarking and enhancing vision-language models for physical world understanding,” arXiv, 2025.
- [141] W. Zhang, W. E. Ng, L. Ma, Y. Wang, J. Zhao, A. Koenecke, B. Li, and L. Wang, “Sphere: Unveiling spatial blind spots in visionlanguage models through hierarchical evaluation,” arXiv, 2024.
- [142] W. Wang, R. Tan, P. Zhu, J. Yang, Z. Yang, L. Wang, A. Kolobov, J. Gao, and B. Gong, “Site: towards spatial intelligence thorough evaluation,” arXiv, 2025.
- [143] J. Lin, C. Zhu, R. Xu, X. Mao, X. Liu, T. Wang, and J. Pang, “Ost-bench: Evaluating the capabilities of mllms in online spatiotemporal scene understanding,” arXiv, 2025.
- [144] Z. Dongfang, X. Zheng, Z. Weng, Y. Lyu, D. P. Paudel, L. Van Gool, K. Yang, and X. Hu, “Are multimodal large language models ready for omnidirectional spatial reasoning?” arXiv, 2025.
- [145] Z. Li, X. Wang, E. Stengel-Eskin, A. Kortylewski, W. Ma, B. Van Durme, and A. L. Yuille, “Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning,” in CVPR, 2023, pp. 14963–14973.
- [146] K. Tang, J. Gao, Y. Zeng, H. Duan, Y. Sun, Z. Xing, W. Liu, K. Lyu, and K. Chen, “Lego-puzzles: How good are mllms at multi-step spatial reasoning?” arXiv, 2025.
- [147] Y. Yuan, R. Dang, L. Li, W. Li, D. Jiao, X. Li, D. Zhao, F. Wang, W. Zhang, J. Xiao et al., “Eoc-bench: Can mllms identify, recall, and forecast objects in an egocentric world?” arXiv, 2025.
- [148] F. Liu, G. Emerson, and N. Collier, “Visual spatial reasoning,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 635–651, 2023.
- [149] X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A. Smith, W.-C. Ma, and R. Krishna, “Blink: Multimodal large language models can see but not perceive,” in ECCV, 2024, pp. 148–166.
- [150] Y. Li, Y. Zhang, T. Lin, X. Liu, W. Cai, Z. Liu, and B. Zhao, “Sti-bench: Are mllms ready for precise spatial-temporal world understanding?” arXiv, 2025.
- [151] J. Zhang, Y. Chen, Y. Zhou, Y. Xu, Z. Huang, J. Mei, J. Chen, Y.-J. Yuan, X. Cai, G. Huang et al., “From flatland to space: Teaching vision-language models to perceive and reason in 3d,” arXiv, 2025.
- [152] Y.-H. Liao, R. Mahmood, S. Fidler, and D. Acuna, “Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models,” arXiv, 2024.
- [153] C.-H. Yeh, C. Wang, S. Tong, T.-Y. Cheng, R. Wang, T. Chu, Y. Zhai, Y. Chen, S. Gao, and Y. Ma, “Seeing from another perspective: Evaluating multi-view understanding in mllms,” arXiv, 2025.
- [154] S. K. Ramakrishnan, E. Wijmans, P. Kraehenbuehl, and V. Koltun, “Does spatial cognition emerge in frontier models?” arXiv, 2024.
- [155] F. Li, D. C. Hogg, and A. G. Cohn, “Reframing spatial reasoning evaluation in language models: A real-world simulation benchmark for qualitative reasoning,” arXiv, 2024.
- [156] J. Li, X. Nan, M. Lu, L. Du, and S. Zhang, “Proximity qa: Unleashing the power of multi-modal large language models for spatial proximity analysis,” arXiv, 2024.
- [157] Z. Liao, Q. Xie, Y. Zhang, Z. Kong, H. Lu, Z. Yang, and Z. Deng, “Improved visual-spatial reasoning via r1-zero-like training,” arXiv, 2025.
- [158] H. Wu, X. Huang, Y. Chen, Y. Zhang, Y. Wang, and W. Xie, “Spatialscore: Towards unified evaluation for multimodal spatial understanding,” arXiv, 2025.
- [159] P. Wu, Y. Liu, M. Liu, and J. Shen, “St-think: How multimodal large language models reason about 4d worlds from ego-centric videos,” arXiv, 2025.
- [160] Z. Lin, S. Cen, D. Jiang, J. Karhade, H. Wang, C. Mitra, T. Ling, Y. Huang, S. Liu, M. Chen et al., “Towards understanding camera motions in any video,” arXiv, 2025.
- [161] W. Han, S. Xiang, C. Liu, R. Wang, and C. Feng, “Spare3d: A dataset for spatial reasoning on three-view line drawings,” in CVPR, 2020, pp. 14690–14699.
- [162] X. Wang, W. Ma, A. Wang, S. Chen, A. Kortylewski, and A. Yuille, “Compositional 4d dynamic scenes understanding with physics priors for video question answering,” arXiv, 2024.

- [163] D. Li, H. Li, Z. Wang, Y. Yan, H. Zhang, S. Chen, G. Hou, S. Jiang, W. Zhang, Y. Shen et al., “Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models,” arXiv, 2025.
- [164] Z. Zhang, F. Hu, J. Lee, F. Shi, P. Kordjamshidi, J. Chai, and Z. Ma, “Do vision-language models represent space and how? evaluating spatial frame of reference under ambiguities,” arXiv, 2024.
- [165] I. Stogiannidis, S. McDonagh, and S. A. Tsaftaris, “Mind the gap: Benchmarking spatial reasoning in vision-language models,” arXiv, 2025.
- [166] X. Ma, S. Yong, Z. Zheng, Q. Li, Y. Liang, S.-C. Zhu, and S. Huang, “Sqa3d: Situated question answering in 3d scenes,” arXiv, 2022.
- [167] J. Wang, Y. Ming, Z. Shi, V. Vineet, X. Wang, S. Li, and N. Joshi, “Is a picture worth a thousand words? delving into spatial reasoning for vision language models,” NeurIPS, vol. 37, pp. 75392–75421, 2024.
- [168] W. Wu, S. Mao, Y. Zhang, Y. Xia, L. Dong, L. Cui, and F. Wei, “Mind’s eye of llms: visualization-of-thought elicits spatial reasoning in large language models,” NeurIPS, vol. 37, pp. 90277– 90317, 2024.
- [169] L. Wang, Z. He, R. Dang, M. Shen, C. Liu, and Q. Chen, “Visionand-language navigation via causal learning,” in CVPR, 2024, pp. 13139–13150.
- [170] Z. Gong, W. Li, O. Ma, S. Li, J. Ji, X. Yang, G. Luo, J. Yan, and R. Ji, “Space-10: A comprehensive benchmark for multimodal large language models in compositional spatial intelligence,” arXiv, 2025.
- [171] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv, 2025.
- [172] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv, 2024.
- [173] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao et al., “Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv, 2025.
- [174] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv, 2025.
- [175] W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi, J. Ji, L. Pan et al., “Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning,” arXiv e-prints, pp. arXiv–2507, 2025.
- [176] D. Guo, F. Wu, F. Zhu, F. Leng, G. Shi, H. Chen, H. Fan, J. Wang, J. Jiang, J. Wang et al., “Seed1. 5-vl technical report,” arXiv, 2025.
- [177] J. Krantz, E. Wijmans, A. Majundar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision and language navigation in continuous environments,” in ECCV, 2020.
- [178] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang et al., “Spatialvla: Exploring spatial representations for visual-language-action model,” arXiv, 2025.
- [179] R. Zheng, Y. Liang, S. Huang, J. Gao, H. Daum´e III, A. Kolobov, F. Huang, and J. Yang, “Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies,” ICLR, 2024.
- [180] Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn et al., “Cot-vla: Visual chain-of-thought reasoning for vision-language-action models,” in CVPR, 2025, pp. 1702– 1713.
- [181] Y. Zhou, L. Huang, Q. Bu, J. Zeng, T. Li, H. Qiu, H. Zhu, M. Guo, Y. Qiao, and H. Li, “Embodied understanding of driving scenarios,” in ECCV, 2024, pp. 129–148.
- [182] X. Cao, T. Zhou, Y. Ma, W. Ye, C. Cui, K. Tang, Z. Cao, K. Liang, Z. Wang, J. M. Rehg et al., “Maplm: A real-world large-scale vision-language benchmark for map and traffic scene understanding,” in CVPR, 2024, pp. 21819–21830.
- [183] K. Tian, J. Mao, Y. Zhang, J. Jiang, Y. Zhou, and Z. Tu, “Nuscenesspatialqa: A spatial understanding and reasoning benchmark for vision-language models in autonomous driving,” arXiv, 2025.
- [184] T. Choudhary, V. Dewangan, S. Chandhok, S. Priyadarshan, A. Jain, A. K. Singh, S. Srivastava, K. M. Jatavallabhula, and K. M. Krishna, “Talk2bev: Language-enhanced bird’s-eye view maps for autonomous driving,” in ICRA, 2024, pp. 16345–16352.
- [185] E. Sachdeva, N. Agarwal, S. Chundi, S. Roelofs, J. Li, M. Kochenderfer, C. Choi, and B. Dariush, “Rank2tell: A multimodal driving dataset for joint importance ranking and reasoning,” in Proceed-

- ings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 7513–7522.
- [186] X. Zhou, M. Liu, E. Yurtsever, B. L. Zagar, W. Zimmer, H. Cao, and A. C. Knoll, “Vision language models in autonomous driving: A survey and outlook,” IEEE Transactions on Intelligent Vehicles, pp. 1–20, 2024.
- [187] J.-J. Hwang, R. Xu, H. Lin, W.-C. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp et al., “Emma: End-to-end multimodal model for autonomous driving,” arXiv, 2024.
- [188] I. Kabir, M. A. Reza, and S. Billah, “Logic-rag: Augmenting large multimodal models with visual-spatial knowledge for road scene understanding,” arXiv, 2025.
- [189] C. H. Song, V. Blukis, J. Tremblay, S. Tyree, Y. Su, and S. Birchfield, “Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics,” in CVPR, 2025, pp. 15768– 15780.
- [190] A. Kamath, J. Hessel, and K.-W. Chang, “What’s” up” with vision-language models? investigating their struggle with spatial reasoning,” arXiv, 2023.
- [191] E. Szymanska,´ M. Dusmanu, J.-W. Buurlage, M. Rad, and M. Pollefeys, “Space3d-bench: Spatial 3d question answering benchmark,” in ECCV, 2024, pp. 68–85.
- [192] M. Du, B. Wu, Z. Li, X. Huang, and Z. Wei, “Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models,” arXiv, 2024.
- [193] J. Liu, Z. Liu, Z. Cen, Y. Zhou, Y. Zou, W. Zhang, H. Jiang, and T. Ruan, “Can multimodal large language models understand spatial relations?” arXiv, 2025.
- [194] W. Ma, H. Chen, G. Zhang, Y.-C. Chou, C. M. de Melo, and A. Yuille, “3dsrbench: A comprehensive 3d spatial reasoning benchmark,” arXiv, 2024.
- [195] M. S. Danish, M. A. Munir, S. R. A. Shah, K. Kuckreja, F. S. Khan, P. Fraccaro, A. Lacoste, and S. Khan, “Geobench-vlm: Benchmarking vision-language models for geospatial tasks,” arXiv, 2024.
- [196] Q. Wu, H. Zhao, M. Saxon, T. Bui, W. Y. Wang, Y. Zhang, and S. Chang, “Vsp: Assessing the dual challenges of perception and reasoning in spatial planning tasks for vlms,” arXiv, 2024.

#### APPENDIX A APPENDIX OUTLINE

This supplementary material includes 6 aspects:

- 1) Data Source: the distribution of the proposed SIBench.
- 2) Adaptation of test data.
- 3) Detailed Comparison of Various Settings.
- 4) Failure case analysis.
- 5) Timeline of Representative VSR Benchmarks.
- 6) Comparison of Benchmarks for Visual Spatial Reasoning.

###### A.1 Data Source

[Figure 160]

- Fig. 10: Data source of SIBench. The distribution of SIBench across three input formats and nearly 20 opensource datasets.

We construct SIBench by aggregating data from approximately 20 open-source datasets into three distinct input formats (see Fig. 10). The breadth of the data guarantees the comprehensiveness of the evaluation.

A.2 Adaptation of test data

[Figure 161]

- Fig. 11: Issues with the test data. i. (Upper) When the test video is too long, the larger sampling interval may cause some targets to be lost, leading to mismatches with the ground truth. ii. (Bottom) The variable sampling interval results in the loss of temporal information.

We conduct a thorough assessment of the task’s reasonableness. In our video test data, for example, we observe that some videos are as long as five minutes, yet the testing

protocol requires sampling only 30 frames for model input. This large sampling interval often leads to target loss, which adversely affects tasks such as object counting (see the upper part of Fig. 11). To mitigate this, we restrict our video selections to a maximum duration of one minute. In another example, when a model is tasked with estimating velocity and acceleration from a video, the testing requirements mandate a fixed number of sampled frames. However, the video durations are variable, resulting in inconsistent sampling intervals and a loss of temporal information, which renders the test meaningless. Our solution is to add a timestamp to each frame, ensuring that effective temporal information is preserved regardless of variations in the sampling interval (see the bottom part of Fig. 11).

###### A.3 Detailed Comparison of Various Settings

SIBench is a comprehensive benchmark consisting of 23 distinct task settings. The detailed results of various models on SIBench and SIBench-mini are shown in Table 4 and Table 5, respectively. Models generally excel at basic perception tasks like identifying the existence of objects but struggle significantly with tasks requiring complex, abstract reasoning, such as spatial imagination, temporal ordering, and precise quantitative estimation. Overall, Gemini-2.5Pro [21] is the top one in the SIBench, while the GPT-5 [20] leads in the SIBench-mini.

- A.3.1 Areas of Model Strength

In Table 4 and Table 5, models consistently perform well on tasks that rely on recognizing objects and simple relationships of objects. On the existence task, Qwen2.5-VL72B [171] achieves a perfect score of 1.0, and many others are above 0.9. This indicates that identifying whether an object is present in a scene is an easy task for current models. On the Spatial Relation task, InternVL-3-78B [173] leads at 0.8402 and Qwen2.5-VL-72B [171] also receives 0.8216. This strength suggests that models have a solid grasp of static relational ability.

- A.3.2 Areas of Model Weakness

The models exhibit several key weaknesses, consistent with the analysis in Section 5.2.2. First, models are weak at some tasks of Foundational Perception, such as Object Size Estimation shown in Table 4 and Table 5. Second, models struggle with quantitative reasoning. This is evident in tasks Relative Distance in Table 4 and Table 5. The scores indicate that models like Gemini-2.5-Flash [21] and InternVL-3-78B [173] rely on coarse visual clues rather than precise metric representations. Third, models show a neartotal absence of spatial imagination. On the Spatial Imagination task, models like GPT-4o-mini [172] and InternVL2.5-78B-MPO [173] receive almost the lowest scores. This indicates that models fail on tasks that require the creation and manipulation of a mental imagination. Models cannot visualize and reason about objects or scenarios that are not explicitly depicted, which is a key component of humanlike spatial intelligence. Second, models are weak at inferring spatial properties from different viewpoints. This is shown by the low scores in Multi-View Reasoning, Camera

[Figure 162]

###### Fig. 12: Limited Robustness of Foundational Perception.

Pose, and Geometric Reasoning. Models like Qwen2.5-VL7B [171], LLaVA-OneVision-72B [18], Gemini-2.5-Flash [21] cannot intuitively understand how a camera’s movement changes a scene’s perspective, leading to poor performance. Similarly, models struggle with temporal appearance order, with LLaVA-OneVision-7B [18] scoring 0.0000. This confirms that models’ unreliable ability for non-static scenes.

###### A.4 Failure case analysis

As mentioned in Section A.3, there are four areas of model weakness. i.e., foundational perception, quantitative capabilities, spatial imagination, and multi-view reasoning. We provide visualizations and analysis of failure cases of these four aspects. For foundational perception, Fig. 12 shows two failure cases, including counting and shape. For counting, GPT-5 predicts close to the GT answer. For shape analysis, GPT-5 predicts only half of GT number of drawers.

For precise and quantitative capabilities, the current VLM also shows limited performance, as shown in Fig. 13. Specifically, GPT-5 provides answers that significantly deviate from the ground truth when tasked with estimating the distance between objects and the size of a specific object from an image. Current VLMs also show deficiencies in spatial imagination and 3D reconstruction (See Fig. 14). In these cases, GPT-5 is unable to correctly identify a rotated 3D shape or determine the relative spatial position of an object from a different perspective. VLMs are weak at dynamics and multi-view reasoning. In Fig. 15, GPT5 incorrectly answers a question that requires it to first identify the largest window dimension from the provided specifications and then use that data to calculate the time needed to open a curtain at a given velocity.

[Figure 163]

- Fig. 13: Lack of Precise and Quantitative Capabilities.

[Figure 164]

- Fig. 14: Deficiency in Spatial Imagination and 3D Reconstruction.

A.6 Comparison between current benchmarks and SIBench

Table 6 presents a comparison of current benchmarks with SIBench. Task settings are categorized as concentrated if there are fewer than 10, and diverse otherwise. The proposed SIBench is characterized by the extensive coverage of input and QA types, all within a large-scale dataset.

###### A.5 Timeline of Representative VSR Benchmarks

We survey over 60 benchmarks in VSR and categorize them into two main types of contributions: methodological advancements and novel benchmarks, as shown in Fig. 16.

###### TABLE 4: Performance evaluation of different models on visual spatial reasoning tasks.

Models Qwen2.5-VL-7B [171] LLaVA-OneVision-7B [18] GPT-4o-mini [172] Gemini-2.5-Flash [21] InternVL-2.5-78B-MPO [173] InternVL-3-78B [173] InternVL-3.5-38B [173]

Settings

Reach Prediction 0.6000 0.6750 0.6500 0.4750 0.6500 0.6000 0.5750 Height 0.5117 0.6117 0.5883 0.6167 0.6167 0.6517 0.5733 Existence 0.9750 0.9000 0.9250 0.9750 0.9750 0.9500 0.9000 Occlusion 0.5660 0.6620 0.5880 0.6160 0.6900 0.7940 0.6160 Object Shape 0.2308 0.2846 0.2692 0.3692 0.3154 0.3385 0.3000 Counting 0.5672 0.5042 0.4454 0.3991 0.4370 0.5462 0.4916 Object Size Estimation 0.5265 0.4679 0.5530 0.4575 0.5590 0.4573 0.6139

Basic Perception

Spatial Compatibility 0.5280 0.4673 0.4813 0.4673 0.6168 0.6075 0.5748 Coordinate Conversion 0.5838 0.5413 0.5113 0.3563 0.6475 0.6475 0.5900 Trajectory Description 0.3205 0.5000 0.2179 0.4615 0.3718 0.2692 0.5513 Geometric Reasoning 0.2103 0.2738 0.2897 0.1786 0.2659 0.2857 0.2659 Spatial Imagination 0.3100 0.3075 0.2325 0.2575 0.2925 0.2800 0.3100 Spatial Grid 0.4675 0.4025 0.4375 0.5650 0.6375 0.6425 0.7525 Temporal Appearance Order 0.1818 0.0000 0.2727 0.3636 0.2727 0.0909 0.2727 Multi-View Reasoning 0.3552 0.3892 0.3287 0.4358 0.4125 0.3934 0.3998 Situational QA 0.4994 0.4972 0.4729 0.4036 0.5329 0.5351 0.5173 Velocity Acceleration 0.3613 0.3882 0.2605 0.3344 0.4202 0.4840 0.5092 Relative Distance 0.4695 0.4629 0.3130 0.3608 0.5812 0.6324 0.5462 Camera Pose 0.4041 0.4477 0.3198 0.3459 0.5029 0.5581 0.5087 Spatial Relation 0.8239 0.7370 0.5892 0.4395 0.8130 0.8435 0.8133 Object Localization 0.4704 0.7290 0.5763 0.5670 0.3832 0.4922 0.2461

Spatial Understanding

Maze Navigation 0.3475 0.3375 0.2950 0.6400 0.5650 0.4775 0.4850 Route Planning 0.3220 0.3220 0.3729 0.4068 0.3220 0.2542 0.4576

Planning

Overall 0.4712 0.4850 0.4278 0.4389 0.5338 0.5481 0.5252

###### TABLE 5: Performance evaluation of different models on the SIBench-mini.

Models LLaVA-OneVision-72B [18] Qwen2.5-VL-72B [171] InternVL3.5-38B [173] GLM4.5-V-106B-A12B [22] Doubao-Seed-1.6-Vision [176] Gemini-2.5-Pro [21] GPT-5 [20]

Settings

Reach Prediction 0.6000 0.6750 0.5750 0.6500 0.5000 0.5750 0.6000 Height 0.6500 0.6000 0.5633 0.7750 0.7500 0.7500 0.8000 Existence 0.9500 0.9500 1.000 0.9750 0.9500 0.9750 0.9250 Occlusion 0.6000 0.5500 0.6160 0.7250 0.8000 0.7250 0.6250 Object Shape 0.2500 0.2750 0.3000 0.4000 0.3250 0.2000 0.4750 Counting 0.8889 0.6111 0.4916 0.7222 0.9444 0.8333 0.7222 Object Size Estimation 0.5150 0.6675 0.6139 0.5800 0.5450 0.6175 0.6425

Basic Perception

Spatial Compatibility 0.6250 0.7000 0.5748 0.7000 0.6500 0.7250 0.6250 Coordinate Conversion 0.7000 0.7000 0.5900 0.8000 0.7250 0.7000 0.6750 Trajectory Description 0.5500 0.6250 0.5513 0.3750 0.4500 0.4500 0.4750 Geometric Reasoning 0.2750 0.2500 0.2659 0.3000 0.3000 0.4000 0.3500 Spatial Imagination 0.3750 0.3000 0.3100 0.3250 0.3250 0.3250 0.3250 Spatial Grid 0.6250 0.8250 0.7265 0.9250 0.9750 1.0000 0.7750 Temporal Appearance Order 0.3636 0.5454 0.2727 0.1818 0.9091 0.1818 1.0000 Multi-View Reasoning 0.5250 0.3250 0.2269 0.3750 0.5250 0.4500 0.5750 Situational QA 0.3750 0.3438 0.5173 0.3438 0.5000 0.4375 0.6250 Velocity Acceleration 0.4000 0.4500 0.5092 0.4750 0.5750 0.5000 0.5250 Relative Distance 0.5325 0.5650 0.4811 0.6775 0.7350 0.6675 0.7075 Camera Pose 0.2250 0.4000 0.4884 0.3000 0.4000 0.5000 0.4000 Spatial Relation 0.6250 0.5750 0.8133 0.7500 0.6500 0.6750 0.6000 Object Localization 0.7250 0.7000 0.2430 0.7500 0.8250 0.7750 0.8000

Spatial Understanding

Maze Navigation 0.4000 0.5000 0.3150 0.7750 0.8500 0.7750 0.7500 Route Planning 0.3750 0.3000 0.4576 0.2500 0.4500 0.3750 0.4500

Planning

Overall 0.5252 0.5168 0.5355 0.5822 0.6216 0.6295 0.6906

[Figure 165]

###### Fig. 15: Insufficiency in Dynamics and Multi-View Reasoning.

[Figure 166]

- Fig. 16: Timeline of Representative VSR Benchmarks. Purple indicates novel evaluation methods and green denotes methodological improvements.

TABLE 6: Comparison of Benchmarks for Visual Spatial Reasoning.

Scale Input Type QA Type

Benchmark Annotation

Diversity (k) Single Multi-View Video T/F Multi-Choice Number

VSR [148] Synthetic 2.81 ✓ ✗ ✗ ✓ ✗ ✗ Concentrated BLINK [149] Manual 0.28 ✓ ✗ ✗ ✗ ✓ ✗ Diverse

VSI-Bench [31] Synthetic 5.00 ✗ ✗ ✓ ✗ ✗ ✓ Diverse

Spatial-MM [32] Synthetic 2.30 ✓ ✗ ✗ ✗ ✓ ✗ Concentrated SpatialRGPT [27] Synthetic 1.40 ✓ ✗ ✗ ✓ ✗ ✓ Concentrated SpatialVLM [26] Synthetic 0.54 ✓ ✗ ✗ ✗ ✗ ✓ Concentrated STI-Bench [150] Synthetic 2.00 ✗ ✗ ✓ ✗ ✓ ✗ Diverse

SpatialBench [34] Synthetic 0.39 ✓ ✗ ✗ ✓ ✓ ✓ Concentrated RoboSpatial [189] Synthetic 6.00 ✓ ✗ ✗ ✓ ✗ ✓ Concentrated

What’s up [190] Synthetic 0.49 ✓ ✗ ✗ ✗ ✓ ✗ Concentrated Space3D-Bench [191] Synthetic 1.00 ✓ ✗ ✗ ✗ ✗ ✓ Concentrated

EmbSpatial-Bench [192] Manual 3.64 ✓ ✗ ✗ ✗ ✓ ✗ Concentrated OmniSpatial [33] Manual 1.50 ✓ ✗ ✗ ✓ ✓ ✗ Diverse PhysBench [140] Manual 10.00 ✓ ✗ ✓ ✗ ✓ ✗ Concentrated

VSI-100K [157] Synthetic 5.00 ✗ ✗ ✓ ✗ ✗ ✓ Concentrated SpatialMQA [193] Manual 5.39 ✓ ✗ ✗ ✗ ✓ ✗ Concentrated SRBench [165] Synthetic 1.80 ✓ ✗ ✗ ✗ ✓ ✗ Concentrated

Open3DVQA [138] Synthetic 1.24 ✗ ✓ ✗ ✓ ✗ ✓ Concentrated Q-Spatial Bench [152] Manual 0.27 ✓ ✗ ✗ ✗ ✗ ✓ Concentrated

VoT [168] Synthetic 3.51 ✓ ✗ ✗ ✗ ✓ ✗ Concentrated 3DSRBench [194] Synthetic 2.77 ✓ ✗ ✗ ✓ ✓ ✗ Diverse Super-CLEVR-3D [145] Synthetic 5.00 ✓ ✗ ✗ ✓ ✓ ✗ Concentrated

SPHERE [141] Manual 2.28 ✓ ✗ ✗ ✓ ✗ ✓ Diverse LEGO-Puzzles [146] Manual 1.10 ✓ ✗ ✓ ✓ ✓ ✓ Diverse

GEOBench-VLM [195] Manual 10.00 ✓ ✗ ✓ ✗ ✓ ✗ Diverse

OpenEQA [137] Manual 1.60 ✗ ✓ ✗ ✗ ✗ ✓ Concentrated SPARE3D [161] Synthetic 2.50 ✗ ✓ ✗ ✗ ✓ ✗ Concentrated VSP [196] Synthetic 4.40 ✓ ✗ ✗ ✗ ✓ ✗ Diverse

DynSuperCLEVR [162] Synthetic 1.00 ✗ ✗ ✓ ✓ ✓ ✗ Concentrated

SITE [142] Synthetic 8.06 ✓ ✗ ✓ ✗ ✓ ✗ Diverse SIBench Manual 9.00 ✓ ✓ ✓ ✓ ✓ ✓ Diverse

