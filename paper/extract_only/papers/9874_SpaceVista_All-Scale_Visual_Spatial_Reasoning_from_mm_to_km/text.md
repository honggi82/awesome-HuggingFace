## SpaceVista: All-Scale Visual Spatial Reasoning from mm to km

Peiwen Sun*12 Shiqiang Lang*3 Dongming Wu1 Yi Ding2 Kaituo Feng1 Huadai Liu4 Zhen Ye4 Rui Liu1 Yun-Hui Liu1 Jianan Wang†2 Xiangyu Yue 1

# arXiv:2510.09606v2[cs.CV]26May2026

### Abstract

With the current surge in spatial reasoning, researchers have made significant progress in understanding indoor scenes, but still struggle with more diverse applications. This paper aims to advance all-scale spatial reasoning by tackling two key challenges: 1) the heavy reliance on indoor 3D scans and labor-intensive annotations for dataset curation; 2) the absence of all-scale modeling, which often leads to overfitting to single scenes. In this paper, we introduce a holistic solution that integrates a structured spatial reasoning knowledge system, scale-aware modeling, and a progressive training paradigm, as the first attempt to broaden the scope of all-scale spatial intelligence. Using a task-specific, specialistdriven automated pipeline, we curate over 38K video scenes across 5 spatial scales to create SpaceVista-1M, a dataset comprising 1M spatial QAs spanning 19 diverse tasks. While specialist models offer valuable domain knowledge, they are often unreliable evaluators. Therefore, we build an all-scale benchmark with precise annotations by manually recording and retrieving videos. Nevertheless, naive training with SpaceVista-1M often yields suboptimal results due to the potential knowledge conflict. Accordingly, we introduce SpaceVista-7B, a spatial reasoning model that accepts inputs beyond semantics and uses scale as an anchor for scale-aware experts and progressive rewards. Finally, extensive evaluations across 5 benchmarks, including our SpaceVista-Bench, demonstrate competitive performance, showcasing generalization across all scales and scenarios. Our demo page is posted on website.

*Equal contribution †Indicates project leader. 1Multimedia Laboratory, The Chinese University of Hong Kong 2Astribot 3Beijing University of Posts and Telecommunications 4Hong Kong University of Science and Technology. Correspondence to: Xiangyu Yue <xyyue@ie.cuhk.edu.hk>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

### 1. Introduction

Spatial reasoning, the ability to sense, interpret, and interact with environments across scales from understanding tiny objects to remote drone sensing, is crucial for next-generation intelligent systems. It significantly enhances 3D and even 4D scene understanding, enabling agents to interpret complex environments from easily obtainable videos. All-scale reasoning capability supports diverse applications: mm for advanced manufacturing (Song et al., 2024), cm and m for embodied intelligence (Pan et al., 2025), 10m for autonomous driving (Liu et al., 2022), and 100m for dronebased sensing (Xiao et al., 2023). Recent research (Yang et al., 2025a), especially on how Multimodal Large Language Models (MLLMs) perceive and recall space, is narrowing the gap in visual spatial reasoning.

The current works on spatial reasoning primarily focus on improvements from two perspectives: data and model. From the data perspective, pioneer works (Ouyang et al., 2025; Zhang et al., 2025c; Deng et al., 2025b) utilize more scanning-based data, or image-based data employing fully automated pipelines to acquire additional information for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). When modeling indoor spatial scenes, Wu et al. (2025c); Zheng et al. (2025) leverage latent features from VGGT (Wang et al., 2025a) by incorporating geometric information to enhance spatial understanding. Concurrently, a series of outstanding works (Ouyang et al., 2025; Zhang et al., 2025c) have improved the performance of existing models by refining the training and thinking approaches. Moreover, Wu et al. (2025d) employs multi-turn dialogues to enhance self-correction capabilities.

Despite these works’ advancements, their spatial perception capabilities are primarily limited to indoor settings, specific objects, and constrained scales, as shown in the bar chart Fig.1. Moreover, current methodologies lack dedicated training frameworks for holistic all-scale scene understanding. To bridge this gap, we introduce the first comprehensive solution to address data, model, and evaluation dimensions for all-scale scenarios.

Previous datasets (Yang et al., 2025a;b; Ouyang et al., 2025; Zhang et al., 2025c) for spatial reasoning have primarily been constructed based on indoor scanning video data (Dai

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Tiny Tabletop Tabletop Indoor Outdoor Drone View

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Closer Closer

[Figure 20]

Closer

Closer

Industrial Manufacture

Robotic System

Embodied Intelligence

Automatic Drive

Remote Sensing

[Figure 21]

|The smaller scale is missing<br><br>|Previous Spatial Reasoning Datasets and Models<br><br>|The larger scale is missing|
|---|---|---|

|Tiny Tabletop Scenes<br><br>[Figure 22]|
|---|

|Indoor Scenes<br><br>[Figure 23]|
|---|

|Drone Scenes<br><br>[Figure 24]|
|---|

[Figure 25]

[Figure 26]

[Figure 27]

|Outdoor Scenes<br><br>[Figure 28]|
|---|

|Tabletop Scenes<br><br>[Figure 29]|
|---|

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

|1e-3m<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]|1m|[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>1e+3m|
|---|---|---|
|Why not only one model to cover all-scale? SpaceVista!<br><br>[Figure 65]| | |

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

Object Location

###### Manipulation Planning

###### Distance Estimation Camera Location

###### Area Estimation

Question: In the 1st frame, where is the coin located relative to the screw at the current view angle?

Question: How can you move the bottle into the box from the perspective of 1st frame?

Question: How far (in meters) from the table and the camera position from the perspective of

Question: Which exact direction the camera is moving towards. Answer: <think>The river is on

Question: How big (in square meters) an area does the garden occupy in the video?

Answer: <think> The overall

Answer: <think>The overall

1st frame?

the left side initially… Then, …

Answer: <think>The overall … The

scenes is ... coin… 2cm… coin…, around 2cm away... </think> <answer>A. Frontward, 2cm </answer>

scenes is … The black spoon, as potential obstacle… </think> <answer>D. 1. Moving Up 10cm. 2. Moving Left 30cm …</answer>

Answer: <think> The overall… According…, the door’s width is around 0.6m… </think> <answer>5</answer>

river move to the right of the video. The others… So,… </think> <answer>C. Left</answer>

white grid seems like the parking spot, which is usually, 4.8m×1.8m. Therefore… </think> <answer>12.2</answer>

Figure 1. Prior works of spatial reasoning have largely focused on indoor (1-30 m) scenes, while our SpaceVista model and dataset span scales from mm (1e-3 m) to km (1e+3 m). Dotted lines represent our contribution in filling the gap. This six-order-of-magnitude range introduces not only scale variation but also rich semantics and diverse tasks. SpaceVista enables all-scale spatial reasoning by integrating cues from micro-objects to macro-scenes.

often deviate significantly, in tabletop and other diverse realworld scenes illustrated in Fig. 2(a). We address this by first injecting SpaceVista-1M knowledge to fine-tune existing models with the self-supervised visual encoder to make compensation for the classic semantic visual tokenizer, enabling extra geometry-based and depth-based spatial understanding. However, naive fine-tuning rarely yields optimal results, largely due to cross-scale conflicts between scenes and objects based on our observation. To address this, we introduce LoRA-like scale experts that cooperates with a scale router during fine-tuning. Moreover, to strengthen the model’s ability to learn scale-centric spatial reasoning processes, we design a training strategy that uses scale as an anchor for progressive rewards. During evaluation, SpaceVista-7B shows superior understanding of spatial layout, size, and comparison, delivering a clear improvement on popular benchmarks and SpaceVista-Bench.

et al., 2017; Yeshwanth et al., 2023) as shown in Fig. 2(b). These indoor datasets often feature relatively simple scenes and depend on manual 3D annotations. Scaling up to build large-scale, wild datasets encompassing video scenes ranging from mm to km presents two major challenges: 1) the high cost of large-scale annotation from complex and wild scenes; 2) the difficulty in obtaining precise evaluations that align with the physical world. To address these challenges, we use an automated pipeline leveraging popular specialized models to generate structured training data across 5 different scales. Since different scales have distinct characteristics and applications, we define several scale-specific tasks for better application, i.e., manipulation planning and area estimation. Overall, we provide over 1 million QA pairs across 19 diverse tasks from around 38K wild video scenes. To adapt to different stages of training, we provide both answers with rationale for SFT and regression/multiple-choice answers for RL. To facilitate accurate evaluation, we collect a highly accurate SpaceVista-Bench through manually recording or retrieving authoritative sources, supplemented with human annotations.

Our key contributions with this comprehensive solution are:

• Developing an automated pipeline to create a diverse, real-world, all-scale reasoning dataset, SpaceVista1M, with 1M QA pairs across 5 scales and 19 tasks (including specific-scale tasks), and supporting both SFT with rationale and high-quality RL.

Most popular reasoning models are optimized for indoor settings, which leads to clear limitations: their responses

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

- • Introducing SpaceVista-7B, a spatial reasoning model that integrates rich spatial information and employs scale experts with a customized training strategy to alleviate potential cross-scale conflicts during all-scale finetuning.
- • Hand-crafting SpaceVista-Bench, an accurate video benchmark spanning all scales, by measuring and recording real-world objects, retrieving authoritative sources, and performing human annotation.

|[Figure 100]<br><br>| |
|---|
|
|---|

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

| |
|---|

[Figure 106]

###### How far between the street lamp and trash can? GT: 7.7m

How long is the screw?

[Figure 107]

[Figure 108]

GT: 0.7cm

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

- 1.3cm
- 2.3cm 0.8cm 3.8cm

1.1cm 10.2m

4.2cm

4.0m

13.0m

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

8.1m 12.3m

7.0m

[Figure 128]

[Figure 129]

[Figure 130]

(a) Case comparison across scales on popular MLLMs

[Figure 131]

[Figure 132]

[Figure 133]

###### SpaceR SpaceVista VG-LLM

[Figure 134]

SpaceVista-1M (Ours)

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

### 2. Related Works

Spatial-MLLM

SpaceR

VSI-Bench Spar-7M

MMSI-Bench

|∆𝟏𝒆 +2|
|---|

Visual Reasoning. Currently, vision-based general reasoning has seen diverse developments (Tan et al., 2025; Wang et al., 2025b; Qiao et al., 2025; Wu et al., 2025a). General MLLMs (Wang et al., 2025c; Bai et al., 2025) first provided the basic understanding ability towards video to the community. Pioneering works (Feng et al., 2025; Liao et al., 2025) started to provide reasonable rewards during model training using Group Relative Policy Optimization (GRPO) for the reasonable Chain of Thought (CoT). Then, visual reasoning (Li et al., 2025c; Chen et al., 2025; Liu et al., 2025c) was considered from broader perspectives, ranging from data to training structure. In general video reasoning, spatial claims are generally divided into two categories: 2D plane-based spatial reasoning (Han et al., 2025; Zhou et al., 2025), and 3D space-based spatial reasoning (Wu et al., 2025c; Zheng et al., 2025). This paper primarily focuses on the latter. Although these general models have achieved a certain degree of spatial ability, spatial MLLM is still in its early stages.

Scale(m)

|∆𝟏e +2|
|---|

|∆𝟏𝒆 +2|
|---|

|∆𝟏𝒆+2|
|---|

|∆𝟏𝒆 +2|
|---|

|∆𝟏𝒆 + 6|
|---|

<video>

<multi-

images>

<video>

<multi-

<video>

images>

<video>

max scale min scale

<video> <Input Type>

∆≈

(b) Scale comparison on popular spatial datasets and benchmarks

Figure 2. Case performance and dataset distribution across scales. Current models and datasets necessitate all-scale spatial reasoning.

inputs. Additionally, spatial reasoning on long (Zhang et al., 2025a), omni (Dongfang et al., 2025), ego-centric (Wu et al.,

- 2025e) and aerial video (Zhang et al., 2025a; Sun et al.,
- 2026) were also explored separately. However, the systematic data and designed model with all-scale video scenes remain unexplored.

Spatial Reasoning. Mainstream spatial reasoning models can be categorized based on input modalities into image (Ma et al., 2025; Liu et al., 2025b; Chen et al., 2024a), multiimage (Xu et al., 2025), multi-view (Li et al., 2025b; Wu et al., 2025b), video (Wu et al., 2025c; Zheng et al., 2025; Ouyang et al., 2025; Zhang et al., 2025a; Ghazanfari et al., 2025), and simulation (Li et al., 2025a; Tang et al., 2025; Zhang et al., 2025b; Wang et al., 2025d; Zhang et al., 2025d). Among these categories, video stands out as a challenging task due to the difficulty of data acquisition and modeling. As the first work in spatial reasoning, VSI-Bench (Yang et al., 2025a) introduced a video-based benchmark that removes linguistic shortcuts and evaluated MLLMs on spatial tasks such as counting, direction, and planning, highlighting substantial performance gaps compared to humans. InternSpatial (Deng et al., 2025b), SPAR (Zhang et al., 2025c), and SpaceR (Ouyang et al., 2025) enriched spatial supervision through extensive QA pairs spanning indoor and other limited settings.

All-Scale Exploration. The challenge of multi-scale in early years lay in information loss within low-resolution image patches (Zhao, 2025; Nikouei et al., 2025), which has almost no effect on spatial reasoning. In this paper, “all-scale” primarily concerns the real scales of the physical world, including distances, semantics, and object states across different scales. Deng et al. (2025a) pushed the limits of 3D perception and reconstruction from meters to kilometers; Wen et al. (2025) extended metric depth estimation from close range to infinity; and Liu et al. (2025a) curated uncommon objects, ranging from screws to airplanes, with object-centric annotations. Together, these developments underscore the need for AI to move beyond simple single-scale memorization toward robust, multiscale, and reasonable visual understanding.

### 3. Dataset

Due to high labeling cost, Tab.1 and Fig.2 show the clear drawback of the previous datasets. The limited data and performance constraints in existing models necessitate the creation of a dataset with all-scale spatial context. We propose SpaceVista-1M, a diverse, real-world, all-scale reasoning dataset, as the first to the best of our knowledge. SpaceVista-1M primarily comprises diverse spatial reason-

Qi et al. (2025) used the bird-view map to aid overall understanding. Then, Spatial-MLLM (Wu et al., 2025c), VGLLM (Zheng et al., 2025), and VLM-3R (Fan et al., 2025) adopted geometry-aware dual encoders to capture geometry cues and inferred occluded structures from monocular

SpaceR 3.8% SPAR-7M 11.4%

Others 4.4%

Drone 0.4%

Outdoor 12.9%

###### Step 1: Data Preparation Output

###### Step 2: Canonical View Transformation

[Figure 139]

[Figure 140]

Tiny 26.8%

[Figure 141]

Camera Intrinsics & Extrinsics

Wild-RGBD

[Figure 142]

[Figure 143]

Camera Intrinsics & Extrinsics

[Figure 144]

Consistency & Occlusion Checking

28.6%

DL3DV

Indoor 27.0%

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Metric Depth Estimation

26.6% uCO-3D

Tabletop 32.9%

(x, y, z)

25.3%

3D Coordination of Objects and Plane

[Figure 150]

[Figure 151]

[Figure 152]

Object Tracking & Segmentation

Key Frame Prompting

[Figure 153]

[Figure 154]

(b) Distribution of Scenes Categories (c) Distribution of Datasets

Objects Grounding

###### Step 3: Task Workflow

[Figure 155]

[Figure 156]

Frequency

Frequency

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Video Filtering & Frame Selection

[Figure 165]

Counting

Matching

[Figure 166]

[Figure 167]

[Figure 168]

Input: Video

###### Step 4: CoT Rationale

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Task-specific Few-shot Prompting

Distance

Relation

[Figure 176]

[Figure 177]

Response

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

1e+3

1e-3 1e-2 1e-1 1 1e+1 1e+2 1e-1 1 1e+1 1e+2

Video &

Fit Answer?

[Figure 185]

Advanced MLLM

(d) Distribution of Object Size (e) Distribution of Object-camera Distance

Input: Cam. Param.

Question

Direction Plan

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Yes

No

[Figure 190]

free reg mc

Template based GPT based

Question Answer Rationale

[Figure 191]

[Figure 192]

[Figure 193]

Discard

[Figure 194]

Question & Answer

[Figure 195]

[Figure 196]

###### Data Evaluation Benchmark

[Figure 197]

[Figure 198]

Normal Map Metric Depth

Bounding Box

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Measure & Record Retrieval

[Figure 203]

Expert Annotation &

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Object ID

Key Frames 3D Coordination

Statue of Benjamin Harrison

Official Info

…

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

Manual Curation

[Figure 212]

…

Camera Parameter Mask Category

(f) Word Cloud of Outdoor Scenes (g) Word Cloud of Tiny Tabletop Scenes

(a) Data Construction Pipeline

- Figure 3. Fig.(a) shows our automated data construction pipeline. The pie charts (b-c) depict the composition of scenes and sources. The bar charts (d–e) show object sizes ranging mm-100m, while object-to-camera distances typically span 10-600m. Accordingly, we claim SpaceVista-1M basically covers the mm-km scale. The word clouds (f-g) provide a glimpse of the scene diversity.

Table 1. Comparison of popular spatial reasoning datasets. Only spatial reasoning QA is included. Lower QA/Scene Ratio usually means more diverse language and visual scenes. “free”,“reg”, and “mc” mean free-form, regression, and multiple-choice, respectively. SpaceVista-1M does not differentiate QA pairs by the type; i.e., the semantically similar questions with reg/mc/free answers are counted only once. The citations here are listed in Appendix §B.5 for conciseness.

ing question–answer pairs, with rich semantic (category, rationale), 2D (mask, box, point), and 3D (depth, camera parameters, point cloud) annotations, obtained either natively or through processing. The construction pipeline in Fig. 3 follows the step-by-step procedure of preparing, transforming, and generating to obtain an all-scale dataset by integrating specialized models.

QA Pairs↑

Video Scenes↑

QA/Scene Ratio↓

Usage Dataset Type

Data Preparation. We begin by selecting widely used video datasets that provide 3D scene modeling (Ling et al., 2024; Xia et al., 2024; Park et al., 2020; Liu et al., 2025a; Dai et al., 2017; Yeshwanth et al., 2023) along with camera intrinsic and extrinsic parameters. Most of these sources are videos of static scenes without moving objects. Leveraging the known camera parameters, we estimate depth maps and normal maps using specialized metric depth models (Hu et al., 2024; Piccinelli et al., 2025) and video depth models (Chen et al., 2025). For semantic understanding, we extract per-frame semantics and bounding boxes using proprietary grounding specialists (Ren et al., 2024; Liu et al., 2023). To establish cross-frame object consistency, by further integrating SAM 2 (Ravi et al., 2024) with the previously mentioned grounding experts, we enable robust object ID association and mask generation. This pipeline ensures both semantic and spatial consistency across frames.

SpaceR reg/mc 191K 1.2K 159 SPAR-7M reg/mc/free 7M 4.5K 1,556 Spatial-MLLM reg/mc/free 120K 1.5K 83 InternSpatial free 2.5M 5.5K 455 SpaceVista-1M (Ours)

Train

free/reg/mc 1M 38K 25

TempCompass mc 7.5K 0.4K 18 VideoMME mc 2.7K 0.9K 3 All-Angles mc 2.1K 90 23 VSI-Bench reg/mc 5.0K 0.3K 17 MMSI-Bench mc 1.0K - SPAR-Bench reg/mc 7.2K - STI-Bench mc 2.0K 0.3K 7 SpaceVista-

Benchmark

reg/mc 1.6K 0.3K 5

-Bench (Ours)

manipulation and drone-view area estimation. Taking object counting as an example, the workflow follows: detect objects, propagate masks across frames, track identities over time, filter out scenes with camera parameters and ambiguous objects, and derive temporally consistent counts. For each task, we obtain the data by using similar, carefully designed computational workflows.

Task Construction. With the help of official camera parameters and the preparations mentioned above, we can obtain the positions and dimensions of target objects. As a common practice (Deng et al., 2025b), we adopt a canonical view space of the reference frame, defined as a 3D Cartesian coordinate system centered at the camera’s optical center. We then design 19 tasks and their corresponding workflows, even including scale-specific tasks such as tabletop object

QA Construction. The pipeline for constructing the QA data is shown in Fig. 3. At the construction level of QA, we employ two strategies: GPT-based and template-based. For relatively fixed questions such as counting and object size, we adopt a template-based approach to obtain reasonable

QA pairs. To ensure the diversity of the questions, we manually curate over 3,000 templates. However, for more flexible questions like planning, we use a GPT-based (OpenAI, 2025) method to generate reasonable answers in natural language. Additionally, through appropriate randomizing and prompting, we obtain multiple options to serve as rewards for RL.

CoT Annotation. To facilitate an efficient cold start, we follow Feng et al. (2025) to leverage cognitioninspired few-shot prompting strategy with Qwen2.5-VL72B-Instruct (Bai et al., 2025) to generate CoT rationales. After employing the filtering policy for low-quality or inconsistent rationale outputs, we obtain the CoT for SpaceVista1M, with high-quality rationale for fundamental knowledge injection for SFT.

Input Extension. Usually, people refer to objects in videos using more than just language. To support this, we extend video-based QA with extra annotations from the video’s key frames. Besides plain visual input, we allow three extra inputs: point, bounding box, and mask, which may support future interactive usage. Each input type is designed to fit its own template and CoT rationales.

Quality Control & Evaluation. 1) SpaceVista-1M (training): To ensure data quality, we manually verify a small portion of the training set for quality control, achieving an average global accuracy of approximately 83%. 2) SpaceVistaBench (evaluation): We choose the reliable pathway for benchmark based on measuring and recording real-world data, retrieving authoritative sources, and performing human annotation for both distance and non-distance problems, shown in the green block Fig.4(a). For tiny and tabletop scenes, we capture and annotate videos of over 50 objects of different sizes. For some indoor and outdoor scenes, we search for the landmarks and retrieve statistics from authoritative sources like Wikipedia. As for other tasks like camera moving, the experts are hired for checking and annotating. By aligning the answer with the physical world, SpaceVista-Bench comprises about 1,600 QA pairs across approximately 300 unique video scenes, with quality ensured through manual review and source verification. Due to the scarcity of public data, the exploration in drone scenarios remains preliminary.

More Information on Dataset and Benchmark: We encourage readers to consult the appendix for more details.

- • Source investigations in Sec. § B.2.
- • Benchmark collection in Sec. §B.2.7.
- • Data preparation in Sec. §B.3.1.
- • Tasks and workflows in Sec. §B.3.
- • In-depth distribution analysis in Sec. §B.4.
- • Data quality control in Sec. §B.4.2.
- • License in Sec. §B.4.3.

In summary, we propose SpaceVista-1M, an open-source, real-world, all-scale dataset with spatial video QA. SpaceVista-1M contains 1 million QA pairs spanning 19 tasks, 5 scale types, and over 50 subscene categories.

### 4. Method

Overview. Our objective is to enhance spatial reasoning by elaborately designing and conditioning the model on explicit and detailed all-scale information. We first utilize a dense, expressive self-supervised encoder beyond semantics to strengthen the model’s overall spatial perception. However, mixing different types of knowledge without distinction hinders, rather than facilitates, the model’s reasoning in Fig. 4(a-d), a problem known as knowledge conflict. In all-scale reasoning, this conflict appears when similar visual patterns are interpreted differently at different scales. To mitigate such conflict, we propose a LoRA-like scale expert architecture to maintain the independence of scalelevel knowledge, while maintaining parameter efficiency, as shown in Fig 4(e). Finally, drawing on human reasoning about scale, we introduce reward-based progressive reasoning paths that employ essential anchors to constrain the reasoning process to a reliable CoT path.

Preliminaries. The number of frames is first denoted as T with the temporal patch size τ. The visual representations from Qwen-2.5-VL visual encoder are denoted as FV ∈ Rt×d

V ×H×W, where t = Tτ is temporal dimension of the feature, dV is the feature dimension per patch, and H and W are the numbers of patches p along the height and width of each frame, respectively. Then, each i ∈ t × dV of FV is directly converted to an image token TVi as input.

Beyond Semantics. Most open-sourced MLLM tokenizers including Qwen-2.5-VL visual encoder are pretrained on semantically rich text–image pairs via contrastive training, and thus often lack a well-formed understanding of information beyond semantics. Meanwhile, El Banani et al. (2024); Tong et al. (2024b;a) draw a valuable conclusion that self-supervised vision models, such as DINO series, learn rich depth, normal, and pattern representations. Therefore, leveraging popular DINOv3 (Siméoni et al., 2025)’s strong dense features seems to be a natural approach beyond simple semantics. The last layer of DINOv3 produces patchlevel dense features FD ∈ RT×d

D×HD×WD. We pad and regularize the original image to align with the patch size p, enforcing HD =H and WD =W. We then apply a simple MLP, Rd

V , to map channel dimensions. For the temporal dimension, we use the same temporal pooling with the previously mentioned temporal patch size τ to aggregate across T, yielding features FD′ ∈ Rt×d

→ Rd

D

V ×H×W. The fusion of the video feature FV and dense feature FD′ is shown as:

FV′ = CA(FV ,FD′ ,FD′ ) + FV (1) where CA(q,k,v) denotes multi-layer cross-attention over

[Figure 213]

[Figure 214]

[Figure 215]

###### SpaceVista Model

Tiny

Human Thinking Process

Not Enough

RatioRatioRatioRatio

[Figure 216]

[Figure 217]

Indoor

What is the distance between the red dot object and the

Scenes

Depth

[Figure 218]

[Figure 219]

[Figure 220]

𝐹

𝐹

𝐹 Semantics

Relations

Normal

Text Tokenizer

Question

small tool? GT: 6cm

Geometry

Perfect for All-Scale Reasoning

- (a) Training on indoor data

[Figure 221]

- (b) Training on tiny tabletop data

[Figure 222]

- (c) Straightforward training on both
- (d) Scale expert training on both

…

Pattern …

- 1. Identify Object Semantics as reference.
- 2. Identify Scale Scene. 3. Imagine the Space.

[Figure 223]

[Figure 224]

[Figure 225]

4. Give final answer.

Video

[Figure 226]

[Figure 227]

Tiny

𝐹

[Figure 228]

###### Proj. LLM

Patch Feature

LLM Visual Encoder

Not Enough

Indoor

[Figure 229]

[Figure 230]

Progressive Reward

𝐹

[Figure 231]

Patch-level Learnable

[Figure 232]

General Reasoning Spatial Reasoning

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

Fusion Proj.

Video Self-supervised Encoder

𝐹

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Patch Feature

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Lora-Like Scale Experts

[Figure 260]

[Figure 261]

Scale

[Figure 262]

Semantic

scale discrepancy

Knowledge Conflict?

SpaceVista: The red dot appears to be a <semantic>toy bear<\semantic>. The overall scene is a view of the tabletop objects and around <scale>10 cm level<\scale> …, The size of the small metal tool… on the round table. According to …, the distance is around <answer>6.3<\answer> cm.

Tiny

[Figure 263]

Indoor

LORA-like Scale Experts

[Figure 264]

Trainable

[Figure 265]

Frozen

[Figure 266]

Input

Output

Pretrained Weights 𝑾𝟎

Correct

[Figure 267]

[Figure 268]

×

𝐴

𝐵

𝛼 ⋅ 𝜆

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

###### Examples

Wrong

[Figure 274]

scale discrepancy

……

×

+

[Figure 275]

[Figure 276]

𝐴

𝐵

Better!

Tiny

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Semantic Recog.

Special Format:

𝛼 ⋅ 𝜆

Indoor

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

Scale Estimation

<scale></…> <semantic></…> <answer></…>

[Figure 287]

×

𝐴

𝐵

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Result Reward

ℎ

𝛼 ⋅ 𝜆

𝑥

[Figure 293]

Scale-aware router 0.5

Normalized Reward 0 0 0.8 1

(e) The pipeline of our SpaceVista model

- Figure 4. The left part (a-d) shows that the undifferentiated mixture of cross-scale knowledge hinders, rather than facilitates, the model’s

reasoning process. The horizontal axis represents the scale discrepancy, defined as answergt (=1 for the ideal situation), and the vertical axis denotes the proportion of answers. Fig.(e) is our SpaceVista model, where “<think>” is omitted for clarity.

the query, key, and value inputs. Then, we convert FV′ into a fused image token TVi , and the remaining calculations proceed as before.

how humans approach spatial observation tasks, we model the reasoning process explicitly. Humans typically proceed by: 1) identifying the task-specified semantics (if they help), 2) perceiving the global scale by inspecting surrounding objects (if it helps), and 3) inferring the answer from spatial relations. Following this paradigm, we construct 3 different anchors for RL that enforce the reasoning path to traverse the resulting anchor states.

Scale Experts Design. During all-scale mixed training in Fig.4(a-d), potential cross-scale knowledge conflicts lead to suboptimal results. This underscores the importance of preserving knowledge independence between scales during training. Inspired by Wu et al. (2024); Buehler & Buehler (2024); Chen et al. (2024b), we further introduce a LoRAlike module that adds scale experts by fine-tuning only 0.5% of the overall parameters for each expert. The original LoRA is using B ∈ Rd×r and A ∈ Rr×d with the rank r ≪ min(d,k) to approximate original weights W0. To construct scale LoRA experts, We attach M scale experts {(Ai,Bi)}Mi=1 to mitigate potential scale-level knowledge interference. Each expert i has a base weight αi and is dynamically scaled by a learned factor λi:

With the minimal, sufficient ground-truth anchors, we design the following three reward components based on these anchor formats: <semantics>, <scale>, and <answer>. Semantic reward Rsemantic is used to identify the referenced objects; Scale reward Rscale is used to estimate the scale of the overall scene; Correctness reward Ranswer is used to ensure the answer is well derived. The updated correctness reward R¯answer can be formed into

3

k

M

R¯answer =

Rj

,

αi∗ BiAix,where αi∗ = αi · λi, (2)

h = W0x +

n

n=1

k=1

i=1

(3)

|log Cans − log Cgt| 2

where Rscale = max 0,1 −

,

where x, h are the input and output of the projection layer, and αi∗ is the scaled factor. The learned factor λi is obtained through a scale router, primarily an MLP and a softmax. We apply M scale experts to each layer of the foundation LLM. Therefore, different layers, according to their respective conditions, obtain appropriate λi to allocate the experts within the layer. Given that scenarios of scales can overlap (for example, an indoor scene may include some tabletop context), in the ideal case, the routers can select the suitable experts at different layers.

SansSgt ∥Sans∥∥Sgt∥

Rsemantic =

,

with (j1,j2,j3) = (answer,scale,semantic) Cans,Cgt is the estimated scene scale in the same measurement; Sans,Sgt is the calculated semantic embedding. Cgt and Sgt can be easily obtained from Sec.§3. It is crucial to note that the order of (j1,...,jn) matters; rewards at the beginning are stricter and more important. Also, because tasks differ, for example in the camera rotation task, Rsemantic and Rscale are not needed. Thus, R¯answer under such circumstances collapses to a standard Ranswer. The calculation of format

Process Reward Design. After basic SFT training, RL is used to align the model with human perception. Inspired by

- Table 2. Performance comparison across five spatial reasoning benchmarks. Open-sourced general models are evaluated with a comparable size. The highest performance of the open-sourced model is marked bold.

Multi-Image Video Model MMSI-Bench SPAR-Bench VSI-Bench STI-Bench SpaceVista-Bench Human 97.2 67.3 79.2 - 81.3

Closed-Sourced Commercial Models & 70B-Class Models

GPT-5 (OpenAI, 2025) 40.7 37.4 44.2 39.3 33.7 Gemini-2.5-pro (DeepMind, 2025) 36.9 36.3 45.0 41.4 33.8 InternVL3.5-38B (Wang et al., 2025c) 36.9 31.0 66.3 39.2 30.7 Qwen2.5-VL-72B (Bai et al., 2025) 30.7 32.4 30.7 40.7 31.1

Open-Sourced General Models

LLAVA-Onevision-7B (Li et al., 2024a) 24.5 30.6 32.4 29.0 13.6 LLaVA-NeXT-Video-7B (Liu et al., 2024a) 26.8 31.3 35.6 29.9 23.7 InternVL3.5-8B (Wang et al., 2025c) 30.9 36.0 38.2 33.2 24.5 Qwen2.5-VL-7B (Bai et al., 2025) 31.7 33.1 32.7 32.1 28.9

Open-Sourced Specialized Models

SpaceR-7B (Ouyang et al., 2025) 26.1 37.6 46.9 37.0 21.2 SpatialMLLM-4B (Wu et al., 2025c) 27.0 31.5 48.4 30.5 24.2 VILASR-7B (Wu et al., 2025d) 30.2 37.6 45.4 31.5 23.6 VG LLM-4B (Zheng et al., 2025) - - 46.1 29.3 28.8

Qwen2.5-VL-7B w/. SpaceVista-1M 27.3 36.9 42.0 35.0 29.5 SpaceVista-7B (Ours) 29.1 38.1 46.3 35.9 35.4 SpaceVista-7B (Ours) w/. RL 32.3 41.6 48.6 38.2 39.2

reward Rformat and answer reward Ranswer remains the same as common practice (Feng et al., 2025; Guo et al., 2025) to encourage the generation of valid and executable answers. Therefore, our reward design forms the accurate reward signals to ensure all-scale spatial compliance and encourage human-like thinking. It is worth noting that the evaluation does not involve these anchors besides the actual answer.

RL Training Objective. For each question i, we define the reward Ri to include both the updated correctness reward R¯answer and Rformat following Guo et al. (2025), and use this overall reward Ri to compute groupwise normalized advantages Ai = Ri−std(mean({R{Rj})

j}) . {Rj} is the response group related to Ri. The final policy πθ is updated by maximizing

G

πθ(oi | q) πθ

1 G

J(θ) =Eq,{o

Ai, clip

min

i}

(oi | q)

old

i=1

πθ(oi | q) πθ

,1 − ϵ, 1 + ϵ Ai − β DKL(πθ ∥πref) ,

(oi | q)

old

(4)

where πθ

and πθ are the old and new policy model respectively. DKL represents KL divergence.

old

Training Strategy. We start with a cold-start phase on SpaceVista-1M, optimizing the input projection, featurefusion modules, and scale experts. Next, we introduce the scale router to further train each scale-specific expert on the appropriate inputs, encouraging specialization. Finally, building on the SFT model, we apply RL training to obtain the final SpaceVista-7B reasoning model.

### 5. Experiment

Datasets. We use SpaceVista-1M in Sec.§ 3 for SFT and RL; its sources are detailed in Appendix.§ B.2.

Model Configurations. Our model is built on Qwen2.5-VL7B for main experiments and Qwen2.5-VL-3B for ablation. Our model is trained on up to 16 NVIDIA A800 (80GB) GPUs. We process a maximum of 32 frames during training, each with a resolution of 128 × 28 × 28 pixels. During inference, we increase the resolution (256 × 28 × 28 pixels) to enhance performance. During the expert training phase, we employ 4 experts, each tailored to a distinct scenario. We set the group size of GRPO to 8. The training data already includes four scale and scene labels from the construction process. These labels are used to divide the data, and each expert is trained on the data that matches its scale. We first perform SFT on CoT data of SpaceVista-1M for two epochs to obtain the SFT model. This is followed by RL training for 2.5k steps to produce the final SpaceVista-7B checkpoints with detailed information shown in Appendix.§ C.1. The semantic embeddings are computed using all-MiniLM-L6v2 with the Sentence-Transformer Toolkit.

Comparison on Spatial Reasoning Datasets. Our method attains competitive performance across all spatial reasoning benchmarks in Tab. 2. On VSI-Bench, we achieve comparable results approaching the state of the art. More importantly, our approach delivers substantially superior performance in our all-scale benchmark SpaceVista-Bench, markedly exceeding 3% compared with proprietary and open-source models. Thus, SpaceVista-1M represents a robust baseline for both indoor and all-scale scenes.

- Table 3. Ablation study on 3B-based SpaceVista model trained in the same condition. We analyze the impact of different modules (top) and extra input modalities (bottom).

Setting VSI-Bench SpaceVista-Bench

Vanilla SFT 44.4 31.0 Vanilla SFT w/. RL 44.9 32.1

Modality Ablation w/. SFT

w/. VGGT 44.3 (-0.1) 31.4 (+0.4) w/. DINOv3 46.4 (+2.0) 32.1 (+1.1) w/. VGGT + DINOv3 45.3 (+0.9) 31.7 (+0.7)

Module Ablation w/. RL

w/. Scale 46.3 (+1.4) 34.8 (+2.7) w/. Scale + Semantic 46.8 (+1.9) 35.4 (+3.3) w/. Expert Finetuning 45.8 (+0.9) 34.8 (+2.8)

- Table 4. Ablation of the number of experts based on the same SFT settings on the 3B model. M = 4 is the default setting.

Num of Expert(s) (M)

SpaceVista -Bench (Ours) None All 44.4 31.0

Training Data (Each Expert)

VSIBench

- 1 All 44.2 (-0.2) 31.0 (0)
- 2 1/2 46.1 (+1.9) 33.1 (+2.1) 4 1/4 46.3 (+2.1) 34.7 (+3.7) 6 1/6 43.1 (-1.1) 26.7 (-4.3)

Benchmarks. We evaluate our model on 5 benchmarks, VSI-Bench , STI-Bench , SpaceVista-Bench (Ours), MMSIBench and SPAR-Bench . Among the benchmarks, the former three are video-based, while the latter two are multiimage benchmarks. We argue that video and multi-image tasks share rather strong similarities and collectively serve as benchmarks for cross-frame understanding. For all evaluations, we follow the configuration used in the official Qwen demo with topp = 0.001 and temperature = 0.01.

Comparison on Spatial Reasoning Datasets. Our method attains competitive performance across all spatial reasoning benchmarks in Tab. 2. On VSI-Bench, we achieve comparable results approaching the state of the art. More importantly, our approach delivers substantially superior performance in all-scale benchmark SpaceVista-Bench, markedly exceeding 3% on average compared with proprietary and open-source models. Thus, SpaceVista-1M represents a robust baseline for both indoor and all-scale scenes.

Comparison on Subsets of SpaceVista-Bench. In Tab.5, we analyze the performance of popular models on each subset of our SpaceVista bench. In general, the small-scale subsets challenge both commercial and general models, likely due to biases in the pretraining corpus.

We also observe that most models perform at a relatively low level on SpaceVista-Bench, indicating that it has the expected discriminative ability for all-scale reasoning and can serve as a foundational benchmark to help the community enrich the overall evaluation ecosystem. Our SpaceVista-7B, although exhibiting minor improvements on indoor scenes,

Table 5. The SpaceVista-Bench leaderboard. We utilize

green (1st) , blue (2nd) , and yellow (3rd) backgrounds to distinguish the top three results within each scene. We employ bold and underlined text to denote the best and second-best results across all open-source models. All the baselines are instructiontuned and are evaluated on the same resolution and fps. The citations here are listed in Appendix §B.5.

SpaceVista-Bench Models Tiny Tabletop Indoor Outdoor Overall

Closed-sourced Commercial Models

GPT-5 32.3 20.3 39.0 43.0 33.7 GPT-4o 21.7 13.3 34.3 38.3 26.9

[Figure 294]

Gemini-2.5-pro 33.0 38.7 34.5 29.0 33.8 Gemini-2.5-flash 20.7 30.0 19.9 26.9 24.4 Claude-Sonnet-4 27.3 19.3 38.1 34.1 29.7 Claude-Opus-4.1 21.7 29.5 24.3 30.0 26.4

[Figure 295]

Open-Source General Models

Internvl3.5-38B 29.3 25.2 41.2 27.0 30.7 Internvl3.5-14B 27.7 22.3 31.3 24.3 26.4

Internvl3-78B 38.3 23.3 42.2 30.3 33.5 Internvl3-38B 18.7 14.3 34.8 38.0 26.5

GLM-4.5V 23.0 17.8 27.3 25.2 23.3 GLM-4.1V-Thinking 30.7 19.3 29.0 13.3 23.1

Qwen2.5VL-72B 27.7 20.3 29.6 28.0 26.4 Qwen2.5VL-32B 25.3 19.3 38.1 30.7 28.4

LLAVA-Onevision-72B 25.0 12.0 15.3 11.7 16.0 LLAVA-Onevision-7B 17.5 8.0 13.3 11.6 12.6

Open-Source Specialized Models

SpaceR 12.9 17.3 34.9 19.8 21.2 Spatial-MLLM 17.3 20.3 36.1 23.1 24.2

VLM-3R 15.1 24.6 45.1 26.9 27.9 SpaceVista-

35.3 38.2 44.1 39.1 39.2

[Figure 296]

-7b (Ours)

attains comparatively high comprehensive scores across other scenarios and in overall evaluations. The results indicate a clear boost of around 6% compared with any size of the open-source models in comprehensive all-scale spatial reasoning.

Ablation on Each Component. 1) Scale Expert: We examine how potential information conflicts during cross-scale training are mitigated. As shown in Tab.3, the experts yield substantial gains. The ablation on the number of experts in Tab. 4. However, more routers are not always better; increasing the number of experts places greater pressure on the router and leads to load imbalance. 2) Reward: In Tab. 3, the progressive reward achieves higher performance than the unconstrained reasoning path. These optional anchors indeed serve as a valuable halfway point in the all-scale reasoning process. This highlights the importance of specifying thinking anchors when designing all-scale reasoning. 3) Modality: As shown in Tab. 3, incorporating DINO v3 yields greater gains than VGGT with its advantage of semantically dense cues.

More Experiments. We provide more visualization, experiments, and analysis in the appendix, for example,

- • Patch-level encoder ablation in Sec. § C.2;
- • All-Scale observation and insights in Sec. §C.5, C.6;

• Out-of-distribution analysis on customized data including Guinness Records in Sec. §C.4 and Fig. C18.

### References

Anthropic. Introducing claude 4. https://www.anthropic.com/news/claude-4, 2025a. Accessed: 2025-07-24.

### 6. Conclusion

In this work, we introduce a novel task for all-scale reasoning from visual spatial context, which requires the machine to understand multimodal information and respond with the correct answer and rationale. To advance this field, we develop the first open-source, all-scale, spatial reasoning dataset, SpaceVista-1M, for cold start and reinforcement learning. Then, we handcraft SpaceVista-Bench, an accurate, multi-scale, video-based benchmark that adheres to physical world measurements and perceptions. During experiments, we compare our SpaceVista-7B model with popular models and demonstrate our proposed model’s promising performance as a robust baseline in all-scale reasoning.

### Acknowledgement

This work is partially supported by the National Natural Science Foundation of China (No. 62306261), HK RGC-Early Career Scheme (No. 24211525), ITSP Platform Project (No. ITS/600/24FP) and the SHIAE Grant (No. 8115074). This study was supported in part by the Centre for Perceptual and Interactive Intelligence, a CUHK-led InnoCentre under the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government. This work is also partially supported by Hong Kong RGC Strategic Topics Grant (No. STG1/E-403/24N), a CUHK-CUHK(SZ)-GDST Joint Collaboration Fund (No. YSP26-4760949), and a University of Sydney-Chinese University of Hong Kong Ignition Grant (No. G232965). This work is also partially supported by Astribot Inc. Special thanks to Xinran Chen for her crucial and diligent data collection.

### Impact Statement

We anticipate that SpaceVista will catalyze widespread advancements across diverse domains by enabling robust allscale spatial reasoning. Beyond enhancing core capabilities like spatial captioning, guided visual generation, and interactive world models, its impact extends to critical real-world applications ranging from the micro-scale—such as precision manufacturing (µm) and medical surgery (mm)—to the macro-scale of remote sensing (km) and cartography (10km). Ultimately, this work provides a foundational step for industrial automation, embedded systems, and autonomous driving, empowering intelligent agents to perceive and reason complex, unconstrained environments in the wild.

Anthropic. Claude opus 4.1. https://www.anthropic.com/news/claude-opus-4-1, 2025b. Version VIII, released in 2025.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu,

- Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang,
- Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Baruch, G., Chen, Z., Dehghan, A., Dimry, T., Feigin, Y., Fu, P., Gebauer, T., Joffe, B., Kurz, D., Schwartz, A., et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021.

Buehler, E. L. and Buehler, M. J. X-lora: Mixture of lowrank adapter experts, a flexible framework for large language models with applications in protein mechanics and molecular design. APL Machine Learning, 2(2), 2024.

Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., and Xia, F. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455–14465, 2024a.

Chen, S., Jie, Z., and Ma, L. Llava-mole: Sparse mixture of lora experts for mitigating data conflicts in instruction finetuning mllms. arXiv preprint arXiv:2401.16160, 2024b.

Chen, S., Guo, H., Zhu, S., Zhang, F., Huang, Z., Feng, J., and Kang, B. Video depth anything: Consistent depth estimation for super-long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22831–22840, 2025.

Chen, Y., Wu, Q., Li, M., Lin, W., Harandi, M., and Cai, J. Fast feedforward 3d gaussian splatting compression. arXiv preprint arXiv:2410.08017, 2024c.

Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., and Shan, Y. Yolo-world: Real-time open-vocabulary object detection. In Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2024.

Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., and Nießner, M. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017.

DeepMind, G. Gemini 2.5: Our most intelligent ai model. https://blog.google/technology/ google-deepmind/, 2025.

Deng, K., Ti, Z., Xu, J., Yang, J., and Xie, J. Vggtlong: Chunk it, loop it, align it–pushing vggt’s limits on kilometer-scale long rgb sequences. arXiv preprint arXiv:2507.16443, 2025a.

Deng, N., Gu, L., Ye, S., He, Y., Chen, Z., Li, S., Wang, H., Wei, X., Yang, T., Dou, M., et al. Internspatial: A comprehensive dataset for spatial reasoning in vision-language models. arXiv preprint arXiv:2506.18385, 2025b.

Dongfang, Z., Zheng, X., Weng, Z., Lyu, Y., Paudel, D. P., Van Gool, L., Yang, K., and Hu, X. Are multimodal large language models ready for omnidirectional spatial reasoning? arXiv preprint arXiv:2505.11907, 2025.

El Banani, M., Raj, A., Maninis, K.-K., Kar, A., Li, Y., Rubinstein, M., Sun, D., Guibas, L., Johnson, J., and Jampani, V. Probing the 3d awareness of visual foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21795– 21806, 2024.

Fan, Z., Zhang, J., Li, R., Zhang, J., Chen, R., Hu, H., Wang, K., Qu, H., Wang, D., Yan, Z., et al. Vlm-3r: Visionlanguage models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025.

Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T., Wu, J., Zhang, X., Wang, B., and Yue, X. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

Ghazanfari, S., Croce, F., Flammarion, N., Krishnamurthy, P., Khorrami, F., and Garg, S. Chain-of-frames: Advancing video understanding in multimodal llms via frameaware reasoning. arXiv preprint arXiv:2506.00318, 2025.

GLM, T., Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D., Rojas, D., Feng, G., Zhao, H., Lai, H., Yu, H., Wang, H., Sun, J., Zhang, J., Cheng, J., Gui, J., Tang, J., Zhang, J.,

- Li, J., Zhao, L., Wu, L., Zhong, L., Liu, M., Huang, M., Zhang, P., Zheng, Q., Lu, R., Duan, S., Zhang, S., Cao, S., Yang, S., Tam, W. L., Zhao, W., Liu, X., Xia, X., Zhang,

- X., Gu, X., Lv, X., Liu, X., Liu, X., Yang, X., Song, X., Zhang, X., An, Y., Xu, Y., Niu, Y., Yang, Y., Li, Y., Bai,
- Y., Dong, Y., Qi, Z., Wang, Z., Yang, Z., Du, Z., Hou,
- Z., and Wang, Z. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024.

Guizilini, V., Ambrus, R., Pillai, S., Raventos, A., and Gaidon, A. 3d packing for self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2485–2494, 2020.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Gupta, A., Dollár, P., and Girshick, R. B. Lvis: A dataset for large vocabulary instance segmentation. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5351–5359, 2019.

Han, S., Huang, W., Shi, H., Zhuo, L., Su, X., Zhang, S., Zhou, X., Qi, X., Liao, Y., and Liu, S. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26181–26191, 2025.

Hu, M., Yin, W., Zhang, C., Cai, Z., Long, X., Chen, H., Wang, K., Yu, G., Shen, C., and Shen, S. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

LaValle, S. Rapidly-exploring random trees: A new tool for path planning. Research Report 9811, 1998.

- Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.
- Li, C., Wu, W., Zhang, H., Xia, Y., Mao, S., Dong, L., Vuli´c, I., and Wei, F. Imagine while reasoning in space: Multimodal visualization-of-thought. arXiv preprint arXiv:2501.07542, 2025a.
- Li, D., Li, H., Wang, Z., Yan, Y., Zhang, H., Chen, S., Hou, G., Jiang, S., Zhang, W., Shen, Y., et al. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint

- arXiv:2505.21500, 2025b.

Li, H., Han, S., Liao, Y., Luo, J., Gao, J., Yan, S., and Liu, S. Reinforcement learning tuning for videollms: Reward design and data efficiency. arXiv preprint

- arXiv:2506.01908, 2025c.

- Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Li, Y., Zhang, Y., Lin, T., Liu, X., Cai, W., Liu, Z., and Zhao, B. Sti-bench: Are mllms ready for precise spatial-temporal world understanding? arXiv preprint arXiv:2503.23765, 2025d.

Liao, Z., Xie, Q., Zhang, Y., Kong, Z., Lu, H., Yang, Z., and Deng, Z. Improved visual-spatial reasoning via r1-zerolike training. arXiv preprint arXiv:2504.00883, 2025.

Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al. Dl3dv-10k: A largescale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22160–22169, 2024.

Liu, F., Lu, Z., and Lin, X. Vision-based environmental perception for autonomous driving. Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering, 239:39 – 69, 2022.

Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL https://llava-vl.github.io/blog/ 2024-01-30-llava-next/.

Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.

- Liu, X., Tayal, P., Wang, J., Zarzar, J., Monnier, T., Tertikas, K., Duan, J., Toisoul, A., Zhang, J. Y., Neverova, N., Vedaldi, A., Shapovalov, R., and Novotny, D. Uncommon objects in 3d. In arXiv, 2025a.
- Liu, Y., Li, S., Liu, Y., Wang, Y., Ren, S., Li, L., Chen, S., Sun, X., and Hou, L. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv: 2403.00476, 2024b.

- Liu, Y., Chi, D., Wu, S., Zhang, Z., Hu, Y., Zhang, L., Zhang, Y., Wu, S., Cao, T., Huang, G., et al. Spatialcot: Advancing spatial reasoning through coordinate alignment and chain-of-thought for embodied task planning. arXiv preprint arXiv:2501.10074, 2025b.
- Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., and Wang, J. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025c.

Ma, W., Chou, Y.-C., Liu, Q., Wang, X., de Melo, C., Xie, J., and Yuille, A. Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. arXiv preprint arXiv:2504.20024, 2025.

Nikouei, M., Baroutian, B., Nabavi, S., Taraghi, F., Aghaei, A., Sajedi, A., and Moghaddam, M. E. Small object detection: A comprehensive survey on challenges, techniques and real-world applications. ArXiv, abs/2503.20516, 2025.

OpenAI. Introducing gpt-4.5. https://openai.com/ index/introducing-gpt-4-5/, 2025.

OpenAI. GPT-5 System Card. Technical report, OpenAI, August 2025. Accessed: 2025-08-10.

Ouyang, K., Liu, Y., Wu, H., Liu, Y., Zhou, H., Zhou, J., Meng, F., and Sun, X. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

Pan, M., Zhang, J., Wu, T., Zhao, Y., Gao, W., and Dong, H. Omnimanip: Towards general robotic manipulation via object-centric interaction primitives as spatial constraints. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17359–17369, 2025.

Park, K., Patten, T., and Vincze, M. Neural object learning for 6d pose estimation using a few cluttered images. In The European Conference on Computer Vision (ECCV), 2020.

Piccinelli, L., Sakaridis, C., Yang, Y.-H., Segu, M., Li, S., Abbeloos, W., and Gool, L. V. UniDepthV2: Universal monocular metric depth estimation made simpler, 2025. URL https://arxiv.org/abs/2502.20110.

Qi, Z., Zhang, Z., Fang, Y., Wang, J., and Zhao, H. Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv preprint arXiv:2501.01428, 2025.

Qiao, R., Tan, Q., Yang, P., Wang, Y., Wang, X., Wan, E., Zhou, S., Dong, G., Zeng, Y., Xu, Y., et al. Wemath 2.0: A versatile mathbook system for incentivizing visual mathematical reasoning. arXiv preprint arXiv:2508.10433, 2025.

Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., and Feichtenhofer, C. Sam 2: Segment anything in images and videos, 2024.

Ren, T., Chen, Y., Jiang, Q., Zeng, Z., Xiong, Y., Liu, W., Ma, Z., Shen, J., Gao, Y., Jiang, X., et al. Dino-x: A unified vision model for open-world object detection and understanding. arXiv preprint arXiv:2411.14347, 2024.

Schönberger, J. L. and Frahm, J.-M. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Schönberger, J. L., Zheng, E., Pollefeys, M., and Frahm, J.-M. Pixelwise view selection for unstructured multiview stereo. In European Conference on Computer Vision (ECCV), 2016.

Silberman, N., Hoiem, D., Kohli, P., and Fergus, R. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pp. 746–760. Springer, 2012.

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Song, C. H., Blukis, V., Tremblay, J., Tyree, S., Su, Y., and Birchfield, S. T. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. ArXiv, abs/2411.16537, 2024.

Sun, X., Si, W., Ni, W., Li, Y., Wu, D., Xie, F., Guan, R., Xu, H.-Y., Ding, H., Wu, Y., et al. Autofly: Vision-languageaction model for uav autonomous navigation in the wild. arXiv preprint arXiv:2602.09657, 2026.

Tan, H., Ji, Y., Hao, X., Lin, M., Wang, P., Wang, Z., and Zhang, S. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025.

Tang, K., Gao, J., Zeng, Y., Duan, H., Sun, Y., Xing, Z., Liu, W., Lyu, K., and Chen, K. Lego-puzzles: How good are mllms at multi-step spatial reasoning? arXiv preprint arXiv:2503.19990, 2025.

Team, V., Hong, W., Yu, W., Gu, X., Wang, G., Gan, G., Tang, H., Cheng, J., Qi, J., Ji, J., Pan, L., Duan, S., Wang, W., Wang, Y., Cheng, Y., He, Z., Su, Z., Yang, Z., Pan, Z., Zeng, A., Wang, B., Chen, B., Shi, B., Pang, C., Zhang, C., Yin, D., Yang, F., Chen, G., Xu, J., Zhu, J., Chen, J., Chen, J., Chen, J., Lin, J., Wang, J., Chen, J., Lei, L., Gong, L., Pan, L., Liu, M., Xu, M., Zhang, M., Zheng, Q., Yang, S., Zhong, S., Huang, S., Zhao, S., Xue, S., Tu, S., Meng, S., Zhang, T., Luo, T., Hao, T., Tong, T., Li,

- W., Jia, W., Liu, X., Zhang, X., Lyu, X., Fan, X., Huang,
- X., Wang, Y., Xue, Y., Wang, Y., Wang, Y., An, Y., Du,
- Y., Shi, Y., Huang, Y., Niu, Y., Wang, Y., Yue, Y., Li, Y., Zhang, Y., Wang, Y., Wang, Y., Zhang, Y., Xue, Z., Hou,
- Z., Du, Z., Wang, Z., Zhang, P., Liu, D., Xu, B., Li, J., Huang, M., Dong, Y., and Tang, J. Glm-4.5v and glm4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025.

Tong, P., Brown, E., Wu, P., Woo, S., IYER, A. J. V., Akula, S. C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024a.

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024b.

Wang, J., Karaev, N., Rupprecht, C., and Novotny, D. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21686– 21697, 2024.

Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., and Novotny, D. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294–5306, 2025a.

Wang, Q., Yu, Y., Yuan, Y., Mao, R., and Zhou, T. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. arXiv preprint arXiv:2505.12434, 2025b.

- Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025c.
- Wang, X., Ma, W., Zhang, T., de Melo, C. M., Chen, J., and Yuille, A. Spatial457: A diagnostic benchmark for 6d spatial reasoning of large multimodal models, 2025d. URL https://arxiv.org/abs/2502.08636.

Wen, T., Wang, J., Chen, Y., Xu, S., Zhang, C., and Li, X. Metric-solver: Sliding anchored metric depth estimation from a single image. arXiv preprint arXiv:2504.12103, 2025.

Wu, D., Fu, Y., Huang, S., Liu, Y., Jia, F., Liu, N., Dai, F., Wang, T., Anwer, R. M., Khan, F. S., et al. Ragnet: Largescale reasoning-based affordance segmentation benchmark towards general grasping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11980–11990, 2025a.

Wu, D., Han, W., Liu, Y., Wang, T., Xu, C.-z., Zhang, X., and Shen, J. Language prompt for autonomous driving. In Proceedings of the AAAI conference on artificial intelligence, volume 39, pp. 8359–8367, 2025b.

Wu, D., Liu, F., Hung, Y.-H., and Duan, Y. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747, 2025c.

Wu, J., Guan, J., Feng, K., Liu, Q., Wu, S., Wang, L., Wu, W., and Tan, T. Reinforcing spatial reasoning in visionlanguage models with interwoven thinking and visual drawing, 2025d. URL https://arxiv.org/abs/ 2506.09965.

Wu, P., Liu, Y., Liu, M., and Shen, J. St-think: How multimodal large language models reason about 4d worlds from ego-centric videos. arXiv preprint arXiv:2503.12542, 2025e.

Wu, X., Huang, S., and Wei, F. Mixture of lora experts. arXiv preprint arXiv:2404.13628, 2024.

Xia, H., Fu, Y., Liu, S., and Wang, X. Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Xiao, J., Zhang, R., Zhang, Y., and Feroskhan, M. Visionbased learning for drones: A survey. IEEE transactions on neural networks and learning systems, PP, 2023.

Xu, R., Wang, W., Tang, H., Chen, X., Wang, X., Chu, F.-J., Lin, D., Feiszli, M., and Liang, K. J. Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models. arXiv preprint arXiv:2505.17015, 2025.

Xu, T. Recent advances in rapidly-exploring random tree: A review. Heliyon, 10(11):e32451, 2024. ISSN 2405-8440. doi: https://doi.org/10.1016/j.heliyon.2024.e32451. URL https://www.sciencedirect.com/ science/article/pii/S2405844024084822.

Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L., and Xie, S. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10632–10643, 2025a.

Yang, S., Xu, R., Xie, Y., Yang, S., Li, M., Lin, J., Zhu, C., Chen, X., Duan, H., Yue, X., Lin, D., Wang, T., and Pang, J. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025b.

Yang, S., Xu, R., Xie, Y., Yang, S., Li, M., Lin, J., Zhu, C., Chen, X., Duan, H., Yue, X., et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025c.

Yeh, C.-H., Wang, C., Tong, S., Cheng, T.-Y., Wang, R., Chu, T., Zhai, Y., Chen, Y., Gao, S., and Ma, Y. Seeing from another perspective: Evaluating multi-view understanding in mllms. arXiv preprint arXiv:2504.15280, 2025.

Yeshwanth, C., Liu, Y.-C., Nießner, M., and Dai, A. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the International Conference on Computer Vision (ICCV), 2023.

Zhang, H., Gu, X., Li, J., Ma, C., Bai, S., Zhang, C., Zhang, B., Zhou, Z., He, D., and Tang, Y. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416, 2025a.

Zhang, H., Liu, M., Li, Z., Wen, H., Guan, W., Wang, Y., and Nie, L. Spatial understanding from videos: Structured prompts meet simulation data. arXiv preprint arXiv:2506.03642, 2025b.

Zhang, J., Chen, Y., Zhou, Y., Xu, Y., Huang, Z., Mei, J., Chen, J., Yuan, Y., Cai, X., Huang, G., Quan, X., Xu, H., and Zhang, L. From flatland to space: Teaching visionlanguage models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976, 2025c.

Zhang, W., Huang, Y., Xu, Y., Huang, J., Zhi, H., Ren, S., Xu, W., and Zhang, J. Why do mllms struggle with spatial understanding? a systematic analysis from data to architecture. arXiv preprint arXiv:2509.02359, 2025d.

Zhao, Z. Advances and challenges in small object detection: A comparative analysis of state-of-the-art models and future directions. Theoretical and Natural Science, 79: 145–153, 01 2025.

Zheng, D., Huang, S., Li, Y., and Wang, L. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. arXiv preprint arXiv:2505.24625, 2025.

Zhou, S., Vilesov, A., He, X., Wan, Z., Zhang, S., Nagachandra, A., Chang, D., Chen, D., Wang, X. E., and Kadambi, A. Vlm4d: Towards spatiotemporal awareness in vision language models. arXiv preprint arXiv:2508.02095, 2025.

Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., Gao, Z., Cui, E., Wang, X., Cao, Y., Liu, Y., Wei, X., Zhang, H., Wang, H., Xu, W., Li, H., Wang, J., Deng, N., Li, S., He, Y., Jiang, T., Luo,

- J., Wang, Y., He, C., Shi, B., Zhang, X., Shao, W., He, J., Xiong, Y., Qu, W., Sun, P., Jiao, P., Lv, H., Wu, L., Zhang,
- K., Deng, H., Ge, J., Chen, K., Wang, L., Dou, M., Lu,
- L., Zhu, X., Lu, T., Lin, D., Qiao, Y., Dai, J., and Wang, W. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025.

### Appendices Contents

#### A Important Information 16

- A.1 Task Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.2 Performance Radar . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

#### B Data Construction 16

- B.1 Data Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.2 Data Source . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- B.2.1 Tiny Tabletop Scene . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2.2 Tabletop Scene . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2.3 Indoor Scene . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2.4 Wild Indoor Scene . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2.5 Outdoor Scene . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.2.6 Drone Scene . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.2.7 Our Own Collected Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- B.3 Task Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- B.3.1 Data Preparation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.3.2 Type: Distance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- B.3.3 Type: Counting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- B.3.4 Type: Planning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.3.5 Type: Relation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.3.6 Data Post-Processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- B.3.7 Benchmark Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- B.4 Data Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- B.4.1 Target Category Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- B.4.2 Data Quality Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- B.4.3 License . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- B.5 Supplementary Citation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

#### C Model Detail 26

- C.1 Parameter Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- C.2 Patch Level Encoder Ablation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- C.3 LoRA Like Expert Ablation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- C.4 Memorization effect observation (Out-of-Distribution Problem) . . . . . . . . . . . . . . . . . . . . . . . 28
- C.5 Challenging Scenario Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- C.6 3D Geometric Feature Observation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

- C.7 Leaderboard Settings Detail . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

#### D FAQ 32

- D.1 Error Accumulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- D.2 All Scale Possibilities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- D.3 Dataset Usage Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

#### E Preview 32

- E.1 Scene Preview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- E.2 Template Preview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

### A. Important Information

|A.1. Task Distribution<br><br>Our SpaceVista-1M consists of a wide range of tasks, including both general tasks and scale-specific tasks. Fig. A5 illustrates the data composition for each scene task, where bubble sizes indicate the relative data volume.<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>Figure A5. Statistical chart of QA types. The spatial reasoning tasks for various scenes include abbreviations, for example, “Est.” for Estimation, “Dist.” for Distance, “Loc.” for Location, and “Com.” for Comparison.<br><br>A.2. Performance Radar<br>|
|---|

The comparison across models is carried out on multiple spatial reasoning benchmarks. We evaluate eight multimodal large models on five distinct benchmarks, with the results visualized in the radar chart in Fig. A6.

SpaceVista-7B achieves significant improvement across the benchmarks, highlighting its superiority in spatial reasoning tasks. While models, including LLAVA-Onevision-7B (Li et al., 2024a), demonstrate competitive performance, SpaceVista7B consistently exhibits superior robustness and adaptability across a range of tasks, thereby solidifying its position as a robust model in spatial reasoning.

### B. Data Construction

Our SpaceVista-1M dataset spans 19 spatial reasoning task types, including scale-specific tasks, comprising 1 million QA pairs and 38 thousand videos collected across diverse scenes. This scale and variety enable large-scale training of perceptual understanding and spatial reasoning, and support comparative analysis across tasks and environments.

[Figure 304]

Figure A6. Performance comparison across popular spatial reasoning benchmarks. Our SpaceVista-7B model achieves certain performance boosts across all benchmarks.

This chapter details the data sources for each scene category (Sec.§ B.2), the end-to-end task construction pipeline (Sec.§ B.3.1), and key dataset statistics (Sec.§ B.4).

#### B.1. Data Comparison

Table B6. The datasets we used to build SpaceVista-1M and SpaceVista-Bench. “†” means the datasets are only used for evaluation in SpaceVista-Bench. “‡” means data collected by us and used for accurate evaluation. The definition of scenes is the number of unique spaces, and one scene can be transformed into multiple questions.

Dataset Type Scenes uCO3D(Liu et al., 2025a) Tiny, Tabletop 10,000 WildRGB-D(Xia et al., 2024) Tabletop 11,300 SMOT(Park et al., 2020) Tabletop 13 SpaceR(Ouyang et al., 2025) Indoor 1,500 Spar-Bench(Zhang et al., 2025c) Indoor 4,500 Scannet Series(Dai et al., 2017; Yeshwanth et al., 2023) Indoor 460 VSI-Bench†(Yang et al., 2025a) Indoor 288 MMSI-Bench†(Yang et al., 2025b) Indoor 231 DL3DV(Ling et al., 2024) Drone, Indoor, Outdoor 10,510 STI-bench†(Li et al., 2025d) Indoor, Outdoor, Tabletop 372 Our own collected data ‡ Tiny, Tabletop, Outdoor 500

Our current dataset encompasses a broad diversity of scene categories, as summarized in Tab. B6. The data sources span a wide range of scenarios, including tiny, tabletop, indoor, outdoor, and drone-view.

To ensure evaluation quality and robustness, we apply multiple rounds of processing and rigorous filtering to all collected data. We remove redundant or inconsistent samples across datasets. Because scenes may overlap across sources, which can compromise the independence of the training and test splits, we removed from the training set any scene that appears in all the benchmarks. This strict separation prevents leakage and enables a fair assessment of generalization. Consequently, the SpaceVista-1M provides broad scene diversity, with a clean, reliable benchmark SpaceVista-Bench.

#### B.2. Data Source

Sec.§ B.2 presents data sources that form our dataset, and systematically describes the provenance and acquisition of seven scene sources. These sources combine multiple public datasets and our own collected data, as detailed in Sec.§ B.2.1- B.2.7.

These scenes span object-centric through scene-level contexts and exhibit substantial variation in scale, shape, pattern, and illumination.

When building the dataset, our foundational data construction process must adhere to the following key criteria:

- • Video Data with 3D Modeling: The data must consist of video sequences accompanied by either official or third-party 3D modeling. This enables effective use of camera parameters for robust data processing.
- • Multi-Frame & Multi-Scale: The dataset should support meaningful spatial reasoning across multiple frames and scales. Its complexity must be sufficient to prevent trivial single-frame assessments from representing the full sequence.
- • Comprehensive Annotations & Metadata: Each sample must include the following: (a) camera intrinsics and extrinsics, (b) detection and segmentation labels, and (c) dense depth maps. These elements support a broad range of downstream tasks.

- B.2.1. TINY TABLETOP SCENE

We curate small-scale, small-object videos from uCO3D (Liu et al., 2025a), selecting sequences where the object size falls below a predefined threshold to instantiate the tiny tabletop scenario. uCO3D comprises approximately 170,000 high-resolution, object-centric 360-degree videos captured via crowdsourcing, covering more than 1,000 LVIS (Gupta et al., 2019) categories grouped into 50 categories. For each video, uCO3D applies VGGSfM (Wang et al., 2024) for motion analysis and 3D Gaussian Splatting to generate accurate camera poses, depth maps, sparse and dense point clouds, and semantic captions. The resulting subset contains everyday small objects, such as stationery, food, and decorative items, placed on flat surfaces such as tables, counters, and shelves. These scenes provide complete viewpoint coverage, precise geometry, and rich semantic labels, which make them well-suited for fine-grained 3D object modeling and spatial video reasoning. Here, we only select a small part of uCO3D for around 10,000 videos for tiny objects after filtering.

- B.2.2. TABLETOP SCENE

For tabletop scene modeling, we select two datasets: WildRGB-D (Xia et al., 2024) and SMOT (Park et al., 2020). WildRGB-D consists of approximately 8,500 objects across 46 categories, recorded in around 20,000 RGB-D videos, with iPhones rotating 360 degrees around objects to replicate real-world interactions. It includes single-object, multi-object, and hand-occlusion videos, all automatically annotated via SLAM-generated camera poses and reconstructed point clouds, making it suitable for spatial reasoning tasks. To select samples for spatial reasoning, we specifically choose around 10,000 videos with multiple objects in a scene. SMOT (Park et al., 2020) is a challenging small dataset collected by a mobile robot, comprising 13 video sequences.

The tabletop, commonly referred to as the “table” scene, encompasses not only the planar surface of a table but also extends to various other surfaces, including sand, beds, wardrobes, floors, and similar environments. In combination, these datasets offer richly varied planar scenes, providing a robust foundation for challenging spatial video reasoning benchmarks.

- B.2.3. INDOOR SCENE

Indoor scenes are among the earliest domains studied in spatial video reasoning. Key datasets, including ScanNet (Dai et al., 2017) and ScanNet++ (Yeshwanth et al., 2023), collect RGB-D scans using handheld cameras, yielding aligned RGB images, depth maps, and 3D reconstructions. ScanNet contains more than 1,500 scenes and 2.5 million frames spanning common indoor spaces, such as offices and bedrooms, with annotations for over twenty object categories. ScanNet++ extends this setting with higher geometric fidelity and more complex layouts. The combination of focused object classes, structured environments, and rich annotations makes these datasets central benchmarks for spatial reasoning.

- B.2.4. WILD INDOOR SCENE

Beyond scan-based indoor modeling, DL3DV (Ling et al., 2024) adopts a video-based pipeline that replaces active scanning with video capture and camera parameter estimation. Building on this framework, and further compressed using 3D Gaussian Splatting (Chen et al., 2024c), DL3DV enables high-precision 3D reconstruction of wild indoor scenes. The dataset covers a broad range of object categories, including challenging reflective and transparent instances. Compared with conventional scan-based datasets, these scenes exhibit greater geometric and appearance variability, providing a more realistic and demanding benchmark for spatial video reasoning.

- B.2.5. OUTDOOR SCENE

In addition to tabletop and indoor scene modeling, DL3DV (Ling et al., 2024) collects extensive in-the-wild outdoor videos encompassing landmarks, street corners, private courtyards, and urban parks. Camera parameters are calibrated using COLMAP (Schönberger et al., 2016; Schönberger & Frahm, 2016). The DL3DV-10K dataset includes 10,510 videos in 4K resolution, totaling about 51.2 million frames, covering 65 types of locations. Each video is annotated for whether it is indoors or outdoors as well as for levels of reflection, transparency, and lighting conditions. Compared to conventional scan-based indoor datasets, these outdoor scenes exhibit richer geometric complexity, greater diversity of materials, and wider environmental variation, offering more challenging benchmarks for spatial video reasoning.

- B.2.6. DRONE SCENE

DL3DV (Ling et al., 2024) extends outdoor scene modeling by incorporating drone-captured videos that provide aerial perspectives to complement ground level views. Videos are recorded using unmanned aerial vehicles (UAVs), and camera parameters are calibrated through COLMAP (Schönberger et al., 2016; Schönberger & Frahm, 2016), following the same reconstruction pipeline applied to handheld footage. The DL3DV Drone subset consists of more than 100 videos covering a variety of scenes, including open plazas, tree-lined pathways, rooftop platforms, and landmark facades. DL3DV enhances spatial video reasoning by introducing unique geometric structures and varied viewpoints.

Although the data scale is not as large as tabletop or indoor, the drone-view scenes establish a more rigorous benchmark for aerial mapping and spatial video reasoning by expanding scene diversity and viewpoint range.

- B.2.7. OUR OWN COLLECTED DATA

The data collection methods described above rely on advanced specialized models and fully automated pipelines. While we incorporate limited manual filtering, whether the resulting data can be used as an accurate evaluation of real-world perception is still a question. This limitation motivates our collection of higher-fidelity data to better align with physical world perception.

Our dataset consists of two types: 1) measured, recorded, and manually annotated data, and 2) existing video data enhanced by retrieving and verifying publicly available information. The former is suitable for tiny objects, tabletop objects, whereas the latter is designed for indoor and outdoor scenarios.

[Figure 305]

[Figure 306]

[Figure 307]

Figure B7. The objects involved in our hand-crafted dataset. Our selfcollected data features various categories of objects, with tabletops and tiny tabletops ranging from 0.4m to 3mm, even including transparent and reflective objects.

Figure B8. A photo of the real scene for the collection of tiny tabletop.

Data from self-recording and measurement. Precise spatial annotations (e.g., location and dimensions) are scarce in existing datasets such as uCO3D and WildRGB-D. To address this, we captured length and positional data for nearly 50 object categories across diverse scenarios. Using GoPro 11, iPhone 15, and Vivo X70, we systematically varied object arrangements, distances, lighting conditions, and backgrounds into over 200 videos and 1,000 QA pairs. As illustrated in Fig. B7 and Fig. B8, they show the objects used for self-collected data and a real scene of tiny tabletop data collection. Although we collected the raw high-resolution videos up to 2.7K/60fps, it is still necessary to resize and resample it for better comparison. The resulting measurements are consolidated into a unified perceptual space that closely approximates physical world geometry.

#### Data Retrieved from authoritative sources.

Adopting a similar rationale, it is apparent that spatial information derived solely from wild videos lacks the precision required for robust evaluation. Consequently, alternative methodologies must be explored. To address this, we propose a systematic approach that first identifies landmark objects within existing datasets and then manually retrieves images of these objects from authoritative sources, such as Wikipedia1, architectural drawings, and official design documents, to obtain accurate spatial information, as shown in Fig. B9. This method ensures that the evaluation data is not only more precise but also more consistent with human perceptual judgments and preferences.

[Figure 308]

- Figure B9. Examples of identifying outdoor landmark objects from existing datasets and retrieving their scale-related ground truth data.

#### B.3. Task Construction

Upon acquiring the appropriate dataset, we initially perform necessary data preparation and processing in Sec.§ B.3.1. Subsequently, we carefully design workflow for each task (Sec.§ B.3.3-B.3.5), and we present detailed task explanations in Tab. B7. The final output consists of high-quality QA pairs, facilitating the cold-start and reinforcement learning processes of MLLMs.

- B.3.1. DATA PREPARATION

Previous popular approaches, such as InternSpatial (Deng et al., 2025b), required estimating camera intrinsic and extrinsic parameters, which introduced cumulative errors that propagated through subsequent tasks. However, since we exclusively utilize datasets with known camera parameters (as detailed in Sec B.2), our framework operates under conditions close to ground truth.

We first employ Metric3Dv2 (Hu et al., 2024) and UniDepthV2 (Piccinelli et al., 2025) to obtain accurate metric depth maps and normal maps. The metric depth maps provide precise distance measurements between the camera and scene objects, while the normal maps facilitate robust plane estimation. There are two challenges during construction. 1) Video consistency: According to observation, the metric depth model may not have that level of consistency across frames. So, we use Video-Depth-Anything (Chen et al., 2025) to ensure consistency by minimizing the energy function,

D∗ = argmin

D

∥D − M∥2F + λ∥∇t(D) − ∇t(N)∥2F , (5)

where M,N represent metric depth model maps and Video-Depth-Anything map . 2) Extreme Scale: Although the metric depth model is trained on the datasets as DDAD (Guizilini et al., 2020) and NYUv2 (Silberman et al., 2012), it may have a

1https://www.wikipedia.org/

Table B7. Detailed explanation of 19 tasks included in SpaceVista-1M.

Task Description

General Indoor Scenes Position Comparison Compare the positions of two objects within or across frames, assessing their spatial relation-

ships in terms of left/right, above/below, and near/far. Size Comparison Compare the positions of two objects within or across frames, involving three pairs of size relationships: wider/thinner, taller/shorter, larger/smaller. Existence Estimation Determine whether there are objects across frames whose positional/size relationships with

the specified object meet the constraint conditions. Object Counting Estimate how many objects meet the constraint conditions across frames. Rotation Estimation Estimate the rotation angle of an object across multiple frames. Absolute Distance Estimate the closest distance between two objects within or across the frames. Object Size Estimate the longest dimension of an object within or across the frames. Route Planning Choose what action should be performed between a sequence of actions within or across the

frames in order to route from a start point to a target. Appearance Order Given a video, determine the N-th appearance order of several objects. Depth Estimation Estimate the relative or absolute distance of objects from the camera viewpoint in a single

image or across multiple images.

View Change Inference Infer how the camera viewpoint has changed (position and orientation) across the video frames. Object Matching Determine whether two objects in the beginning and end frames of a video are the same

physical object instance or different instances of the same object type. Spatial Relation Analyze and describe the spatial relationships (e.g., support, hanging, adhesion, stacking, encircling, plug-in) between multiple objects or cameras across the frames.

Indoor Scenes Every Type in General All task types from Indoor Scenes can be applied to drone-view perspectives. Room Size Estimate the volume of the room(s) across the frames.

Outdoor Scenes Every Type in General All task types from Indoor Scenes apply to Outdoor Scenes except for Room Size estimation. Navigation Determine the optimal path or movement strategy to navigate from one location to another

across different views (similar to the Route Planning mentioned in Indoor Scenes).

Drone-View Scenes Every Type in General All task types from Indoor Scenes can be applied to drone-view perspectives. Route Plan Given a series of aerial images, choose what action should be performed between a sequence of

actions in order to route from a start point to a target (similar to the Route Planning mentioned in Indoor Scenes).

Area Estimation Estimate the size or area of regions or objects from an aerial perspective.

Tabletop Scenes Every Type in General All task types from Indoor Scenes can be applied to drone-view perspectives. Object Location Determine the precise position of objects on a table surface, typically corresponding to other

objects. Destination Location Identify target positions related to single objects (i.e. left, right, front ...) as part of manipulation planning. Obstacles Location Identify and locate objects with the AABB box that may interfere with manipulation as part of manipulation planning. Manipulation Planning Determine the sequence of actions needed to rearrange objects or achieve a specific configuration on the table.

certain level of adaptation to the extreme situations. For extreme situations, including drone-view and tiny objects, it is still necessary to provide a prerequisite to adjust the depth normalization accordingly.

For fine-grained semantic understanding at the pixel level, we leverage the advanced proprietary model DINO-X (Ren et al., 2024) to extract semantic information and bounding boxes for complex scenes, while relying on Grounding DINO (Liu et al., 2023) for simpler samples. To address cross-frame consistency challenges in video data, we integrate the aforementioned grounding models with SAM2’s (Ravi et al., 2024) advanced tracking capabilities, generating temporally consistent masks and unique object IDs across frames based on Grounded-SAM22.

By this stage, we obtain a comprehensive understanding of each frame, including bounding boxes, masks, categories, and object IDs, laying a solid foundation for downstream task formulation.

- B.3.2. TYPE: DISTANCE

The distance-related tasks, including object size, room size, object distance, and relative distance, rely on depth maps and computer vision techniques to measure object and spatial dimensions from monocular images. The method converts 2D depth keypoints into 3D point clouds using camera calibration parameters and applies Principal Component Analysis (PCA) to extract dimensional information, focusing on objects larger than 20×20 pixels. For object size estimation, the system segments visible objects using instance masks and projects the masked depth values into 3D space. PCA determines the principal axes of the point cloud, with height measured along the vertical axis and width derived from the convex hull of points projected onto the dominant plane. Relative distances are calculated by comparing 3D centroids in world coordinates, and room dimensions are estimated by analyzing the spatial distribution of depth points and identifying major planar surfaces corresponding to walls.

The method uses camera intrinsics and extrinsics to express all measurements in a consistent world coordinate system, addressing the scale ambiguity of monocular systems. Multiple frames are processed to improve robustness, with temporal averaging reducing noise in the estimates. The technique assumes piecewise rigid scenes, operates on standard RGB images, and produces metric-scale measurements. Accuracy depends on the quality of depth estimation and segmentation. Overall, it demonstrates how 2D computer vision pipelines can be extended to 3D measurement tasks through precise geometric reasoning.

- B.3.3. TYPE: COUNTING

Camera Parameter Positioning Direct Coordinates

Confirm Counting Video

Dataset Video

Object Tracking

Available Object Video

Object Detection

Bbox Threshold

Manually

Refinement

Object Appearance

Object Counting Selection

Object Bbox Filtered

Feature Detection

Filtered Small Object Video

Counting

Preliminary Filter Counting Video

Dateset

- Figure B10. Automatic Processing Pipeline for Counting Task Scenes. Through data filtering, object tracking, and counting, the final counting video is obtained after data confirmation.

Object counting across real-world scenes faces diverse visual conditions and a high cost of manual labeling, which motivates an automatic pipeline that adapts to scene type. The automatic pipeline addresses object counting through two methodologies tailored to specific scenarios, and Fig. B10 illustrates the workflow that maintains high accuracy while reducing manual effort across indoor, outdoor, and tabletop scenes. For outdoor video sequences, the open-vocabulary detection model (Ren et al., 2024; Cheng et al., 2024) uses text prompts with a confidence threshold of 0.3 for zero-shot detection, projects

2https://github.com/IDEA-Research/Grounded-SAM-2

- 2D observations into 3D world coordinates to enforce spatial consistency, and tracks objects via motion prediction with confirmation after at least ten consistent detections. Given the difficulty of reliably detecting very small objects in outdoor scenes and to mitigate ID switching and trajectory fragmentation under severe occlusions, scenes are prefiltered to those containing 2 to 10 objects with a minimum bounding-box size of 32 pixels. For tabletop scenarios, grounding model (Ren et al., 2024; Liu et al., 2023) and SAM2 (Ravi et al., 2024) are employed, where open-vocabulary detection uses text and bounding box thresholds of 0.4, and mask propagation applies IoU and center distance thresholds of 0.4 and 32 pixels, respectively, to distinguish instances. Both methodologies output object categories and their corresponding counts for each video.

B.3.4. TYPE: PLANNING

0.2 0.1 0.0 0.1 0.2

0.2

0.1

0.0

0.1

0.2

0.0

0.1

0.2

0.3

0.4

3D Obstacles and Path Visualization (Choice D)

Path

[Figure 309]

(a)3D Path Planning Visualization (b)Trajectory Execution Process

Figure B11. Visualization of robotic manipulation planning. Fig.(a) visualizes the option for moving the red box to the left of the upper box. Fig.(b) represents the key frame to carry out the manipulation.

In robotic manipulation tasks, effective route planning is essential to ensuring smooth and accurate object movement. The route planning pipeline proceeds as follows. First, depth information and object detection are utilized to identify the category, position, shape, and size of all objects within the image. Subsequently, an arbitrary object is selected as the manipulation source and another as the target position, with the objective being to relocate the source object to a designated position (e.g., front, back, left, right, or above) relative to the target object. Based on this configuration, an LLM generates corresponding manipulation instructions, such as “What is the correct route of placing the apple on the box”. Next, the actual spatial positions of the objects are computed using both intrinsic and extrinsic camera parameters. The Rapidly-exploring Random Tree (RRT) (LaValle, 1998; Xu, 2024) algorithm is then employed to plan a collision-free path, where the bounding boxes of objects serve as obstacle constraints during path computation. Finally, two types of data are generated from the planned path: 1) multiple paths are projected onto the camera plane, with the correct trajectory serving as the ground truth answer, and 2) the coordinate variations along the path are translated into natural language instructions via the LLM. For instance, when the x-coordinate of the object decreases while the y-coordinate remains constant in the camera space, the LLM produces the instruction “move the object to the left.” Fig. B11 demonstrates the visualization of robotic manipulation under the option, showing the planned movement of the red box to the left of the upper box. This figure highlights the spatial relationship and intended positioning within the manipulation task.

B.3.5. TYPE: RELATION

In spatial relation analysis, we combine semantic information with 3D positional data through an automatic reasoning process to ensure consistency in both semantic and spatial aspects. Our analysis operates primarily at the semantic level. We first identify and extract common candidate relations, such as support, attach, insert, and surround. Based on the consistent

- 3D keypoint semantics established earlier, we generate potential relation pairs that may exhibit these spatial relationships. These candidate pairs are then evaluated for spatial plausibility by integrating 3D positional data with the few-shot prompt through Chain-of-Thought (CoT) reasoning using the foundation model. Finally, the validated pairs are processed by GPT

for transformation and answer generation, ensuring semantically and spatially consistent outputs.

- B.3.6. DATA POST-PROCESSING

To address the cold-start challenge in SFT, we prioritize the acquisition of explicit “thinking process” rationales—step-bystep explanations that clarify how answers are derived. For example, in object counting, the model is prompted to articulate intermediate reasoning (e.g., “there are 2 cups on the table and 3 on the chair, totaling 5”), enriching task understanding and facilitating more robust generalization.

Following common practice (Feng et al., 2025), we acquire high-quality rationales by distilling from advanced open-source and proprietary large models. Specifically, we use Qwen2.5-VL-72B and Gemini-2.5-Pro for complex tasks, and Qwen2.5VL-32B for simpler ones, balancing reasoning depth with efficiency. We then compare these generated rationales and their corresponding answers with previously collected cases. When GPT answers are different from the answers from previous workflows, we apply a confidence-based filtering strategy to curate the training set, retaining only instances with consistent, well-supported reasoning. This pipeline generates a cleaner, rationale-augmented dataset, mitigating SFT cold-start effects and enhancing downstream performance.

- B.3.7. BENCHMARK CONSTRUCTION

Our benchmark comprises two components: 1) Measurement-Related. For the scale-related portion requiring precise scale annotations, we collect approximately 300 videos across diverse scenes using the two methods described in Appendix B.2.7 and human annotation for other spatial tasks, covering tiny, tabletop, and outdoor settings. For the indoor evaluation set, we instead selected suitable data from ScanNet-based datasets (e.g., VSI-bench and SPAR-bench) and constructed a series of scale-focused questions on top of these bases. 2) Non-Measurement. For the non-measurement questions, we manually annotate the data collected in the previous step to produce additional spatial reasoning QA pairs. In total, we curate about 1,600 fully human-annotated QA pairs for model evaluation.

#### B.4. Data Statistics

From a visual perspective, our dataset comprises wild scenes spanning scales from millimeters to kilometers. Although the raw dataset contains over 100 million frames, we calculate unsupervised annotations as intermediate information at both the pixel and semantic levels for a curated subset of 10 million frames. These frames vary in resolution from 480p to 2.7K, with frame rates ranging from 24 to 30 fps. During data processing, we preserve the original resolution whenever possible and apply uniform sampling during training as needed.

In terms of the QA component, we employ a combination of templated generation and GPT-based methods to produce 1 million QA pairs with a theoretical duplication rate of only 0.0005%. These pairs are structured into diverse answer formats, including free-form, multiple-choice, and regression-based responses, catering to different analytical needs. Rigorous quality control measures are implemented, with detailed analyses provided in Sec.§ B.4.2.

- B.4.1. TARGET CATEGORY DISTRIBUTION

The introduction of diverse scenarios, such as tabletop, indoor, and outdoor, aims to establish a more inclusive object composition system. Due to the limited drone data, we incorporate drone-view data into the outdoor analysis. By approximating complex object distribution patterns to the real world, this approach enhances the scene adaptation capabilities of visual reasoning models. To quantitatively assess the impact of scene diversity on model generalization, we use the word cloud to compare object distribution characteristics across different scenarios, as shown in Figs. B12–B17. The results reveal that indoor scenes are predominantly composed of rigid objects such as furniture and electronics, exhibiting a highly structured spatial layout. In contrast, outdoor scenes feature more scale-varying objects like vehicles and natural landscapes, demonstrating spatial openness. Meanwhile, tabletop scenes focus on manipulable items such as tools and daily necessities, reflecting precise spatial arrangements. These cross-scene differences provide complementary training samples, effectively mitigating the risk of overfitting to specific scenarios. Thus, the necessity of a multi-scenario strategy to enhance cross-domain generalization is validated.

Overall, each subset scenario differs significantly from the previous indoor-dominated setting, highlighting the diversity of our scenes.

- B.4.2. DATA QUALITY CONTROL

During the construction of our dataset, we distinguish between two notions of answer assessment: 1) strict correctness, which requires that an answer conform to objective physical reality, and 2) human preference, which requires that an answer align with typical human judgments. Since strict correctness is difficult to establish for training data derived from in-the-wild videos (due to issues like missing calibration, occlusions, and limited metadata), we adopt the human-preference criterion

[Figure 310]

[Figure 311]

- Figure B12. The word cloud of the previous indoor spatial reasoning datasets. Figure B13. The word cloud of our indoor subset.

[Figure 312]

[Figure 313]

Figure B14. The word cloud of our outdoor subset. Figure B15. The word cloud of our tabletop subset.

[Figure 314]

[Figure 315]

Figure B16. The word cloud of our tiny tabletop subset. Figure B17. The word cloud of the self-collected subset. Note: We use standard ISO 7046 to denote the models of the screw, which looks like “m4*10”.

for training data quality control. Specifically, during validation, we present annotators with both the question and a candidate answer and ask them to judge whether the answer is acceptable from a human perceptual perspective. Consequently, the reported human preference rate should be interpreted as agreement with human perception rather than strict fidelity to physical-world quantities or metric scale. For these statistics and the user study, we use MTurk3. SpaceVista-1M human preference rates are shown in Tab. B8. SpaceVista-Bench is curated through source verification and manual review, while model evaluation follows strict correctness.

Table B8. Human preference rate over each task category in our training set, SpaceVista-1M. “∼” means we observe unusual variation for different annotators. The standards for Route Planning, Navigation, and Obstacle Avoidance are notably stringent, as these are inherently multi-step processes where a single error can invalidate an entire sample. However, even if a training sample contains minor discrepancies in step, sequence, or distance, the descriptive knowledge within the remaining sections remains valuable for comprehension.

Task Categories in SpaceVista-1M for training Task

Position Comp.

Size Comp.

Existence Est.

Rotation Est.

Relative Dist.

- Human Preference Rate 95% 84% 94% 95% 82%

Task

Room Size

Object Count

Object Size

Route Plan

Appear. Order Human Preference Rate 84% 87% 81% ∼69% 80% Task

View Change

Object Match

Spatial Rel.

Navigation

Area Est.

- Human Preference Rate 96% 93% 95% ∼67% 78%

Manip. Plan

Absolute Dist.

Depth Est.

Task

Obstacles Human Preference Rate 73% 84% 95% 74%

- B.4.3. LICENSE

We conduct a systematic review of the open-source licenses for the datasets we use, with the results summarized in Tab. B9. The analysis indicates that CC BY 4.0 and Apache License 2.0 are the most widely adopted. After comprehensive consideration, our SpaceVista-1M dataset adopts the Creative Commons Attribution (CC BY) 4.0 or Apache License 2.0 for different sources of data, which is already used by most of the source data.

Table B9. The licenses for the dataset and benchmark included in this paper.

Dataset Type License

Benchmarks VSI-Bench(Yang et al., 2025a) Indoor Apache License 2.0 STI-bench(Li et al., 2025d) Indoor Apache License 2.0 MMSI-Bench(Yang et al., 2025b) Indoor CC BY 4.0 STI-Bench(Li et al., 2025d) Outdoor, Tabletop Apache License 2.0 Spar-Bench(Zhang et al., 2025c) Indoor Apache License 2.0 SpaceVista-Bench (Ours) Tiny, Tabletop, Indoor, Outdoor Apache License License 2.0 & CC BY 4.0 Training Datasets uCO3D(Liu et al., 2025a) Tiny, Tabletop CC BY 4.0 SMOT(Park et al., 2020) Tabletop Unknown WildRGBD(Xia et al., 2024) Tabletop None SpaceR(Ouyang et al., 2025) Indoor CC BY-NC 4.0 Scannet Series(Yeshwanth et al., 2023) Indoor ScanNet Terms of Use DL3DV(Ling et al., 2024) Indoor, Outdoor, Drone DL3DV-10K Terms of Use SpaceVista-1M (Ours) Tiny, Tabletop, Outdoor Apache License License 2.0 & CC BY 4.0

- B.5. Supplementary Citation Due to the page limit, we have omitted some citations in Tab. 1. Here, we provide a supplementary table of citations.

### C. Model Detail

#### C.1. Parameter Setting

SFT. The model architecture is based on Qwen2.5-VL-7B-Instruct, a 7-billion parameter vision-language model capable of processing both images (resized to 100,352 pixels) and videos (16,384 pixels at 16/32 frames). In the ablation study, we use the 3B model for efficiency. For fine-tuning, we employ a selective freezing strategy: while the vision tower and multi-modal

3https://www.mturk.com/

Table B10. Supplementary citation of Tab. 1 dataset comparison.

|Dataset Citation<br><br>|Dataset Citation|
|---|---|
|SpaceR Ouyang et al. (2025) SPAR-7M Zhang et al. (2025c) Spatial-MLLM Wu et al. (2025c) InternSpatial Deng et al. (2025b) Video-MME Fu et al. (2024) TempCompass Liu et al. (2024b)|All-Angles Yeh et al. (2025) MVBench Li et al. (2024b)<br><br>VSI-Bench Yang et al. (2025a) MMSI-Bench Yang et al. (2025c) SPAR-Bench Zhang et al. (2025c)<br><br>STI-Bench Li et al. (2025d)|

Table B11. Supplementary citation of models in Tab. 5 SpaceVista-Bench leaderboard.

Model Citation Model Citation

GPT-5 (OpenAI, 2025) Internvl3-38B (Zhu et al., 2025) GPT-4o (Hurst et al., 2024) GLM-4.5V (Team et al., 2025) Gemini-2.5-pro (DeepMind, 2025) GLM-4.1V-Thinking (GLM et al., 2024) Gemini-2.5-flash (DeepMind, 2025) Qwen2.5VL-72B (Bai et al., 2025) Claude-Sonnet-4 (Anthropic, 2025a) Qwen2.5VL-32B (Bai et al., 2025) Claude-Opus-4.1 (Anthropic, 2025b) LLAVA-Onevision-72B (Li et al., 2024a) Internvl3.5-38B (Wang et al., 2025c) LLAVA-Onevision-7B (Li et al., 2024a) Internvl3.5-14B (Wang et al., 2025c) SpaceR (Ouyang et al., 2025) Internvl3-78B (Zhu et al., 2025) Spatial-MLLM (Wu et al., 2025c)

projector remain frozen to preserve pretrained visual representations, the language model is fully trainable. Training utilizes full parameter fine-tuning with a DeepSpeed4 ZeRO-2 configuration for memory optimization. The model is trained on our proposed dataset for spatial understanding in indoor environments, with samples truncated at 32,768 tokens. We implement a cosine learning rate schedule (initial LR=5e-7) with 10% warmup over 2 epochs. We maintain computational efficiency through mixed-precision bfloat16 training.

RL. We conduct our experiments using the Qwen2.5-VL (Bai et al., 2025) on a custom spatial dataset. The training utilizes 7 GPUs with DeepSpeed acceleration and mixed-precision bf16 training with flash attention. Key hyperparameters include a batch size of 1 per device, gradient accumulation steps of 1, an initial learning rate of 1e-6 with cosine scheduling, and weight decay of 0.01. The model processes input sequences up to 16,384 tokens long while generating outputs up to 1,024 tokens. Training runs for 2 epochs with evaluation performed every 200 steps. For inference, we use vLLM on a separate GPU with temperature 1.0 and generate 8 samples per input.

Other Setting. We set the number of experts M to 4 in most cases. We also add LoRA with the same default behavior as PEFT. Additionally, we apply expert scaling factors on a layer-wise basis rather than globally.

Ablation Setting. Unless otherwise noted, we conduct all ablation experiments using the Qwen2.5-VL-3B model because of resource constraints; all other settings are identical to those described above.

#### C.2. Patch Level Encoder Ablation

We evaluate several visual encoders with dense feature or geometry-aware representations, including VGGT-1B (Deng et al., 2025a)(the only publicly available model) and the generalDINOv3 ViT-Base, and perform ablations on the patch encoder. Tab. C12 reports the performance gains and computational costs associated with each model. Across encoders, DINOv3 achieves more favorable efficiency–accuracy trade-offs with a smaller parameter budget. We attribute this to its self-supervised pretraining, which is not constrained by labeled data and thus confers stronger generalization. In contrast, VGGT exhibits strong reconstruction capabilities but depends on annotations that lack rich semantic content and further relies on a large decoder to recover geometry. Consequently, compared to VGGT, DINOv3 features are more readily consumed by the fusion module, facilitating more effective mapping.

#### C.3. LoRA Like Expert Ablation

On top of the same 3B pretrained base model, we compare three training strategies: 1) Full-parameter Fine-tuning, 2) Vanilla LoRA, and 3) LoRA-like Expert, with the results shown in Fig. C13. We observe that vanilla SFT-based fine-tuning still suffers from latent cross-scale information conflicts. The difference between model-wise and layer-wise is that, for

4https://github.com/deepspeedai/DeepSpeed

Table C12. Ablation Comparison of the patch-level encoder across different sizes of models on the indoor set VSI-Bench based on the same SFT training settings.

Model&Parameter Video-Only +VGGT +DINO v3 +VGGT +DINO v3 SpaceVista-3B (Ours) 41.9 43.3 43.5 43 .3

SpaceVista-3B (Ours) w/o. fusion module - 42.0 44.8 44.7 SpaceVista-7B (Ours) 45.0 45.7 46.3 46.0

Extra Parameter 0 909M 303M 1,320M

Table C13. Ablation comparison of the LoRA-like expert in the SFT training stage.

w/. Full-parameter Fine-tuning

w/. Vanilla LoRA Fine-tuning

w/. LoRA-like Expert (model-wise)

w/. LoRA-like Expert (layer-wise)

Model Benchmark

VSI-Bench 43.5 42.9 43.9 45.3 SpaceVista-Bench 29.5 29.4 32.5 33.0

SpaceVista-3B

Trainable Parameters 3B 20M 80M+30M 80M+34M

each input, the router is calculated and implemented to the whole model or to separate layers, respectively. In contrast, the model-wise LoRA-like Expert yields clear gains over both full-parameter fine-tuning and vanilla LoRA. Furthermore, scaling to a higher-capacity, layer-wise LoRA-like Expert delivers additional improvements.

#### C.4. Memorization effect observation (Out-of-Distribution Problem)

In our experiments, we observe that models often exhibit a strong bias toward memorizing fixed sizes for certain objects—for instance, chairs are typically assumed to be 50-70 cm tall. Consequently, the network tends to rely on memorized size priors rather than reasoning about object scale. However, this phenomenon presents a dual nature. On one hand, human perception of size and scale also depends on reference objects and familiar benchmarks, which are essential for intuitive understanding. On the other hand, since real-world spatial relationships can vary significantly, such biases may lead to erroneous judgments in atypical cases.

We argue there is two types of Out-of-Distribution (OOD) that should be discussed separately. 1) OOD category with normal size 2) normal category with OOD size.

For normal category with OOD size, we need to develop a dataset with precise annotation. The Guinness World Records (GWR) is a globally recognized organization5 that catalogs uncommon objects and forms. We obtain precise size measurements along with the corresponding images/videos, and construct a series of QA pairs about object sizes as shown in Fig. C18. The GWR data comprises diverse scenes, including outdoor, indoor, and drone, with over 50 images and over 50 questions. Because only a small portion of the records is documented on the website, we used nearly all available website content to construct this GWR test set. All questions were created through human annotation to ensure dataset quality. This data is used solely for insight and analysis, not for official purposes. The licensing status of GWR content is unclear. If the license permits, we will release this GWR set on Hugging Face.

Table C14. Performance comparison across GWR dataset.

|Size-Related QA<br><br>|Qwen2.5VL-7B Qwen2.5VL-3B SpaceVista-7B SpaceVista-3B|
|---|---|
|SpaceVista-Bench GWR set|49.9 44.0 58.3 49.3 27.8 23.1 31.1 27.3<br><br>|

As shown in Table C14, we evaluate the popular Qwen2.5-VL model and our SpaceVista-7B model. Because the GWR data contain only size-related questions, we select the size-related subset of SpaceVista-Bench to ensure a fair comparison. We find that these OOD data are challenging for both the general-purpose model and our specialist model. However, the OOD challenge does not produce a clear performance gap between Qwen2.5-VL and SpaceVista. Although our model is not designed for purely image-based tasks, this potential bias suggests a promising direction for future work in VLLMs.

For OOD category with normal size, to systematically evaluate the impact of this bias and its potential implications for advancing the field, we design three specialized subsets at the same scale:

5https://www.guinnessworldrecords.com/records/showcase

[Figure 316]

Figure C18. Data preview in the Guinness World Records (GWR). GWR is a globally recognized organization that catalogs uncommon objects and forms. We scraped precise size measurements along with the corresponding images/videos, and constructed a series of QA pairs about object sizes.

- • Seen Set: Common object categories from the training distribution (i.e., bicycle, table, chair).
- • Seen Set with Various Scales: bjects of the same category (i.e., different sizes and shapes of screw).
- • Unseen Set: Rare or culturally specific objects requiring contextual size reasoning (i.e., ethnic items with regional characteristics, such as a traditional food).

The Seen Set provides baseline performance metrics for familiar objects but may overlook biases due to training conformity. The Seen Set with scale variety directly probes size generalization for known categories, but it is limited to variations within seen objects. The Unseen Set evaluates robustness to novel, culturally diverse scenarios but risks introducing confounders beyond scale bias. Collectively, these subsets balance ecological validity with experimental control, offering a comprehensive framework to diagnose size-related biases. This structured approach enables us to analyze how size biases manifest under different conditions, combining ecological validity with controlled experimentation. As shown in Fig. C15, all-scale training benefits the overall reasoning model; however, the general models still tend to memorize the regular size of the target object.

Table C15. Reasoning VS memorizing analysis of different subsets.

Seen Set (Normal)

Seen Set (Various Scales)

Unseen Set

Model

Qwen2.5-VL-3B-Instruct 35.7 34.7 23.1 Qwen2.5-VL-7B-Instruct 37.0 38.9 28.0 SpaceVista-7B (Ours) 37.3 41.0 32.8

Our analysis of potential bias has two parts:

- 1. Depth Knowledge. Current metric depth models estimate distance primarily based on accurate camera parameters, such as focal length. These parameters vary across different scales, which is why our model performs slightly better than a general model.
- 2. Scale Prior. Human distance estimation also strongly relies on reference objects (i.e., scale priors in question). When these references are unusual, humans also unavoidably exhibit bias. Thus, scale priors are a double-edged sword and cannot be simply described as good or bad.

- Table C16. Results analysis of different scenes. The model mentioned below is trained in a balanced subset of SpaceVista-1M for better control of experiment conditions.

Model

SpaceVista-Bench (Ours) Indoor Outdoor Tabletop Tabletop Qwen2.5-VL-7B 30.34 18.31 23.79 19.37

w/. balance training 38.77 24.90 30.17 20.86

C.5. Challenging Scenario Analysis

When testing scenes at varying scales, several critical questions arise: Which scenarios pose greater challenges, and to what extent is data complexity the primary bottleneck? To systematically investigate these issues, we design a controlled observational experiment.

We identify tasks that exhibit consistent properties across different scales, including object size, object comparison, absolute and relative distance, and depth estimation. For fairness in comparison, we train models using videos from diverse scenes while maintaining similar quantities of QA pairs and video samples. Under these controlled conditions, we evaluate and compared performance across different scale-dependent scenarios. In Tab.C16, it seems indoor data is the easiest task. We hypothesize that a human-scale estimation bias—arising because both humans and GPT focus on objects expressible in basic units like meters in pretraining corpora—leads to this preference.

C.6. 3D Geometric Feature Observation

- Table C17. Comparison of the robustness of the model training of 3D and 2.5D. All the models are trained on 3D or 2.5D data along with the video. However, we vary the evaluation input of these models to see the robustness. “–” denotes experiments we consider unnecessary. “low” means using low resolution visual for 3D reconstruction. This table includes only the popular model for which a detailed score is available. For average-score comparisons, see Table 2. “(n%)” means the relative decrease compared to the original input.

###### Settings Eval Input VSI-bench SpaceVista-Bench

visual w/. 3D 44.3 31.4 visual w/. 3D (low) 38.1 (-14%) – visual w/o. 3D 34.0 (-23%) –

Training with w/. 3D

visual w/. 2.5D 45.6 33.0 visual w/. 2.5D (low) 43.9 (-4%) 32.3 (-2%) visual w/o. 2.5D 40.7 (-10%) 29.1(-12%)

Training with w/. 2.5D

In addition to introducing VGGT(Wang et al., 2025a) and DINO v3(Siméoni et al., 2025) as extra signals, we conduct a series of targeted ablation studies. This suggests that representation formats like VGGT, when used in their native encoder output, are wonderful for capturing geometry information, but suboptimal for capturing semantic information or overall scenes, especially for low resolution and uncommon scenarios. In Tab.C17, we use “3D” to denote the pure geometric features from VGGT, and “2.5D” to denote the additional 12 viewing angles of the overall scene rendered by the decoder and the renderer. We use the special prompt and the image token to provide

As shown in Tab.C17, 2.5D is usually more robust in spatial reasoning. Rendering to 2.5D enables effective exploitation of pretrained image tokenizers, which in turn provides more reliable semantic information.

Below is the special prompt for 2.5D finetuning.

“Please think about this question as if you were a human pondering deeply. Consider detailed information from the video frames and coarse spatial information from the 3D point cloud image. Provide the model’s thought process and reasoning between the <think> </think> tags, and give your final answer between the <answer> </answer> tags. <video> The images below are obtained from the 3D point clouds based on the video frames above. The following point cloud images are randomly selected viewpoints; some may be completely unhelpful, while others may contain important information. Please discern carefully. <image> Provide your reasoning between the <think> </think> tags and your final answer between the

<answer> </answer> tags.”

###### Table C18. The release time and model source of LLMs used.

Model Release Time Source GPT-5(OpenAI, 2025) 2025-08 https://openai.com/gpt-5/ GPT-4o(Hurst et al., 2024) 2024-05 https://gpt4o.ai/ Claude-Opus-4.1(Anthropic, 2025b) 2025-08 https://www.anthropic.com/news/claude-opus-4-1 Claude-Sonnet-4(Anthropic, 2025a) 2025-05 https://www.anthropic.com/claude/sonnet Gemini-2.5-Pro(DeepMind, 2025) 2025-06 https://deepmind.google/technologies/gemini/pro/ Gemini-2.5-Flash(DeepMind, 2025) 2025-06 https://deepmind.google/models/gemini/flash/ Internvl3.5-38B (Wang et al., 2025c) 2025-08 https://huggingface.co/OpenGVLab/InternVL3_5-38B-Instruct Internvl3.5-14B (Wang et al., 2025c) 2025-08 https://huggingface.co/OpenGVLab/InternVL3_5-14B-Instruct Internvl3-78B (Zhu et al., 2025) 2025-04 https://huggingface.co/OpenGVLab/InternVL3-78B Internvl3-38B (Zhu et al., 2025) 2025-04 https://huggingface.co/OpenGVLab/InternVL3-38B GLM-4.5V (Team et al., 2025) 2025-08 https://www.glm45.com/glm45v GLM-4.1V-Thinking (GLM et al., 2024) 2025-07 https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking Qwen2.5VL-72B (Bai et al., 2025) 2025-01 https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct Qwen2.5VL-32B (Bai et al., 2025) 2025-01 https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct LLAVA-Onevision-72B (Li et al., 2024a) 2024-08 https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf LLAVA-Onevision-7B (Li et al., 2024a) 2024-08 https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov

###### Table C19. Parameter settings for Closed-Source LLMs generation.

Model Generation Setup GPT-5 "model" : "gpt-5", "temperature" : 0, "max_tokens" : 1024 GPT-4o "model" : "gpt-4o", "temperature" : 0, "max_tokens" : 1024

Claude-Opus-4.1 "model" : "claude-opus-4.1", "temperature" : 0, "max_tokens" : 1024 Claude-Sonnet-4 "model" : "claude-sonnet-4", "temperature" : 0, "max_tokens" : 1024 Gemini-2.5-Pro "model" : "gemini-2.5-pro", "temperature" : 0, "max_tokens" : 1024 Gemini-2.5-Flash "model" : "gemini-2.5-flash", "temperature" : 0, "max_tokens" : 1024

#### C.7. Leaderboard Settings Detail

To assess the spatial reasoning ability of both closed-source and open-source models, we evaluate the latest available versions. Tab. 5 presents their performance across the Tiny Tabletop, Tabletop, Indoor, and Outdoor scenarios, whereas

- Tab. C18 provides an overview of their release dates and sources. For closed-source models accessed via API and open-source models, the generation configurations are summarized in
- Tab. C19 and C20, respectively.

Table C20. Generating parameters for Open-Source LLMs.

Model Generation Setup

Internvl3.5-38B do_sample = False, temperature = 0, max_new_tokens = 512 Internvl3.5-14B do_sample = False, temperature = 0, max_new_tokens = 512

Internvl3-38B do_sample = False, temperature = 0, max_new_tokens = 512 Internvl3-78B do_sample = False, temperature = 0, max_new_tokens = 512

GLM-4.5V do_sample = False, temperature = 0, max_new_tokens = 1024 GLM-4.1V-Thinking do_sample = False, temperature = 0, max_new_tokens = 1024

Qwen2.5VL-32B do_sample = False, max_new_tokens = 1024 Qwen2.5VL-72B do_sample = False, max_new_tokens = 1024

LLAVA-Onevision-7B do_sample = False, temperature = 0, max_new_tokens = 1024 LLAVA-Onevision-72B do_sample = False, temperature = 0, max_new_tokens = 1024

### D. FAQ

- D.1. Error Accumulation

Our data construction pipeline is primarily based on metric depth estimation and the corresponding transformation to canonical view space. It should be noted that this approach may introduce potential error accumulation, especially considering that current metric depth estimation models have not yet achieved high performance at full scale.

To address concerns regarding error accumulation, we justify our methodology from the following perspectives: 1) data quality assurance: To ensure alignment with human perception, we implement a multi-tiered validation process. Specifically, we conduct manual verification on a subset of the training set, perform full human annotation on the entire test set, and additionally collect real-world measured data to construct a dedicated test subset. These measures effectively ensure that the automatically generated data remains suitable for learning human perceptual models. We argue that even if minor error accumulation exists, it does not compromise the overall quality and contribution of the dataset. 2) forward-looking methodological contribution: The proposed data construction framework and model architecture will have a significant impact on the field of all-scale spatial reasoning. Importantly, as more accurate all-scale inference methods emerge in the future, we will continuously integrate higher-quality data to refine this work. This dynamic updating mechanism ensures the long-term relevance and value of our research.

- D.2. All Scale Possibilities

Currently, our data coverage remains limited in addressing the full spectrum of spatial scales, despite the equal importance of spatial understanding across these domains. At fine scales, domains such as minimally invasive surgery call for millimeterlevel models, while precision manufacturing—especially semiconductor production—pushes into the nanometer range. These capabilities underpin progress in healthcare and technology. In contrast, large-scale applications, including satellite remote sensing and cartography, typically work with resolutions of 10 kilometers or greater.

While spatial understanding is equally essential across these extremes, the imaging and 3D modeling techniques involved extend well beyond conventional real-world sensing methods. As a result, our current work does not fully address these diverse scales. Nevertheless, we aim to expand our capabilities in the future by integrating modeling across a broader range of dimensions, thereby bridging these gaps and enabling more unified spatial analysis.

- D.3. Dataset Usage Discussion

We use the free-form subset of SPAR-7M(Zhang et al., 2025c), which consists of approximately 100K samples, about 1% of the original dataset. This part of the data is later processed and filtered with original Scannet (Dai et al., 2017), Scannet++ (Yeshwanth et al., 2023), and ARKitScenes (Baruch et al., 2021) to fit the requirements of our dataset. However, we do not consider our model to be trained on SPAR-7M, nor do we compare it against models trained on SPAR-7M in SparBench. We observe that SPAR-7M’s data design leads to over 200 QA pairs per scene on average, which can cause overfitting in indoor scenarios. Instead, we leverage SPAR-7M’s scan-based characteristics to construct our own CoT for cold-start purposes. It is important to note that neither SpaceR nor SPAR-7M includes CoT reasoning. We generate CoT following the method described in Sec.§§ 3 and apply filtering and screening to ensure quality. These processed data sources, along with the wild video dataset, are integrated into SpaceVista-1M, while acknowledging the additional labeling and filtering steps involved in our pipeline. Overall, these decisions support our position that our data retains a meaningful degree of independence from SPAR-7M and SpaceR.

### E. Preview

#### E.1. Scene Preview

Indoor Scenes. Our indoor dataset consists of simple and clean room-scale environments such as living rooms, meeting rooms, and classrooms. An overview of the data is provided in Fig. E19, highlighting the simplicity and cleanliness of our indoor scenes compared to more complex wild indoor environments. Living rooms feature sofas, coffee tables, and shelves arranged along walls with open floor space. Meeting rooms include evenly spaced chairs around a central table, while classrooms have rows of desks facing a blackboard or screen. These scenes show limited object variety and limited scene complexity.

Wild Indoor Scenes. Representative wild indoor scenes, captured via multi-view smartphone recordings in complex and unconstrained environments such as shopping malls, banquet halls, and art galleries, are illustrated in Fig. E20. These scenes exhibit diverse architectural layouts and high object density. Like in shopping malls, elements such as escalators, display shelves, and glass facades create multi-layered structures with frequent reflections and occlusions. Compared to previous indoor scenes, wild indoor scenes have irregular layouts, dense furniture, diverse objects, and uneven lighting, leading to more complex spatial arrangements. This contrast underscores the structured and clear nature of our data, which supports controlled spatial reasoning evaluation.

Outdoor Scenes. Our outdoor scenes include various environments such as parks, tourist landmarks, and others, captured from both ground and aerial views, as shown in Fig. E21. Parks contain irregularly shaped walking paths winding through dense clusters of trees, shrubs, and open lawns, creating a mix of natural textures and spatial variations. These areas often include water features, benches, and varied terrain elevations.

Therefore, outdoor scene layouts usually involve plazas, staircases, and structured open spaces that introduce rich geometric complexity.

Drone Scenes. Fig. E22 shows examples from a drone’s perspective. Aerial, low-angle, and oblique views offer detailed spatial structures that are not easily visible from the ground. Playgrounds exhibit clear arrangements of play equipment and open spaces, while parking lots display orderly rows of vehicles and marked boundaries. Parks show clusters of trees, pathways, and water bodies, revealing a layered combination of natural and built elements.

These diverse viewpoints provide a more complete understanding of scene layout and environmental features, supporting improved spatial reasoning.

Tabletop Scenes. Examples of tabletop scenes are illustrated in Fig. E23. These scenes capture everyday objects such as keyboards, boxes, and fruits arranged on tabletops, characterized by natural occlusions, varying object placements, and diverse background textures. The dataset employs dynamic multi-view acquisition using mobile devices, enabling richer structural coverage compared to traditional static indoor datasets. This approach captures subtle interactions between objects and background elements, as well as changes in viewpoint and lighting conditions.

Tiny Tabletop Scenes. The Fig. E24 shows the tiny tabletop scenes from our dataset. These data are 360-degree turntable videos to capture objects from every angle, solving occlusion issues and improving scene completeness.

Our Collected Scenes.We use mobile devices to capture and collect data for some Tabletop and Tiny Tabletop scenes. Our collected data, shown in Fig. E25, features diverse objects and detailed multi-view coverage, enabling fine-grained spatial analysis. The data is similar to the previously mentioned tabletop and tiny tabletop. Tabletop scenes have relatively large objects and rich and diverse backgrounds, which are suitable for capturing diverse objects and natural environments in daily life; while Tiny Tabletop scenes focus on smaller objects, emphasizing detail integrity and multi-view coverage, which facilitates in-depth research on the subtle structure and morphology of these scenes.

#### E.2. Template Preview

As shown in Tab. E21, we present three exemplar applications: point input for Object Counting, bounding box input for Object Distance, and original input for Spatial Relation. Other scenes and tasks are similar to the example template.

Table E21. Multi-type template preview. Examples using the point input for Object Counting, the bounding-box input for Object Distance, and the original input for Spatial Relation.

###### Point Input Template

- - Refer to the red point in the starting frame and count how many objects are of that type.
- - Count the number of objects whose class is referred to by the red point in the first frame throughout the video.
- - Using the red point in the first frame as reference, count how many objects of that class appear in the entire video.
- - Count every object like the one highlighted by the red point in the video’s first frame.
- - Find all video objects that are of the same kind as the one identified by the red point.
- - Identify the class from the red point in frame one and tally all instances of that class in the video.
- - How many objects in the video resemble the one tagged with the red point in the first frame?
- - Search for all items that belong to the same class as the one shown by the red point in frame one.
- - Track all objects of the same category as the red-point one from the first frame and count them.
- - Count the total number of objects in the video that correspond to the class defined by the red point in the first frame.
- - Use the red point to find a class and count how many such instances are there in the video.
- - Using the initial frame’s red point as a guide, total up all objects of that class.
- - From the first frame’s red point, find that class and count its appearances across the video.
- - Match the object under the red point to others in the video and count them.
- - Take the red-pointed object as example and count all others like it in the video. Bounding Box Input Template

- - How far apart do the objects enclosed by the red bounding box and blue bounding box appear in these frames?
- - What space lies between the red bounding box and the blue bounding box in these frames?
- - What is the distance measurement between the red bounding box and blue bounding box in the video?
- - What is the distance between the red-bounded object and the blue-bounded object in the video?
- - Measure the distance separating the red bounding box and blue bounding box in the video frames.
- - What is the estimated distance between the red bounding box and the blue bounding box in the video?
- - What is the measured distance between the red bounding box and blue bounding box in the footage?
- - Calculate the ground distance from the red bounding box to the blue bounding box based on the frames.
- - Find the ground distance between the red bounding box and the blue bounding box in these images.
- - How wide is the space between the red bounding box and the blue bounding box in the video?
- - Based on the frames, what is the distance from the red bounding box to the blue bounding box?
- - Please estimate the ground distance between the red bounding box and the blue bounding box in these images.
- - What is the approximate distance between the red bounding box and blue bounding box in these images?
- - Provide an estimate for the distance between the red bounding box and blue bounding box seen in the footage.
- - How far is the red bounding box from the blue bounding box in the frames? Original Input Template

- - Describe how desk and chair are spatially positioned relative to each other.
- - What is the spatial relation type between desk and chair in the video?
- - What type of spatial relationship exists between desk and chair in these frames?
- - Estimate the spatial relation (such as support, stacking, adhesion, hanging, plug-in) between desk and chair in these frames.
- - What is the most likely spatial relationship (support, stacking, adhesion, hanging, plug-in) between cabinet and book?
- - Can you describe the spatial relationship type of awning and awning?
- - Identify how picture and ceiling are spatially related in the video sequence.
- - Between desk and chair, what spatial link exists?
- - What spatial relation links tag to hat in the given frames?
- - What spatial relation best fits cable and computer mouse in the video frames?
- - Identify how cable and socket are spatially related in the video sequence.
- - Describe the spatial relation (e.g., support, stacking, adhesion, hanging, plug-in) between fork and spoon.
- - Explain the spatial relation between toy camera and building blocks in the video.
- - How would you classify the spatial relation between sticky note and tumbler?
- - What type of spatial relationship exists between toy block and toy train in these frames?.

[Figure 317]

###### Figure E19. Preview of indoor data. Indoor data are rather simple and clean scenes inside a room. The overall scene is not as complex as the wild indoor scene.

[Figure 318]

###### Figure E20. Preview of wild indoor data. Wild indoor data includes more light changes, reflections, and transparency. The objects included are more diverse.

[Figure 319]

###### Figure E21. Preview of outdoor data. Outdoor data is jointly collected from ground views, incorporating street, park, building and so on.

[Figure 320]

###### Figure E22. Preview of drone data. Drone data captures ground objects from above at oblique angles, providing more complete structural coverage than traditional ground-based capture methods.

[Figure 321]

Figure E23. Preview of tabletop data. In this tabletop scene, videos capture tabletop objects exhibiting rich background variation and natural occlusions, delivering clearer structural coverage of the objects than traditional static indoor datasets.

[Figure 322]

- Figure E24. Preview of tiny tabletop data. Tiny tabletop objects captured with rich details for small objects, focusing on fine-scale scenes, unlike typical large or complex indoor or outdoor datasets.

[Figure 323]

###### Figure E25. Preview of self-collected data. These samples are collected by us. As small-scale, Tabletop, and Tiny Tabletop datasets offer rich details with accurate annotations.

