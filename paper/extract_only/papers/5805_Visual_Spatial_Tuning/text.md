# arXiv:2511.05491v1[cs.CV]7Nov2025

[Figure 1]

## Visual Spatial Tuning

### Rui Yang1∗, Ziyu Zhu3∗, Yanwei Li2†, Jingjia Huang2, Shen Yan2, Siyuan Zhou2, Zhe Liu1, Xiangtai Li2, Shuangye Li2, Wenqian Wang2, Yi Lin2§, Hengshuang Zhao1§

1The University of Hong Kong, 2ByteDance Seed, 3Tsinghua University

### Abstract

Capturing spatial relationships from visual inputs is a cornerstone of human-like general intelligence. Several previous studies have tried to enhance the spatial awareness of Vision-Language Models (VLMs) by adding extra expert encoders, which brings extra overhead and usually harms general capabilities. To enhance the spatial ability in general architectures, we introduce Visual Spatial Tuning (VST), a comprehensive framework to cultivate VLMs with human-like visuospatial abilities, from spatial perception to reasoning. We first attempt to enhance spatial perception in VLMs by constructing a large-scale dataset termed VST-P, which comprises 4.1 million samples spanning 19 skills across single views, multiple images, and videos. Then, we present VST-R, a curated dataset with 135K samples that instruct models to reason in space. In particular, we adopt a progressive training pipeline: supervised fine-tuning to build foundational spatial knowledge, followed by reinforcement learning to further improve spatial reasoning abilities. Without the side-effect to general capabilities, the proposed VST consistently achieves state-of-the-art results on several spatial benchmarks, including 34.8% on MMSI-Bench and 61.2% on VSIBench. It turns out that the Vision-Language-Action models can be significantly enhanced with the proposed spatial tuning paradigm, paving the way for more physically grounded AI.

Project Leader †: Yanwei Li at yanwei.li@bytedance.com Correspondence §: Yi Lin at linyi.james@bytedance.com, Hengshuang Zhao at hszhao@cs.hku.hk Code: https://github.com/Yangr116/VST

### 1 Introduction

Vision-Language Models (VLMs) [1, 13, 17, 27, 62] have achieved remarkable success across a wide range of domains, such as visual question answering [38, 79], document understanding [39, 44], and autonomous GUI agents [67]. However, these models exhibit limitations in capturing spatial relationships from sequential visual observations [72, 75]. This spatial understanding ability is a foundational component of general intelligence, presents across a broad spectrum of animals, including humans [28, 48]. The deficiency significantly constrains current VLMs to effectively interact with the physical world, thereby limiting their application in fields such as robotics [8, 86], autonomous driving [59], and augmented/virtual reality (AR/VR) [25]. To mitigate this issue, several studies have explored the incorporation of additional expert encoders [6, 22]. However, this approach often introduces extra complexity and can negatively impact the general capabilities of the models. Alternatively, other research efforts have focused on the development of specialized datasets [11, 19, 47, 69, 77, 82], aiming to enhance the spatial understanding abilities of VLMs.

∗Equal contribution.

Data Type Data Usage

Method

SI MI Video SFT RL

SpatialVLM [11] ✓ ✗ ✗ ✓ ✗ SAT [49] ✓ ✗ ✗ ✓ ✗ MM-Spatial [19] ✓ ✗ ✗ ✓ ✗ SPAR [82] ✓ ✓ ✓ ✓ ✗ Space-R [47] ✗ ✗ ✓ ✓ ✓ VLM-3R [22] ✗ ✗ ✓ ✓ ✗

VST (ours) ✓ ✓ ✓ ✓ ✓

VLM VST for Spatial Understanding

###### VST Dataset

Response

Depth Estimation

3D Object Detection

[Figure 2]

[Figure 3]

Distance Estimation

Language Model

VST for Robot Action

Position Relationship

…

ViT Tokenizer

Spatial Reasoning

SI/MI/Video

Text

Table 1 Comparison with spatial dataset.

Figure 1 Overview of our VST framework.

Nevertheless, these arts have typically concentrated on limited or isolated aspects of spatial understanding. As summarized in Table 1, some studies focus only on the supervised fine-tuning stage, while others are restricted to the single scenario, overlooking the diversity of visual input. To this end, we introduce a comprehensive and integrated framework, termed Visual Spatial Tuning (VST), which is designed to holistically cultivate human-like visuospatial abilities in VLMs. As illustrated in Figure 1, VST effectively augments the spatial capabilities of existing VLMs through the construction of an extensive and carefully curated dataset. This enhancement proves advantageous for downstream Vision-Language-Action (VLA) tasks.

To develop the VST, we deconstruct spatial ability into two key components: spatial perception and spatial reasoning. We define spatial perception as the ability to discern the spatial relationships between objects, and spatial reasoning as the ability to build and mentally manipulate an internal model of an environment. These two components correspond to the concepts of perceptual and conceptual spatial ability, respectively, as proposed in cognitive science [48]. Effective spatial perception requires the model to possess foundational spatial knowledge—specifically, the ability to identify both "what is it?" and "where is it?" within its peripersonal space. While existing VLMs can accurately recognize objects and locate them within pixel space using 2D points or bounding boxes [3, 20, 62], their ability to determine object positions in 3D space remains limited [41, 60]. Therefore, we introduce the VST-Perception (VST-P) dataset, comprising 4.1 million samples across 19 diverse tasks. The dataset incorporates single-image data to facilitate VLMs in discerning spatial relationships beyond the pixel level, which is an essential step towards bridging the gap between pixel space and 3D space. In addition, multi-image data is included to enhance the ability to comprehend spatial relationships from multiple viewpoints, and video data enables the capture of spatiotemporal relationships. Collectively, this dataset provides a comprehensive foundation for advancing spatial perception in VLMs.

Beyond foundational spatial perception, we expect the model to mentally represent spatial relationships beyond its own body, thereby engaging in advanced spatial reasoning. To this end, we introduce the VSTReasoning (VST-R) dataset, which comprises samples featuring chain-of-thought (CoT) processes to facilitate the spatial reasoning ability, as well as samples with rule-checkable answers to further enhance its reasoning capabilities. In spatial reasoning, we place particular emphasis on multi-image scenarios, as these necessitate the model’s ability to identify connections among objects and cameras, and to mentally reconstruct spatial layouts. However, when generating spatial CoT, the limited multi-view spatial understanding of current large VLMs [72, 75] poses challenges for directly synthesizing accurate layout descriptions and coherent reasoning chains. Drawing inspiration from human cognition, we propose prompting with Bird’s-Eye View (BEV) annotation. It leverages a top-down perspective to explicitly convey spatial relationships between objects, thereby improving the quality of both generated layout descriptions and CoT reasoning process.

Building upon the introduced VST-P and VST-R datasets, we demonstrate that it is unnecessary to incorporate a special encoder with 3D inductive biases into VLMs for achieving strong spatial capabilities. Instead, we propose to inject visual spatial knowledge into VLMs through supervised fine-tuning and further enhance spatial reasoning capabilities via reinforcement learning. This progressive approach mirrors the development of human spatial intelligence [48], i.e., establishing a foundation in spatial perception before developing higherlevel spatial reasoning abilities. As a result, our proposed VST framework consistently achieves state-of-the-art

Single-Image (64.8%)

CoT (77.8%)

3D Object Detection (1586K) 3D Grounding (170K) Depth Comparison (412K) Distance Prediction (251K) Scene Caption (106K) Measurement (175K)

SR-SI (15K) SR-Camera Motion (6K) SR-Object-Object (15K) SR-Camera-Camera (6K) GR-Math (31K) GR-OCR (9K) GR-Knowledge (24K)

###### Multi-Image (33.1%)

###### VST-R

###### VST-P

Correspondence (361K) 3D Object Detection-MI (600K) Object-Object (230K) Camera-Camera (63K) Camera Motion (88K) Scene Caption (35K)

135K

4.1M

###### RL (22.2%)

SR-SI (2K) SR-Camera Motion (1K) SR-Object-Object (5K) SR-Camera-Camera (2K) GR-Math (8K) GR-OCR (2K) GR-Knowledge (10K)

###### Video (2.1%)

Measurement (3K) Object-Object (19K) Object Counting (12K) Route Plan (4K) Spatiotemporal (10K) Scene Caption (5K) Others (35K)

(a) Perception data distribution.

(b) Reasoning data distribution

Figure 2 Overview of the VST dataset. (a) The distribution of VST-P, which is used for SFT. (b) The distribution of VST-R, which is used for CoT cold start and RL. ‘SR’ denotes spatial reasoning, and ‘GR’ denotes general reasoning.

performance on multiple spatial benchmarks, attaining 87.8% on CVBench [60], 34.8% on MMSI-Bench [75], and 61.2% on VSIBench [72], while preserving the general multi-modal capabilities. Furthermore, the spatial proficiency from VST demonstrably enhances broader VLA tasks. For instance, Qwen2.5VL-3B [3] fine-tuned on our VST yields an 8.6% improvement on the LIBERO benchmark [35].

### 2 Dataset

In this section, we introduce the VST dataset, specifically developed to enhance the spatial perception and reasoning capabilities of VLMs. First, we construct a large-scale dataset, VST-Perception (VST-P), to equip VLMs with comprehensive spatial knowledge. Building upon this foundation, we further create the VST-Reasoning (VST-R) dataset to enable VLMs to reason in space.

#### 2.1 VST-Perception

As illustrated in Figure 2a, the VST-P dataset contains 4.1 M samples across 19 different tasks for supervised fine-tuning, covering three primary vision scenarios, i.e., single-image, multi-image, and video. Specifically, single-image data constitutes the majority (64.8%), multi-image data accounts for 33.1%, and video data makes up the remaining small portion (2.1%).

Single-image. Since monocular images are easily obtainable, single-image data constitutes the largest category. This category primarily encompasses tasks such as relative depth estimation (2.5D), 3D object detection, and distance estimation. These tasks bridge the gap between 2D pixel coordinates and the 3D physical world, thereby facilitating the acquisition of spatial knowledge and the development of spatial awareness in VLMs. To collect this data, we create dedicated data engines to gather data with depth maps and data with 3D bounding box annotations, as shown in the top left of Figure 3. The depth data mainly comes from public datasets and synthetic data. The open-source data originates from ScanNet++ [76], which is collected using real-world devices, and Hypersim [50], which is generated by a simulator. To increase the diversity of depth data, we use a depth expert model [73] to create pseudo labels for wild images from the COCO dataset [34]. After obtaining the depth maps, we convert them to the same coordinate system and generate depth-related visual instruction samples. The reference formats for depth-related samples encompass text-based, point-based, box-based, and visual-prompt-based representations. These diverse formats enable VLMs to infer the relative distance from objects to the camera plane.

For the 3D data engine, we use two main approaches. The first approach relies on open-source datasets, including ScanNet [18], ARKitScenes [4], Hypersim [50], SUN-RGBD [54], Matterport3D [10], and Objectron [2]. Since the 3D bounding boxes from ScanNet and Matterport3D are axis-aligned, we use the corrected versions from EmbodiedScan [63] to ensure greater accuracy. Notably, each dataset is designed for distinct applications, provides visual data in either video or image format, and annotates objects in varying coordinate systems. Therefore, we standardize all collected 3D bounding boxes to a unified camera coordinate system and process the raw visual data to reduce repetition and occlusion. The second approach is generating data using a

##### VST Data Engine

##### VST Model Ability

Single-image Engine (2.5D)

###### 2.5D Spatial Perception from Monocular Images

Question Template

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Filtering

RGB-D Image

[Figure 11]

Unified Format

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Q: There are several points in the image. Present the object represented by these points in an order that goes from far to close. A: D, C, A, E, B

Q: Which one in boat-A (bbox_2d: [207, 118, 416, 306]) and boat-B (bbox_2d: [0, 120, 220, 390]) is farthest from the camera? A: boat-A.

Q: Based on your analysis of the image, which object—bottle (red point) and trash can (green point)—do you think is farther from the camera? A: trash can

Expert Model

Web RGB Images

Depth

Single-image Engine (3D)

3D Spatial Perception from Monocular Images

[Figure 16]

[Figure 17]

Question Template

[Figure 18]

[Figure 19]

[Figure 20]

###### 3D Object Detection

###### Attribute Measurement

Distance Estimation

[Figure 21]

Filtering

[Figure 22]

Unified Format

[Figure 23]

[Figure 24]

RGB-D Scan

[Figure 25]

Q: Detect the 3D bounding box of the chair pillow, … A:{“chair”: 1.23, 0.84, 4.55, 0.98, 0.70, 0.94, 0.01, 0.00, 0.00], “pillow”: …}

Q: Provide the height of the table (pink point-1)? Specify the measurement in centimeters. A: 71 cm

Q: By estimating the real-world distances, identify which of the mirror, or the blinds is nearest to the counter. A: blinds.

Simulator

Synthetic Scene Rendering

Multi-image and Video Engine

Multi-image and Video Spatial Perception

[Figure 26]

[Figure 27]

Question Template

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

###### Correspondence

Position Relationship

###### Camera Motion

[Figure 33]

Unified Format

[Figure 34]

Sampling

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

…

Q: In the first image, a point is circled in pink. The second image presents several points labeled '1, 2, 3, 4’. Which corresponds to the original? A: point-4

Q: At the camera position of the second image, the direction toward backpack (gold point) is south, which direction is blanket (blue point) relative to me? A: southwest

Q: Images are recorded from a first-person perspective. Determine the camera's direction of sideways translation (left or right) shown in these images. A: Left

Raw Data

CoT Engine

Spatial Reasoning

[Figure 42]

VLM Auto Write

[Figure 43]

[Figure 44]

Q: If the small metal trash can on the floor near the white storage cabinets in the first image is north of the small printer on the white cabinet near the door in the second image, what direction is the clear plastic bottle on the white desk with two monitors in the second image from the printer? Options: A. southeast, B. northeast, C. south, D. northwest A: <think>First, I look at the first image. The scene has a long row of white storage cabinets along the left wall a white desk with shelves holding folders on the right, and a small metal trash can sitting on the floor between the cabinets and the desk. There's also a coat hanger in the corner and some boxes stacked nearby. … So, from the printer's view: the bottle is to the left (the same direction as north, since north is toward the trash can on the left) and forward (since the desk is closer than the trash can). Combining these, the bottle is in the northwest direction from the printer. That matches option D.</think>D. northwest

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Large VLM

[Figure 49]

QA data 3D boxes

BEV Images

Figure 3 Data engines of VST (left) and the capabilities they enable in VST-Model (right).

simulator. Specifically, we use Isaac Sim1 to synthesize data, and scenes are from the GUTopia [61]. With the large-scale data with 3D bounding boxes, we create visual instruction samples for 3D object detection, 3D grounding, attribute measurement, and distance estimation tasks.

In the 3D object detection task, we predict the 9-DoF bounding box in the camera coordinate system. Specifically, the 3D bounding box is defined by (x,y,z,xl,yl,zl,p,y,r), where (x,y,z) are the box center, (xl,yl,zl) are the length along the X, Y, and Z axes, and (p,y,r) are the rotation angles. However, a significant challenge in utilizing datasets aggregated from disparate sources is the inherent variability in camera intrinsics, which introduces geometric inconsistencies that can hinder model generalization and scalability. To mitigate this issue, we introduce a Field of View (FoV) unification strategy. This approach normalizes the input data by projecting all images onto a virtual camera with a predefined, uniform FoV. This process creates a standardized visual input, akin to data captured by a single virtual camera, thereby eliminating intrinsicrelated discrepancies for the 3D object detection task. In addition, when creating the instruction data, we mix single-turn and multi-turn formats. The multi-turn format data allows each subsequent box to reference the previous one during training, helping the model learn the layout information.

Furthermore, if we rely solely on template-based 3D object detection data for training, the VLM may overfit to the specific numerical values and fail to generalize spatial understanding. Therefore, to help the VLM better comprehend spatial information at the language level, we introduce the scene caption. Unlike general

1https://developer.nvidia.com/isaac/sim

captions, which primarily describe image content, scene captions focus on the layout information and spatial relationships within the image. To obtain such scene captions, we prompt a large VLM [27] with ground-truth

- 3D bounding boxes and object relationships extracted from the scene graph [85]. The resulting captions not only describe the objects present in the image, but also provide detailed layout information and spatial arrangements.

Multi-image. The second category comprises multi-image data, which supports tasks such as multi-view 3D object detection, multi-view correspondence, object-object relationship understanding, and camera motion analysis. These tasks are designed to enhance VLMs to comprehend spatial relationships across different viewpoints. As illustrated in the third data engine of Figure 3, we sample multi-image data from RGB-D scans sourced from ScanNet [18], ScanNet++ [76], and ARKitScenes [4]. For correspondence tasks, we utilize point clouds and depth maps from various viewpoints to identify matched points. To unify object information across multiple images, we transform all objects into the camera coordinate system of the first image. For camera motion data, we represent camera poses using Euler angles. After these unification steps, we generate template-based visual instruction samples. In the multi-image scenario, we also create the scene caption to reconstruct the scene layout by text and describe the spatial information represented by multiple RGB images.

Video. The third category consists of video data, which enables the model to capture spatiotemporal relationships through tasks such as identifying the order of appearances and counting objects. To construct the video dataset, we employ the same data engine used for multi-image data. The only difference is that we add the appearance time for each object. Furthermore, we enhance the video dataset by sampling two-thirds of the data from VLM-3R [22], reorganizing it into a multi-turn format rather than a single-turn format.

With the introduction of the VST-P dataset, the VLM exhibits significantly enhanced fundamental capabilities in comprehending spatial relationships. Notably, there is a ∼20% improvement on CVBench-3D [60], a ∼5% increase on BLINK [23], and a ∼16% gain on VSIBench [72], as illustrated in Tables 5, 6, and 7.

#### 2.2 VST-Reasoning

As shown in Figure 2b, the VST-Reasoning (VST-R) dataset contains 135K samples with two parts: one part includes CoT steps to teach the model how to reason, and the other part provides rule-checkable data used in online RL to improve the reasoning ability. Besides spatial data, both parts include general data to preserve the original general abilities. Most spatial reasoning samples come from multi-image scenarios, which require reconstructing scene details and inferring spatial relations.

For the spatial reasoning samples with the CoT process, we develop a data engine, as illustrated in the bottom left of Figure 3. Specifically, we sample data from template-based question-answer pairs and employ a large VLM [27] as the teacher to generate detailed CoT reasoning steps. Recognizing that the multi-view spatial understanding of the current large VLMs remains limited relative to their general multi-modal abilities, we introduce a novel strategy named prompting with BEV annotation. Specifically, this method leverages ground-truth 3D bounding boxes to visualize the BEV image of the scene represented by multiple images. During generation, we provide RGB images, the corresponding BEV visualizations, detailed object information, and question-answer pairs to prompt the teacher VLM. The BEV images serve as an auxiliary spatial prompt, allowing the teacher model to better capture spatial relationships compared to using only RGB images. As a result, the generated reasoning processes are more coherent and accurate. For the CoT format, we adopt a textual representation rather than utilizing 3D bounding boxes or cognition maps [77], as the textual format offers greater generality. In particular, during the reasoning process, the model first reconstructs the spatial layout by text and subsequently infers the correct answer.

With the VST-R dataset, the VLM demonstrates significantly enhanced spatial reasoning abilities. As illustrated in Table 11, there is an 8.9% improvement on MMSI-Bench [75].

### 3 Method

Our target is to equip general VLMs with 3D knowledge for better spatial understanding and reasoning from common visual inputs. Therefore, we chose Qwen2.5-VL [3] as the base model because it can accurately identify

Output Text Action De-Tokenizer

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Language Model

Language Model

Vision Encoder Tokenizer

Vision Encoder Tokenizer

Input Image

Language Instruction

Input Image

Language Instruction

(a) VST Model (b) VST-based VLA Model

Figure 4 (a) The VST model, which incorporates spatial perception and reasoning capabilities. (b) The VST-based VLA model, capable of generating action sequences through an action de-tokenizer.

objects and locate them in pixel space. As shown in Figure 4 (a), it follows the widely used "ViT-MLP-LLM" paradigm: a pre-trained Vision Transformer (ViT) is combined with a large language model (LLM) via an MLP merger.

#### 3.1 Training Strategy

We continued training the base model to endow it with spatial perception and reasoning capabilities. The training process can be divided into three stages.

- Stage 1: Supervised Fine-tuning. At this stage, we incorporate the foundational spatial understanding into the base model by utilizing the proposed VST-P dataset. To maintain the original capabilities of the base model, we also incorporate a portion of general multi-model data from open-source datasets [20, 33]. Assume the base model is parameterized by θ, which can simultaneously process text, images, and video. For any

given training sample x = [x1,...,xL] of length L, we employ visual tokens as the conditioning context for text prediction and adopt the standard auto-regressive objective:

Lθ(x) = −

L

i=2,xi∈text

wi log pθ(xi | x1,...,xi−1), (1)

The resulting model exhibits foundational spatial perception abilities, as illustrated in Table 5, 6, and 7.

- Stage 2: CoT Cold Start. This stage leverages chain-of-thought (CoT) data to instruct the model utilizing reasoning patterns. For example, in spatial reasoning scenarios with limited viewpoints, the model first reconstructs the layout of the scene using text, and then reasons through the given question. To preserve the model’s reasoning ability on general tasks, we also take some general reasoning data. The training objective remains the same as in Equation 1. The resulting model from this stage has basic spatial reasoning capabilities (Table 10), which serves as the initial RL actor.
- Stage 3: Reinforcement Learning. In this stage, we employ RL to further enhance the spatial reasoning capabilities of the stage-2 model. For this purpose, we utilize the Group Relative Policy Optimization (GRPO) algorithm [51], which bypasses the need for a value model by computing the relative advantage of each response within a group of responses to the same question. To facilitate this process, we curated a verification dataset comprising tasks related to spatial understanding, 3D object detection, and general multi-modal understanding. This dataset is categorized into four task types: multiple-choice, open-ended, OCR, and 3D detection. In the GRPO framework, we employ a mixed rule-based reward to evaluate the generated responses. For a given response yˆ and its corresponding ground truth y, the overall reward function is defined as:

R(y,yˆ) = Racc(y,yˆ) + Rformat(y,yˆ). (2)

This function combines an accuracy reward, Racc(·,·), which scores the correctness of the response, with a format reward, Rformat(·,·), which incentivizes adherence to a specified output format. For multiple-choice,

open-ended, and OCR tasks, the accuracy reward is calculated using standard evaluation protocols [24, 43, 44, 53]. For 3D object detection tasks, the reward is a linear combination of the 3D Intersection over Union (IoU) score and the F1 score:

R3d(y,yˆ) = αRiou(y,yˆ) + (1 − α)RF1(y,yˆ)), (3)

where α is a hyperparameter that defaults to 0.5. In detail, given N predicted and M ground-truth 3D bounding boxes, we first establish a bipartite matching [32] between the predictions and the ground truth; Riou(·,·) is then calculated as the average IoU of the successfully matched pairs. To calculate RF1(·,·), we define a true positive as a match with an IoU score exceeding a threshold of 0.25. Following this stage, the model exhibits superior spatial reasoning abilities relative to the cold-start model, as shown in Table 11.

#### 3.2 Expanding to Vision-Language Action Model

With the spatial-enhanced model, a natural question emerges: can the integration of spatial priors improve the performance of Vision-Language-Action (VLA) models in robotic manipulation tasks? To this end, we adapt the pretrained VLM into a VLA model, following the methodology of OpenVLA [31], as illustrated in Figure 4. Specifically, we formulate the action prediction problem as a vision-language task where, given an observation image and a natural language instruction, the model auto-regressively predicts the actions. To accomplish this, we discretize the action space into 256 bins, where each bin corresponds to a special token in the language tokenizer. With the actions tokenized, the entire model is fine-tuned using the objective function defined in Eq 1.

### 4 Experiment

In this section, we first outline our experimental setup, detailing the training and evaluation protocols. Following this, we compare our model’s performance against leading methods across several key benchmarks. Finally, we conclude with a detailed analysis of data effectiveness, the impact of data and model scaling, and other ablation studies.

4.1 Implementation Details

- Stage 1. The initial training stage aims to establish a strong foundation of spatial understanding capabilities. For this stage, we use a global batch size of 128, a sequence length of 16,384, and a dynamic data packing strategy to accelerate the training process. We employ the AdamW [40] optimizer, setting the base learning rate to 5 × 10−5 and the vision encoder’s learning rate to 5 × 10−6. During this phase, we combine our VST data with general multi-modal data from LLaVA-OneVision [33]. This approach allows the model to learn new spatial understanding knowledge while mitigating catastrophic forgetting of its original capabilities. For our ablation studies, we use Qwen2.5-VL-3B [3] as the base model, training it on a mixture of one-third of the VST data and 800K general multi-modal samples. For our final models, we employ Qwen2.5-VL-3B, Qwen2.5-VL-7B, and Qwen2.5-VL-32B as base models, utilizing the entire VST dataset combined with 2.4M general multi-modal samples.
- Stage 2. In the CoT cold-start stage, we continue training the model from the initial foundation stage. The hyper-parameters are adjusted to a global batch size of 128, a base learning rate of 1 × 10−5, a vision encoder learning rate of 1 × 10−6, and a sequence length of 16,384. In this stage, the training data is a mixture of spatial reasoning data and general multimodal reasoning data. We train the model for 2 epochs, as we observed that smaller-scale models require extended training to effectively master the long-form CoT reasoning process.
- Stage 3. In the RL stage, we further refine the model from the second stage using the VeRL [52] framework. For the training objective, we adopt a revised version of the GRPO algorithm [78]. This stage utilizes the AdamW [40] optimizer with a constant learning rate of 1 × 10−6 and a global batch size of 128.

Expanding to VLA Model. When expanding to the VLA model, we still use the AdamW [40] optimizer, but with a modified learning rate schedule: a base learning rate of 5 × 10−5 and a vision encoder learning rate of

###### 5 × 10−6. We set the global batch size to 128 and the max sequence length of data packing to 1024. This

###### Methods CV 3DSR MMSI BLINK VSI MMStar MMB RealworldQA MMMU OCRB AI2D

GPT-4o [1] 76.0 45.3 30.3 65.9 34.0 65.1 84.3 76.2 70.7 80.6 84.9 Gemini-2.5-Pro [17] - - 36.9 70.6 - 77.5 90.1 78.0 81.7 86.6 88.4 Seed1.5-VL [27] 85.2 61.6 29.7 72.1 41.5 77.8 89.9 78.4 77.9 86.1 87.3

LLava-OneVision-7B [33] 61.9 54.4 26.6 48.2 32.4 61.7 80.8 66.3 48.8 62.2 81.4 Qwen2.5-VL-3B [3] 71.8 50.2 26.5 47.6 29.6 55.9 79.9 65.4 47.9 79.7 81.6 Qwen2.5-VL-7B [3] 75.4 53.2 25.9 56.4 38.9 63.9 83.5 68.5 58.6 86.4 83.9 Qwen2.5-VL-32B [3] 81.9 56.7 27.7 59.9 40.9 70.3 84.0 71.2 68.9 85.6 85.4 InternVL3-8B [84] 81.0 55.7 25.7 55.5 42.1 68.2 83.4 70.8 62.7 88.0 85.2 InternVL3-38B [84] 84.9 59.0 26.3 64.0 48.9 71.5 87.6 75.6 70.1 88.6 88.9 MiMo-VL-7B-RL [58] 82.3 50.8 29.3 62.4 37.2 65.1 84.4 68.2 66.7 86.6 83.5 SpaceR-7B [47] 74.8 53.3 20.1 55.4 43.5 61.6 84.3 64.7 53.1 85.9 85.5 SPAR-8B [82] 80.7 57.5 - 43.9 41.1 - 79.9 64.7 - - -

VST-3B-SFT (ours) 84.4 54.1 30.2 59.1 57.9 58.0 80.9 68.4 45.2 83.7 82.5 VST-3B-RL (ours) 84.2 56.5 31.3 57.2 57.7 58.9 80.5 68.5 49.8 80.9 82.4 VST-7B-SFT (ours) 85.5 54.6 32.0 62.1 60.6 63.1 83.3 72.2 50.6 85.5 84.9 VST-7B-RL (ours) 86.5 60.1 34.8 62.6 61.2 63.5 83.0 68.5 49.4 86.1 83.5

Table 2 Comparison with state-of-the-art VLMs on spatial benchmarks and general benchmarks.

adjustment is necessary to compensate for the relatively short action sequences and the small resolution of the training images (256 × 256). The model is finetuned for 2.5K steps and 10K on the LIBERO benchmark [35] in total.

Evaluation. For our evaluation, we assess spatial understanding across three distinct modalities: single-image ability is benchmarked with CVBench [60] and 3DSRBench [41], multi-image ability with BLINK [23] and MMSI-Bench [75], and video-based ability with VSIBench [72]. The average score (S-AVG) across these benchmarks is then calculated to quantify the model’s overall spatial understanding capabilities. To verify its general multi-modal understanding, we also report the average score (MM-AVG) across a suite of standard benchmarks: MMStar [12], MMBench [38], RealworldQA [66], MMMU [79], OCRBench [39], and AI2D [30]. For 3D object detection, we evaluate the model on both the SUN RGB-D [54] validation set (Total3D version [46]) and the ARKitScenes [4] test set (Omni3D version [7]). Performance is measured using standard metrics: average precision (AP) at IoU thresholds of 0.15, 0.25, and 0.50 (denoted as AP@15, AP@25, and AP@50, respectively), and average recall for the top 100 predictions (AR@100).

#### 4.2 Main Results

As shown in Table 2, our VST models achieve competitive results across both spatial and general benchmarks. Notably, VST-7B-SFT and VST-7B-RL deliver leading performance on mainstream spatial understanding tasks. On the CVBench [55], VST-7B-SFT attains 85.5, surpassing the proprietary Seed1.5-VL [27] with 85.2. On MMSI-Bench [75], VST-7B-SFT achieves 32.0, outperforming GPT-4o [1] at 30.3, while RL further boosts VST-7B-RL to 34.8, approaching the proprietary state of the art Gemini-2.5-Pro [17] at 36.9. On BLINK [23], VST-7B-SFT yields 62.1, surpassing Qwen2.5-VL-7B [71] at 56.4. Notably, the VSI-Bench [72] highlights the strength of our models in video spatial understanding. VST-7B-SFT reaches 60.6, and VST-3B-SFT achieves 57.9, substantially ahead of GPT-4o with 34.0. Detailed results are reported in Table 3. Without any specialized 3D encoder, VST-7B-RL delivers the best overall average among comparable VLMs, achieving 61.2. Although VLM-3R-7B [22] attains a similar score, it relies on an additional expert 3D encoder, whereas VST operates with a standard vision backbone. Beyond the overall average, VST shows clear strengths on fine-grained spatial sub-tasks: it leads in Object Size and Room Size estimation, and performs strongly in Relative Direction and Appearance Order. These gains reveal that VST’s visuospatial training enables robust metric and ordinal reasoning over scenes, even without explicit 3D features. Although VST-SFT models obtain only moderate results on general benchmarks—scoring 83.3 for the 7B size and 80.9 for the 3B size on MMBench [38], and 84.9 and 82.5 on AI2D [30], slightly below proprietary state of the art—they still provide a well-balanced overall performance. These outcomes highlight the clear advantage of VST in spatial perception and reasoning while maintaining strong competitiveness in multi-modal understanding.

Obj. Count

Abs. Dist.

Obj. Size

Room Size

Rel. Dist

Rel. Dir.

Route Plan

Appr. Order

Methods Avg.

GPT-4o [1] 34.0 46.2 5.3 43.8 38.2 37.0 41.3 31.5 28.5

- Gemini-1.5-Pro [57] 45.4 56.2 30.9 64.1 43.6 51.3 46.3 36.0 34.6 LLaVA-OneVision-7B [33] 32.4 47.7 20.2 47.4 12.3 42.5 35.2 29.4 24.4 LLaVA-Video-7B [83] 35.6 48.5 14.0 47.8 24.2 43.5 42.4 34.0 30.6 Qwen2.5-VL-7B [3] 32.7 34.5 19.4 47.6 40.8 32.8 24.5 32.5 29.4 SAT-7B [49] - - - - - 47.3 41.1 37.1 36.1 InternVL-Spatial-8B [21] - 68.7 40.9 63.1 54.3 47.7 - 29.9 60.5 SpaceR-7B [47] 43.5 61.9 28.6 60.9 35.2 38.2 46.0 31.4 45.6 VILASR-7B [64] 45.4 63.5 34.4 60.6 30.9 48.9 45.2 30.4 49.2 VLM-3R-7B [22] 60.9 70.2 49.4 69.2 67.1 65.4 80.5 45.4 40.1 VST-3B-SFT (ours) 57.9 69.3 45.4 71.8 62.4 59.0 46.0 38.7 70.2 VST-3B-RL (ours) 57.7 66.6 45.0 72.8 60.9 59.9 47.6 40.7 68.3 VST-7B-SFT (ours) 60.6 72.0 44.4 74.3 68.3 59.7 55.8 44.9 65.2 VST-7B-RL (ours) 61.2 71.6 43.8 75.5 69.2 60.0 55.6 44.3 69.2

Object Count

Appearance Order

Absolute Distance

Object Size

Route Plan

Relative Distance

Relative Direction

Room Size

72.0

70.2

45.4

74.3

44.9

59.7

55.8

68.3

Gemini-1.5 Pro

LLaVA-Video-7B

Qwen2.5-VL-7B

SpaceR-7B

VILASR-7B

VST-7B-SFT (ours)

Table 3 Comparison with state-of-the-art VLMs on VSI-Bench [72].

Methods AP@15

Seed1.5-VL [27] 33.5

- Gemini-2.0-Pro [17] 32.5 Gemini Robotics-ER [56] 48.3

Expert Models

48.3

50

Proprietary Models VST (ours)

###### 44.2

41.6 40.1

| |
|---|

37.3

40

33.5 32.5

AP@15

30

24.1

Total3DU [46] 14.3 Implicit3D [81] 24.1

20

14.3

VST-3B-SFT (ours) 37.3 VST-3B-RL (ours) 40.1 VST-7B-SFT (ours) 41.6 VST-7B-RL (ours) 44.2

10

0

GeminiRobotics-ER VST-7B-RLVST-7B-SFT VST-3B-RLVST-3B-SFT Seed1.5-VLGemini-2.0-Pro Implicit3D Total3DU

Table 4 Comparison AP@15 on SUN RGB-D 3D object detection benchmark [54].

Table 4 summarizes results on the SUN RGB-D [54] 3D object detection benchmark. VST-7B-SFT reaches 41.6 AP@15, while VST-7B-RL improves to 44.2, ranking first among both general VLMs [17, 27] and expert methods [46, 81]. The RL model outperforms Gemini-2.0-Pro [17] at 32.5 and Seed1.5-VL [27] at 33.5, as well as specialized systems such as Implicit3D [81] at 24.1 and Total3DU [46] at 14.3. These findings show that VST, even without auxiliary 3D encoders, can achieve strong 3D object detection purely through visual spatial tuning.

#### 4.3 Ablation Study

Ablation for the single-image data. Our baseline model is the Qwen2.5-VL-3B fine-tuned on a general dataset of 800K samples. From this baseline, we incrementally introduce different types of data to enhance its capabilities. The results are recorded in Table 5. Given that the baseline model already possesses the 2D perception ability, our initial goal is to endow it with 3D spatial awareness by incorporating the 3D object detection task, using only monocular RGB inputs. Introducing these 3D object detection data improves the performance on spatial understanding benchmarks by 1.0%. Notably, this data leads to a 5.7% improvement on CVBench-3D [60], demonstrating that the model has successfully acquired foundational 3D perception skills from the detection task. However, the model fails to generalize these newly learned 3D representations to other tasks, i.e., the model exhibits only marginal improvements or a slight degradation in performance on other benchmarks. To address this, we design a series of auxiliary tasks. The first auxiliary task involves 3D grounding data, designed to teach the model to interpret the spatial meaning of a given 3D bounding box. Although this data does not yield an overall performance boost, we retain it in our training set to maintain

the diversity of the overall data and tasks. The second task utilizes scene captioning data, which primarily describes the spatial relationships between objects within an image. This addition resulted in a 2.3% overall improvement on spatial understanding benchmarks, with specific gains of 11% on CVBench-3D [60], 1.7% on 3DSRBench [41], and 2.1% on VSIBench [72]. Third, we introduce measurement data to inject priors for an object’s length, width, and height, which obtains a 0.8% improvement on 3DSRBench [41]. Surprisingly, this measurement priors also generalized to video tasks, yielding a 2.4% improvement on VSIBench [72]. The final auxiliary task incorporated depth data. Explicitly training on depth data led to significant improvements, including a 10.4% gain on CVBench-3D [60], a 1.2% gain on BLINK [23], and a 3.2% gain on VSIBench [72]. However, the results show that incorporating single-image data for spatial understanding fails to bring significant improvements on the MMSI-Bench [75]. This may be because the MMSI-Bench requires models to perform more advanced spatial reasoning.

Single-image Multi-image Video

Data S-AVG

MM-AVG CV-2D CV-3D 3DSR MMSI BLINK VSI

Baseline 49.9 71.2 72.6 50.5 26.1 49.2 29.6 68.3 + 3D Object Detection 50.9 71.5 78.3 51.0 25.5 48.2 30.9 70.0 + 3D Grounding 50.5 73.3 72.3 50.5 27.7 48.4 31.0 69.5 + Scene Caption (si) 52.8 72.4 83.8 52.2 25.9 49.4 33.1 69.8 + Measurement (si) 53.3 71.9 83.0 53.0 26.7 49.4 35.5 69.2 + Depth and Distance Data 56.4 73.5 93.4 53.2 28.8 50.6 38.7 69.7

Table 5 Ablation for the single-image data. si denotes single-image data.

Ablation for the multi-image data. Building on the model’s ability to understand spatial relationships in single images, we aim to extend its capabilities to comprehend correspondences, object relationships, and camera relationships in multi-image scenarios. Therefore, we use Qwen2.5-VL-3B, fine-tuned on single-image data, as our baseline model and further enhance it by incorporating multi-image data. The results are shown in Table 6. First, we introduce multi-view correspondence data, which leads to a 1.8% improvement on the BLINK [23]. Second, we incorporate multi-image object detection data, which requires the model to identify corresponding objects across multiple views and represent them using 3D bounding boxes. This addition yields a 2.1% improvement on MMSI-Bench [75], demonstrating that reconstructing layouts from limited views enhances spatial reasoning. Third, we include rule-based data that captures relationships between objects and cameras. This data yields gains of 1.0% on BLINK [23], 0.8% on MMSI-Bench [75], and 1.1% on VSI-Bench [72]. Similar to single-image scene captioning, we aim for the model to understand scenes depicted across multiple images. Thus, we create multi-image scene caption data, which contributes an additional 0.3% gain on MMSI-Bench [75]. Next, we add data related to camera motion, resulting in a 2% improvement on BLINK. Finally, we incorporate general multi-image understanding data to maintain the overall capabilities of the model. Overall, incorporating multi-image data results in a 1.3% improvement in the average score for spatial understanding, but it does not lead to any progress on single-image benchmarks.

Single-image Multi-image Video

Data S-AVG

MM-AVG CV-2D CV-3D 3DSR MMSI BLINK VSI

Baseline 56.4 73.5 93.4 53.2 28.8 50.6 38.7 69.7 + Corespondence 56.7 74.1 92.5 53.9 27.9 52.4 39.1 69.7 + 3D Object Detection (mi) 56.5 73.3 92.3 53.1 30.0 52.7 37.7 69.2 + Object-object Relation 57.1 74.3 91.9 53.2 31.9 53.2 38.3 69.4 + Camera-camera Relation 57.3 72.7 93.1 53.6 31.8 53.7 38.8 68.8 + Scene Caption (mi) 57.4 73.2 92.8 54.0 32.1 53.0 39.3 69.2 + Camera Motion 57.7 73.9 92.5 54.0 32.4 55.0 38.2 68.7 + General Data (mi) 57.7 73.8 92.4 53.7 32.7 55.4 38.4 69.2

Table 6 Ablation for the multi-image data. mi denotes mingle-image data.

Ablation for the video data. Since single-image and multi-image data provide minimal improvement on video benchmarks, we further constructed video data to enhance spatial understanding for video inputs. We use

Qwen2.5-VL-3B, a model fine-tuned on both single-image and multi-image data, as our baseline model. As demonstrated in Table 7, when incorporating a portion of general video data from LLaVA-OneVision [33], the VSI-Bench score decreases by 0.3%, indicating the limitations of general video data in video spatial understanding tasks. When our VST-video data is introduced, the model achieves a 16.6% improvement on VSI-Bench [72]. Moreover, it is able to maintain performance on other benchmarks.

Single-image Multi-image Video

Data S-AVG

MM-AVG CV-2D CV-3D 3DSR MMSI BLINK VSI

Baseline 54.8 73.8 92.4 53.7 32.7 55.4 38.4 69.2 + General Video 57.9 74.3 92.3 53.5 31.9 57.6 38.1 69.4 + VST-video 60.6 75.1 93.1 54.0 31.3 55.6 54.7 69.4

Table 7 Ablation for the video data.

Scaling up model size yields consistent gains in spatial benchmarks, whereas for 3D object detection, performance improvements saturate at the 7B model. We further investigate the relationship between model size and spatial understanding performance, as shown in Table 8. Increasing the model size from 3B to 7B results in a 1.3% improvement in average scores on spatial understanding benchmarks. Further increasing the model size to 32B yields a 1.7% improvement in average scores on these benchmarks. These results suggest that larger models achieve greater improvements on spatial understanding tasks.

The relationship between model size and 3D object detection is presented in Table 9. When increasing the model size from 3B to 7B, we observe a 4.2 AP improvement on SUN RGB-D [54] and a 3.5 AP improvement on ARKitScenes [4]. However, when increasing the model size from 7B to 32B, the performance does not exhibit a positive correlation as seen in the spatial understanding benchmarks. This may be because a model with 7B parameters is already sufficient to handle this fundamental perception task.

Data scaling boosts performance for all models, yet larger models gain more on spatial benchmarks while smaller models benefit most on 3D object detection. When the dataset is increased three times, all models show improvement. As shown in Table 8, the tripling of the data scale resulted in improvements of 1.1%, 1.5%, and 1.7% for the 3B, 7B, and 32B models, respectively, on spatial benchmarks. As recorded in Table 9, data scaling also enhances the performance of 3D object detection tasks.

70

Single-image Multi-image Video

Model Size

Data Scale

1× 3×

S-AVG

MM-AVG CV-2D CV-3D 3DSR MMSI BLINK VSI

| |
|---|

+1.7

65

###### S-AVG(%)

1× 60.6 75.1 93.1 54.0 31.3 55.6 54.7 69.4 3× 61.7 75.0 93.8 54.1 30.2 59.1 57.9 69.9

+1.5

3B

+1.1

1× 61.9 76.5 94.4 53.5 31.9 58.3 57.0 73.1 3× 63.4 75.2 95.7 54.6 32.0 62.1 60.6 73.3

60

7B

1× 63.6 77.5 93.1 55.7 36.9 61.4 57.1 75.6

32B

3× 65.3 80.7 94.8 56.1 36.0 65.4 58.7 75.9 55 3B 7B 32B

Table 8 Results of scaling model and data size

A clearer reasoning trace from ’prompting with BEV’ can improve spatial reasoning performance. We conduct an ablation study on CoT data for spatial reasoning, as recorded in Table 10. Our baseline model is Qwen2.5VL-7B [71], fine-tuned on one-third of VST-P single-image data, VST-P multi-view correspondence data, VST-P multi-image 3D object-detection data, and 800K general samples. Our first attempt is to represent the layout of multi-image scenes with 3D bounding boxes and infer spatial relations through numerical computation, denoted as numerical CoT (‘Num’). This method achieves a score of 29.2 on MMSI-Bench, outperforming the baseline by 2.8%. It performs well on the attribute subset because 3D object detection aligns objects in a shared 3D coordinate, avoiding parallax effects in the pixel space (e.g., distant objects occupy fewer pixels while nearby objects occupy more). However, estimating camera poses across multiple

55

SUN RGB-D ARKitScenes

Model Size

Data Scale

SUN 1× SUN 3×

ARK 1× ARK 3×

| |
|---|

50

| |
|---|

| |
|---|

AP AP@15 AP@25 AP@50 AR@100 AP AP@15 AP@25 AP@50 AR@100

45

+6.5 +4.1

40

1× 20.2 30.3 20.6 4.5 33.5 31.5 45.1 34.8 8.3 46.6 3× 26.5 37.3 28.9 7.1 39.7 38.0 51.7 41.5 14.3 53.4

3B

+2.5

35

###### AP

+3.8

30

+6.3

1× 24.2 37.4 24.6 5.4 37.7 35.0 48.8 38.0 11.3 50.1 3× 28.0 41.6 29.5 7.7 42.1 39.1 52.8 42.3 15.3 54.3

+2.9

25

7B

20

1× 19.6 29.5 19.5 4.3 32.7 31.1 44.3 33.7 8.9 46.3

15

32B

3× 22.5 33.2 23.3 5.1 36.1 33.6 47.6 35.8 11.3 49.1 10 3B 7B 32B

Table 9 3D object detection results of scaling model and data size

images can be difficult, especially when viewpoints vary greatly, which can reduce the accuracy of 3D object detection. Moreover, human beings do not reason about space through direct numerical calculation. Inspired by this, we propose reconstructing the scene using text within the CoT (RT-CoT). Compared with Num-CoT, RT-CoT handles object, camera, and scene relations much better; for example, on the camera–object (CO) subset of MMSI-Bench, it delivers a 3.5% improvement over Num-CoT. Moreover, because current large models are weak at spatial reasoning, the CoT may misidentify object relations when constructing training data. To address this, we condition CoT generation on a BEV image to improve its accuracy. This yields a further 1.1% gain. We further add data about camera–camera relationships and camera motion to increase data diversity, which pushes the score to 31.7%.

Positional Relationship Attribute Motion MSR CC OO RR CO OR CR M A C O -

CoT Type Data Overall

- - 26.4 22.6 28.7 17.3 39.5 38.8 28.9 20.3 18.2 21.6 28.9 25.3 Num-CoT OO 29.2 28.0 33.0 28.4 40.7 25.9 36.1 39.1 36.4 20.3 31.6 18.7 RT-CoT OO 30.0 35.5 31.9 25.9 44.2 36.5 32.5 32.8 30.3 21.6 27.6 21.2 RT-CoTBEV OO 31.1 31.2 36.2 27.2 46.5 40.0 36.1 26.6 30.3 17.6 30.3 24.8 RT-CoTBEV Mix 31.7 35.5 39.4 37.0 44.2 36.5 43.4 28.1 22.7 17.6 30.3 21.7

Table 10 Cold-start results for spatial reasoning on the MMSI-Bench [75]. CoT Types: ‘Num.’ denotes numerical calculation; ‘RT’ involves layout reconstruction using text from visual input; ‘RTBEV’ denotes the generated CoT is grounded on the BEV image. Data: OO refers to the object-object subset, while Mix includes all data types.

RL facilitates the development of spatial reasoning ability. However, we found that after CoT cold start, the model performs better without CoT inference than with it. As shown in Table 11, the model achieves 33.6% accuracy on MMSI-Bench [75] without CoT inference, surpassing the CoT setting by 1.9%. This suggests the model learns spatial knowledge from the CoT process, but its CoT reasoning ability is weak, leading to poorer results. Therefore, to reinforce the spatial reasoning ability, we apply online RL [51] as detailed in Section 3.1. After RL training, the model reaches 35.3% on MMSI-Bench [75] when using CoT inference.

Positional relationship Attribute Motion MSR CC OO RR CO OR CR M A C O -

Training Stage

CoT Inference

Overall

Cold start ✗ 33.6 43.0 35.1 33.3 43.0 36.5 44.6 32.8 27.3 28.4 32.9 23.2 Cold start ✓ 31.7 35.5 39.4 37.0 44.2 36.5 43.4 28.1 22.7 17.6 30.3 21.7 RL ✗ 34.7 34.4 38.3 29.6 53.5 37.6 43.4 25.0 36.4 27.0 40.8 25.3 RL ✓ 35.3 35.5 38.3 30.9 54.6 35.3 47.0 26.6 48.5 33.8 31.6 22.7

Table 11 RL results for spatial reasoning on the MMSI-Bench [75].

Ablation for the 3D object detection. We conduct ablation studies on 3D object detection settings in Table 12. The baseline model is trained on a mix of 600K 3D detection samples and 800K general samples. First, lifting

- 2D to 3D needs camera intrinsics or FoV, and different devices have different intrinsics, which forces the VLM

to fit many camera models. To address this, we unify the FoV across datasets with different FoVs. With a unified FoV, the baseline drops by 0.1 AP on SUN RGB-D but improves by 2.5 AP on ARKitScenes. Second, we study angle representations for 3D bounding boxes. When we replace Euler angles with quaternions, the baseline performs worse on SUN RGB-D and ARKitScenes. We propose mixing single-turn and multi-turn data in the 3D detection corpus, as we believe multi-turn data helps the VLM build layouts from context and learn spatial information better. When we replace all multi-turn data with the same amount of single-turn data, the model drops by 1.7 AP and 1.8 AP on SUN RGB-D and ARKitScenes, respectively.

SUN RGB-D ARKitScenes

Settings

AP AP@15 AP@25 AP@50 AR@100 AP AP@15 AP@25 AP@50 AR@100

Baseline 18.5 28.5 19.1 3.2 30.7 29.4 42.7 31.8 7.2 44.6 w/o FoV Unification 18.6 29.1 19.2 2.2 30.4 26.9 40.9 28.2 5.4 42.0 Euler Angle –> Quaternions 18.3 29.4 17.8 2.8 30.6 28.6 42.4 30.8 6.3 43.9 Singe-turn Data 16.8 26.0 17.0 2.7 28.4 27.6 41.9 29.0 6.0 42.5

Table 12 Ablation study of different settings for 3D object detection. All models are trained on a mixture of 600K 3D object detection samples and 800K general data samples.

- 3D IoU combined with F1 score serves as an effective accuracy reward for the 3D object detection task. We conduct ablation studies on the accuracy reward designed for 3D object detection, as recorded in Table 13. First, we use 3D IoU and recall as the accuracy reward. After RL training, performance drops markedly because each ground-truth box is matched with too many false-positive predictions. Therefore, we switch to IoU and F1 score as the accuracy reward, which yields a 4.2 AP improvement.

Accuracy Reward AP AP@15 AP@25 AP@50 AR@100

Baseline 20.2 30.3 20.6 4.5 33.5 3D IoU + Recall 13.8 20.9 14.7 2.6 33.6 3D IoU + F1 Score 24.4 36.0 26.3 5.8 37.2

Table 13 Ablation for the accuracy reward on 3D object detection task on SUN RGB-D [54]

#### 4.4 Expanding to VLA Model

VLA Backbone Steps Spatial Object Goal 10 AVG

|Qwen2.5-VL-3B [3] VST-tuned Qwen2.5-VL-3B (ours)<br><br>|2.5K 2.5K<br><br>|56.6 86.6 53.8 15.2 65.0 (+8.4) 88.4 (+1.8) 67.8 (+4.0) 25.6 (+10.4)<br><br>|53.1 61.7 (+8.6)<br><br>|
|---|---|---|---|
|Qwen2.5-VL-3B [3] VST-tuned Qwen2.5-VL-3B (ours)<br><br>|10K 10K<br><br>|76.0 83.8 72.0 39.6 78.4 (+2.4) 87.8 (+4.0) 76.0 (+4.0) 41.0 (+1.4)<br><br>|67.9 70.8 (+2.9)<br><br>|

Table 14 Success rate comparison on the LIBERO benchmark [35].

As detailed in Section 3.2, we adapt our VST-tuned VLM into a VLA model. In contrast to the approach used by OpenVLA [31], we do not utilize any pre-trained data on robotic learning. Instead, we directly fine-tune the VLM and its action embeddings on the small-scale LIBERO benchmark [35] from scratch for action prediction. The results are presented in Table 14. Notably, the VLA model based on the VST-tuned model, which incorporates spatial knowledge, surpasses the one based on Qwen2.5-VL-3B [3] by an average of 8.6% in success rate. Even with a fourfold increase in training steps, the VST-tuned model still demonstrated an improvement of 2.9%. This improvement clearly demonstrates that the integration of spatial knowledge helps robotic learning.

### 5 Related Work

Large Vision-Language Models. Recently, Large Vision–Language Models (LVLMs) have become a pivotal technology in artificial intelligence, able to understand and integrate information across multiple modalities such

as text, images, and video. Most LVLMs use the same efficient architecture [3, 13, 20, 33, 37, 62]. Specifically, a pre-trained vision encoder is employed to extract visual embeddings, which are subsequently projected into the language space via a projector composed of multilayer perceptron (MLP) layers. Various strategies have been implemented to enhance the performance of VLMs, including the utilization of more advanced vision encoders [14, 80], increasing input resolution [13, 36], adopting dynamic resolution techniques [62], refining multimodal positional embeddings [3, 71], and synthesizing high-quality training datasets [16, 33]. Additionally, some works [5, 74] aim to develop LVLMs as unified models that consolidate vision and language capabilities. Moreover, some studies [29, 45, 70] explore the use of reinforcement learning in the post-training stage to enhance the visual reasoning capabilities of LVLMs, an approach inspired mainly by DeepSeek-R1 [26]. The majority of these works apply Generalized Reinforcement Learning from Preference Optimization (GRPO) [51] to train LVLMs, achieving significant improvements in many tasks.

Large Vision-Language Models for Spatial Understanding and Reasoning. Despite the remarkable progress of current LVLMs in visual tasks [38, 39, 44, 67, 79], numerous benchmarks [41, 72, 75] have highlighted persistent challenges in spatial understanding and reasoning. To address these issues, SpatialVLM [11] pioneered the application of VLMs to spatial understanding by constructing VQA datasets using expert models. Similarly, SpatialRGPT [15] expanded RGB-based spatial understanding to the RGB-D domain by generating spatial datasets from 3D scene graphs. Recognizing the prohibitive cost of collecting and annotating real-world data, SAT [49] employed simulators to generate training data, thereby extending its focus from static to dynamic tasks. SpatialBot [9] enables VLMs to invoke external tools for depth estimation, thereby improving their ability to interpret spatial information in input images. Subsequent studies [19, 21, 68, 82] have further advanced the field by constructing more comprehensive datasets to enhance the spatial understanding capabilities of VLMs. In parallel, another line of research focuses on enhancing the spatial reasoning abilities of VLMs. For example, MVoT [65] leverages multimodal representations within its reasoning traces to strengthen spatial reasoning. SpaceR [47] and MindCube [77] incorporate textual cognition maps into their reasoning traces to enhance spatial reasoning, further improving performance through reinforcement learning. Similarly, Spatialreasoner [42] performs spatial reasoning by predicting 3D locations and poses as intermediate results. VILASR [64] enhances spatial reasoning by incorporating visual tools and introducing visual prompting into the reasoning process. In contrast to these prior studies, which typically focus exclusively on either spatial understanding or spatial reasoning, our approach begins with foundational capabilities and builds upon them to enhance the model’s overall reasoning skills.

### 6 Conclusion

We present Visual Spatial Tuning (VST), a general and scalable framework that endows vision-language models with human-like spatial perception and reasoning abilities. With the large-scale perception data (VST-P) and curated reasoning instructions (VST-R), VST effectively acquires spatial awareness without degrading general capabilities. The proposed approach achieves state-of-the-art performance on multiple spatial benchmarks, demonstrating that spatial abilities in foundation models can be systematically scaled. Moreover, the Vision-Language-Action (VLA) models have been proven to enhance visuospatial skills, enabling more grounded interaction with the physical world. The generality, scalability, and effectiveness of VST highlight a promising direction toward building physical AI systems that reason and act in space with human-like intelligence.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv:2303.08774, 2023.

- [2] Adel Ahmadyan, Liangkai Zhang, Artsiom Ablavatski, Jianing Wei, and Matthias Grundmann. Objectron: A large scale dataset of object-centric videos in the wild with pose annotations. In CVPR, 2021.

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv:2502.13923, 2025.

- [4] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv:2111.08897, 2021.

- [5] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sağnak Taşırlar. Introducing our multimodal models, 2023. URL https://www.adept.ai/blog/fuyu-8b.
- [6] Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G Shapiro, and Ranjay Krishna. Perception tokens enhance visual reasoning in multimodal language models. In CVPR, 2025.

- [7] Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. Omni3d: A large benchmark and model for 3d object detection in the wild. In CVPR, 2023.

- [8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv:2212.06817, 2022.

- [9] Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. arXiv:2406.13642, 2024.

- [10] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv:1709.06158, 2017.

- [11] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024.

- [12] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv:2403.20330, 2024.

- [13] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv:2412.05271, 2024.

- [14] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024.

- [15] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. NeurIPS, 2024.

- [16] Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, et al. Perceptionlm: Open-access data and models for detailed visual understanding. arXiv:2504.13180, 2025.

- [17] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv:2507.06261, 2025.

- [18] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.

- [19] Erik Daxberger, Nina Wenzel, David Griffiths, Haiming Gang, Justin Lazarow, Gefen Kohavi, Kai Kang, Marcin Eichner, Yinfei Yang, Afshin Dehghan, et al. Mm-spatial: Exploring 3d spatial understanding in multimodal llms. arXiv:2503.13111, 2025.

- [20] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In CVPR, 2025.

- [21] Nianchen Deng, Lixin Gu, Shenglong Ye, Yinan He, Zhe Chen, Songze Li, Haomin Wang, Xingguang Wei, Tianshuo Yang, Min Dou, et al. Internspatial: A comprehensive dataset for spatial reasoning in vision-language models. arXiv:2506.18385, 2025.

- [22] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv:2505.20279, 2025.

- [23] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In ECCV, 2024.

- [24] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.

- [25] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, 2022.

- [26] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv:2501.12948, 2025.

- [27] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv:2505.07062, 2025.

- [28] Mary Hegarty, Daniel R Montello, Anthony E Richardson, Toru Ishikawa, and Kristin Lovelace. Spatial abilities at different scales: Individual differences in aptitude-test performance and spatial-layout learning. Intelligence, 34

(2):151–176, 2006.

- [29] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv:2503.06749, 2025.

- [30] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.

- [31] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv:2406.09246, 2024.

- [32] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2): 83–97, 1955.

- [33] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv:2408.03326, 2024.

- [34] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.

- [35] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. NeurIPS, 2023.

- [36] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.

- [37] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024.

- [38] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024.

- [39] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024.

- [40] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

- [41] Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Celso M de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. arXiv:2412.07825, 2024.

- [42] Wufei Ma, Yu-Cheng Chou, Qihao Liu, Xingrui Wang, Celso de Melo, Jianwen Xie, and Alan Yuille. Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. arXiv:2504.20024, 2025.

- [43] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv:2203.10244, 2022.

- [44] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021.

- [45] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. CoRR, 2025.

- [46] Yinyu Nie, Xiaoguang Han, Shihui Guo, Yujian Zheng, Jian Chang, and Jian Jun Zhang. Total3dunderstanding: Joint layout, object pose and mesh reconstruction for indoor scenes from a single image. In CVPR, 2020.

- [47] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv:2504.01805, 2025.

- [48] Jean Piaget. Child’s Conception of Space: Selected Works vol 4. Routledge, 2013.

- [49] Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, et al. Sat: Dynamic spatial aptitude training for multimodal language models. arXiv:2412.07755, 2024.

- [50] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021.

- [51] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300, 2024.

- [52] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In EuroSys, 2025.

- [53] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019.

- [54] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In CVPR, 2015.

- [55] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv:2405.09818, 2024.

- [56] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv:2312.11805, 2023.

- [57] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv:2403.05530, 2024.

- [58] MiMo-VL Team. Mimo-vl technical report, 2025. URL https://arxiv.org/abs/2506.03569.
- [59] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv:2402.12289, 2024.

- [60] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. NeurIPS, 2024.

- [61] Hanqing Wang, Jiahe Chen, Wensi Huang, Qingwei Ben, Tai Wang, Boyu Mi, Tao Huang, Siheng Zhao, Yilun Chen, Sizhe Yang, et al. Grutopia: Dream general robots in a city at scale. arXiv:2407.10943, 2024.

- [62] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv:2409.12191, 2024.

- [63] Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, et al. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In CVPR, 2024.

- [64] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv:2506.09965, 2025.

- [65] Wenshan Wu, Shaoguang Mao, Yadong Zhang, Yan Xia, Li Dong, Lei Cui, and Furu Wei. Mind’s eye of llms: visualization-of-thought elicits spatial reasoning in large language models. NeurIPS, 2024.

- [66] x.ai. Grok-1.5 vision preview, 2024. URL https://x.ai/blog/grok-1.5v.
- [67] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. NeurIPS, 2024.

- [68] Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, and Kevin J Liang. Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models. arXiv:2505.17015, 2025.

- [69] Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, and Kevin J Liang. Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models. arXiv:2505.17015, 2025.

- [70] Shilin Xu, Yanwei Li, Rui Yang, Tao Zhang, Yueyi Sun, Wei Chow, Linfeng Li, Hang Song, Qi Xu, Yunhai Tong, et al. Mixed-r1: Unified reward perspective for reasoning capability in multimodal large language models. arXiv:2505.24164, 2025.

- [71] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv:2407.10671, 2024.

- [72] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, 2025.

- [73] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. NeurIPS, 2024.

- [74] Rui Yang, Lin Song, Yicheng Xiao, Runhui Huang, Yixiao Ge, Ying Shan, and Hengshuang Zhao. Haplovl: A single-transformer baseline for multi-modal understanding. arXiv:2503.14694, 2025.

- [75] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv:2505.23764, 2025.

- [76] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In ICCV, 2023.

- [77] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. arXiv:2506.21458, 2025.

- [78] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv:2503.14476, 2025.

- [79] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.

- [80] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.

- [81] Cheng Zhang, Zhaopeng Cui, Yinda Zhang, Bing Zeng, Marc Pollefeys, and Shuaicheng Liu. Holistic 3d scene understanding from a single image with implicit representation. In CVPR, 2021.

- [82] Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv:2503.22976, 2025.

- [83] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. Transactions on Machine Learning Research, 2025.

- [84] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv:2504.10479, 2025.

- [85] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, and Qing Li. 3d-vista: Pre-trained transformer for 3d vision and text alignment. In ICCV, 2023.

- [86] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In CoRL, 2023.

## Appendix

### A More Implementation Details

#### A.1 Coordinate

Camera coordinate. We define the camera coordinate system based on the right-hand rule. The camera center is taken as the origin, with the X-axis pointing to the right (parallel to the image plane), the Y-axis pointing downward (also parallel to the image plane), and the Z-axis pointing forward along the optical axis.

Bounding box coordinate. Within the defined camera system, a 3D bounding box is specified by its center coordinates (x,y,z), its size (xl,yl,zl), and its orientation (pitch,yaw,roll). In our convention, the Xdimension corresponds to the front–back size of the box, Y represents the vertical extent, and Z denotes the lateral (side) extent. In our definition of a 3D bounding box, the X-axis points to the front, the Y-axis points downward, and the Z-axis points sideways. The rotation angles are defined as the transformations from the camera axes to the box axes, measured in degrees and normalized by π (i.e., divided by 180◦). The center and dimension are given in meters. All values are rounded to two decimal places.

#### A.2 Instruction Format The CoT Format

<|im_start|>system

You are a helpful assistant. You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here.<|im_end|>

<|im_start|>user <|vision_start|>image.jpg<|vision_end|>{question}<|im_end|> <|im_start|>assistant <think>{thinking content}</think> {answer}<|im_end|>

#### A.3 FoV Unification

We show the process of FoV unification in Algorithm 1. We resize the input image according to a pre-defined focal length fnew. This approach ensures that all images fed into the VLM for the 3D object detection task share a uniform focal length, thereby mitigating the potential challenges arising from disparate camera parameters when training on mixed datasets.

Algorithm 1 FoV Unification

- 1 # I: Input image with size (W, H)
- 2 # f: Focal length of the original camera
- 3 # f_new: Unified focal length
- 4 # hfov and wfov are the horizontal and vertical fields of view.
- 5 hfov = 2 * atan(W / (2 * f))
- 6 wfov = 2 * atan(H / (2 * f))
- 7 W_new = 2 * f_new * tan(hfov / 2)
- 8 H_new = 2 * f_new * tan(wfov / 2)
- 9 # Resize the original image
- 10 I_new = resize(I, (W_new , H_new))

#### A.4 Settings

VLM. We summarize the training details in Table 15, 16, and 17.

VLA. We select four task suites on the LIBERO benchmark [35]: LIBERO-Spatial, LIBERO-Object, LIBEROGoal, and LIBERO-Long (LIBERO-10). In line with the methodology of OpenVLA [31], we preprocess

Model Size Optimizer LR ViT LR Packing length Global batch size Warmup ratio Epoch

3B AdamW 5.00e-5 5.00e-6 16384 128 0.03 1 7B AdamW 5.00e-5 5.00e-6 16384 128 0.03 1 32B AdamW 5.00e-5 5.00e-6 16384 128 0.03 1

###### Table 15 Summary of training parameters used in the SFT stage.

Model Size Optimizer LR ViT LR Packing length Global batch size Warmup ratio Epoch

3B AdamW 1.00e-5 1.00e-6 16384 128 0.03 2 7B AdamW 1.00e-5 1.00e-6 16384 128 0.03 2 32B AdamW 1.00e-5 1.00e-6 16384 128 0.03 2

###### Table 16 Summary of training parameters used in the CoT cold start stage.

Model Size Optimizer LR ViT LR Max length Global batch size Rollout number KL Clip_low Clip_high

3B AdamW 1.00e-6 1.00e-6 8192 128 5 disable 0.2 0.28 7B AdamW 1.00e-6 1.00e-6 8192 128 5 disable 0.2 0.28 32B AdamW 1.00e-6 1.00e-6 8192 128 5 disable 0.2 0.28

Table 17 Summary of training parameters used in the RL stage.

the dataset by filtering out "no-operation" actions and unsuccessful demonstrations to accommodate the single-step policy. The visual input is a third-person camera image, resized to a resolution of 256 × 256 pixels. To evaluate performance, we execute 50 independent rollouts for each task and compute the average success rate.

Simulator. We use Isaac Sim-4.2 to generate the 3D bounding box of a scene.

### B More Results

Data scaling of spatial foundational tasks. To enable the VLM to perceive the positions of objects in 3D space, we selected monocular 3D object detection and depth estimation as our foundational tasks. We then incrementally scaled the volume of training data to validate the VLM’s emerging spatial perception capabilities. We use Qwen2.5-VL-3B [3] as a baseline model. As shown in Figure 5a, the AP@15 on the SUN RGB-D [54] and ARKitScenes [4] datasets progressively improved as the amount of 3D detection data increased, demonstrating its ability to learn how to perceive the 3D spatial positions of objects from visual input. Furthermore, by gradually introducing depth-related data, we discovered that the VLM could learn to judge the relative distances between objects and the camera, even with a comparatively small amount of data, as illustrated in Figure 5b.

### C More Data Engine Details

#### C.1 Prompting with BEV Annotation

In the process of generating spatial reasoning data for object relationships, we augment the object information extracted from multi-view images with a visualized BEV image as additional contextual input, as demonstrated in Table 18. The corresponding prompt is detailed in Table 19. Given the current limitations of large VLMs in spatial perception, incorporating a BEV image as an auxiliary input enhances the model to comprehend spatial relationships among objects within RGB images. This integration facilitates more accurate caption generation and enables more effective text-based reasoning.

AP@15-SUN AP@15-ARK S-AVG MM-AVG

70

60

50

Performance

40

30

20

10

0

0K 200K 400K 600K 800K 1600K

Number of 3D Detection Data

CV-3D BLINK-D S-AVG MM-AVG

95

90

85

Performance

80

75

70

65

60

55

0K 200K 400K 600K 800K 1600K

Number of Depth Data

(a) Data scaling of 3D detection data.

(b) Data scaling of depth-related data. Figure 5 Data scaling of spatial foundational tasks.

#### C.2 Dataset Visualization

Depth comparison. The depth-related data are visualized in Table 20 and Table 21. We have devised multiple representation formats to refer to objects, including visual box, visual point, textual box, textual point, and text. For data involving points, we utilize the mask center to ensure that the point is located on the object. Additionally, we incorporate samples to estimate the relative depth at randomly selected points.

Distance Comparison. The distance-related data are presented in Table 22, where objects are referenced using visual boxes or points.

###### 3D Object Detection. The 3D object detection data are presented in Table 23. We employ 9-DoF (Degrees of Freedom) 3D bounding boxes, with further details provided in Section A.1. All images are resized using the method described in Algorithm 1 to ensure a unified field of view (FoV).

Measurement. Measurement data includes information related to object height and size, as presented in Table 24. Due to ambiguities in specifying the width and length of objects, we only annotate the height. Otherwise, we assign the maximum value among width, height, and length. Additionally, the data is organized in a multi-turn format, as we aim for the model to learn the relative relationships between objects from multi-turn data.

Scene caption. Scene caption data for single-view and multi-view scenarios are presented in Table 25. The captions are generated based on the prompt outlined in Table 26. In the generation process, we incorporate detailed object information—including category, spatial center, and size—as well as object relationships derived from the scene graph, to enrich the contextual understanding.

Correspondence. The correspondence data are presented in Table 27. Given a reference visual point, the model is tasked with identifying the corresponding visual point in the second image. To obtain the correspondence points, we begin with the ground truth instance masks of a point cloud. For each object instance, we randomly sample a set of N points from its surface. Each sampled 3D point is then projected into the 2D image plane to determine its visibility. A point is considered a valid correspondence and thus “visible” only if it satisfies all of the following criteria:

- 1. Image Boundary Check: The projected 2D coordinates (u,v) must lie within the image boundaries, i.e., 0 ≤ u < W and 0 ≤ v < H, where W and H are the width and height of the image, respectively. Points projecting outside these bounds are discarded.
- 2. Depth Validity Check: The point’s depth value in the camera’s coordinate system must be positive (z > 0), ensuring it is in front of the camera plane.
- 3. Occlusion Check: The point must not be occluded by other geometry. We verify this by comparing the point’s ground true depth, zgt, with the depth value zdepth recorded in the depth map at its projected

###### Prompting with BEV annotation

[Figure 50]

[Figure 51]

[Figure 52]

Prompt <system prompt> <ground-truth> <question>

If the small metal trash can on the floor near the white storage cabinets in the first image is north of the small printer on the white cabinet near the door in the second image, what direction is the clear plastic bottle on the white desk with two monitors in the second image from the printer? Options: A. southeast, B. northeast, C. south, D. northwest

###### Teacher model

First, I look at the first image. The scene has a long row of white storage cabinets along the left wall, a white desk with shelves holding folders on the right, and a small metal trash can sitting on the floor between the cabinets and the desk. There’s also a coat hanger in the corner and some boxes stacked nearby. The trash can is clearly positioned on the left side of the room, close to the storage cabinets. Next, I examine the second image, which was taken from a closer, more forward angle-like the camera moved toward the desk and tilted up slightly. The desk is now in the foreground, holding two black monitors, a keyboard, a mouse, a telephone, and a clear plastic bottle. Against the back wall near the door, there’s a small printer on a white cabinet. The door and coat hanger from the first image are still visible in the background, but the trash can isn’t present here. Putting both images together, the room’s layout makes sense: the left wall has storage cabinets, the center/right has a desk, and the back wall has a door. The trash can (from image 1) is on the left side near the cabinets, the printer (from image 2) is on the back right near the door, and the bottle (from image 2) is on the foreground desk. The question says the trash can is north of the printer. Since the trash can is on the left side of the room and the printer is on the back right, "north" from the printer’s perspective must be toward the left-toward the trash can. Now, I need to figure out where the bottle is relative to the printer. The bottle is on the desk, which is in the foreground of the second image. From the printer’s position (back right near the door), the desk is in front of it-closer to the camera. The bottle is on the part of the desk that’s to the left of the printer (since the desk extends from the foreground to the back wall, and the printer is on a cabinet against the back wall to the right of the desk). So, from the printer’s view: the bottle is to the left (the same direction as north, since north is toward the trash can on the left) and forward (since the desk is closer than the trash can). Combining these, the bottle is in the northwest direction from the printer. That matches option D.

Table 18 An example to illustrate the prompting with BEV annotation.

2D location. The point is deemed occluded (and thus invisible) if the relative depth difference exceeds a threshold of 5%: |zgt−zzdepth|

> 0.05. Points that pass this check are considered visible.

gt

Only the points that successfully pass all three checks are retained as the final set of valid correspondence points.

3D Object Detection-MI. We adopt the same 3D bounding box representation as used in single-image 3D object detection tasks. Note that we designate the camera coordinate system of the first frame as the world coordinate system.

Object-Object Relationship. Object-related data is visualized in Table 28. We select pairs of images that contain at least one common object (object-level correspondence), as well as objects that are unique to a single image. Anchor objects are defined as those present exclusively in one image and whose center point height difference is less than one meter. For these anchor objects, we formulate questions pertaining to spatial direction and distance. Direction-related queries necessitate a reference direction: given objects A, B, and C, the vector from object A to object B serves as the reference direction, and we examine the spatial relationship between object A and object C accordingly. Alternatively, the camera position may substitute for object A as

You are an expert in 3D annotation. Given a set of images, ground-truth camera poses, ground-truth objects, and a specific question-answer pair, your task is to describe the given objects, rewrite the question and generate a chain-of-thought for the answer.

# Basic Information Coordinate System:

- - Origin: The first image’s camera viewpoint at (0,0,0)
- - Axes:

→ X: Rightward (horizontal) ↓ Y: Downward (vertical, perpendicular to the ground) ↗ Z: Forward

# Objects: Each object is annotated with a class name, 3D position (x_center, y_center, z_center), size (x_size, y_size, z_size), and in_frames flag.

- * x_center, y_center, z_center: Center position of the object in the frame1 coordinate system, in meters. If x_center is positive, it means the object is to the right of the camera in frame1. If y_center is positive, it means the object is below the camera in frame1. If z_center is positive, it means the object is in front of the camera in frame1.
- * x_size, y_size, z_size: The dimensions of the object along the (XY Z) axes, in meters, when the rotation angles are zero. y_size is the height.
- * in_frames: A list indicating whether the object is present in each frame. The length of this list should match the number of frames. For instance, if an object is present in frame1 and frame2, the in_frames list will be [True, True]. If it is only present in frame1, the list will be [True, False].

# Input and Output format The input is a set of images, ground-truth camera pose and 3D bounding boxes, question type, question and answer. The output should be a triplet in json format, containing a revised question, a chain-of-thought process, and an answer: {

’question’: revised_question, ’thought’: generated_thought, ’answer’: given_answer

} # Requirement When revising the question, you must describe each object using a concise caption that reflects its spatial position, material, shape, relationship with the environment, and other relevant features. Use these short captions to refer to objects in the question instead of their ’label’ directly. Your reasoning should be presented in the first person perspective, a detailed, step-by-step manner, forming a logical chain of thought that leads to the answer. This process must incorporate visual content from both image-1 and image-2, describing the entire scene. Structure your response in four parts:

- 1. Describe the visual content of the first image, providing your initial observations of the scene.
- 2. Describe the visual content of the second image, emphasizing correspondences with the first image and noting any new elements or changes. If feasible, also describe the second image’s camera rotation and translation relative to the first image’s camera.
- 3. Summarize the overall layout of the scene as inferred from the two images and the TOP VIEW image. Then, provide a detailed, logical reasoning process based on your observations.
- 4. Summarize your answer.

Always use the TOP VIEW image to ensure your spatial descriptions are consistent with the overall scene layout, but do not explicitly reference the TOP VIEW image in your wording. Ensure all spatial information in your descriptions aligns with what is visualized in the TOP VIEW image. Do NOT refer to objects using ’color point’ or similar terms, as these are not used in real-world visualizations. Do NOT include any numerical calculations. Do NOT include any coordinate-related information and their values, such as x`-axis`, `z-axis`. Do NOT use phrases like àccording to ground-truthòr similar expressions. The question must be clear and free of any thought guidance.

# Examples and question type description These are examples for reference. You can freely play with them. {few_shot_examples}

# Input These two images were taken consecutively, recording information about the same scene:

- image-1:<|image_pad|>
- image-2:<|image_pad|> Ground-truth camera poses are: {camera_info} Ground-truth objects are: {object_info} The TOP VIEW of the scene described by the two images:<|image_pad|> Question type is camera-object orientation: {question}

The calculation process is for your reference (You don’t have to use it): {text_orientation_process} Now, generate response following task description, input and output format, and examples. Please strictly follow the json format. Use different phrasings or styles to ask the questions while maintaining consistency with the intent and structure of the examples.

Table 19 Prompt used in the prompting with bev annotation. {camera_info} is the camera motion information, {object_info} will be replaced by object information (category, center, and size), and {text_orientation_process}

is the textual calculation process.

the reference point.

Camera-Camera Relationship. As shown in Table 29, similar to the object-object data, we select pairs of images that share at least one common object (i.e., object-level correspondence), as well as images containing objects unique to a single frame. The objective is to enable the model to infer the spatial relationship between the camera positions in the two images, based on their respective layouts.

Camera Motion. As shown in Table 30, we divide the camera motion into rotation and translation movements. The rotation category encompasses the following motions: panning to the right, panning to the left, tilting upward, tilting downward, rolling clockwise, and rolling counterclockwise. The translation category includes: rightward, leftward, upward, downward, forward, and backward movements. It is important to note that the precise definitions of rotation and translation will be provided within the specific context.

Video Object-object Relationship. We present video data related to object-object relationships in Table 31, which includes information on object distances and object direction types. Furthermore, we organize the data into a multi-turn format, enabling the model to learn object-relative relationships within the context. This approach also enhances training efficiency, as it mitigates the computational overhead associated with loading video data.

Video Counting Data. We present video counting data in Table 32. Video Spatiotemporal Data. We present video spatiotemporal data in Table 33.

General Reasoning Data. We select data from LLaVA-Onevision [33] that can be verified through predefined rules and categorize it into three types: math, OCR, and knowledge. Each sample is inferred four times using the Qwen2.5-VL [3], and we filter out samples that are either entirely correct or entirely incorrect, retaining only those with partial correctness. Subsequently, the selected samples are inferred using the Seed1.5-VL [27], during which we record the reasoning process. Since this reasoning often contains repetitive patterns (e.g., repeated use of the term "wait"), we further utilize Seed1.5-VL [27] to rewrite and refine the recorded reasoning steps, enhancing their clarity.

Spatial Reasoning data. For spatial reasoning data concerning the relationships between objects and cameras, we employ prompting with BEV annotations (refer to Section C.1) to facilitate the generation of the reasoning process. For data related to camera motion, we utilize the prompts outlined in Table 34 and Table 35, which instruct the model to leverage the parallax effect when reasoning about the types of camera motion. The generated samples are shown in Table 36 and Table 37.

[Figure 53]

Prompt Tell me the depth relationship of the objects of A.sink, B.shelves, C.blinds, D.lamp. Answer The objects from near to far is A, D, B, C

[Figure 54]

Prompt There are several boxes in the image: box-A, box-B. Each box represents an object. Present the object

represented by these boxes in an order that goes from close to far and give their names. Answer box-A (chair), box-B (picture)

[Figure 55]

Prompt You are given several 2D bounding boxes in the image: [3, 132, 98, 468] [322, 83, 475, 481] Arrange the object represented by these boxes from the farthest to the nearest based on their depth relationship and give their names. Output the sorted bboxes and labels using JSON format.

###### Answer

[

{"bbox_2d":[324,81,477,475],"label":"person"}, {"bbox_2d":[3,130,98,462],"label":"person"}

]

[Figure 56]

Prompt After reviewing the image, which object—table (cyan box), window-A (orange box), window-B (pink box)

and door (purple box)—would you say is nearer to the camera? Answer table

###### Table 20 Examples of the depth-related data.

[Figure 57]

Prompt Based on the image, arrange the object represented by these sentences in a list from the closest to the farthest.

1. the chair with the cat laying in it 2. hanging plant sitting on ground

Answer 1. the chair with the cat laying in it, 2. hanging plant sitting on ground

[Figure 58]

Prompt The image contains red point-A and point-B. Please decide which point is closer to the camera. Answer point-A

Table 21 Examples of the depth-related data.

Distance-related Data

[Figure 59]

Prompt Considering the positions, which object—the toilet-A (orange box) or the toilet-B (green box)—do you think

is farthest from the door (purple box)? Answer toilet-A

[Figure 60]

Prompt Estimate the real distances and identify which object—the pillow-A (orange point) or the pillow-B (purple

point)—is closest to the chair (yellow point).

Answer The distance relationships are: Distance[chair, pillow-A]=3.51m Distance[chair, pillow-B]=3.67m So, the

answer is pillow-A.

###### Table 22 Examples of the distance-related data.

###### 3D Object Detection Data

[Figure 61]

Prompt Here are the detailed camera parameters for the image. Camera intrinsic parameters: Horizontal fov, hfov=69.16, and vertical fov, vfov=53.17. Image width=959 and height=696. We do not consider distortion parameters here. Camera coordinate: X-axis points rightward, Y-axis points downward, and Z-axis points forward. The origin point is the camera location. We take the camera coordinate system as the world coordinate system. 3D bounding box format: [x_center, y_center, z_center, x_size, y_size, z_size, pitch, yaw, roll]

- * x_center, y_center, z_center: the center of the object in the camera coordinate, in meters. z_center is the depth of the object in space.
- * x_size, y_size, z_size: The dimensions of the object along the ( XYZ ) axes, in meters, when the rotation angles are zero.
- * pitch, yaw, roll: Euler angles representing rotations around the X, Y, and Z axes, respectively. Each angle is normalized to the range of (-1, 1) and is multiplied by 180 to convert it into degrees. Output a json list where each entry contains the object name in "label" and its 3D bounding box in "bbox_3d". Detect the 3D bounding boxes of printer.

[Figure 62]

###### Answer

[

{"bbox_3d":[-0.16,0.12,1.56,0.44,0.51,0.41,0.11,0.28,0.05],"label":"printer"}, {"bbox_3d":[0.40,-0.02,1.96,0.45,0.51,0.36,0.11,0.27,0.05],"label":"printer"}

]

Prompt Detect the 3D bounding boxes of table.

[Figure 63]

###### Answer

[

{"bbox_3d":[0.19,0.56,1.78,0.53,0.52,1.36,0.1,0.26,0.04],"label":"table"} ]

###### Table 23 Examples of 3D object detection data.

###### Measurement-related Data

[Figure 64]

Prompt Could you provide the height of the table (specified at the pink point-1)? Specify the measurement in

centimeters. Answer the question using a single word or phrase. Answer 71 cm

[Figure 65]

Prompt What is the length of the dimension with the maximum value (length, width, or height) of the chair (signified

by the brown point-A) in meters? Please try to answer the question with short words or phrases if possible. Answer 0.97 m Prompt What is the length of the largest dimension (length, width, or height) of the window (signified by the orange

point-B) in centimeters? Answer 2031 cm

###### Table 24 Examples of the measurement-related data.

###### Scene Caption

[Figure 66]

Prompt Analyze the spatial organization and positional relationship in this image.

Answer Starting from the camera’s viewpoint, the nearest object is a chair (orange point-2) with a depth of 1.72m. Adjacent to it, behind, is another chair (orange point-3) at 2.72m depth. Moving forward, a table (blue point-1) spans 3.00m depth. To the table’s left, chairs (orange points-4,5,6) are aligned at depths 3.20m, 3.63m, and 4.11m respectively. On the table’s right, chairs (orange points-7,8) are aligned, with chair (orange

- point-7) beside chair (orange point-8). Chair (orange point-6) is in front of chair (orange point-8). Distances: chair (orange point-2) is 1.79m from the table; chair (orange point-3) is 1.58m from the table. Chairs (orange points-3,4,5,6) form a left-aligned row. Chairs (orange points-6,7,8) form a right-aligned row. Chair (orange
- point-8) is 0.52m from chair (orange point-7).

[Figure 67]

Prompt The observer’s line of sight is preserved as images are captured one by one. Describe these two frames in

detail.

Answer The first image shows a room scene with a whiteboard on the left wall, a chair in the lower central area, a window on the right side with a view of a building outside, a heater below the window, a box on the windowsill, sockets on the wall near the heater, and a light switch on the wall. Key holders with colorful keys are attached to the whiteboard. The second image, taken with the camera rotated right from the first perspective, shows part of the window and the heater still visible. New elements include a trash can on the floor, a table partially in view, and additional sockets. Objects present in the first image such as the whiteboard, light switch, and the box on the windowsill are no longer in view, while the chair remains present.

###### Table 25 Examples of the scene caption data.

You are a 3D scene understanding expert tasked with generating ego-centric spatial descriptions for the given image. Coordinate System

- - Origin: Camera viewpoint at (0,0,0)
- - Axes:

- X (Rightward, horizontal)
- Y (Downward, vertical)
- Z (Depth, forward)

Input Data Specification For each object:

- - name: "object (color point-x)"
- - size: [X-width, Y-height, Z-length] in meters
- - centroid: (x, y, z) where z is depth from image plane
- - distance_to_camera: distance from object center to camera, in meters Relationships include:

- 1. Binary relationships: [Object-A, Object-B, spatial_relation] (direct positional relationships)
- 2. Multi-object relationships: [[Object-A, Object-B, Object-C...], spatial_relation] (group spatial relationships)
- 3. Distance between two objects in 3D space: [Object-A, Object-B, "distance = X.XX m"] (quantitative supplements)

Scene information Camera: center_x = 0, center_y = 0, center_z (depth) = 0

{object_info}

Output Guidelines

- 1. Depth Handling:

- - Explicitly state depth comparisons using meter values (e.g., "1.2 m closer")
- - Use "closer/farther" for relative depth comparisons

- 2. Spatial Relations:

- - You can refer to "Binary relationships" and "Multi-object relationships"
- - You can also summarize from the original data:

- X-axis: absolute delta x > 0.5 m implies clear left/right
- Y-axis: absolute delta y > 0.3 m implies taller/shorter (Y-axis)
- Z-axis: absolute delta z > 1 m implies clear in front of/behind

- 3. Relationship Coverage:

- - 100 percent coverage of provided binary/multi-object relationships
- - Include more than 70 percent of distance metrics (prioritize distances less than 3 m)

- 4. Conflict Resolution:

- - Prioritize explicit relationship declarations over coordinate calculations
- - Use distance metrics only for quantitative enhancement, not directional correction

- 5. Viewpoint Consistency:

- - All directional descriptions must be strictly from camera perspective
- - Never use absolute coordinates except for Z-axis depth values

- 6. Structural Flow:

- - Camera to nearest object to depth progression (describe the depth value if necessary)
- - Anchor object (table) to surrounding objects
- - Group formations to individual outliers

- 7. Format Requirements (MUST follow):

- - Output 4 to 20 sentences
- - Plain text only (no markdown)
- - Strictly preserve object names (exact "object (color point-x)" format)
- - When making numerical references, specify the name of the numerical relationship, for example, (depth = x m). Unit is required.
- - Prohibited phrases: "according to the coordinates", "as shown in the data", axis terminology, "The multi - object relationships show", "Binary relationships:"
- - Prohibited raw data regurgitation, relationship lists
- - When referring to object quantity, count the objects in the image and text instead of referring to the point index. Now, generate a caption for this scene based on the reference image and the above information, enclosed in <caption> </caption>

###### Table 26 Prompt used in the scene caption generation. {object_info} will be replaced by object information (category, center, and size) and relationships extracted from the scene graph.

###### Correspondence Data

[Figure 68]

Prompt The first image shows a point circled in gold. After adjusting the camera or lighting, the second image presents several gold-circled points labeled ’A, B, C, D’. Which matches the original? Options: A: point-A, B: point-B, C: point-C, D: point-D

Answer B: point-B

Table 27 Examples of the correspondence data.

Object-related Data

[Figure 69]

Prompt If the small white cabinet under the white desk is north of the black monitor on the left side of the desk, what direction is the chair on the right side of the room from the black monitor? Options: A. southeast B. north C. south D. southwest

- Answer A. southeast

[Figure 70]

Prompt If, from the camera position of the first image, the direction toward the hanging jacket (visible in the first image) is north, then in which direction does the window (visible in the second image) lie relative to the first image’s camera? Options: A. northeast, B. southeast, C. east, D. south

- Answer A. northeast

###### Table 28 Examples of the object-related data.

###### Camera-related Data

[Figure 71]

Prompt The frames are acquired in a continuous sequence from a first-person perspective. If the first picture was taken with the camera facing west, what is the direction for the second picture? Options: A. southeast, B. north, C. south, D. northwest

Answer D. northwest

[Figure 72]

Prompt Images are shot one after another from a first-person perspective. When positioned at the second photo spot,

how is the first camera placed relative to me? Options: A. right, B. back, C. front, D. front right

- Answer B. back

Table 29 Examples of the camera-related data.

Camera Motion Data

[Figure 73]

Prompt The frames are captured in a continuous manner from a first-person perspective. You are to determine the main direction in which the camera is translated, disregarding small shakes or jitters and concentrating on the overall intentional movement. Which way is the camera’s perspective moving? Options: A. moving backward, B. moving rightward and forward, C. moving backward and upward, D. moving leftward

- Answer B. moving rightward and forward

[Figure 74]

Prompt The visual narrative unfolds through a series of images, all from a first-person angle. If we’re only considering horizontal rotation, does the camera pan left or right from image one to image two? Options: A. panning to the left, B. panning to the right

Answer A. panning to the left

###### Table 30 Examples of the camera motion data.

###### Object-Object Video Data

[Figure 75]

Prompt During the course of this video, which of the objects flowerpot, vase, chandelier or stool is closest to the light?

Options: A. flowerpot, B. stool, C. chandelier, D. vase

- Answer C. chandelier

[Figure 76]

Prompt Consider the scene in the video. You are positioned at shoe, with your gaze fixed on basket. In which direction

is bucket? Options: A. back, B. back-left, C. left, D. front-right

- Answer D. front-right

Prompt Assuming you are at bucket and looking at basket, determine the location of shoe relative to you. Options: A.

back-left, B. front-right, C. right, D. front

- Answer A. back-left

- Table 31 Examples of the object-object video data.

Video Counting Data

[Figure 77]

Prompt How many chairs can you spot in this part of the video? Answer the question using a single word or phrase. Answer 8 Prompt How many stools show up in the video? Answer 1

- Table 32 Examples of the video counting data.

Video Spatiotemporal Data

[Figure 78]

Prompt Provide the appearance order for the initial sighting of these objects within the video: trash bin, pillar, table divider. Options: A. pillar, trash bin, table divider B. table divider, pillar, trash bin C. trash bin, table divider, pillar

Answer C. trash bin, table divider, pillar Prompt Arrange the given objects based on the timestamp of their first appearance in the video: cabinet, trash bin,

table divider. Options: A. table divider, trash bin, cabinet B. cabinet, trash bin, table divider C. trash bin, cabinet, table divider

- Answer B. cabinet, trash bin, table divider

###### Table 33 Examples of the video spatiotemporal data.

You are an expert in 3D spatial reasoning and camera motion analysis. Your task is to analyze a pair of images that represent a sequence of camera movements. Given a specific question-answer pair about the camera’s motion, you will first rewrite the question to be more precise and then generate a detailed, step-by-step chain-of-thought reasoning to justify the provided answer.

# Coordinate System:

- - Origin: The first image’s camera viewpoint at (0,0,0)
- - Axes:

→ X: Rightward (horizontal) ↓ Y: Downward (vertical, perpendicular to the ground) ↗ Z: Forward (along the camera’s viewing direction)

# Camera Motion Types (Rotations)

- 1. Pan (Yaw): Rotation around the vertical axis. This is like turning your head from side to side.

- * Pan right: The camera rotates to its right. From the camera’s perspective, the visual content appears to shift to the left. Objects on the left side of the frame may move out of view, while new objects on the right side may enter the frame.
- * Pan left: The camera rotates to its left. The visual content appears to shift to the right.

- 2. Tilt (Pitch): Rotation around the horizontal axis. This is like nodding your head up or down.

- * Tilt Up: The camera points upwards. From the camera’s perspective, the scene appears to shift downwards. Objects originally in the center may move towards the bottom of the frame, and new content may appear at the top.
- * Tilt Down: The camera points downwards. The scene appears to shift upwards.

- 3. Roll: Rotation around the forward viewing direction. This is like tilting your head to touch your ear to your shoulder.

- * Roll Clockwise: The camera rotates clockwise. From the camera’s perspective, the entire scene and all objects within it will appear to rotate counter-clockwise. You should pay more
- * Roll Counter-clockwise: The camera rotates counter-clockwise. The scene will appear to rotate clockwise.

# Input and Output format The input is a set of images, question type, question, and answer. The output should be a triplet in JSON format, containing a revised question, a chain-of-thought process, and an answer: {{ ’question’: revised_question, ’thought’: generated_thought, ’answer’: given_answer }} # Requirement To generate high-quality Question, Answer, and Reasoning triplets. The generated content must be coherent, logical, and adhere to the specified format and persona.

- 1. Rewrite the Provided Question

- * Your primary task is to rephrase the original, user-provided question.
- * The rewritten question must be clear, specific, and well-formulated. It should eliminate any ambiguity or vagueness present in the original.
- * The core intent and subject matter of the original question must be preserved. Do not change the topic.

- 2. For the answer, provide an accurate answer that directly corresponds to the **rewritten question**. The question and answer should be match.
- 3. Construct a Detailed Reasoning Process

- * This is a critical component. You must articulate the step-by-step thought process that leads from the question to the answer.
- * The reasoning should be logical, transparent, and demonstrate how you arrived at the final answer. It should explain *why* you chose certain information, how you connected different concepts, and the structure of your final response.
- * Your reasoning should be presented from a first-person perspective (e.g., "First, I will examine...", "I can see that...")

- 4. Embrace Creativity and Expansion

* You are encouraged to be creative and expand on the given basic knowledge. Do NOT include markdown format. Do NOT include any numerical calculations. Do NOT include any coordinate-related information and their values, such as ‘x-axis‘, ‘y-axis‘ , ‘z-axis‘, ‘X-axis‘, ‘Y-axis‘ , ‘Z-axis‘. Do NOT use phrases like ‘according to ground-truth‘ or similar expressions. # Example These are some examples but don’t feel constrained by the few-shot examples; you have creative freedom.

{few_shot_examples}

# Input

{input}

Now, generate response following the task description, input and output format, and examples. Please strictly follow the JSON format. Use different phrasings or styles to ask the questions while maintaining consistency with the intent and structure of the examples.

###### Table 34 Prompt used in the generation of camera rotation reasoning data. {few_shot_examples} will be replaced by manually verified samples. {input} is the question and answer.

You are an expert in 3D spatial reasoning and camera motion analysis. Your task is to analyze a pair of images that represent a sequence of camera movements. Given a specific question-answer pair about the camera’s motion, you will first rewrite the question to be more precise and then generate a detailed, step-by-step chain-of-thought reasoning to justify the provided answer.

# Coordinate System:

- - Origin: The first image’s camera viewpoint at (0,0,0)
- - Axes:

→ X: Rightward (horizontal) ↓ Y: Downward (vertical, perpendicular to the ground) ↗ Z: Forward (along the camera’s viewing direction)

# Parallax Effect: The parallax effect is the cornerstone of your analysis. It is the apparent displacement of an object when viewed from different lines of sight.

- - Key Insight: Objects closer to the camera (foreground) will appear to move more significantly against the background than objects farther away. # Camera Motion Types (Translation)

- 1. Translation along Z-axis (Forward/Backward):

- - Moving Forward: All objects appear larger in appearent size. Foreground objects expand significantly more than background objects. Objects appear to move radially outward from the center of the view.
- - Moving Backward: All objects appear smaller in appearent object size. Foreground objects shrink significantly more than background objects. Objects appear to move radially inward toward the center of the view. When referring to ’object size,’ it is important to distinguish between ’apparent object size’ and ’real object size in 3D space.’ This is because the apparent size of an object changes as the camera moves, while its real size in 3D space remains constant.

- 2. Translation along X-axis (Left/Right):

- - Moving Rightward: Foreground objects appear to shift significantly to the LEFT relative to the background.
- - Moving Leftward: Foreground objects appear to shift significantly to the RIGHT relative to the background.

- 3. Translation along Y-axis (Up/Down):

- - Moving Upward: Foreground objects appear to shift significantly DOWNWARD relative to the background.
- - Moving Downward (+Y): Foreground objects appear to shift significantly UPWARD relative to the background.

# Input and Output format The input is a set of images, question type, question, and answer. The output should be a triplet in JSON format, containing a revised question, a chain-of-thought process, and an answer: {{ ’question’: revised_question, ’thought’: generated_thought, ’answer’: given_answer }}

# Requirement To generate high-quality Question, Answer, and Reasoning triplets. The generated content must be coherent, logical, and adhere to the specified format and persona.

- 1. Rewrite the Provided Question

- * Your primary task is to rephrase the original, user-provided question.
- * The rewritten question must be clear, specific, and well-formulated. It should eliminate any ambiguity or vagueness present in the original.
- * The core intent and subject matter of the original question must be preserved. Do not change the topic.

- 2. For the answer, provide an accurate answer that directly corresponds to the rewritten question. The question and answer should be match.
- 3. Construct a Detailed Reasoning Process

- * This is a critical component. You must articulate the step-by-step thought process that leads from the question to the answer.
- * The reasoning should be logical, transparent, and demonstrate how you arrived at the final answer. It should explain *why* you chose certain information, how you connected different concepts, and the structure of your final response.
- * Your reasoning should be presented from a first-person perspective (e.g., "First, I will examine...", "I can see that...")
- * When referring to ’object size,’ it is important to distinguish between ’apparent object size’ and ’real object size in 3D space.’ This is because the apparent size of an object changes as the camera moves, while its real size in 3D space remains constant.

- 4. Embrace Creativity and Expansion

* You are encouraged to be creative and expand on the given basic knowledge. Do NOT include markdown format. Do NOT include any numerical calculations. Do NOT include any coordinate-related information and their values, such as ‘x-axis‘, ‘y-axis‘ , ‘z-axis‘, ‘X-axis‘, ‘Y-axis‘ , ‘Z-axis‘. Do NOT use phrases like ‘according to ground-truth‘ or similar expressions. # Example These are some examples but don’t feel constrained by the few-shot examples; you have creative freedom.

{few_shot_examples}

# Input

{input}

Now, generate response following the task description, input and output format, and examples. Please strictly follow the JSON format. Use different phrasings or styles to ask the questions while maintaining consistency with the intent and structure of the examples.

###### Table 35 Prompt used in the generation of camera translation reasoning data. {few_shot_examples} will be replaced by manually verified samples. {input} is the question and answer.

###### Spatial Reasoning for Camera Motion

[Figure 79]

Prompt You take two consecutive photos (Image 1 and Image 2) from a first-person perspective. How does the camera move in space? Options: A. forward and left, B. upward, C. backward, leftward and downward, D. leftward, backward and upward

Answer <think>1. Start by comparing the two images to identify changes in object visibility and apparent size:

-In Image 1, a large machine is on the right side, and the left-side workbench shows fewer detailed objects. In Image 2, that machine is no longer visible, while the left-side workbench has more detailed items and appears more prominent. 2. Analyze the apparent size of the workbench: -The workbench in Image 2 has larger-appearing objects (like the blue cylinder and tools) with more details. When a camera moves closer to an object, the object’s apparent size increases. So this change implies the camera moved forward towards the workbench. 3. Evaluate the disappearance of the machine: -The machine that was on the right in Image 1 is not in Image 2. For an object on the right to leave the frame, the camera must have shifted to the left. This leftward movement means the machine is no longer within the camera’s field of view. 4. Combine the observations: -The increased apparent size of the workbench points to forward movement, and the machine’s disappearance from the right indicates leftward movement. These two movements match option A.</think> A. forward and left

Table 36 Examples of the spatial reasoning data for camera translation.

Spatial Reasoning for Camera Motion

[Figure 80]

Prompt Two consecutive photos show a scene from a first-person view. When considering only horizontal rotation (pan), determine if the camera pans to the right or left from the first photo to the second. Options: A. Panning to the right, B. Panning to the left

Answer <think>1. First, I need to recall what pan right and pan left mean. When the camera pans right, from its perspective, the scene shifts to the left. This means objects on the left side of the first frame may move out of view, and new objects on the right side may appear. When the camera pans left, the scene shifts to the right, with right-side objects moving out and left-side new objects appearing. 2. Now, I’ll compare the two images.

- In image-1, on the left side, there’s a bright area with what seems to be a doorway or sliding door structure.
- In image-2, this left-side doorway structure is no longer visible. That indicates that the part of the scene that was on the left in image-1 has moved out of the frame, which is a sign of a rightward pan. 3. Next, look at the right side. In image-1, the cabinet is partially visible on the right. In image-2, more of the cabinet’s right side is shown, including a dark sliding door and additional greenery (plants) on the far right. These are new elements that entered the frame, which happens when the camera pans right and reveals more of the right-hand side of the scene. 4. Putting this together: the disappearance of the left-side doorway and the appearance of new right-side elements match the behavior of a camera panning to the right. Because when the camera pans right, the scene shifts left from the camera’s view, causing left-side objects to exit and right-side objects to enter.</think> A. Panning to the right

###### Table 37 Examples of the spatial reasoning data for camera rotation.

