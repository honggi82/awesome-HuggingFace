# arXiv:2512.12799v1[cs.CV]14Dec2025

## DrivePI: Spatial-aware 4D MLLM for Unified Autonomous Driving Understanding, Perception, Prediction and Planning

Zhe Liu1, Runhui Huang1, Rui Yang1, Siming Yan2, Zining Wang2, Lu Hou2, Di Lin3, Xiang Bai4, Hengshuang Zhao1,

1The University of Hong Kong, 2Yinwang Intelligent Technology Co. Ltd., 3Tianjin University, 4Huazhong University of Science and Technology

#### Abstract

- (a) Vision-Action (VA) Models
- (b) Vision-Language-Action (VLA) Models
- (c) Vision-Language-Perception/Prediction/Action Models (Ours)

[Figure 1]

|3D Perception|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

Although multi-modal large language models (MLLMs) have shown strong capabilities across diverse domains, their application in generating fine-grained 3D perception and prediction outputs in autonomous driving remains underexplored. In this paper, we propose DrivePI, a novel spatial-aware 4D MLLM that serves as a unified VisionLanguage-Action (VLA) framework that is also compatible with vision-action (VA) models. Our method jointly performs spatial understanding, 3D perception (i.e., 3D occupancy), prediction (i.e., occupancy flow), and planning (i.e., action outputs) in parallel through end-to-end optimization. To obtain both precise geometric information and rich visual appearance, our approach integrates point clouds, multi-view images, and language instructions within a unified MLLM architecture. We further develop a data engine to generate text-occupancy and text-flow QA pairs for 4D spatial understanding. Remarkably, with only a 0.5B Qwen2.5 model as MLLM backbone, DrivePI as a single unified model matches or exceeds both existing VLA models and specialized VA models. Specifically, compared to VLA models, DrivePI outperforms OpenDriveVLA-7B by 2.5% mean accuracy on nuScenes-QA and reduces collision rate by 70% over ORION (from 0.37% to 0.11%) on nuScenes. Against specialized VA models, DrivePI surpasses FB-OCC by 10.3 RayIoU for 3D occupancy on OpenOcc, reduces the mAVE from 0.591 to 0.509 for occupancy flow on OpenOcc, and achieves 32% lower L2 error than VAD (from 0.72m to

[Figure 5]

[Figure 6]

Vision Models

|Action|
|---|

[Figure 7]

|[Figure 8]|
|---|

|Prediction|
|---|

[Figure 9]

Question-Answer

[Figure 10]

|Scene Descriptions|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Action

[Figure 16]

MLLM

|Text Instructions| |
|---|---|
| | |

|Action|
|---|

|3D Perception|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

|Prediction|
|---|

[Figure 23]

Question-Answer

MLLM

[Figure 24]

[Figure 25]

|[Figure 26]|
|---|

DriveVLX

|Scene Descriptions|
|---|

3D Occupancy

Occupancy Flow Action

Text Instructions

|Action|
|---|

Figure 1. (a) presents the pipeline of mainstream visionaction (VA) models for end-to-end autonomous driving. (b) illustrates mainstream Vision-Language-Action (VLA) models. (c) shows our DrivePI, which combines coarse-grained linguistic understanding with fine-grained 3D perception and prediction, inheriting advantages both existing VA models and VLA models.

nals, achieving substantial progress. Specifically, as shown in Figure 1 (a), UniAD [14] and VAD [20] adopt a modular framework that progresses from 3D perception to prediction, subsequently aggregating this information to generate final driving actions. Furthermore, FusionAD [64] combines LiDAR point clouds and camera images to further enhance UniAD’s performance. Although these methods achieve promising results through their accurate spatial perception capabilities and the modular designs, they are limited in language-based scene interaction, which reduces user-friendliness.

- 0.49m) for planning on nuScenes. Code will be available at https://github.com/happinesslz/DrivePI.
- 1. Introduction

In end-to-end autonomous driving systems, VisionAction (VA) models [14, 20] take vision information (LiDAR point clouds, images) as inputs and output action sig-

To address these limitations, researchers [9, 53, 71] have explored leveraging the powerful reasoning and humanlike decision-making capabilities of MLLMs. Specifi-

Corresponding author

cally, OpenDriveVLA [71] and ORION [9] adopt a VisionLanguage-Action (VLA) framework that takes multi-view images and language instructions as inputs and generates actions, as illustrated in Figure 1 (b). These VLA-based methods achieve superior interaction capabilities with scenarios and demonstrate enhanced user engagement. However, these VLA-based approaches usually struggle to guarantee reliable outputs due to the absence of fine-grained intermediate 3D perception and prediction outputs compared to the modular-design of VA models, consequently compromising interpretability and safety assurances.

Therefore, a natural question arises: Can we develop a unified framework that combines the precise spatial perception of VA models with the natural language interaction of VLA-based approaches? In this paper, we propose DrivePI, a novel spatial-aware 4D MLLM that serves as a unified Vision-Language-Action (VLA) framework for autonomous driving, as illustrated in Figure 1(c). Here, we term it 4D MLLM as it outputs both 3D occupancy and flow, capturing fine-grained spatial-temporal dynamics. Unlike existing approaches that treat VA and VLA-based methods as separate paradigms, DrivePI establishes a unified architecture that seamlessly integrates the spatial precision of VA models with the interpretability and interactive capabilities of VLA frameworks.

Specifically, DrivePI exhibits four distinctive characteristics that differentiate it from existing approaches. first, unlike mainstream VLA-based methods that rely solely on camera images as input, DrivePI introduces LiDAR as a complementary sensing modality, providing precise 3D geometric information that better elicits the spatial understanding capabilities of MLLMs. Second, DrivePI generates intermediate fine-grained 3D perception (e.g., 3D occupancy) and prediction (e.g., occupancy flow) representations to ensure that the output features of the MLLM maintain reliable spatial perception capabilities, thereby enhancing both interpretability and safety guarantees for autonomous driving systems. Third, we develop enriched data engines that seamlessly integrate 3D occupancy and occupancy flow representations into natural language scene descriptions, enabling the model to reason about complex spatial-temporal dynamics through textual understanding. Fourth, DrivePI serves as a unified model, which employs end-to-end joint optimization across all tasks including 3D perception, prediction, planning, and scene understanding.

In summary, our contributions are as follows:

• We propose DrivePI, the first unified spatial-aware 4D MLLM framework that seamlessly integrates coarsegrained linguistic spatial understanding with fine-grained 3D perception capabilities, bridging the gap between VAbased and VLA-based paradigms in autonomous driving while inheriting the complementary strengths of both approaches;

- • We incorporate LiDAR as a complementary sensing modality alongside camera imagery, providing highprecision 3D geometric information that better elicits the spatial understanding capabilities of MLLMs. Furthermore, DrivePI enables accurate 3D perception (e.g., occupancy prediction) and prediction (e.g., occupancy flow), which effectively enhances the interpretability and safety assurances;
- • We develop three complementary spatial understanding benchmarks based on our data engine by constructing multiple question-answer (QA) pairs: 3D occupancy perception for static scene understanding, occupancy flow prediction for dynamic motion analysis, and trajectory planning for decision-making evaluation. These benchmarks collectively assess different aspects of linguistic spatial reasoning capabilities across the temporal and spatial dimensions.
- • Despite utilizing only a compact 0.5B parameter MLLM backbone, DrivePI even outperforms existing VA models in 3D occupancy and occupancy flow while maintaining comparable interactive capabilities with existing VLA frameworks in autonomous driving.

#### 2. Related Works

Multimodal Large Language Model. With the rapid development of large language models (LLMs) [2, 31, 46, 61], numerous works have attempted to incorporate additional modalities into LLMs, thereby expanding their application capabilities. Currently, multimodal large language models (MLLMs) have been successfully adopted across various tasks, including image understanding [3, 7, 32, 34, 37], video understanding [30, 55, 56], and semantic understanding [4, 22, 42, 51, 67, 69], demonstrating the enormous potential of LLMs for downstream applications. Consequently, an increasing number of works [1, 38, 39, 58, 62, 70] have begun exploring how to achieve spatial intelligence through MLLMs. Specifically, VSI-Bench [62] establishes a comprehensive benchmark that collects diverse indoor images and generates a series of questions designed to evaluate 3D spatial understanding, thereby validating the spatial reasoning capabilities of MLLMs. Gemini RoboticsER [1] predicts metric 3D bounding boxes from single images and further achieves open-vocabulary 3D object detection. Following this direction, Seed1.5-VL [12] enhances the perception capabilities of MLLMs and simultaneously processes 2D or 3D grounding tasks alongside other tasks (e.g., OCR, spatial understanding) from images using a unified MLLM architecture. However, these methods mainly achieve coarse-grained spatial perception such as describing relationships among objects or predicting object-level 3D bounding boxes.

End-to-End Autonomous Driving. The autonomous driving system has undergone tremendous changes due to the

[Figure 27]

- Figure 2. The pipeline of DrivePI consists of the following steps. First, we employ a vision encoder to extract features from images and LiDAR data, obtaining latent BEV features that are then converted into vision tokens by a spatial projector. Next, we feed both vision tokens and text tokens into the MLLM to generate output tokens. The MLLM produces responses through four specialized heads: a text head for scene understanding in an auto-regressive manner, a 3D occupancy head for accurate spatial perception, an occupancy flow head for pixel-level motion prediction, and an action diffusion head for trajectory planning.

development of the end-to-end model. Many works [14, 20, 27, 45, 57, 64] have explored how to perform perception, prediction, and planning tasks in an end-to-end model, greatly improving the upper limit of planning performance and reducing the complexity of the autonomous driving system. UniAD [14] adopts a modular architecture to process each task by different modules and jointly train these tasks in an end-to-end manner. VAD [20] proposes a vectorized representation and enables interaction among perception, mapping, and planning queries through different transformer decoders [50]. Although these methods demonstrate promising planning results, they lack the ability to interact with users through natural language descriptions, resulting in reduced user-friendliness.

Vision Language Action Model. Leveraging the powerful and versatile capabilities of MLLMs, several researches [43, 44, 48, 53, 54, 59, 60] have successfully developed MLLMs for autonomous driving systems. These methods exploit the advanced reasoning capabilities of MLLMs to generate rich scene descriptions and high-level driving commands, thereby enhancing interpretability of end-to-end planning. Furthermore, to fully unleash comprehension abilities of MLLMs, recent works [9, 18, 19, 23, 24, 71] integrate the planning module directly into MLLMs to achieve vision-language-action (VLA) frameworks, enabling the integration of reasoning and planning tasks. For example, OpenDriveVLA [71] employs a Bird’sEye-View (BEV) representation to generate distinct tokens corresponding to agents, static maps, and scene context. These tokens are subsequently projected into a unified semantic space and processed by the MLLM to facilitate information interactions for generating trajectories by a planning decoder. In this paper, we propose DrivePI, a new VLA framework that seamlessly integrates both coarse-grained and fine-grained granularities. DrivePI preserves the interpretability of textual representations for perception and planning reasoning while simultaneously employing finegrained decoding mechanisms to generate precise 3D occu-

pancy and trajectory predictions.

#### 3. Methods

Multi-modal large language models have garnered increasing attention in autonomous driving due to their capabilities for user interaction and human-like decision-making. However, existing LLM-based methods struggle to directly output fine-grained perception results (e.g., 3D occupancy and occupancy flow) through next-token prediction, which are readily achievable by VA approaches. To address this limitation, we propose DrivePI, a new VLA framework that achieves both coarse-grained linguistic understanding and fine-grained spatial perception, thereby inheriting the complementary advantages of both VA models and VLA frameworks. Next, we will introduce the details of DrivePI.

##### 3.1. Overview

As shown in Figure 2, we present the pipeline of DrivePI. First, to ensure that the inputs of MLLMs contain accurate geometric information, we incorporate LiDAR point clouds as additional inputs (Note: LiDAR point clouds contains temporal information on the nuScenes dataset.), which provide precise 3D spatial information compared to camera images alone. This enhancement is instrumental in exploring the spatial perception capabilities of MLLMs. Second, we employ an advanced multi-modal vision encoder [36] to process multi-view images and LiDAR point clouds, subsequently converting them into a compact latent BEV feature representation. Next, a spatial projector is utilized to map the latent BEV features into the language space and obtain vision tokens. These vision tokens, along with text tokens, are then fed into the MLLM. Finally, we utilize four specialized heads: a text head to generate responses for scene understanding in an auto-regressive manner, a 3D occupancy head for accurate 3D perception, an occupancy flow head for fine-grained motion prediction, and an action diffusion head for trajectory planning. Note that the MLLM is train-

able, and all tasks are jointly optimized during training.

Spatial Projector. Given the latent BEV feature Fbev ∈ RH×W×C, where H × W typically exceeds a resolution of 100 × 100, directly inputting these features at the pixel level into the MLLM would incur prohibitive computational costs. Here, H,W,C are the height, width and channel dimensions of Fbev, respectively. To address this challenge, we first patchify Fbev into N patches of size K × K, producing visual features Fpatch ∈ RN×K

2×C, where

N = KH × WK . A conventional approach involves applying pooling operations to aggregate the K × K spatial fea-

tures into a single representation, yielding pooled features Fpool ∈ RN×1×C. However, this approach typically results in the loss of fine-grained spatial information. Therefore, following [17] for handling high-resolution image inputs in

- 2D MLLMs, we adopt a cross-attention mechanism with

Fpool, Fpatch, and Fpatch serving as query, key, and value, respectively. This design preserves more detailed spatial information. Finally, we employ a linear layer to transform the processed features to match the channel dimension Cl of the MLLM’s input hidden states, producing the final vision tokens Fv ∈ RN×C

l.

##### 3.2. Coarse-grained Spatial Understanding

In this paper, we define coarse-grained spatial understanding as the text-based descriptions that arises from the highly compressed nature of language compared to rich visual information (e.g., images, LiDAR point clouds). Although MLLMs demonstrate remarkable capabilities in many domains, exploring their spatial understanding remains a critical challenge, particularly in autonomous driving where precise spatial reasoning is essential. Therefore, we develop three complementary spatial understanding benchmarks based on our data engine by constructing multiple question-answer (QA) pairs: 3D occupancy perception for static scene understanding, occupancy flow prediction for dynamic motion analysis, and trajectory planning for decision-making evaluation. These benchmarks collectively assess different aspects of linguistic spatial reasoning capabilities across the temporal and spatial dimensions.

Data Engine. As shown in Figure 3, our pipeline consists of three main stages: caption annotation for scene understanding, 4D spatial understanding annotation, and planning reasoning annotation. In the first stage, we employ InternVL3-78B [72] to generate scene descriptions for the front and back views separately. This strategy prevents potential confusion that may arise when MLLMs struggle to distinguish between different viewpoints. The captions from both views are then merged to construct a comprehensive description of the entire scene, which is further refined to ensure high descriptive quality. The second stage aims to equip the MLLM with 4D spatial understanding capabilities. Specifically, we leverage ground-truth occupancy

and flow data to generate diverse text-occupancy and textflow QA pairs. These QA pairs focus on key tasks such as determining whether given positions are occupied, identifying corresponding object categories, and predicting velocity information. This enables DrivePI to explore finegrained 3D occupancy and flow occupancy in textual format compared to previous methods [24, 43, 59]. In the final stage, we generate text-planning QA pairs to enhance planning interpretability based on future trajectory annotations of the ego-vehicle. These pairs require the MLLM to analyze the surrounding environment and provide high-level driving commands along with suggested trajectories. This multi-stage pipeline ensures both versatility and high quality of the generated language dataset, enabling the MLLM to develop 4D spatial understanding and planning capabilities. For more details of data engine (e.g., the design of prompts), please refer to our supplemental materials.

##### 3.3. Fine-grained Spatial Learning

In this paper, we define fine-grained spatial learning as the integration of explicit spatial capabilities, including 3D occupancy, occupancy flow, and trajectory planning. While language descriptions effectively capture high-level semantic concepts and global spatial arrangements, they inherently lack the precision required for detailed spatial localization and geometric understanding essential in autonomous driving tasks. Therefore, we introduce finegrained vision heads to address this.

Fine-grained Visual Heads. We adopt three fine-grained visual heads to enable precise spatial capabilities: a 3D occupancy head for volumetric scene understanding, an occupancy flow head for temporal dynamics modeling, and an action diffusion head for trajectory planning. Specifically, to implement these fine-grained heads, we first extract the corresponding vision tokens Fv∗ ∈ RN×C

l from the multimodal representation. Subsequently, we employ a linear projection layer to transform Fv∗ into Fvout ∈ RN×K

2C. We then reshape Fvout into a spatial feature map Fout ∈ RH×W×C, where H and W denote the height and width of the feature map, respectively. Based on the spatiallyorganized features Fout, we can seamlessly integrate our three prediction heads following existing VA models. For more details of each head, please refer to the supplementary material.

##### 3.4. Loss Function

We define the total loss as a weighted summation of four components: Lllm for the text head, Locc for the occupancy head, Lflow for the occupancy flow head, and Laction for the action diffusion head. Therefore, the total loss Ltotal is formulated as:

###### Ltotal = λ1Lllm + λ2Locc + λ3Lflow + λ4Laction, (1)

Front views

Trajectory

[Figure 28]

|[Figure 29]|
|---|

Stage I: Scene Description

Stage II: 4D spatial understanding

Stage III: Planning reasoning

🗎 🗎

Multi-turn Scene Description 3D Occupancy Flow Prediction Action Prediction Trajectory Prediction

Q: What is the safe action

[Figure 30]

Caption

Q: What object is occupying (70, 120, 15) A: car

[Figure 31]

[Figure 32]

of the ego car? A: Go straight.

[Figure 33]

Language Data

Raw Data

[Figure 34]

Merging and Rewriting

[Figure 35]

Q: Predict the future 6-frame trajectory A: Future 6-frame trajectory: [(3.24, -0.03), …, (3.81, -0.08), (4.67, -0.08)]

Multi-turn

⛜

Q: please provide its velocity A: vx: 1.5, vy: 2.5

[Figure 36]

[Figure 37]

[Figure 38]

Caption

⛜

[Figure 39]

|[Figure 40]|
|---|

|[Figure 41]|
|---|

Back views

Occupancy and Flow

- Figure 3. The illustration of our multi-stage data pipeline. We first generate captions of front and back views, respectively. Then, we use InternVL3-78B (adopts Qwen2.5-72B [3] as the language model) to combine these captions to merge and polish generated scene descriptions. Moreover, we generate text-occupancy and text-flow QA pairs based on occupancy and flow ground truth by multi-turn conversations to improve the 4D spatial understanding ability. Finally, we generate text-planning QA pairs to allow MLLM to predict the future actions of ego-vehicle.

where λ1,λ2,λ3,λ4 are the balancing weights for text scene understanding, 3D occupancy perception, occupancy flow prediction, and trajectory planning, respectively. All tasks are optimized jointly in an end-to-end manner.

- 4. Experiments

erage Velocity Error (mAVE) [5], respectively. For planning, we follow previous methods [45, 71] and use L2 distance error and collision rate metrics. To evaluate text understanding, we adopt the official evaluation metrics from the nuScenes-QA benchmark [41]. For 4D spatial understanding (see Table 6), we adopt the following metrics: 1) accuracy for occupancy status, occupancy classification, action status, 2) mAVE for occupancy flow, 3) L2 distance error and collision rate for planning.

##### 4.1. Datasets and Evaluation Metrics

In this paper, we conduct comprehensive experiments on the nuScenes [5] dataset, a large-scale autonomous driving benchmark. The dataset comprises 750 training scenes, 150 validation scenes, and 150 test scenes. nuScenes provides synchronized multi-modal sensor data including LiDAR point clouds and multi-view images from 6 cameras. For occupancy prediction, we employ OpenOcc [49] as the primary occupancy evaluation. Moreover, to make a comprehensive comparison with most 3D occupancy methods, we evaluate the reuslts on Occ3D [47]. For occupancy flow, we utilize the occupancy flow annotations provided by OpenOcc [49] to evaluate the performance of DrivePI in fine-grained motion prediction. For trajectory planning evaluation, we follow established protocols from prior works [53, 71] and employ open-loop evaluation metrics on the nuScenes dataset. For text understanding, we conduct experiments on the nuScenes-QA [41] dataset containing 377k training question-answer (QA) pairs, and employ our proposed data generation pipeline to generate 84k training scene descriptions, 560k QA pairs for 4D spatial reasoning and 24k for planning reasoning, culminating in a comprehensive training dataset of over 1.0M QA pairs.

##### 4.2. Implementation Details

To balance computational efficiency, we adopt Qwen2.50.5B [3] as our base model. The training of DrivePI comprises two stages. In the first stage, we freeze the vision encoder and the MLLM, then train the spatial projector to align visual representations with the language embedding space for 1 epoch using the generated captions of our data engine. In the second stage, we jointly optimize the spatial projector, MLLM, and all task-specific heads (i.e., text head, 3D occupancy head, occupancy flow head, and action diffusion head) for 1 epoch while keeping only the vision encoder frozen. For loss balancing, we adopt the balancing weights of λ1 = λ2 = λ3 = λ4 = 1. All experiments are conducted on 8×NVIDIA L40S GPUs.

##### 4.3. Main Results

We comprehensively evaluate DrivePI across four key tasks: text understanding, 3D occupancy, occupancy flow, and trajectory planning. For 3D occupancy and occupancy flow evaluation, we conduct comparisons on the OpenOcc [49] benchmark, which provides annotations for both 3D occupancy and flow on nuScenes. For trajectory planning, we directly adopt the annotations provided by the offi-

For evaluation metrics, we adopt the established metrics to evaluate the performance of DrivePI. For 3D occupancy and occupancy flow, we use RayIoU [33] and the mean Av-

Table 1. 3D occupancy and occupancy flow performance on the OpenOcc validation set.

Method VLM-based OccScore↑ RayIoU (3D Occ.)↑ mAVE (Occ. Flow)↓ RayIoU1m RayIoU2m RayIoU4m

OccNeRF [66] 28.5 31.7 – 16.6 29.3 49.2 RenderOcc [40] 33.0 36.7 – 20.3 32.7 49.9 LetOccFlow [35] 36.4 40.5 – 25.5 39.7 56.3 OccNet [49] 35.7 39.7 – 29.3 39.7 50.0 BEVDetOcc-SF [15] 33.0 36.7 1.420 31.6 37.3 41.1 FB-Occ [26] 39.2 39.0 0.591 32.7 39.9 44.4 F-Occ [68] 41.0 39.9 0.491 33.9 40.7 45.2 CascadeFlow [29] 40.9 39.6 0.470 33.5 40.3 45.0 ALOcc-Flow-3D [6] 43.0 41.9 0.556 35.6 42.8 47.4

DrivePI (Ours) ✓ 49.3 49.3 0.509 45.0 50.0 52.9

- Table 2. Planning performance on the nuScenes validation set. Note that our unified model DrivePI does not incorporate ego status during training by default to avoid potential shortcut learning.

L2 (m)↓ Col. (%)↓

Method VLM-based Ego Status 1s 2s 3s avg. 1s 2s 3s avg. ST-P3 [] 1.33 2.11 2.90 2.11 0.23 0.62 1.27 0.71 FF [13] 0.55 1.20 2.54 1.43 0.06 0.17 1.07 0.43 EO [21] 0.67 1.36 2.78 1.60 0.04 0.09 0.88 0.33 UniAD [14] 0.48 0.96 1.65 1.03 0.05 0.17 0.71 0.31 VAD [20] 0.41 0.70 1.05 0.72 0.07 0.17 0.41 0.22 VAD [20] ✓ 0.17 0.34 0.60 0.37 0.07 0.10 0.24 0.14 OmniDrive [53] ✓ ✓ 0.14 0.29 0.55 0.33 0.00 0.13 0.78 0.30 ORION [9] ✓ ✓ 0.17 0.31 0.55 0.34 0.05 0.25 0.80 0.37 OpenDriveVLA-7B [71] ✓ ✓ 0.20 0.58 1.21 0.66 0.00 0.22 0.55 0.25 DrivePI (Ours) ✓ 0.24 0.46 0.78 0.49 0.38 0.27 0.48 0.38 DrivePI (Ours) ✓ ✓ 0.19 0.36 0.64 0.40 0.00 0.05 0.28 0.11

cial nuScenes dataset. Moreover, to demonstrate the text understanding capability of DrivePI, we report results on the nuScenes-QA benchmark, which offers diverse visual question-answering pairs tailored to autonomous driving scenarios. Unless otherwise specified, all experiments are conducted using our unified 0.5B model architecture with shared parameter weights across all tasks.

- 3D Ocuupancy and Occupancy Flow on OpenOcc. As shown in Table 1, we compare the 3D occupancy and occupancy flow performance of DrivePI with existing approaches. DrivePI achieves superior performance across multiple metrics, with an OccScore of 49.3%, RayIoU of 49.3%, and mAVE of 0.509. DrivePI surpasses the representative method FB-OCC [26] by 10.3 RayIoU for 3D occupancy and reduces flow mAVE from 0.591 to 0.509. Notably, DrivePI outperforms the previous state-of-the-art method ALOcc-Flow-3D [6] by 6.3% in OccScore, 7.4% in RayIoU, and reduces mAVE by 0.047, establishing new state-of-the-art results with only a 0.5B MLLM backbone. This demonstrates the effectiveness of DrivePI in achieving remarkable 4D fine-grained spatial perception capabilities within a VLA framework. Planning on nuScenes. To further evaluate the planning capabilities of DrivePI as a VLA model, we conduct exper-

iments on the nuScenes benchmark [5]. As shown in Table 2, we compare the performance of our DrivePI against both traditional end-to-end VA methods and VLM-based approaches. DrivePI achieves an L2 error of 0.40 and a collision rate of 0.11% with ego status, outperforming the end-to-end VA method VAD [20] and VLA method OpenDriveVLA-7B [71] by 0.03% and 0.14% in collision rate, respectively. Notably, DrivePI reduces collision rate by 70% compared to the recent ORION [9] (from 0.37% to 0.11%). Without ego status, DrivePI achieves 32% lower L2 error than VAD (from 0.72m to 0.49m). These results demonstrate the effectiveness of DrivePI as a VLA model for planning tasks.

Text Understanding on nuScenes-QA. Text understanding capability is crucial for VLA models, as it enables autonomous driving systems to interpret and reason based on natural language, thereby supporting more human-like decision-making. We validate the text understanding performance of DrivePI on the nuScenes-QA [41] benchmark. As shown in Table 3, DrivePI with only a 0.5B model size achieves 60.7% accuracy, outperforming OpenDriveVLA7B [71] by 2.5%. These results demonstrate the promising text understanding capabilities of DrivePI.

###### 3D Occupancy on Occ3D. Beyond the unified model

- Table 3. Text Understanding performance on the nuScenes-QA validation set. Ext., Cnt., Obj., Sts., Cmp. and Acc. are short for exist, count, object, status, comparison, and the overall accuracy.

Method Ext.↑ Cnt.↑ Obj.↑ Sts.↑ Cmp.↑ Acc.↑ LLaMA-AdapV2 [11] 19.3 2.7 7.6 10.8 1.6 9.6 LLaVA1.5 [34] 45.8 7.7 7.8 9.0 52.1 26.2 LiDAR-LLM [63] 74.5 15.0 37.8 45.9 57.8 48.6 BEVDet+BUTD [41] 83.7 20.9 48.8 52.0 67.7 57.0 OpenDriveVLA-0.5B [71] 83.9 22.0 50.2 57.0 68.4 58.4 OpenDriveVLA-3B [71] 84.0 22.3 50.3 56.9 68.5 58.5 OpenDriveVLA-7B [71] 84.2 22.7 49.6 54.5 68.8 58.2

DrivePI (Ours) 85.3 22.4 57.5 59.1 68.3 60.7

- Table 4. 3D occupancy performance on the Occ3D-nuScenes validation set. * indicates that DrivePI is trained exclusively on the 3D occupancy task of Occ3D-nuScenes.

Method VLM-based RayIoU↑ RayIoU1m RayIoU2m RayIoU4m

RenderOcc [40] 19.5 13.4 19.6 25.5 SimpleOcc [10] 22.5 17.0 22.7 27.9 BEVFormer [25] 32.4 26.1 32.9 38.0 BEVDet-Occ [16] 32.6 26.6 33.1 38.2 FB-Occ [26] 33.5 26.7 34.1 39.7 SparseOcc [33] 36.1 30.2 36.8 41.2 OPUS [52] 41.2 34.7 42.1 46.7

DrivePI* (Ours) ✓ 46.0 42.2 46.7 49.2

- Table 5. Ablation study for text head and vision head in DrivePI.

ponent, we conduct ablation studies by removing individual heads, with quantitative results summarized in Table 5. When only the text head is enabled (I), DrivePI achieves competitive performance with 61.2% accuracy on the text understanding task. When only the vision head (i.e., occupancy, flow, and planning) is enabled (II), DrivePI still delivers promising results: 47.5% RayIoU for 3D occupancy, 0.69 mAVE for occupancy flow, and 1.02 L2 error with 0.39 collision rate for trajectory planning (we not use ego-vehicle information during training process). When both text and vision heads are combined (III), DrivePI achieves superior performance across most tasks. Compared to the visiononly setting (II), the unified model (III) increases RayIoU by 1.8%, reduces mAVE by 0.18, and decreases L2 error by 0.52. The main reason is that text understanding helps better align with the appropriate feature space for vision tasks. Furthermore, it maintains comparable text understanding performance with 60.7% accuracy, close to the text-only setting (I). These results verify the effectiveness of unifying text understanding with 3D perception, prediction, and planning tasks within a single VLA framework.

3D Occ. Occ. Flow Planning QA

# Text Head Vision Head

RayIoU↑ mAVE↓ L2↓ Col.↓ Acc.↑

- I ✓ – – – – – 61.2

- II – ✓ 47.5 0.69 1.02 0.39 –

- III ✓ ✓ 49.3 0.51 0.50 0.38 60.7

DrivePI trained jointly on all tasks, we also train DrivePI exclusively on the 3D occupancy task of Occ3D to enable comprehensive comparisons with existing approaches on the Occ3D benchmark [47]. As shown in Table 4, DrivePI achieves state-of-the-art performance with 46.0% RayIoU, surpassing the previous best method OPUS [52] by a significant margin of 4.8%. Notably, DrivePI is built on a MLLM architecture, which highlights its strong capabilities in finegrained 3D perception, despite being primarily designed for multi-modal understanding.

Scaling Text Data. To explore the impact of different sizes of text data, we conduct a series of ablation studies by varying the amount of instruction-tuning data across multiple tasks, including predicting occupancy status, occupancy category, action status, and trajectory planning from our data engine, as well as QA from nuScenes-QA. As summarized in Table 6, we mainly conduct the experiments on the Qwen-2.5 3B model using only the text head. When trained with only 112K samples (84K captions + 28K occupancy QA pairs), DrivePI achieves only 73% accuracy in occupied status prediction and 14.3% accuracy in occupancy category prediction. When scaling the occupancy QA pairs from 28K to 560K, there is an obvious performance improvement with 14% accuracy on predicting occupancy status and 44.9% accuracy on predicting occupancy category. When further adding QA pairs for occupancy flow, action, and the official nuScenes-QA, the accuracy of predicting occupancy status and category brings performance improvements of 1.6% and 0.6%, respectively. Besides, the

##### 4.4. Ablation Study

In this section, we conduct the ablation studies to illustrate the effectiveness of our designs in DrivePI. Unless otherwise specified, we use the DrivePI-0.5B by default.

Ablation on Text Head and Vision Head. DrivePI integrates text understanding, 3D perception, prediction, and planning within a unified VLA framework. For simplicity, we refer to the combination of the fine-grained 3D occupancy head, occupancy flow head, and trajectory planning head as the vision head. To evaluate the impact of each com-

- Table 6. The ablation study of DrivePI exploring data scaling. The columns of Occ. Status, Occ. Class, Occ. Flow, Action Status denote the occupancy status (i.e., “yes” or “no”), occupancy category, the occupancy flow, the action commands (i.e., “straight”, “right”, “left”, and “stop”), respectively.

Training Data size Occ. Status Occ. Class Occ. Flow Action Status Planning QA

Model Size

Caption OCC. Flow Action QA Acc.↑ Acc.↑ mAVE↓ Acc.↑ L2↓ Col.↓ Acc.↑ 0.5B 84k 420k 140k 24k 377k 86.0 50.4 0.91 83.8 0.79 0.63 60.7

84k 28k – – – 73.0 14.3 – – – – 84k 56k – – – 74.2 22.4 – – – – – 84k 280k – – – 86.8 54.7 – – – – – 84k 560k – – – 87.0 59.2 – – – – – 84k 420k 140k 24k 377k 88.6 59.8 0.69 83.3 0.86 0.62 63.0

3B

Coarse-grained Understanding Fine-grained Understanding

[Figure 42]

[Figure 43]

Scene Description

The scene depicts an urban street environment with various vehicles and buildings. There are multiple cars, some parked and others moving, along with a few pedestrians. Construction barriers and traffic cones are present, indicating ongoing construction or roadwork. The area includes modern buildings, traffic lights, and a pedestrian crossing. The sky is partly cloudy, suggesting a typical day in the city.

|[Figure 44]<br><br>Occ Pred.|
|---|

|[Figure 45]<br><br>Trajectory Pred.|
|---|

|[Figure 46]<br><br>[(3.46, -0.07), (3.65, -0.11), (3.89, 0.15), (4.09, -0.21), (4.29, -0.21), (4.24, -0.22)].<br><br>Trajectory Pred.|
|---|

Q: What object is occupying position (65, 136, 7)?

A: car

Occ Pred.

|[Figure 47]|
|---|

Q: What is the safe action of the ego car? Predict the future 6-frame trajectory of the ego car in the last.

|[Figure 48]<br><br>GT Occ|
|---|

|[Figure 49]<br><br>Flow Pred.|
|---|

|[Figure 50]<br><br>GT Trajectory|
|---|

A: Go straight. Future 6-frame trajectory: [(3.46, -0.07), …, (4.24, 0.22)].

Figure 4. The visualization of coarse-grained and fine-grained understanding of DrivePI. We present the results of scene description, 3D occupancy, trajectory prediction results to illustrate the coarse-grained understanding of DrivePI.

corresponding performance for occupancy flow, action status, and planning is 0.69 mAVE, 83.3% Acc., and 0.62 collision rate. Finally, we also evaluate the 0.5B model with joint training of text and vision heads. We find that our 0.5B model even achieves better performance on action status prediction and L2 error despite having lower QA accuracy, demonstrating the effectiveness of DrivePI in unifying text understanding, 3D perception, prediction, and planning.

##### 4.5. Visualization

In this section, we present qualitative visualizations to showcase the multi-granularity understanding capabilities of DrivePI. As illustrated in Figure 4, we present the visualization of DrivePI on scene descriptions, 3D occupancy, occupancy flow, action, and trajectory predictions. At a coarse-grained level, DrivePI generates detailed scene descriptions and reasonable answers based on different instructions. While at a fine-grained level, it produces finegrained results (e.g., 3D occupancy, occupancy flow, and

trajectory planning) through specialized decoding heads, enabling more precise prediction results. Besides, we have two important observations. First, regarding scene descriptions, DrivePI demonstrates the ability to understand detailed appearance information (such as “the sky is partly cloudy”), indicating that the vision encoder effectively preserves critical visual information even when transforming front-view images into Bird’s-Eye-View representation. Second, we observe strong alignment between coarse-grained and fine-grained predictions. For instance, the object category described linguistically at a coarse level corresponds to the fine-grained 3D occupancy prediction at the grid coordinate (65,136,7). Similarly, the predicted trajectory exhibits similar consistency. This alignment across different levels of understanding validates the effectiveness of DrivePI in unifying coarse-grained linguistic spatial understanding with fine-grained 3D perception capabilities, which enhances the interpretability and explainable decision-making of autonomous driving systems.

#### 5. Conclusion

In this paper, we have presented a novel VLA framework, DrivePI, which achieves both coarse-grained spatial understanding in text formats and fine-grained spatial perception comparable to VA models, thereby inheriting the advantages of both existing VA models and VLA frameworks. Notably, our DrivePI employs only a 0.5B parameter LLM as its backbone, demonstrating remarkable efficiency. Despite utilizing only a compact 0.5B parameter MLLM backbone, DrivePI with promising text understanding, even outperforms existing VA models in 3D occupancy and occupancy flow while maintaining comparable interactive capabilities with existing VLA-based frameworks in autonomous driving. Finally, we hope this new VLA framework can inspire future research to enhance autonomous driving systems with improved interpretability and explainable decision-making through language reasoning and finegrained 3D outputs.

Limitations. Our approach still has two main limitations that warrant future investigation. First, we use a simple multi-task learning strategy to balance loss weights across tasks, which may not be optimal to obtain the best trade-off between competing objectives. Second, we do not incorporate reinforcement learning techniques that could enhance reasoning capabilities through trial-and-error learning, particularly for complex planning scenarios.

#### References

- [1] Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, Michael Bloesch, et al. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025. 2
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 2
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 5
- [4] Zechen Bai, Tong He, Haiyang Mei, Pichao Wang, Ziteng Gao, Joya Chen, Zheng Zhang, and Mike Zheng Shou. One token to seg them all: Language instructed reasoning segmentation in videos. In NeurIPS, 2024. 2
- [5] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020. 5, 6, 3
- [6] Dubing Chen, Jin Fang, Wencheng Han, Xinjing Cheng, Junbo Yin, Chengzhong Xu, Fahad Shahbaz Khan, and Jian-

- bing Shen. Alocc: Adaptive lifting-based 3d semantic occupancy and cost volume-based flow predictions. In CVPR, 2025. 6
- [7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR,

2024. 2

- [8] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44 (10-11):1684–1704, 2025. 3
- [9] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-toend autonomous driving framework by vision-language instructed action generation. In ICCV, 2025. 1, 2, 3, 6
- [10] Wanshui Gan, Ningkai Mo, Hongbin Xu, and Naoto Yokoya. A simple attempt for 3d occupancy estimation in autonomous driving. arXiv preprint arXiv:2303.10076, 2023. 7
- [11] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010,

2023. 7

- [12] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 2
- [13] Peiyun Hu, Aaron Huang, John Dolan, David Held, and Deva Ramanan. Safe local motion planning with selfsupervised freespace forecasting. In CVPR, 2021. 6
- [14] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In CVPR, 2023. 1, 3, 6
- [15] Junjie Huang and Guan Huang. Bevdet4d: Exploit temporal cues in multi-camera 3d object detection. arXiv preprint arXiv:2203.17054, 2022. 6
- [16] Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790,

2021. 7

- [17] Runhui Huang, Xinpeng Ding, Chunwei Wang, Jianhua Han, Yulong Liu, Hengshuang Zhao, Hang Xu, Lu Hou, Wei Zhang, and Xiaodan Liang. Hires-llava: Restoring fragmentation input in high-resolution large vision-language models. In CVPR, pages 29814–29824, 2025. 4
- [18] Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262,

2024. 3

- [19] Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun Wang, Jinghao Chai, Qian Cao, Yuweng Heng, Hao Jiang,

- Yunda Dong, et al. Diffvla: Vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381, 2025. 3
- [20] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In ICCV, 2023. 1, 3, 6
- [21] Tarasha Khurana, Peiyun Hu, Achal Dave, Jason Ziglar, David Held, and Deva Ramanan. Differentiable raycasting for self-supervised occupancy forecasting. In ECCV, 2022. 6
- [22] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In CVPR, 2024. 2
- [23] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796, 2025. 3
- [24] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052, 2025. 3, 4
- [25] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In ECCV, 2022. 7
- [26] Zhiqi Li, Zhiding Yu, David Austin, Mingsheng Fang, Shiyi Lan, Jan Kautz, and Jose M Alvarez. Fb-occ: 3d occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492, 2023. 6, 7
- [27] Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for openloop end-to-end autonomous driving? In CVPR, 2024. 3
- [28] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, and Xinggang Wang. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In CVPR, 2025. 3
- [29] Zhimin Liao and Ping Wei. Cascadeflow: 3d occupancy and flow prediction with cascaded sparsity sampling refinement framework. CVPR2024 Autonomous Grand Challenge Track On Occupancy and Flow, 2024. 6
- [30] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, 2024. 2
- [31] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 2
- [32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. 2023. 2
- [33] Haisong Liu, Yang Chen, Haiguang Wang, Zetong Yang, Tianyu Li, Jia Zeng, Li Chen, Hongyang Li, and Limin Wang. Fully sparse 3d occupancy prediction. In ECCV,

2024. 5, 7

- [34] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR,

2024. 2, 7

- [35] Yili Liu, Linzhan Mou, Xuan Yu, Chenrui Han, Sitong Mao, Rong Xiong, and Yue Wang. Let occ flow: Self-supervised 3d occupancy flow prediction. In CoRL, 2025. 6
- [36] Zhe Liu, Jinghua Hou, Xiaoqing Ye, Jingdong Wang, Hengshuang Zhao, and Xiang Bai. Unilion: Towards unified autonomous driving model with linear group rnns. arXiv preprint arXiv:2511.01768, 2025. 3
- [37] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525,

2024. 2

- [38] Wufei Ma, Luoxin Ye, Celso M de Melo, Alan Yuille, and Jieneng Chen. Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models. In CVPR, 2025. 2
- [39] Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, and Zihan Zhou. Spatiallm: Training large language models for structured indoor modeling. In NeurIPS, 2025. 2
- [40] Mingjie Pan, Jiaming Liu, Renrui Zhang, Peixiang Huang, Xiaoqi Li, Hongwei Xie, Bing Wang, Li Liu, and Shanghang Zhang. Renderocc: Vision-centric 3d occupancy prediction with 2d rendering supervision. In ICRA, 2024. 6, 7
- [41] Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In AAAI, 2024. 5, 6, 7
- [42] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. In CVPR, 2024. 2
- [43] Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In CVPR, 2024. 3, 4
- [44] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In ECCV, 2024. 3
- [45] Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu, and Sifa Zheng. Sparsedrive: End-to-end autonomous driving via sparse scene representation. In ICRA,

2025. 3, 5

- [46] Qwen Team et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 2
- [47] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. In NeurIPS, 2023. 5, 7
- [48] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024. 3

- [49] Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, et al. Scene as occupancy. In ICCV, 2023. 5, 6
- [50] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 3
- [51] Haiyang Wang, Hao Tang, Li Jiang, Shaoshuai Shi, Muhammad Ferjad Naeem, Hongsheng Li, Bernt Schiele, and Liwei Wang. Git: Towards generalist vision transformer through universal language interface. In ECCV, 2024. 2
- [52] Jiabao Wang, Zhaojiang Liu, Qiang Meng, Liujiang Yan, Ke Wang, Jie Yang, Wei Liu, Qibin Hou, and Ming-Ming Cheng. Opus: occupancy prediction using a sparse set. In NeurIPS, 2024. 7
- [53] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In CVPR,

2025. 1, 3, 5, 6

- [54] Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, et al. Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving. arXiv preprint arXiv:2312.09245, 2023. 3
- [55] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 2
- [56] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In ECCV, 2024. 2
- [57] Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for realtime autonomous driving. In CVPR, 2024. 3
- [58] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. In NeurIPS, 2025. 2
- [59] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE RAL, 2024. 3, 4
- [60] Zhenhua Xu, Yan Bai, Yujia Zhang, Zhuoling Li, Fei Xia, Kwan-Yee K Wong, Jianqiang Wang, and Hengshuang Zhao. Drivegpt4-v2: Harnessing large language model capabilities for enhanced closed-loop autonomous driving. In CVPR,

2025. 3

- [61] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2
- [62] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, 2025. 2

- [63] Senqiao Yang, Jiaming Liu, Renrui Zhang, Mingjie Pan, Ziyu Guo, Xiaoqi Li, Zehui Chen, Peng Gao, Hongsheng Li, Yandong Guo, et al. Lidar-llm: Exploring the potential of large language models for 3d lidar understanding. In AAAI,

2025. 7

- [64] Tengju Ye, Wei Jing, Chunyong Hu, Shikun Huang, Lingping Gao, Fangzhen Li, Jingke Wang, Ke Guo, Wencong Xiao, Weibo Mao, et al. Fusionad: Multi-modality fusion for prediction and planning tasks of autonomous driving. arXiv preprint arXiv:2308.01006, 2023. 1, 3
- [65] Zichen Yu, Changyong Shu, Jiajun Deng, Kangjie Lu, Liu. Zongdai, Jiangyong Yu, Dawei Yang, Hui Li, and Yan Chen. Flashocc: Fast and memory-efficient occupancy prediction via channel-to-height plugin. arXiv preprint arXiv:2311.12058, 2023. 2, 3
- [66] Chubin Zhang, Juncheng Yan, Yi Wei, Jiaxin Li, Li Liu, Yansong Tang, Yueqi Duan, and Jiwen Lu. Occnerf: Selfsupervised multi-camera occupancy prediction with neural radiance fields. arXiv preprint arXiv:2312.09243, 2023. 6
- [67] Zheng Zhang, Yeyao Ma, Enming Zhang, and Xiang Bai. Psalm: Pixelwise segmentation with large multi-modal model. In ECCV, 2024. 2
- [68] Yun Zhao, Peiru Zheng, and Zhan Gong. 3d occupancy and flow prediction based on forward view transformation. CVPR2024 Autonomous Grand Challenge Track On Occupancy and Flow, 2024. 6
- [69] Rongkun Zheng, Lu Qi, Xi Chen, Yi Wang, Kun Wang, and Hengshuang Zhao. Villa: Video reasoning segmentation with large language model. In ICCV, 2025. 2
- [70] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. In NeurIPS, 2025. 2
- [71] Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, and Alois C Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463, 2025. 1, 2, 3, 5, 6, 7
- [72] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 4

## DrivePI: Spatial-aware 4D MLLM for Unified Autonomous Driving Understanding, Perception, Prediction and Planning

### Supplementary Material

In this supplementary material, we present more ablation studies, the details of coarse-grained spatial understanding, the details of fine-grained spatial learning, and more visualizations in Sections 6, 7, 8 and 9, respectively.

- 6. More Ablation Studies Table 7. Ablation study for the balancing weights in DrivePI.

3D Occ. Occ. Flow Planning QA

# Occ. Weight Flow Weight

RayIoU↑ mAVE↓ L2↓ Col.↓ Acc.↑

- I 0.2 0.2 48.1 0.57 0.46 0.19 61.1

- II 0.5 0.5 49.3 0.54 0.49 0.40 60.9

- III 1.0 1.0 49.3 0.51 0.50 0.38 60.7

Different Balancing Weights. In DrivePI, we adopt multitask learning to train our model in an end-to-end manner. Here, we conduct experiments to investigate the effect of different loss balancing weights on overall performance. Through preliminary analysis, we observe that the 3D occupancy and occupancy flow losses dominate the total loss function, accounting for over 60% of the combined loss magnitude. To mitigate potential optimization imbalances, we systematically reduce their weights from the default value of 1.0 (III) to 0.5 (II) and 0.2 (I) for both 3D occupancy and occupancy flow tasks. The corresponding results are presented in Table 7. We find that higher occupancy and flow weights yield improved performance on 3D occupancy and occupancy flow, but result in slight degradation in planning accuracy (L2 error) and text understanding in the QA task. Therefore, setting proper balancing weights in multi-task learning remains challenging. For simplicity, we adopt the default weight of 1.0 for both 3D occupancy and flow losses in our final implementation.

Learned Weights of Different Hidden States. For the MLLM (i.e., Qwen2.5-0.5B/3B model), we investigate what is the importance weight of each hidden state of MLLMs in DrivePI? To answer this question, we conduct an analysis of the learned importance weights of all hidden states in the MLLM, as shown in Table 8. Specifically, we replace the default setting of using only the last hid-

den state with a weighted combination h = li=0 Fih · wi, where Fih ∈ RN×C represents the features of the i-th hidden state, wi ∈ R1×1 is the corresponding learnable importance weight, and l denotes the total number of hidden states in the MLLM. Here, the 0.5B (or 3B) MLLM contains 24 (or 36) transformer layers, and when including the input embeddings, we obtain a total of 25 (or 37) hidden

- Table 8. The learned importance weights of all hidden states in the MLLM with Qwen-2.5 0.5B model, including the input embedding (indexed as 0). The Index and Weight column indicates the index and the learned importance weight of each hidden state.

Index Weight Index Weight Index Weight

- 0 0.0328 10 0.0375 20 0.0463
- 1 0.0332 11 0.0381 21 0.0466
- 2 0.0337 12 0.0388 22 0.0472
- 3 0.0341 13 0.0397 23 0.0477
- 4 0.0346 14 0.0409 24 0.0468
- 5 0.0350 15 0.0422
- 6 0.0355 16 0.0435
- 7 0.0360 17 0.0449
- 8 0.0365 18 0.0455
- 9 0.0370 19 0.0458

- Table 9. The learned importance weights of all hidden states in the MLLM with Qwen-2.5 3B model, including the input embedding (indexed as 0). The Index and Weight column indicates the index and the learned importance weight of each hidden state.

Index Weight Index Weight Index Weight

- 0 0.0254 13 0.0259 26 0.0277

- 1 0.0251 14 0.0260 27 0.0283

- 2 0.0251 15 0.0261 28 0.0287

- 3 0.0252 16 0.0262 29 0.0294

- 4 0.0253 17 0.0263 30 0.0297

- 5 0.0252 18 0.0264 31 0.0303

- 6 0.0253 19 0.0265 32 0.0304

- 7 0.0254 20 0.0266 33 0.0304

- 8 0.0255 21 0.0271 34 0.0302

- 9 0.0255 22 0.0265 35 0.0303

- 10 0.0256 23 0.0251 36 0.0331

- 11 0.0257 24 0.0258

- 12 0.0258 25 0.0267

states with corresponding learnable weights. As shown in Table 8 and Table 9, we observe an overall trend where deeper layers tend to receive larger weights, indicating that higher-level features extracted by deeper transformer layers are more crucial for the effectiveness of our DrivePI.

7. Details of Coarse-grained Spatial Understanding

- Table 10 presents a variety of prompts generated by our multi-stage data engine. The data engine first produces individual captions for both front and back views, which are subsequently merged and refined into a coherent final caption. The corresponding prompts are shown as “Prompt 13” in Table 10. To enhance DrivePI’s comprehensive spatial

Table 10. An example generated by our multi-stage data engine.

[Figure 51]

- Prompt 1: Front Caption The front scene depicts a street intersection with several notable elements. On the left side of the front scene, there is a modern building with a white facade and large windows. In front of the building, there is a traffic light displaying a red signal. A parked vehicle is visible near the building, and there is a grassy area with a few trees and a sidewalk. In the center of the front scene...<omitted>
- Prompt 2: Back Caption The back scene depicts a street intersection with various elements captured from different angles. On the left side of the back scene, there is a traffic light and a construction area with a fence and some equipment, including a small vehicle or machinery. Two traffic cones are placed near the fence, indicating a possible work zone or restricted area. The background features a multi-story building with visible structural elements...<omitted>
- Prompt 3: Rewriting Caption The scene depicts a street intersection with various vehicles and elements. There are multiple vehicles, including trucks and cars, some of which are moving and others parked. Traffic cones are present in some parts of the scene, indicating possible construction or restricted areas. The area is surrounded by buildings and greenery, with traffic lights controlling the flow of vehicles. There are no visible pedestrians sitting or lying down. The setting appears to be an urban environment with infrastructure for traffic management.

- Prompt 4: 3D Occupancy and Flow prediction Your task is to predict the 3D occupancy of the scene. Assume you are located at the point (0,0,0). The scene area around you (in front, behind, left, and right) is divided into a 200×200 grid, with the bottom-left corner at (−100,−100) and the top-right corner at (100,100). The height region is divided into 16 bins. We use < OCC > (x,y,z) < /OCC > to represent the point at location (x,y) with a height of z. Assume you are located at the point < OCC > (100,100,0) < /OCC >. Answer the below question. Is the position {position} occupied? What object is occupying position < OCC > (x,y,z) < /OCC >? If there is an object, please provide its name and predict the velocity; otherwise, answer ‘free’.
- Prompt 5: Action Prediction What is the safe action of the ego car?
- Prompt 6: Trajectory Prediction Predict the future 6-frame trajectory of the ego car in the last.

understanding capabilities, we design instruction questionanswering (QA) pairs covering four core tasks: 3D occupancy prediction, flow prediction, action prediction, and trajectory prediction. For 3D occupancy and flow prediction tasks, we generate questions about the occupancy status, category, and velocity of given 3D locations using occupancy and flow ground truth. The corresponding prompts are as shown in “Prompt 4: Occupancy and Flow Prediction” in Table 10. For action prediction, we design prompts to predict subsequent driving actions, as demonstrated in “Prompt 5: Action Prediction” in Table 10. For trajectory prediction, we create prompts that guide the model to pre-

dict future ego-vehicle trajectories, as shown in “Prompt 6: Trajectory Prediction” in Table 10.

#### 8. Details of Fine-grained Spatial Learning

We provide details of task-specific heads, including 3D occupancy, flow, and action diffusion heads for fine-grained spatial learning.

Details of 3D Occupancy Head. For the 3D occupancy head, we mainly follow the previous superior method FlashOcc [65]. Specifically, given the spatial feature map Fout ∈ RH×W×C (refer to the main paper), we perform a

reshape operation along the channel dimension to transform Fout from a shape of H × W × C to H × W × Z × C′, where C′ and Z represent the channel dimension and depth dimension of the features, respectively. Then, we use an MLP to predict the category of 3D occupancy. For loss functions, we adopt the same losses as in [65], including the focal loss Lfocal, geometric loss Lgeo, semantic loss Lsem, and Lov´asz loss Llovasz.

Details of Occupancy Flow Head. We use an additional MLP to predict the velocity of 3D occupancy based on the occupancy head. To address the imbalance in the distribution of static and moving objects, we employ L1 loss and apply distinct loss weights to the flow prediction task for static and dynamic occupancy, respectively. Specifically, we assign a weight of 1.0 to each dynamic occupancy flow for computing the flow loss, while applying a smaller weight of 0.01 to static occupancy flow during training.

Details of Action Diffusion Head. We refer to previous action diffusion methods [8, 28] to implement a simple action diffusion head that predicts trajectories based on the denoising procedure. On nuScenes [5], DiffusionDrive [28] applies SparseDrive [45] to achieve plan query initialization for better trajectory planning results. In contrast, our approach directly generates trajectories without requiring SparseDrive for plan query initialization. For the loss of action diffusion head, we employ L1 loss for the action diffusion head during training.

#### 9. More Visualization

In this section, we provide additional visualizations to demonstrate the coarse-grained and fine-grained understanding capabilities of DrivePI. As shown in Table 11, DrivePI generates accurate 3D occupancy and flow predictions in complex congested environments, and successfully implements appropriate actions for stationary waiting scenarios. Furthermore, we visualize a straight-ahead driving scene in Table 12, where DrivePI demonstrates high consistency between coarse-grained and fine-grained understanding. We also present a challenging nighttime turning scenario in Table 13, which is typically difficult for VLA models. In this scene, leveraging multi-modal information, DrivePI accurately describes the environment despite the very dark imagery. Therefore, our DrivePI delivers accurate and reasonable results in this challenging low-light condition, demonstrating its robustness and effectiveness.

[Figure 52]

###### Coarse-grained Perception, Prediction, and Planning

- Task 1: Caption The scene depicts a quiet residential street lined with houses and lush greenery. The street is mostly empty, with a few parked vehicles visible in some parts. There are trees providing shade along the sidewalks, and the houses have gated entrances and fences. A bicycle is parked on the sidewalk near one of the houses. The overall atmosphere is calm and suburban, with no pedestrians or moving vehicles present.
- Task 2: Occupancy and Flow prediction Question: What can be seen at position < OCC > (83,99,5) < /OCC >? If there is an object, please provide its name and predict the velocity; otherwise, answer ‘free’ Answer: {label: car}, {vx: 0.1, vy: 0.06}
- Task 3: Action decision Question: What is the safe action of the ego car? Answer: Go straight.
- Task 4: Trajectory prediction Question: Predict the future 6-frame trajectory of the ego car in the last. Answer: Future 6-frame trajectory: [(0.12, 0.00), (0.05, 0.00), (-0.01, 0.00), (0.00, 0.00), (0.00, 0.00), (0.00, 0.00)]. Fine-grained Perception, Prediction, and Planning

[Figure 53]

[Figure 54]

[Figure 55]

Planning Occupancy Flow

Table 11. The visualization of remaining stationary with coarse-grained and fine-grained results.

[Figure 56]

###### Coarse-grained Perception, Prediction, and Planning

- Task 1: Caption The scene depicts a busy urban street with several vehicles, both parked and moving. There are multiple cars, some stopped at intersections and others in motion. Pedestrians are visible walking along the sidewalks. Construction cranes and buildings under construction are present in the background, indicating ongoing development. Traffic cones are placed on the road, likely for directing traffic or marking construction areas. The environment appears to be a bustling city area with a mix of modern architecture and active traffic.
- Task 2: Occupancy and Flow prediction Question: What can be seen at position < OCC > (159,78,4) < /OCC >? If there is an object, please provide its name and predict the velocity; otherwise, answer ‘free’ Answer: {label: car}, {vx: 0.1, vy: -0.07}
- Task 3: Action decision Question: What is the safe action of the ego car? Answer: Go straight.
- Task 4: Trajectory prediction Question: Predict the future 6-frame trajectory of the ego car in the last. Answer: Future 6-frame trajectory: [(3.42, -0.03), (3.52, -0.02), (3.59, -0.04), (3.72, -0.07), (3.87, -0.10), (3.88, -0.09)]. Fine-grained Perception, Prediction, and Planning

[Figure 57]

[Figure 58]

[Figure 59]

Planning Occupancy Flow

###### Table 12. The visualization of going straight with coarse-grained and fine-grained results.

[Figure 60]

###### Coarse-grained Perception, Prediction, and Planning Task 1: Caption

The scene depicts a quiet residential street lined with houses and lush greenery. The road is paved and has a few parked vehicles, including a white van and a car. There are trees and plants along the sidewalks, and some trash bins are visible near the houses. The area appears calm and suburban, with no pedestrians or moving vehicles in sight.

Task 2: Occupancy and Flow prediction Question: What can be seen at position < OCC > (173,102,8) < /OCC >? If there is an object, please provide its name and predict the velocity; otherwise, answer ‘free’ Answer: {label: vegetation}, {vx: 0.0, vy: 0.0} Task 3: Action decision Question: What is the safe action of the ego car? Answer: Turn left. Task 4: Trajectory prediction Question: Predict the future 6-frame trajectory of the ego car in the last. Answer: Future 6-frame trajectory: [(3.55, -0.02), (3.34, -0.25), (3.12, -0.68), (2.98, -1.17), (2.69, -1.74), (2.47, -2.18)].

###### Fine-grained Perception, Prediction, and Planning

[Figure 61]

[Figure 62]

[Figure 63]

Planning Occupancy Flow

Table 13. The visualization of turning with coarse-grained and fine-grained results.

