### LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences

# arXiv:2412.01292v2[cs.CV]2Feb2025

Hongyan Zhi1* Peihao Chen2* Junyan Li4* Shuailei Ma3 Xinyu Sun1 Tianhang Xiang1 Yinjie Lei7 Mingkui Tan16† Chuang Gan45

1South China University of Technology, 2Tencent Robotics X, 3Northeastern University, 4UMass Amherst, 5MIT-IBM Watson AI Lab, 6Pazhou Laboratory, 7Sichuan University

#### Abstract

Research on 3D Vision-Language Models (3D-VLMs) is gaining increasing attention, which is crucial for developing embodied AI within 3D scenes, such as visual navigation and embodied question answering. Due to the high density of visual features, especially in large 3D scenes, accurately locating task-relevant visual information is challenging. Existing works attempt to segment all objects and consider their features as scene representations. However, these task-agnostic object features include much redundant information and missing details for the task-relevant area. To tackle these problems, we propose LSceneLLM, an adaptive framework that automatically identifies taskrelevant areas by leveraging LLM’s visual preference for different tasks, followed by a plug-and-play scene magnifier module to capture fine-grained details in focused areas. Specifically, a dense token selector examines the attention map of LLM to identify visual preferences for the instruction input. It then magnifies fine-grained details of the focusing area. An adaptive self-attention module is leveraged to fuse the coarse-grained and selected finegrained visual information. To comprehensively evaluate the large scene understanding ability of 3D-VLMs, we further introduce a cross-room understanding benchmark, XRScene, which contains a series of large scene understanding tasks including XR-QA, XR-EmbodiedPlanning, and XRSceneCaption. Experiments show that our method surpasses existing methods on both large scene understanding and existing scene understanding benchmarks. Plunging our scene magnifier module into the existing 3D-VLMs also brings significant improvement. Code and data are available at https://github.com/Hoyyyaard/LSceneLLM

*Equal contribution. Email: hoyard1212@gmail.com †Corresponding author. Email: mingkuitan@scut.edu.cn

(a)

[Figure 1]

- (b)

(c)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Down Sample

Dense Token Selector

Black and White

[Figure 13]

Ours

coarse scene understand module

scene magnifier module

[Figure 14]

Low Computation

[Figure 15]

[Figure 16]

Visual Preference

[Figure 17]

LLM

[Figure 18]

Task–related Details

[Figure 19]

[Figure 20]

[Figure 21]

LSceneLLM Chat3D-v2 Leo

OccLLaMA Chat-Scene LL3DA Lidar-LLM 3D-LLM

[Figure 22]

Figure 1. We propose LSceneLLM, a novel framework for adaptive large 3D scene understanding. (a) Existing methods struggle to locate task-relevant visual information when facing large scenes. (b) We are committed to precisely identifying finegrain task-related visual features through adaptive scene modeling.

- (c) Our method outperforms existing approaches across various benchmarks.

#### 1. Introduction

3D scene understanding is essential for various tasks, including robotic manipulation [19,48], navigation [35], and embodied long-horizon planning [7, 22]. With the rapid development of large language models (LLMs) [3,10], researchers are increasingly focusing on leveraging LLMs’ impressive reasoning and summarization capabilities to enhance the understanding of 3D point clouds, building 3D Vision-Language Models (3D-VLMs) [8,15,17,18,36,44].

Most efforts on 3D-VLMs have concentrated on objectlevel point cloud understanding [14, 36, 44], while scenelevel understanding remains underexplored due to the larger and more intricate nature of 3D environments compared to individual objects. To introduce the scene-level understanding ability to 3D-VLMs, existing works [8,16–18,41] seek help from object detection [9] and instance segmentation [32, 47] technique to delineate all objects within a scene. All the object features extracted by an object-level 3D

features extractor are considered to represent the scene-level 3D features, as illustrated in Fig. 1(a). Despite the advancements, existing 3D-VLMs face two major limitations: 1) When facing large scenes, the scale of task-related visual information is significantly smaller than that of the entire scene. This disparity poses a significant challenge for existing 3D-VLMs in accurately focusing on pertinent visual information, as they require all segmented objects as input, which is task agnostic. 2) To balance the computational load, existing approaches utilize sparse object point clouds as input to capture object features within an entire scene, which leads to the loss of critical details related to small objects. LL3DA [8] attempts to address the above issues by employing a Qformer [23] to query the task-relevant scene features. However, it fails to fully utilize the substantial reasoning capabilities of LLMs to assist in selecting taskrelevant visual information and is hindered by the limitations of low-resolution scene features. Moreover, Current 3DVLM are predominantly benchmarked on the datasets that are annotated in ScanNet [11], 3RScan [40], etc., which mainly consists of single-room [2,6,40]. The understanding of large scenes such as multi-room scenarios remains underexplored.

When faced with complex visual input, humans selectively focus on certain regions first, then search for information within those relevant areas [31,38,43]. Mimicking the mechanisms humans use to process complex visual input, we posit that a model can first obtain visual preference within sparse scene representation, followed by a fine-grained analysis of focus areas. It’s similar to how people, when reading a bulletin board, first focus on a specific topic before paying closer attention to particular details.

Inspired by the above motivation, we propose LSceneLLM, a 3D-VLM framework that contains a coarse scene understanding module and a scene magnifier module for adaptive modeling of large scenes. As shown in Fig. 1(b), to obtain a preliminary understanding of the various areas of the scene, we utilize a scene encoder to encode the downsampled point cloud. Additionally, the scene magnifier module is proposed to identify the visual preference of LLM while extracting and fusing selected detailed visual information. Specifically, a dense token selector leverages the attention map of LLM to identify the visual preferences relevant to the instruction and then collects dense visual features from the areas of interest, which is then fused with coarse-grained scene information through the adaptive self-attention module, enabling large-scale scene understanding with limited point cloud input. The scene magnifier module can be easily inserted into existing 3D-VLMs by replacing the corresponding self-attention module in the LLM.

To provide a comprehensive evaluation for large scene understanding, we additionally propose a cross-room understanding benchmark XR-Scene, which includes XR-QA,

XR-EmbodiedPlanning, and XR-SceneCaption. It has an average scene area size of 132 m2, which is significantly larger than the 29 m2 in ScanQA [2]. Our experimental results demonstrate that LSceneLLM achieves state-of-the-art performance on a wide range of 3D tasks and benchmarks, including single-room scene benchmarks and large scene benchmarks, as shown in Fig. 1(c). Our contributions are summarized as follows:

- • We present LSceneLLM, a 3D-VLM framework that automatically identifies and magnifies detailed information in task-relevant areas. This helps the model accurately localize the important information within the large 3D scene.
- • To comprehensively benchmark 3D-VLMs in large scene understanding, we present the XR-Scene, a collection of cross-room understanding tasks that includes question-answering, embodied planning, and scene caption, with an average scene area approximately four times larger than that of ScanQA, which offers a more challenging evaluation environment.
- • Our approach consistently demonstrates superior performance in both indoor and outdoor large-scene understanding benchmarks. Integrating our scene magnifier module with existing 3D-VLMs also brings significant improvement.

#### 2. Related Work

##### 2.1. 3D Vision Language Models

3D-VLMs have recently developed rapidly, thanks to the emergence of excellent work in LLMs. Researchers initially focused on understanding 3D object point clouds [14,36,44]. PointLLM [44] first leverages the pretraining and instruction tuning paradigm to empower LLM to understand object point clouds. MiniGPT-3D [36] introduced an efficient 3D-VLM that aligns 3D point clouds with LLMs by leveraging 2D priors from 2D-LLMs, capitalizing on the similarities between 2D and 3D visual information. Point-Bind [14] constructs a joint embedding space that integrates 3D and multimodal data, projecting 3D semantic features into LLMs to enhance their capacity for 3D object-level question answering. Scenelevel point clouds are inherently more complex than objectlevel point clouds due to the high density of visual features. This complexity poses a greater challenge for LLMs to understand scene-level point clouds effectively. Object-centric works [16–18, 41] first leverage instance segmentation or object detection to extract all objects within the scene and model the spatial relationship between objects through a relation module. When facing large scenes, the scale of task-related visual information is significantly smaller than that of the objects features within the scene, as it involves

Cross-Room Scene Caption

[Figure 23]

[Figure 24]

[Figure 25]

LSceneLLM: Adaptive Framework For Large Scene Understanding

Question

[Figure 26]

[Figure 27]

Describe the dining area.

Projector

[Figure 28]

LSceneLLM

Downsample

LLM

The first room appears to be a kitchen or dining area, equipped with kitchen cabinets, a refrigerator, a dining table, and multiple chairs. This room is well-suited for cooking, dining, and socializing.

The dining area is equipped with kitchen cabinets, a refrigerator, a dining table, and multiple chairs. This room is well-suited for cooking, dining, and socializing.

Dense Vision Token Seletor

FFN

FFN

Scene Magnifier Module

[Figure 29]

xNMA What color is the monitor？

Cross-Room Scene Embodied Planning

[Figure 30]

Attention Map

[Figure 31]

Question

[Figure 32]

I want to create a cozy reading nook in the minimalistic room. I have done these things: 1. selected a comfortable chair. What should i do next?

FFN

Tokenizer

Sample & Group

xNSA

Adaptive SelfAttention

SelfAttention

"scan_id": "00823-7MXmsvcQjpJ", "room_id": "sub012-sub013-sub014-sub019"

[Figure 33]

LSceneLLM

2. Place a small table next to the chair. 3. Place a lamp on the table. 4. Add a few books on the table. 5. Arrange a pillow on the chair for added comfort.

[Figure 34]

Sample Groups

Text Tokens

Sparse Vision Tokens

Dense Vision Tokens

Selection Process

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Cross-Room Scene Question-Answering

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Outdoor Scene QuestionAnswering

[Figure 52]

###### Question

###### Question

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Where is the candle in relation to the fireplace ?

[Figure 57]

[Figure 58]

What color is the faucet?

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Question

[Figure 66]

[Figure 67]

[Figure 68]

LSceneLLM

LSceneLLM

How many moving pedestrians are to the back of me?

[Figure 69]

[Figure 70]

[Figure 71]

On the left side of the fireplace.

Silver.

[Figure 72]

LSceneLLM

One.

- Figure 2. An Overview of LSceneLLM. LSceneLLM first perceives the scene through sparse vision tokens at the coarse level and then enhances regions of interest using dense vision tokens. Our method can effectively handle various visual language tasks in large scenes.

querying a single object among hundreds of others. This challenges the task-agnostic visual feature construction process, making it difficult for the model to locate task-relevant information accurately. A recent work, LL3DA [8], proposed using QFormer [23] to extract visual features related to task instructions. It employs a set of learnable queries to summarize the object features in the scene that are relevant to the task. However, this process occurs outside the LLM and is a shallow task-related information extraction process that demands a substantial amount of training data [46]. We propose to leverage the LLM’s powerful reasoning capabilities to select visual preferences based on different instructions.

large scene understanding benchmark based on HM3D for cross-room scenarios.

#### 3. LSceneLLM: Adaptive Framework For Large Scene Understanding

When facing large scenes, to precisely identify taskrelevant visual information within high-density visual contexts, we introduce LSceneLLM, which automatically identifies task-relevant areas by leveraging LLM’s visual preference and then searching for the desired information within the focus areas. The adaptive framework consists of a coarse scene understanding module and a scene magnifier module, allowing for the comprehension of scenes from coarsegrained overviews to fine-grained details of significant regions. The scene magnifier module can be seamlessly integrated into most 3D-VLMs by simply replacing their corresponding self-attention modules in LLM.

##### 2.2. 3D Understanding Benchmarks

Early datasets for 3D understanding, such as NYUv2 [33] and SUN RGB-D [34], comprise short RGBD sequences with low resolution and limited annotations. ScanNet [11] is the first dataset that provides 3D reconstructions and annotations at scale, including 1,201 and 312 different and complex indoor 3D scenes. Most existing indoor scene understanding benchmarks [1,2,6] are built from ScanNet. In addition, benchmarks [18] based on scenes like 3RScan [40] and ARKitScenes [5] are also widely discussed. Despite the above advancements, existing 3D understanding benchmarks are largely limited to small scenes, such as single-room scenarios, and benchmarks for large scene understanding in cross-room and outdoor environments [29] remain underexplored. HM3D [30] is the largest-ever dataset of 3D indoor spaces, consisting of 1,000 high-resolution 3D scans of building-scale residential, commercial, and civic spaces generated from real-world environments. We construct a

##### 3.1. Overall Architecture

As shown in Fig. 2, given the dense point cloud of a 3D scene, we first obtain dense point cloud features through a scene encoder [27]. These features are then down-sampled to sparse point cloud features. Through a coarse scene understanding module modified from SA module [28] that random sample points and groups a wide range of nearby features for each sample point, the sparse point cloud features are converted into sparse vision tokens, representing a coarse scene depiction. For fine-grain feature construction, we identify the center points of a preferred region and extract a specific number of point cloud features from the dense point cloud features in its vicinity. These local region point

[Figure 73]

[Figure 74]

[Figure 75]

Adaptive Self-attention Module of Layeri-1……

Text Tokens

Dense Vision Tokens

Sparse Vision Tokens

Dense Vision Token Selector

Adaptive Self-attention Module of Layeri

[Figure 76]

Text Tokens

[Figure 77]

[Figure 78]

[Figure 79]

Key Value

Share

……

Q u e r y

K

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | |Att|ent|ion|We|igh|t|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

Dense Pointcloud

Q

Sparse Vision Tokens

Dense Vision Tokens

| | | | | | |Attention Map<br><br>Threshold >|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 80]

Matrix Multiplication

T

Hidden State

Sparse Pointcloud

Dense Vision Token Selector

Last Text Token

(a) Adaptive Self-attention Module (b) Dense Vision Token Selector

- Figure 3. Illustration of Adaptive Self-attention Module and Dense Vision Token Selector. We first obtain the focused regions by analyzing the attention map of LLM. Then we extract dense point cloud features from the region of interest and parse dense vision tokens through sampling and grouping operations.

cloud features go through the SA module [28] to obtain the dense vision tokens. After receiving the visual features, we modify the LLM with our proposed scene magnifier module to achieve a fine-grained scene perception. Given an LLM with N layers of self-attention, we replace the last NMA layers’ self-attention modules with our scene magnifier module, while keeping the first NSA layers unchanged because the first few layers of attention tend to focus on the global information of the vision input. To reduce computational complexity, we input only the sparse vision tokens and text tokens into the first NSA layers. For the last NMA layers which utilize the scene magnifier module, we additionally incorporate selected dense vision tokens to enhance the model’s understanding of regions of interest in greater detail. Specifically, the scene magnifier module comprises two sub-modules: a dense vision token selector and an adaptive self-attention mechanism. The dense vision token selector dynamically identifies and selects dense vision tokens for regions of interest guided by the attention map from the selfattention mechanism, rather than inputting all dense vision tokens. The adaptive self-attention mechanism integrates the information from the selected dense vision tokens into the original hidden states, thereby enriching the model’s contextual understanding.

weights of the last text token to all sparse vision tokens and normalize them to a range of 0 to 255. Next, we select 10% to 20% sparse vision tokens whose weights exceed a specified threshold. For each selected sparse vision token, we identify the corresponding region in the scene and sample dense vision tokens from that region to provide richer visual information, thereby enhancing the model’s understanding of the scene.

##### 3.3. Adaptive Self-attention Module

The adaptive Self-attention Module is a crucial mechanism that integrates dense vision tokens D with the hidden state H containing text tokens and sparse vision tokens. As shown in Fig. 3(a), this module takes dense vision tokens and the hidden state as input and outputs the fused hidden state. Unlike standard self-attention, our attention map captures interactions between the hidden state and dense vision tokens. We eliminate the attention map for this interaction component, and the remaining attention map is used to select dense visual tokens in the subsequent layer. It is important to note that each text token interacts solely with the selected dense vision tokens, which we achieve through the use of an attention mask. The calculation of the adaptive self-attention is summarized as follows:

Q = HWQ Kall = Concat(HWK,DW′

##### 3.2. Dense Vision Tokens Selector

K) Vall = Concat(HWV ,DW′

In an auto-regressive model, the prediction of the next token is contingent upon the hidden state of the last token processed. As shown in Fig. 3(b), by analyzing the activation values of the attention map for the last token to the sparse vision tokens, we can identify specific visual information that the model focuses on when making predictions. Although sparse vision tokens provide limited information, we enhance the model’s understanding by retrieving dense point cloud features in focused areas. We first extract the attention

V ) Adaptive Self-attention(H,D) = softmax

QKallT √dk

Vall

where WQ,WK,WV ,WK′ ,WV′ ∈ RD×d are learnable linear projection matrices. Kall and Vall are the key and value matrices that aggregate information from dense point cloud features.

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

(a) Sub-Regions Generation (c) Question-Related Object Counts

(b) Area Size Comparison

[Figure 88]

XR-EmbodiedPlanning

[Figure 89]

[Figure 90]

Question: I want to set up a cozy reading corner. what should i do? Answers: 1. go to the room with the single lamp 2. place a chair near the lamp 3. add a table next to the chair 4. place a few books on the table 5. switch on the lamp to create a warm ambiance

XR-QA

XR-SceneCaption

Question: Where is the black-framed window? Answer: On the wall above the cluttered pile.

Question: Describe the scene. Answers: The multi-room scene includes a living room, a bedroom, a utility/storage room, a

[Figure 91]

[Figure 92]

minimalistic room with a lamp, and a bathroom. The living room …… The …… The utility/storage room contains two vents ……

Question: Describe the living room. Answers: The living room is well-equipped with various furniture including cabinets, chairs, pillows, a TV, tables, and stools. Additional objects include smoke detectors, lamps ……

Question: What is the color of the dishwasher? Answer: Silver.

XR-QA

(d) Dataset Visualization

- Figure 4. Examples of dataset XR-Scene. XR-Scene contains three cross-room scene benchmarks that comprehensively evaluate different understanding abilities.

- Table 1. 3D large scene understanding results. All use Ll3da and XR-Scene data for training. ∗ means do not identify the question-related objects for the model. # means requiring images and point clouds as input.

XR-QA XR-SceneCaption XR-EmbodiedPlanning CIDEr METEOR ROUGE CIDEr METEOR ROUGE CIDEr METEOR ROUGE

Methods

Zero-Shot Chat-Scene# [16] 69.55 26.63 10.06 0.01 5.94 1.52 32.64 20.71 10.26 Leo∗ [18] 55.40 22.71 6.96 0.02 1.92 2.92 9.74 16.84 6.88 Ll3da [8] 24.78 12.66 5.31 0.12 8.71 5.14 7.02 15.21 7.17

###### Finetuning

Chat-Scene# [16] 114.10 35.93 14.32 3.58 17.49 11.59 46.18 22.34 36.71 Leo∗ [18] 112.09 35.47 14.02 2.42 15.96 10.25 39.45 18.99 33.31 Ll3da [8] 112.80 36.94 18.68 3.22 20.95 13.49 35.96 15.74 31.50 LSceneLLM(Ours) 117.21 38.18 19.30 4.59 23.43 16.16 63.08 22.97 36.96

#### 4. XR-Scene: Cross-Room Scene Understanding Benchmark

sive evaluation of large scene understanding, we propose a cross-room understanding benchmark XR-Scene, which includes three tasks, XR-QA, XR-EmbodiedPlanning, and XR-SceneCaption, as shown in Fig. 4(d).

Current benchmarks for 3D-VLMs primarily benchmark on the datasets that are annotated in ScanNet [11], etc., which mainly consists of single-room scenes, such as ScanQA [2] and ScanRefer [6]. Benchmarking in cross-room scenes remains underexplored. Cross-room scenes exhibit higher spatial complexity and object diversity, posing a challenge for 3D-VLMs to comprehend complex scenes with limited scene point cloud inputs. To conduct a comprehen-

We generate cross-room scenes from HM3D [30] using ground-truth room positions, selecting the nearest N rooms to form each scene. Object annotations from SceneVerse [20] are used to assist in the generation of different types of QA pairs. We prompt GPT-4o with annotations from each room and a top-down view of the whole scene to generate XRScene. XR-QA necessitates that 3D-VLM initially iden-

- Table 2. 3D question answering results on the ScanQA [2] validation dataset. ∗ means do not identify the question-related objects for the model.

Method LLM Training Data ROUGE METEOR CIDEr

3D-VLP [21] - - 34.51 13.53 66.97 ScanQA [2] - - 33.33 13.14 64.86 Chat3D [41] Vicuna-7b - 28.5 11.9 53.2 Chat3D-v2 [17] Vicuna-7b 204k 40.1 16.1 77.1 3D-LLM [15] BLIP2-flanT5 675k 35.7 14.5 69.4 SceneLLM [12] Llama2-7b 690k 35.9 15.8 80.00 Chat-Scene [16] Vicuna-7b 145k 37.79 15.94 77.75 Leo∗ [18] Vicuna-7b 1034k+145k 40.24 16.68 80.20 Ll3da [8] Opt-1.3b 145k 37.02 15.37 75.67 Ll3da [8] Llama2-7b 145k 38.31 15.91 79.08 LSceneLLM(Ours) Llama2-7b 145k 40.82 17.95 88.24

tifies the specific region within a vast scene to which the object associated with the query pertains. Furthermore, it must comprehend the relationships between this object and its surrounding entities to deliver precise responses. XREmbodiedPlanning requires 3D-VLM to comprehend complex inter-object and room relationships to produce subtasks based on high-level goals. Meanwhile, XR-SceneCaption challenges 3D-VLM to generate comprehensive scene descriptions and captions for specific rooms while inferring attributes based on present objects. Overall, these tasks demand advanced spatial reasoning and contextual awareness to navigate the intricacies of cross-room environments effectively. For more detailed information on dataset generation, please refer to Appendix A.

- 4.1. Analysis of XR-Scene

XR-Scene consists of more than 1000 unique scenes, with an average area of 132m2, significantly larger than the average scene area of 29m2 in ScanQA [2], as mentioned in Fig. 4(b). Moreover, the number of inquiries about objects in the scene of XR-QA is relatively balanced, whereas for ScanQA [2], there is a heavy focus on asking numerous questions about chairs, and tables, resulting in a lack of diversity, as shown in Fig. 4(c). Furthermore, due to the visual density in large scenes, which often require downsampling and can result in the loss of scene details, we selected a subset from XR-QA based on the bounding box sizes of the question-specific objects. This subset, XR-QAS, has a bounding box size threshold of 0.05 m2, equivalent to the size of a keyboard. This subset assesses the model’s ability to comprehend fine-grained information in various environments.

- 5. Experiment

- 5.1. Datasets, Metrics and Implementation Details

Datasets. In this paper, we conduct experiments using 3D data from ScanNet [11] and HM3D [30]. The annotated training data is sourced from ScanRefer [6], Nr3D [1], ScanQA [2], the ScanNet subset of 3D-LLM [15] and our

XR-Scene. The annotations cover various 3D tasks such as object captions, scene descriptions, scene question answering, and task planning. We also conduct experiments on the outdoor large scenes QA benchmark, Nuscenes-QA [29], to validate whether our method effectively handles sparse and wide-ranging point clouds. Due to the huge gap between indoor and outdoor scenes, we tested with additional models, trained with Nuscenes-QA data.

Metrics. Here, we adopt CiDEr [39], METEOR [4], Rouge [24] and accuracy to evaluate the quality of the generated textual responses.

Implementation Details. Following previous works on 3D vision language tasks [8], we randomly sample 40k point clouds from each 3D scene as the 3D visual input. We use the LLama2-7b [37] as our causal LLM backbone, which is fully fine-tuned to enable the model to attend to the areas of interest accurately. We adopt the AdamW [26] optimizer with a weight decay of 0.1 and a learning rate decaying from 10−5 to 10−6 with a cosine annealing scheduler. In XRScene, We select Leo [18], Chat-Scene [16] and Ll3da [8] as the typical methods. We train all baselines with the same data in order to conduct a fair comparison.

##### 5.2. Comparison with SoTA Specialists

###### 5.2.1 Indoor Scene Understanding

We benchmark existing methods on XR-Scene to evaluate the model’s understanding of large scenes, as shown in Tab. 1. We first evaluate existing methods in a zero-shot manner, and experiment results indicate that those models perform poorly without having undergone a learning process on large scene data. For question-answering task XR-QA, when facing high-density vision input, the performance of the Ll3da [8] is poor due to the suboptimal process of extracting task-related visual information and detail loss in feature compressing process. Object-centric methods like Leo [18] and ChatScene [16] also perform poorly due to the overwhelming number of objects in the scene, which hinder the model’s ability to focus on the relevant objects precisely. For XRSceneCaption and XR-EmbodiedPlanning, which require a deep understanding of the relationships between regions (rooms) and objects to generate captions for specific areas or to complete high-level planning tasks within a region, existing methods yielded unsatisfactory results. This is because they modeled the scene only at the object level, neglecting the dependency information between regions and objects within the broader context of the scene. On the contrary, our proposed method, LSceneLLM, can identify task-relevant regions and facilitate a fine-grained understanding within focus areas. As a result, our approach outperforms existing methods across all three tasks by a large margin.

We also evaluate our method on ScanQA [2], as shown in Tab. 2. The experimental results indicate when facing

Table 3. 3D question answering results on outdoor scene benchmark NuscenesQA [29]. * means downstream specialist model.

Exist Count Object Status Comparison

Method

Acc

H0 H1 All H0 H1 All H0 H1 All H0 H1 All H0 H1 All NuscenesQA* [29] 87.7 81.1 84.1 21.9 20.7 21.3 70.2 45.6 49.2 62.8 52.4 55.9 81.6 68.0 69.2 58.1 LLaVA-Adaptaer-v2 [13] 34.2 6.3 19.3 5.0 0.1 2.7 23.7 4.6 7.6 9.8 11.3 10.8 2.6 1.5 1.6 9.6 LLaVA [25] 38.9 51.9 45.8 7.7 7.6 7.7 10.5 7.4 7.8 7.0 9.9 9.0 64.5 50.8 52.1 26.2 LidarLLM [45] 79.1 70.6 74.5 15.3 14.7 15.0 59.6 34.1 37.8 53.4 42.0 45.9 67.0 57.0 57.8 48.6 OccLLaMA [42] 80.6 79.3 79.9 18.6 19.1 18.9 64.9 39.0 42.8 48.0 49.6 49.1 80.6 63.7 65.2 53.4 LSceneLLM(Ours) 86.4 81.3 83.6 19.4 19.8 19.6 64.4 41.3 44.8 58.8 51.2 53.8 81.0 67.5 68.7 56.4

- Table 4. More results on the XR-QA validation dataset and challenge subset XR-QA-S. # We re-implement Leo [18] and Ll3da [8] keeping all other settings the same as ours to conduct a fair and further comparison.

Method Scene Magnifier Module

XR-QA XR-QA-S

ROUGE METEOR CIDEr ROUGE METEOR CIDEr Leo# [18] ✗ 36.56 18.61 110.33 36.10 18.06 103.16 Leo# [18] ✓ 37.53(+0.97) 19.00(+0.39) 113.46(+3.13) 36.88(+0.77) 18.47(+0.41) 107.56(+5.29) Ll3da# [8] ✗ 37.19 18.51 111.35 36.04 17.61 95.65 Ll3da# [8] ✓ 37.85(+0.65) 19.15(+0.56) 115.79(+4.44) 37.23(+1.19) 18.60(+0.99) 106.73(+11.09) LSceneLLM ✗ 36.58 18.65 109.92 35.47 17.91 97.57 LSceneLLM(Ours) ✓ 38.18(+1.60) 19.30(+0.65) 117.21(+7.29) 38.15(+2.68) 18.69(+0.78) 109.42(+11.85)

- Table 5. Ablation studies. ATR: the activate token ratio of sparse vision tokens. #: do not use the scene magnifier module.

image input or specialized framework design, with an increase of 3.0 in accuracy compared to the previous best method.

Parameter ROUGE METEOR CIDEr

96(AT: 10%-20%) 38.18 19.30 117.21 127(AT: 3%-5%) 37.89 19.26 115.92 64(AT: 40%-50%) 37.68 19.07 114.69

##### 5.3. More Insights into Fine-Grain Large Scene Understanding

Threshold

2 37.91 19.14 115.32 4 38.18 19.30 117.21 6 37.54 19.03 115.14

To reveal the capability of 3D-VLMs to understand small objects in scenes, we further conducted a more in-depth analysis of existing methods on XR-QA and XR-QA-S, as shown in Tab. 4. We re-implement Leo [18] and Ll3da [8] to keep all other settings the same as ours to conduct a fair comparison. The performance of existing methods on XR-QA-S has significantly declined compared to XR-QA. This is due to the loss of details during downsampling required by object recognition techniques, making it difficult to perceive small objects in the scene accurately. Our proposed LSceneLLM can automatically complete fine-grained visual information and performs best on XR-QA-S, outperforming the existing methods on CIDEr by a large margin.

Dense Token Num

Attention Map 38.18 19.30 117.21 Random 37.64 19.18 115.66

Select Strategy

512 37.27 18.80 112.19 128 36.58 18.65 109.92

Vision Token Num

128# 38.18 19.30 117.21

small-scale scenes, our method can also achieve optimal performance with limited training data, further validating the effectiveness of the hierarchical scene understanding approach we proposed. More ScanNet understanding results can be found in Appendix C.

###### 5.2.2 Outdoor Scene Understanding

Outdoor scenes are larger in scale compared to indoor scenes, and the visual information is more sparse, which poses greater challenges for 3D-VLMs in understanding outdoor sparse point clouds. We also conducted experiments on the NuscenesQA [29] benchmark to assess the model’s capability in handling large outdoor scenes. Results in Tab. 3 show that our LSceneLLM achieves state-of-the-art performance across all generative methods without requiring multi-view

##### 5.4. Module Plug-and-Play Analysis

To further verify that the scene magnifier module can be easily integrated into most existing 3D-VLM frameworks to enhance fine-grain understanding, we integrated it with two major architectures Leo [18] and Ll3da [8], as shown in Tab. 4. For the Ll3da framework, adding the scene magnifier module resulted in a 4.44 improvement in XR-QA and an 11.09 improvement in XR-QA-S. In comparison, the improvement was less significant compared to ours. We attribute this to the fact that the use of modules that compress

High Low

[Figure 93]

[Figure 94]

Target Object LSceneLLM Ll3da# Leo#

| |
|---|

[Figure 95]

[Figure 96]

[Figure 97]

###### Question

[Figure 98]

[Figure 99]

[Figure 100]

Where is the magazine located?

[Figure 101]

LSceneLLM

[Figure 102]

On the table below the window.

[Figure 103]

###### GT

On the dinner table, close to the window.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

###### Question

Where is the table located in the room?

[Figure 112]

LSceneLLM

Focus on query object actually

Focus on irrelevant objects

In the middle.

[Figure 113]

###### GT

In the center.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

###### Question

[Figure 119]

What is the purpose of the lamp?

Fail to locate small item

[Figure 120]

[Figure 121]

LSceneLLM

For lighting.

[Figure 122]

GT

Used for lighting.

Figure 5. Visualization of attention map of LLM. Red represents high activation values, while blue represents low activation values.

visual information like Qformer [23] has resulted in the loss of visual details. Leo models the entire scene directly at the object level, so the improvement after adding the scene magnifier module is less compared to the other two frameworks due to the absence of the coarse scene understanding module.

cation of the magazine which is on the table in the scene, LSceneLLM accurately identifies the magazine’s position and correctly focuses on the table mentioned in the response. The other two methods only roughly focus on the magazine’s location and output the wrong answers. When asked about smaller objects (the third row), LSceneLLM is also able to accurately locate the position of the small objects, while the other two methods fail to do so.

##### 5.5. Ablation Studies

We conduct several ablation studies, including the number of vision tokens, the dense vision token selection strategy, the selection threshold of attention value, and the number of dense vision tokens that interact with sparse vision tokens using the XR-QA dataset, as shown in Tab. 5. We experimentally found that using the attention map from LLM to identify the areas of interest and activating 10%-20% of these areas yields the best performance for our method. For more detailed information, please refer to Appendix B.

#### 6. Conclusion

In this paper, we investigate the paradigm of 3D-VLMs for large 3D scene understanding. To precisely locate taskrelated visual information, we propose an adaptive framework that automatically identifies task-relevant areas by leveraging LLM’s visual preference when tackling different tasks, followed by a plug-and-play scene magnifier module to capture fine-grained details in focused areas. Experimental results demonstrate that our approach achieves significant performance improvements on large 3D scene understanding benchmarks. We further propose the XR-Scene cross-room understanding benchmark to complete the benchmarking process for 3D-VLMs in large-scale environments. We hope that our findings will inspire further advancements in the development of large 3D scene understanding methodologies.

##### 5.6. Qualitative Analysis

Visualization of Attention Map We explore the areas the model focuses on when answering questions by visualizing the attention maps of the generated sequence of LLM to scene vision tokens, as shown in Fig. 5. Experiments were conducted using LSceneLLM and two other commonly used 3D-VLM frameworks Ll3da [8] and Leo [18]. As illustrated by the example in the first row, when asked about the lo-

#### References

- [1] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pages 422–440. Springer, 2020.
- [2] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19129– 19139, 2022.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [4] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pages 65–72, 2005.
- [5] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021.
- [6] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In European conference on computer vision, pages 202–221. Springer, 2020.
- [7] Peihao Chen, Xinyu Sun, Hongyan Zhi, Runhao Zeng, Thomas H Li, Gaowen Liu, Mingkui Tan, and Chuang Gan. A2nav: Action-aware zero-shot robot navigation by exploiting vision-and-language ability of foundation models. arXiv preprint arXiv:2308.07997, 2023.
- [8] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26428–26438, 2024.
- [9] Sijin Chen, Hongyuan Zhu, Xin Chen, Yinjie Lei, Gang Yu, and Tao Chen. End-to-end 3d dense captioning with vote2capdetr. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11124–11133, 2023.
- [10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023.
- [11] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet:

- Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.
- [12] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-llm: Extending language model for 3d visual understanding and reasoning. arXiv preprint arXiv:2403.11401, 2024.
- [13] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023.
- [14] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, et al. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615, 2023.
- [15] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023.
- [16] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, and Zhou Zhao. Chat-scene: Bridging 3d scene and large language models with object identifiers, 2024.
- [17] Haifeng Huang, Zehan Wang, Rongjie Huang, Luping Liu, Xize Cheng, Yang Zhao, Tao Jin, and Zhou Zhao. Chat-3d v2: Bridging 3d scene and large language models with object identifiers. arXiv preprint arXiv:2312.08168, 2023.
- [18] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023.
- [19] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.
- [20] Baoxiong Jia, Yixin Chen, Huangyue Yu, Yan Wang, Xuesong Niu, Tengyu Liu, Qing Li, and Siyuan Huang. Sceneverse: Scaling 3d vision-language learning for grounded scene understanding. arXiv preprint arXiv:2401.09340, 2024.
- [21] Zhao Jin, Munawar Hayat, Yuwei Yang, Yulan Guo, and Yinjie Lei. Context-aware alignment and mutual masking for 3d-language pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10984–10994, 2023.
- [22] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16, pages 104–

120. Springer, 2020.

- [23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen

- image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.
- [24] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004.
- [25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
- [26] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [27] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 815–824, 2023.
- [28] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017.
- [29] Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4542–4550, 2024.
- [30] Santhosh K Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alex Clegg, John Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, et al. Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238, 2021.
- [31] Rajesh PN Rao, Gregory J Zelinsky, Mary M Hayhoe, and Dana H Ballard. Eye movements in iconic visual search. Vision research, 42(11):1447–1463, 2002.
- [32] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3d: Mask transformer for 3d semantic instance segmentation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 8216–8223. IEEE, 2023.
- [33] Nathan Silberman and Rob Fergus. Indoor scene segmentation using a structured light sensor. In 2011 IEEE international conference on computer vision workshops (ICCV workshops), pages 601–608. IEEE, 2011.
- [34] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 567–576, 2015.
- [35] Xinyu Sun, Peihao Chen, Jugang Fan, Jian Chen, Thomas Li, and Mingkui Tan. Fgprompt: fine-grained goal prompting for image-goal navigation. Advances in Neural Information Processing Systems, 36, 2024.
- [36] Yuan Tang, Xu Han, Xianzhi Li, Qiao Yu, Yixue Hao, Long Hu, and Min Chen. Minigpt-3d: Efficiently aligning 3d point clouds with large language models using 2d priors. arXiv preprint arXiv:2405.01413, 2024.

- [37] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [38] Anne M Treisman and Garry Gelade. A feature-integration theory of attention. Cognitive psychology, 12(1):97–136, 1980.
- [39] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575, 2015.
- [40] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner. Rio: 3d object instance re-localization in changing indoor environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7658–7667, 2019.
- [41] Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, and Zhou Zhao. Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes. arXiv preprint arXiv:2308.08769, 2023.
- [42] Julong Wei, Shanshuai Yuan, Pengfei Li, Qingda Hu, Zhongxue Gan, and Wenchao Ding. Occllama: An occupancy-language-action generative world model for autonomous driving. arXiv preprint arXiv:2409.03272, 2024.
- [43] Jeremy M Wolfe and Todd S Horowitz. What attributes guide the deployment of visual attention and how do they do it? Nature reviews neuroscience, 5(6):495–501, 2004.
- [44] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. arXiv preprint arXiv:2308.16911, 2023.
- [45] Senqiao Yang, Jiaming Liu, Ray Zhang, Mingjie Pan, Zoey Guo, Xiaoqi Li, Zehui Chen, Peng Gao, Yandong Guo, and Shanghang Zhang. Lidar-llm: Exploring the potential of large language models for 3d lidar understanding. arXiv preprint arXiv:2312.14074, 2023.
- [46] Linli Yao, Lei Li, Shuhuai Ren, Lean Wang, Yuanxin Liu, Xu Sun, and Lu Hou. Deco: Decoupling token compression from semantic abstraction in multimodal large language models. arXiv preprint arXiv:2405.20985, 2024.
- [47] Bo Zhang, Jiakang Yuan, Botian Shi, Tao Chen, Yikang Li, and Yu Qiao. Uni3d: A unified baseline for multi-dataset 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9253– 9262, 2023.
- [48] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024.

## APPENDIX

#### A. More Details on Generation of XR-Scene

Generation of Cross-Room Scenes HM3D [30] contains several cross-room, multi-floor 3D scenes. In SceneVerse [20], annotations are generated for each room in the HM3D scenes, including object properties and spatial relationships with surrounding objects. As shown in Fig. 4(a). For a given scene, we leverage the ground-true central positions of each room in HM3D. We randomly sample one room and calculate the Euclidean distances between other rooms, the nearest N rooms are selected to form a cross-room scene.

XR-QA Generation For each cross-room scene containing N rooms, we retrieve object annotations from SceneVerse [20] for these N rooms and filter out objects that appear exactly once in the scene to ensure uniqueness corresponding to the question. For each annotated object, we use GPT-4 to generate two types of questions: object properties and spatial relationships with surrounding objects based on the annotations.

XR-Planning and XR-EmbodiedPlanning Generation The embodied planning task requires the model to understand the objects in the scene and their specific locations. Given a high-level task, the model needs to use the objects in the scene to generate a series of subtasks. In contrast to single-room scenes, embodied planning in cross-room scenes is more complex for the model, as it needs to understand the relationships between objects and the rooms, not just the relationships between the objects themselves. The scene captioning task requires the model to provide a general description of the current scene, including the relationships between objects and their attributes. In larger scenes, scene captioning demands a stronger spatial understanding of the model. The model not only needs to perceive the positional relationships between objects but also pay attention to the areas to which the objects belong. Our tasks will include generating captions for the entire large scene as well as requiring the model to caption only a specific room. Furthermore, the model needs to infer room attributes based on the objects present, making scene captioning in cross-scene Scenes more challenging than in single-room Scenes. We generate the top-down view of the cross-room scene and use bounding boxes to specify that a certain annotation corresponds to a specific room. Follow Leo [18], We use prompt engineering to guide GPT-4o in understanding the scene and generating scene captions and QA pairs for embodied planning. Additionally, we provide the model with a real RGB-rendered top-down view of the scene to further reduce model hallucinations, as shown in Fig. 6.

Table 6. Ablation studies of selection threshold

Threshold Activate Token Ratio ROUGE METEOR CIDEr

64 40% - 50% 37.68 19.07 114.69 96 10% - 20% 38.18 19.30 117.21

127 3% - 5% 37.89 19.26 115.92

Table 7. Ablation studies of the number of vision tokens

Vision Token Num Scene Magnifier Module ROUGE METEOR CIDEr

512 ✗ 37.27 18.80 112.89 128 ✗ 36.58 18.65 109.92 128 ✓ 38.18 19.30 117.21

Table 8. Ablation studies of dense token

Dense Token Num ROUGE METEOR CIDEr

2 37.91 19.14 115.32 4 38.18 19.30 117.21 6 37.54 19.03 115.14

Table 9. Ablation studies of selection strategies

Select Strategy ROUGE METEOR CIDEr Attention Map 38.18 19.30 117.21

Random 37.64 19.18 115.66

#### B. Ablation Study

Selection Threshold of Attention Weight. We also explored the threshold for the confidence of text tokens to vision tokens in the attention map. We normalized the attention weight of a text token to all vision tokens to a range of 0-255. The experimental results show that the model performs best when the chosen threshold is 96, meaning 10%-20% of the vision tokens are selected to interact with the corresponding fine-grained scene features. If too many tokens are selected, the model cannot accurately focus on the local areas, while if too few tokens are selected, the finegrained scene information provided is insufficient, offering limited help in understanding the scene, as shown in Tab. 6.

Numbers of Dense Vision Token Interact With Sparse Vision Token. This ablation experiment investigates the optimal number of dense vision tokens with which each sparse vision token should interact. We sample a certain number of point cloud features around the center point of the sparse vision token from the dense point cloud features and then aggregate them. As shown in Tab. 8, using 4 dense vision tokens to represent the fine-grained features of a local region provides the greatest benefit to the model.

The number of Vision Tokens We first explored whether sampling more visual information from the environment would improve the model’s performance. As shown in Tab. 7,

[Figure 123]

[Figure 124]

In this room, there are two vents, two cabinets, one hose, and two potted plants. The room appears to be well-equipped with

In this room, there is a solitary bathtub, standing as the sole object of interest. Its presence suggests a space dedicated to relaxation and cleansing. The room's purpose is clear……

ventilation and storage options. The presence of a hose suggests the possibility of cleaning or watering tasks……

In this room, there is a sink, nine cabinets, a kitchen counter, an oven, two mugs, three books, four hoses, four pictures, eight lamps, a TV, eight chairs, five potted plants, a refrigerator, a rug, four tables, four stools, four vents, and a light switch……

In this room, there are 4 pillows, 2 lamps, 1 decoration, 2 cabinets, 1 picture, and 2 storage boxes. The pillows are aligned with each other and are lower than the lamps. The lamps are higher than the cabinets and the pillows. One lamp is below another lamp……

Large scene caption

The scene features multiple distinct areas. One area is a restful bedroom with a bed …… The

<top-down-view-image> Using the provided scene realistic top-down view and room

practical and organized area contains pictures, boxes for storage …… a minimalistic area illuminated by a single lamp exudes calm and simplicity, offering a peaceful environment.

caption per room, design a high-level task that can be performed

in this 3D scene. Besides, decomposing this high-level task into a sequence of action steps that can be performed using the

[Figure 125]

instances in this 3D scene.(You need to provide a whole summary for the multi-room scene. This summary should contain a rough summary of what rooms are in the whole scene......)

Large scene embodied planning

Question: i want to set up a cozy reading corner. what should i do? Answers: 1. go to the room with the single lamp 2. place a chair near the lamp 3. add a table next to the chair 4. place a few books on the table 5. switch on the lamp to create a warm ambiance

Please strictly follow the rules below:<rule> Room Captions:<caption per room> or <object list> Example: <example> You need to generate 5 question-answer pairs follow example

strictlly in json format and re-thinkng the rules before output.

###### Figure 6. Generation pipeline of XR-SceneCaption and XR-EmbodiedPlanning.

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

What color are the tables? Brown.

Where is the clutter in relation to the stairs? On left side of stairs.

What is the color of the toilet paper? White.

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

[Figure 150]

Where is the curtain in relation to the window? Left.

What color is the bed? White.

What color is the faucet? Silver.

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

What is the shape of the bed? Rectangular.

What color is the toilet? White.

Where is the table in relation to the_door? To right of door.

Figure 7. More Attention Visualization of LSceneLLM.

Table 10. More 3D scene understanding results. ∗ means do not identify the question-related objects for the model.

Scene Caption Embodied Planning Embodied QA ROUGE CIDEr METEOR ROUGE CIDEr METEOR ROUGE CIDEr METEOR

Method

Leo* [18] 1.80 20.84 13.29 46.40 204.78 19.86 30.89 86.14 18.81 Chat-Scene [16] 3.67 21.05 12.60 40.03 210.86 20.71 34.23 99.01 18.48 Ll3da [8] 1.44 24.62 12.93 45.34 186.13 19.60 33.75 95.53 19.81 LSceneLLM(Ours) 3.07 21.88 14.79 47.05 214.63 21.05 36.00 104.98 21.26

Table 11. Computational complexity results on XR-QA

Method Scene Magnifier Module Vision Token Num Flops CIDEr Leo ✗ 200 6.55 110.33

Ll3da ✗ 32 4.11 111.35 LSceneLLM ✗ 128 5.3 109.92 LSceneLLM ✓ 128 6.33 117.21

although using four times the number of vision tokens does lead to some performance improvement, the enhancement is not as significant as the improvement achieved by incorporating the LSceneLLM module, which validates the efficiency of our approach.

Dense Vision Token Selection Strategy. We conducted ablation experiments to verify that the attention map in the selfattention module reflects the visual information the model focuses on when answering questions. As shown in Tab. 9, the selection strategy based on attention weight outperforms the random selection strategy, demonstrating that the information about the regions that the model is currently focusing on aids in understanding the scene, while the random selection strategy provides little benefit to the model.

#### C. More Scene Understanding Results on ScanNet

We also test our method on scene caption, embodied planning, and embodied qa, these datasets are sourced from the ScanNet part of 3D-LLM [15] and organized by Ll3da [8]. Embodied QA requires the model to answer questions from the perspective of an agent, considering the agent’s position and orientation within the environment. All of these tasks demand the model to have a holistic understanding of the entire scene. As shown in Tab. 10, our method outperforms the current state-of-the-art approaches on most metrics, demonstrating that the proposed approach not only captures fine-grained details in the scene but also achieves an accurate overall understanding of the entire scene.

#### D. More Attention Visualization of LSceneLLM on XR-QA

We provide more attention map visualization results when LSceneLLM deals with different instructions on XR-QA.

Experiment results show that our proposed method can accurately locate the task-relevant visual features using adaptive visual preferences from LLM.

#### E. Computational Complexity Analysis

We analyze the computational complexity of the Ll3da [8], Leo [18], and our method when faced with the same scene and identical text input. In Leo, the number of visual tokens corresponds to the number of objects in the scene, while both the Ll3da and our method use a fixed number of visual tokens when dealing with scenes of varying sizes, as shown in Tab. 11. The computational complexity of our method is situated between the two baseline methods. As the scene size increases further, LSceneLLM can maintain a constant computational complexity while preserving the scene details. In contrast, the computational complexity of the object-centric method increases with the growing number of objects in the scene.

