# arXiv:2506.06205v1[cs.RO]6Jun2025

[Figure 1]

## Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning

ByteDance Seed Full author list in Contributions

### Abstract

Modern robot navigation systems encounter difficulties in diverse and complex indoor environments. Traditional approaches rely on multiple modules with small models or rule-based systems and thus lack adaptability to new environments. To address this, we developed Astra, a comprehensive dualmodel architecture, Astra-Global and Astra-Local, for mobile robot navigation. Astra-Global, a multimodal LLM, processes vision and language inputs to perform self and goal localization using a hybrid topological-semantic graph as the global map, and outperforms traditional visual place recognition methods. Astra-Local, a multitask network, handles local path planning and odometry estimation. Its 4D spatial-temporal encoder, trained through self-supervised learning, generates robust 4D features for downstream tasks. The planning head utilizes flow matching and a novel masked ESDF loss to minimize collision risks for generating local trajectories, and the odometry head integrates multi-sensor inputs via a transformer encoder to predict the relative pose of the robot. Deployed on real in-house mobile robots, Astra achieves high end-to-end mission success rate across diverse indoor environments.

Date: June 6, 2025 Project Page: https://astra-mobility.github.io/

### 1 Introduction

Modern robotic systems face increasing demands for adaptive navigation in diverse and complex environments. In a known environment, navigation can be decomposed into three major challenges. Goal Localization: in certain applications, instead of directly providing a pose for navigation goal, the goal may be specified by natural language or goal image prompt. In these cases, we need a system to understand the prompt and localize the goal within the map. Self-Localization: the robot needs to localize itself in the map. In complex scenarios like warehouses, the environment is highly repetitive with few global landmarks. Traditional navigation system often needs to rely on artificial landmarks like QR code. Path Planning: path planning can be divided into global planning and local planning. Global planning generates a coarse route based on the robot pose and goal pose. Given way points along global path, local path planning is responsible for reaching intermediate way points while avoiding obstacles.

To address goal localization, self-localization, and path planning tasks, traditional navigation systems generally consist of multiple modules, e.g., localization, perception, prediction, planning, and control. Different modules often contain multiple small models or rule-based systems. In recent years, the emergence of foundation models has spurred a trend to integrate small models into larger ones to solve more tasks. However, the question of how many models are necessary remains unanswered. In this report, we propose Astra, a dual-model architecture that tackles the above three navigation problems in diverse indoor environments. Astra adheres to the System 1/System

###### 2 philosophy [1], where Astra-Global is responsible for low frequency tasks such as goal and self-localization, and

1

[Figure 2]

- Figure 1 Astra addresses three key navigation challenges: goal localization, self-localization, and path planning, with the holistic integration of Astra-Global and Astra-Local. Astra-Global is responsible for goal and self-localization. For the goal

localization, it locates the landmark and the corresponding goal pose pG from the map based on user text prompts. For self-localization, Astra-Global identifies visual landmarks from images and then integrates this information with the odometry estimated by Astra-Local through multi-sensor fusion to obtain the robot’s global pose pi. Meanwhile, Astra-Local takes an additional subgoal gi as input for path planning and generates a local path for the robot to follow.

Astra-Local manages high frequency tasks including local path planning and odometry estimation.

Astra-Global is a multimodal LLM (MLLM) responsible for self and goal localization within a global map. We represent the known environment as a topological-semantic map and use it directly as context input for the MLLM. Given this map representation along with a query image or text prompt, the model locates the query within the map. Whether the query is a text prompt from a user, such as ‘I’d like to find somewhere to rest’, in which case the model locates the goal in the map, or an image the robot currently perceives, in which case it locates the robot itself, Astra-Global can handle both cases. Unlike traditional mobile robot global localization methods that often rely on artificial landmarks like QR codes or additional sensors in complex scenes, Astra-Global leverages natural human built landmarks and functions effectively in diverse environments.

On the other hand, Astra-Local is a multi-task network tasked for local path planning and odometry estimation. The model takes sensor inputs (e.g., multi-view images, IMU, wheel), a local goal, and other robot states as inputs. Based on these, it plans a local path and estimates the robot’s odometry. Astra-Local features a 4D spatial-temporal encoder that fuses images from a multi-camera setup across multiple frames. Pretrained on a large scale datasets, the encoder is capable of generating a coherent 4D spatial-temporal representation and predicting future representations. The encoder is connected to two task heads. The planning head, taking additional inputs such as the local goal and robot states, employs flow matching to generate a local path that guides the robot towards the local goal while avoiding obstacles. The odometry head uses a transformer to fuse vision features with other sensor inputs to estimate robot relative movement.

Our key contributions include:

- • A novel dual-model system Astra that effectively solves key mobile navigation problems, as depicted in Fig. 1. We developed and deployed Astra to our in-house built mobile robots and tested across different indoor environments including warehouse, office building, house. In all environments, Astra achieves high end-to-end mission success rate from locating the user query to navigating to the goal safely.

- • Astra-Global, a MLLM designed to handle low frequency tasks in mobile navigation, effectively addresses both goal and self-localization within a single model. Built upon a foundation MLLM, we train Astra-Global using Supervised Finetuning (SFT) and Reinforcement Learning (RL). Astra-Global outperforms traditional Visual Place Recognition (VPR) methods across all environments and works zero-shot in new environments. Our experiments demonstrate that RL is an effective approach for enhancing the model’s generalization capabilities and is more data efficient than using SFT alone.
- • Astra-Local is a multi-task network designed for high frequency tasks in mobile navigation, such as odometry estimation and local path planning. Its shared encoder, pretrained on large scale datasets in a self-supervised manner, provides robust spatial-temporal visual features and outperforms state-of-the-art (SOTA) methods in downstream tasks like occupancy forecasting. The odometry head, which fuses camera, IMU, wheel data within a novel transformer framework, has been shown by experimental results to be effective in multi-sensor data fusion. The planning head employs flow matching for local path planning. When trained with a novel masked ESDF loss function, it achieves a significant reduction in the collision rate.

The remainder of this report is organized as follows: Section 3 elaborates on the details of Astra, including the overall architecture, model design, and training procedures for both Astra-Global and Astra-Local. Section 4 presents the experimental results for the entire system and each model. Section 2 reviews related studies, and we conclude in Section 5.

### 2 Related Work

#### 2.1 Global Localization

The goal of visual-based global localization, a critical component of visual navigation, is to determine the position of the current image within a known scene. Traditional approaches like Visual Place Recognition [2–4] rely on image retrieval to match the current scene with the most similar image in a pre-built map, offering robustness to environmental and seasonal changes but suffering from limitations such as loss of image details in global descriptions—leading to poor performance in repetitive scenes with minor differences—and an inability to directly output camera pose as they focus on retrieval rather than pose estimation. In contrast, end-to-end (E2E) methods have emerged to directly estimate ego pose from sensor inputs, leveraging architectures that bypass complex geometric calculations (e.g., PixLoc [5], BEV-Locator [6], EgoVM [7], MapLocNet [8]). While these E2E approaches streamline the pipeline and enable direct pose output, they often struggle with generalization and performance degradation under significant scene and season variations, highlighting the need for more robust strategies to balance accuracy and adaptability across diverse environments.

Recently, the rapid advancement of Vision-Language Models (VLMs) [9–11], powered by large-scale pretraining on billions of image-text pairs, has revolutionized cross-modal understanding in robotics domain. While existing VLM research focuses on navigation [12, 13] and manipulation [14, 15], their application to visual localization remains underexplored. Works like MobilityVLA [16] use long-context VLMs for topological localization but struggle with low precision due to semantic-geometric mismatches. Our Astra-Global addresses this by introducing a two-stage localization framework paired with a rigorously constructed multi-scenario dataset, enhancing both accuracy and real-world adaptability.

#### 2.2 Odometry Estimation

Traditional multi-sensor fusion odometry methods typically rely on probabilistic frameworks like Kalman filters, particle filters, or factor graph optimization to integrate data from LiDAR, cameras, and IMU, etc. Existing multi-sensor fusion odometry frameworks, such as filtering-based approaches exemplified by R3LIVE [17] and factor graph optimization methods like LVI-SAM [18], demonstrate satisfactory accuracy under nominal operating conditions. However, these methodologies predominantly rely on handcrafted feature extraction and incorporate simplified assumptions in uncertainty modeling. Such limitations hinder their capacity to fully characterize the intrinsic sensor observation properties, leading to performance degradation in edge cases such as geometrically degenerate environments or adverse weather conditions.

Recent advancements in deep learning have catalyzed paradigm shifts in state estimation. While single-modality odometry solutions (e.g., DROID-SLAM [19], DPVO [20], BEV-ODOM [21], RoNIN [22]) demonstrate domain-

specific competence, they exhibit limitations in accuracy and robustness compared to multi-sensor counterparts which effectively exploit complementary sensor characteristics. Transformer-based architectures, in particular, have gained prominence due to their superior cross-temporal correlation modeling and cross-modal interaction capabilities, such as TransFusionOdom [23] and VIFT [24]. The integration of temporal dynamics with multi-modal fusion presents a promising direction for advancing odometry systems. Therefore, our method integrated extended multi-view and multi-modal temporal data to achieve superior accuracy and robustness.

#### 2.3 End-to-End Planning

End-to-end planning offers a compelling vision for mobile robot navigation, as well as autonomous driving and robot manipulation, by simpler and more adaptive systems. From early demonstrations [25] to recent innovations incorporating transformers, diffusion models and LLMs [26–29], the field has made significant strides. Industry efforts, such as Tesla’s Full Self-Driving (FSD) [30], leverage end-to-end learning to process vast datasets, aiming for scalable performance [31]. However, since most approaches apply imitation learning that relies on massive expert data, challenges like out-of-distribution problems, system robustness and interpretability of results remain, requiring ongoing research to ensure practical deployment [32]. Comparing with existing methods, our proposed masked ESDF loss, can significantly reduce collision rates while preserving high modality of trajectories.

#### 2.4 3D and 4D Encoders

In vision-based 3D occupancy prediction, camera inputs are utilized to predict the occupancy status, offering a cost-effective alternative to LiDAR-based systems [33–35]. Recently, self-supervised learning for occupancy gained widespread attention. Self-supervised learning addresses the data scarcity challenge by enabling models to learn from unlabeled data, reducing the need for costly manual annotations. The volume rendering technique is generally adopted, which back-projects the 3D voxel volume into the 2D image space (represented by depth) [36, 37]. Then, the rendered depth can be optimized using photometric consistency or temporal coherence [38, 39].

4D occupancy forecasting extends 3D occupancy prediction into the temporal domain, predicting how the environment will evolve over time. In [40], a GPT-like spatial-temporal generative transformer is utilized to generate subsequent scene and ego tokens, which are decoded into the future occupancy and ego trajectory simultaneously. [41] predicts future occupancy and flow, conditioned on ego-vehicle actions like velocity and steering angle, and integrates with end-to-end planning. In [42], a benchmark for camera-only 4D occupancy forecasting is proposed. It evaluates sequential occupancy states and 3D backward centripetal flow, highlighting challenges in long-term prediction. Different from these methods, our proposed 4D Spatial-Temporal encoder in Astra-Local does not require 3D semantic labels, which is a highly cost-effective solution.

### 3 Approach

We now formally formulate the mobile navigation problem. We consider a wheeled mobile robot operating in a pre-mapped environment, for which one or more demonstration tour videos are available. Given an instruction I in the form of a text prompt from the user, the robot’s task is two-fold. First, it needs to locate the goal. Then, it must navigate towards the goal while avoiding any obstacles in its path.

Fig. 1 illustrates the overall navigation process with Astra. Given a demonstration tour of the indoor environment, we first build a map G offline. When a user provides a prompt, Astra-Global searches the map G for landmarks that match the prompt and sets the corresponding pose as the navigation goal. For self-localization, Astra-Global performs low frequency global localization of the robot in the map using robot’s onboard sensor data, specifically images. Meanwhile, Astra-Local fuses images with other sensors such as IMU, wheel to predict the local relative pose at a high frequency. By combining these two, we obtain high frequency localization results. This mirrors human behavior in most environments, where we use sparse landmarks for global positioning and estimate our current location through dead-reckoning, calculating relative movements (odometry) from our last known position. In the path planning task, a simple search-based global path planner generates a route to the goal and selects a local subgoal based on the robot’s current pose. Astra-Local then uses sensor data to generate a local path to the subgoal while avoiding obstacles. Algorithm 1 shows the main navigation loop.

Algorithm 1 Astra for mobile navigation Require: Map G, user instruction I, robot sensor data O

Locate goal pose pG from I and robot pose p0 with Astra Global Plan a global trajectory Tr while pG is not reached do

Select a subgoal gi from Tr based on pi−1 Plan a local trajectory to the gi using Astra-Local Update pi with Astra-Local and Astra-Global

end while

#### 3.1 Astra-Global

Astra-Global focuses on two tasks: self-localization and goal localization. Traditional indoor localization methods either depend on artificial landmarks such as QR codes or require a complex map-building process, making the entire system difficult to deploy or adapt to new environments. In contrast, humans often localize themselves in complex scenes using high-level semantic landmarks. Inspired by this and the fact that self and goal localization only need to operate at a low frequency, we design Astra-Global as a MLLM as shown in Fig. 2. Thanks to recent advancements, MLLMs have excellent scene understanding and grounding capabilities and can handle multimodal inputs naturally. We first create a comprehensive map that includes geometric poses, visual landmarks, and connectivity constraints. This map provides the foundation for the robot to understand the scene. By using the map as context input for Astra-Global, it enables vision-language localization for prior-free self-localization and language-based goal localization. Next, we introduce the offline mapping module.

[Figure 3]

- Figure 2 Astra-Global follows most modern MLLMs like [9] where images are encoded via a separate visual encoder and further aligned with text tokens with a projector. Map is represented as a combination of images and texts depending on localization stage. The encoded vision tokens and text tokens are fed into a LLM to generate the final results.

##### 3.1.1 Offline Mapping

Map is an important prior knowledge the robot can leverage in an environment. From the perspective of an agentic framework, the map functions as an integral part of the robot’s memory which allows the robot to efficiently determine object positions, accurately localize itself, and plan paths efficiently. Similar to [16], we assume a demonstration tour video is given for the environment where the video can be taken from the robot or any other device.

We propose an offline approach to construct a hybrid topological-semantic graph G = (V,E,L) as our map representation that integrates geometric poses, visual landmarks, and connectivity constraints. Here, V represents the set of nodes in the graph. E is the set of undirected edges established between nodes based on their relative pose relationships, which is crucial for global path planning. The set L holds the landmark information. For each visual landmark in the environment, L records landmark details and all the node IDs where the landmark appears, effectively creating a centralized registry of its spatial occurrences. This allows the robot to utilize these landmarks for more accurate localization and navigation.

The map construction process has three parts. Topological map construction sets up V and defines E based on relative poses for navigation. Landmark semantic enrichment extracts landmark details from nodes in V to enhance semantic understanding. Landmark co-visibility graph construction perform cross-frame analysis to identify shared landmark features among multiple nodes, ensuring semantic consistency across the map.

[Figure 4]

- Figure 3 Hybrid topological-semantic map structure. Nodes encode camera poses and landmark references; edges represent geometric connectivity; landmarks store semantic attributes and link to multiple nodes via co-visibility relationships.

Topological Map Construction: The input video first undergoes temporal downsampling to reduce redundancy while preserving structural continuity. We employ [43], an off-the-shelf structure-from-motion pipeline, to estimate approximate 6-Degree-of-Freedom (DoF) camera poses T ∈ SE(3) for each processed frame. The keyframes selected subsequently serve as nodes V in the hybrid map. Undirected edges E are established between nodes based on their relative pose relationships, thereby enabling global path planning and related navigation tasks.

Landmark Semantic Enrichment: Astra-Global is employed to extract semantic landmarks from each node’s visual data, thereby enriching the map with high-level environmental semantics. For a given node vi, Astra-Global combines linguistic guidance and image to identify a set of landmarks Li = {li,1,li,2,...,li,k}, where each landmark li,m ∈ L is characterized by the following attributes:

- • Object/Text Category: Semantic labels such as "sofa," "A-001," or "door", providing fundamental entity identification;
- • Visual Attributes: Color (e.g., "gray," "brown"), material (e.g., "fabric," "wood") or background (e.g. white wall, glass door), capturing perceptual characteristics;
- • Functional Description: Natural-language annotations describing usage purposes (e.g., "for resting in living areas", "for document storage"), supporting language-based goal localization tasks.

These landmark attributes are primarily represented in textual form. The first two attribute categories (object/text and visual attributes) are utilized across all localization tasks, while functional descriptions are specifically designed for language-based goal localization tasks.

Landmark Co-Visibility Graph Construction: To ensure semantic consistency across the map, we use Astra-Global to perform cross-frame analysis to identify shared landmarks features visible from multiple nodes. When the same landmark (e.g., a gray sofa in a living room for resting) is detected in nodes vi and vj, a bidirectional co-visibility relationship is established through two operations:

- • Landmark-to-Node Association: The landmark’s entry in L is updated to include all node IDs where it appears (e.g., nodes = [node1,node2,node3]), creating a centralized registry of its spatial occurrences;
- • Node-to-Landmark Reference: Each node vi records the landmark’s unique ID in its landmarks field, forming a many-to-many association that records which landmarks are visible from each node.

This mechanism enables spatial relationship inference between non-adjacent nodes - for instance, determining navigable paths between rooms through shared landmarks like "door", even without direct geometric connections.

The resulting hybrid map integrates both geometric navigability (through topological edges) and semantic understanding (via landmark annotations), establishing a comprehensive prior knowledge for robotic navigation. As illustrated in Fig. 3, this dual-representation architecture empowers the robot to: (1) localize itself or goal based on either language or image prompt, and (2) compute optimal navigation paths through the topological graph, thereby effectively bridging the semantic gap between high-level user instructions and low-level physical movement.

##### 3.1.2 Self & Goal Localization

Given the map defined above, Astra-Global supports multi-modal localization requests, addressing two core tasks: vision-language localization and language-based goal localization as shown in Fig. 2.

Visual-Language Localization: Astra-Global estimates an input image’s pose in the map via a coarse-to-fine two-stage process. Coarse localization shrinks the pose search space, and then fine localization refines the result for higher-precision localization.

In the coarse localization stage, the input to Astra-Global consists of query image, localization prompt, and the prebuilt landmark map. It analyzes the input image and takes into account the localization prompt to detect landmarks and establish correspondences with the pre-built landmark map. The model outputs detected landmarks in the query image : Lquery = {lj|lj = (type,color or background)} together with the matched landmarks Lmatched ⊆ L. In this step, the model mainly focuses on semantic matching that considers both categorical alignment (e.g., "sofa" ↔ "couch") and attribute consistency (color/texture/background similarity). For example, a query for a "gray sofa" retrieves map nodes containing corresponding "light-gray couch" entries via the Astra-Global’s semantic reasoning.

However, the landmark results obtained above are merely matching results based on the text descriptions of landmarks. To further optimize the coarse localization process, it is crucial to also consider image similarity. We term this step the visual consistency filtering. Specifically Astra-Global determines whether there is a co-visible area between a candidate image corresponding to each landmark and the query image. Landmarks without a co-visible area are filtered out. This additional step refines the set of matched landmarks Lfiltered, ensuring that the results

are more accurate and relevant to the actual visual content of the query image, thus enhancing the effectiveness of the coarse localization process.

The final candidate nodes are determined by:

{vi|vi ↔ lk}.

Vcandidate =

lk∈Lfiltered

In the fine localization stage, the goal is to leverage the query image and Vcandidate output from the coarse localization to get a more precise localization result. To bridge the gap between coarse candidate regions and accurate pose estimation, we sample reference map nodes Vref from offline map G that are in close proximity with Vcandidate. Proximity is simply measured by the Euclidean distance between the location and the difference between the pose. Vref serves as anchor points that allows for more detailed comparison with the query for better localization accuracy.

With Vref established, we feed both the query image and reference nodes (each containing an image and its known pose) to Astra-Global which directly outputs the predicted pose of the query image by leveraging the combined visual and positional information from the reference nodes. The prediction is formulated as:

pˆ = Astra-Global(Iquery,Vref). (1)

Language-based Goal Localization: In this task, the overarching goal is to retrieve the image of the target location within the map G = (V,E,L) that corresponds to the given natural language instruction. As landmarks in our map G already contains functional description, it’s straightforward to use Astra-Global to identify the relevant landmarks l ∈ L that satisfy the query language instruction. Subsequently, through the landmark-to-node association mechanism, we can locate the relevant nodes. The nodes, in turn, provide the images and 6-DoF poses Tv ∈ SE(3) that contain the desired goal.

To ensure efficient retrieval, the system implements a spatial partitioning strategy that divides the search space S into sub-regions {S1,S2,...,Sn}. Formally, using a Euclidean distance metric d(·,·) in 3D space, the retrieval process first focuses on landmarks l within sub - region Sj satisfying ∀p ∈ Sj,d(p,pr) ≤ r, where r represents a predefined search radius. Initially, the system gives priority to searching the area near the current location. If the target location image is not found in this nearby area, the search will then be expanded to the surrounding regions. This process enables the system to narrow down the potential locations within the map, with the ultimate goal of accurately identifying the image of the target location as specified by the user’s natural language input.

##### 3.1.3 Model Training

Astra-Global leverages Qwen2.5-VL [9] as its backbone, combining supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO)[44] to specialize the model for localization tasks while retaining its general multimodal capabilities.

In the SFT stage, we prepare diverse datasets with different tasks to fine-tune the model. Besides coarse and fine localization datasets as the main tasks, we also constructed a series of auxiliary tasks to help improve model’s spatial understanding including:

- • Co-Visibility Detection: given two images, the model must determine whether a target image and a reference image share a co-visible region.
- • Selection of Co-Visible Image: the model needs to identify the image from several candidate images that has co-visibility with the query image.
- • Estimation of Movement Trend: given two images, the model needs to infer the relative movement trend from a reference image to a target image.

These auxilary tasks fosters the model’s general understanding of spatial relationships and task constraints, enabling it to derive actionable decisions through sequential reasoning rather than direct input-output mapping. Notably,

all datasets are engineered to minimize annotation overhead: data collection relies solely on image-pose pairs, a standard output of robotic SLAM systems or open-source datasets.

Following the SFT stage, we employ GRPO to train the model for visual-language localization tasks, leveraging rule-based reward functions tailored to coarse localization subtasks. To facilitate the training of reinforcement fine-tuning, the model formats the obtained results from the coarse localization process into a predefined, structured output. This format is specifically designed to simplify the calculation of the reward. The reward function Rcourse consists of four normalized components.

Rcoarse = Rformat + Rlandmark + Rmap + Rextra, where each reward is defined as follows: Format Reward (Rformat): Checks if the output adheres to the predefined format:

Rformat =

1 if format is valid, 0 otherwise.

Landmark Extraction Reward (Rlandmark): Rewards accurate extraction of semantic landmarks by matching against ground-truth landmarks Glandmark:

Rlandmark = |Plandmark ∩ Glandmark| |Glandmark|

,

where Plandmark is the set of predicted landmarks. Map Matching Reward (Rmap): Computed as the Intersection-over-Union (IoU) between predicted and ground-truth landmark ids:

Rmap = |idspred ∩ idsgt| |idspred ∪ idsgt|

.

Extra Landmark Reward (Rextra): Rewards novel landmarks that are not in the ground truth but correct, weighted by their pose error to query image’s ground truth pose.

Rextra = exp(−λ(wd · d(ppred,pgt) + wθ · |ϕpred − ϕgt|)), where d(.,.) is the Euclidean distance, wd + wθ = 1 are used to balance position and angular errors.

While landmark-based coarse retrieval effectively narrows down candidate locations using semantic landmarks (e.g., objects and visual attributes), this process may inadvertently discard fine-grained visual features that are crucial for precise localization. To mitigate this limitation, we introduce a visual consistency filtering mechanism that evaluates co-visibility relationships between query images and retrieved landmark candidates. This step ensures that only candidates exhibiting both semantic relevance and visual consistency are advanced to the fine localization stage.

For the visual consistency filtering method, the reward combines two components: format consistency and co-visibility score consistency. The reward of co-visibility score consistency is computed as:

Rcovis = 1 − |Sgt − Spred|, (2) where:

- • Sgt is the ground truth co-visibility score
- • Spred is the predicted co-visibility score from model inference

The total reward for visual consistency filtering is then calculated as:

Rtotal = Rformat + λRcovis, (3)

where λ is a weighting hyperparameter balancing the importance between format consistency and co-visibility consistency.

#### 3.2 Astra-Local

[Figure 5]

Figure 4 Model Architecture of Astra-Local. The multi-view images first go through a 3D spatial encoder to get a voxel feature VfT for current frame. Combined with voxel features from previous frames and query embeddings for future frames,

- 4D Temporal Encoder predicts voxel features for future timestamps. The Odometry head incorporates the current and prior voxel features along with additional sensor data. it leverages a transformer encode to fuse multiple modalities and outputs a relative pose. The planning head takes all predicted voxel features with local goal and other robot states as inputs and formulates local path planning as conditional flow matching.

Astra-Local is a multi-task network that generates local paths and estimates odometry from sensor data in an end-to-end fashion. As shown in Fig. 4, the architecture of Astra-Local comprises three main components. First, the 4D Spatial-Temporal encoder processes multi-frame, multi-camera inputs to generate 4D features aligned with real-world coordinates, which serve as the foundation for downstream tasks. Second, the Planning Head takes these 4D features along with local goal and robot status and generates executable trajectories for the robot to follow. Third, the Odometry Head uses the same 4D features with additional sensor inputs to estimate the relative pose between the current frame and previous frames.

- 3.2.1 4D Spatial-Temporal Encoder

In traditional mobility stack, perception and prediction modules serve as critical components by enabling agents to fuse information from the past and acquire both current and future environmental states. In Astra-Local, we propose a unified 4D spatial-temporal encoder to supplant the two distinct modules. The 3D spatial encoder is trained at first to generate common representations through vast amounts of unlabeled data via a self-supervised learning paradigm. Then, the 4D spatial-temporal encoder is trained to forecast future environmental representations on top of the 3D encoder.

###### 3D Spatial Encoder: Given N surround-view images, the 3D encoder M3D is designed to encode them into geometric

- 3D representations:

N

- i
- Ii,Ki,Ti), (4)

Vf = M3D(

where Ii, Ki, and Ti correspond to the ith image and its intrinsic and extrinsic. V denotes the voxel-level 3D representations.

Specifically, we employ a Vision Transformer (ViT) [45] to encode input images into discriminative feature representations Fi ∈ Rh×w×C from the ith image Ii ∈ RH×W×3. Then, we use lift-splat-shoot [46] to convert 2D image features into 3D voxel features:

N

Vf = P(

i

π(Fi ⊗ Ddisti ,Ki,Ti)), (5)

where Ddisti represents the estimated depth distribution. π refers to the projection which transforms a 2D image pixel into a 3D point using the camera intrinsic K and extrinsic T. The function P denotes voxel pooling.

We train the 3D Spatial Encoder in a self-supervised learning manner and is achieved via 3D volumetric differentiable neural rendering where the depth and color image are rendered from Vf. Specifically, to render the depth image, we use a multi-layer perceptron (MLP) [47] to transform the voxel feature Vf into a signed distance field (SDF) Vs. Then, given a set of rays r consisted of camera origins and view directions, the depth values are computed through a weighted integration along the rays. The weighting coefficients are derived from the opacity and the accumulated transmittance as in MonoSDF. Similarly, we apply an MLP to Vf to get Vcolor and use the same logic to render a color image from Vcolor. For the ith image, the neural rendering is represented by:

Dri = R(Vs,ri), Iri = R(Vcolor,ri). (6)

The ground truth depth labels are needed for supervision along with the original color images. For certain open-source datasets, depth are provided. However, in real-world scenarios, depth labels are typically unavailable. To address this limitation, we leverage large-scale mono depth estimation model, i.e. DepthAnything-V2 [48], to generate pseudo depth labels. The pseudo depth labels are subsequently aligned with depth measurements from depth sensors such as Lidar or RGBD cameras if available through RANSAC, to produce dense and accurate depth maps for training. The loss function in a batch can be formulated as:

1 N

L3D =

N

(||Dgti (ri) − Dri||1 + ||Ii(ri) − Iri||1). (7)

i

Dgti (ri) and Ii(ri) represent the ground truth depth values and color values sampled by ri.

- 4D Spatial-Temporal Encoder: Vf representing the current environment serves perception clues for planning where Vs can be easily converted into occupancy Vo based on the sign of an SDF value. However, besides perception, prediction is also critical for local planning and it models the temporal dynamics of environmental evolution. To this end, we propose a model that directly forecasts future voxel-level spatiotemporal representations.

Specifically, the 4D encoder M4D takes the past voxel features and future timestamps as input, and outputs future voxel features at corresponding timestamps:

{Vˆfj|j = T + 1,··· ,T + F} = M4D({Vfj|j = T − P,··· ,T},{tj|j = T + 1,··· ,T + F}). (8)

where tj denotes the timestamp of the jth frame. The past and current P + 1 voxel features are concatenated in channels and encoded into multi-scale features by a ResNet, which are composed of 3D convolutions. The prediction module is inserted at each feature level. It is implemented by DiT [49] blocks, where encoded features serve as input while future timestamps serve as condition. Then, we adopt an FPN3D to fuse the multi-scale predicted features and obtain final future F voxel features.

The prediction module is also trained in a self-supervised fashion following the pretraining of the 3D Spatial Encoder.

The loss function is computed at both voxel level and pixel level:

1 F

L4D =

F

j

(||Vfj − Vˆfj||1)+

F

N

1 FN

(||Dgtij(rij) − R(Vˆsj,rij)||1+

j

i

||Iij(rij) − R(Vˆcolorj ,rij)||1). (9) The first term measures the voxel-level differences between the original and predicted future voxel features. The second term is computed at the pixel level, where Dgtij(rij) and Iij(rij) represent the ground truth depth values and color values sampled from the i th view of the j th frame by a set of rays rij, respectively. Vˆsj and Vˆcolorj are obtained from the predicted voxel features Vˆfj via MLPs.

After pretraining, the full 4D Sptial-Temporal Encoder is able to produce both current and future environmental states, represented by 4D voxel features. The framework is trained in a self-supervised manner where only depth labels are needed. Also note that the model design and training procedure can support different number of input views which means besides our own data, we can also leverage diverse open-source datasets like depth estimation, autonomous driving and etc.

- 3.2.2 Planning Head For our local path planning problem, we define the action trajectory in the format of

X =< (∆x1,∆y1,∆θ1),··· ,(∆xn,∆yn,∆θn) >, (10)

where each (∆xi,∆yi,∆θi) represents the relative coordinates between adjacent poses. Our planning head takes the pretrained 4D features as condition, along with robot velocity and task information like goal pose, and reconstructs an action trajectory from a Gaussian noise with flow matching [50].

Transformer-based flow matching: Given that our planning task deals with complex environments where trajectories often have multi-modal characteristics, generative methods like diffusion models and flow matching are better suited as are also commonly used in autonomous driving and robot manipulation [28, 51, 52].

We choose flow matching as our main planning method due to its high efficiency which is vital to a real-time system. More specifically, we aim to minimize the Conditional Flow Matching (CFM) objective [50]:

1,Xt∥vt(Xt;θ) − ut(Xt|X1)∥2, (11)

LCFM(θ) = Et,X

where θ is the model parameter, t ∈ U[0,1] is the timestep, X0 ∈ p is the source distribution (usually a normal distribution), X1 ∈ q(·|C) is the target distribution and C is the condition mentioned above.

A transformer-based model is used to represent the vector field, i.e. vt(X;θ|t,C). The condition C consists of four parts: robot velocities, goal points, voxel features {Vfj} and occupancy maps {Voj}. Robot velocities and goal points are normalized and projected to the robot’s ego coordinate respectively, and then tokenized by a linear projection. We concatenate the channel and time dimensions of voxel features, and further combine it at channel dimension with occupancy maps encoded by a convolution neural network. The combined features are tokenized by unfolding its spatial dimensions (width and height) and a 2D positional embedding is also applied. Afterwards, a transformer encoder takes all the tokens above, as well as the timestep features as inputs, and sends the results to a transformer decoder, followed by an MLP-based action head that outputs the corresponding vector.

Masked ESDF loss: A major challenge of our task is to avoid collision with various types of obstacles (static or dynamic) in the environment. Although some guidance-based techniques have been developed [53] to control the vector field of flow matching with a cost function, they may also introduce additional computational overhead, e.g. either conducting Monte Carlo estimation at inference time [54], or learning a surrogate model at training time [55].

To achieve the required low inference latency in our real-time system while maintaining the training cost unchanged, we design a novel technique named masked ESDF loss. Specifically, given a 3D occupancy map Vo, we calculate its

euclidean space distance field (ESDF) map Φ(x,y) by: 1). compressing the 3D map to a 2D binary map Vm(x,y) by taking maximum on the z-dimension, 2). calculating the minimum Euclidean distance from each free pixel to the nearest obstacle pixel [56]:

+D(x,y) if Vm(x,y) = 0 (free space) −D′(x,y) if Vm(x,y) = 1 (obstacle)

(12)

Φ(x,y) =

where D and D′ represent the unsigned Euclidean distance transform and the interior distance transform respectively. During the training phase of flow matching, we approximate the action trajectory with the predicted vector vt by:

X˜ = Xt − t · vt(Xt;θ|t,C). (13) We further calculate the pose trajectory X˜p based on:

 

cosθk−1 −sinθk−1 sinθk−1 cosθk−1

xk yk

xk−1 yk−1

∆xk ∆yk

=

+



θk = θk−1 + ∆θk

Then the ESDF value of each point on the trajectory can be queried by Φ(x,y),(x,y) ∈ X˜p.

However, directly minimizing the ESDF values will result in heading errors, i.e. the trajectory will always head to areas with fewer obstacles rather than the areas of goal points. To address this issue, we add a 2D ground-truth trajectory mask on the ESDF map:

Φ(˜ x,y) = Φ(x,y) · (1 − α · I((x,y) ∈ Ugt)), (14)

where Ugt is the expanded area of the ground-truth trajectory and we use α to punish reconstructed trajectories that are too far away from the ground-truth.

The total loss function with masked ESDF loss can be written as: Lplanning = LCFM − λ ·

Φ(˜ x,y). (15)

(x,y)∈X˜p

In our implementation, the masked ESDF loss is calculated by bilinear grid sampling.

The proposed masked ESDF loss only introduces O(n) additional computational overhead where n is the size of action trajectory, and brings no extra cost to the inference time. We demonstrate in experiments that the collision rate can be significantly reduced with this proposed loss.

##### 3.2.3 Odometry Head

The odometry head predicts relative robot pose given current and past 4D features from the 4D encoder and additional sensor including IMU, wheel. We train a transformer model to fuse information from different sensors (see Fig. 4). For each time step, each modality (4D vision feature, IMU, wheel) goes through a modality-specific tokenizer to get tokens for that time step. Combined with learned modality embedding and temporal position embedding, the tokens from a series are fed into a transformer encoder together with a CLS token which is then used to predict the relative robot pose. Specifically, for 4D vision feature tokenizer, we took inspiration from [21] and computed a correlation volume between two consecutive 3D voxel features. For IMU and wheel data, we use a small LSTM to encode raw data between two image frames into a single token. Training for the odometry head follows a straightforward supervised learning setup where the objective is to minimize the L1 loss between the predicted delta pose and ground truth pose.

### 4 Experiments

We collected data from our in-house built robots operating in diverse environments such as warehouses, office buildings, and homes to train and test Astra. We optimized and deployed Astra on our robots. Specifically, Astra-Local runs on the on-bot edge device, while Astra-Global runs on the cloud.

##### Main Results 1

Astra achieves high end-to-end mission success rate across diverse environments.

To assess the end-to- end performance of Astra, we adopt the settings from [16], where user instructions are randomly selected and the robot is initially placed at random locations within the environment, and then measure the overall success rate of the robot reaching the goal. Additionally, we evaluate the performance of Astra for three navigation sub-tasks, i.e. goal localization, self-localization, and path planning. For goal localization, similar to [16], we measure the success rate (SR) of Astra in correctly localizing user instructions. For self-localization, we compare the estimated robot trajectory with the ground-truth trajectory and consider it a success if the translational error for the entire trajectory is within 1m. For path planning, we implemented a fallback system on the robot. When the trajectory generated by Astra-Local violates collision constraints, the system falls back to a traditional optimization-based planning method. We report the fallback rate (FR) as an indicator of how often the system relies on this fallback mechanism.

| |Warehouse|Office Building|
|---|---|---|
|End-to-end SR|84.2%<br><br>|99.1%|
|Goal Localization SR|98.3%<br><br>|99.1%|
|Self-Localization SR<br><br>|85.9%|100%|
|Path Planning FR<br><br>|8.3%<br><br>|15.6%|

Table 1 Success Rate (SR) of Astra for end-to-end mission, goal localization, self-localization and Fallback Rate (FR) for path planning across diverse environments

As shown in Tab. 1 Astra achieves a high end-to-end success rate in all types of environments. Among the failures, the primary reason for failure in warehouses is due to the fact that at the starting point the robot is not able to self localize. This is because of the highly repetitive environment and the lack of visible landmarks around some random starting locations. When excluding these challenging starting points, the success rate increases to 91.2%. In the office building environment, although the end-to-end success rate is high, the path planning fallback rate increases due to the presence of more dynamic obstacles.

##### Main Results 2

Astra-Global can handle multi-modal localization query across diverse environment.

With our map representation, Astra-Global supports text and image localization queries, as illustrated in Fig. 5, 6. For goal localization, Astra-Global can effectively identify map images and poses matching text instructions.

For robot localization, Astra-Global showcases excellent performance across a wide range of scenarios. As Fig. 7 and Fig. 6(a)(b) show, whether tested in large-scale industrial-looking warehouses or visually distinct office buildings, the model demonstrates its effectiveness. It can adapt well to the significant scale and visual differences between these diverse scenarios, delivering reliable localization results.

We compare Astra-Global with traditional visual place recognition (VPR) method that often relies on models to provide an image embedding and formulate the problem as retrieval. To make a fair comparison between our method and traditional VPR methods, we ensure that the recall rates of the two methods are the same and focus on comparing their precision. The evaluation metrics include the accuracy of pose within a 1-meter distance error and 5-degree angular error and the results are presented in Fig. 7.

The results demonstrate that our method significantly outperforms [2] in all scenarios. Key advantages include:

[Figure 6]

Figure 5 Language-based goal localization task examples.

[Figure 7]

###### (a) Warehouse Localization (b) Office Localization (c) Home Localization

Figure 6 Self-Localization across diverse scenarios.

[Figure 8]

Figure 7 Performance (%) comparison with VPR method (MixVPR) across diverse indoor environments.

- • Better detail capture: VPR, like [2], uses global features, often misses fine details like room numbers. Our method catches these details, avoiding failures in similar scenes with repetitive layouts, as shown in Fig. 8(a).
- • More robust to viewpoint changes: VPR often has trouble with big viewpoint shifts while Astra-Global is more robust as it relies on semantic landmarks. The relative positions between landmarks stay the same even when the camera angle changes. An example is shown in Fig. 8(b).
- • Higher pose accuracy: In the presence of multiple similar candidate positions, our method leverages landmark spatial relationships to select the best-matching pose, achieving significantly higher accuracy within pose error compared to VPR (as shown in Fig. 7).

We also validate the cross-scene generalization capability of Astra-Global in a zero-shot transfer experiment: the model is trained on warehouse and office building datasets, then directly deployed to home environments without any parameter fine-tuning. As shown in Fig. 6(c) and Fig. 8, our method achieves 81.8% pose accuracy under 1m-5° criteria, demonstrating more than 20% percentage points improvement over MixVPR’s 57.7%. When finetuned on limited home environment data, the performance of Astra-Global further boosted to 91.1%.

##### Main Results 3

GRPO improves Astra-Global’s generalization.

To validate the effectiveness of GRPO in our framework, we conduct comprehensive ablation studies on the coarse localization task, evaluating two training paradigms: (1) supervised fine-tuning (SFT) alone and (2) SFT followed by GRPO reinforcement learning. Tab. 2 presents the localization accuracy (%, within 10m and 180°) across different environments. Notably, the Home environment was employed as a zero-shot scenario, enabling an assessment of the generalization capacity of the methods.

scenario Warehouse Office Home

Method #samples

SFT-only 100k 89.9 93.8 93.7 SFT-only 300k 93.1 94.6 97.3 SFT+GRPO 100k + 20k 93.3 95.3 99.9 SFT+GRPO 100k + 200k 95.5 95.3 99.9

###### Table 2 Ablation results of GRPO on coarse localization

[Figure 9]

- (a) Warehouse Localization

[Figure 10]

- (b) Office Localization

Figure 8 Astra-Global’s robust localization performance in cases where VPR fails

The experimental results demonstrate GRPO’s effectiveness in improving performance for coarse localization. In the zero-shot home scene, SFT+GRPO (100k+20k) attains an impressive accuracy of 99.9%, significantly outperforming SFT-only method (93.7%) even with a comparable number of samples. Moreover, within the previously-seen warehouse and office environments, GRPO consistently yields performance gains in the range of 0.7-2.4%. This not only validate its capability to enhance generalization but also highlight its potential to improve sample efficiency through the application of reinforcement learning techniques.

##### Main Results 4

Flow matching with proposed Masked ESDF loss gives the best local trajectory in Astra-Local.

We train our planning head in Astra-Local on 10M trajectory samples recorded via human remote control and evaluate the models on two test dataset: ID (in-distribution) and OOD (out-of-distribution). The OOD dataset exists numerous unseen congested scenarios that are absent from the training data. We mainly compare our methods with ACT [27] and diffusion policy (DP) [51] while the pretrained encoders remain the same. We also compare results w/ and w/o the proposed masked ESDF to exam its usefulness. We use collision rate and velocity as our main evaluation metrics. As safety and velocity often exhibit a strong trade-off, we also use a score function as a supplementary metric, which is learned from data annotated by human experts.

The effectiveness of the proposed masked ESDF loss is shown in Fig. 9a. We can observe that the ESDF loss can significantly reduce the collision rates of all approaches both on ID and OOD datasets. The comparison of different planning heads on OOD dataset is shown in Fig. 9b, where all methods apply the same encoder and ESDF loss. The superscript * represents that we use the trajectory of the highest score out of 10 generated samples. We can see that FM can achieve higher score and velocity while the collision rate can be maintained on the same level. If we sample multiple trajectories, FM* can dominate other approaches in terms of all three metrics.

[Figure 11]

[Figure 12]

(a) Effectiveness of ESDF loss. (b) Comparison of planning heads. Figure 9 Ablation of different planning head & ESDF loss.

The results varying different model size × training dataset size are shown in Tab. 3. We can see that on the ID dataset, as the model scale and dataset size increases, the collision rate will decrease and the velocity will increase, which aligns with the expectation. However, on the OOD dataset, the collision rate does not show obvious improvement, while the velocity and score does increase. Nevertheless, the improvement on the score which reflects the overall performance of trajectories still demonstrate the effectiveness of increasing model and dataset sizes.

ID OOD Collision rate Velocity Score Collision rate Velocity Score

Model Scale Data Scale

1M 3.5% 0.63 0.96 11.4% 0.45 0.52 5M 1.4% 0.75 1.21 11.9% 0.50 0.59 10M 1.3% 0.68 1.10 11.3% 0.47 0.54

Small

1M 3.2% 0.67 0.98 12.8% 0.42 0.48 5M 1.8% 0.78 1.15 15.7% 0.53 0.58 10M 1.1% 0.77 1.17 12.5% 0.55 0.64

Medium

1M 1.5% 0.80 1.26 15.3% 0.55 0.61 5M 1.0% 0.80 1.26 14.2% 0.58 0.68 10M 0.7% 0.87 1.34 8.0% 0.60 0.77

Large

* Velocity values are normalized to [0,1]

Table 3 Ablation of different model size & dataset size for Astra-Local planning head.

###### In addition, we present open-loop qualitative results in Fig. 10, 11, and 12. The left three images depict the

tri-camera color views, and the rightmost one shows the 2D occupancy map. The red trajectories indicate the ground truth, while the blue ones are the model’s outputs, and the green squares mark the local goal. Fig. 10 depicts scenarios where the robots interact with various static obstacles such as picking carts, forklifts, and pallet jacks. Our model can successfully plan a collision-free path to navigate around these obstacles, as shown in the figure. Fig. 11 illustrates the interaction between the robot and an operator moving with a pallet jack. When the operator is at a distance, the predicted trajectory tends to turn right in front of them. However, when the operator gets closer, the trajectory shortens rapidly, indicating deceleration. As the operator is about to leave, the trajectory lengthens again, corresponding to an acceleration.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Figure 10 Planing head case study: interaction with different types of static obstacles.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Figure 11 Planing head case study: interaction with an operator pulling a pallet jack in a warehouse.

###### However, there are also some corner cases. Representative examples are shown in Fig. 12. In the first case, while the

trajectory remains collision-free, it fails to circumvent the obstacle, demonstrating the limitations of relying solely on collision rate. In the second case, the model commits a directional error, resulting in choosing an incorrect path.

[Figure 21]

[Figure 22]

Figure 12 Planing head case study: corner cases

##### Main Results 5

Proposed transformer encoder in Astra-Local is effective for odometry estimation by fusing multi-sensor multi-frame inputs.

We evaluate the odometry head in Astra-Local on our multi-modal dataset containing synchronized image sequences, IMU and wheel measurements, together with ground truth poses. We trained our model and our re-implementation of [21] which serve as our baseline on this dataset.

As in [21], it processes consecutive image frames by first extracting Bird’s Eye View (BEV) features through a depth-aware perspective-to-BEV encoder. A correlation volume is computed from adjacent BEV features, followed by MLP-based regression to estimate relative poses (3-DoF). Consistent with the original methodology, we adopted a 7×7 search window configuration with a feature resolution of 0.2 m/pixel. This implementation achieved mean positional error of 0.006m and angular error of 0.0014 rad per frame. Subsequent trajectory propagation using these relative poses demonstrated approximately 5% relative error compared to ground truth trajectories.

In contrast, our transformer-based odometry head employs temporal modeling of multi-frame sensor data. Specifically, the network architecture integrates both measurements from current frame and historical sensor inputs from the preceding 9 frames to predict inter-frame motion. Note that our BEV correlation tokenizer uses identical configuration parameters as our re-implementation of [21]. We compare different variant of the odometry head in Astra-Local to the baseline [21]. Additionally, to exam the effect of the temporal modeling and multi-sensor fusion, we trained Astra-Local (bev-only) where we only fuse multi-frame visual data in our transformer encoder, Astra-Local (BEV + IMU) where wheel data was not used and Astra-Local where all sensor inputs were used.

- Tab. 4 and Fig. 13 present quantitative results and trajectory visualizations, showing significant improvements over the two-frame BEV-ODOM baseline. Specifically, incorporating IMU measurements boosts rotational estimation accuracy, cutting the overall trajectory error to around 2%. Moreover, integrating wheel data further enhances scale stability and estimation accuracy. These enhancements highlight the advantages of combining temporal fusion and multi-sensor data in our odometry head.

Methods RTE(%) RRE(◦/10m) ATE(m) BEV-ODOM 5.46% 6.36 1.27

Astra-Local (visual only) 3.13% 2.85 1.08 Astra-Local (visual + imu) 2.04% 1.19 0.48

Astra-Local 1.92% 0.66 0.26

###### Table 4 Comparison of odometry performance

[Figure 23]

[Figure 24]

[Figure 25]

(a) Sequence 1 (b) Sequence 2 (c) Sequence 3 Figure 13 Odometry trajectory comparisons between Astra-Local and baselines.

##### Main Results 6

The 4D Spatial-Temporal Encoder in Astra-Local provides robust feature for downstream tasks.

We examine the effectiveness of the 4D Spatial-Temporal encoder in Astra-Local by comparing two downstream tasks: the occupancy prediction and the local path planning.

- Tab. 5 shows the results of occupancy prediction where we took the 3D spatial encoder and train it on an occupancy prediction dataset w/ and w/o the pretrained weights. The self-supervised pretraining of the 3D encoder improves the results by a decent margin.

Pretrained Weights

Office Building Warehouse Total IoU Obstacle IoU Total IoU Obstacle IoU

w/o pretrain 41.10 30.78 40.91 22.45 w/ pretrain 42.16 32.26 42.55 25.81

Table 5 Effect of 3D spatial encoder on downstream occupancy prediction task.

- Tab. 6 reveals that, compared to using only a 3D spatial encoder, the addition of the 4D Spatial-Temporal module can increase the overall velocity and decrease the collision rate in most settings. This demonstrates the significance of prediction in local path planning.

ID OOD Collision rate Velocity Collision rate Velocity

Encoder

###### VAE + ESDF

- 3D encoder 0.6% 0.75 8.4% 0.35
- 4D encoder 0.0% 0.72 2.0% 0.40

###### FM + ESDF

- 3D encoder 0.9% 0.78 10.3% 0.52
- 4D encoder 0.7% 0.87 8.0% 0.60

Table 6 Planning Performance Comparison of Different Encoders

We also investigate the effect of model scale and dataset size for the 4D Spatial-Temporal Encoder. The full pretraining dataset contains about 10M training samples. We progressively varied the scale of pretraining dataset from 250k to 1.5M and finally to 10M samples. We also evaluate by finetuning the pretrained models on occupancy prediction tasks. It can be seen in Tab. 7 that the model performs the best at the scale of 1.5M. When it comes to

the 10M, the performance degrades a little. The observed performance plateau could potentially be attributed to the model’s limited capacity, constraining its ability to benefit from larger-scale pretraining dataset. Therefore, we also experimented with larger backbones, switching from the original ViT-S to ViT-L architectures, while keeping the scale of pretraining dataset at 10M. We can see that with increased model capacity, the largest model now performs the best.

Office Building Warehouse Total IoU Obstacle IoU Total IoU Obstacle IoU

Dataset & Model Scale

250K (ViT-S) 40.14 30.84 36.64 25.68 1.5M (ViT-S) 42.30 33.17 37.57 27.12 10M (ViT-S) 41.65 32.40 36.27 25.37 10M (ViT-L) 43.27 33.94 41.64 26.91

Table 7 Ablation of dataset and model size of 3D encoder for downstream occupancy prediction.

Finally, we compare the 4D encoder with the SOTA occupancy forecasting methods. The 4D encoder takes the past

- 4 frames of voxel features as input and output the future 4 frames, where each frame of voxel features is produced by the pretrained 3D encoder. We compare our 4D encode with the our re-implementation of OccWorld [40] and Cam4DOcc [42] which are trained using Equation 9, with the 3D encoder frozen. We evaluate the predicted voxel features from different aspects. As the voxel features can be converted into SDF via the fixed MLP from 3D encoder which can further be transformed to occupancy grid. We evaluate the predicted occupancy grid. Besides, the voxel features can be rendered into depth and color images, we use AbsRel and PSNR to evaluate the quality of the rendered future depth and color image respectively. The results are shown in Tab. 8. Benefiting from the multi-scale architecture and the DiT block, voxel features predicted from our 4D encoder can generate better occupancy, depth and color images. In Fig. 14, we show some examples of rendered depth from the predicted future voxels. More visualization results can be found on our website.

Methods

Warehouse Total IoU Obstacle IoU AbsRel (Depth) PSNR (Color)

OccWorld 82.90 41.05 0.1752 29.1943 Cam4DOcc 81.17 51.01 0.1376 28.9919 Astra-Local 84.80 56.66 0.1179 29.3021

Table 8 Comparison of 4D encoder to the SOTA methods

- 5 Conclusion and Future Work

In this report, we introduce our newly developed Astra, a dual-model architecture for mobile robot navigation. Unlike traditional modular systems, it integrates key navigation tasks—such as goal/robot localization and local path planning—into two cohesive models. Astra-Global, a multimodal LLM, localizes multimodal queries (e.g., text or images) within a pre-built hybrid topological-semantic map. Our results demonstrate its robustness across diverse environments and ability to generalize to unseen scenarios with minimal or no additional data. Astra-Local, a multi-sensor multi-task network focused on path planning and odometry estimation, leverages a pretrained 4D spatial-temporal encoder to generate robust features for downstream tasks. Its planning head uses a novel masked ESDF loss combined with flow matching to minimize collision rates, while the odometry head fuses multi-sensor data via a transformer encoder for accurate relative pose estimation, effectively integrating diverse sensor inputs. In the future, we plan to deploy Astra across broader scenarios and continue enhancing its robustness and generalization capabilities. For Astra-Global: Although the current map representation balances information loss and token length, it may still lack certain semantic details critical for localization. We intend to investigate alternative map compression approaches to retain essential semantic information while optimizing efficiency. Additionally, the current localization relies solely on single-frame observations, which can fail in scenarios where even humans would

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

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

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Figure 14 Visualization for 4D Spatial-Temporal Encoder. First column shows the current color image. Second column is the rendered depth from voxel features in current frame. Column 3-6 shows rendered depth from predicted voxel features in 0.5s, 1s, 1.5s, 2s. Colder colors indicate objects that are closer, while warmer colors indicate objects that are farther away.

struggle (e.g., featureless or highly repetitive environments). To address this, we plan to enable the robot to actively explore its surroundings and incorporate temporal reasoning into the model, leveraging sequential observations for more robust localization. For Astra-Local: Our real-robot deployments reveal a non-negligible fallback rate, stemming from both the model’s generalization limitations and the tendency of the rule-based fallback system to trigger erroneously in edge cases. We aim to enhance the model’s robustness to out-of-distribution (OOD) scenarios and redesign the fallback system to be a more seamless, integrated component of the system. Furthermore, we plan to integrate instruction-following capabilities into the model, enabling natural human-robot interaction and expanding usability in dynamic, human-centric environments.

- 6 Contributions and Acknowledgments The names are sorted in alphabetical order of the last name.

#### Core Contributors

Sheng Chen, Peiyu He, Jiaxin Hu, Ziyang Liu, Yansheng Wang, Tao Xu, Chi Zhang, Chongchong Zhang

#### Contributors

Chao An, Shiyu Cai, Duo Cao, Kangping Chen, Shuai Chu, Tianwei Chu, Mingdi Dan, Min Du, Weiwei Fang, Pengyou Fu, Junkai Hu, Xiaowei Jiang, Zhaodi Jiang, Fuxuan Li, Jun Li, Minghui Li, Mingyao Li, Yanchang Li, Zhibin Li, Guangming Liu, Kairui Liu, Lihao Liu, Weizhi Liu, Xiaoshun Liu, Yufei Liu, Yunfei Liu, Qiang Lu, Yuanfei Luo, Xiang Lv, Hongying Ma, Sai Ma, Lingxian Mi, Sha Sa, Hongxiang Shu, Lei Tian, Chengzhi Wang, Jiayu Wang, Kaijie Wang, Qingyi Wang, Renwen Wang, Tao Wang, Wei Wang, Xirui Wang, Chao Wei, Xuguang Wei, Zijun Xia, Zhaohao Xiao, Tingshuai Yan, Liyan Yang, Yifan Yang, Zhikai Yang, Zhong Yin, Li Yuan, Liuchun Yuan, Chi Zhang, Jinyang Zhang, Junhui Zhang, Linge Zhang, Zhenyi Zhang, Zheyu Zhang, Dongjie Zhu

#### Team Lead

Hang Li, Yangang Zhang

### References

- [1] Daniel Kahneman. Thinking, Fast and Slow. Farrar, Straus and Giroux, 2011.

- [2] Amar Ali-Bey, Brahim Chaib-Draa, and Philippe Giguere. Mixvpr: Feature mixing for visual place recognition. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2998–3007, 2023.

- [3] Ruotong Wang, Yanqing Shen, Weiliang Zuo, Sanping Zhou, and Nanning Zheng. Transvpr: Transformer-based place recognition with multi-level attention aggregation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13648–13657, 2022.

- [4] Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford, and Tobias Fischer. Patch-netvlad: Multi-scale fusion of locally-global descriptors for place recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14141–14152, 2021.

- [5] Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Larsson, and Marc Pollefeys. Pixel-perfect structure-from-motion with featuremetric refinement. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5987–5997, 2021.

- [6] Zhihuang Zhang, Meng Xu, Wenqiang Zhou, Tao Peng, Liang Li, and Stefan Poslad. Bev-locator: An end-to-end visual semantic localization network using multi-view images. Science China Information Sciences, 68(2):122106, 2025.

- [7] Yuzhe He, Shuang Liang, Xiaofei Rui, Chengying Cai, and Guowei Wan. Egovm: Achieving precise ego-localization using lightweight vectorized maps. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 12248–12255. IEEE, 2024.

- [8] Hang Wu, Zhenghao Zhang, Siyuan Lin, Xiangru Mu, Qiang Zhao, Ming Yang, and Tong Qin. Maplocnet: Coarse-to-fine feature registration for visual re-localization in navigation maps. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 13198–13205. IEEE, 2024.

- [9] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [10] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

- [11] Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, et al. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024.

- [12] Gengze Zhou, Yicong Hong, and Qi Wu. Navgpt: Explicit reasoning in vision-and-language navigation with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7641–7649, 2024.

- [13] Hang Yin, Xiuwei Xu, Lingqing Zhao, Ziwei Wang, Jie Zhou, and Jiwen Lu. Unigoal: Towards universal zero-shot goal-oriented navigation. arXiv preprint arXiv:2503.10630, 2025.

- [14] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

- [15] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

- [16] Hao-Tien L. Chiang etc. Mobility vla: Multimodal instruction navigation with long-context vlms and topological graphs. In arxiv, 2024.

- [17] Jiarong Lin and Fu Zhang. R 3 live: A robust, real-time, rgb-colored, lidar-inertial-visual tightly-coupled state estimation and mapping package. In 2022 International Conference on Robotics and Automation (ICRA), pages 10672–10678. IEEE, 2022.

- [18] Tixiao Shan, Brendan Englot, Carlo Ratti, and Daniela Rus. Lvi-sam: Tightly-coupled lidar-visual-inertial odometry via smoothing and mapping. In 2021 IEEE international conference on robotics and automation (ICRA), pages 5692–5698. IEEE, 2021.

- [19] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558–16569, 2021.

- [20] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch visual odometry. Advances in Neural Information Processing Systems, 36:39033–39051, 2023.

- [21] Yufei Wei, Sha Lu, Fuzhang Han, Rong Xiong, and Yue Wang. Bev-odom: Reducing scale drift in monocular visual odometry with bev representation. In IROS, 2024.

- [22] Sachini Herath, Hang Yan, and Yasutaka Furukawa. Ronin: Robust neural inertial navigation in the wild: Benchmark, evaluations, & new methods. In 2020 IEEE international conference on robotics and automation (ICRA), pages 3146–3152. IEEE, 2020.

- [23] Leyuan Sun, Guanqun Ding, Yue Qiu, Yusuke Yoshiyasu, and Fumio Kanehiro. Transfusionodom: Transformer-based lidar-inertial fusion odometry estimation. IEEE Sensors Journal, 23(18):22064–22079, 2023.

- [24] Yunus Bilge Kurt, Ahmet Akman, and A Aydın Alatan. Causal transformer for fusion and pose estimation in deep visual inertial odometry. arXiv preprint arXiv:2409.08769, 2024.

- [25] Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, et al. End to end learning for self-driving cars. CoRR, abs/1604.07316, 2016.

- [26] Yihan Hu, Jiazhi Yang, Li Chen, et al. Planning-oriented autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 17853–17862. IEEE, 2023.

- [27] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

- [28] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

- [29] Zhenhua Xu, Yujia Zhang, Enze Xie, et al. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics Autom. Lett., 9(10):8186–8193, 2024.

- [30] Tesla. Autopilot and full self-driving (supervised), 2025.
- [31] Gongjin Lan and Qi Hao. End-to-end planning of autonomous driving in industry and academia: 2022-2023. CoRR, abs/2401.08658, 2024.

- [32] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. IEEE Trans. Pattern Anal. Mach. Intell., 46(12):10164–10183, 2024.

- [33] Anh-Quan Cao and Raoul De Charette. Monoscene: Monocular 3d semantic scene completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3991–4001, 2022.

- [34] Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Tri-perspective view for vision-based 3d semantic occupancy prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9223–9232, 2023.

- [35] Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21729–21740, 2023.

- [36] Honghui Yang, Sha Zhang, Di Huang, Xiaoyang Wu, Haoyi Zhu, Tong He, Shixiang Tang, Hengshuang Zhao, Qibo Qiu, Binbin Lin, et al. Unipad: A universal pre-training paradigm for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15238–15250, 2024.

- [37] Mingjie Pan, Jiaming Liu, Renrui Zhang, Peixiang Huang, Xiaoqi Li, Hongwei Xie, Bing Wang, Li Liu, and Shanghang Zhang. Renderocc: Vision-centric 3d occupancy prediction with 2d rendering supervision. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 12404–12411. IEEE, 2024.

- [38] Yuanhui Huang, Wenzhao Zheng, Borui Zhang, Jie Zhou, and Jiwen Lu. Selfocc: Self-supervised vision-based 3d occupancy prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19946–19956, 2024.

- [39] Chubin Zhang, Juncheng Yan, Yi Wei, Jiaxin Li, Li Liu, Yansong Tang, Yueqi Duan, and Jiwen Lu. Occnerf: Self-supervised multi-camera occupancy prediction with neural radiance fields. CoRR, 2023.

- [40] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In European conference on computer vision, pages 55–72. Springer, 2024.

- [41] Yu Yang, Jianbiao Mei, Yukai Ma, Siliang Du, Wenqing Chen, Yijie Qian, Yuxiang Feng, and Yong Liu. Driving in the occupancy world: Vision-centric 4d occupancy forecasting and planning via world models for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9327–9335, 2025.

- [42] Junyi Ma, Xieyuanli Chen, Jiawei Huang, Jingyi Xu, Zhen Luo, Jintao Xu, Weihao Gu, Rui Ai, and Hesheng Wang. Cam4docc: Benchmark for camera-only 4d occupancy forecasting in autonomous driving applications. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21486–21495, 2024.

- [43] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016.

- [44] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [45] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.

- [46] Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16, pages 194–210. Springer, 2020.

- [47] Simon Haykin. Neural networks: a comprehensive foundation. Prentice Hall PTR, 1994.

- [48] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024.

- [49] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [50] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, 2023.

- [51] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023.

- [52] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. arXiv preprint arXiv:2411.15139, 2024.

- [53] Ruiqi Feng, Tailin Wu, Chenglei Yu, Wenhao Deng, and Peiyan Hu. On the guidance of flow matching. CoRR, abs/2502.02150, 2025.

- [54] Ricky T. Q. Chen and Yaron Lipman. Flow matching on general geometries. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, 2024.

- [55] Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, and Jun Zhu. Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 22825–22855. PMLR, 2023.

- [56] Pedro F. Felzenszwalb and Daniel P. Huttenlocher. Distance transforms of sampled functions. Theory Comput., 8(1):415–428, 2012.

